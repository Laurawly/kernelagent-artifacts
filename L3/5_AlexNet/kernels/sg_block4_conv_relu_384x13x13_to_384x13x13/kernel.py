import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_CO": 64, "BLOCK_P": 64, "BLOCK_CI": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_CO": 64, "BLOCK_P": 32, "BLOCK_CI": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_CO": 32, "BLOCK_P": 64, "BLOCK_CI": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_CO": 32, "BLOCK_P": 32, "BLOCK_CI": 32}, num_warps=4, num_stages=2),
    ],
    key=["N", "CIN", "COUT", "H", "W"],
)
@triton.jit
def _conv2d_bias_relu_nchw_3x3(
    x_ptr, w_ptr, b_ptr, y_ptr,
    N, CIN, COUT, H, W, P,  # P = H*W
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wco, stride_wci, stride_wkh, stride_wkw,
    stride_yn, stride_yc, stride_yh, stride_yw,
    # Compile-time constants
    BLOCK_CO: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    PAD_H: tl.constexpr,
    PAD_W: tl.constexpr,
):
    """
    NCHW convolution with 3x3 kernel, stride=1, padding=1, dilation=1, groups=1, fused with bias and ReLU.
    - Computes a tile [BLOCK_CO x BLOCK_P] for a single batch 'n'.
    - Accumulates in fp32; inputs/weights are typically bf16.
    """
    pid_p = tl.program_id(0)  # tiles over spatial positions (H*W)
    pid_n = tl.program_id(1)  # batch index
    pid_co = tl.program_id(2)  # tiles over output channels

    n = pid_n

    # Offsets for output channels and positions
    offs_co = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    offs_p = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)

    # Mask for valid tiles
    mask_co = offs_co < COUT
    mask_p = offs_p < P

    # Map linear position (H*W) to (y, x)
    oy = offs_p // W
    ox = offs_p - oy * W  # equivalent to offs_p % W

    # Initialize fp32 accumulator
    acc = tl.zeros((BLOCK_CO, BLOCK_P), dtype=tl.float32)

    # Loop over input channels in chunks and over 3x3 kernel
    # Accumulate acc += W[co, ci, ky, kx] @ X[n, ci, oy+ky-PAD_H, ox+kx-PAD_W]
    # X tile: [BLOCK_CI, BLOCK_P], W tile: [BLOCK_CO, BLOCK_CI]
    for ci0 in range(0, CIN, BLOCK_CI):
        offs_ci = ci0 + tl.arange(0, BLOCK_CI)
        valid_ci = offs_ci < CIN

        # Process 3x3 kernel
        for ky in range(KH):
            iy = oy + ky - PAD_H  # [BLOCK_P]
            in_y_ok = (iy >= 0) & (iy < H)
            for kx in range(KW):
                ix = ox + kx - PAD_W  # [BLOCK_P]
                in_x_ok = (ix >= 0) & (ix < W)
                x_mask = (valid_ci[:, None]) & (mask_p[None, :]) & in_y_ok[None, :] & in_x_ok[None, :]

                # Build pointers for X tile and load
                x_ptrs = (
                    x_ptr
                    + n * stride_xn
                    + offs_ci[:, None] * stride_xc
                    + iy[None, :] * stride_xh
                    + ix[None, :] * stride_xw
                )
                x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)
                x_tile_f32 = x_tile.to(tl.float32)

                # Build pointers for W tile and load
                w_ptrs = (
                    w_ptr
                    + offs_co[:, None] * stride_wco
                    + offs_ci[None, :] * stride_wci
                    + ky * stride_wkh
                    + kx * stride_wkw
                )
                w_mask = (mask_co[:, None]) & (valid_ci[None, :])
                w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)
                w_tile_f32 = w_tile.to(tl.float32)

                # Accumulate using matrix multiply: (co x ci) @ (ci x p) -> (co x p)
                acc += tl.dot(w_tile_f32, x_tile_f32)

    # Add bias and apply ReLU
    # bias is per-output-channel: shape [COUT]
    b = tl.load(b_ptr + offs_co, mask=mask_co, other=0.0).to(tl.float32)
    acc = acc + b[:, None]
    zero = tl.zeros((1,), dtype=tl.float32)
    acc = tl.maximum(acc, zero)

    # Store result to output tensor, cast to output dtype
    # We'll cast to the destination pointer's element type.
    # Target shape NCHW: y[n, co, oy, ox]
    y_ptrs = (
        y_ptr
        + n * stride_yn
        + offs_co[:, None] * stride_yc
        + oy[None, :] * stride_yh
        + ox[None, :] * stride_yw
    )
    store_mask = (mask_co[:, None]) & (mask_p[None, :])
    # Cast accumulator to the output dtype
    out_val = acc.to(y_ptr.dtype.element_ty)
    tl.store(y_ptrs, out_val, mask=store_mask)


def kernel_function(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Fused Conv2D (3x3, stride=1, padding=1, dilation=1, groups=1) + Bias + ReLU for NCHW tensors.

    What is fused:
    - Convolution accumulation in fp32
    - Bias add (per output channel) in the epilogue
    - ReLU activation in the epilogue
    All three stages occur in a single Triton kernel to minimize memory traffic and kernel launches.

    Constraints:
    - Layout: NCHW
    - Kernel: 3x3, stride=1, padding=1, dilation=1, groups=1
    - Dtype: works with bf16/fp16/fp32 inputs/weights; accumulation is in fp32; result cast to x.dtype.
    - The wrapper only validates, allocates, and launches. All math runs inside the Triton kernel.

    Args:
        x: Input tensor of shape [N, C_in, H, W], dtype typically torch.bfloat16
        weight: Weights tensor of shape [C_out, C_in, 3, 3], same dtype as x
        bias: Bias tensor of shape [C_out], same dtype as x

    Returns:
        Output tensor y of shape [N, C_out, H, W], dtype same as input x.
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda, "All tensors must be on CUDA device."
    assert x.ndim == 4 and weight.ndim == 4 and bias.ndim == 1, "Invalid tensor ranks."
    N, CIN, H, W = x.shape
    COUT, WCIN, KH, KW = weight.shape
    assert CIN == WCIN, "C_in mismatch between input and weight."
    assert KH == 3 and KW == 3, "This implementation expects a 3x3 kernel."
    assert bias.shape[0] == COUT, "Bias shape mismatch."
    # Stride=1, padding=1, dilation=1, groups=1 semantics are assumed by kernel math
    # Test case uses these exact parameters.

    # Allocate output; dtype same as input
    y = torch.empty((N, COUT, H, W), device=x.device, dtype=x.dtype)

    # Flattened spatial size
    P = H * W

    # Extract strides in elements
    sxn, sxc, sxh, sxw = x.stride()
    swco, swci, swkh, swkw = weight.stride()
    syn, syc, syh, syw = y.stride()

    # Grid: tiles over (P, N, COUT)
    # BLOCK sizes are autotuned, so grid expressed in terms of meta
    def grid(meta):
        return (
            triton.cdiv(P, meta["BLOCK_P"]),  # tiles of spatial positions
            N,                                 # batch dimension
            triton.cdiv(COUT, meta["BLOCK_CO"])  # tiles of output channels
        )

    # Launch kernel
    _conv2d_bias_relu_nchw_3x3[grid](
        x, weight, bias, y,
        N, CIN, COUT, H, W, P,
        sxn, sxc, sxh, sxw,
        swco, swci, swkh, swkw,
        syn, syc, syh, syw,
        KH=3, KW=3, PAD_H=1, PAD_W=1,
    )

    return y