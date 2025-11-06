import torch
import triton
import triton.language as tl


@triton.jit
def _fused_conv_relu_pool_kernel_nchw(
    x_ptr,                # *bf16 [N, C_in, H, W]
    w_ptr,                # *bf16 [C_out, C_in, 3, 3]
    b_ptr,                # *bf16 [C_out]
    out_ptr,              # *bf16 [N, C_out, PH, PW]
    N, C_in, H, W, C_out,
    PH, PW,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wo, stride_wi, stride_wk, stride_wl,  # strides for weight [O, I, KH, KW]
    stride_on, stride_oc, stride_oh, stride_ow,  # strides for output [N, O, PH, PW]
    BLOCK_OC: tl.constexpr,
):
    """
    Fused Conv3x3 (stride=1, pad=1) + ReLU + MaxPool3x3 (stride=2, pad=1) for NCHW tensors.

    Each program instance computes one pooled output location (y_pool, x_pool) and a block of output channels.
    It performs the 3x3 max pooling over the ReLU(conv) results by streaming through input channels and kernel taps,
    accumulating nine separate conv sums for the nine contributing conv positions, and then taking the max of those
    nine positions, adding bias, and applying ReLU once.

    Correctness note:
    - Since bias is constant across spatial positions, max_p(ReLU(conv_p + b)) == ReLU(max_p(conv_p) + b).
      We therefore compute max over pre-bias conv sums and apply bias + ReLU once at the end.
    - Pool padding introduces virtual zeros into the pooled window; this is equivalent to ignoring positions whose
      corresponding conv index is outside [0..H-1]x[0..W-1]. We implement this by marking those positions invalid and
      excluding them from the final max (treat as -inf before the max).
    - Conv padding (pad=1) is handled by masking input loads outside image borders to zero.

    Data types:
    - Inputs and weights are expected in bf16; accumulation is done in fp32, and the final result is cast back to the
      output dtype (bf16).
    """
    pid_spatial = tl.program_id(axis=0)  # pooled spatial index
    pid_no = tl.program_id(axis=1)       # combined (N, OC-block) index

    # Decode pooled spatial coordinates
    y_pool = pid_spatial // PW
    x_pool = pid_spatial % PW

    # Map pid_no into n and oc-block
    num_oc_blocks = tl.cdiv(C_out, BLOCK_OC)
    n = pid_no // num_oc_blocks
    oc_block = pid_no % num_oc_blocks

    # Offsets for output channels this program computes
    oc_start = oc_block * BLOCK_OC
    offs_oc = oc_start + tl.arange(0, BLOCK_OC)
    oc_mask = offs_oc < C_out

    # Prepare nine accumulators for the nine conv positions in the pooling window
    acc00 = tl.zeros((BLOCK_OC,), dtype=tl.float32)
    acc01 = tl.zeros((BLOCK_OC,), dtype=tl.float32)
    acc02 = tl.zeros((BLOCK_OC,), dtype=tl.float32)
    acc10 = tl.zeros((BLOCK_OC,), dtype=tl.float32)
    acc11 = tl.zeros((BLOCK_OC,), dtype=tl.float32)
    acc12 = tl.zeros((BLOCK_OC,), dtype=tl.float32)
    acc20 = tl.zeros((BLOCK_OC,), dtype=tl.float32)
    acc21 = tl.zeros((BLOCK_OC,), dtype=tl.float32)
    acc22 = tl.zeros((BLOCK_OC,), dtype=tl.float32)

    # Precompute conv-grid indices for the 3x3 pooling window centered at (2*y_pool, 2*x_pool)
    yprime0 = 2 * y_pool - 1
    yprime1 = 2 * y_pool + 0
    yprime2 = 2 * y_pool + 1
    xprime0 = 2 * x_pool - 1
    xprime1 = 2 * x_pool + 0
    xprime2 = 2 * x_pool + 1

    # Validity of conv positions within conv output grid [0..H-1]x[0..W-1]
    valid_y0 = (yprime0 >= 0) & (yprime0 < H)
    valid_y1 = (yprime1 >= 0) & (yprime1 < H)
    valid_y2 = (yprime2 >= 0) & (yprime2 < H)
    valid_x0 = (xprime0 >= 0) & (xprime0 < W)
    valid_x1 = (xprime1 >= 0) & (xprime1 < W)
    valid_x2 = (xprime2 >= 0) & (xprime2 < W)

    v00 = valid_y0 & valid_x0
    v01 = valid_y0 & valid_x1
    v02 = valid_y0 & valid_x2
    v10 = valid_y1 & valid_x0
    v11 = valid_y1 & valid_x1
    v12 = valid_y1 & valid_x2
    v20 = valid_y2 & valid_x0
    v21 = valid_y2 & valid_x1
    v22 = valid_y2 & valid_x2

    # Loop over input channels and 3x3 kernel taps
    # We keep weights loaded once per (c, kh, kw) and accumulate contributions to all nine positions.
    for c in tl.range(0, C_in):
        # kernel rows/cols are small -> unroll statically
        for kh in tl.static_range(0, 3):
            for kw in tl.static_range(0, 3):
                # Load weight vector for oc tile
                w_ptrs = w_ptr + offs_oc * stride_wo + c * stride_wi + kh * stride_wk + kw * stride_wl
                w_vec = tl.load(w_ptrs, mask=oc_mask, other=0.0).to(tl.float32)

                # For each pooling position, compute the input coordinate for this kernel tap
                # and accumulate w_vec * input_val into the appropriate accumulator.
                # Position (0,0)
                y_in = yprime0 + kh - 1
                x_in = xprime0 + kw - 1
                in_bounds = (y_in >= 0) & (y_in < H) & (x_in >= 0) & (x_in < W)
                x_ptrs = x_ptr + n * stride_xn + c * stride_xc + y_in * stride_xh + x_in * stride_xw
                val = tl.load(x_ptrs, mask=in_bounds, other=0.0).to(tl.float32)
                val = tl.where(v00, val, 0.0)
                acc00 += w_vec * val

                # Position (0,1)
                y_in = yprime0 + kh - 1
                x_in = xprime1 + kw - 1
                in_bounds = (y_in >= 0) & (y_in < H) & (x_in >= 0) & (x_in < W)
                x_ptrs = x_ptr + n * stride_xn + c * stride_xc + y_in * stride_xh + x_in * stride_xw
                val = tl.load(x_ptrs, mask=in_bounds, other=0.0).to(tl.float32)
                val = tl.where(v01, val, 0.0)
                acc01 += w_vec * val

                # Position (0,2)
                y_in = yprime0 + kh - 1
                x_in = xprime2 + kw - 1
                in_bounds = (y_in >= 0) & (y_in < H) & (x_in >= 0) & (x_in < W)
                x_ptrs = x_ptr + n * stride_xn + c * stride_xc + y_in * stride_xh + x_in * stride_xw
                val = tl.load(x_ptrs, mask=in_bounds, other=0.0).to(tl.float32)
                val = tl.where(v02, val, 0.0)
                acc02 += w_vec * val

                # Position (1,0)
                y_in = yprime1 + kh - 1
                x_in = xprime0 + kw - 1
                in_bounds = (y_in >= 0) & (y_in < H) & (x_in >= 0) & (x_in < W)
                x_ptrs = x_ptr + n * stride_xn + c * stride_xc + y_in * stride_xh + x_in * stride_xw
                val = tl.load(x_ptrs, mask=in_bounds, other=0.0).to(tl.float32)
                val = tl.where(v10, val, 0.0)
                acc10 += w_vec * val

                # Position (1,1)
                y_in = yprime1 + kh - 1
                x_in = xprime1 + kw - 1
                in_bounds = (y_in >= 0) & (y_in < H) & (x_in >= 0) & (x_in < W)
                x_ptrs = x_ptr + n * stride_xn + c * stride_xc + y_in * stride_xh + x_in * stride_xw
                val = tl.load(x_ptrs, mask=in_bounds, other=0.0).to(tl.float32)
                val = tl.where(v11, val, 0.0)
                acc11 += w_vec * val

                # Position (1,2)
                y_in = yprime1 + kh - 1
                x_in = xprime2 + kw - 1
                in_bounds = (y_in >= 0) & (y_in < H) & (x_in >= 0) & (x_in < W)
                x_ptrs = x_ptr + n * stride_xn + c * stride_xc + y_in * stride_xh + x_in * stride_xw
                val = tl.load(x_ptrs, mask=in_bounds, other=0.0).to(tl.float32)
                val = tl.where(v12, val, 0.0)
                acc12 += w_vec * val

                # Position (2,0)
                y_in = yprime2 + kh - 1
                x_in = xprime0 + kw - 1
                in_bounds = (y_in >= 0) & (y_in < H) & (x_in >= 0) & (x_in < W)
                x_ptrs = x_ptr + n * stride_xn + c * stride_xc + y_in * stride_xh + x_in * stride_xw
                val = tl.load(x_ptrs, mask=in_bounds, other=0.0).to(tl.float32)
                val = tl.where(v20, val, 0.0)
                acc20 += w_vec * val

                # Position (2,1)
                y_in = yprime2 + kh - 1
                x_in = xprime1 + kw - 1
                in_bounds = (y_in >= 0) & (y_in < H) & (x_in >= 0) & (x_in < W)
                x_ptrs = x_ptr + n * stride_xn + c * stride_xc + y_in * stride_xh + x_in * stride_xw
                val = tl.load(x_ptrs, mask=in_bounds, other=0.0).to(tl.float32)
                val = tl.where(v21, val, 0.0)
                acc21 += w_vec * val

                # Position (2,2)
                y_in = yprime2 + kh - 1
                x_in = xprime2 + kw - 1
                in_bounds = (y_in >= 0) & (y_in < H) & (x_in >= 0) & (x_in < W)
                x_ptrs = x_ptr + n * stride_xn + c * stride_xc + y_in * stride_xh + x_in * stride_xw
                val = tl.load(x_ptrs, mask=in_bounds, other=0.0).to(tl.float32)
                val = tl.where(v22, val, 0.0)
                acc22 += w_vec * val

    # Exclude invalid conv positions from the max by setting them to -inf
    neg_inf = -float("inf")
    acc00 = tl.where(v00, acc00, neg_inf)
    acc01 = tl.where(v01, acc01, neg_inf)
    acc02 = tl.where(v02, acc02, neg_inf)
    acc10 = tl.where(v10, acc10, neg_inf)
    acc11 = tl.where(v11, acc11, neg_inf)
    acc12 = tl.where(v12, acc12, neg_inf)
    acc20 = tl.where(v20, acc20, neg_inf)
    acc21 = tl.where(v21, acc21, neg_inf)
    acc22 = tl.where(v22, acc22, neg_inf)

    # Max across the nine positions
    smax = tl.maximum(acc00, acc01)
    smax = tl.maximum(smax, acc02)
    smax = tl.maximum(smax, acc10)
    smax = tl.maximum(smax, acc11)
    smax = tl.maximum(smax, acc12)
    smax = tl.maximum(smax, acc20)
    smax = tl.maximum(smax, acc21)
    smax = tl.maximum(smax, acc22)

    # Add bias and apply ReLU
    b_ptrs = b_ptr + offs_oc
    bias = tl.load(b_ptrs, mask=oc_mask, other=0.0).to(tl.float32)
    smax = smax + bias
    zero = tl.zeros((BLOCK_OC,), dtype=tl.float32)
    smax = tl.maximum(smax, zero)

    # Store to output
    out_ptrs = out_ptr + n * stride_on + offs_oc * stride_oc + y_pool * stride_oh + x_pool * stride_ow
    tl.store(out_ptrs, smax.to(out_ptr.dtype.element_ty), mask=oc_mask)


def kernel_function(x: torch.Tensor, conv_weight: torch.Tensor, conv_bias: torch.Tensor) -> torch.Tensor:
    """
    Fused Conv3x3 (stride=1, pad=1) -> ReLU -> MaxPool3x3 (stride=2, pad=1) for NCHW tensors.

    What is fused:
    - 3x3 convolution (with bias) is fused with ReLU and 3x3 max pooling (stride 2) into a single Triton kernel.
    - We compute the nine conv outputs that would feed the pooled value implicitly and take their spatial max
      inside the kernel, then add bias and apply ReLU once:
        out[n, oc, y, x] = ReLU(max_{dy,dx in {0,1,2}} conv[n, oc, 2y + dy - 1, 2x + dx - 1] + bias[oc])
      with correct handling of both conv padding (pad=1 -> zero input) and pool padding (pad=1 -> ignore outside).

    Runtime behavior:
    - The wrapper only validates shapes/dtypes, allocates the output tensor, computes launch grid, and invokes the
      Triton kernel. All math (conv, bias, ReLU, pooling) is computed inside the Triton kernel.

    Args:
      x:           Input tensor [N, C_in, H, W], dtype typically torch.bfloat16, device CUDA.
      conv_weight: Weights [C_out, C_in, 3, 3], same dtype/device as x.
      conv_bias:   Bias [C_out], same dtype/device as x.

    Returns:
      Output tensor [N, C_out, H_out_pool, W_out_pool] with dtype = x.dtype and device = x.device.
      For stride=1 pad=1 conv and 3x3 pool stride=2 pad=1, shapes shrink H,W -> ceil(H/2), ceil(W/2).
    """
    # Basic validations
    assert x.is_cuda and conv_weight.is_cuda and conv_bias.is_cuda, "All tensors must be on CUDA"
    assert x.ndim == 4 and conv_weight.ndim == 4 and conv_bias.ndim == 1, "Invalid tensor ranks"
    N, C_in, H, W = x.shape
    C_out, C_in_w, KH, KW = conv_weight.shape
    assert C_in_w == C_in and KH == 3 and KW == 3, "Expected weight shape [C_out, C_in, 3, 3]"
    assert conv_bias.shape[0] == C_out, "Bias must have shape [C_out]"
    assert x.dtype == conv_weight.dtype == conv_bias.dtype, "All tensors must have the same dtype"
    assert x.dtype in (torch.bfloat16, torch.float16, torch.float32), "Supported dtypes: bf16, fp16, fp32"

    # Conv parameters fixed by test scenario
    stride_conv = (1, 1)
    pad_conv = (1, 1)
    dilation_conv = (1, 1)
    assert stride_conv == (1, 1) and pad_conv == (1, 1) and dilation_conv == (1, 1), "Kernel assumes 3x3 s=1 p=1"

    # Derived conv output size (same spatial as input for s=1,p=1,3x3)
    Hc = (H + 2 * pad_conv[0] - dilation_conv[0] * (KH - 1) - 1) // stride_conv[0] + 1
    Wc = (W + 2 * pad_conv[1] - dilation_conv[1] * (KW - 1) - 1) // stride_conv[1] + 1
    assert Hc == H and Wc == W, "For s=1,p=1,3x3 we expect conv output spatial == input spatial"

    # Pool parameters (per test)
    pool_k = (3, 3)
    pool_stride = (2, 2)
    pool_pad = (1, 1)

    # Pooled output size
    PH = (Hc + 2 * pool_pad[0] - pool_k[0]) // pool_stride[0] + 1
    PW = (Wc + 2 * pool_pad[1] - pool_k[1]) // pool_stride[1] + 1

    # Allocate output
    out = torch.empty((N, C_out, PH, PW), device=x.device, dtype=x.dtype)

    # Strides for NCHW tensors
    sxn, sxc, sxh, sxw = x.stride()
    swo, swi, swk, swl = conv_weight.stride()  # [C_out, C_in, 3, 3]
    son, soc, soh, sow = out.stride()

    # Launch grid:
    #  - axis 0: pooled spatial (PH*PW)
    #  - axis 1: N * ceil_div(C_out, BLOCK_OC)
    BLOCK_OC = 32
    grid = (PH * PW, N * triton.cdiv(C_out, BLOCK_OC))

    # Launch kernel
    _fused_conv_relu_pool_kernel_nchw[grid](
        x, conv_weight, conv_bias, out,
        N, C_in, H, W, C_out,
        PH, PW,
        sxn, sxc, sxh, sxw,
        swo, swi, swk, swl,
        son, soc, soh, sow,
        BLOCK_OC=BLOCK_OC,
        num_warps=4,
        num_stages=2,
    )

    return out