import torch
import triton
import triton.language as tl


@triton.jit
def _conv3d_fused_kernel(
    x_ptr,                # *bfloat16 [N, Cin, D, H, W]
    w_ptr,                # *bfloat16 [Cout, Cin, Kd, Kh, Kw]
    cbias_ptr,            # *bfloat16 [Cout]
    scale_ptr,            # *bfloat16 [Cout, 1, 1, 1] (index by channel 0 only)
    bmul_ptr,             # *bfloat16 [Cout, 1, 1, 1] (index by channel 0 only)
    out_ptr,              # *bfloat16 [N, Cout, D_out, H_out, W_out]
    N, Cin, Cout, D, H, W,
    D_out, H_out, W_out,
    stride_xn, stride_xc, stride_xd, stride_xh, stride_xw,
    stride_wn, stride_wc, stride_wd, stride_wh, stride_ww,
    stride_on, stride_oc, stride_od, stride_oh, stride_ow,
    stride_scalec, stride_bmulc,
    BLOCK_W: tl.constexpr,
    K_D: tl.constexpr,
    K_H: tl.constexpr,
    K_W: tl.constexpr,
):
    # Each program handles one (n, co, d_out, h_out) "row" and a tile along W_out
    pid_ncdh = tl.program_id(axis=0)
    pid_wtile = tl.program_id(axis=1)

    # Decode n, co, d_out, h_out from pid_ncdh
    tmp = pid_ncdh
    h_out_idx = tmp % H_out
    tmp = tmp // H_out
    d_out_idx = tmp % D_out
    tmp = tmp // D_out
    co = tmp % Cout
    n = tmp // Cout

    w_block_start = pid_wtile * BLOCK_W
    offs_w = w_block_start + tl.arange(0, BLOCK_W)
    mask_w = offs_w < W_out

    # Accumulator in fp32 over the BLOCK_W vector
    acc = tl.zeros((BLOCK_W,), dtype=tl.float32)

    # Convolution accumulation: sum_{ci, kd, kh, kw} x * w
    # Input indices:
    #   di = d_out_idx + kd
    #   hi = h_out_idx + kh
    #   wi = offs_w + kw
    # Strides are passed in, we just compute pointer offsets.
    for ci in range(0, K_D * K_H * K_W * Cin, 1):
        # We will unroll ci into (c_in, kd, kh, kw) manually to let compiler optimize
        # but Triton requires compile-time loops for best performance. Here, we map linear index to tuple.
        # However, to keep it simple and clear, we explicitly loop the 4 dimensions instead:
        pass  # Placeholder to allow explicit nested loops below

    # Explicit nested loops (preferred for clarity and correctness)
    for c_in in range(0, Cin):
        for kd in range(0, K_D):
            di = d_out_idx + kd
            for kh in range(0, K_H):
                hi = h_out_idx + kh
                for kw in range(0, K_W):
                    wi = offs_w + kw
                    # Compute input pointers vector for this slice
                    x_ptrs = (
                        x_ptr
                        + n * stride_xn
                        + c_in * stride_xc
                        + di * stride_xd
                        + hi * stride_xh
                        + wi * stride_xw
                    )
                    # Load input vector in bf16 and cast to fp32
                    x_vals = tl.load(x_ptrs, mask=mask_w, other=0).to(tl.float32)
                    # Load weight scalar (bf16 -> fp32)
                    w_off = (
                        co * stride_wn
                        + c_in * stride_wc
                        + kd * stride_wd
                        + kh * stride_wh
                        + kw * stride_ww
                    )
                    w_val = tl.load(w_ptr + w_off).to(tl.float32)
                    # FMA
                    acc += x_vals * w_val

    # Add per-output-channel conv bias
    cbias = tl.load(cbias_ptr + co).to(tl.float32)
    acc = acc + cbias

    # Fused post-ops:
    # 1) multiply by scaling_factor (C,1,1,1) -> broadcast by channel
    # 2) tanh
    # 3) multiply by bias_param (C,1,1,1) -> broadcast by channel
    # 4) sigmoid
    scale = tl.load(scale_ptr + co * stride_scalec).to(tl.float32)
    bmul = tl.load(bmul_ptr + co * stride_bmulc).to(tl.float32)

    acc = acc * scale

    # tanh(x) = 2 * sigmoid(2x) - 1; we implement with exp for compatibility
    two = 2.0
    one = 1.0
    acc = two / (one + tl.exp(-two * acc)) - one

    acc = acc * bmul

    # sigmoid
    acc = one / (one + tl.exp(-acc))

    # Store to output (bf16)
    out_ptrs = (
        out_ptr
        + n * stride_on
        + co * stride_oc
        + d_out_idx * stride_od
        + h_out_idx * stride_oh
        + offs_w * stride_ow
    )
    tl.store(out_ptrs, acc.to(tl.bfloat16), mask=mask_w)


def kernel_function(x, weight, conv_bias, scaling_param, bias_param):
    """
    Fused Triton kernel wrapper: Conv3D (k=3,s=1,p=0) -> per-channel scale -> tanh -> per-channel mul -> sigmoid.

    Fusion details (single pass inside the kernel):
    - Computes 3D convolution accumulation entirely in the kernel (BF16 loads, FP32 accumulate).
    - Adds Conv3d bias (per out-channel).
    - Multiplies by scaling_param (shape [Cout, 1, 1, 1], broadcasted per out-channel).
    - Applies tanh (implemented via 2*sigmoid(2x)-1 for libdevice portability).
    - Multiplies by bias_param (shape [Cout, 1, 1, 1], broadcasted per out-channel).
    - Applies final sigmoid.
    - Stores BF16 output.

    Wrapper only does:
    - Argument validation and shape checks
    - Output allocation
    - Grid configuration and kernel launch

    All math runs in the Triton kernel per runtime constraints.
    """
    # Basic validations
    assert x.is_cuda and weight.is_cuda and conv_bias.is_cuda and scaling_param.is_cuda and bias_param.is_cuda, \
        "All tensors must be CUDA tensors."
    assert x.dtype == torch.bfloat16, "Input x must be bfloat16."
    assert weight.dtype == torch.bfloat16, "Weight must be bfloat16."
    assert conv_bias.dtype == torch.bfloat16, "conv_bias must be bfloat16."
    assert scaling_param.dtype == torch.bfloat16, "scaling_param must be bfloat16."
    assert bias_param.dtype == torch.bfloat16, "bias_param must be bfloat16."
    assert x.ndim == 5, "x must be [N, Cin, D, H, W]"
    assert weight.ndim == 5, "weight must be [Cout, Cin, Kd, Kh, Kw]"
    assert conv_bias.ndim == 1, "conv_bias must be [Cout]"
    assert scaling_param.shape[0] == weight.shape[0], "scaling_param first dim must match Cout."
    assert bias_param.shape[0] == weight.shape[0], "bias_param first dim must match Cout."

    N, Cin, D, H, W = x.shape
    Cout, Cin_w, Kd, Kh, Kw = weight.shape
    assert Cin == Cin_w, "in_channels mismatch between x and weight."
    assert (Kd, Kh, Kw) == (3, 3, 3), "Kernel size must be (3,3,3) per requirements."
    D_out = D - Kd + 1
    H_out = H - Kh + 1
    W_out = W - Kw + 1
    assert D_out > 0 and H_out > 0 and W_out > 0, "Invalid spatial sizes for 'valid' conv."

    # Output allocation (BF16)
    out = torch.empty((N, Cout, D_out, H_out, W_out), device=x.device, dtype=torch.bfloat16)

    # Strides (in elements, not bytes)
    stride_xn, stride_xc, stride_xd, stride_xh, stride_xw = x.stride()
    stride_wn, stride_wc, stride_wd, stride_wh, stride_ww = weight.stride()
    stride_on, stride_oc, stride_od, stride_oh, stride_ow = out.stride()

    # Per-channel scale/bias first-dimension strides
    # For shapes (Cout, 1, 1, 1), stride along dim-0 is fine for channel indexing; we only index by channel
    stride_scalec = scaling_param.stride(0)
    stride_bmulc = bias_param.stride(0)

    # Grid
    # axis 0: all (N, Cout, D_out, H_out) rows
    # axis 1: tiles along W_out
    BLOCK_W = 64  # power of 2 for coalesced access; W_out=62 in test -> 1 tile
    grid = (
        N * Cout * D_out * H_out,
        triton.cdiv(W_out, BLOCK_W),
    )

    # Launch kernel
    _conv3d_fused_kernel[grid](
        x, weight, conv_bias, scaling_param, bias_param, out,
        N, Cin, Cout, D, H, W,
        D_out, H_out, W_out,
        stride_xn, stride_xc, stride_xd, stride_xh, stride_xw,
        stride_wn, stride_wc, stride_wd, stride_wh, stride_ww,
        stride_on, stride_oc, stride_od, stride_oh, stride_ow,
        stride_scalec, stride_bmulc,
        BLOCK_W=BLOCK_W,
        K_D=3, K_H=3, K_W=3,
        num_warps=4,
        num_stages=2,
    )
    return out