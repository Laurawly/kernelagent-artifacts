import torch
import triton
import triton.language as tl


@triton.jit
def _bn_relu_conv1x1_avgpool2_kernel(
    x_ptr,                   # *bfloat16/float32 [N, C_in, H, W]
    gamma_ptr, beta_ptr,     # *bfloat16/float32 [C_in]
    mean_ptr, var_ptr,       # *bfloat16/float32 [C_in]
    w_ptr,                   # *bfloat16/float32 [C_out, C_in, 1, 1] (treated as [C_out, C_in])
    y_ptr,                   # *bfloat16/float32 [N, C_out, H//2, W//2]
    N, C_in, H, W, C_out, H_out, W_out,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wco, stride_wci,
    stride_yn, stride_yc, stride_yh, stride_yw,
    eps,  # float32 scalar
    BLOCK_CI: tl.constexpr,
    BLOCK_CO: tl.constexpr,
):
    # Program IDs:
    # axis 0 -> flattened (n, ho, wo), axis 1 -> blocks of output channels
    pid_nhw = tl.program_id(axis=0)
    pid_co_block = tl.program_id(axis=1)

    # Decode pid_nhw into (n, ho, wo)
    hw_out = H_out * W_out
    n = pid_nhw // hw_out
    rem = pid_nhw % hw_out
    ho = rem // W_out
    wo = rem % W_out

    # Output channel block
    co_start = pid_co_block * BLOCK_CO
    offs_co = co_start + tl.arange(0, BLOCK_CO)
    mask_co = offs_co < C_out

    # Prepare spatial indices for 2x2 avgpool window
    h0 = 2 * ho
    h1 = h0 + 1
    w0 = 2 * wo
    w1 = w0 + 1
    h0_in = h0 < H
    h1_in = h1 < H
    w0_in = w0 < W
    w1_in = w1 < W

    # Initialize accumulator for output channels in fp32
    acc = tl.zeros((BLOCK_CO,), dtype=tl.float32)

    # Loop over input channels in tiles of BLOCK_CI
    for ci_start in range(0, C_in, BLOCK_CI):
        offs_ci = ci_start + tl.arange(0, BLOCK_CI)
        mask_ci = offs_ci < C_in

        # Load BN parameters (gamma, beta, mean, var) for this tile, cast to fp32
        gamma = tl.load(gamma_ptr + offs_ci, mask=mask_ci, other=0.0).to(tl.float32)
        beta = tl.load(beta_ptr + offs_ci, mask=mask_ci, other=0.0).to(tl.float32)
        mean = tl.load(mean_ptr + offs_ci, mask=mask_ci, other=0.0).to(tl.float32)
        var = tl.load(var_ptr + offs_ci, mask=mask_ci, other=0.0).to(tl.float32)
        inv_std = 1.0 / tl.sqrt(var + eps)

        # Sum of ReLU(BN(x)) over the 2x2 window for each input channel
        s = tl.zeros((BLOCK_CI,), dtype=tl.float32)

        # Helper to load and accumulate one spatial position
        # pos (h, w)
        # (x - mean) * inv_std * gamma + beta, then ReLU, then add to s
        # We implement inline to ensure JIT friendly code.

        # (h0, w0)
        ptr00 = x_ptr + n * stride_xn + offs_ci * stride_xc + h0 * stride_xh + w0 * stride_xw
        x00 = tl.load(ptr00, mask=mask_ci & h0_in & w0_in, other=0.0).to(tl.float32)
        y00 = ((x00 - mean) * inv_std) * gamma + beta
        y00 = tl.maximum(y00, 0.0)
        s += y00

        # (h0, w1)
        ptr01 = x_ptr + n * stride_xn + offs_ci * stride_xc + h0 * stride_xh + w1 * stride_xw
        x01 = tl.load(ptr01, mask=mask_ci & h0_in & w1_in, other=0.0).to(tl.float32)
        y01 = ((x01 - mean) * inv_std) * gamma + beta
        y01 = tl.maximum(y01, 0.0)
        s += y01

        # (h1, w0)
        ptr10 = x_ptr + n * stride_xn + offs_ci * stride_xc + h1 * stride_xh + w0 * stride_xw
        x10 = tl.load(ptr10, mask=mask_ci & h1_in & w0_in, other=0.0).to(tl.float32)
        y10 = ((x10 - mean) * inv_std) * gamma + beta
        y10 = tl.maximum(y10, 0.0)
        s += y10

        # (h1, w1)
        ptr11 = x_ptr + n * stride_xn + offs_ci * stride_xc + h1 * stride_xh + w1 * stride_xw
        x11 = tl.load(ptr11, mask=mask_ci & h1_in & w1_in, other=0.0).to(tl.float32)
        y11 = ((x11 - mean) * inv_std) * gamma + beta
        y11 = tl.maximum(y11, 0.0)
        s += y11

        # Load weight tile [BLOCK_CO, BLOCK_CI]
        w_ptrs = w_ptr + (offs_co[:, None] * stride_wco + offs_ci[None, :] * stride_wci)
        w_tile = tl.load(w_ptrs, mask=mask_co[:, None] & mask_ci[None, :], other=0.0).to(tl.float32)

        # GEMV: acc += w_tile @ s
        # Using elementwise multiply + reduction along BLOCK_CI
        contrib = tl.sum(w_tile * s[None, :], axis=1)  # shape: [BLOCK_CO]
        acc += contrib

    # Apply average pooling scale (1/4)
    acc = acc * 0.25

    # Store result
    y_ptrs = y_ptr + n * stride_yn + offs_co * stride_yc + ho * stride_yh + wo * stride_yw
    # Cast accumulator to output dtype
    out_dtype = y_ptr.dtype.element_ty
    tl.store(y_ptrs, acc.to(out_dtype), mask=mask_co)


def kernel_function(
    x: torch.Tensor,
    bn_weight: torch.Tensor,
    bn_bias: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    conv_weight: torch.Tensor,
):
    """
    Fused Triton kernel: BatchNorm (eval, running stats) -> ReLU -> Conv2d(1x1, 256->128) -> AvgPool2d(kernel=2, stride=2)

    What is fused:
    - BatchNorm (with running stats) and ReLU are applied to input tiles on-the-fly in the kernel.
    - The 1x1 convolution is computed immediately using the BN-ReLU outputs without writing intermediates to memory.
    - The 2x2 average pooling with stride 2 is folded into the same pass by summing the four BN-ReLU activations
      from the corresponding 2x2 region first, then doing the 1x1 convolution on that sum, and finally multiplying by 1/4.
      Concretely, for each output location (ho, wo) we compute s_ci = sum over the 2x2 window of ReLU(BN(x[n, ci, h, w]))
      and then out[n, co, ho, wo] = 0.25 * sum_ci conv_weight[co, ci] * s_ci.

    This design completely avoids writing intermediate tensors for BN, ReLU, Conv, or Pool, minimizing memory traffic
    and kernel launch overhead.

    Runtime notes:
    - Wrapper performs only validation, allocation, and launch configuration.
    - All math runs in the Triton kernel.
    - Supports float32 and bfloat16 inputs/weights/BN params. Accumulates in float32 and stores to input dtype.
    """
    # Basic checks
    assert x.is_cuda, "x must be on CUDA"
    device = x.device
    assert x.dim() == 4, "x must be NCHW"
    N, C_in, H, W = x.shape

    # BN parameter checks
    for t, name in [(bn_weight, "bn_weight"), (bn_bias, "bn_bias"), (running_mean, "running_mean"), (running_var, "running_var")]:
        assert t.is_cuda, f"{name} must be on CUDA"
        assert t.dim() == 1 and t.shape[0] == C_in, f"{name} must have shape [C_in]"

    # Conv weight checks: [C_out, C_in, 1, 1]
    assert conv_weight.is_cuda, "conv_weight must be on CUDA"
    assert conv_weight.dim() == 4 and conv_weight.shape[2] == 1 and conv_weight.shape[3] == 1, "conv_weight must be [C_out, C_in, 1, 1]"
    C_out = conv_weight.shape[0]
    assert conv_weight.shape[1] == C_in, "conv_weight C_in mismatch"

    # Dtype support: bf16 or fp32
    assert x.dtype in (torch.bfloat16, torch.float32), "Only bfloat16 and float32 inputs are supported"
    # Enforce all params/weights to match x dtype for simplicity; cast lightweight vectors if needed
    if bn_weight.dtype != x.dtype:
        bn_weight = bn_weight.to(dtype=x.dtype)
    if bn_bias.dtype != x.dtype:
        bn_bias = bn_bias.to(dtype=x.dtype)
    if running_mean.dtype != x.dtype:
        running_mean = running_mean.to(dtype=x.dtype)
    if running_var.dtype != x.dtype:
        running_var = running_var.to(dtype=x.dtype)
    if conv_weight.dtype != x.dtype:
        conv_weight = conv_weight.to(dtype=x.dtype)

    # Output allocation [N, C_out, H//2, W//2]
    H_out = (H + 1) // 2  # robust for odd, though test uses even
    W_out = (W + 1) // 2
    y = torch.empty((N, C_out, H_out, W_out), device=device, dtype=x.dtype)

    # Strides
    stride_xn, stride_xc, stride_xh, stride_xw = x.stride()
    # Treat conv_weight as [C_out, C_in]
    stride_wco = conv_weight.stride(0)
    stride_wci = conv_weight.stride(1)
    stride_yn, stride_yc, stride_yh, stride_yw = y.stride()

    # Launch grid
    def grid(meta):
        BLOCK_CO = meta["BLOCK_CO"]
        return (N * H_out * W_out, triton.cdiv(C_out, BLOCK_CO))

    # Reasonable tile sizes for this problem size
    _bn_relu_conv1x1_avgpool2_kernel[grid](
        x,
        bn_weight, bn_bias,
        running_mean, running_var,
        conv_weight,
        y,
        N, C_in, H, W, C_out, H_out, W_out,
        stride_xn, stride_xc, stride_xh, stride_xw,
        stride_wco, stride_wci,
        stride_yn, stride_yc, stride_yh, stride_yw,
        1e-5,  # eps
        BLOCK_CI=64,
        BLOCK_CO=32,
        num_warps=4,
        num_stages=2,
    )

    return y