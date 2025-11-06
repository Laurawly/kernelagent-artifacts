import torch
import triton
import triton.language as tl


@triton.jit
def _copy_front_channels_kernel(
    src_ptr, dst_ptr,
    N, C0, H, W,
    stride_sn, stride_sc, stride_sh, stride_sw,
    stride_dn, stride_dc, stride_dh, stride_dw,
    BLOCK_SIZE: tl.constexpr,
):
    # 1D grid over all elements of [N, C0, H, W]
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    total = N * C0 * H * W
    mask = offsets < total

    w_idx = offsets % W
    tmp0 = offsets // W
    h_idx = tmp0 % H
    tmp1 = tmp0 // H
    c_idx = tmp1 % C0
    n_idx = tmp1 // C0

    w_idx = tl.where(mask, w_idx, 0)
    h_idx = tl.where(mask, h_idx, 0)
    c_idx = tl.where(mask, c_idx, 0)
    n_idx = tl.where(mask, n_idx, 0)

    src_ptrs = src_ptr + n_idx * stride_sn + c_idx * stride_sc + h_idx * stride_sh + w_idx * stride_sw
    dst_ptrs = dst_ptr + n_idx * stride_dn + c_idx * stride_dc + h_idx * stride_dh + w_idx * stride_dw
    vals = tl.load(src_ptrs, mask=mask, other=0)
    tl.store(dst_ptrs, vals, mask=mask)


@triton.jit
def _bn_stats_kernel(
    x_ptr, mean_ptr, var_ptr,
    N, C, H, W,
    stride_n, stride_c, stride_h, stride_w,
    BLOCK_ELEMS: tl.constexpr,
):
    """
    Compute per-channel mean and variance across N*H*W in fp32 (training stats).
    unbiased=False => var = E[x^2] - mean^2
    """
    c = tl.program_id(axis=0)
    if c >= C:
        return

    total = N * H * W
    sum_val = tl.zeros((), dtype=tl.float32)
    sum_sq = tl.zeros((), dtype=tl.float32)

    for start in tl.range(0, total, BLOCK_ELEMS, num_stages=1):
        offs = start + tl.arange(0, BLOCK_ELEMS)
        m = offs < total

        w_idx = offs % W
        tmp0 = offs // W
        h_idx = tmp0 % H
        n_idx = tmp0 // H

        w_idx = tl.where(m, w_idx, 0)
        h_idx = tl.where(m, h_idx, 0)
        n_idx = tl.where(m, n_idx, 0)

        x_ptrs = x_ptr + n_idx * stride_n + c * stride_c + h_idx * stride_h + w_idx * stride_w
        x_bf = tl.load(x_ptrs, mask=m, other=0)
        x = x_bf.to(tl.float32)

        sum_val += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)

    denom = tl.full((), total, dtype=tl.float32)
    mean = sum_val / denom
    ex2 = sum_sq / denom
    var = ex2 - mean * mean
    var = tl.maximum(var, 0.0)

    tl.store(mean_ptr + c, mean)
    tl.store(var_ptr + c, var)


@triton.jit
def _bn_relu_conv3x3_kernel(
    x_ptr,        # input/output tensor (we read first C_IN channels, write new channels)
    out_ptr,      # same tensor; write into [C_OUT_START : C_OUT_START + GROWTH)
    gamma_ptr, beta_ptr,   # [C_IN]
    mean_ptr, var_ptr,     # [C_IN], fp32
    w_ptr,                 # weights [GROWTH, C_IN, 3, 3]
    N, C_IN, H, W,
    C_OUT_START, GROWTH,
    eps,                   # fp32
    # strides for input/output tensor (NCHW)
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_on, stride_oc, stride_oh, stride_ow,
    # weight strides (OIHW)
    stride_wo, stride_wi, stride_wh, stride_ww,
    OUT_DTYPE_CODE: tl.constexpr,  # 0=fp16, 1=bf16, 2=fp32
    BLOCK_HW: tl.constexpr,
):
    """
    Fused BN + ReLU + Conv2D(3x3, stride=1, padding=1, no bias) producing 'GROWTH' channels.
    Grid:
      axis0 -> tiles over H*W
      axis1 -> batch N
      axis2 -> output channel within growth (0..GROWTH-1)
    BN computed using provided per-channel mean/var (training).
    ReLU is applied after casting BN output to request dtype to match PyTorch semantics.
    Padding: out-of-bounds contributions are zeroed BEFORE BN/ReLU (correct padding semantics).
    """
    pid_hw = tl.program_id(axis=0)
    n = tl.program_id(axis=1)
    oc = tl.program_id(axis=2)  # 0..GROWTH-1

    start_hw = pid_hw * BLOCK_HW
    offs = start_hw + tl.arange(0, BLOCK_HW)
    mask_hw = offs < (H * W)

    h_idx = offs // W
    w_idx = offs % W

    acc = tl.zeros((BLOCK_HW,), dtype=tl.float32)

    # iterate over input channels and 3x3 kernel
    for ic in tl.range(0, C_IN):
        # Preload BN params for this ic in fp32
        gamma = tl.load(gamma_ptr + ic).to(tl.float32)
        beta = tl.load(beta_ptr + ic).to(tl.float32)
        mean = tl.load(mean_ptr + ic)  # float32
        var = tl.load(var_ptr + ic)    # float32
        inv_std = 1.0 / tl.sqrt(var + eps)
        scale = gamma * inv_std
        shift = beta - mean * scale

        base_x = x_ptr + n * stride_xn + ic * stride_xc

        for kh in range(3):
            for kw in range(3):
                dh = kh - 1
                dw = kw - 1
                ih = h_idx + dh
                iw = w_idx + dw

                in_bounds = mask_hw & (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
                ih_safe = tl.where(in_bounds, ih, 0)
                iw_safe = tl.where(in_bounds, iw, 0)

                # load activation; out-of-bounds is ignored later (zero contribution)
                x_ptrs = base_x + ih_safe * stride_xh + iw_safe * stride_xw
                x_vals_bf = tl.load(x_ptrs, mask=in_bounds, other=0)
                x_vals = x_vals_bf.to(tl.float32)

                # BN in fp32
                y32 = x_vals * scale + shift

                # Cast-before-ReLU to match PyTorch semantics, then cast back to fp32 for accumulation
                if OUT_DTYPE_CODE == 1:  # bf16
                    y_cast = y32.to(tl.bfloat16)
                    zero_cast = tl.zeros_like(y_cast)
                    y_act = tl.maximum(y_cast, zero_cast).to(tl.float32)
                elif OUT_DTYPE_CODE == 0:  # fp16
                    y_cast = y32.to(tl.float16)
                    zero_cast = tl.zeros_like(y_cast)
                    y_act = tl.maximum(y_cast, zero_cast).to(tl.float32)
                else:  # fp32
                    y_act = tl.maximum(y32, 0.0)

                # IMPORTANT: padding semantics -> zero-out contributions that come from out-of-bounds
                y_act = tl.where(in_bounds, y_act, 0.0)

                w_val = tl.load(w_ptr + oc * stride_wo + ic * stride_wi + kh * stride_wh + kw * stride_ww).to(tl.float32)
                acc += y_act * w_val

    out_ch = C_OUT_START + oc
    out_ptrs = out_ptr + n * stride_on + out_ch * stride_oc + h_idx * stride_oh + w_idx * stride_ow
    if OUT_DTYPE_CODE == 1:
        out_vals = acc.to(tl.bfloat16)
    elif OUT_DTYPE_CODE == 0:
        out_vals = acc.to(tl.float16)
    else:
        out_vals = acc
    tl.store(out_ptrs, out_vals, mask=mask_hw)


def kernel_function(x, bn_weights, bn_biases, conv_weights, p=0.0, eps=1e-5):
    """
    DenseBlock: 6 layers, NCHW
    Per-layer: BN(train stats over current channels) -> ReLU -> Conv3x3(stride=1,pad=1) -> Dropout(p) -> concat
    All math executed in Triton kernels.
    We fuse BN+ReLU+Conv per layer. BN statistics require a reduction over N*H*W,
    which is done in a separate kernel since it cannot be computed on-the-fly per 3x3 sample without two passes.
    """
    # Basic validation
    assert isinstance(bn_weights, (list, tuple)) and isinstance(bn_biases, (list, tuple)) and isinstance(conv_weights, (list, tuple))
    layers = len(conv_weights)
    assert layers == 6, "This implementation assumes 6 layers"
    assert len(bn_weights) == layers and len(bn_biases) == layers

    assert x.is_cuda, "Input must be CUDA tensor"
    assert x.ndim == 4
    N, C0, H, W = x.shape
    device = x.device
    dtype = x.dtype
    assert dtype in (torch.bfloat16, torch.float16, torch.float32)

    # Determine dtype code for kernels
    if dtype == torch.bfloat16:
        OUT_DTYPE_CODE = 1
    elif dtype == torch.float16:
        OUT_DTYPE_CODE = 0
    else:
        OUT_DTYPE_CODE = 2

    # Validate parameter shapes and derive in_channels per layer
    growth_rates = []
    in_channels_per_layer = []
    for i in range(layers):
        w = conv_weights[i]
        assert w.is_cuda and w.ndim == 4 and w.shape[2] == 3 and w.shape[3] == 3, f"conv_weights[{i}] must be [growth, in_ch, 3, 3]"
        gr = w.shape[0]
        inferred_in = C0 + i * gr
        assert w.shape[1] == inferred_in, f"conv_weights[{i}].shape[1]={w.shape[1]} != expected {inferred_in}"
        assert bn_weights[i].numel() == inferred_in, f"bn_weights[{i}] size mismatch"
        assert bn_biases[i].numel() == inferred_in, f"bn_biases[{i}] size mismatch"
        growth_rates.append(gr)
        in_channels_per_layer.append(inferred_in)

    # Allocate final output tensor with all concatenated features
    C_total = C0 + sum(growth_rates)
    out = torch.empty((N, C_total, H, W), device=device, dtype=dtype)

    # Step 0: copy original input into the front of 'out'
    BLOCK_COPY = 1024
    grid_copy = (triton.cdiv(N * C0 * H * W, BLOCK_COPY),)
    _copy_front_channels_kernel[grid_copy](
        x, out,
        N, C0, H, W,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        BLOCK_SIZE=BLOCK_COPY,
        num_warps=4,
        num_stages=2,
    )

    # Sequential dense layers
    c_out_start = C0
    for i in range(layers):
        w = conv_weights[i]
        gamma = bn_weights[i]
        beta = bn_biases[i]
        C_in = in_channels_per_layer[i]
        growth = growth_rates[i]

        # 1) BN stats for current 'C_in' channels at the front of 'out'
        mean = torch.empty((C_in,), device=device, dtype=torch.float32)
        var = torch.empty((C_in,), device=device, dtype=torch.float32)

        BLOCK_ELEMS = 256
        grid_stats = (C_in,)
        _bn_stats_kernel[grid_stats](
            out, mean, var,
            N, C_in, H, W,
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            BLOCK_ELEMS=BLOCK_ELEMS,
            num_warps=4,
            num_stages=2,
        )

        # 2) Fused BN + ReLU + Conv3x3 -> write next 'growth' channels into 'out'
        BLOCK_HW = 256
        grid_hw = triton.cdiv(H * W, BLOCK_HW)
        grid = (grid_hw, N, growth)
        _bn_relu_conv3x3_kernel[grid](
            out, out,
            gamma, beta,
            mean, var,
            w,
            N, C_in, H, W,
            c_out_start, growth,
            float(eps),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            w.stride(0), w.stride(1), w.stride(2), w.stride(3),
            OUT_DTYPE_CODE=OUT_DTYPE_CODE,
            BLOCK_HW=BLOCK_HW,
            num_warps=4,
            num_stages=2,
        )

        # New features are now part of the "current" front section for the next layer
        c_out_start += growth

    return out