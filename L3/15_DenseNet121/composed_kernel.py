import torch
import triton
import triton.language as tl
import math
import sys

# =========================
# Stem: Conv7x7 s=2 p=3 + BN(eval) + ReLU -> MaxPool3x3 s=2 p=1
# =========================

@triton.jit
def _conv_bn_relu_7x7s2_kernel(
    x_ptr,                # *bfloat16 or *float16 or *float32, NCHW input
    w_ptr,                # *bfloat16 or *float16 or *float32, OIHW weights
    gamma_ptr, beta_ptr,  # *bfloat16/float16/float32, BN weight/bias
    mean_ptr, var_ptr,    # *bfloat16/float16/float32, BN running stats
    y_ptr,                # *float32, output activation after BN+ReLU
    N, C_in, H, W,        # input dims
    C_out, H_out, W_out,  # output dims (112, 112)
    stride_in_n, stride_in_c, stride_in_h, stride_in_w,         # input strides
    stride_w_o, stride_w_i, stride_w_h, stride_w_w,             # weight strides
    stride_y_n, stride_y_c, stride_y_h, stride_y_w,             # output (activation) strides
    EPS: tl.constexpr,                                          # BN eps
    BLOCK_CO: tl.constexpr,                                     # out-channels per program
    BLOCK_HW: tl.constexpr,                                     # HW-out elements per program
):
    pid_hw = tl.program_id(axis=0)  # tile id across flattened H_out*W_out
    pid_n  = tl.program_id(axis=1)  # batch
    pid_co = tl.program_id(axis=2)  # out-channel tile

    co_start = pid_co * BLOCK_CO
    co_offsets = co_start + tl.arange(0, BLOCK_CO)
    mask_co = co_offsets < C_out

    hw_start = pid_hw * BLOCK_HW
    hw_offsets = hw_start + tl.arange(0, BLOCK_HW)
    mask_hw = hw_offsets < (H_out * W_out)

    ow = hw_offsets % W_out
    oh = hw_offsets // W_out

    in_y_base = oh * 2 - 3
    in_x_base = ow * 2 - 3

    acc = tl.zeros((BLOCK_CO, BLOCK_HW), dtype=tl.float32)
    x_batch_ptr = x_ptr + pid_n * stride_in_n

    for ci in range(0, C_in):
        x_c_ptr = x_batch_ptr + ci * stride_in_c
        for ky in range(0, 7):
            y_coords = in_y_base + ky
            y_inbounds = (y_coords >= 0) & (y_coords < H) & mask_hw
            y_offsets = y_coords * stride_in_h
            for kx in range(0, 7):
                x_coords = in_x_base + kx
                in_bounds = y_inbounds & (x_coords >= 0) & (x_coords < W)
                x_offsets = x_coords * stride_in_w
                x_ptrs = x_c_ptr + y_offsets + x_offsets
                x_vals = tl.load(x_ptrs, mask=in_bounds, other=0.0).to(tl.float32)

                w_ptrs = w_ptr + co_offsets * stride_w_o + ci * stride_w_i + ky * stride_w_h + kx * stride_w_w
                w_vals = tl.load(w_ptrs, mask=mask_co, other=0.0).to(tl.float32)

                acc += w_vals[:, None] * x_vals[None, :]

    gamma = tl.load(gamma_ptr + co_offsets, mask=mask_co, other=0.0).to(tl.float32)
    beta  = tl.load(beta_ptr  + co_offsets, mask=mask_co, other=0.0).to(tl.float32)
    mean  = tl.load(mean_ptr  + co_offsets, mask=mask_co, other=0.0).to(tl.float32)
    var   = tl.load(var_ptr   + co_offsets, mask=mask_co, other=0.0).to(tl.float32)

    inv_std = 1.0 / tl.sqrt(var + EPS)

    acc = (acc - mean[:, None]) * inv_std[:, None]
    acc = acc * gamma[:, None] + beta[:, None]
    acc = tl.maximum(acc, 0.0)

    y_base_ptr = y_ptr + pid_n * stride_y_n
    y_ptrs = y_base_ptr + co_offsets[:, None] * stride_y_c + oh[None, :] * stride_y_h + ow[None, :] * stride_y_w
    tl.store(y_ptrs, acc, mask=(mask_co[:, None] & mask_hw[None, :]))


@triton.jit
def _maxpool3x3_s2_p1_kernel(
    a_ptr,                 # *float32 activation after BN+ReLU (N, C, 112, 112)
    o_ptr,                 # *float32 output (N, C, 56, 56)
    N, C, H_in, W_in,      # activation dims (112, 112)
    H_out, W_out,          # pooled dims (56, 56)
    stride_a_n, stride_a_c, stride_a_h, stride_a_w,   # activation strides
    stride_o_n, stride_o_c, stride_o_h, stride_o_w,   # output strides
    BLOCK_CO: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    pid_hw = tl.program_id(axis=0)
    pid_n  = tl.program_id(axis=1)
    pid_c  = tl.program_id(axis=2)

    co_start = pid_c * BLOCK_CO
    co_offsets = co_start + tl.arange(0, BLOCK_CO)
    mask_c = co_offsets < C

    hw_start = pid_hw * BLOCK_HW
    hw_offsets = hw_start + tl.arange(0, BLOCK_HW)
    mask_hw = hw_offsets < (H_out * W_out)

    ow = hw_offsets % W_out
    oh = hw_offsets // W_out

    y0 = oh * 2 - 1
    x0 = ow * 2 - 1

    neg_inf = -float("inf")
    acc = tl.zeros((BLOCK_CO, BLOCK_HW), dtype=tl.float32) + neg_inf

    a_base_ptr = a_ptr + pid_n * stride_a_n

    for dy in range(0, 3):
        y = y0 + dy
        y_valid = (y >= 0) & (y < H_in) & mask_hw
        y_offs = y * stride_a_h
        for dx in range(0, 3):
            x = x0 + dx
            valid = y_valid & (x >= 0) & (x < W_in)
            x_offs = x * stride_a_w
            a_ptrs = a_base_ptr + co_offsets[:, None] * stride_a_c + y_offs[None, :] + x_offs[None, :]
            vals = tl.load(a_ptrs, mask=(mask_c[:, None] & valid[None, :]), other=neg_inf)
            acc = tl.maximum(acc, vals)

    o_base_ptr = o_ptr + pid_n * stride_o_n
    o_ptrs = o_base_ptr + co_offsets[:, None] * stride_o_c + oh[None, :] * stride_o_h + ow[None, :] * stride_o_w
    tl.store(o_ptrs, acc, mask=(mask_c[:, None] & mask_hw[None, :]))


def _stem_forward(x, conv_weight, bn_weight, bn_bias, bn_running_mean, bn_running_var):
    assert x.is_cuda
    device = x.device
    N, C_in, H, W = x.shape
    C_out = conv_weight.shape[0]
    pad = 3
    stride = 2
    KH = KW = 7
    H_out = (H + 2*pad - KH)//stride + 1
    W_out = (W + 2*pad - KW)//stride + 1
    assert H_out == 112 and W_out == 112
    pool_k = 3
    pool_stride = 2
    pool_pad = 1
    HP = (H_out + 2*pool_pad - pool_k)//pool_stride + 1
    WP = (W_out + 2*pool_pad - pool_k)//pool_stride + 1
    assert HP == 56 and WP == 56

    act = torch.empty((N, C_out, H_out, W_out), device=device, dtype=torch.float32)
    out = torch.empty((N, C_out, HP, WP), device=device, dtype=torch.float32)

    s_in_n, s_in_c, s_in_h, s_in_w = x.stride()
    s_w_o, s_w_i, s_w_h, s_w_w = conv_weight.stride()
    s_y_n, s_y_c, s_y_h, s_y_w = act.stride()
    s_a_n, s_a_c, s_a_h, s_a_w = act.stride()
    s_o_n, s_o_c, s_o_h, s_o_w = out.stride()

    BLOCK_CO = 32
    BLOCK_HW = 64
    grid_conv = (triton.cdiv(H_out*W_out, BLOCK_HW), N, triton.cdiv(C_out, BLOCK_CO))
    _conv_bn_relu_7x7s2_kernel[grid_conv](
        x, conv_weight,
        bn_weight, bn_bias, bn_running_mean, bn_running_var,
        act,
        N, C_in, H, W,
        C_out, H_out, W_out,
        s_in_n, s_in_c, s_in_h, s_in_w,
        s_w_o, s_w_i, s_w_h, s_w_w,
        s_y_n, s_y_c, s_y_h, s_y_w,
        EPS=1e-5, BLOCK_CO=BLOCK_CO, BLOCK_HW=BLOCK_HW,
        num_warps=4, num_stages=2
    )
    grid_pool = (triton.cdiv(HP*WP, BLOCK_HW), N, triton.cdiv(C_out, BLOCK_CO))
    _maxpool3x3_s2_p1_kernel[grid_pool](
        act, out,
        N, C_out, H_out, W_out,
        HP, WP,
        s_a_n, s_a_c, s_a_h, s_a_w,
        s_o_n, s_o_c, s_o_h, s_o_w,
        BLOCK_CO=BLOCK_CO, BLOCK_HW=BLOCK_HW,
        num_warps=4, num_stages=2
    )
    return out

# =========================
# DenseBlock (inference BN) generic fused BN+ReLU+Conv3x3 per layer
# =========================

@triton.jit
def _copy_nchw_kernel(
    src_ptr, dst_ptr,
    N, C, H, W,
    stride_sn, stride_sc, stride_sh, stride_sw,
    stride_dn, stride_dc, stride_dh, stride_dw,
    c_dst_offset,
    BLOCK_C: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    pid_n = tl.program_id(axis=0)
    pid_c = tl.program_id(axis=1)
    pid_hw = tl.program_id(axis=2)

    c_start = pid_c * BLOCK_C
    hw_start = pid_hw * BLOCK_HW

    offs_c = c_start + tl.arange(0, BLOCK_C)
    offs_hw = hw_start + tl.arange(0, BLOCK_HW)
    mask_c = offs_c < C
    mask_hw = offs_hw < (H * W)

    oy = offs_hw // W
    ox = offs_hw % W

    src_ptrs = src_ptr + pid_n * stride_sn + offs_c[:, None] * stride_sc + oy[None, :] * stride_sh + ox[None, :] * stride_sw
    dst_ptrs = dst_ptr + pid_n * stride_dn + (offs_c + c_dst_offset)[:, None] * stride_dc + oy[None, :] * stride_dh + ox[None, :] * stride_dw

    m = mask_c[:, None] & mask_hw[None, :]
    vals = tl.load(src_ptrs, mask=m, other=0.0)
    tl.store(dst_ptrs, vals, mask=m)


@triton.jit
def _dense_layer_kernel(x_ptr,                # float32 [N, C_total, H, W]
                        y_ptr,                # float32 [N, C_total, H, W] base
                        w_ptr,                # weights [C_OUT, C_IN, 3, 3]
                        gamma_ptr, beta_ptr, mean_ptr, var_ptr,  # BN params [C_IN]
                        N, C_IN, C_BASE, H, W,
                        stride_x_n, stride_x_c, stride_x_h, stride_x_w,
                        stride_y_n, stride_y_c, stride_y_h, stride_y_w,
                        stride_w_oc, stride_w_ic, stride_w_kh, stride_w_kw,
                        eps,
                        BLOCK_HW: tl.constexpr,
                        BLOCK_OC: tl.constexpr,
                        BLOCK_C: tl.constexpr,
                        C_OUT: tl.constexpr):
    pid_n = tl.program_id(axis=0)
    pid_oc = tl.program_id(axis=1)
    pid_hw = tl.program_id(axis=2)

    M = H * W
    start_hw = pid_hw * BLOCK_HW
    offs_hw = start_hw + tl.arange(0, BLOCK_HW)
    mask_hw = offs_hw < M

    W_i = W
    h = offs_hw // W_i
    w = offs_hw % W_i

    oc_start = pid_oc * BLOCK_OC
    offs_oc = oc_start + tl.arange(0, BLOCK_OC)
    oc_mask = offs_oc < C_OUT

    base_x_n = x_ptr + pid_n * stride_x_n
    base_y_n = y_ptr + pid_n * stride_y_n

    acc = tl.zeros((BLOCK_OC, BLOCK_HW), dtype=tl.float32)

    for ci0 in range(0, C_IN, BLOCK_C):
        offs_ci = ci0 + tl.arange(0, BLOCK_C)
        ci_mask = offs_ci < C_IN

        gamma = tl.load(gamma_ptr + offs_ci, mask=ci_mask, other=0.0).to(tl.float32)
        beta = tl.load(beta_ptr + offs_ci, mask=ci_mask, other=0.0).to(tl.float32)
        run_mean = tl.load(mean_ptr + offs_ci, mask=ci_mask, other=0.0).to(tl.float32)
        run_var = tl.load(var_ptr + offs_ci, mask=ci_mask, other=0.0).to(tl.float32)
        inv_std = 1.0 / tl.sqrt(run_var + eps)
        scale = gamma * inv_std
        shift = beta - run_mean * scale

        for kh in range(0, 3):
            for kw in range(0, 3):
                in_h = h + (kh - 1)
                in_w = w + (kw - 1)
                in_bounds = mask_hw & (in_h >= 0) & (in_h < H) & (in_w >= 0) & (in_w < W)

                x_ptrs = base_x_n + offs_ci[:, None] * stride_x_c + in_h[None, :] * stride_x_h + in_w[None, :] * stride_x_w
                x_mask = ci_mask[:, None] & in_bounds[None, :]
                x_vals = tl.load(x_ptrs, mask=x_mask, other=0.0).to(tl.float32)

                y_vals = scale[:, None] * x_vals + shift[:, None]
                y_vals = tl.maximum(y_vals, 0.0)
                y_vals = tl.where(in_bounds[None, :], y_vals, 0.0)
                y_vals = tl.where(ci_mask[:, None], y_vals, 0.0)

                w_ptrs = w_ptr + offs_oc[:, None] * stride_w_oc + offs_ci[None, :] * stride_w_ic + kh * stride_w_kh + kw * stride_w_kw
                w_mask = oc_mask[:, None] & ci_mask[None, :]
                w_vals = tl.load(w_ptrs, mask=w_mask, other=0.0).to(tl.float32)

                # acc += w_vals @ y_vals
                acc += tl.sum(w_vals[:, :, None] * y_vals[None, :, :], axis=1)

    y_ptrs = base_y_n + (C_BASE + offs_oc)[:, None] * stride_y_c + h[None, :] * stride_y_h + w[None, :] * stride_y_w
    store_mask = oc_mask[:, None] & mask_hw[None, :]
    tl.store(y_ptrs, acc, mask=store_mask)


def _denseblock_infer(x, conv_weights, bn_weight, bn_bias, bn_running_mean, bn_running_var, eps=1e-5):
    """
    Generic DenseBlock (eval BN):
      Repeats L = len(conv_weights) times:
        y = Conv3x3(ReLU(BN(x_current)))
        x_current = concat(x_current, y)
    All BN uses running stats.
    """
    assert isinstance(conv_weights, (list, tuple))
    layers = len(conv_weights)
    assert layers == len(bn_weight) == len(bn_bias) == len(bn_running_mean) == len(bn_running_var)
    assert x.is_cuda
    N, C0, H, W = x.shape
    device = x.device
    dtype = torch.float32  # keep fp32 all along for numerical stability

    growth = conv_weights[0].shape[0]
    for i in range(layers):
        C_in = C0 + i * growth
        w = conv_weights[i]
        assert w.shape == (growth, C_in, 3, 3)
        assert bn_weight[i].shape == (C_in,)
        assert bn_bias[i].shape == (C_in,)
        assert bn_running_mean[i].shape == (C_in,)
        assert bn_running_var[i].shape == (C_in,)

    C_total = C0 + growth * layers
    out = torch.empty((N, C_total, H, W), device=device, dtype=dtype)

    # copy input into out[:, :C0]
    BLOCK_C = 64
    BLOCK_HW = 64
    grid_copy = (N, triton.cdiv(C0, BLOCK_C), triton.cdiv(H*W, BLOCK_HW))
    _copy_nchw_kernel[grid_copy](
        x, out,
        N, C0, H, W,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        0,
        BLOCK_C=BLOCK_C, BLOCK_HW=BLOCK_HW,
        num_warps=4
    )

    # Per-layer fused BN+ReLU+Conv3x3 writing at offset C_in
    BLOCK_HW = 64
    BLOCK_OC = growth
    BLOCK_CIN = 64
    for i in range(layers):
        C_in = C0 + i * growth
        w = conv_weights[i]
        gamma = bn_weight[i]
        beta = bn_bias[i]
        rm = bn_running_mean[i]
        rv = bn_running_var[i]

        grid = (N, triton.cdiv(BLOCK_OC, BLOCK_OC), triton.cdiv(H*W, BLOCK_HW))
        _dense_layer_kernel[grid](
            out, out, w, gamma, beta, rm, rv,
            N, C_in, C_in, H, W,
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            w.stride(0), w.stride(1), w.stride(2), w.stride(3),
            eps,
            BLOCK_HW=BLOCK_HW, BLOCK_OC=BLOCK_OC, BLOCK_C=BLOCK_CIN, C_OUT=growth,
            num_warps=4
        )
    return out

# =========================
# Transition: BN(eval) + ReLU + Conv1x1 + AvgPool2d(2x2, s=2)
# =========================

@triton.jit
def _fused_bn_relu_conv1x1_avgpool2_kernel(
    x_ptr,                            # *T [N, C_in, H, W]
    gamma_ptr, beta_ptr,              # *T [C_in], [C_in]
    running_mean_ptr, running_var_ptr,# *T [C_in], [C_in]
    w_ptr,                            # *T [C_out, C_in, 1, 1] (OIHW)
    out_ptr,                          # *f32 [N, C_out, H_out, W_out]

    # sizes
    N, C_IN, H, W, C_OUT, H_OUT, W_OUT,

    # strides for x (NCHW)
    stride_xn, stride_xc, stride_xh, stride_xw,
    # strides for w (OIHW) - only first two matter for 1x1
    stride_wco, stride_wci,
    # strides for out (NCHW)
    stride_on, stride_oc, stride_oh, stride_ow,

    EPS: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_CO: tl.constexpr,
    NUM_CO_TILES: tl.constexpr
):
    pid_w = tl.program_id(axis=0)  # wo
    pid_h = tl.program_id(axis=1)  # ho
    pid_n = tl.program_id(axis=2)  # n

    ih0 = pid_h * 2
    iw0 = pid_w * 2

    x_n_off = pid_n * stride_xn
    out_base_off = pid_n * stride_on + pid_h * stride_oh + pid_w * stride_ow

    num_k_tiles = tl.cdiv(C_IN, BLOCK_K)

    for tile_idx in range(NUM_CO_TILES):
        co_start = tile_idx * BLOCK_CO
        offs_co = co_start + tl.arange(0, BLOCK_CO)
        mask_co = offs_co < C_OUT

        acc = tl.zeros((BLOCK_CO,), dtype=tl.float32)

        for kt in range(num_k_tiles):
            k0 = kt * BLOCK_K
            offs_k = k0 + tl.arange(0, BLOCK_K)
            mask_k = offs_k < C_IN

            gamma = tl.load(gamma_ptr + offs_k, mask=mask_k, other=0).to(tl.float32)
            beta = tl.load(beta_ptr + offs_k, mask=mask_k, other=0).to(tl.float32)
            rm = tl.load(running_mean_ptr + offs_k, mask=mask_k, other=0).to(tl.float32)
            rv = tl.load(running_var_ptr + offs_k, mask=mask_k, other=0).to(tl.float32)
            scale = gamma * tl.rsqrt(rv + EPS)
            bias = beta - rm * scale

            base_c_offs = offs_k * stride_xc
            ptr00 = x_ptr + x_n_off + base_c_offs + ih0 * stride_xh + iw0 * stride_xw
            ptr01 = ptr00 + 1 * stride_xw
            ptr10 = ptr00 + 1 * stride_xh
            ptr11 = ptr10 + 1 * stride_xw

            x00 = tl.load(ptr00, mask=mask_k, other=0).to(tl.float32)
            x01 = tl.load(ptr01, mask=mask_k, other=0).to(tl.float32)
            x10 = tl.load(ptr10, mask=mask_k, other=0).to(tl.float32)
            x11 = tl.load(ptr11, mask=mask_k, other=0).to(tl.float32)

            y00 = tl.maximum(x00 * scale + bias, 0.0)
            y01 = tl.maximum(x01 * scale + bias, 0.0)
            y10 = tl.maximum(x10 * scale + bias, 0.0)
            y11 = tl.maximum(x11 * scale + bias, 0.0)

            s_k = (y00 + y01 + y10 + y11) * 0.25

            w_ptrs = w_ptr + (offs_co[None, :] * stride_wco + offs_k[:, None] * stride_wci)
            w_blk = tl.load(w_ptrs, mask=(mask_k[:, None] & mask_co[None, :]), other=0).to(tl.float32)

            acc += tl.sum(w_blk * s_k[:, None], axis=0)

        out_ptrs = out_ptr + out_base_off + offs_co * stride_oc
        tl.store(out_ptrs, acc, mask=mask_co)


def _transition_forward(x, bn_weight, bn_bias, bn_running_mean, bn_running_var, conv_weight):
    assert x.is_cuda
    device = x.device
    N, C_in, H, W = x.shape
    assert H % 2 == 0 and W % 2 == 0
    C_out = conv_weight.shape[0]
    H_out = H // 2
    W_out = W // 2
    out = torch.empty((N, C_out, H_out, W_out), device=device, dtype=torch.float32)

    BLOCK_K = 64
    BLOCK_CO = 64
    NUM_CO_TILES = (C_out + BLOCK_CO - 1) // BLOCK_CO
    grid = (W_out, H_out, N)
    _fused_bn_relu_conv1x1_avgpool2_kernel[grid](
        x, bn_weight, bn_bias, bn_running_mean, bn_running_var, conv_weight, out,
        N, C_in, H, W, C_out, H_out, W_out,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        conv_weight.stride(0), conv_weight.stride(1),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        EPS=1e-5, BLOCK_K=BLOCK_K, BLOCK_CO=BLOCK_CO, NUM_CO_TILES=NUM_CO_TILES,
        num_warps=4, num_stages=2
    )
    return out

# =========================
# Head: BN(eval) + ReLU + AdaptiveAvgPool2d(1,1) + Linear
# =========================

@triton.jit
def _bn_relu_gap_kernel(
    x_ptr,         # *T [N, C, H, W]
    gamma_ptr, beta_ptr, mean_ptr, var_ptr,  # *T [C]
    y_ptr,         # *f32 [N, C]
    N, C, H: tl.constexpr, W: tl.constexpr,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_yn, stride_yc,
    EPS: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_c_blk = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    c_start = pid_c_blk * BLOCK_C
    offs_c = c_start + tl.arange(0, BLOCK_C)
    mask_c = offs_c < C

    # Load BN params
    gamma = tl.load(gamma_ptr + offs_c, mask=mask_c, other=0.0).to(tl.float32)
    beta  = tl.load(beta_ptr  + offs_c, mask=mask_c, other=0.0).to(tl.float32)
    mean  = tl.load(mean_ptr  + offs_c, mask=mask_c, other=0.0).to(tl.float32)
    var   = tl.load(var_ptr   + offs_c, mask=mask_c, other=0.0).to(tl.float32)
    inv_std = 1.0 / tl.sqrt(var + EPS)
    scale = gamma * inv_std
    shift = beta - mean * scale

    acc = tl.zeros((BLOCK_C,), dtype=tl.float32)

    # Iterate over all HxW positions (H and W small, e.g., 7)
    for h in range(0, H):
        for w in range(0, W):
            x_ptrs = x_ptr + pid_n * stride_xn + offs_c * stride_xc + h * stride_xh + w * stride_xw
            x = tl.load(x_ptrs, mask=mask_c, other=0.0).to(tl.float32)
            y = x * scale + shift
            y = tl.maximum(y, 0.0)
            acc += y

    denom = 1.0 / (H * W)
    acc = acc * denom

    y_ptrs = y_ptr + pid_n * stride_yn + offs_c * stride_yc
    tl.store(y_ptrs, acc, mask=mask_c)


@triton.jit
def _linear_gemm_kernel(
    x_ptr,     # *f32 [N, K]
    w_ptr,     # *f32 [M, K] (OI)
    b_ptr,     # *f32 [M]
    y_ptr,     # *f32 [N, M]
    N, M, K,
    stride_xn, stride_xk,
    stride_wm, stride_wk,
    stride_yn, stride_ym,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)  # output channel blocks
    pid_n = tl.program_id(axis=1)  # batch index

    m_start = pid_m * BLOCK_M
    offs_m = m_start + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M

    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
    # initialize with bias
    b_vals = tl.load(b_ptr + offs_m, mask=mask_m, other=0.0).to(tl.float32)
    acc += b_vals

    # Reduction along K
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        # Load input row x[n, k]
        x_ptrs = x_ptr + pid_n * stride_xn + offs_k * stride_xk
        x_vec = tl.load(x_ptrs, mask=mask_k, other=0.0).to(tl.float32)

        # Load weights w[m, k] tile
        w_ptrs = w_ptr + offs_m[:, None] * stride_wm + offs_k[None, :] * stride_wk
        w_tile = tl.load(w_ptrs, mask=(mask_m[:, None] & mask_k[None, :]), other=0.0).to(tl.float32)

        # acc += w_tile @ x_vec
        acc += tl.sum(w_tile * x_vec[None, :], axis=1)

    # Store
    y_ptrs = y_ptr + pid_n * stride_yn + offs_m * stride_ym
    tl.store(y_ptrs, acc, mask=mask_m)


def _head_forward(x, bn_weight, bn_bias, bn_running_mean, bn_running_var, fc_weight, fc_bias):
    assert x.is_cuda
    device = x.device
    N, C, H, W = x.shape
    # 1) BN+ReLU+GAP -> [N, C]
    y_gap = torch.empty((N, C), device=device, dtype=torch.float32)
    BLOCK_C = 128
    grid_gap = (triton.cdiv(C, BLOCK_C), N)
    _bn_relu_gap_kernel[grid_gap](
        x, bn_weight, bn_bias, bn_running_mean, bn_running_var, y_gap,
        N, C, H, W,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        y_gap.stride(0), y_gap.stride(1),
        EPS=1e-5, BLOCK_C=BLOCK_C,
        num_warps=4
    )
    # 2) Linear: [N, C] x [M, C]^T + b -> [N, M]
    M = fc_weight.shape[0]
    out = torch.empty((N, M), device=device, dtype=torch.float32)
    BLOCK_M = 64
    BLOCK_K = 128
    grid_fc = (triton.cdiv(M, BLOCK_M), N)
    _linear_gemm_kernel[grid_fc](
        y_gap, fc_weight, fc_bias, out,
        N, M, C,
        y_gap.stride(0), y_gap.stride(1),
        fc_weight.stride(0), fc_weight.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K,
        num_warps=4
    )
    return out

# =========================
# Top-level wrapper
# =========================

def kernel_function(x, params):
    """
    End-to-end Triton implementation of the provided DenseNet-like model for the fixed shapes.

    Args:
      x: (N, 3, 224, 224) float32 CUDA tensor.
      params: dict with following keys (tensors on same device as x):
        - 'stem': dict with keys:
            'conv_w': (64,3,7,7)
            'bn_w', 'bn_b', 'bn_rm', 'bn_rv': (64,)
        - 'db1': dict with keys:
            'conv_ws': list of 6 tensors [32, C_in_i, 3,3]
            'bn_w', 'bn_b', 'bn_rm', 'bn_rv': lists of 6 tensors, length matching C_in_i per layer
        - 't1': dict with keys:
            'bn_w','bn_b','bn_rm','bn_rv': (256,)
            'conv_w': (128,256,1,1)
        - 'db2': like db1, lists length 12 (input 128-> output 512)
        - 't2': BN(512), conv_w (256,512,1,1)
        - 'db3': lists length 24 (input 256 -> output 1024)
        - 't3': BN(1024), conv_w (512,1024,1,1)
        - 'db4': lists length 16 (input 512 -> output 1024)
        - 'head': dict with keys:
            'bn_w','bn_b','bn_rm','bn_rv': (1024,)
            'fc_w': (10,1024), 'fc_b': (10,)
    Returns:
      out: (N, 10) float32 CUDA tensor
    """
    assert x.is_cuda and x.dtype == torch.float32
    device = x.device
    N, Cin, H, W = x.shape
    assert (N, Cin, H, W) == (10, 3, 224, 224), "This implementation assumes fixed problem sizes"
    # Stem
    stem = params['stem']
    y = _stem_forward(
        x,
        stem['conv_w'], stem['bn_w'], stem['bn_b'], stem['bn_rm'], stem['bn_rv']
    )  # (N,64,56,56)

    # DenseBlock 1: 6 layers -> 256 ch
    db1 = params['db1']
    y = _denseblock_infer(
        y,
        db1['conv_ws'], db1['bn_w'], db1['bn_b'], db1['bn_rm'], db1['bn_rv'],
        eps=1e-5
    )  # (N,256,56,56)

    # Transition 1: -> (N,128,28,28)
    t1 = params['t1']
    y = _transition_forward(
        y,
        t1['bn_w'], t1['bn_b'], t1['bn_rm'], t1['bn_rv'],
        t1['conv_w']
    )

    # DenseBlock 2: 12 layers -> 512 ch
    db2 = params['db2']
    y = _denseblock_infer(
        y,
        db2['conv_ws'], db2['bn_w'], db2['bn_b'], db2['bn_rm'], db2['bn_rv'],
        eps=1e-5
    )  # (N,512,28,28)

    # Transition 2: -> (N,256,14,14)
    t2 = params['t2']
    y = _transition_forward(
        y,
        t2['bn_w'], t2['bn_b'], t2['bn_rm'], t2['bn_rv'],
        t2['conv_w']
    )

    # DenseBlock 3: 24 layers -> 1024 ch
    db3 = params['db3']
    y = _denseblock_infer(
        y,
        db3['conv_ws'], db3['bn_w'], db3['bn_b'], db3['bn_rm'], db3['bn_rv'],
        eps=1e-5
    )

    # Transition 3: -> (N,512,7,7)
    t3 = params['t3']
    y = _transition_forward(
        y,
        t3['bn_w'], t3['bn_b'], t3['bn_rm'], t3['bn_rv'],
        t3['conv_w']
    )

    # DenseBlock 4: 16 layers -> 1024 ch
    db4 = params['db4']
    y = _denseblock_infer(
        y,
        db4['conv_ws'], db4['bn_w'], db4['bn_b'], db4['bn_rm'], db4['bn_rv'],
        eps=1e-5
    )

    # Head: BN + ReLU + GAP + FC
    head = params['head']
    out = _head_forward(
        y,
        head['bn_w'], head['bn_b'], head['bn_rm'], head['bn_rv'],
        head['fc_w'], head['fc_b']
    )
    return out

# =========================
# Reference model (from original problem) to build params and test
# =========================

import torch.nn as nn
import torch.nn.functional as F

class DenseBlockRef(nn.Module):
    def __init__(self, num_layers: int, num_input_features: int, growth_rate: int):
        super(DenseBlockRef, self).__init__()
        layers = []
        for i in range(num_layers):
            layers.append(self._make_layer(num_input_features + i * growth_rate, growth_rate))
        self.layers = nn.ModuleList(layers)

    def _make_layer(self, in_features: int, growth_rate: int):
        return nn.Sequential(
            nn.BatchNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_features, growth_rate, kernel_size=3, padding=1, bias=False),
            nn.Dropout(0.0)
        )

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            new_feature = layer(x)
            features.append(new_feature)
            x = torch.cat(features, 1)
        return x

class TransitionLayerRef(nn.Module):
    def __init__(self, num_input_features: int, num_output_features: int):
        super(TransitionLayerRef, self).__init__()
        self.transition = nn.Sequential(
            nn.BatchNorm2d(num_input_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_input_features, num_output_features, kernel_size=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.transition(x)

class ModelRef(nn.Module):
    def __init__(self, growth_rate: int = 32, num_classes: int = 1000):
        super(ModelRef, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        num_features = 64
        block_layers = [6, 12, 24, 16]
        self.dense_blocks = nn.ModuleList()
        self.transition_layers = nn.ModuleList()
        for i, num_layers in enumerate(block_layers):
            block = DenseBlockRef(num_layers=num_layers, num_input_features=num_features, growth_rate=growth_rate)
            self.dense_blocks.append(block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_layers) - 1:
                transition = TransitionLayerRef(num_input_features=num_features, num_output_features=num_features // 2)
                self.transition_layers.append(transition)
                num_features = num_features // 2
        self.final_bn = nn.BatchNorm2d(num_features)
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        for i, block in enumerate(self.dense_blocks):
            x = block(x)
            if i != len(self.dense_blocks) - 1:
                x = self.transition_layers[i](x)
        x = self.final_bn(x)
        x = F.relu(x, inplace=True)
        x = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Inputs helpers from original problem
batch_size = 10
num_classes = 10
height, width = 224, 224
def get_inputs():
    return [torch.rand(batch_size, 3, height, width)]
def get_init_inputs():
    return [32, num_classes]

# =========================
# Build params from reference model
# =========================

def build_params_from_model(m: nn.Module):
    device = next(m.parameters()).device
    params = {}
    # stem
    stem = {}
    stem['conv_w'] = m.features[0].weight.detach().to(device)
    bn = m.features[1]
    stem['bn_w'] = bn.weight.detach().to(device)
    stem['bn_b'] = bn.bias.detach().to(device)
    stem['bn_rm'] = bn.running_mean.detach().to(device)
    stem['bn_rv'] = bn.running_var.detach().to(device)
    params['stem'] = stem

    # dense blocks and transitions
    block_layers = [6, 12, 24, 16]
    num_features = 64
    for i, num_layers in enumerate(block_layers):
        block = m.dense_blocks[i]
        db_key = f'db{i+1}'
        conv_ws = []
        bn_w = []
        bn_b = []
        bn_rm = []
        bn_rv = []
        for l in range(num_layers):
            seq = block.layers[l]
            bn_mod = seq[0]
            conv_mod = seq[2]
            conv_ws.append(conv_mod.weight.detach().to(device))
            bn_w.append(bn_mod.weight.detach().to(device))
            bn_b.append(bn_mod.bias.detach().to(device))
            bn_rm.append(bn_mod.running_mean.detach().to(device))
            bn_rv.append(bn_mod.running_var.detach().to(device))
        params[db_key] = dict(conv_ws=conv_ws, bn_w=bn_w, bn_b=bn_b, bn_rm=bn_rm, bn_rv=bn_rv)
        num_features = num_features + num_layers * 32
        if i != len(block_layers) - 1:
            t_key = f't{i+1}'
            trans = m.transition_layers[i].transition
            bn_t = trans[0]
            conv_t = trans[2]
            params[t_key] = dict(
                bn_w=bn_t.weight.detach().to(device),
                bn_b=bn_t.bias.detach().to(device),
                bn_rm=bn_t.running_mean.detach().to(device),
                bn_rv=bn_t.running_var.detach().to(device),
                conv_w=conv_t.weight.detach().to(device),
            )
            num_features = num_features // 2

    # head
    head = {}
    head_bn = m.final_bn
    head['bn_w'] = head_bn.weight.detach().to(device)
    head['bn_b'] = head_bn.bias.detach().to(device)
    head['bn_rm'] = head_bn.running_mean.detach().to(device)
    head['bn_rv'] = head_bn.running_var.detach().to(device)
    head['fc_w'] = m.classifier.weight.detach().to(device)
    head['fc_b'] = m.classifier.bias.detach().to(device)
    params['head'] = head
    return params

# =========================
# Self-test
# =========================

def run_tests():
    torch.manual_seed(0)
    # Instantiate model and input
    growth_rate, n_classes = get_init_inputs()
    model = ModelRef(growth_rate, n_classes).cuda().eval()  # eval mode to use running stats BN (matches kernels)
    x = get_inputs()[0].cuda().to(torch.float32)

    # Build params
    params = build_params_from_model(model)

    # Triton result
    with torch.no_grad():
        y_triton = kernel_function(x, params)
        # Reference
        y_ref = model(x)

    ok = torch.allclose(y_triton, y_ref, rtol=1e-3, atol=1e-3)
    if ok:
        print("PASS")
        return 0
    else:
        max_abs = (y_triton - y_ref).abs().max().item()
        max_rel = ((y_triton - y_ref).abs() / (y_ref.abs() + 1e-8)).max().item()
        print("FAIL: max_abs_err=", max_abs, " max_rel_err=", max_rel)
        return 1

if __name__ == "__main__":
    sys.exit(run_tests())
