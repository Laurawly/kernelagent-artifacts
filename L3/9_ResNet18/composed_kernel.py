import math
import sys
import torch
import triton
import triton.language as tl


# ---------------------------
# Stem: Conv7x7 /2 + BN + ReLU (NCHW, fp32 I/O)
# ---------------------------

@triton.jit
def _conv7x7_s2_bn_relu_kernel(
    x_ptr,            # *fp32 [N, Cin=3, H, W]
    w_ptr,            # *fp32 [Cout, Cin=3, 7, 7]
    gamma_ptr,        # *fp32 [Cout]
    beta_ptr,         # *fp32 [Cout]
    mean_ptr,         # *fp32 [Cout]
    var_ptr,          # *fp32 [Cout]
    y_ptr,            # *fp32 [N, Cout, H_out, W_out]
    N, H, W, C_OUT, H_OUT, W_OUT,
    sXn, sXc, sXh, sXw,
    sWoc, sWci, sWkh, sWkw,
    sYn, sYc, sYh, sYw,
    EPS: tl.constexpr,
    C_IN: tl.constexpr,            # 3
    KH: tl.constexpr, KW: tl.constexpr,  # 7,7
    STRIDE_H: tl.constexpr, STRIDE_W: tl.constexpr,  # 2,2
    PAD_H: tl.constexpr, PAD_W: tl.constexpr,        # 3,3
    DIL_H: tl.constexpr, DIL_W: tl.constexpr,        # 1,1
    BLOCK_OC: tl.constexpr,        # e.g., 32
):
    pid_spatial = tl.program_id(axis=0)  # over N * H_OUT * W_OUT
    pid_oc_blk = tl.program_id(axis=1)   # over OC tiles

    ohw = H_OUT * W_OUT
    n = pid_spatial // ohw
    rem = pid_spatial % ohw
    oh = rem // W_OUT
    ow = rem % W_OUT

    oc_start = pid_oc_blk * BLOCK_OC
    oc_offsets = oc_start + tl.arange(0, BLOCK_OC)
    oc_mask = oc_offsets < C_OUT

    acc = tl.zeros((BLOCK_OC,), dtype=tl.float32)

    base_x_n = n * sXn

    for ci in range(C_IN):
        base_x_ci = base_x_n + ci * sXc
        for kh in range(KH):
            h_in = oh * STRIDE_H - PAD_H + kh * DIL_H
            h_in_valid = (h_in >= 0) & (h_in < H)
            for kw in range(KW):
                w_in = ow * STRIDE_W - PAD_W + kw * DIL_W
                in_bounds = h_in_valid & (w_in >= 0) & (w_in < W)

                h_safe = tl.where(in_bounds, h_in, 0)
                w_safe = tl.where(in_bounds, w_in, 0)

                x_off = base_x_ci + h_safe * sXh + w_safe * sXw
                x_val = tl.load(x_ptr + x_off, mask=in_bounds, other=0.0)

                w_offs = oc_offsets * sWoc + ci * sWci + kh * sWkh + kw * sWkw
                w_vec = tl.load(w_ptr + w_offs, mask=oc_mask, other=0.0)

                acc += x_val * w_vec

    gamma = tl.load(gamma_ptr + oc_offsets, mask=oc_mask, other=1.0)
    beta = tl.load(beta_ptr + oc_offsets, mask=oc_mask, other=0.0)
    running_mean = tl.load(mean_ptr + oc_offsets, mask=oc_mask, other=0.0)
    running_var = tl.load(var_ptr + oc_offsets, mask=oc_mask, other=1.0)

    bn_norm = (acc - running_mean) / tl.sqrt(running_var + EPS)
    bn_out = bn_norm * gamma + beta
    relu_out = tl.maximum(bn_out, 0.0)

    base_y = n * sYn + oh * sYh + ow * sYw
    y_ptrs = y_ptr + base_y + oc_offsets * sYc
    tl.store(y_ptrs, relu_out, mask=oc_mask)


def stem_conv7x7_s2_bn_relu(x, conv_weight, bn_weight, bn_bias, bn_mean, bn_var):
    assert x.is_cuda and conv_weight.is_cuda
    assert x.dtype == torch.float32
    N, C_in, H, W = x.shape
    C_out, C_in_w, KH, KW = conv_weight.shape
    assert C_in == 3 and C_in_w == 3 and KH == 7 and KW == 7
    stride_h = stride_w = 2
    pad_h = pad_w = 3
    dil_h = dil_w = 1
    H_out = (H + 2 * pad_h - dil_h * (KH - 1) - 1) // stride_h + 1
    W_out = (W + 2 * pad_w - dil_w * (KW - 1) - 1) // stride_w + 1

    y = torch.empty((N, C_out, H_out, W_out), device=x.device, dtype=torch.float32)

    sXn, sXc, sXh, sXw = x.stride()
    sWoc, sWci, sWkh, sWkw = conv_weight.stride()
    sYn, sYc, sYh, sYw = y.stride()

    BLOCK_OC = 32
    grid = (N * H_out * W_out, triton.cdiv(C_out, BLOCK_OC))

    _conv7x7_s2_bn_relu_kernel[grid](
        x, conv_weight, bn_weight, bn_bias, bn_mean, bn_var, y,
        N, H, W, C_out, H_out, W_out,
        sXn, sXc, sXh, sXw,
        sWoc, sWci, sWkh, sWkw,
        sYn, sYc, sYh, sYw,
        EPS=1e-5,
        C_IN=3, KH=7, KW=7,
        STRIDE_H=2, STRIDE_W=2,
        PAD_H=3, PAD_W=3,
        DIL_H=1, DIL_W=1,
        BLOCK_OC=BLOCK_OC,
        num_warps=4, num_stages=2,
    )
    return y


# ---------------------------
# MaxPool 3x3 s=2 p=1 (NCHW, dtype-preserving)
# ---------------------------

@triton.jit
def _maxpool2d_3x3_s2_nchw(
    x_ptr, y_ptr,
    N, C, H, W,
    OH, OW,
    stride_n, stride_c, stride_h, stride_w,
    y_stride_n, y_stride_c, y_stride_h, y_stride_w,
    PAD_H: tl.constexpr, PAD_W: tl.constexpr,
    STRIDE_H: tl.constexpr, STRIDE_W: tl.constexpr,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr,
):
    pid_w = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_nc = tl.program_id(2)

    n = pid_nc // C
    c = pid_nc % C

    oh_start = pid_h * BLOCK_H
    ow_start = pid_w * BLOCK_W
    oh = oh_start + tl.arange(0, BLOCK_H)
    ow = ow_start + tl.arange(0, BLOCK_W)

    mask_oh = oh < OH
    mask_ow = ow < OW
    mask_out = mask_oh[:, None] & mask_ow[None, :]

    ih0 = oh * STRIDE_H - PAD_H
    iw0 = ow * STRIDE_W - PAD_W
    ih_base = ih0[:, None]
    iw_base = iw0[None, :]

    x_base = x_ptr + n * stride_n + c * stride_c

    acc = tl.full((BLOCK_H, BLOCK_W), -float("inf"), dtype=tl.float32)

    for ky in range(3):
        ih = ih_base + ky
        mask_h = (ih >= 0) & (ih < H)
        for kx in range(3):
            iw = iw_base + kx
            mask_w = (iw >= 0) & (iw < W)
            m = mask_out & mask_h & mask_w
            ptrs = x_base + ih * stride_h + iw * stride_w
            vals = tl.load(ptrs, mask=m, other=0.0).to(tl.float32)
            acc = tl.maximum(acc, vals)

    y_base = y_ptr + n * y_stride_n + c * y_stride_c
    out_ptrs = y_base + oh[:, None] * y_stride_h + ow[None, :] * y_stride_w
    tl.store(out_ptrs, acc.to(y_ptr.dtype.element_ty), mask=mask_out)


def maxpool2d_3x3_s2_triton(x: torch.Tensor):
    assert x.is_cuda and x.ndim == 4
    N, C, H, W = x.shape
    KH, KW = 3, 3
    SH, SW = 2, 2
    PH, PW = 1, 1
    OH = (H + 2 * PH - KH) // SH + 1
    OW = (W + 2 * PW - KW) // SW + 1
    y = torch.empty((N, C, OH, OW), device=x.device, dtype=x.dtype)

    BLOCK_H = 16
    BLOCK_W = 64
    grid = (triton.cdiv(OW, BLOCK_W), triton.cdiv(OH, BLOCK_H), N * C)

    _maxpool2d_3x3_s2_nchw[grid](
        x, y,
        N, C, H, W,
        OH, OW,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        PAD_H=PH, PAD_W=PW,
        STRIDE_H=SH, STRIDE_W=SW,
        BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W,
        num_warps=4, num_stages=2,
    )
    return y


# ---------------------------
# BasicBlock s=1 (3x3 + BN + ReLU) x2 + Add + ReLU, NCHW (fp32)
# ---------------------------

@triton.jit
def _bb_s1_conv_bn_relu(
    x_ptr,            # *: input NCHW
    w_ptr,            # *: [C, C, 3, 3]
    gamma_ptr, beta_ptr, mean_ptr, var_ptr,
    y_ptr,            # *: output NCHW (fp32)
    N, C, H, W,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wn, stride_wc, stride_wk, stride_wl,
    stride_yn, stride_yc, stride_yh, stride_yw,
    eps,  # float32
    BLOCK_W: tl.constexpr,
):
    pid_w = tl.program_id(axis=0)
    h = tl.program_id(axis=1)
    nc = tl.program_id(axis=2)

    n = nc // C
    co = nc % C

    offs_w = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    mask_w = offs_w < W
    mask_h = h < H
    full_mask_out = mask_w & mask_h & (n < N) & (co < C)

    acc = tl.zeros((BLOCK_W,), dtype=tl.float32)

    for ci in tl.range(0, C):
        for ky in range(0, 3):
            iy = h + ky - 1
            valid_y = (iy >= 0) & (iy < H)
            for kx in range(0, 3):
                ix = offs_w + kx - 1
                mask_x = (ix >= 0) & (ix < W)
                mask = full_mask_out & valid_y & mask_x

                x_ptrs = x_ptr + n * stride_xn + ci * stride_xc + iy * stride_xh + ix * stride_xw
                x_val = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)

                w_ptrs = w_ptr + co * stride_wn + ci * stride_wc + ky * stride_wk + kx * stride_wl
                w_val = tl.load(w_ptrs).to(tl.float32)

                acc += x_val * w_val

    mean = tl.load(mean_ptr + co).to(tl.float32)
    var = tl.load(var_ptr + co).to(tl.float32)
    gamma = tl.load(gamma_ptr + co).to(tl.float32)
    beta = tl.load(beta_ptr + co).to(tl.float32)

    inv_std = 1.0 / tl.sqrt(var + eps)
    y = (acc - mean) * inv_std
    y = y * gamma + beta
    y = tl.maximum(y, 0.0)

    y_ptrs = y_ptr + n * stride_yn + co * stride_yc + h * stride_yh + offs_w * stride_yw
    tl.store(y_ptrs, y, mask=full_mask_out)


@triton.jit
def _bb_s1_conv_bn_add_relu(
    x_id_ptr,         # *: identity input NCHW
    z_ptr,            # *: input from previous stage (y1_relu) NCHW
    w_ptr,            # *: [C, C, 3, 3]
    gamma_ptr, beta_ptr, mean_ptr, var_ptr,
    y_ptr,            # *: output NCHW
    N, C, H, W,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_zn, stride_zc, stride_zh, stride_zw,
    stride_wn, stride_wc, stride_wk, stride_wl,
    stride_yn, stride_yc, stride_yh, stride_yw,
    eps,  # float32
    BLOCK_W: tl.constexpr,
):
    pid_w = tl.program_id(axis=0)
    h = tl.program_id(axis=1)
    nc = tl.program_id(axis=2)

    n = nc // C
    co = nc % C

    offs_w = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    mask_w = offs_w < W
    mask_h = h < H
    full_mask_out = mask_w & mask_h & (n < N) & (co < C)

    acc = tl.zeros((BLOCK_W,), dtype=tl.float32)

    for ci in tl.range(0, C):
        for ky in range(0, 3):
            iy = h + ky - 1
            valid_y = (iy >= 0) & (iy < H)
            for kx in range(0, 3):
                ix = offs_w + kx - 1
                mask_x = (ix >= 0) & (ix < W)
                mask = full_mask_out & valid_y & mask_x

                z_ptrs = z_ptr + n * stride_zn + ci * stride_zc + iy * stride_zh + ix * stride_zw
                z_val = tl.load(z_ptrs, mask=mask, other=0.0).to(tl.float32)

                w_ptrs = w_ptr + co * stride_wn + ci * stride_wc + ky * stride_wk + kx * stride_wl
                w_val = tl.load(w_ptrs).to(tl.float32)

                acc += z_val * w_val

    mean = tl.load(mean_ptr + co).to(tl.float32)
    var = tl.load(var_ptr + co).to(tl.float32)
    gamma = tl.load(gamma_ptr + co).to(tl.float32)
    beta = tl.load(beta_ptr + co).to(tl.float32)

    inv_std = 1.0 / tl.sqrt(var + eps)
    y = (acc - mean) * inv_std
    y = y * gamma + beta

    x_id_ptrs = x_id_ptr + n * stride_xn + co * stride_xc + h * stride_xh + offs_w * stride_xw
    x_id_val = tl.load(x_id_ptrs, mask=full_mask_out, other=0.0).to(tl.float32)
    y = y + x_id_val
    y = tl.maximum(y, 0.0)

    y_ptrs = y_ptr + n * stride_yn + co * stride_yc + h * stride_yh + offs_w * stride_yw
    tl.store(y_ptrs, y, mask=full_mask_out)


def basicblock_s1_triton(
    x: torch.Tensor,
    conv1_weight: torch.Tensor,
    bn1_weight: torch.Tensor,
    bn1_bias: torch.Tensor,
    bn1_running_mean: torch.Tensor,
    bn1_running_var: torch.Tensor,
    conv2_weight: torch.Tensor,
    bn2_weight: torch.Tensor,
    bn2_bias: torch.Tensor,
    bn2_running_mean: torch.Tensor,
    bn2_running_var: torch.Tensor,
):
    assert x.is_cuda and x.dtype == torch.float32
    device = x.device
    N, C, H, W = x.shape
    assert conv1_weight.shape == (C, C, 3, 3)
    assert conv2_weight.shape == (C, C, 3, 3)
    for t in (bn1_weight, bn1_bias, bn1_running_mean, bn1_running_var,
              bn2_weight, bn2_bias, bn2_running_mean, bn2_running_var):
        assert t.shape == (C,)

    y1_relu = torch.empty((N, C, H, W), device=device, dtype=torch.float32)
    y_out = torch.empty_like(x)

    sxn, sxc, sxh, sxw = x.stride()
    swn, swc, swk, swl = conv1_weight.stride()
    s1n, s1c, s1h, s1w = y1_relu.stride()

    BLOCK_W = 64
    grid_1 = (triton.cdiv(W, BLOCK_W), H, N * C)

    _bb_s1_conv_bn_relu[grid_1](
        x, conv1_weight,
        bn1_weight, bn1_bias, bn1_running_mean, bn1_running_var,
        y1_relu,
        N, C, H, W,
        sxn, sxc, sxh, sxw,
        swn, swc, swk, swl,
        s1n, s1c, s1h, s1w,
        1e-5,
        BLOCK_W=BLOCK_W,
        num_warps=4, num_stages=2,
    )

    szn, szc, szh, szw = y1_relu.stride()
    s2n, s2c, s2h, s2w = y_out.stride()
    swn2, swc2, swk2, swl2 = conv2_weight.stride()

    grid_2 = (triton.cdiv(W, BLOCK_W), H, N * C)

    _bb_s1_conv_bn_add_relu[grid_2](
        x, y1_relu, conv2_weight,
        bn2_weight, bn2_bias, bn2_running_mean, bn2_running_var,
        y_out,
        N, C, H, W,
        sxn, sxc, sxh, sxw,
        szn, szc, szh, szw,
        swn2, swc2, swk2, swl2,
        s2n, s2c, s2h, s2w,
        1e-5,
        BLOCK_W=BLOCK_W,
        num_warps=4, num_stages=2,
    )
    return y_out


# ---------------------------
# BasicBlock downsample (s=2 on main conv1 and skip 1x1 s=2)
# ---------------------------

@triton.jit
def _down_s2_conv3x3_bn_relu(
    x_ptr, w_ptr,
    bn_w_ptr, bn_b_ptr, bn_mean_ptr, bn_var_ptr,
    out_ptr,
    N, Cin, Hin, Win,
    Cout, Hout, Wout,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wo, stride_wi, stride_wky, stride_wkx,
    stride_on, stride_oc, stride_oh, stride_ow,
    eps: tl.constexpr,
    BLOCK_OC: tl.constexpr,
):
    pid_hw = tl.program_id(0)
    pid_oc = tl.program_id(1)

    HW = Hout * Wout
    n = pid_hw // HW
    hw = pid_hw % HW
    oh = hw // Wout
    ow = hw % Wout

    oc_start = pid_oc * BLOCK_OC
    oc_offsets = oc_start + tl.arange(0, BLOCK_OC)
    oc_mask = oc_offsets < Cout

    base_iy = oh * 2 - 1
    base_ix = ow * 2 - 1

    acc = tl.zeros((BLOCK_OC,), dtype=tl.float32)

    for ic in tl.range(0, Cin):
        for ky in range(0, 3):
            iy = base_iy + ky
            y_in_bounds = (iy >= 0) & (iy < Hin)
            for kx in range(0, 3):
                ix = base_ix + kx
                in_bounds = y_in_bounds & (ix >= 0) & (ix < Win)
                x_offset = n * stride_xn + ic * stride_xc + iy * stride_xh + ix * stride_xw
                x_val = tl.load(x_ptr + x_offset, mask=in_bounds, other=0.0).to(tl.float32)

                w_ptrs = w_ptr + oc_offsets * stride_wo + ic * stride_wi + ky * stride_wky + kx * stride_wkx
                w_vec = tl.load(w_ptrs, mask=oc_mask, other=0.0).to(tl.float32)

                acc += w_vec * x_val

    gamma = tl.load(bn_w_ptr + oc_offsets, mask=oc_mask, other=0).to(tl.float32)
    beta = tl.load(bn_b_ptr + oc_offsets, mask=oc_mask, other=0).to(tl.float32)
    mean = tl.load(bn_mean_ptr + oc_offsets, mask=oc_mask, other=0).to(tl.float32)
    var = tl.load(bn_var_ptr + oc_offsets, mask=oc_mask, other=0).to(tl.float32)

    denom = tl.sqrt(var + eps)
    scale = gamma / denom
    shift = beta - mean * scale

    y = acc * scale + shift
    y = tl.maximum(y, 0)

    out_ptrs = out_ptr + n * stride_on + oc_offsets * stride_oc + oh * stride_oh + ow * stride_ow
    tl.store(out_ptrs, y, mask=oc_mask)


@triton.jit
def _down_s2_conv3x3_bn_add_skip1x1_bn_relu(
    x_ptr, act_ptr,
    w2_ptr,
    bn2_w_ptr, bn2_b_ptr, bn2_mean_ptr, bn2_var_ptr,
    ds_w_ptr,
    ds_bn_w_ptr, ds_bn_b_ptr, ds_bn_mean_ptr, ds_bn_var_ptr,
    out_ptr,
    N, Cin, Hin, Win,
    Cmid, Hout, Wout, Cout,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_a_n, stride_a_c, stride_a_h, stride_a_w,
    stride_w2o, stride_w2i, stride_w2ky, stride_w2kx,
    stride_dso, stride_dsi,
    stride_on, stride_oc, stride_oh, stride_ow,
    eps: tl.constexpr,
    BLOCK_OC: tl.constexpr,
):
    pid_hw = tl.program_id(0)
    pid_oc = tl.program_id(1)

    HW = Hout * Wout
    n = pid_hw // HW
    hw = pid_hw % HW
    oh = hw // Wout
    ow = hw % Wout

    oc_start = pid_oc * BLOCK_OC
    oc_offsets = oc_start + tl.arange(0, BLOCK_OC)
    oc_mask = oc_offsets < Cout

    acc_main = tl.zeros((BLOCK_OC,), dtype=tl.float32)
    for ic in tl.range(0, Cmid):
        for ky in range(0, 3):
            iy = oh + ky - 1
            y_in_bounds = (iy >= 0) & (iy < Hout)
            for kx in range(0, 3):
                ix = ow + kx - 1
                in_bounds = y_in_bounds & (ix >= 0) & (ix < Wout)

                a_off = n * stride_a_n + ic * stride_a_c + iy * stride_a_h + ix * stride_a_w
                a_val = tl.load(act_ptr + a_off, mask=in_bounds, other=0.0).to(tl.float32)

                w2_ptrs = w2_ptr + oc_offsets * stride_w2o + ic * stride_w2i + ky * stride_w2ky + kx * stride_w2kx
                w2_vec = tl.load(w2_ptrs, mask=oc_mask, other=0.0).to(tl.float32)
                acc_main += w2_vec * a_val

    gamma2 = tl.load(bn2_w_ptr + oc_offsets, mask=oc_mask, other=0).to(tl.float32)
    beta2 = tl.load(bn2_b_ptr + oc_offsets, mask=oc_mask, other=0).to(tl.float32)
    mean2 = tl.load(bn2_mean_ptr + oc_offsets, mask=oc_mask, other=0).to(tl.float32)
    var2 = tl.load(bn2_var_ptr + oc_offsets, mask=oc_mask, other=0).to(tl.float32)
    denom2 = tl.sqrt(var2 + eps)
    scale2 = gamma2 / denom2
    shift2 = beta2 - mean2 * scale2
    y_main = acc_main * scale2 + shift2

    iy = oh * 2
    ix = ow * 2
    valid_in = (iy >= 0) & (iy < Hin) & (ix >= 0) & (ix < Win)

    acc_skip = tl.zeros((BLOCK_OC,), dtype=tl.float32)
    for ic in tl.range(0, Cin):
        x_off = n * stride_xn + ic * stride_xc + iy * stride_xh + ix * stride_xw
        x_val = tl.load(x_ptr + x_off, mask=valid_in, other=0.0).to(tl.float32)
        ds_w_ptrs = ds_w_ptr + oc_offsets * stride_dso + ic * stride_dsi
        w_vec = tl.load(ds_w_ptrs, mask=oc_mask, other=0.0).to(tl.float32)
        acc_skip += w_vec * x_val

    dsg = tl.load(ds_bn_w_ptr + oc_offsets, mask=oc_mask, other=0).to(tl.float32)
    dsb = tl.load(ds_bn_b_ptr + oc_offsets, mask=oc_mask, other=0).to(tl.float32)
    dsm = tl.load(ds_bn_mean_ptr + oc_offsets, mask=oc_mask, other=0).to(tl.float32)
    dsv = tl.load(ds_bn_var_ptr + oc_offsets, mask=oc_mask, other=0).to(tl.float32)
    ds_denom = tl.sqrt(dsv + eps)
    ds_scale = dsg / ds_denom
    ds_shift = dsb - dsm * ds_scale
    y_skip = acc_skip * ds_scale + ds_shift

    y = tl.maximum(y_main + y_skip, 0)

    out_ptrs = out_ptr + n * stride_on + oc_offsets * stride_oc + oh * stride_oh + ow * stride_ow
    tl.store(out_ptrs, y, mask=oc_mask)


def basicblock_down_s2_triton(
    x: torch.Tensor,
    *,
    conv1_weight: torch.Tensor,
    bn1_weight: torch.Tensor,
    bn1_bias: torch.Tensor,
    bn1_running_mean: torch.Tensor,
    bn1_running_var: torch.Tensor,
    conv2_weight: torch.Tensor,
    bn2_weight: torch.Tensor,
    bn2_bias: torch.Tensor,
    bn2_running_mean: torch.Tensor,
    bn2_running_var: torch.Tensor,
    downsample_conv_weight: torch.Tensor,
    downsample_bn_weight: torch.Tensor,
    downsample_bn_bias: torch.Tensor,
    downsample_bn_running_mean: torch.Tensor,
    downsample_bn_running_var: torch.Tensor,
    eps: float = 1e-5,
):
    assert x.is_cuda and x.dtype == torch.float32
    device = x.device
    N, Cin, Hin, Win = x.shape
    Cout1 = conv1_weight.shape[0]
    Cmid = Cout1
    Cout = conv2_weight.shape[0]
    assert conv1_weight.shape == (Cout1, Cin, 3, 3)
    assert conv2_weight.shape == (Cout, Cmid, 3, 3)
    assert downsample_conv_weight.shape == (Cout, Cin, 1, 1)

    Hmid = (Hin + 2 * 1 - 3) // 2 + 1
    Wmid = (Win + 2 * 1 - 3) // 2 + 1
    Hout, Wout = Hmid, Wmid

    act = torch.empty((N, Cmid, Hmid, Wmid), device=device, dtype=torch.float32)
    out = torch.empty((N, Cout, Hout, Wout), device=device, dtype=torch.float32)

    x_c = x.contiguous()
    w1_c = conv1_weight.contiguous()
    w2_c = conv2_weight.contiguous()
    ds_w_c = downsample_conv_weight.contiguous()

    bn1_w_c = bn1_weight.contiguous()
    bn1_b_c = bn1_bias.contiguous()
    bn1_mean_c = bn1_running_mean.contiguous()
    bn1_var_c = bn1_running_var.contiguous()

    bn2_w_c = bn2_weight.contiguous()
    bn2_b_c = bn2_bias.contiguous()
    bn2_mean_c = bn2_running_mean.contiguous()
    bn2_var_c = bn2_running_var.contiguous()

    ds_bn_w_c = downsample_bn_weight.contiguous()
    ds_bn_b_c = downsample_bn_bias.contiguous()
    ds_bn_mean_c = downsample_bn_running_mean.contiguous()
    ds_bn_var_c = downsample_bn_running_var.contiguous()

    sxn, sxc, sxh, sxw = x_c.stride()
    s1o, s1i, s1ky, s1kx = w1_c.stride()
    s2o, s2i, s2ky, s2kx = w2_c.stride()
    sdsn, sdsi = ds_w_c.stride()[:2]

    san, sac, sah, saw = act.stride()
    son, soc, soh, sow = out.stride()

    BLOCK_OC = 64
    grid1 = (N * Hmid * Wmid, triton.cdiv(Cout1, BLOCK_OC))
    grid2 = (N * Hout * Wout, triton.cdiv(Cout, BLOCK_OC))

    _down_s2_conv3x3_bn_relu[grid1](
        x_c, w1_c,
        bn1_w_c, bn1_b_c, bn1_mean_c, bn1_var_c,
        act,
        N, Cin, Hin, Win,
        Cout1, Hmid, Wmid,
        sxn, sxc, sxh, sxw,
        s1o, s1i, s1ky, s1kx,
        san, sac, sah, saw,
        eps,
        BLOCK_OC=BLOCK_OC,
        num_warps=4, num_stages=2,
    )

    _down_s2_conv3x3_bn_add_skip1x1_bn_relu[grid2](
        x_c, act,
        w2_c,
        bn2_w_c, bn2_b_c, bn2_mean_c, bn2_var_c,
        ds_w_c,
        ds_bn_w_c, ds_bn_b_c, ds_bn_mean_c, ds_bn_var_c,
        out,
        N, Cin, Hin, Win,
        Cmid, Hout, Wout, Cout,
        sxn, sxc, sxh, sxw,
        san, sac, sah, saw,
        s2o, s2i, s2ky, s2kx,
        sdsn, sdsi,
        son, soc, soh, sow,
        eps,
        BLOCK_OC=BLOCK_OC,
        num_warps=4, num_stages=2,
    )
    return out


# ---------------------------
# AdaptiveAvgPool2d((1,1)) + Flatten(start_dim=1) (NCHW)
# ---------------------------

@triton.jit
def _adaptive_avgpool2d_flatten_kernel(
    x_ptr,
    y_ptr,
    N, C, H, W,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_yn, stride_yc,
    BLOCK_W: tl.constexpr,
):
    pid_c = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    n = pid_n
    c = pid_c
    valid_nc = (n < N) & (c < C)

    x_base = x_ptr + n * stride_xn + c * stride_xc

    acc = tl.zeros((), dtype=tl.float32)

    offs_w = tl.arange(0, BLOCK_W)
    for h in tl.range(0, H):
        row_base = x_base + h * stride_xh
        for start_w in tl.range(0, W, BLOCK_W):
            w_idx = start_w + offs_w
            m = (w_idx < W) & valid_nc
            ptrs = row_base + w_idx * stride_xw
            vals = tl.load(ptrs, mask=m, other=0.0)
            acc += tl.sum(vals.to(tl.float32), axis=0)

    acc = acc / W
    acc = acc / H

    y_ptrs = y_ptr + n * stride_yn + c * stride_yc
    tl.store(y_ptrs, acc.to(y_ptr.dtype.element_ty), mask=valid_nc)


def adaptive_avgpool_flatten_triton(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and x.ndim == 4
    N, C, H, W = x.shape
    y = torch.empty((N, C), device=x.device, dtype=x.dtype)
    stride_xn, stride_xc, stride_xh, stride_xw = x.stride()
    stride_yn, stride_yc = y.stride()
    BLOCK_W = 64
    grid = (C, N)
    _adaptive_avgpool2d_flatten_kernel[grid](
        x, y,
        N, C, H, W,
        stride_xn, stride_xc, stride_xh, stride_xw,
        stride_yn, stride_yc,
        BLOCK_W=BLOCK_W,
        num_warps=2, num_stages=2,
    )
    return y


# ---------------------------
# Linear (matmul) with bias: [N,K] x [O,K]^T + bias -> [N,O]
# ---------------------------

@triton.jit
def _linear_bias_kernel(
    x_ptr,         # *fp32 [N, K]
    w_ptr,         # *fp32 [O, K] (row-major by O)
    b_ptr,         # *fp32 [O]
    y_ptr,         # *fp32 [N, O]
    N, K, O,
    sXn, sXk,
    sWo, sWk,
    sYn, sYo,
    BLOCK_O: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(axis=0)
    pid_o_blk = tl.program_id(axis=1)

    n = pid_n
    o_start = pid_o_blk * BLOCK_O
    o_offsets = o_start + tl.arange(0, BLOCK_O)
    o_mask = o_offsets < O
    k_offsets_base = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_O,), dtype=tl.float32)

    for k0 in tl.range(0, K, BLOCK_K):
        k_offsets = k0 + k_offsets_base
        k_mask = k_offsets < K

        x_ptrs = x_ptr + n * sXn + k_offsets * sXk
        x_vals = tl.load(x_ptrs, mask=k_mask, other=0.0)  # [BLOCK_K]
        x_vals = x_vals.to(tl.float32)

        w_ptrs = w_ptr + o_offsets[:, None] * sWo + k_offsets[None, :] * sWk
        w_mask = o_mask[:, None] & k_mask[None, :]
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0).to(tl.float32)  # [BLOCK_O, BLOCK_K]

        acc += tl.sum(w_tile * x_vals[None, :], axis=1)

    b_vals = tl.load(b_ptr + o_offsets, mask=o_mask, other=0.0).to(tl.float32)
    acc = acc + b_vals

    y_ptrs = y_ptr + n * sYn + o_offsets * sYo
    tl.store(y_ptrs, acc, mask=o_mask)


def linear_triton(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and weight.is_cuda and bias.is_cuda
    assert x.dtype == torch.float32 and weight.dtype == torch.float32 and bias.dtype == torch.float32
    N, K = x.shape
    O, Kw = weight.shape
    assert Kw == K
    y = torch.empty((N, O), device=x.device, dtype=torch.float32)

    sXn, sXk = x.stride()
    sWo, sWk = weight.stride()
    sYn, sYo = y.stride()

    BLOCK_O = 128
    BLOCK_K = 128
    grid = (N, triton.cdiv(O, BLOCK_O))
    _linear_bias_kernel[grid](
        x, weight, bias, y,
        N, K, O,
        sXn, sXk,
        sWo, sWk,
        sYn, sYo,
        BLOCK_O=BLOCK_O, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=2,
    )
    return y


# ---------------------------
# Top-level orchestration wrapper
# ---------------------------

def kernel_function(x: torch.Tensor, params: dict) -> torch.Tensor:
    """
    End-to-end Triton implementation of the provided ResNet-like model.

    Args:
      x: [N, 3, 224, 224] float32 CUDA tensor
      params: dict mapping module names to tensors. Expected keys (PyTorch naming):
        - 'conv1.weight', 'bn1.weight', 'bn1.bias', 'bn1.running_mean', 'bn1.running_var'
        - For each layer i=1..4 and block j=0..1:
            f'layer{i}.{j}.conv1.weight'
            f'layer{i}.{j}.bn1.weight', f'layer{i}.{j}.bn1.bias', f'layer{i}.{j}.bn1.running_mean', f'layer{i}.{j}.bn1.running_var'
            f'layer{i}.{j}.conv2.weight'
            f'layer{i}.{j}.bn2.weight', f'layer{i}.{j}.bn2.bias', f'layer{i}.{j}.bn2.running_mean', f'layer{i}.{j}.bn2.running_var'
          For j==0 when i in {2,3,4} (downsample present):
            f'layer{i}.{j}.downsample.0.weight'
            f'layer{i}.{j}.downsample.1.weight', f'layer{i}.{j}.downsample.1.bias', f'layer{i}.{j}.downsample.1.running_mean', f'layer{i}.{j}.downsample.1.running_var'
        - 'fc.weight', 'fc.bias'

    Returns:
      [N, 1000] float32 CUDA tensor.
    """
    assert x.is_cuda and x.dtype == torch.float32
    # Stem
    y = stem_conv7x7_s2_bn_relu(
        x,
        params['conv1.weight'].contiguous(),
        params['bn1.weight'].contiguous(),
        params['bn1.bias'].contiguous(),
        params['bn1.running_mean'].contiguous(),
        params['bn1.running_var'].contiguous(),
    )
    # MaxPool 3x3 s=2
    y = maxpool2d_3x3_s2_triton(y)

    # Layer1: 64 channels, two s=1 blocks
    y = basicblock_s1_triton(
        y,
        params['layer1.0.conv1.weight'].contiguous(),
        params['layer1.0.bn1.weight'].contiguous(),
        params['layer1.0.bn1.bias'].contiguous(),
        params['layer1.0.bn1.running_mean'].contiguous(),
        params['layer1.0.bn1.running_var'].contiguous(),
        params['layer1.0.conv2.weight'].contiguous(),
        params['layer1.0.bn2.weight'].contiguous(),
        params['layer1.0.bn2.bias'].contiguous(),
        params['layer1.0.bn2.running_mean'].contiguous(),
        params['layer1.0.bn2.running_var'].contiguous(),
    )
    y = basicblock_s1_triton(
        y,
        params['layer1.1.conv1.weight'].contiguous(),
        params['layer1.1.bn1.weight'].contiguous(),
        params['layer1.1.bn1.bias'].contiguous(),
        params['layer1.1.bn1.running_mean'].contiguous(),
        params['layer1.1.bn1.running_var'].contiguous(),
        params['layer1.1.conv2.weight'].contiguous(),
        params['layer1.1.bn2.weight'].contiguous(),
        params['layer1.1.bn2.bias'].contiguous(),
        params['layer1.1.bn2.running_mean'].contiguous(),
        params['layer1.1.bn2.running_var'].contiguous(),
    )

    # Layer2: 64->128 downsample block + s=1 block
    y = basicblock_down_s2_triton(
        y,
        conv1_weight=params['layer2.0.conv1.weight'].contiguous(),
        bn1_weight=params['layer2.0.bn1.weight'].contiguous(),
        bn1_bias=params['layer2.0.bn1.bias'].contiguous(),
        bn1_running_mean=params['layer2.0.bn1.running_mean'].contiguous(),
        bn1_running_var=params['layer2.0.bn1.running_var'].contiguous(),
        conv2_weight=params['layer2.0.conv2.weight'].contiguous(),
        bn2_weight=params['layer2.0.bn2.weight'].contiguous(),
        bn2_bias=params['layer2.0.bn2.bias'].contiguous(),
        bn2_running_mean=params['layer2.0.bn2.running_mean'].contiguous(),
        bn2_running_var=params['layer2.0.bn2.running_var'].contiguous(),
        downsample_conv_weight=params['layer2.0.downsample.0.weight'].contiguous(),
        downsample_bn_weight=params['layer2.0.downsample.1.weight'].contiguous(),
        downsample_bn_bias=params['layer2.0.downsample.1.bias'].contiguous(),
        downsample_bn_running_mean=params['layer2.0.downsample.1.running_mean'].contiguous(),
        downsample_bn_running_var=params['layer2.0.downsample.1.running_var'].contiguous(),
    )
    y = basicblock_s1_triton(
        y,
        params['layer2.1.conv1.weight'].contiguous(),
        params['layer2.1.bn1.weight'].contiguous(),
        params['layer2.1.bn1.bias'].contiguous(),
        params['layer2.1.bn1.running_mean'].contiguous(),
        params['layer2.1.bn1.running_var'].contiguous(),
        params['layer2.1.conv2.weight'].contiguous(),
        params['layer2.1.bn2.weight'].contiguous(),
        params['layer2.1.bn2.bias'].contiguous(),
        params['layer2.1.bn2.running_mean'].contiguous(),
        params['layer2.1.bn2.running_var'].contiguous(),
    )

    # Layer3: 128->256 downsample + s=1
    y = basicblock_down_s2_triton(
        y,
        conv1_weight=params['layer3.0.conv1.weight'].contiguous(),
        bn1_weight=params['layer3.0.bn1.weight'].contiguous(),
        bn1_bias=params['layer3.0.bn1.bias'].contiguous(),
        bn1_running_mean=params['layer3.0.bn1.running_mean'].contiguous(),
        bn1_running_var=params['layer3.0.bn1.running_var'].contiguous(),
        conv2_weight=params['layer3.0.conv2.weight'].contiguous(),
        bn2_weight=params['layer3.0.bn2.weight'].contiguous(),
        bn2_bias=params['layer3.0.bn2.bias'].contiguous(),
        bn2_running_mean=params['layer3.0.bn2.running_mean'].contiguous(),
        bn2_running_var=params['layer3.0.bn2.running_var'].contiguous(),
        downsample_conv_weight=params['layer3.0.downsample.0.weight'].contiguous(),
        downsample_bn_weight=params['layer3.0.downsample.1.weight'].contiguous(),
        downsample_bn_bias=params['layer3.0.downsample.1.bias'].contiguous(),
        downsample_bn_running_mean=params['layer3.0.downsample.1.running_mean'].contiguous(),
        downsample_bn_running_var=params['layer3.0.downsample.1.running_var'].contiguous(),
    )
    y = basicblock_s1_triton(
        y,
        params['layer3.1.conv1.weight'].contiguous(),
        params['layer3.1.bn1.weight'].contiguous(),
        params['layer3.1.bn1.bias'].contiguous(),
        params['layer3.1.bn1.running_mean'].contiguous(),
        params['layer3.1.bn1.running_var'].contiguous(),
        params['layer3.1.conv2.weight'].contiguous(),
        params['layer3.1.bn2.weight'].contiguous(),
        params['layer3.1.bn2.bias'].contiguous(),
        params['layer3.1.bn2.running_mean'].contiguous(),
        params['layer3.1.bn2.running_var'].contiguous(),
    )

    # Layer4: 256->512 downsample + s=1
    y = basicblock_down_s2_triton(
        y,
        conv1_weight=params['layer4.0.conv1.weight'].contiguous(),
        bn1_weight=params['layer4.0.bn1.weight'].contiguous(),
        bn1_bias=params['layer4.0.bn1.bias'].contiguous(),
        bn1_running_mean=params['layer4.0.bn1.running_mean'].contiguous(),
        bn1_running_var=params['layer4.0.bn1.running_var'].contiguous(),
        conv2_weight=params['layer4.0.conv2.weight'].contiguous(),
        bn2_weight=params['layer4.0.bn2.weight'].contiguous(),
        bn2_bias=params['layer4.0.bn2.bias'].contiguous(),
        bn2_running_mean=params['layer4.0.bn2.running_mean'].contiguous(),
        bn2_running_var=params['layer4.0.bn2.running_var'].contiguous(),
        downsample_conv_weight=params['layer4.0.downsample.0.weight'].contiguous(),
        downsample_bn_weight=params['layer4.0.downsample.1.weight'].contiguous(),
        downsample_bn_bias=params['layer4.0.downsample.1.bias'].contiguous(),
        downsample_bn_running_mean=params['layer4.0.downsample.1.running_mean'].contiguous(),
        downsample_bn_running_var=params['layer4.0.downsample.1.running_var'].contiguous(),
    )
    y = basicblock_s1_triton(
        y,
        params['layer4.1.conv1.weight'].contiguous(),
        params['layer4.1.bn1.weight'].contiguous(),
        params['layer4.1.bn1.bias'].contiguous(),
        params['layer4.1.bn1.running_mean'].contiguous(),
        params['layer4.1.bn1.running_var'].contiguous(),
        params['layer4.1.conv2.weight'].contiguous(),
        params['layer4.1.bn2.weight'].contiguous(),
        params['layer4.1.bn2.bias'].contiguous(),
        params['layer4.1.bn2.running_mean'].contiguous(),
        params['layer4.1.bn2.running_var'].contiguous(),
    )

    # AdaptiveAvgPool + flatten
    y = adaptive_avgpool_flatten_triton(y)  # [N, 512]

    # Linear
    logits = linear_triton(y, params['fc.weight'].contiguous(), params['fc.bias'].contiguous())  # [N,1000]
    return logits


# ---------------------------
# Reference PyTorch model for self-test
# ---------------------------

import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride
    def forward(self, x):
        identity = x
        out = self.conv1(x); out = self.bn1(out); out = self.relu(out)
        out = self.conv2(out); out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class Model(nn.Module):
    def __init__(self, num_classes=1000):
        super(Model, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)
    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.conv1(x); x = self.bn1(x); x = self.relu(x); x = self.maxpool(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        x = self.avgpool(x); x = torch.flatten(x, 1); x = self.fc(x)
        return x


# ---------------------------
# Utilities to extract parameters into dict
# ---------------------------

def build_params_dict(model: Model) -> dict:
    p = {}
    # Stem
    p['conv1.weight'] = model.conv1.weight.data
    p['bn1.weight'] = model.bn1.weight.data
    p['bn1.bias'] = model.bn1.bias.data
    p['bn1.running_mean'] = model.bn1.running_mean.data
    p['bn1.running_var'] = model.bn1.running_var.data
    # Layers
    for li in range(1, 5):
        layer = getattr(model, f'layer{li}')
        for bj in range(2):
            blk = getattr(layer, str(bj))
            p[f'layer{li}.{bj}.conv1.weight'] = blk.conv1.weight.data
            p[f'layer{li}.{bj}.bn1.weight'] = blk.bn1.weight.data
            p[f'layer{li}.{bj}.bn1.bias'] = blk.bn1.bias.data
            p[f'layer{li}.{bj}.bn1.running_mean'] = blk.bn1.running_mean.data
            p[f'layer{li}.{bj}.bn1.running_var'] = blk.bn1.running_var.data

            p[f'layer{li}.{bj}.conv2.weight'] = blk.conv2.weight.data
            p[f'layer{li}.{bj}.bn2.weight'] = blk.bn2.weight.data
            p[f'layer{li}.{bj}.bn2.bias'] = blk.bn2.bias.data
            p[f'layer{li}.{bj}.bn2.running_mean'] = blk.bn2.running_mean.data
            p[f'layer{li}.{bj}.bn2.running_var'] = blk.bn2.running_var.data

            if blk.downsample is not None:
                ds_conv = blk.downsample[0]
                ds_bn = blk.downsample[1]
                p[f'layer{li}.{bj}.downsample.0.weight'] = ds_conv.weight.data
                p[f'layer{li}.{bj}.downsample.1.weight'] = ds_bn.weight.data
                p[f'layer{li}.{bj}.downsample.1.bias'] = ds_bn.bias.data
                p[f'layer{li}.{bj}.downsample.1.running_mean'] = ds_bn.running_mean.data
                p[f'layer{li}.{bj}.downsample.1.running_var'] = ds_bn.running_var.data
    # FC
    p['fc.weight'] = model.fc.weight.data
    p['fc.bias'] = model.fc.bias.data
    return p


def to_device_and_fp32(p: dict, device: torch.device) -> dict:
    q = {}
    for k, v in p.items():
        q[k] = v.detach().to(device=device, dtype=torch.float32).contiguous()
    return q


# ---------------------------
# Self-test
# ---------------------------

def run_tests():
    torch.manual_seed(0)
    device = torch.device('cuda')
    batch_size = 2
    num_classes = 1000
    input_shape = (batch_size, 3, 224, 224)

    # Build reference model and data
    model = Model(num_classes).eval().to(device)
    x = torch.rand(input_shape, device=device, dtype=torch.float32)

    with torch.no_grad():
        # Collect params
        params = build_params_dict(model)
        params = to_device_and_fp32(params, device)

        # Triton result
        y_triton = kernel_function(x, params)

        # Reference
        y_ref = model(x)

        # Compare
        ok = torch.allclose(y_triton, y_ref, rtol=1e-3, atol=1e-3)
        max_err = (y_triton - y_ref).abs().max().item()
        print("max_abs_err:", max_err)
        if ok:
            print("PASS")
            sys.exit(0)
        else:
            print("FAIL")
            sys.exit(1)


if __name__ == "__main__":
    run_tests()
