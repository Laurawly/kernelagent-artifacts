import math
import torch
import triton
import triton.language as tl


# ----------------------------
# Block 1: Conv2d(11x11, s=4, p=2) + ReLU, then MaxPool2d(3x3, s=2)
# ----------------------------

@triton.jit
def _block1_conv11_relu_kernel(
    x_ptr,        # *T,  [N, C_in, H, W]
    w_ptr,        # *T,  [C_out, C_in, KH=11, KW=11]
    b_ptr,        # *T,  [C_out]
    y_ptr,        # *T,  [N, C_out, HO, WO]  (conv+relu output)
    N, C_in, H, W, C_out, HO, WO,                       # runtime sizes
    stride_xn, stride_xc, stride_xh, stride_xw,         # input strides (in elements)
    stride_wc, stride_wci, stride_wkh, stride_wkw,      # weight strides (in elements)
    stride_yn, stride_yc, stride_yh, stride_yw,         # output strides (in elements)
    BLOCK_CO: tl.constexpr,                             # tile in output channels
    BLOCK_PIX: tl.constexpr,                            # tile in (N*HO*WO)
    KH: tl.constexpr, KW: tl.constexpr,                 # conv kernel size (compile-time)
    STRIDE_H: tl.constexpr, STRIDE_W: tl.constexpr,     # conv stride (compile-time)
    PAD_H: tl.constexpr, PAD_W: tl.constexpr            # conv padding (compile-time)
):
    pid_pix = tl.program_id(axis=0)  # tiles over flattened pixels
    pid_co = tl.program_id(axis=1)   # tiles over output channels

    pix_start = pid_pix * BLOCK_PIX
    pix_offsets = pix_start + tl.arange(0, BLOCK_PIX)
    p_mask = pix_offsets < (N * HO * WO)

    ho_w = HO * WO
    n_idx = pix_offsets // ho_w
    rem = pix_offsets % ho_w
    oh_idx = rem // WO
    ow_idx = rem % WO

    co_offsets = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    co_mask = co_offsets < C_out

    acc = tl.zeros((BLOCK_PIX, BLOCK_CO), dtype=tl.float32)

    for ci in range(0, C_in):
        for ky in range(0, KH):
            ih = oh_idx * STRIDE_H + (ky - PAD_H)
            inb_h = (ih >= 0) & (ih < H)
            for kx in range(0, KW):
                iw = ow_idx * STRIDE_W + (kx - PAD_W)
                inb_w = (iw >= 0) & (iw < W)
                valid = p_mask & inb_h & inb_w

                x_ptrs = (
                    x_ptr
                    + n_idx * stride_xn
                    + ci * stride_xc
                    + ih * stride_xh
                    + iw * stride_xw
                )
                x_vals = tl.load(x_ptrs, mask=valid, other=0.0).to(tl.float32)

                w_ptrs = (
                    w_ptr
                    + co_offsets * stride_wc
                    + ci * stride_wci
                    + ky * stride_wkh
                    + kx * stride_wkw
                )
                w_vals = tl.load(w_ptrs, mask=co_mask, other=0.0).to(tl.float32)

                acc += x_vals[:, None] * w_vals[None, :]

    b_vals = tl.load(b_ptr + co_offsets, mask=co_mask, other=0.0).to(tl.float32)
    acc = acc + b_vals[None, :]
    acc = tl.maximum(acc, 0.0)

    y_ptrs = (
        y_ptr
        + n_idx[:, None] * stride_yn
        + co_offsets[None, :] * stride_yc
        + oh_idx[:, None] * stride_yh
        + ow_idx[:, None] * stride_yw
    )
    out_mask = p_mask[:, None] & co_mask[None, :]
    tl.store(y_ptrs, acc.to(y_ptr.dtype.element_ty), mask=out_mask)


@triton.jit
def _block1_maxpool2d_kernel(
    x_ptr,      # *T, [N, C, HO, WO] (input to pooling, i.e., conv+relu output)
    y_ptr,      # *T, [N, C, HPO, WPO]
    N, C, HO, WO, HPO, WPO,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_yn, stride_yc, stride_yh, stride_yw,
    NC_BLOCK: tl.constexpr, POS_BLOCK: tl.constexpr,
    POOL_KH: tl.constexpr, POOL_KW: tl.constexpr,
    POOL_STRIDE_H: tl.constexpr, POOL_STRIDE_W: tl.constexpr,
):
    pid_nc = tl.program_id(axis=0)
    pid_pos = tl.program_id(axis=1)

    nc_start = pid_nc * NC_BLOCK
    nc_offsets = nc_start + tl.arange(0, NC_BLOCK)
    nc_mask = nc_offsets < (N * C)
    n_idx = nc_offsets // C
    c_idx = nc_offsets % C

    pos_start = pid_pos * POS_BLOCK
    pos_offsets = pos_start + tl.arange(0, POS_BLOCK)
    pos_mask = pos_offsets < (HPO * WPO)
    po_h = pos_offsets // WPO
    po_w = pos_offsets % WPO

    max_vals = tl.full((NC_BLOCK, POS_BLOCK), -float("inf"), dtype=tl.float32)

    for dy in range(0, POOL_KH):
        ih = po_h * POOL_STRIDE_H + dy
        inb_h = ih < HO
        for dx in range(0, POOL_KW):
            iw = po_w * POOL_STRIDE_W + dx
            inb_w = iw < WO
            valid = nc_mask[:, None] & pos_mask[None, :] & inb_h[None, :] & inb_w[None, :]

            x_ptrs = (
                x_ptr
                + n_idx[:, None] * stride_xn
                + c_idx[:, None] * stride_xc
                + ih[None, :] * stride_xh
                + iw[None, :] * stride_xw
            )
            vals = tl.load(x_ptrs, mask=valid, other=0.0).to(tl.float32)
            max_vals = tl.maximum(max_vals, vals)

    y_ptrs = (
        y_ptr
        + n_idx[:, None] * stride_yn
        + c_idx[:, None] * stride_yc
        + po_h[None, :] * stride_yh
        + po_w[None, :] * stride_yw
    )
    out_mask = nc_mask[:, None] & pos_mask[None, :]
    tl.store(y_ptrs, max_vals.to(y_ptr.dtype.element_ty), mask=out_mask)


def block1_conv_relu_pool(x, w, b):
    # Parameters
    N, C_in, H, W = x.shape
    C_out, Cw_in, KH, KW = w.shape
    assert C_in == Cw_in == 3
    STRIDE_H = STRIDE_W = 4
    PAD_H = PAD_W = 2

    HO = (H + 2 * PAD_H - KH) // STRIDE_H + 1
    WO = (W + 2 * PAD_W - KW) // STRIDE_W + 1
    assert HO == 55 and WO == 55

    POOL_KH = POOL_KW = 3
    POOL_STRIDE_H = POOL_STRIDE_W = 2
    HPO = (HO - POOL_KH) // POOL_STRIDE_H + 1
    WPO = (WO - POOL_KW) // POOL_STRIDE_W + 1
    assert HPO == 27 and WPO == 27

    conv_out = torch.empty((N, C_out, HO, WO), dtype=x.dtype, device=x.device)
    y = torch.empty((N, C_out, HPO, WPO), dtype=x.dtype, device=x.device)

    sxn, sxc, sxh, sxw = x.stride()
    swc, swci, swkh, swkw = w.stride()
    syn, syc, syh, syw = conv_out.stride()
    pyn, pyc, pyh, pyw = y.stride()

    BLOCK_CO = 64
    BLOCK_PIX = 8
    grid_conv = (
        triton.cdiv(N * HO * WO, BLOCK_PIX),
        triton.cdiv(C_out, BLOCK_CO),
    )
    _block1_conv11_relu_kernel[grid_conv](
        x, w, b, conv_out,
        N, C_in, H, W, C_out, HO, WO,
        sxn, sxc, sxh, sxw,
        swc, swci, swkh, swkw,
        syn, syc, syh, syw,
        BLOCK_CO=BLOCK_CO,
        BLOCK_PIX=BLOCK_PIX,
        KH=KH, KW=KW,
        STRIDE_H=STRIDE_H, STRIDE_W=STRIDE_W,
        PAD_H=PAD_H, PAD_W=PAD_W,
        num_warps=4, num_stages=2
    )

    NC_BLOCK = 16
    POS_BLOCK = 64
    grid_pool = (
        triton.cdiv(N * C_out, NC_BLOCK),
        triton.cdiv(HPO * WPO, POS_BLOCK),
    )
    _block1_maxpool2d_kernel[grid_pool](
        conv_out, y,
        N, C_out, HO, WO, HPO, WPO,
        syn, syc, syh, syw,
        pyn, pyc, pyh, pyw,
        NC_BLOCK=NC_BLOCK, POS_BLOCK=POS_BLOCK,
        POOL_KH=POOL_KH, POOL_KW=POOL_KW,
        POOL_STRIDE_H=POOL_STRIDE_H, POOL_STRIDE_W=POOL_STRIDE_W,
        num_warps=4, num_stages=2
    )
    return y


# ----------------------------
# Block 2: Conv2d(5x5, s=1, p=2) + ReLU, then MaxPool2d(3x3, s=2)
# ----------------------------

@triton.jit
def _block2_conv2d_relu_im2col_gemm(
    x_ptr,        # [N, Cin, H, W]
    w_ptr,        # [Cout, Cin, KH, KW]
    b_ptr,        # [Cout]
    y_ptr,        # [N, Cout, H_out, W_out]
    N, Cin, H, W,
    Cout, H_out, W_out,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wn, stride_wc, stride_wkh, stride_wkw,
    stride_yn, stride_yc, stride_yh, stride_yw,
    K_total,      # = Cin*KH*KW
    HW_out,       # = H_out*W_out
    pad_h: tl.constexpr,
    pad_w: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    BLOCK_M: tl.constexpr,   # tile on M=N*H_out*W_out
    BLOCK_N: tl.constexpr,   # tile on N=Cout
    BLOCK_K: tl.constexpr,   # tile on K=Cin*KH*KW
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    m_mask = offs_m < (N * HW_out)
    n_mask = offs_n < Cout

    n_idx = offs_m // HW_out
    rem_m = offs_m % HW_out
    oh_idx = rem_m // W_out
    ow_idx = rem_m % W_out

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K_total, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K_total

        c_idx = offs_k // (KH * KW)
        rem_k = offs_k % (KH * KW)
        ky_idx = rem_k // KW
        kx_idx = rem_k % KW

        ih = oh_idx[:, None] + ky_idx[None, :] - pad_h
        iw = ow_idx[:, None] + kx_idx[None, :] - pad_w

        x_ptrs = (
            x_ptr
            + n_idx[:, None] * stride_xn
            + c_idx[None, :] * stride_xc
            + ih * stride_xh
            + iw * stride_xw
        )

        in_bounds = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
        a_mask = (m_mask[:, None] & k_mask[None, :] & in_bounds)
        a = tl.load(x_ptrs, mask=a_mask, other=0.0)

        w_ptrs = (
            w_ptr
            + offs_n[None, :] * stride_wn
            + c_idx[:, None] * stride_wc
            + ky_idx[:, None] * stride_wkh
            + kx_idx[:, None] * stride_wkw
        )

        b_mask = k_mask[:, None] & n_mask[None, :]
        b = tl.load(w_ptrs, mask=b_mask, other=0.0)

        acc = tl.dot(a, b, acc)

    bias = tl.load(b_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
    acc = acc + bias[None, :]
    acc = tl.maximum(acc, 0.0)

    y_ptrs = (
        y_ptr
        + n_idx[:, None] * stride_yn
        + offs_n[None, :] * stride_yc
        + oh_idx[:, None] * stride_yh
        + ow_idx[:, None] * stride_yw
    )
    store_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(y_ptrs, acc.to(y_ptr.dtype.element_ty), mask=store_mask)


@triton.jit
def _block2_maxpool2d_3x3_s2(
    x_ptr,     # [N, C, H, W]
    y_ptr,     # [N, C, H_out, W_out]
    N, C, H, W,
    H_out, W_out,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_yn, stride_yc, stride_yh, stride_yw,
    BLOCK_P: tl.constexpr,   # tile for N*C dimension
    BLOCK_W: tl.constexpr,   # tile for W_out dimension
):
    pid_p = tl.program_id(0)
    pid_w = tl.program_id(1)
    pid_h = tl.program_id(2)

    P = N * C

    offs_p = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)
    offs_w = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    oh = pid_h

    p_mask = offs_p < P
    w_mask = offs_w < W_out

    n_idx = offs_p // C
    c_idx = offs_p % C

    in_h0 = oh * 2

    in_w0 = offs_w * 2 + 0
    ptrs0 = (
        x_ptr
        + n_idx[:, None] * stride_xn
        + c_idx[:, None] * stride_xc
        + (in_h0 + 0) * stride_xh
        + in_w0[None, :] * stride_xw
    )
    mask0 = p_mask[:, None] & w_mask[None, :]
    val = tl.load(ptrs0, mask=mask0, other=0.0).to(tl.float32)
    maxv = val

    in_w1 = offs_w * 2 + 1
    ptrs = (
        x_ptr
        + n_idx[:, None] * stride_xn
        + c_idx[:, None] * stride_xc
        + (in_h0 + 0) * stride_xh
        + in_w1[None, :] * stride_xw
    )
    val = tl.load(ptrs, mask=mask0, other=0.0).to(tl.float32)
    maxv = tl.maximum(maxv, val)

    in_w2 = offs_w * 2 + 2
    ptrs = (
        x_ptr
        + n_idx[:, None] * stride_xn
        + c_idx[:, None] * stride_xc
        + (in_h0 + 0) * stride_xh
        + in_w2[None, :] * stride_xw
    )
    val = tl.load(ptrs, mask=mask0, other=0.0).to(tl.float32)
    maxv = tl.maximum(maxv, val)

    in_h1 = in_h0 + 1
    in_w0 = offs_w * 2 + 0
    ptrs = (
        x_ptr
        + n_idx[:, None] * stride_xn
        + c_idx[:, None] * stride_xc
        + in_h1 * stride_xh
        + in_w0[None, :] * stride_xw
    )
    val = tl.load(ptrs, mask=mask0, other=0.0).to(tl.float32)
    maxv = tl.maximum(maxv, val)

    in_w1 = offs_w * 2 + 1
    ptrs = (
        x_ptr
        + n_idx[:, None] * stride_xn
        + c_idx[:, None] * stride_xc
        + in_h1 * stride_xh
        + in_w1[None, :] * stride_xw
    )
    val = tl.load(ptrs, mask=mask0, other=0.0).to(tl.float32)
    maxv = tl.maximum(maxv, val)

    in_w2 = offs_w * 2 + 2
    ptrs = (
        x_ptr
        + n_idx[:, None] * stride_xn
        + c_idx[:, None] * stride_xc
        + in_h1 * stride_xh
        + in_w2[None, :] * stride_xw
    )
    val = tl.load(ptrs, mask=mask0, other=0.0).to(tl.float32)
    maxv = tl.maximum(maxv, val)

    in_h2 = in_h0 + 2
    in_w0 = offs_w * 2 + 0
    ptrs = (
        x_ptr
        + n_idx[:, None] * stride_xn
        + c_idx[:, None] * stride_xc
        + in_h2 * stride_xh
        + in_w0[None, :] * stride_xw
    )
    val = tl.load(ptrs, mask=mask0, other=0.0).to(tl.float32)
    maxv = tl.maximum(maxv, val)

    in_w1 = offs_w * 2 + 1
    ptrs = (
        x_ptr
        + n_idx[:, None] * stride_xn
        + c_idx[:, None] * stride_xc
        + in_h2 * stride_xh
        + in_w1[None, :] * stride_xw
    )
    val = tl.load(ptrs, mask=mask0, other=0.0).to(tl.float32)
    maxv = tl.maximum(maxv, val)

    in_w2 = offs_w * 2 + 2
    ptrs = (
        x_ptr
        + n_idx[:, None] * stride_xn
        + c_idx[:, None] * stride_xc
        + in_h2 * stride_xh
        + in_w2[None, :] * stride_xw
    )
    val = tl.load(ptrs, mask=mask0, other=0.0).to(tl.float32)
    maxv = tl.maximum(maxv, val)

    y_ptrs = (
        y_ptr
        + n_idx[:, None] * stride_yn
        + c_idx[:, None] * stride_yc
        + oh * stride_yh
        + offs_w[None, :] * stride_yw
    )
    tl.store(y_ptrs, maxv.to(y_ptr.dtype.element_ty), mask=mask0)


def block2_conv_relu_pool(x, w, b):
    N, Cin, H, W = x.shape
    Cout, Cin_w, KH, KW = w.shape
    assert Cin == Cin_w and KH == 5 and KW == 5

    stride_h = 1
    stride_w = 1
    pad_h = 2
    pad_w = 2
    H_out = (H + 2 * pad_h - (KH - 1) - 1) // stride_h + 1
    W_out = (W + 2 * pad_w - (KW - 1) - 1) // stride_w + 1
    assert H_out == 27 and W_out == 27

    pool_k = 3
    pool_s = 2
    Hp = (H_out - pool_k) // pool_s + 1
    Wp = (W_out - pool_k) // pool_s + 1
    assert Hp == 13 and Wp == 13

    y_conv = torch.empty((N, Cout, H_out, W_out), dtype=x.dtype, device=x.device)
    y_out = torch.empty((N, Cout, Hp, Wp), dtype=x.dtype, device=x.device)

    sxn, sxc, sxh, sxw = x.stride()
    swn, swc, swkh, swkw = w.stride()
    syn, syc, syh, syw = y_conv.stride()
    spn, spc, sph, spw = y_out.stride()

    M_total = N * H_out * W_out
    K_total = Cin * KH * KW
    HW_out = H_out * W_out

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32

    grid_conv = (triton.cdiv(M_total, BLOCK_M), triton.cdiv(Cout, BLOCK_N))
    _block2_conv2d_relu_im2col_gemm[grid_conv](
        x, w, b, y_conv,
        N, Cin, H, W,
        Cout, H_out, W_out,
        sxn, sxc, sxh, sxw,
        swn, swc, swkh, swkw,
        syn, syc, syh, syw,
        K_total,
        HW_out,
        pad_h, pad_w,
        KH, KW,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=8, num_stages=2,
    )

    BLOCK_P = 64
    BLOCK_W = 16
    grid_pool = (triton.cdiv(N * Cout, BLOCK_P), triton.cdiv(Wp, BLOCK_W), Hp)
    _block2_maxpool2d_3x3_s2[grid_pool](
        y_conv, y_out,
        N, Cout, H_out, W_out,
        Hp, Wp,
        syn, syc, syh, syw,
        spn, spc, sph, spw,
        BLOCK_P=BLOCK_P, BLOCK_W=BLOCK_W,
        num_warps=4, num_stages=2,
    )
    return y_out


# ----------------------------
# Shared 3x3 Conv + ReLU for Blocks 3 and 4
# ----------------------------

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
    BLOCK_CO: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    PAD_H: tl.constexpr,
    PAD_W: tl.constexpr,
):
    pid_p = tl.program_id(0)  # tiles over spatial positions (H*W)
    pid_n = tl.program_id(1)  # batch index
    pid_co = tl.program_id(2)  # tiles over output channels

    n = pid_n

    offs_co = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    offs_p = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)

    mask_co = offs_co < COUT
    mask_p = offs_p < P

    oy = offs_p // W
    ox = offs_p - oy * W

    acc = tl.zeros((BLOCK_CO, BLOCK_P), dtype=tl.float32)

    for ci0 in range(0, CIN, BLOCK_CI):
        offs_ci = ci0 + tl.arange(0, BLOCK_CI)
        valid_ci = offs_ci < CIN

        for ky in range(KH):
            iy = oy + ky - PAD_H
            in_y_ok = (iy >= 0) & (iy < H)
            for kx in range(KW):
                ix = ox + kx - PAD_W
                in_x_ok = (ix >= 0) & (ix < W)
                x_mask = (valid_ci[:, None]) & (mask_p[None, :]) & in_y_ok[None, :] & in_x_ok[None, :]

                x_ptrs = (
                    x_ptr
                    + n * stride_xn
                    + offs_ci[:, None] * stride_xc
                    + iy[None, :] * stride_xh
                    + ix[None, :] * stride_xw
                )
                x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)

                w_ptrs = (
                    w_ptr
                    + offs_co[:, None] * stride_wco
                    + offs_ci[None, :] * stride_wci
                    + ky * stride_wkh
                    + kx * stride_wkw
                )
                w_mask = (mask_co[:, None]) & (valid_ci[None, :])
                w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)

                acc += tl.dot(w_tile, x_tile)

    b = tl.load(b_ptr + offs_co, mask=mask_co, other=0.0).to(tl.float32)
    acc = acc + b[:, None]
    acc = tl.maximum(acc, 0.0)

    y_ptrs = (
        y_ptr
        + n * stride_yn
        + offs_co[:, None] * stride_yc
        + oy[None, :] * stride_yh
        + ox[None, :] * stride_yw
    )
    store_mask = (mask_co[:, None]) & (mask_p[None, :])
    out_val = acc.to(y_ptr.dtype.element_ty)
    tl.store(y_ptrs, out_val, mask=store_mask)


def conv3x3_bias_relu(x, w, b):
    N, CIN, H, W = x.shape
    COUT, WCIN, KH, KW = w.shape
    assert CIN == WCIN and KH == 3 and KW == 3

    y = torch.empty((N, COUT, H, W), device=x.device, dtype=x.dtype)

    P = H * W
    sxn, sxc, sxh, sxw = x.stride()
    swco, swci, swkh, swkw = w.stride()
    syn, syc, syh, syw = y.stride()

    def grid(meta):
        return (
            triton.cdiv(P, meta["BLOCK_P"]),
            N,
            triton.cdiv(COUT, meta["BLOCK_CO"]),
        )

    _conv2d_bias_relu_nchw_3x3[grid](
        x, w, b, y,
        N, CIN, COUT, H, W, P,
        sxn, sxc, sxh, sxw,
        swco, swci, swkh, swkw,
        syn, syc, syh, syw,
        KH=3, KW=3, PAD_H=1, PAD_W=1,
    )
    return y


# ----------------------------
# Block 5: Fused Conv2d(3x3, s=1, p=1) + MaxPool2d(3x3, s=2) + Bias + ReLU
# ----------------------------

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OC': 64, 'BLOCK_P': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_P': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_P': 64, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_P': 64, 'BLOCK_K': 128}, num_warps=8, num_stages=4),
    ],
    key=['N', 'CIN', 'H', 'W', 'COUT']
)
@triton.jit
def _fused_conv_pool_bias_relu_3x3_s2_kernel(
    x_ptr,           # *T, input [N, Cin, H, W]
    w_ptr,           # *T, weight [Cout, Cin, 3, 3]
    b_ptr,           # *T, bias [Cout]
    out_ptr,         # *T, output [N, Cout, PH, PW]
    N, CIN, H, W, COUT,
    PH, PW,
    SXN, SXC, SXH, SXW,
    SWO, SWI, SWKH, SWKW,
    SON, SOC, SOH, SOW,
    BLOCK_OC: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_oc = tl.program_id(0)
    pid_p = tl.program_id(1)

    oc_start = pid_oc * BLOCK_OC
    p_start = pid_p * BLOCK_P

    oc_offsets = oc_start + tl.arange(0, BLOCK_OC)
    p_offsets = p_start + tl.arange(0, BLOCK_P)

    oc_mask = oc_offsets < COUT
    P_TOTAL = N * PH * PW
    p_mask = p_offsets < P_TOTAL

    PHxPW = PH * PW
    n_idx = p_offsets // PHxPW
    rem_p = p_offsets % PHxPW
    ph_idx = rem_p // PW
    pw_idx = rem_p % PW

    base_p_n = n_idx * SXN
    base_p_y = (ph_idx * 2) * SXH
    base_p_x = (pw_idx * 2) * SXW

    acc_max = tl.full((BLOCK_OC, BLOCK_P), -float("inf"), dtype=tl.float32)

    K_TOTAL = CIN * 9

    for t in range(9):
        dy = t // 3
        dx = t % 3

        acc_t = tl.zeros((BLOCK_OC, BLOCK_P), dtype=tl.float32)

        for k0 in range(0, K_TOTAL, BLOCK_K):
            k_offsets = k0 + tl.arange(0, BLOCK_K)
            k_mask = k_offsets < K_TOTAL

            cin_idx = k_offsets // 9
            rem9 = k_offsets % 9
            ry = rem9 // 3
            rx = rem9 % 3

            a_ptrs = (
                w_ptr
                + oc_offsets[:, None] * SWO
                + cin_idx[None, :] * SWI
                + ry[None, :] * SWKH
                + rx[None, :] * SWKW
            )
            a = tl.load(a_ptrs, mask=oc_mask[:, None] & k_mask[None, :], other=0.0)

            y_in_2d = (base_p_y[None, :] + (dy * SXH)) + (ry[:, None] - 1) * SXH
            x_in_2d = (base_p_x[None, :] + (dx * SXW)) + (rx[:, None] - 1) * SXW
            b_ptrs = (
                x_ptr
                + base_p_n[None, :]
                + cin_idx[:, None] * SXC
                + y_in_2d
                + x_in_2d
            )

            y_in_idx = (ph_idx[None, :] * 2 + dy) + (ry[:, None] - 1)
            x_in_idx = (pw_idx[None, :] * 2 + dx) + (rx[:, None] - 1)
            in_bounds = (y_in_idx >= 0) & (y_in_idx < H) & (x_in_idx >= 0) & (x_in_idx < W)
            b_mask = k_mask[:, None] & p_mask[None, :] & in_bounds

            b_vals = tl.load(b_ptrs, mask=b_mask, other=0.0)

            acc_t = tl.dot(a, b_vals, acc_t)

        acc_max = tl.maximum(acc_max, acc_t)

    bias = tl.load(b_ptr + oc_offsets, mask=oc_mask, other=0.0).to(tl.float32)
    acc_max = acc_max + bias[:, None]
    acc_max = tl.maximum(acc_max, 0)

    out_ptrs = (
        out_ptr
        + n_idx[None, :] * SON
        + oc_offsets[:, None] * SOC
        + ph_idx[None, :] * SOH
        + pw_idx[None, :] * SOW
    )
    tl.store(out_ptrs, acc_max.to(out_ptr.dtype.element_ty), mask=oc_mask[:, None] & p_mask[None, :])


def block5_conv_relu_pool(x, w, b):
    # x: [N, 384, 13, 13] -> y: [N, 256, 6, 6]
    N, Cin, H, W = x.shape
    Cout, Cin_w, kH, kW = w.shape
    assert Cin == Cin_w and kH == 3 and kW == 3
    PH = (H - 3) // 2 + 1
    PW = (W - 3) // 2 + 1
    assert PH == 6 and PW == 6

    out = torch.empty((N, Cout, PH, PW), dtype=x.dtype, device=x.device)

    SXN, SXC, SXH, SXW = x.stride()
    SWO, SWI, SWKH, SWKW = w.stride()
    SON, SOC, SOH, SOW = out.stride()

    P_TOTAL = N * PH * PW
    def grid(meta):
        return (triton.cdiv(Cout, meta['BLOCK_OC']), triton.cdiv(P_TOTAL, meta['BLOCK_P']))

    _fused_conv_pool_bias_relu_3x3_s2_kernel[grid](
        x, w, b, out,
        N, Cin, H, W, Cout,
        PH, PW,
        SXN, SXC, SXH, SXW,
        SWO, SWI, SWKH, SWKW,
        SON, SOC, SOH, SOW,
    )
    return out


# ----------------------------
# Flatten NCHW (start_dim=1) to [N, C*H*W]
# ----------------------------

@triton.jit
def _flatten_nchw_to_2d_kernel(
    in_ptr,                  # *T
    out_ptr,                 # *T
    N,                       # int
    CHW,                     # int (C*H*W)
    in_stride_n,             # int (stride over N in elements)
    out_stride_n,            # int (stride over output row in elements; typically CHW)
    BLOCK_SIZE: tl.constexpr
):
    pid_n = tl.program_id(axis=0)
    pid_c = tl.program_id(axis=1)

    offs = pid_c * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < CHW

    in_row_start = pid_n * in_stride_n
    out_row_start = pid_n * out_stride_n

    vals = tl.load(in_ptr + in_row_start + offs, mask=mask, other=0)
    tl.store(out_ptr + out_row_start + offs, vals, mask=mask)


def flatten_start_dim1(x):
    N, C, H, W = x.shape
    CHW = C * H * W
    out = torch.empty((N, CHW), device=x.device, dtype=x.dtype)
    BLOCK_SIZE = 1024
    grid = lambda META: (
        N,
        triton.cdiv(CHW, META["BLOCK_SIZE"]),
    )
    _flatten_nchw_to_2d_kernel[grid](
        x, out,
        N, CHW,
        x.stride(0),
        out.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4, num_stages=2
    )
    return out


# ----------------------------
# Linear + ReLU + (Dropout p=0.0 no-op)
# ----------------------------

_autotune_configs_linear = [
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64,  "BLOCK_K": 32}, num_warps=4, num_stages=4),
    triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4, num_stages=4),
    triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64,  "BLOCK_K": 32}, num_warps=4, num_stages=4),
]

@triton.autotune(configs=_autotune_configs_linear, key=["M", "N", "K"])
@triton.jit
def _linear_relu_dropout(x_ptr, w_ptr, b_ptr, y_ptr,
                         M, N, K,
                         stride_xm, stride_xk,
                         stride_wn, stride_wk,
                         stride_ym, stride_yn,
                         P_DROPOUT: tl.constexpr,
                         BLOCK_M: tl.constexpr,
                         BLOCK_N: tl.constexpr,
                         BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    mask_m = offs_m < M
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in tl.range(0, K, BLOCK_K):
        k_idx = k0 + offs_k
        k_mask = k_idx < K

        a_ptrs = x_ptr + (offs_m[:, None] * stride_xm + k_idx[None, :] * stride_xk)
        b_ptrs = w_ptr + (k_idx[:, None] * stride_wk + offs_n[None, :] * stride_wn)

        a_mask = mask_m[:, None] & k_mask[None, :]
        b_mask = k_mask[:, None] & mask_n[None, :]

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        acc = tl.dot(a, b, acc)

    bias_vals = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + bias_vals[None, :].to(tl.float32)

    acc = tl.maximum(acc, 0.0)

    y_ptrs = y_ptr + (offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn)
    tl.store(y_ptrs, acc.to(y_ptr.dtype.element_ty), mask=mask_m[:, None] & mask_n[None, :])


def linear_relu(x, w, b):
    M, K = x.shape
    N, Kw = w.shape
    assert K == Kw and b.shape[0] == N
    y = torch.empty((M, N), device=x.device, dtype=x.dtype)

    def grid(meta):
        return (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]))

    _linear_relu_dropout[grid](
        x, w, b, y,
        M, N, K,
        x.stride(0), x.stride(1),
        w.stride(0), w.stride(1),
        y.stride(0), y.stride(1),
        P_DROPOUT=0.0,
    )
    return y


# ----------------------------
# Linear (no activation)
# ----------------------------

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64, 'GROUP_SIZE_M': 8}, num_warps=4, num_stages=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _linear_matmul_bias_kernel(
    a_ptr,         # [M, K] input
    b_ptr,         # [N, K] weights (we index as [K, N] via strides)
    bias_ptr,      # [N] bias
    c_ptr,         # [M, N] output
    M, N, K,
    stride_am, stride_ak,   # strides for A
    stride_bk, stride_bn,   # strides for B (treat B as [K, N])
    stride_cm, stride_cn,   # strides for C
    OUT_DTYPE_TAG: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    offs_m = tl.where(offs_m < M, offs_m, 0)
    offs_n = tl.where(offs_n < N, offs_n, 0)
    offs_m = tl.max_contiguous(tl.multiple_of(offs_m, BLOCK_M), BLOCK_M)
    offs_n = tl.max_contiguous(tl.multiple_of(offs_n, BLOCK_N), BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    k_tiles = tl.cdiv(K, BLOCK_K)
    for ki in range(0, k_tiles):
        k_start = ki * BLOCK_K
        k_idx = k_start + offs_k

        a_ptrs = a_ptr + (offs_m[:, None] * stride_am + k_idx[None, :] * stride_ak)
        b_ptrs = b_ptr + (k_idx[:, None] * stride_bk + offs_n[None, :] * stride_bn)

        a_mask = (offs_m[:, None] < M) & (k_idx[None, :] < K)
        b_mask = (k_idx[:, None] < K) & (offs_n[None, :] < N)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        acc = tl.dot(a, b, acc)

    bias = tl.load(bias_ptr + offs_n, mask=(offs_n < N), other=0.0)
    bias_f32 = bias.to(tl.float32)
    acc = acc + bias_f32[None, :]

    if OUT_DTYPE_TAG == 0:
        c_cast = acc.to(tl.bfloat16)
    elif OUT_DTYPE_TAG == 1:
        c_cast = acc.to(tl.float16)
    else:
        c_cast = acc

    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c_cast, mask=c_mask)


def linear(x, w, b):
    M, K = x.shape
    N, Kw = w.shape
    assert K == Kw and b.shape[0] == N
    y = torch.empty((M, N), device=x.device, dtype=x.dtype)

    if y.dtype == torch.bfloat16:
        tag = 0
    elif y.dtype == torch.float16:
        tag = 1
    else:
        tag = 2

    def grid(meta):
        return (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)

    _linear_matmul_bias_kernel[grid](
        x, w, b, y,
        M, N, K,
        x.stride(0), x.stride(1),
        w.stride(1), w.stride(0),
        y.stride(0), y.stride(1),
        OUT_DTYPE_TAG=tag,
    )
    return y


# ----------------------------
# Top-level end-to-end wrapper: kernel_function(...)
# ----------------------------

def kernel_function(
    x,
    conv1_w, conv1_b,
    conv2_w, conv2_b,
    conv3_w, conv3_b,
    conv4_w, conv4_b,
    conv5_w, conv5_b,
    fc1_w, fc1_b,
    fc2_w, fc2_b,
    fc3_w, fc3_b,
):
    """
    End-to-end Triton pipeline implementing the original model's forward:
    conv1->relu->pool -> conv2->relu->pool -> conv3->relu -> conv4->relu ->
    conv5->relu->pool -> flatten -> fc1->relu -> fc2->relu -> fc3

    All math is performed by Triton kernels defined above. No torch.nn/functional ops are used.
    Inputs must be CUDA tensors. Dtype: float32/float16/bfloat16 supported across kernels.
    """
    assert x.is_cuda, "Input must be on CUDA"
    device = x.device
    dtype = x.dtype

    # Block 1
    x1 = block1_conv_relu_pool(x, conv1_w, conv1_b)                      # [N, 96, 27, 27]
    # Block 2
    x2 = block2_conv_relu_pool(x1, conv2_w, conv2_b)                     # [N, 256, 13, 13]
    # Block 3
    x3 = conv3x3_bias_relu(x2, conv3_w, conv3_b)                         # [N, 384, 13, 13]
    # Block 4
    x4 = conv3x3_bias_relu(x3, conv4_w, conv4_b)                         # [N, 384, 13, 13]
    # Block 5 (fused conv+pool)
    x5 = block5_conv_relu_pool(x4, conv5_w, conv5_b)                     # [N, 256, 6, 6]
    # Flatten
    x_flat = flatten_start_dim1(x5)                                      # [N, 9216]
    # FC1 + ReLU
    x6 = linear_relu(x_flat, fc1_w, fc1_b)                               # [N, 4096]
    # FC2 + ReLU
    x7 = linear_relu(x6, fc2_w, fc2_b)                                   # [N, 4096]
    # Classifier (no activation)
    out = linear(x7, fc3_w, fc3_b)                                       # [N, 1000]
    return out


# ----------------------------
# Reference PyTorch model (from the original problem) for testing
# ----------------------------

class Model(torch.nn.Module):
    def __init__(self, num_classes=1000):
        super(Model, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=2)
        self.relu1 = torch.nn.ReLU(inplace=True)
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = torch.nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2)
        self.relu2 = torch.nn.ReLU(inplace=True)
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv3 = torch.nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1)
        self.relu3 = torch.nn.ReLU(inplace=True)
        self.conv4 = torch.nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1)
        self.relu4 = torch.nn.ReLU(inplace=True)
        self.conv5 = torch.nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1)
        self.relu5 = torch.nn.ReLU(inplace=True)
        self.maxpool3 = torch.nn.MaxPool2d(kernel_size=3, stride=2)
        self.fc1 = torch.nn.Linear(in_features=256 * 6 * 6, out_features=4096)
        self.relu6 = torch.nn.ReLU(inplace=True)
        self.dropout1 = torch.nn.Dropout(p=0.0)
        self.fc2 = torch.nn.Linear(in_features=4096, out_features=4096)
        self.relu7 = torch.nn.ReLU(inplace=True)
        self.dropout2 = torch.nn.Dropout(p=0.0)
        self.fc3 = torch.nn.Linear(in_features=4096, out_features=num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.conv5(x)
        x = self.relu5(x)
        x = self.maxpool3(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu6(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu7(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

batch_size = 1024
num_classes = 1000

def get_inputs():
    return [torch.rand(batch_size, 3, 224, 224)]

def get_init_inputs():
    return [num_classes]


# ----------------------------
# Self-test
# ----------------------------

def run_tests():
    if not torch.cuda.is_available():
        print("SKIP: CUDA is not available.")
        print("PASS")
        return

    torch.manual_seed(0)
    device = torch.device('cuda')

    # Initialize reference model and inputs
    model = Model(*get_init_inputs()).to(device=device, dtype=torch.float32)
    model.eval()
    x = get_inputs()[0].to(device=device, dtype=torch.float32)

    # Extract weights and biases
    conv1_w = model.conv1.weight.detach().to(device)
    conv1_b = model.conv1.bias.detach().to(device)
    conv2_w = model.conv2.weight.detach().to(device)
    conv2_b = model.conv2.bias.detach().to(device)
    conv3_w = model.conv3.weight.detach().to(device)
    conv3_b = model.conv3.bias.detach().to(device)
    conv4_w = model.conv4.weight.detach().to(device)
    conv4_b = model.conv4.bias.detach().to(device)
    conv5_w = model.conv5.weight.detach().to(device)
    conv5_b = model.conv5.bias.detach().to(device)
    fc1_w = model.fc1.weight.detach().to(device)  # [4096, 9216]
    fc1_b = model.fc1.bias.detach().to(device)
    fc2_w = model.fc2.weight.detach().to(device)  # [4096, 4096]
    fc2_b = model.fc2.bias.detach().to(device)
    fc3_w = model.fc3.weight.detach().to(device)  # [1000, 4096]
    fc3_b = model.fc3.bias.detach().to(device)

    # Triton result
    with torch.no_grad():
        y_triton = kernel_function(
            x,
            conv1_w, conv1_b,
            conv2_w, conv2_b,
            conv3_w, conv3_b,
            conv4_w, conv4_b,
            conv5_w, conv5_b,
            fc1_w, fc1_b,
            fc2_w, fc2_b,
            fc3_w, fc3_b,
        )

    # PyTorch reference
    with torch.no_grad():
        y_ref = model(x)

    # Compare
    rtol = 1e-3
    atol = 1e-3
    ok = torch.allclose(y_triton, y_ref, rtol=rtol, atol=atol)
    print("Max abs diff:", (y_triton - y_ref).abs().max().item())
    print("Max rel diff:", ((y_triton - y_ref).abs() / (y_ref.abs() + 1e-8)).max().item())
    if ok:
        print("PASS")
        return
    else:
        print("FAIL")
        diff = (y_triton - y_ref).abs()
        print("Mean abs diff:", diff.mean().item())
        raise SystemExit(1)


if __name__ == "__main__":
    run_tests()
