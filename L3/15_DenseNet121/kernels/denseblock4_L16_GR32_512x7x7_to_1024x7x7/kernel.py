import torch
import triton
import triton.language as tl

# -------------------------------------------------------------------------------------
# Fused DenseBlock layer kernel:
# - Per layer ops: BatchNorm (inference) -> ReLU -> Conv2d(3x3, stride=1, pad=1, no bias)
# - Layout: NCHW (contiguous)
# - DType: bf16/fp16/fp32 inputs supported; math in fp32; stores cast back to output dtype.
# - This kernel computes a single spatial position (n, y, x) for BLOCK_CO output channels,
#   accumulating over all input channels (in BLOCK_CI tiles) and the 3x3 neighborhood.
#
# Notes:
# - y_ptr is the base pointer to the full output tensor; C_OUT_BASE is a channel offset
#   for this layer's 32 new output channels. Do NOT pre-offset y_ptr in Python; pass the
#   base pointer and offset as scalars to avoid unintended PyTorch ops/allocations.
# -------------------------------------------------------------------------------------

@triton.jit
def _conv3x3_bn_relu_kernel(
    x_ptr,                  # *T: input buffer containing cumulative features [N, C_MAX, H, W]
    gamma_ptr,              # *T: BN gamma [C_IN]
    beta_ptr,               # *T: BN beta  [C_IN]
    rm_ptr,                 # *T: BN running mean [C_IN]
    rv_ptr,                 # *T: BN running var  [C_IN]
    w_ptr,                  # *T: conv weights [C_OUT, C_IN, 3, 3] (OIHW contiguous)
    y_ptr,                  # *T: base output pointer (same storage as x_ptr)
    N, C_IN, C_OUT, H, W,   # sizes
    stride_xn, stride_xc, stride_xh, stride_xw,   # strides for x (NCHW)
    stride_wo, stride_wi, stride_wkh, stride_wkw, # strides for w (OIHW)
    stride_yn, stride_yc, stride_yh, stride_yw,   # strides for y (NCHW)
    C_OUT_BASE,             # channel offset in y where this layer stores its outputs
    eps,                    # BN epsilon
    BLOCK_CI: tl.constexpr, # tile size along input channels
    BLOCK_CO: tl.constexpr, # tile size along output channels (use 32 here)
):
    # Tile IDs
    pid_hw = tl.program_id(axis=0)  # over N*H*W
    pid_co = tl.program_id(axis=1)  # over output channel tiles

    # Decode N/H/W position
    NHW = H * W
    n = pid_hw // NHW
    hw = pid_hw % NHW
    oy = hw // W
    ox = hw % W

    # Output channel tile
    oc_start = pid_co * BLOCK_CO
    oc_offsets = oc_start + tl.arange(0, BLOCK_CO)
    oc_mask = oc_offsets < C_OUT

    # Accumulator for BLOCK_CO output channels at a single (n, oy, ox)
    acc = tl.zeros((BLOCK_CO,), dtype=tl.float32)

    # Iterate 3x3 spatial window (padding=1)
    for ky in range(3):
        iy = oy + (ky - 1)
        inb_y = (iy >= 0) & (iy < H)
        for kx in range(3):
            ix = ox + (kx - 1)
            inb = inb_y & (ix >= 0) & (ix < W)

            # Iterate over input channels in tiles
            for ci_start in range(0, C_IN, BLOCK_CI):
                ci_offsets = ci_start + tl.arange(0, BLOCK_CI)
                ci_mask = ci_offsets < C_IN

                # Load BN params for current ci tile
                g = tl.load(gamma_ptr + ci_offsets, mask=ci_mask, other=0.0).to(tl.float32)
                b = tl.load(beta_ptr + ci_offsets, mask=ci_mask, other=0.0).to(tl.float32)
                rm = tl.load(rm_ptr + ci_offsets, mask=ci_mask, other=0.0).to(tl.float32)
                rv = tl.load(rv_ptr + ci_offsets, mask=ci_mask, other=0.0).to(tl.float32)
                inv_std = 1.0 / tl.sqrt(rv + eps)

                # Load input features X[n, ci, iy, ix]
                x_ptrs = x_ptr + n * stride_xn + ci_offsets * stride_xc + iy * stride_xh + ix * stride_xw
                xvals = tl.load(x_ptrs, mask=(ci_mask & inb), other=0.0).to(tl.float32)

                # BN + ReLU
                yvals = (xvals - rm) * inv_std
                yvals = yvals * g + b
                yvals = tl.maximum(yvals, 0.0)

                # Zero-out if spatially out-of-bounds (padding)
                yvals = yvals * tl.where(inb, 1.0, 0.0)

                # Load weight tile [BLOCK_CO, BLOCK_CI] for this (ky, kx)
                w_ptrs = w_ptr + oc_offsets[:, None] * stride_wo + ci_offsets[None, :] * stride_wi \
                                  + ky * stride_wkh + kx * stride_wkw
                wvals = tl.load(w_ptrs, mask=(oc_mask[:, None] & ci_mask[None, :]), other=0.0).to(tl.float32)

                # Accumulate over input channels in this tile
                prod = tl.sum(wvals * yvals[None, :], axis=1)
                acc += prod

    # Store accumulated outputs to y at channel offset C_OUT_BASE
    y_ptrs = y_ptr + n * stride_yn + (oc_offsets + C_OUT_BASE) * stride_yc + oy * stride_yh + ox * stride_yw
    tl.store(y_ptrs, acc.to(y_ptr.dtype.element_ty), mask=oc_mask)


# NCHW copy kernel to place src into dst[:, c_dst_offset:c_dst_offset+C]
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


def _is_list_of_tensors(lst, dim=None):
    if not isinstance(lst, (list, tuple)):
        return False
    for t in lst:
        if not isinstance(t, torch.Tensor):
            return False
        if dim is not None and t.ndim != dim:
            return False
    return True


def kernel_function(*args, **kwargs):
    """
    Fused DenseBlock (16 layers, growth_rate=32) for NCHW tensors using Triton kernels.
    For each layer:
      - BatchNorm (inference, using running mean/var, gamma/beta) in fp32
      - ReLU
      - Conv2d 3x3, stride=1, padding=1, groups=1, no bias
    New 32-channel features are appended along C; the cumulative buffer is used as input
    for subsequent layers.

    Expected call pattern (used by the test):
      kernel_function(x, conv_ws, bn_ws, bn_bs, bn_rm, bn_rv, p=0.0, eps=1e-5)

    Runtime rules:
      - Only the wrapper validates/allocates/launches.
      - All math is inside Triton kernels; no torch.nn.functional calls.
    """
    if len(args) < 6:
        raise TypeError("Expected at least 6 positional arguments: x, conv_ws, bn_ws, bn_bs, bn_rm, bn_rv.")
    x = args[0]
    conv_ws = args[1]
    bn_ws = args[2]
    bn_bs = args[3]
    bn_rm = args[4]
    bn_rv = args[5]
    p = float(kwargs.get("p", args[6] if len(args) >= 7 else 0.0))
    eps = float(kwargs.get("eps", args[7] if len(args) >= 8 else 1e-5))

    # Basic validation
    assert isinstance(x, torch.Tensor) and x.is_cuda, "x must be a CUDA tensor"
    assert x.ndim == 4 and x.shape[1:] == (512, 7, 7), f"Expected x shape [*, 512, 7, 7], got {tuple(x.shape)}"
    assert _is_list_of_tensors(conv_ws, dim=4) and len(conv_ws) == 16, "conv_ws must be a list of 16 [32, C_in, 3, 3] tensors"
    assert _is_list_of_tensors(bn_ws, dim=1) and len(bn_ws) == 16, "bn_ws must be a list of 16 vectors"
    assert _is_list_of_tensors(bn_bs, dim=1) and len(bn_bs) == 16, "bn_bs must be a list of 16 vectors"
    assert _is_list_of_tensors(bn_rm, dim=1) and len(bn_rm) == 16, "bn_rm must be a list of 16 vectors"
    assert _is_list_of_tensors(bn_rv, dim=1) and len(bn_rv) == 16, "bn_rv must be a list of 16 vectors"
    assert all(w.is_cuda for w in conv_ws), "All conv weights must be on CUDA"
    assert all(t.is_cuda for t in (bn_ws + bn_bs + bn_rm + bn_rv)), "All BN params must be on CUDA"
    assert p == 0.0, "Dropout with p != 0.0 is not supported; test uses p=0.0"

    device = x.device
    dtype = x.dtype
    N, C0, H, W = x.shape
    layers = 16
    growth = 32
    C_out_total = C0 + growth * layers

    # Allocate final output buffer and copy input x into the first channels
    out = torch.empty((N, C_out_total, H, W), device=device, dtype=dtype)

    stride_yn, stride_yc, stride_yh, stride_yw = out.stride()
    stride_xn, stride_xc, stride_xh, stride_xw = x.stride()

    # Copy input into out[:, :C0]
    BLOCK_C = 64
    BLOCK_HW = 64
    grid_copy = (N, triton.cdiv(C0, BLOCK_C), triton.cdiv(H * W, BLOCK_HW))
    _copy_nchw_kernel[grid_copy](
        x, out,
        N, C0, H, W,
        stride_xn, stride_xc, stride_xh, stride_xw,
        stride_yn, stride_yc, stride_yh, stride_yw,
        0,  # destination channel offset
        BLOCK_C=BLOCK_C,
        BLOCK_HW=BLOCK_HW,
        num_warps=4,
    )

    # Per-layer fused BN+ReLU+Conv3x3
    BLOCK_CI = 64  # tile size over input channels
    BLOCK_CO = 32  # growth rate; one tile covers all OC
    for i in range(layers):
        C_in = C0 + growth * i
        C_out = growth
        w = conv_ws[i]
        gamma = bn_ws[i]
        beta = bn_bs[i]
        rm = bn_rm[i]
        rv = bn_rv[i]
        # Validate shapes quickly
        assert w.ndim == 4 and w.shape[0] == C_out and w.shape[1] == C_in and w.shape[2:] == (3, 3), \
            f"conv_ws[{i}] expected shape {(C_out, C_in, 3, 3)}, got {tuple(w.shape)}"
        assert gamma.numel() == C_in and beta.numel() == C_in and rm.numel() == C_in and rv.numel() == C_in, \
            f"BN params at layer {i} must have length {C_in}"

        stride_wo, stride_wi, stride_wkh, stride_wkw = w.stride()
        c_dest_offset = C0 + growth * i

        # Grid: one program per (n, y, x), and tiles over output channels
        grid_hw = N * H * W
        grid_co = triton.cdiv(C_out, BLOCK_CO)
        _conv3x3_bn_relu_kernel[(grid_hw, grid_co)](
            out,                               # read from cumulative buffer
            gamma, beta, rm, rv,               # BN params (inference)
            w,                                 # conv weights
            out,                               # base out pointer (no pre-offset!)
            N, C_in, C_out, H, W,
            stride_yn, stride_yc, stride_yh, stride_yw,   # x strides (same as out)
            stride_wo, stride_wi, stride_wkh, stride_wkw, # weight strides
            stride_yn, stride_yc, stride_yh, stride_yw,   # y strides
            c_dest_offset,                     # channel offset for this layer's outputs
            eps,
            BLOCK_CI=BLOCK_CI,
            BLOCK_CO=BLOCK_CO,
            num_warps=4,
        )

    return out