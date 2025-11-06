import torch
import triton
import triton.language as tl
import math
import sys
import os


# ======================================================================================
# Generic grouped Conv2D (NCHW) with optional BatchNorm, activation, and residual add.
# - Supports arbitrary groups (including depthwise: groups=C_in=C_out).
# - Supports stride, padding, dilation.
# - Works on fp32/bf16/fp16 inputs; accumulates in fp32; casts to output dtype on store.
# - BN parameters are per-output-channel; residual is added AFTER BN and BEFORE activation.
# Tiling strategy:
#   - grid0 tiles spatial positions (H_out * W_out) by BLOCK_P
#   - grid1 iterates output channels (one oc per program)
#   - grid2 iterates batch
# This design is simple and robust for correctness-focused implementations.
# ======================================================================================

@triton.jit
def _conv2d_nchw_grouped_fused(
    x_ptr, w_ptr, bias_ptr, y_ptr,
    # sizes
    N, C_IN, H, W, C_OUT, H_OUT, W_OUT,
    GROUPS, C_IN_PER_G, C_OUT_PER_G,
    # hyper-params
    STRIDE_H, STRIDE_W, PAD_H, PAD_W, DIL_H, DIL_W,
    # strides
    SXN, SXC, SXH, SXW,   # input strides
    SWO, SWI, SWKH, SWKW, # weight strides: [C_OUT, C_IN_PER_G, KH, KW]
    SYN, SYC, SYH, SYW,   # output strides
    # BN params (per oc)
    bn_w_ptr, bn_b_ptr, bn_rm_ptr, bn_rv_ptr, eps,
    # residual (optional; same shape as y)
    res_ptr, SRN, SRC, SRH, SRW,
    # compile-time
    BLOCK_P: tl.constexpr,
    KH: tl.constexpr, KW: tl.constexpr,
    FUSE_BN: tl.constexpr, ACT: tl.constexpr, ADD_RESIDUAL: tl.constexpr
):
    pid_sp = tl.program_id(0)
    oc = tl.program_id(1)
    n = tl.program_id(2)

    # spatial offsets (linearized)
    p_start = pid_sp * BLOCK_P
    offs_p = p_start + tl.arange(0, BLOCK_P)
    mask_p = offs_p < (H_OUT * W_OUT)

    # map to (oh, ow)
    oh = offs_p // W_OUT
    ow = offs_p - oh * W_OUT

    # base input indices
    ih0 = oh * STRIDE_H - PAD_H
    iw0 = ow * STRIDE_W - PAD_W

    # determine group for this oc
    g = oc // C_OUT_PER_G

    # accumulator
    acc = tl.zeros((BLOCK_P,), dtype=tl.float32)

    # iterate over input channels belonging to this group
    # Tiling across input channels is not strictly necessary for correctness; we iterate elementwise here.
    for ci_g in tl.range(0, C_IN_PER_G):
        ci = g * C_IN_PER_G + ci_g
        # loop over kernel window
        for kh in range(KH):
            ih = ih0 + kh * DIL_H
            for kw in range(KW):
                iw = iw0 + kw * DIL_W
                in_bounds = mask_p & (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)

                # load input pixel vector across BLOCK_P spatial positions
                x_ptrs = x_ptr + n * SXN + ci * SXC + ih * SXH + iw * SXW
                x_val = tl.load(x_ptrs, mask=in_bounds, other=0.0).to(tl.float32)

                # load weight scalar for (oc, ci_g, kh, kw)
                w_off = oc * SWO + ci_g * SWI + kh * SWKH + kw * SWKW
                w_val = tl.load(w_ptr + w_off).to(tl.float32)

                acc += x_val * w_val

    # bias (if provided); safe to always add since pointer points to zeros if bias=None
    b = tl.load(bias_ptr + oc)
    acc += b.to(tl.float32)

    # BatchNorm (inference)
    if FUSE_BN:
        gamma = tl.load(bn_w_ptr + oc).to(tl.float32)
        beta  = tl.load(bn_b_ptr + oc).to(tl.float32)
        mean  = tl.load(bn_rm_ptr + oc).to(tl.float32)
        var   = tl.load(bn_rv_ptr + oc).to(tl.float32)
        inv_std = 1.0 / tl.sqrt(var + eps)
        acc = (acc - mean) * inv_std
        acc = acc * gamma + beta

    # Residual add (post-BN, pre-activation)
    if ADD_RESIDUAL:
        res_ptrs = res_ptr + n * SRN + oc * SRC + oh * SRH + ow * SRW
        res_val = tl.load(res_ptrs, mask=mask_p, other=0.0).to(tl.float32)
        acc = acc + res_val

    # Activation
    if ACT == 1:  # ReLU
        acc = tl.maximum(acc, 0.0)
    elif ACT == 2:  # ReLU6
        acc = tl.minimum(tl.maximum(acc, 0.0), 6.0)

    # store
    y_ptrs = y_ptr + n * SYN + oc * SYC + oh * SYH + ow * SYW
    tl.store(y_ptrs, acc.to(y_ptr.dtype.element_ty), mask=mask_p)


def conv2d_fused(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    bn_w: torch.Tensor,
    bn_b: torch.Tensor,
    bn_rm: torch.Tensor,
    bn_rv: torch.Tensor,
    stride=(1, 1),
    padding=(0, 0),
    dilation=(1, 1),
    groups=1,
    activation=None,  # None | "relu" | "relu6"
    residual: torch.Tensor = None,
    eps: float = 1e-5,
):
    """
    Wrapper around _conv2d_nchw_grouped_fused Triton kernel.
    Args:
      x: [N, C_in, H, W]
      weight: [C_out, C_in/groups, KH, KW]
      bias: [C_out] or None (will be treated as zeros)
      bn_*: [C_out], BatchNorm running stats and affine
      stride, padding, dilation: tuples
      groups: int
      activation: None/"relu"/"relu6"
      residual: optional tensor [N, C_out, H_out, W_out] added post-BN/pre-activation
      eps: float BN epsilon
    Returns:
      y: [N, C_out, H_out, W_out] same dtype/device as x
    """
    assert x.is_cuda and weight.is_cuda
    assert x.ndim == 4 and weight.ndim == 4
    assert x.dtype in (torch.float32, torch.float16, torch.bfloat16)
    N, C_in, H, W = x.shape
    C_out, C_in_per_g, KH, KW = weight.shape
    assert C_in % groups == 0 and C_out % groups == 0
    assert C_in_per_g == C_in // groups

    sh, sw = (stride if isinstance(stride, (tuple, list)) else (stride, stride))
    ph, pw = (padding if isinstance(padding, (tuple, list)) else (padding, padding))
    dh, dw = (dilation if isinstance(dilation, (tuple, list)) else (dilation, dilation))

    H_out = (H + 2 * ph - dh * (KH - 1) - 1) // sh + 1
    W_out = (W + 2 * pw - dw * (KW - 1) - 1) // sw + 1
    assert H_out > 0 and W_out > 0

    # Prepare tensors
    x_ = x.contiguous()
    w_ = weight.contiguous()
    if bias is None:
        bias_ = torch.zeros((C_out,), device=x.device, dtype=x.dtype)
    else:
        bias_ = bias.contiguous()
    # BN params
    if bn_w is None:
        bn_w_ = torch.ones((C_out,), device=x.device, dtype=torch.float32)
        bn_b_ = torch.zeros((C_out,), device=x.device, dtype=torch.float32)
        bn_rm_ = torch.zeros((C_out,), device=x.device, dtype=torch.float32)
        bn_rv_ = torch.ones((C_out,), device=x.device, dtype=torch.float32)
        FUSE_BN = 0
    else:
        bn_w_ = bn_w.contiguous().to(torch.float32)
        bn_b_ = bn_b.contiguous().to(torch.float32)
        bn_rm_ = bn_rm.contiguous().to(torch.float32)
        bn_rv_ = bn_rv.contiguous().to(torch.float32)
        FUSE_BN = 1

    # Output
    y = torch.empty((N, C_out, H_out, W_out), device=x.device, dtype=x.dtype)

    # Strides
    SXN, SXC, SXH, SXW = x_.stride()
    SWO, SWI, SWKH, SWKW = w_.stride()
    SYN, SYC, SYH, SYW = y.stride()

    # Residual ptr + strides
    if residual is not None:
        assert residual.is_cuda and residual.shape == (N, C_out, H_out, W_out)
        res = residual.contiguous()
        SRN, SRC, SRH, SRW = res.stride()
        res_ptr = res
        ADD_RESIDUAL = 1
    else:
        # dummy tensor (not used; pass y as placeholder)
        res_ptr = y
        SRN = SRC = SRH = SRW = 0
        ADD_RESIDUAL = 0

    # Act code
    if activation is None:
        ACT = 0
    elif activation == "relu":
        ACT = 1
    elif activation == "relu6":
        ACT = 2
    else:
        raise ValueError("Unknown activation")

    BLOCK_P = 128
    grid = (triton.cdiv(H_out * W_out, BLOCK_P), C_out, N)

    _conv2d_nchw_grouped_fused[grid](
        x_, w_, bias_, y,
        N, C_in, H, W, C_out, H_out, W_out,
        groups, C_in // groups, C_out // groups,
        sh, sw, ph, pw, dh, dw,
        SXN, SXC, SXH, SXW,
        SWO, SWI, SWKH, SWKW,
        SYN, SYC, SYH, SYW,
        bn_w_, bn_b_, bn_rm_, bn_rv_, float(eps),
        res_ptr, SRN, SRC, SRH, SRW,
        BLOCK_P=BLOCK_P, KH=KH, KW=KW,
        FUSE_BN=FUSE_BN, ACT=ACT, ADD_RESIDUAL=ADD_RESIDUAL,
        num_warps=4, num_stages=2,
    )
    return y


# ======================================================================================
# 1x1 Conv2D (stride=1, padding=0) as GEMM, followed by BN + optional activation/residual
# ======================================================================================

# Autotune configurations for GEMM tiles
_pw_configs = [
    triton.Config({'BLOCK_M': 32,  'BLOCK_N': 64,  'BLOCK_K': 32},  num_stages=2, num_warps=4),
    triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64},  num_stages=3, num_warps=4),
    triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64},  num_stages=3, num_warps=8),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 64},  num_stages=3, num_warps=8),
]


@triton.autotune(configs=_pw_configs, key=["M", "N", "K"])
@triton.jit
def _conv1x1_s1_nchw_gemm(
    x_ptr, w_ptr, y_ptr,
    # sizes
    M, N, K,
    # spatial dims
    H, W,
    # strides
    sxn, sxc, sxh, sxw,
    swo, swc,
    syn, syc, syh, syw,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    HW = H * W
    n_idx = offs_m // HW
    rem = offs_m - n_idx * HW
    h_idx = rem // W
    w_idx = rem - h_idx * W

    base_x = n_idx * sxn + h_idx * sxh + w_idx * sxw
    base_y = n_idx * syn + h_idx * syh + w_idx * syw

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    k_tiles = tl.cdiv(K, BLOCK_K)
    for kt in range(0, k_tiles):
        k_idx = kt * BLOCK_K + offs_k

        a_ptrs = x_ptr + base_x[:, None] + k_idx[None, :] * sxc
        a_mask = (offs_m[:, None] < M) & (k_idx[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        b_ptrs = w_ptr + k_idx[:, None] * swc + offs_n[None, :] * swo
        b_mask = (k_idx[:, None] < K) & (offs_n[None, :] < N)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        acc = tl.dot(a, b, acc)

    y_ptrs = y_ptr + base_y[:, None] + offs_n[None, :] * syc
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(y_ptrs, acc.to(y_ptr.dtype.element_ty), mask=c_mask)


@triton.jit
def _bn_act_residual_kernel(
    y_ptr, out_ptr,
    gamma_ptr, beta_ptr, mean_ptr, var_ptr, eps,
    res_ptr,  # optional residual (may be same shape as y); ignored if ADD_RESIDUAL==0
    N, C, H, W,
    syn, syc, syh, syw,  # strides for y
    son, soc, soh, sow,  # strides for out
    srn, src, srh, srw,  # strides for residual
    ACT: tl.constexpr, ADD_RESIDUAL: tl.constexpr,
    BLOCK_C: tl.constexpr, BLOCK_S: tl.constexpr
):
    # program tiles across spatial positions (S=N*H*W) and channels
    pid_s = tl.program_id(0)
    pid_c = tl.program_id(1)

    S = N * H * W
    offs_s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_s = offs_s < S
    mask_c = offs_c < C

    # Decode offs_s -> (n, h, w)
    HW = H * W
    n_idx = offs_s // HW
    rem = offs_s - n_idx * HW
    h_idx = rem // W
    w_idx = rem - h_idx * W

    gamma = tl.load(gamma_ptr + offs_c, mask=mask_c, other=0.0).to(tl.float32)
    beta  = tl.load(beta_ptr  + offs_c, mask=mask_c, other=0.0).to(tl.float32)
    mean  = tl.load(mean_ptr  + offs_c, mask=mask_c, other=0.0).to(tl.float32)
    var   = tl.load(var_ptr   + offs_c, mask=mask_c, other=0.0).to(tl.float32)
    inv_std = 1.0 / tl.sqrt(var + eps)
    scale = gamma * inv_std
    shift = beta - mean * scale

    # Load y
    y_ptrs = y_ptr + (n_idx[:, None] * syn + offs_c[None, :] * syc + h_idx[:, None] * syh + w_idx[:, None] * syw)
    y = tl.load(y_ptrs, mask=mask_s[:, None] & mask_c[None, :], other=0.0).to(tl.float32)

    # BN
    y = y * scale[None, :] + shift[None, :]

    # Residual add
    if ADD_RESIDUAL:
        r_ptrs = res_ptr + (n_idx[:, None] * srn + offs_c[None, :] * src + h_idx[:, None] * srh + w_idx[:, None] * srw)
        r = tl.load(r_ptrs, mask=mask_s[:, None] & mask_c[None, :], other=0.0).to(tl.float32)
        y = y + r

    # Activation
    if ACT == 1:
        y = tl.maximum(y, 0.0)
    elif ACT == 2:
        y = tl.minimum(tl.maximum(y, 0.0), 6.0)

    # Store
    out_ptrs = out_ptr + (n_idx[:, None] * son + offs_c[None, :] * soc + h_idx[:, None] * soh + w_idx[:, None] * sow)
    tl.store(out_ptrs, y.to(out_ptr.dtype.element_ty), mask=mask_s[:, None] & mask_c[None, :])


def conv1x1_bn_act(
    x: torch.Tensor,
    weight: torch.Tensor,  # [C_out, C_in, 1, 1]
    bn_w: torch.Tensor, bn_b: torch.Tensor, bn_rm: torch.Tensor, bn_rv: torch.Tensor,
    activation=None,  # None/"relu"/"relu6"
    residual: torch.Tensor = None,
    eps: float = 1e-5,
):
    """
    1x1 Conv2D (stride=1, padding=0) via GEMM + BN + optional activation/residual.
    """
    assert x.is_cuda and weight.is_cuda
    assert weight.shape[2:] == (1, 1)
    N, C_in, H, W = x.shape
    C_out = weight.shape[0]
    assert weight.shape[1] == C_in

    # 1) GEMM conv
    y = torch.empty((N, C_out, H, W), device=x.device, dtype=x.dtype)
    M = N * H * W
    K = C_in
    Ncols = C_out
    sxn, sxc, sxh, sxw = x.stride()
    swo, swc, _, _ = weight.stride()
    syn, syc, syh, syw = y.stride()

    def grid(meta):
        return (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(Ncols, meta["BLOCK_N"]))
    _conv1x1_s1_nchw_gemm[grid](
        x, weight, y,
        M, Ncols, K,
        H, W,
        sxn, sxc, sxh, sxw,
        swo, swc,
        syn, syc, syh, syw,
    )

    # 2) BN + residual + activation
    out = torch.empty_like(y)
    ACT = 0 if activation is None else (1 if activation == "relu" else 2)
    ADD_R = 1 if residual is not None else 0
    if residual is None:
        residual = y  # placeholder; not used

    BLOCK_C = 64
    BLOCK_S = 64
    grid2 = (triton.cdiv(N * H * W, BLOCK_S), triton.cdiv(C_out, BLOCK_C))
    _bn_act_residual_kernel[grid2](
        y, out,
        bn_w.to(torch.float32), bn_b.to(torch.float32), bn_rm.to(torch.float32), bn_rv.to(torch.float32), float(eps),
        residual,
        N, C_out, H, W,
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        residual.stride(0), residual.stride(1), residual.stride(2), residual.stride(3),
        ACT=ACT, ADD_RESIDUAL=ADD_R, BLOCK_C=BLOCK_C, BLOCK_S=BLOCK_S,
        num_warps=4, num_stages=2,
    )
    return out


# ======================================================================================
# Global Average Pooling [N, C, H, W] -> [N, C] fused with flatten
# ======================================================================================

@triton.jit
def _gap_flatten_kernel(
    x_ptr, y_ptr,
    N, C, H, W,
    sN, sC, sH, sW,
    oN, oC,
    inv_hw,
    BLOCK_HW: tl.constexpr
):
    pid = tl.program_id(0)
    n = pid // C
    c = pid % C
    base = n * sN + c * sC

    total = H * W
    acc = tl.zeros((), dtype=tl.float32)
    for start in range(0, total, BLOCK_HW):
        offs = start + tl.arange(0, BLOCK_HW)
        mask = offs < total
        h = offs // W
        w = offs % W
        ptrs = x_ptr + base + h * sH + w * sW
        vals = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)
        acc += tl.sum(vals, axis=0)
    mean = acc * inv_hw
    out_ptr = y_ptr + n * oN + c * oC
    tl.store(out_ptr, mean.to(y_ptr.dtype.element_ty))


def gap_flatten(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda
    N, C, H, W = x.shape
    y = torch.empty((N, C), device=x.device, dtype=x.dtype)
    BLOCK_HW = 128
    grid = (N * C,)
    _gap_flatten_kernel[grid](
        x, y,
        N, C, H, W,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        y.stride(0), y.stride(1),
        float(1.0 / (H * W)),
        BLOCK_HW=BLOCK_HW,
    )
    return y


# ======================================================================================
# Linear y = x @ weight.T + bias
# ======================================================================================

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=4),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _linear_fused_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    M, N, K,
    sxm, sxk,
    swn, swk,
    sym, syn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    k_tiles = tl.cdiv(K, BLOCK_K)
    for kt in range(0, k_tiles):
        k_idx = kt * BLOCK_K + offs_k

        x_ptrs = x_ptr + offs_m[:, None] * sxm + k_idx[None, :] * sxk
        x_mask = (offs_m[:, None] < M) & (k_idx[None, :] < K)
        a = tl.load(x_ptrs, mask=x_mask, other=0.0)

        w_ptrs = w_ptr + offs_n[None, :] * swn + k_idx[:, None] * swk
        w_mask = (offs_n[None, :] < N) & (k_idx[:, None] < K)
        b = tl.load(w_ptrs, mask=w_mask, other=0.0)

        acc = tl.dot(a, b, acc)

    bias = tl.load(b_ptr + offs_n, mask=(offs_n < N), other=0.0).to(tl.float32)
    acc = acc + bias[None, :]

    y_ptrs = y_ptr + offs_m[:, None] * sym + offs_n[None, :] * syn
    y_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(y_ptrs, acc.to(y_ptr.dtype.element_ty), mask=y_mask)


def linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and weight.is_cuda and bias.is_cuda
    M, K = x.shape
    N, K2 = weight.shape
    assert K == K2 and bias.shape[0] == N
    y = torch.empty((M, N), device=x.device, dtype=x.dtype)

    def grid(meta):
        return (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]))
    _linear_fused_kernel[grid](
        x, weight, bias, y,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
        y.stride(0), y.stride(1),
    )
    return y


# ======================================================================================
# End-to-end EfficientNetB0 forward built from Triton kernels above.
# Top-level wrapper: kernel_function(x, params) -> logits
# params is a dict containing all conv/bn/linear weights.
# ======================================================================================

def _mbconv_block(
    x: torch.Tensor,
    params: dict,
    prefix: str,
    kernel_size: int,
    stride: int,
    expand_ratio: int,
):
    """
    MBConv block built from:
      [optional] 1x1 expand -> BN -> ReLU6
      depthwise KxK (groups=C_mid) stride -> BN -> ReLU6
      1x1 project -> BN -> [optional residual add]
    params keys:
      if expand_ratio != 1:
        f"{prefix}_expand_w"        [C_mid, C_in, 1, 1]
        f"{prefix}_expand_bn_w/b/rm/rv"
      depthwise:
        f"{prefix}_dw_w"            [C_mid, 1, K, K]
        f"{prefix}_dw_bn_w/b/rm/rv"
      project:
        f"{prefix}_proj_w"          [C_out, C_mid, 1, 1]
        f"{prefix}_proj_bn_w/b/rm/rv"
    """
    N, C_in, H, W = x.shape

    # Expand (if needed)
    if expand_ratio != 1:
        x = conv1x1_bn_act(
            x,
            params[f"{prefix}_expand_w"],
            params[f"{prefix}_expand_bn_w"], params[f"{prefix}_expand_bn_b"],
            params[f"{prefix}_expand_bn_rm"], params[f"{prefix}_expand_bn_rv"],
            activation="relu6",
            residual=None,
            eps=1e-5,
        )
    C_mid = x.shape[1]

    # Depthwise
    dw_w = params[f"{prefix}_dw_w"]
    assert dw_w.shape[2] == kernel_size and dw_w.shape[3] == kernel_size
    x = conv2d_fused(
        x, dw_w, None,
        params[f"{prefix}_dw_bn_w"], params[f"{prefix}_dw_bn_b"], params[f"{prefix}_dw_bn_rm"], params[f"{prefix}_dw_bn_rv"],
        stride=(stride, stride),
        padding=((kernel_size - 1)//2, (kernel_size - 1)//2),
        dilation=(1, 1),
        groups=C_mid,
        activation="relu6",
        residual=None,
        eps=1e-5,
    )

    # Project
    use_residual = (stride == 1) and (C_in == params[f"{prefix}_proj_w"].shape[0])
    residual = x if False else None  # placeholder to satisfy linter

    proj_in = x
    res = None
    if use_residual:
        # residual should be the input to the block (before expand/depthwise), i.e., the original x_in
        # We need the original input which is the incoming tensor to this block.
        # To pass it in, we stash it earlier and supply here as a function argument.
        # In our orchestrator below, we'll call _mbconv_block with x_in saved separately.
        pass  # handled by orchestrator wrapper

    # We'll return a closure signal to the orchestrator telling it to add residual after project BN
    return x  # temporary; real logic implemented in orchestrator below (we need x_in)


def kernel_function(x: torch.Tensor, params: dict) -> torch.Tensor:
    """
    End-to-end EfficientNetB0 forward using Triton kernels only.

    Args:
      x: [N, 3, 224, 224] float32 CUDA tensor
      params: dict carrying all weights/bn params extracted from a PyTorch Model instance.

    Returns:
      logits: [N, 1000] float32 CUDA tensor
    """
    assert x.is_cuda and x.dtype == torch.float32
    N = x.shape[0]

    # Stem: conv3x3 s2 + BN + ReLU
    x = conv2d_fused(
        x,
        params["stem_conv_w"], None,
        params["stem_bn_w"], params["stem_bn_b"], params["stem_bn_rm"], params["stem_bn_rv"],
        stride=(2, 2), padding=(1, 1), dilation=(1, 1), groups=1, activation="relu", residual=None, eps=1e-5,
    )

    # Helper to run MBConv by explicitly handling residual addition at the project stage
    def run_mbconv(x_in, name_prefix, k, s, e, Cout):
        x = x_in
        # Expand
        if e != 1:
            x = conv1x1_bn_act(
                x,
                params[f"{name_prefix}_expand_w"],
                params[f"{name_prefix}_expand_bn_w"], params[f"{name_prefix}_expand_bn_b"],
                params[f"{name_prefix}_expand_bn_rm"], params[f"{name_prefix}_expand_bn_rv"],
                activation="relu6",
                residual=None,
                eps=1e-5,
            )
        C_mid = x.shape[1]
        # Depthwise
        x = conv2d_fused(
            x, params[f"{name_prefix}_dw_w"], None,
            params[f"{name_prefix}_dw_bn_w"], params[f"{name_prefix}_dw_bn_b"],
            params[f"{name_prefix}_dw_bn_rm"], params[f"{name_prefix}_dw_bn_rv"],
            stride=(s, s), padding=((k - 1)//2, (k - 1)//2), dilation=(1, 1),
            groups=C_mid, activation="relu6", residual=None, eps=1e-5,
        )
        # Project 1x1
        use_res = (s == 1) and (x_in.shape[1] == Cout) and (x_in.shape[2] == x.shape[2]) and (x_in.shape[3] == x.shape[3])
        res = x_in if use_res else None
        x = conv1x1_bn_act(
            x,
            params[f"{name_prefix}_proj_w"],
            params[f"{name_prefix}_proj_bn_w"], params[f"{name_prefix}_proj_bn_b"],
            params[f"{name_prefix}_proj_bn_rm"], params[f"{name_prefix}_proj_bn_rv"],
            activation=None,
            residual=res,
            eps=1e-5,
        )
        return x

    # Blocks (names and metadata must match our param packing function in tests)
    # 1) MBConv(32 -> 16, k3, s1, e1)
    x = run_mbconv(x, "b0", k=3, s=1, e=1, Cout=16)
    # 2) MBConv(16 -> 24, k3, s2, e6)
    x = run_mbconv(x, "b1", k=3, s=2, e=6, Cout=24)
    # 3) MBConv(24 -> 24, k3, s1, e6) residual
    x = run_mbconv(x, "b2", k=3, s=1, e=6, Cout=24)
    # 4) MBConv(24 -> 40, k5, s2, e6)
    x = run_mbconv(x, "b3", k=5, s=2, e=6, Cout=40)
    # 5) MBConv(40 -> 40, k5, s1, e6) residual
    x = run_mbconv(x, "b4", k=5, s=1, e=6, Cout=40)
    # 6) MBConv(40 -> 80, k3, s2, e6)
    x = run_mbconv(x, "b5", k=3, s=2, e=6, Cout=80)
    # 7) MBConv(80 -> 80, k3, s1, e6) residual
    x = run_mbconv(x, "b6", k=3, s=1, e=6, Cout=80)
    # 8) MBConv(80 -> 112, k5, s1, e6)
    x = run_mbconv(x, "b7", k=5, s=1, e=6, Cout=112)
    # 9) MBConv(112 -> 112, k5, s1, e6) residual
    x = run_mbconv(x, "b8", k=5, s=1, e=6, Cout=112)
    # 10) MBConv(112 -> 192, k5, s2, e6)
    x = run_mbconv(x, "b9", k=5, s=2, e=6, Cout=192)
    # 11) MBConv(192 -> 192, k5, s1, e6) residual
    x = run_mbconv(x, "b10", k=5, s=1, e=6, Cout=192)
    # 12) MBConv(192 -> 192, k5, s1, e6) residual
    x = run_mbconv(x, "b11", k=5, s=1, e=6, Cout=192)
    # 13) MBConv(192 -> 320, k3, s1, e6)
    x = run_mbconv(x, "b12", k=3, s=1, e=6, Cout=320)

    # Head: conv1x1 320->1280 + BN + ReLU
    x = conv1x1_bn_act(
        x,
        params["head_conv_w"],
        params["head_bn_w"], params["head_bn_b"], params["head_bn_rm"], params["head_bn_rv"],
        activation="relu",
        residual=None,
        eps=1e-5,
    )

    # GAP + flatten -> [N, 1280]
    x = gap_flatten(x)

    # Linear [N, 1000]
    x = linear(x, params["fc_w"], params["fc_b"])
    return x


# ======================================================================================
# Utilities: reference model and parameter extractor for self-test
# ======================================================================================

# Original reference model from the problem statement (copied verbatim)
import torch.nn as nn
import torch.nn.functional as F

class MBConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio):
        super(MBConv, self).__init__()
        self.use_residual = (stride == 1 and in_channels == out_channels)
        hidden_dim = in_channels * expand_ratio

        if expand_ratio != 1:
            self.expand_conv = nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True)
            )

        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True)
        )

        self.project_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        identity = x
        if hasattr(self, 'expand_conv'):
            x = self.expand_conv(x)
        x = self.depthwise_conv(x)
        x = self.project_conv(x)
        if self.use_residual:
            x += identity
        return x

class Model(nn.Module):
    def __init__(self, num_classes=1000):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)

        self.blocks = nn.Sequential(
            MBConv(32, 16, kernel_size=3, stride=1, expand_ratio=1),
            MBConv(16, 24, kernel_size=3, stride=2, expand_ratio=6),
            MBConv(24, 24, kernel_size=3, stride=1, expand_ratio=6),
            MBConv(24, 40, kernel_size=5, stride=2, expand_ratio=6),
            MBConv(40, 40, kernel_size=5, stride=1, expand_ratio=6),
            MBConv(40, 80, kernel_size=3, stride=2, expand_ratio=6),
            MBConv(80, 80, kernel_size=3, stride=1, expand_ratio=6),
            MBConv(80, 112, kernel_size=5, stride=1, expand_ratio=6),
            MBConv(112, 112, kernel_size=5, stride=1, expand_ratio=6),
            MBConv(112, 192, kernel_size=5, stride=2, expand_ratio=6),
            MBConv(192, 192, kernel_size=5, stride=1, expand_ratio=6),
            MBConv(192, 192, kernel_size=5, stride=1, expand_ratio=6),
            MBConv(192, 320, kernel_size=3, stride=1, expand_ratio=6)
        )

        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)

        self.fc = nn.Linear(1280, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.blocks(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def get_inputs():
    return [torch.rand(10, 3, 224, 224)]

def get_init_inputs():
    return [1000]


def extract_params(model: Model) -> dict:
    """
    Collect all needed tensors from the PyTorch model into a flat dict for kernel_function.
    All tensors are moved to CUDA and kept in float32.
    """
    device = torch.device('cuda')
    d = {}
    # Stem
    d["stem_conv_w"] = model.conv1.weight.detach().to(device)
    d["stem_bn_w"] = model.bn1.weight.detach().to(device, dtype=torch.float32)
    d["stem_bn_b"] = model.bn1.bias.detach().to(device, dtype=torch.float32)
    d["stem_bn_rm"] = model.bn1.running_mean.detach().to(device, dtype=torch.float32)
    d["stem_bn_rv"] = model.bn1.running_var.detach().to(device, dtype=torch.float32)

    # Blocks
    for i, blk in enumerate(model.blocks):
        prefix = f"b{i}"
        # expand (if exists)
        if hasattr(blk, "expand_conv"):
            conv = blk.expand_conv[0]
            bn = blk.expand_conv[1]
            d[f"{prefix}_expand_w"] = conv.weight.detach().to(device)
            d[f"{prefix}_expand_bn_w"] = bn.weight.detach().to(device, dtype=torch.float32)
            d[f"{prefix}_expand_bn_b"] = bn.bias.detach().to(device, dtype=torch.float32)
            d[f"{prefix}_expand_bn_rm"] = bn.running_mean.detach().to(device, dtype=torch.float32)
            d[f"{prefix}_expand_bn_rv"] = bn.running_var.detach().to(device, dtype=torch.float32)
        else:
            # create placeholders to avoid KeyError if accessed (won't be used)
            pass

        # depthwise
        conv = blk.depthwise_conv[0]
        bn = blk.depthwise_conv[1]
        d[f"{prefix}_dw_w"] = conv.weight.detach().to(device)
        d[f"{prefix}_dw_bn_w"] = bn.weight.detach().to(device, dtype=torch.float32)
        d[f"{prefix}_dw_bn_b"] = bn.bias.detach().to(device, dtype=torch.float32)
        d[f"{prefix}_dw_bn_rm"] = bn.running_mean.detach().to(device, dtype=torch.float32)
        d[f"{prefix}_dw_bn_rv"] = bn.running_var.detach().to(device, dtype=torch.float32)

        # project
        conv = blk.project_conv[0]
        bn = blk.project_conv[1]
        d[f"{prefix}_proj_w"] = conv.weight.detach().to(device)
        d[f"{prefix}_proj_bn_w"] = bn.weight.detach().to(device, dtype=torch.float32)
        d[f"{prefix}_proj_bn_b"] = bn.bias.detach().to(device, dtype=torch.float32)
        d[f"{prefix}_proj_bn_rm"] = bn.running_mean.detach().to(device, dtype=torch.float32)
        d[f"{prefix}_proj_bn_rv"] = bn.running_var.detach().to(device, dtype=torch.float32)

    # Head
    d["head_conv_w"] = model.conv2.weight.detach().to(device)
    d["head_bn_w"] = model.bn2.weight.detach().to(device, dtype=torch.float32)
    d["head_bn_b"] = model.bn2.bias.detach().to(device, dtype=torch.float32)
    d["head_bn_rm"] = model.bn2.running_mean.detach().to(device, dtype=torch.float32)
    d["head_bn_rv"] = model.bn2.running_var.detach().to(device, dtype=torch.float32)

    # FC
    d["fc_w"] = model.fc.weight.detach().to(device)
    d["fc_b"] = model.fc.bias.detach().to(device)

    return d


# Modify run_mbconv to add residual properly by wrapping kernel_function's inner implementation
def _build_run_mbconv_with_params(params: dict):
    def run_mbconv(x_in, name_prefix, k, s, e, Cout):
        x = x_in
        if e != 1:
            x = conv1x1_bn_act(
                x,
                params[f"{name_prefix}_expand_w"],
                params[f"{name_prefix}_expand_bn_w"], params[f"{name_prefix}_expand_bn_b"],
                params[f"{name_prefix}_expand_bn_rm"], params[f"{name_prefix}_expand_bn_rv"],
                activation="relu6",
                residual=None,
                eps=1e-5,
            )
        C_mid = x.shape[1]
        x = conv2d_fused(
            x, params[f"{name_prefix}_dw_w"], None,
            params[f"{name_prefix}_dw_bn_w"], params[f"{name_prefix}_dw_bn_b"],
            params[f"{name_prefix}_dw_bn_rm"], params[f"{name_prefix}_dw_bn_rv"],
            stride=(s, s), padding=((k - 1)//2, (k - 1)//2), dilation=(1, 1),
            groups=C_mid, activation="relu6", residual=None, eps=1e-5,
        )
        use_res = (s == 1) and (x_in.shape[1] == Cout) and (x_in.shape[2] == x.shape[2]) and (x_in.shape[3] == x.shape[3])
        res = x_in if use_res else None
        x = conv1x1_bn_act(
            x,
            params[f"{name_prefix}_proj_w"],
            params[f"{name_prefix}_proj_bn_w"], params[f"{name_prefix}_proj_bn_b"],
            params[f"{name_prefix}_proj_bn_rm"], params[f"{name_prefix}_proj_bn_rv"],
            activation=None, residual=res, eps=1e-5,
        )
        return x
    return run_mbconv


def kernel_function_reference(x: torch.Tensor, params: dict) -> torch.Tensor:
    """
    Same as kernel_function but implemented with the helper run_mbconv built on params
    (kept for clarity; kernel_function uses identical logic).
    """
    return kernel_function(x, params)


# Override kernel_function to use the run_mbconv closure with params
def kernel_function(x: torch.Tensor, params: dict) -> torch.Tensor:
    assert x.is_cuda and x.dtype == torch.float32
    # Stem
    x = conv2d_fused(
        x,
        params["stem_conv_w"], None,
        params["stem_bn_w"], params["stem_bn_b"], params["stem_bn_rm"], params["stem_bn_rv"],
        stride=(2, 2), padding=(1, 1), dilation=(1, 1), groups=1, activation="relu", residual=None, eps=1e-5,
    )
    run_mbconv = _build_run_mbconv_with_params(params)
    x = run_mbconv(x, "b0", k=3, s=1, e=1, Cout=16)
    x = run_mbconv(x, "b1", k=3, s=2, e=6, Cout=24)
    x = run_mbconv(x, "b2", k=3, s=1, e=6, Cout=24)
    x = run_mbconv(x, "b3", k=5, s=2, e=6, Cout=40)
    x = run_mbconv(x, "b4", k=5, s=1, e=6, Cout=40)
    x = run_mbconv(x, "b5", k=3, s=2, e=6, Cout=80)
    x = run_mbconv(x, "b6", k=3, s=1, e=6, Cout=80)
    x = run_mbconv(x, "b7", k=5, s=1, e=6, Cout=112)
    x = run_mbconv(x, "b8", k=5, s=1, e=6, Cout=112)
    x = run_mbconv(x, "b9", k=5, s=2, e=6, Cout=192)
    x = run_mbconv(x, "b10", k=5, s=1, e=6, Cout=192)
    x = run_mbconv(x, "b11", k=5, s=1, e=6, Cout=192)
    x = run_mbconv(x, "b12", k=3, s=1, e=6, Cout=320)

    x = conv1x1_bn_act(
        x,
        params["head_conv_w"],
        params["head_bn_w"], params["head_bn_b"], params["head_bn_rm"], params["head_bn_rv"],
        activation="relu", residual=None, eps=1e-5,
    )
    x = gap_flatten(x)
    x = linear(x, params["fc_w"], params["fc_b"])
    return x


# ======================================================================================
# Self-test
# ======================================================================================

def run_tests():
    torch.manual_seed(0)
    assert torch.cuda.is_available(), "CUDA required"
    device = torch.device('cuda')

    # Instantiate reference model and set to eval for BN inference semantics
    model = Model(*get_init_inputs()).to(device).eval()

    # Prepare inputs
    x = get_inputs()[0].to(device, dtype=torch.float32)

    # Run PyTorch reference
    with torch.no_grad():
        ref = model(x)

    # Collect params
    params = extract_params(model)

    # Run Triton implementation
    with torch.no_grad():
        out = kernel_function(x, params)

    # Compare
    atol = 1e-3
    rtol = 1e-3
    ok = torch.allclose(out, ref, rtol=rtol, atol=atol)
    if ok:
        print("PASS")
        return 0
    else:
        # Print diagnostic
        max_abs = (out - ref).abs().max().item()
        max_rel = ((out - ref).abs() / (ref.abs() + 1e-8)).max().item()
        print("Mismatch! max_abs=", max_abs, "max_rel=", max_rel)
        return 1


if __name__ == "__main__":
    sys.exit(run_tests())
