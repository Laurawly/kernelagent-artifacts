# kernel.py
# Fused ResNet BasicBlock (NCHW, s=1, k=3x3) implemented with Triton kernels.
# Pipeline fused as much as is practical without recomputation:
#   Kernel 1: conv3x3 -> BN (inference) -> ReLU
#   Kernel 2: conv3x3 -> BN (inference) -> Add(identity) -> ReLU
# Rationale: Fully fusing both convolutions into a single kernel would require
# reusing a 3x3 neighborhood of the first conv's BN+ReLU outputs to produce each
# output pixel of the second conv. Doing this without writing out the intermediate
# would either require recomputing many conv1 outputs or substantial on-chip tiling
# complexity across both spatial and channel dimensions including halos. We keep a
# single intermediate tensor (y1_relu) in VRAM to preserve correct data dependency
# and keep the wrapper free of compute logic, as required.

import torch
import triton
import triton.language as tl


@triton.jit
def _conv_bn_relu_3x3_s1(
    x_ptr,            # *: input NCHW
    w_ptr,            # *: weights [C_out, C_in, 3, 3]
    gamma_ptr,        # *: BN gamma [C_out]
    beta_ptr,         # *: BN beta [C_out]
    mean_ptr,         # *: BN running mean [C_out]
    var_ptr,          # *: BN running var [C_out]
    y_ptr,            # *: output NCHW (e.g., fp32 for stability)
    N, C, H, W, C_OUT,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wn, stride_wc, stride_wk, stride_wl,  # weight strides: [co, ci, kh, kw]
    stride_yn, stride_yc, stride_yh, stride_yw,
    eps,  # float32
    BLOCK_W: tl.constexpr,
):
    # program ids
    pid_w = tl.program_id(axis=0)
    h = tl.program_id(axis=1)
    nc = tl.program_id(axis=2)

    n = nc // C_OUT
    co = nc % C_OUT

    # vector of W indices
    offs_w = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    mask_w = offs_w < W
    mask_h = h < H
    full_mask_out = mask_w & mask_h & (n < N) & (co < C_OUT)

    # initialize accumulator in fp32
    acc = tl.zeros((BLOCK_W,), dtype=tl.float32)

    # accumulate over input channels and 3x3 spatial kernel
    for ci in range(0, C):
        for ky in range(0, 3):
            iy = h + ky - 1  # padding = 1
            valid_y = (iy >= 0) & (iy < H)
            for kx in range(0, 3):
                ix = offs_w + kx - 1
                mask_x = (ix >= 0) & (ix < W)
                mask = full_mask_out & valid_y & mask_x

                # input pointer for vector load
                x_ptrs = x_ptr + n * stride_xn + ci * stride_xc + iy * stride_xh + ix * stride_xw
                x_val = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)

                # scalar weight for this (co, ci, ky, kx)
                w_ptrs = w_ptr + co * stride_wn + ci * stride_wc + ky * stride_wk + kx * stride_wl
                w_val = tl.load(w_ptrs).to(tl.float32)

                acc += x_val * w_val

    # batch-norm parameters for channel co
    mean = tl.load(mean_ptr + co).to(tl.float32)
    var = tl.load(var_ptr + co).to(tl.float32)
    gamma = tl.load(gamma_ptr + co).to(tl.float32)
    beta = tl.load(beta_ptr + co).to(tl.float32)

    inv_std = 1.0 / tl.sqrt(var + eps)
    y = (acc - mean) * inv_std
    y = y * gamma + beta
    # ReLU
    y = tl.maximum(y, 0.0)

    # store
    y_ptrs = y_ptr + n * stride_yn + co * stride_yc + h * stride_yh + offs_w * stride_yw
    tl.store(y_ptrs, y, mask=full_mask_out)


@triton.jit
def _conv_bn_add_relu_3x3_s1(
    x_id_ptr,         # *: identity input NCHW (original x)
    z_ptr,            # *: input from previous stage (y1_relu), NCHW
    w_ptr,            # *: weights [C_out, C_in, 3, 3]
    gamma_ptr,        # *: BN2 gamma [C_out]
    beta_ptr,         # *: BN2 beta [C_out]
    mean_ptr,         # *: BN2 running mean [C_out]
    var_ptr,          # *: BN2 running var [C_out]
    y_ptr,            # *: output NCHW (dtype can match x)
    N, C, H, W, C_OUT,
    stride_xn, stride_xc, stride_xh, stride_xw,  # identity x strides
    stride_zn, stride_zc, stride_zh, stride_zw,  # z (y1_relu) strides
    stride_wn, stride_wc, stride_wk, stride_wl,  # weight strides
    stride_yn, stride_yc, stride_yh, stride_yw,
    eps,  # float32
    BLOCK_W: tl.constexpr,
):
    # program ids
    pid_w = tl.program_id(axis=0)
    h = tl.program_id(axis=1)
    nc = tl.program_id(axis=2)

    n = nc // C_OUT
    co = nc % C_OUT

    # vector of W indices
    offs_w = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    mask_w = offs_w < W
    mask_h = h < H
    full_mask_out = mask_w & mask_h & (n < N) & (co < C_OUT)

    # initialize accumulator in fp32
    acc = tl.zeros((BLOCK_W,), dtype=tl.float32)

    # accumulate over input channels and 3x3 spatial kernel (input is z=y1_relu)
    for ci in range(0, C):
        for ky in range(0, 3):
            iy = h + ky - 1  # padding = 1
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

    # BN2 params
    mean = tl.load(mean_ptr + co).to(tl.float32)
    var = tl.load(var_ptr + co).to(tl.float32)
    gamma = tl.load(gamma_ptr + co).to(tl.float32)
    beta = tl.load(beta_ptr + co).to(tl.float32)

    inv_std = 1.0 / tl.sqrt(var + eps)
    y = (acc - mean) * inv_std
    y = y * gamma + beta

    # Add identity
    x_id_ptrs = x_id_ptr + n * stride_xn + co * stride_xc + h * stride_xh + offs_w * stride_xw
    x_id_val = tl.load(x_id_ptrs, mask=full_mask_out, other=0.0).to(tl.float32)
    y = y + x_id_val

    # ReLU
    y = tl.maximum(y, 0.0)

    # store (cast handled by tl.store into pointer dtype)
    y_ptrs = y_ptr + n * stride_yn + co * stride_yc + h * stride_yh + offs_w * stride_yw
    tl.store(y_ptrs, y, mask=full_mask_out)


def kernel_function(
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
    """
    Launch wrapper for the fused BasicBlock:
    - Stage 1 (Triton): Conv3x3(s=1,p=1) -> BatchNorm (inference) -> ReLU
    - Stage 2 (Triton): Conv3x3(s=1,p=1) -> BatchNorm (inference) -> Add(identity) -> ReLU

    Notes:
    - All math is in Triton kernels; the wrapper only validates, allocates, and launches.
    - Intermediate y1_relu is stored in float32 to better match "float32 semantics" and improve numerical stability.
    - Final output dtype matches input x.dtype.
    """
    assert x.is_cuda, "Input must be on CUDA device"
    device = x.device
    assert conv1_weight.is_cuda and conv2_weight.is_cuda
    assert bn1_weight.is_cuda and bn1_bias.is_cuda and bn1_running_mean.is_cuda and bn1_running_var.is_cuda
    assert bn2_weight.is_cuda and bn2_bias.is_cuda and bn2_running_mean.is_cuda and bn2_running_var.is_cuda

    # Shapes checks: NCHW, 3x3 convs, stride=1, padding=1 (implicit in kernels)
    assert x.ndim == 4, "x must be NCHW"
    N, C, H, W = x.shape
    assert conv1_weight.shape == (C, C, 3, 3), "conv1_weight must be [C_out, C_in, 3, 3] with C_out=C_in=C"
    assert conv2_weight.shape == (C, C, 3, 3), "conv2_weight must be [C_out, C_in, 3, 3] with C_out=C_in=C"
    for t in (bn1_weight, bn1_bias, bn1_running_mean, bn1_running_var,
              bn2_weight, bn2_bias, bn2_running_mean, bn2_running_var):
        assert t.shape == (C,), "BN parameter tensors must be 1D of shape [C]"

    # Allocate intermediate and output
    # y1_relu in fp32 for accuracy; final output dtype matches input
    y1_relu = torch.empty((N, C, H, W), device=device, dtype=torch.float32)
    y_out = torch.empty_like(x)

    # Strides (in elements)
    sxn, sxc, sxh, sxw = x.stride()
    swn, swc, swk, swl = conv1_weight.stride()  # same layout for both weights
    s1n, s1c, s1h, s1w = y1_relu.stride()

    # Grid configuration
    BLOCK_W = 64  # power-of-2 tile along W; 56x56 case -> single tile per row; still fine
    grid_1 = (
        triton.cdiv(W, BLOCK_W),
        H,
        N * C,
    )

    # Kernel 1: conv1 -> bn1 -> relu
    _conv_bn_relu_3x3_s1[grid_1](
        x, conv1_weight,
        bn1_weight, bn1_bias, bn1_running_mean, bn1_running_var,
        y1_relu,
        N, C, H, W, C,
        sxn, sxc, sxh, sxw,
        swn, swc, swk, swl,
        s1n, s1c, s1h, s1w,
        1e-5,
        BLOCK_W=BLOCK_W,
        num_warps=4,
        num_stages=2,
    )

    # Strides for conv2 input (y1_relu) and identity (x) and output (y_out)
    szn, szc, szh, szw = y1_relu.stride()
    s2n, s2c, s2h, s2w = y_out.stride()

    grid_2 = (
        triton.cdiv(W, BLOCK_W),
        H,
        N * C,
    )

    # Kernel 2: conv2 -> bn2 -> add(x) -> relu
    _conv_bn_add_relu_3x3_s1[grid_2](
        x, y1_relu, conv2_weight,
        bn2_weight, bn2_bias, bn2_running_mean, bn2_running_var,
        y_out,
        N, C, H, W, C,
        sxn, sxc, sxh, sxw,
        szn, szc, szh, szw,
        swn, swc, swk, swl,
        s2n, s2c, s2h, s2w,
        1e-5,
        BLOCK_W=BLOCK_W,
        num_warps=4,
        num_stages=2,
    )

    return y_out