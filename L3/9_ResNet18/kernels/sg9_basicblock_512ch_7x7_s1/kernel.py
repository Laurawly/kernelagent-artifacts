# kernel.py
# Fused ResNet BasicBlock (two 3x3 Conv2d with BN + ReLU and residual add) implemented using Triton.
# We use two Triton kernels:
#   1) conv1 + BN1 + ReLU (fused)
#   2) conv2 + BN2 + residual add (identity) + ReLU (fused)
# Fusion rationale:
#   - Fully fusing both convolutions is not feasible without either excessive recomputation
#     or very large on-chip tiling across C and spatial neighborhoods. We instead fuse all
#     compatible stages within each conv while materializing the intermediate activation.
# Accuracy considerations:
#   - The reference computes in float32 regardless of input dtype, then casts final output.
#   - To match that, we:
#       * Cast inputs/weights to float32 before MAC (tl.dot).
#       * Accumulate in float32.
#       * Keep the intermediate activation buffer (between conv1 and conv2) in float32.
#       * Only cast to the input dtype on the final store.
#   - This eliminates precision loss from intermediate quantization (bf16/fp16) and significantly
#     reduces numerical differences vs the FP32 reference.

import triton
import triton.language as tl
import torch


@triton.jit
def _conv_bn_relu_kernel(
    x_ptr,                # *const input  [N, C, H, W]
    w_ptr,                # *const weight [C_out=C, C_in=C, KH, KW]
    y_ptr,                # *mut  output  [N, C, H, W] (FP32 buffer)
    bn_w_ptr,             # *const BN gamma [C]
    bn_b_ptr,             # *const BN beta  [C]
    bn_rm_ptr,            # *const BN running_mean [C]
    bn_rv_ptr,            # *const BN running_var  [C]
    N, C, H, W,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wo, stride_wi, stride_wkh, stride_wkw,
    stride_yn, stride_yc, stride_yh, stride_yw,
    eps,                  # float32 epsilon
    BLOCK_HW: tl.constexpr,
    BLOCK_CO: tl.constexpr,
    BLOCK_CI: tl.constexpr,
):
    # tile ids
    pid_hw = tl.program_id(0)
    pid_co = tl.program_id(1)

    P = N * H * W
    offs_p = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    offs_co = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    mask_p = offs_p < P
    mask_co = offs_co < C

    HW = H * W
    n_idx = offs_p // HW
    hw_idx = offs_p % HW
    h_idx = hw_idx // W
    w_idx = hw_idx % W

    acc = tl.zeros((BLOCK_HW, BLOCK_CO), dtype=tl.float32)

    for ci_start in range(0, C, BLOCK_CI):
        offs_ci = ci_start + tl.arange(0, BLOCK_CI)
        mask_ci = offs_ci < C

        for kh in range(3):
            for kw in range(3):
                h_in = h_idx + (kh - 1)
                w_in = w_idx + (kw - 1)
                in_bounds = (h_in >= 0) & (h_in < H) & (w_in >= 0) & (w_in < W) & mask_p

                # pointers
                x_ptrs = (
                    x_ptr
                    + n_idx[:, None] * stride_xn
                    + offs_ci[None, :] * stride_xc
                    + h_in[:, None] * stride_xh
                    + w_in[:, None] * stride_xw
                )
                w_ptrs = (
                    w_ptr
                    + offs_co[:, None] * stride_wo
                    + offs_ci[None, :] * stride_wi
                    + kh * stride_wkh
                    + kw * stride_wkw
                )

                x_mask = in_bounds[:, None] & mask_ci[None, :]
                w_mask = mask_co[:, None] & mask_ci[None, :]

                # load as source dtype then cast to fp32 for MAC to match fp32 reference semantics
                x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0).to(tl.float32)  # [M, K]
                w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0).to(tl.float32)  # [N, K]

                acc = tl.dot(x_tile, tl.trans(w_tile), acc)

    # BN per out-channel in fp32
    bn_mean = tl.load(bn_rm_ptr + offs_co, mask=mask_co, other=0.0).to(tl.float32)
    bn_var  = tl.load(bn_rv_ptr + offs_co, mask=mask_co, other=0.0).to(tl.float32)
    gamma   = tl.load(bn_w_ptr + offs_co,  mask=mask_co, other=0.0).to(tl.float32)
    beta    = tl.load(bn_b_ptr + offs_co,  mask=mask_co, other=0.0).to(tl.float32)

    rstd = 1.0 / tl.sqrt(bn_var + eps)
    acc = (acc - bn_mean[None, :]) * rstd[None, :]
    acc = acc * gamma[None, :] + beta[None, :]

    # ReLU
    acc = tl.maximum(acc, 0.0)

    # store to fp32 intermediate buffer
    y_ptrs = (
        y_ptr
        + n_idx[:, None] * stride_yn
        + offs_co[None, :] * stride_yc
        + h_idx[:, None] * stride_yh
        + w_idx[:, None] * stride_yw
    )
    y_mask = mask_p[:, None] & mask_co[None, :]
    tl.store(y_ptrs, acc, mask=y_mask)


@triton.jit
def _conv_bn_residual_relu_kernel(
    inp_ptr,             # *const input  [N, C, H, W] (FP32 intermediate)
    w_ptr,               # *const weight [C_out=C, C_in=C, KH, KW]
    identity_ptr,        # *const identity input x [N, C, H, W] (bf16/fp16/fp32)
    out_ptr,             # *mut  output [N, C, H, W] (final dtype == x.dtype)
    bn_w_ptr,            # *const BN gamma [C]
    bn_b_ptr,            # *const BN beta  [C]
    bn_rm_ptr,           # *const BN running_mean [C]
    bn_rv_ptr,           # *const BN running_var  [C]
    N, C, H, W,
    stride_inpn, stride_inpc, stride_inph, stride_inpw,
    stride_wo, stride_wi, stride_wkh, stride_wkw,
    stride_idn, stride_idc, stride_idh, stride_idw,
    stride_outn, stride_outc, stride_outh, stride_outw,
    eps,
    BLOCK_HW: tl.constexpr,
    BLOCK_CO: tl.constexpr,
    BLOCK_CI: tl.constexpr,
):
    pid_hw = tl.program_id(0)
    pid_co = tl.program_id(1)

    P = N * H * W
    offs_p = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    offs_co = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    mask_p = offs_p < P
    mask_co = offs_co < C

    HW = H * W
    n_idx = offs_p // HW
    hw_idx = offs_p % HW
    h_idx = hw_idx // W
    w_idx = hw_idx % W

    acc = tl.zeros((BLOCK_HW, BLOCK_CO), dtype=tl.float32)

    for ci_start in range(0, C, BLOCK_CI):
        offs_ci = ci_start + tl.arange(0, BLOCK_CI)
        mask_ci = offs_ci < C

        for kh in range(3):
            for kw in range(3):
                h_in = h_idx + (kh - 1)
                w_in = w_idx + (kw - 1)
                in_bounds = (h_in >= 0) & (h_in < H) & (w_in >= 0) & (w_in < W) & mask_p

                inp_ptrs = (
                    inp_ptr
                    + n_idx[:, None] * stride_inpn
                    + offs_ci[None, :] * stride_inpc
                    + h_in[:, None] * stride_inph
                    + w_in[:, None] * stride_inpw
                )
                w_ptrs = (
                    w_ptr
                    + offs_co[:, None] * stride_wo
                    + offs_ci[None, :] * stride_wi
                    + kh * stride_wkh
                    + kw * stride_wkw
                )

                inp_mask = in_bounds[:, None] & mask_ci[None, :]
                w_mask = mask_co[:, None] & mask_ci[None, :]

                # load tiles; cast to fp32 for MAC
                inp_tile = tl.load(inp_ptrs, mask=inp_mask, other=0.0).to(tl.float32)
                w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0).to(tl.float32)

                acc = tl.dot(inp_tile, tl.trans(w_tile), acc)

    # BN in fp32
    bn_mean = tl.load(bn_rm_ptr + offs_co, mask=mask_co, other=0.0).to(tl.float32)
    bn_var  = tl.load(bn_rv_ptr + offs_co, mask=mask_co, other=0.0).to(tl.float32)
    gamma   = tl.load(bn_w_ptr + offs_co,  mask=mask_co, other=0.0).to(tl.float32)
    beta    = tl.load(bn_b_ptr + offs_co,  mask=mask_co, other=0.0).to(tl.float32)

    rstd = 1.0 / tl.sqrt(bn_var + eps)
    acc = (acc - bn_mean[None, :]) * rstd[None, :]
    acc = acc * gamma[None, :] + beta[None, :]

    # Residual add in fp32 using identity (casted to fp32 on load)
    id_ptrs = (
        identity_ptr
        + n_idx[:, None] * stride_idn
        + offs_co[None, :] * stride_idc
        + h_idx[:, None] * stride_idh
        + w_idx[:, None] * stride_idw
    )
    id_mask = mask_p[:, None] & mask_co[None, :]
    identity_val = tl.load(id_ptrs, mask=id_mask, other=0.0).to(tl.float32)
    acc = acc + identity_val

    # ReLU
    acc = tl.maximum(acc, 0.0)

    # store to output dtype
    out_ptrs = (
        out_ptr
        + n_idx[:, None] * stride_outn
        + offs_co[None, :] * stride_outc
        + h_idx[:, None] * stride_outh
        + w_idx[:, None] * stride_outw
    )
    out_mask = mask_p[:, None] & mask_co[None, :]
    tl.store(out_ptrs, acc.to(out_ptr.dtype.element_ty), mask=out_mask)


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
    Fused BasicBlock forward using Triton:
      Stage A (kernel 1): Conv3x3(x, w1, s=1, p=1) + BN1 + ReLU (fp32 output)
      Stage B (kernel 2): Conv3x3(Act1, w2, s=1, p=1) + BN2 + ResidualAdd(x) + ReLU (cast to x.dtype on store)

    Constraints:
      - No torch.nn/torch.nn.functional compute ops; all math in Triton kernels.
      - Wrapper only validates, allocates, and launches.

    Returns:
      out: [N, C, H, W], same dtype/device as x
    """
    assert x.is_cuda and conv1_weight.is_cuda and conv2_weight.is_cuda, "All tensors must be on CUDA."
    assert x.ndim == 4, "x must be NCHW"
    N, C, H, W = x.shape

    # Validate shapes / device
    def _check_param(t, shape):
        assert tuple(t.shape) == tuple(shape), f"Expected shape {shape}, got {tuple(t.shape)}"
        assert t.device == x.device, "All tensors must be on the same device"

    _check_param(conv1_weight, (C, C, 3, 3))
    _check_param(conv2_weight, (C, C, 3, 3))
    _check_param(bn1_weight, (C,))
    _check_param(bn1_bias, (C,))
    _check_param(bn1_running_mean, (C,))
    _check_param(bn1_running_var, (C,))
    _check_param(bn2_weight, (C,))
    _check_param(bn2_bias, (C,))
    _check_param(bn2_running_mean, (C,))
    _check_param(bn2_running_var, (C,))

    # Supported dtypes
    assert x.dtype in (torch.float16, torch.bfloat16, torch.float32), "Supported dtypes: fp16, bf16, fp32"

    # Allocate intermediate in fp32 to preserve reference semantics across stages
    act1 = torch.empty((N, C, H, W), device=x.device, dtype=torch.float32)
    out = torch.empty_like(x)

    # Strides (in elements)
    sxn, sxc, sxh, sxw = x.stride()
    syn, syc, syh, syw = act1.stride()
    swo, swi, swkh, swkw = conv1_weight.stride()
    swo2, swi2, swkh2, swkw2 = conv2_weight.stride()
    sidn, sidc, sidh, sidw = x.stride()
    soutn, soutc, south, soutw = out.stride()

    # Launch grid
    P = N * H * W
    BLOCK_HW = 32
    BLOCK_CO = 64
    BLOCK_CI = 64
    grid = (
        triton.cdiv(P, BLOCK_HW),
        triton.cdiv(C, BLOCK_CO),
    )

    eps = 1e-5

    # Kernel 1: conv1 + BN1 + ReLU -> fp32 intermediate
    _conv_bn_relu_kernel[grid](
        x, conv1_weight, act1,
        bn1_weight, bn1_bias, bn1_running_mean, bn1_running_var,
        N, C, H, W,
        sxn, sxc, sxh, sxw,
        swo, swi, swkh, swkw,
        syn, syc, syh, syw,
        eps,
        BLOCK_HW=BLOCK_HW, BLOCK_CO=BLOCK_CO, BLOCK_CI=BLOCK_CI,
        num_warps=4, num_stages=2,
    )

    # Kernel 2: conv2 + BN2 + residual add + ReLU -> cast to x.dtype on store
    _conv_bn_residual_relu_kernel[grid](
        act1, conv2_weight, x, out,
        bn2_weight, bn2_bias, bn2_running_mean, bn2_running_var,
        N, C, H, W,
        syn, syc, syh, syw,
        swo2, swi2, swkh2, swkw2,
        sidn, sidc, sidh, sidw,
        soutn, soutc, south, soutw,
        eps,
        BLOCK_HW=BLOCK_HW, BLOCK_CO=BLOCK_CO, BLOCK_CI=BLOCK_CI,
        num_warps=4, num_stages=2,
    )

    return out