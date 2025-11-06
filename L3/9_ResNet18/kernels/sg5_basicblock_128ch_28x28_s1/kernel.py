import torch
import triton
import triton.language as tl

# Fused residual block for: conv3x3(s=1,p=1) -> BN -> ReLU -> conv3x3(s=1,p=1) -> BN -> Add(residual) -> ReLU
# Layout: NCHW, inputs/weights BF16; compute in FP32. Intermediate between conv1 and conv2 kept in FP32
# to match reference semantics and accuracy.
#
# Fusion rationale:
# - We fuse BatchNorm and activation into each convolution stage.
# - We do not fuse both convolutions into a single kernel because conv2 depends on a 3x3 spatial
#   neighborhood of conv1 outputs, which are computed by different Triton programs with no cross-program
#   synchronization available. Hence, writing the intermediate to global memory is required.


@triton.jit
def _conv3x3_bn_relu_kernel(
    x_ptr,                  # *bf16 [N, C, H, W]
    w_ptr,                  # *bf16 [C_out, C_in, 3, 3]
    y_ptr,                  # *fp32 or *bf16 [N, C_out, H, W] (we'll store to pointer dtype)
    bn_weight_ptr,          # *bf16 [C_out]
    bn_bias_ptr,            # *bf16 [C_out]
    bn_mean_ptr,            # *bf16 [C_out]
    bn_var_ptr,             # *bf16 [C_out]
    N, C_in, H, W, C_out,   # int32
    stride_xn, stride_xc, stride_xh, stride_xw,      # int64
    stride_woc, stride_wic, stride_wkh, stride_wkw,  # int64
    stride_yn, stride_yc, stride_yh, stride_yw,      # int64
    EPS: tl.constexpr,                                # float compile-time constant
    BLOCK_OC: tl.constexpr,                           # tile size along output channels
    BLOCK_P: tl.constexpr,                            # tile size along positions (N*H*W)
    CHUNK_C: tl.constexpr                             # chunk size along input channels
):
    # Program IDs
    pid_oc = tl.program_id(0)
    pid_p = tl.program_id(1)

    # Offsets
    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    p_offs = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)
    P = N * H * W

    # Masks
    oc_mask = oc_offs < C_out
    p_mask = p_offs < P

    # Map flattened positions to (n, h, w)
    HW = H * W
    n_idx = p_offs // HW
    rem = p_offs % HW
    h_idx = rem // W
    w_idx = rem % W

    # Safe indexing for masked positions
    n_idx = tl.where(p_mask, n_idx, 0)
    h_idx = tl.where(p_mask, h_idx, 0)
    w_idx = tl.where(p_mask, w_idx, 0)

    # FP32 accumulator
    acc = tl.zeros((BLOCK_OC, BLOCK_P), dtype=tl.float32)

    ic_range = tl.arange(0, CHUNK_C)

    # Convolution 3x3, stride=1, padding=1
    for ic0 in range(0, C_in, CHUNK_C):
        ic_idxs = ic0 + ic_range
        ic_mask = ic_idxs < C_in

        for ky in range(3):
            dy = ky - 1
            hi = h_idx + dy
            for kx in range(3):
                dx = kx - 1
                wi = w_idx + dx

                # Spatial validity mask
                valid_sp = (hi >= 0) & (hi < H) & (wi >= 0) & (wi < W)
                load_in_mask = ic_mask[:, None] & valid_sp[None, :] & p_mask[None, :]

                # Input tile [CHUNK_C, BLOCK_P]
                in_ptrs = (
                    x_ptr
                    + n_idx[None, :] * stride_xn
                    + ic_idxs[:, None] * stride_xc
                    + hi[None, :] * stride_xh
                    + wi[None, :] * stride_xw
                )
                x_tile = tl.load(in_ptrs, mask=load_in_mask, other=0.0).to(tl.float32)

                # Weight tile [BLOCK_OC, CHUNK_C]
                w_ptrs = (
                    w_ptr
                    + oc_offs[:, None] * stride_woc
                    + ic_idxs[None, :] * stride_wic
                    + ky * stride_wkh
                    + kx * stride_wkw
                )
                w_mask = oc_mask[:, None] & ic_mask[None, :]
                w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0).to(tl.float32)

                # Accumulate: [BLOCK_OC, CHUNK_C] x [CHUNK_C, BLOCK_P] -> [BLOCK_OC, BLOCK_P]
                acc += tl.dot(w_tile, x_tile)

    # BatchNorm (inference)
    bn_mean = tl.load(bn_mean_ptr + oc_offs, mask=oc_mask, other=0.0).to(tl.float32)
    bn_var = tl.load(bn_var_ptr + oc_offs, mask=oc_mask, other=0.0).to(tl.float32)
    bn_w = tl.load(bn_weight_ptr + oc_offs, mask=oc_mask, other=0.0).to(tl.float32)
    bn_b = tl.load(bn_bias_ptr + oc_offs, mask=oc_mask, other=0.0).to(tl.float32)

    inv_std = 1.0 / tl.sqrt(bn_var + EPS)
    acc = (acc - bn_mean[:, None]) * inv_std[:, None]
    acc = acc * bn_w[:, None] + bn_b[:, None]

    # ReLU
    acc = tl.maximum(acc, 0.0)

    # Store y in the dtype of y_ptr (we pass FP32 for the intermediate to preserve accuracy)
    y_ptrs = (
        y_ptr
        + n_idx[None, :] * stride_yn
        + oc_offs[:, None] * stride_yc
        + h_idx[None, :] * stride_yh
        + w_idx[None, :] * stride_yw
    )
    y_mask = oc_mask[:, None] & p_mask[None, :]
    tl.store(y_ptrs, acc.to(y_ptr.dtype.element_ty), mask=y_mask)


@triton.jit
def _conv3x3_bn_add_relu_kernel(
    x_res_ptr,              # *bf16 residual input [N, C, H, W]
    x_ptr,                  # *fp32 or *bf16 input (stage1 output) [N, C, H, W]
    w_ptr,                  # *bf16 [C_out, C_in, 3, 3]
    y_ptr,                  # *fp32 or *bf16 [N, C_out, H, W] (store to pointer dtype)
    bn_weight_ptr,          # *bf16 [C_out]
    bn_bias_ptr,            # *bf16 [C_out]
    bn_mean_ptr,            # *bf16 [C_out]
    bn_var_ptr,             # *bf16 [C_out]
    N, C_in, H, W, C_out,   # int32
    stride_xn, stride_xc, stride_xh, stride_xw,      # int64 (stage1 output strides)
    stride_woc, stride_wic, stride_wkh, stride_wkw,  # int64
    stride_yn, stride_yc, stride_yh, stride_yw,      # int64
    stride_rn, stride_rc, stride_rh, stride_rw,      # int64 (residual input x strides)
    EPS: tl.constexpr,                                # float compile-time constant
    BLOCK_OC: tl.constexpr,                           # tile size along output channels
    BLOCK_P: tl.constexpr,                            # tile size along positions (N*H*W)
    CHUNK_C: tl.constexpr                             # chunk size along input channels
):
    # Program IDs
    pid_oc = tl.program_id(0)
    pid_p = tl.program_id(1)

    # Offsets
    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    p_offs = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)
    P = N * H * W

    # Masks
    oc_mask = oc_offs < C_out
    p_mask = p_offs < P

    # Map flattened positions to (n, h, w)
    HW = H * W
    n_idx = p_offs // HW
    rem = p_offs % HW
    h_idx = rem // W
    w_idx = rem % W

    # Safe indexing for masked positions
    n_idx = tl.where(p_mask, n_idx, 0)
    h_idx = tl.where(p_mask, h_idx, 0)
    w_idx = tl.where(p_mask, w_idx, 0)

    # FP32 accumulator
    acc = tl.zeros((BLOCK_OC, BLOCK_P), dtype=tl.float32)

    ic_range = tl.arange(0, CHUNK_C)

    # Convolution 3x3, stride=1, padding=1 on stage1 output
    for ic0 in range(0, C_in, CHUNK_C):
        ic_idxs = ic0 + ic_range
        ic_mask = ic_idxs < C_in

        for ky in range(3):
            dy = ky - 1
            hi = h_idx + dy
            for kx in range(3):
                dx = kx - 1
                wi = w_idx + dx

                # Spatial validity mask
                valid_sp = (hi >= 0) & (hi < H) & (wi >= 0) & (wi < W)
                load_in_mask = ic_mask[:, None] & valid_sp[None, :] & p_mask[None, :]

                # Input tile [CHUNK_C, BLOCK_P] from stage1 output (may be fp32)
                in_ptrs = (
                    x_ptr
                    + n_idx[None, :] * stride_xn
                    + ic_idxs[:, None] * stride_xc
                    + hi[None, :] * stride_xh
                    + wi[None, :] * stride_xw
                )
                x_tile = tl.load(in_ptrs, mask=load_in_mask, other=0.0).to(tl.float32)

                # Weight tile [BLOCK_OC, CHUNK_C]
                w_ptrs = (
                    w_ptr
                    + oc_offs[:, None] * stride_woc
                    + ic_idxs[None, :] * stride_wic
                    + ky * stride_wkh
                    + kx * stride_wkw
                )
                w_mask = oc_mask[:, None] & ic_mask[None, :]
                w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0).to(tl.float32)

                # Accumulate: [BLOCK_OC, CHUNK_C] x [CHUNK_C, BLOCK_P] -> [BLOCK_OC, BLOCK_P]
                acc += tl.dot(w_tile, x_tile)

    # BatchNorm (inference)
    bn_mean = tl.load(bn_mean_ptr + oc_offs, mask=oc_mask, other=0.0).to(tl.float32)
    bn_var = tl.load(bn_var_ptr + oc_offs, mask=oc_mask, other=0.0).to(tl.float32)
    bn_w = tl.load(bn_weight_ptr + oc_offs, mask=oc_mask, other=0.0).to(tl.float32)
    bn_b = tl.load(bn_bias_ptr + oc_offs, mask=oc_mask, other=0.0).to(tl.float32)

    inv_std = 1.0 / tl.sqrt(bn_var + EPS)
    acc = (acc - bn_mean[:, None]) * inv_std[:, None]
    acc = acc * bn_w[:, None] + bn_b[:, None]

    # Residual add (residual x is BF16; cast to FP32)
    res_ptrs = (
        x_res_ptr
        + n_idx[None, :] * stride_rn
        + oc_offs[:, None] * stride_rc
        + h_idx[None, :] * stride_rh
        + w_idx[None, :] * stride_rw
    )
    res = tl.load(res_ptrs, mask=(oc_mask[:, None] & p_mask[None, :]), other=0.0).to(tl.float32)
    acc = acc + res

    # ReLU
    acc = tl.maximum(acc, 0.0)

    # Store y in the dtype of y_ptr (we'll pass FP32 to match reference accuracy)
    y_ptrs = (
        y_ptr
        + n_idx[None, :] * stride_yn
        + oc_offs[:, None] * stride_yc
        + h_idx[None, :] * stride_yh
        + w_idx[None, :] * stride_yw
    )
    y_mask = oc_mask[:, None] & p_mask[None, :]
    tl.store(y_ptrs, acc.to(y_ptr.dtype.element_ty), mask=y_mask)


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
    Residual basic block:
      y = ReLU(BN2(Conv2(ReLU(BN1(Conv1(x))))) + x)

    - All math runs inside Triton kernels.
    - We keep the intermediate (after first Conv-BN-ReLU) in FP32 to match reference semantics/accuracy.
    """
    # Validate
    assert x.is_cuda, "x must be CUDA"
    assert x.dtype == torch.bfloat16, "Test policy uses bf16 inputs"
    device = x.device
    N, C, H, W = x.shape
    assert conv1_weight.shape == (C, C, 3, 3)
    assert conv2_weight.shape == (C, C, 3, 3)
    for t in [conv1_weight, conv2_weight,
              bn1_weight, bn1_bias, bn1_running_mean, bn1_running_var,
              bn2_weight, bn2_bias, bn2_running_mean, bn2_running_var]:
        assert t.is_cuda and t.dtype == torch.bfloat16

    # Allocate intermediate (FP32 to avoid precision loss between convs) and output (FP32 to match reference)
    y1 = torch.empty((N, C, H, W), device=device, dtype=torch.float32)
    y2 = torch.empty((N, C, H, W), device=device, dtype=torch.float32)

    # Strides in elements
    sxn, sxc, sxh, sxw = x.stride()
    syn, syc, syh, syw = y1.stride()
    s2yn, s2yc, s2yh, s2yw = y2.stride()
    sw1oc, sw1ic, sw1kh, sw1kw = conv1_weight.stride()
    sw2oc, sw2ic, sw2kh, sw2kw = conv2_weight.stride()
    srn, src, srh, srw = x.stride()

    # Tiles
    P = N * H * W
    BLOCK_OC = 64
    BLOCK_P = 128
    CHUNK_C = 32

    grid = (triton.cdiv(C, BLOCK_OC), triton.cdiv(P, BLOCK_P))

    # Stage 1: conv -> bn -> relu (store FP32)
    _conv3x3_bn_relu_kernel[grid](
        x, conv1_weight, y1,
        bn1_weight, bn1_bias, bn1_running_mean, bn1_running_var,
        N, C, H, W, C,
        sxn, sxc, sxh, sxw,
        sw1oc, sw1ic, sw1kh, sw1kw,
        syn, syc, syh, syw,
        EPS=1e-5,
        BLOCK_OC=BLOCK_OC,
        BLOCK_P=BLOCK_P,
        CHUNK_C=CHUNK_C,
        num_warps=4,
        num_stages=3,
    )

    # Stage 2: conv -> bn -> add(residual) -> relu (store FP32)
    _conv3x3_bn_add_relu_kernel[grid](
        x, y1, conv2_weight, y2,
        bn2_weight, bn2_bias, bn2_running_mean, bn2_running_var,
        N, C, H, W, C,
        syn, syc, syh, syw,       # stage1 output strides
        sw2oc, sw2ic, sw2kh, sw2kw,
        s2yn, s2yc, s2yh, s2yw,
        srn, src, srh, srw,       # residual x strides
        EPS=1e-5,
        BLOCK_OC=BLOCK_OC,
        BLOCK_P=BLOCK_P,
        CHUNK_C=CHUNK_C,
        num_warps=4,
        num_stages=3,
    )

    return y2