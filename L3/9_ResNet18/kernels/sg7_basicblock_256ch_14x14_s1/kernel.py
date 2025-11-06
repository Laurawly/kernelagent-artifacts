import torch
import triton
import triton.language as tl


@triton.jit
def _conv3x3_bn_relu(
    x_ptr,               # [N, C, H, W] input
    w_ptr,               # [C_out, C_in, 3, 3] conv weights
    gamma_ptr,           # [C_out] BN weight
    beta_ptr,            # [C_out] BN bias
    mean_ptr,            # [C_out] BN running mean
    var_ptr,             # [C_out] BN running var
    y_ptr,               # [N, C_out, H, W] output
    N, C, H, W,          # sizes (C_out == C_in == C for this test)
    eps,                 # BN epsilon
    BLOCK_W: tl.constexpr,  # tile size over W
):
    # Program IDs: tile across width (axis 0), and across rows (n, co, h) (axis 1)
    pid_w = tl.program_id(axis=0)
    pid_row = tl.program_id(axis=1)

    # Decode (n, co, h) from pid_row
    rows_per_n = C * H
    n = pid_row // rows_per_n
    rem = pid_row % rows_per_n
    co = rem // H
    h = rem % H

    # Offsets along width tile
    w_start = pid_w * BLOCK_W
    offs_w = w_start + tl.arange(0, BLOCK_W)
    mask_w = offs_w < W

    # Accumulator for conv in fp32
    acc = tl.zeros([BLOCK_W], dtype=tl.float32)

    # Convolution: 3x3, stride=1, padding=1
    # Loop over input channels
    for ci in tl.range(0, C, 1):
        # Loop over kernel height/width
        # kh, kw in [0, 1, 2] correspond to offsets [-1, 0, 1]
        for kh in range(0, 3):
            ih = h + kh - 1
            h_valid = (ih >= 0) & (ih < H)
            # Row base for input (scalar)
            in_row_base = ((n * C + ci) * H + ih) * W
            for kw in range(0, 3):
                iw = offs_w + (kw - 1)
                w_valid = (iw >= 0) & (iw < W)
                m = mask_w & h_valid & w_valid

                # Load input slice (bf16/half -> f32)
                in_ptrs = x_ptr + in_row_base + iw
                x_vals = tl.load(in_ptrs, mask=m, other=0.0)
                x_vals_f32 = x_vals.to(tl.float32)

                # Load weight scalar for [co, ci, kh, kw]
                w_index = ((co * C + ci) * 3 + kh) * 3 + kw
                w_val = tl.load(w_ptr + w_index)
                w_val_f32 = w_val.to(tl.float32)

                # Accumulate
                acc += x_vals_f32 * w_val_f32

    # BatchNorm (inference): y = gamma * (x - mean) / sqrt(var + eps) + beta
    mean_f32 = tl.load(mean_ptr + co).to(tl.float32)
    var_f32 = tl.load(var_ptr + co).to(tl.float32)
    gamma_f32 = tl.load(gamma_ptr + co).to(tl.float32)
    beta_f32 = tl.load(beta_ptr + co).to(tl.float32)

    inv_std = 1.0 / tl.sqrt(var_f32 + eps)
    y_f32 = (acc - mean_f32) * inv_std
    y_f32 = y_f32 * gamma_f32 + beta_f32

    # ReLU
    y_f32 = tl.maximum(y_f32, 0.0)

    # Store
    out_row_base = ((n * C + co) * H + h) * W
    out_ptrs = y_ptr + out_row_base + offs_w
    tl.store(out_ptrs, y_f32.to(y_ptr.dtype.element_ty), mask=mask_w)


@triton.jit
def _conv3x3_bn_add_relu(
    x_ptr,               # [N, C, H, W] identity input for residual add
    y1_ptr,              # [N, C, H, W] input from first block (after BN+ReLU)
    w2_ptr,              # [C_out, C_in, 3, 3] conv2 weights
    gamma2_ptr,          # [C_out] BN2 weight
    beta2_ptr,           # [C_out] BN2 bias
    mean2_ptr,           # [C_out] BN2 running mean
    var2_ptr,            # [C_out] BN2 running var
    y2_ptr,              # [N, C_out, H, W] final output
    N, C, H, W,          # sizes
    eps,                 # BN epsilon
    BLOCK_W: tl.constexpr,
):
    # Program IDs: tile across width (axis 0), and across rows (n, co, h) (axis 1)
    pid_w = tl.program_id(axis=0)
    pid_row = tl.program_id(axis=1)

    # Decode (n, co, h) from pid_row
    rows_per_n = C * H
    n = pid_row // rows_per_n
    rem = pid_row % rows_per_n
    co = rem // H
    h = rem % H

    # Offsets along width tile
    w_start = pid_w * BLOCK_W
    offs_w = w_start + tl.arange(0, BLOCK_W)
    mask_w = offs_w < W

    # Accumulator for conv2 in fp32
    acc2 = tl.zeros([BLOCK_W], dtype=tl.float32)

    # Convolution: 3x3 on y1
    for ci in tl.range(0, C, 1):
        for kh in range(0, 3):
            ih = h + kh - 1
            h_valid = (ih >= 0) & (ih < H)
            in_row_base = ((n * C + ci) * H + ih) * W
            for kw in range(0, 3):
                iw = offs_w + (kw - 1)
                w_valid = (iw >= 0) & (iw < W)
                m = mask_w & h_valid & w_valid

                in_ptrs = y1_ptr + in_row_base + iw
                x_vals = tl.load(in_ptrs, mask=m, other=0.0)
                x_vals_f32 = x_vals.to(tl.float32)

                w_index = ((co * C + ci) * 3 + kh) * 3 + kw
                w_val = tl.load(w2_ptr + w_index)
                w_val_f32 = w_val.to(tl.float32)

                acc2 += x_vals_f32 * w_val_f32

    # BN2
    mean2_f32 = tl.load(mean2_ptr + co).to(tl.float32)
    var2_f32 = tl.load(var2_ptr + co).to(tl.float32)
    gamma2_f32 = tl.load(gamma2_ptr + co).to(tl.float32)
    beta2_f32 = tl.load(beta2_ptr + co).to(tl.float32)

    inv_std2 = 1.0 / tl.sqrt(var2_f32 + eps)
    y2_f32 = (acc2 - mean2_f32) * inv_std2
    y2_f32 = y2_f32 * gamma2_f32 + beta2_f32

    # Residual add with identity input x
    out_row_base = ((n * C + co) * H + h) * W
    id_ptrs = x_ptr + out_row_base + offs_w
    identity_vals = tl.load(id_ptrs, mask=mask_w, other=0.0).to(tl.float32)

    y2_f32 = y2_f32 + identity_vals

    # Final ReLU
    y2_f32 = tl.maximum(y2_f32, 0.0)

    # Store
    out_ptrs = y2_ptr + out_row_base + offs_w
    tl.store(out_ptrs, y2_f32.to(y2_ptr.dtype.element_ty), mask=mask_w)


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
    Fused BasicBlock (NCHW) implemented with Triton kernels.

    Fused stages:
    - Kernel 1: Conv3x3(s=1,p=1) + BatchNorm (inference) + ReLU
    - Kernel 2: Conv3x3(s=1,p=1) + BatchNorm (inference) + Add(identity) + ReLU

    Notes:
    - All math executes inside Triton kernels. The wrapper only validates inputs,
      allocates outputs, and launches kernels.
    - Accumulations and BN arithmetic are performed in float32 for numerical stability.
      I/O tensors may be bfloat16/float16; stores are cast back to the output dtype.
    - The skip connection uses the original input x (identity) as in the reference.

    Args follow the test specification. Expected shapes:
      x:               [N, C, H, W]
      conv1_weight:    [C, C, 3, 3]
      bn1_*(gamma,beta,mean,var): [C]
      conv2_weight:    [C, C, 3, 3]
      bn2_*(gamma,beta,mean,var): [C]

    Returns:
      torch.Tensor with shape [N, C, H, W], same device and dtype as x.
    """
    assert x.is_cuda, "Input must be on CUDA device"
    device = x.device

    # Validate shapes (minimal checks tailored to the test)
    N, C, H, W = x.shape
    assert conv1_weight.shape == (C, C, 3, 3)
    assert conv2_weight.shape == (C, C, 3, 3)
    for t in [bn1_weight, bn1_bias, bn1_running_mean, bn1_running_var,
              bn2_weight, bn2_bias, bn2_running_mean, bn2_running_var]:
        assert t.shape == (C,), "BatchNorm parameter shape must be (C,)"

    # Dtypes: allow bf16/fp16/fp32 inputs; all compute done in fp32 in kernels
    out_dtype = x.dtype
    assert out_dtype in (torch.bfloat16, torch.float16, torch.float32), "Unsupported input dtype"

    # Allocate intermediate and output
    y1 = torch.empty_like(x, device=device, dtype=out_dtype)
    y2 = torch.empty_like(x, device=device, dtype=out_dtype)

    # Launch configs
    BLOCK_W = 64  # works well for W=14; masked stores handle tail
    grid = (triton.cdiv(W, BLOCK_W), N * C * H)

    # BN epsilon
    eps = 1e-5

    # Launch kernel 1: conv1 + BN1 + ReLU
    _conv3x3_bn_relu[grid](
        x,
        conv1_weight,
        bn1_weight, bn1_bias, bn1_running_mean, bn1_running_var,
        y1,
        N, C, H, W,
        eps,
        BLOCK_W=BLOCK_W,
        num_warps=4,
    )

    # Launch kernel 2: conv2 + BN2 + Add(x) + ReLU
    _conv3x3_bn_add_relu[grid](
        x,
        y1,
        conv2_weight,
        bn2_weight, bn2_bias, bn2_running_mean, bn2_running_var,
        y2,
        N, C, H, W,
        eps,
        BLOCK_W=BLOCK_W,
        num_warps=4,
    )

    return y2