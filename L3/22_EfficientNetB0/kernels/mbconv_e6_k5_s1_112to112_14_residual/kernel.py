import torch
import triton
import triton.language as tl


# ============================
# Pointwise 1x1 GEMM + BN + ReLU6
# Used for expand stage (C_in -> C_mid)
# ============================

_pw_configs = [
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
]


@triton.autotune(configs=_pw_configs, key=["S", "K", "N"])
@triton.jit
def _pw1_bn_relu6_kernel(
    x_ptr,                      # BF16 input [N, C_in, H, W]
    w_ptr,                      # BF16 weights [N_out, K_in, 1, 1] - treated as [N_out, K_in]
    out_ptr,                    # FP32 output [N, N_out, H, W]
    gamma_ptr, beta_ptr, mean_ptr, var_ptr,  # FP32 BN params for output channels
    eps,                        # FP32
    # shapes
    N, H, W, K, N_out, S,       # ints: K=C_in, N_out=C_mid, S=N*H*W
    # strides (in elements, not bytes)
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wo, stride_wi,
    stride_on, stride_oc, stride_oh, stride_ow,
    # tile sizes
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid_m = tl.program_id(0)  # tiles over S (spatial-batch)
    pid_n = tl.program_id(1)  # tiles over output channels

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < S
    mask_n = offs_n < N_out

    # Decode offs_m -> (n, h, w)
    HW = H * W
    n_idx = offs_m // HW
    rem = offs_m - n_idx * HW
    h_idx = rem // W
    w_idx = rem - h_idx * W

    # Base input address for each row in this tile (without channel offset)
    x_base = n_idx * stride_xn + h_idx * stride_xh + w_idx * stride_xw

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # K loop
    for kk in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = kk * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        # A tile: [BLOCK_M, BLOCK_K] from input x
        a_ptrs = x_ptr + x_base[:, None] + offs_k[None, :] * stride_xc
        a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        a = a.to(tl.float32)

        # B tile: [BLOCK_K, BLOCK_N] from weights (transposed use)
        b_ptrs = w_ptr + offs_k[:, None] * stride_wi + offs_n[None, :] * stride_wo
        b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        b = b.to(tl.float32)

        # Multiply-accumulate: acc += sum_k a[:, k] * b[k, :]
        # Compute partial = sum over k axis (axis=1 of the 3D tensor)
        partial = tl.sum(a[:, :, None] * b[None, :, :], axis=1)
        acc += partial

    # BatchNorm inference per output-channel j
    gamma = tl.load(gamma_ptr + offs_n, mask=mask_n, other=0.0)
    beta = tl.load(beta_ptr + offs_n, mask=mask_n, other=0.0)
    mean = tl.load(mean_ptr + offs_n, mask=mask_n, other=0.0)
    var = tl.load(var_ptr + offs_n, mask=mask_n, other=0.0)

    inv_std = 1.0 / tl.sqrt(var + eps)
    scale = gamma * inv_std
    bias = beta - mean * scale

    # Apply BN and ReLU6
    acc = acc * scale[None, :] + bias[None, :]
    acc = tl.maximum(acc, 0.0)
    acc = tl.minimum(acc, 6.0)

    # Store to output [N, N_out, H, W]
    out_ptrs = out_ptr + (
        n_idx[:, None] * stride_on
        + offs_n[None, :] * stride_oc
        + h_idx[:, None] * stride_oh
        + w_idx[:, None] * stride_ow
    )
    tl.store(out_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


# ============================
# Depthwise 5x5 + BN + ReLU6
# Used for middle depthwise stage (C_mid groups)
# ============================

@triton.jit
def _dw5x5_bn_relu6_kernel(
    inp_ptr,                    # FP32 input [N, C, H, W]
    w_ptr,                      # BF16 weights [C, 1, 5, 5]
    out_ptr,                    # FP32 output [N, C, H, W]
    gamma_ptr, beta_ptr, mean_ptr, var_ptr,  # FP32 BN params per C
    eps,                        # FP32
    # shapes
    N, C, H, W, S,              # S=N*H*W
    # strides
    stride_in_n, stride_in_c, stride_in_h, stride_in_w,
    stride_w_c, stride_w_kh, stride_w_kw,
    stride_out_n, stride_out_c, stride_out_h, stride_out_w,
    # tiling
    BLOCK_C: tl.constexpr, BLOCK_S: tl.constexpr, PAD: tl.constexpr
):
    pid_c = tl.program_id(0)  # tile over channels
    pid_s = tl.program_id(1)  # tile over spatial positions

    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    offs_s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    mask_c = offs_c < C
    mask_s = offs_s < S

    # Decode offs_s -> (n, h, w)
    HW = H * W
    n_idx = offs_s // HW
    rem = offs_s - n_idx * HW
    h_idx = rem // W
    w_idx = rem - h_idx * W

    # Base output addresses
    out_base = n_idx * stride_out_n + h_idx * stride_out_h + w_idx * stride_out_w

    acc = tl.zeros((BLOCK_S, BLOCK_C), dtype=tl.float32)

    for kh in range(0, 5):
        hh = h_idx + kh - PAD
        in_h_ok = (hh >= 0) & (hh < H)
        for kw in range(0, 5):
            ww = w_idx + kw - PAD
            in_w_ok = (ww >= 0) & (ww < W)
            valid_spatial = in_h_ok & in_w_ok

            in_base = n_idx * stride_in_n + hh * stride_in_h + ww * stride_in_w
            in_ptrs = inp_ptr + in_base[:, None] + offs_c[None, :] * stride_in_c
            x = tl.load(in_ptrs, mask=mask_s[:, None] & mask_c[None, :] & valid_spatial[:, None], other=0.0)
            # weights per channel
            w_vec = tl.load(w_ptr + offs_c * stride_w_c + kh * stride_w_kh + kw * stride_w_kw,
                            mask=mask_c, other=0.0)
            w_vec = w_vec.to(tl.float32)
            acc += x * w_vec[None, :]

    # BatchNorm inference per channel
    gamma = tl.load(gamma_ptr + offs_c, mask=mask_c, other=0.0)
    beta = tl.load(beta_ptr + offs_c, mask=mask_c, other=0.0)
    mean = tl.load(mean_ptr + offs_c, mask=mask_c, other=0.0)
    var = tl.load(var_ptr + offs_c, mask=mask_c, other=0.0)

    inv_std = 1.0 / tl.sqrt(var + eps)
    scale = gamma * inv_std
    bias = beta - mean * scale

    acc = acc * scale[None, :] + bias[None, :]
    acc = tl.maximum(acc, 0.0)
    acc = tl.minimum(acc, 6.0)

    out_ptrs = out_ptr + out_base[:, None] + offs_c[None, :] * stride_out_c
    tl.store(out_ptrs, acc, mask=mask_s[:, None] & mask_c[None, :])


# ============================
# Pointwise 1x1 GEMM + BN + Residual Add
# Used for projection stage (C_mid -> C_out) + residual connection
# ============================

@triton.autotune(configs=_pw_configs, key=["S", "K", "N"])
@triton.jit
def _pw2_bn_residual_kernel(
    x_ptr,                      # FP32 input [N, K_in, H, W] from depthwise output
    w_ptr,                      # BF16 weights [N_out, K_in, 1, 1]
    out_ptr,                    # BF16 output [N, N_out, H, W]
    gamma_ptr, beta_ptr, mean_ptr, var_ptr,  # FP32 BN params per output channel
    eps,                        # FP32
    skip_ptr,                   # BF16 residual input [N, N_out, H, W] (original x)
    # shapes
    N, H, W, K, N_out, S,
    # strides
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wo, stride_wi,
    stride_skip_n, stride_skip_c, stride_skip_h, stride_skip_w,
    stride_on, stride_oc, stride_oh, stride_ow,
    # tiling
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid_m = tl.program_id(0)  # tiles over S
    pid_n = tl.program_id(1)  # tiles over output channels

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < S
    mask_n = offs_n < N_out

    # Decode (n, h, w)
    HW = H * W
    n_idx = offs_m // HW
    rem = offs_m - n_idx * HW
    h_idx = rem // W
    w_idx = rem - h_idx * W

    x_base = n_idx * stride_xn + h_idx * stride_xh + w_idx * stride_xw

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for kk in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = kk * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        a_ptrs = x_ptr + x_base[:, None] + offs_k[None, :] * stride_xc
        a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)  # FP32
        a = a.to(tl.float32)

        b_ptrs = w_ptr + offs_k[:, None] * stride_wi + offs_n[None, :] * stride_wo
        b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0).to(tl.float32)

        partial = tl.sum(a[:, :, None] * b[None, :, :], axis=1)
        acc += partial

    # BN per output channel
    gamma = tl.load(gamma_ptr + offs_n, mask=mask_n, other=0.0)
    beta = tl.load(beta_ptr + offs_n, mask=mask_n, other=0.0)
    mean = tl.load(mean_ptr + offs_n, mask=mask_n, other=0.0)
    var = tl.load(var_ptr + offs_n, mask=mask_n, other=0.0)

    inv_std = 1.0 / tl.sqrt(var + eps)
    scale = gamma * inv_std
    bias = beta - mean * scale

    y = acc * scale[None, :] + bias[None, :]

    # Residual add from skip (BF16)
    skip_ptrs = skip_ptr + (
        n_idx[:, None] * stride_skip_n
        + offs_n[None, :] * stride_skip_c
        + h_idx[:, None] * stride_skip_h
        + w_idx[:, None] * stride_skip_w
    )
    skip_val = tl.load(skip_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0).to(tl.float32)
    y = y + skip_val

    # Store BF16 output
    out_ptrs = out_ptr + (
        n_idx[:, None] * stride_on
        + offs_n[None, :] * stride_oc
        + h_idx[:, None] * stride_oh
        + w_idx[:, None] * stride_ow
    )
    tl.store(out_ptrs, y.to(tl.bfloat16), mask=mask_m[:, None] & mask_n[None, :])


def kernel_function(
    x,
    w_expand,
    gamma_expand, beta_expand, mean_expand, var_expand,
    w_depthwise,
    gamma_dw, beta_dw, mean_dw, var_dw,
    w_project,
    gamma_proj, beta_proj, mean_proj, var_proj,
):
    """
    Fused MBConv (EfficientNet-style) subgraph using Triton.

    Fused stages across three Triton kernels:
    1) Expand 1x1 convolution (C_in -> C_mid) + BatchNorm-inference + ReLU6 (fully fused).
       Implemented as tiled GEMM over S=N*H*W and K=C_in, N=C_mid. Accumulate in fp32,
       apply BN+ReLU6 in-kernel, store intermediate in fp32 tensor y1.

    2) Depthwise 5x5 convolution (groups=C_mid, stride=1, padding=2) + BatchNorm + ReLU6 (fused).
       Direct sliding-window accumulation in fp32 across 5x5 neighborhood per channel/tile, then BN+ReLU6.
       Store intermediate in fp32 tensor y2.

    3) Projection 1x1 convolution (C_mid -> C_out) + BatchNorm + residual add (input skip).
       Tiled GEMM in fp32 (accumulate) then BN, add BF16 skip-connection, and write BF16 output.

    Notes:
    - Wrapper performs only validation, allocation, and kernel launch. All math executes in Triton kernels.
    - Intermediate tensors y1 and y2 use float32 to match reference numerics more closely; inputs/outputs use bfloat16.
    - BN parameters (gamma, beta, mean, var) are read in-kernel and combined as per inference formula.
    - Activation ReLU6 is applied only in the first two stages per the test; final stage has BN then residual add.

    Args:
        x:            [N, C_in, H, W] bfloat16 input (contiguous NCHW)
        w_expand:     [C_mid, C_in, 1, 1] bfloat16
        gamma_expand, beta_expand, mean_expand, var_expand: fp32 tensors of shape [C_mid]
        w_depthwise:  [C_mid, 1, 5, 5] bfloat16
        gamma_dw, beta_dw, mean_dw, var_dw: fp32 tensors of shape [C_mid]
        w_project:    [C_out, C_mid, 1, 1] bfloat16
        gamma_proj, beta_proj, mean_proj, var_proj: fp32 tensors of shape [C_out]

    Returns:
        y: [N, C_out, H, W] bfloat16 output after MBConv block with residual connection
    """
    # Basic validations
    assert isinstance(x, torch.Tensor) and x.is_cuda, "x must be a CUDA tensor"
    assert x.dtype == torch.bfloat16, "Input dtype must be bfloat16"
    device = x.device
    N, C_in, H, W = x.shape
    C_mid = w_expand.shape[0]
    assert w_expand.shape[:2] == (C_mid, C_in)
    assert w_expand.shape[2:] == (1, 1)
    assert w_depthwise.shape == (C_mid, 1, 5, 5)
    C_out = w_project.shape[0]
    assert w_project.shape[:2] == (C_out, C_mid)
    assert w_project.shape[2:] == (1, 1)
    # BN params dtype/device
    for t in [gamma_expand, beta_expand, mean_expand, var_expand,
              gamma_dw, beta_dw, mean_dw, var_dw,
              gamma_proj, beta_proj, mean_proj, var_proj]:
        assert isinstance(t, torch.Tensor) and t.is_cuda and t.dtype == torch.float32
    assert gamma_expand.numel() == C_mid and beta_expand.numel() == C_mid
    assert mean_expand.numel() == C_mid and var_expand.numel() == C_mid
    assert gamma_dw.numel() == C_mid and beta_dw.numel() == C_mid
    assert mean_dw.numel() == C_mid and var_dw.numel() == C_mid
    assert gamma_proj.numel() == C_out and beta_proj.numel() == C_out
    assert mean_proj.numel() == C_out and var_proj.numel() == C_out

    # Allocate intermediates (fp32) and output (bf16)
    y1 = torch.empty((N, C_mid, H, W), device=device, dtype=torch.float32)
    y2 = torch.empty((N, C_mid, H, W), device=device, dtype=torch.float32)
    y_out = torch.empty((N, C_out, H, W), device=device, dtype=torch.bfloat16)

    # Common constants and dims
    eps = 1e-5
    S = N * H * W

    # ============== Launch expand 1x1 + BN + ReLU6 ==============
    # Grid: tiles over S and C_mid
    def grid_pw1(META):
        return (triton.cdiv(S, META["BLOCK_M"]), triton.cdiv(C_mid, META["BLOCK_N"]))

    _pw1_bn_relu6_kernel[grid_pw1](
        x, w_expand, y1,
        gamma_expand, beta_expand, mean_expand, var_expand,
        eps,
        N, H, W, C_in, C_mid, S,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        w_expand.stride(0), w_expand.stride(1),
        y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
    )

    # ============== Launch depthwise 5x5 + BN + ReLU6 ==============
    # Grid: tiles over channels and spatial positions
    BLOCK_C = 64
    BLOCK_S = 64
    grid_dw = (triton.cdiv(C_mid, BLOCK_C), triton.cdiv(S, BLOCK_S))
    _dw5x5_bn_relu6_kernel[grid_dw](
        y1, w_depthwise, y2,
        gamma_dw, beta_dw, mean_dw, var_dw,
        eps,
        N, C_mid, H, W, S,
        y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
        w_depthwise.stride(0), w_depthwise.stride(2), w_depthwise.stride(3),
        y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3),
        BLOCK_C=BLOCK_C, BLOCK_S=BLOCK_S, PAD=2,
    )

    # ============== Launch projection 1x1 + BN + Residual ==============
    def grid_pw2(META):
        return (triton.cdiv(S, META["BLOCK_M"]), triton.cdiv(C_out, META["BLOCK_N"]))

    _pw2_bn_residual_kernel[grid_pw2](
        y2, w_project, y_out,
        gamma_proj, beta_proj, mean_proj, var_proj,
        eps,
        x,  # residual
        N, H, W, C_mid, C_out, S,
        y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3),
        w_project.stride(0), w_project.stride(1),
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        y_out.stride(0), y_out.stride(1), y_out.stride(2), y_out.stride(3),
    )

    return y_out