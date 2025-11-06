import math
import torch
import triton
import triton.language as tl


# --------------------------------------------------------------------------------------
# Fused GEMM + Bias + Sigmoid + Scaling + ResidualAdd Triton kernel
# Computes: Y = sigmoid(X @ W^T + b) * scale + (X @ W^T + b)
#
# Notes on fusion:
# - We fuse the full pipeline in a single kernel:
#   1) Matrix multiplication: X[M, K] @ W^T[K, N]  -> accumulator in fp32
#   2) Bias add: + b[N]
#   3) Activation: sigmoid()
#   4) Scaling: * scale
#   5) Residual add: + linear_output
# - Bias and activation/epilogue math are performed directly on the accumulator to avoid
#   extra global memory traffic and kernel launches.
# --------------------------------------------------------------------------------------

# A small but effective autotune set for large square-ish GEMMs
_autotune_configs = [
    triton.Config(
        {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8},
        num_stages=3, num_warps=8
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8},
        num_stages=4, num_warps=8
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8},
        num_stages=4, num_warps=8
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8},
        num_stages=3, num_warps=4
    ),
]


@triton.autotune(configs=_autotune_configs, key=["M", "N", "K"])
@triton.jit
def _fused_gemm_sigmoid_residual_kernel(
    a_ptr,  # X: [M, K]
    b_ptr,  # W^T: [K, N]
    bias_ptr,  # b: [N]
    out_ptr,   # Y: [M, N]
    M, N, K,
    stride_am, stride_ak,  # strides for X
    stride_bk, stride_bn,  # strides for W^T
    stride_om, stride_on,  # strides for Y
    scale,  # scaling factor (fp32 scalar)
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # Program id with grouping along M for better L2 locality
    pid = tl.program_id(axis=0)

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Tile corner indices
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Pointers for tiles
    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)  # [BM, BK]
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)  # [BK, BN]

    # Accumulator in FP32
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Main K loop
    k_iter = tl.cdiv(K, BLOCK_SIZE_K)
    for ki in range(0, k_iter):
        k_base = ki * BLOCK_SIZE_K
        k_mask = (k_base + offs_k) < K

        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (k_mask[None, :]), other=0.0)
        b = tl.load(b_ptrs, mask=(k_mask[:, None]) & (offs_n[None, :] < N), other=0.0)

        acc = tl.dot(a, b, acc)

        # Advance pointers along K
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # Add bias
    bias = tl.load(bias_ptr + offs_n, mask=(offs_n < N), other=0.0).to(tl.float32)
    acc = acc + bias[None, :]

    # Save linear output for residual add
    lin = acc

    # Sigmoid in fp32: 1 / (1 + exp(-x))
    sig = 1.0 / (1.0 + tl.exp(-acc))

    # y = sigmoid(lin) * scale + lin
    y = sig * scale + lin

    # Compute output pointers and mask
    out_ptrs = out_ptr + (offs_m[:, None] * stride_om + offs_n[None, :] * stride_on)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    # Cast to output dtype before store
    # If out dtype is bf16 / fp16 / fp32 we handle appropriately
    out_dtype = out_ptr.dtype.element_ty
    y_cast = y.to(out_dtype)
    tl.store(out_ptrs, y_cast, mask=mask)


def kernel_function(
    x: torch.Tensor,
    weight: torch.Tensor = None,
    bias: torch.Tensor = None,
    scaling_factor: float = 1.0,
    out: torch.Tensor = None,
    weight_t: torch.Tensor = None,
    **kwargs,
):
    """
    Fused Gemm+Sigmoid+Scaling+ResidualAdd:
      y = sigmoid(x @ W^T + b) * scaling_factor + (x @ W^T + b)

    Fusion rationale (single-pass):
      - We compute the GEMM tile-by-tile, accumulate in fp32.
      - Epilogue fuses bias add, sigmoid activation, scaling, and residual add directly on the accumulator.
      - This eliminates intermediate global memory writes and additional kernel launches.

    Arguments:
      x         : [M, K] input activations (recommended torch.bfloat16 or torch.float16; fp32 also supported)
      weight    : [N, K] weight matrix (non-transposed). If provided, an internal transposed view is used.
      weight_t  : [K, N] transposed weight. If provided, takes precedence over 'weight'.
      bias      : [N] bias vector
      scaling_factor : float scalar multiplier applied to sigmoid output
      out       : optional preallocated output tensor [M, N]; if None a new tensor is allocated

    Constraints:
      - All compute is done inside the Triton kernel. The wrapper only validates, allocates, and launches.
      - No torch.nn/torch.nn.functional usage.

    Returns:
      out: [M, N] tensor on same device as x, with dtype matching x (or 'out' dtype if provided)
    """
    assert x is not None and (weight is not None or weight_t is not None) and bias is not None, \
        "x, weight/weight_t, and bias are required"

    # Basic shape checks
    assert x.ndim == 2, "x must be 2D [M, K]"
    M, K_x = x.shape

    # Resolve W^T view for kernel (expects [K, N])
    if weight_t is not None:
        wt = weight_t
        assert wt.ndim == 2, "weight_t must be 2D [K, N]"
        K_wt, N = wt.shape
        assert K_wt == K_x, f"K mismatch: x.shape[1]={K_x}, weight_t.shape[0]={K_wt}"
    else:
        assert weight is not None, "Either weight or weight_t must be provided"
        assert weight.ndim == 2, "weight must be 2D [N, K]"
        N, K_w = weight.shape
        assert K_w == K_x, f"K mismatch: x.shape[1]={K_x}, weight.shape[1]={K_w}"
        # Use a transposed VIEW; this is not compute and avoids extra device work
        wt = weight.t()

    # Bias check
    assert bias.ndim == 1 and bias.shape[0] == N, f"bias must be [N], got {bias.shape}"

    # Dtype/device checks
    assert x.is_cuda and wt.is_cuda and bias.is_cuda, "All tensors must be on CUDA device"
    assert x.device == wt.device == bias.device, "Device mismatch among inputs"

    # Output allocation
    if out is None:
        # Match dtype of x (test uses bfloat16)
        out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    else:
        assert out.shape == (M, N), f"out must be {(M, N)}, got {out.shape}"
        assert out.is_cuda and out.device == x.device, "out must be CUDA tensor on same device as x"

    # Strides for inputs/outputs (row-major by default)
    stride_am, stride_ak = x.stride(0), x.stride(1)                  # X [M, K]
    stride_bk, stride_bn = wt.stride(0), wt.stride(1)                # W^T [K, N]
    stride_om, stride_on = out.stride(0), out.stride(1)              # Y [M, N]

    # Grid setup: 1D launch using grouped tiling inside the kernel
    def grid(meta):
        return (triton.cdiv(M, meta["BLOCK_SIZE_M"]) * triton.cdiv(N, meta["BLOCK_SIZE_N"]),)

    # Launch kernel
    _fused_gemm_sigmoid_residual_kernel[grid](
        x, wt, bias, out,
        M, N, K_x,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_om, stride_on,
        float(scaling_factor),
    )

    return out