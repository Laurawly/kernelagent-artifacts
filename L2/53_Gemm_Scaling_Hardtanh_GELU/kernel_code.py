import torch
import triton
import triton.language as tl

# -----------------------------------------------------------------------------
# Fused Linear + Bias + Scaling + Hardtanh + GELU in a single Triton kernel.
#
# This kernel computes:
#   y = GELU(Hardtanh((x @ W^T + b) * scaling_factor))
#
# Fusion details (single pass):
# - Tiled matmul: C = X @ W^T
# - Fused epilogue:
#     1) Add bias
#     2) Multiply by scaling_factor
#     3) Hardtanh clamp to [hardtanh_min, hardtanh_max]
#     4) GELU (exact, erf-based) in float32
# - Store as output dtype (bf16)
#
# Runtime wrapper only validates shapes/dtypes, allocates output, and launches the kernel.
# No torch.nn / torch.nn.functional or other PyTorch compute ops are used in the math path.
# -----------------------------------------------------------------------------


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64,  'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64,  'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 64,  'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=8),
    ],
    key=['M', 'N', 'K']
)
@triton.jit
def _fused_linear_activations_kernel(
    a_ptr,        # X: [M, K], row-major
    b_ptr,        # W: [N, K] as stored; we access as B = W^T: [K, N] via strides
    bias_ptr,     # bias: [N]
    c_ptr,        # Out: [M, N]
    M, N, K,
    stride_am, stride_ak,   # strides for A (M, K)
    stride_bk, stride_bn,   # strides for B viewed as (K, N) via W's strides: bk = W.stride(1), bn = W.stride(0)
    stride_cm, stride_cn,   # strides for C (M, N)
    scaling, hard_min, hard_max,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # 2D tile (M x N), grouped ordering for better L2 caching
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

    # Make indices more compiler-friendly for coalescing
    offs_m = tl.max_contiguous(tl.multiple_of(offs_m, BLOCK_M), BLOCK_M)
    offs_n = tl.max_contiguous(tl.multiple_of(offs_n, BLOCK_N), BLOCK_N)

    # Accumulator in fp32 for numerical stability
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # K loop
    for k0 in range(0, K, BLOCK_K):
        k = k0 + offs_k

        a_ptrs = a_ptr + (offs_m[:, None] * stride_am + k[None, :] * stride_ak)
        b_ptrs = b_ptr + (k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

        a_mask = (offs_m[:, None] < M) & (k[None, :] < K)
        b_mask = (k[:, None] < K) & (offs_n[None, :] < N)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Dot product; A and B expected in bf16/fp16; accumulate in fp32
        acc = tl.dot(a, b, acc)

    # Fused epilogue: bias add, scaling, hardtanh, gelu
    # Load bias for the N tile; broadcast along M dimension
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
    acc = acc + bias[None, :]

    # scaling
    acc = acc * scaling

    # hardtanh clamp
    acc = tl.minimum(tl.maximum(acc, hard_min), hard_max)

    # GELU exact: 0.5 * x * (1 + erf(x / sqrt(2)))
    inv_sqrt2 = 0.7071067811865476  # 1 / sqrt(2)
    t = acc * inv_sqrt2
    gelu = 0.5 * acc * (1.0 + tl.math.erf(t))

    # Store result in the output dtype
    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    out = gelu.to(c_ptr.dtype.element_ty)
    tl.store(c_ptrs, out, mask=c_mask)


def kernel_function(x, weight, bias, scaling_factor, hardtanh_min, hardtanh_max):
    """
    Launch wrapper for the fused Triton kernel.

    Computes:
      y = GELU(Hardtanh((x @ W^T + b) * scaling_factor))

    Fusion stages inside the Triton kernel (single pass over output tiles):
      - Tiled MatMul: X @ W^T (accumulation in float32)
      - Bias addition (broadcast over rows)
      - Scaling by 'scaling_factor'
      - Hardtanh clamp to [hardtanh_min, hardtanh_max]
      - GELU (exact, erf-based) in float32
      - Cast to output dtype (bf16) and store

    Args:
      x:       [batch_size, in_features], bf16/half tensor on CUDA
      weight:  [out_features, in_features], bf16/half tensor on CUDA
      bias:    [out_features], bf16/half tensor on CUDA
      scaling_factor: float
      hardtanh_min: float
      hardtanh_max: float

    Returns:
      y: [batch_size, out_features], bf16 (same dtype as x)
    """
    if not (isinstance(x, torch.Tensor) and isinstance(weight, torch.Tensor) and isinstance(bias, torch.Tensor)):
        raise TypeError("x, weight, bias must be torch.Tensors")

    if not (x.is_cuda and weight.is_cuda and bias.is_cuda):
        raise RuntimeError("All inputs must be CUDA tensors")

    if x.dim() != 2 or weight.dim() != 2:
        raise ValueError("x and weight must be 2D tensors")
    if bias.dim() != 1:
        raise ValueError("bias must be 1D tensor")

    M, Kx = x.shape
    N, Kw = weight.shape
    if Kx != Kw:
        raise ValueError(f"Incompatible shapes: x is (*, {Kx}), weight is ({N}, {Kw})")

    if bias.shape[0] != N:
        raise ValueError(f"bias shape mismatch: expected ({N},), got {bias.shape}")

    # Restrict dtypes to half/bfloat16 to ensure tl.dot fast path
    if x.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError("x must be torch.bfloat16 or torch.float16")
    if weight.dtype != x.dtype or bias.dtype != x.dtype:
        raise TypeError("weight and bias must have the same dtype as x")

    # Allocate output
    y = torch.empty((M, N), device=x.device, dtype=x.dtype)

    # Strides
    stride_am = x.stride(0)
    stride_ak = x.stride(1)
    # Treat B as W^T (K, N). For W [N, K], we set:
    stride_bk = weight.stride(1)  # along K in W (i.e., 1 for contiguous)
    stride_bn = weight.stride(0)  # along N in W (i.e., K for contiguous)
    stride_cm = y.stride(0)
    stride_cn = y.stride(1)

    # Grid: 1D over tiles with grouping inside kernel
    def grid(meta):
        return (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)

    _fused_linear_activations_kernel[grid](
        x, weight, bias, y,
        M, N, Kx,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        float(scaling_factor), float(hardtanh_min), float(hardtanh_max),
    )
    return y