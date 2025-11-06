import torch
import triton
import triton.language as tl


# This kernel computes y = mish(mish(x @ W^T + b)) in a single pass:
# - Tiled matmul: [M, K] x [N, K]^T -> [M, N] with fp32 accumulation
# - Fused epilogue: add bias, apply Mish twice, then store
# Notes:
# - Mish(x) = x * tanh(softplus(x)), softplus(x) = log(1 + exp(-|x|)) + max(x, 0)
# - To avoid numerical issues and reliance on possibly unavailable tl.tanh/tl.log1p,
#   we implement tanh via sigmoid: tanh(z) = 2 * sigmoid(2z) - 1 with a stable sigmoid.
# - We mask loads/stores for generality, even though provided sizes are multiples of common BLOCK sizes.
#
# Memory layout assumptions:
# - x: [M, K], row-major
# - W: [N, K], row-major; we use it as W^T by reading [K, N] tiles via strides
# - b: [N], added per output column
# - y: [M, N], row-major
#
# Grid:
# - program_id(0) -> blocks along M dimension
# - program_id(1) -> blocks along N dimension

configs = [
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=4),
]


@triton.autotune(configs=configs, key=['M', 'N', 'K'])
@triton.jit
def _fused_linear_mish2_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Program ids for 2D grid
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # Offsets for rows/cols of the output tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    k_tiles = tl.cdiv(K, BLOCK_K)

    # Make offsets more compiler-friendly for coalescing
    offs_m = tl.max_contiguous(tl.multiple_of(offs_m, BLOCK_M), BLOCK_M)
    offs_n = tl.max_contiguous(tl.multiple_of(offs_n, BLOCK_N), BLOCK_N)

    # Pointers to x and W tiles (will be advanced along K)
    x_ptrs = x_ptr + (offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk)
    w_ptrs = w_ptr + (offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn)

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Masks for OOB accesses
    mask_m = offs_m < M
    mask_n = offs_n < N
    mask_k_base = offs_k  # will be compared to remaining K in-loop

    # Matmul loop over K
    for ki in range(0, k_tiles):
        k_remaining = K - ki * BLOCK_K
        mask_x = mask_m[:, None] & (mask_k_base[None, :] < k_remaining)
        mask_w = (mask_k_base[:, None] < k_remaining) & mask_n[None, :]

        a = tl.load(x_ptrs, mask=mask_x, other=0.0)
        b = tl.load(w_ptrs, mask=mask_w, other=0.0)
        acc = tl.dot(a, b, acc)

        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    # Fused epilogue: add bias, apply Mish twice
    # Load bias for this N tile
    bias = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + bias[None, :]

    # First Mish
    # softplus(z) = log(1 + exp(-|z|)) + max(z, 0)
    z = acc
    abs_z = tl.abs(z)
    sp = tl.log(1.0 + tl.exp(-abs_z)) + tl.maximum(z, 0.0)
    u = 2.0 * sp
    # stable sigmoid(u)
    abs_u = tl.abs(u)
    e = tl.exp(-abs_u)
    sig = tl.where(u >= 0, 1.0 / (1.0 + e), e / (1.0 + e))
    tanh_sp = 2.0 * sig - 1.0
    mish1 = z * tanh_sp

    # Second Mish
    z2 = mish1
    abs_z2 = tl.abs(z2)
    sp2 = tl.log(1.0 + tl.exp(-abs_z2)) + tl.maximum(z2, 0.0)
    u2 = 2.0 * sp2
    abs_u2 = tl.abs(u2)
    e2 = tl.exp(-abs_u2)
    sig2 = tl.where(u2 >= 0, 1.0 / (1.0 + e2), e2 / (1.0 + e2))
    tanh_sp2 = 2.0 * sig2 - 1.0
    mish2 = z2 * tanh_sp2

    # Store
    y_ptrs = y_ptr + (offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn)
    out_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(y_ptrs, mish2, mask=out_mask)


def kernel_function(x: torch.Tensor, W: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Fused Triton implementation of:
        y = mish(mish(x @ W^T + b))

    What is fused in this kernel:
    - Tiled matmul between x [M, K] and W^T [K, N] (by reading W as [N, K] with K-major access).
    - Bias addition: broadcast add b[n] to each output column n.
    - Two consecutive Mish activations in epilogue:
        Mish(x) = x * tanh(softplus(x)),
        softplus(x) = log(1 + exp(-|x|)) + max(x, 0)
      We implement tanh via a stable sigmoid: tanh(z) = 2 * sigmoid(2z) - 1.

    Wrapper responsibilities per guidelines:
    - Validate inputs (dtype/device/shape), allocate output, and launch Triton kernel.
    - No math is performed in Python; all compute happens inside the Triton kernel.

    Args:
        x: [M, K] float32 CUDA tensor
        W: [N, K] float32 CUDA tensor (row-major). We compute x @ W^T.
        b: [N] float32 CUDA tensor

    Returns:
        y: [M, N] float32 CUDA tensor on the same device as x
    """
    # Basic validations
    if not (x.is_cuda and W.is_cuda and b.is_cuda):
        raise ValueError("All inputs must be CUDA tensors")
    if x.dtype != torch.float32 or W.dtype != torch.float32 or b.dtype != torch.float32:
        raise TypeError("All inputs must be torch.float32")
    if x.ndim != 2 or W.ndim != 2 or b.ndim != 1:
        raise ValueError("Expected shapes: x [M, K], W [N, K], b [N]")
    M, Kx = x.shape
    Nw, Kw = W.shape
    if Kx != Kw:
        raise ValueError(f"Incompatible inner dims: x.shape[1]={Kx} vs W.shape[1]={Kw}")
    if b.shape[0] != Nw:
        raise ValueError(f"Bias shape mismatch: b.shape[0]={b.shape[0]} vs W.shape[0]={Nw}")
    if x.device != W.device or x.device != b.device:
        raise ValueError("All tensors must be on the same CUDA device")

    M, K, N = x.shape[0], x.shape[1], W.shape[0]

    # Allocate output
    y = torch.empty((M, N), device=x.device, dtype=torch.float32)

    # Define launch grid
    def grid(meta):
        return (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))

    # Launch kernel
    _fused_linear_mish2_kernel[grid](
        x, W, b, y,
        M, N, K,
        x.stride(0), x.stride(1),
        W.stride(0), W.stride(1),
        y.stride(0), y.stride(1),
    )
    return y