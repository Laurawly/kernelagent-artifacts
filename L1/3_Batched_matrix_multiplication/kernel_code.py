import torch
import triton
import triton.language as tl


# -----------------------------
# Batched Matrix Multiplication (C = A @ B) implemented in Triton
# - A: [batch, M, K]
# - B: [batch, K, N]
# - C: [batch, M, N]
#
# Notes:
# - All computation is performed in the Triton kernel via tl.load/tl.store/tl.dot.
# - The Python wrapper (kernel_function) performs only validation, allocations, and launch config.
# - No PyTorch compute helpers (torch.nn, torch.nn.functional, torch.bmm, etc.) are used here.
# - We accumulate in fp32 for better numerical stability with large K (e.g., 1024).
# - Nothing to fuse beyond pure matmul for this task; we keep a clean single-pass matmul kernel.
# -----------------------------

# Some reasonable autotune configs; the autotuner will pick the best one for the given shapes.
_autotune_configs = [
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=4, num_warps=8),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=8),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=4, num_warps=8),
]


@triton.autotune(configs=_autotune_configs, key=['M', 'N', 'K'])
@triton.jit
def _bmm_kernel(
    a_ptr, b_ptr, c_ptr,
    BATCH, M, N, K,
    stride_aB, stride_aM, stride_aK,
    stride_bB, stride_bK, stride_bN,
    stride_cB, stride_cM, stride_cN,
    USE_BF16: tl.constexpr,  # True if output dtype is bfloat16, otherwise fp16
    BLOCK_M: tl.constexpr,   # tile size in M dimension
    BLOCK_N: tl.constexpr,   # tile size in N dimension
    BLOCK_K: tl.constexpr,   # tile size in K dimension
):
    """
    A Triton kernel that computes a single [BLOCK_M x BLOCK_N] tile of C for a given batch index.
    - Launch grid is 3D: (M tiles, N tiles, Batch)
    - Accumulates in fp32 for improved stability, casts to fp16/bf16 at store.
    """

    # Program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_b = tl.program_id(2)

    # Compute the tile's starting indices along M and N
    start_m = pid_m * BLOCK_M
    start_n = pid_n * BLOCK_N

    # Offsets within the tile
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_n = start_n + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Base pointers to the current batch
    a_batch_ptr = a_ptr + pid_b * stride_aB
    b_batch_ptr = b_ptr + pid_b * stride_bB
    c_batch_ptr = c_ptr + pid_b * stride_cB

    # Initialize accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k0 in range(0, K, BLOCK_K):
        k_idx_a = k0 + offs_k[None, :]    # [1, BLOCK_K] broadcasted over rows
        k_idx_b = k0 + offs_k[:, None]    # [BLOCK_K, 1] broadcasted over cols

        # Compute pointers for this K-slice
        a_ptrs = a_batch_ptr + (offs_m[:, None] * stride_aM) + (k_idx_a * stride_aK)
        b_ptrs = b_batch_ptr + (k_idx_b * stride_bK) + (offs_n[None, :] * stride_bN)

        # Masks for boundary conditions
        a_mask = (offs_m[:, None] < M) & (k_idx_a < K)
        b_mask = (k_idx_b < K) & (offs_n[None, :] < N)

        # Load tiles; out-of-bounds elements are set to 0
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Multiply-accumulate into fp32 accumulator
        acc = tl.dot(a, b, acc)

    # Write back result; cast to requested output dtype
    c_ptrs = c_batch_ptr + (offs_m[:, None] * stride_cM) + (offs_n[None, :] * stride_cN)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    # Cast accumulator to the appropriate low-precision output type
    c_val = acc.to(tl.bfloat16) if USE_BF16 else acc.to(tl.float16)
    tl.store(c_ptrs, c_val, mask=c_mask)


def kernel_function(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Batched matrix multiplication using Triton (C = A @ B).

    Args:
        A: Tensor of shape [batch, M, K], on CUDA, dtype float16 or bfloat16.
        B: Tensor of shape [batch, K, N], on CUDA, dtype must match A.

    Returns:
        C: Tensor of shape [batch, M, N], same dtype/device as A/B.

    Implementation details:
    - This wrapper performs only:
      - argument checks
      - output allocation
      - grid configuration and kernel launch
    - The entire computation (multiply-accumulate) is executed in the Triton kernel.
    - There is no opportunity for multi-op fusion beyond pure matmul in this task, so the kernel
      is a single-pass BMM without epilogue ops (e.g., bias or activation). If such ops existed,
      we would fuse them here to reduce memory traffic and launch overhead.
    """
    # Basic validations
    assert A.ndim == 3 and B.ndim == 3, "A and B must be 3D tensors: [batch, M, K] and [batch, K, N]"
    batch_a, M, K_a = A.shape
    batch_b, K_b, N = B.shape
    assert batch_a == batch_b, "Batch sizes of A and B must match"
    assert K_a == K_b, "Inner dimensions must match: A[..., K] and B[K, ...]"
    assert A.device.type == 'cuda' and B.device.type == 'cuda', "Tensors must be on CUDA device"
    assert A.dtype == B.dtype, "A and B must have the same dtype"
    assert A.dtype in (torch.float16, torch.bfloat16), "Supported dtypes: float16 and bfloat16"

    BATCH = batch_a
    K = K_a
    use_bf16 = (A.dtype == torch.bfloat16)

    # Allocate output on same device/dtype
    C = torch.empty((BATCH, M, N), device=A.device, dtype=A.dtype)

    # Compute launch grid: one program per output tile and per batch
    def grid(meta):
        return (
            triton.cdiv(M, meta['BLOCK_M']),
            triton.cdiv(N, meta['BLOCK_N']),
            BATCH,
        )

    # Launch Triton kernel
    _bmm_kernel[grid](
        A, B, C,
        BATCH, M, N, K,
        A.stride(0), A.stride(1), A.stride(2),
        B.stride(0), B.stride(1), B.stride(2),
        C.stride(0), C.stride(1), C.stride(2),
        USE_BF16=use_bf16,
    )

    return C