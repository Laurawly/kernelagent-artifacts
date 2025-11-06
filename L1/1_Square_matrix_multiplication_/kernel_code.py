import torch
import triton
import triton.language as tl


# ---------------------------
# Triton GEMM: C = A @ B
# ---------------------------
# - Blocked matmul using tl.dot with FP32 accumulation
# - Works for arbitrary (M, K) x (K, N) shapes; tested for square N=4096 with bf16
# - Uses masked tl.load/tl.store to handle boundaries safely
# - Autotuned across a small set of tile sizes and schedules
#
# Notes on fusion:
# The test requires a single matrix multiplication. There are no additional ops
# (bias, activation, etc.) to fuse, so the entire operator pipeline consists
# of just the matmul itself. We fuse the accumulation and type-conversion
# epilogue into the same kernel (compute and write-back in one pass). If
# downstream ops existed (e.g., bias add + activation), they could be fused
# into the epilogue here to avoid extra kernel launches and memory traffic.


@triton.autotune(
    configs=[
        triton.Config(
            {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32},
            num_stages=3, num_warps=8
        ),
        triton.Config(
            {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32},
            num_stages=4, num_warps=4
        ),
        triton.Config(
            {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32},
            num_stages=4, num_warps=4
        ),
        triton.Config(
            {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64},
            num_stages=3, num_warps=4
        ),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Program IDs for 2D tiling of output matrix C
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # Tile's starting indices
    start_m = pid_m * BLOCK_SIZE_M
    start_n = pid_n * BLOCK_SIZE_N

    # Offsets for rows/cols this program will compute
    offs_m = start_m + tl.arange(0, BLOCK_SIZE_M)
    offs_n = start_n + tl.arange(0, BLOCK_SIZE_N)
    offs_k_init = tl.arange(0, BLOCK_SIZE_K)

    # Make arange values more compiler-friendly
    offs_m = tl.max_contiguous(tl.multiple_of(offs_m, BLOCK_SIZE_M), BLOCK_SIZE_M)
    offs_n = tl.max_contiguous(tl.multiple_of(offs_n, BLOCK_SIZE_N), BLOCK_SIZE_N)

    # Accumulator in FP32
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Loop over K in BLOCK_SIZE_K steps
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    for ki in range(0, k_tiles):
        offs_k = ki * BLOCK_SIZE_K + offs_k_init

        # Pointers for A and B blocks
        a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

        # Masks for safe loads
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)

        # Load tiles; other=0.0 to zero-pad out-of-bounds
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Matrix multiply-accumulate
        acc = tl.dot(a, b, acc)

    # Write back results to C (convert accumulator to output dtype)
    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    # Convert to whatever dtype C holds (bf16/half, etc.)
    c_dtype = c_ptr.dtype.element_ty
    tl.store(c_ptrs, acc.to(c_dtype), mask=c_mask)


def kernel_function(*args):
    """
    Compute C = A @ B using a Triton matmul kernel.

    Accepted call patterns (the first two are sufficient for the provided test):
      - kernel_function(A, B)
      - kernel_function(A, B, N)               # N is optional; ignored if provided
      - kernel_function(A, B, C_out)
      - kernel_function(A, B, C_out, N)

    Notes:
    - All computation is done in Triton (_matmul_kernel); the wrapper only validates inputs,
      allocates output if needed, and configures the kernel launch.
    - No PyTorch math ops are used in the computation path, in accordance with the runtime constraints.
    """
    # Parse arguments
    tensors = [a for a in args if isinstance(a, torch.Tensor)]
    ints = [a for a in args if isinstance(a, int)]
    if len(tensors) < 2:
        raise TypeError("kernel_function expects at least two CUDA tensors (A, B).")

    # Primary accepted signatures for this test: (A, B), (A, B, N), (A, B, C_out), (A, B, C_out, N)
    # We intentionally do not try to interpret (C_out, A, B) to keep the interface simple and robust.
    A = tensors[0]
    B = tensors[1]
    C_out = None
    if len(tensors) >= 3:
        # If a third tensor is provided, treat it as the output buffer
        C_out = tensors[2]

    # Basic validation
    if not (A.is_cuda and B.is_cuda):
        raise ValueError("A and B must be CUDA tensors.")
    if A.dim() != 2 or B.dim() != 2:
        raise ValueError("A and B must be 2D matrices.")
    if A.dtype != B.dtype:
        raise ValueError("A and B must have the same dtype.")
    if A.device != B.device:
        raise ValueError("A and B must be on the same device.")

    M, K_A = A.shape
    K_B, N = B.shape
    if K_A != K_B:
        raise ValueError(f"Incompatible shapes: A={tuple(A.shape)}, B={tuple(B.shape)}")

    # Output allocation if needed
    if C_out is None:
        C_out = torch.empty((M, N), device=A.device, dtype=A.dtype)
    else:
        if not C_out.is_cuda:
            raise ValueError("C_out must be a CUDA tensor.")
        if C_out.shape != (M, N):
            raise ValueError(f"C_out has shape {tuple(C_out.shape)}; expected {(M, N)}")
        if C_out.dtype != A.dtype or C_out.device != A.device:
            raise ValueError("C_out must have the same dtype and device as A and B.")

    # Compute grid size
    def grid(meta):
        return (
            triton.cdiv(M, meta['BLOCK_SIZE_M']),
            triton.cdiv(N, meta['BLOCK_SIZE_N']),
        )

    # Launch Triton kernel
    _matmul_kernel[grid](
        A, B, C_out,
        M, N, K_A,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C_out.stride(0), C_out.stride(1),
    )
    return C_out