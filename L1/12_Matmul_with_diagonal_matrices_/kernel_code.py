import torch
import triton
import triton.language as tl


@triton.jit
def _scale_rows_kernel(
    a_ptr,            # *A: [N]
    b_ptr,            # *B: [N, M]
    c_ptr,            # *C: [N, M]
    N,                # rows
    M,                # cols
    stride_bm,        # B.stride(0)
    stride_bn,        # B.stride(1)
    stride_cm,        # C.stride(0)
    stride_cn,        # C.stride(1)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Tile kernel that computes C = diag(A) @ B == rowwise_scale(A, B).

    Each program instance handles a BLOCK_M x BLOCK_N tile of the output C.
    - Loads a BLOCK_M slice of A once.
    - Loads a BLOCK_M x BLOCK_N tile of B.
    - Multiplies row-wise and stores to C with proper masking.
    """
    pid_m = tl.program_id(axis=0)  # tile index along rows
    pid_n = tl.program_id(axis=1)  # tile index along cols

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]

    mask_m = offs_m < N
    mask_n = offs_n < M
    mask_2d = mask_m[:, None] & mask_n[None, :]

    # Load a BLOCK_M slice of A (vector), mask for bounds. Cast to fp32 for better precision.
    a_vals = tl.load(a_ptr + offs_m, mask=mask_m, other=0).to(tl.float32)  # [BLOCK_M]

    # Compute pointers into the B and C tiles
    b_ptrs = b_ptr + (offs_m[:, None] * stride_bm + offs_n[None, :] * stride_bn)
    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)

    # Load B tile (masked), cast to fp32 for compute
    b_tile = tl.load(b_ptrs, mask=mask_2d, other=0).to(tl.float32)  # [BLOCK_M, BLOCK_N]

    # Row-wise scaling: C[i, j] = A[i] * B[i, j]
    c_tile = a_vals[:, None] * b_tile

    # Cast back to output dtype
    out_dtype = c_ptr.dtype.element_ty
    c_tile = c_tile.to(out_dtype)

    # Store to C (masked)
    tl.store(c_ptrs, c_tile, mask=mask_2d)


def kernel_function(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Compute C = diag(A) @ B using a single Triton kernel.

    Fusion note:
    - We fuse the diagonal materialization and the matrix multiply into a single pass.
    - Instead of building diag(A) and calling a GEMM, we directly scale each row of B by A[i].
    - This eliminates the need to write/read an intermediate diagonal matrix and reduces memory traffic.

    Constraints/behavior:
    - A must be 1D of shape (N,)
    - B must be 2D of shape (N, M)
    - All computation happens inside the Triton kernel. The wrapper only validates inputs,
      allocates the output, and configures/launches the kernel.
    - Supports common floating dtypes (bfloat16, float16, float32). Computation is done in fp32
      for improved numerical accuracy and then cast back to B's dtype on store.

    Returns:
        A newly allocated tensor C on the same device/dtype as B with shape (N, M).
    """
    # Basic validations (wrapper allowed to do setup/validation only)
    if not (isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor)):
        raise TypeError("A and B must be torch.Tensors")

    if A.device.type != "cuda" or B.device.type != "cuda":
        raise ValueError("A and B must be CUDA tensors")

    if A.dim() != 1:
        raise ValueError(f"A must be 1D, got shape {tuple(A.shape)}")

    if B.dim() != 2:
        raise ValueError(f"B must be 2D, got shape {tuple(B.shape)}")

    N = A.shape[0]
    if B.shape[0] != N:
        raise ValueError(f"Shape mismatch: A.shape[0]={N} must equal B.shape[0]={B.shape[0]}")

    # Dtype checks: keep it simple and safe
    if A.dtype != B.dtype:
        raise ValueError(f"Dtype mismatch: A.dtype={A.dtype} must equal B.dtype={B.dtype}")

    if A.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise TypeError(f"Unsupported dtype: {A.dtype}. Supported: bfloat16, float16, float32")

    M = B.shape[1]

    # Allocate output
    C = torch.empty_like(B)

    # Launch configuration
    # Choose tile sizes that coalesce accesses along columns (contiguous dimension).
    # BLOCK_N is chosen larger to load/store along contiguous memory in each row.
    BLOCK_M = 128
    BLOCK_N = 256
    grid = (triton.cdiv(N, BLOCK_M), triton.cdiv(M, BLOCK_N))

    # Launch kernel
    _scale_rows_kernel[grid](
        A, B, C,
        N, M,
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=4,
    )

    return C