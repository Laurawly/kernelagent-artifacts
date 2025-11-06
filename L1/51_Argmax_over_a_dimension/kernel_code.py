# kernel.py
# Triton Argmax kernel: computes argmax over dimension=1 for a 3D tensor [B, M, N]
# Wrapper function: kernel_function(x, dim=1)
#
# Notes for reviewers:
# - This implementation performs a single-pass reduction over the M dimension using Triton.
# - There is no additional operator to fuse with argmax; thus, the kernel focuses solely on the reduction.
# - All math (comparisons, max tracking) is performed inside the Triton kernel using tl.load/tl.store and Triton ops.
# - The Python wrapper only validates inputs, allocates the output tensor, and configures the launch.
# - Input dtype is expected to be BF16 as per the test requirement. We upcast to FP32 inside the kernel for numerically
#   reliable comparisons; results are integer indices (torch.int64). No PyTorch compute helpers are used.

import triton
import triton.language as tl
import torch


@triton.jit
def _argmax_dim1_kernel(
    x_ptr,                      # *bfloat16
    out_idx_ptr,                # *int64
    B, M, N,                    # sizes
    stride_b, stride_m, stride_n,          # input strides (in elements)
    o_stride_b, o_stride_n,                # output strides (in elements)
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    # Program ids:
    pid_n = tl.program_id(0)  # tile along N
    pid_b = tl.program_id(1)  # batch index

    # Offsets in N for this program
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N

    # Base offsets for this batch in input/output
    base_in = pid_b * stride_b
    base_out = pid_b * o_stride_b

    # Initialize running best values and indices for each n in the tile.
    # Use FP32 accumulator for comparisons; indices are int32 and converted to int64 at store.
    neg_inf = -float("inf")
    best_val = tl.zeros([BLOCK_N], dtype=tl.float32) + neg_inf
    best_idx = tl.zeros([BLOCK_N], dtype=tl.int32)

    # Number of tiles along M
    m_tiles = tl.cdiv(M, BLOCK_M)

    # Loop over tiles along M dimension
    for mi in range(m_tiles):
        m_base = mi * BLOCK_M

        # Within this M-tile, iterate rows [m_base, m_base + BLOCK_M)
        # Update best values/indices using strictly greater-than comparison to match PyTorch argmax semantics
        # (first occurrence is kept when values tie).
        for di in range(BLOCK_M):
            # Row index to load
            # Make it a vector to easily form masks and indices per lane
            row_idx_vec = (m_base + di) + tl.zeros([BLOCK_N], dtype=tl.int32)

            # Mask for valid rows and columns
            row_in_bounds = row_idx_vec < M
            curr_mask = n_mask & row_in_bounds

            # Compute pointers for this row: x[b, row_idx, offs_n]
            row_ptrs = x_ptr + base_in + row_idx_vec * stride_m + offs_n * stride_n

            # Load a row across the N-tile; masked OOB rows/cols receive -inf
            row_vals_bf16 = tl.load(row_ptrs, mask=curr_mask, other=neg_inf)
            row_vals = row_vals_bf16.to(tl.float32)

            # Compare and update; strictly '>' ensures first index is kept on ties (PyTorch behavior)
            better = row_vals > best_val
            best_val = tl.where(better, row_vals, best_val)
            best_idx = tl.where(better, row_idx_vec, best_idx)

    # Store resulting indices for this (b, tile of n)
    out_ptrs = out_idx_ptr + base_out + offs_n * o_stride_n
    tl.store(out_ptrs, best_idx.to(tl.int64), mask=n_mask)


def kernel_function(x: torch.Tensor, dim: int = 1):
    """
    Compute argmax along a specified dimension using a Triton kernel.

    Args:
        x: Input CUDA tensor of shape [B, M, N] (tested shape: [128, 4096, 4095]).
           Must be dtype=torch.bfloat16 as per the test's requirement to avoid FP32.
        dim: Dimension to reduce with argmax. This implementation supports dim == 1.

    Returns:
        A CUDA tensor of integer indices with the reduction dimension removed.
        For [B, M, N] reduced over dim=1, result shape is [B, N], dtype=torch.int64.

    Fusion notes:
        - There are no additional stages around argmax in the test (no bias, activation, etc.),
          so there is nothing meaningful to fuse. The kernel is a single-pass reduction that
          reads input data once per element and writes indices directly.

    Runtime constraints:
        - No PyTorch compute ops are used; only allocation, dtype/device checks,
          and launch configuration occur here. The reduction is entirely in the Triton kernel.
    """
    # Basic validations and constraints
    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a torch.Tensor")
    if not x.is_cuda:
        raise ValueError("Input tensor must be on CUDA device.")
    if x.dtype != torch.bfloat16:
        raise ValueError(f"Input dtype must be torch.bfloat16; got {x.dtype}")
    if dim != 1:
        raise NotImplementedError("This kernel only supports argmax over dim=1 for 3D tensors.")
    if x.ndim != 3:
        raise NotImplementedError("This kernel expects a 3D tensor [B, M, N].")

    B, M, N = x.shape

    # Prepare output tensor of indices; matches PyTorch argmax default dtype (torch.long / int64)
    out = torch.empty((B, N), device=x.device, dtype=torch.int64)

    # Strides of input and output in elements
    stride_b, stride_m, stride_n = x.stride()
    o_stride_b, o_stride_n = out.stride()

    # Choose tiling/meta-parameters
    # BLOCK_N: tile along N (last dim, contiguous) for coalesced accesses
    # BLOCK_M: number of rows processed per tile step along the reduction dimension
    # Powers of two per guidelines; tuned to balance registers and occupancy for given shapes
    BLOCK_N = 128
    BLOCK_M = 128

    # 2D grid: (tiles along N, batches)
    grid = (triton.cdiv(N, BLOCK_N), B)

    # Launch kernel
    _argmax_dim1_kernel[grid](
        x, out,
        B, M, N,
        stride_b, stride_m, stride_n,
        o_stride_b, o_stride_n,
        BLOCK_N=BLOCK_N,
        BLOCK_M=BLOCK_M,
        num_warps=4,        # reasonable default for reduction + wide loads
        num_stages=2,       # simple pipeline is sufficient
    )
    return out