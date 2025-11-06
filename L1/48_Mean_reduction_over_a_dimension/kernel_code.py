# kernel.py
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # A single, sensible configuration to avoid heavy autotune time on very large tensors.
        triton.Config({"BLOCK_K": 256, "BLOCK_C": 128}, num_warps=8, num_stages=2),
    ],
    key=["K", "C"],
)
@triton.jit
def _mean_reduce_dim1_kernel(
    x_ptr,                   # *const T
    out_ptr,                 # *mut T
    B, C, K,                 # sizes: x shape is [B, C, K]; reduce over C (dim=1)
    stride_xb, stride_xc, stride_xk,  # x strides
    stride_ob, stride_ok,            # out strides, out shape [B, K]
    BLOCK_K: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """
    Compute mean over dimension 1 (C) for a 3D tensor X[B, C, K].

    Each program instance computes one [BLOCK_K] slice of the output for a fixed batch index b
    and a contiguous tile of the last dimension (K). It iterates over the reduction dimension (C)
    in chunks of BLOCK_C, loading tiles [BLOCK_C, BLOCK_K] with proper masking, accumulating in fp32,
    and finally writing the mean value back to out[b, k_tile].
    """
    pid_k = tl.program_id(0)  # tile along K
    pid_b = tl.program_id(1)  # batch index

    b = pid_b
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    k_mask = offs_k < K

    # Accumulator in fp32 for numerical stability
    acc = tl.zeros([BLOCK_K], dtype=tl.float32)

    # Loop over C in BLOCK_C chunks
    num_c_tiles = tl.cdiv(C, BLOCK_C)
    offs_c_base = tl.arange(0, BLOCK_C)

    for tc in range(0, num_c_tiles):
        c_start = tc * BLOCK_C
        offs_c = c_start + offs_c_base
        c_mask = offs_c < C

        # Compute pointers for a [BLOCK_C, BLOCK_K] tile: x[b, offs_c, offs_k]
        x_ptrs = x_ptr + b * stride_xb + offs_c[:, None] * stride_xc + offs_k[None, :] * stride_xk
        mask = c_mask[:, None] & k_mask[None, :]

        # Load, cast to fp32, sum along C-chunk axis (axis=0) and accumulate
        x_tile = tl.load(x_ptrs, mask=mask, other=0)
        x_tile_f32 = x_tile.to(tl.float32)
        acc += tl.sum(x_tile_f32, axis=0)

    # Compute mean by dividing by C
    invC = 1.0 / C
    mean_vals = acc * invC

    # Cast to output dtype if needed
    # We only need float16/bfloat16/float32 for this problem, but we handle the common types.
    if out_ptr.dtype.element_ty == tl.float16:
        mean_vals = mean_vals.to(tl.float16)
    elif out_ptr.dtype.element_ty == tl.bfloat16:
        mean_vals = mean_vals.to(tl.bfloat16)
    elif out_ptr.dtype.element_ty == tl.float32:
        pass  # already fp32
    else:
        # Fallback: try converting to the pointer element type if it's a float-like type
        # (kept for forward-compatibility; in practice this kernel targets bf16/fp16/fp32)
        mean_vals = mean_vals.to(out_ptr.dtype.element_ty)

    # Store results
    out_ptrs = out_ptr + b * stride_ob + offs_k * stride_ok
    tl.store(out_ptrs, mean_vals, mask=k_mask)


def kernel_function(x: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """
    Compute torch.mean(x, dim=dim) using a Triton kernel (reduction over a single dimension).
    This implementation specifically targets dim=1 for a 3D tensor of shape [B, C, K].

    Fusion and rationale:
    - The operation is a single mean-reduction over one dimension. There are no additional ops
      (e.g., bias, activation) to fuse logically with it. Therefore, the "fused kernel" is exactly
      the reduction itself: Load -> Reduce (sum) -> Normalize by C -> Store, in one pass.

    Runtime behavior:
    - The wrapper validates inputs, allocates the output tensor, and configures the launch.
      All math is performed inside the Triton kernel (_mean_reduce_dim1_kernel).
    - No torch.nn or torch.nn.functional calls are used.
    - No PyTorch math operations (sum/mean/etc.) are used for the computation.

    Constraints:
    - x must be a CUDA tensor.
    - dim must be 1 (i.e., reduce over the middle dimension) for a 3D input.

    Args:
        x: Input tensor, expected shape [B, C, K], dtype bf16/fp16/fp32, on CUDA device.
        dim: Reduction dimension (must be 1 for this kernel).

    Returns:
        A tensor of shape [B, K] with the mean over the C dimension, dtype same as x.
    """
    # Argument validation (wrapper is kept lightweight per guidelines)
    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a torch.Tensor")
    if not x.is_cuda:
        raise ValueError("x must be on CUDA device")
    if x.ndim != 3:
        raise ValueError(f"Expected x.ndim == 3, got {x.ndim}")
    # Normalize dim (allow -2 which equals 1 for a 3D tensor)
    if dim < 0:
        dim = x.ndim + dim
    if dim != 1:
        raise ValueError(f"This kernel implementation only supports reduction over dim=1. Got dim={dim}.")

    B, C, K = x.shape
    # Allocate output tensor: shape [B, K]
    out = torch.empty((B, K), device=x.device, dtype=x.dtype)

    # Extract strides in elements
    stride_xb = x.stride(0)
    stride_xc = x.stride(1)
    stride_xk = x.stride(2)

    stride_ob = out.stride(0)
    stride_ok = out.stride(1)

    # Configure launch grid: 2D
    def grid(meta):
        return (triton.cdiv(K, meta["BLOCK_K"]), B)

    # Launch kernel (no PyTorch compute; all math inside Triton)
    _mean_reduce_dim1_kernel[grid](
        x, out,
        B, C, K,
        stride_xb, stride_xc, stride_xk,
        stride_ob, stride_ok,
    )
    return out