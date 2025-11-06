import torch
import triton
import triton.language as tl


# -----------------------------------------------------------------------------
# Fused Linear: y = x @ weight.T + bias
# - Computes the matrix multiplication and adds bias in a single Triton kernel.
# - Accumulates in float32 for numerical stability (bf16/fp16 inputs supported).
# - Wrapper performs only validation, allocation, and launch; all math is in the kernel.
# -----------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=4),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _linear_fused_kernel(
    x_ptr,          # *[M, K]
    w_ptr,          # *[N, K]   (weight, NOT transposed; kernel uses it as KxN)
    b_ptr,          # *[N]
    y_ptr,          # *[M, N]
    M, N, K,        # sizes
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Program IDs for tile indices along M and N
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # Offsets for the M and N dimensions for this tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Create accumulator in fp32 for numerical stability
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension tiles
    k_tiles = tl.cdiv(K, BLOCK_K)
    for kt in range(0, k_tiles):
        k_start = kt * BLOCK_K
        k_idx = k_start + offs_k

        # Pointers for a tile of X: shape [BLOCK_M, BLOCK_K]
        x_ptrs = x_ptr + (offs_m[:, None] * stride_xm + k_idx[None, :] * stride_xk)
        x_mask = (offs_m[:, None] < M) & (k_idx[None, :] < K)
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)

        # Pointers for a tile of W (weights): we need (BLOCK_K, BLOCK_N) = weight[k, n]
        # Weight tensor is [N, K] contiguous in row-major:
        # index weight[n, k] => base + n*stride_wn + k*stride_wk
        w_ptrs = w_ptr + (offs_n[None, :] * stride_wn + k_idx[:, None] * stride_wk)
        w_mask = (offs_n[None, :] < N) & (k_idx[:, None] < K)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)

        # Matrix multiply accumulate
        acc = tl.dot(x, w, acc)

    # Add bias (broadcast across rows)
    b = tl.load(b_ptr + offs_n, mask=(offs_n < N), other=0.0).to(tl.float32)
    acc = acc + b[None, :]

    # Store result to output in the output dtype
    y_ptrs = y_ptr + (offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn)
    y_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(y_ptrs, acc.to(y_ptr.dtype.element_ty), mask=y_mask)


def kernel_function(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Fused linear operator implemented with Triton:
      y = x @ weight.T + bias

    Fusion details:
    - The matrix multiplication (x @ weight.T) and the bias addition are fused in a single kernel.
    - Accumulation is performed in float32 to match reference float32 semantics even when inputs are bf16/fp16.
    - The wrapper only validates inputs, allocates output, and launches the Triton kernel; all math is in Triton.

    Args:
      x:      [M, K] input tensor (bf16/fp16/fp32), CUDA
      weight: [N, K] weight tensor (same dtype as x), CUDA
      bias:   [N]    bias tensor (same dtype as x), CUDA

    Returns:
      y: [M, N] tensor with same dtype/device as x
    """
    # Basic validation (wrapper does not perform math per runtime constraints)
    assert x.is_cuda and weight.is_cuda and bias.is_cuda, "All tensors must be on CUDA."
    assert x.ndim == 2 and weight.ndim == 2 and bias.ndim == 1, "Invalid input ranks."
    M, Kx = x.shape
    Nw, Kw = weight.shape
    assert Kx == Kw, f"Incompatible shapes: x has K={Kx}, weight has K={Kw}"
    assert bias.shape[0] == Nw, f"Bias shape mismatch: expected {Nw}, got {bias.shape[0]}"
    assert x.dtype == weight.dtype == bias.dtype, "x, weight, and bias must have the same dtype."
    assert x.dtype in (torch.bfloat16, torch.float16, torch.float32), "Supported dtypes: bf16, fp16, fp32."

    # Allocate output tensor (same dtype/device as inputs)
    y = torch.empty((M, Nw), device=x.device, dtype=x.dtype)

    # Define launch grid: 2D tiling across M and N
    def grid(meta):
        return (
            triton.cdiv(M, meta["BLOCK_M"]),
            triton.cdiv(Nw, meta["BLOCK_N"]),
        )

    # Launch Triton kernel
    _linear_fused_kernel[grid](
        x, weight, bias, y,
        M, Nw, Kx,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
        y.stride(0), y.stride(1),
    )

    return y