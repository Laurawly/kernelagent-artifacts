import torch
import triton
import triton.language as tl


@triton.jit
def _linear_bias_relu_kernel(
    x_ptr,          # (M, K) input
    w_ptr,          # (N, K) weight (out_features, in_features)
    b_ptr,          # (N,) bias
    out_ptr,        # (M, N) output
    M, N, K,        # sizes
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Compute: out = relu(x @ w.T + b)
    x: [M, K], w: [N, K], b: [N], out: [M, N]
    Accumulate in fp32 for numerical stability; store to out dtype.
    Fuses matmul + bias add + ReLU in a single pass.
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < M
    mask_n = offs_n < N

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    num_k_tiles = tl.cdiv(K, BLOCK_K)
    for ki in range(0, num_k_tiles):
        offs_k = ki * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        # A tile: [BLOCK_M, BLOCK_K] from x
        a_ptrs = x_ptr + (offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk)
        a = tl.load(a_ptrs, mask=(mask_m[:, None] & mask_k[None, :]), other=0.0)

        # B tile: [BLOCK_K, BLOCK_N] from w (note: w is [N, K])
        # We load w[k, n] by indexing w[n, k] using strides: offs_k along K and offs_n along N
        b_ptrs = w_ptr + (offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn)
        b = tl.load(b_ptrs, mask=(mask_k[:, None] & mask_n[None, :]), other=0.0)

        # Dot product accumulates into fp32
        acc = tl.dot(a, b, acc)

    # Bias add: load bias for the current N-block
    bias = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
    acc = acc + bias[None, :]

    # ReLU
    zero = tl.zeros((), dtype=tl.float32)
    acc = tl.maximum(acc, zero)

    # Store result, cast to output dtype (matches out_ptr dtype)
    out = acc.to(out_ptr.dtype.element_ty)

    out_ptrs = out_ptr + (offs_m[:, None] * stride_om + offs_n[None, :] * stride_on)
    tl.store(out_ptrs, out, mask=(mask_m[:, None] & mask_n[None, :]))


def kernel_function(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor):
    """
    Fused linear + ReLU on GPU using Triton.

    What is fused:
    - A single Triton kernel performs:
      1) Matrix multiplication: y = x @ w.T   where x is [M, K], w is [N, K]
      2) Bias addition: y += b
      3) ReLU activation: y = max(y, 0)
    This avoids intermediate global memory writes between stages.

    Args:
      x: [M, K] tensor (expected bf16 per test policy; other floating types supported)
      w: [N, K] tensor (out_features, in_features)
      b: [N] tensor

    Returns:
      out: [M, N] tensor with same dtype/device as x

    Notes:
    - The wrapper performs only validation, allocation, and kernel launch.
    - All math is executed inside the Triton kernel.
    """
    # Basic validation, per runtime constraints
    assert x.is_cuda and w.is_cuda and b.is_cuda, "All tensors must be on CUDA device."
    assert x.ndim == 2 and w.ndim == 2 and b.ndim == 1, "Shapes must be (M,K), (N,K), (N,)."
    M, Kx = x.shape
    N, Kw = w.shape
    assert Kx == Kw, f"Incompatible shapes: x.shape[1]={Kx} must equal w.shape[1]={Kw}."
    assert b.shape[0] == N, f"Bias shape {b.shape} incompatible with out_features N={N}."
    # Allocate output (same dtype/device as x; test compares in fp32)
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)

    # Extract strides (in elements, not bytes)
    stride_xm, stride_xk = x.stride()
    stride_wn, stride_wk = w.stride()
    stride_om, stride_on = out.stride()

    # Kernel launch configuration
    # For this problem (M=4096, K=120, N=84), these blocks work well.
    BLOCK_M = 128
    BLOCK_N = 64
    BLOCK_K = 32

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    # Launch kernel
    _linear_bias_relu_kernel[grid](
        x, w, b, out,
        M, N, Kx,
        stride_xm, stride_xk,
        stride_wn, stride_wk,
        stride_om, stride_on,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=4,     # reasonable default for these tile sizes
        num_stages=3,    # pipeline stages to help latency hiding
    )

    return out