import torch
import triton
import triton.language as tl


@triton.jit
def _linear_bias_fwd(
    x_ptr,        # [M, K] input
    w_ptr,        # [N, K] weights (note: row-major as stored; we use w.T in math)
    b_ptr,        # [N] bias
    y_ptr,        # [M, N] output
    M, K, N,      # dimensions
    stride_xm, stride_xk,   # x strides
    stride_wn, stride_wk,   # w strides (N, K)
    stride_ym, stride_yn,   # y strides
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Fused Linear Forward: y = x @ w.T + b
    - x: [M, K]
    - w: [N, K] (we use w.T in the matmul)
    - b: [N]
    - y: [M, N]
    Accumulates in float32, stores to y dtype.
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Accumulator in fp32 for numerical stability
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    k_tiles = tl.cdiv(K, BLOCK_K)
    for kt in range(k_tiles):
        k_start = kt * BLOCK_K
        k_idx = k_start + offs_k

        # Pointers for X tile: [BLOCK_M, BLOCK_K]
        x_ptrs = x_ptr + (offs_m[:, None] * stride_xm + k_idx[None, :] * stride_xk)
        # Pointers for W tile: want [BLOCK_K, BLOCK_N] = w.T tile, but w stored [N, K]
        # Index as w[k, n] => w_ptr + k * stride_wk + n * stride_wn
        w_ptrs = w_ptr + (k_idx[:, None] * stride_wk + offs_n[None, :] * stride_wn)

        # Masks
        x_mask = (offs_m[:, None] < M) & (k_idx[None, :] < K)
        w_mask = (k_idx[:, None] < K) & (offs_n[None, :] < N)

        # Load tiles
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)
        w_t = tl.load(w_ptrs, mask=w_mask, other=0.0)  # shape [BLOCK_K, BLOCK_N]

        # Accumulate
        acc = tl.dot(x, w_t, acc)

    # Fuse bias addition in epilogue
    b = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc += b[None, :].to(tl.float32)

    # Store result to output with proper masking
    y_ptrs = y_ptr + (offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn)
    y_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    # Cast accumulator to destination dtype
    out_dtype = y_ptr.dtype.element_ty
    tl.store(y_ptrs, acc.to(out_dtype), mask=y_mask)


def kernel_function(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Fused linear forward in a single Triton kernel: y = x @ w.T + b

    What is fused:
    - Matrix multiplication (x @ w.T) and bias addition are fused into one kernel.
      This minimizes global memory traffic and avoids an extra kernel launch.

    Constraints and behavior:
    - Shapes must be: x: [M, K], w: [N, K], b: [N], y: [M, N]
    - Dtypes supported: bfloat16, float16, float32 (accumulation is always float32)
    - Device: CUDA only
    - Wrapper only validates inputs, allocates output, computes grid, and launches the Triton kernel.
      All math executes inside the Triton kernel per runtime constraints.

    Parameters:
        x: [M, K] input tensor (e.g., [4096, 84]), dtype typically bfloat16 per test
        w: [N, K] weight tensor (e.g., [20, 84])
        b: [N]    bias tensor (e.g., [20])

    Returns:
        y: [M, N] output tensor on the same device as x, with dtype = x.dtype
    """
    # Basic validation
    assert x.is_cuda and w.is_cuda and b.is_cuda, "All tensors must be CUDA tensors."
    assert x.ndim == 2 and w.ndim == 2 and b.ndim == 1, "Invalid tensor ranks."
    M, Kx = x.shape
    Nw, Kw = w.shape
    assert Kx == Kw, f"Incompatible inner dimensions: x.K={Kx}, w.K={Kw}"
    assert b.shape[0] == Nw, f"Bias shape mismatch: b.N={b.shape[0]}, w.N={Nw}"
    assert x.device == w.device == b.device, "All tensors must be on the same device."
    # Output dtype: follow input dtype (test uses bfloat16)
    out_dtype = x.dtype
    # Allocate output
    y = torch.empty((M, Nw), device=x.device, dtype=out_dtype)

    # Extract strides in elements (Triton expects element-based strides)
    stride_xm, stride_xk = x.stride()
    stride_wn, stride_wk = w.stride()
    stride_ym, stride_yn = y.stride()

    # Kernel launch configuration
    BLOCK_M = 128
    BLOCK_N = 32
    BLOCK_K = 32

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(Nw, BLOCK_N))

    # Launch kernel
    _linear_bias_fwd[grid](
        x, w, b, y,
        M, Kx, Nw,
        stride_xm, stride_xk,
        stride_wn, stride_wk,
        stride_ym, stride_yn,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=4,
        num_stages=3,
    )

    return y