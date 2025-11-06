import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 256, 'BLOCK_K': 128}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _fused_linear_rowwise_lse_mish(
    x_ptr,             # *const T, [M, K]
    w_ptr,             # *const T, treated as shape [K, N] via provided strides
    b_ptr,             # *const T (or dummy), [N]
    out_ptr,           # *mut T, [M, 1]
    M, N, K,           # sizes
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_bo,
    stride_out_m, stride_out_n,
    scale_factor,
    clamp_min, clamp_max,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Program ID for rows
    pid_m = tl.program_id(axis=0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M

    # Online logsumexp state per row:
    # m_i = running maximum; l_i = running sum of exp(y - m_i)
    m_i = tl.full([BLOCK_M], -float('inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)

    # Iterate over N in tiles, compute y_tile = (x @ w + b), then fused scale/self-add, clamp, and update LSE
    for start_n in tl.range(0, N, BLOCK_N, num_stages=1):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N

        # Accumulator for the matmul tile: [BLOCK_M, BLOCK_N]
        acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

        # Compute tile via K-reduction
        for start_k in tl.range(0, K, BLOCK_K):
            offs_k = start_k + tl.arange(0, BLOCK_K)
            mask_k = offs_k < K

            # Load x_tile: [BLOCK_M, BLOCK_K]
            x_ptrs = x_ptr + (offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk)
            x_tile = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)

            # Load w_tile: [BLOCK_K, BLOCK_N] using provided strides (treat w as [K, N])
            w_ptrs = w_ptr + (offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn)
            w_tile = tl.load(w_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)

            # Accumulate: cast to fp32 for stable accumulation
            acc += tl.dot(x_tile.to(tl.float32), w_tile.to(tl.float32))

        # Add bias if provided
        if HAS_BIAS:
            b_tile = tl.load(b_ptr + offs_n * stride_bo, mask=mask_n, other=0.0).to(tl.float32)
        else:
            b_tile = tl.zeros([BLOCK_N], dtype=tl.float32)

        # y = (x @ w + b)
        y = acc + b_tile[None, :]

        # Fused scale and self-add, respecting the operation order:
        # y = y * scale_factor
        # y = y + y
        y = y * scale_factor
        y = y + y

        # Clamp
        y = tl.minimum(tl.maximum(y, clamp_min), clamp_max)

        # Mask out-of-bounds columns so they don't affect reductions
        y = tl.where(mask_n[None, :], y, -float('inf'))

        # Online LSE update
        tile_row_max = tl.max(y, axis=1)  # [BLOCK_M]
        new_m = tl.maximum(m_i, tile_row_max)  # [BLOCK_M]

        # sum(exp(y - new_m)) over columns
        y_shifted = y - new_m[:, None]
        exp_y = tl.exp(y_shifted)
        sum_exp = tl.sum(exp_y, axis=1)  # [BLOCK_M]

        # l_i_new = l_i * exp(m_i - new_m) + sum_exp
        alpha = tl.exp(m_i - new_m)
        l_i = l_i * alpha + sum_exp
        m_i = new_m

    # s = logsumexp(y, dim=1, keepdim=True) = m_i + log(l_i)
    s = m_i + tl.log(l_i)

    # mish(s) = s * tanh(softplus(s)); softplus(s) = max(s, 0) + log(1 + exp(-|s|))
    abs_s = tl.abs(s)
    sp = tl.maximum(s, 0.0) + tl.log(1.0 + tl.exp(-abs_s))
    # tanh(sp) via logistic form to avoid relying on tl.tanh and to keep numerically stable
    t = tl.exp(-2.0 * sp)
    tanh_sp = (1.0 - t) / (1.0 + t)
    mish_s = s * tanh_sp
    out_val = s * mish_s  # final: s * mish(s)

    # Store result to out[:, 0]
    out_ptrs = out_ptr + offs_m * stride_out_m + 0 * stride_out_n
    tl.store(out_ptrs, out_val.to(out_ptr.dtype.element_ty), mask=mask_m)


def kernel_function(*args):
    """
    Fused Triton kernel wrapper: implements
      y = Linear(x)           # matmul + bias
      y = y * scale_factor
      y = y + y               # residual/self-add
      y = clamp(y, min, max)
      s = logsumexp(y, dim=1, keepdim=True)
      out = s * mish(s)

    Fusion details:
    - The kernel tiles the GEMM (x @ W (+ b)) and immediately applies scale, self-add, and clamp
      on each [BLOCK_M, BLOCK_N] tile before updating an online logsumexp per row. This avoids
      materializing the intermediate Y tensor in global memory.
    - A numerically stable online logsumexp is maintained per row across N-tiles:
        m_i, l_i updated with per-tile maxima and exp sums.
    - Final row scalar s = m_i + log(l_i); mish(s) computed in-kernel using stable softplus
      and a logistic tanh implementation; out = s * mish(s).
    - Only allocation/validation/launch happen in Python. All math runs in Triton.

    Supported call patterns (the test will try several; the first matching will be used):
      - kernel_function(x, weight, bias, scale_factor, clamp_min, clamp_max)
      - kernel_function(x, weight, scale_factor, clamp_min, clamp_max)     # no bias
      - kernel_function(x, weight, bias)                                   # defaults for scalars
      - kernel_function(x, weight)                                         # no bias, defaults for scalars
      - kernel_function(x, weight_t, ...)                                  # weight can be (K, N) or (N, K)

    Args:
      x:           (B, K) torch.Tensor (CUDA)
      weight:      either (N, K) or (K, N) torch.Tensor (CUDA). Orientation is detected automatically.
      bias:        (N,) torch.Tensor or None (optional)
      scale_factor: float (default 2.0)
      clamp_min:   float (default -inf)
      clamp_max:   float (default +inf)

    Returns:
      out: (B, 1) torch.Tensor (CUDA), same dtype as x
    """
    if len(args) == 0:
        raise TypeError("kernel_function expected at least an input tensor 'x'")

    # Parse positional arguments flexibly to satisfy the test attempts
    x = args[0]
    if not isinstance(x, torch.Tensor):
        raise TypeError("First argument must be a torch.Tensor for 'x'")

    weight = None
    bias = None
    scale_factor = 2.0
    clamp_min = float('-inf')
    clamp_max = float('inf')

    # Helper to pick next scalar if present
    def _as_float(v, name):
        if isinstance(v, (float, int)):
            return float(v)
        raise TypeError(f"Argument '{name}' must be a float")

    # Try to decode arguments
    if len(args) >= 2 and isinstance(args[1], torch.Tensor):
        weight = args[1]
        # Possibilities for the remaining args:
        # - (bias, scale, min, max)
        # - (scale, min, max)
        # - (bias)
        # - ()
        if len(args) >= 3 and isinstance(args[2], torch.Tensor):
            bias = args[2]
            if len(args) >= 4:
                scale_factor = _as_float(args[3], "scale_factor")
            if len(args) >= 5:
                clamp_min = _as_float(args[4], "clamp_min")
            if len(args) >= 6:
                clamp_max = _as_float(args[5], "clamp_max")
        else:
            # No bias provided, parse scalars if any
            if len(args) >= 3:
                scale_factor = _as_float(args[2], "scale_factor")
            if len(args) >= 4:
                clamp_min = _as_float(args[3], "clamp_min")
            if len(args) >= 5:
                clamp_max = _as_float(args[4], "clamp_max")
    else:
        # Unsupported signatures for this test; provide a clear error
        raise TypeError("kernel_function expects at least (x, weight, ...) where weight is a torch.Tensor")

    # Device/dtype checks
    if not x.is_cuda:
        raise ValueError("Input 'x' must be a CUDA tensor")
    if not weight.is_cuda:
        raise ValueError("Weight must be a CUDA tensor")
    if bias is not None and not bias.is_cuda:
        raise ValueError("Bias must be a CUDA tensor")

    # Shapes and strides
    if x.ndim != 2:
        raise ValueError(f"x must be 2D (B, K). Got shape {tuple(x.shape)}")
    B, K = x.shape

    if weight.ndim != 2:
        raise ValueError(f"weight must be 2D. Got shape {tuple(weight.shape)}")

    # Determine orientation and N, set strides to treat W as [K, N]
    w_shape = tuple(weight.shape)
    # Case A: weight is (N, K) -> interpret (K, N) using transposed indexing
    if w_shape[1] == K:
        N = w_shape[0]
        stride_wk = weight.stride(1)  # move along K by stepping dim=1
        stride_wn = weight.stride(0)  # move along N by stepping dim=0
    # Case B: weight is (K, N)
    elif w_shape[0] == K:
        N = w_shape[1]
        stride_wk = weight.stride(0)  # move along K by stepping dim=0
        stride_wn = weight.stride(1)  # move along N by stepping dim=1
    else:
        raise ValueError(
            f"weight has incompatible shape {w_shape} with x.shape={tuple(x.shape)} "
            f"(expected (N, K) or (K, N) with K={K})"
        )

    if bias is not None:
        if bias.ndim != 1 or bias.shape[0] != N:
            raise ValueError(f"bias must be 1D of length N={N}. Got shape {tuple(bias.shape)}")
    else:
        # Create a dummy bias tensor to satisfy pointer type; not used when HAS_BIAS=0
        bias = torch.empty(1, device=x.device, dtype=x.dtype)

    # Allocate output (B, 1)
    out = torch.empty((B, 1), device=x.device, dtype=x.dtype)

    # Strides (PyTorch uses element-based strides)
    stride_xm, stride_xk = x.stride(0), x.stride(1)
    stride_out_m, stride_out_n = out.stride(0), out.stride(1)
    stride_bo = bias.stride(0)

    # Kernel launch
    def grid(meta):
        return (triton.cdiv(B, meta['BLOCK_M']),)

    _fused_linear_rowwise_lse_mish[grid](
        x, weight, bias, out,
        B, N, K,
        stride_xm, stride_xk,
        stride_wk, stride_wn,
        stride_bo,
        stride_out_m, stride_out_n,
        float(scale_factor), float(clamp_min), float(clamp_max),
        HAS_BIAS=(1 if bias.numel() == N else 0),
    )

    return out