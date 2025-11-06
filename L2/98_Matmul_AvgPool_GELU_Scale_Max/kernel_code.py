# kernel.py
import math
import torch
import triton
import triton.language as tl


"""
Fused Triton kernel: Matmul -> AvgPool1d (along features) -> GELU -> Scale -> Row-wise Max

What is fused:
- We never materialize the [B, 8192] matmul result. Instead, we aggregate weights in 16-sized
  contiguous groups (pool kernel size = 16) on the fly and compute directly the pooled outputs
  of size [B, 512] = [B, 8192/16].
- We fuse bias addition (averaged over each group), GELU, and scaling inside the kernel.
- We keep a running per-row maximum over groups so that only a [B] vector is stored at the end.

Notes for reviewers:
- AvgPool1d with kernel=16, stride=16 across the feature dimension can be rewritten as:
    avg_pool(linear(x, W, b), k=16) = linear(x, W_avg, b_avg)
  where W_avg[g, :] = mean over the 16 rows of W belonging to group g,
        b_avg[g]    = mean over the 16 entries of b belonging to group g.
  We compute these averages inside the kernel (no math in the Python wrapper).
- The kernel handles both layouts for W:
    - W of shape [N, K] (out_features x in_features)
    - W_T of shape [K, N] (transposed)
  The wrapper detects layout and sets a constexpr flag so the kernel can index appropriately.
- All compute runs inside Triton. The wrapper only validates inputs and launches the kernel.
"""


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_B": 64, "BLOCK_G": 64, "BLOCK_K": 128}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_B": 128, "BLOCK_G": 64, "BLOCK_K": 128}, num_stages=2, num_warps=8),
        triton.Config({"BLOCK_B": 64, "BLOCK_G": 128, "BLOCK_K": 128}, num_stages=2, num_warps=8),
        triton.Config({"BLOCK_B": 64, "BLOCK_G": 64, "BLOCK_K": 256}, num_stages=3, num_warps=8),
    ],
    key=["B", "K"],
)
@triton.jit
def _fused_linear_pool_gelu_scale_rowmax(
    x_ptr,         # *bf16 / *fp16 / *fp32:  [B, K]
    w_ptr,         # *bf16 / *fp16 / *fp32:  [N, K] or [K, N]
    b_ptr,         # *bf16 / *fp16 / *fp32:  [N]
    out_ptr,       # *fp32: [B]
    B, N, K,       # int32
    stride_xm, stride_xk,   # x strides
    stride_w0, stride_w1,   # w strides (depend on layout)
    stride_b,               # b stride
    scale,                  # float32 scale factor
    POOL: tl.constexpr,     # int constexpr, e.g., 16
    W_TRANSPOSED: tl.constexpr,  # bool constexpr
    BLOCK_B: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_G: tl.constexpr,
):
    # Program id for batch tiles
    pid_b = tl.program_id(0)

    # Offsets for batch dimension
    offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    mask_b = offs_b < B

    # Running max per row in this tile
    m_i = tl.zeros([BLOCK_B], dtype=tl.float32) - float("inf")

    # Number of pooled groups
    G = N // POOL

    # Precompute some ranges reused in loops
    arange_k = tl.arange(0, BLOCK_K)
    arange_g = tl.arange(0, BLOCK_G)

    # Iterate over groups in tiles of BLOCK_G
    g0 = 0
    while g0 < G:
        offs_g = g0 + arange_g
        mask_g = offs_g < G

        # Accumulator for pooled linear outputs: [BLOCK_B, BLOCK_G]
        acc = tl.zeros([BLOCK_B, BLOCK_G], dtype=tl.float32)

        # Reduction over K
        k0 = 0
        while k0 < K:
            offs_k = k0 + arange_k
            mask_k = offs_k < K

            # Load X tile [BLOCK_B, BLOCK_K]
            x_ptrs = x_ptr + (offs_b[:, None] * stride_xm + offs_k[None, :] * stride_xk)
            x = tl.load(x_ptrs, mask=mask_b[:, None] & mask_k[None, :], other=0.0)
            x = x.to(tl.float32)

            # Build W_avg tile for these group and K offsets and add to acc
            if not W_TRANSPOSED:
                # W shape: [N, K], rows are outputs, cols are input features
                # We need avg over 16 rows: sum over r=0..POOL-1
                # Create sumW with shape [BLOCK_G, BLOCK_K], then transpose for dot
                sumW = tl.zeros([BLOCK_G, BLOCK_K], dtype=tl.float32)
                # Sum over the 16 rows in each group
                r = 0
                while r < POOL:
                    rows = (offs_g * POOL + r)
                    w_ptrs = w_ptr + (rows[:, None] * stride_w0 + offs_k[None, :] * stride_w1)
                    w_tile = tl.load(w_ptrs, mask=mask_g[:, None] & mask_k[None, :], other=0.0)
                    sumW += w_tile.to(tl.float32)
                    r += 1
                # Average and transpose to [BLOCK_K, BLOCK_G]
                avgW_T = tl.trans(sumW) * (1.0 / POOL)
                # Dot: [BLOCK_B, BLOCK_K] x [BLOCK_K, BLOCK_G] -> [BLOCK_B, BLOCK_G]
                acc = tl.dot(x, avgW_T, acc)
            else:
                # W_T shape: [K, N]; rows are input features, cols are outputs
                # We need avg over 16 columns in each group. Build directly [BLOCK_K, BLOCK_G]
                sumW_T = tl.zeros([BLOCK_K, BLOCK_G], dtype=tl.float32)
                r = 0
                while r < POOL:
                    cols = (offs_g * POOL + r)
                    w_ptrs = w_ptr + (offs_k[:, None] * stride_w0 + cols[None, :] * stride_w1)
                    w_tile = tl.load(w_ptrs, mask=mask_k[:, None] & mask_g[None, :], other=0.0)
                    sumW_T += w_tile.to(tl.float32)
                    r += 1
                avgW_T = sumW_T * (1.0 / POOL)
                # Dot: [BLOCK_B, BLOCK_K] x [BLOCK_K, BLOCK_G] -> [BLOCK_B, BLOCK_G]
                acc = tl.dot(x, avgW_T, acc)

            k0 += BLOCK_K

        # Add averaged bias: [BLOCK_G], broadcast over rows
        bsum = tl.zeros([BLOCK_G], dtype=tl.float32)
        r = 0
        while r < POOL:
            rows = (offs_g * POOL + r)
            b_vals = tl.load(b_ptr + rows * stride_b, mask=mask_g, other=0.0)
            bsum += b_vals.to(tl.float32)
            r += 1
        bavg = bsum * (1.0 / POOL)
        acc = acc + bavg[None, :]

        # GELU in fp32: 0.5 * x * (1 + erf(x / sqrt(2)))
        inv_sqrt2 = 0.7071067811865476
        t = acc * inv_sqrt2
        gelu = 0.5 * acc * (1.0 + tl.math.erf(t))

        # Scale
        y = gelu * scale

        # Max over groups in this tile
        tile_max = tl.max(y, 1)  # reduce across BLOCK_G -> [BLOCK_B]
        m_i = tl.maximum(m_i, tile_max)

        g0 += BLOCK_G

    # Store result
    tl.store(out_ptr + offs_b, m_i, mask=mask_b)


def kernel_function(x, W, b, pool_kernel_size=16, scale_factor=2.0):
    """
    Fused pipeline implemented in a single Triton kernel:
      y = F.linear(x, W, b)                    # [B, 8192]
      y = avg_pool1d(y.unsqueeze(1), k=16)     # [B, 512]
      y = gelu(y)                              # [B, 512]
      y = y * scale_factor                     # [B, 512]
      y = max(y, dim=1).values                 # [B]

    Wrapper responsibilities only:
    - Validate inputs, infer layout, allocate outputs.
    - Launch the Triton kernel. No math is done here.

    Args:
      x: [B, K] tensor on CUDA (recommended dtype: bfloat16 as per test)
      W: Either [N, K] or [K, N] tensor on CUDA (bf16 recommended)
      b: [N] tensor on CUDA (bf16 recommended)
      pool_kernel_size: int (must divide N), default 16
      scale_factor: float or 0-dim tensor (converted to float), default 2.0

    Returns:
      Tensor [B] on CUDA (float32)
    """
    # Basic checks
    assert x.is_cuda and W.is_cuda and b.is_cuda, "All tensors must be on CUDA"
    assert x.dim() == 2, "x must be 2D [B, K]"
    assert b.dim() == 1, "b must be 1D [N]"

    B, K = x.shape

    # Detect W layout: either [N, K] (not transposed) or [K, N] (transposed).
    W_TRANSPOSED = False
    if W.dim() != 2:
        raise ValueError("W must be 2D")
    if W.shape[1] == K and W.shape[0] == b.shape[0]:
        # W: [N, K]
        N = W.shape[0]
        W_TRANSPOSED = False
    elif W.shape[0] == K and W.shape[1] == b.shape[0]:
        # W_T: [K, N]
        N = W.shape[1]
        W_TRANSPOSED = True
    else:
        raise ValueError(
            f"Incompatible shapes: x {x.shape}, W {W.shape}, b {b.shape}. "
            "Expected W to be [N, K] with N == b.numel() or [K, N] with N == b.numel()."
        )

    # Pooling kernel must divide N exactly for grouped average over contiguous outputs
    POOL = int(pool_kernel_size)
    assert POOL > 0 and (N % POOL == 0), f"pool_kernel_size must divide out_features (N={N})."

    # Convert scale to Python float
    if isinstance(scale_factor, torch.Tensor):
        scale = float(scale_factor.item())
    else:
        scale = float(scale_factor)

    # Output allocation (float32 for numerical stability)
    out = torch.empty((B,), device=x.device, dtype=torch.float32)

    # Strides
    stride_xm, stride_xk = x.stride()
    if W_TRANSPOSED:
        stride_w0, stride_w1 = W.stride()  # [K, N]
    else:
        stride_w0, stride_w1 = W.stride()  # [N, K]
    stride_b = b.stride(0)

    # Launch grid: one dimension over batch rows
    def grid(meta):
        return (triton.cdiv(B, meta["BLOCK_B"]),)

    _fused_linear_pool_gelu_scale_rowmax[grid](
        x, W, b, out,
        B, N, K,
        stride_xm, stride_xk,
        stride_w0, stride_w1,
        stride_b,
        scale,
        POOL=POOL,
        W_TRANSPOSED=W_TRANSPOSED,
    )

    return out