import torch
import triton
import triton.language as tl


# FUSED KERNEL: adaptive_avg_pool2d(NCHW, out=1x1) -> flatten(start_dim=1) -> dropout(p=0.0) -> linear(1024->1000, bias)
# We fuse all math into a single Triton kernel:
#  - For each (batch_tile, out_feature_tile), we iterate over channel tiles.
#  - Within each channel tile, we compute the per-(n, c) spatial average (1/(H*W) * sum_{h,w} x[n,c,h,w]) in FP32.
#  - We then GEMM the pooled [BLOCK_N, BLOCK_C] tile with weight [BLOCK_C, BLOCK_O] and accumulate into [BLOCK_N, BLOCK_O].
#  - After the C-loop, we add bias and store in the desired output dtype.
#
# Notes:
#  - We perform compute in float32 for numerical robustness ("FP32 semantics") and cast on store.
#  - Dropout with p=0.0 is a no-op and thus not applied.
#  - All math is done inside Triton; the Python wrapper only allocates/launches.
#
# Dimensions / strides:
#   x: [N, C, H, W] with strides (sxn, sxc, sxh, sxw)
#   w: [O, C]       with strides (swo, swc)  (row-major: swo = C, swc = 1)
#   b: [O]
#   out: [N, O]     with strides (son, soo)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 1, 'BLOCK_O': 128, 'BLOCK_C': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 2, 'BLOCK_O': 128, 'BLOCK_C': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_N': 4, 'BLOCK_O': 128, 'BLOCK_C': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_N': 2, 'BLOCK_O': 256, 'BLOCK_C': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_N': 1, 'BLOCK_O': 256, 'BLOCK_C': 128}, num_warps=8, num_stages=4),
    ],
    key=['N', 'O', 'C', 'H', 'W'],
)
@triton.jit
def _avgpool_flatten_linear_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, C, H, W, O,
    sxn, sxc, sxh, sxw,
    swo, swc,
    son, soo,
    BLOCK_N: tl.constexpr,
    BLOCK_O: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    # Program ids map tiles over (O, N)
    pid_o = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_o = pid_o * BLOCK_O + tl.arange(0, BLOCK_O)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_o = offs_o < O
    mask_n = offs_n < N

    # Accumulator for [BLOCK_N, BLOCK_O] in FP32
    acc = tl.zeros((BLOCK_N, BLOCK_O), dtype=tl.float32)

    # Iterate over C in tiles
    num_ctiles = tl.cdiv(C, BLOCK_C)
    for ci in range(0, num_ctiles):
        offs_c = ci * BLOCK_C + tl.arange(0, BLOCK_C)
        mask_c = offs_c < C

        # Compute pooled X tile: shape [BLOCK_N, BLOCK_C]
        # pooled[n, c] = (1/(H*W)) * sum_{h=0..H-1} sum_{w=0..W-1} x[n, c, h, w]
        sum_hw = tl.zeros((BLOCK_N, BLOCK_C), dtype=tl.float32)

        # Sum over spatial dims in FP32
        # Using nested loops to remain general in H, W; each load is [BLOCK_N, BLOCK_C]
        for h in range(0, H):
            # loop over W
            for w in range(0, W):
                x_ptrs = x_ptr + \
                    (offs_n[:, None] * sxn) + \
                    (offs_c[None, :] * sxc) + \
                    (h * sxh) + (w * sxw)
                x_mask = (mask_n[:, None] & mask_c[None, :])
                x_val = tl.load(x_ptrs, mask=x_mask, other=0.0)
                x_val_f32 = x_val.to(tl.float32)
                sum_hw += x_val_f32

        inv_hw = 1.0 / (H * W)
        pooled = sum_hw * inv_hw  # [BLOCK_N, BLOCK_C], fp32

        # Load weight tile [BLOCK_C, BLOCK_O]
        w_ptrs = w_ptr + (offs_c[:, None] * swc) + (offs_o[None, :] * swo)
        w_mask = (mask_c[:, None] & mask_o[None, :])
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0).to(tl.float32)

        # GEMM: [BLOCK_N, BLOCK_C] x [BLOCK_C, BLOCK_O] -> [BLOCK_N, BLOCK_O]
        acc = tl.dot(pooled, w_tile, acc)

    # Add bias [BLOCK_O], broadcast over BLOCK_N
    b_vals = tl.load(b_ptr + offs_o, mask=mask_o, other=0.0).to(tl.float32)
    acc = acc + b_vals[None, :]

    # Store result (cast to output dtype)
    out_ptrs = out_ptr + (offs_n[:, None] * son) + (offs_o[None, :] * soo)
    out_mask = (mask_n[:, None] & mask_o[None, :])
    out = acc.to(out_ptr.dtype.element_ty)
    tl.store(out_ptrs, out, mask=out_mask)


def kernel_function(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Fused kernel wrapper:
      Computes: adaptive_avg_pool2d(x, output_size=1x1) -> flatten(start_dim=1) -> dropout(p=0.0) -> linear(w, b)
      Data layout: NCHW for x; Linear is y = x_flat @ w.T + b with w shape [O, C], b shape [O].

    Fusion rationale:
      - We fuse spatial average pooling, flatten, and the linear layer in one pass:
        - For each tile, we directly accumulate the spatial sum over HxW in FP32, normalize by 1/(H*W),
          and immediately perform the matrix multiply with the corresponding weight tile.
        - This removes the need to materialize intermediate pooled/flattened tensors and eliminates
          extra global memory traffic and kernel launches.
      - Dropout(p=0.0) is a no-op; intentionally omitted to simplify and keep correctness.

    Constraints:
      - All computation is performed in the Triton kernel with FP32 accumulation; wrapper only handles validation,
        allocation, and launch.

    Args:
      x: [N, C, H, W], NCHW, typically bfloat16/float16/float32
      w: [O, C], same device as x
      b: [O], same device as x

    Returns:
      out: [N, O], same dtype as x (cast from FP32 accumulation on store)
    """
    assert x.is_cuda and w.is_cuda and b.is_cuda, "All tensors must be CUDA tensors"
    assert x.ndim == 4, "x must be [N, C, H, W]"
    assert w.ndim == 2 and b.ndim == 1, "w must be [O, C], b must be [O]"
    N, C, H, W = x.shape
    O, Cw = w.shape
    assert C == Cw, f"Incompatible shapes: C={C}, w.shape[1]={Cw}"
    assert b.shape[0] == O, f"Incompatible bias shape: {b.shape} vs O={O}"
    assert x.device == w.device == b.device, "All tensors must be on the same device"

    # Output dtype: follow input dtype; all math will be done in FP32 and cast on store
    out_dtype = x.dtype
    out = torch.empty((N, O), device=x.device, dtype=out_dtype)

    # Strides in elements
    sxn, sxc, sxh, sxw = x.stride()
    swo, swc = w.stride()
    son, soo = out.stride()

    # Grid config: tiles over (O, N)
    def grid(meta):
        return (
            triton.cdiv(O, meta['BLOCK_O']),
            triton.cdiv(N, meta['BLOCK_N']),
        )

    _avgpool_flatten_linear_kernel[grid](
        x, w, b, out,
        N, C, H, W, O,
        sxn, sxc, sxh, sxw,
        swo, swc,
        son, soo,
    )

    return out