# kernel.py
# Fused Linear + Bias + ReLU (+ optional Dropout) implemented with Triton.
# - All math runs inside the Triton kernel.
# - Accumulates in fp32, stores in input/output dtype (bf16 in the test).
# - Weights are provided as [OUT_FEATURES, IN_FEATURES] and we compute x @ W^T + b.
# - Dropout is supported via a compile-time flag; test sets p=0.0 so it is a no-op.

import torch
import triton
import triton.language as tl


# Autotune configurations: tile sizes and scheduling parameters.
# We keep a small, safe set that works well for 4096x4096 matmul while remaining general.
_autotune_configs = [
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
    triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
    triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
]


@triton.autotune(configs=_autotune_configs, key=["M", "N", "K"])
@triton.jit
def _linear_bias_relu_dropout_kernel(
    x_ptr,         # *bf16/half/fp32  [M, K]
    w_ptr,         # *bf16/half/fp32  [N, K] logically as (K, N) via strides (stride_wk, stride_wn)
    b_ptr,         # *bf16/half/fp32  [N]
    o_ptr,         # *bf16/half/fp32  [M, N]
    M, N, K,       # int32 sizes
    stride_xm, stride_xk,     # x strides
    stride_wk, stride_wn,     # w viewed as (K, N) strides
    stride_om, stride_on,     # out strides
    p,                        # dropout probability (float32 runtime) — ignored if DO_DROPOUT = False
    scale,                    # 1/(1-p) (float32 runtime) — ignored if DO_DROPOUT = False
    seed,                     # int32 RNG seed — ignored if DO_DROPOUT = False
    DO_DROPOUT: tl.constexpr, # compile-time flag to include/exclude dropout
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # Program IDs for 2D launch: tiles of [M x N]
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # Compute tile start indices
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Guard against out-of-bounds indices by sanitizing pointers (coalesced access)
    offs_m_sanitized = tl.where(offs_m < M, offs_m, 0)
    offs_n_sanitized = tl.where(offs_n < N, offs_n, 0)

    # Create accumulator in fp32 for numerical stability
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Iterate along K dimension
    for k0 in tl.range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        offs_k_sanitized = tl.where(offs_k < K, offs_k, 0)

        # Pointers for a tile of X: [BLOCK_M, BLOCK_K]
        x_ptrs = x_ptr + (offs_m_sanitized[:, None] * stride_xm + offs_k_sanitized[None, :] * stride_xk)
        # Pointers for a tile of W^T (we treat W as (K, N) via given strides): [BLOCK_K, BLOCK_N]
        w_ptrs = w_ptr + (offs_k_sanitized[:, None] * stride_wk + offs_n_sanitized[None, :] * stride_wn)

        # Masks
        x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        w_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)

        # Load tiles; dtypes are inferred from pointers; other=0 for OOB
        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)

        # Fused multiply-accumulate using tl.dot; accumulate in fp32
        acc = tl.dot(x_tile, w_tile, acc)

    # Add bias (load in fp32) and apply ReLU
    b_vals = tl.load(b_ptr + offs_n_sanitized, mask=(offs_n < N), other=0.0).to(tl.float32)
    acc = acc + b_vals[None, :]
    # ReLU
    zero = tl.zeros((), dtype=tl.float32)
    acc = tl.maximum(acc, zero)

    # Optional dropout (training style: scale by 1/(1-p) on kept activations)
    if DO_DROPOUT:
        # Make per-element RNG using global element indices
        # Global indices for the output tile
        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        # Compute a unique counter per element
        counters = rows[:, None] * N + cols[None, :]
        key = tl.full([1], seed, dtype=tl.int32)
        rnd = tl.rand(key, counters)  # uniform [0, 1)
        keep = rnd > p
        acc = tl.where(keep, acc * scale, 0.0)

    # Store to output (cast to destination dtype)
    o_ptrs = o_ptr + (offs_m_sanitized[:, None] * stride_om + offs_n_sanitized[None, :] * stride_on)
    o_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    out_dtype = o_ptr.dtype.element_ty
    tl.store(o_ptrs, acc.to(out_dtype), mask=o_mask)


def kernel_function(x, weight, bias, p: float = 0.0, *, seed: int = 0):
    """
    Fused linear -> bias -> ReLU -> dropout (p=0.0 by default, i.e., no-op) using a single Triton kernel.

    What is fused (single pass over output tile):
    - Matmul: y = x @ W^T
    - Bias add: y += b (broadcasted over rows)
    - ReLU: y = max(y, 0)
    - Optional dropout: if p > 0, we apply training-style dropout: y = Bernoulli(1-p) * y / (1-p)
      Note: For numerical parity with the test, p defaults to 0.0, which makes dropout a no-op.

    Runtime behavior/constraints:
    - All math is performed inside the Triton kernel. The wrapper validates inputs, allocates the output,
      chooses the launch grid, and passes metadata only.
    - Accumulation is in float32, then cast back to the original dtype on store (bf16 in the test) for
      accuracy similar to common mixed-precision practices.

    Args:
      x:       [M, K] input tensor, CUDA, typically bfloat16 for this test
      weight:  [N, K] weight tensor (as given by the test: [OUT_FEATURES, IN_FEATURES])
      bias:    [N] bias tensor
      p:       dropout probability (default 0.0; test uses 0.0)
      seed:    RNG seed if dropout is enabled (int)

    Returns:
      out: [M, N] tensor on CUDA device, same dtype as x
    """
    # Basic validations (no compute)
    assert isinstance(x, torch.Tensor) and isinstance(weight, torch.Tensor) and isinstance(bias, torch.Tensor)
    assert x.is_cuda and weight.is_cuda and bias.is_cuda, "All tensors must be on CUDA device."
    assert x.ndim == 2 and weight.ndim == 2 and bias.ndim == 1, "Shapes must be [M, K], [N, K], [N]."
    M, Kx = x.shape
    Nw, Kw = weight.shape
    assert Kx == Kw, f"Incompatible shapes: x has K={Kx}, weight has K={Kw}."
    assert Nw == bias.shape[0], f"Incompatible shapes: weight N={Nw}, bias N={bias.shape[0]}."

    # Output allocation; follow input dtype and device
    out = torch.empty((M, Nw), device=x.device, dtype=x.dtype)

    # Determine stride metadata to view weight as (K, N) with strides (stride_wk, stride_wn)
    stride_xm, stride_xk = x.stride()
    # weight is [N, K] row-major; to view as (K, N): stride over k is weight.stride(1), over n is weight.stride(0)
    stride_wk, stride_wn = weight.stride(1), weight.stride(0)
    stride_om, stride_on = out.stride()

    # Dropout settings
    do_dropout = bool(p is not None and p > 0.0)
    if do_dropout:
        assert 0.0 <= p < 1.0, "Dropout probability must be in [0, 1)."
        scale = 1.0 / (1.0 - float(p))
        # seed is provided by user or default; ensure int32 range
        seed_i32 = int(seed) & 0x7FFFFFFF
    else:
        p = 0.0
        scale = 1.0
        seed_i32 = 0

    # 2D launch grid over output tiles
    def grid(meta):
        return (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(Nw, meta["BLOCK_N"]))

    # Launch kernel
    _linear_bias_relu_dropout_kernel[grid](
        x, weight, bias, out,
        M, Nw, Kx,
        stride_xm, stride_xk,
        stride_wk, stride_wn,
        stride_om, stride_on,
        float(p), float(scale), seed_i32,
        DO_DROPOUT=do_dropout,
    )

    return out