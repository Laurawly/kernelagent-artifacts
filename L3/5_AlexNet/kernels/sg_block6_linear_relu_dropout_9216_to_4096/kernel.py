import torch
import triton
import triton.language as tl


"""
Fused kernel: linear (x @ weight.T + bias) -> ReLU -> Dropout(p=0.0)

- All compute is performed inside the Triton kernel _linear_relu_dropout.
- The wrapper kernel_function only validates inputs, allocates the output, configures the grid, and launches the kernel.
- Fusion: The matmul accumulation, bias addition, ReLU activation, and (no-op) dropout(p=0.0) are fused into a single kernel to avoid extra memory traffic and kernel launches.

Shapes:
  x:       [M, K]          (batch, in_features)
  weight:  [N, K]          (out_features, in_features), as per PyTorch Linear
  bias:    [N]
  result:  [M, N]
Dtypes:
  Test uses torch.bfloat16 for x/weight/bias. We support float16, bfloat16, or float32 inputs,
  accumulate in float32, then cast back to the output dtype on store.

Runtime constraints: no PyTorch compute ops are used; all math runs in Triton.
"""

# A small set of reasonable autotune configs; will select the best for given M, N, K.
_autotune_configs = [
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64,  "BLOCK_K": 32}, num_warps=4, num_stages=4),
    triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4, num_stages=4),
    triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64,  "BLOCK_K": 32}, num_warps=4, num_stages=4),
]


@triton.autotune(configs=_autotune_configs, key=["M", "N", "K"])
@triton.jit
def _linear_relu_dropout(x_ptr, w_ptr, b_ptr, y_ptr,
                         M, N, K,
                         stride_xm, stride_xk,
                         stride_wn, stride_wk,
                         stride_ym, stride_yn,
                         P_DROPOUT: tl.constexpr,     # compile-time constant; here we pass 0.0 (no-op)
                         BLOCK_M: tl.constexpr,
                         BLOCK_N: tl.constexpr,
                         BLOCK_K: tl.constexpr):
    # Program IDs for 2D tiling over M and N
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # Offsets for rows/cols handled by this program
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Masks for M and N bounds
    mask_m = offs_m < M
    mask_n = offs_n < N

    # Accumulator in FP32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension in BLOCK_K steps
    # Use tl.range for dynamic loop bounds
    for k0 in tl.range(0, K, BLOCK_K):
        k_idx = k0 + offs_k
        k_mask = k_idx < K

        # Pointers for X tile [BLOCK_M, BLOCK_K]
        a_ptrs = x_ptr + (offs_m[:, None] * stride_xm + k_idx[None, :] * stride_xk)
        # Pointers for W tile (note: W has shape [N, K]; we need [K, N] for the dot)
        # So we index as [k, n] by broadcasting K along rows and N along columns.
        b_ptrs = w_ptr + (k_idx[:, None] * stride_wk + offs_n[None, :] * stride_wn)

        # Masks for loads
        a_mask = mask_m[:, None] & k_mask[None, :]
        b_mask = k_mask[:, None] & mask_n[None, :]

        # Load tiles (use other=0.0 for out-of-bounds)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Accumulate
        acc = tl.dot(a, b, acc)

    # Add bias: broadcast bias[N] across rows
    bias_vals = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + bias_vals[None, :]

    # ReLU
    acc = tl.maximum(acc, 0.0)

    # Dropout p=0.0 => identity; keep here for completeness and future extension.
    if P_DROPOUT > 0.0:
        # With P_DROPOUT as constexpr=0.0 in this test, this branch is compiled out.
        # A proper dropout implementation would sample RNG and apply scaling.
        keep_prob = 1.0 - P_DROPOUT
        acc = acc * keep_prob

    # Store result cast to output pointer dtype
    y_ptrs = y_ptr + (offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn)
    out_dtype = y_ptr.dtype.element_ty
    tl.store(y_ptrs, acc.to(out_dtype), mask=mask_m[:, None] & mask_n[None, :])


def kernel_function(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Fused linear + ReLU + Dropout(p=0.0) using a single Triton kernel.

    What is fused inside the kernel:
    - Matrix multiply with FP32 accumulation: x @ weight.T
    - Bias addition: + bias
    - ReLU activation: relu()
    - Dropout(p=0.0): identity (kept in-kernel for pipeline completeness; no RNG used)

    Wrapper responsibilities (no math, per runtime constraints):
    - Validate arguments (shapes/dtypes/devices)
    - Allocate output tensor
    - Configure and launch the Triton kernel
    - Return the result

    Args:
        x:      [M, K] input tensor
        weight: [N, K] weight tensor (same layout as torch.nn.Linear.weight)
        bias:   [N]    bias tensor
    Returns:
        y:      [M, N] fused output with same dtype/device as x
    """
    # Basic validations
    assert isinstance(x, torch.Tensor) and isinstance(weight, torch.Tensor) and isinstance(bias, torch.Tensor), \
        "x, weight, bias must be torch.Tensors"
    assert x.is_cuda and weight.is_cuda and bias.is_cuda, "All tensors must be on CUDA device"
    assert x.ndim == 2 and weight.ndim == 2 and bias.ndim == 1, "x:[M,K], weight:[N,K], bias:[N]"
    M, Kx = x.shape
    Nw, Kw = weight.shape
    assert Kx == Kw, f"Incompatible in_features: x.shape[1]={Kx} vs weight.shape[1]={Kw}"
    assert Nw == bias.shape[0], f"Bias shape mismatch: bias.shape[0]={bias.shape[0]} vs weight.shape[0]={Nw}"

    # Dtype checks: allow bf16/fp16/fp32; enforce all inputs share the same dtype
    assert x.dtype == weight.dtype == bias.dtype, "x, weight, bias must have the same dtype"
    assert x.dtype in (torch.bfloat16, torch.float16, torch.float32), \
        "Supported dtypes: bfloat16, float16, float32"

    # Allocate output
    y = torch.empty((M, Nw), device=x.device, dtype=x.dtype)

    # Grid: 2D over M and N tiles
    def grid(meta):
        return (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(Nw, meta["BLOCK_N"]))

    # Launch fused kernel; dropout p=0.0 per test requirements
    _linear_relu_dropout[grid](
        x, weight, bias, y,
        M, Nw, Kx,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
        y.stride(0), y.stride(1),
        P_DROPOUT=0.0,  # identity
    )
    return y