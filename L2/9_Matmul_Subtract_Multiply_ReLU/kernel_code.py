import torch
import triton
import triton.language as tl


"""
This module provides a single fused Triton kernel that computes:
    y = relu((x @ W^T + b - subtract_value) * multiply_value)

Fusion details:
- We fuse the following stages into a single kernel to minimize memory traffic and kernel launches:
  1) Matrix multiplication: x @ W^T
  2) Bias addition: + b (broadcast over the output features)
  3) Scalar epilogue: - subtract_value, then * multiply_value
  4) ReLU activation
- Accumulation is performed in fp32 for numerical stability; inputs/outputs are bf16 per the requirement.

Runtime constraints:
- The Python wrapper 'kernel_function' only validates inputs, allocates the output, configures the launch grid,
  and launches the Triton kernel. All math happens inside the Triton kernel.
- No torch.nn or torch.nn.functional calls are used. No PyTorch compute ops are used to implement the math.
"""


def _mm_configs():
    # A modest set of autotune configs targeting BF16 GEMM on modern NVIDIA GPUs.
    # BLOCK sizes are powers of two as recommended.
    return [
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32},  num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64},  num_warps=8, num_stages=4),
    ]


@triton.autotune(configs=_mm_configs(), key=["M", "N", "K"])
@triton.jit
def _linear_bias_relu_kernel(
    x_ptr,         # *bf16, [M, K]
    w_ptr,         # *bf16, [N, K] (row-major), we will treat it as transposed logically (K, N) via strides
    b_ptr,         # *bf16, [N]
    y_ptr,         # *bf16, [M, N]
    M, N, K,       # int32 sizes
    stride_xm, stride_xk,   # x strides
    stride_wk, stride_wn,   # weight strides to index as [K, N]: wk for K, wn for N
    stride_ym, stride_yn,   # y strides
    subtract_value,         # fp32 scalar
    multiply_value,         # fp32 scalar
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Program IDs for block tiles along M and N
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # Offsets for M and N within the tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.max_contiguous(tl.multiple_of(offs_m, BLOCK_M), BLOCK_M)
    offs_n = tl.max_contiguous(tl.multiple_of(offs_n, BLOCK_N), BLOCK_N)

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # K loop
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)

        # Compute pointers for A = x[offs_m, offs_k]
        a_ptrs = x_ptr + (offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk)
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # Compute pointers for B = w_t[offs_k, offs_n] -> using w stored as [N, K] with strides (wn, wk)
        # Logical B(K, N): first index over K uses stride_wk; second over N uses stride_wn
        b_ptrs = w_ptr + (offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn)
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Ensure inputs are BF16/FP16 for tensor cores; Triton expects a and b in low precision for tl.dot.
        # If the buffer is bf16, tl.load already returns tl.bfloat16; if it's fp16, it'll be tl.float16.
        # Accumulate into fp32.
        acc = tl.dot(a, b, acc)

    # Fused epilogue:
    # 1) + bias (broadcast along M)
    bias_ptrs = b_ptr + offs_n
    bias = tl.load(bias_ptrs, mask=(offs_n < N), other=0.0).to(tl.float32)
    acc = acc + bias[None, :]

    # 2) - subtract_value, then * multiply_value
    acc = (acc - subtract_value) * multiply_value

    # 3) ReLU
    zero = 0.0
    acc = tl.maximum(acc, zero)

    # Store to output, cast to the output dtype
    y_ptrs = y_ptr + (offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn)
    y_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    y_val = acc.to(y_ptr.dtype.element_ty)
    tl.store(y_ptrs, y_val, mask=y_mask)


def kernel_function(x: torch.Tensor,
                    weight: torch.Tensor,
                    bias: torch.Tensor,
                    subtract_value: float,
                    multiply_value: float) -> torch.Tensor:
    """
    Fused linear + bias + scalar epilogue + ReLU using Triton.

    Computes:
        y = relu((x @ W^T + b - subtract_value) * multiply_value)

    Arguments:
        x:      [batch_size, in_features], dtype must be torch.bfloat16 or torch.float16
        weight: [out_features, in_features], same dtype and device as x
        bias:   [out_features], same dtype and device as x
        subtract_value: scalar float to subtract after bias-add
        multiply_value: scalar float to multiply after subtraction

    Returns:
        y: [batch_size, out_features], same dtype and device as x

    Notes on fusion:
    - The kernel fuses matmul + bias add + scalar epilogue (sub and mul) + ReLU in a single pass.
    - Accumulation is in fp32 for numerical stability; inputs/outputs are kept in bf16/fp16.
    - This design reduces memory traffic and avoids multiple kernel launches.

    Runtime behavior:
    - The wrapper only validates inputs, allocates output, and launches the Triton kernel.
      All math executes on the device within the Triton kernel as required.
    """
    # Basic checks
    assert x.is_cuda and weight.is_cuda and bias.is_cuda, "All inputs must be CUDA tensors"
    assert x.device == weight.device == bias.device, "All tensors must be on the same device"
    assert x.ndim == 2 and weight.ndim == 2 and bias.ndim == 1, "Invalid shapes"
    M, Kx = x.shape
    Nw, Kw = weight.shape
    assert Kx == Kw, f"Incompatible inner dimensions: x.shape[1]={Kx}, weight.shape[1]={Kw}"
    N = Nw
    assert bias.shape == (N,), f"Bias must have shape ({N},), got {tuple(bias.shape)}"
    # Enforce low-precision inputs (bf16 preferred)
    assert x.dtype in (torch.bfloat16, torch.float16), "x must be bf16 or fp16"
    assert weight.dtype == x.dtype, "weight dtype must match x dtype"
    assert bias.dtype == x.dtype, "bias dtype must match x dtype"

    # Allocate output
    y = torch.empty((M, N), device=x.device, dtype=x.dtype)

    # Compute strides (in elements)
    stride_xm, stride_xk = x.stride(0), x.stride(1)
    # NOTE: weight is provided as [N, K], but we logically use its transpose [K, N].
    # For indexing B(k, n), we need stride_wk (along K) and stride_wn (along N)
    stride_wn, stride_wk = weight.stride(0), weight.stride(1)
    stride_ym, stride_yn = y.stride(0), y.stride(1)

    # Launch grid: 2D over tiles of M and N
    def grid(meta):
        return (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]))

    # Launch kernel
    _linear_bias_relu_kernel[grid](
        x, weight, bias, y,
        M, N, Kx,
        stride_xm, stride_xk,
        stride_wk, stride_wn,
        stride_ym, stride_yn,
        float(subtract_value), float(multiply_value),
    )

    return y