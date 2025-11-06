# kernel.py
import triton
import triton.language as tl
import torch

"""
Fused linear + bias + ReLU + divide kernel.

This module provides a single entry-point function: kernel_function(x, weight, bias, divisor)
that computes:
    y = relu(x @ weight.T + bias) / divisor
in a single Triton kernel. The kernel fuses:
- GEMM (matmul) with accumulation in fp32
- Bias add (broadcast over rows)
- ReLU activation
- Final scalar division

All math happens inside the Triton kernel. The Python wrapper only validates inputs, allocates
the output, computes launch grid, and dispatches the kernel.

Shapes expected by the test:
- x: [M=1024, K=8192]
- weight: [N=8192, K=8192]  (we multiply by weight.T so the GEMM uses K dimension = 8192)
- bias: [N=8192]
- output: [M=1024, N=8192]

Memory layout: contiguous row-major (as constructed by the test); strides are passed explicitly
so the kernel supports general row-major contiguous layouts and is not hardcoded to these sizes.

Performance considerations:
- Uses tile sizes tuned via @triton.autotune to pick a good configuration for the provided M, N, K
- Coalesced loads/stores and masking for boundary protection
- Accumulation in float32 for numerical stability when inputs are float16/bfloat16
"""

# A small set of reasonable autotune configs. Triton will pick the fastest at runtime for given M, N, K.
_autotune_configs = [
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=8),
    triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_stages=3, num_warps=4),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
]


@triton.autotune(configs=_autotune_configs, key=["M", "N", "K"])
@triton.jit
def _fused_linear_bias_relu_div_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,   # pointers
    M, N, K,                      # sizes
    stride_xm, stride_xk,         # x strides
    stride_wn, stride_wk,         # w strides (n, k) to access weight as transposed [K, N]
    stride_ym, stride_yn,         # y strides
    divisor,                      # scalar divisor
    BLOCK_M: tl.constexpr,        # tile size M
    BLOCK_N: tl.constexpr,        # tile size N
    BLOCK_K: tl.constexpr,        # tile size K
):
    # Program IDs for 2D grid over tiles of the output
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # Offsets for rows/cols this program handles
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Make offsets more compiler-friendly for vectorized access
    offs_m = tl.max_contiguous(tl.multiple_of(offs_m, BLOCK_M), BLOCK_M)
    offs_n = tl.max_contiguous(tl.multiple_of(offs_n, BLOCK_N), BLOCK_N)

    # Pointer to the output tile
    y_ptrs = y_ptr + (offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn)

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # K-loop
    for k0 in tl.range(0, K, BLOCK_K):
        k_ids = k0 + offs_k

        # Compute pointers for the tiles of x and w
        # x tile: [BLOCK_M, BLOCK_K] -> x_ptr + m*stride_xm + k*stride_xk
        x_ptrs = x_ptr + (offs_m[:, None] * stride_xm + k_ids[None, :] * stride_xk)
        # w tile as transposed: we want [BLOCK_K, BLOCK_N] -> w_ptr + n*stride_wn + k*stride_wk
        w_ptrs = w_ptr + (offs_n[None, :] * stride_wn + k_ids[:, None] * stride_wk)

        # Masks for OOB
        x_mask = (offs_m[:, None] < M) & (k_ids[None, :] < K)
        w_mask = (offs_n[None, :] < N) & (k_ids[:, None] < K)

        # Load x and w tiles
        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)

        # FMA using tensor cores if available; accumulates into acc (fp32)
        acc = tl.dot(x_tile, w_tile, acc)

    # Epilogue: add bias, ReLU, divide, and store
    # Load bias for current N tile and broadcast across M
    bias_vals = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0)
    bias_f32 = bias_vals.to(tl.float32)
    acc = acc + bias_f32[None, :]

    # ReLU
    zero = tl.zeros_like(acc)
    acc = tl.where(acc > 0, acc, zero)

    # Divide by scalar divisor
    acc = acc / divisor

    # Store with correct dtype
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    out = acc.to(y_ptr.dtype.element_ty)
    tl.store(y_ptrs, out, mask=out_mask)


def kernel_function(x, weight, bias, divisor=1.0, in_features=None, out_features=None):
    """
    Fused Triton implementation of:
        y = relu(x @ weight.T + bias) / divisor

    Fused stages inside the single Triton kernel:
    - Matrix multiplication (GEMM) with fp32 accumulation
    - Bias addition (broadcast)
    - ReLU activation
    - Final scalar division
    All of the above are done in one pass over the output tile to minimize memory traffic and kernel launch overhead.

    Args:
        x:       CUDA tensor of shape [M, K], dtype in {bfloat16, float16, float32}
        weight:  CUDA tensor of shape [N, K], same dtype as x
        bias:    CUDA tensor of shape [N],   same dtype as x
        divisor: Python float or scalar convertible to float32
        in_features, out_features: optional ints (ignored if shapes are provided)

    Returns:
        y: CUDA tensor of shape [M, N], dtype same as x
    """
    # Basic validations and setup-only ops (no math on data)
    assert isinstance(x, torch.Tensor) and isinstance(weight, torch.Tensor) and isinstance(bias, torch.Tensor)
    assert x.is_cuda and weight.is_cuda and bias.is_cuda, "All inputs must be CUDA tensors"
    assert x.dtype == weight.dtype == bias.dtype, "All tensors must have the same dtype"
    assert x.ndim == 2 and weight.ndim == 2 and bias.ndim == 1, "Shapes must be x[M,K], weight[N,K], bias[N]"
    M, K_x = x.shape
    N_w, K_w = weight.shape
    assert K_x == K_w, "Inner dimensions must match (x.shape[1] == weight.shape[1])"
    assert bias.shape[0] == N_w, "Bias must have length equal to weight.shape[0]"
    if in_features is not None:
        assert in_features == K_x, "Provided in_features doesn't match x.shape[1]"
    if out_features is not None:
        assert out_features == N_w, "Provided out_features doesn't match weight.shape[0]"
    # Dtype checks
    assert x.dtype in (torch.bfloat16, torch.float16, torch.float32), "Supported dtypes: bfloat16, float16, float32"

    # Make sure tensors are contiguous; allowed as setup
    x_contig = x.contiguous()
    w_contig = weight.contiguous()
    b_contig = bias.contiguous()

    # Allocate output
    y = torch.empty((M, N_w), device=x.device, dtype=x.dtype)

    # Strides
    stride_xm, stride_xk = x_contig.stride()
    stride_wn, stride_wk = w_contig.stride()  # note: we use these to access weight as transposed
    stride_ym, stride_yn = y.stride()

    # Launch grid: 2D over M and N tiles
    def grid(meta):
        return (
            triton.cdiv(M, meta["BLOCK_M"]),
            triton.cdiv(N_w, meta["BLOCK_N"]),
        )

    # Launch the Triton kernel
    _fused_linear_bias_relu_div_kernel[grid](
        x_contig, w_contig, b_contig, y,
        M, N_w, K_x,
        stride_xm, stride_xk,
        stride_wn, stride_wk,
        stride_ym, stride_yn,
        float(divisor),
    )

    return y