import torch
import triton
import triton.language as tl


"""
Fused Linear + Bias + ReLU using a single Triton kernel.

Operation:
  y = ReLU(x @ weight.T + bias)

Shapes for the test:
  - x      : [N, IN_FEAT]  = [4096, 400]
  - weight : [OUT_FEAT, IN_FEAT] = [120, 400]
  - bias   : [OUT_FEAT] = [120]
  - y      : [N, OUT_FEAT] = [4096, 120]

Fusion rationale:
- We fuse the matrix multiplication with bias addition and ReLU into a single kernel:
  - MatMul accumulation is done in fp32 for numerical stability.
  - Bias is broadcast and added in the epilogue, also in fp32.
  - ReLU activation is then applied before storing.
- This avoids materializing the intermediate result in global memory and eliminates extra kernel launches.

Runtime notes:
- Wrapper performs only validation/allocation/launch; all math stays inside the Triton kernel.
- Supports float32/bfloat16 inputs. Accumulates in float32 and casts to output dtype at store.
- The kernel indexes weight as if it is transposed by passing appropriate strides, so no PyTorch transpose is performed.
"""


# A small set of autotuned configurations for 2D matmul tiles
_configs = [
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=4),
    triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=8, num_stages=4),
]


@triton.autotune(configs=_configs, key=['M', 'N', 'K'])
@triton.jit
def _linear_bias_relu_kernel(
    x_ptr,          # *dtype, [M, K]
    w_ptr,          # *dtype, [N, K] logically as weight.T with custom strides
    b_ptr,          # *dtype, [N]
    y_ptr,          # *dtype, [M, N]
    M, N, K,        # int
    stride_xm, stride_xk,      # strides of x
    stride_wk, stride_wn,      # strides treating w as [K, N] (i.e., stride over K and over N)
    stride_ym, stride_yn,      # strides of y
    BLOCK_M: tl.constexpr,     # tile M
    BLOCK_N: tl.constexpr,     # tile N
    BLOCK_K: tl.constexpr,     # tile K
):
    # Program IDs for tiling along M and N dimensions
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute tile indices
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Masks for boundaries
    mask_m = offs_m < M
    mask_n = offs_n < N

    # Accumulator in fp32 for numerical stability
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k0 in range(0, K, BLOCK_K):
        k_idxs = k0 + offs_k
        mask_k = k_idxs < K

        # Pointers for current tiles in X and W (treat W as transposed via strides)
        x_ptrs = x_ptr + (offs_m[:, None] * stride_xm + k_idxs[None, :] * stride_xk)
        w_ptrs = w_ptr + (k_idxs[:, None] * stride_wk + offs_n[None, :] * stride_wn)

        # Load with masking and coalesced access
        a = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        b = tl.load(w_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)

        # FMA on Tensor Cores when possible; accumulate in fp32
        acc = tl.dot(a, b, acc)

    # Epilogue: add bias (broadcast) in fp32 and apply ReLU
    bias = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
    acc = acc + bias[None, :]
    acc = tl.maximum(acc, 0.0)

    # Store results, casting to output dtype
    y_ptrs = y_ptr + (offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn)
    out_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(y_ptrs, acc.to(y_ptr.dtype.element_ty), mask=out_mask)


def kernel_function(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Fused linear + bias + ReLU.

    Computes:
        y = ReLU(x @ weight.T + bias)

    Args:
        x:      [M, K] input tensor (float32 or bfloat16), CUDA
        weight: [N, K] row-major as [OUT_FEAT, IN_FEAT], CUDA, same dtype as x preferred
        bias:   [N] bias, CUDA, same dtype as x preferred

    Returns:
        y: [M, N] tensor on CUDA, same dtype as x

    Notes:
    - The entire pipeline (matmul, bias add, ReLU) is fused inside a single Triton kernel.
    - Wrapper performs only validation, allocation, and launch. No compute occurs outside the kernel.
    - Accumulates in fp32 for numerical stability, casts to output dtype upon store.
    """
    # Basic validations
    assert x.is_cuda and weight.is_cuda and bias.is_cuda, "All tensors must be on CUDA"
    assert x.ndim == 2 and weight.ndim == 2 and bias.ndim == 1, "Invalid tensor ranks"
    M, Kx = x.shape
    Nw, Kw = weight.shape
    assert Kx == Kw, f"Incompatible shapes: x(K={Kx}) vs weight(K={Kw})"
    N = Nw
    assert bias.shape[0] == N, f"Bias shape {bias.shape} incompatible with weight.out={N}"

    # Allocate output; by default return same dtype as input x
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)

    # Prepare strides; treat weight as [K, N] by swapping strides (no actual transpose)
    stride_xm, stride_xk = x.stride()
    # weight is [N, K] row-major => stride_wk = stride along K, stride_wn = stride along N when seen as [K, N]
    stride_wn, stride_wk = weight.stride()  # original [N, K] -> (stride_N, stride_K)
    # out strides
    stride_ym, stride_yn = out.stride()

    # Compute launch grid
    def grid(meta):
        return (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))

    # Launch kernel
    _linear_bias_relu_kernel[grid](
        x, weight, bias, out,
        M, N, Kx,
        stride_xm, stride_xk,
        stride_wk, stride_wn,   # treat weight as [K, N]
        stride_ym, stride_yn,
    )

    return out