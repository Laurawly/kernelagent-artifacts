import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64, 'GROUP_SIZE_M': 8}, num_warps=4, num_stages=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _linear_matmul_bias_kernel(
    a_ptr,         # [M, K] input
    b_ptr,         # [N, K] weights (note: stored as [OUT, IN], we index as [K, N] via strides)
    bias_ptr,      # [N] bias
    c_ptr,         # [M, N] output
    M, N, K,
    stride_am, stride_ak,   # strides for A
    stride_bk, stride_bn,   # strides for B (treat B as [K, N])
    stride_cm, stride_cn,   # strides for C
    OUT_DTYPE_TAG: tl.constexpr,   # 0: bf16, 1: fp16, 2: fp32
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # 1D program id that is grouped along the M dimension to improve L2 locality
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Offsets for the tile this program will compute
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Make them "nice" for codegen
    offs_m = tl.where(offs_m < M, offs_m, 0)
    offs_n = tl.where(offs_n < N, offs_n, 0)
    offs_m = tl.max_contiguous(tl.multiple_of(offs_m, BLOCK_M), BLOCK_M)
    offs_n = tl.max_contiguous(tl.multiple_of(offs_n, BLOCK_N), BLOCK_N)

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Iterate over K
    k_tiles = tl.cdiv(K, BLOCK_K)
    for ki in range(0, k_tiles):
        k_start = ki * BLOCK_K
        k_idx = k_start + offs_k

        # Pointers for A and B tiles
        a_ptrs = a_ptr + (offs_m[:, None] * stride_am + k_idx[None, :] * stride_ak)
        b_ptrs = b_ptr + (k_idx[:, None] * stride_bk + offs_n[None, :] * stride_bn)

        # Masks for boundary handling
        a_mask = (offs_m[:, None] < M) & (k_idx[None, :] < K)
        b_mask = (k_idx[:, None] < K) & (offs_n[None, :] < N)

        # Load tiles; tl.dot will upcast bf16/fp16 to fp32 accumulator automatically
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Accumulate
        acc = tl.dot(a, b, acc)

    # Fused epilogue: add bias and store
    # Load bias for the N tile and broadcast across rows
    bias = tl.load(bias_ptr + offs_n, mask=(offs_n < N), other=0.0)
    bias_f32 = bias.to(tl.float32)
    acc = acc + bias_f32[None, :]

    # Cast to output dtype
    if OUT_DTYPE_TAG == 0:
        c_cast = acc.to(tl.bfloat16)
    elif OUT_DTYPE_TAG == 1:
        c_cast = acc.to(tl.float16)
    else:
        c_cast = acc  # keep float32

    # Store results with mask
    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c_cast, mask=c_mask)


def kernel_function(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Fused Linear Layer: y = x @ weight.T + bias

    What is fused:
    - Matrix multiplication and bias addition are fused into a single Triton kernel epilogue.
      This avoids an additional kernel launch and extra memory reads/writes.

    Runtime behavior:
    - The wrapper only validates inputs, allocates the output, and launches the Triton kernel.
      All math (matmul, bias add) is performed in the Triton kernel.

    Args:
      x:      [M, K] input tensor
      weight: [N, K] weights (as per PyTorch Linear with out_features=N, in_features=K)
      bias:   [N] bias vector

    Returns:
      y: [M, N] tensor with same dtype/device as x
    """
    # Basic validation
    assert x.ndim == 2, "x must be 2D: [M, K]"
    assert weight.ndim == 2, "weight must be 2D: [N, K]"
    assert bias.ndim == 1, "bias must be 1D: [N]"
    assert x.device.type == 'cuda', "x must be on CUDA device"
    assert weight.device == x.device and bias.device == x.device, "All tensors must be on the same CUDA device"
    assert x.shape[1] == weight.shape[1], f"Incompatible shapes: x.shape={x.shape}, weight.shape={weight.shape}"
    assert weight.shape[0] == bias.shape[0], "Bias length must match weight out_features"

    # Dtype checks: support bf16/fp16/fp32, compute in fp32 accumulator
    supported_dtypes = (torch.bfloat16, torch.float16, torch.float32)
    assert x.dtype in supported_dtypes, f"Unsupported dtype for x: {x.dtype}"
    assert weight.dtype == x.dtype, "x and weight must have the same dtype"
    assert bias.dtype == x.dtype, "bias must have the same dtype as x"

    M, K = x.shape
    N = weight.shape[0]

    # Output allocation
    y = torch.empty((M, N), device=x.device, dtype=x.dtype)

    # Choose dtype tag for kernel casting
    if x.dtype == torch.bfloat16:
        out_dtype_tag = 0
    elif x.dtype == torch.float16:
        out_dtype_tag = 1
    else:
        out_dtype_tag = 2

    # Grid configuration: 1D launch over tiles
    def grid(meta):
        return (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)

    # Launch kernel
    _linear_matmul_bias_kernel[grid](
        x, weight, bias, y,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(1), weight.stride(0),  # treat weight as [K, N] via strides
        y.stride(0), y.stride(1),
        OUT_DTYPE_TAG=out_dtype_tag,
    )

    return y