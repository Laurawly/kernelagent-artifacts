import triton
import triton.language as tl
import torch


@triton.jit
def _softsign_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    Triton kernel that applies the Softsign activation:
        y = x / (1 + |x|)
    to a 1D flattened array of n_elements.

    - Uses masked loads/stores for boundary handling.
    - Computes in float32 for improved numerical stability, then casts back to input dtype.
    - Coalesced memory accesses: consecutive threads access consecutive elements.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0)  # load as input dtype
    x32 = x.to(tl.float32)
    denom = 1.0 + tl.abs(x32)
    y32 = x32 / denom
    y = y32.to(x.dtype)  # cast back to original dtype for store
    tl.store(y_ptr + offsets, y, mask=mask)


def kernel_function(x: torch.Tensor) -> torch.Tensor:
    """
    Applies Softsign activation using a single fused Triton kernel:
        y = x / (1 + |x|)

    What is fused:
    - Absolute value, addition, and division are performed in one pass directly from global memory:
      Load -> abs -> add 1 -> divide -> Store.
    - No intermediate tensors or multiple kernel launches; all math is in the Triton kernel.

    Runtime behavior:
    - Wrapper only validates inputs, allocates the output tensor, and configures the launch.
    - All math is performed on the GPU by the Triton kernel (no torch.nn or torch.* compute ops).

    Args:
        x: CUDA tensor of shape (any), typically very large as per the test, dtype in {bfloat16, float16, float32}.

    Returns:
        Tensor y with the same shape and dtype as x.
    """
    if not x.is_cuda:
        raise ValueError("Input must be a CUDA tensor.")
    if x.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError(f"Unsupported dtype {x.dtype}. Use bfloat16, float16, or float32.")
    # Ensure contiguous memory for coalesced access; torch.rand returns contiguous tensors.
    x_in = x if x.is_contiguous() else x.contiguous()

    y = torch.empty_like(x_in)
    n_elements = x_in.numel()

    # Choose a power-of-two BLOCK_SIZE for good performance; 1024 is a solid default.
    BLOCK_SIZE = 1024

    # 1D grid where each program handles BLOCK_SIZE elements.
    grid = lambda META: (triton.cdiv(n_elements, META["BLOCK_SIZE"]),)

    _softsign_kernel[grid](x_in, y, n_elements, BLOCK_SIZE=BLOCK_SIZE, num_warps=8)
    return y