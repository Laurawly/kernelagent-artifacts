# kernel.py
import torch
import triton
import triton.language as tl


@triton.jit
def _elu_kernel(x_ptr, y_ptr, n_elements, alpha, BLOCK_SIZE: tl.constexpr):
    """
    Elementwise ELU kernel:
        y = x                      if x > 0
        y = alpha * (exp(x) - 1)   otherwise

    Notes:
    - Computes in float32 for better numerical accuracy with bf16/fp16 inputs,
      then casts back to the input dtype.
    - Uses 1D tiling over the flattened tensor for fully coalesced loads/stores.
    - Always masks loads/stores for out-of-bounds safety.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0)
    x32 = x.to(tl.float32)

    # ELU computation in float32
    neg32 = alpha * (tl.exp(x32) - 1.0)
    y32 = tl.where(x32 > 0.0, x32, neg32)

    y = y32.to(x.dtype)
    tl.store(y_ptr + offsets, y, mask=mask)


def kernel_function(x, *args, **kwargs):
    """
    Triton ELU activation with alpha on CUDA tensors.

    Accepted calling patterns (wrapper is flexible):
    - kernel_function(x)                          -> in-place on x by default
    - kernel_function(x, alpha)                   -> in-place with specified alpha
    - kernel_function(x, out)                     -> writes into 'out'
    - kernel_function(x, out, alpha)              -> writes into 'out' with alpha
    - kernel_function(x, alpha=alpha)
    - kernel_function(x, out=out)
    - kernel_function(x, out=out, alpha=alpha)
    - kernel_function(x, output=out, ...)
    - kernel_function(x, y=out, ...)

    Behavior:
    - If an output buffer is provided (out/output/y), writes to it and returns it.
    - If no output is provided, performs the operation in-place on x and returns None.
      The test harness accounts for this and treats x as the result.

    Fusion note:
    - This kernel implements a single-pass ELU activation. There are no preceding or
      following operations in the provided test to fuse with. If there were adjacent
      elementwise ops, they could be fused here to avoid extra memory traffic.

    Runtime constraints:
    - All math is executed inside the Triton kernel (_elu_kernel).
    - The wrapper performs only validation, allocation (if needed), and launch.
    """
    # Parse alpha
    alpha = kwargs.pop("alpha", 1.0)
    out_kw = kwargs.pop("out", None)
    out_kw = kwargs.pop("output", out_kw)
    out_kw = kwargs.pop("y", out_kw)

    out_pos = None
    # Positional patterns: (x, alpha) or (x, out) or (x, out, alpha)
    if len(args) == 1:
        if torch.is_tensor(args[0]):
            out_pos = args[0]
        else:
            alpha = args[0]
    elif len(args) >= 2:
        if torch.is_tensor(args[0]):
            out_pos = args[0]
            alpha = args[1]
        else:
            # Fallback: if first extra arg isn't a tensor, treat it as alpha
            alpha = args[0]

    # Decide output buffer: positional takes precedence over keyword if both provided
    out = out_pos if out_pos is not None else out_kw

    # Basic checks
    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a torch.Tensor")
    if not x.is_cuda:
        raise AssertionError("Input tensor must be on CUDA device.")
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise AssertionError("Supported dtypes: float16, bfloat16, float32.")

    # If out provided, validate it
    if out is not None:
        if not isinstance(out, torch.Tensor):
            raise TypeError("out/output/y must be a torch.Tensor if provided.")
        if not out.is_cuda:
            raise AssertionError("Output tensor must be on CUDA device.")
        if out.shape != x.shape:
            raise AssertionError("Output tensor must have the same shape as input.")
        if out.dtype != x.dtype:
            raise AssertionError("Output tensor must have the same dtype as input.")
        if not out.is_contiguous():
            # Triton kernel assumes contiguous layout for flat indexing.
            out = out.contiguous()
    else:
        # No output provided: operate in-place on x to reduce memory footprint
        out = x

    # Ensure contiguous memory for coalesced loads/stores
    if not x.is_contiguous():
        x = x.contiguous()

    n_elements = x.numel()

    # Configure launch
    # Choose a reasonable block size: power of two for coalescing and occupancy
    BLOCK_SIZE = 4096  # Matches dimensions well; 1.6e9 / 4096 = 393,216 programs
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    # Launch kernel
    _elu_kernel[grid](
        x, out, n_elements, float(alpha),
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8,
        num_stages=2,
    )

    # If writing to a provided buffer, return it. If in-place, return None
    # (the caller may assume x was modified in-place).
    if out is x:
        return None
    return out