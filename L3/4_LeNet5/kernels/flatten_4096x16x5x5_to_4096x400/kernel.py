import torch
import triton
import triton.language as tl


@triton.jit
def _flatten_nchw_to_nm_kernel(
    in_ptr, out_ptr,
    N, C, H, W,
    stride_n, stride_c, stride_h, stride_w,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Generic NCHW -> [N, C*H*W] flatten kernel.

    Copies input values into a 2D flattened output buffer while preserving
    row-major order. Works for any (possibly strided) NCHW input; when input
    is contiguous this reduces to a linear memcpy. All math is done inside
    Triton; the wrapper only allocates and launches.

    Args:
      in_ptr: pointer to input tensor (N, C, H, W)
      out_ptr: pointer to output tensor (N, C*H*W)
      N, C, H, W: input dimensions
      stride_n, stride_c, stride_h, stride_w: input strides
      n_elements: total number of elements = N*C*H*W
      BLOCK_SIZE: compile-time block size
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # Decompose linear index into (n, c, h, w) to support arbitrary input strides.
    CHW = C * H * W
    HW = H * W

    n = offs // CHW
    rem0 = offs % CHW
    c = rem0 // HW
    rem1 = rem0 % HW
    h = rem1 // W
    w = rem1 % W

    in_idx = n * stride_n + c * stride_c + h * stride_h + w * stride_w

    vals = tl.load(in_ptr + in_idx, mask=mask)
    # Output is contiguous [N, C*H*W], so linear index is the same as offs
    tl.store(out_ptr + offs, vals, mask=mask)


def kernel_function(x: torch.Tensor) -> torch.Tensor:
    """
    Flatten NCHW tensor to [N, C*H*W] using a Triton kernel.

    What is fused: This operator is just a reshape/copy; there are no additional
    arithmetic stages to fuse. The entire flatten is handled in a single kernel
    pass that reads from the NCHW layout and writes to a 2D [N, M] buffer
    with M = C*H*W.

    Runtime policy:
    - Wrapper performs validation, allocation, and launch configuration only.
    - All indexing/math to map linear indices to (n, c, h, w) is performed
      inside the Triton kernel; no torch compute ops are used.

    Args:
      x: Input tensor with shape [N, C, H, W] on CUDA.

    Returns:
      Tensor with shape [N, C*H*W], same dtype/device as x.
    """
    # Basic validation
    if not isinstance(x, torch.Tensor):
        raise TypeError("kernel_function expects a torch.Tensor as input")
    if not x.is_cuda:
        raise RuntimeError("Input tensor must be on CUDA device")
    if x.dim() != 4:
        raise ValueError(f"Expected 4D NCHW input, got shape {tuple(x.shape)}")

    N, C, H, W = x.shape
    M = C * H * W
    n_elements = N * M

    # Allocate output buffer [N, C*H*W], same dtype/device as input
    out = torch.empty((N, M), device=x.device, dtype=x.dtype)

    # Compute strides for generalized support (no torch compute used)
    sN, sC, sH, sW = x.stride()

    # Launch configuration
    BLOCK_SIZE = 1024  # power-of-two per guidelines
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    # Launch Triton kernel
    _flatten_nchw_to_nm_kernel[grid](
        x, out,
        N, C, H, W,
        sN, sC, sH, sW,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out