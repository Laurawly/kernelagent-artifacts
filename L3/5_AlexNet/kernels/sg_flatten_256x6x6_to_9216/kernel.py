# kernel.py
# Triton kernel implementing flatten(start_dim=1) for NCHW tensors.
# Target case from test: input [1024, 256, 6, 6] float32 -> output [1024, 9216]
#
# Design:
# - Single-pass, tiled, coalesced copy that treats each N slice as one row of length C*H*W.
# - 2D launch grid: axis 0 over batch rows (N), axis 1 over column tiles.
# - Each program loads/stores a BLOCK_SIZE chunk from a row with proper masking for tail tiles.
# - No PyTorch compute ops are used; all data movement happens inside the Triton kernel.
#
# Fusion note:
# - The pipeline consists of a single flatten operation (a reshape logically).
# - There are no additional ops to fuse; thus we implement the entire operation in one kernel.

import torch
import triton
import triton.language as tl


@triton.jit
def _flatten_nchw_to_2d_kernel(
    in_ptr,                  # *const T
    out_ptr,                 # *T
    N,                       # int
    CHW,                     # int (C*H*W)
    in_stride_n,             # int (stride over N in elements)
    out_stride_n,            # int (stride over output row in elements; typically CHW)
    BLOCK_SIZE: tl.constexpr
):
    """
    Kernel copies each N slice (size CHW) from input 4D NCHW tensor to 2D output [N, CHW].
    Assumes the input slice over channels/height/width is stored contiguously so that
    the slice can be addressed as a flat block of CHW elements, starting at n * in_stride_n.
    """
    # Program IDs:
    pid_n = tl.program_id(axis=0)  # row index (N)
    pid_c = tl.program_id(axis=1)  # column tile index

    # Offsets within the row for this tile
    offs = pid_c * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # Bounds mask
    mask = offs < CHW

    # Compute base addresses for this row in input and output
    in_row_start = pid_n * in_stride_n
    out_row_start = pid_n * out_stride_n

    # Coalesced load/store along the row
    vals = tl.load(in_ptr + in_row_start + offs, mask=mask, other=0)
    tl.store(out_ptr + out_row_start + offs, vals, mask=mask)


def kernel_function(x: torch.Tensor) -> torch.Tensor:
    """
    Flatten(start_dim=1) for NCHW tensors using a Triton kernel.

    What is fused:
    - The entire flatten operation is executed in a single pass kernel that
      reinterprets each N slice as a contiguous row and copies it directly.
      There are no additional ops to fuse for this subgraph.

    Constraints/Assumptions:
    - x is a 4D tensor with layout NCHW and contiguous last three dimensions.
      For standard PyTorch-contiguous NCHW tensors this holds:
        stride(0) = C*H*W, stride(1) = H*W, stride(2) = W, stride(3) = 1.
    - The test uses float32 and standard contiguous layout; we validate basic conditions.

    Wrapper responsibilities:
    - Validate inputs (shape, device).
    - Allocate the output tensor.
    - Configure the launch grid and meta-parameters.
    - Launch the Triton kernel.

    All data movement is done by the Triton kernel; no PyTorch compute ops are used.
    """
    # Basic validations allowed in wrapper
    assert isinstance(x, torch.Tensor), "Input must be a torch.Tensor"
    assert x.device.type == "cuda", "Input tensor must be on CUDA device"
    assert x.ndim == 4, "Expected a 4D NCHW tensor"
    N, C, H, W = x.shape

    # Ensure the inner N-slice (C*H*W) is contiguous so we can copy as a flat block per row
    # This is true for default-contiguous NCHW tensors created by torch.randn with no transposes.
    expected_strides = (C * H * W, H * W, W, 1)
    if tuple(x.stride()) != expected_strides:
        # For simplicity and to keep the compute in Triton, enforce standard contiguous NCHW.
        # If needed, the caller can call x = x.contiguous() before passing it here.
        raise AssertionError(
            f"Input must be contiguous NCHW with strides {expected_strides}, got {tuple(x.stride())}"
        )

    # Allocate output [N, C*H*W]
    CHW = C * H * W
    out = torch.empty((N, CHW), device=x.device, dtype=x.dtype)

    # Launch configuration
    BLOCK_SIZE = 1024  # power-of-two tile width for good coalescing
    grid = lambda META: (
        N,  # one program per row
        triton.cdiv(CHW, META["BLOCK_SIZE"]),  # tile across columns
    )

    # Launch kernel
    _flatten_nchw_to_2d_kernel[grid](
        x,                        # in_ptr
        out,                      # out_ptr
        N,                        # N
        CHW,                      # CHW
        x.stride(0),              # in_stride_n (== CHW for contiguous NCHW)
        out.stride(0),            # out_stride_n (== CHW for contiguous [N, CHW])
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
        num_stages=2,
    )

    return out