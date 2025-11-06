import torch
import triton
import triton.language as tl


@triton.jit
def _gap_flatten_kernel(
    x_ptr,                 # *const T, input tensor [N, C, H, W]
    y_ptr,                 # *T, output tensor [N, C]
    N, C, H, W,            # dimensions
    stride_n, stride_c, stride_h, stride_w,    # input strides
    out_stride_n, out_stride_c,                # output strides
    inv_hw,                # float32: 1.0 / (H * W) for FP32-semantics averaging
    BLOCK_HW: tl.constexpr # tile size for H*W reduction
):
    """
    Global Average Pool (NCHW) fused with flatten (start_dim=1).
    Each program instance computes one (n, c) output by reducing over H*W.

    - Accumulate in float32 for numerical stability (float32 semantics).
    - Store in the same dtype as y_ptr (typically matches input dtype).
    """
    pid = tl.program_id(axis=0)            # program id over N*C
    n = pid // C
    c = pid % C

    # Base pointer for this (n, c) plane
    base = n * stride_n + c * stride_c

    total_hw = H * W
    acc = tl.zeros((), dtype=tl.float32)

    # Reduce over H*W in BLOCK_HW-sized tiles
    for start in range(0, total_hw, BLOCK_HW):
        offs = start + tl.arange(0, BLOCK_HW)
        mask = offs < total_hw

        # Map linear offsets to (h, w)
        h_idx = offs // W
        w_idx = offs % W

        # Compute pointers for this tile
        ptrs = x_ptr + base + h_idx * stride_h + w_idx * stride_w

        # Load with masking, accumulate in float32
        vals = tl.load(ptrs, mask=mask, other=0.0)
        vals_f32 = vals.to(tl.float32)
        acc += tl.sum(vals_f32, axis=0)

    # Compute mean in float32, then cast to output dtype
    mean_f32 = acc * inv_hw
    out_ptr = y_ptr + n * out_stride_n + c * out_stride_c
    tl.store(out_ptr, mean_f32.to(y_ptr.dtype.element_ty))


def kernel_function(x: torch.Tensor) -> torch.Tensor:
    """
    Fused Global Average Pool 2D (adaptive_avg_pool2d to 1x1) + Flatten(start_dim=1)
    implemented in a single Triton kernel.

    What is fused:
    - Global average pooling over spatial dimensions (H, W).
    - Flattening from [N, C, 1, 1] to [N, C] is achieved by directly writing to [N, C] output.
      No separate reshape/copy step is needed.

    Notes:
    - All math is performed inside the Triton kernel.
    - Accumulation uses float32 semantics for numerical stability; results are cast back to x.dtype on store.
    - Wrapper only validates inputs, allocates outputs, computes launch grid, and passes arguments.

    Args:
        x: Input tensor of shape [N, C, H, W], on CUDA device. Supports float16/bfloat16/float32 input.

    Returns:
        Tensor of shape [N, C], same dtype/device as input.
    """
    # Basic validation
    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a torch.Tensor")
    if x.ndim != 4:
        raise ValueError(f"Expected x.ndim == 4 (NCHW), got {x.ndim}")
    if not x.is_cuda:
        raise ValueError("Input must be on CUDA device")
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(f"Unsupported dtype: {x.dtype}")

    N, C, H, W = x.shape

    # Allocate output [N, C] with same dtype/device as input
    y = torch.empty((N, C), device=x.device, dtype=x.dtype)

    # Compute scaling factor in float32 to enforce FP32 semantics for averaging
    inv_hw = float(1.0 / (H * W))

    # Strides for input and output
    sN, sC, sH, sW = x.stride()
    oN, oC = y.stride()

    # Choose reasonable tile size for spatial reduction; small H*W typical for GAP
    BLOCK_HW = 128

    # Grid: one program per (n, c)
    grid = (N * C,)

    _gap_flatten_kernel[grid](
        x, y,
        N, C, H, W,
        sN, sC, sH, sW,
        oN, oC,
        inv_hw,
        BLOCK_HW=BLOCK_HW,
    )
    return y