import math
import torch
import triton
import triton.language as tl


@triton.jit
def _maxpool2d_3x3_s2_nchw(
    x_ptr, y_ptr,
    N, C, H, W,
    OH, OW,
    stride_n, stride_c, stride_h, stride_w,
    y_stride_n, y_stride_c, y_stride_h, y_stride_w,
    PAD_H: tl.constexpr, PAD_W: tl.constexpr,
    STRIDE_H: tl.constexpr, STRIDE_W: tl.constexpr,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr,
):
    """
    Triton kernel: NCHW MaxPool2d with kernel_size=3x3, stride=2x2, padding=1x1, dilation=1.
    - Computes in float32 for numerical stability (FP32 semantics) even if I/O is bf16/fp16.
    - Treats out-of-bounds (padding) as -inf so they do not affect maxima.
    Tiling:
      - program_id(0): tiles along output width (OW)
      - program_id(1): tiles along output height (OH)
      - program_id(2): flattened N*C grid
    """
    pid_w = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_nc = tl.program_id(2)

    # Map pid_nc into n and c
    n = pid_nc // C
    c = pid_nc % C

    # Tile coordinates in output space
    oh_start = pid_h * BLOCK_H
    ow_start = pid_w * BLOCK_W
    oh = oh_start + tl.arange(0, BLOCK_H)
    ow = ow_start + tl.arange(0, BLOCK_W)

    # Masks for valid output indices
    mask_oh = oh < OH
    mask_ow = ow < OW
    mask_out = mask_oh[:, None] & mask_ow[None, :]

    # Input top-left coordinates for each output position
    ih0 = oh * STRIDE_H - PAD_H  # [BH]
    iw0 = ow * STRIDE_W - PAD_W  # [BW]
    ih_base = ih0[:, None]       # [BH, 1]
    iw_base = iw0[None, :]       # [1, BW]

    # Pointer base for this (n,c)
    x_base = x_ptr + n * stride_n + c * stride_c

    # Prepare 'other' value for masked loads as -inf, with input pointer dtype
    other_val = tl.full((BLOCK_H, BLOCK_W), -float("inf"), dtype=tl.float32).to(x_ptr.dtype.element_ty)

    # Accumulator in fp32
    acc = tl.full((BLOCK_H, BLOCK_W), -float("inf"), dtype=tl.float32)

    # Unrolled 3x3 max over the window
    for ky in range(3):
        ih = ih_base + ky
        mask_h = (ih >= 0) & (ih < H)
        for kx in range(3):
            iw = iw_base + kx
            mask_w = (iw >= 0) & (iw < W)
            m = mask_out & mask_h & mask_w
            ptrs = x_base + ih * stride_h + iw * stride_w
            vals = tl.load(ptrs, mask=m, other=other_val)
            vals_f32 = vals.to(tl.float32)
            acc = tl.maximum(acc, vals_f32)

    # Store results to output with appropriate dtype cast
    y_base = y_ptr + n * y_stride_n + c * y_stride_c
    out_ptrs = y_base + oh[:, None] * y_stride_h + ow[None, :] * y_stride_w
    tl.store(out_ptrs, acc.to(y_ptr.dtype.element_ty), mask=mask_out)


def kernel_function(
    x: torch.Tensor,
    kernel_size=(3, 3),
    stride=(2, 2),
    padding=(1, 1),
    dilation=(1, 1),
    ceil_mode: bool = False,
):
    """
    MaxPool2d NCHW implementation using Triton.
    - Fused pipeline: load -> 3x3 max reduction (in fp32) -> cast/store.
      No intermediate tensors, no extra kernel launches.
    - Parameters default to the test's target configuration:
        kernel_size=(3,3), stride=(2,2), padding=(1,1), dilation=(1,1), ceil_mode=False

    Args:
      x: Input tensor [N, C, H, W], device=cuda, dtype=float16/bfloat16/float32
      kernel_size, stride, padding, dilation, ceil_mode: pooling parameters

    Returns:
      y: Output tensor [N, C, OH, OW] on same device and dtype as x.
    """
    # Basic validations and constraints
    assert x.is_cuda, "Input must be on CUDA device."
    assert x.ndim == 4, "Input must be NCHW tensor."
    assert isinstance(kernel_size, (tuple, list)) and kernel_size == (3, 3), "Only kernel_size=(3,3) is supported."
    assert isinstance(stride, (tuple, list)) and stride == (2, 2), "Only stride=(2,2) is supported."
    assert isinstance(padding, (tuple, list)) and padding == (1, 1), "Only padding=(1,1) is supported."
    assert isinstance(dilation, (tuple, list)) and dilation == (1, 1), "Only dilation=(1,1) is supported."
    assert ceil_mode is False, "ceil_mode=False is required."

    N, C, H, W = x.shape
    KH, KW = kernel_size
    SH, SW = stride
    PH, PW = padding
    DH, DW = dilation

    # Compute output spatial dimensions (ceil_mode=False)
    eff_kh = DH * (KH - 1) + 1
    eff_kw = DW * (KW - 1) + 1
    OH = (H + 2 * PH - eff_kh) // SH + 1
    OW = (W + 2 * PW - eff_kw) // SW + 1
    assert OH > 0 and OW > 0, "Invalid output dimensions."

    # Allocate output; I/O dtype matches input; math is performed in fp32 inside the kernel.
    y = torch.empty((N, C, OH, OW), device=x.device, dtype=x.dtype)

    # Grid configuration: tile over OW, OH, and over flattened N*C
    BLOCK_H = 16
    BLOCK_W = 64
    grid = (
        triton.cdiv(OW, BLOCK_W),
        triton.cdiv(OH, BLOCK_H),
        N * C,
    )

    # Launch kernel
    _maxpool2d_3x3_s2_nchw[grid](
        x, y,
        N, C, H, W,
        OH, OW,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        PAD_H=PH, PAD_W=PW,
        STRIDE_H=SH, STRIDE_W=SW,
        BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W,
        num_warps=4, num_stages=2,
    )
    return y