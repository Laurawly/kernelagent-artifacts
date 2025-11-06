# kernel.py
# Fused 3x3 Conv2d + Bias + ReLU (NCHW) implemented in Triton.
# Target test: input [1024, 256, 13, 13] -> output [1024, 384, 13, 13], stride=1, padding=1, dilation=1, groups=1
# BF16 inputs/weights/bias/outputs; all math accumulated in FP32.
#
# Notes on fusion:
# - We fuse three stages in a single Triton kernel pass:
#   1) Convolution accumulation in fp32
#   2) Per-channel bias addition
#   3) ReLU activation
# - This avoids extra memory traffic and kernel launches versus separate ops.
#
# Wrapper responsibilities:
# - Validate shapes/dtypes/devices, ensure contiguous memory, allocate output, compute grid, and launch kernel.
# - No PyTorch compute calls are used; all math is inside the Triton kernel.

import torch
import triton
import triton.language as tl


@triton.jit
def _conv3x3_bias_relu_nchw_kernel(
    x_ptr,                # *const T: input  (N, Cin, H, W)
    w_ptr,                # *const T: weight (Cout, Cin, 3, 3)
    b_ptr,                # *const T: bias   (Cout,)
    y_ptr,                # *T:       output (N, Cout, H, W)
    N, Cin, H, W, Cout,   # problem sizes
    sxN, sxC, sxH, sxW,   # input strides
    swO, swI, swK, swL,   # weight strides (O,I,KH,KW)
    syN, syC, syH, syW,   # output strides
    BLOCK_OC: tl.constexpr,   # tile size in output channels
    BLOCK_HW: tl.constexpr,   # tile size in output H*W positions
):
    # Program IDs for tiling:
    pid_oc = tl.program_id(0)  # tile along output channels
    pid_n  = tl.program_id(1)  # batch index
    pid_hw = tl.program_id(2)  # tile along H*W

    # Compute ranges this program will handle
    oc_start = pid_oc * BLOCK_OC
    hw_start = pid_hw * BLOCK_HW

    oc_offsets = oc_start + tl.arange(0, BLOCK_OC)
    hw_offsets = hw_start + tl.arange(0, BLOCK_HW)

    # Masks for boundary checks
    oc_mask = oc_offsets < Cout
    hw_mask = hw_offsets < (H * W)

    # Convert linear HW offsets to (h, w)
    h_idxs = hw_offsets // W
    w_idxs = hw_offsets % W

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_OC, BLOCK_HW), dtype=tl.float32)

    # Iterate over input channels and 3x3 kernel
    # Padding=1, stride=1, dilation=1
    # y[n, oc, h, w] = sum_{ci, ky, kx} w[oc, ci, ky, kx] * x[n, ci, h+ky-1, w+kx-1]
    for ci in range(0, Cin):
        # Unroll the 3x3 kernel spatially
        for ky in range(0, 3):
            iy = h_idxs + (ky - 1)
            mask_y = (iy >= 0) & (iy < H)
            iy_safe = tl.where(mask_y, iy, 0)
            for kx in range(0, 3):
                ix = w_idxs + (kx - 1)
                mask_x = (ix >= 0) & (ix < W)
                in_mask = hw_mask & mask_y & mask_x

                # Input pointers for the BLOCK_HW pixel positions
                # x_ptr + n*sxN + ci*sxC + iy*sxH + ix*sxW
                x_ptrs = x_ptr \
                         + pid_n * sxN \
                         + ci * sxC \
                         + iy_safe * sxH \
                         + tl.where(mask_x, ix, 0) * sxW

                x_vals = tl.load(x_ptrs, mask=in_mask, other=0).to(tl.float32)

                # Weight pointers for the BLOCK_OC output channels
                # w_ptr + oc*swO + ci*swI + ky*swK + kx*swL
                w_ptrs = w_ptr + oc_offsets * swO + ci * swI + ky * swK + kx * swL
                w_vals = tl.load(w_ptrs, mask=oc_mask, other=0).to(tl.float32)

                # Outer product update: (BLOCK_OC, 1) * (1, BLOCK_HW)
                acc += w_vals[:, None] * x_vals[None, :]

    # Add bias per out-channel and apply ReLU
    bias_vals = tl.load(b_ptr + oc_offsets, mask=oc_mask, other=0).to(tl.float32)
    acc = acc + bias_vals[:, None]
    acc = tl.maximum(acc, 0.0)

    # Store results
    # y_ptr + n*syN + oc*syC + h*syH + w*syW
    y_ptrs = y_ptr \
             + pid_n * syN \
             + oc_offsets[:, None] * syC \
             + h_idxs[None, :] * syH \
             + w_idxs[None, :] * syW

    store_mask = oc_mask[:, None] & hw_mask[None, :]
    tl.store(y_ptrs, acc.to(y_ptr.dtype.element_ty), mask=store_mask)


def kernel_function(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Fused Conv2d(3x3, stride=1, padding=1, dilation=1, groups=1) + Bias + ReLU in Triton.

    What is fused:
    - Convolution accumulation (in fp32)
    - Bias addition (per output channel)
    - ReLU activation
    All performed in a single Triton kernel pass with no intermediate tensors.

    Args:
        x:      Input tensor [N, Cin, H, W], dtype typically torch.bfloat16, device CUDA, NCHW layout.
        weight: Weight tensor [Cout, Cin, 3, 3], same dtype/device as x, contiguous.
        bias:   Bias tensor [Cout], same dtype/device.

    Returns:
        Output tensor [N, Cout, H, W] with same dtype/device as x.

    Runtime policy:
    - Only validation, allocation, and kernel launch occur in Python.
    - All math (conv, bias add, ReLU) happens inside the Triton kernel.
    """
    # Basic checks
    assert x.is_cuda and weight.is_cuda and bias.is_cuda, "All tensors must be CUDA tensors."
    assert x.ndim == 4 and weight.ndim == 4 and bias.ndim == 1, "Invalid tensor ranks."
    N, Cin, H, W = x.shape
    Cout, Cin_w, KH, KW = weight.shape
    assert Cin == Cin_w, "Input channels mismatch between x and weight."
    assert KH == 3 and KW == 3, "This kernel only supports 3x3 filters."
    assert bias.shape[0] == Cout, "Bias shape mismatch."
    # Default layout: NCHW; enforce contiguous for predictable strides
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()

    # Allocate output
    y = torch.empty((N, Cout, H, W), device=x.device, dtype=x.dtype)

    # Extract strides (in elements)
    sxN, sxC, sxH, sxW = x.stride()
    swO, swI, swK, swL = weight.stride()  # (Cout, Cin, KH, KW)
    syN, syC, syH, syW = y.stride()

    # Tiling configuration: tuned for the given problem shape (small H/W, large N and Cout)
    BLOCK_OC = 64   # tile size along output channels (power of 2)
    BLOCK_HW = 64   # tile size along H*W (power of 2)

    # 3D launch grid: (tiles over Cout) x (N) x (tiles over H*W)
    grid = (
        triton.cdiv(Cout, BLOCK_OC),
        N,
        triton.cdiv(H * W, BLOCK_HW),
    )

    # Launch Triton kernel
    _conv3x3_bias_relu_nchw_kernel[grid](
        x, weight, bias, y,
        N, Cin, H, W, Cout,
        sxN, sxC, sxH, sxW,
        swO, swI, swK, swL,
        syN, syC, syH, syW,
        BLOCK_OC=BLOCK_OC,
        BLOCK_HW=BLOCK_HW,
        num_warps=4,
        num_stages=2,
    )

    return y