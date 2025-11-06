# kernel.py
import torch
import triton
import triton.language as tl

"""
Fused Conv2d(7x7, stride=2, pad=3, groups=1) + BatchNorm (inference) + ReLU
NCHW layout, BF16 inputs/weights/params; float32 math inside kernel, BF16 on store.

Fusion rationale:
- We fuse Conv2d + BatchNorm (inference) + ReLU in a single Triton kernel to avoid
  writing/reading the intermediate activation to global memory and to reduce kernel
  launch overhead. Accumulation is performed in float32 for accuracy, and BN+ReLU
  are applied per output channel tile before storing the final BF16 result.

Runtime constraints:
- The wrapper only validates inputs, computes grid, and launches the Triton kernel.
- All compute (conv, BN, ReLU) is implemented inside the Triton kernel using
  tl.load/tl.store and arithmetic; no torch.nn or torch.nn.functional are used.
"""

@triton.jit
def _conv_bn_relu_nchw_kernel(
    x_ptr,            # *BF16  [N, C_in, H, W]
    w_ptr,            # *BF16  [C_out, C_in, KH, KW]
    gamma_ptr,        # *BF16  [C_out]
    beta_ptr,         # *BF16  [C_out]
    mean_ptr,         # *BF16  [C_out]
    var_ptr,          # *BF16  [C_out]
    y_ptr,            # *BF16  [N, C_out, H_out, W_out]

    # Sizes (runtime integers)
    N, H, W, C_OUT, H_OUT, W_OUT,

    # Strides in elements (runtime integers)
    sXn, sXc, sXh, sXw,
    sWoc, sWci, sWkh, sWkw,
    sYn, sYc, sYh, sYw,

    # Compile-time constants
    EPS: tl.constexpr,             # epsilon for BN
    C_IN: tl.constexpr,            # input channels (3 for this test)
    KH: tl.constexpr, KW: tl.constexpr,
    STRIDE_H: tl.constexpr, STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr, PAD_W: tl.constexpr,
    DIL_H: tl.constexpr, DIL_W: tl.constexpr,
    BLOCK_OC: tl.constexpr,        # OC tile size (e.g., 32)
):
    # Program ids
    pid_spatial = tl.program_id(axis=0)  # over N * H_OUT * W_OUT
    pid_oc_blk = tl.program_id(axis=1)   # over OC tiles

    # Decode spatial program id into (n, oh, ow)
    ohw = H_OUT * W_OUT
    n = pid_spatial // ohw
    rem = pid_spatial % ohw
    oh = rem // W_OUT
    ow = rem % W_OUT

    # Tile of output channels this program will compute
    oc_start = pid_oc_blk * BLOCK_OC
    oc_offsets = oc_start + tl.arange(0, BLOCK_OC)
    oc_mask = oc_offsets < C_OUT

    # Accumulator for this output pixel across a tile of output channels
    acc = tl.zeros((BLOCK_OC,), dtype=tl.float32)

    # Base for input at batch n
    base_x_n = n * sXn

    # Compute convolution: single output pixel (oh, ow) for a tile of OC
    # Iterate over input channels and filter window (KH, KW)
    for ci in range(C_IN):
        base_x_ci = base_x_n + ci * sXc
        for kh in range(KH):
            h_in = oh * STRIDE_H - PAD_H + kh * DIL_H
            h_in_valid = (h_in >= 0) & (h_in < H)
            for kw in range(KW):
                w_in = ow * STRIDE_W - PAD_W + kw * DIL_W
                in_bounds = h_in_valid & (w_in >= 0) & (w_in < W)

                # Safe coordinates for masked load
                h_safe = tl.where(in_bounds, h_in, 0)
                w_safe = tl.where(in_bounds, w_in, 0)

                # Load input scalar (BF16 -> FP32), masked with zero outside bounds
                x_off = base_x_ci + h_safe * sXh + w_safe * sXw
                x_val = tl.load(x_ptr + x_off, mask=in_bounds, other=0.0)
                x_f32 = x_val.to(tl.float32)

                # Load weights for the OC tile (BF16 -> FP32)
                w_offs = oc_offsets * sWoc + ci * sWci + kh * sWkh + kw * sWkw
                w_vec = tl.load(w_ptr + w_offs, mask=oc_mask, other=0.0)
                w_f32 = w_vec.to(tl.float32)

                # Accumulate
                acc += x_f32 * w_f32

    # BatchNorm inference and ReLU per output channel
    gamma = tl.load(gamma_ptr + oc_offsets, mask=oc_mask, other=1.0).to(tl.float32)
    beta = tl.load(beta_ptr + oc_offsets, mask=oc_mask, other=0.0).to(tl.float32)
    running_mean = tl.load(mean_ptr + oc_offsets, mask=oc_mask, other=0.0).to(tl.float32)
    running_var = tl.load(var_ptr + oc_offsets, mask=oc_mask, other=1.0).to(tl.float32)

    bn_norm = (acc - running_mean) / tl.sqrt(running_var + EPS)
    bn_out = bn_norm * gamma + beta
    relu_out = tl.maximum(bn_out, 0.0)

    # Store to output (BF16)
    base_y = n * sYn + oh * sYh + ow * sYw
    y_ptrs = y_ptr + base_y + oc_offsets * sYc
    tl.store(y_ptrs, relu_out.to(y_ptr.dtype.element_ty), mask=oc_mask)


def kernel_function(
    x: torch.Tensor,
    conv_weight: torch.Tensor,
    bn_weight: torch.Tensor,
    bn_bias: torch.Tensor,
    bn_running_mean: torch.Tensor,
    bn_running_var: torch.Tensor,
):
    """
    Fused Conv2d(7x7, stride=2, padding=3, dilation=1, groups=1) -> BatchNorm(inference) -> ReLU
    Implementation notes:
    - NCHW layout
    - BF16 inputs/weights/BN params are supported, with FP32 math inside kernel
    - Outputs are written as BF16
    - All compute (conv, BN, ReLU) is inside the Triton kernel

    Arguments:
        x:              [N, C_in, H, W], torch.bfloat16, CUDA
        conv_weight:    [C_out, C_in, KH, KW], torch.bfloat16, CUDA
        bn_weight:      [C_out], torch.bfloat16, CUDA (gamma)
        bn_bias:        [C_out], torch.bfloat16, CUDA (beta)
        bn_running_mean:[C_out], torch.bfloat16, CUDA
        bn_running_var: [C_out], torch.bfloat16, CUDA

    Returns:
        y:              [N, C_out, H_out, W_out], torch.bfloat16, CUDA

    Fused stages:
        1) Spatial convolution (accumulate in fp32)
        2) BatchNorm inference: y = gamma * (conv - mean) / sqrt(var + eps) + beta
        3) ReLU activation
    """
    assert x.is_cuda and conv_weight.is_cuda, "Tensors must be on CUDA device."
    assert x.dtype == torch.bfloat16 and conv_weight.dtype == torch.bfloat16, "x and conv_weight must be bf16."
    assert bn_weight.dtype == torch.bfloat16 and bn_bias.dtype == torch.bfloat16
    assert bn_running_mean.dtype == torch.bfloat16 and bn_running_var.dtype == torch.bfloat16

    # Shapes
    N, C_in, H, W = x.shape
    C_out, C_in_w, KH, KW = conv_weight.shape
    assert C_in == C_in_w, "C_in mismatch between x and conv_weight."
    assert bn_weight.numel() == C_out and bn_bias.numel() == C_out
    assert bn_running_mean.numel() == C_out and bn_running_var.numel() == C_out

    # Fixed conv params as per test
    stride_h, stride_w = 2, 2
    pad_h, pad_w = 3, 3
    dil_h, dil_w = 1, 1
    groups = 1
    assert groups == 1, "This kernel only supports groups=1."
    assert KH == 7 and KW == 7, "This kernel is specialized for 7x7 kernel."
    assert C_in == 3, "This kernel is specialized for C_in=3."

    # Compute output shape
    H_out = (H + 2 * pad_h - dil_h * (KH - 1) - 1) // stride_h + 1
    W_out = (W + 2 * pad_w - dil_w * (KW - 1) - 1) // stride_w + 1

    # Allocate output
    y = torch.empty((N, C_out, H_out, W_out), device=x.device, dtype=torch.bfloat16)

    # Strides in elements (PyTorch strides are already in elements)
    sXn, sXc, sXh, sXw = x.stride()
    sWoc, sWci, sWkh, sWkw = conv_weight.stride()
    sYn, sYc, sYh, sYw = y.stride()

    # Launch parameters
    BLOCK_OC = 32  # compute 32 output channels per program
    grid = (
        N * H_out * W_out,
        triton.cdiv(C_out, BLOCK_OC),
    )

    # Epsilon for BN (matches torch.nn.BatchNorm2d default in test)
    eps = 1e-5

    _conv_bn_relu_nchw_kernel[grid](
        x, conv_weight, bn_weight, bn_bias, bn_running_mean, bn_running_var, y,
        N, H, W, C_out, H_out, W_out,
        sXn, sXc, sXh, sXw,
        sWoc, sWci, sWkh, sWkw,
        sYn, sYc, sYh, sYw,
        EPS=eps,
        C_IN=C_in, KH=KH, KW=KW,
        STRIDE_H=stride_h, STRIDE_W=stride_w,
        PAD_H=pad_h, PAD_W=pad_w,
        DIL_H=dil_h, DIL_W=dil_w,
        BLOCK_OC=BLOCK_OC,
        num_warps=4,
        num_stages=2,
    )

    return y