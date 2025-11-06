import torch
import triton
import triton.language as tl


# Fused Conv2d (stride=1, no padding, 5x5) + Bias + ReLU + MaxPool2d (2x2, stride=2)
# Data layout: NCHW for input and output; OIHW for weights.
# DType: supports bf16/fp16/fp32 I/O; accumulates in fp32 for numerical stability.
#
# Kernel computes one pooled output element [n, oc, py, px] per program:
#   - Computes the 2x2 window of conv outputs at conv coordinates:
#       (y0, x0), (y0, x0+1), (y0+1, x0), (y0+1, x0+1) with y0 = 2*py, x0 = 2*px
#   - Fuses bias add and ReLU
#   - Applies max over the 2x2 window to produce pooled result
#
# This design aggressively fuses all stages to avoid intermediate tensors and extra memory traffic.
@triton.jit
def _conv_relu_maxpool2d_fused(
    x_ptr, w_ptr, b_ptr, y_ptr,
    N, C_IN, H, W, C_OUT,
    P_H, P_W,
    x_stride_n, x_stride_c, x_stride_h, x_stride_w,
    w_stride_o, w_stride_i, w_stride_h, w_stride_w,
    y_stride_n, y_stride_c, y_stride_h, y_stride_w,
    KH: tl.constexpr, KW: tl.constexpr, CIN: tl.constexpr,
):
    # Program IDs: map to a single output pooled element
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_p = tl.program_id(2)

    py = pid_p // P_W
    px = pid_p % P_W

    # Out-of-bounds protection mask (for generality)
    oob = (pid_n >= N) | (pid_oc >= C_OUT) | (py >= P_H) | (px >= P_W)
    # Early-exit guard via masked store later; compute regardless as shapes are expected valid
    # but keep mask for safety and to follow guidelines.

    # Top-left conv output coordinates for the 2x2 maxpool window
    y0 = 2 * py
    x0 = 2 * px

    # Four accumulators for the 2x2 conv outputs (fp32 accumulation)
    acc00 = tl.zeros((), dtype=tl.float32)
    acc01 = tl.zeros((), dtype=tl.float32)
    acc10 = tl.zeros((), dtype=tl.float32)
    acc11 = tl.zeros((), dtype=tl.float32)

    # Compute convolution for four neighboring output points simultaneously to reuse weights
    # and reduce redundant loads.
    for ci in range(CIN):
        # Base pointer offset for the current input channel
        base_ic = pid_n * x_stride_n + ci * x_stride_c
        for ky in range(KH):
            yi = y0 + ky
            base_y = base_ic + yi * x_stride_h
            for kx in range(KW):
                xi = x0 + kx
                # Weight load [oc, ci, ky, kx]
                w_off = pid_oc * w_stride_o + ci * w_stride_i + ky * w_stride_h + kx * w_stride_w
                w_val = tl.load(w_ptr + w_off)
                w_f32 = w_val.to(tl.float32)
                # Input loads for the 2x2 positions:
                # (yi, xi), (yi, xi+1), (yi+1, xi), (yi+1, xi+1)
                base_xy = base_y + xi * x_stride_w
                i00 = tl.load(x_ptr + base_xy)
                i01 = tl.load(x_ptr + base_xy + x_stride_w)
                i10 = tl.load(x_ptr + base_xy + x_stride_h)
                i11 = tl.load(x_ptr + base_xy + x_stride_h + x_stride_w)
                # Accumulate (cast to fp32)
                i00 = i00.to(tl.float32)
                i01 = i01.to(tl.float32)
                i10 = i10.to(tl.float32)
                i11 = i11.to(tl.float32)
                acc00 += i00 * w_f32
                acc01 += i01 * w_f32
                acc10 += i10 * w_f32
                acc11 += i11 * w_f32

    # Bias add (per-output-channel)
    b_val = tl.load(b_ptr + pid_oc).to(tl.float32)
    acc00 += b_val
    acc01 += b_val
    acc10 += b_val
    acc11 += b_val

    # ReLU
    zero = tl.zeros((), dtype=tl.float32)
    acc00 = tl.maximum(acc00, zero)
    acc01 = tl.maximum(acc01, zero)
    acc10 = tl.maximum(acc10, zero)
    acc11 = tl.maximum(acc11, zero)

    # MaxPool 2x2 over the four conv outputs
    m0 = tl.maximum(acc00, acc01)
    m1 = tl.maximum(acc10, acc11)
    pooled = tl.maximum(m0, m1)

    # Store result (cast to output dtype)
    y_off = pid_n * y_stride_n + pid_oc * y_stride_c + py * y_stride_h + px * y_stride_w
    out_val = pooled.to(y_ptr.dtype.element_ty)
    tl.store(y_ptr + y_off, out_val, mask=~oob)


def kernel_function(x, conv_weight, conv_bias):
    """
    Fused Triton kernel: conv2d (5x5, stride=1, no padding) + bias + ReLU + maxpool2d (2x2, stride=2).

    What is fused:
    - Convolution accumulation directly followed by bias addition and ReLU, then a 2x2 max-pooling
      reduction, all computed in a single pass without materializing intermediate tensors.

    Restrictions and assumptions matched to the test:
    - Input layout: NCHW
    - Weight layout: OIHW
    - Kernel size: 5x5
    - Conv stride: 1
    - Conv padding: 0
    - MaxPool: kernel_size=2x2, stride=2, padding=0
    - DTypes: bf16 (as used in the test), also works with fp16/fp32 inputs (accumulation in fp32)

    Runtime responsibilities:
    - Validate inputs and shapes
    - Allocate output tensor
    - Configure and launch Triton kernel
    """
    # Basic validations
    assert isinstance(x, torch.Tensor) and isinstance(conv_weight, torch.Tensor) and isinstance(conv_bias, torch.Tensor)
    assert x.is_cuda and conv_weight.is_cuda and conv_bias.is_cuda, "All tensors must be on CUDA device"
    assert x.ndim == 4 and conv_weight.ndim == 4 and conv_bias.ndim == 1, "Shapes must be NCHW / OIHW and bias of shape [C_out]"
    N, C_in, H, W = x.shape
    C_out, C_in_w, KH, KW = conv_weight.shape
    assert C_in == C_in_w, "Input channels mismatch between x and weight"
    assert conv_bias.shape[0] == C_out, "Bias shape mismatch"
    assert KH == 5 and KW == 5, "This fused kernel expects a 5x5 convolution kernel"
    # Conv output spatial size (no padding, stride=1)
    H_conv = H - KH + 1
    W_conv = W - KW + 1
    assert H_conv > 0 and W_conv > 0, "Invalid spatial sizes for convolution"
    # MaxPool 2x2 stride 2
    pool_ks = 2
    pool_stride = 2
    assert H_conv % pool_stride == 0 and W_conv % pool_stride == 0, "Conv output must be divisible by pooling stride=2"
    P_H = H_conv // pool_stride
    P_W = W_conv // pool_stride

    # Supported dtypes: bf16/fp16/fp32. Accumulation always in fp32.
    assert x.dtype in (torch.bfloat16, torch.float16, torch.float32), "Unsupported input dtype"
    assert conv_weight.dtype == x.dtype and conv_bias.dtype == x.dtype, "All dtypes must match"

    # Allocate output
    y = torch.empty((N, C_out, P_H, P_W), device=x.device, dtype=x.dtype)

    # Strides
    x_sn, x_sc, x_sh, x_sw = x.stride()
    w_so, w_si, w_sh, w_sw = conv_weight.stride()
    y_sn, y_sc, y_sh, y_sw = y.stride()

    # Grid: one program per (n, oc, pooled_yx)
    grid = (N, C_out, P_H * P_W)

    # Launch kernel
    _conv_relu_maxpool2d_fused[grid](
        x, conv_weight, conv_bias, y,
        N, C_in, H, W, C_out,
        P_H, P_W,
        x_sn, x_sc, x_sh, x_sw,
        w_so, w_si, w_sh, w_sw,
        y_sn, y_sc, y_sh, y_sw,
        KH=KH, KW=KW, CIN=C_in,
        num_warps=4, num_stages=2,
    )

    return y