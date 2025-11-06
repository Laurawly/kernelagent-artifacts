import torch
import triton
import triton.language as tl


# -----------------------------------------------------------------------------
# Fused Conv7x7(s=2,p=3) + BN (inference) + ReLU
# Produces activation of shape (N, C_out, 112, 112) from input (N, 3, 224, 224)
# -----------------------------------------------------------------------------
@triton.jit
def _conv_bn_relu_7x7s2_kernel(
    x_ptr,                # *bfloat16 or *float16 or *float32, NCHW input
    w_ptr,                # *bfloat16 or *float16 or *float32, OIHW weights
    gamma_ptr, beta_ptr,  # *bfloat16/float16/float32, BN weight/bias
    mean_ptr, var_ptr,    # *bfloat16/float16/float32, BN running stats
    y_ptr,                # *float32, output activation after BN+ReLU
    N, C_in, H, W,        # input dims
    C_out, H_out, W_out,  # output dims (112, 112)
    stride_in_n, stride_in_c, stride_in_h, stride_in_w,         # input strides
    stride_w_o, stride_w_i, stride_w_h, stride_w_w,             # weight strides
    stride_y_n, stride_y_c, stride_y_h, stride_y_w,             # output (activation) strides
    EPS: tl.constexpr,                                          # BN eps
    BLOCK_CO: tl.constexpr,                                     # out-channels per program
    BLOCK_HW: tl.constexpr,                                     # HW-out elements per program
):
    # Program IDs
    pid_hw = tl.program_id(axis=0)  # tile id across flattened H_out*W_out
    pid_n  = tl.program_id(axis=1)  # batch
    pid_co = tl.program_id(axis=2)  # out-channel tile

    # Offsets in channel and flattened HW
    co_start = pid_co * BLOCK_CO
    co_offsets = co_start + tl.arange(0, BLOCK_CO)
    mask_co = co_offsets < C_out

    hw_start = pid_hw * BLOCK_HW
    hw_offsets = hw_start + tl.arange(0, BLOCK_HW)
    mask_hw = hw_offsets < (H_out * W_out)

    # Map flattened offsets to (oh, ow)
    ow = hw_offsets % W_out
    oh = hw_offsets // W_out

    # Compute base input coords given conv stride=2, pad=3, kernel=7
    # input coords for a given kernel (ky, kx): y = oh*2 + ky - 3; x = ow*2 + kx - 3
    # Precompute (oh*2 - 3), (ow*2 - 3)
    in_y_base = oh * 2 - 3
    in_x_base = ow * 2 - 3

    # Accumulator for convolution: shape (BLOCK_CO, BLOCK_HW)
    acc = tl.zeros((BLOCK_CO, BLOCK_HW), dtype=tl.float32)

    # Base pointers: input and (batch/channel) offsets
    x_batch_ptr = x_ptr + pid_n * stride_in_n

    # Loop over input channels and 7x7 kernel
    for ci in range(0, C_in):
        x_c_ptr = x_batch_ptr + ci * stride_in_c
        for ky in range(0, 7):
            y_coords = in_y_base + ky
            y_inbounds = (y_coords >= 0) & (y_coords < H) & mask_hw
            y_offsets = y_coords * stride_in_h
            for kx in range(0, 7):
                x_coords = in_x_base + kx
                in_bounds = y_inbounds & (x_coords >= 0) & (x_coords < W)
                x_offsets = x_coords
                # Input pointers for current (ci, ky, kx) across BLOCK_HW positions
                x_ptrs = x_c_ptr + y_offsets + x_offsets
                # Load input vector (BLOCK_HW,)
                x_vals = tl.load(x_ptrs, mask=in_bounds, other=0.0).to(tl.float32)

                # Load weights for (co_offsets, ci, ky, kx)
                w_ptrs = w_ptr + co_offsets * stride_w_o + ci * stride_w_i + ky * stride_w_h + kx * stride_w_w
                w_vals = tl.load(w_ptrs, mask=mask_co, other=0.0).to(tl.float32)

                # Outer-product accumulation: (BLOCK_CO, BLOCK_HW)
                acc += w_vals[:, None] * x_vals[None, :]

    # BatchNorm inference and ReLU
    # Load BN params for co tile
    gamma = tl.load(gamma_ptr + co_offsets, mask=mask_co, other=0.0).to(tl.float32)
    beta  = tl.load(beta_ptr  + co_offsets, mask=mask_co, other=0.0).to(tl.float32)
    mean  = tl.load(mean_ptr  + co_offsets, mask=mask_co, other=0.0).to(tl.float32)
    var   = tl.load(var_ptr   + co_offsets, mask=mask_co, other=0.0).to(tl.float32)

    inv_std = 1.0 / tl.sqrt(var + EPS)

    # Normalize, scale/shift
    acc = (acc - mean[:, None]) * inv_std[:, None]
    acc = acc * gamma[:, None] + beta[:, None]
    # ReLU
    acc = tl.maximum(acc, 0.0)

    # Store to activation tensor y_ptr (float32)
    y_base_ptr = y_ptr + pid_n * stride_y_n
    y_ptrs = y_base_ptr + co_offsets[:, None] * stride_y_c + oh[None, :] * stride_y_h + ow[None, :] * stride_y_w
    tl.store(y_ptrs, acc, mask=(mask_co[:, None] & mask_hw[None, :]))


# -----------------------------------------------------------------------------
# MaxPool 3x3 stride=2 pad=1 over activations (N, C_out, 112, 112) -> (N, C_out, 56, 56)
# -----------------------------------------------------------------------------
@triton.jit
def _maxpool3x3_s2_p1_kernel(
    a_ptr,                 # *float32 activation after BN+ReLU (N, C, 112, 112)
    o_ptr,                 # *float32 output (N, C, 56, 56)
    N, C, H_in, W_in,      # activation dims (112, 112)
    H_out, W_out,          # pooled dims (56, 56)
    stride_a_n, stride_a_c, stride_a_h, stride_a_w,   # activation strides
    stride_o_n, stride_o_c, stride_o_h, stride_o_w,   # output strides
    BLOCK_CO: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    pid_hw = tl.program_id(axis=0)  # tile across flattened H_out*W_out
    pid_n  = tl.program_id(axis=1)  # batch
    pid_c  = tl.program_id(axis=2)  # channel tile

    co_start = pid_c * BLOCK_CO
    co_offsets = co_start + tl.arange(0, BLOCK_CO)
    mask_c = co_offsets < C

    hw_start = pid_hw * BLOCK_HW
    hw_offsets = hw_start + tl.arange(0, BLOCK_HW)
    mask_hw = hw_offsets < (H_out * W_out)

    ow = hw_offsets % W_out
    oh = hw_offsets // W_out

    # Pool window start (with padding=1)
    y0 = oh * 2 - 1
    x0 = ow * 2 - 1

    # Initialize with -inf
    neg_inf = -float("inf")
    acc = tl.zeros((BLOCK_CO, BLOCK_HW), dtype=tl.float32) + neg_inf

    a_base_ptr = a_ptr + pid_n * stride_a_n

    # Iterate over 3x3 window
    for dy in range(0, 3):
        y = y0 + dy
        y_valid = (y >= 0) & (y < H_in) & mask_hw
        y_offs = y * stride_a_h
        for dx in range(0, 3):
            x = x0 + dx
            valid = y_valid & (x >= 0) & (x < W_in)
            x_offs = x
            a_ptrs = a_base_ptr + co_offsets[:, None] * stride_a_c + y_offs[None, :] + x_offs[None, :]
            vals = tl.load(a_ptrs, mask=(mask_c[:, None] & valid[None, :]), other=neg_inf)
            acc = tl.maximum(acc, vals)

    o_base_ptr = o_ptr + pid_n * stride_o_n
    o_ptrs = o_base_ptr + co_offsets[:, None] * stride_o_c + oh[None, :] * stride_o_h + ow[None, :] * stride_o_w
    tl.store(o_ptrs, acc, mask=(mask_c[:, None] & mask_hw[None, :]))


def kernel_function(x, conv_weight, bn_weight, bn_bias, bn_running_mean, bn_running_var):
    """
    Fused Triton implementation for:
      Conv2d(7x7, stride=2, padding=3) -> BatchNorm2d (inference, eps=1e-5) -> ReLU -> MaxPool2d(3x3, stride=2, pad=1)
    for NCHW input with C_in=3 and C_out arbitrary (here test uses 64).

    Fusion strategy:
      - Kernel 1 fuses Conv + BN (using running stats) + ReLU in one pass, writing (N, C_out, 112, 112) activations.
      - Kernel 2 performs MaxPool 3x3 s=2 p=1 over the activations, producing final (N, C_out, 56, 56).
    Rationale: Conv+BN+ReLU fusion avoids an extra round-trip to memory for BN/ReLU. Pooling is kept in a second kernel
    for simplicity and to avoid recomputing nine neighboring convolutions per pooled output.

    Notes:
      - All math is implemented in Triton kernels. The wrapper does only validation, allocation, and kernel launch.
      - Accumulation and BN are done in float32 for numerical stability. Final output is float32.
      - Inputs and parameters may be bf16/fp16/fp32; they are cast to fp32 inside kernels as needed.

    Args:
      x:               (N, C_in=3, 224, 224)
      conv_weight:     (C_out, C_in=3, 7, 7)
      bn_weight:       (C_out,)
      bn_bias:         (C_out,)
      bn_running_mean: (C_out,)
      bn_running_var:  (C_out,)

    Returns:
      Tensor (N, C_out, 56, 56) on same device as x, dtype=torch.float32
    """
    # Basic validation
    assert isinstance(x, torch.Tensor) and isinstance(conv_weight, torch.Tensor)
    assert isinstance(bn_weight, torch.Tensor) and isinstance(bn_bias, torch.Tensor)
    assert isinstance(bn_running_mean, torch.Tensor) and isinstance(bn_running_var, torch.Tensor)
    assert x.ndim == 4 and x.shape[1] == 3, "Expected NCHW input with C_in=3"
    assert conv_weight.ndim == 4 and conv_weight.shape[1] == 3 and conv_weight.shape[2:] == (7, 7), "Expected OIHW weights with I=3, K=7x7"
    assert bn_weight.shape == bn_bias.shape == bn_running_mean.shape == bn_running_var.shape == (conv_weight.shape[0],), "BN params must match C_out"
    assert x.device == conv_weight.device == bn_weight.device == bn_bias.device == bn_running_mean.device == bn_running_var.device, "All tensors must be on the same device"
    device = x.device
    if device.type != "cuda":
        raise RuntimeError("This Triton kernel requires a CUDA device.")

    N, C_in, H, W = x.shape
    C_out = conv_weight.shape[0]
    stride = 2
    pad = 3
    KH = KW = 7
    # Conv output dims
    H_out = (H + 2 * pad - KH) // stride + 1
    W_out = (W + 2 * pad - KW) // stride + 1
    assert H_out == 112 and W_out == 112, "Expected 112x112 conv output for 224x224 input with 7x7 s=2 p=3"

    # MaxPool dims
    pool_k = 3
    pool_stride = 2
    pool_pad = 1
    HP = (H_out + 2 * pool_pad - pool_k) // pool_stride + 1
    WP = (W_out + 2 * pool_pad - pool_k) // pool_stride + 1
    assert HP == 56 and WP == 56, "Expected 56x56 pooled output"

    # Allocate intermediate and output tensors (float32 for stability)
    act = torch.empty((N, C_out, H_out, W_out), device=device, dtype=torch.float32)
    out = torch.empty((N, C_out, HP, WP), device=device, dtype=torch.float32)

    # Strides
    s_in_n, s_in_c, s_in_h, s_in_w = x.stride()
    s_w_o, s_w_i, s_w_h, s_w_w = conv_weight.stride()
    s_y_n, s_y_c, s_y_h, s_y_w = act.stride()
    s_a_n, s_a_c, s_a_h, s_a_w = act.stride()
    s_o_n, s_o_c, s_o_h, s_o_w = out.stride()

    # Kernel launch configuration
    # Tiling choices are empirically reasonable for this workload
    BLOCK_CO = 32  # 2 tiles for 64 out-channels
    BLOCK_HW = 64  # 12544 / 64 = 196 tiles along HW

    grid_conv = (
        triton.cdiv(H_out * W_out, BLOCK_HW),  # hw tiles
        N,                                     # batches
        triton.cdiv(C_out, BLOCK_CO),          # out-channel tiles
    )
    _conv_bn_relu_7x7s2_kernel[grid_conv](
        x, conv_weight,
        bn_weight, bn_bias,
        bn_running_mean, bn_running_var,
        act,
        N, C_in, H, W,
        C_out, H_out, W_out,
        s_in_n, s_in_c, s_in_h, s_in_w,
        s_w_o, s_w_i, s_w_h, s_w_w,
        s_y_n, s_y_c, s_y_h, s_y_w,
        EPS=1e-5,
        BLOCK_CO=BLOCK_CO,
        BLOCK_HW=BLOCK_HW,
        num_warps=4, num_stages=2,
    )

    grid_pool = (
        triton.cdiv(HP * WP, BLOCK_HW),
        N,
        triton.cdiv(C_out, BLOCK_CO),
    )
    _maxpool3x3_s2_p1_kernel[grid_pool](
        act, out,
        N, C_out, H_out, W_out,
        HP, WP,
        s_a_n, s_a_c, s_a_h, s_a_w,
        s_o_n, s_o_c, s_o_h, s_o_w,
        BLOCK_CO=BLOCK_CO,
        BLOCK_HW=BLOCK_HW,
        num_warps=4, num_stages=2,
    )

    return out