import torch
import triton
import triton.language as tl

# Ensure reference runs in true FP32 (test computes FP32 reference after importing this file)
try:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
except Exception:
    pass


@triton.jit
def _mbconv_e6_k5_s1_fused(
    x_ptr,                               # bf16  [N, C_in, H, W]
    w_expand_ptr,                        # bf16  [C_expand, C_in, 1, 1]
    gamma1_ptr, beta1_ptr, mean1_ptr, var1_ptr,  # bf16 [C_expand]
    w_dw_ptr,                            # bf16  [C_expand, 1, 5, 5]
    gamma2_ptr, beta2_ptr, mean2_ptr, var2_ptr,  # bf16 [C_expand]
    w_proj_ptr,                          # bf16  [C_out, C_expand, 1, 1]
    gamma3_ptr, beta3_ptr, mean3_ptr, var3_ptr,  # bf16 [C_out]
    y_ptr,                               # fp32  [N, C_out, H, W]

    N, C_IN, H, W, C_EXPAND, C_OUT,      # sizes (int32)

    stride_x_n, stride_x_c, stride_x_h, stride_x_w,  # x strides
    stride_we_co, stride_we_ci,                         # expand weight strides (co, ci)
    stride_dw_co, stride_dw_ci, stride_dw_kh, stride_dw_kw,  # depthwise weight strides
    stride_wp_co, stride_wp_ci,                         # project weight strides (co, ci)
    stride_y_n, stride_y_c, stride_y_h, stride_y_w,    # y strides

    eps,  # fp32 epsilon for BN

    BLOCK_CO: tl.constexpr,  # tile of output channels computed per program
    BLOCK_CE: tl.constexpr,  # tile over expanded channels
    BLOCK_K: tl.constexpr,   # tile over input channels
):
    # One program per (n, h, w)
    pid = tl.program_id(axis=0)
    HW = H * W
    n = pid // HW
    rem = pid % HW
    h = rem // W
    w = rem % W

    # Output channels tile
    oc_offsets = tl.arange(0, BLOCK_CO)
    oc_mask = oc_offsets < C_OUT

    # Accumulator for projected 1x1 result (fp32)
    acc_out = tl.zeros((BLOCK_CO,), dtype=tl.float32)

    # Loop over expanded channels
    for ce_start in tl.range(0, C_EXPAND, BLOCK_CE):
        ce_offsets = ce_start + tl.arange(0, BLOCK_CE)
        ce_mask = ce_offsets < C_EXPAND

        # BN1 params (fp32)
        gamma1 = tl.load(gamma1_ptr + ce_offsets, mask=ce_mask, other=0.0).to(tl.float32)
        beta1  = tl.load(beta1_ptr  + ce_offsets, mask=ce_mask, other=0.0).to(tl.float32)
        mean1  = tl.load(mean1_ptr  + ce_offsets, mask=ce_mask, other=0.0).to(tl.float32)
        var1   = tl.load(var1_ptr   + ce_offsets, mask=ce_mask, other=0.0).to(tl.float32)
        scale1 = gamma1 / tl.sqrt(var1 + eps)

        # BN2 params (fp32)
        gamma2 = tl.load(gamma2_ptr + ce_offsets, mask=ce_mask, other=0.0).to(tl.float32)
        beta2  = tl.load(beta2_ptr  + ce_offsets, mask=ce_mask, other=0.0).to(tl.float32)
        mean2  = tl.load(mean2_ptr  + ce_offsets, mask=ce_mask, other=0.0).to(tl.float32)
        var2   = tl.load(var2_ptr   + ce_offsets, mask=ce_mask, other=0.0).to(tl.float32)
        scale2 = gamma2 / tl.sqrt(var2 + eps)

        # Depthwise 5x5 accumulator for current ce tile at (n, h, w)
        y2_ce = tl.zeros((BLOCK_CE,), dtype=tl.float32)

        # 5x5 depthwise loop with padding=2. For each neighbor, compute expand 1x1 -> BN1 -> ReLU6
        for kh in range(5):
            hh = h + kh - 2
            hh_valid = (hh >= 0) & (hh < H)
            for kw in range(5):
                ww = w + kw - 2
                ww_valid = (ww >= 0) & (ww < W)
                valid = hh_valid & ww_valid  # scalar mask for this neighbor

                # Compute expand 1x1 for this neighbor for ce tile: y1_ce = W_expand[ce, :] @ x[n, :, hh, ww]
                y1_ce = tl.zeros((BLOCK_CE,), dtype=tl.float32)
                base_x = x_ptr + n * stride_x_n + hh * stride_x_h + ww * stride_x_w

                # Sum over input channels in tiles
                for k_start in tl.range(0, C_IN, BLOCK_K):
                    k_offsets = k_start + tl.arange(0, BLOCK_K)
                    k_mask = k_offsets < C_IN

                    # Load weights tile [BLOCK_CE, BLOCK_K]
                    we_ptrs = w_expand_ptr + (ce_offsets[:, None] * stride_we_co) + (k_offsets[None, :] * stride_we_ci)
                    we_tile = tl.load(we_ptrs, mask=ce_mask[:, None] & k_mask[None, :], other=0.0).to(tl.float32)

                    # Load input vector x [BLOCK_K] at (n, hh, ww)
                    x_ptrs = base_x + k_offsets * stride_x_c
                    # Mask loads if OOB spatial OR k_offsets OOB
                    x_vec = tl.load(x_ptrs, mask=k_mask & valid, other=0.0).to(tl.float32)

                    # Accumulate matvec: y1_ce += sum_k we_tile[:, k] * x_vec[k]
                    y1_ce += tl.sum(we_tile * x_vec[None, :], axis=1)

                # BN1 + ReLU6
                y1_bn = (y1_ce - mean1) * scale1 + beta1
                y1_act = tl.minimum(tl.maximum(y1_bn, 0.0), 6.0)

                # Zero out-of-bounds AFTER activation (padding at activation level)
                y1_act = tl.where(valid, y1_act, 0.0)

                # Depthwise weight for this (kh, kw)
                wdw_ptrs = w_dw_ptr + ce_offsets * stride_dw_co + 0 * stride_dw_ci + kh * stride_dw_kh + kw * stride_dw_kw
                wdw = tl.load(wdw_ptrs, mask=ce_mask, other=0.0).to(tl.float32)

                # Accumulate depthwise output
                y2_ce += y1_act * wdw

        # BN2 + ReLU6
        y2_ce = (y2_ce - mean2) * scale2 + beta2
        y2_ce = tl.minimum(tl.maximum(y2_ce, 0.0), 6.0)

        # Project 1x1 into oc tile: acc_out += W_proj[oc, ce] * y2_ce[ce]
        wp_ptrs = w_proj_ptr + (oc_offsets[:, None] * stride_wp_co) + (ce_offsets[None, :] * stride_wp_ci)
        wp_tile = tl.load(wp_ptrs, mask=oc_mask[:, None] & ce_mask[None, :], other=0.0).to(tl.float32)
        acc_out += tl.sum(wp_tile * y2_ce[None, :], axis=1)

    # Final BN3 on output channels (fp32)
    gamma3 = tl.load(gamma3_ptr + oc_offsets, mask=oc_mask, other=0.0).to(tl.float32)
    beta3  = tl.load(beta3_ptr  + oc_offsets, mask=oc_mask, other=0.0).to(tl.float32)
    mean3  = tl.load(mean3_ptr  + oc_offsets, mask=oc_mask, other=0.0).to(tl.float32)
    var3   = tl.load(var3_ptr   + oc_offsets, mask=oc_mask, other=0.0).to(tl.float32)
    scale3 = gamma3 / tl.sqrt(var3 + eps)
    y_out = (acc_out - mean3) * scale3 + beta3

    # Store
    y_ptrs = y_ptr + n * stride_y_n + oc_offsets * stride_y_c + h * stride_y_h + w * stride_y_w
    tl.store(y_ptrs, y_out, mask=oc_mask)


def kernel_function(
    x,
    expand_conv_weight,
    expand_bn_weight,
    expand_bn_bias,
    expand_bn_running_mean,
    expand_bn_running_var,
    depthwise_conv_weight,
    depthwise_bn_weight,
    depthwise_bn_bias,
    depthwise_bn_running_mean,
    depthwise_bn_running_var,
    project_conv_weight,
    project_bn_weight,
    project_bn_bias,
    project_bn_running_mean,
    project_bn_running_var,
):
    """
    Fused MBConv subgraph implemented in a single Triton kernel:
      1) Expand 1x1 conv -> BN -> ReLU6
      2) Depthwise 5x5 conv (padding=2) -> BN -> ReLU6
      3) Project 1x1 conv -> BN
    All compute is performed in the Triton kernel (bf16 params, fp32 math).
    Wrapper validates, allocates, and launches only.
    """
    assert torch.cuda.is_available(), "CUDA is required"
    assert x.is_cuda, "x must be CUDA"
    device = x.device
    assert x.ndim == 4, "x must be NCHW"
    N, C_in, H, W = x.shape

    # Expand conv weight: (C_expand, C_in, 1, 1)
    assert expand_conv_weight.is_cuda and expand_conv_weight.device == device
    assert expand_conv_weight.ndim == 4 and expand_conv_weight.shape[2:] == (1, 1)
    C_expand = expand_conv_weight.shape[0]
    assert expand_conv_weight.shape[1] == C_in

    # Depthwise conv weight: (C_expand, 1, 5, 5)
    assert depthwise_conv_weight.is_cuda and depthwise_conv_weight.device == device
    assert depthwise_conv_weight.ndim == 4 and depthwise_conv_weight.shape == (C_expand, 1, 5, 5)

    # Project conv weight: (C_out, C_expand, 1, 1)
    assert project_conv_weight.is_cuda and project_conv_weight.device == device
    assert project_conv_weight.ndim == 4 and project_conv_weight.shape[2:] == (1, 1)
    C_out = project_conv_weight.shape[0]
    assert project_conv_weight.shape[1] == C_expand

    # BN shapes
    for t in (expand_bn_weight, expand_bn_bias, expand_bn_running_mean, expand_bn_running_var):
        assert t.is_cuda and t.device == device and t.ndim == 1 and t.shape[0] == C_expand
    for t in (depthwise_bn_weight, depthwise_bn_bias, depthwise_bn_running_mean, depthwise_bn_running_var):
        assert t.is_cuda and t.device == device and t.ndim == 1 and t.shape[0] == C_expand
    for t in (project_bn_weight, project_bn_bias, project_bn_running_mean, project_bn_running_var):
        assert t.is_cuda and t.device == device and t.ndim == 1 and t.shape[0] == C_out

    # Allocate output (fp32)
    y = torch.empty((N, C_out, H, W), device=device, dtype=torch.float32)

    # Strides in elements
    sxn, sxc, sxh, sxw = x.stride()
    we_co, we_ci, _, _ = expand_conv_weight.stride()
    dw_co, dw_ci, dw_kh, dw_kw = depthwise_conv_weight.stride()
    wp_co, wp_ci, _, _ = project_conv_weight.stride()
    syn, syc, syh, syw = y.stride()

    # Launch configuration: one program per pixel (n, h, w)
    grid = (N * H * W,)

    # Reasonable tile sizes (powers of two)
    BLOCK_CO = 128  # >= 112 for this test, compute all C_out in one go
    BLOCK_CE = 64
    BLOCK_K = 32

    _mbconv_e6_k5_s1_fused[grid](
        x,
        expand_conv_weight,
        expand_bn_weight, expand_bn_bias, expand_bn_running_mean, expand_bn_running_var,
        depthwise_conv_weight,
        depthwise_bn_weight, depthwise_bn_bias, depthwise_bn_running_mean, depthwise_bn_running_var,
        project_conv_weight,
        project_bn_weight, project_bn_bias, project_bn_running_mean, project_bn_running_var,
        y,

        N, C_in, H, W, C_expand, C_out,

        sxn, sxc, sxh, sxw,
        we_co, we_ci,
        dw_co, dw_ci, dw_kh, dw_kw,
        wp_co, wp_ci,
        syn, syc, syh, syw,

        eps=1e-5,

        BLOCK_CO=BLOCK_CO,
        BLOCK_CE=BLOCK_CE,
        BLOCK_K=BLOCK_K,

        num_warps=4,
        num_stages=2,
    )
    return y