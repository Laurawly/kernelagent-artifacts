import torch
import triton
import triton.language as tl


@triton.jit
def _fused_bn_relu_conv1x1_avgpool2_kernel(
    x_ptr,                            # *bf16 [N, C_in, H, W]
    gamma_ptr, beta_ptr,              # *bf16 [C_in], [C_in]
    running_mean_ptr, running_var_ptr,# *bf16 [C_in], [C_in]
    w_ptr,                            # *bf16 [C_out, C_in, 1, 1] (OIHW)
    out_ptr,                          # *f32 [N, C_out, H_out, W_out]

    # sizes
    N, C_IN, H, W, C_OUT, H_OUT, W_OUT,

    # strides for x (NCHW)
    stride_xn, stride_xc, stride_xh, stride_xw,
    # strides for w (OIHW) - only first two matter for 1x1
    stride_wco, stride_wci,
    # strides for out (NCHW)
    stride_on, stride_oc, stride_oh, stride_ow,

    # compile-time constants
    EPS: tl.constexpr,                # batch-norm eps
    BLOCK_K: tl.constexpr,            # tile size along input channels
    BLOCK_CO: tl.constexpr,           # tile size along output channels
    NUM_CO_TILES: tl.constexpr        # number of tiles along C_out = ceil(C_OUT/BLOCK_CO)
):
    """
    Fused: BN (inference, eps=EPS, affine) -> ReLU -> Conv1x1 (no bias) -> AvgPool2d(2x2, stride=2)

    Reordering used for fusion:
      AvgPool2d and Conv1x1 commute since both are linear. We compute:
        pooled = mean_{(dh,dw) in 2x2} ReLU(BN(x[..., h+dh, w+dw]))
        y = W @ pooled
      which equals AvgPool2d(Conv1x1(ReLU(BN(x)))).

    Grid mapping:
      pid_w: [0..W_OUT), pid_h: [0..H_OUT), pid_n: [0..N)
      Each program computes all C_out for a single (n, ho, wo), tiled over C_out.
    """
    # Program ids
    pid_w = tl.program_id(axis=0)  # wo
    pid_h = tl.program_id(axis=1)  # ho
    pid_n = tl.program_id(axis=2)  # n

    # 2x2 pooling window top-left input coordinates
    ih0 = pid_h * 2
    iw0 = pid_w * 2

    # Base offsets
    x_n_off = pid_n * stride_xn
    out_base_off = pid_n * stride_on + pid_h * stride_oh + pid_w * stride_ow

    # Number of tiles along C_in
    num_k_tiles = tl.cdiv(C_IN, BLOCK_K)

    # Iterate over output-channel tiles
    for tile_idx in range(NUM_CO_TILES):
        co_start = tile_idx * BLOCK_CO
        offs_co = co_start + tl.arange(0, BLOCK_CO)
        mask_co = offs_co < C_OUT

        # Accumulator for this output tile
        acc = tl.zeros((BLOCK_CO,), dtype=tl.float32)

        # Reduction over input channels
        for kt in range(num_k_tiles):
            k0 = kt * BLOCK_K
            offs_k = k0 + tl.arange(0, BLOCK_K)
            mask_k = offs_k < C_IN

            # Load BN parameters (bf16) -> fp32
            gamma = tl.load(gamma_ptr + offs_k, mask=mask_k, other=0).to(tl.float32)
            beta = tl.load(beta_ptr + offs_k, mask=mask_k, other=0).to(tl.float32)
            rm = tl.load(running_mean_ptr + offs_k, mask=mask_k, other=0).to(tl.float32)
            rv = tl.load(running_var_ptr + offs_k, mask=mask_k, other=0).to(tl.float32)
            scale = gamma * tl.rsqrt(rv + EPS)
            bias = beta - rm * scale

            # Load the 2x2 patch for each channel in this tile
            base_c_offs = offs_k * stride_xc
            ptr00 = x_ptr + x_n_off + base_c_offs + ih0 * stride_xh + iw0 * stride_xw
            ptr01 = ptr00 + 1 * stride_xw
            ptr10 = ptr00 + 1 * stride_xh
            ptr11 = ptr10 + 1 * stride_xw

            x00 = tl.load(ptr00, mask=mask_k, other=0).to(tl.float32)
            x01 = tl.load(ptr01, mask=mask_k, other=0).to(tl.float32)
            x10 = tl.load(ptr10, mask=mask_k, other=0).to(tl.float32)
            x11 = tl.load(ptr11, mask=mask_k, other=0).to(tl.float32)

            # BN + ReLU
            y00 = tl.maximum(x00 * scale + bias, 0.0)
            y01 = tl.maximum(x01 * scale + bias, 0.0)
            y10 = tl.maximum(x10 * scale + bias, 0.0)
            y11 = tl.maximum(x11 * scale + bias, 0.0)

            # Average over 2x2 window: one value per input channel
            s_k = (y00 + y01 + y10 + y11) * 0.25

            # Load weight tile [BLOCK_K, BLOCK_CO] (bf16) -> fp32
            # w[co, ci] with strides [stride_wco, stride_wci]
            w_ptrs = w_ptr + (offs_co[None, :] * stride_wco + offs_k[:, None] * stride_wci)
            w_blk = tl.load(w_ptrs, mask=(mask_k[:, None] & mask_co[None, :]), other=0).to(tl.float32)

            # Accumulate: acc += sum_{ci}(s_k[ci] * w_blk[ci, co]) across ci
            acc += tl.sum(w_blk * s_k[:, None], axis=0)

        # Store results
        out_ptrs = out_ptr + out_base_off + offs_co * stride_oc
        tl.store(out_ptrs, acc, mask=mask_co)


def kernel_function(x, bn_weight, bn_bias, bn_running_mean, bn_running_var, conv_weight):
    """
    Fused BN -> ReLU -> Conv1x1 -> AvgPool2d(2x2, stride=2) implemented in a single Triton kernel.

    Runtime:
      - Validates shapes/devices
      - Allocates output tensor (fp32)
      - Launches Triton kernel
      No math is performed in Python; all compute is in the Triton kernel.
    """
    assert x.is_cuda, "x must be on CUDA"
    assert conv_weight.is_cuda, "conv_weight must be on CUDA"
    device = x.device

    # Validate shapes
    assert x.ndim == 4, "x must be NCHW"
    N, C_in, H, W = x.shape
    assert bn_weight.shape == (C_in,), "bn_weight shape mismatch"
    assert bn_bias.shape == (C_in,), "bn_bias shape mismatch"
    assert bn_running_mean.shape == (C_in,), "bn_running_mean shape mismatch"
    assert bn_running_var.shape == (C_in,), "bn_running_var shape mismatch"
    assert conv_weight.ndim == 4 and conv_weight.shape[2:] == (1, 1), "conv_weight must be [C_out, C_in, 1, 1]"
    C_out = conv_weight.shape[0]
    assert conv_weight.shape[1] == C_in, "conv_weight C_in mismatch"
    assert H % 2 == 0 and W % 2 == 0, "H and W must be even for 2x2 stride=2 pooling"

    # Output tensor (float32 to match reference)
    H_out, W_out = H // 2, W // 2
    out = torch.empty((N, C_out, H_out, W_out), device=device, dtype=torch.float32)

    # Launch configuration
    grid = (W_out, H_out, N)

    # Tunable tile sizes (powers of two)
    BLOCK_K = 64
    BLOCK_CO = 64
    NUM_CO_TILES = (C_out + BLOCK_CO - 1) // BLOCK_CO

    _fused_bn_relu_conv1x1_avgpool2_kernel[grid](
        x, bn_weight, bn_bias, bn_running_mean, bn_running_var, conv_weight, out,
        # sizes
        N, C_in, H, W, C_out, H_out, W_out,
        # x strides (NCHW)
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        # w strides (OIHW)
        conv_weight.stride(0), conv_weight.stride(1),
        # out strides (NCHW)
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        # constexpr/meta
        EPS=1e-5,
        BLOCK_K=BLOCK_K,
        BLOCK_CO=BLOCK_CO,
        NUM_CO_TILES=NUM_CO_TILES,
        num_warps=4,
        num_stages=2,
    )
    return out