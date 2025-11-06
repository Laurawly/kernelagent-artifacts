import torch
import triton
import triton.language as tl


# ------------------------------
# Utility kernels
# ------------------------------

@triton.jit
def _copy_cast_flat(dst_ptr, src_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    Generic flat copy+cast kernel.
    Loads from src_ptr, casts to dst element type, stores to dst_ptr.
    """
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(src_ptr + offs, mask=mask, other=0)
    x = x.to(dst_ptr.dtype.element_ty)
    tl.store(dst_ptr + offs, x, mask=mask)


@triton.jit
def _copy_cast_slice_nchw(dst_ptr, src_ptr,
                          N, C, H, W,
                          dst_stride_n, dst_stride_c, dst_stride_h, dst_stride_w,
                          src_stride_n, src_stride_c, src_stride_h, src_stride_w,
                          BLOCK_SIZE: tl.constexpr):
    """
    Copy + cast a 4D NCHW slice [N, C, H, W] from src to dst.
    """
    n_elems = N * C * H * W
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elems

    W_i = W
    H_i = H
    C_i = C

    tmp = offs // W_i
    w = offs % W_i
    tmp2 = tmp // H_i
    h = tmp % H_i
    n = tmp2 // C_i
    c = tmp2 % C_i

    src_ptrs = src_ptr + n * src_stride_n + c * src_stride_c + h * src_stride_h + w * src_stride_w
    dst_ptrs = dst_ptr + n * dst_stride_n + c * dst_stride_c + h * dst_stride_h + w * dst_stride_w

    vals = tl.load(src_ptrs, mask=mask, other=0)
    vals = vals.to(dst_ptr.dtype.element_ty)
    tl.store(dst_ptrs, vals, mask=mask)


# ------------------------------
# Fused Dense layer kernel
# ------------------------------

@triton.jit
def _dense_layer_kernel(x_ptr,                # float32 [N, C_total, H, W] workspace (read first C_IN channels)
                        y_ptr,                # float32 base pointer to full [N, C_total, H, W]; write starts at C_BASE
                        w_ptr,                # bf16/fp16/fp32 weights [C_OUT, C_IN, 3, 3]
                        gamma_ptr, beta_ptr, mean_ptr, var_ptr,  # bf16/fp16/fp32 BN params [C_IN]
                        N, C_IN, C_BASE, H, W,
                        stride_x_n, stride_x_c, stride_x_h, stride_x_w,
                        stride_y_n, stride_y_c, stride_y_h, stride_y_w,
                        stride_w_oc, stride_w_ic, stride_w_kh, stride_w_kw,
                        eps,
                        BLOCK_HW: tl.constexpr,
                        BLOCK_OC: tl.constexpr,
                        BLOCK_C: tl.constexpr,
                        C_OUT: tl.constexpr):
    """
    Fused BN(inference) -> ReLU -> Conv2d(3x3, stride=1, padding=1, no bias)
    Implements one DenseNet layer producing C_OUT=32 new channels appended to current features.
    Computes:
        acc[oc, hw] = sum_{ci, kh, kw} W[oc, ci, kh, kw] * ReLU(BN(X[n, ci, h+kh-1, w+kw-1]))
    with zero-padding for OOB spatial accesses (i.e., padded values are zeros after BN+ReLU).
    All math in float32; loads from lower-precision inputs are cast to float32.
    """
    pid_n = tl.program_id(axis=0)
    pid_oc = tl.program_id(axis=1)
    pid_hw = tl.program_id(axis=2)

    M = H * W
    start_hw = pid_hw * BLOCK_HW
    offs_hw = start_hw + tl.arange(0, BLOCK_HW)
    mask_hw = offs_hw < M
    # recover h, w for current tile
    W_i = W
    h = offs_hw // W_i
    w = offs_hw % W_i

    # output-channel tile
    oc_start = pid_oc * BLOCK_OC
    offs_oc = oc_start + tl.arange(0, BLOCK_OC)
    oc_mask = offs_oc < C_OUT

    # base pointers for this batch n
    base_x_n = x_ptr + pid_n * stride_x_n
    base_y_n = y_ptr + pid_n * stride_y_n

    # accumulator in fp32
    acc = tl.zeros((BLOCK_OC, BLOCK_HW), dtype=tl.float32)

    # iterate over input channels in tiles of BLOCK_C
    for ci0 in tl.range(0, C_IN, BLOCK_C):
        offs_ci = ci0 + tl.arange(0, BLOCK_C)
        ci_mask = offs_ci < C_IN

        # load BN params for this ci-tile
        gamma = tl.load(gamma_ptr + offs_ci, mask=ci_mask, other=0.0).to(tl.float32)
        beta = tl.load(beta_ptr + offs_ci, mask=ci_mask, other=0.0).to(tl.float32)
        run_mean = tl.load(mean_ptr + offs_ci, mask=ci_mask, other=0.0).to(tl.float32)
        run_var = tl.load(var_ptr + offs_ci, mask=ci_mask, other=0.0).to(tl.float32)
        inv_std = 1.0 / tl.sqrt(run_var + eps)
        scale = gamma * inv_std
        # beta - mean * scale
        shift = beta - run_mean * scale

        # 3x3 kernel positions
        for kh in range(0, 3):
            for kw in range(0, 3):
                in_h = h + (kh - 1)
                in_w = w + (kw - 1)
                in_bounds = mask_hw & (in_h >= 0) & (in_h < H) & (in_w >= 0) & (in_w < W)

                # input tile X: [BLOCK_C, BLOCK_HW]
                x_ptrs = base_x_n + offs_ci[:, None] * stride_x_c + in_h[None, :] * stride_x_h + in_w[None, :] * stride_x_w
                x_mask = ci_mask[:, None] & in_bounds[None, :]
                x_vals = tl.load(x_ptrs, mask=x_mask, other=0.0).to(tl.float32)

                # BN + ReLU
                y_vals = scale[:, None] * x_vals + shift[:, None]
                y_vals = tl.maximum(y_vals, 0.0)
                # ensure OOB spatial contributes 0 (padding after activation)
                y_vals = tl.where(in_bounds[None, :], y_vals, 0.0)
                # ensure masked channels contribute 0
                y_vals = tl.where(ci_mask[:, None], y_vals, 0.0)

                # weights tile W: [BLOCK_OC, BLOCK_C] for (kh,kw)
                w_ptrs = w_ptr + offs_oc[:, None] * stride_w_oc + offs_ci[None, :] * stride_w_ic + kh * stride_w_kh + kw * stride_w_kw
                w_mask = oc_mask[:, None] & ci_mask[None, :]
                w_vals = tl.load(w_ptrs, mask=w_mask, other=0.0).to(tl.float32)

                # GEMM-style accumulation
                acc = tl.dot(w_vals, y_vals, acc)

    # store results to y buffer at [n, C_BASE + oc, h, w]
    y_ptrs = base_y_n + (C_BASE + offs_oc)[:, None] * stride_y_c + h[None, :] * stride_y_h + w[None, :] * stride_y_w
    store_mask = oc_mask[:, None] & mask_hw[None, :]
    tl.store(y_ptrs, acc, mask=store_mask)


# ------------------------------
# Wrapper
# ------------------------------

def kernel_function(x,
                    conv_weights,
                    bn_weight,
                    bn_bias,
                    bn_running_mean,
                    bn_running_var,
                    dropout_p=0.0,
                    eps=1e-5):
    """
    DenseNet-like dense block:
    Repeats 24 times: [BatchNorm2d(inference) -> ReLU -> Conv2d 3x3, stride=1, padding=1, no bias] then concatenate along channels.
    Layout: NCHW
    Input:  [N, 256, 14, 14] (typically bfloat16)
    Output: [N, 1024, 14, 14] (same dtype as input)

    Fusion inside each layer:
      - BN (inference) -> ReLU -> Conv3x3 (no bias) fused in one Triton kernel
      - Concatenation implemented by writing into the correct slice of a pre-allocated workspace
    All math is performed in float32. Inputs/params may be bf16/fp16/fp32 and are cast on load.

    Note:
      - The wrapper performs only allocation, validation, and kernel launches (no PyTorch compute ops).
      - We avoid Python-side pointer arithmetic; all addressing is done inside Triton kernels.
    """
    # Validate inputs and params
    assert isinstance(conv_weights, (list, tuple)) and len(conv_weights) == 24
    assert all(isinstance(lst, (list, tuple)) and len(lst) == 24
               for lst in [bn_weight, bn_bias, bn_running_mean, bn_running_var]), "BN param lists must be length 24"
    assert x.is_cuda, "Input must be CUDA tensor"
    assert x.ndim == 4, "Input must be NCHW"

    N, C0, H, W = x.shape
    growth = 32
    layers = 24
    C_total = C0 + growth * layers
    device = x.device
    dtype_in = x.dtype
    assert dtype_in in (torch.bfloat16, torch.float16, torch.float32)

    for l in range(layers):
        C_in = C0 + l * growth
        w = conv_weights[l]
        assert w.is_cuda and w.ndim == 4 and w.shape[0] == growth and w.shape[1] == C_in and tuple(w.shape[2:]) == (3, 3), \
            f"conv_weights[{l}] must be [32,{C_in},3,3] on CUDA"
        assert bn_weight[l].numel() == C_in and bn_bias[l].numel() == C_in and \
               bn_running_mean[l].numel() == C_in and bn_running_var[l].numel() == C_in, \
               f"BN params for layer {l} must have length {C_in}"
        assert bn_weight[l].device == device and bn_bias[l].device == device and \
               bn_running_mean[l].device == device and bn_running_var[l].device == device, \
               f"BN params for layer {l} must be on the same device as input"

    # Allocate float32 workspace and copy input slice
    out_f32 = torch.empty((N, C_total, H, W), device=device, dtype=torch.float32)

    n_elem_in = N * C0 * H * W
    BLOCK = 1024
    grid_copy_in = (triton.cdiv(n_elem_in, BLOCK),)
    _copy_cast_slice_nchw[grid_copy_in](
        out_f32, x,
        N, C0, H, W,
        out_f32.stride(0), out_f32.stride(1), out_f32.stride(2), out_f32.stride(3),
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        BLOCK_SIZE=BLOCK,
    )

    # Kernel tiling parameters
    BLOCK_HW = 64
    BLOCK_OC = 32   # write all 32 output channels in one go
    BLOCK_C = 32    # input-channel tile
    C_OUT = growth  # 32

    # Run 24 fused layers sequentially
    for l in range(layers):
        C_in = C0 + l * growth

        w = conv_weights[l]
        g = bn_weight[l]
        b = bn_bias[l]
        rm = bn_running_mean[l]
        rv = bn_running_var[l]

        # Strides
        sxn, sxc, sxh, sxw = out_f32.stride(0), out_f32.stride(1), out_f32.stride(2), out_f32.stride(3)
        syn, syc, syh, syw = out_f32.stride(0), out_f32.stride(1), out_f32.stride(2), out_f32.stride(3)
        swoc, swic, swkh, swkw = w.stride(0), w.stride(1), w.stride(2), w.stride(3)

        # Grid: (N, 1, ceil_div(H*W, BLOCK_HW))
        grid = (N, triton.cdiv(C_OUT, BLOCK_OC), triton.cdiv(H * W, BLOCK_HW))
        _dense_layer_kernel[grid](
            out_f32,                             # x_ptr: read from workspace
            out_f32,                             # y_ptr: base workspace; C_BASE selects the write offset
            w,                                   # weights
            g, b, rm, rv,                        # BN params
            N, C_in, C_in, H, W,                 # C_BASE = C_in (write starting channel)
            sxn, sxc, sxh, sxw,
            syn, syc, syh, syw,
            swoc, swic, swkh, swkw,
            eps,
            BLOCK_HW=BLOCK_HW,
            BLOCK_OC=BLOCK_OC,
            BLOCK_C=BLOCK_C,
            C_OUT=C_OUT,
        )

    # Cast final workspace to original dtype
    result = torch.empty((N, C_total, H, W), device=device, dtype=dtype_in)
    n_elem_out = N * C_total * H * W
    grid_cast_out = (triton.cdiv(n_elem_out, BLOCK),)
    _copy_cast_flat[grid_cast_out](result, out_f32, n_elem_out, BLOCK_SIZE=BLOCK)

    return result