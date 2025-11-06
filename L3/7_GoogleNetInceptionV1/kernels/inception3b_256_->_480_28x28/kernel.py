import torch
import triton
import triton.language as tl


@triton.jit
def _conv1x1_kernel(
    x_ptr,          # *[N, C_IN, H, W]
    w_ptr,          # *[C_OUT, C_IN, 1, 1]
    b_ptr,          # *[C_OUT]
    y_ptr,          # *[N, C_OUT_TOTAL, H, W] (or intermediate when CO_OFFSET=0 and C_OUT_TOTAL=C_OUT)
    N, C_IN, H, W, C_OUT, CO_OFFSET,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wn, stride_wc, stride_wk, stride_wl,  # weight strides: co, ci, kh, kw
    stride_yn, stride_yc, stride_yh, stride_yw,
    BLOCK_OC: tl.constexpr,
    BLOCK_PX: tl.constexpr,
):
    # Block indices
    pid_oc = tl.program_id(axis=0)
    pid_px = tl.program_id(axis=1)

    oc_start = pid_oc * BLOCK_OC
    px_start = pid_px * BLOCK_PX

    oc_offsets = oc_start + tl.arange(0, BLOCK_OC)
    px_offsets = px_start + tl.arange(0, BLOCK_PX)

    mask_oc = oc_offsets < C_OUT
    mask_px = px_offsets < (N * H * W)

    # Compute (n, y, x) for each pixel index in the tile
    HW = H * W
    n_idx = px_offsets // HW
    rem = px_offsets % HW
    y_idx = rem // W
    x_idx = rem % W

    # Base input pointers for each pixel (without channel contribution)
    x_base_ptrs = x_ptr + n_idx * stride_xn + y_idx * stride_xh + x_idx * stride_xw

    # Accumulator [BLOCK_PX, BLOCK_OC] in fp32
    acc = tl.zeros((BLOCK_PX, BLOCK_OC), dtype=tl.float32)

    # Loop over input channels
    ci = 0
    while ci < C_IN:
        # Load input values for each pixel at this channel
        x_ptrs_ci = x_base_ptrs + ci * stride_xc
        x_vals = tl.load(x_ptrs_ci, mask=mask_px, other=0.0).to(tl.float32)  # [BLOCK_PX]

        # Load weights for this channel across OC tile
        w_ptrs_ci = w_ptr + oc_offsets * stride_wn + ci * stride_wc  # kh=0, kw=0 for 1x1
        w_vals = tl.load(w_ptrs_ci, mask=mask_oc, other=0.0).to(tl.float32)  # [BLOCK_OC]

        # FMA: broadcast
        acc += x_vals[:, None] * w_vals[None, :]

        ci += 1

    # Add bias
    b_vals = tl.load(b_ptr + oc_offsets, mask=mask_oc, other=0.0).to(tl.float32)
    acc += b_vals[None, :]

    # Store to output with output channel offset
    oc_global = CO_OFFSET + oc_offsets
    y_base_px = y_ptr + n_idx * stride_yn + y_idx * stride_yh + x_idx * stride_yw  # [BLOCK_PX]
    y_ptrs = y_base_px[:, None] + oc_global[None, :] * stride_yc  # [BLOCK_PX, BLOCK_OC]
    out_mask = mask_px[:, None] & mask_oc[None, :]
    tl.store(y_ptrs, acc.to(y_ptr.dtype.element_ty), mask=out_mask)


@triton.jit
def _convk_kernel(
    x_ptr,          # *[N, C_IN, H, W]
    w_ptr,          # *[C_OUT, C_IN, K, K]
    b_ptr,          # *[C_OUT]
    y_ptr,          # *[N, C_OUT_TOTAL, H, W]
    N, C_IN, H, W, C_OUT, CO_OFFSET,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wn, stride_wc, stride_wk, stride_wl,
    stride_yn, stride_yc, stride_yh, stride_yw,
    K: tl.constexpr,
    PAD: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_PX: tl.constexpr,
):
    pid_oc = tl.program_id(axis=0)
    pid_px = tl.program_id(axis=1)

    oc_start = pid_oc * BLOCK_OC
    px_start = pid_px * BLOCK_PX

    oc_offsets = oc_start + tl.arange(0, BLOCK_OC)
    px_offsets = px_start + tl.arange(0, BLOCK_PX)

    mask_oc = oc_offsets < C_OUT
    mask_px = px_offsets < (N * H * W)

    HW = H * W
    n_idx = px_offsets // HW
    rem = px_offsets % HW
    y_idx = rem // W
    x_idx = rem % W

    # Accumulator [BLOCK_PX, BLOCK_OC]
    acc = tl.zeros((BLOCK_PX, BLOCK_OC), dtype=tl.float32)

    # Convolution loops: over spatial kernel (ky, kx) and over channels
    ky = 0
    while ky < K:
        kx = 0
        while kx < K:
            y_nei = y_idx + ky - PAD
            x_nei = x_idx + kx - PAD
            in_bounds = (y_nei >= 0) & (y_nei < H) & (x_nei >= 0) & (x_nei < W)
            x_base_ptrs = x_ptr + n_idx * stride_xn + y_nei * stride_xh + x_nei * stride_xw
            ci = 0
            while ci < C_IN:
                # load input tile values at this (ky,kx,ci)
                x_ptrs_ci = x_base_ptrs + ci * stride_xc
                # zero padding for out-of-bounds pixels
                x_vals = tl.load(x_ptrs_ci, mask=(mask_px & in_bounds), other=0.0).to(tl.float32)  # [BLOCK_PX]
                # load weights across OC tile for this (ky,kx,ci)
                w_ptrs = w_ptr + oc_offsets * stride_wn + ci * stride_wc + ky * stride_wk + kx * stride_wl
                w_vals = tl.load(w_ptrs, mask=mask_oc, other=0.0).to(tl.float32)  # [BLOCK_OC]
                # FMA
                acc += x_vals[:, None] * w_vals[None, :]
                ci += 1
            kx += 1
        ky += 1

    # Add bias
    b_vals = tl.load(b_ptr + oc_offsets, mask=mask_oc, other=0.0).to(tl.float32)
    acc += b_vals[None, :]

    # Store to output with channel offset
    oc_global = CO_OFFSET + oc_offsets
    y_base_px = y_ptr + n_idx * stride_yn + y_idx * stride_yh + x_idx * stride_yw
    y_ptrs = y_base_px[:, None] + oc_global[None, :] * stride_yc
    out_mask = mask_px[:, None] & mask_oc[None, :]
    tl.store(y_ptrs, acc.to(y_ptr.dtype.element_ty), mask=out_mask)


@triton.jit
def _maxpool3x3_kernel(
    x_ptr,      # *[N, C, H, W]
    y_ptr,      # *[N, C, H, W]
    N, C, H, W,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_yn, stride_yc, stride_yh, stride_yw,
    BLOCK_C: tl.constexpr,
    BLOCK_PX: tl.constexpr,
):
    pid_c = tl.program_id(axis=0)
    pid_px = tl.program_id(axis=1)

    c_start = pid_c * BLOCK_C
    px_start = pid_px * BLOCK_PX

    c_offsets = c_start + tl.arange(0, BLOCK_C)
    px_offsets = px_start + tl.arange(0, BLOCK_PX)

    mask_c = c_offsets < C
    mask_px = px_offsets < (N * H * W)

    HW = H * W
    n_idx = px_offsets // HW
    rem = px_offsets % HW
    y_idx = rem // W
    x_idx = rem % W

    # Initialize max values in fp32 to -inf
    neg_inf = -float("inf")
    max_vals = tl.full((BLOCK_PX, BLOCK_C), neg_inf, dtype=tl.float32)

    ky = 0
    while ky < 3:
        kx = 0
        while kx < 3:
            y_nei = y_idx + ky - 1
            x_nei = x_idx + kx - 1
            in_bounds = (y_nei >= 0) & (y_nei < H) & (x_nei >= 0) & (x_nei < W)
            x_base_ptrs = x_ptr + n_idx * stride_xn + y_nei * stride_xh + x_nei * stride_xw
            # 2D pointer: [px, c]
            x_ptrs = x_base_ptrs[:, None] + c_offsets[None, :] * stride_xc
            mask = mask_px[:, None] & mask_c[None, :] & in_bounds[:, None]
            # Use other = -inf so padded elements don't affect max
            vals = tl.load(x_ptrs, mask=mask, other=neg_inf).to(tl.float32)
            max_vals = tl.maximum(max_vals, vals)
            kx += 1
        ky += 1

    # Store results
    y_base_ptrs = y_ptr + n_idx * stride_yn + y_idx * stride_yh + x_idx * stride_yw
    y_ptrs = y_base_ptrs[:, None] + c_offsets[None, :] * stride_yc
    out_mask = mask_px[:, None] & mask_c[None, :]
    tl.store(y_ptrs, max_vals.to(y_ptr.dtype.element_ty), mask=out_mask)


def kernel_function(
    x,
    b1_w, b1_b,
    b2r_w, b2r_b, b2_w, b2_b,
    b3r_w, b3r_b, b3_w, b3_b,
    b4_w, b4_b,
):
    """
    Fused Inception-3b block (NCHW) using Triton kernels.

    Branches:
      - b1: 1x1 conv
      - b2: 1x1 reduce -> 3x3 conv (pad=1)
      - b3: 1x1 reduce -> 5x5 conv (pad=2)
      - b4: 3x3 maxpool (stride=1,pad=1) -> 1x1 conv
    Output: cat([b1, b2, b3, b4], dim=1) -> [N, 480, H, W]

    Precision note:
      - All accumulations are done in fp32.
      - To match the float32-reference more closely under bf16 inputs/weights, we
        keep reduction and pooling intermediates in fp32 and only cast to the
        final output dtype when writing to the final output tensor. This removes
        an extra bf16 quantization step for b2/b3 and fixes the test mismatch.

    Fusion rationale:
      - 1x1-reduction followed by spatial KxK conv is not fused into a single
        pass because the reduced feature maps must be reused across K*K taps.
        Recomputing them would be prohibitively expensive; thus we materialize
        compact fp32 intermediates and then run the spatial convolution.
    """
    assert isinstance(x, torch.Tensor) and x.is_cuda, "x must be a CUDA tensor"
    device = x.device
    dtype = x.dtype
    assert dtype in (torch.bfloat16, torch.float16, torch.float32), "Supported dtypes: bf16, fp16, fp32"

    # Shapes as per test
    N, C_in, H, W = x.shape
    assert x.ndim == 4 and C_in == 256 and H == 28 and W == 28, "Input must be [*, 256, 28, 28] for this test"

    # Validate parameter shapes
    assert b1_w.shape == (128, 256, 1, 1) and b1_b.shape == (128,)
    assert b2r_w.shape == (128, 256, 1, 1) and b2r_b.shape == (128,)
    assert b2_w.shape == (192, 128, 3, 3) and b2_b.shape == (192,)
    assert b3r_w.shape == (32, 256, 1, 1) and b3r_b.shape == (32,)
    assert b3_w.shape == (96, 32, 5, 5) and b3_b.shape == (96,)
    assert b4_w.shape == (64, 256, 1, 1) and b4_b.shape == (64,)

    # Ensure all params on same device/dtype as input
    params = [b1_w, b1_b, b2r_w, b2r_b, b2_w, b2_b, b3r_w, b3r_b, b3_w, b3_b, b4_w, b4_b]
    for t in params:
        assert t.device == device, "All parameters must be on the same CUDA device as input"
        assert t.dtype == dtype, "Parameters must use same dtype as input"

    # Allocate outputs and intermediates
    out = torch.empty((N, 480, H, W), device=device, dtype=dtype)

    # Critical fix: keep intermediates in fp32 to avoid extra bf16 quantization
    b2_reduce = torch.empty((N, 128, H, W), device=device, dtype=torch.float32)
    b3_reduce = torch.empty((N, 32, H, W), device=device, dtype=torch.float32)
    pooled = torch.empty((N, 256, H, W), device=device, dtype=torch.float32)

    # Tiling parameters
    BLOCK_OC_1x1 = 64
    BLOCK_PX = 64
    BLOCK_OC_3x3_5x5 = 64
    BLOCK_C_POOL = 64

    # Strides (for grid-independent kernels)
    sxn, sxc, sxh, sxw = x.stride()
    syn, syc, syh, syw = out.stride()
    pyn, pyc, pyh, pyw = pooled.stride()

    # Branch 4: MaxPool 3x3 stride=1 pad=1 -> pooled (fp32)
    grid_pool = (
        triton.cdiv(C_in, BLOCK_C_POOL),
        triton.cdiv(N * H * W, BLOCK_PX),
    )
    _maxpool3x3_kernel[grid_pool](
        x, pooled,
        N, C_in, H, W,
        sxn, sxc, sxh, sxw,
        pyn, pyc, pyh, pyw,
        BLOCK_C=BLOCK_C_POOL,
        BLOCK_PX=BLOCK_PX,
    )

    # Helper: 1x1 conv
    def launch_conv1x1(x_t, w_t, b_t, y_t, C_IN, C_OUT, CO_OFFSET):
        sxn_, sxc_, sxh_, sxw_ = x_t.stride()
        syn_, syc_, syh_, syw_ = y_t.stride()
        wsn, wsc, wsk, wsl = w_t.stride()  # co, ci, kh, kw
        grid = (
            triton.cdiv(C_OUT, BLOCK_OC_1x1),
            triton.cdiv(N * H * W, BLOCK_PX),
        )
        _conv1x1_kernel[grid](
            x_t, w_t, b_t, y_t,
            N, C_IN, H, W, C_OUT, CO_OFFSET,
            sxn_, sxc_, sxh_, sxw_,
            wsn, wsc, wsk, wsl,
            syn_, syc_, syh_, syw_,
            BLOCK_OC=BLOCK_OC_1x1,
            BLOCK_PX=BLOCK_PX,
        )

    # Helper: KxK conv
    def launch_convk(x_t, w_t, b_t, y_t, C_IN, C_OUT, CO_OFFSET, K, PAD):
        sxn_, sxc_, sxh_, sxw_ = x_t.stride()
        syn_, syc_, syh_, syw_ = y_t.stride()
        wsn, wsc, wsk, wsl = w_t.stride()
        grid = (
            triton.cdiv(C_OUT, BLOCK_OC_3x3_5x5),
            triton.cdiv(N * H * W, BLOCK_PX),
        )
        _convk_kernel[grid](
            x_t, w_t, b_t, y_t,
            N, C_IN, H, W, C_OUT, CO_OFFSET,
            sxn_, sxc_, sxh_, sxw_,
            wsn, wsc, wsk, wsl,
            syn_, syc_, syh_, syw_,
            K=K, PAD=PAD,
            BLOCK_OC=BLOCK_OC_3x3_5x5,
            BLOCK_PX=BLOCK_PX,
        )

    # Branch 1: 1x1 conv -> out[:, 0:128]
    launch_conv1x1(x, b1_w, b1_b, out, C_IN=256, C_OUT=128, CO_OFFSET=0)

    # Branch 2: 1x1 reduce (to fp32) -> 3x3 conv (pad=1) -> out[:, 128:320]
    launch_conv1x1(x, b2r_w, b2r_b, b2_reduce, C_IN=256, C_OUT=128, CO_OFFSET=0)
    launch_convk(b2_reduce, b2_w, b2_b, out, C_IN=128, C_OUT=192, CO_OFFSET=128, K=3, PAD=1)

    # Branch 3: 1x1 reduce (to fp32) -> 5x5 conv (pad=2) -> out[:, 320:416]
    launch_conv1x1(x, b3r_w, b3r_b, b3_reduce, C_IN=256, C_OUT=32, CO_OFFSET=0)
    launch_convk(b3_reduce, b3_w, b3_b, out, C_IN=32, C_OUT=96, CO_OFFSET=320, K=5, PAD=2)

    # Branch 4: pooled (fp32) -> 1x1 proj -> out[:, 416:480]
    launch_conv1x1(pooled, b4_w, b4_b, out, C_IN=256, C_OUT=64, CO_OFFSET=416)

    return out