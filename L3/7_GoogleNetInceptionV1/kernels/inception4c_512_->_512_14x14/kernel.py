import torch
import triton
import triton.language as tl


# -----------------------------------------------------------------------------
# Triton kernels
# -----------------------------------------------------------------------------

@triton.jit
def _maxpool3x3_s1p1_kernel(
    x_ptr, y_ptr,
    N, C, H, W,
    x_stride_n, x_stride_c, x_stride_h, x_stride_w,
    y_stride_n, y_stride_c, y_stride_h, y_stride_w,
    BLOCK_POS: tl.constexpr,   # number of (n,h,w) positions per program
    BLOCK_C: tl.constexpr,     # number of channels per program
):
    # Program ids along position-tiles and channel-tiles
    pid_pos = tl.program_id(axis=0)
    pid_c = tl.program_id(axis=1)

    # Derived sizes
    HW = H * W
    NHW = N * HW

    # Offsets
    pos_start = pid_pos * BLOCK_POS
    offs_pos = pos_start + tl.arange(0, BLOCK_POS)
    mask_pos = offs_pos < NHW

    c_start = pid_c * BLOCK_C
    offs_c = c_start + tl.arange(0, BLOCK_C)
    mask_c = offs_c < C

    # Decode (n, h, w) from flattened position
    n = offs_pos // HW
    hw = offs_pos % HW
    h = hw // W
    w = hw % W

    # Accumulator for max, in fp32 for stability and to match reference numerics
    amax = tl.full((BLOCK_POS, BLOCK_C), -float("inf"), dtype=tl.float32)

    # 3x3 window with padding=1 and stride=1
    for kh in range(0, 3):
        for kw in range(0, 3):
            ih = h + kh - 1
            iw = w + kw - 1
            in_bounds = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
            row_mask = mask_pos & in_bounds
            # Pointers to load input tile [BLOCK_POS, BLOCK_C]
            x_ptrs = x_ptr + (
                n[:, None] * x_stride_n
                + offs_c[None, :] * x_stride_c
                + ih[:, None] * x_stride_h
                + iw[:, None] * x_stride_w
            )
            mask2d = row_mask[:, None] & mask_c[None, :]
            x_vals = tl.load(x_ptrs, mask=mask2d, other=-float("inf"))
            amax = tl.maximum(amax, x_vals.to(tl.float32))

    # Store pooled result
    y_ptrs = y_ptr + (
        n[:, None] * y_stride_n
        + offs_c[None, :] * y_stride_c
        + h[:, None] * y_stride_h
        + w[:, None] * y_stride_w
    )
    y_dtype = y_ptr.dtype.element_ty
    tl.store(y_ptrs, amax.to(y_dtype), mask=mask_pos[:, None] & mask_c[None, :])


@triton.jit
def _conv2d_nchw_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    N, Cin, H, W, Cout,
    x_stride_n, x_stride_c, x_stride_h, x_stride_w,
    w_stride_co, w_stride_ci, w_stride_kh, w_stride_kw,
    y_stride_n, y_stride_c, y_stride_h, y_stride_w,
    CO_BASE: tl.constexpr,   # starting output channel offset within y_ptr (for in-place concatenation)
    PAD_H: tl.constexpr, PAD_W: tl.constexpr,
    KH: tl.constexpr, KW: tl.constexpr,
    BLOCK_POS: tl.constexpr, BLOCK_OC: tl.constexpr, KBLOCK: tl.constexpr,
):
    # Tile IDs
    pid_pos = tl.program_id(axis=0)
    pid_oc = tl.program_id(axis=1)

    # Derived sizes
    HW = H * W
    NHW = N * HW

    # Tile ranges
    pos_start = pid_pos * BLOCK_POS
    offs_pos = pos_start + tl.arange(0, BLOCK_POS)
    mask_pos = offs_pos < NHW

    oc_start = pid_oc * BLOCK_OC
    offs_oc = oc_start + tl.arange(0, BLOCK_OC)
    mask_oc = offs_oc < Cout

    # Decode (n, h, w) from flattened positions
    n = offs_pos // HW
    hw = offs_pos % HW
    oh = hw // W
    ow = hw % W

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_POS, BLOCK_OC), dtype=tl.float32)

    # Reduction over input channels and kernel window
    k_tiles = tl.cdiv(Cin, KBLOCK)
    offs_k = tl.arange(0, KBLOCK)

    # Compute in fp32 to match the reference (conv2d on .float())
    for kh in range(0, KH):
        for kw in range(0, KW):
            ih = oh + kh - PAD_H
            iw = ow + kw - PAD_W
            in_bounds = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
            row_mask = mask_pos & in_bounds
            for kt in range(0, k_tiles):
                k_idx = kt * KBLOCK + offs_k
                mask_k = k_idx < Cin

                # Input tile [BLOCK_POS, KBLOCK]
                x_ptrs = x_ptr + (
                    n[:, None] * x_stride_n
                    + k_idx[None, :] * x_stride_c
                    + ih[:, None] * x_stride_h
                    + iw[:, None] * x_stride_w
                )
                mask_x = row_mask[:, None] & mask_k[None, :]
                x_tile = tl.load(x_ptrs, mask=mask_x, other=0.0).to(tl.float32)  # [BLOCK_POS, KBLOCK]

                # Weight tile [KBLOCK, BLOCK_OC]
                w_ptrs = w_ptr + (
                    offs_oc[None, :] * w_stride_co
                    + k_idx[:, None] * w_stride_ci
                    + kh * w_stride_kh
                    + kw * w_stride_kw
                )
                mask_w = mask_k[:, None] & mask_oc[None, :]
                w_tile = tl.load(w_ptrs, mask=mask_w, other=0.0).to(tl.float32)  # [KBLOCK, BLOCK_OC]

                # Vectorized FMA: (BLOCK_POS, KBLOCK) x (KBLOCK, BLOCK_OC)
                # Avoid scalar indexing to keep Triton happy
                # acc += sum_k x_tile[:, k]*w_tile[k, :]
                acc += tl.sum(x_tile[:, :, None] * w_tile[None, :, :], axis=1)

    # Add bias
    b_vals = tl.load(b_ptr + offs_oc, mask=mask_oc, other=0.0).to(tl.float32)
    acc = acc + b_vals[None, :]

    # Store to output with channel offset CO_BASE (for on-the-fly concatenation)
    y_co = offs_oc + CO_BASE
    y_ptrs = y_ptr + (
        n[:, None] * y_stride_n
        + y_co[None, :] * y_stride_c
        + oh[:, None] * y_stride_h
        + ow[:, None] * y_stride_w
    )
    y_dtype = y_ptr.dtype.element_ty
    tl.store(y_ptrs, acc.to(y_dtype), mask=mask_pos[:, None] & mask_oc[None, :])


# -----------------------------------------------------------------------------
# Wrapper
# -----------------------------------------------------------------------------

def kernel_function(
    x,
    *args,
    weights=None,
    **kwargs
):
    """
    Inception-v1 Inception(4c) block (NCHW) using Triton kernels only.

    Branches:
      - b1: 1x1 conv (512 -> 128)
      - b2: 1x1 reduce (512 -> 128), then 3x3 conv (128 -> 256, pad=1)
      - b3: 1x1 reduce (512 -> 24),  then 5x5 conv (24  -> 64,  pad=2)
      - b4: 3x3 maxpool (s=1,p=1),   then 1x1 conv (512 -> 64)

    Fusion choices:
      - All convs are fused with bias and write directly into their slice of the final output tensor
        (on-the-fly concatenation), eliminating extra concat passes.
      - Pooling remains a small standalone kernel; fusing pool+1x1 would require recomputing/wiring
        a 3x3 max per position per channel within the conv loop, increasing register pressure and
        reducing memory coalescing with little upside for this configuration.

    Runtime policy:
      - Wrapper only validates, allocates, and launches. All math happens inside Triton kernels.
    """
    assert x.is_cuda, "Input must be on CUDA device"
    assert x.dim() == 4, "Expected NCHW input"
    N, Cin, H, W = x.shape
    device = x.device
    dtype = x.dtype
    assert Cin == 512 and H == 14 and W == 14, "Test expects input (N,512,14,14)"
    assert dtype in (torch.bfloat16, torch.float16, torch.float32), "Supported dtypes: bf16/fp16/fp32"

    # Parse weights/biases from args/kwargs/weights dict
    def extract_from_dict(d):
        return (
            d["b1_conv1x1.weight"], d["b1_conv1x1.bias"],
            d["b2_reduce.weight"], d["b2_reduce.bias"],
            d["b2_conv3x3.weight"], d["b2_conv3x3.bias"],
            d["b3_reduce.weight"], d["b3_reduce.bias"],
            d["b3_conv5x5.weight"], d["b3_conv5x5.bias"],
            d["b4_proj.weight"], d["b4_proj.bias"],
        )

    b1_w = b1_b = b2r_w = b2r_b = b2_w = b2_b = None
    b3r_w = b3r_b = b3_w = b3_b = b4_w = b4_b = None

    if weights is None and len(args) == 1 and isinstance(args[0], dict):
        weights = args[0]
        args = ()

    if weights is not None:
        (b1_w, b1_b,
         b2r_w, b2r_b,
         b2_w, b2_b,
         b3r_w, b3r_b,
         b3_w, b3_b,
         b4_w, b4_b) = extract_from_dict(weights)
    elif len(args) == 12:
        (b1_w, b1_b,
         b2r_w, b2r_b,
         b2_w, b2_b,
         b3r_w, b3r_b,
         b3_w, b3_b,
         b4_w, b4_b) = args
    else:
        kw = kwargs
        required = [
            "b1_conv1x1_weight", "b1_conv1x1_bias",
            "b2_reduce_weight", "b2_reduce_bias",
            "b2_conv3x3_weight", "b2_conv3x3_bias",
            "b3_reduce_weight", "b3_reduce_bias",
            "b3_conv5x5_weight", "b3_conv5x5_bias",
            "b4_proj_weight", "b4_proj_bias",
        ]
        if all(k in kw for k in required):
            (b1_w, b1_b,
             b2r_w, b2r_b,
             b2_w, b2_b,
             b3r_w, b3r_b,
             b3_w, b3_b,
             b4_w, b4_b) = (
                kw["b1_conv1x1_weight"], kw["b1_conv1x1_bias"],
                kw["b2_reduce_weight"], kw["b2_reduce_bias"],
                kw["b2_conv3x3_weight"], kw["b2_conv3x3_bias"],
                kw["b3_reduce_weight"], kw["b3_reduce_bias"],
                kw["b3_conv5x5_weight"], kw["b3_conv5x5_bias"],
                kw["b4_proj_weight"], kw["b4_proj_bias"],
            )
        else:
            raise TypeError("Invalid arguments for kernel_function: provide 'weights' dict, 12 positional tensors, or underscore-named kwargs.")

    # Basic checks
    tensors = [b1_w, b1_b, b2r_w, b2r_b, b2_w, b2_b, b3r_w, b3r_b, b3_w, b3_b, b4_w, b4_b]
    for t in tensors:
        assert isinstance(t, torch.Tensor) and t.is_cuda, "All weights/bias must be CUDA tensors"
        assert t.dtype == dtype, "All weights/bias must have same dtype as input (per test)"

    # Shapes based on inception4c config
    assert b1_w.shape == (128, 512, 1, 1) and b1_b.shape == (128,)
    assert b2r_w.shape == (128, 512, 1, 1) and b2r_b.shape == (128,)
    assert b2_w.shape == (256, 128, 3, 3) and b2_b.shape == (256,)
    assert b3r_w.shape == (24, 512, 1, 1) and b3r_b.shape == (24,)
    assert b3_w.shape == (64, 24, 5, 5) and b3_b.shape == (64,)
    assert b4_w.shape == (64, 512, 1, 1) and b4_b.shape == (64,)

    # Allocate outputs: final Y and required intermediates
    Cout_total = 128 + 256 + 64 + 64
    assert Cout_total == 512
    y = torch.empty((N, Cout_total, H, W), device=device, dtype=dtype)

    # Intermediates for b2 and b3 reductions and b4 pooling
    b2r = torch.empty((N, 128, H, W), device=device, dtype=dtype)
    b3r = torch.empty((N, 24, H, W), device=device, dtype=dtype)
    b4p = torch.empty_like(x)  # (N, 512, H, W)

    # Common strides
    xs0, xs1, xs2, xs3 = x.stride()
    ys0, ys1, ys2, ys3 = y.stride()

    # Launch config helpers
    def grid_conv(NHW, Cout, BLOCK_POS, BLOCK_OC):
        return (triton.cdiv(NHW, BLOCK_POS), triton.cdiv(Cout, BLOCK_OC))

    def grid_pool(NHW, C, BLOCK_POS, BLOCK_C):
        return (triton.cdiv(NHW, BLOCK_POS), triton.cdiv(C, BLOCK_C))

    NHW = N * H * W

    # Tunable meta-parameters
    BLOCK_POS = 64   # tile over (n,h,w)
    BLOCK_OC = 64    # tile over output channels
    KBLOCK = 64      # reduction tile over input channels

    # 1) Branch 4: maxpool 3x3 s=1, p=1
    _maxpool3x3_s1p1_kernel[grid_pool(NHW, Cin, BLOCK_POS, BLOCK_OC)](
        x, b4p,
        N, Cin, H, W,
        xs0, xs1, xs2, xs3,
        xs0, xs1, xs2, xs3,
        BLOCK_POS=BLOCK_POS, BLOCK_C=BLOCK_OC
    )

    # 2) Branch 2 reduce: 1x1 conv 512->128, pad=0
    _conv2d_nchw_kernel[grid_conv(NHW, 128, BLOCK_POS, BLOCK_OC)](
        x, b2r_w, b2r_b, b2r,
        N, Cin, H, W, 128,
        xs0, xs1, xs2, xs3,
        b2r_w.stride(0), b2r_w.stride(1), b2r_w.stride(2), b2r_w.stride(3),
        b2r.stride(0), b2r.stride(1), b2r.stride(2), b2r.stride(3),
        CO_BASE=0, PAD_H=0, PAD_W=0, KH=1, KW=1,
        BLOCK_POS=BLOCK_POS, BLOCK_OC=BLOCK_OC, KBLOCK=KBLOCK
    )

    # 3) Branch 3 reduce: 1x1 conv 512->24, pad=0
    _conv2d_nchw_kernel[grid_conv(NHW, 24, BLOCK_POS, BLOCK_OC)](
        x, b3r_w, b3r_b, b3r,
        N, Cin, H, W, 24,
        xs0, xs1, xs2, xs3,
        b3r_w.stride(0), b3r_w.stride(1), b3r_w.stride(2), b3r_w.stride(3),
        b3r.stride(0), b3r.stride(1), b3r.stride(2), b3r.stride(3),
        CO_BASE=0, PAD_H=0, PAD_W=0, KH=1, KW=1,
        BLOCK_POS=BLOCK_POS, BLOCK_OC=BLOCK_OC, KBLOCK=KBLOCK
    )

    # 4) Branch 1 final: 1x1 conv 512->128, pad=0; write directly into y channels [0:128]
    _conv2d_nchw_kernel[grid_conv(NHW, 128, BLOCK_POS, BLOCK_OC)](
        x, b1_w, b1_b, y,
        N, Cin, H, W, 128,
        xs0, xs1, xs2, xs3,
        b1_w.stride(0), b1_w.stride(1), b1_w.stride(2), b1_w.stride(3),
        ys0, ys1, ys2, ys3,
        CO_BASE=0, PAD_H=0, PAD_W=0, KH=1, KW=1,
        BLOCK_POS=BLOCK_POS, BLOCK_OC=BLOCK_OC, KBLOCK=KBLOCK
    )

    # 5) Branch 2 final: 3x3 conv 128->256, pad=1; write into y channels [128:384]
    _conv2d_nchw_kernel[grid_conv(NHW, 256, BLOCK_POS, BLOCK_OC)](
        b2r, b2_w, b2_b, y,
        N, 128, H, W, 256,
        b2r.stride(0), b2r.stride(1), b2r.stride(2), b2r.stride(3),
        b2_w.stride(0), b2_w.stride(1), b2_w.stride(2), b2_w.stride(3),
        ys0, ys1, ys2, ys3,
        CO_BASE=128, PAD_H=1, PAD_W=1, KH=3, KW=3,
        BLOCK_POS=BLOCK_POS, BLOCK_OC=BLOCK_OC, KBLOCK=KBLOCK
    )

    # 6) Branch 3 final: 5x5 conv 24->64, pad=2; write into y channels [384:448]
    _conv2d_nchw_kernel[grid_conv(NHW, 64, BLOCK_POS, BLOCK_OC)](
        b3r, b3_w, b3_b, y,
        N, 24, H, W, 64,
        b3r.stride(0), b3r.stride(1), b3r.stride(2), b3r.stride(3),
        b3_w.stride(0), b3_w.stride(1), b3_w.stride(2), b3_w.stride(3),
        ys0, ys1, ys2, ys3,
        CO_BASE=128 + 256, PAD_H=2, PAD_W=2, KH=5, KW=5,
        BLOCK_POS=BLOCK_POS, BLOCK_OC=BLOCK_OC, KBLOCK=KBLOCK
    )

    # 7) Branch 4 final: 1x1 conv 512->64 on pooled input, pad=0; write into y channels [448:512]
    _conv2d_nchw_kernel[grid_conv(NHW, 64, BLOCK_POS, BLOCK_OC)](
        b4p, b4_w, b4_b, y,
        N, Cin, H, W, 64,
        b4p.stride(0), b4p.stride(1), b4p.stride(2), b4p.stride(3),
        b4_w.stride(0), b4_w.stride(1), b4_w.stride(2), b4_w.stride(3),
        ys0, ys1, ys2, ys3,
        CO_BASE=128 + 256 + 64, PAD_H=0, PAD_W=0, KH=1, KW=1,
        BLOCK_POS=BLOCK_POS, BLOCK_OC=BLOCK_OC, KBLOCK=KBLOCK
    )

    return y