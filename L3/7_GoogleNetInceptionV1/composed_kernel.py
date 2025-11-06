import torch
import triton
import triton.language as tl
import math
import sys

# -----------------------------------------------------------------------------
# Stem: Conv7x7 s=2 p=3 -> ReLU -> MaxPool3x3 s=2 p=1 (fused)
# -----------------------------------------------------------------------------

@triton.jit
def _fused_conv7x7_relu_maxpool3x3(
    x_ptr,  # NCHW input
    w_ptr,  # OIHW weights
    b_ptr,  # bias [O]
    out_ptr,  # NCHW output after ReLU+MaxPool
    N, Cin, H, W, Cout,
    OHc, OWc,  # conv output spatial (112, 112)
    OHp, OWp,  # pooled output spatial (56, 56)
    # input strides
    sx_n, sx_c, sx_h, sx_w,
    # weight strides
    sw_o, sw_i, sw_kh, sw_kw,
    # output strides
    so_n, so_c, so_h, so_w,
    BLOCK_CO: tl.constexpr,  # tile size over output channels
    KH: tl.constexpr, KW: tl.constexpr,  # 7x7 conv kernel
    STRIDE_CONV: tl.constexpr, PAD_CONV: tl.constexpr,  # conv stride and padding
    POOL_K: tl.constexpr, POOL_STRIDE: tl.constexpr,  # maxpool 3x3/2/1
):
    # Program IDs:
    # pid0 walks over all pooled output locations flattened together with N
    # pid1 walks over output channels in tiles of BLOCK_CO
    pid0 = tl.program_id(axis=0)
    pid1 = tl.program_id(axis=1)

    # Which output channel indices does this program compute?
    oc_offsets = pid1 * BLOCK_CO + tl.arange(0, BLOCK_CO)
    oc_mask = oc_offsets < Cout

    # Decode (n, ph, pw) from flattened pid0 over N * OHp * OWp
    per_n = OHp * OWp
    n = pid0 // per_n
    rem = pid0 % per_n
    ph = rem // OWp
    pw = rem % OWp

    # Initialize running max for 3x3 max-pool window with -inf
    max_val = tl.full([BLOCK_CO], -float("inf"), dtype=tl.float32)

    # Load bias upfront (vector for BLOCK_CO channels)
    bias = tl.load(b_ptr + oc_offsets, mask=oc_mask, other=0.0).to(tl.float32)

    # Iterate over 3x3 pooling window centered around (ph*2, pw*2) on conv output
    # Pool padding = 1 -> offsets are {-1, 0, +1}
    for pdy in range(POOL_K):  # 0..2
        dy = pdy - (POOL_K // 2)  # -1, 0, 1
        oh = ph * POOL_STRIDE + dy  # conv output y-index
        for pdx in range(POOL_K):
            dx = pdx - (POOL_K // 2)  # -1, 0, 1
            ow = pw * POOL_STRIDE + dx  # conv output x-index

            # Accumulator for the conv result at (oh, ow) for BLOCK_CO channels
            acc = tl.zeros([BLOCK_CO], dtype=tl.float32)

            # Map conv output index (oh, ow) -> input base with stride/pad
            base_ih = oh * STRIDE_CONV - PAD_CONV
            base_iw = ow * STRIDE_CONV - PAD_CONV

            # Convolution: sum over Cin, KH, KW
            # We mask input reads out-of-bounds to zero.
            for ci in tl.range(0, Cin):
                for kh in range(KH):
                    ih = base_ih + kh
                    in_h_valid = (ih >= 0) & (ih < H)
                    for kw in range(KW):
                        jw = base_iw + kw
                        in_w_valid = (jw >= 0) & (jw < W)
                        valid_in = in_h_valid & in_w_valid & (n < N)

                        # Load input scalar x[n, ci, ih, jw] with mask
                        x_offset = (n * sx_n + ci * sx_c + ih * sx_h + jw * sx_w)
                        x_val = tl.load(x_ptr + x_offset, mask=valid_in, other=0.0).to(tl.float32)

                        # Load weight vector w[oc, ci, kh, kw] for BLOCK_CO channels
                        w_offsets = oc_offsets * sw_o + ci * sw_i + kh * sw_kh + kw * sw_kw
                        w_val = tl.load(w_ptr + w_offsets, mask=oc_mask, other=0.0).to(tl.float32)

                        # FMA accumulate
                        acc += w_val * x_val

            # Add bias and apply ReLU
            y = acc + bias
            y = tl.maximum(y, 0.0)

            # Pool padding behavior: if conv output index is out-of-range, treat as -inf
            valid_y = (oh >= 0) & (oh < OHc) & (ow >= 0) & (ow < OWc)
            y = tl.where(valid_y, y, tl.full([BLOCK_CO], -float("inf"), dtype=tl.float32))

            # Max-pool accumulation
            max_val = tl.maximum(max_val, y)

    # Store pooled result at out[n, oc, ph, pw]
    out_offsets = n * so_n + oc_offsets * so_c + ph * so_h + pw * so_w
    tl.store(out_ptr + out_offsets, max_val.to(out_ptr.dtype.element_ty), mask=oc_mask)


def stem_fused(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor):
    """
    Fused Conv7x7 (stride=2, pad=3) -> ReLU -> MaxPool3x3 (stride=2, pad=1) for NCHW tensors.
    Uses a single Triton kernel to compute the final [N, Cout, 56, 56] activation from [N, Cin, 224, 224].

    Args:
        x: Input [N, 3, 224, 224]
        w: Weights [64, 3, 7, 7]
        b: Bias [64]
    """
    assert x.is_cuda and w.is_cuda and b.is_cuda
    assert x.ndim == 4 and w.ndim == 4 and b.ndim == 1
    N, Cin, H, W = x.shape
    Cout, Cin_w, KH, KW = w.shape
    assert Cin == Cin_w and KH == 7 and KW == 7 and Cout == 64 and Cin == 3
    # Dtype support
    assert x.dtype in (torch.bfloat16, torch.float16, torch.float32)
    assert w.dtype == x.dtype and b.dtype == x.dtype
    # Conv params
    stride_conv = 2
    pad_conv = 3
    OHc = (H + 2 * pad_conv - KH) // stride_conv + 1
    OWc = (W + 2 * pad_conv - KW) // stride_conv + 1
    # Pool params
    pool_k = 3
    pool_stride = 2
    pool_pad = 1
    OHp = (OHc + 2 * pool_pad - pool_k) // pool_stride + 1
    OWp = (OWc + 2 * pool_pad - pool_k) // pool_stride + 1
    # Allocate out
    out = torch.empty((N, Cout, OHp, OWp), device=x.device, dtype=x.dtype)

    # Strides (elements)
    sx_n, sx_c, sx_h, sx_w = x.stride()
    sw_o, sw_i, sw_kh, sw_kw = w.stride()
    so_n, so_c, so_h, so_w = out.stride()

    BLOCK_CO = 32
    grid = (
        N * OHp * OWp,
        triton.cdiv(Cout, BLOCK_CO),
    )

    _fused_conv7x7_relu_maxpool3x3[grid](
        x, w, b, out,
        N, Cin, H, W, Cout,
        OHc, OWc,
        OHp, OWp,
        sx_n, sx_c, sx_h, sx_w,
        sw_o, sw_i, sw_kh, sw_kw,
        so_n, so_c, so_h, so_w,
        BLOCK_CO=BLOCK_CO,
        KH=7, KW=7,
        STRIDE_CONV=2, PAD_CONV=3,
        POOL_K=3, POOL_STRIDE=2,
        num_warps=4, num_stages=2,
    )
    return out

# -----------------------------------------------------------------------------
# Conv 1x1 + Bias + ReLU (fused)
# -----------------------------------------------------------------------------

_autotune_configs_1x1 = [
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
    triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
]


@triton.autotune(configs=_autotune_configs_1x1, key=['M', 'N', 'K'])
@triton.jit
def _conv1x1_bias_relu_kernel(
    x_ptr,           # *bf16/fp16/fp32, input: [N, C_in, H, W] (NCHW, contiguous)
    w_ptr,           # *bf16/fp16/fp32, weight: [C_out, C_in, 1, 1] contiguous
    b_ptr,           # *bf16/fp16/fp32, bias: [C_out]
    y_ptr,           # *bf16/fp16/fp32, output: [N, C_out, H, W] (NCHW, contiguous)
    M,               # int32, M = N * H * W  (flattened NHW)
    N,               # int32, N = C_out
    K,               # int32, K = C_in
    HW,              # int32, H * W (to avoid extra muls)
    # compile-time constants
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # Row/column tile coordinates
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = offs_m < M
    n_mask = offs_n < N

    # Compute (n_idx, rem) for each row in this tile
    HW_i = HW
    n_idx = offs_m // HW_i
    rem   = offs_m - n_idx * HW_i

    # Prepare base pointers for X and Y row starts (for c=0 / oc=0)
    x_row_bases = n_idx * (K * HW_i) + rem                     # [BLOCK_M]
    y_row_bases = n_idx * (N * HW_i) + rem                     # [BLOCK_M]

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # K loop
    k_tiles = tl.cdiv(K, BLOCK_K)
    for kt in tl.range(0, k_tiles):
        k_offs = kt * BLOCK_K + tl.arange(0, BLOCK_K)
        k_mask = k_offs < K

        # A tile (X): [BLOCK_M, BLOCK_K]
        a_ptrs = x_ptr + x_row_bases[:, None] + k_offs[None, :] * HW_i
        a = tl.load(a_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)

        # B tile (W): [BLOCK_K, BLOCK_N] where B[k, n] = W[n, k]
        b_ptrs = w_ptr + offs_n[None, :] * K + k_offs[:, None]
        b = tl.load(b_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)

        acc = tl.dot(a, b, acc)

    # Add bias (broadcast across rows)
    bias = tl.load(b_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
    acc = acc + bias[None, :]

    # ReLU
    acc = tl.maximum(acc, 0.0)

    # Store to output Y
    y_ptrs = y_ptr + y_row_bases[:, None] + offs_n[None, :] * HW_i
    out_dtype = y_ptr.dtype.element_ty
    tl.store(y_ptrs, acc.to(out_dtype), mask=m_mask[:, None] & n_mask[None, :])


def conv1x1_bias_relu(x, weight, bias):
    """
    Fused conv2d 1x1 (stride=1, padding=0, dilation=1, groups=1) + bias + ReLU using a single Triton kernel.
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda
    N, C_in, H, W = x.shape
    C_out, C_in_w, KH, KW = weight.shape
    assert KH == 1 and KW == 1 and C_in_w == C_in
    assert bias.shape[0] == C_out
    assert x.dtype in (torch.bfloat16, torch.float16, torch.float32)
    assert weight.dtype == x.dtype and bias.dtype == x.dtype

    y = torch.empty((N, C_out, H, W), dtype=x.dtype, device=x.device)

    M = N * H * W
    K = C_in
    Ncol = C_out
    HW = H * W

    def grid(meta):
        BM = meta['BLOCK_M']
        BN = meta['BLOCK_N']
        return (triton.cdiv(M, BM), triton.cdiv(Ncol, BN))

    _conv1x1_bias_relu_kernel[grid](
        x, weight, bias, y,
        M, Ncol, K, HW,
    )

    return y


# -----------------------------------------------------------------------------
# Conv3x3 s=1 p=1 + ReLU + MaxPool3x3 s=2 p=1 (fused)
# -----------------------------------------------------------------------------

@triton.jit
def _fused_conv_relu_pool_kernel_nchw(
    x_ptr,                # *bf16 [N, C_in, H, W]
    w_ptr,                # *bf16 [C_out, C_in, 3, 3]
    b_ptr,                # *bf16 [C_out]
    out_ptr,              # *bf16 [N, C_out, PH, PW]
    N, C_in, H, W, C_out,
    PH, PW,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wo, stride_wi, stride_wk, stride_wl,  # strides for weight [O, I, KH, KW]
    stride_on, stride_oc, stride_oh, stride_ow,  # strides for output [N, O, PH, PW]
    BLOCK_OC: tl.constexpr,
):
    pid_spatial = tl.program_id(axis=0)  # pooled spatial index
    pid_no = tl.program_id(axis=1)       # combined (N, OC-block) index

    # Decode pooled spatial coordinates
    y_pool = pid_spatial // PW
    x_pool = pid_spatial % PW

    # Map pid_no into n and oc-block
    num_oc_blocks = tl.cdiv(C_out, BLOCK_OC)
    n = pid_no // num_oc_blocks
    oc_block = pid_no % num_oc_blocks

    # Offsets for output channels this program computes
    oc_start = oc_block * BLOCK_OC
    offs_oc = oc_start + tl.arange(0, BLOCK_OC)
    oc_mask = offs_oc < C_out

    # Prepare nine accumulators for the nine conv positions in the pooling window
    acc00 = tl.zeros((BLOCK_OC,), dtype=tl.float32)
    acc01 = tl.zeros((BLOCK_OC,), dtype=tl.float32)
    acc02 = tl.zeros((BLOCK_OC,), dtype=tl.float32)
    acc10 = tl.zeros((BLOCK_OC,), dtype=tl.float32)
    acc11 = tl.zeros((BLOCK_OC,), dtype=tl.float32)
    acc12 = tl.zeros((BLOCK_OC,), dtype=tl.float32)
    acc20 = tl.zeros((BLOCK_OC,), dtype=tl.float32)
    acc21 = tl.zeros((BLOCK_OC,), dtype=tl.float32)
    acc22 = tl.zeros((BLOCK_OC,), dtype=tl.float32)

    # Precompute conv-grid indices for the 3x3 pooling window centered at (2*y_pool, 2*x_pool)
    yprime0 = 2 * y_pool - 1
    yprime1 = 2 * y_pool + 0
    yprime2 = 2 * y_pool + 1
    xprime0 = 2 * x_pool - 1
    xprime1 = 2 * x_pool + 0
    xprime2 = 2 * x_pool + 1

    # Validity of conv positions within conv output grid [0..H-1]x[0..W-1]
    valid_y0 = (yprime0 >= 0) & (yprime0 < H)
    valid_y1 = (yprime1 >= 0) & (yprime1 < H)
    valid_y2 = (yprime2 >= 0) & (yprime2 < H)
    valid_x0 = (xprime0 >= 0) & (xprime0 < W)
    valid_x1 = (xprime1 >= 0) & (xprime1 < W)
    valid_x2 = (xprime2 >= 0) & (xprime2 < W)

    v00 = valid_y0 & valid_x0
    v01 = valid_y0 & valid_x1
    v02 = valid_y0 & valid_x2
    v10 = valid_y1 & valid_x0
    v11 = valid_y1 & valid_x1
    v12 = valid_y1 & valid_x2
    v20 = valid_y2 & valid_x0
    v21 = valid_y2 & valid_x1
    v22 = valid_y2 & valid_x2

    # Loop over input channels and 3x3 kernel taps
    for c in tl.range(0, C_in):
        for kh in tl.static_range(0, 3):
            for kw in tl.static_range(0, 3):
                # Load weight vector for oc tile
                w_ptrs = w_ptr + offs_oc * stride_wo + c * stride_wi + kh * stride_wk + kw * stride_wl
                w_vec = tl.load(w_ptrs, mask=oc_mask, other=0.0).to(tl.float32)

                # Position (0,0)
                y_in = yprime0 + kh - 1
                x_in = xprime0 + kw - 1
                in_bounds = (y_in >= 0) & (y_in < H) & (x_in >= 0) & (x_in < W)
                x_ptrs = x_ptr + n * stride_xn + c * stride_xc + y_in * stride_xh + x_in * stride_xw
                val = tl.load(x_ptrs, mask=in_bounds, other=0.0).to(tl.float32)
                val = tl.where(v00, val, 0.0)
                acc00 += w_vec * val

                # Position (0,1)
                y_in = yprime0 + kh - 1
                x_in = xprime1 + kw - 1
                in_bounds = (y_in >= 0) & (y_in < H) & (x_in >= 0) & (x_in < W)
                x_ptrs = x_ptr + n * stride_xn + c * stride_xc + y_in * stride_xh + x_in * stride_xw
                val = tl.load(x_ptrs, mask=in_bounds, other=0.0).to(tl.float32)
                val = tl.where(v01, val, 0.0)
                acc01 += w_vec * val

                # Position (0,2)
                y_in = yprime0 + kh - 1
                x_in = xprime2 + kw - 1
                in_bounds = (y_in >= 0) & (y_in < H) & (x_in >= 0) & (x_in < W)
                x_ptrs = x_ptr + n * stride_xn + c * stride_xc + y_in * stride_xh + x_in * stride_xw
                val = tl.load(x_ptrs, mask=in_bounds, other=0.0).to(tl.float32)
                val = tl.where(v02, val, 0.0)
                acc02 += w_vec * val

                # Position (1,0)
                y_in = yprime1 + kh - 1
                x_in = xprime0 + kw - 1
                in_bounds = (y_in >= 0) & (y_in < H) & (x_in >= 0) & (x_in < W)
                x_ptrs = x_ptr + n * stride_xn + c * stride_xc + y_in * stride_xh + x_in * stride_xw
                val = tl.load(x_ptrs, mask=in_bounds, other=0.0).to(tl.float32)
                val = tl.where(v10, val, 0.0)
                acc10 += w_vec * val

                # Position (1,1)
                y_in = yprime1 + kh - 1
                x_in = xprime1 + kw - 1
                in_bounds = (y_in >= 0) & (y_in < H) & (x_in >= 0) & (x_in < W)
                x_ptrs = x_ptr + n * stride_xn + c * stride_xc + y_in * stride_xh + x_in * stride_xw
                val = tl.load(x_ptrs, mask=in_bounds, other=0.0).to(tl.float32)
                val = tl.where(v11, val, 0.0)
                acc11 += w_vec * val

                # Position (1,2)
                y_in = yprime1 + kh - 1
                x_in = xprime2 + kw - 1
                in_bounds = (y_in >= 0) & (y_in < H) & (x_in >= 0) & (x_in < W)
                x_ptrs = x_ptr + n * stride_xn + c * stride_xc + y_in * stride_xh + x_in * stride_xw
                val = tl.load(x_ptrs, mask=in_bounds, other=0.0).to(tl.float32)
                val = tl.where(v12, val, 0.0)
                acc12 += w_vec * val

                # Position (2,0)
                y_in = yprime2 + kh - 1
                x_in = xprime0 + kw - 1
                in_bounds = (y_in >= 0) & (y_in < H) & (x_in >= 0) & (x_in < W)
                x_ptrs = x_ptr + n * stride_xn + c * stride_xc + y_in * stride_xh + x_in * stride_xw
                val = tl.load(x_ptrs, mask=in_bounds, other=0.0).to(tl.float32)
                val = tl.where(v20, val, 0.0)
                acc20 += w_vec * val

                # Position (2,1)
                y_in = yprime2 + kh - 1
                x_in = xprime1 + kw - 1
                in_bounds = (y_in >= 0) & (y_in < H) & (x_in >= 0) & (x_in < W)
                x_ptrs = x_ptr + n * stride_xn + c * stride_xc + y_in * stride_xh + x_in * stride_xw
                val = tl.load(x_ptrs, mask=in_bounds, other=0.0).to(tl.float32)
                val = tl.where(v21, val, 0.0)
                acc21 += w_vec * val

                # Position (2,2)
                y_in = yprime2 + kh - 1
                x_in = xprime2 + kw - 1
                in_bounds = (y_in >= 0) & (y_in < H) & (x_in >= 0) & (x_in < W)
                x_ptrs = x_ptr + n * stride_xn + c * stride_xc + y_in * stride_xh + x_in * stride_xw
                val = tl.load(x_ptrs, mask=in_bounds, other=0.0).to(tl.float32)
                val = tl.where(v22, val, 0.0)
                acc22 += w_vec * val

    # Exclude invalid conv positions from the max by setting them to -inf
    neg_inf = -float("inf")
    acc00 = tl.where(v00, acc00, neg_inf)
    acc01 = tl.where(v01, acc01, neg_inf)
    acc02 = tl.where(v02, acc02, neg_inf)
    acc10 = tl.where(v10, acc10, neg_inf)
    acc11 = tl.where(v11, acc11, neg_inf)
    acc12 = tl.where(v12, acc12, neg_inf)
    acc20 = tl.where(v20, acc20, neg_inf)
    acc21 = tl.where(v21, acc21, neg_inf)
    acc22 = tl.where(v22, acc22, neg_inf)

    # Max across the nine positions
    smax = tl.maximum(acc00, acc01)
    smax = tl.maximum(smax, acc02)
    smax = tl.maximum(smax, acc10)
    smax = tl.maximum(smax, acc11)
    smax = tl.maximum(smax, acc12)
    smax = tl.maximum(smax, acc20)
    smax = tl.maximum(smax, acc21)
    smax = tl.maximum(smax, acc22)

    # Add bias and apply ReLU
    b_ptrs = b_ptr + offs_oc
    bias = tl.load(b_ptrs, mask=oc_mask, other=0.0).to(tl.float32)
    smax = smax + bias
    zero = tl.zeros((BLOCK_OC,), dtype=tl.float32)
    smax = tl.maximum(smax, zero)

    # Store to output
    out_ptrs = out_ptr + n * stride_on + offs_oc * stride_oc + y_pool * stride_oh + x_pool * stride_ow
    tl.store(out_ptrs, smax.to(out_ptr.dtype.element_ty), mask=oc_mask)


def conv3x3_relu_maxpool(x: torch.Tensor, conv_weight: torch.Tensor, conv_bias: torch.Tensor) -> torch.Tensor:
    """
    Fused Conv3x3 (stride=1, pad=1) -> ReLU -> MaxPool3x3 (stride=2, pad=1) for NCHW tensors.
    """
    assert x.is_cuda and conv_weight.is_cuda and conv_bias.is_cuda
    N, C_in, H, W = x.shape
    C_out, C_in_w, KH, KW = conv_weight.shape
    assert C_in_w == C_in and KH == 3 and KW == 3
    assert conv_bias.shape[0] == C_out
    assert x.dtype == conv_weight.dtype == conv_bias.dtype
    assert x.dtype in (torch.bfloat16, torch.float16, torch.float32)

    # Conv output size equals input size for s=1,p=1
    Hc, Wc = H, W
    # Pool params
    pool_k = (3, 3)
    pool_stride = (2, 2)
    pool_pad = (1, 1)
    PH = (Hc + 2 * pool_pad[0] - pool_k[0]) // pool_stride[0] + 1
    PW = (Wc + 2 * pool_pad[1] - pool_k[1]) // pool_stride[1] + 1

    out = torch.empty((N, C_out, PH, PW), device=x.device, dtype=x.dtype)

    sxn, sxc, sxh, sxw = x.stride()
    swo, swi, swk, swl = conv_weight.stride()
    son, soc, soh, sow = out.stride()

    BLOCK_OC = 32
    grid = (PH * PW, N * triton.cdiv(C_out, BLOCK_OC))

    _fused_conv_relu_pool_kernel_nchw[grid](
        x, conv_weight, conv_bias, out,
        N, C_in, H, W, C_out,
        PH, PW,
        sxn, sxc, sxh, sxw,
        swo, swi, swk, swl,
        son, soc, soh, sow,
        BLOCK_OC=BLOCK_OC,
        num_warps=4,
        num_stages=2,
    )
    return out


# -----------------------------------------------------------------------------
# Generic MaxPool2d NCHW (3x3 s=2 p=1)
# -----------------------------------------------------------------------------

@triton.jit
def _maxpool2d_nchw_kernel(
    x_ptr, y_ptr,
    N, C, H, W,
    OH, OW,
    stride_h, stride_w,
    pad_h, pad_w,
    in_stride_n, in_stride_c, in_stride_h, in_stride_w,
    out_stride_n, out_stride_c, out_stride_h, out_stride_w,
    K_H: tl.constexpr, K_W: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    """
    NCHW MaxPool2d with kernel K_H x K_W, stride (stride_h, stride_w), padding (pad_h, pad_w).
    Grid:
      - axis 0: tiles along output width (OW) of size BLOCK_W
      - axis 1: output height (OH)
      - axis 2: packed (N*C)
    """
    pid_w = tl.program_id(axis=0)
    out_h = tl.program_id(axis=1)
    nc = tl.program_id(axis=2)

    # Derive n, c from packed nc
    n = nc // C
    c = nc % C

    # Compute the starting output w indices this program handles
    offs_ow = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    mask_ow = offs_ow < OW

    # Input window top-left for this output row
    in_h0 = out_h * stride_h - pad_h
    in_w0 = offs_ow * stride_w - pad_w

    # Base pointers for this (n, c) slice
    base_in = n * in_stride_n + c * in_stride_c
    base_out = n * out_stride_n + c * out_stride_c

    # Accumulator for max in fp32
    acc = tl.full((BLOCK_W,), -float("inf"), dtype=tl.float32)

    # Iterate over the pooling window
    for ky in range(K_H):
        ih = in_h0 + ky
        valid_h = (ih >= 0) & (ih < H)

        for kx in range(K_W):
            iw = in_w0 + kx
            valid_w = (iw >= 0) & (iw < W) & mask_ow

            x_ptrs = x_ptr + base_in + ih * in_stride_h + iw * in_stride_w
            x_vals = tl.load(x_ptrs, mask=valid_w & valid_h, other=-float("inf"))
            x_vals_f32 = x_vals.to(tl.float32)
            acc = tl.maximum(acc, x_vals_f32)

    # Store result
    y_ptrs = y_ptr + base_out + out_h * out_stride_h + offs_ow * out_stride_w
    tl.store(y_ptrs, acc.to(y_ptr.dtype.element_ty), mask=mask_ow)


def maxpool3x3_s2(x, kernel_size=3, stride=2, padding=1):
    assert x.is_cuda
    kH = kW = kernel_size
    sH = sW = stride
    pH = pW = padding
    N, C, H, W = x.shape
    OH = (H + 2 * pH - kH) // sH + 1
    OW = (W + 2 * pW - kW) // sW + 1
    y = torch.empty((N, C, OH, OW), device=x.device, dtype=x.dtype)

    in_stride_n, in_stride_c, in_stride_h, in_stride_w = x.stride()
    out_stride_n, out_stride_c, out_stride_h, out_stride_w = y.stride()

    BLOCK_W = 16
    grid = (triton.cdiv(OW, BLOCK_W), OH, N * C)

    _maxpool2d_nchw_kernel[grid](
        x, y,
        N, C, H, W,
        OH, OW,
        sH, sW,
        pH, pW,
        in_stride_n, in_stride_c, in_stride_h, in_stride_w,
        out_stride_n, out_stride_c, out_stride_h, out_stride_w,
        K_H=kH, K_W=kW,
        BLOCK_W=BLOCK_W,
        num_warps=4,
        num_stages=2,
    )
    return y


# -----------------------------------------------------------------------------
# Generic Conv2d NCHW (1x1, 3x3, 5x5 etc.) + Bias
# Also a MaxPool3x3 s=1 p=1 for inception branches
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
    pid_pos = tl.program_id(axis=0)
    pid_c = tl.program_id(axis=1)

    HW = H * W
    NHW = N * HW

    pos_start = pid_pos * BLOCK_POS
    offs_pos = pos_start + tl.arange(0, BLOCK_POS)
    mask_pos = offs_pos < NHW

    c_start = pid_c * BLOCK_C
    offs_c = c_start + tl.arange(0, BLOCK_C)
    mask_c = offs_c < C

    n = offs_pos // HW
    hw = offs_pos % HW
    h = hw // W
    w = hw % W

    amax = tl.full((BLOCK_POS, BLOCK_C), -float("inf"), dtype=tl.float32)

    for kh in range(0, 3):
        for kw in range(0, 3):
            ih = h + kh - 1
            iw = w + kw - 1
            in_bounds = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
            row_mask = mask_pos & in_bounds
            x_ptrs = x_ptr + (
                n[:, None] * x_stride_n
                + offs_c[None, :] * x_stride_c
                + ih[:, None] * x_stride_h
                + iw[:, None] * x_stride_w
            )
            mask2d = row_mask[:, None] & mask_c[None, :]
            x_vals = tl.load(x_ptrs, mask=mask2d, other=-float("inf"))
            amax = tl.maximum(amax, x_vals.to(tl.float32))

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
    pid_pos = tl.program_id(axis=0)
    pid_oc = tl.program_id(axis=1)

    HW = H * W
    NHW = N * HW

    pos_start = pid_pos * BLOCK_POS
    offs_pos = pos_start + tl.arange(0, BLOCK_POS)
    mask_pos = offs_pos < NHW

    oc_start = pid_oc * BLOCK_OC
    offs_oc = oc_start + tl.arange(0, BLOCK_OC)
    mask_oc = offs_oc < Cout

    n = offs_pos // HW
    hw = offs_pos % HW
    oh = hw // W
    ow = hw % W

    acc = tl.zeros((BLOCK_POS, BLOCK_OC), dtype=tl.float32)

    k_tiles = tl.cdiv(Cin, KBLOCK)
    offs_k = tl.arange(0, KBLOCK)

    for kh in range(0, KH):
        for kw in range(0, KW):
            ih = oh + kh - PAD_H
            iw = ow + kw - PAD_W
            in_bounds = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
            row_mask = mask_pos & in_bounds
            for kt in range(0, k_tiles):
                k_idx = kt * KBLOCK + offs_k
                mask_k = k_idx < Cin

                x_ptrs = x_ptr + (
                    n[:, None] * x_stride_n
                    + k_idx[None, :] * x_stride_c
                    + ih[:, None] * x_stride_h
                    + iw[:, None] * x_stride_w
                )
                mask_x = row_mask[:, None] & mask_k[None, :]
                x_tile = tl.load(x_ptrs, mask=mask_x, other=0.0).to(tl.float32)

                w_ptrs = w_ptr + (
                    offs_oc[None, :] * w_stride_co
                    + k_idx[:, None] * w_stride_ci
                    + kh * w_stride_kh
                    + kw * w_stride_kw
                )
                mask_w = mask_k[:, None] & mask_oc[None, :]
                w_tile = tl.load(w_ptrs, mask=mask_w, other=0.0).to(tl.float32)

                acc += tl.sum(x_tile[:, :, None] * w_tile[None, :, :], axis=1)

    b_vals = tl.load(b_ptr + offs_oc, mask=mask_oc, other=0.0).to(tl.float32)
    acc = acc + b_vals[None, :]

    y_co = offs_oc + CO_BASE
    y_ptrs = y_ptr + (
        n[:, None] * y_stride_n
        + y_co[None, :] * y_stride_c
        + oh[:, None] * y_stride_h
        + ow[:, None] * y_stride_w
    )
    y_dtype = y_ptr.dtype.element_ty
    tl.store(y_ptrs, acc.to(y_dtype), mask=mask_pos[:, None] & mask_oc[None, :])


def _launch_maxpool3x3_s1p1(x, out):
    N, C, H, W = x.shape
    xs0, xs1, xs2, xs3 = x.stride()
    ys0, ys1, ys2, ys3 = out.stride()

    BLOCK_POS = 64
    BLOCK_C = 64
    NHW = N * H * W

    grid = (triton.cdiv(NHW, BLOCK_POS), triton.cdiv(C, BLOCK_C))
    _maxpool3x3_s1p1_kernel[grid](
        x, out,
        N, C, H, W,
        xs0, xs1, xs2, xs3,
        ys0, ys1, ys2, ys3,
        BLOCK_POS=BLOCK_POS, BLOCK_C=BLOCK_C
    )


def _launch_conv2d(
    x, w, b, y,
    Cout, CO_BASE, KH, KW, PAD_H, PAD_W,
    Cin_override=None
):
    N, Cin, H, W = x.shape
    xs0, xs1, xs2, xs3 = x.stride()
    ys0, ys1, ys2, ys3 = y.stride()
    wsn, wsc, wsk, wsl = w.stride()
    if Cin_override is not None:
        Cin = Cin_override

    BLOCK_POS = 64
    BLOCK_OC = 64
    KBLOCK = 64
    NHW = N * H * W

    grid = (triton.cdiv(NHW, BLOCK_POS), triton.cdiv(Cout, BLOCK_OC))
    _conv2d_nchw_kernel[grid](
        x, w, b, y,
        N, Cin, H, W, Cout,
        xs0, xs1, xs2, xs3,
        wsn, wsc, wsk, wsl,
        ys0, ys1, ys2, ys3,
        CO_BASE=CO_BASE,
        PAD_H=PAD_H, PAD_W=PAD_W,
        KH=KH, KW=KW,
        BLOCK_POS=BLOCK_POS, BLOCK_OC=BLOCK_OC, KBLOCK=KBLOCK
    )


def inception_block(x, weights_dict):
    """
    Generic InceptionModule (as in this problem's Model) using Triton only.
    Branches:
      - b1: 1x1 conv
      - b2: 1x1 reduce -> 3x3 conv (pad=1)
      - b3: 1x1 reduce -> 5x5 conv (pad=2)
      - b4: maxpool 3x3 s=1,p=1 -> 1x1 conv
    Concatenate along channels.
    weights_dict keys:
      b1_w, b1_b,
      b2r_w, b2r_b, b2_w, b2_b,
      b3r_w, b3r_b, b3_w, b3_b,
      b4_w, b4_b
    """
    assert x.is_cuda
    dtype = x.dtype
    device = x.device
    N, Cin, H, W = x.shape

    b1_w = weights_dict['b1_w']; b1_b = weights_dict['b1_b']
    b2r_w = weights_dict['b2r_w']; b2r_b = weights_dict['b2r_b']
    b2_w = weights_dict['b2_w']; b2_b = weights_dict['b2_b']
    b3r_w = weights_dict['b3r_w']; b3r_b = weights_dict['b3r_b']
    b3_w = weights_dict['b3_w']; b3_b = weights_dict['b3_b']
    b4_w = weights_dict['b4_w']; b4_b = weights_dict['b4_b']

    # Channels per branch from weights
    c_b1 = b1_w.shape[0]
    c_b2 = b2_w.shape[0]
    c_b3 = b3_w.shape[0]
    c_b4 = b4_w.shape[0]
    Cout_total = c_b1 + c_b2 + c_b3 + c_b4

    # Output and intermediates
    y = torch.empty((N, Cout_total, H, W), device=device, dtype=dtype)
    b2r = torch.empty((N, b2r_w.shape[0], H, W), device=device, dtype=dtype)
    b3r = torch.empty((N, b3r_w.shape[0], H, W), device=device, dtype=dtype)
    b4p = torch.empty_like(x)

    # Branch 4: pool 3x3 s=1 p=1
    _launch_maxpool3x3_s1p1(x, b4p)

    # Branch 2 reduce: 1x1 conv
    _launch_conv2d(x, b2r_w, b2r_b, b2r, Cout=b2r_w.shape[0], CO_BASE=0, KH=1, KW=1, PAD_H=0, PAD_W=0)

    # Branch 3 reduce: 1x1 conv
    _launch_conv2d(x, b3r_w, b3r_b, b3r, Cout=b3r_w.shape[0], CO_BASE=0, KH=1, KW=1, PAD_H=0, PAD_W=0)

    # Branch 1 final: 1x1 conv -> y[:, :c_b1]
    _launch_conv2d(x, b1_w, b1_b, y, Cout=c_b1, CO_BASE=0, KH=1, KW=1, PAD_H=0, PAD_W=0)

    # Branch 2 final: 3x3 conv (pad=1) -> y[:, c_b1 : c_b1+c_b2]
    _launch_conv2d(b2r, b2_w, b2_b, y, Cout=c_b2, CO_BASE=c_b1, KH=3, KW=3, PAD_H=1, PAD_W=1, Cin_override=b2r.shape[1])

    # Branch 3 final: 5x5 conv (pad=2) -> y[:, c_b1+c_b2 : c_b1+c_b2+c_b3]
    _launch_conv2d(b3r, b3_w, b3_b, y, Cout=c_b3, CO_BASE=c_b1 + c_b2, KH=5, KW=5, PAD_H=2, PAD_W=2, Cin_override=b3r.shape[1])

    # Branch 4 final: 1x1 conv on pooled -> y[:, c_b1+c_b2+c_b3 : ]
    _launch_conv2d(b4p, b4_w, b4_b, y, Cout=c_b4, CO_BASE=c_b1 + c_b2 + c_b3, KH=1, KW=1, PAD_H=0, PAD_W=0)

    return y


# -----------------------------------------------------------------------------
# Head: AdaptiveAvgPool2d(1x1) + Flatten + Linear (fused)
# -----------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 1, 'BLOCK_O': 128, 'BLOCK_C': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 2, 'BLOCK_O': 128, 'BLOCK_C': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_N': 4, 'BLOCK_O': 128, 'BLOCK_C': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_N': 2, 'BLOCK_O': 256, 'BLOCK_C': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_N': 1, 'BLOCK_O': 256, 'BLOCK_C': 128}, num_warps=8, num_stages=4),
    ],
    key=['N', 'O', 'C', 'H', 'W'],
)
@triton.jit
def _avgpool_flatten_linear_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, C, H, W, O,
    sxn, sxc, sxh, sxw,
    swo, swc,
    son, soo,
    BLOCK_N: tl.constexpr,
    BLOCK_O: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_o = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_o = pid_o * BLOCK_O + tl.arange(0, BLOCK_O)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_o = offs_o < O
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_N, BLOCK_O), dtype=tl.float32)

    num_ctiles = tl.cdiv(C, BLOCK_C)
    for ci in range(0, num_ctiles):
        offs_c = ci * BLOCK_C + tl.arange(0, BLOCK_C)
        mask_c = offs_c < C

        sum_hw = tl.zeros((BLOCK_N, BLOCK_C), dtype=tl.float32)

        for h in range(0, H):
            for w in range(0, W):
                x_ptrs = x_ptr + \
                    (offs_n[:, None] * sxn) + \
                    (offs_c[None, :] * sxc) + \
                    (h * sxh) + (w * sxw)
                x_mask = (mask_n[:, None] & mask_c[None, :])
                x_val = tl.load(x_ptrs, mask=x_mask, other=0.0)
                x_val_f32 = x_val.to(tl.float32)
                sum_hw += x_val_f32

        inv_hw = 1.0 / (H * W)
        pooled = sum_hw * inv_hw

        w_ptrs = w_ptr + (offs_c[:, None] * swc) + (offs_o[None, :] * swo)
        w_mask = (mask_c[:, None] & mask_o[None, :])
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0).to(tl.float32)

        acc = tl.dot(pooled, w_tile, acc)

    b_vals = tl.load(b_ptr + offs_o, mask=mask_o, other=0.0).to(tl.float32)
    acc = acc + b_vals[None, :]

    out_ptrs = out_ptr + (offs_n[:, None] * son) + (offs_o[None, :] * soo)
    out_mask = (mask_n[:, None] & mask_o[None, :])
    out = acc.to(out_ptr.dtype.element_ty)
    tl.store(out_ptrs, out, mask=out_mask)


def head_avgpool_linear(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and w.is_cuda and b.is_cuda
    assert x.ndim == 4 and w.ndim == 2 and b.ndim == 1
    N, C, H, W = x.shape
    O, Cw = w.shape
    assert C == Cw and b.shape[0] == O
    out = torch.empty((N, O), device=x.device, dtype=x.dtype)
    sxn, sxc, sxh, sxw = x.stride()
    swo, swc = w.stride()
    son, soo = out.stride()

    def grid(meta):
        return (
            triton.cdiv(O, meta['BLOCK_O']),
            triton.cdiv(N, meta['BLOCK_N']),
        )

    _avgpool_flatten_linear_kernel[grid](
        x, w, b, out,
        N, C, H, W, O,
        sxn, sxc, sxh, sxw,
        swo, swc,
        son, soo,
    )
    return out


# -----------------------------------------------------------------------------
# End-to-end Triton wrapper (top-level API)
# -----------------------------------------------------------------------------

def kernel_function(x: torch.Tensor, weights: dict):
    """
    End-to-end Triton forward pass that replicates the original Model.forward for the provided shapes.

    Args:
      x:        [10, 3, 224, 224] tensor on CUDA
      weights:  dict with all module weights/biases extracted from a Model instance:
                Required keys:
                  'conv1_w', 'conv1_b',
                  'conv2_w', 'conv2_b',
                  'conv3_w', 'conv3_b',
                  'inception3a', 'inception3b',
                  'inception4a', 'inception4b', 'inception4c', 'inception4d', 'inception4e',
                  'inception5a', 'inception5b',
                  'fc_w', 'fc_b'
                where each inception entry is itself a dict with keys:
                  b1_w, b1_b, b2r_w, b2r_b, b2_w, b2_b, b3r_w, b3r_b, b3_w, b3_b, b4_w, b4_b
    Returns:
      y: [10, 1000]
    """
    assert x.is_cuda, "Input must be CUDA"
    assert x.dtype in (torch.bfloat16, torch.float16, torch.float32), "Supported dtypes: bf16/fp16/fp32"

    # Stem: conv7x7->relu->maxpool
    x = stem_fused(x, weights['conv1_w'], weights['conv1_b'])             # [N,64,56,56]
    # conv2 1x1 + ReLU
    x = conv1x1_bias_relu(x, weights['conv2_w'], weights['conv2_b'])      # [N,64,56,56]
    # conv3 3x3 + ReLU + MaxPool s=2
    x = conv3x3_relu_maxpool(x, weights['conv3_w'], weights['conv3_b'])   # [N,192,28,28]

    # Inception 3a, 3b
    x = inception_block(x, weights['inception3a'])                         # [N,256,28,28]
    x = inception_block(x, weights['inception3b'])                         # [N,480,28,28]
    # MaxPool3 3x3 s=2
    x = maxpool3x3_s2(x)                                                   # [N,480,14,14]

    # Inception 4a..4e
    x = inception_block(x, weights['inception4a'])                         # [N,512,14,14]
    x = inception_block(x, weights['inception4b'])                         # [N,512,14,14]
    x = inception_block(x, weights['inception4c'])                         # [N,512,14,14]
    x = inception_block(x, weights['inception4d'])                         # [N,528,14,14]
    x = inception_block(x, weights['inception4e'])                         # [N,832,14,14]
    # MaxPool4 3x3 s=2
    x = maxpool3x3_s2(x)                                                   # [N,832,7,7]

    # Inception 5a, 5b
    x = inception_block(x, weights['inception5a'])                         # [N,832,7,7]
    x = inception_block(x, weights['inception5b'])                         # [N,1024,7,7]

    # Head: adaptive avg pool to 1x1, flatten, linear
    y = head_avgpool_linear(x, weights['fc_w'], weights['fc_b'])           # [N,1000]
    return y


# -----------------------------------------------------------------------------
# Reference Model (for self-test)
# -----------------------------------------------------------------------------

import torch.nn as nn
import torch.nn.functional as F

class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_1x1, reduce_3x3, out_3x3, reduce_5x5, out_5x5, pool_proj):
        super(InceptionModule, self).__init__()
        self.branch1x1 = nn.Conv2d(in_channels, out_1x1, kernel_size=1)
        self.branch3x3 = nn.Sequential(
            nn.Conv2d(in_channels, reduce_3x3, kernel_size=1),
            nn.Conv2d(reduce_3x3, out_3x3, kernel_size=3, padding=1)
        )
        self.branch5x5 = nn.Sequential(
            nn.Conv2d(in_channels, reduce_5x5, kernel_size=1),
            nn.Conv2d(reduce_5x5, out_5x5, kernel_size=5, padding=2)
        )
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1)
        )
    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3(x)
        branch5x5 = self.branch5x5(x)
        branch_pool = self.branch_pool(x)
        outputs = [branch1x1, branch3x3, branch5x5, branch_pool]
        return torch.cat(outputs, 1)

class Model(nn.Module):
    def __init__(self, num_classes=1000):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=1)
        self.conv3 = nn.Conv2d(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, padding=1)
        self.inception3a = InceptionModule(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = InceptionModule(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, padding=1)
        self.inception4a = InceptionModule(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = InceptionModule(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = InceptionModule(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = InceptionModule(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = InceptionModule(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(3, stride=2, padding=1)
        self.inception5a = InceptionModule(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = InceptionModule(832, 384, 192, 384, 48, 128, 128)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.0)
        self.fc = nn.Linear(1024, num_classes)
    def forward(self, x):
        x = self.maxpool1(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = self.maxpool2(F.relu(self.conv3(x)))
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)
        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)
        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

# Test shapes
batch_size = 10
input_channels = 3
height = 224
width = 224
num_classes = 1000

def get_inputs():
    return [torch.rand(batch_size, input_channels, height, width)]

def get_init_inputs():
    return [num_classes]


# -----------------------------------------------------------------------------
# Helpers to extract weights for the Triton pipeline
# -----------------------------------------------------------------------------

def _pack_inception_weights(mod: InceptionModule):
    return {
        'b1_w': mod.branch1x1.weight.detach(),
        'b1_b': mod.branch1x1.bias.detach(),
        'b2r_w': mod.branch3x3[0].weight.detach(),
        'b2r_b': mod.branch3x3[0].bias.detach(),
        'b2_w': mod.branch3x3[1].weight.detach(),
        'b2_b': mod.branch3x3[1].bias.detach(),
        'b3r_w': mod.branch5x5[0].weight.detach(),
        'b3r_b': mod.branch5x5[0].bias.detach(),
        'b3_w': mod.branch5x5[1].weight.detach(),
        'b3_b': mod.branch5x5[1].bias.detach(),
        'b4_w': mod.branch_pool[1].weight.detach(),
        'b4_b': mod.branch_pool[1].bias.detach(),
    }

def extract_all_weights(model: Model, dtype=torch.float32, device='cuda'):
    # Move everything to device/dtype
    def to_dd(t):
        return t.to(device=device, dtype=dtype).contiguous()
    w = {}
    w['conv1_w'] = to_dd(model.conv1.weight.detach())
    w['conv1_b'] = to_dd(model.conv1.bias.detach())
    w['conv2_w'] = to_dd(model.conv2.weight.detach())
    w['conv2_b'] = to_dd(model.conv2.bias.detach())
    w['conv3_w'] = to_dd(model.conv3.weight.detach())
    w['conv3_b'] = to_dd(model.conv3.bias.detach())

    w['inception3a'] = {k: to_dd(v) for k, v in _pack_inception_weights(model.inception3a).items()}
    w['inception3b'] = {k: to_dd(v) for k, v in _pack_inception_weights(model.inception3b).items()}
    w['inception4a'] = {k: to_dd(v) for k, v in _pack_inception_weights(model.inception4a).items()}
    w['inception4b'] = {k: to_dd(v) for k, v in _pack_inception_weights(model.inception4b).items()}
    w['inception4c'] = {k: to_dd(v) for k, v in _pack_inception_weights(model.inception4c).items()}
    w['inception4d'] = {k: to_dd(v) for k, v in _pack_inception_weights(model.inception4d).items()}
    w['inception4e'] = {k: to_dd(v) for k, v in _pack_inception_weights(model.inception4e).items()}
    w['inception5a'] = {k: to_dd(v) for k, v in _pack_inception_weights(model.inception5a).items()}
    w['inception5b'] = {k: to_dd(v) for k, v in _pack_inception_weights(model.inception5b).items()}

    w['fc_w'] = to_dd(model.fc.weight.detach())
    w['fc_b'] = to_dd(model.fc.bias.detach())
    return w


# -----------------------------------------------------------------------------
# Self-test
# -----------------------------------------------------------------------------

def run_tests():
    torch.manual_seed(0)
    # Instantiate reference model
    model = Model(*get_init_inputs()).cuda().eval().to(dtype=torch.float32)

    # Input
    x = get_inputs()[0].cuda().to(dtype=torch.float32)

    # Reference forward (PyTorch)
    with torch.no_grad():
        ref = model(x)

    # Triton forward
    weights = extract_all_weights(model, dtype=torch.float32, device=x.device)
    with torch.no_grad():
        out = kernel_function(x, weights)

    # Compare
    rtol = 1e-3
    atol = 1e-3
    ok = torch.allclose(out, ref, rtol=rtol, atol=atol)
    print("Output shape:", out.shape, "| Reference shape:", ref.shape)
    max_abs = (out - ref).abs().max().item()
    print(f"Max abs diff: {max_abs:.6f}")
    if ok:
        print("PASS")
        return 0
    else:
        print("FAIL")
        return 1


if __name__ == "__main__":
    sys.exit(run_tests())
