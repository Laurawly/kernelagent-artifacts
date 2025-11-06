import torch
import triton
import triton.language as tl

"""
Kernel pipeline implemented:

Stage 1: Conv2D (NCHW, stride=1, pad=2, dilation=1, groups=1) + Bias + ReLU
- Implemented as an implicit im2col x GEMM: (N*H_out*W_out, Cin*KH*KW) @ (Cin*KH*KW, Cout)
- Accumulation in fp32, input/weights in bf16, output cast back to input dtype.
- Fused epilogue: bias addition and ReLU are applied before storing.

Stage 2: MaxPool2d (kernel=3, stride=2, pad=0, ceil_mode=False)
- Implemented as a separate Triton kernel over the conv output.
- Computes 3x3 window maxima in fp32 for stability, stores to output dtype.

Notes on fusion decision:
- Fully fusing pool into conv would require computing each pooled output by re-computing overlapping conv outputs multiple times (3x3 windows with stride 2), roughly 2x redundant compute over the full 27x27 conv grid. This is infeasible at these sizes (N=1024, Cout=256).
- Therefore, we aggressively fuse bias+ReLU with conv, and keep pooling as a separate kernel to avoid redundant conv compute and keep memory traffic reasonable.

Runtime restrictions adhered to:
- All computation is done inside Triton kernels (no torch.nn.functional or other PyTorch compute ops).
- Wrapper only validates inputs, allocates outputs, computes grids, and launches kernels.
"""


@triton.jit
def _conv2d_relu_im2col_gemm(
    x_ptr,        # bf16 [N, Cin, H, W]
    w_ptr,        # bf16 [Cout, Cin, KH, KW]
    b_ptr,        # bf16 [Cout]
    y_ptr,        # bf16 [N, Cout, H_out, W_out]
    N, Cin, H, W,
    Cout, H_out, W_out,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wn, stride_wc, stride_wkh, stride_wkw,
    stride_yn, stride_yc, stride_yh, stride_yw,
    K_total,      # = Cin*KH*KW
    HW_out,       # = H_out*W_out
    pad_h: tl.constexpr,
    pad_w: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    BLOCK_M: tl.constexpr,   # tile on M=N*H_out*W_out
    BLOCK_N: tl.constexpr,   # tile on N=Cout
    BLOCK_K: tl.constexpr,   # tile on K=Cin*KH*KW
):
    # Program ids
    pid_m = tl.program_id(0)  # along M dimension (flattened output pixels)
    pid_n = tl.program_id(1)  # along output channels (Cout)

    # Offsets for this tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    m_mask = offs_m < (N * HW_out)
    n_mask = offs_n < Cout

    # Decode M -> (n, oh, ow)
    n_idx = offs_m // HW_out
    rem_m = offs_m % HW_out
    oh_idx = rem_m // W_out
    ow_idx = rem_m % W_out

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K in BLOCK_K
    for k0 in range(0, K_total, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K_total

        # Decode K -> (c, ky, kx)
        c_idx = offs_k // (KH * KW)
        rem_k = offs_k % (KH * KW)
        ky_idx = rem_k // KW
        kx_idx = rem_k % KW

        # Compute input coordinates for A-tile (implicit im2col)
        ih = oh_idx[:, None] + ky_idx[None, :] - pad_h
        iw = ow_idx[:, None] + kx_idx[None, :] - pad_w

        # Pointer matrix for input X
        x_ptrs = (
            x_ptr
            + n_idx[:, None] * stride_xn
            + c_idx[None, :] * stride_xc
            + ih * stride_xh
            + iw * stride_xw
        )

        # Mask for valid loads
        in_bounds = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
        a_mask = (m_mask[:, None] & k_mask[None, :] & in_bounds)

        # Load A tile
        a = tl.load(x_ptrs, mask=a_mask, other=0.0)

        # Pointer matrix for weights B (shape [K, N])
        w_ptrs = (
            w_ptr
            + offs_n[None, :] * stride_wn
            + c_idx[:, None] * stride_wc
            + ky_idx[:, None] * stride_wkh
            + kx_idx[:, None] * stride_wkw
        )

        b_mask = k_mask[:, None] & n_mask[None, :]
        b = tl.load(w_ptrs, mask=b_mask, other=0.0)

        # Multiply-accumulate
        acc = tl.dot(a, b, acc)

    # Add bias and ReLU epilogue
    bias = tl.load(b_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
    acc = acc + bias[None, :]
    acc = tl.maximum(acc, 0.0)

    # Store to output Y (N, Cout, H_out, W_out)
    y_ptrs = (
        y_ptr
        + n_idx[:, None] * stride_yn
        + offs_n[None, :] * stride_yc
        + oh_idx[:, None] * stride_yh
        + ow_idx[:, None] * stride_yw
    )
    store_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(y_ptrs, acc.to(y_ptr.dtype.element_ty), mask=store_mask)


@triton.jit
def _maxpool2d_3x3_s2(
    x_ptr,     # [N, C, H, W] bf16 - input (conv+relu)
    y_ptr,     # [N, C, H_out, W_out] bf16 - output pooled
    N, C, H, W,
    H_out, W_out,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_yn, stride_yc, stride_yh, stride_yw,
    BLOCK_P: tl.constexpr,   # tile for N*C dimension
    BLOCK_W: tl.constexpr,   # tile for W_out dimension
):
    # Grid: (ceil((N*C)/BLOCK_P), ceil(W_out/BLOCK_W), H_out)
    pid_p = tl.program_id(0)
    pid_w = tl.program_id(1)
    pid_h = tl.program_id(2)

    P = N * C

    offs_p = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)      # [BLOCK_P]
    offs_w = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)      # [BLOCK_W]
    oh = pid_h

    p_mask = offs_p < P
    w_mask = offs_w < W_out

    # Decode p -> (n, c)
    n_idx = offs_p // C
    c_idx = offs_p % C

    # Compute input base h index for the 3x3 window
    in_h0 = oh * 2  # stride = 2

    # First load initializes max
    in_w0 = offs_w * 2 + 0
    ptrs0 = (
        x_ptr
        + n_idx[:, None] * stride_xn
        + c_idx[:, None] * stride_xc
        + (in_h0 + 0) * stride_xh
        + in_w0[None, :] * stride_xw
    )
    mask0 = p_mask[:, None] & w_mask[None, :]
    val = tl.load(ptrs0, mask=mask0, other=0.0).to(tl.float32)
    maxv = val

    # Remaining 8 elements of the 3x3 window
    # krow=0, kcol=1..2
    in_w1 = offs_w * 2 + 1
    ptrs = (
        x_ptr
        + n_idx[:, None] * stride_xn
        + c_idx[:, None] * stride_xc
        + (in_h0 + 0) * stride_xh
        + in_w1[None, :] * stride_xw
    )
    val = tl.load(ptrs, mask=mask0, other=0.0).to(tl.float32)
    maxv = tl.maximum(maxv, val)

    in_w2 = offs_w * 2 + 2
    ptrs = (
        x_ptr
        + n_idx[:, None] * stride_xn
        + c_idx[:, None] * stride_xc
        + (in_h0 + 0) * stride_xh
        + in_w2[None, :] * stride_xw
    )
    val = tl.load(ptrs, mask=mask0, other=0.0).to(tl.float32)
    maxv = tl.maximum(maxv, val)

    # krow=1, kcol=0..2
    in_h1 = in_h0 + 1
    in_w0 = offs_w * 2 + 0
    ptrs = (
        x_ptr
        + n_idx[:, None] * stride_xn
        + c_idx[:, None] * stride_xc
        + in_h1 * stride_xh
        + in_w0[None, :] * stride_xw
    )
    val = tl.load(ptrs, mask=mask0, other=0.0).to(tl.float32)
    maxv = tl.maximum(maxv, val)

    in_w1 = offs_w * 2 + 1
    ptrs = (
        x_ptr
        + n_idx[:, None] * stride_xn
        + c_idx[:, None] * stride_xc
        + in_h1 * stride_xh
        + in_w1[None, :] * stride_xw
    )
    val = tl.load(ptrs, mask=mask0, other=0.0).to(tl.float32)
    maxv = tl.maximum(maxv, val)

    in_w2 = offs_w * 2 + 2
    ptrs = (
        x_ptr
        + n_idx[:, None] * stride_xn
        + c_idx[:, None] * stride_xc
        + in_h1 * stride_xh
        + in_w2[None, :] * stride_xw
    )
    val = tl.load(ptrs, mask=mask0, other=0.0).to(tl.float32)
    maxv = tl.maximum(maxv, val)

    # krow=2, kcol=0..2
    in_h2 = in_h0 + 2
    in_w0 = offs_w * 2 + 0
    ptrs = (
        x_ptr
        + n_idx[:, None] * stride_xn
        + c_idx[:, None] * stride_xc
        + in_h2 * stride_xh
        + in_w0[None, :] * stride_xw
    )
    val = tl.load(ptrs, mask=mask0, other=0.0).to(tl.float32)
    maxv = tl.maximum(maxv, val)

    in_w1 = offs_w * 2 + 1
    ptrs = (
        x_ptr
        + n_idx[:, None] * stride_xn
        + c_idx[:, None] * stride_xc
        + in_h2 * stride_xh
        + in_w1[None, :] * stride_xw
    )
    val = tl.load(ptrs, mask=mask0, other=0.0).to(tl.float32)
    maxv = tl.maximum(maxv, val)

    in_w2 = offs_w * 2 + 2
    ptrs = (
        x_ptr
        + n_idx[:, None] * stride_xn
        + c_idx[:, None] * stride_xc
        + in_h2 * stride_xh
        + in_w2[None, :] * stride_xw
    )
    val = tl.load(ptrs, mask=mask0, other=0.0).to(tl.float32)
    maxv = tl.maximum(maxv, val)

    # Store pooled result
    y_ptrs = (
        y_ptr
        + n_idx[:, None] * stride_yn
        + c_idx[:, None] * stride_yc
        + oh * stride_yh
        + offs_w[None, :] * stride_yw
    )
    tl.store(y_ptrs, maxv.to(y_ptr.dtype.element_ty), mask=mask0)


def kernel_function(x, weight, bias):
    """
    Fused conv->ReLU->maxpool pipeline using Triton.

    Args:
      x:      Input tensor [N, C_in, H, W], dtype bfloat16/float16/float32 (tested with bfloat16), NCHW contiguous.
      weight: Weights [C_out, C_in, 5, 5], same dtype as x.
      bias:   Bias [C_out], same dtype as x.

    Returns:
      Tensor [N, C_out, 13, 13] of same dtype/device as inputs.
    """
    # Argument validation and setup (no compute)
    assert x.device.type == "cuda", "Input must be on CUDA"
    assert x.is_contiguous(), "Input x must be contiguous (NCHW)"
    assert weight.is_contiguous(), "Weight must be contiguous (O, I, KH, KW)"
    assert bias.is_contiguous(), "Bias must be contiguous"
    assert x.ndim == 4 and weight.ndim == 4 and bias.ndim == 1
    N, Cin, H, W = x.shape
    Cout, Cin_w, KH, KW = weight.shape
    assert Cin == Cin_w, "Cin mismatch"
    assert KH == 5 and KW == 5, "Kernel must be 5x5 per test spec"
    assert bias.shape[0] == Cout, "Bias size mismatch"

    # Convolution parameters (fixed by test)
    stride_h = 1
    stride_w = 1
    pad_h = 2
    pad_w = 2
    dilation_h = 1
    dilation_w = 1
    groups = 1
    assert groups == 1, "This kernel supports groups=1"
    assert dilation_h == 1 and dilation_w == 1
    # Output shapes
    H_out = (H + 2 * pad_h - dilation_h * (KH - 1) - 1) // stride_h + 1
    W_out = (W + 2 * pad_w - dilation_w * (KW - 1) - 1) // stride_w + 1
    assert H_out == 27 and W_out == 27, "Output spatial must be 27x27 for given test"
    # Pool params (fixed)
    pool_k = 3
    pool_s = 2
    pool_p = 0
    Hp = (H_out + 2 * pool_p - pool_k) // pool_s + 1
    Wp = (W_out + 2 * pool_p - pool_k) // pool_s + 1
    assert Hp == 13 and Wp == 13

    # Dtype: operate inputs/weights in their dtype, accumulate in fp32
    dtype = x.dtype
    assert dtype in (torch.bfloat16, torch.float16, torch.float32), "Supported dtypes: bf16/fp16/fp32"

    device = x.device
    # Allocate intermediate conv+relu output and final pooled output
    y_conv = torch.empty((N, Cout, H_out, W_out), dtype=dtype, device=device)
    y_out = torch.empty((N, Cout, Hp, Wp), dtype=dtype, device=device)

    # Strides
    sxn, sxc, sxh, sxw = x.stride()
    swn, swc, swkh, swkw = weight.stride()
    syn, syc, syh, syw = y_conv.stride()
    spn, spc, sph, spw = y_out.stride()

    # Launch Conv+ReLU kernel
    M_total = N * H_out * W_out
    K_total = Cin * KH * KW
    HW_out = H_out * W_out

    # Choose tile sizes (powers of two, reasonably balanced)
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32

    grid_conv = (triton.cdiv(M_total, BLOCK_M), triton.cdiv(Cout, BLOCK_N))
    _conv2d_relu_im2col_gemm[grid_conv](
        x, weight, bias, y_conv,
        N, Cin, H, W,
        Cout, H_out, W_out,
        sxn, sxc, sxh, sxw,
        swn, swc, swkh, swkw,
        syn, syc, syh, syw,
        K_total,
        HW_out,
        pad_h, pad_w,
        KH, KW,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=8, num_stages=2,
    )

    # Launch MaxPool 3x3 stride 2 kernel
    BLOCK_P = 64
    BLOCK_W = 16
    grid_pool = (triton.cdiv(N * Cout, BLOCK_P), triton.cdiv(Wp, BLOCK_W), Hp)
    _maxpool2d_3x3_s2[grid_pool](
        y_conv, y_out,
        N, Cout, H_out, W_out,
        Hp, Wp,
        syn, syc, syh, syw,
        spn, spc, sph, spw,
        BLOCK_P=BLOCK_P, BLOCK_W=BLOCK_W,
        num_warps=4, num_stages=2,
    )

    return y_out

# Alias per problem statement: the callable wrapper is kernel_function
# The Triton kernels are defined above and launched here.