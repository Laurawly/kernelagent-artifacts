import sys
import torch
import triton
import triton.language as tl


# ============================
# Fused Conv(5x5, s=1, p=0) + ReLU + MaxPool2d(2x2, s=2)
# Kernel variant 1 (used for conv1)
# ============================
@triton.jit
def _conv_relu_maxpool2x2_nchw_fused(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, C_in, H, W,
    C_out, KH: tl.constexpr, KW: tl.constexpr,
    H_OUT, W_OUT,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wn, stride_wc, stride_wh, stride_ww,
    stride_on, stride_oc, stride_oh, stride_ow,
    NW_TILES: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    """
    Fused Conv2d (stride=1, pad=0, 5x5), ReLU, MaxPool2d(2x2 stride=2) kernel.
    - Layout: NCHW
    - x_ptr:  [N, C_in, H, W]
    - w_ptr:  [C_out, C_in, KH, KW]
    - b_ptr:  [C_out]
    - out_ptr:[N, C_out, H_OUT, W_OUT]
    Fuses:
      conv (fp32 accumulation) -> bias add -> ReLU -> 2x2 maxpool with stride 2
    The kernel computes a single output-channel (oc) slice for a batch item (n) and
    a tile of pooled width locations at a fixed pooled row (oh).
    """
    # Program IDs
    pid_n = tl.program_id(0)  # N dimension
    pid_oc = tl.program_id(1)  # C_out dimension
    pid_oh_tile = tl.program_id(2)  # combined oh and ow-tile

    oh = pid_oh_tile // NW_TILES
    wtile = pid_oh_tile % NW_TILES

    # Offsets for pooled width within the row
    ow_start = wtile * BLOCK_W
    offs_w = ow_start + tl.arange(0, BLOCK_W)
    mask_ow = offs_w < W_OUT

    # Load bias for current output channel and broadcast to accumulators
    b_f32 = tl.load(b_ptr + pid_oc, mask=True).to(tl.float32)
    acc00 = tl.full((BLOCK_W,), b_f32, dtype=tl.float32)
    acc01 = tl.full((BLOCK_W,), b_f32, dtype=tl.float32)
    acc10 = tl.full((BLOCK_W,), b_f32, dtype=tl.float32)
    acc11 = tl.full((BLOCK_W,), b_f32, dtype=tl.float32)

    # Precompute some terms
    # Each pooled output (oh, ow) pools over conv outputs at (2*oh + dh, 2*ow + dw) with dh,dw in {0,1}
    base_h0 = 2 * oh  # scalar
    base_w = 2 * offs_w  # vector

    # Base pointers for input x: select batch n
    x_base_n = x_ptr + pid_n * stride_xn

    # Loop over input channels and kernel spatial
    for ci in range(0, C_in):
        x_base_nc = x_base_n + ci * stride_xc
        # For each kernel point (ky, kx)
        for ky in range(0, KH):
            # Two row offsets for pooling dh in {0,1}
            h_in0 = base_h0 + 0 + ky  # scalar
            h_in1 = base_h0 + 1 + ky  # scalar

            # Since we assume valid shapes (pad=0, stride=1), these are always in-bounds.
            # Vectorized across BLOCK_W pooled positions for each of the four pooled conv sites.
            for kx in range(0, KW):
                # Load weight for this oc, ci, ky, kx (scalar)
                w_val = tl.load(w_ptr + pid_oc * stride_wn + ci * stride_wc + ky * stride_wh + kx * stride_ww).to(tl.float32)

                # dw = 0
                w_in00 = base_w + 0 + kx
                ptr_x00 = x_base_nc + h_in0 * stride_xh + w_in00 * stride_xw
                x00 = tl.load(ptr_x00, mask=mask_ow, other=0.0).to(tl.float32)
                acc00 += x00 * w_val

                w_in10 = base_w + 0 + kx
                ptr_x10 = x_base_nc + h_in1 * stride_xh + w_in10 * stride_xw
                x10 = tl.load(ptr_x10, mask=mask_ow, other=0.0).to(tl.float32)
                acc10 += x10 * w_val

                # dw = 1
                w_in01 = base_w + 1 + kx
                ptr_x01 = x_base_nc + h_in0 * stride_xh + w_in01 * stride_xw
                x01 = tl.load(ptr_x01, mask=mask_ow, other=0.0).to(tl.float32)
                acc01 += x01 * w_val

                w_in11 = base_w + 1 + kx
                ptr_x11 = x_base_nc + h_in1 * stride_xh + w_in11 * stride_xw
                x11 = tl.load(ptr_x11, mask=mask_ow, other=0.0).to(tl.float32)
                acc11 += x11 * w_val

    # ReLU
    zero = tl.zeros((BLOCK_W,), dtype=tl.float32)
    acc00 = tl.maximum(acc00, zero)
    acc01 = tl.maximum(acc01, zero)
    acc10 = tl.maximum(acc10, zero)
    acc11 = tl.maximum(acc11, zero)

    # MaxPool 2x2 over the four conv outputs
    max0 = tl.maximum(acc00, acc01)
    max1 = tl.maximum(acc10, acc11)
    pooled = tl.maximum(max0, max1)

    # Store to output
    out_ptrs = out_ptr + pid_n * stride_on + pid_oc * stride_oc + oh * stride_oh + offs_w * stride_ow
    tl.store(out_ptrs, pooled, mask=mask_ow)


def conv_relu_maxpool_fused1(x: torch.Tensor, conv_weight: torch.Tensor, conv_bias: torch.Tensor):
    """
    Fused Conv2d (5x5, stride=1, pad=0, bias) + ReLU + MaxPool2d (2x2, stride=2) implemented in Triton.
    Used for the first convolutional block (1->6 channels).
    """
    assert x.is_cuda and conv_weight.is_cuda and conv_bias.is_cuda, "All tensors must be on CUDA device."
    assert x.ndim == 4 and conv_weight.ndim == 4 and conv_bias.ndim == 1
    N, C_in, H, W = x.shape
    C_out, C_in_w, KH, KW = conv_weight.shape
    assert C_in == C_in_w
    assert KH == 5 and KW == 5, "Kernel expects 5x5 conv."

    H_conv = H - KH + 1
    W_conv = W - KW + 1
    assert H_conv > 0 and W_conv > 0

    # MaxPool 2x2, stride 2
    H_out = (H_conv - 2) // 2 + 1
    W_out = (W_conv - 2) // 2 + 1

    # Allocate output in fp32 for stability
    out = torch.empty((N, C_out, H_out, W_out), device=x.device, dtype=torch.float32)

    # Compute strides in elements
    s_xn, s_xc, s_xh, s_xw = x.stride()
    s_wn, s_wc, s_wh, s_ww = conv_weight.stride()
    s_on, s_oc, s_oh, s_ow = out.stride()

    # Launch configuration
    BLOCK_W = 16
    NW_TILES = triton.cdiv(W_out, BLOCK_W)
    grid = (N, C_out, H_out * NW_TILES)

    _conv_relu_maxpool2x2_nchw_fused[grid](
        x, conv_weight, conv_bias, out,
        N, C_in, H, W,
        C_out, KH, KW,
        H_out, W_out,
        s_xn, s_xc, s_xh, s_xw,
        s_wn, s_wc, s_wh, s_ww,
        s_on, s_oc, s_oh, s_ow,
        NW_TILES=NW_TILES,
        BLOCK_W=BLOCK_W,
        num_warps=4,
        num_stages=2,
    )
    return out


# ============================
# Fused Conv(5x5, s=1, p=0) + ReLU + MaxPool2d(2x2, s=2)
# Kernel variant 2 (used for conv2)
# ============================
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

    oob = (pid_n >= N) | (pid_oc >= C_OUT) | (py >= P_H) | (px >= P_W)

    # Top-left conv output coordinates for the 2x2 maxpool window
    y0 = 2 * py
    x0 = 2 * px

    # Four accumulators for the 2x2 conv outputs (fp32 accumulation)
    acc00 = tl.zeros((), dtype=tl.float32)
    acc01 = tl.zeros((), dtype=tl.float32)
    acc10 = tl.zeros((), dtype=tl.float32)
    acc11 = tl.zeros((), dtype=tl.float32)

    # Compute convolution for four neighboring output points simultaneously
    for ci in range(CIN):
        base_ic = pid_n * x_stride_n + ci * x_stride_c
        for ky in range(KH):
            yi = y0 + ky
            base_y = base_ic + yi * x_stride_h
            for kx in range(KW):
                xi = x0 + kx
                w_off = pid_oc * w_stride_o + ci * w_stride_i + ky * w_stride_h + kx * w_stride_w
                w_val = tl.load(w_ptr + w_off)
                w_f32 = w_val.to(tl.float32)
                base_xy = base_y + xi * x_stride_w
                i00 = tl.load(x_ptr + base_xy)
                i01 = tl.load(x_ptr + base_xy + x_stride_w)
                i10 = tl.load(x_ptr + base_xy + x_stride_h)
                i11 = tl.load(x_ptr + base_xy + x_stride_h + x_stride_w)
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


def conv_relu_maxpool_fused2(x, conv_weight, conv_bias):
    """
    Fused Triton kernel: conv2d (5x5, stride=1, no padding) + bias + ReLU + maxpool2d (2x2, stride=2).
    Used for the second convolutional block (6->16 channels).
    """
    assert isinstance(x, torch.Tensor) and isinstance(conv_weight, torch.Tensor) and isinstance(conv_bias, torch.Tensor)
    assert x.is_cuda and conv_weight.is_cuda and conv_bias.is_cuda
    N, C_in, H, W = x.shape
    C_out, C_in_w, KH, KW = conv_weight.shape
    assert C_in == C_in_w
    assert KH == 5 and KW == 5

    H_conv = H - KH + 1
    W_conv = W - KW + 1
    assert H_conv > 0 and W_conv > 0
    pool_stride = 2
    assert H_conv % pool_stride == 0 and W_conv % pool_stride == 0
    P_H = H_conv // pool_stride
    P_W = W_conv // pool_stride

    # Output dtype same as input
    y = torch.empty((N, C_out, P_H, P_W), device=x.device, dtype=x.dtype)

    # Strides
    x_sn, x_sc, x_sh, x_sw = x.stride()
    w_so, w_si, w_sh, w_sw = conv_weight.stride()
    y_sn, y_sc, y_sh, y_sw = y.stride()

    grid = (N, C_out, P_H * P_W)

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


# ============================
# Flatten NCHW -> [N, C*H*W]
# ============================
@triton.jit
def _flatten_nchw_to_nm_kernel(
    in_ptr, out_ptr,
    N, C, H, W,
    stride_n, stride_c, stride_h, stride_w,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Generic NCHW -> [N, C*H*W] flatten kernel.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    CHW = C * H * W
    HW = H * W

    n = offs // CHW
    rem0 = offs % CHW
    c = rem0 // HW
    rem1 = rem0 % HW
    h = rem1 // W
    w = rem1 % W

    in_idx = n * stride_n + c * stride_c + h * stride_h + w * stride_w

    vals = tl.load(in_ptr + in_idx, mask=mask)
    tl.store(out_ptr + offs, vals, mask=mask)


def flatten_nchw(x: torch.Tensor) -> torch.Tensor:
    """
    Flatten NCHW tensor to [N, C*H*W] using a Triton kernel.
    """
    assert x.is_cuda and x.dim() == 4
    N, C, H, W = x.shape
    M = C * H * W
    n_elements = N * M
    out = torch.empty((N, M), device=x.device, dtype=x.dtype)
    sN, sC, sH, sW = x.stride()

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    _flatten_nchw_to_nm_kernel[grid](
        x, out,
        N, C, H, W,
        sN, sC, sH, sW,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


# ============================
# Fused Linear + Bias + ReLU
# ============================
_configs_linear_relu = [
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=4),
    triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=8, num_stages=4),
]


@triton.autotune(configs=_configs_linear_relu, key=['M', 'N', 'K'])
@triton.jit
def _linear_bias_relu_kernel(
    x_ptr,          # *dtype, [M, K]
    w_ptr,          # *dtype, [N, K] logically as weight.T with custom strides
    b_ptr,          # *dtype, [N]
    y_ptr,          # *dtype, [M, N]
    M, N, K,        # int
    stride_xm, stride_xk,      # strides of x
    stride_wk, stride_wn,      # strides treating w as [K, N]
    stride_ym, stride_yn,      # strides of y
    BLOCK_M: tl.constexpr,     # tile M
    BLOCK_N: tl.constexpr,     # tile N
    BLOCK_K: tl.constexpr,     # tile K
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    mask_m = offs_m < M
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        k_idxs = k0 + offs_k
        mask_k = k_idxs < K

        x_ptrs = x_ptr + (offs_m[:, None] * stride_xm + k_idxs[None, :] * stride_xk)
        w_ptrs = w_ptr + (k_idxs[:, None] * stride_wk + offs_n[None, :] * stride_wn)

        a = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        b = tl.load(w_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)

        acc = tl.dot(a, b, acc)

    bias = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
    acc = acc + bias[None, :]
    acc = tl.maximum(acc, 0.0)

    y_ptrs = y_ptr + (offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn)
    out_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(y_ptrs, acc.to(y_ptr.dtype.element_ty), mask=out_mask)


def linear_bias_relu(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Fused linear + bias + ReLU.
    Computes y = ReLU(x @ weight.T + bias)
    x: [M, K], weight: [N, K], bias: [N]
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda
    assert x.ndim == 2 and weight.ndim == 2 and bias.ndim == 1
    M, Kx = x.shape
    Nw, Kw = weight.shape
    assert Kx == Kw
    assert bias.shape[0] == Nw

    out = torch.empty((M, Nw), device=x.device, dtype=x.dtype)

    stride_xm, stride_xk = x.stride()
    stride_wn, stride_wk = weight.stride()  # [N, K] -> (stride_N, stride_K)
    stride_ym, stride_yn = out.stride()

    def grid(meta):
        return (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(Nw, meta['BLOCK_N']))

    _linear_bias_relu_kernel[grid](
        x, weight, bias, out,
        M, Nw, Kx,
        stride_xm, stride_xk,
        stride_wk, stride_wn,   # pass strides as if weight is [K, N]
        stride_ym, stride_yn,
    )
    return out


# ============================
# Fused Linear + Bias (no activation)
# ============================
@triton.jit
def _linear_bias_fwd(
    x_ptr,        # [M, K] input
    w_ptr,        # [N, K] weights (we use w.T in the matmul)
    b_ptr,        # [N] bias
    y_ptr,        # [M, N] output
    M, K, N,      # dimensions
    stride_xm, stride_xk,   # x strides
    stride_wn, stride_wk,   # w strides (N, K)
    stride_ym, stride_yn,   # y strides
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Fused Linear Forward: y = x @ w.T + b
    - x: [M, K]
    - w: [N, K]
    - b: [N]
    - y: [M, N]
    Accumulates in float32, stores to y dtype.
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    k_tiles = tl.cdiv(K, BLOCK_K)
    for kt in range(k_tiles):
        k_start = kt * BLOCK_K
        k_idx = k_start + offs_k

        x_ptrs = x_ptr + (offs_m[:, None] * stride_xm + k_idx[None, :] * stride_xk)
        w_ptrs = w_ptr + (k_idx[:, None] * stride_wk + offs_n[None, :] * stride_wn)

        x = tl.load(x_ptrs, mask=(offs_m[:, None] < M) & (k_idx[None, :] < K), other=0.0)
        w_t = tl.load(w_ptrs, mask=(k_idx[:, None] < K) & (offs_n[None, :] < N), other=0.0)

        acc = tl.dot(x, w_t, acc)

    b = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc += b[None, :].to(tl.float32)

    y_ptrs = y_ptr + (offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn)
    y_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(y_ptrs, acc.to(y_ptr.dtype.element_ty), mask=y_mask)


def linear_bias(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Fused linear: y = x @ w.T + b
    x: [M, K], w: [N, K], b: [N] -> y: [M, N]
    """
    assert x.is_cuda and w.is_cuda and b.is_cuda
    assert x.ndim == 2 and w.ndim == 2 and b.ndim == 1
    M, Kx = x.shape
    Nw, Kw = w.shape
    assert Kx == Kw and b.shape[0] == Nw

    y = torch.empty((M, Nw), device=x.device, dtype=x.dtype)

    stride_xm, stride_xk = x.stride()
    stride_wn, stride_wk = w.stride()
    stride_ym, stride_yn = y.stride()

    BLOCK_M = 128
    BLOCK_N = 32
    BLOCK_K = 32

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(Nw, BLOCK_N))

    _linear_bias_fwd[grid](
        x, w, b, y,
        M, Kx, Nw,
        stride_xm, stride_xk,
        stride_wn, stride_wk,
        stride_ym, stride_yn,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=3,
    )
    return y


# ============================
# Top-level end-to-end Triton wrapper
# ============================
def kernel_function(
    x: torch.Tensor,
    conv1_weight: torch.Tensor, conv1_bias: torch.Tensor,
    conv2_weight: torch.Tensor, conv2_bias: torch.Tensor,
    fc1_weight: torch.Tensor, fc1_bias: torch.Tensor,
    fc2_weight: torch.Tensor, fc2_bias: torch.Tensor,
    fc3_weight: torch.Tensor, fc3_bias: torch.Tensor,
) -> torch.Tensor:
    """
    End-to-end Triton implementation of the LeNet-5 forward pass for the given shapes:
      - Input: x [N, 1, 32, 32] (N=4096 for the test)
      - conv1 (5x5, s=1) + ReLU + maxpool(2x2, s=2) -> [N, 6, 14, 14]
      - conv2 (5x5, s=1) + ReLU + maxpool(2x2, s=2) -> [N, 16, 5, 5]
      - flatten -> [N, 400]
      - fc1 + ReLU -> [N, 120]
      - fc2 + ReLU -> [N, 84]
      - fc3 -> [N, 20]

    All computations are performed by Triton kernels; no PyTorch math ops are used.
    """
    # Basic validation and device checks
    tensors = [x, conv1_weight, conv1_bias, conv2_weight, conv2_bias,
               fc1_weight, fc1_bias, fc2_weight, fc2_bias, fc3_weight, fc3_bias]
    devices = {t.device for t in tensors}
    assert len(devices) == 1, "All tensors must be on the same device."
    assert x.is_cuda, "CUDA device required."
    # Ensure dtypes are consistent (float32 preferred)
    dtype = x.dtype
    for t in tensors[1:]:
        assert t.dtype == dtype, "All tensors must share the same dtype as input."

    # Stage 1: conv1 + relu + pool
    y1 = conv_relu_maxpool_fused1(x, conv1_weight, conv1_bias)  # [N, 6, 14, 14], fp32

    # Stage 2: conv2 + relu + pool
    y2 = conv_relu_maxpool_fused2(y1, conv2_weight, conv2_bias)  # [N, 16, 5, 5]

    # Flatten
    y3 = flatten_nchw(y2)  # [N, 400]

    # FC1 + ReLU
    y4 = linear_bias_relu(y3, fc1_weight, fc1_bias)  # [N, 120]

    # FC2 + ReLU
    y5 = linear_bias_relu(y4, fc2_weight, fc2_bias)  # [N, 84]

    # FC3 (no activation)
    out = linear_bias(y5, fc3_weight, fc3_bias)  # [N, 20]
    return out


# ============================
# Reference PyTorch model (from the original problem)
# ============================
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, num_classes):
        """
        LeNet-5 architecture implementation in PyTorch.
        """
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        self.fc1 = nn.Linear(in_features=16 * 5 * 5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


batch_size = 4096
num_classes = 20


def get_inputs():
    return [torch.rand(batch_size, 1, 32, 32)]


def get_init_inputs():
    return [num_classes]


# ============================
# Self-test
# ============================
def run_tests():
    torch.manual_seed(0)
    device = torch.device('cuda')
    # Initialize model and data
    num_classes = get_init_inputs()[0]
    model = Model(num_classes).to(device).eval()

    x = get_inputs()[0].to(device).to(torch.float32).contiguous()

    with torch.no_grad():
        ref = model(x)

    # Extract weights/biases
    conv1_w = model.conv1.weight.detach().contiguous()
    conv1_b = model.conv1.bias.detach().contiguous()
    conv2_w = model.conv2.weight.detach().contiguous()
    conv2_b = model.conv2.bias.detach().contiguous()

    fc1_w = model.fc1.weight.detach().contiguous()
    fc1_b = model.fc1.bias.detach().contiguous()
    fc2_w = model.fc2.weight.detach().contiguous()
    fc2_b = model.fc2.bias.detach().contiguous()
    fc3_w = model.fc3.weight.detach().contiguous()
    fc3_b = model.fc3.bias.detach().contiguous()

    # Triton path
    out = kernel_function(
        x,
        conv1_w, conv1_b,
        conv2_w, conv2_b,
        fc1_w, fc1_b,
        fc2_w, fc2_b,
        fc3_w, fc3_b,
    )

    # Compare
    atol = 1e-3
    rtol = 1e-3
    ok = torch.allclose(out, ref, rtol=rtol, atol=atol)
    if not ok:
        max_abs = (out - ref).abs().max().item()
        max_rel = ((out - ref).abs() / (ref.abs() + 1e-8)).max().item()
        print("Mismatch!")
        print(f"Max abs diff: {max_abs:.6e}, Max rel diff: {max_rel:.6e}")
        sys.exit(1)
    print("PASS")
    sys.exit(0)


if __name__ == "__main__":
    run_tests()
