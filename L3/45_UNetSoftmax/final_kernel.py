# kernel.py
# Triton-based implementation of a U-Net-like forward with DoubleConv blocks (Conv3x3 + BN + Softmax)
# All math is done inside Triton kernels; the Python wrapper only allocates and launches.
import torch
import triton
import triton.language as tl


# ============================
# 3x3 Conv + BN + Softmax (dim=-1)
# ============================
@triton.jit
def _conv3_bn_sm_row_kernel(
    x_ptr, w_ptr, b_ptr, gamma_ptr, beta_ptr, mean_ptr, var_ptr, y_ptr,
    N, Cin, H, W, Cout,
    sXN, sXC, sXH, sXW,
    sWO, sWI, sWKH, sWKW,
    sYN, sYC, sYH, sYW,
    EPS: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid_w = tl.program_id(axis=0)
    pid_r = tl.program_id(axis=1)

    row_size = Cout * H
    n = pid_r // row_size
    rem = pid_r % row_size
    oc = rem // H
    h = rem % H

    w_start = pid_w * BLOCK_W
    offs_w = w_start + tl.arange(0, BLOCK_W)
    mask_w = offs_w < W

    acc = tl.zeros((BLOCK_W,), dtype=tl.float32)

    # bias load
    b = tl.load(b_ptr + oc).to(tl.float32)

    # 3x3 conv with padding=1
    for ic in tl.range(0, Cin):
        for kh in range(3):
            ih = h + kh - 1
            h_ok = (ih >= 0) & (ih < H)
            for kw in range(3):
                iw = offs_w + kw - 1
                m = mask_w & h_ok & (iw >= 0) & (iw < W)
                x_ptrs = x_ptr + n * sXN + ic * sXC + ih * sXH + iw * sXW
                x_vals = tl.load(x_ptrs, mask=m, other=0.0).to(tl.float32)
                w_val = tl.load(w_ptr + (oc * sWO + ic * sWI + kh * sWKH + kw * sWKW)).to(tl.float32)
                acc += x_vals * w_val

    acc += b

    # BatchNorm2d (eval): y = (x - mean) / sqrt(var + eps) * gamma + beta
    mean = tl.load(mean_ptr + oc).to(tl.float32)
    var = tl.load(var_ptr + oc).to(tl.float32)
    gamma = tl.load(gamma_ptr + oc).to(tl.float32)
    beta = tl.load(beta_ptr + oc).to(tl.float32)
    inv_std = 1.0 / tl.sqrt(var + EPS)
    bn_out = (acc - mean) * inv_std
    bn_out = bn_out * gamma + beta

    # Softmax along last dim (width)
    # Stability: subtract row max
    neg_large = tl.full((BLOCK_W,), -1.0e30, dtype=tl.float32)
    masked_vals = tl.where(mask_w, bn_out, neg_large)
    row_max = tl.max(masked_vals, axis=0)
    shifted = masked_vals - row_max
    ex = tl.exp(shifted)
    ex = tl.where(mask_w, ex, 0.0)
    denom = tl.sum(ex, axis=0)
    sm = ex / denom

    y_ptrs = y_ptr + n * sYN + oc * sYC + h * sYH + offs_w * sYW
    tl.store(y_ptrs, sm.to(y_ptr.dtype.element_ty), mask=mask_w)


def conv3_bn_softmax(x, w, b, gamma, beta, running_mean, running_var, eps=1e-5, block_w=512):
    # x: [N,Cin,H,W], w: [Cout,Cin,3,3]
    assert x.is_cuda and w.is_cuda
    N, Cin, H, W = x.shape
    Cout = w.shape[0]
    y = torch.empty((N, Cout, H, W), device=x.device, dtype=x.dtype)

    grid = (triton.cdiv(W, block_w), N * Cout * H)
    _conv3_bn_sm_row_kernel[grid](
        x, w, b, gamma, beta, running_mean, running_var, y,
        N, Cin, H, W, Cout,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        w.stride(0), w.stride(1), w.stride(2), w.stride(3),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        EPS=eps, BLOCK_W=block_w
    )
    return y


# ============================
# 1x1 Conv (no BN / no Softmax)
# ============================
@triton.jit
def _conv1x1_row_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    N, Cin, H, W, Cout,
    sXN, sXC, sXH, sXW,
    sWO, sWI, sWKH, sWKW,  # still pass 4D strides for consistency (KH=KW=1)
    sYN, sYC, sYH, sYW,
    BLOCK_W: tl.constexpr,
):
    pid_w = tl.program_id(axis=0)
    pid_r = tl.program_id(axis=1)

    row_size = Cout * H
    n = pid_r // row_size
    rem = pid_r % row_size
    oc = rem // H
    h = rem % H

    w_start = pid_w * BLOCK_W
    offs_w = w_start + tl.arange(0, BLOCK_W)
    mask_w = offs_w < W

    acc = tl.zeros((BLOCK_W,), dtype=tl.float32)
    b = tl.load(b_ptr + oc).to(tl.float32)

    for ic in tl.range(0, Cin):
        x_ptrs = x_ptr + n * sXN + ic * sXC + h * sXH + offs_w * sXW
        x_vals = tl.load(x_ptrs, mask=mask_w, other=0.0).to(tl.float32)
        w_val = tl.load(w_ptr + (oc * sWO + ic * sWI)).to(tl.float32)
        acc += x_vals * w_val

    acc += b
    y_ptrs = y_ptr + n * sYN + oc * sYC + h * sYH + offs_w * sYW
    tl.store(y_ptrs, acc.to(y_ptr.dtype.element_ty), mask=mask_w)


def conv1x1(x, w, b, block_w=512):
    # x: [N,Cin,H,W], w: [Cout,Cin,1,1]
    N, Cin, H, W = x.shape
    Cout = w.shape[0]
    y = torch.empty((N, Cout, H, W), device=x.device, dtype=x.dtype)
    grid = (triton.cdiv(W, block_w), N * Cout * H)
    _conv1x1_row_kernel[grid](
        x, w, b, y,
        N, Cin, H, W, Cout,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        w.stride(0), w.stride(1), w.stride(2), w.stride(3),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        BLOCK_W=block_w
    )
    return y


# ============================
# MaxPool2d kernel_size=2, stride=2
# ============================
@triton.jit
def _maxpool2x2_row_kernel(
    x_ptr, y_ptr,
    N, C, H_in, W_in, H_out, W_out,
    sXN, sXC, sXH, sXW,
    sYN, sYC, sYH, sYW,
    BLOCK_W: tl.constexpr,
):
    pid_w = tl.program_id(axis=0)
    pid_r = tl.program_id(axis=1)

    row_size = C * H_out
    n = pid_r // row_size
    rem = pid_r % row_size
    c = rem // H_out
    ho = rem % H_out

    w_start = pid_w * BLOCK_W
    offs_wo = w_start + tl.arange(0, BLOCK_W)
    mask_w = offs_wo < W_out

    hi0 = 2 * ho
    wi0 = 2 * offs_wo

    x00_ptrs = x_ptr + n * sXN + c * sXC + (hi0 + 0) * sXH + (wi0 + 0) * sXW
    x01_ptrs = x_ptr + n * sXN + c * sXC + (hi0 + 0) * sXH + (wi0 + 1) * sXW
    x10_ptrs = x_ptr + n * sXN + c * sXC + (hi0 + 1) * sXH + (wi0 + 0) * sXW
    x11_ptrs = x_ptr + n * sXN + c * sXC + (hi0 + 1) * sXH + (wi0 + 1) * sXW

    # Inside bounds guards not necessary for even shapes here, but keep mask for safety
    valid = mask_w  # shapes are exact halves in the test
    x00 = tl.load(x00_ptrs, mask=valid, other=-1.0e30).to(tl.float32)
    x01 = tl.load(x01_ptrs, mask=valid, other=-1.0e30).to(tl.float32)
    x10 = tl.load(x10_ptrs, mask=valid, other=-1.0e30).to(tl.float32)
    x11 = tl.load(x11_ptrs, mask=valid, other=-1.0e30).to(tl.float32)

    m0 = tl.maximum(x00, x01)
    m1 = tl.maximum(x10, x11)
    m = tl.maximum(m0, m1)

    y_ptrs = y_ptr + n * sYN + c * sYC + ho * sYH + offs_wo * sYW
    tl.store(y_ptrs, m.to(y_ptr.dtype.element_ty), mask=mask_w)


def maxpool2x2(x, block_w=512):
    N, C, H_in, W_in = x.shape
    assert H_in % 2 == 0 and W_in % 2 == 0
    H_out, W_out = H_in // 2, W_in // 2
    y = torch.empty((N, C, H_out, W_out), device=x.device, dtype=x.dtype)
    grid = (triton.cdiv(W_out, block_w), N * C * H_out)
    _maxpool2x2_row_kernel[grid](
        x, y,
        N, C, H_in, W_in, H_out, W_out,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        BLOCK_W=block_w
    )
    return y


# ============================
# ConvTranspose2d k=2, stride=2, padding=0
# For this configuration, each output element depends on exactly one kernel offset:
# kh = h_out % 2, kw = w_out % 2, and input location is (h_out//2, w_out//2)
# Weight layout: [Cin, Cout, 2, 2] (PyTorch)
# ============================
@triton.jit
def _deconv2x2s2_row_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    N, Cin, H_in, W_in, Cout, H_out, W_out,
    sXN, sXC, sXH, sXW,
    sWO, sWI, sWKH, sWKW,  # For convtranspose: O==Cin, I==Cout here (using generic names)
    sYN, sYC, sYH, sYW,
    BLOCK_W: tl.constexpr,
):
    pid_w = tl.program_id(axis=0)
    pid_r = tl.program_id(axis=1)

    row_size = Cout * H_out
    n = pid_r // row_size
    rem = pid_r % row_size
    oc = rem // H_out
    ho = rem % H_out

    w_start = pid_w * BLOCK_W
    offs_wo = w_start + tl.arange(0, BLOCK_W)
    mask_w = offs_wo < W_out

    # map output positions to input positions (h_out -> h_in, w_out -> w_in)
    hi = ho // 2
    kh = ho % 2
    wi = offs_wo // 2
    kw_mask = (offs_wo % 2) == 1  # kw=1 if True else 0

    acc = tl.zeros((BLOCK_W,), dtype=tl.float32)
    b = tl.load(b_ptr + oc).to(tl.float32)
    acc += b

    for ic in tl.range(0, Cin):
        x_ptrs = x_ptr + n * sXN + ic * sXC + hi * sXH + wi * sXW
        x_vals = tl.load(x_ptrs, mask=mask_w, other=0.0).to(tl.float32)
        # two possible kw per lane; pick via kw_mask
        w0 = tl.load(w_ptr + (ic * sWO + oc * sWI + kh * sWKH + 0 * sWKW)).to(tl.float32)
        w1 = tl.load(w_ptr + (ic * sWO + oc * sWI + kh * sWKH + 1 * sWKW)).to(tl.float32)
        wv = tl.where(kw_mask, w1, w0)
        acc += x_vals * wv

    y_ptrs = y_ptr + n * sYN + oc * sYC + ho * sYH + offs_wo * sYW
    tl.store(y_ptrs, acc.to(y_ptr.dtype.element_ty), mask=mask_w)


def deconv2x2s2(x, w, b, block_w=512):
    # x: [N,Cin,H_in,W_in], w: [Cin,Cout,2,2]
    N, Cin, H_in, W_in = x.shape
    Cout = w.shape[1]
    H_out, W_out = H_in * 2, W_in * 2
    y = torch.empty((N, Cout, H_out, W_out), device=x.device, dtype=x.dtype)
    grid = (triton.cdiv(W_out, block_w), N * Cout * H_out)
    _deconv2x2s2_row_kernel[grid](
        x, w, b, y,
        N, Cin, H_in, W_in, Cout, H_out, W_out,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        w.stride(0), w.stride(1), w.stride(2), w.stride(3),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        BLOCK_W=block_w
    )
    return y


# ============================
# Channel-wise Concatenation via Copy with Channel Offset
# ============================
@triton.jit
def _cat_copy_row_kernel(
    src_ptr, dst_ptr,
    N, C, H, W, c_offset,
    sSN, sSC, sSH, sSW,
    sDN, sDC, sDH, sDW,
    BLOCK_W: tl.constexpr,
):
    pid_w = tl.program_id(axis=0)
    pid_r = tl.program_id(axis=1)

    row_size = C * H
    n = pid_r // row_size
    rem = pid_r % row_size
    c = rem // H
    h = rem % H

    w_start = pid_w * BLOCK_W
    offs_w = w_start + tl.arange(0, BLOCK_W)
    mask_w = offs_w < W

    src_ptrs = src_ptr + n * sSN + c * sSC + h * sSH + offs_w * sSW
    vals = tl.load(src_ptrs, mask=mask_w, other=0.0)
    dst_ptrs = dst_ptr + n * sDN + (c + c_offset) * sDC + h * sDH + offs_w * sDW
    tl.store(dst_ptrs, vals, mask=mask_w)


def cat_channels(x1, x2, block_w=512):
    # Concatenate along channel dimension (N, C1 + C2, H, W) = cat([x1, x2], dim=1)
    assert x1.shape[0] == x2.shape[0] and x1.shape[2:] == x2.shape[2:]
    N, C1, H, W = x1.shape
    C2 = x2.shape[1]
    y = torch.empty((N, C1 + C2, H, W), device=x1.device, dtype=x1.dtype)

    grid1 = (triton.cdiv(W, block_w), N * C1 * H)
    _cat_copy_row_kernel[grid1](
        x1, y,
        N, C1, H, W, 0,
        x1.stride(0), x1.stride(1), x1.stride(2), x1.stride(3),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        BLOCK_W=block_w
    )
    grid2 = (triton.cdiv(W, block_w), N * C2 * H)
    _cat_copy_row_kernel[grid2](
        x2, y,
        N, C2, H, W, C1,
        x2.stride(0), x2.stride(1), x2.stride(2), x2.stride(3),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        BLOCK_W=block_w
    )
    return y


# ============================
# Helpers to extract parameters from the provided state_dict
# ============================
def _dc_params(state, prefix):
    # DoubleConv params under f"{prefix}.double_conv"
    p = f"{prefix}.double_conv"
    conv1_w = state[f"{p}.0.weight"]
    conv1_b = state[f"{p}.0.bias"]
    bn1_w = state[f"{p}.1.weight"]
    bn1_b = state[f"{p}.1.bias"]
    bn1_mean = state[f"{p}.1.running_mean"]
    bn1_var = state[f"{p}.1.running_var"]

    conv2_w = state[f"{p}.3.weight"]
    conv2_b = state[f"{p}.3.bias"]
    bn2_w = state[f"{p}.4.weight"]
    bn2_b = state[f"{p}.4.bias"]
    bn2_mean = state[f"{p}.4.running_mean"]
    bn2_var = state[f"{p}.4.running_var"]

    return (conv1_w, conv1_b, bn1_w, bn1_b, bn1_mean, bn1_var,
            conv2_w, conv2_b, bn2_w, bn2_b, bn2_mean, bn2_var)


def _up_params(state, name):
    # ConvTranspose2d params: weight [Cin, Cout, 2, 2], bias [Cout]
    w = state[f"{name}.weight"]
    b = state[f"{name}.bias"]
    return w, b


def _final_params(state):
    w = state["final_conv.weight"]
    b = state["final_conv.bias"]
    return w, b


# ============================
# Orchestration: forward pass
# ============================
def _double_conv_block(x, params, eps=1e-5):
    (w1, b1, g1, be1, m1, v1,
     w2, b2, g2, be2, m2, v2) = params
    y = conv3_bn_softmax(x, w1, b1, g1, be1, m1, v1, eps=eps)
    y = conv3_bn_softmax(y, w2, b2, g2, be2, m2, v2, eps=eps)
    return y


def _forward_unet(x, state):
    # Encoder
    enc1 = _double_conv_block(x, _dc_params(state, "encoder1"))
    p1 = maxpool2x2(enc1)
    enc2 = _double_conv_block(p1, _dc_params(state, "encoder2"))
    p2 = maxpool2x2(enc2)
    enc3 = _double_conv_block(p2, _dc_params(state, "encoder3"))
    p3 = maxpool2x2(enc3)
    enc4 = _double_conv_block(p3, _dc_params(state, "encoder4"))
    p4 = maxpool2x2(enc4)

    # Bottleneck
    bottleneck = _double_conv_block(p4, _dc_params(state, "bottleneck"))

    # Decoder
    up4_w, up4_b = _up_params(state, "upconv4")
    dec4 = deconv2x2s2(bottleneck, up4_w, up4_b)
    dec4 = cat_channels(dec4, enc4)
    dec4 = _double_conv_block(dec4, _dc_params(state, "decoder4"))

    up3_w, up3_b = _up_params(state, "upconv3")
    dec3 = deconv2x2s2(dec4, up3_w, up3_b)
    dec3 = cat_channels(dec3, enc3)
    dec3 = _double_conv_block(dec3, _dc_params(state, "decoder3"))

    up2_w, up2_b = _up_params(state, "upconv2")
    dec2 = deconv2x2s2(dec3, up2_w, up2_b)
    dec2 = cat_channels(dec2, enc2)
    dec2 = _double_conv_block(dec2, _dc_params(state, "decoder2"))

    up1_w, up1_b = _up_params(state, "upconv1")
    dec1 = deconv2x2s2(dec2, up1_w, up1_b)
    dec1 = cat_channels(dec1, enc1)
    dec1 = _double_conv_block(dec1, _dc_params(state, "decoder1"))

    # Final 1x1 conv to out_channels
    fw, fb = _final_params(state)
    out = conv1x1(dec1, fw, fb)
    return out


# ============================
# Public API
# ============================
def kernel_function(x, state_dict):
    """
    Execute the full forward pass of the provided U-Net-like model using Triton kernels.

    Args:
      x: Input tensor of shape [N, C_in, H, W], on CUDA, dtype bf16 preferred.
      state_dict: A PyTorch state dict (tensors on same device/dtype) containing all model weights/buffers.

    Returns:
      Output tensor [N, out_channels, H, W], same device/dtype as x.
    """
    # Validation
    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a torch.Tensor")
    if not x.is_cuda:
        raise ValueError("x must be on CUDA device")
    if not x.is_contiguous():
        x = x.contiguous()
    # Basic dtype/device checks on state_dict
    if not isinstance(state_dict, dict):
        raise TypeError("state_dict must be a dict of tensors")
    dev = x.device
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor):
            if v.device != dev:
                raise ValueError(f"State tensor {k} is on {v.device}, expected {dev}")
            # ensure contiguous
            if not v.is_contiguous():
                state_dict[k] = v.contiguous()

    # Run the network
    with torch.no_grad():
        y = _forward_unet(x, state_dict)
    return y