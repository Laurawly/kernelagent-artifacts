import torch
import triton
import triton.language as tl


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


def kernel_function(x: torch.Tensor, conv_weight: torch.Tensor, conv_bias: torch.Tensor):
    """
    Fused Conv2d (5x5, stride=1, pad=0, bias) + ReLU + MaxPool2d (2x2, stride=2) implemented in Triton.

    Fusion rationale:
    - We directly compute the pooled outputs without materializing the intermediate 28x28 conv map.
    - For each pooled output position (oh, ow), we accumulate four neighboring conv outputs
      (corresponding to the 2x2 pooling window at stride=2) in fp32, add bias, apply ReLU,
      and then take the max across those four values.
    - This removes the need to write/read the 28x28 conv activation to/from global memory,
      reduces bandwidth, and eliminates two separate kernel launches.

    Runtime behavior:
    - The wrapper performs only validation, allocation, and Triton launch configuration.
    - All math (convolution, bias-add, ReLU, pooling) is inside the Triton kernel.
    """
    # Basic validation (no compute here)
    assert x.is_cuda and conv_weight.is_cuda and conv_bias.is_cuda, "All tensors must be on CUDA device."
    assert x.ndim == 4 and x.shape[1] == conv_weight.shape[1], "Input/weight channel mismatch."
    assert conv_weight.ndim == 4, "Weight must be 4D [C_out, C_in, KH, KW]."
    assert conv_bias.ndim == 1 and conv_bias.shape[0] == conv_weight.shape[0], "Bias shape must match C_out."

    N, C_in, H, W = x.shape
    C_out, C_in_w, KH, KW = conv_weight.shape
    assert C_in == C_in_w, "C_in mismatch."

    # Conv out (no padding, stride=1)
    H_conv = H - KH + 1
    W_conv = W - KW + 1
    assert H_conv > 0 and W_conv > 0, "Invalid convolution geometry."

    # MaxPool 2x2, stride 2, pad 0
    H_out = (H_conv - 2) // 2 + 1
    W_out = (W_conv - 2) // 2 + 1

    # Allocate output in fp32 for stable accuracy
    out = torch.empty((N, C_out, H_out, W_out), device=x.device, dtype=torch.float32)

    # Compute strides in elements
    s_xn, s_xc, s_xh, s_xw = x.stride()
    s_wn, s_wc, s_wh, s_ww = conv_weight.stride()
    s_on, s_oc, s_oh, s_ow = out.stride()

    # Launch configuration
    # Tile across width of pooled output. With W_out=14, BLOCK_W=16 results in a single tile.
    BLOCK_W = 16
    NW_TILES = triton.cdiv(W_out, BLOCK_W)

    grid = (N, C_out, H_out * NW_TILES)

    # Launch Triton kernel
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