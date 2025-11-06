# kernel.py
import torch
import triton
import triton.language as tl


@triton.jit
def _conv3x3_bn_relu_s2_kernel(
    x_ptr,               # *bf16  [N, Cin, Hin, Win]
    w_ptr,               # *bf16  [Cout, Cin, 3, 3]
    bn_w_ptr,            # *bf16  [Cout]
    bn_b_ptr,            # *bf16  [Cout]
    bn_mean_ptr,         # *bf16  [Cout]
    bn_var_ptr,          # *bf16  [Cout]
    out_ptr,             # *fp32  [N, Cout, Hout, Wout]  (intermediate activation)
    N, Cin, Hin, Win,
    Cout, Hout, Wout,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wo, stride_wi, stride_wky, stride_wkx,
    stride_on, stride_oc, stride_oh, stride_ow,
    eps: tl.constexpr,   # compile-time constant epsilon used by BN
    BLOCK_OC: tl.constexpr,
):
    # Program IDs
    pid_hw = tl.program_id(0)
    pid_oc = tl.program_id(1)

    # Derive (n, oh, ow)
    HW = Hout * Wout
    n = pid_hw // HW
    hw = pid_hw % HW
    oh = hw // Wout
    ow = hw % Wout

    # Select OC tile
    oc_start = pid_oc * BLOCK_OC
    oc_offsets = oc_start + tl.arange(0, BLOCK_OC)
    oc_mask = oc_offsets < Cout

    # Base input coordinate for stride=2, pad=1
    base_iy = oh * 2 - 1
    base_ix = ow * 2 - 1

    # Accumulator in fp32 for a vector of OC
    acc = tl.zeros((BLOCK_OC,), dtype=tl.float32)

    # Convolution 3x3 over input channels
    for ic in tl.range(0, Cin):
        for ky in range(0, 3):
            iy = base_iy + ky
            y_in_bounds = (iy >= 0) & (iy < Hin)
            for kx in range(0, 3):
                ix = base_ix + kx
                in_bounds = y_in_bounds & (ix >= 0) & (ix < Win)

                # Load input scalar (bf16 -> fp32), masked for borders
                x_offset = n * stride_xn + ic * stride_xc + iy * stride_xh + ix * stride_xw
                x_val = tl.load(x_ptr + x_offset, mask=in_bounds, other=0).to(tl.float32)

                # Load weight vector across OC tile (bf16 -> fp32)
                w_ptrs = w_ptr + oc_offsets * stride_wo + ic * stride_wi + ky * stride_wky + kx * stride_wkx
                w_vec = tl.load(w_ptrs, mask=oc_mask, other=0).to(tl.float32)

                acc += w_vec * x_val

    # BatchNorm inference: scale = gamma / sqrt(var + eps), shift = beta - mean*scale
    gamma = tl.load(bn_w_ptr + oc_offsets, mask=oc_mask, other=0).to(tl.float32)
    beta = tl.load(bn_b_ptr + oc_offsets, mask=oc_mask, other=0).to(tl.float32)
    mean = tl.load(bn_mean_ptr + oc_offsets, mask=oc_mask, other=0).to(tl.float32)
    var = tl.load(bn_var_ptr + oc_offsets, mask=oc_mask, other=0).to(tl.float32)

    denom = tl.sqrt(var + eps)
    scale = gamma / denom
    shift = beta - mean * scale

    y = acc * scale + shift
    y = tl.maximum(y, 0)  # ReLU

    # Store intermediate activation in fp32
    out_ptrs = out_ptr + n * stride_on + oc_offsets * stride_oc + oh * stride_oh + ow * stride_ow
    tl.store(out_ptrs, y, mask=oc_mask)


@triton.jit
def _conv3x3_bn_add_skip1x1_bn_relu_kernel(
    x_ptr,                # *bf16  [N, Cin, Hin, Win]
    act_ptr,              # *fp32  [N, Cmid, Hout, Wout] from previous kernel
    w2_ptr,               # *bf16  [Cout, Cmid, 3, 3]
    bn2_w_ptr,            # *bf16  [Cout]
    bn2_b_ptr,            # *bf16  [Cout]
    bn2_mean_ptr,         # *bf16  [Cout]
    bn2_var_ptr,          # *bf16  [Cout]
    ds_w_ptr,             # *bf16  [Cout, Cin, 1, 1]
    ds_bn_w_ptr,          # *bf16  [Cout]
    ds_bn_b_ptr,          # *bf16  [Cout]
    ds_bn_mean_ptr,       # *bf16  [Cout]
    ds_bn_var_ptr,        # *bf16  [Cout]
    out_ptr,              # *fp32  [N, Cout, Hout, Wout]
    N, Cin, Hin, Win,
    Cmid, Hout, Wout, Cout,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_a_n, stride_a_c, stride_a_h, stride_a_w,
    stride_w2o, stride_w2i, stride_w2ky, stride_w2kx,
    stride_dso, stride_dsi,  # ds_w is 1x1 so only oc/in strides matter
    stride_on, stride_oc, stride_oh, stride_ow,
    eps: tl.constexpr,
    BLOCK_OC: tl.constexpr,
):
    pid_hw = tl.program_id(0)
    pid_oc = tl.program_id(1)

    HW = Hout * Wout
    n = pid_hw // HW
    hw = pid_hw % HW
    oh = hw // Wout
    ow = hw % Wout

    oc_start = pid_oc * BLOCK_OC
    oc_offsets = oc_start + tl.arange(0, BLOCK_OC)
    oc_mask = oc_offsets < Cout

    # Main path: conv3x3 s1 p1 over act_ptr (fp32)
    acc_main = tl.zeros((BLOCK_OC,), dtype=tl.float32)
    for ic in tl.range(0, Cmid):
        for ky in range(0, 3):
            iy = oh + ky - 1
            y_in_bounds = (iy >= 0) & (iy < Hout)
            for kx in range(0, 3):
                ix = ow + kx - 1
                in_bounds = y_in_bounds & (ix >= 0) & (ix < Wout)

                a_off = n * stride_a_n + ic * stride_a_c + iy * stride_a_h + ix * stride_a_w
                a_val = tl.load(act_ptr + a_off, mask=in_bounds, other=0.0)  # already fp32

                w2_ptrs = w2_ptr + oc_offsets * stride_w2o + ic * stride_w2i + ky * stride_w2ky + kx * stride_w2kx
                w2_vec = tl.load(w2_ptrs, mask=oc_mask, other=0).to(tl.float32)
                acc_main += w2_vec * a_val

    # BN for main path
    gamma2 = tl.load(bn2_w_ptr + oc_offsets, mask=oc_mask, other=0).to(tl.float32)
    beta2 = tl.load(bn2_b_ptr + oc_offsets, mask=oc_mask, other=0).to(tl.float32)
    mean2 = tl.load(bn2_mean_ptr + oc_offsets, mask=oc_mask, other=0).to(tl.float32)
    var2 = tl.load(bn2_var_ptr + oc_offsets, mask=oc_mask, other=0).to(tl.float32)
    denom2 = tl.sqrt(var2 + eps)
    scale2 = gamma2 / denom2
    shift2 = beta2 - mean2 * scale2
    y_main = acc_main * scale2 + shift2  # fp32

    # Skip path: conv1x1 stride2 (no padding) over input x_ptr bf16
    # Target top-left input coordinate for stride2
    iy = oh * 2
    ix = ow * 2
    # These are always within bounds for this problem's shapes, but keep masks robust
    valid_in = (iy >= 0) & (iy < Hin) & (ix >= 0) & (ix < Win)

    acc_skip = tl.zeros((BLOCK_OC,), dtype=tl.float32)
    for ic in tl.range(0, Cin):
        x_off = n * stride_xn + ic * stride_xc + iy * stride_xh + ix * stride_xw
        x_val = tl.load(x_ptr + x_off, mask=valid_in, other=0).to(tl.float32)
        ds_w_ptrs = ds_w_ptr + oc_offsets * stride_dso + ic * stride_dsi
        w_vec = tl.load(ds_w_ptrs, mask=oc_mask, other=0).to(tl.float32)
        acc_skip += w_vec * x_val

    # BN for skip path
    dsg = tl.load(ds_bn_w_ptr + oc_offsets, mask=oc_mask, other=0).to(tl.float32)
    dsb = tl.load(ds_bn_b_ptr + oc_offsets, mask=oc_mask, other=0).to(tl.float32)
    dsm = tl.load(ds_bn_mean_ptr + oc_offsets, mask=oc_mask, other=0).to(tl.float32)
    dsv = tl.load(ds_bn_var_ptr + oc_offsets, mask=oc_mask, other=0).to(tl.float32)
    ds_denom = tl.sqrt(dsv + eps)
    ds_scale = dsg / ds_denom
    ds_shift = dsb - dsm * ds_scale
    y_skip = acc_skip * ds_scale + ds_shift  # fp32

    # Add + ReLU
    y = tl.maximum(y_main + y_skip, 0)

    out_ptrs = out_ptr + n * stride_on + oc_offsets * stride_oc + oh * stride_oh + ow * stride_ow
    tl.store(out_ptrs, y, mask=oc_mask)


def kernel_function(
    x: torch.Tensor,
    *,
    conv1_weight: torch.Tensor,
    bn1_weight: torch.Tensor,
    bn1_bias: torch.Tensor,
    bn1_running_mean: torch.Tensor,
    bn1_running_var: torch.Tensor,
    conv2_weight: torch.Tensor,
    bn2_weight: torch.Tensor,
    bn2_bias: torch.Tensor,
    bn2_running_mean: torch.Tensor,
    bn2_running_var: torch.Tensor,
    downsample_conv_weight: torch.Tensor,
    downsample_bn_weight: torch.Tensor,
    downsample_bn_bias: torch.Tensor,
    downsample_bn_running_mean: torch.Tensor,
    downsample_bn_running_var: torch.Tensor,
    eps: float = 1e-5,
):
    """
    Fused ResNet BasicBlock (downsample variant) in Triton.

    Pipeline implemented:
      Main path:
        conv3x3 (s=2, p=1) -> BN -> ReLU -> conv3x3 (s=1, p=1) -> BN
      Skip path:
        conv1x1 (s=2) -> BN
      Output:
        Add(main, skip) -> ReLU

    Fusion decision:
      - The first conv3x3 produces an activation that is spatially reused by the second conv3x3.
        Fully fusing both convs into a single pass would require recomputing up to 9 neighborhoods
        of the first conv for each output position, multiplying work by ~9x and significantly
        increasing on-chip pressure. To avoid exponential recomputation, we fuse operations into
        two Triton kernels:
          1) conv3x3 s2 p1 + BN + ReLU -> writes fp32 intermediate
          2) conv3x3 s1 p1 + BN, skip 1x1 s2 + BN, Add + ReLU -> writes final output
      - All math is performed inside Triton kernels; the wrapper only validates, allocates, and launches.

    Returns:
      Output tensor [N, 128, 28, 28] on the same device as x, dtype float32.
    """
    assert x.is_cuda, "Input must be on CUDA"
    device = x.device

    # DType: tests use bfloat16 for inputs/weights; kernels accumulate in fp32 and store fp32 for stability
    # Shapes from the test (but we keep generic logic for safety)
    N, Cin, Hin, Win = x.shape
    Cout1 = conv1_weight.shape[0]   # 128
    Cmid = Cout1                    # 128
    Cout = conv2_weight.shape[0]    # 128

    # Validate basic consistency
    assert conv1_weight.shape == (Cout1, Cin, 3, 3)
    assert conv2_weight.shape == (Cout, Cmid, 3, 3)
    assert downsample_conv_weight.shape == (Cout, Cin, 1, 1)

    # Compute output spatial sizes for stride/padding given in the test
    # Conv1: k=3, s=2, p=1
    Hmid = (Hin + 2 * 1 - 3) // 2 + 1
    Wmid = (Win + 2 * 1 - 3) // 2 + 1
    # Conv2: k=3, s=1, p=1 -> (Hmid, Wmid)
    Hout, Wout = Hmid, Wmid

    # Allocate intermediate and output in fp32 for accuracy
    act = torch.empty((N, Cmid, Hmid, Wmid), device=device, dtype=torch.float32)
    out = torch.empty((N, Cout, Hout, Wout), device=device, dtype=torch.float32)

    # Ensure contiguous memory
    x_c = x.contiguous()
    w1_c = conv1_weight.contiguous()
    w2_c = conv2_weight.contiguous()
    ds_w_c = downsample_conv_weight.contiguous()

    bn1_w_c = bn1_weight.contiguous()
    bn1_b_c = bn1_bias.contiguous()
    bn1_mean_c = bn1_running_mean.contiguous()
    bn1_var_c = bn1_running_var.contiguous()

    bn2_w_c = bn2_weight.contiguous()
    bn2_b_c = bn2_bias.contiguous()
    bn2_mean_c = bn2_running_mean.contiguous()
    bn2_var_c = bn2_running_var.contiguous()

    ds_bn_w_c = downsample_bn_weight.contiguous()
    ds_bn_b_c = downsample_bn_bias.contiguous()
    ds_bn_mean_c = downsample_bn_running_mean.contiguous()
    ds_bn_var_c = downsample_bn_running_var.contiguous()

    # Strides in elements
    sxn, sxc, sxh, sxw = x_c.stride()
    s1o, s1i, s1ky, s1kx = w1_c.stride()
    s2o, s2i, s2ky, s2kx = w2_c.stride()
    sdsn, sdsi, = ds_w_c.stride()[:2]  # [Cout, Cin, 1, 1], so only oc/in matter

    san, sac, sah, saw = act.stride()
    son, soc, soh, sow = out.stride()

    # Kernel launch configurations
    BLOCK_OC = 64  # power of 2; 128/64=2 tiles along OC
    grid1 = (triton.cdiv(N * Hmid * Wmid, 1), triton.cdiv(Cout1, BLOCK_OC))
    grid2 = (triton.cdiv(N * Hout * Wout, 1), triton.cdiv(Cout, BLOCK_OC))

    # Launch 1: conv3x3 s2 p1 + BN + ReLU -> act (fp32)
    _conv3x3_bn_relu_s2_kernel[grid1](
        x_c, w1_c,
        bn1_w_c, bn1_b_c, bn1_mean_c, bn1_var_c,
        act,
        N, Cin, Hin, Win,
        Cout1, Hmid, Wmid,
        sxn, sxc, sxh, sxw,
        s1o, s1i, s1ky, s1kx,
        san, sac, sah, saw,
        eps,
        BLOCK_OC=BLOCK_OC,
        num_warps=4,
        num_stages=2,
    )

    # Launch 2: conv3x3 s1 p1 + BN, skip 1x1 s2 + BN, Add + ReLU -> out (fp32)
    _conv3x3_bn_add_skip1x1_bn_relu_kernel[grid2](
        x_c, act,
        w2_c,
        bn2_w_c, bn2_b_c, bn2_mean_c, bn2_var_c,
        ds_w_c,
        ds_bn_w_c, ds_bn_b_c, ds_bn_mean_c, ds_bn_var_c,
        out,
        N, Cin, Hin, Win,
        Cmid, Hout, Wout, Cout,
        sxn, sxc, sxh, sxw,
        san, sac, sah, saw,
        s2o, s2i, s2ky, s2kx,
        sdsn, sdsi,
        son, soc, soh, sow,
        eps,
        BLOCK_OC=BLOCK_OC,
        num_warps=4,
        num_stages=2,
    )

    return out