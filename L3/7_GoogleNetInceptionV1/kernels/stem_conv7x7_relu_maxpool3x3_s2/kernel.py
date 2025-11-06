import torch
import triton
import triton.language as tl


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


def kernel_function(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor):
    """
    Fused Conv7x7 (stride=2, pad=3) -> ReLU -> MaxPool3x3 (stride=2, pad=1) for NCHW tensors.

    What is fused:
    - We compute the max-pooling result directly from the input and weights without materializing
      the intermediate conv output nor the ReLU output. For each pooled output pixel, we evaluate
      the 3x3 neighborhood of conv outputs (applying ReLU to each) and take the maximum in a
      single Triton kernel. This eliminates the need to write/read the intermediate 112x112
      activation tensor, significantly reducing memory traffic and kernel launch overhead.

    Runtime policy:
    - Wrapper only validates arguments, allocates the output, computes launch grid, and invokes the
      Triton kernel. All math (conv, ReLU, pooling) is performed inside the Triton kernel.
    - Accumulation is in float32 for numerical stability; inputs/weights/bias/output use bf16 as provided.

    Args:
        x: Input tensor (N, Cin, H, W), dtype=torch.bfloat16, device='cuda'
        w: Weights tensor (Cout, Cin, 7, 7), dtype=torch.bfloat16, device='cuda'
        b: Bias tensor (Cout,), dtype=torch.bfloat16, device='cuda'

    Returns:
        out: Output tensor after fused pipeline with shape (N, Cout, 56, 56), dtype matches x.dtype
    """
    assert x.is_cuda and w.is_cuda and b.is_cuda, "All tensors must be on CUDA"
    assert x.dtype == torch.bfloat16 and w.dtype == torch.bfloat16 and b.dtype == torch.bfloat16, \
        "This fused kernel expects bfloat16 inputs/weights/bias"
    assert x.dim() == 4 and w.dim() == 4 and b.dim() == 1, "Invalid tensor dimensions"
    N, Cin, H, W = x.shape
    Cout, Cin_w, KH, KW = w.shape
    assert Cin == Cin_w, f"Input channels mismatch: x has {Cin}, w has {Cin_w}"
    assert KH == 7 and KW == 7, "This kernel is specialized for 7x7 convolution"
    assert b.shape[0] == Cout, "Bias shape must match output channels"
    # Conv stride and padding
    stride_conv = 2
    pad_conv = 3
    # Conv output spatial size (OHc x OWc) = (112 x 112) for 224 input with stride=2, pad=3, k=7
    OHc = (H + 2 * pad_conv - KH) // stride_conv + 1
    OWc = (W + 2 * pad_conv - KW) // stride_conv + 1
    # MaxPool3x3 stride=2 pad=1 output spatial (OHp x OWp) = (56 x 56)
    pool_k = 3
    pool_stride = 2
    pool_pad = 1
    OHp = (OHc + 2 * pool_pad - pool_k) // pool_stride + 1
    OWp = (OWc + 2 * pool_pad - pool_k) // pool_stride + 1

    # Expected shapes from test
    assert OHp == 56 and OWp == 56, f"Expected pooled output spatial (56,56), got ({OHp},{OWp})"
    assert OHc == 112 and OWc == 112, f"Expected conv output spatial (112,112), got ({OHc},{OWc})"

    # Allocate output
    out = torch.empty((N, Cout, OHp, OWp), device=x.device, dtype=x.dtype)

    # Extract strides (in elements)
    sx_n, sx_c, sx_h, sx_w = x.stride()
    sw_o, sw_i, sw_kh, sw_kw = w.stride()
    so_n, so_c, so_h, so_w = out.stride()

    # Grid configuration
    BLOCK_CO = 32  # tile 32 output channels per program; 64 channels -> 2 programs in axis=1
    grid = (
        N * OHp * OWp,                 # one program per (n, ph, pw)
        triton.cdiv(Cout, BLOCK_CO),   # tiling over output channels
    )

    # Launch kernel
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