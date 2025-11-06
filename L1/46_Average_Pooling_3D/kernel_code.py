import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_WO': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_WO': 128}, num_warps=8, num_stages=2),
    ],
    key=['W_OUT'],
)
@triton.jit
def _avg_pool3d_kernel(
    x_ptr, y_ptr,
    N, C, D, H, W,
    D_OUT, H_OUT, W_OUT,
    stride_n, stride_c, stride_d, stride_h, stride_w,
    out_stride_n, out_stride_c, out_stride_d, out_stride_h, out_stride_w,
    STRIDE: tl.constexpr,
    PADDING: tl.constexpr,
    KERNEL_SIZE: tl.constexpr,
    BLOCK_WO: tl.constexpr,
):
    """
    3D Average Pooling (N, C, D, H, W) -> (N, C, D_OUT, H_OUT, W_OUT)

    Each program instance computes a horizontal stripe (BLOCK_WO outputs) for a fixed (n, c, od, oh).
    Padding is handled via masked loads; since count_include_pad=True (PyTorch default), the divisor is K^3.
    Accumulation is performed in FP32 for numerical stability; results are cast back to input dtype.
    """
    # Program ids map to output coordinates
    pid_w = tl.program_id(axis=0)  # tile along output width
    oh = tl.program_id(axis=1)     # output height index
    z = tl.program_id(axis=2)      # fused (n, c, od)

    # Decode fused z -> (n, c, od)
    od = z % D_OUT
    z = z // D_OUT
    c = z % C
    n = z // C

    # Vector of output width indices this program will compute
    offs_ow = pid_w * BLOCK_WO + tl.arange(0, BLOCK_WO)
    ow_mask = offs_ow < W_OUT

    # Compute input starting coordinates for this output location
    in_d0 = od * STRIDE - PADDING
    in_h0 = oh * STRIDE - PADDING
    in_w0 = offs_ow * STRIDE - PADDING

    # FP32 accumulator for better accuracy
    acc = tl.zeros((BLOCK_WO,), dtype=tl.float32)

    # Base pointer increments for the fixed (n, c) pair
    base_nc = n * stride_n + c * stride_c

    # Iterate over the kernel window (KxKxK)
    for kd in range(KERNEL_SIZE):
        in_d = in_d0 + kd
        valid_d = (0 <= in_d) & (in_d < D)

        for kh in range(KERNEL_SIZE):
            in_h = in_h0 + kh
            valid_dh = valid_d & (0 <= in_h) & (in_h < H)

            # Base offset for this (n, c, in_d, in_h)
            base_offset = base_nc + in_d * stride_d + in_h * stride_h

            for kw in range(KERNEL_SIZE):
                in_w = in_w0 + kw
                valid_w = (in_w >= 0) & (in_w < W)
                mask = ow_mask & valid_dh & valid_w

                ptrs = x_ptr + base_offset + in_w * stride_w
                vals = tl.load(ptrs, mask=mask, other=0.0)
                acc += vals.to(tl.float32)

    # Average: count_include_pad=True => divisor is K^3
    denom = KERNEL_SIZE * KERNEL_SIZE * KERNEL_SIZE
    acc = acc * (1.0 / denom)

    # Store results
    out_base = y_ptr + (n * out_stride_n + c * out_stride_c + od * out_stride_d + oh * out_stride_h)
    out_ptrs = out_base + offs_ow * out_stride_w
    tl.store(out_ptrs, acc.to(y_ptr.dtype.element_ty), mask=ow_mask)


def _normalize_triple(x):
    """Helper to normalize kernel_size/stride/padding inputs to a 3-tuple."""
    if isinstance(x, int):
        return (x, x, x)
    if isinstance(x, (tuple, list)) and len(x) == 3:
        return tuple(int(v) for v in x)
    raise ValueError("Expected int or 3-tuple.")


def kernel_function(x, kernel_size=3, stride=2, padding=1):
    """
    3D Average Pooling implemented in a single Triton kernel.

    Fusion notes:
    - Padding handling and summation over the pooling window are fused into one pass.
    - We perform on-the-fly masked loads for out-of-bounds indices, avoiding explicit padding buffers.
    - Accumulate in FP32 and cast to input dtype (typically BF16) at the epilogue.
    - No additional stages (bias, activations) are present in the test, so there is nothing else to fuse.

    Runtime behavior:
    - This function only validates arguments, computes output shape, allocates the output tensor,
      and launches the Triton kernel. All compute (including reductions) occurs inside Triton.

    Args:
        x: Input tensor, shape [N, C, D, H, W], dtype typically torch.bfloat16, CUDA device.
        kernel_size: int or 3-tuple. The test uses 3.
        stride: int or 3-tuple. The test uses 2.
        padding: int or 3-tuple. The test uses 1.

    Returns:
        y: Output tensor with shape computed as in PyTorch AvgPool3d with count_include_pad=True.
    """
    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a torch.Tensor")
    if not x.is_cuda:
        raise ValueError("x must be on CUDA device")
    if x.ndim != 5:
        raise ValueError("x must have shape [N, C, D, H, W]")

    # BF16 is preferred per test; we allow other dtypes but keep output same dtype as input.
    if x.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError("Unsupported dtype. Use bfloat16/float16/float32.")

    N, C, D, H, W = map(int, x.shape)

    kD, kH, kW = _normalize_triple(kernel_size)
    sD, sH, sW = _normalize_triple(stride)
    pD, pH, pW = _normalize_triple(padding)

    # The test uses identical parameters across spatial dims; we assert that here to match kernel impl.
    if not (kD == kH == kW and sD == sH == sW and pD == pH == pW):
        raise ValueError("This implementation expects isotropic kernel_size/stride/padding (same value for D/H/W).")

    K = int(kD)
    S = int(sD)
    P = int(pD)

    # Output sizes (PyTorch AvgPool3d, count_include_pad=True, ceil_mode=False)
    D_OUT = (D + 2 * P - K) // S + 1
    H_OUT = (H + 2 * P - K) // S + 1
    W_OUT = (W + 2 * P - K) // S + 1
    if D_OUT <= 0 or H_OUT <= 0 or W_OUT <= 0:
        raise ValueError("Computed non-positive output dimension. Check kernel_size/stride/padding.")

    # Allocate output tensor on same device/dtype
    y = torch.empty((N, C, D_OUT, H_OUT, W_OUT), device=x.device, dtype=x.dtype)

    # Strides in element units
    sN, sC, sD_in, sH_in, sW_in = x.stride()
    sN_o, sC_o, sD_o, sH_o, sW_o = y.stride()

    # Triton launch grid:
    # - axis 0: tiles along output width
    # - axis 1: output height rows
    # - axis 2: fused (N, C, D_OUT)
    def grid(META):
        BW = META['BLOCK_WO']
        return (triton.cdiv(W_OUT, BW), H_OUT, N * C * D_OUT)

    _avg_pool3d_kernel[grid](
        x, y,
        N, C, D, H, W,
        D_OUT, H_OUT, W_OUT,
        sN, sC, sD_in, sH_in, sW_in,
        sN_o, sC_o, sD_o, sH_o, sW_o,
        STRIDE=S,
        PADDING=P,
        KERNEL_SIZE=K,
    )

    return y