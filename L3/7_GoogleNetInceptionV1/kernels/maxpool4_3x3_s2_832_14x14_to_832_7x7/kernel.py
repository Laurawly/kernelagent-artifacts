import torch
import triton
import triton.language as tl


@triton.jit
def _maxpool2d_nchw_ks3_s2_p1_kernel(
    x_ptr, y_ptr,
    N, C, H, W, HO, WO,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_yn, stride_yc, stride_yh, stride_yw,
    KSIZE: tl.constexpr,
    STRIDE: tl.constexpr,
    PADDING: tl.constexpr,
    BLOCK_OW: tl.constexpr,
):
    """
    MaxPool2d kernel for NCHW layout with:
      - kernel size KSIZE x KSIZE (here KSIZE=3)
      - stride STRIDE (here 2)
      - padding PADDING (here 1)
    Computes y[n, c, oh, ow] = max_{kh, kw in [0..KSIZE-1]} x_padded[n, c, ih0+kh, iw0+kw]
    where ih0 = oh*STRIDE - PADDING, iw0 = ow*STRIDE - PADDING, and x_padded is -inf outside [0..H-1]x[0..W-1].

    Grid:
      pid_nc = program_id(0) over N*C
      pid_oh = program_id(1) over HO
      pid_ob = program_id(2) over ceil_div(WO, BLOCK_OW)
    Each program computes one output row segment (BLOCK_OW wide) for a given (n, c, oh).
    Accumulation is done in fp32 for numerical robustness; cast back to input dtype on store.
    """
    pid_nc = tl.program_id(0)
    pid_oh = tl.program_id(1)
    pid_ob = tl.program_id(2)

    # Decode (n, c) from combined pid
    n = pid_nc // C
    c = pid_nc % C

    # Offsets along output width processed by this program
    offs_ow = pid_ob * BLOCK_OW + tl.arange(0, BLOCK_OW)
    mask_ow = offs_ow < WO

    # Base pointers for input and output for this (n, c)
    x_base = x_ptr + n * stride_xn + c * stride_xc
    y_base = y_ptr + n * stride_yn + c * stride_yc + pid_oh * stride_yh

    # Compute pooling window start coordinates
    h_start = pid_oh * STRIDE - PADDING
    w_starts = offs_ow * STRIDE - PADDING

    # Initialize accumulator in fp32 to -inf
    acc = tl.full((BLOCK_OW,), -float("inf"), dtype=tl.float32)

    # Iterate over 3x3 window
    for kh in range(KSIZE):
        ih = h_start + kh
        mask_h = (ih >= 0) & (ih < H)
        row_ptr = x_base + ih * stride_xh
        for kw in range(KSIZE):
            iw = w_starts + kw
            mask_w = (iw >= 0) & (iw < W)
            valid = mask_ow & mask_h & mask_w
            ptrs = row_ptr + iw * stride_xw
            # Load input in its native dtype with masking; replace invalid with 0 then manually set -inf
            vals = tl.load(ptrs, mask=valid, other=0.0)
            vals_f32 = vals.to(tl.float32)
            neg_inf = tl.full((BLOCK_OW,), -float("inf"), dtype=tl.float32)
            vals_f32 = tl.where(valid, vals_f32, neg_inf)
            acc = tl.maximum(acc, vals_f32)

    # Store result back in original dtype with masking
    y_ptrs = y_base + offs_ow * stride_yw
    tl.store(y_ptrs, acc.to(y_ptr.dtype.element_ty), mask=mask_ow)


def kernel_function(x: torch.Tensor) -> torch.Tensor:
    """
    Triton MaxPool2d(NCHW) wrapper for kernel_size=3, stride=2, padding=1.

    What is fused:
    - The entire max pooling computation (window loads, boundary masking with -inf, and reduction)
      is executed in a single Triton kernel without staging intermediate tensors or launching multiple kernels.

    Runtime behavior:
    - This wrapper performs only validation, allocation, and kernel launch.
    - All math is done inside the Triton kernel as required.

    Args:
        x: Input tensor of shape [N, C, H, W], device=cuda, dtype in {bf16, f16, f32}

    Returns:
        y: Output tensor of shape [N, C, HO, WO] with HO = (H + 2*1 - 3)//2 + 1, WO = (W + 2*1 - 3)//2 + 1
    """
    assert x.is_cuda, "Input must be on CUDA device"
    assert x.ndim == 4, "Input must be a 4D NCHW tensor"
    assert x.dtype in (torch.bfloat16, torch.float16, torch.float32), "Supported dtypes: bf16, f16, f32"

    N, C, H, W = x.shape
    KSIZE = 3
    STRIDE = 2
    PADDING = 1
    HO = (H + 2 * PADDING - KSIZE) // STRIDE + 1
    WO = (W + 2 * PADDING - KSIZE) // STRIDE + 1

    y = torch.empty((N, C, HO, WO), device=x.device, dtype=x.dtype)

    # Strides in elements (PyTorch gives strides already in element units)
    sxn, sxc, sxh, sxw = x.stride()
    syn, syc, syh, syw = y.stride()

    # Grid: (N*C, HO, ceil_div(WO, BLOCK_OW))
    def grid(meta):
        block_ow = meta["BLOCK_OW"]
        return (N * C, HO, triton.cdiv(WO, block_ow))

    # Use a power-of-two BLOCK_OW for good vectorization; 8 fits WO=7 well with minimal masking.
    _maxpool2d_nchw_ks3_s2_p1_kernel[grid](
        x, y,
        N, C, H, W, HO, WO,
        sxn, sxc, sxh, sxw,
        syn, syc, syh, syw,
        KSIZE=KSIZE,
        STRIDE=STRIDE,
        PADDING=PADDING,
        BLOCK_OW=8,
        num_warps=2,
        num_stages=2,
    )
    return y