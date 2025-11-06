import torch
import triton
import triton.language as tl


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

    # Accumulator for max in fp32 for numerical robustness (bf16 inputs are exactly representable in fp32)
    acc = tl.full((BLOCK_W,), -float("inf"), dtype=tl.float32)

    # Iterate over the pooling window
    for ky in range(K_H):
        ih = in_h0 + ky
        valid_h = (ih >= 0) & (ih < H)
        # Safe h index for pointer arithmetic
        ih_safe = tl.where(valid_h, ih, 0)

        for kx in range(K_W):
            iw = in_w0 + kx
            valid_w = (iw >= 0) & (iw < W) & mask_ow
            iw_safe = tl.where(valid_w, iw, 0)

            # Compute input pointers for this (ky, kx) window position
            x_ptrs = x_ptr + base_in + ih_safe * in_stride_h + iw_safe * in_stride_w

            # Load with masking; use -inf for invalid lanes so they don't affect max
            x_vals = tl.load(x_ptrs, mask=valid_w & valid_h, other=-float("inf"))
            x_vals_f32 = x_vals.to(tl.float32)
            acc = tl.maximum(acc, x_vals_f32)

    # Store result
    y_ptrs = y_ptr + base_out + out_h * out_stride_h + offs_ow * out_stride_w
    tl.store(y_ptrs, acc.to(y_ptr.dtype.element_ty), mask=mask_ow)


def _normalize_2tuple(x):
    if isinstance(x, (tuple, list)):
        assert len(x) == 2, "Expected a 2-tuple"
        return int(x[0]), int(x[1])
    else:
        v = int(x)
        return v, v


def kernel_function(x, kernel_size=3, stride=2, padding=1):
    """
    Fused 2D MaxPool (NCHW) implemented with a single Triton kernel.

    What is fused:
    - Padding, neighborhood selection (im2col-like), and max reduction are performed in one pass.
      There is no intermediate tensor materialization; each output element reads its 3x3 window
      directly from global memory with masking for padding, and reduces via max on the fly.

    Constraints:
    - Input layout: NCHW
    - Dtypes: bfloat16, float16, float32 (accumulation in float32, cast back on store)
    - This wrapper performs only validation, allocation, and kernel launch; all math is in Triton.

    Acceptable call signatures (per test):
      kernel_function(x)
      kernel_function(x, kernel_size=3, stride=2, padding=1)
      kernel_function(x, kernel_size=(3,3), stride=(2,2), padding=(1,1))
    """
    assert isinstance(x, torch.Tensor), "x must be a torch.Tensor"
    assert x.is_cuda, "x must be on CUDA device"
    assert x.layout == torch.strided, "x must be a strided tensor (contiguous or strided)"
    assert x.ndim == 4, "x must be 4D NCHW"

    # Normalize params
    kH, kW = _normalize_2tuple(kernel_size)
    sH, sW = _normalize_2tuple(stride)
    pH, pW = _normalize_2tuple(padding)

    N, C, H, W = x.shape
    # Compute output spatial dims (PyTorch's floor division semantics)
    OH = (H + 2 * pH - kH) // sH + 1
    OW = (W + 2 * pW - kW) // sW + 1
    assert OH > 0 and OW > 0, "Invalid output size; check kernel_size/stride/padding"

    # Supported dtypes
    assert x.dtype in (torch.bfloat16, torch.float16, torch.float32), "Unsupported dtype"

    # Allocate output
    y = torch.empty((N, C, OH, OW), device=x.device, dtype=x.dtype)

    # Extract element-wise strides (not bytes)
    in_stride_n, in_stride_c, in_stride_h, in_stride_w = x.stride()
    out_stride_n, out_stride_c, out_stride_h, out_stride_w = y.stride()

    # Choose a BLOCK_W that is a power of two and fits OW nicely
    # For this problem OW=14; BLOCK_W=16 provides good coalescing and simple masking
    BLOCK_W = 16

    # 3D grid: tiles along W, rows along H, packed N*C on axis=2
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