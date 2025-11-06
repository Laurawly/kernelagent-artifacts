import torch
import triton
import triton.language as tl


@triton.jit
def _adaptive_avgpool2d_flatten_kernel(
    x_ptr,                # *bfloat16/float16/float32
    y_ptr,                # *bfloat16/float16/float32
    N, C, H, W,           # dimensions (runtime ints)
    stride_xn, stride_xc, stride_xh, stride_xw,   # input strides
    stride_yn, stride_yc,                         # output strides
    BLOCK_W: tl.constexpr,                        # compile-time constant for W tile size
):
    """
    Fused AdaptiveAvgPool2d(output_size=(1,1)) + Flatten(start_dim=1) for NCHW input.

    Each program instance computes the average over the HxW spatial region for a single (n, c)
    pair and writes the scalar result into y[n, c]. Accumulation is performed in fp32.
    """
    pid_c = tl.program_id(axis=0)  # channel index (0..C-1)
    pid_n = tl.program_id(axis=1)  # batch index   (0..N-1)

    n = pid_n
    c = pid_c
    valid_nc = (n < N) & (c < C)

    # Base pointer to the (n, c, :, :) plane
    x_base = x_ptr + n * stride_xn + c * stride_xc

    # Accumulate in float32 for numerical stability
    acc = tl.zeros((), dtype=tl.float32)

    offs_w = tl.arange(0, BLOCK_W)

    # Iterate rows
    for h in tl.range(0, H):
        row_base = x_base + h * stride_xh
        # Iterate columns in tiles of BLOCK_W
        for start_w in tl.range(0, W, BLOCK_W):
            w_idx = start_w + offs_w
            m = (w_idx < W) & valid_nc
            ptrs = row_base + w_idx * stride_xw
            vals = tl.load(ptrs, mask=m, other=0.0)
            acc += tl.sum(vals.to(tl.float32), axis=0)

    # Compute mean: divide by H*W, done in float32 then cast to output dtype
    acc = acc / W
    acc = acc / H

    y_ptrs = y_ptr + n * stride_yn + c * stride_yc
    tl.store(y_ptrs, acc.to(y_ptr.dtype.element_ty), mask=valid_nc)


def kernel_function(x: torch.Tensor) -> torch.Tensor:
    """
    Fused implementation of:
      adaptive_avg_pool2d(output_size=(1,1)) -> flatten(start_dim=1)
    for NCHW layout.

    Fusion notes:
    - We fuse the global average pooling over spatial dimensions (H, W) with the subsequent
      flatten(start_dim=1). Instead of producing an intermediate [N, C, 1, 1] tensor and then
      reshaping, the kernel directly writes [N, C] where each element is the average of the
      corresponding (n, c) spatial plane. This eliminates intermediate memory traffic and a
      separate kernel launch.

    Constraints/behavior:
    - All math happens inside the Triton kernel. The wrapper only validates inputs, allocates
      outputs, and launches the kernel.
    - Accumulation is done in float32 for numerical stability; results are cast back to the
      input dtype on store (bf16 in the provided test).
    """
    assert x.is_cuda, "Input must be a CUDA tensor."
    assert x.ndim == 4, f"Expected 4D NCHW input, got shape {tuple(x.shape)}"
    N, C, H, W = x.shape

    # Allocate output [N, C], same dtype/device as input
    y = torch.empty((N, C), device=x.device, dtype=x.dtype)

    # Extract strides as element counts (PyTorch .stride() is already in elements)
    stride_xn, stride_xc, stride_xh, stride_xw = x.stride()
    # Output layout [N, C]
    stride_yn, stride_yc = y.stride()

    # Choose a reasonable tile size for W; coalesces along the innermost dimension (W).
    BLOCK_W = 64

    # 2D grid over channels and batch
    grid = (C, N)

    _adaptive_avgpool2d_flatten_kernel[grid](
        x, y,
        N, C, H, W,
        stride_xn, stride_xc, stride_xh, stride_xw,
        stride_yn, stride_yc,
        BLOCK_W=BLOCK_W,
        num_warps=2,
        num_stages=2,
    )

    return y