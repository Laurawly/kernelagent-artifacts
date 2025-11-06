import torch
import triton
import triton.language as tl


# Minimal autotune set to keep compile time reasonable while still following guidelines.
_autotune_configs = [
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
    triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
]


@triton.autotune(configs=_autotune_configs, key=['M', 'N', 'K'])
@triton.jit
def _conv1x1_bias_relu_kernel(
    x_ptr,           # *bf16/fp16/fp32, input: [N, C_in, H, W] (NCHW, contiguous)
    w_ptr,           # *bf16/fp16/fp32, weight: [C_out, C_in, 1, 1] contiguous
    b_ptr,           # *bf16/fp16/fp32, bias: [C_out]
    y_ptr,           # *bf16/fp16/fp32, output: [N, C_out, H, W] (NCHW, contiguous)
    M,               # int32, M = N * H * W  (flattened NHW)
    N,               # int32, N = C_out
    K,               # int32, K = C_in
    HW,              # int32, H * W (to avoid extra muls)
    # compile-time constants
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Fused 1x1 Conv2D + Bias + ReLU implemented as a tiled GEMM over NHW x (C_in->C_out).

    Shapes:
      X: [N, C_in, H, W]  -> flattened rows M = N*H*W
      W: [C_out, C_in]    -> treated as K x N (transposed for dot)
      Y: [N, C_out, H, W] -> flattened rows M = N*H*W

    Indexing details (NCHW contiguous):
      X[n, c, h, w] offset = ((n*C_in + c)*H + h)*W + w
      For a given row m over NHW:
         m -> n = m // (H*W), rem = m % (H*W) == h*W + w
         X base address for c=0 row m: n*C_in*H*W + rem
         Advancing c adds +HW per channel
      Y[n, oc, h, w] offset = ((n*C_out + oc)*H + h)*W + w
         Y base for oc=0 row m: n*C_out*H*W + rem
         Advancing oc adds +HW per channel
      W[oc, ic] offset = oc*K + ic
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # Row/column tile coordinates
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = offs_m < M
    n_mask = offs_n < N

    # Compute (n_idx, rem) for each row in this tile
    # n_idx = offs_m // HW; rem = offs_m % HW
    HW_i = HW
    n_idx = offs_m // HW_i
    rem   = offs_m - n_idx * HW_i

    # Prepare base pointers for X and Y row starts (for c=0 / oc=0)
    # X base: n_idx * (K*HW) + rem
    # Y base: n_idx * (N*HW) + rem
    x_row_bases = n_idx * (K * HW_i) + rem                     # [BLOCK_M]
    y_row_bases = n_idx * (N * HW_i) + rem                     # [BLOCK_M]

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # K loop
    k_tiles = tl.cdiv(K, BLOCK_K)
    for kt in tl.range(0, k_tiles):
        k_offs = kt * BLOCK_K + tl.arange(0, BLOCK_K)
        k_mask = k_offs < K

        # A tile (X): [BLOCK_M, BLOCK_K]
        # Pointer math: base per row + k_offs*HW
        a_ptrs = x_ptr + x_row_bases[:, None] + k_offs[None, :] * HW_i
        a = tl.load(a_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)

        # B tile (W): [BLOCK_K, BLOCK_N] where B[k, n] = W[n, k]
        # W layout: [C_out, C_in] contiguous -> offset = oc*K + ic
        b_ptrs = w_ptr + offs_n[None, :] * K + k_offs[:, None]
        b = tl.load(b_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)

        # Accumulate
        acc = tl.dot(a, b, acc)

    # Add bias (broadcast across rows)
    bias = tl.load(b_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
    acc = acc + bias[None, :]

    # ReLU
    acc = tl.maximum(acc, 0.0)

    # Store to output Y
    # Y pointers: y_ptr + y_row_bases[:, None] + offs_n[None, :] * HW
    y_ptrs = y_ptr + y_row_bases[:, None] + offs_n[None, :] * HW_i
    out_dtype = y_ptr.dtype.element_ty
    tl.store(y_ptrs, acc.to(out_dtype), mask=m_mask[:, None] & n_mask[None, :])


def kernel_function(x, weight, bias):
    """
    Fused conv2d 1x1 (stride=1, padding=0, dilation=1, groups=1) + bias + ReLU using a single Triton kernel.

    Fused stages inside the kernel:
      1) Matrix multiply equivalent to 1x1 conv: Y = X(NHW x Cin) @ W(Cin x Cout)
      2) Bias addition per output channel (broadcast over NHW)
      3) ReLU activation

    Wrapper responsibilities (no math here):
      - Validate inputs, dtypes, devices, and shapes
      - Allocate output tensor
      - Configure and launch Triton kernel

    Args:
      x:      [N, C_in, H, W], float or bf16/half on CUDA
      weight: [C_out, C_in, 1, 1], same dtype/device as x
      bias:   [C_out], same dtype/device as x

    Returns:
      y: [N, C_out, H, W], same dtype/device as x
    """
    assert isinstance(x, torch.Tensor), "x must be a torch.Tensor"
    assert isinstance(weight, torch.Tensor), "weight must be a torch.Tensor"
    assert isinstance(bias, torch.Tensor), "bias must be a torch.Tensor"
    assert x.is_cuda and weight.is_cuda and bias.is_cuda, "All tensors must be CUDA tensors"
    assert x.ndim == 4 and weight.ndim == 4 and bias.ndim == 1, "Invalid tensor ranks"
    N, C_in, H, W = x.shape
    C_out, C_in_w, KH, KW = weight.shape
    assert KH == 1 and KW == 1, "This kernel only implements 1x1 convolutions"
    assert C_in_w == C_in, "weight C_in must match input C_in"
    assert bias.shape[0] == C_out, "bias size must match C_out"

    # Dtype checks: Allow bfloat16, float16, float32
    allowed_dtypes = {torch.bfloat16, torch.float16, torch.float32}
    if x.dtype not in allowed_dtypes:
        raise TypeError(f"Unsupported x dtype {x.dtype}; supported: {allowed_dtypes}")
    if weight.dtype != x.dtype or bias.dtype != x.dtype:
        raise TypeError("x, weight, and bias must have the same dtype")

    # Output allocation (same dtype/device as input)
    y = torch.empty((N, C_out, H, W), dtype=x.dtype, device=x.device)

    # Flattened GEMM dims
    M = N * H * W            # rows over NHW
    K = C_in                 # input channels
    Ncol = C_out             # output channels
    HW = H * W

    # Compute grid
    def grid(meta):
        BM = meta['BLOCK_M']
        BN = meta['BLOCK_N']
        return (triton.cdiv(M, BM), triton.cdiv(Ncol, BN))

    # Launch kernel
    _conv1x1_bias_relu_kernel[grid](
        x, weight, bias, y,
        M, Ncol, K, HW,
    )

    return y