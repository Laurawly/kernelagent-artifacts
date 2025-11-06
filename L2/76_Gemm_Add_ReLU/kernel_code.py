import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _linear_bias_relu_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,  # w is [N, K]; treated as B with shape [K, N] by using (k, n) strides
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Program ids for the output tile
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # Compute row/col offsets for this tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Make offsets more compiler-friendly for vectorization
    offs_m = tl.max_contiguous(tl.multiple_of(offs_m, BLOCK_M), BLOCK_M)
    offs_n = tl.max_contiguous(tl.multiple_of(offs_n, BLOCK_N), BLOCK_N)

    # Accumulator in fp32 for numerical stability
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    k_tiles = tl.cdiv(K, BLOCK_K)
    for ki in range(0, k_tiles):
        k_offs = ki * BLOCK_K + tl.arange(0, BLOCK_K)
        k_mask = k_offs < K

        # Pointers for A = X[M, K]
        a_ptrs = x_ptr + (offs_m[:, None] * stride_xm + k_offs[None, :] * stride_xk)
        a_mask = (offs_m[:, None] < M) & (k_mask[None, :])
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # Pointers for B = W^T[K, N] but W is stored as [N, K]
        # We index W with (k, n) using strides (stride_wk, stride_wn)
        b_ptrs = w_ptr + (k_offs[:, None] * stride_wk + offs_n[None, :] * stride_wn)
        b_mask = (k_mask[:, None]) & (offs_n[None, :] < N)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Accumulate partial matmul using tensor cores when possible
        acc = tl.dot(a, b, acc)

    # Add bias in fp32 and apply ReLU
    bias = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
    acc = acc + bias[None, :]
    acc = tl.maximum(acc, 0.0)

    # Store result: cast to BF16 by default (as per requirements)
    c = acc.to(tl.bfloat16)

    # Output pointers and mask
    y_ptrs = y_ptr + (offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(y_ptrs, c, mask=c_mask)


def kernel_function(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor, out: torch.Tensor = None):
    """
    Fused linear + bias + ReLU kernel implemented in Triton.

    Computes: y = ReLU(x @ W^T + b)

    Shapes:
      - x: [M, K] = [batch_size, in_features]
      - W: [N, K] = [out_features, in_features] (note: W is provided as [out, in])
      - b: [N]     bias broadcast over the batch dimension
      - y: [M, N]

    Fusion rationale:
      - We fuse the matrix multiplication, bias addition, and ReLU into a single Triton kernel.
      - This avoids writing the intermediate matmul result to memory and reading it back for bias/activation,
        minimizing memory traffic and kernel launch overhead.

    Runtime behavior:
      - Inputs/outputs are expected to be bfloat16; accumulation is performed in float32 for numeric stability.
      - The wrapper only validates, allocates, and launches the Triton kernel. All math is inside the kernel.
    """
    # Basic validations (no math here)
    assert x.is_cuda and w.is_cuda and b.is_cuda, "All tensors must be on CUDA device."
    assert x.ndim == 2 and w.ndim == 2 and b.ndim == 1, "Invalid tensor ranks."
    M, Kx = x.shape
    Nw, Kw = w.shape
    assert Kx == Kw, "Incompatible shapes: x.shape[1] must equal w.shape[1] (in_features)."
    assert b.shape[0] == Nw, "Bias shape must match out_features."
    M_out, N_out = M, Nw

    # Enforce BF16 inputs/outputs as per critical requirement
    if x.dtype != torch.bfloat16 or w.dtype != torch.bfloat16 or b.dtype != torch.bfloat16:
        raise AssertionError("This kernel requires BF16 inputs (x, w, b).")

    device = x.device
    assert w.device == device and b.device == device, "All tensors must be on the same device."

    # Allocate output if not provided; must be BF16
    if out is None:
        out = torch.empty((M_out, N_out), dtype=torch.bfloat16, device=device)
    else:
        assert out.shape == (M_out, N_out), "Output tensor has incorrect shape."
        assert out.device == device, "Output tensor must be on the same device."
        if out.dtype != torch.bfloat16:
            raise AssertionError("Output tensor must be BF16.")

    # Strides: Triton expects element-wise strides (like PyTorch)
    stride_xm, stride_xk = x.stride()
    stride_wn, stride_wk = w.stride()  # w is [N, K]
    stride_ym, stride_yn = out.stride()

    # Grid: 2D over M and N tiles
    def grid(meta):
        return (
            triton.cdiv(M, meta["BLOCK_M"]),
            triton.cdiv(Nw, meta["BLOCK_N"]),
        )

    # Launch the fused kernel
    _linear_bias_relu_kernel[grid](
        x, w, b, out,
        M, Nw, Kx,
        stride_xm, stride_xk,
        stride_wn, stride_wk,  # interpret w as (k, n) via strides (wk, wn)
        stride_ym, stride_yn,
    )

    return out