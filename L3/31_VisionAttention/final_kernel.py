import torch
import triton
import triton.language as tl


# ------------------------------------------------------------
# Matmul + optional bias: C[M,N] = A[M,K] @ B[N,K]^T + bias[N]
# A is [M,K], B is [N,K] in memory; we tile B as (K, BN) to compute B^T
# ------------------------------------------------------------
@triton.jit
def _matmul_linear_kernel(
    a_ptr, b_ptr, c_ptr, bias_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    ADD_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_n[None, :] * stride_bn + offs_k[:, None] * stride_bk)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    k_iter = tl.cdiv(K, BLOCK_K)
    for ki in tl.range(0, k_iter):
        k_base = ki * BLOCK_K
        mask_a = (offs_m[:, None] < M) & (k_base + offs_k[None, :] < K)
        mask_b = (offs_n[None, :] < N) & (k_base + offs_k[:, None] < K)
        a = tl.load(a_ptrs, mask=mask_a, other=0.0)
        b = tl.load(b_ptrs, mask=mask_b, other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    if ADD_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
        acc += bias[None, :]

    c = acc.to(c_ptr.dtype.element_ty)
    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


# ------------------------------------------------------------
# Split QKV: qkv[M2, 3E] -> Q/K/V[NH, L, Dh]
# Correctly map flattened row index m -> (l, n) with:
#   l = m // N, n = m % N
# ------------------------------------------------------------
@triton.jit
def _split_qkv_kernel(
    qkv_ptr,
    Q_ptr, K_ptr, V_ptr,
    M2, L, N,
    stride_qkv_m, stride_qkv_n,
    stride_q_h, stride_q_l, stride_q_d,
    stride_k_h, stride_k_l, stride_k_d,
    stride_v_h, stride_v_l, stride_v_d,
    H: tl.constexpr,
    E: tl.constexpr,
    Dh: tl.constexpr,
):
    m = tl.program_id(axis=0)
    if m >= M2:
        return

    # Correct mapping from flattened (L,N) -> (l,n)
    l = m // N
    n = m - l * N  # n = m % N

    d_offs = tl.arange(0, Dh)
    d_mask = d_offs < Dh

    base_m = qkv_ptr + m * stride_qkv_m
    for h in range(0, H):
        e_start = h * Dh
        # Q slice
        q_src = base_m + (0 * E + e_start + d_offs) * stride_qkv_n
        q_seg = tl.load(q_src, mask=d_mask, other=0.0)
        # K slice
        k_src = base_m + (1 * E + e_start + d_offs) * stride_qkv_n
        k_seg = tl.load(k_src, mask=d_mask, other=0.0)
        # V slice
        v_src = base_m + (2 * E + e_start + d_offs) * stride_qkv_n
        v_seg = tl.load(v_src, mask=d_mask, other=0.0)

        nh = n * H + h
        q_dst = Q_ptr + nh * stride_q_h + l * stride_q_l + d_offs * stride_q_d
        k_dst = K_ptr + nh * stride_k_h + l * stride_k_l + d_offs * stride_k_d
        v_dst = V_ptr + nh * stride_v_h + l * stride_v_l + d_offs * stride_v_d
        tl.store(q_dst, q_seg, mask=d_mask)
        tl.store(k_dst, k_seg, mask=d_mask)
        tl.store(v_dst, v_seg, mask=d_mask)


# ------------------------------------------------------------
# FlashAttention-like forward: O[NH, L, Dh] from Q/K/V[NH, L, Dh]
# Online softmax, numerically stable, no causal mask
# ------------------------------------------------------------
@triton.jit
def _flash_attn_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    L, scale,
    stride_q_h, stride_q_l, stride_q_d,
    stride_k_h, stride_k_l, stride_k_d,
    stride_v_h, stride_v_l, stride_v_d,
    stride_o_h, stride_o_l, stride_o_d,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    Dh: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)  # sequence block id
    pid_h = tl.program_id(axis=1)  # head id in NH

    start_m = pid_m * BLOCK_M
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, Dh)

    # Load Q [BM, Dh]
    q_ptrs = Q_ptr + pid_h * stride_q_h + (offs_m[:, None] * stride_q_l + offs_d[None, :] * stride_q_d)
    q_mask = (offs_m[:, None] < L)
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    # Online softmax state
    m_i = tl.zeros((BLOCK_M,), dtype=tl.float32) - float("inf")
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, Dh), dtype=tl.float32)

    n_blocks = tl.cdiv(L, BLOCK_N)
    for nb in tl.range(0, n_blocks):
        start_n = nb * BLOCK_N
        offsn = start_n + offs_n

        # K tile (Dh, BN)
        k_ptrs = K_ptr + pid_h * stride_k_h + (offsn[None, :] * stride_k_l + offs_d[:, None] * stride_k_d)
        k_mask = (offsn[None, :] < L)
        k = tl.load(k_ptrs, mask=k_mask, other=0.0)

        # qk [BM, BN]
        qk = tl.dot(q, k).to(tl.float32)
        qk = qk * scale

        # Online softmax update
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp(qk - m_ij[:, None])
        alpha = tl.exp(m_i - m_ij)
        l_ij = tl.sum(p, 1)

        # V tile (BN, Dh)
        v_ptrs = V_ptr + pid_h * stride_v_h + (offsn[:, None] * stride_v_l + offs_d[None, :] * stride_v_d)
        v_mask = (offsn[:, None] < L)
        v = tl.load(v_ptrs, mask=v_mask, other=0.0).to(tl.float32)

        acc = acc * alpha[:, None] + tl.dot(p.to(tl.float32), v)
        l_i = l_i * alpha + l_ij
        m_i = m_ij

    out = acc * (1.0 / l_i)[:, None]
    o_ptrs = O_ptr + pid_h * stride_o_h + (offs_m[:, None] * stride_o_l + offs_d[None, :] * stride_o_d)
    o_mask = (offs_m[:, None] < L)
    tl.store(o_ptrs, out.to(O_ptr.dtype.element_ty), mask=o_mask)


# ------------------------------------------------------------
# Merge heads: O[NH,L,Dh] -> merged[M2,E] with M2=L*N, E=H*Dh
# Correct mapping m -> (l, n): l = m // N, n = m % N
# ------------------------------------------------------------
@triton.jit
def _merge_heads_kernel(
    O_ptr, merged_ptr,
    M2, L, N,
    stride_o_h, stride_o_l, stride_o_d,
    stride_m_m, stride_m_e,
    H: tl.constexpr,
    Dh: tl.constexpr,
    E: tl.constexpr,
):
    m = tl.program_id(axis=0)
    if m >= M2:
        return

    l = m // N
    n = m - l * N  # n = m % N

    d_offs = tl.arange(0, Dh)
    mask = d_offs < Dh
    for h in range(0, H):
        nh = n * H + h
        o_ptr_row = O_ptr + nh * stride_o_h + l * stride_o_l + d_offs * stride_o_d
        seg = tl.load(o_ptr_row, mask=mask, other=0.0)
        out_ptr_row = merged_ptr + m * stride_m_m + (h * Dh + d_offs) * stride_m_e
        tl.store(out_ptr_row, seg, mask=mask)


# ------------------------------------------------------------
# Residual + LayerNorm per row over E: y = LN(x + residual)
# Accumulate statistics in fp32; cast on store.
# ------------------------------------------------------------
@triton.jit
def _residual_layernorm_kernel(
    x_ptr, residual_ptr, y_ptr,
    gamma_ptr, beta_ptr,
    M2, E, eps,
    stride_x_m, stride_x_e,
    stride_r_m, stride_r_e,
    stride_y_m, stride_y_e,
    BLOCK_E: tl.constexpr,
):
    m = tl.program_id(axis=0)
    offs_e = tl.arange(0, BLOCK_E)
    mask = (m < M2) & (offs_e < E)

    x_row = tl.load(x_ptr + m * stride_x_m + offs_e * stride_x_e, mask=mask, other=0.0).to(tl.float32)
    r_row = tl.load(residual_ptr + m * stride_r_m + offs_e * stride_r_e, mask=mask, other=0.0).to(tl.float32)
    v = x_row + r_row

    mean = tl.sum(v, axis=0) / E
    diff = v - mean
    var = tl.sum(diff * diff, axis=0) / E
    inv_std = 1.0 / tl.sqrt(var + eps)

    gamma = tl.load(gamma_ptr + offs_e, mask=offs_e < E, other=0.0).to(tl.float32)
    beta = tl.load(beta_ptr + offs_e, mask=offs_e < E, other=0.0).to(tl.float32)

    y = diff * inv_std
    y = y * gamma + beta
    tl.store(y_ptr + m * stride_y_m + offs_e * stride_y_e, y.to(y_ptr.dtype.element_ty), mask=mask)


# ------------------------------------------------------------
# Public API: kernel_function
# Notes on fusion:
# - Heavy per-row fusion of residual+LN is applied.
# - Projections and attention are separate kernels for register/SRAM and occupancy reasons at this size.
# ------------------------------------------------------------
def kernel_function(
    x,
    embed_dim=None,
    num_heads=None,
    in_proj_weight=None,
    in_proj_bias=None,
    out_proj_weight=None,
    out_proj_bias=None,
    norm_weight=None,
    norm_bias=None,
    eps: float = 1e-5,
):
    assert isinstance(x, torch.Tensor) and x.is_cuda, "x must be a CUDA tensor"
    device = x.device
    dtype = x.dtype

    if embed_dim is None:
        embed_dim = x.shape[1]
    if num_heads is None:
        num_heads = 1

    B, C, H_img, W_img = x.shape
    E = embed_dim
    assert C == E, f"C ({C}) must equal embed_dim ({E})"
    assert E % num_heads == 0, "embed_dim must be divisible by num_heads"
    Dh = E // num_heads
    L = H_img * W_img
    N = B
    M2 = L * N
    NH = N * num_heads

    # Initialize weights if not provided
    def maybe_init_weights():
        nonlocal in_proj_weight, in_proj_bias, out_proj_weight, out_proj_bias, norm_weight, norm_bias
        gen = torch.Generator(device=device)
        gen.manual_seed(2025)
        scale = 0.05
        if in_proj_weight is None:
            in_proj_weight = (torch.randn(3 * E, E, device=device, dtype=dtype, generator=gen) * scale).contiguous()
        if in_proj_bias is None:
            in_proj_bias = (torch.randn(3 * E, device=device, dtype=dtype, generator=gen) * scale).contiguous()
        if out_proj_weight is None:
            out_proj_weight = (torch.randn(E, E, device=device, dtype=dtype, generator=gen) * scale).contiguous()
        if out_proj_bias is None:
            out_proj_bias = (torch.randn(E, device=device, dtype=dtype, generator=gen) * scale).contiguous()
        if norm_weight is None:
            norm_weight = torch.ones(E, device=device, dtype=dtype).contiguous()
        if norm_bias is None:
            norm_bias = torch.zeros(E, device=device, dtype=dtype).contiguous()

    maybe_init_weights()

    # Weight sanity checks
    assert in_proj_weight.shape == (3 * E, E)
    assert in_proj_bias.shape == (3 * E,)
    assert out_proj_weight.shape == (E, E)
    assert out_proj_bias.shape == (E,)
    assert norm_weight.shape == (E,)
    assert norm_bias.shape == (E,)
    for w in [in_proj_weight, in_proj_bias, out_proj_weight, out_proj_bias, norm_weight, norm_bias]:
        assert w.is_cuda, "All weights must reside on CUDA"

    # Reshape to (L, N, E) then to (M2, E)
    x_lne = x.view(B, C, L).permute(2, 0, 1).contiguous()  # (L, N, E)
    x_rows = x_lne.view(M2, E).contiguous()                # (M2, E)

    # 1) In-projection: (M2, E) x (3E, E)^T -> (M2, 3E)
    qkv = torch.empty((M2, 3 * E), device=device, dtype=dtype)
    BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 32
    grid_inproj = (triton.cdiv(M2, BLOCK_M), triton.cdiv(3 * E, BLOCK_N))
    _matmul_linear_kernel[grid_inproj](
        x_rows, in_proj_weight, qkv, in_proj_bias,
        M2, 3 * E, E,
        x_rows.stride(0), x_rows.stride(1),
        in_proj_weight.stride(0), in_proj_weight.stride(1),
        qkv.stride(0), qkv.stride(1),
        ADD_BIAS=True,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )

    # 2) Split Q,K,V to [NH, L, Dh] (correct (l,n) mapping)
    Q = torch.empty((NH, L, Dh), device=device, dtype=dtype)
    K = torch.empty((NH, L, Dh), device=device, dtype=dtype)
    V = torch.empty((NH, L, Dh), device=device, dtype=dtype)
    grid_split = (M2,)
    _split_qkv_kernel[grid_split](
        qkv, Q, K, V,
        M2, L, N,
        qkv.stride(0), qkv.stride(1),
        Q.stride(0), Q.stride(1), Q.stride(2),
        K.stride(0), K.stride(1), K.stride(2),
        V.stride(0), V.stride(1), V.stride(2),
        H=num_heads, E=E, Dh=Dh,
    )

    # 3) Attention forward
    O = torch.empty_like(Q)
    BLOCK_M_ATT = 64
    BLOCK_N_ATT = 64
    scale = 1.0 / (float(Dh) ** 0.5)
    grid_attn = (triton.cdiv(L, BLOCK_M_ATT), NH)
    _flash_attn_fwd_kernel[grid_attn](
        Q, K, V, O,
        L, scale,
        Q.stride(0), Q.stride(1), Q.stride(2),
        K.stride(0), K.stride(1), K.stride(2),
        V.stride(0), V.stride(1), V.stride(2),
        O.stride(0), O.stride(1), O.stride(2),
        BLOCK_M=BLOCK_M_ATT,
        BLOCK_N=BLOCK_N_ATT,
        Dh=Dh,
    )

    # 4) Merge heads to (M2, E) with correct (l,n) mapping
    merged = torch.empty((M2, E), device=device, dtype=dtype)
    grid_merge = (M2,)
    _merge_heads_kernel[grid_merge](
        O, merged,
        M2, L, N,
        O.stride(0), O.stride(1), O.stride(2),
        merged.stride(0), merged.stride(1),
        H=num_heads, Dh=Dh, E=E,
    )

    # 5) Out-projection: (M2, E) x (E, E)^T -> (M2, E)
    out_proj = torch.empty((M2, E), device=device, dtype=dtype)
    grid_outproj = (triton.cdiv(M2, BLOCK_M), triton.cdiv(E, BLOCK_N))
    _matmul_linear_kernel[grid_outproj](
        merged, out_proj_weight, out_proj, out_proj_bias,
        M2, E, E,
        merged.stride(0), merged.stride(1),
        out_proj_weight.stride(0), out_proj_weight.stride(1),
        out_proj.stride(0), out_proj.stride(1),
        ADD_BIAS=True,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )

    # 6) Residual + LayerNorm
    y_rows = torch.empty((M2, E), device=device, dtype=dtype)
    grid_ln = (M2,)
    _residual_layernorm_kernel[grid_ln](
        out_proj, x_rows, y_rows,
        norm_weight, norm_bias,
        M2, E, eps,
        out_proj.stride(0), out_proj.stride(1),
        x_rows.stride(0), x_rows.stride(1),
        y_rows.stride(0), y_rows.stride(1),
        BLOCK_E=E,
    )

    # Reshape back to (B, C, H, W)
    y = y_rows.view(L, N, E).permute(1, 2, 0).contiguous().view(B, C, H_img, W_img)
    return y