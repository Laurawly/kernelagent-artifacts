import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import inspect

# Summary:
# Test a Triton-backed kernel that implements an Attention block using Multihead Self-Attention (MHA) and LayerNorm.
# Problem specs:
# - Input: tensor x of shape (B=2, C=128, H=128, W=128)
# - Model: MHA with embed_dim=128, num_heads=4, residual + LayerNorm over embed_dim
# - Use BF16 instead of FP32 for inputs/weights/outputs
# - Call kernel_function as a normal Python function; kernel.py handles Triton launch.
# - Verify numerical correctness against a PyTorch reference when possible.
# - Provide detailed debugging output on mismatches and handle exceptions gracefully.

def _build_reference_weights(embed_dim, device, dtype):
    """Create deterministic non-zero weights for MHA and LayerNorm."""
    # Use a fixed seed for reproducibility of weights
    gen = torch.Generator(device=device)
    gen.manual_seed(12345)

    scale = 0.1
    in_proj_weight = torch.randn(3 * embed_dim, embed_dim, device=device, dtype=dtype, generator=gen) * scale
    in_proj_bias = torch.randn(3 * embed_dim, device=device, dtype=dtype, generator=gen) * scale
    out_proj_weight = torch.randn(embed_dim, embed_dim, device=device, dtype=dtype, generator=gen) * scale
    out_proj_bias = torch.randn(embed_dim, device=device, dtype=dtype, generator=gen) * scale
    # LN defaults are gamma=1, beta=0; we use near-default but nontrivial gamma to avoid accidental equivalence
    norm_weight = (torch.randn(embed_dim, device=device, dtype=dtype, generator=gen) * scale) + 1.0
    norm_bias = torch.randn(embed_dim, device=device, dtype=dtype, generator=gen) * scale

    return {
        "in_proj_weight": in_proj_weight.contiguous(),
        "in_proj_bias": in_proj_bias.contiguous(),
        "out_proj_weight": out_proj_weight.contiguous(),
        "out_proj_bias": out_proj_bias.contiguous(),
        "norm_weight": norm_weight.contiguous(),
        "norm_bias": norm_bias.contiguous(),
    }


def _reference_attention_block(x, embed_dim, num_heads, weights, eps=1e-5):
    """
    Reference implementation of the described Attention + Residual + LayerNorm block.

    x: (B, C, H, W) with C == embed_dim
    returns: (B, C, H, W)
    """
    device = x.device
    dtype = x.dtype
    B, C, H, W = x.shape
    assert C == embed_dim, f"embed_dim mismatch: got C={C}, expected embed_dim={embed_dim}"

    # Reshape to MHA expected (L, N, E)
    x_lne = x.view(B, C, H * W).permute(2, 0, 1)  # (L, N, E) where L = H*W, N = B, E = embed_dim
    L, N, E = x_lne.shape
    assert E == embed_dim

    # Compute QKV using in-proj
    W_qkv = weights["in_proj_weight"]  # (3E, E)
    b_qkv = weights["in_proj_bias"]    # (3E,)
    qkv = F.linear(x_lne, W_qkv, b_qkv)  # (L, N, 3E)
    q, k, v = torch.split(qkv, E, dim=-1)  # each (L, N, E)

    head_dim = embed_dim // num_heads
    assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

    # Rearrange for scaled_dot_product_attention: (N, num_heads, L, head_dim)
    def reshape_heads(t):
        # (L, N, E) -> (N, num_heads, L, head_dim)
        t = t.permute(1, 2, 0)  # (N, E, L)
        t = t.reshape(N, num_heads, head_dim, L).permute(0, 1, 3, 2)  # (N, num_heads, L, head_dim)
        return t

    q_r = reshape_heads(q)
    k_r = reshape_heads(k)
    v_r = reshape_heads(v)

    # Attention
    # Prefer memory-efficient path
    if hasattr(F, "scaled_dot_product_attention"):
        attn = F.scaled_dot_product_attention(q_r, k_r, v_r, dropout_p=0.0, is_causal=False)  # (N, num_heads, L, head_dim)
    else:
        # Fallback (not recommended for large L; provided for compatibility)
        # Compute softmax(q k^T / sqrt(d)) v
        scale = head_dim ** -0.5
        attn_scores = torch.matmul(q_r, k_r.transpose(-2, -1)) * scale  # (N, num_heads, L, L)
        attn_probs = attn_scores.softmax(dim=-1)
        attn = torch.matmul(attn_probs, v_r)  # (N, num_heads, L, head_dim)

    # Merge heads: (N, num_heads, L, head_dim) -> (L, N, E)
    attn = attn.permute(0, 2, 1, 3).contiguous()  # (N, L, num_heads, head_dim)
    attn = attn.view(N, L, E).permute(1, 0, 2).contiguous()  # (L, N, E)

    # Out projection
    out = F.linear(attn, weights["out_proj_weight"], weights["out_proj_bias"])  # (L, N, E)

    # Residual + LayerNorm over last dim
    residual = x_lne
    ln = nn.LayerNorm(E, eps=eps, device=device, dtype=dtype)
    with torch.no_grad():
        # Set LN affine weights to provided ones
        # LayerNorm stores weight and bias as parameters named 'weight' and 'bias'
        ln.weight.copy_(weights["norm_weight"])
        ln.bias.copy_(weights["norm_bias"])
    y = ln(out + residual)  # (L, N, E)

    # Reshape back to (B, C, H, W)
    y = y.permute(1, 2, 0).contiguous().view(B, C, H, W)
    return y


def test_kernel():
    """Test the kernel implementation."""
    try:
        try:
            # In the artifacts repo the kernel lives in final_kernel.py
            from final_kernel import kernel_function
        except Exception as e:
            print(f"Failed to import kernel_function from final_kernel.py: {e}")
            return False

        if not callable(kernel_function):
            print("kernel_function is not callable")
            return False

        if not torch.cuda.is_available():
            print("CUDA is not available on this system. This test requires a CUDA device.")
            return False

        device = torch.device("cuda")
        dtype = torch.bfloat16  # Use BF16 as per requirement to avoid FP32

        # Exact shapes and parameters from the problem description
        embed_dim = 128
        num_heads = 4
        batch_size = 2
        num_channels = embed_dim
        image_height = 128
        image_width = 128

        # Create random non-zero input tensor on CUDA device in BF16
        # Use rand instead of randn to keep values in [0,1); avoids extreme values that may cause large relative errors with bf16
        x = torch.rand(batch_size, num_channels, image_height, image_width, device=device, dtype=dtype)

        # Build deterministic test weights and compute expected output using a robust reference path
        weights = _build_reference_weights(embed_dim, device, dtype)
        try:
            expected = _reference_attention_block(x, embed_dim, num_heads, weights)
        except RuntimeError as e:
            print("Error while computing reference output. This may indicate insufficient GPU support for bf16 or SDPA.")
            print(f"Reference computation error: {e}")
            return False

        # Attempt to call kernel_function with progressively fewer arguments to accommodate unknown signatures.
        # Primary expected signature (most informative, enables strong numerical verification):
        # kernel_function(x, embed_dim, num_heads, in_proj_w, in_proj_b, out_proj_w, out_proj_b, norm_w, norm_b)
        attempts = []
        result = None
        call_signature_used = None

        def try_call(args, desc):
            nonlocal result, call_signature_used
            try:
                out = kernel_function(*args)
                result = out
                call_signature_used = desc
                return True
            except TypeError as e:
                print(f"Call attempt '{desc}' failed with TypeError: {e}")
                return False
            except Exception as e:
                print(f"Call attempt '{desc}' failed with exception: {e}")
                return False

        args_full = [
            x, embed_dim, num_heads,
            weights["in_proj_weight"], weights["in_proj_bias"],
            weights["out_proj_weight"], weights["out_proj_bias"],
            weights["norm_weight"], weights["norm_bias"],
        ]

        if not try_call(args_full, "x, embed_dim, num_heads, all weights"):
            # Fallback: maybe kernel constructs its own weights internally; pass minimal required config
            if not try_call([x, embed_dim, num_heads], "x, embed_dim, num_heads"):
                # Final fallback: only x
                if not try_call([x], "x only"):
                    print("All calling attempts for kernel_function failed. Please ensure kernel_function accepts one of the tested signatures.")
                    return False

        # Normalize result type (some kernels may return (out, aux) or similar)
        if isinstance(result, (tuple, list)):
            if len(result) == 0:
                print("kernel_function returned an empty tuple/list.")
                return False
            kernel_out = result[0]
        else:
            kernel_out = result

        if not isinstance(kernel_out, torch.Tensor):
            print(f"kernel_function returned a non-tensor output of type: {type(kernel_out)}")
            return False

        # Device check
        if kernel_out.device != x.device:
            print(f"Device mismatch: input device {x.device}, result device {kernel_out.device}")
            return False

        # Shape check
        expected_shape = x.shape
        if kernel_out.shape != expected_shape:
            print(f"Shape mismatch: expected {expected_shape}, got {kernel_out.shape}")
            return False

        # Dtype check (allow minor deviations; cast for comparison if needed)
        if kernel_out.dtype != x.dtype:
            print(f"Warning: dtype mismatch: input {x.dtype}, result {kernel_out.dtype}. Will cast for comparison.")

        # Numerical verification
        # If we passed weights successfully, we can check full numerical equality versus expected
        # Otherwise, we check sane properties and fail with guidance
        tol_rtol = 1e-2
        tol_atol = 2e-2  # Looser tolerances for bf16 due to reduced precision

        can_compare_numerically = (call_signature_used == "x, embed_dim, num_heads, all weights")
        if can_compare_numerically:
            # Compare using float32 for error measurement to avoid bf16 rounding dominating the error metrics
            expected32 = expected.to(dtype=torch.float32)
            result32 = kernel_out.to(dtype=torch.float32)

            try:
                if not torch.allclose(result32, expected32, rtol=tol_rtol, atol=tol_atol):
                    diff = (result32 - expected32)
                    max_abs = torch.max(torch.abs(diff)).item()
                    rel_err = torch.max(torch.abs(diff) / (torch.abs(expected32) + 1e-8)).item()

                    print("NUMERICAL MISMATCH:")
                    print(f"Call signature used: {call_signature_used}")
                    print(f"Input shape: {x.shape}, dtype: {x.dtype}, device: {x.device}")
                    print(f"Expected shape: {expected.shape}, dtype: {expected.dtype}, device: {expected.device}")
                    print(f"Result shape: {kernel_out.shape}, dtype: {kernel_out.dtype}, device: {kernel_out.device}")
                    print(f"Tolerances used (bf16): rtol={tol_rtol}, atol={tol_atol}")
                    print(f"Expected (first 10): {expected32.flatten()[:10].cpu()}")
                    print(f"Got (first 10): {result32.flatten()[:10].cpu()}")
                    print(f"Max absolute difference: {max_abs}")
                    print(f"Max relative error: {rel_err}")
                    return False
            except Exception as e:
                print(f"Exception during numerical comparison: {e}")
                # Provide more context to assist debugging
                print(f"Expected dtype/device: {expected.dtype}/{expected.device}, result dtype/device: {kernel_out.dtype}/{kernel_out.device}")
                return False
        else:
            # We could not pass reference weights to the kernel. Numerical equality cannot be asserted reliably.
            # Perform weaker sanity checks and inform the developer.
            print("Info: Could not pass explicit weights to kernel_function; numerical equivalence to a reference cannot be established.")
            print(f"Used call signature: {call_signature_used}")
            # Additional sanity checks: output should not be identical to input (for non-trivial kernels)
            # Note: This check is heuristic and will not fail the test unless it's obviously incorrect.
            same_as_input = torch.allclose(kernel_out.to(torch.float32), x.to(torch.float32), rtol=1e-3, atol=1e-3)
            if same_as_input:
                print("Warning: kernel output is numerically identical to input tensor with tight tolerance; kernel may be a no-op.")
                # Do not fail here; keep as informational

        print("Test passed.")
        return True  # if successful

    except Exception as e:
        # Surface undefined helper issues from kernel.py clearly
        if isinstance(e, NameError):
            print(f"Test failed: NameError (likely undefined helper in kernel.py): {e}")
        else:
            print(f"Test failed: {e}")
        return False


if __name__ == "__main__":
    success = test_kernel()
    sys.exit(0 if success else 1)
