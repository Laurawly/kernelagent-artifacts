import torch
import sys
import inspect

# Summary:
# The original problem describes a Model consisting of:
# - Multihead Self-Attention (nn.MultiheadAttention) with embed_dim=128, num_heads=4
# - Residual connection with LayerNorm over the channel dimension
# Input tensor shape is exactly (B, C, H, W) = (2, 128, 128, 128)
# The forward pass:
#   x -> reshape to (H*W, B, C), self-attn, residual add, LayerNorm, reshape back to (B, C, H, W)
#
# Test requirements:
# - Import kernel_function from kernel.py and call it as a normal Python function
# - Create CUDA tensors, preferably non-zero random data
# - Use BF16 instead of FP32 when not specified explicitly
# - Verify results with device checks, shape/dtype checks, and numerical checks
# - Avoid Triton-specific launch syntax (handled inside kernel.py)
# - Handle exceptions and print debug info
# - Exit code 0 on success, 1 on failure


def _first_tensor_from_output(out):
    """Extract the first torch.Tensor from output (Tensor, list/tuple of Tensors, or others)."""
    if torch.is_tensor(out):
        return out
    if isinstance(out, (list, tuple)):
        for item in out:
            if torch.is_tensor(item):
                return item
    return None


def _try_call_kernel_variants(kernel_function, x, embed_dim, num_heads):
    """
    Try calling kernel_function with a few plausible signatures:
    1) kernel_function(x, embed_dim, num_heads)
    2) kernel_function(x)
    3) kernel_function(x, ...) with weight keyword arguments if those names are accepted
    """
    last_error = None

    # Variant 1: (x, embed_dim, num_heads)
    try:
        print("Attempting kernel_function(x, embed_dim, num_heads)...")
        out = kernel_function(x, embed_dim, num_heads)
        return out, "variant_1"
    except TypeError as e:
        print(f"Call variant_1 failed (TypeError): {e}")
        last_error = e
    except Exception as e:
        print(f"Call variant_1 failed: {e}")
        last_error = e

    # Variant 2: (x,)
    try:
        print("Attempting kernel_function(x)...")
        out = kernel_function(x)
        return out, "variant_2"
    except TypeError as e:
        print(f"Call variant_2 failed (TypeError): {e}")
        last_error = e
    except Exception as e:
        print(f"Call variant_2 failed: {e}")
        last_error = e

    # Variant 3: try to pass weights if the kernel supports them
    # We'll prepare a reference PyTorch module only to source canonical parameter tensors.
    # We DO NOT run the reference forward due to extremely large sequence length.
    try:
        print("Preparing weight tensors for a weight-explicit call attempt (variant_3)...")
        # Build reference modules to obtain consistent parameter tensors
        class RefModel(torch.nn.Module):
            def __init__(self, embed_dim, num_heads):
                super().__init__()
                self.attn = torch.nn.MultiheadAttention(embed_dim, num_heads, batch_first=False)
                self.norm = torch.nn.LayerNorm(embed_dim)

        ref = RefModel(embed_dim, num_heads).to(x.device).to(torch.float32).eval()
        # Extract parameters
        in_proj_weight = ref.attn.in_proj_weight.detach()
        in_proj_bias = ref.attn.in_proj_bias.detach()
        out_proj_weight = ref.attn.out_proj.weight.detach()
        out_proj_bias = ref.attn.out_proj.bias.detach()
        norm_weight = ref.norm.weight.detach()
        norm_bias = ref.norm.bias.detach()
        eps = ref.norm.eps

        # Cast to input dtype (BF16)
        in_proj_weight_bf16 = in_proj_weight.to(x.dtype)
        in_proj_bias_bf16 = in_proj_bias.to(x.dtype)
        out_proj_weight_bf16 = out_proj_weight.to(x.dtype)
        out_proj_bias_bf16 = out_proj_bias.to(x.dtype)
        norm_weight_bf16 = norm_weight.to(x.dtype)
        norm_bias_bf16 = norm_bias.to(x.dtype)

        sig = None
        try:
            sig = inspect.signature(kernel_function)
            print(f"kernel_function signature: {sig}")
        except Exception as e:
            print(f"Could not introspect kernel_function signature: {e}")

        # Prepare two common naming schemes
        kw_map_a = {
            "in_proj_weight": in_proj_weight_bf16,
            "in_proj_bias": in_proj_bias_bf16,
            "out_proj_weight": out_proj_weight_bf16,
            "out_proj_bias": out_proj_bias_bf16,
            "norm_weight": norm_weight_bf16,
            "norm_bias": norm_bias_bf16,
            "embed_dim": embed_dim,
            "num_heads": num_heads,
            "eps": eps,
        }

        kw_map_b = {
            "qkv_weight": in_proj_weight_bf16,
            "qkv_bias": in_proj_bias_bf16,
            "proj_weight": out_proj_weight_bf16,
            "proj_bias": out_proj_bias_bf16,
            "ln_weight": norm_weight_bf16,
            "ln_bias": norm_bias_bf16,
            "embed_dim": embed_dim,
            "num_heads": num_heads,
            "eps": eps,
        }

        # Try variant 3a: use kw_map_a filtered by signature
        if sig is not None:
            accepted = set(sig.parameters.keys())
            filtered_a = {k: v for k, v in kw_map_a.items() if k in accepted}
            try:
                print("Attempting kernel_function(x, **filtered_a)...")
                out = kernel_function(x, **filtered_a)
                return out, "variant_3a"
            except TypeError as e:
                print(f"Call variant_3a failed (TypeError): {e}")
                last_error = e
            except Exception as e:
                print(f"Call variant_3a failed: {e}")
                last_error = e

            # Try variant 3b: use kw_map_b names
            filtered_b = {k: v for k, v in kw_map_b.items() if k in accepted}
            try:
                print("Attempting kernel_function(x, **filtered_b)...")
                out = kernel_function(x, **filtered_b)
                return out, "variant_3b"
            except TypeError as e:
                print(f"Call variant_3b failed (TypeError): {e}")
                last_error = e
            except Exception as e:
                print(f"Call variant_3b failed: {e}")
                last_error = e
        else:
            # If signature is unavailable, attempt with full kw maps directly (may error)
            try:
                print("Attempting kernel_function(x, **kw_map_a) without signature introspection...")
                out = kernel_function(x, **kw_map_a)
                return out, "variant_3a_no_sig"
            except Exception as e:
                print(f"Call variant_3a_no_sig failed: {e}")
                last_error = e
            try:
                print("Attempting kernel_function(x, **kw_map_b) without signature introspection...")
                out = kernel_function(x, **kw_map_b)
                return out, "variant_3b_no_sig"
            except Exception as e:
                print(f"Call variant_3b_no_sig failed: {e}")
                last_error = e

    except Exception as e:
        print(f"Preparing or invoking weight-explicit call (variant_3) failed: {e}")
        last_error = e

    raise RuntimeError(f"All kernel_function call variants failed. Last error: {last_error}")


def test_kernel():
    """Test the kernel implementation."""
    try:
        from kernel import kernel_function
    except Exception as e:
        print(f"Failed to import kernel_function from kernel.py: {e}")
        return False

    try:
        if not callable(kernel_function):
            print("kernel_function is not callable")
            return False

        # Ensure CUDA is available as the kernel is expected to run on GPU
        if not torch.cuda.is_available():
            print("CUDA device is not available. This test requires a CUDA-capable device.")
            return False

        device = torch.device("cuda")
        dtype = torch.bfloat16  # Use BF16 per requirement to avoid pure FP32 tests

        # Exact specifications from the problem description
        embed_dim = 128
        num_heads = 4
        batch_size = 2
        num_channels = embed_dim
        image_height = 128
        image_width = 128

        # Create test input tensor with non-zero random data
        torch.manual_seed(42)
        x = torch.rand(batch_size, num_channels, image_height, image_width, device=device, dtype=dtype)
        # Make sure it's not trivially zero to avoid masking missing computation
        x = x + torch.finfo(dtype).eps

        print(f"Input tensor: shape={x.shape}, dtype={x.dtype}, device={x.device}, "
              f"min={float(x.min())}, max={float(x.max())}")

        # Call kernel_function as a normal Python function (no Triton launch here)
        try:
            out, used_variant = _try_call_kernel_variants(kernel_function, x, embed_dim, num_heads)
            print(f"kernel_function call succeeded using {used_variant}.")
        except Exception as e:
            if isinstance(e, NameError):
                print(f"Test failed: NameError (likely undefined helper in kernel.py): {e}")
            else:
                print(f"Test failed during kernel invocation: {e}")
            return False

        result = _first_tensor_from_output(out)
        if result is None:
            print("kernel_function did not return a torch.Tensor or a container with a Tensor as the first element.")
            print(f"Raw return value type: {type(out)}")
            return False

        # Device check: result device should match input or be on any CUDA device
        if (result.device != x.device) and (result.device.type != 'cuda'):
            print(f"Device mismatch: input on {x.device}, result on {result.device}")
            return False

        # Shape check: result must have the same shape as input (B,C,H,W)
        if result.shape != x.shape:
            print(f"Shape mismatch: expected {x.shape}, got {result.shape}")
            return False

        # Dtype check: allow either same dtype or promotion to float32
        if result.dtype != x.dtype and result.dtype != torch.float32:
            print(f"Dtype unexpected: input dtype={x.dtype}, result dtype={result.dtype} "
                  f"(allowed: {x.dtype} or torch.float32)")
            return False

        # Finite values check
        if not torch.isfinite(result).all():
            finite_ratio = float(torch.isfinite(result).float().mean())
            print(f"Non-finite values detected in result. Finite ratio: {finite_ratio:.6f}")
            return False

        # Basic non-triviality check: the result should not be almost identical to the input
        try:
            # Compare after casting result to input dtype for fairness
            result_cast = result.to(x.dtype)
            # Allow fairly small tolerance to avoid false positives; we expect a residual+LN change
            if torch.allclose(result_cast, x, rtol=1e-3, atol=1e-3):
                print("Result is unexpectedly too close to input; computation may be missing.")
                # Print some samples for debugging
                diff = (result_cast - x).abs()
                print(f"Max abs diff: {float(diff.max())}, mean abs diff: {float(diff.mean())}")
                return False
        except Exception as e:
            print(f"Non-triviality comparison failed: {e}")
            return False

        # LayerNorm property check (assuming default gamma=1, beta=0 as in problem description)
        # For each token (per position), LN should make mean ~ 0 and variance ~ 1 across channel dimension.
        try:
            res32 = result.to(torch.float32)
            # Compute stats across channel dimension (C) for every (B,H,W) location
            means = res32.mean(dim=1)  # shape: (B, H, W)
            vars_ = res32.var(dim=1, unbiased=False)  # LN uses population variance
            zeros = torch.zeros_like(means)
            ones = torch.ones_like(vars_)

            # Use looser tolerances due to BF16 precision and large-tensor accumulation effects
            # BF16: rtol=1e-2, atol=2e-2, but variance can deviate more; allow atol=5e-2
            mean_close = torch.allclose(means, zeros, rtol=1e-2, atol=2e-2)
            var_close = torch.allclose(vars_, ones, rtol=1e-2, atol=5e-2)  # var more sensitive

            if not mean_close or not var_close:
                print("LayerNorm statistical check failed.")
                print(f"Means: mean(abs(mean))={float(means.abs().mean())}, "
                      f"max(abs(mean))={float(means.abs().max())}")
                print(f"Vars: mean(abs(var-1))={float((vars_-1).abs().mean())}, "
                      f"max(abs(var-1))={float((vars_-1).abs().max())}")
                # Print samples (flatten a bit for readability)
                print("Sample means (first 10):", means.flatten()[:10].detach().cpu())
                print("Sample vars (first 10):", vars_.flatten()[:10].detach().cpu())
                return False
        except Exception as e:
            print(f"LayerNorm property check encountered an error: {e}")
            return False

        print("All checks passed.")
        return True  # Success

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