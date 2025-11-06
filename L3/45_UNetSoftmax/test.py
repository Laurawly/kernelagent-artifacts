import torch
import inspect

# Summary: Test a Triton-backed kernel against a U-Net-related problem.
# We will:
# - Create CUDA BF16 input with exact shape (8, 8, 64, 512)
# - Try calling kernel_function with flexible signatures (input only, input+dim, input+model, etc.)
# - If output matches input shape, validate it implements softmax over last dim
# - If output matches U-Net output shape and the kernel accepted a model, validate numerically vs PyTorch Model
# - Provide detailed debug info on mismatches or exceptions
# - Return True on pass, False on fail, and exit with code accordingly


class DoubleConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.Softmax(dim=-1),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.Softmax(dim=-1),
        )

    def forward(self, x):
        return self.double_conv(x)


class Model(torch.nn.Module):
    def __init__(self, in_channels, out_channels, features):
        super(Model, self).__init__()
        self.encoder1 = DoubleConv(in_channels, features)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = DoubleConv(features, features * 2)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = DoubleConv(features * 2, features * 4)
        self.pool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = DoubleConv(features * 4, features * 8)
        self.pool4 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = DoubleConv(features * 8, features * 16)

        self.upconv4 = torch.nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = DoubleConv(features * 16, features * 8)
        self.upconv3 = torch.nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = DoubleConv(features * 8, features * 4)
        self.upconv2 = torch.nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = DoubleConv(features * 4, features * 2)
        self.upconv1 = torch.nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = DoubleConv(features * 2, features)

        self.final_conv = torch.nn.Conv2d(features, out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        return self.final_conv(dec1)


def _dtype_tolerances(dtype):
    # Default: rtol=1e-3, atol=1e-3
    # For bfloat16: loosen as per instructions
    if dtype in (torch.bfloat16, torch.float16):
        return 1e-2, 2e-2
    return 1e-3, 1e-3


def _first_tensor_from(result):
    if isinstance(result, torch.Tensor):
        return result
    if isinstance(result, (list, tuple)):
        for x in result:
            if isinstance(x, torch.Tensor):
                return x
    return None


def test_kernel():
    """Test the kernel implementation."""
    try:
        from kernel import kernel_function
        if not callable(kernel_function):
            print("kernel_function is not callable")
            return False
    except Exception as e:
        if isinstance(e, NameError):
            print(f"Test failed: NameError (likely undefined helper in kernel.py): {e}")
        else:
            print(f"Test failed importing kernel_function: {e}")
        return False

    try:
        # Exact problem specs
        batch_size = 8
        in_channels = 8
        out_channels = 4
        height = 64
        width = 512
        features = 64

        if not torch.cuda.is_available():
            print("CUDA is not available; this test requires a CUDA device.")
            return False
        device = torch.device("cuda")

        # Use BF16 as required (avoid FP32 if specified FP32 in original)
        # Non-zero random input
        x = torch.rand(batch_size, in_channels, height, width, device=device, dtype=torch.bfloat16)

        # Build a reference model to compare against if the kernel accepts a model
        # Use eval mode to avoid batch-stat updates causing nondeterminism
        model = Model(in_channels, out_channels, features).to(device).eval()

        # Introspect the kernel signature to attempt the most plausible call
        sig = None
        try:
            sig = inspect.signature(kernel_function)
        except Exception:
            sig = None

        used_strategy = None
        used_model_for_kernel = False
        last_exception = None
        result = None

        # Strategy 1: keyword-argument mapping based on common parameter names
        if sig is not None:
            params = list(sig.parameters.values())
            kwargs_map = {}
            name_to_value = {
                # Input tensor
                "x": x,
                "input": x,
                "inp": x,
                "data": x,
                "tensor": x,
                "a": x,
                # Dimensions
                "dim": -1,
                "axis": -1,
                "height": height,
                "H": height,
                "h": height,
                "width": width,
                "W": width,
                "w": width,
                # Channels
                "in_channels": in_channels,
                "in_ch": in_channels,
                "cin": in_channels,
                "C_in": in_channels,
                "out_channels": out_channels,
                "out_ch": out_channels,
                "cout": out_channels,
                "C_out": out_channels,
                # Features
                "features": features,
                "base_features": features,
                "feat": features,
                "hidden": features,
                # Model/module
                "model": model,
                "module": model,
                "net": model,
                # Dtype/device
                "dtype": torch.bfloat16,
                "precision": torch.bfloat16,
                "device": device,
            }
            for p in params:
                if p.kind in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL):
                    continue
                if p.name in name_to_value:
                    kwargs_map[p.name] = name_to_value[p.name]

            try:
                if kwargs_map:
                    used_strategy = "kwargs by name mapping"
                    used_model_for_kernel = any(k in kwargs_map for k in ("model", "module", "net"))
                    result = kernel_function(**kwargs_map)
            except Exception as e:
                last_exception = e
                result = None

        # Strategy 2: simple positional calls
        if result is None:
            try:
                used_strategy = "positional: (x,)"
                used_model_for_kernel = False
                result = kernel_function(x)
            except Exception as e:
                last_exception = e
                result = None

        if result is None:
            try:
                used_strategy = "positional: (x, -1)"
                used_model_for_kernel = False
                result = kernel_function(x, -1)
            except Exception as e:
                last_exception = e
                result = None

        if result is None:
            try:
                used_strategy = "positional: (x, in_channels, out_channels, features)"
                used_model_for_kernel = False
                result = kernel_function(x, in_channels, out_channels, features)
            except Exception as e:
                last_exception = e
                result = None

        if result is None:
            try:
                used_strategy = "positional: (x, in_channels, out_channels, features, height, width)"
                used_model_for_kernel = False
                result = kernel_function(x, in_channels, out_channels, features, height, width)
            except Exception as e:
                last_exception = e
                result = None

        if result is None and sig is not None:
            try:
                used_strategy = "kwargs: (x) + dim=-1"
                used_model_for_kernel = False
                result = kernel_function(x=x, dim=-1)
            except Exception as e:
                last_exception = e
                result = None

        if result is None and sig is not None and "model" in (p.name for p in sig.parameters.values()):
            try:
                used_strategy = "kwargs: (x, model)"
                used_model_for_kernel = True
                result = kernel_function(x=x, model=model)
            except Exception as e:
                last_exception = e
                result = None

        # If still no result, report error
        if result is None:
            print("Failed to call kernel_function with multiple strategies.")
            if last_exception is not None:
                if isinstance(last_exception, NameError):
                    print(f"NameError (likely undefined helper in kernel.py): {last_exception}")
                else:
                    print(f"Last exception: {last_exception}")
            return False

        # If the kernel did in-place, result may be None; attempt to use input as output
        result_tensor = _first_tensor_from(result)
        if result_tensor is None:
            # If it returned None, assume in-place on x
            if result is None:
                print("kernel_function returned None; assuming in-place modification of the input tensor.")
                result_tensor = x
            else:
                print(f"kernel_function did not return a tensor. Returned type: {type(result)}")
                return False

        # Device checks
        if result_tensor.device != x.device:
            print("Device mismatch:")
            print(f"Input device: {x.device}, Result device: {result_tensor.device}")
            return False

        # Basic sanity on shape and dtype
        print(f"Kernel call strategy used: {used_strategy}")
        print(f"Input shape/dtype: {tuple(x.shape)}/{x.dtype}, Result shape/dtype: {tuple(result_tensor.shape)}/{result_tensor.dtype}")

        # Determine expected behavior:
        # Case A: same shape => expect softmax over last dim
        if tuple(result_tensor.shape) == (batch_size, in_channels, height, width):
            # Expected: softmax over last dimension; compute in fp32 for stability, cast back
            expected32 = torch.softmax(x.to(torch.float32), dim=-1)
            expected = expected32.to(result_tensor.dtype)
            rtol, atol = _dtype_tolerances(result_tensor.dtype)
            try:
                if not torch.allclose(result_tensor, expected, rtol=rtol, atol=atol):
                    print("NUMERICAL MISMATCH for softmax over last dimension:")
                    print(f"rtol={rtol}, atol={atol}")
                    print(f"Input shape: {x.shape}, dtype: {x.dtype}")
                    print(f"Expected shape: {expected.shape}, dtype: {expected.dtype}")
                    print(f"Result shape: {result_tensor.shape}, dtype: {result_tensor.dtype}")
                    print(f"Expected (first few): {expected.flatten()[:10].cpu()}")
                    print(f"Got (first few): {result_tensor.flatten()[:10].float().cpu()}")
                    max_abs = torch.max(torch.abs(result_tensor.to(torch.float32) - expected32)).item()
                    rel_err = torch.max(torch.abs((result_tensor.to(torch.float32) - expected32) / (expected32 + 1e-8))).item()
                    print(f"Max absolute difference: {max_abs}")
                    print(f"Max relative error: {rel_err}")
                    # Additional diagnostics: sum over last dim should be ~1
                    sums = result_tensor.to(torch.float32).sum(dim=-1)
                    sums_dev = torch.max(torch.abs(sums - 1.0)).item()
                    print(f"Max deviation from 1 for softmax sums: {sums_dev}")
                    return False

                # Additional checks: values in [0, 1], sum along last dim ~ 1
                if not torch.isfinite(result_tensor).all():
                    print("Result contains non-finite values.")
                    return False
                sums = result_tensor.to(torch.float32).sum(dim=-1)
                if not torch.allclose(sums, torch.ones_like(sums), rtol=rtol, atol=atol):
                    print("Softmax sum along last dimension is not approximately 1.")
                    print(f"Max deviation: {torch.max(torch.abs(sums - 1.0)).item()}")
                    return False

                print("Softmax verification passed.")
                return True
            except Exception as e:
                print(f"Exception during numerical comparison for softmax: {e}")
                return False

        # Case B: U-Net full output shape and kernel accepted model => compare to model(x)
        elif tuple(result_tensor.shape) == (batch_size, out_channels, height, width):
            if used_model_for_kernel:
                # Compute reference in fp32 for stability, cast to result dtype
                with torch.no_grad():
                    x32 = x.to(torch.float32)
                    model.eval()
                    expected32 = model(x32)
                    expected = expected32.to(result_tensor.dtype)
                rtol, atol = _dtype_tolerances(result_tensor.dtype)
                try:
                    if not torch.allclose(result_tensor, expected, rtol=rtol, atol=atol):
                        print("NUMERICAL MISMATCH for U-Net output:")
                        print(f"rtol={rtol}, atol={atol}")
                        print(f"Input shape: {x.shape}, dtype: {x.dtype}")
                        print(f"Expected shape: {expected.shape}, dtype: {expected.dtype}")
                        print(f"Result shape: {result_tensor.shape}, dtype: {result_tensor.dtype}")
                        print(f"Expected (first few): {expected.flatten()[:10].cpu()}")
                        print(f"Got (first few): {result_tensor.flatten()[:10].float().cpu()}")
                        max_abs = torch.max(torch.abs(result_tensor.to(torch.float32) - expected32)).item()
                        rel_err = torch.max(torch.abs((result_tensor.to(torch.float32) - expected32) / (expected32 + 1e-8))).item()
                        print(f"Max absolute difference: {max_abs}")
                        print(f"Max relative error: {rel_err}")
                        return False
                    if not torch.isfinite(result_tensor).all():
                        print("Result contains non-finite values.")
                        return False
                    print("U-Net verification passed.")
                    return True
                except Exception as e:
                    print(f"Exception during numerical comparison for U-Net: {e}")
                    return False
            else:
                print("Kernel output matches U-Net output shape but no model was provided to kernel; cannot verify numerically.")
                return False

        else:
            print("Unknown output shape; cannot verify correctness against a reference.")
            print(f"Expected either {x.shape} (softmax case) or {(batch_size, out_channels, height, width)} (U-Net case).")
            return False

    except Exception as e:
        if isinstance(e, NameError):
            print(f"Test failed: NameError (likely undefined helper in kernel.py): {e}")
        else:
            print(f"Test failed: {e}")
        return False


if __name__ == "__main__":
    import sys
    success = test_kernel()
    sys.exit(0 if success else 1)