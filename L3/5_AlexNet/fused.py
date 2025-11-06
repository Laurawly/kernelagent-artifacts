import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# Reference (original) model
# -----------------------------
class OriginalModel(nn.Module):
    def __init__(self, num_classes=1000):
        """
        :param num_classes: The number of output classes (default is 1000 for ImageNet)
        """
        super(OriginalModel, self).__init__()

        # First convolutional layer
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=2)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        # Second convolutional layer
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        # Third convolutional layer
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)

        # Fourth convolutional layer
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU(inplace=True)

        # Fifth convolutional layer
        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1)
        self.relu5 = nn.ReLU(inplace=True)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2)

        # Fully connected layers
        self.fc1 = nn.Linear(in_features=256 * 6 * 6, out_features=4096)
        self.relu6 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(p=0.0)

        self.fc2 = nn.Linear(in_features=4096, out_features=4096)
        self.relu7 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(p=0.0)

        self.fc3 = nn.Linear(in_features=4096, out_features=num_classes)

    def forward(self, x):
        """
        :param x: The input tensor, shape (batch_size, 3, 224, 224)
        :return: The output tensor, shape (batch_size, num_classes)
        """
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        x = self.conv3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.relu4(x)

        x = self.conv5(x)
        x = self.relu5(x)
        x = self.maxpool3(x)

        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = self.relu6(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.relu7(x)
        x = self.dropout2(x)

        x = self.fc3(x)

        return x

# Helpers from the problem file (redeclared to keep this single file self-contained)
batch_size = 1024
num_classes = 1000

def get_inputs():
    return [torch.rand(batch_size, 3, 224, 224)]

def get_init_inputs():
    return [num_classes]

# -----------------------------
# Fused subgraph modules
# -----------------------------
class ConvReLUPool(nn.Module):
    """
    Fused subgraph: Conv2d -> ReLU -> MaxPool2d

    Forward I/O:
    - Input:  x of shape (N, C_in, H_in, W_in)
    - Output: y of shape (N, C_out, H_out, W_out)

    Shapes:
      After Conv2d (kernel_size=k, stride=s, padding=p):
        Hc = floor((H_in + 2*p - k) / s) + 1
        Wc = floor((W_in + 2*p - k) / s) + 1

      After MaxPool2d (pool_kernel=pk, pool_stride=ps, pool_padding=pp=0 here):
        H_out = floor((Hc - pk) / ps) + 1
        W_out = floor((Wc - pk) / ps) + 1
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 conv_kernel_size: int,
                 conv_stride: int = 1,
                 conv_padding: int = 0,
                 pool_kernel_size: int = 2,
                 pool_stride: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=conv_kernel_size,
                              stride=conv_stride,
                              padding=conv_padding)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: Tensor of shape (N, C_in, H_in, W_in)
        :return: Tensor of shape (N, C_out, H_out, W_out)
        """
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


class ConvReLU(nn.Module):
    """
    Fused subgraph: Conv2d -> ReLU

    Forward I/O:
    - Input:  x of shape (N, C_in, H_in, W_in)
    - Output: y of shape (N, C_out, H_out, W_out)

    Shapes after Conv2d (kernel_size=k, stride=s, padding=p):
      H_out = floor((H_in + 2*p - k) / s) + 1
      W_out = floor((W_in + 2*p - k) / s) + 1
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: int = 0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: Tensor of shape (N, C_in, H_in, W_in)
        :return: Tensor of shape (N, C_out, H_out, W_out)
        """
        x = self.conv(x)
        x = self.relu(x)
        return x


class Flatten(nn.Module):
    """
    Fused subgraph: Flatten (feature-only)

    Forward I/O:
    - Input:  x of shape (N, C, H, W)
    - Output: y of shape (N, C*H*W)
    """
    def __init__(self, start_dim: int = 1):
        super().__init__()
        self.start_dim = start_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: Tensor of shape (N, C, H, W)
        :return: Tensor of shape (N, C*H*W)
        """
        return torch.flatten(x, self.start_dim)


class LinearReLUDropout(nn.Module):
    """
    Fused subgraph: Linear -> ReLU -> Dropout

    Forward I/O:
    - Input:  x of shape (N, in_features)
    - Output: y of shape (N, out_features)

    Notes:
    - Dropout is applied with probability p during training. In this model, p=0.0 by default (no-op),
      but the subgraph preserves structure for fusion-friendly representation.
    """
    def __init__(self, in_features: int, out_features: int, p: float = 0.0, inplace_relu: bool = True):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU(inplace=inplace_relu)
        self.dropout = nn.Dropout(p=p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: Tensor of shape (N, in_features)
        :return: Tensor of shape (N, out_features)
        """
        x = self.fc(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class LinearOnly(nn.Module):
    """
    Fused subgraph: Linear (identity activation)

    Forward I/O:
    - Input:  x of shape (N, in_features)
    - Output: y of shape (N, out_features)
    """
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: Tensor of shape (N, in_features)
        :return: Tensor of shape (N, out_features)
        """
        return self.fc(x)

# -----------------------------
# Fused top-level model
# -----------------------------
class Model(nn.Module):
    """
    Fused AlexNet-style model where each subgraph is encapsulated into a module.

    Graph decomposition:
      - block1: Conv(3->96, k=11, s=4, p=2) + ReLU + MaxPool(3, s=2)
      - block2: Conv(96->256, k=5, p=2) + ReLU + MaxPool(3, s=2)
      - block3: Conv(256->384, k=3, p=1) + ReLU
      - block4: Conv(384->384, k=3, p=1) + ReLU
      - block5: Conv(384->256, k=3, p=1) + ReLU + MaxPool(3, s=2)
      - flatten: Flatten to (N, 256*6*6)
      - block6: Linear(9216->4096) + ReLU + Dropout(p=0.0)
      - block7: Linear(4096->4096) + ReLU + Dropout(p=0.0)
      - classifier: Linear(4096->num_classes)

    End-to-end I/O:
      - Input:  x of shape (N, 3, 224, 224)
      - Output: y of shape (N, num_classes)
    """
    def __init__(self, num_classes: int = 1000, dropout_p: float = 0.0):
        super().__init__()
        # Feature extractor
        self.block1 = ConvReLUPool(in_channels=3, out_channels=96,
                                   conv_kernel_size=11, conv_stride=4, conv_padding=2,
                                   pool_kernel_size=3, pool_stride=2)
        self.block2 = ConvReLUPool(in_channels=96, out_channels=256,
                                   conv_kernel_size=5, conv_stride=1, conv_padding=2,
                                   pool_kernel_size=3, pool_stride=2)
        self.block3 = ConvReLU(in_channels=256, out_channels=384,
                               kernel_size=3, stride=1, padding=1)
        self.block4 = ConvReLU(in_channels=384, out_channels=384,
                               kernel_size=3, stride=1, padding=1)
        self.block5 = ConvReLUPool(in_channels=384, out_channels=256,
                                   conv_kernel_size=3, conv_stride=1, conv_padding=1,
                                   pool_kernel_size=3, pool_stride=2)
        self.flatten = Flatten(start_dim=1)

        # Classifier
        self.block6 = LinearReLUDropout(in_features=256 * 6 * 6, out_features=4096, p=dropout_p, inplace_relu=True)
        self.block7 = LinearReLUDropout(in_features=4096, out_features=4096, p=dropout_p, inplace_relu=True)
        self.classifier = LinearOnly(in_features=4096, out_features=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: Tensor of shape (N, 3, 224, 224)
        :return: Tensor of shape (N, num_classes)
        """
        x = self.block1(x)         # (N, 96, 27, 27)
        x = self.block2(x)         # (N, 256, 13, 13)
        x = self.block3(x)         # (N, 384, 13, 13)
        x = self.block4(x)         # (N, 384, 13, 13)
        x = self.block5(x)         # (N, 256, 6, 6)
        x = self.flatten(x)        # (N, 9216)
        x = self.block6(x)         # (N, 4096)
        x = self.block7(x)         # (N, 4096)
        x = self.classifier(x)     # (N, num_classes)
        return x

    def load_from_original(self, original: OriginalModel):
        """
        Copy weights and configuration from an instance of OriginalModel into this fused Model.
        Must be called on architecture-compatible models.
        """
        # Conv + ReLU + Pool blocks
        with torch.no_grad():
            # block1: conv1 -> relu1 -> maxpool1
            self.block1.conv.weight.copy_(original.conv1.weight)
            if original.conv1.bias is not None:
                self.block1.conv.bias.copy_(original.conv1.bias)

            # block2: conv2 -> relu2 -> maxpool2
            self.block2.conv.weight.copy_(original.conv2.weight)
            if original.conv2.bias is not None:
                self.block2.conv.bias.copy_(original.conv2.bias)

            # block3: conv3 -> relu3
            self.block3.conv.weight.copy_(original.conv3.weight)
            if original.conv3.bias is not None:
                self.block3.conv.bias.copy_(original.conv3.bias)

            # block4: conv4 -> relu4
            self.block4.conv.weight.copy_(original.conv4.weight)
            if original.conv4.bias is not None:
                self.block4.conv.bias.copy_(original.conv4.bias)

            # block5: conv5 -> relu5 -> maxpool3
            self.block5.conv.weight.copy_(original.conv5.weight)
            if original.conv5.bias is not None:
                self.block5.conv.bias.copy_(original.conv5.bias)

            # block6: fc1 -> relu6 -> dropout1
            self.block6.fc.weight.copy_(original.fc1.weight)
            self.block6.fc.bias.copy_(original.fc1.bias)
            self.block6.dropout.p = float(original.dropout1.p)

            # block7: fc2 -> relu7 -> dropout2
            self.block7.fc.weight.copy_(original.fc2.weight)
            self.block7.fc.bias.copy_(original.fc2.bias)
            self.block7.dropout.p = float(original.dropout2.p)

            # classifier: fc3
            self.classifier.fc.weight.copy_(original.fc3.weight)
            self.classifier.fc.bias.copy_(original.fc3.bias)

# -----------------------------
# Test harness
# -----------------------------
def _max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a - b).abs().max().item()

def run_tests():
    """
    Validate numerical equivalence between the original and the fused model on representative inputs.
    On success, prints 'PASS' and exits(0).
    """
    torch.manual_seed(0)
    torch.set_grad_enabled(False)

    # Initialize models
    init_args = get_init_inputs()
    if isinstance(init_args, list) or isinstance(init_args, tuple):
        init_args = tuple(init_args)
    else:
        init_args = (init_args,)

    original = OriginalModel(*init_args)
    fused = Model(*init_args)

    # Copy weights from original to fused
    fused.load_from_original(original)

    # Eval mode for deterministic behavior
    original.eval()
    fused.eval()

    # Use a small batch for memory safety while preserving input dimensionality
    x_small = torch.rand(2, 3, 224, 224)

    # Forward pass
    with torch.no_grad():
        y_ref = original(x_small)
        y_fused = fused(x_small)

    # Validate shapes
    assert y_ref.shape == y_fused.shape, f"Shape mismatch: ref {y_ref.shape} vs fused {y_fused.shape}"

    # Validate numerical equivalence
    diff = _max_abs_diff(y_ref, y_fused)
    atol = 1e-6
    if diff > atol:
        raise AssertionError(f"Outputs differ more than {atol}: max abs diff = {diff}")

    print("PASS")
    sys.exit(0)

if __name__ == "__main__":
    run_tests()
