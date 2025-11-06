import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# Determinism for reproducibility
torch.manual_seed(0)


class FusedConvReLUPool(nn.Module):
    """
    Fused subgraph: Conv2d -> ReLU -> MaxPool2d

    Shape contract:
    - Input:  (N, C_in, H_in, W_in)
    - Conv2d(kernel_size=k, stride=s, padding=p=0) produces Hc = floor((H_in + 2p - k)/s) + 1, Wc analogously.
    - ReLU:   elementwise, no shape change: (N, C_out, Hc, Wc)
    - MaxPool2d(kernel_size=pk, stride=ps) produces Hp = floor((Hc - pk)/ps) + 1, Wp analogously.
    - Output: (N, C_out, Hp, Wp)

    For this LeNet use-case:
    - Block 1: in (N, 1, 32, 32)  -> out (N, 6, 14, 14) with k=5,s=1,pk=2,ps=2
    - Block 2: in (N, 6, 14, 14) -> out (N, 16, 5, 5) with k=5,s=1,pk=2,ps=2
    """
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, pool_kernel=2, pool_stride=2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, stride=stride)
        self.pool_kernel = pool_kernel
        self.pool_stride = pool_stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward:
        - Input:  x of shape (N, C_in, H_in, W_in)
        - Output: y of shape (N, C_out, H_out, W_out) per the contract above.
        """
        x = self.conv(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=self.pool_kernel, stride=self.pool_stride)
        return x


class Flatten(nn.Module):
    """
    Fused subgraph: 4D feature map -> 2D batch-major vector

    Shape contract:
    - Input:  (N, C, H, W)
    - Output: (N, C*H*W)
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward:
        - Input:  x of shape (N, C, H, W)
        - Output: y of shape (N, C*H*W)
        """
        return x.view(x.size(0), -1)


class FusedLinearReLU(nn.Module):
    """
    Fused subgraph: Linear -> ReLU

    Shape contract:
    - Input:  (N, D_in)
    - Output: (N, D_out)
    """
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features=in_features, out_features=out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward:
        - Input:  x of shape (N, D_in)
        - Output: y of shape (N, D_out)
        """
        return F.relu(self.linear(x))


class OutputLinear(nn.Module):
    """
    Fused subgraph: Linear (classifier head)

    Shape contract:
    - Input:  (N, D_in)
    - Output: (N, D_out)
    """
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features=in_features, out_features=out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward:
        - Input:  x of shape (N, D_in)
        - Output: y of shape (N, D_out)
        """
        return self.linear(x)


class Model(nn.Module):
    """
    Fused LeNet-5-style model assembled from fusable subgraphs only.

    End-to-end shape contract (for 32x32 grayscale input):
    - Input:  (N, 1, 32, 32)
    - After conv_block1 (Conv5x5 s=1 -> ReLU -> MaxPool2x2 s=2): (N, 6, 14, 14)
    - After conv_block2 (Conv5x5 s=1 -> ReLU -> MaxPool2x2 s=2): (N, 16, 5, 5)
    - After flatten:                                          (N, 400)
    - After fc_block1 (Linear->ReLU):                         (N, 120)
    - After fc_block2 (Linear->ReLU):                         (N, 84)
    - After classifier (Linear):                              (N, num_classes)
    """
    def __init__(self, num_classes: int):
        super().__init__()
        # Feature extractor
        self.conv_block1 = FusedConvReLUPool(in_channels=1, out_channels=6, kernel_size=5, stride=1,
                                             pool_kernel=2, pool_stride=2)
        self.conv_block2 = FusedConvReLUPool(in_channels=6, out_channels=16, kernel_size=5, stride=1,
                                             pool_kernel=2, pool_stride=2)
        self.flatten = Flatten()

        # Classifier
        self.fc_block1 = FusedLinearReLU(in_features=16 * 5 * 5, out_features=120)
        self.fc_block2 = FusedLinearReLU(in_features=120, out_features=84)
        self.classifier = OutputLinear(in_features=84, out_features=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.flatten(x)
        x = self.fc_block1(x)
        x = self.fc_block2(x)
        x = self.classifier(x)
        return x


# ----------------------- Reference (non-fused) model for verification -----------------------

class ReferenceLeNet5(nn.Module):
    """
    Original unfused LeNet-5 variant copied from the problem for equivalence testing.
    """
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        self.fc1 = nn.Linear(in_features=16*5*5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# ----------------------- Helpers (mirroring problem file) -----------------------

batch_size = 4096
num_classes_default = 20

def get_inputs():
    # Deterministic input generation
    torch.manual_seed(0)
    return [torch.rand(batch_size, 1, 32, 32)]

def get_init_inputs():
    return [num_classes_default]


# ----------------------- Weight copy utility -----------------------

def copy_reference_to_fused(ref: ReferenceLeNet5, fused: Model) -> None:
    """
    Copies parameters from the reference model into the fused model by explicit mapping.
    """
    with torch.no_grad():
        # Conv blocks
        fused.conv_block1.conv.weight.copy_(ref.conv1.weight)
        fused.conv_block1.conv.bias.copy_(ref.conv1.bias)

        fused.conv_block2.conv.weight.copy_(ref.conv2.weight)
        fused.conv_block2.conv.bias.copy_(ref.conv2.bias)

        # FC blocks
        fused.fc_block1.linear.weight.copy_(ref.fc1.weight)
        fused.fc_block1.linear.bias.copy_(ref.fc1.bias)

        fused.fc_block2.linear.weight.copy_(ref.fc2.weight)
        fused.fc_block2.linear.bias.copy_(ref.fc2.bias)

        fused.classifier.linear.weight.copy_(ref.fc3.weight)
        fused.classifier.linear.bias.copy_(ref.fc3.bias)


# ----------------------- Tests -----------------------

def run_tests():
    # Seed everything deterministically
    torch.manual_seed(0)

    # Build models
    init_args = get_init_inputs()
    ref = ReferenceLeNet5(*init_args).eval()
    fused = Model(*init_args).eval()

    # Align weights so both graphs compute identically
    copy_reference_to_fused(ref, fused)

    # Single pass equivalence test
    inputs = get_inputs()
    with torch.no_grad():
        y_ref = ref(*inputs)
        y_fused = fused(*inputs)

    # Validate shape and value equivalence
    assert y_ref.shape == y_fused.shape, f"Shape mismatch: ref {y_ref.shape} vs fused {y_fused.shape}"
    # Use exact comparison; operations and order are identical
    if not torch.allclose(y_ref, y_fused, rtol=0.0, atol=0.0):
        # Allow a tiny tolerance just in case of backend nuances
        assert torch.allclose(y_ref, y_fused, rtol=1e-7, atol=1e-7), "Numerical mismatch between reference and fused outputs"

    print("PASS")
    sys.exit(0)


if __name__ == "__main__":
    run_tests()
