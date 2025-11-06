import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================
# Reference (Original) Model from problem file
# =========================
class OriginalMBConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio):
        super(OriginalMBConv, self).__init__()
        self.use_residual = (stride == 1 and in_channels == out_channels)
        hidden_dim = in_channels * expand_ratio

        if expand_ratio != 1:
            self.expand_conv = nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True)
            )

        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True)
        )

        self.project_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        identity = x

        if hasattr(self, 'expand_conv'):
            x = self.expand_conv(x)

        x = self.depthwise_conv(x)
        x = self.project_conv(x)

        if self.use_residual:
            x += identity

        return x

class OriginalModel(nn.Module):
    def __init__(self, num_classes=1000):
        super(OriginalModel, self).__init__()

        # Initial convolutional layer
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)

        # MBConv blocks
        self.blocks = nn.Sequential(
            # MBConv1 (32, 16, 1, 1)
            OriginalMBConv(32, 16, kernel_size=3, stride=1, expand_ratio=1),
            # MBConv6 (16, 24, 2, 6)
            OriginalMBConv(16, 24, kernel_size=3, stride=2, expand_ratio=6),
            # MBConv6 (24, 24, 1, 6)
            OriginalMBConv(24, 24, kernel_size=3, stride=1, expand_ratio=6),
            # MBConv6 (24, 40, 2, 6)
            OriginalMBConv(24, 40, kernel_size=5, stride=2, expand_ratio=6),
            # MBConv6 (40, 40, 1, 6)
            OriginalMBConv(40, 40, kernel_size=5, stride=1, expand_ratio=6),
            # MBConv6 (40, 80, 2, 6)
            OriginalMBConv(40, 80, kernel_size=3, stride=2, expand_ratio=6),
            # MBConv6 (80, 80, 1, 6)
            OriginalMBConv(80, 80, kernel_size=3, stride=1, expand_ratio=6),
            # MBConv6 (80, 112, 1, 6)
            OriginalMBConv(80, 112, kernel_size=5, stride=1, expand_ratio=6),
            # MBConv6 (112, 112, 1, 6)
            OriginalMBConv(112, 112, kernel_size=5, stride=1, expand_ratio=6),
            # MBConv6 (112, 192, 2, 6)
            OriginalMBConv(112, 192, kernel_size=5, stride=2, expand_ratio=6),
            # MBConv6 (192, 192, 1, 6)
            OriginalMBConv(192, 192, kernel_size=5, stride=1, expand_ratio=6),
            # MBConv6 (192, 192, 1, 6)
            OriginalMBConv(192, 192, kernel_size=5, stride=1, expand_ratio=6),
            # MBConv6 (192, 320, 1, 6)
            OriginalMBConv(192, 320, kernel_size=3, stride=1, expand_ratio=6)
        )

        # Final convolutional layer
        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)

        # Fully connected layer
        self.fc = nn.Linear(1280, num_classes)

    def forward(self, x):
        # x: (N, 3, 224, 224)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.blocks(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Test code inputs from problem file
batch_size = 10
num_classes = 1000

def get_inputs():
    return [torch.rand(batch_size, 3, 224, 224)]

def get_init_inputs():
    return [num_classes]

# =========================
# Fused Subgraph Modules
# =========================

class ConvBNActivation2d(nn.Module):
    """
    Fused 2D Conv + BatchNorm (+ optional activation) module.

    Forward:
      - Input: x, shape (N, Cin, H, W)
      - Output: y, shape (N, Cout, H', W')

    H' and W' depend on stride/padding/dilation similarly to nn.Conv2d.
    Activation can be 'relu', 'relu6', or None.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 groups=1, bias=False, activation='relu'):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        if activation is None:
            self.act = None
        elif activation == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif activation == 'relu6':
            self.act = nn.ReLU6(inplace=True)
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, x):
        y = self.conv(x)
        y = self.bn(y)
        if self.act is not None:
            y = self.act(y)
        return y

    def load_from_conv_bn(self, ref_conv: nn.Conv2d, ref_bn: nn.BatchNorm2d):
        # Copy Conv2d weights/bias
        with torch.no_grad():
            self.conv.weight.copy_(ref_conv.weight)
            if self.conv.bias is not None and ref_conv.bias is not None:
                self.conv.bias.copy_(ref_conv.bias)
            # Copy BN parameters
            self.bn.weight.copy_(ref_bn.weight)
            self.bn.bias.copy_(ref_bn.bias)
            self.bn.running_mean.copy_(ref_bn.running_mean)
            self.bn.running_var.copy_(ref_bn.running_var)
            if hasattr(self.bn, "num_batches_tracked") and hasattr(ref_bn, "num_batches_tracked"):
                self.bn.num_batches_tracked.copy_(ref_bn.num_batches_tracked)
            # Copy eps and momentum for safety
            self.bn.eps = ref_bn.eps
            self.bn.momentum = ref_bn.momentum


class GlobalAvgPoolFlatten(nn.Module):
    """
    Fused Global Average Pooling to 1x1 and flatten to (N, C).

    Forward:
      - Input: x, shape (N, C, H, W)
      - Output: y, shape (N, C)
    """
    def __init__(self):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        y = self.pool(x)
        y = y.view(y.size(0), -1)
        return y


class FusedMBConv(nn.Module):
    """
    Fused Mobile Inverted Bottleneck Convolution (MBConv) block.

    Structure:
      - Optional expand: 1x1 Conv-BN-ReLU6 (Cin -> hidden_dim = Cin * expand_ratio)
      - Depthwise: kxk Depthwise Conv-BN-ReLU6 (hidden_dim groups = hidden_dim)
      - Project: 1x1 Conv-BN (hidden_dim -> Cout)
      - Residual: add input if stride == 1 and Cin == Cout

    Forward:
      - Input: x, shape (N, Cin, H, W)
      - Output: y, shape (N, Cout, H', W')
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio):
        super().__init__()
        self.use_residual = (stride == 1 and in_channels == out_channels)
        hidden_dim = in_channels * expand_ratio

        # Optional expand
        if expand_ratio != 1:
            self.expand = ConvBNActivation2d(
                in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, activation='relu6'
            )
        else:
            self.expand = None

        # Depthwise
        self.depthwise = ConvBNActivation2d(
            hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride,
            padding=(kernel_size - 1) // 2, groups=hidden_dim, activation='relu6'
        )

        # Project
        self.project = ConvBNActivation2d(
            hidden_dim, out_channels, kernel_size=1, stride=1, padding=0, activation=None
        )

    def forward(self, x):
        identity = x
        if self.expand is not None:
            x = self.expand(x)
        x = self.depthwise(x)
        x = self.project(x)
        if self.use_residual:
            x = x + identity
        return x

    def load_from_reference(self, ref_mbconv: OriginalMBConv):
        # Copy expand if present
        if self.expand is not None:
            assert hasattr(ref_mbconv, 'expand_conv'), "Reference MBConv missing expand_conv"
            self.expand.load_from_conv_bn(ref_mbconv.expand_conv[0], ref_mbconv.expand_conv[1])
        else:
            assert not hasattr(ref_mbconv, 'expand_conv'), "Reference MBConv has expand_conv but fused does not"

        # Copy depthwise
        self.depthwise.load_from_conv_bn(ref_mbconv.depthwise_conv[0], ref_mbconv.depthwise_conv[1])

        # Copy project
        self.project.load_from_conv_bn(ref_mbconv.project_conv[0], ref_mbconv.project_conv[1])


# =========================
# Fused Top-level Model
# =========================
class Model(nn.Module):
    """
    Fused EfficientNetB0-style model assembled from larger subgraph modules.

    Graph submodules:
      - stem: Conv-BN-ReLU
      - blocks: sequence of fused MBConv blocks
      - head: Conv-BN-ReLU
      - pool: global average pooling + flatten
      - fc: linear classifier

    Forward:
      - Input: x, shape (N, 3, 224, 224)
      - Output: logits, shape (N, num_classes)
    """
    def __init__(self, num_classes=1000):
        super().__init__()
        # Stem
        self.stem = ConvBNActivation2d(3, 32, kernel_size=3, stride=2, padding=1, activation='relu')

        # Blocks (MBConv)
        self.blocks = nn.Sequential(
            FusedMBConv(32, 16, kernel_size=3, stride=1, expand_ratio=1),
            FusedMBConv(16, 24, kernel_size=3, stride=2, expand_ratio=6),
            FusedMBConv(24, 24, kernel_size=3, stride=1, expand_ratio=6),
            FusedMBConv(24, 40, kernel_size=5, stride=2, expand_ratio=6),
            FusedMBConv(40, 40, kernel_size=5, stride=1, expand_ratio=6),
            FusedMBConv(40, 80, kernel_size=3, stride=2, expand_ratio=6),
            FusedMBConv(80, 80, kernel_size=3, stride=1, expand_ratio=6),
            FusedMBConv(80, 112, kernel_size=5, stride=1, expand_ratio=6),
            FusedMBConv(112, 112, kernel_size=5, stride=1, expand_ratio=6),
            FusedMBConv(112, 192, kernel_size=5, stride=2, expand_ratio=6),
            FusedMBConv(192, 192, kernel_size=5, stride=1, expand_ratio=6),
            FusedMBConv(192, 192, kernel_size=5, stride=1, expand_ratio=6),
            FusedMBConv(192, 320, kernel_size=3, stride=1, expand_ratio=6),
        )

        # Head
        self.head = ConvBNActivation2d(320, 1280, kernel_size=1, stride=1, padding=0, activation='relu')

        # Pool + classifier
        self.pool = GlobalAvgPoolFlatten()
        self.fc = nn.Linear(1280, num_classes)

    def forward(self, x):
        x = self.stem(x)             # (N, 32, 112, 112)
        x = self.blocks(x)           # (N, 320, H, W)
        x = self.head(x)             # (N, 1280, H, W)
        x = self.pool(x)             # (N, 1280)
        x = self.fc(x)               # (N, num_classes)
        return x

    def load_from_reference(self, ref: OriginalModel):
        # Copy stem conv/bn
        self.stem.load_from_conv_bn(ref.conv1, ref.bn1)

        # Copy MBConv blocks
        assert len(self.blocks) == len(ref.blocks), "Block count mismatch"
        for fused_block, ref_block in zip(self.blocks, ref.blocks):
            fused_block.load_from_reference(ref_block)

        # Copy head conv/bn
        self.head.load_from_conv_bn(ref.conv2, ref.bn2)

        # Copy FC
        with torch.no_grad():
            self.fc.weight.copy_(ref.fc.weight)
            if self.fc.bias is not None and ref.fc.bias is not None:
                self.fc.bias.copy_(ref.fc.bias)


# =========================
# Tests
# =========================
def run_tests():
    torch.manual_seed(0)

    # Instantiate reference and fused models
    init_args = get_init_inputs()
    ref_model = OriginalModel(*init_args)
    fused_model = Model(*init_args)

    # Copy weights from reference into fused
    fused_model.load_from_reference(ref_model)

    # Eval mode for deterministic BN
    ref_model.eval()
    fused_model.eval()

    # Deterministic input
    torch.manual_seed(123)
    inputs = get_inputs()
    x = inputs[0]

    with torch.no_grad():
        out_ref = ref_model(x)
        out_fused = fused_model(x)

    # Numerical equivalence check
    if not torch.allclose(out_ref, out_fused, rtol=1e-5, atol=1e-6):
        max_abs = (out_ref - out_fused).abs().max().item()
        print(f"FAIL: outputs differ. max_abs_diff={max_abs}")
        sys.exit(1)

    print("PASS")
    sys.exit(0)


if __name__ == "__main__":
    run_tests()
