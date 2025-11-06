import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------
# Reference (original) network from the problem file
# ------------------------------
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        """
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param stride: Stride for the first convolutional layer
        :param downsample: Downsample layer for the shortcut connection
        """
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        """
        :param x: Input tensor, shape (batch_size, in_channels, height, width)
        :return: Output tensor, shape (batch_size, out_channels, height, width)
        """
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Model(nn.Module):
    def __init__(self, num_classes=1000):
        """
        :param num_classes: Number of output classes
        """
        super(Model, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: Input tensor, shape (batch_size, 3, height, width)
        :return: Output tensor, shape (batch_size, num_classes)
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

# Test helpers (as in the problem file)
batch_size = 2
num_classes = 1000
input_shape = (batch_size, 3, 224, 224)
def get_inputs():
    return [torch.rand(input_shape)]
def get_init_inputs():
    return [num_classes]

# ------------------------------
# Fused subgraphs (inference-friendly modules) with explicit shape contracts
# ------------------------------

class ConvBNReLU2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, inplace=True):
        """
        Fused Conv2d + BatchNorm2d + ReLU (in-place).

        Shape contract:
        - Input:  (N, C_in, H, W)
        - Output: (N, C_out, H_out, W_out)
          where H_out = floor((H + 2*padding - kernel_size)/stride) + 1
                W_out = floor((W + 2*padding - kernel_size)/stride) + 1
        """
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        """
        :param x: Tensor of shape (N, C_in, H, W)
        :return: Tensor of shape (N, C_out, H_out, W_out)
        """
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class MaxPool2dModule(nn.Module):
    def __init__(self, kernel_size=3, stride=2, padding=1):
        """
        Wrapper for MaxPool2d.

        Shape contract:
        - Input:  (N, C, H, W)
        - Output: (N, C, H_out, W_out)
          where H_out = floor((H + 2*padding - kernel_size)/stride) + 1
                W_out = floor((W + 2*padding - kernel_size)/stride) + 1
        """
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        """
        :param x: Tensor of shape (N, C, H, W)
        :return: Tensor of shape (N, C, H_out, W_out)
        """
        return self.pool(x)


class DownsampleConvBN(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        """
        Fused 1x1 Conv2d + BatchNorm2d used on residual shortcuts.

        Shape contract:
        - Input:  (N, C_in, H, W)
        - Output: (N, C_out, H_out, W_out)
          where H_out = floor((H - 1)/stride) + 1  [for 1x1 conv with no padding]
                W_out = floor((W - 1)/stride) + 1
        """
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        """
        :param x: Tensor of shape (N, C_in, H, W)
        :return: Tensor of shape (N, C_out, H_out, W_out)
        """
        return self.bn(self.conv(x))


class BasicBlockFused(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1, downsample: DownsampleConvBN = None, inplace_relu=True):
        """
        Fused residual BasicBlock: [Conv3x3->BN->ReLU] -> [Conv3x3->BN] + identity (with optional downsample) -> ReLU

        Shape contract:
        - Input:  (N, C_in, H, W)
        - Output: (N, C_out, H', W')
          where:
            If stride == 1 and no downsample:
                H' = H, W' = W, C_out == out_channels
            If stride > 1 or C_in != C_out:
                identity path is downsampled to match (N, C_out, H', W')
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=inplace_relu)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        """
        :param x: Tensor of shape (N, C_in, H, W)
        :return: Tensor of shape (N, C_out, H', W')
        """
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.relu(out + identity)
        return out


class ResNetStage(nn.Module):
    def __init__(self, blocks):
        """
        Fused stage that stacks multiple BasicBlockFused modules.

        Shape contract:
        - Input:  (N, C_in, H, W)
        - Output: (N, C_out, H', W')
          where the first block may change spatial resolution and channels;
          subsequent blocks keep (C_out, H', W') constant.
        """
        super().__init__()
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        """
        :param x: Tensor of shape (N, C_in, H, W)
        :return: Tensor of shape (N, C_out, H', W')
        """
        for b in self.blocks:
            x = b(x)
        return x


class AdaptiveAvgPoolFlatten(nn.Module):
    def __init__(self):
        """
        Fused global average pooling to (1, 1) followed by flatten.

        Shape contract:
        - Input:  (N, C, H, W)
        - Output: (N, C)
        """
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        """
        :param x: Tensor of shape (N, C, H, W)
        :return: Tensor of shape (N, C)
        """
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return x


class LinearClassifier(nn.Module):
    def __init__(self, in_features, num_classes):
        """
        Linear classifier head.

        Shape contract:
        - Input:  (N, D) where D == in_features
        - Output: (N, num_classes)
        """
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        """
        :param x: Tensor of shape (N, D)
        :return: Tensor of shape (N, num_classes)
        """
        return self.fc(x)


class FusedResNet(nn.Module):
    def __init__(self, num_classes=1000, inplace_relu=True):
        """
        End-to-end fused model that mirrors the reference architecture but decomposed
        into fusable subgraphs.

        Subgraphs:
        - stem: Conv7x7->BN->ReLU
        - pool: MaxPool 3x3 /2
        - layer1..layer4: ResNetStage of BasicBlockFused
        - head: AdaptiveAvgPool2d->Flatten->Linear
        """
        super().__init__()
        # Stem
        self.stem = ConvBNReLU2d(3, 64, kernel_size=7, stride=2, padding=3, inplace=inplace_relu)
        self.pool = MaxPool2dModule(kernel_size=3, stride=2, padding=1)

        # Stages (2 blocks each), mirroring the reference
        self.layer1 = self._make_stage(in_ch=64, out_ch=64, blocks=2, first_stride=1, inplace_relu=inplace_relu)
        self.layer2 = self._make_stage(in_ch=64, out_ch=128, blocks=2, first_stride=2, inplace_relu=inplace_relu)
        self.layer3 = self._make_stage(in_ch=128, out_ch=256, blocks=2, first_stride=2, inplace_relu=inplace_relu)
        self.layer4 = self._make_stage(in_ch=256, out_ch=512, blocks=2, first_stride=2, inplace_relu=inplace_relu)

        # Head
        self.avgpool_flatten = AdaptiveAvgPoolFlatten()
        self.classifier = LinearClassifier(in_features=512 * BasicBlockFused.expansion, num_classes=num_classes)

    def _make_stage(self, in_ch, out_ch, blocks, first_stride=1, inplace_relu=True):
        downsample = None
        if first_stride != 1 or in_ch != out_ch * BasicBlockFused.expansion:
            downsample = DownsampleConvBN(in_channels=in_ch, out_channels=out_ch * BasicBlockFused.expansion,
                                          stride=first_stride)
        layers = []
        layers.append(BasicBlockFused(in_ch, out_ch, stride=first_stride, downsample=downsample,
                                      inplace_relu=inplace_relu))
        in_ch = out_ch * BasicBlockFused.expansion
        for _ in range(1, blocks):
            layers.append(BasicBlockFused(in_ch, out_ch, stride=1, downsample=None, inplace_relu=inplace_relu))
        return ResNetStage(layers)

    def forward(self, x):
        """
        :param x: Input tensor, shape (N, 3, H, W)
        :return: Output tensor, shape (N, num_classes)
        """
        x = self.stem(x)
        x = self.pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool_flatten(x)
        x = self.classifier(x)
        return x


# ------------------------------
# Utilities to copy weights from the reference Model to FusedResNet
# ------------------------------

def _copy_bn_(dst_bn: nn.BatchNorm2d, src_bn: nn.BatchNorm2d):
    with torch.no_grad():
        dst_bn.weight.copy_(src_bn.weight)
        dst_bn.bias.copy_(src_bn.bias)
        dst_bn.running_mean.copy_(src_bn.running_mean)
        dst_bn.running_var.copy_(src_bn.running_var)
        if hasattr(dst_bn, "num_batches_tracked") and hasattr(src_bn, "num_batches_tracked"):
            dst_bn.num_batches_tracked.copy_(src_bn.num_batches_tracked)

def _copy_conv_(dst_conv: nn.Conv2d, src_conv: nn.Conv2d):
    with torch.no_grad():
        dst_conv.weight.copy_(src_conv.weight)
        if src_conv.bias is not None and dst_conv.bias is not None:
            dst_conv.bias.copy_(src_conv.bias)

def _copy_linear_(dst_linear: nn.Linear, src_linear: nn.Linear):
    with torch.no_grad():
        dst_linear.weight.copy_(src_linear.weight)
        if src_linear.bias is not None and dst_linear.bias is not None:
            dst_linear.bias.copy_(src_linear.bias)

def convert_basicblock_(dst: BasicBlockFused, src: BasicBlock):
    # conv1/bn1
    _copy_conv_(dst.conv1, src.conv1)
    _copy_bn_(dst.bn1, src.bn1)
    # conv2/bn2
    _copy_conv_(dst.conv2, src.conv2)
    _copy_bn_(dst.bn2, src.bn2)
    # downsample if present
    if src.downsample is not None:
        assert isinstance(dst.downsample, DownsampleConvBN) and isinstance(src.downsample, nn.Sequential)
        src_ds_conv = src.downsample[0]
        src_ds_bn = src.downsample[1]
        _copy_conv_(dst.downsample.conv, src_ds_conv)
        _copy_bn_(dst.downsample.bn, src_ds_bn)

def load_from_reference(fused: FusedResNet, ref: Model):
    # Stem
    _copy_conv_(fused.stem.conv, ref.conv1)
    _copy_bn_(fused.stem.bn, ref.bn1)
    # Pool: no params

    # Layers: layer1..4 each with 2 blocks
    ref_layers = [ref.layer1, ref.layer2, ref.layer3, ref.layer4]
    fused_layers = [fused.layer1, fused.layer2, fused.layer3, fused.layer4]
    for ref_stage, fused_stage in zip(ref_layers, fused_layers):
        for ref_b, fused_b in zip(ref_stage, fused_stage.blocks):
            convert_basicblock_(fused_b, ref_b)

    # Head
    _copy_linear_(fused.classifier.fc, ref.fc)


# ------------------------------
# Tests
# ------------------------------

def run_tests():
    # Determinism
    torch.manual_seed(0)

    # Initialize reference model and fused model with same topology
    init_args = get_init_inputs()
    ref_model = Model(*init_args).eval()
    fused_model = FusedResNet(*init_args).eval()

    # Copy weights from ref to fused
    load_from_reference(fused_model, ref_model)

    # Prepare inputs
    torch.manual_seed(0)
    inputs = get_inputs()
    x = inputs[0]

    # Run both models
    with torch.no_grad():
        y_ref = ref_model(x)
        y_fused = fused_model(x)

    # Validate shapes
    assert y_ref.shape == y_fused.shape, f"Shape mismatch: ref {y_ref.shape} vs fused {y_fused.shape}"

    # Validate numerical closeness
    max_abs_err = (y_ref - y_fused).abs().max().item()
    max_rel_err = ((y_ref - y_fused).abs() / (y_ref.abs().clamp_min(1e-8))).max().item()
    atol = 1e-6
    rtol = 1e-6
    close = torch.allclose(y_ref, y_fused, atol=atol, rtol=rtol)

    if not close:
        # Allow slightly higher tolerance to account for floating point nuances
        close = torch.allclose(y_ref, y_fused, atol=1e-5, rtol=1e-5)

    if not close:
        print(f"FAILED: max_abs_err={max_abs_err:.3e}, max_rel_err={max_rel_err:.3e}")
        sys.exit(1)

    print("PASS")
    sys.exit(0)


if __name__ == "__main__":
    run_tests()
