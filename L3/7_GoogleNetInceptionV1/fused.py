import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================
# Original model (baseline)
# =========================

class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_1x1, reduce_3x3, out_3x3, reduce_5x5, out_5x5, pool_proj):
        """
        :param in_channels: Number of input channels
        :param out_1x1: Number of output channels for the 1x1 convolution
        :param reduce_3x3: Number of output channels for the 1x1 reduction before 3x3 convolution
        :param out_3x3: Number of output channels for the 3x3 convolution
        :param reduce_5x5: Number of output channels for the 1x1 reduction before 5x5 convolution
        :param out_5x5: Number of output channels for the 5x5 convolution
        :param pool_proj: Number of output channels for the pooling projection
        """
        super(InceptionModule, self).__init__()

        # 1x1 convolution branch
        self.branch1x1 = nn.Conv2d(in_channels, out_1x1, kernel_size=1)

        # 3x3 convolution branch
        self.branch3x3 = nn.Sequential(
            nn.Conv2d(in_channels, reduce_3x3, kernel_size=1),
            nn.Conv2d(reduce_3x3, out_3x3, kernel_size=3, padding=1)
        )

        # 5x5 convolution branch
        self.branch5x5 = nn.Sequential(
            nn.Conv2d(in_channels, reduce_5x5, kernel_size=1),
            nn.Conv2d(reduce_5x5, out_5x5, kernel_size=5, padding=2)
        )

        # Max pooling branch
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1)
        )

    def forward(self, x):
        """
        :param x: Input tensor, shape (batch_size, in_channels, height, width)
        :return: Output tensor, shape (batch_size, out_channels, height, width)
        """
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3(x)
        branch5x5 = self.branch5x5(x)
        branch_pool = self.branch_pool(x)

        outputs = [branch1x1, branch3x3, branch5x5, branch_pool]
        return torch.cat(outputs, 1)

class Model(nn.Module):
    def __init__(self, num_classes=1000):
        """
        :param num_classes: Number of output classes
        """
        super(Model, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=1)
        self.conv3 = nn.Conv2d(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, padding=1)

        self.inception3a = InceptionModule(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = InceptionModule(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, padding=1)

        self.inception4a = InceptionModule(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = InceptionModule(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = InceptionModule(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = InceptionModule(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = InceptionModule(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(3, stride=2, padding=1)

        self.inception5a = InceptionModule(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = InceptionModule(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.0)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        """
        :param x: Input tensor, shape (batch_size, 3, height, width)
        :return: Output tensor, shape (batch_size, num_classes)
        """
        x = self.maxpool1(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = self.maxpool2(F.relu(self.conv3(x)))

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)

        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)

        x = self.inception5a(x)
        x = self.inception5b(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        return x

# =========================
# Refactored, fusion-friendly modules
# =========================

class ConvReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=True):
        """
        Fused Conv2d + ReLU.
        - Input: (N, C_in, H, W)
        - Output after conv: (N, C_out, Hc, Wc)
          Hc = floor((H + 2*padding - dilation*(kernel_size-1) - 1)/stride) + 1
          Wc = floor((W + 2*padding - dilation*(kernel_size-1) - 1)/stride) + 1
        - Output after ReLU: same shape as conv output.

        Example for this model:
          - conv2: (N, 64, 56, 56) -> (N, 64, 56, 56) -> (N, 64, 56, 56)
        """
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              padding, dilation, groups, bias)

    def forward(self, x):
        return F.relu(self.conv(x))


class ConvReLUPool(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool, stride=1, padding=0,
                 dilation=1, groups=1, bias=True):
        """
        Fused Conv2d + ReLU + Pool.
        - Input: (N, C_in, H, W)
        - After conv: (N, C_out, Hc, Wc)
          Hc = floor((H + 2*padding - dilation*(kernel_size-1) - 1)/stride) + 1
        - After ReLU: same shape
        - After pooled output Hp, Wp depend on pool hyperparams.

        Examples for this model:
          - stem conv1+relu+pool1:
              conv1: (N, 3, 224, 224) -> (N, 64, 112, 112) with k=7,s=2,p=3
              pool1: -> (N, 64, 56, 56) with k=3,s=2,p=1
          - conv3+relu+pool2:
              conv3: (N, 64, 56, 56) -> (N, 192, 56, 56) with k=3,s=1,p=1
              pool2: -> (N, 192, 28, 28) with k=3,s=2,p=1
        """
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              padding, dilation, groups, bias)
        self.pool = pool

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = self.pool(x)
        return x


class InceptionBranch1x1(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        A single 1x1 conv branch.
        - Input: (N, C_in, H, W)
        - Output: (N, out_channels, H, W)
        """
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv1x1(x)


class InceptionBranch3x3(nn.Module):
    def __init__(self, in_channels, reduce_channels, out_channels):
        """
        A 1x1 reduction followed by 3x3 conv branch (no activations in the original).
        - Input: (N, C_in, H, W)
        - After 1x1: (N, reduce_channels, H, W)
        - After 3x3: (N, out_channels, H, W) with padding=1
        """
        super().__init__()
        self.reduce = nn.Conv2d(in_channels, reduce_channels, kernel_size=1)
        self.conv3x3 = nn.Conv2d(reduce_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.reduce(x)
        x = self.conv3x3(x)
        return x


class InceptionBranch5x5(nn.Module):
    def __init__(self, in_channels, reduce_channels, out_channels):
        """
        A 1x1 reduction followed by 5x5 conv branch (no activations in the original).
        - Input: (N, C_in, H, W)
        - After 1x1: (N, reduce_channels, H, W)
        - After 5x5: (N, out_channels, H, W) with padding=2
        """
        super().__init__()
        self.reduce = nn.Conv2d(in_channels, reduce_channels, kernel_size=1)
        self.conv5x5 = nn.Conv2d(reduce_channels, out_channels, kernel_size=5, padding=2)

    def forward(self, x):
        x = self.reduce(x)
        x = self.conv5x5(x)
        return x


class InceptionBranchPoolProj(nn.Module):
    def __init__(self, in_channels, proj_channels):
        """
        A 3x3 max-pool (stride=1, padding=1) followed by 1x1 conv projection.
        - Input: (N, C_in, H, W)
        - After pool: (N, C_in, H, W)
        - After proj 1x1: (N, proj_channels, H, W)
        """
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.proj = nn.Conv2d(in_channels, proj_channels, kernel_size=1)

    def forward(self, x):
        x = self.pool(x)
        x = self.proj(x)
        return x


class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_1x1, reduce_3x3, out_3x3, reduce_5x5, out_5x5, pool_proj):
        """
        Fused Inception block aggregating four parallel branches and concatenating.
        No activations are used inside branches (matching the original implementation).
        - Input:  (N, in_channels, H, W)
        - Branch outputs:
            b1: (N, out_1x1, H, W)
            b2: (N, out_3x3, H, W)
            b3: (N, out_5x5, H, W)
            b4: (N, pool_proj, H, W)
        - Concat along C: (N, out_1x1 + out_3x3 + out_5x5 + pool_proj, H, W)
        """
        super().__init__()
        self.branch1x1 = InceptionBranch1x1(in_channels, out_1x1)
        self.branch3x3 = InceptionBranch3x3(in_channels, reduce_3x3, out_3x3)
        self.branch5x5 = InceptionBranch5x5(in_channels, reduce_5x5, out_5x5)
        self.branch_pool = InceptionBranchPoolProj(in_channels, pool_proj)

    def forward(self, x):
        b1 = self.branch1x1(x)
        b2 = self.branch3x3(x)
        b3 = self.branch5x5(x)
        b4 = self.branch_pool(x)
        return torch.cat([b1, b2, b3, b4], dim=1)


class AvgPoolFlattenDropoutLinear(nn.Module):
    def __init__(self, in_features, num_classes, dropout_p=0.0):
        """
        Fused classification head.
        Steps:
          - AdaptiveAvgPool2d to (1,1)
          - Flatten to (N, in_features)
          - Dropout(p)
          - Linear to (N, num_classes)

        Shapes for this model:
          - Input: (N, 1024, 7, 7) after final inception
          - After avgpool: (N, 1024, 1, 1)
          - Flatten: (N, 1024)
          - Linear: (N, num_classes)
        """
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout_p)
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class FusedModel(nn.Module):
    def __init__(self, num_classes=1000):
        """
        Refactored model using fusion-friendly submodules.

        High-level shapes through the network (for input (N,3,224,224)):
          - stem (conv7x7 s2 + relu + maxpool s2): (N,64,56,56)
          - conv2 (1x1 + relu): (N,64,56,56)
          - conv3 (3x3 + relu + maxpool s2): (N,192,28,28)
          - inception3a: (N,256,28,28)
          - inception3b: (N,480,28,28)
          - maxpool3 s2: (N,480,14,14)
          - inception4a: (N,512,14,14)
          - inception4b: (N,512,14,14)
          - inception4c: (N,512,14,14)
          - inception4d: (N,528,14,14)
          - inception4e: (N,832,14,14)
          - maxpool4 s2: (N,832,7,7)
          - inception5a: (N,832,7,7)
          - inception5b: (N,1024,7,7)
          - head: (N,num_classes)
        """
        super().__init__()

        # Fused stem: conv1 + relu + maxpool1
        self.stem1 = ConvReLUPool(
            in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3,
            pool=nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        # Conv2 + ReLU
        self.conv2_relu = ConvReLU(64, 64, kernel_size=1)
        # Conv3 + ReLU + MaxPool2
        self.conv3_relu_pool = ConvReLUPool(
            in_channels=64, out_channels=192, kernel_size=3, padding=1,
            pool=nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # Inception blocks and pools
        self.inception3a = InceptionBlock(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = InceptionBlock(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception4a = InceptionBlock(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = InceptionBlock(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = InceptionBlock(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = InceptionBlock(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = InceptionBlock(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception5a = InceptionBlock(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = InceptionBlock(832, 384, 192, 384, 48, 128, 128)

        # Fused classification head
        self.head = AvgPoolFlattenDropoutLinear(in_features=1024, num_classes=num_classes, dropout_p=0.0)

    def forward(self, x):
        x = self.stem1(x)             # (N, 64, 56, 56)
        x = self.conv2_relu(x)        # (N, 64, 56, 56)
        x = self.conv3_relu_pool(x)   # (N, 192, 28, 28)

        x = self.inception3a(x)       # (N, 256, 28, 28)
        x = self.inception3b(x)       # (N, 480, 28, 28)
        x = self.maxpool3(x)          # (N, 480, 14, 14)

        x = self.inception4a(x)       # (N, 512, 14, 14)
        x = self.inception4b(x)       # (N, 512, 14, 14)
        x = self.inception4c(x)       # (N, 512, 14, 14)
        x = self.inception4d(x)       # (N, 528, 14, 14)
        x = self.inception4e(x)       # (N, 832, 14, 14)
        x = self.maxpool4(x)          # (N, 832, 7, 7)

        x = self.inception5a(x)       # (N, 832, 7, 7)
        x = self.inception5b(x)       # (N, 1024, 7, 7)

        x = self.head(x)              # (N, num_classes)
        return x


# =========================
# Test helpers (from problem file)
# =========================

batch_size = 10
input_channels = 3
height = 224
width = 224
num_classes = 1000

def get_inputs():
    return [torch.rand(batch_size, input_channels, height, width)]

def get_init_inputs():
    return [num_classes]


# =========================
# Weight copy utilities
# =========================

def _copy_conv(dst: nn.Conv2d, src: nn.Conv2d):
    with torch.no_grad():
        dst.weight.copy_(src.weight)
        if dst.bias is not None and src.bias is not None:
            dst.bias.copy_(src.bias)

def _copy_inception_block(dst: InceptionBlock, src: InceptionModule):
    # branch1x1
    _copy_conv(dst.branch1x1.conv1x1, src.branch1x1)
    # branch3x3 (reduce then conv)
    _copy_conv(dst.branch3x3.reduce, src.branch3x3[0])
    _copy_conv(dst.branch3x3.conv3x3, src.branch3x3[1])
    # branch5x5 (reduce then conv)
    _copy_conv(dst.branch5x5.reduce, src.branch5x5[0])
    _copy_conv(dst.branch5x5.conv5x5, src.branch5x5[1])
    # pool->proj (only conv has params)
    _copy_conv(dst.branch_pool.proj, src.branch_pool[1])

def copy_weights_from_original(fused: FusedModel, orig: Model):
    # Stem conv1
    _copy_conv(fused.stem1.conv, orig.conv1)
    # conv2
    _copy_conv(fused.conv2_relu.conv, orig.conv2)
    # conv3
    _copy_conv(fused.conv3_relu_pool.conv, orig.conv3)

    # Inception blocks
    _copy_inception_block(fused.inception3a, orig.inception3a)
    _copy_inception_block(fused.inception3b, orig.inception3b)

    _copy_inception_block(fused.inception4a, orig.inception4a)
    _copy_inception_block(fused.inception4b, orig.inception4b)
    _copy_inception_block(fused.inception4c, orig.inception4c)
    _copy_inception_block(fused.inception4d, orig.inception4d)
    _copy_inception_block(fused.inception4e, orig.inception4e)

    _copy_inception_block(fused.inception5a, orig.inception5a)
    _copy_inception_block(fused.inception5b, orig.inception5b)

    # Head FC
    with torch.no_grad():
        fused.head.fc.weight.copy_(orig.fc.weight)
        fused.head.fc.bias.copy_(orig.fc.bias)


# =========================
# Tests
# =========================

def run_tests():
    # Determinism
    torch.manual_seed(0)
    torch.set_grad_enabled(False)

    # Initialize models
    init_args = get_init_inputs()
    num_classes_local = init_args[0] if len(init_args) > 0 else 1000
    orig = Model(num_classes=num_classes_local).eval()
    fused = FusedModel(num_classes=num_classes_local).eval()

    # Copy weights
    copy_weights_from_original(fused, orig)

    # Create identical inputs
    torch.manual_seed(1234)
    inputs = get_inputs()
    x = inputs[0]

    with torch.no_grad():
        y_orig = orig(x)
        y_fused = fused(x)

    # Validate shapes
    assert y_orig.shape == y_fused.shape, f"Shape mismatch: {y_orig.shape} vs {y_fused.shape}"

    # Validate numerical equivalence
    if not torch.allclose(y_orig, y_fused, rtol=0, atol=1e-6):
        max_abs = (y_orig - y_fused).abs().max().item()
        raise AssertionError(f"Outputs differ, max abs diff = {max_abs}")

    print("PASS")
    sys.exit(0)


if __name__ == "__main__":
    run_tests()
