import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import random

# ------------------------
# Reference (original) model from the problem statement
# ------------------------

class ReferenceDenseBlock(nn.Module):
    def __init__(self, num_layers: int, num_input_features: int, growth_rate: int):
        super(ReferenceDenseBlock, self).__init__()
        layers = []
        for i in range(num_layers):
            layers.append(self._make_layer(num_input_features + i * growth_rate, growth_rate))
        self.layers = nn.ModuleList(layers)

    def _make_layer(self, in_features: int, growth_rate: int):
        return nn.Sequential(
            nn.BatchNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_features, growth_rate, kernel_size=3, padding=1, bias=False),
            nn.Dropout(0.0)
        )

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            new_feature = layer(x)
            features.append(new_feature)
            x = torch.cat(features, 1)
        return x

class ReferenceTransitionLayer(nn.Module):
    def __init__(self, num_input_features: int, num_output_features: int):
        super(ReferenceTransitionLayer, self).__init__()
        self.transition = nn.Sequential(
            nn.BatchNorm2d(num_input_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_input_features, num_output_features, kernel_size=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.transition(x)

class ReferenceModel(nn.Module):
    def __init__(self, growth_rate: int = 32, num_classes: int = 1000):
        super(ReferenceModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        num_features = 64
        block_layers = [6, 12, 24, 16]
        self.dense_blocks = nn.ModuleList()
        self.transition_layers = nn.ModuleList()

        for i, num_layers in enumerate(block_layers):
            block = ReferenceDenseBlock(num_layers=num_layers, num_input_features=num_features, growth_rate=growth_rate)
            self.dense_blocks.append(block)
            num_features = num_features + num_layers * growth_rate

            if i != len(block_layers) - 1:
                transition = ReferenceTransitionLayer(num_input_features=num_features, num_output_features=num_features // 2)
                self.transition_layers.append(transition)
                num_features = num_features // 2

        self.final_bn = nn.BatchNorm2d(num_features)
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)

        for i, block in enumerate(self.dense_blocks):
            x = block(x)
            if i != len(self.dense_blocks) - 1:
                x = self.transition_layers[i](x)

        x = self.final_bn(x)
        x = F.relu(x, inplace=True)
        x = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)
        x = self.classifier(x)
        return x

# ------------------------
# Fused modules
# ------------------------

class FusedStem(nn.Module):
    """
    Fused initial stem: Conv7x7(stride=2) -> BatchNorm -> ReLU -> MaxPool(3x3, stride=2)

    Shape contract:
    - Input:  x of shape (N, 3, H, W)
    - Output: y of shape (N, 64, H2, W2)
      where H1 = floor((H + 2*3 - 7)/2 + 1) = floor((H + 1)/2)
            H2 = floor((H1 + 2*1 - 3)/2 + 1) = floor((H1 + 1)/2)
      Similarly for W.
    """
    def __init__(self):
        super(FusedStem, self).__init__()
        self.conv = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x

class FusedDenseLayer(nn.Module):
    """
    One DenseNet layer: BatchNorm -> ReLU -> Conv3x3 -> Dropout

    Shape contract:
    - Input:  x of shape (N, C_in, H, W)
    - Output: y of shape (N, growth_rate, H, W)
    """
    def __init__(self, in_features: int, growth_rate: int):
        super(FusedDenseLayer, self).__init__()
        self.bn = nn.BatchNorm2d(in_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_features, growth_rate, kernel_size=3, padding=1, bias=False)
        self.dropout = nn.Dropout(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv(x)
        x = self.dropout(x)
        return x

class FusedDenseBlock(nn.Module):
    """
    Fused DenseBlock that encapsulates the dense connectivity pattern.

    Shape contract:
    - Input:  x of shape (N, C_in, H, W)
    - Output: y of shape (N, C_out, H, W), where
              C_out = C_in + num_layers * growth_rate
    """
    def __init__(self, num_layers: int, num_input_features: int, growth_rate: int):
        super(FusedDenseBlock, self).__init__()
        self.num_layers = num_layers
        self.num_input_features = num_input_features
        self.growth_rate = growth_rate
        layers = []
        for i in range(num_layers):
            in_features = num_input_features + i * growth_rate
            layers.append(FusedDenseLayer(in_features, growth_rate))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = [x]
        current = x
        for layer in self.layers:
            new_feature = layer(current)
            features.append(new_feature)
            current = torch.cat(features, dim=1)
        return current

class FusedTransition(nn.Module):
    """
    Fused transition layer: BatchNorm -> ReLU -> 1x1 Conv -> AvgPool(stride=2)

    Shape contract:
    - Input:  x of shape (N, C_in, H, W)
    - Output: y of shape (N, C_out, H2, W2), where H2 = floor(H/2), W2 = floor(W/2)
    """
    def __init__(self, num_input_features: int, num_output_features: int):
        super(FusedTransition, self).__init__()
        self.bn = nn.BatchNorm2d(num_input_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(num_input_features, num_output_features, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv(x)
        x = self.pool(x)
        return x

class FusedHead(nn.Module):
    """
    Fused classification head: BatchNorm -> ReLU -> Global AvgPool -> Flatten -> Linear

    Shape contract:
    - Input:  x of shape (N, C_in, H, W)
    - Output: y of shape (N, num_classes)
    """
    def __init__(self, num_features: int, num_classes: int):
        super(FusedHead, self).__init__()
        self.bn = nn.BatchNorm2d(num_features)
        self.num_features = num_features
        self.fc = nn.Linear(num_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn(x)
        x = F.relu(x, inplace=True)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)  # (N, C)
        x = self.fc(x)
        return x

class Model(nn.Module):
    """
    Fused DenseNet-like model composed entirely of fused subgraphs.

    Topology:
    - FusedStem
    - [FusedDenseBlock -> FusedTransition]* for all but last block
    - FusedDenseBlock (last, no transition)
    - FusedHead
    """
    def __init__(self, growth_rate: int = 32, num_classes: int = 1000):
        super(Model, self).__init__()
        self.stem = FusedStem()

        num_features = 64
        self.block_layers = [6, 12, 24, 16]
        self.dense_blocks = nn.ModuleList()
        self.transitions = nn.ModuleList()

        for i, num_layers in enumerate(self.block_layers):
            block = FusedDenseBlock(num_layers=num_layers, num_input_features=num_features, growth_rate=growth_rate)
            self.dense_blocks.append(block)
            num_features = num_features + num_layers * growth_rate
            if i != len(self.block_layers) - 1:
                transition = FusedTransition(num_input_features=num_features, num_output_features=num_features // 2)
                self.transitions.append(transition)
                num_features = num_features // 2

        self.head = FusedHead(num_features=num_features, num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for i, block in enumerate(self.dense_blocks):
            x = block(x)
            if i != len(self.dense_blocks) - 1:
                x = self.transitions[i](x)
        x = self.head(x)
        return x

# ------------------------
# Helpers from the problem file (exact replicas)
# ------------------------
batch_size = 10
num_classes = 10
height, width = 224, 224  # Standard input size for DenseNet

def get_inputs():
    return [torch.rand(batch_size, 3, height, width)]

def get_init_inputs():
    return [32, num_classes]

# ------------------------
# Weight copying utilities
# ------------------------

def _copy_bn_(dst_bn: nn.BatchNorm2d, src_bn: nn.BatchNorm2d):
    dst_bn.weight.data.copy_(src_bn.weight.data)
    dst_bn.bias.data.copy_(src_bn.bias.data)
    dst_bn.running_mean.data.copy_(src_bn.running_mean.data)
    dst_bn.running_var.data.copy_(src_bn.running_var.data)
    if hasattr(dst_bn, "num_batches_tracked") and hasattr(src_bn, "num_batches_tracked"):
        dst_bn.num_batches_tracked.data.copy_(src_bn.num_batches_tracked.data)

def copy_state_from_reference(fused: Model, ref: ReferenceModel):
    # Stem
    fused.stem.conv.weight.data.copy_(ref.features[0].weight.data)
    _copy_bn_(fused.stem.bn, ref.features[1])

    # Blocks and transitions
    for bi, (ref_block, fused_block) in enumerate(zip(ref.dense_blocks, fused.dense_blocks)):
        for li, (ref_layer_seq, fused_layer) in enumerate(zip(ref_block.layers, fused_block.layers)):
            # ref_layer_seq: [BN, ReLU, Conv, Dropout]
            _copy_bn_(fused_layer.bn, ref_layer_seq[0])
            fused_layer.conv.weight.data.copy_(ref_layer_seq[2].weight.data)

        # Transition if not last block
        if bi != len(ref.dense_blocks) - 1:
            ref_trans_seq = ref.transition_layers[bi].transition  # [BN, ReLU, Conv, AvgPool]
            fused_trans = fused.transitions[bi]
            _copy_bn_(fused_trans.bn, ref_trans_seq[0])
            fused_trans.conv.weight.data.copy_(ref_trans_seq[2].weight.data)

    # Head
    _copy_bn_(fused.head.bn, ref.final_bn)
    fused.head.fc.weight.data.copy_(ref.classifier.weight.data)
    fused.head.fc.bias.data.copy_(ref.classifier.bias.data)

# ------------------------
# Tests
# ------------------------

def set_seed(seed: int = 0):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def run_tests():
    set_seed(0)
    device = torch.device("cpu")  # keep deterministic and broadly compatible

    # Instantiate models
    growth_rate, num_cls = get_init_inputs()
    ref = ReferenceModel(growth_rate, num_cls).to(device).eval()
    fused = Model(growth_rate, num_cls).to(device).eval()

    # Copy weights from reference into fused model
    copy_state_from_reference(fused, ref)

    # Prepare inputs
    set_seed(0)  # ensure deterministic inputs
    inputs = get_inputs()[0].to(device)

    # Run both
    with torch.no_grad():
        out_ref = ref(inputs)
        out_fused = fused(inputs)

    # Check numerical equivalence
    if not torch.allclose(out_ref, out_fused, rtol=0, atol=0):
        # If strict equality fails due to subtle ordering, relax slightly
        if not torch.allclose(out_ref, out_fused, rtol=1e-6, atol=1e-6):
            max_abs = (out_ref - out_fused).abs().max().item()
            print(f"Mismatch: max abs diff {max_abs}")
            sys.exit(1)

    print("PASS")
    sys.exit(0)

if __name__ == "__main__":
    run_tests()
