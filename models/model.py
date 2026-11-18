"""TREF-Net reconstructed from the manuscript's architecture and equations.

This is a new implementation. Train a checkpoint for this model before reporting
accuracy or timing. The existing 7.58 M checkpoint is for a different model.
"""

from typing import NamedTuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2


def conv_norm_act(in_channels, out_channels, kernel_size, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                  kernel_size // 2, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU6(inplace=True),
    )


def resize(feature, size):
    return F.interpolate(feature, size=size, mode="bilinear", align_corners=False)


class MobileViTBlock(nn.Module):
    """Merge local convolutional features with attention across image patches."""

    def __init__(self, channels=128, transformer_dim=128, ffn_dim=512):
        super().__init__()
        self.local = conv_norm_act(channels, channels, 3)
        self.token_projection = nn.Conv2d(channels, transformer_dim, 1)
        layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim, nhead=4, dim_feedforward=ffn_dim,
            dropout=0.0, activation="gelu", batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=2)
        self.token_norm = nn.LayerNorm(transformer_dim)
        self.feature_projection = conv_norm_act(transformer_dim, channels, 1)
        self.fuse = conv_norm_act(2 * channels, channels, 3)

    def forward(self, x):
        features = self.token_projection(self.local(x))
        batch, channels, height, width = features.shape
        pad_height, pad_width = height % 2, width % 2
        features = F.pad(features, (0, pad_width, 0, pad_height))
        rows, columns = features.shape[-2] // 2, features.shape[-1] // 2
        # Group pixels at the same within-patch position into one sequence.
        tokens = features.reshape(batch, channels, rows, 2, columns, 2)
        tokens = tokens.permute(0, 3, 5, 2, 4, 1).reshape(batch * 4, rows * columns, channels)
        tokens = self.token_norm(self.transformer(tokens))
        features = tokens.reshape(batch, 2, 2, rows, columns, channels)
        features = features.permute(0, 5, 3, 1, 4, 2).reshape(batch, channels, 2 * rows, 2 * columns)
        features = self.feature_projection(features[..., :height, :width])
        return self.fuse(torch.cat((x, features), dim=1))


class SharedEncoder(nn.Module):
    def __init__(self, pretrained=False, transformer_ffn_dim=512):
        super().__init__()
        weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
        layers = mobilenet_v2(weights=weights).features
        self.stem = layers[0]
        self.inverted_residuals = layers[1:4]
        self.mid_projection = conv_norm_act(24, 64, 1)
        self.mobilevit_input = conv_norm_act(64, 128, 3, stride=2)
        self.mobilevit = MobileViTBlock(ffn_dim=transformer_ffn_dim)
        self.downsample = conv_norm_act(128, 256, 3, stride=2)

    def forward(self, x):
        shallow = self.stem(x)
        mid = self.mid_projection(self.inverted_residuals(shallow))
        semantic = self.mobilevit(self.mobilevit_input(mid))
        deep = self.downsample(semantic)
        return shallow, semantic, deep


class SpatialWeightAggregation(nn.Module):
    """Map the three encoder taps to 128 channels and concatenate them."""

    def __init__(self):
        super().__init__()
        self.shallow = conv_norm_act(32, 128, 1)
        self.mid = conv_norm_act(128, 128, 3)
        self.deep = conv_norm_act(256, 128, 3)

    def forward(self, features):
        shallow, mid, deep = features
        size = shallow.shape[-2:]
        return torch.cat((self.shallow(shallow), resize(self.mid(mid), size),
                          resize(self.deep(deep), size)), dim=1)


class CTAF_STDA(nn.Module):
    def __init__(self, channels=384, gamma=0.7):
        super().__init__()
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1]")
        self.gamma = gamma
        self.delta = 1.0 - gamma
        self.channel_map = nn.Conv2d(channels, channels, 1)
        self.attention = nn.Conv2d(channels, channels, 1)

    def calibrate(self, feature):
        mapped = self.channel_map(feature)
        gate = self.attention(F.adaptive_avg_pool2d(mapped, 1)).sigmoid()
        return mapped * gate

    def forward(self, first, second):
        first_calibrated = self.calibrate(first)
        second_calibrated = self.calibrate(second)
        difference = (first_calibrated - second_calibrated).abs()
        first_residual = (first_calibrated - first).abs()
        second_residual = (second_calibrated - second).abs()
        return self.gamma * difference + self.delta * (first_residual + second_residual)


class PyramidFusionNode(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.align = nn.Conv2d(channels, channels, 1)
        self.top_weight = nn.Conv2d(channels, channels, 1)
        self.lateral_weight = nn.Conv2d(channels, channels, 1)

    def forward(self, top, lateral):
        top = resize(top, lateral.shape[-2:])
        lateral = self.align(lateral)
        top_gate = self.top_weight(F.adaptive_avg_pool2d(top, 1)).sigmoid()
        lateral_gate = self.lateral_weight(F.adaptive_avg_pool2d(lateral, 1)).sigmoid()
        return top * top_gate + lateral * lateral_gate


class LDW_FPN(nn.Module):
    def __init__(self, channels=384, levels=4):
        super().__init__()
        self.input_projection = conv_norm_act(channels, channels, 1)
        self.downsample = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(channels, channels, 3, stride=2, padding=1,
                          groups=channels, bias=False),
                conv_norm_act(channels, channels, 1),
            )
            for _ in range(levels - 1)
        )
        self.top_projection = conv_norm_act(channels, channels, 1)
        self.fusion = nn.ModuleList(PyramidFusionNode(channels) for _ in range(levels - 1))

    def forward(self, difference):
        hierarchy = [self.input_projection(difference)]
        for downsample in self.downsample:
            hierarchy.append(downsample(hierarchy[-1]))
        top = self.top_projection(hierarchy[-1])
        for node, lateral in zip(self.fusion, reversed(hierarchy[:-1])):
            top = node(top, lateral)
        return top


class BA_AT_GCCM(nn.Module):
    def __init__(self, in_channels=384, out_channels=128):
        super().__init__()
        self.boundary = nn.Sequential(
            conv_norm_act(in_channels, out_channels, 3),
            nn.Conv2d(out_channels, out_channels, 1),
        )
        self.context = nn.Conv2d(in_channels, out_channels, 1)
        self.threshold = nn.Conv2d(out_channels, out_channels, 1)

    def forward(self, fused):
        boundary = self.boundary(fused)
        context = self.context(F.adaptive_avg_pool2d(fused, 1))
        coefficient = self.threshold(F.adaptive_avg_pool2d(boundary + context, 1)).sigmoid()
        return boundary * coefficient + context


class SAM(nn.Module):
    def __init__(self, channels=128):
        super().__init__()
        self.classifier = nn.Conv2d(channels, 1, 1)
        self.context_gate = nn.Conv2d(channels * 2, channels, 1)
        self.refine = conv_norm_act(channels, channels, 3)

    def forward(self, feature):
        foreground = self.classifier(feature).sigmoid()
        background = 1.0 - foreground
        foreground_context = F.adaptive_avg_pool2d(feature * foreground, 1)
        background_context = F.adaptive_avg_pool2d(feature * background, 1)
        gate = self.context_gate(torch.cat((foreground_context, background_context), dim=1)).sigmoid()
        return self.refine(feature + feature * gate)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.sam = SAM()
        self.refine = conv_norm_act(128, 64, 3)
        self.classifier = nn.Conv2d(64, 1, 1)

    def forward(self, refined, output_size):
        attended = self.sam(refined)
        full_resolution = resize(attended, output_size)
        return self.classifier(self.refine(full_resolution)).sigmoid()


class Prediction(NamedTuple):
    probability: Tensor


class TREF_Net(nn.Module):
    def __init__(self, pretrained=False, gamma=0.7, transformer_ffn_dim=512):
        super().__init__()
        self.encoder = SharedEncoder(pretrained, transformer_ffn_dim)
        self.swa = SpatialWeightAggregation()
        self.ctaf_stda = CTAF_STDA(gamma=gamma)
        self.ldw_fpn = LDW_FPN()
        self.ba_at_gccm = BA_AT_GCCM()
        self.decoder = Decoder()

    def forward(self, first, second, targets=None, train_mode=False):
        if targets is not None or train_mode:
            raise ValueError("Call model(first, second) and pass its probability to paper_loss().")
        if first.ndim != 4 or first.shape != second.shape or first.shape[1] != 3:
            raise ValueError("Expected matching Bx3xHxW image tensors")
        output_size = first.shape[-2:]
        first = self.swa(self.encoder(first))
        second = self.swa(self.encoder(second))
        difference = self.ctaf_stda(first, second)
        fused = self.ldw_fpn(difference)
        refined = self.ba_at_gccm(fused)
        return Prediction(self.decoder(refined, output_size))


BaseNet = TREF_Net


def paper_loss(probability, target, epsilon=1e-4):
    """Equal-weight BCE and Dice on the final prediction of the decoder."""
    if target.ndim == 3:
        target = target.unsqueeze(1)
    target = target.to(dtype=torch.float32, device=probability.device)
    probability = probability.float()
    bce = F.binary_cross_entropy(probability, target)
    overlap = (probability * target).sum()
    dice = 1.0 - 2.0 * overlap / (probability.sum() + target.sum() + epsilon)
    return 0.5 * (bce + dice)


if __name__ == "__main__":
    model = TREF_Net().eval()
    count = sum(parameter.numel() for parameter in model.parameters())
    print(f"Parameters: {count:,} ({count / 1e6:.2f} M)")
    first = torch.rand(1, 3, 256, 256)
    second = torch.rand_like(first)
    with torch.inference_mode():
        print(f"Output: {tuple(model(first, second).probability.shape)}")
