from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class DeblurTorchModelConfig:
    image_size: tuple[int, int] = (256, 256)
    weights: Optional[Literal["imagenet"]] = "imagenet"
    backbone_trainable: bool = False


class _UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch + skip_ch, out_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        return x


class DeblurMobileNetV2UNet(nn.Module):
    """MobileNetV2 encoder + lightweight U-Net-ish decoder.

    Input/Output:
      - input:  (N, 3, H, W) float32 in [0,1]
      - output: (N, 3, H, W) float32 in [0,1]

    Notes:
      - Uses MobileNetV2 features from torchvision.
      - Uses a simple [-1,1] scaling (like TF MobileNetV2 preprocess) instead of mean/std.
    """

    def __init__(self, weights: str | None = "imagenet", backbone_trainable: bool = False):
        super().__init__()
        try:
            from torchvision.models import mobilenet_v2
            from torchvision.models import MobileNet_V2_Weights
        except Exception as e:  # pragma: no cover
            raise RuntimeError("torchvision is required for DeblurMobileNetV2UNet") from e

        if weights is None or str(weights).lower() in ("none", "null"):
            tv_weights = None
        else:
            tv_weights = MobileNet_V2_Weights.DEFAULT

        backbone = mobilenet_v2(weights=tv_weights)
        self.encoder = backbone.features

        for p in self.encoder.parameters():
            p.requires_grad = bool(backbone_trainable)

        # Skip taps chosen to roughly match TF version stages.
        # Indices correspond to torchvision MobileNetV2 feature blocks.
        self._skip_idxs = [1, 3, 6, 13]  # progressively lower resolution
        self._deep_idx = 18

        # Decoder channel sizes (match expected encoder output channels)
        # MobileNetV2 deep output channels is 1280.
        self.up4 = _UpBlock(in_ch=1280, skip_ch=96, out_ch=256)
        self.up3 = _UpBlock(in_ch=256, skip_ch=32, out_ch=192)
        self.up2 = _UpBlock(in_ch=192, skip_ch=24, out_ch=128)
        self.up1 = _UpBlock(in_ch=128, skip_ch=16, out_ch=96)

        self.dec0_conv1 = nn.Conv2d(96, 64, kernel_size=3, padding=1)
        self.dec0_conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.out_conv = nn.Conv2d(64, 3, kernel_size=1)
        self.act = nn.ReLU(inplace=True)

    @staticmethod
    def _preprocess(x: torch.Tensor) -> torch.Tensor:
        # x in [0,1] -> [-1,1]
        return x * 2.0 - 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._preprocess(x)

        skips: list[torch.Tensor] = []
        out = x
        for i, layer in enumerate(self.encoder):
            out = layer(out)
            if i in self._skip_idxs:
                skips.append(out)
            if i == self._deep_idx:
                break

        # The collected skips are low->high level; we want deepest skip last.
        # We use: skip13(96ch@16x16), skip6(32ch@32x32), skip3(24ch@64x64), skip1(16ch@128x128)
        # skips order by idxs: [1,3,6,13]
        skip1, skip3, skip6, skip13 = skips

        out = self.up4(out, skip13)
        out = self.up3(out, skip6)
        out = self.up2(out, skip3)
        out = self.up1(out, skip1)

        out = F.interpolate(out, scale_factor=2.0, mode="bilinear", align_corners=False)
        out = self.act(self.dec0_conv1(out))
        out = self.act(self.dec0_conv2(out))

        out = torch.sigmoid(self.out_conv(out))
        return out


def build_deblur_mobilenetv2_torch(
    weights: str | None = "imagenet",
    backbone_trainable: bool = False,
) -> nn.Module:
    return DeblurMobileNetV2UNet(weights=weights, backbone_trainable=backbone_trainable)
