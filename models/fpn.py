import torch
from torch import nn
import torch.nn.functional as F


class FPN(nn.Module):
    """Simple top-down Feature Pyramid Network for MobileViT-XS features."""
    def __init__(self, in_channels=(64, 80, 96), out_channels=96):
        super().__init__()
        self.lateral3 = nn.Conv2d(in_channels[0], out_channels, 1)
        self.lateral4 = nn.Conv2d(in_channels[1], out_channels, 1)
        self.lateral5 = nn.Conv2d(in_channels[2], out_channels, 1)

        self.smooth3 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.smooth4 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.smooth5 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

    def forward(self, features):
        c3, c4, c5 = features["f3"], features["f4"], features["f5"]

        p5 = self.lateral5(c5)
        p4 = self.lateral4(c4) + F.interpolate(
            p5, size=c4.shape[-2:], mode="nearest"
        )
        p3 = self.lateral3(c3) + F.interpolate(
            p4, size=c3.shape[-2:], mode="nearest"
        )

        return {
            "p3": self.smooth3(p3),
            "p4": self.smooth4(p4),
            "p5": self.smooth5(p5),
        }
