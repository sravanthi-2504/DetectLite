import math

import torch
from torch import nn
import torch.nn.functional as F


class DetectionHead(nn.Module):
    """
    Anchor-free dense detection head.

    Each FPN location predicts:
      - class logits
      - box distances: left, top, right, bottom
      - centerness

    V4 change:
      - Box regression is initialized to approximately
        2 feature-map cells instead of 4.
      - Softplus keeps predicted box distances positive.
    """

    def __init__(
        self,
        channels=96,
        num_classes=20,
        hidden=96,
    ):
        super().__init__()

        self.num_classes = num_classes

        def tower():
            return nn.Sequential(
                nn.Conv2d(
                    channels,
                    hidden,
                    3,
                    padding=1,
                ),
                nn.GroupNorm(
                    8,
                    hidden,
                ),
                nn.ReLU(
                    inplace=True,
                ),
                nn.Conv2d(
                    hidden,
                    hidden,
                    3,
                    padding=1,
                ),
                nn.GroupNorm(
                    8,
                    hidden,
                ),
                nn.ReLU(
                    inplace=True,
                ),
            )

        # Separate classification and box towers.
        self.cls_tower = nn.ModuleList(
            [
                tower()
                for _ in range(3)
            ]
        )

        self.box_tower = nn.ModuleList(
            [
                tower()
                for _ in range(3)
            ]
        )

        # Classification prediction.
        self.cls_pred = nn.ModuleList(
            [
                nn.Conv2d(
                    hidden,
                    num_classes,
                    1,
                )
                for _ in range(3)
            ]
        )

        # Box distance prediction.
        self.box_pred = nn.ModuleList(
            [
                nn.Conv2d(
                    hidden,
                    4,
                    1,
                )
                for _ in range(3)
            ]
        )

        # Centerness prediction.
        self.ctr_pred = nn.ModuleList(
            [
                nn.Conv2d(
                    hidden,
                    1,
                    1,
                )
                for _ in range(3)
            ]
        )

        # ---------------------------------------------------------
        # V4 initialization
        # ---------------------------------------------------------
        #
        # Softplus(x) ~= 2 when:
        #
        # x = log(exp(2) - 1)
        #
        # Therefore the initial predicted box distance is
        # approximately 2 feature-map cells.
        #
        box_bias = math.log(
            math.exp(2.0) - 1.0
        )

        for layer in self.box_pred:
            nn.init.constant_(
                layer.bias,
                box_bias,
            )

    def forward(self, pyramid):
        outputs = {}

        for i, level in enumerate(
            ("p3", "p4", "p5")
        ):
            x = pyramid[level]

            # Classification branch.
            c = self.cls_tower[i](x)

            # Box + centerness branch.
            b = self.box_tower[i](x)

            # Positive box distances.
            box = F.softplus(
                self.box_pred[i](b)
            )

            outputs[level] = {
                "cls": self.cls_pred[i](c),

                "box": box,

                "center": self.ctr_pred[i](b),
            }

        return outputs
