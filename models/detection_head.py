import math

import torch
from torch import nn
import torch.nn.functional as F


class DetectionHead(nn.Module):
    """
    Lightweight anchor-free detection head.

    Each FPN location predicts:
      - class logits
      - normalized l/t/r/b box distances
      - centerness/objectness logit

    Box distances are normalized by the FPN stride.
    """

    def __init__(self, channels=96, num_classes=20, hidden=96):
        super().__init__()

        self.num_classes = num_classes

        def tower():
            return nn.Sequential(
                nn.Conv2d(
                    channels,
                    hidden,
                    kernel_size=3,
                    padding=1,
                ),
                nn.GroupNorm(8, hidden),
                nn.ReLU(inplace=True),

                nn.Conv2d(
                    hidden,
                    hidden,
                    kernel_size=3,
                    padding=1,
                ),
                nn.GroupNorm(8, hidden),
                nn.ReLU(inplace=True),
            )

        self.cls_tower = nn.ModuleList(
            [tower() for _ in range(3)]
        )

        self.box_tower = nn.ModuleList(
            [tower() for _ in range(3)]
        )

        self.cls_pred = nn.ModuleList(
            [
                nn.Conv2d(hidden, num_classes, 1)
                for _ in range(3)
            ]
        )

        self.box_pred = nn.ModuleList(
            [
                nn.Conv2d(hidden, 4, 1)
                for _ in range(3)
            ]
        )

        self.ctr_pred = nn.ModuleList(
            [
                nn.Conv2d(hidden, 1, 1)
                for _ in range(3)
            ]
        )

        self._initialize_heads()

    def _initialize_heads(self):
        """
        Initialize dense prediction heads.

        Classification and centerness start with a low foreground
        probability.

        Box distances start around 4 feature-map cells instead of
        approximately 1 pixel-scale unit.
        """

        prior_prob = 0.01

        prior_bias = -math.log(
            (1.0 - prior_prob) / prior_prob
        )

        for layer in self.cls_pred:
            nn.init.normal_(
                layer.weight,
                mean=0.0,
                std=0.01,
            )
            nn.init.constant_(
                layer.bias,
                prior_bias,
            )

        for layer in self.ctr_pred:
            nn.init.normal_(
                layer.weight,
                mean=0.0,
                std=0.01,
            )
            nn.init.constant_(
                layer.bias,
                prior_bias,
            )

        # softplus(approximately 4) gives an initial normalized
        # distance of approximately four feature-map cells.
        box_bias = math.log(
            math.exp(4.0) - 1.0
        )

        for layer in self.box_pred:
            nn.init.normal_(
                layer.weight,
                mean=0.0,
                std=0.01,
            )
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

            cls_features = self.cls_tower[i](x)
            box_features = self.box_tower[i](x)

            cls_logits = self.cls_pred[i](
                cls_features
            )

            # Positive normalized l/t/r/b distances.
            box_distances = F.softplus(
                self.box_pred[i](box_features)
            )

            center_logits = self.ctr_pred[i](
                box_features
            )

            outputs[level] = {
                "cls": cls_logits,
                "box": box_distances,
                "center": center_logits,
            }

        return outputs
