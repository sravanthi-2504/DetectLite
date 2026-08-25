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
      - center/objectness logit
    """

    def __init__(self, channels=96, num_classes=20, hidden=96):
        super().__init__()

        self.num_classes = num_classes

        def tower():
            return nn.Sequential(
                nn.Conv2d(channels, hidden, 3, padding=1),
                nn.GroupNorm(8, hidden),
                nn.ReLU(inplace=True),

                nn.Conv2d(hidden, hidden, 3, padding=1),
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
        Initialize the dense detection heads with a low foreground
        prior. This prevents the classifier from starting with
        extremely large foreground probabilities everywhere.
        """

        prior_prob = 0.01
        bias_value = -math.log(
            (1.0 - prior_prob) / prior_prob
        )

        for cls_layer in self.cls_pred:
            nn.init.normal_(
                cls_layer.weight,
                mean=0.0,
                std=0.01,
            )
            nn.init.constant_(
                cls_layer.bias,
                bias_value,
            )

        for ctr_layer in self.ctr_pred:
            nn.init.normal_(
                ctr_layer.weight,
                mean=0.0,
                std=0.01,
            )
            nn.init.constant_(
                ctr_layer.bias,
                bias_value,
            )

        for box_layer in self.box_pred:
            nn.init.normal_(
                box_layer.weight,
                mean=0.0,
                std=0.01,
            )
            nn.init.constant_(
                box_layer.bias,
                1.0,
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
