import torch
from torch import nn

from models.mobilevit_backbone import MobileViTBackbone
from models.fpn import FPN
from models.detection_head import DetectionHead


class MobileViTDetector(nn.Module):
    def __init__(self, num_classes=20, backbone_mode="x_small"):
        super().__init__()
        self.backbone = MobileViTBackbone(mode=backbone_mode)
        self.fpn = FPN(in_channels=(64, 80, 96), out_channels=96)
        self.head = DetectionHead(channels=96, num_classes=num_classes)

    def forward(self, images):
        features = self.backbone(images)
        pyramid = self.fpn(features)
        predictions = self.head(pyramid)
        return predictions

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True
