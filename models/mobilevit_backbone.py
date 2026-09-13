import torch
import torch.nn as nn

try:
    import timm
except ImportError as e:
    raise ImportError(
        "timm is required for the pretrained MobileViT backbone. "
        "Install it with: pip install timm"
    ) from e


class MobileViTBackbone(nn.Module):
    """
    MobileViT feature backbone.

    Backed by timm's `*.cvnets_in1k` weights, which are the same
    ImageNet-1k checkpoints Apple released with the original MobileViT
    paper/ml-cvnets repo, repackaged for easy loading. Using them
    (pretrained=True) replaces random-init transformer features with
    features that already understand natural images -- this is one of
    the highest-leverage changes for training a ViT-based detector on
    a small dataset like Pascal VOC.

    drop_rate / drop_path_rate add regularization inside the backbone
    (previously hardcoded to 0.0), which matters once you're not
    massively underfitting anymore.

    Output feature maps (for `mode="x_small"`, 224x224 input):
        f1: (B, 32, 112, 112)  stride 2
        f2: (B, 48, 56, 56)    stride 4
        f3: (B, 64, 28, 28)    stride 8
        f4: (B, 80, 14, 14)    stride 16
        f5: (B, 384, 7, 7)     stride 32

    Note: f5 has 384 channels here (timm keeps MobileViT's final
    1x1 expansion conv), vs. 96 in a from-scratch build that drops
    it. `models/detector.py` FPN is configured to match
    (in_channels=(64, 80, 384)).
    """

    _TIMM_NAMES = {
        "xx_small": "mobilevit_xxs",
        "x_small": "mobilevit_xs",
        "small": "mobilevit_s",
    }

    def __init__(self, mode="x_small", pretrained=True, drop_rate=0.1, drop_path_rate=0.1):
        super().__init__()
        if mode not in self._TIMM_NAMES:
            raise ValueError(
                f"Unknown mode '{mode}'. Expected one of {list(self._TIMM_NAMES)}"
            )

        self.model = timm.create_model(
            self._TIMM_NAMES[mode],
            pretrained=pretrained,
            features_only=True,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
        )

    def forward(self, x):
        f1, f2, f3, f4, f5 = self.model(x)
        return {"f1": f1, "f2": f2, "f3": f3, "f4": f4, "f5": f5}


if __name__ == "__main__":

    device = (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cuda") if torch.cuda.is_available()
        else torch.device("cpu")
    )

    print("Creating MobileViT-XS (ImageNet-1k pretrained)...")
    print("Device:", device)

    model = MobileViTBackbone(mode="x_small", pretrained=True).to(device)
    model.eval()

    x = torch.randn(1, 3, 224, 224, device=device)

    print("Running forward pass...")
    with torch.no_grad():
        features = model(x)

    print()
    print("MobileViT Backbone Test")
    print("-----------------------")
    for name, feature in features.items():
        print(f"{name}: shape={tuple(feature.shape)}")
