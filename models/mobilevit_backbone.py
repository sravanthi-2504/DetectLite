import argparse
import torch

from cvnets.models.classification.mobilevit import MobileViT


class MobileViTOptions(argparse.Namespace):
    """
    Minimal argparse.Namespace containing the options required
    by Apple's CVNets MobileViT implementation.
    """

    def __init__(self, mode="x_small"):
        super().__init__()

        values = {
            # MobileViT
            "model.classification.mit.mode": mode,
            "model.classification.mit.attn_dropout": 0.0,
            "model.classification.mit.ffn_dropout": 0.0,
            "model.classification.mit.dropout": 0.0,
            "model.classification.mit.transformer_norm_layer": "layer_norm",
            "model.classification.mit.no_fuse_local_global_features": False,
            "model.classification.mit.conv_kernel_size": 3,
            "model.classification.mit.head_dim": None,
            "model.classification.mit.number_heads": 4,

            # Classification
            "model.classification.n_classes": 1000,
            "model.classification.classifier_dropout": 0.0,
            "model.classification.gradient_checkpointing": False,
            "model.classification.enable_layer_wise_lr_decay": False,
            "model.classification.layer_wise_lr_decay_rate": 1.0,

            # Backbone
            "model.layer.global_pool": "mean",

            # Normalization
            "model.normalization.name": "batch_norm",
            "model.normalization.groups": 1,
            "model.normalization.momentum": 0.1,

            # Activation
            "model.activation.name": "swish",
            "model.activation.inplace": False,
            "model.activation.neg_slope": 0.1,

            # Initialization
            "model.layer.linear_init": "normal",

            # Neural augmentor
            "neural_augmentor.enable": False,

            # Misc
            "model.resume_exclude_scopes": "",
            "model.ignore_missing_scopes": "",
            "model.rename_scopes_map": None,
            "model.freeze_modules": "",
        }

        for key, value in values.items():
            setattr(self, key, value)


class MobileViTBackbone(torch.nn.Module):

    def __init__(self, mode="x_small"):
        super().__init__()

        opts = MobileViTOptions(mode=mode)

        self.model = MobileViT(opts)

        # Remove classification head.
        self.model.classifier = torch.nn.Identity()

    def forward(self, x):

        x = self.model.conv_1(x)

        x = self.model.layer_1(x)
        f1 = x

        x = self.model.layer_2(x)
        f2 = x

        x = self.model.layer_3(x)
        f3 = x

        x = self.model.layer_4(x)
        f4 = x

        x = self.model.layer_5(x)
        f5 = x

        return {
            "f1": f1,
            "f2": f2,
            "f3": f3,
            "f4": f4,
            "f5": f5,
        }


if __name__ == "__main__":

    device = (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )

    print("Creating MobileViT-XS...")
    print("Device:", device)

    model = MobileViTBackbone(mode="x_small").to(device)
    model.eval()

    x = torch.randn(
        1,
        3,
        224,
        224,
        device=device
    )

    print("Running forward pass...")

    with torch.no_grad():
        features = model(x)

    print()
    print("MobileViT Backbone Test")
    print("-----------------------")

    for name, feature in features.items():
        print(
            f"{name}: "
            f"shape={tuple(feature.shape)}"
        )
