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
    def __init__(
        self,
        mode="x_small",
        pretrained_path="checkpoints/pretrained/mobilevit_xs.pt",
    ):
        super().__init__()

        opts = MobileViTOptions(mode=mode)
        self.model = MobileViT(opts)

        if pretrained_path is not None:
            checkpoint = torch.load(
                pretrained_path,
                map_location="cpu",
            )

            model_state = self.model.state_dict()

            compatible = {
                key: value
                for key, value in checkpoint.items()
                if key in model_state
                and model_state[key].shape == value.shape
            }

            missing = [
                key
                for key in model_state
                if key not in compatible
                and not key.startswith("classifier.")
            ]

            if missing:
                raise RuntimeError(
                    "Missing pretrained MobileViT parameters: "
                    + str(missing)
                )

            self.model.load_state_dict(
                compatible,
                strict=False,
            )

            print(
                f"Loaded pretrained MobileViT weights: "
                f"{len(compatible)} compatible parameters"
            )

        # Remove the ImageNet classification head.
        self.model.classifier = torch.nn.Identity()

    def forward(self, x):
        features = self.model.extract_end_points_all(
            x,
            use_l5=True,
            use_l5_exp=False,
        )

        return {
            "f3": features["out_l3"],
            "f4": features["out_l4"],
            "f5": features["out_l5"],
        }
