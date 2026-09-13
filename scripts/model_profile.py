import argparse
import json
import torch

from models.detector import MobileViTDetector
from evaluation.profiler import profile_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint")
    args = p.parse_args()

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = MobileViTDetector(num_classes=20).to(device)
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model"])

    result = profile_model(model, device=str(device))
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
