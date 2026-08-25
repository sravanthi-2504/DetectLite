import torch
from models.detector import MobileViTDetector


def main():
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = MobileViTDetector(num_classes=20).to(device)
    model.eval()

    x = torch.randn(1, 3, 224, 224, device=device)

    with torch.no_grad():
        y = model(x)

    print("Device:", device)
    for level, out in y.items():
        print(
            level,
            "cls=", tuple(out["cls"].shape),
            "box=", tuple(out["box"].shape),
            "center=", tuple(out["center"].shape),
        )


if __name__ == "__main__":
    main()
