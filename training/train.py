import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import VOCDataset, collate_fn
from models.detector import MobileViTDetector
from models.losses import detection_loss


def main():
    p = argparse.ArgumentParser()

    p.add_argument("--data", required=True, help="VOC root directory")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--out", default="checkpoints/model_30epoch.pt")
    p.add_argument("--resume", default=None)

    args = p.parse_args()

    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    ds = VOCDataset(
        args.data,
        split="train",
        image_size=224
    )

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_fn
    )

    model = MobileViTDetector(num_classes=20).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4
    )

    start_epoch = 0

    if args.resume:
        resume_path = Path(args.resume)

        if not resume_path.exists():
            raise FileNotFoundError(
                f"Resume checkpoint not found: {resume_path}"
            )

        print("Loading checkpoint:", resume_path)

        checkpoint = torch.load(
            resume_path,
            map_location="cpu"
        )

        model.load_state_dict(checkpoint["model"])

        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])

        start_epoch = int(checkpoint.get("epoch", 0))

        print("Resumed from epoch:", start_epoch)

    Path(args.out).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    if start_epoch >= args.epochs:
        raise ValueError(
            f"Checkpoint is already at epoch {start_epoch}, "
            f"but --epochs is {args.epochs}."
        )

    for epoch in range(start_epoch, args.epochs):
        model.train()

        running = 0.0

        bar = tqdm(
            loader,
            desc=f"Epoch {epoch + 1}/{args.epochs}"
        )

        for images, targets in bar:
            images = images.to(device)

            optimizer.zero_grad(set_to_none=True)

            outputs = model(images)

            loss = detection_loss(
                outputs,
                targets
            )

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite loss detected at epoch {epoch + 1}: "
                    f"{float(loss.detach())}"
                )

            loss.backward()

            optimizer.step()

            loss_value = float(loss.detach())

            running += loss_value

            bar.set_postfix(
                loss=f"{loss_value:.4f}"
            )

        avg = running / max(len(loader), 1)

        print(
            f"Epoch {epoch + 1}: "
            f"loss={avg:.5f}"
        )

        torch.save(
            {
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            },
            args.out
        )

        print("Checkpoint saved:", args.out)

    print("Training complete.")
    print("Saved:", args.out)


if __name__ == "__main__":
    main()
