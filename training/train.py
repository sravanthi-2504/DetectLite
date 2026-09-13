import argparse
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import VOCDataset, collate_fn
from models.detector import MobileViTDetector
from models.losses import detection_loss
from evaluation.voc_map import compute_map


def build_lr_lambda(total_epochs, warmup_epochs, min_lr_ratio=0.01):
    """
    Linear warmup for `warmup_epochs`, then cosine decay down to
    `min_lr_ratio * base_lr` for the remaining epochs.

    Returned as an epoch-indexed multiplier for LambdaLR (so it's
    called once per epoch, not per step -- simple and enough here
    given how few steps/epoch this dataset has).
    """
    def lr_lambda(epoch):
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return min_lr_ratio + (1 - min_lr_ratio) * cosine
    return lr_lambda


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="VOC root directory")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--out", default="checkpoints/baseline.pt")

    # Validation / model selection
    p.add_argument("--val-split", default="val",
                    help="Split name to validate on each epoch (set to '' to disable)")
    p.add_argument("--val-every", type=int, default=1,
                    help="Run validation every N epochs")
    p.add_argument("--patience", type=int, default=7,
                    help="Stop if val mAP hasn't improved for this many validations. 0 disables.")

    # LR schedule
    p.add_argument("--warmup-epochs", type=int, default=2)
    p.add_argument("--min-lr-ratio", type=float, default=0.01,
                    help="Cosine decay floor, as a fraction of --lr")

    # Backbone
    p.add_argument("--pretrained-backbone", action="store_true", default=True)
    p.add_argument("--no-pretrained-backbone", dest="pretrained_backbone", action="store_false")
    p.add_argument("--drop-rate", type=float, default=0.1)
    p.add_argument("--drop-path-rate", type=float, default=0.1)
    p.add_argument("--freeze-backbone-epochs", type=int, default=0,
                    help="Freeze the pretrained backbone for this many initial epochs "
                         "so the randomly-initialized head/FPN stabilize first")

    args = p.parse_args()

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Device:", device)

    train_ds = VOCDataset(args.data, split="train", image_size=224, augment=True)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, collate_fn=collate_fn
    )

    val_loader = None
    if args.val_split:
        val_ds = VOCDataset(args.data, split=args.val_split, image_size=224, augment=False)
        val_loader = DataLoader(
            val_ds, batch_size=1, shuffle=False,
            num_workers=args.workers, collate_fn=collate_fn
        )

    model = MobileViTDetector(
        num_classes=20,
        pretrained_backbone=args.pretrained_backbone,
        drop_rate=args.drop_rate,
        drop_path_rate=args.drop_path_rate,
    ).to(device)

    if args.freeze_backbone_epochs > 0:
        model.freeze_backbone()
        print(f"Backbone frozen for the first {args.freeze_backbone_epochs} epoch(s).")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, build_lr_lambda(args.epochs, args.warmup_epochs, args.min_lr_ratio)
    )

    out_path = Path(args.out)
    best_path = out_path.with_name(out_path.stem + "_best" + out_path.suffix)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    best_map = -1.0
    epochs_since_improve = 0

    for epoch in range(args.epochs):
        # Unfreeze the backbone once the warm-up window is over.
        if args.freeze_backbone_epochs > 0 and epoch == args.freeze_backbone_epochs:
            model.unfreeze_backbone()
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=optimizer.param_groups[0]["lr"], weight_decay=1e-4
            )
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                build_lr_lambda(args.epochs - epoch, 0, args.min_lr_ratio),
            )
            print(f"Backbone unfrozen at epoch {epoch + 1}.")

        model.train()
        running = 0.0

        bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for images, targets in bar:
            images = images.to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = detection_loss(outputs, targets)
            loss.backward()
            optimizer.step()

            running += float(loss.detach())
            bar.set_postfix(
                loss=f"{float(loss.detach()):.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )

        scheduler.step()
        avg_train_loss = running / max(len(train_loader), 1)
        log_line = f"Epoch {epoch + 1}: train_loss={avg_train_loss:.5f}"

        run_val = val_loader is not None and (epoch + 1) % args.val_every == 0
        val_map = None
        if run_val:
            val_map, _ = compute_map(model, val_loader, device, iou_threshold=0.5)
            log_line += f"  val_mAP@0.5={val_map:.4f}"

        print(log_line)

        # Always keep a "last" checkpoint so you can resume/inspect.
        torch.save({
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "val_map": val_map,
        }, out_path)

        if run_val:
            if val_map > best_map:
                best_map = val_map
                epochs_since_improve = 0
                torch.save({
                    "epoch": epoch + 1,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "val_map": val_map,
                }, best_path)
                print(f"  -> new best (val_mAP@0.5={best_map:.4f}), saved to {best_path}")
            else:
                epochs_since_improve += 1
                if args.patience > 0 and epochs_since_improve >= args.patience:
                    print(
                        f"No val mAP improvement for {epochs_since_improve} validations "
                        f"(patience={args.patience}). Stopping early at epoch {epoch + 1}."
                    )
                    break

    if val_loader is not None:
        print(f"Done. Best val mAP@0.5={best_map:.4f} -> {best_path}")
    else:
        print(f"Done. Last checkpoint -> {out_path} (no --val-split given, so no best checkpoint was tracked)")


if __name__ == "__main__":
    main()
