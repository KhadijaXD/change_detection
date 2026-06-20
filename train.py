"""
Training Script
===============
Full training loop for the Siamese change detection network.
Supports mixed precision, early stopping, and best-checkpoint saving.

Usage:
    python train.py                          # uses configs/default.yaml
    python train.py --config my_config.yaml  # custom config
"""

import argparse
import json
import os
import random
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

# --- LOCAL IMPORTS ---
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data.dataset import LEVIRDataset
from models.siamese import SiameseChangeDetector
from models.losses import CombinedLoss
from utils.metrics import compute_metrics, AverageMeter, MetricTracker
from utils.visualize import plot_training_curves


def load_config(path: str) -> dict:
    """Load YAML config and return as a nested dict."""
    try:
        from omegaconf import OmegaConf

        cfg = OmegaConf.load(path)
        return OmegaConf.to_container(cfg, resolve=True)
    except ImportError:
        import yaml

        with open(path) as f:
            return yaml.safe_load(f)


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def get_device(requested: str) -> torch.device:
    """Resolve device with graceful CPU fallback."""
    if requested == "cuda" and not torch.cuda.is_available():
        warnings.warn("CUDA requested but not available — falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested)


# --- TRAINING ONE EPOCH ---
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    log_interval: int,
    use_amp: bool,
) -> float:
    """Train for one epoch.

    Returns:
        Average training loss for the epoch.
    """
    model.train()
    loss_meter = AverageMeter()

    pbar = tqdm(loader, desc="  Train", leave=False)
    for i, batch in enumerate(pbar):
        img_A = batch["A"].to(device)
        img_B = batch["B"].to(device)
        mask = batch["mask"].to(device)

        optimizer.zero_grad()

        with autocast(enabled=use_amp):
            logits = model(img_A, img_B)
            loss = criterion(logits, mask)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loss_meter.update(loss.item(), img_A.size(0))

        if (i + 1) % log_interval == 0:
            pbar.set_postfix(loss=f"{loss_meter.avg:.4f}")

    return loss_meter.avg


# --- VALIDATION ONE EPOCH ---
@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
) -> dict:
    """Validate and compute metrics.

    Returns:
        Dict with 'loss', 'iou', 'f1', 'precision', 'recall'.
    """
    model.eval()
    loss_meter = AverageMeter()
    tracker = MetricTracker()

    for batch in tqdm(loader, desc="  Val  ", leave=False):
        img_A = batch["A"].to(device)
        img_B = batch["B"].to(device)
        mask = batch["mask"].to(device)

        with autocast(enabled=use_amp):
            logits = model(img_A, img_B)
            loss = criterion(logits, mask)

        loss_meter.update(loss.item(), img_A.size(0))
        tracker.update(logits, mask)

    metrics = tracker.compute()
    metrics["loss"] = loss_meter.avg
    return metrics


# --- MAIN ---
def main():
    parser = argparse.ArgumentParser(description="Train Siamese Change Detector")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML config file.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(42)

    device = get_device(cfg["training"]["device"])
    use_amp = device.type == "cuda"
    print(f"[INFO] Using device: {device} | AMP: {use_amp}")

    # --- DATA ---
    train_ds = LEVIRDataset(
        root=cfg["data"]["root"],
        split="train",
        tile_size=cfg["data"]["tile_size"],
        stride=cfg["data"]["stride"],
    )
    val_ds = LEVIRDataset(
        root=cfg["data"]["root"],
        split="val",
        tile_size=cfg["data"]["tile_size"],
        stride=cfg["data"]["stride"],
    )
    print(f"[INFO] Train tiles: {len(train_ds)} | Val tiles: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
    )

    # --- MODEL ---
    model = SiameseChangeDetector(
        backbone=cfg["model"]["backbone"],
        pretrained=cfg["model"]["pretrained"],
        diff_mode=cfg["model"]["diff_mode"],
        decoder_channels=cfg["model"]["decoder_channels"],
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Model parameters: {total_params:,}")

    # --- LOSS, OPTIMIZER, SCHEDULER ---
    criterion = CombinedLoss(bce_weight=0.5, dice_weight=0.5, pos_weight=8.0)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["training"]["epochs"], eta_min=1e-6
    )
    scaler = GradScaler(enabled=use_amp)

    # --- TRAINING LOOP ---
    save_dir = Path(cfg["logging"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    best_f1 = 0.0
    patience_counter = 0
    train_losses, val_losses, val_f1s = [], [], []

    for epoch in range(1, cfg["training"]["epochs"] + 1):
        print(f"\nEpoch {epoch}/{cfg['training']['epochs']}")

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler,
            device, cfg["logging"]["log_interval"], use_amp,
        )

        val_metrics = validate(model, val_loader, criterion, device, use_amp)
        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_metrics["loss"])
        val_f1s.append(val_metrics["f1"])

        print(
            f"  Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val F1: {val_metrics['f1']:.4f} | "
            f"Val IoU: {val_metrics['iou']:.4f} | "
            f"Val Prec: {val_metrics['precision']:.4f} | "
            f"Val Rec: {val_metrics['recall']:.4f} | "
            f"LR: {scheduler.get_last_lr()[0]:.2e}"
        )

        # --- CHECKPOINTING ---
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            patience_counter = 0
            ckpt_path = save_dir / "best.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_f1": best_f1,
                    "config": cfg,
                },
                ckpt_path,
            )
            print(f"  ✓ Saved best checkpoint (F1={best_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= cfg["training"]["patience"]:
                print(f"  Early stopping at epoch {epoch} (no improvement for {cfg['training']['patience']} epochs)")
                break

    # --- SAVE TRAINING CURVES ---
    curves_path = save_dir / "training_curves.png"
    plot_training_curves(train_losses, val_losses, val_f1s, str(curves_path))
    print(f"\n[INFO] Training curves saved to {curves_path}")
    print(f"[INFO] Best val F1: {best_f1:.4f}")


if __name__ == "__main__":
    main()
