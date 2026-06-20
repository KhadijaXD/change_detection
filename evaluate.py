"""
Evaluation Script
=================
Load a trained checkpoint, evaluate on the test split, print metrics,
save a JSON report, and generate qualitative sample visualizations.

Usage:
    python evaluate.py --config configs/default.yaml --checkpoint checkpoints/best.pth
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data.dataset import LEVIRDataset
from models.siamese import SiameseChangeDetector
from utils.metrics import compute_metrics, MetricTracker
from utils.visualize import plot_sample


def load_config(path: str) -> dict:
    try:
        from omegaconf import OmegaConf
        return OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    except ImportError:
        import yaml
        with open(path) as f:
            return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Siamese Change Detector")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pth")
    parser.add_argument("--output_dir", type=str, default="eval_results")
    parser.add_argument("--num_samples", type=int, default=8, help="Number of qualitative samples to save.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- LOAD MODEL ---
    print(f"[INFO] Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    ckpt_cfg = ckpt.get("config", cfg)

    model = SiameseChangeDetector(
        backbone=ckpt_cfg["model"]["backbone"],
        pretrained=False,  # weights come from checkpoint
        diff_mode=ckpt_cfg["model"]["diff_mode"],
        decoder_channels=ckpt_cfg["model"]["decoder_channels"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"[INFO] Model loaded (trained epoch {ckpt.get('epoch', '?')}, best F1 {ckpt.get('best_f1', '?'):.4f})")

    # --- TEST DATA ---
    test_ds = LEVIRDataset(
        root=cfg["data"]["root"],
        split="test",
        tile_size=cfg["data"]["tile_size"],
        stride=cfg["data"]["stride"],
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
    )
    print(f"[INFO] Test tiles: {len(test_ds)}")

    # --- EVALUATE ---
    tracker = MetricTracker()
    sample_indices = sorted(random.sample(range(len(test_ds)), min(args.num_samples, len(test_ds))))
    collected_samples = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            img_A = batch["A"].to(device)
            img_B = batch["B"].to(device)
            mask = batch["mask"].to(device)

            with autocast(enabled=use_amp):
                logits = model(img_A, img_B)

            tracker.update(logits, mask)

    all_metrics = tracker.compute()

    # --- PRINT RESULTS ---
    print("\n" + "=" * 50)
    print("TEST RESULTS (change class)")
    print("=" * 50)
    for k, v in all_metrics.items():
        print(f"  {k.upper():>10s}: {v:.4f}")
    print("=" * 50)

    # --- SAVE METRICS JSON ---
    metrics_path = output_dir / "test_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n[INFO] Metrics saved to {metrics_path}")

    # --- QUALITATIVE SAMPLES ---
    print(f"[INFO] Generating {len(sample_indices)} qualitative samples...")
    for i, idx in enumerate(sample_indices):
        sample = test_ds[idx]
        img_A = sample["A"].unsqueeze(0).to(device)
        img_B = sample["B"].unsqueeze(0).to(device)
        mask = sample["mask"]

        with torch.no_grad(), autocast(enabled=use_amp):
            logits = model(img_A, img_B)

        pred = (torch.sigmoid(logits) > 0.5).float().cpu().squeeze(0)

        save_path = output_dir / f"sample_{i:02d}.png"
        plot_sample(
            sample["A"], sample["B"], mask, pred, str(save_path)
        )

    print(f"[INFO] Samples saved to {output_dir}/")


if __name__ == "__main__":
    main()
