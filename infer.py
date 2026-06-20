"""
Inference Script
================
Run change detection on a single image pair. Automatically tiles large images,
runs inference, and stitches the output mask back together.

Usage:
    python infer.py --img_a before.png --img_b after.png --checkpoint best.pth --output mask.png
"""

import argparse
import os
import sys
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from PIL import Image
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data.dataset import IMAGENET_MEAN, IMAGENET_STD
from models.siamese import SiameseChangeDetector


def load_config_from_checkpoint(ckpt: dict) -> dict:
    """Extract config from checkpoint, with sensible defaults."""
    return ckpt.get("config", {
        "model": {
            "backbone": "resnet34",
            "diff_mode": "subtract",
            "decoder_channels": [256, 128, 64, 32],
        }
    })


def normalize_image(img: np.ndarray) -> np.ndarray:
    """Normalize a (H, W, 3) uint8 image to float32 with ImageNet stats."""
    img = img.astype(np.float32) / 255.0
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std = np.array(IMAGENET_STD, dtype=np.float32)
    img = (img - mean) / std
    return img.astype(np.float32)


def tile_and_infer(
    img_A: np.ndarray,
    img_B: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
    tile_size: int = 256,
    stride: int = 256,
    use_amp: bool = False,
) -> np.ndarray:
    """Tile large images, run inference on each tile, stitch mask.

    Args:
        img_A: Before image (H, W, 3) uint8.
        img_B: After image (H, W, 3) uint8.
        model: Loaded SiameseChangeDetector.
        device: Torch device.
        tile_size: Size of square tiles.
        stride: Stride between tiles.
        use_amp: Whether to use mixed precision.

    Returns:
        Binary mask (H, W) as float32 (0.0 or 1.0).
    """
    h, w = img_A.shape[:2]

    # Pad to be divisible by tile_size
    pad_h = (tile_size - h % tile_size) % tile_size
    pad_w = (tile_size - w % tile_size) % tile_size
    if pad_h > 0 or pad_w > 0:
        img_A = np.pad(img_A, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
        img_B = np.pad(img_B, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")

    h_pad, w_pad = img_A.shape[:2]

    # Normalize
    norm_A = normalize_image(img_A)
    norm_B = normalize_image(img_B)

    # Accumulator for averaging overlapping predictions
    prob_map = np.zeros((h_pad, w_pad), dtype=np.float32)
    count_map = np.zeros((h_pad, w_pad), dtype=np.float32)

    model.eval()
    with torch.no_grad():
        for r in range(0, h_pad - tile_size + 1, stride):
            for c in range(0, w_pad - tile_size + 1, stride):
                tile_A = norm_A[r : r + tile_size, c : c + tile_size]
                tile_B = norm_B[r : r + tile_size, c : c + tile_size]

                # (H,W,C) → (1,C,H,W)
                t_A = torch.from_numpy(tile_A.transpose(2, 0, 1)).unsqueeze(0).to(device)
                t_B = torch.from_numpy(tile_B.transpose(2, 0, 1)).unsqueeze(0).to(device)

                with autocast(enabled=use_amp):
                    logits = model(t_A, t_B)

                probs = torch.sigmoid(logits).squeeze().cpu().numpy()
                prob_map[r : r + tile_size, c : c + tile_size] += probs
                count_map[r : r + tile_size, c : c + tile_size] += 1.0

    # Average overlapping regions
    prob_map /= np.maximum(count_map, 1.0)

    # Crop back to original size
    prob_map = prob_map[:h, :w]

    return (prob_map > 0.5).astype(np.float32)


def save_visualization(
    img_A: np.ndarray, img_B: np.ndarray, mask: np.ndarray, save_path: str
) -> None:
    """Save a 3-panel visualization: A | B | Predicted Mask."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(img_A)
    axes[0].set_title("Before (A)")
    axes[0].axis("off")

    axes[1].imshow(img_B)
    axes[1].set_title("After (B)")
    axes[1].axis("off")

    axes[2].imshow(mask, cmap="hot", vmin=0, vmax=1)
    axes[2].set_title("Predicted Change")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Visualization saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Run change detection inference")
    parser.add_argument("--img_a", type=str, required=True, help="Path to before image.")
    parser.add_argument("--img_b", type=str, required=True, help="Path to after image.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint.")
    parser.add_argument("--output", type=str, default="mask.png", help="Output mask path.")
    parser.add_argument("--tile_size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=256)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"[INFO] Device: {device}")

    # --- LOAD MODEL ---
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = load_config_from_checkpoint(ckpt)

    model = SiameseChangeDetector(
        backbone=cfg["model"]["backbone"],
        pretrained=False,
        diff_mode=cfg["model"]["diff_mode"],
        decoder_channels=cfg["model"]["decoder_channels"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    print("[INFO] Model loaded.")

    # --- LOAD IMAGES ---
    img_A = np.array(Image.open(args.img_a).convert("RGB"))
    img_B = np.array(Image.open(args.img_b).convert("RGB"))
    assert img_A.shape == img_B.shape, (
        f"Image shapes must match: A={img_A.shape}, B={img_B.shape}"
    )
    print(f"[INFO] Image size: {img_A.shape[1]}×{img_A.shape[0]}")

    # --- INFERENCE ---
    mask = tile_and_infer(img_A, img_B, model, device, args.tile_size, args.stride, use_amp)

    # --- SAVE MASK ---
    mask_uint8 = (mask * 255).astype(np.uint8)
    Image.fromarray(mask_uint8).save(args.output)
    print(f"[INFO] Binary mask saved to {args.output}")

    # --- SAVE VISUALIZATION ---
    vis_path = args.output.replace(".png", "_vis.png").replace(".jpg", "_vis.jpg")
    if vis_path == args.output:
        vis_path = args.output + "_vis.png"
    save_visualization(img_A, img_B, mask, vis_path)


if __name__ == "__main__":
    main()
