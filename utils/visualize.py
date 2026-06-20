"""
Visualization Utilities
=======================
Side-by-side plotting of image pairs, masks, and training curves.
"""

from typing import List, Optional
from pathlib import Path

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt


# --- IMAGENET DENORMALIZATION ---
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def _denormalize(tensor: torch.Tensor) -> np.ndarray:
    """Convert a normalized (C, H, W) tensor back to a displayable (H, W, 3) uint8 array."""
    img = tensor.cpu().numpy().transpose(1, 2, 0)  # (H, W, 3)
    img = img * IMAGENET_STD + IMAGENET_MEAN
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img


def plot_sample(
    img_A: torch.Tensor,
    img_B: torch.Tensor,
    gt_mask: Optional[torch.Tensor],
    pred_mask: Optional[torch.Tensor],
    save_path: str,
) -> None:
    """Plot a 4-panel figure: Before | After | Ground Truth | Prediction.

    Args:
        img_A: Normalized image tensor (3, H, W).
        img_B: Normalized image tensor (3, H, W).
        gt_mask: Ground truth mask (1, H, W) or (H, W), or None.
        pred_mask: Predicted mask (1, H, W) or (H, W), or None.
        save_path: File path to save the figure.
    """
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].imshow(_denormalize(img_A))
    axes[0].set_title("Before (A)")
    axes[0].axis("off")

    axes[1].imshow(_denormalize(img_B))
    axes[1].set_title("After (B)")
    axes[1].axis("off")

    if gt_mask is not None:
        gt = gt_mask.squeeze().cpu().numpy()
    else:
        gt = np.zeros_like(_denormalize(img_A)[:, :, 0])
    axes[2].imshow(gt, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("Ground Truth")
    axes[2].axis("off")

    if pred_mask is not None:
        pr = pred_mask.squeeze().cpu().numpy()
    else:
        pr = np.zeros_like(gt)
    axes[3].imshow(pr, cmap="gray", vmin=0, vmax=1)
    axes[3].set_title("Prediction")
    axes[3].axis("off")

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    val_f1s: List[float],
    save_path: str,
) -> None:
    """Plot training/validation loss and validation F1 on a dual-axis figure.

    Args:
        train_losses: Per-epoch training loss.
        val_losses: Per-epoch validation loss.
        val_f1s: Per-epoch validation F1 score.
        save_path: File path to save the figure.
    """
    epochs = range(1, len(train_losses) + 1)

    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Loss axis
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color="tab:blue")
    ax1.plot(epochs, train_losses, "b-", label="Train Loss")
    ax1.plot(epochs, val_losses, "b--", label="Val Loss")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    # F1 axis
    ax2 = ax1.twinx()
    ax2.set_ylabel("F1 Score", color="tab:red")
    ax2.plot(epochs, val_f1s, "r-o", markersize=3, label="Val F1")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    ax2.set_ylim(0, 1)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")

    plt.title("Training Progress")
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
