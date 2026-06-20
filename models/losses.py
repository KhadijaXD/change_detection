"""
Loss Functions
==============
Combined BCE + Dice loss for binary change detection, with support
for class-imbalance weighting via pos_weight.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Soft Dice loss computed on sigmoid-activated predictions.

    Args:
        smooth: Smoothing constant to avoid division by zero.
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Raw model output (B, 1, H, W).
            targets: Binary ground truth (B, 1, H, W).

        Returns:
            Scalar Dice loss.
        """
        probs = torch.sigmoid(logits)
        probs_flat = probs.reshape(-1)
        targets_flat = targets.reshape(-1)

        intersection = (probs_flat * targets_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (
            probs_flat.sum() + targets_flat.sum() + self.smooth
        )
        return 1.0 - dice


class CombinedLoss(nn.Module):
    """Weighted combination of BCE and Dice losses.

    Args:
        bce_weight: Weight for the BCE component.
        dice_weight: Weight for the Dice component.
        pos_weight: Positive class weight for BCE to handle imbalance.
                    If None, no weighting is applied.
    """

    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        pos_weight: Optional[float] = None,
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

        pw = torch.tensor([pos_weight]) if pos_weight is not None else None
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pw)
        self.dice = DiceLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Raw model output (B, 1, H, W).
            targets: Binary ground truth (B, 1, H, W).

        Returns:
            Scalar combined loss.
        """
        # Move pos_weight to same device if needed
        if self.bce.pos_weight is not None:
            self.bce.pos_weight = self.bce.pos_weight.to(logits.device)

        loss_bce = self.bce(logits, targets)
        loss_dice = self.dice(logits, targets)
        return self.bce_weight * loss_bce + self.dice_weight * loss_dice

