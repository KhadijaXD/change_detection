"""
Metrics
=======
Evaluation metrics for binary change detection, computed only for
the foreground (change) class. Includes a MetricTracker for correct
dataset-wide accumulation.
"""

from typing import Dict

import torch
import numpy as np


class MetricTracker:
    """Accumulates true positives, false positives, and false negatives
    across multiple batches to compute mathematically correct dataset-wide
    metrics (Precision, Recall, F1, and IoU), avoiding batch-averaging bias.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.tp = 0.0
        self.fp = 0.0
        self.fn = 0.0

    def update(self, preds: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5):
        """Update confusion matrix elements from a batch of predictions and targets."""
        with torch.no_grad():
            probs = torch.sigmoid(preds)
            pred_bin = (probs > threshold).float().reshape(-1)
            target_bin = targets.float().reshape(-1)

            self.tp += (pred_bin * target_bin).sum().item()
            self.fp += (pred_bin * (1 - target_bin)).sum().item()
            self.fn += ((1 - pred_bin) * target_bin).sum().item()

    def compute(self) -> Dict[str, float]:
        """Compute dataset-wide metrics based on accumulated confusion matrix elements."""
        precision = self.tp / (self.tp + self.fp + 1e-8)
        recall = self.tp / (self.tp + self.fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        iou = self.tp / (self.tp + self.fp + self.fn + 1e-8)

        return {
            "iou": iou,
            "f1": f1,
            "precision": precision,
            "recall": recall,
        }


def compute_metrics(
    preds: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5
) -> Dict[str, float]:
    """Compute change-class metrics from raw logits and binary targets for a single batch.

    WARNING: Averaging these values across batches is mathematically incorrect and leads to
    underestimating the F1/IoU scores (especially on batches with no change pixels).
    Use MetricTracker for evaluating over multiple batches.
    """
    with torch.no_grad():
        probs = torch.sigmoid(preds)
        pred_bin = (probs > threshold).float().reshape(-1)
        target_bin = targets.float().reshape(-1)

        tp = (pred_bin * target_bin).sum().item()
        fp = (pred_bin * (1 - target_bin)).sum().item()
        fn = ((1 - pred_bin) * target_bin).sum().item()

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        iou = tp / (tp + fp + fn + 1e-8)

    return {
        "iou": iou,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }


class AverageMeter:
    """Tracks a running average of a scalar value.

    Usage:
        meter = AverageMeter()
        for val in values:
            meter.update(val)
        print(meter.avg)
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        """Update with a new value.

        Args:
            val: New value (or batch-average value).
            n: Batch size (number of samples this value represents).
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
