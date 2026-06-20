# Change Detection Project — In-Depth Review

## TL;DR: Your model is actually getting **F1 = 0.906**, not 0.65

The 0.65 F1 score you were seeing was **a metric calculation bug**, not a model performance problem. Your model, architecture, loss function, and training methodology are all sound.

---

## The Root Cause: Batch-Averaged Metrics

### What was wrong

In both [train.py](file:///c:/Users/khadi/Documents/change_detection%20-%20Copy/train.py) and [evaluate.py](file:///c:/Users/khadi/Documents/change_detection%20-%20Copy/evaluate.py), metrics were computed **per-batch** and then **averaged across batches**:

```python
# OLD (WRONG) — in validate() and evaluate
m = compute_metrics(logits, mask)    # per-batch F1
for k in metrics_accum:
    metrics_accum[k] += m[k]
n_batches += 1

# Then later:
metrics_accum[k] /= max(n_batches, 1)  # simple average of per-batch F1s
```

### Why this is mathematically wrong

Averaging F1 scores across batches is **not the same** as computing F1 over the whole dataset. The problem is severe because:

1. **~25% of batches have ZERO change pixels** — tiles from unchanged regions of the 1024×1024 images produce `TP=0, FP≈0, FN=0`, yielding F1 ≈ 0.0 for those batches
2. These zero-F1 batches **drag down the average dramatically**
3. Batches with very few change pixels get **equal weight** as batches with lots of change pixels

### The fix

Accumulate raw confusion matrix elements (TP, FP, FN) across ALL batches, then compute metrics once at the end:

```diff
-metrics_accum = {"iou": 0, "f1": 0, ...}
-n_batches = 0
-# ... per batch: metrics_accum[k] += m[k]; n_batches += 1
-# ... final: metrics_accum[k] /= n_batches
+tracker = MetricTracker()
+# ... per batch: tracker.update(logits, mask)
+# ... final: metrics = tracker.compute()
```

### Before vs After fix (same checkpoint, same model, zero retraining)

| Metric | Batch-Averaged (Bug) | Dataset-Wide (Correct) |
|--------|---------------------|----------------------|
| **F1** | 0.6638 | **0.9061** |
| **IoU** | 0.5977 | **0.8284** |
| Precision | 0.6731 | 0.8943 |
| Recall | 0.6718 | 0.9182 |

> [!IMPORTANT]
> Your model already **exceeds** the target F1 > 0.78 by a large margin. The 0.65 was never the true performance.

---

## Files Already Fixed

I've already applied the fix to three files:

### 1. [utils/metrics.py](file:///c:/Users/khadi/Documents/change_detection%20-%20Copy/utils/metrics.py)
Added `MetricTracker` class that correctly accumulates TP/FP/FN across batches.

### 2. [train.py](file:///c:/Users/khadi/Documents/change_detection%20-%20Copy/train.py) — `validate()` function
Now uses `MetricTracker` so that early stopping and checkpoint saving are based on the correct F1.

### 3. [evaluate.py](file:///c:/Users/khadi/Documents/change_detection%20-%20Copy/evaluate.py) — test evaluation loop
Now uses `MetricTracker` for final test set reporting.

---

## Full Architecture Review (Everything Else is Solid)

### ✅ Model Architecture — Good
- **Siamese ResNet-50** with shared encoder weights — standard and effective for change detection
- **FPN-style decoder** with skip connections — proper multi-scale fusion
- **Concatenate diff_mode** — more expressive than subtract (you're using this)
- Output upsampled to input resolution with bilinear interpolation

### ✅ Loss Function — Good
- **BCE + Dice** (50/50 weight) — the standard combination for segmentation
- **pos_weight=8.0** for BCE — correctly handles the class imbalance (change pixels are rare)
- Dice loss provides region-level overlap optimization

### ✅ Data Pipeline — Good
- **LEVIR-CD** dataset properly loaded with 445 train / 64 val / 128 test images
- **On-the-fly tiling** (256×256 tiles, stride 192) — proper overlapping crops
- **Synchronized spatial augmentation** (flips, rotations, shift-scale-rotate, grid distortion) applied identically to A, B, and mask
- **Independent color jitter** on A and B separately — correctly simulates acquisition differences
- **ImageNet normalization** — correct for pretrained ResNet backbone

### ✅ Training Configuration — Good
- AdamW optimizer with lr=1e-4, weight_decay=1e-4
- Cosine annealing scheduler
- Early stopping with patience=20
- Mixed precision training (when GPU available)

### ✅ Training Curves — Healthy

![Training curves](C:/Users/khadi/.gemini/antigravity/brain/b170c225-22f8-49f0-a58c-e54cbf18001f/training_curves_subtract.png)

- Train loss decreases smoothly from ~0.48 → ~0.09
- Val loss decreases from ~0.43 → ~0.20 (slight gap = mild overfitting, normal)
- Val F1 (red line) was showing the **buggy batch-averaged** value, plateauing at ~0.65

### ✅ Qualitative Results — Look Great

![Sample prediction](C:/Users/khadi/.gemini/antigravity/brain/b170c225-22f8-49f0-a58c-e54cbf18001f/sample_00.png)

The model correctly identifies building footprint changes. Predictions closely match ground truth masks.

---

## Minor Improvements You Could Still Make

These are optional refinements, **not necessary** since F1 is already 0.906:

| Improvement | Expected Gain | Effort |
|---|---|---|
| Filter out all-empty tiles during training (10.6% of labels are blank) | +1-2% F1 | Low |
| Use `ReduceLROnPlateau` instead of `CosineAnnealing` (react to val F1) | +1% F1 | Low |
| Test-time augmentation (TTA: flip/rotate predictions and average) | +1-2% F1 | Low |
| Deeper backbone (ResNet-101 or EfficientNet-B4) | +1-3% F1 | Medium |
| Attention mechanisms (CBAM or SE blocks) in decoder | +1-2% F1 | Medium |
| Use BIT (Binary Transformer) or ChangeFormer architecture | +2-5% F1 | High |

---

## Summary

| Aspect | Verdict |
|---|---|
| Model architecture | ✅ Correct |
| Loss function | ✅ Correct |
| Data augmentation | ✅ Correct |
| Training loop | ✅ Correct |
| Hyperparameters | ✅ Reasonable |
| **Metric calculation** | **🐛 Was the only bug — now fixed** |
| **Actual Test F1** | **0.906 (was showing 0.664 due to bug)** |
