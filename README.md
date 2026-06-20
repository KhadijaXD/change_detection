# Satellite Image Change Detection — Siamese ResNet-34

Binary change detection on satellite imagery using a Siamese network with a shared ResNet-34 backbone and FPN-style decoder. Trained on LEVIR-CD, with scripts for Sentinel-2 inference on Lahore urban growth and the 2022 Indus River floods.

## Quick Start

```bash
# 1. Install dependencies
cd change_detection
pip install -r requirements.txt

# 2. Download LEVIR-CD from Kaggle and arrange as:
#    images/{train,val,test}/{A,B}/   — satellite image pairs
#    labels/{train,val,test}/         — binary masks (0/255)

# 3. Train (default config: subtract mode, ResNet-34, 50 epochs)
python train.py --config configs/default.yaml

# 4. Evaluate on test split
python evaluate.py --config configs/default.yaml --checkpoint checkpoints/best.pth

# 5. Run inference on a single image pair
python infer.py --img_a path/to/before.png --img_b path/to/after.png \
    --checkpoint checkpoints/best.pth --output change_mask.png

# 6. Download Sentinel-2 data (requires Copernicus credentials)
python download_sentinel.py --region lahore --year 2019
python download_sentinel.py --region lahore --year 2024
python download_sentinel.py --region floods --year 2022 --month 8

# 7. Ablation: switch to concatenation mode
#    Edit configs/default.yaml → diff_mode: "concatenate"
#    Then re-run train.py
```

## Switching Diff Modes

Change `diff_mode` in `configs/default.yaml`:
- `"subtract"` — element-wise feature subtraction (default, fewer parameters)
- `"concatenate"` — channel-wise concatenation (more expressive, larger decoder)

## Evaluation Targets

| Metric    | Target |
|-----------|--------|
| Test IoU  | > 0.65 |
| Test F1   | > 0.78 |

---

## Experiment Log Template

| diff_mode   | epochs | lr     | val_F1 | test_F1 | test_IoU | notes |
|-------------|--------|--------|--------|---------|----------|-------|
| subtract    |        | 1e-4   |        |         |          |       |
| concatenate |        | 1e-4   |        |         |          |       |
|             |        |        |        |         |          |       |
|             |        |        |        |         |          |       |
