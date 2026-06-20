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
## Training Logs 
<img width="931" height="730" alt="image" src="https://github.com/user-attachments/assets/fdba6c94-55ab-4bd4-858b-b92ca9ec1a3e" />

## Training Curve
<img width="980" height="474" alt="image" src="https://github.com/user-attachments/assets/346f609e-0611-45f9-a6f4-bf27b5a7a415" />

## Test Results
<img width="394" height="149" alt="image" src="https://github.com/user-attachments/assets/8221f0b9-93f1-47ce-8a87-826139e8386d" />

## Change Mask VIS
<img width="983" height="342" alt="image" src="https://github.com/user-attachments/assets/dba6f29f-8de1-4593-a823-d81e2f75b447" />






