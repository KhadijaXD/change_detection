"""
LEVIR-CD Dataset Loader
=======================
Loads co-registered satellite image pairs (A, B) and optional binary change masks.
Automatically tiles 1024×1024 images into smaller crops at runtime.
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2


# --- IMAGENET NORMALIZATION STATS ---
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(split: str, tile_size: int) -> A.Compose:
    """Build augmentation pipeline.

    For training: random flips, 90° rotations, and independent color jitter.
    For val/test: normalization only.

    Args:
        split: One of 'train', 'val', 'test'.
        tile_size: Spatial size of each crop (not used for resize here since
                   tiling handles sizing, but kept for future flexibility).

    Returns:
        An albumentations Compose pipeline.
    """
    if split == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.3),
                A.GridDistortion(p=0.2),
                A.GaussNoise(p=0.2),
                A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ToTensorV2(),
            ],
            additional_targets={"image_B": "image"},
        )
    else:
        return A.Compose(
            [
                A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ToTensorV2(),
            ],
            additional_targets={"image_B": "image"},
        )


def build_color_jitter() -> A.Compose:
    """Separate color jitter applied independently to A and B (NOT synchronized)."""
    return A.Compose(
        [
            A.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.5
            ),
        ]
    )


class LEVIRDataset(Dataset):
    """LEVIR-CD dataset with on-the-fly tiling.

    Loads full 1024×1024 images and tiles them into `tile_size × tile_size`
    crops at runtime, avoiding redundant disk writes.

    Args:
        root: Path to the images folder (containing train/val/test sub-dirs).
        split: One of 'train', 'val', 'test'.
        tile_size: Size of square crops.
        stride: Stride between consecutive tiles.
        transform: Optional custom transform (overrides default).
    """

    def __init__(
        self,
        root: str,
        split: str,
        tile_size: int = 256,
        stride: int = 256,
        transform: Optional[A.Compose] = None,
    ):
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.tile_size = tile_size
        self.stride = stride
        self.transform = transform or build_transforms(split, tile_size)
        self.color_jitter = build_color_jitter() if split == "train" else None

        # --- LOCATE IMAGES ---
        self.dir_A = self.root / split / "A"
        self.dir_B = self.root / split / "B"
        assert self.dir_A.exists(), f"Directory not found: {self.dir_A}"
        assert self.dir_B.exists(), f"Directory not found: {self.dir_B}"

        self.filenames = sorted(
            [f for f in os.listdir(self.dir_A) if f.lower().endswith((".png", ".jpg", ".tif"))]
        )
        assert len(self.filenames) > 0, f"No images found in {self.dir_A}"

        # --- LOCATE MASKS (optional) ---
        label_root = self.root.parent / "labels" / split
        if not label_root.exists():
            # Try alternative structure: masks alongside images
            label_root = self.root / split / "label"
        self.label_dir = label_root if label_root.exists() else None
        if self.label_dir is None:
            print(f"[WARN] No mask directory found for split '{split}'. Masks will be None.")

        # --- BUILD TILE INDEX ---
        self.tile_index = self._build_tile_index()

    def _build_tile_index(self) -> List[Tuple[int, int, int]]:
        """Pre-compute (file_idx, row_start, col_start) for every tile.

        Returns:
            List of (file_index, row_offset, col_offset) tuples.
        """
        # Read the first image to determine spatial dimensions
        sample = np.array(Image.open(self.dir_A / self.filenames[0]))
        h, w = sample.shape[:2]

        tiles = []
        for fi in range(len(self.filenames)):
            for r in range(0, h - self.tile_size + 1, self.stride):
                for c in range(0, w - self.tile_size + 1, self.stride):
                    tiles.append((fi, r, c))
        return tiles

    def __len__(self) -> int:
        return len(self.tile_index)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Load a single tile.

        Returns:
            Dict with keys: 'A' (C,H,W tensor), 'B' (C,H,W tensor),
            'mask' (1,H,W tensor or dummy zeros), 'meta' (dict).
        """
        fi, r, c = self.tile_index[idx]
        fname = self.filenames[fi]
        ts = self.tile_size

        # --- LOAD FULL IMAGES AND CROP ---
        img_A = np.array(Image.open(self.dir_A / fname).convert("RGB"))
        img_B = np.array(Image.open(self.dir_B / fname).convert("RGB"))
        tile_A = img_A[r : r + ts, c : c + ts]
        tile_B = img_B[r : r + ts, c : c + ts]

        # --- LOAD MASK ---
        has_mask = False
        if self.label_dir is not None:
            mask_path = self.label_dir / fname
            if mask_path.exists():
                mask_full = np.array(Image.open(mask_path).convert("L"))
                tile_mask = mask_full[r : r + ts, c : c + ts]
                # Convert 255 → 1
                tile_mask = (tile_mask > 127).astype(np.float32)
                has_mask = True

        if not has_mask:
            tile_mask = np.zeros((ts, ts), dtype=np.float32)

        # --- INDEPENDENT COLOR JITTER (train only) ---
        if self.color_jitter is not None:
            tile_A = self.color_jitter(image=tile_A)["image"]
            tile_B = self.color_jitter(image=tile_B)["image"]

        # --- SYNCHRONIZED SPATIAL AUGMENTATIONS + NORMALIZE ---
        augmented = self.transform(
            image=tile_A, image_B=tile_B, mask=tile_mask
        )
        tensor_A = augmented["image"]           # (3, H, W)
        tensor_B = augmented["image_B"]         # (3, H, W)
        tensor_mask = augmented["mask"]         # (H, W)

        # Ensure mask is (1, H, W)
        if tensor_mask.ndim == 2:
            tensor_mask = tensor_mask.unsqueeze(0)

        return {
            "A": tensor_A,
            "B": tensor_B,
            "mask": tensor_mask,
            "has_mask": has_mask,
            "meta": {"file": fname, "tile_idx": idx, "row": r, "col": c},
        }
