"""
Siamese Change Detection Network
=================================
A Siamese encoder (shared ResNet-34 backbone via segmentation-models-pytorch)
with configurable feature difference mode and an FPN-style lightweight decoder.
"""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp


# --- DECODER BLOCK ---
class DecoderBlock(nn.Module):
    """Single decoder stage: upsample + concat skip + conv-bn-relu × 2.

    Args:
        in_ch: Input channels (from deeper level).
        skip_ch: Channels from the skip connection.
        out_ch: Output channels.
    """

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        if skip is not None:
            # Handle size mismatch from odd dimensions
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


# --- LIGHTWEIGHT FPN-STYLE DECODER ---
class FPNDecoder(nn.Module):
    """Multi-scale decoder that progressively upsamples difference features.

    Args:
        encoder_channels: List of channel counts from the encoder (low-res → high-res).
        decoder_channels: List of output channel counts per decoder stage.
        diff_mode: 'subtract' or 'concatenate' — affects input channel widths.
    """

    def __init__(
        self,
        encoder_channels: List[int],
        decoder_channels: List[int],
        diff_mode: str = "subtract",
    ):
        super().__init__()
        self.diff_mode = diff_mode
        ch_mult = 2 if diff_mode == "concatenate" else 1

        # Effective channels after differencing
        enc_ch = [c * ch_mult for c in encoder_channels]

        # Build decoder blocks from deepest to shallowest
        blocks = []
        in_ch = enc_ch[-1]  # deepest level
        for i, out_ch in enumerate(decoder_channels):
            # Skip connection from the next shallower encoder level
            skip_idx = len(enc_ch) - 2 - i
            skip_ch = enc_ch[skip_idx] if skip_idx >= 0 else 0
            blocks.append(DecoderBlock(in_ch, skip_ch, out_ch))
            in_ch = out_ch
        self.blocks = nn.ModuleList(blocks)

    def forward(self, diff_features: List[torch.Tensor]) -> torch.Tensor:
        """Decode difference feature maps.

        Args:
            diff_features: List ordered from shallowest to deepest resolution.

        Returns:
            Decoded feature map at the shallowest decoder level.
        """
        # Start from deepest
        x = diff_features[-1]
        for i, block in enumerate(self.blocks):
            skip_idx = len(diff_features) - 2 - i
            skip = diff_features[skip_idx] if skip_idx >= 0 else None
            x = block(x, skip)
        return x


# --- SIAMESE CHANGE DETECTOR ---
class SiameseChangeDetector(nn.Module):
    """Siamese network for binary change detection.

    Uses a shared encoder (e.g. ResNet-34) to extract multi-scale features
    from both before (A) and after (B) images. Features are compared via
    element-wise subtraction or concatenation, then decoded to a binary mask.

    Args:
        backbone: Encoder backbone name (e.g. 'resnet34').
        pretrained: Whether to load ImageNet-pretrained weights.
        diff_mode: 'subtract' for element-wise difference, 'concatenate' for
                   channel-wise concatenation.
        decoder_channels: List of channel widths for each decoder stage.
    """

    def __init__(
        self,
        backbone: str = "resnet34",
        pretrained: bool = True,
        diff_mode: str = "subtract",
        decoder_channels: Optional[List[int]] = None,
    ):
        super().__init__()
        self.diff_mode = diff_mode
        if decoder_channels is None:
            decoder_channels = [256, 128, 64, 32]

        # --- SHARED ENCODER ---
        weights = "imagenet" if pretrained else None
        self.encoder = smp.encoders.get_encoder(backbone, weights=weights)
        enc_channels = list(self.encoder.out_channels)  # e.g. [3, 64, 64, 128, 256, 512]

        # We skip the first entry (stem input channels = 3) for the decoder
        self.enc_channels_for_decoder = enc_channels[1:]  # [64, 64, 128, 256, 512]

        # --- DECODER ---
        self.decoder = FPNDecoder(
            encoder_channels=self.enc_channels_for_decoder,
            decoder_channels=decoder_channels,
            diff_mode=diff_mode,
        )

        # --- SEGMENTATION HEAD ---
        self.seg_head = nn.Sequential(
            nn.Conv2d(decoder_channels[-1], 1, kernel_size=1),
        )

    def _encode(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Run the shared encoder on a single image.

        Args:
            x: Input image tensor (B, 3, H, W).

        Returns:
            List of feature maps at increasing depth (shallowest first).
        """
        features = self.encoder(x)
        return features[1:]  # drop the stem-level (raw 3-ch) output

    def _compute_diff(
        self, feats_A: List[torch.Tensor], feats_B: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """Compute per-scale feature differences.

        Args:
            feats_A: Encoder features for image A.
            feats_B: Encoder features for image B.

        Returns:
            List of difference feature maps.
        """
        diffs = []
        for fa, fb in zip(feats_A, feats_B):
            if self.diff_mode == "subtract":
                diffs.append(fa - fb)
            elif self.diff_mode == "concatenate":
                diffs.append(torch.cat([fa, fb], dim=1))
            else:
                raise ValueError(f"Unknown diff_mode: {self.diff_mode}")
        return diffs

    def forward(self, img_A: torch.Tensor, img_B: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            img_A: Before image (B, 3, H, W).
            img_B: After image  (B, 3, H, W).

        Returns:
            Logit map (B, 1, H, W) — apply sigmoid for probabilities.
        """
        feats_A = self._encode(img_A)
        feats_B = self._encode(img_B)
        diff = self._compute_diff(feats_A, feats_B)
        decoded = self.decoder(diff)

        # Upsample decoded features to input resolution
        logits = self.seg_head(decoded)
        logits = F.interpolate(
            logits, size=img_A.shape[2:], mode="bilinear", align_corners=False
        )
        return logits
