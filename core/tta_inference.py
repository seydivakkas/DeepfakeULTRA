"""
Test-Time Augmentation (TTA) — Cross-Dataset Generalizasyon

Mevcut modele dokunmadan inference sirasinda N augmented kopya
uzerinden ortalama alarak daha guvenilir tahmin uretir.

Kullanim:
    from core.tta_inference import TTAPredictor
    predictor = TTAPredictor(model, n_aug=10)
    probs = predictor.predict_batch(rgb, freq, mesh)  # (batch, 2)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from PIL import Image
import io
import random


class TTAPredictor:
    """
    Test-Time Augmentation ile batch tahmin.

    Strateji:
        1. Orijinal girdi → tahmin
        2. Horizontal flip → tahmin
        3. N-2 rastgele augmented kopya → tahmin
        4. Tum tahminlerin ortalamasini al

    Augmentasyonlar:
        - Horizontal Flip
        - Rastgele JPEG sikistirma (Q=40-95)
        - Kucuk olcekli crop + resize
        - Hafif Gaussian blur
        - Hafif renk perturbasyonu
    """

    def __init__(self, model: nn.Module, n_aug: int = 10, device: str = "cuda"):
        self.model = model
        self.n_aug = max(n_aug, 2)  # Minimum: orijinal + flip
        self.device = device
        self.model.eval()

    @torch.no_grad()
    def predict_batch(
        self,
        rgb: torch.Tensor,
        freq: torch.Tensor,
        mesh: torch.Tensor,
    ) -> torch.Tensor:
        """
        TTA ile batch tahmin.

        Args:
            rgb:  (batch, 3, H, W)
            freq: (batch, C, H, W)
            mesh: (batch, 1404)

        Returns:
            probs: (batch, num_classes) — ortalama softmax olasiliklari
        """
        all_probs = []

        # 1. Orijinal girdi
        logits = self.model(rgb, freq, mesh)
        all_probs.append(F.softmax(logits, dim=1))

        # 2. Horizontal flip
        rgb_flip = torch.flip(rgb, dims=[3])
        freq_flip = torch.flip(freq, dims=[3])
        logits_flip = self.model(rgb_flip, freq_flip, mesh)
        all_probs.append(F.softmax(logits_flip, dim=1))

        # 3. Ek augmentasyonlar
        for i in range(self.n_aug - 2):
            aug_rgb, aug_freq = self._augment_batch(rgb, freq, i)
            logits_aug = self.model(aug_rgb, aug_freq, mesh)
            all_probs.append(F.softmax(logits_aug, dim=1))

        # Ortalama
        stacked = torch.stack(all_probs, dim=0)  # (n_aug, batch, classes)
        mean_probs = stacked.mean(dim=0)  # (batch, classes)
        return mean_probs

    def _augment_batch(
        self,
        rgb: torch.Tensor,
        freq: torch.Tensor,
        aug_idx: int,
    ) -> tuple:
        """GPU-native augmentasyon uygula."""
        B, C, H, W = rgb.shape

        # Augmentasyon secimi (dengeli dagilim)
        aug_type = aug_idx % 6

        if aug_type == 0:
            # Hafif Gaussian blur
            return self._gaussian_blur(rgb, freq)
        elif aug_type == 1:
            # Kucuk olcekli center crop
            return self._center_crop(rgb, freq, scale=0.9)
        elif aug_type == 2:
            # Renk perturbasyonu (brightness)
            factor = random.uniform(0.85, 1.15)
            return rgb * factor, freq
        elif aug_type == 3:
            # Renk perturbasyonu (contrast)
            mean = rgb.mean(dim=[2, 3], keepdim=True)
            factor = random.uniform(0.85, 1.15)
            return (rgb - mean) * factor + mean, freq
        elif aug_type == 4:
            # Kucuk gaussian noise
            noise = torch.randn_like(rgb) * 0.02
            return rgb + noise, freq
        else:
            # Daha buyuk center crop
            return self._center_crop(rgb, freq, scale=0.85)

    def _gaussian_blur(
        self,
        rgb: torch.Tensor,
        freq: torch.Tensor,
        kernel_size: int = 3,
    ) -> tuple:
        """Hafif Gaussian blur."""
        # Basit box blur (3x3 ortalama)
        pad = kernel_size // 2
        weight = torch.ones(1, 1, kernel_size, kernel_size, device=rgb.device)
        weight = weight / (kernel_size * kernel_size)

        B, C_rgb, H, W = rgb.shape
        # Her kanal icin ayri blur
        blurred_channels = []
        for c in range(C_rgb):
            ch = rgb[:, c:c+1, :, :]
            ch_padded = F.pad(ch, [pad]*4, mode='reflect')
            ch_blurred = F.conv2d(ch_padded, weight)
            blurred_channels.append(ch_blurred)
        blurred_rgb = torch.cat(blurred_channels, dim=1)

        return blurred_rgb, freq

    def _center_crop(
        self,
        rgb: torch.Tensor,
        freq: torch.Tensor,
        scale: float = 0.9,
    ) -> tuple:
        """Center crop + resize."""
        B, C_rgb, H, W = rgb.shape
        crop_h = int(H * scale)
        crop_w = int(W * scale)
        start_h = (H - crop_h) // 2
        start_w = (W - crop_w) // 2

        cropped_rgb = rgb[:, :, start_h:start_h+crop_h, start_w:start_w+crop_w]
        cropped_rgb = F.interpolate(cropped_rgb, size=(H, W), mode='bilinear', align_corners=False)

        # Freq branch icin de crop (ayni spatial boyut)
        B, C_freq, Hf, Wf = freq.shape
        crop_hf = int(Hf * scale)
        crop_wf = int(Wf * scale)
        start_hf = (Hf - crop_hf) // 2
        start_wf = (Wf - crop_wf) // 2
        cropped_freq = freq[:, :, start_hf:start_hf+crop_hf, start_wf:start_wf+crop_wf]
        cropped_freq = F.interpolate(cropped_freq, size=(Hf, Wf), mode='bilinear', align_corners=False)

        return cropped_rgb, cropped_freq
