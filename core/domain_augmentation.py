"""
Domain Generalization Augmentation — Cross-Dataset Robustness

Farkli dataset'lerin sikistirma, cozunurluk, renk profili ve
post-processing farkliliklarini simule eden augmentasyonlar.

Amac: Modelin dataset-spesifik artefaktlara degil, genel deepfake
ozelliklerine odaklanmasini saglamak.
"""

import io
import random
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import torch


class DomainAugmentation:
    """
    Domain-invariant ozellik ogrenimi icin agresif augmentasyonlar.

    Augmentasyon cesitleri:
        1. Random JPEG Quality (Q=30-95) — sikistirma cesitliligi
        2. Random Downscale-Upscale — cozunurluk cesitliligi
        3. Random Gaussian Noise — sensor gurultu simulasyonu
        4. Random Sharpen/Blur — post-processing cesitliligi
        5. Random Color Shift — renk uzayi cesitliligi
        6. Random Gamma — aydinlatma cesitliligi
    """

    def __init__(self, prob: float = 0.5):
        """
        Args:
            prob: Her augmentasyon tipinin uygulanma olasiligi
        """
        self.prob = prob

    def __call__(self, img: Image.Image) -> Image.Image:
        """Rastgele domain augmentasyonu uygula."""
        # Her augmentasyon bagimsiz prob ile
        if random.random() < self.prob:
            img = self._random_jpeg_quality(img)

        if random.random() < self.prob * 0.6:
            img = self._random_downscale_upscale(img)

        if random.random() < self.prob * 0.4:
            img = self._random_noise(img)

        if random.random() < self.prob * 0.5:
            img = self._random_sharpen_blur(img)

        if random.random() < self.prob * 0.5:
            img = self._random_color_shift(img)

        if random.random() < self.prob * 0.4:
            img = self._random_gamma(img)

        return img

    def _random_jpeg_quality(self, img: Image.Image) -> Image.Image:
        """Rastgele JPEG kalitesinde yeniden sikistir (Q=30-95)."""
        quality = random.randint(30, 95)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")

    def _random_downscale_upscale(self, img: Image.Image) -> Image.Image:
        """
        Rastgele kucult + buyut (cozunurluk cesitliligi).
        Farkli dataset'ler farkli cozunurluklerde — bunu simule eder.
        """
        w, h = img.size
        scale = random.uniform(0.4, 0.85)
        small_w, small_h = int(w * scale), int(h * scale)
        if small_w < 32 or small_h < 32:
            return img

        # Downscale
        resample_down = random.choice([Image.BILINEAR, Image.NEAREST, Image.BICUBIC])
        img_small = img.resize((small_w, small_h), resample_down)

        # Upscale (geri)
        resample_up = random.choice([Image.BILINEAR, Image.BICUBIC])
        img_up = img_small.resize((w, h), resample_up)

        return img_up

    def _random_noise(self, img: Image.Image) -> Image.Image:
        """Rastgele Gaussian noise ekle."""
        arr = np.array(img, dtype=np.float32)
        sigma = random.uniform(2, 15)
        noise = np.random.normal(0, sigma, arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    def _random_sharpen_blur(self, img: Image.Image) -> Image.Image:
        """Rastgele sharpen veya blur uygula."""
        choice = random.choice(["sharpen", "blur", "median"])
        if choice == "sharpen":
            enhancer = ImageEnhance.Sharpness(img)
            factor = random.uniform(1.5, 3.0)
            return enhancer.enhance(factor)
        elif choice == "blur":
            radius = random.uniform(0.5, 2.5)
            return img.filter(ImageFilter.GaussianBlur(radius=radius))
        else:
            return img.filter(ImageFilter.MedianFilter(size=3))

    def _random_color_shift(self, img: Image.Image) -> Image.Image:
        """
        Rastgele renk kanal kaydirmasi.
        Farkli kameralar/codec'ler farkli renk profilleri uretir.
        """
        arr = np.array(img, dtype=np.float32)

        # Her kanal icin kucuk shift
        for c in range(3):
            shift = random.uniform(-15, 15)
            arr[:, :, c] = np.clip(arr[:, :, c] + shift, 0, 255)

        return Image.fromarray(arr.astype(np.uint8))

    def _random_gamma(self, img: Image.Image) -> Image.Image:
        """Rastgele gamma duzeltmesi — aydinlatma cesitliligi."""
        gamma = random.uniform(0.7, 1.5)
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = np.power(arr, gamma)
        arr = (arr * 255).clip(0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    def __repr__(self):
        return f"DomainAugmentation(prob={self.prob})"
