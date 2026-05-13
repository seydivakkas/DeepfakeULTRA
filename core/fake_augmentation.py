"""
GÖREV 3: FAKE-Only Frekans-Güvenli Augmentation
FAKE görsellere sadece geometrik + hafif renk augmentasyonu uygular.
Frekans domain imzasını (GAN/Diffusion artefaktları) KORUR.

İZİN VERİLEN:
  - Geometrik: horizontal flip, rotation (±10°), mild crop
  - Renk: çok hafif brightness/contrast (±5%)
  - FrequencyMask: DCT domain'de rastgele maskeleme (YENİ)

YASAK (frekans imzasını bozar):
  - JPEG recompression
  - Gaussian blur / noise
  - Bilateral filter / smoothing
  - Heavy color jitter
"""
import random
import numpy as np
from PIL import Image, ImageEnhance

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


class FakeAwareAugmentation:
    """
    FAKE görsellere frekans-güvenli augmentation.

    Frekans domain'deki sahtecilik imzalarını koruyarak
    sadece spatial/geometrik dönüşümler uygular.
    """

    def __init__(self, prob: float = 0.5, use_freq_mask: bool = True):
        """
        Args:
            prob: Augmentation uygulama olasılığı.
            use_freq_mask: DCT domain frequency masking aktif mi.
        """
        self.prob = prob
        self.use_freq_mask = use_freq_mask and HAS_CV2

    def __call__(self, image: Image.Image) -> Image.Image:
        """FAKE görsele frekans-güvenli augmentation uygula."""
        if random.random() > self.prob:
            return image

        # Rastgele 1-2 transform seç ve sırayla uygula
        transforms = [
            self._horizontal_flip,
            self._mild_rotation,
            self._mild_color,
        ]

        if self.use_freq_mask:
            transforms.append(self._frequency_mask_dct)

        num_transforms = random.randint(1, 2)
        selected = random.sample(transforms, min(num_transforms, len(transforms)))

        for t in selected:
            try:
                image = t(image)
            except Exception:
                continue

        return image

    def _horizontal_flip(self, img: Image.Image) -> Image.Image:
        """%50 olasılıkla yatay çevirme."""
        if random.random() < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        return img

    def _mild_rotation(self, img: Image.Image) -> Image.Image:
        """±10° hafif döndürme — frekans imzasını minimal etkiler."""
        angle = random.uniform(-10, 10)
        return img.rotate(angle, resample=Image.BILINEAR, expand=False, fillcolor=(0, 0, 0))

    def _mild_color(self, img: Image.Image) -> Image.Image:
        """Çok hafif brightness/contrast ayarı (±5%)."""
        # Brightness
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(random.uniform(0.95, 1.05))

        # Contrast
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(random.uniform(0.95, 1.05))

        return img

    def _frequency_mask_dct(self, img: Image.Image) -> Image.Image:
        """
        DCT domain'de frekans maskeleme.
        Yüksek frekanslardaki belirli bantları rastgele maskeler.
        Bu, modelin farklı frekans bantlarına genelleme yapmasını sağlar.
        """
        if not HAS_CV2:
            return img

        img_np = np.array(img, dtype=np.float32)
        result = img_np.copy()

        for ch in range(3):
            channel = img_np[:, :, ch]

            # DCT
            dct = cv2.dct(channel)

            # Rastgele frekans bant maskeleme
            h, w = dct.shape
            mask_size = random.randint(2, max(3, h // 16))
            mask_x = random.randint(h // 4, h - mask_size)
            mask_y = random.randint(w // 4, w - mask_size)

            # Maskeleme oranı (hafif — imzayı tamamen silmemeli)
            damping = random.uniform(0.3, 0.7)
            dct[mask_x:mask_x + mask_size, mask_y:mask_y + mask_size] *= damping

            # Inverse DCT
            result[:, :, ch] = cv2.idct(dct)

        result = np.clip(result, 0, 255).astype(np.uint8)
        return Image.fromarray(result)
