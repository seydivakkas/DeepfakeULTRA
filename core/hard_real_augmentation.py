"""
GÖREV 3: Sınıf-Bilinçli Augmentation Pipeline
REAL görsellerden beauty filter/HDR/low-quality simülasyonu uygular.
FAKE görsellere sadece geometrik augmentation (frekans bozulmaz).

Kullanım:
    Eğitim pipeline'ında otomatik çağrılır (data_pipeline.py entegrasyonu).
"""
import io
import random
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


class HardRealAugmentation:
    """
    REAL görsellere kozmetik/düzenleme simülasyonu uygular.
    FAKE görsellere UYGULANMAZ (frekans imzasını bozar).

    Eğitim sırasında on-the-fly çalışır, epoch-aware değildir.
    Curriculum Learning ile birlikte kullanılır (trainer kontrol eder).
    """

    def __init__(self, prob: float = 0.3):
        """
        Args:
            prob: Augmentation uygulama olasılığı (0-1).
                  0.3 = her 3 REAL görselden 1'ine uygulanır.
        """
        self.prob = prob
        self.transforms = [
            self._beauty_filter,
            self._hdr_edit,
            self._low_quality,
            self._heavy_makeup,
            self._aggressive_jpeg,
            self._dct_quantization_noise,
        ]

    def __call__(self, image: Image.Image, label: int) -> Image.Image:
        """
        Args:
            image: PIL Image (RGB)
            label: 0=REAL, 1=FAKE

        Returns:
            Augmented PIL Image (sadece REAL ise augmentation uygulanır)
        """
        # FAKE görsellere kozmetik augmentation UYGULANMAZ
        if label == 1:
            return image

        # Olasılık kontrolü
        if random.random() > self.prob:
            return image

        # Rastgele bir transform seç ve uygula
        transform = random.choice(self.transforms)
        try:
            return transform(image)
        except Exception:
            return image

    def _beauty_filter(self, img: Image.Image) -> Image.Image:
        """Skin smoothing + saturation boost."""
        if HAS_CV2:
            img_np = np.array(img, dtype=np.uint8)
            smoothed = cv2.bilateralFilter(
                img_np, d=7,
                sigmaColor=random.randint(40, 70),
                sigmaSpace=random.randint(40, 70)
            )
            img = Image.fromarray(smoothed)
        else:
            img = img.filter(ImageFilter.SMOOTH_MORE)

        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(random.uniform(1.1, 1.3))
        return img

    def _hdr_edit(self, img: Image.Image) -> Image.Image:
        """CLAHE + gamma + color grading."""
        img_np = np.array(img, dtype=np.uint8)

        if HAS_CV2:
            lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
            clahe = cv2.createCLAHE(clipLimit=random.uniform(1.5, 3.0), tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            img_np = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        # Gamma
        gamma = random.uniform(0.75, 1.3)
        img_np = np.clip(np.power(img_np / 255.0, gamma) * 255, 0, 255).astype(np.uint8)
        img = Image.fromarray(img_np)

        enhancer = ImageEnhance.Contrast(img)
        return enhancer.enhance(random.uniform(1.05, 1.3))

    def _low_quality(self, img: Image.Image) -> Image.Image:
        """Downscale + noise + heavy JPEG."""
        w, h = img.size
        target = random.choice([64, 96, 128])
        img = img.resize((target, target), Image.BILINEAR)
        img = img.resize((w, h), Image.BILINEAR)

        # Noise
        img_np = np.array(img, dtype=np.float32)
        sigma = random.uniform(2, 8)
        noise = np.random.normal(0, sigma, img_np.shape)
        img_np = np.clip(img_np + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(img_np)

        # JPEG
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=random.randint(25, 50))
        buf.seek(0)
        return Image.open(buf).convert("RGB")

    def _heavy_makeup(self, img: Image.Image) -> Image.Image:
        """Saturation + contrast + smoothing."""
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(random.uniform(1.2, 1.5))

        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(random.uniform(1.1, 1.25))

        if HAS_CV2:
            img_np = np.array(img, dtype=np.uint8)
            smoothed = cv2.bilateralFilter(img_np, d=5, sigmaColor=50, sigmaSpace=50)
            img = Image.fromarray(smoothed)

        return img

    def _aggressive_jpeg(self, img: Image.Image) -> Image.Image:
        """Çok düşük kalite JPEG sıkıştırma."""
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=random.randint(15, 35))
        buf.seek(0)
        return Image.open(buf).convert("RGB")

    def _dct_quantization_noise(self, img: Image.Image) -> Image.Image:
        """
        DCT domain'de quantization noise ekler.
        Gerçek JPEG sıkıştırma artefaktlarını simüle eder.
        REAL görsellerin frekans profilini gerçekçi hale getirir.
        """
        if not HAS_CV2:
            return img

        img_np = np.array(img, dtype=np.float32)
        result = img_np.copy()

        # Quantization şiddeti
        q_strength = random.uniform(5.0, 25.0)

        for ch in range(3):
            channel = result[:, :, ch]
            h, w = channel.shape

            # 8x8 blok tabanlı DCT quantization (JPEG standardı)
            block_size = 8
            for y in range(0, h - block_size + 1, block_size):
                for x in range(0, w - block_size + 1, block_size):
                    block = channel[y:y + block_size, x:x + block_size]
                    dct_block = cv2.dct(block)
                    # Quantize: round(dct / q) * q
                    dct_block = np.round(dct_block / q_strength) * q_strength
                    channel[y:y + block_size, x:x + block_size] = cv2.idct(dct_block)

            result[:, :, ch] = channel

        result = np.clip(result, 0, 255).astype(np.uint8)
        return Image.fromarray(result)

