"""
Faz 2 — Self-Blended Images (SBI) Augmentation
Kendi yüz swap'lerinden öğrenen augmentation.

Referans: Shiohara & Yamasaki, "Detecting Deepfakes with Self-Blended Images" (CVPR 2022)

Mantik:
    1. Bir REAL gorsel al
    2. Ayni gorselin yuzunu landmark-guided transform et (warp, scale, rotate)
    3. Convex hull mask ile self-blend yap
    4. Sonuc: FAKE gibi gorunen ama REAL kaynakli gorsel
    → Model, blending artifact'lerini ogrenmeye zorlanir

Kullanim:
    sbi = SelfBlendedAugmentation()
    fake_image = sbi(real_image_np)  # numpy HWC -> numpy HWC
"""

import numpy as np
from PIL import Image
import random
import io

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False


# ═══════════════════════════════════════════════════════════
# YARDIMCI FONKSİYONLAR
# ═══════════════════════════════════════════════════════════

def _get_face_landmarks(image_rgb: np.ndarray) -> np.ndarray:
    """MediaPipe ile yuz landmark noktalarini cikar.

    Returns:
        (468, 2) numpy array — piksel koordinatlari, veya None
    """
    if not HAS_MEDIAPIPE:
        return None

    try:
        face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.3,
        )
        results = face_mesh.process(image_rgb)
        face_mesh.close()

        if not results.multi_face_landmarks:
            return None

        h, w = image_rgb.shape[:2]
        landmarks = results.multi_face_landmarks[0]
        coords = np.array([
            [lm.x * w, lm.y * h] for lm in landmarks.landmark
        ], dtype=np.float32)
        return coords
    except Exception:
        return None


def _get_convex_hull_mask(landmarks: np.ndarray, shape: tuple) -> np.ndarray:
    """Landmark noktalarindan convex hull mask olustur.

    Args:
        landmarks: (N, 2) piksel koordinatlari
        shape: (H, W) gorsel boyutu

    Returns:
        (H, W) binary mask [0, 1] float32
    """
    if not HAS_CV2:
        return np.ones(shape, dtype=np.float32)

    # Yuz kontur indeksleri (face oval)
    FACE_OVAL = [
        10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
        397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
        172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
    ]

    # Kullanilabilir indeksler
    valid_indices = [i for i in FACE_OVAL if i < len(landmarks)]
    if len(valid_indices) < 10:
        # Fallback: tum landmark'lardan convex hull
        hull_points = landmarks.astype(np.int32)
    else:
        hull_points = landmarks[valid_indices].astype(np.int32)

    hull = cv2.convexHull(hull_points)
    mask = np.zeros(shape, dtype=np.float32)
    cv2.fillConvexPoly(mask, hull, 1.0)

    # Kenar yumusatma (Gaussian blur)
    kernel_size = max(3, int(min(shape) * 0.05) | 1)  # Tek sayi olmali
    mask = cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)
    mask = np.clip(mask, 0, 1)

    return mask


def _random_face_transform(image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    """Yuze rastgele geometrik donusum uygula.

    Uygulanabilecek donusumler:
        - Hafif olcekleme (0.95-1.05)
        - Hafif dondurme (-5 ila +5 derece)
        - Hafif kayma (translate)
        - Color jitter (renk fark ayristirmasi icin)

    Returns:
        Donusturulmus gorsel (ayni boyut)
    """
    if not HAS_CV2:
        return image

    h, w = image.shape[:2]
    center = landmarks.mean(axis=0)

    # Rastgele parametreler
    scale = random.uniform(0.96, 1.04)
    angle = random.uniform(-4, 4)
    tx = random.uniform(-3, 3)
    ty = random.uniform(-3, 3)

    # Affine transform matrisi
    M = cv2.getRotationMatrix2D(tuple(center), angle, scale)
    M[0, 2] += tx
    M[1, 2] += ty

    transformed = cv2.warpAffine(
        image, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    return transformed


def _color_transfer(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Basit renk transferi — source'un renklerini target'a uydur.
    Lab color space'de mean/std matching.
    """
    if not HAS_CV2:
        return source

    try:
        source_lab = cv2.cvtColor(source, cv2.COLOR_RGB2LAB).astype(np.float32)
        target_lab = cv2.cvtColor(target, cv2.COLOR_RGB2LAB).astype(np.float32)

        # Her kanal icin mean/std eslestirme
        for ch in range(3):
            s_mean, s_std = source_lab[:, :, ch].mean(), source_lab[:, :, ch].std() + 1e-6
            t_mean, t_std = target_lab[:, :, ch].mean(), target_lab[:, :, ch].std() + 1e-6

            source_lab[:, :, ch] = (source_lab[:, :, ch] - s_mean) * (t_std / s_std) + t_mean

        source_lab = np.clip(source_lab, 0, 255).astype(np.uint8)
        return cv2.cvtColor(source_lab, cv2.COLOR_LAB2RGB)
    except Exception:
        return source


def _add_jpeg_artifact(image: np.ndarray, quality_range=(60, 85)) -> np.ndarray:
    """JPEG sikistirma artefakti ekle (gercekci forgery simulasyonu)."""
    try:
        pil_img = Image.fromarray(image)
        buffer = io.BytesIO()
        quality = random.randint(*quality_range)
        pil_img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return np.array(Image.open(buffer).convert("RGB"))
    except Exception:
        return image


# ═══════════════════════════════════════════════════════════
# ANA SBI SINIFI
# ═══════════════════════════════════════════════════════════

class SelfBlendedAugmentation:
    """
    Self-Blended Images (SBI) Augmentation.

    Bir REAL gorsel alir, yuzunu kendi uzerine self-blend ederek
    deepfake benzeri artefaktlar olusturur.

    Args:
        blend_alpha_range: (min, max) blend alpha araligi
        apply_color_transfer: Renk transferi uygula
        add_jpeg_compression: JPEG artefakti ekle
        jpeg_quality_range: JPEG kalite araligi
    """

    def __init__(
        self,
        blend_alpha_range: tuple = (0.3, 0.7),
        apply_color_transfer: bool = True,
        add_jpeg_compression: bool = True,
        jpeg_quality_range: tuple = (60, 85),
    ):
        self.blend_alpha_range = blend_alpha_range
        self.apply_color_transfer = apply_color_transfer
        self.add_jpeg_compression = add_jpeg_compression
        self.jpeg_quality_range = jpeg_quality_range

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        SBI uygula.

        Args:
            image: (H, W, 3) uint8 RGB numpy array

        Returns:
            (H, W, 3) uint8 RGB — self-blended gorsel (FAKE benzeri)
        """
        if not HAS_CV2 or not HAS_MEDIAPIPE:
            return image

        # 1. Landmark cikar
        landmarks = _get_face_landmarks(image)
        if landmarks is None:
            return image  # Yuz bulunamadi — orijinal don

        h, w = image.shape[:2]

        # 2. Yuz bolgesine rastgele geometrik donusum
        transformed = _random_face_transform(image.copy(), landmarks)

        # 3. Color transfer (opsiyonel — renk tutarsizligi ekler)
        if self.apply_color_transfer and random.random() < 0.5:
            # Hafif renk perturbasyonu
            brightness_shift = random.randint(-15, 15)
            transformed = np.clip(
                transformed.astype(np.int16) + brightness_shift, 0, 255
            ).astype(np.uint8)

        # 4. Convex hull mask olustur
        mask = _get_convex_hull_mask(landmarks, (h, w))

        # 5. Blend alpha
        alpha = random.uniform(*self.blend_alpha_range)

        # 6. Alpha blending: result = mask * (alpha*transformed + (1-alpha)*original) + (1-mask)*original
        mask_3d = mask[:, :, np.newaxis]
        blended_face = (alpha * transformed.astype(np.float32) +
                        (1 - alpha) * image.astype(np.float32))
        result = (mask_3d * blended_face +
                  (1 - mask_3d) * image.astype(np.float32))
        result = np.clip(result, 0, 255).astype(np.uint8)

        # 7. JPEG artefakti (opsiyonel)
        if self.add_jpeg_compression and random.random() < 0.4:
            result = _add_jpeg_artifact(result, self.jpeg_quality_range)

        return result

    def __repr__(self):
        return (f"SelfBlendedAugmentation("
                f"blend_alpha={self.blend_alpha_range}, "
                f"color_transfer={self.apply_color_transfer}, "
                f"jpeg={self.add_jpeg_compression})")


class SBITransform:
    """
    torchvision.transforms uyumlu SBI wrapper.
    PIL Image alir, SBI uygular, PIL Image dondurur.

    Kullanim:
        transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomApply([SBITransform()], p=0.3),
            transforms.ToTensor(),
            ...
        ])
    """

    def __init__(self, **kwargs):
        self.sbi = SelfBlendedAugmentation(**kwargs)

    def __call__(self, img):
        if isinstance(img, Image.Image):
            img_np = np.array(img)
            result = self.sbi(img_np)
            return Image.fromarray(result)
        return img

    def __repr__(self):
        return f"SBITransform({self.sbi})"


# ═══════════════════════════════════════════════════════════
# TEST
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=== SBI Augmentation Test ===")

    # Dummy test
    dummy = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    sbi = SelfBlendedAugmentation()
    result = sbi(dummy)
    print(f"Input: {dummy.shape} -> Output: {result.shape}")
    print(f"SBI: {sbi}")

    # PIL Transform test
    pil_img = Image.fromarray(dummy)
    transform = SBITransform()
    pil_result = transform(pil_img)
    print(f"PIL Transform: {pil_img.size} -> {pil_result.size}")

    print("\n✅ SBI modulu hazir!")
