"""
Deepfake Detection System v3.0 — Compression Analysis Module
JPEG kalite tahmini, sikistirma robustness sweep, platform simulasyonu.
"""
import io
import numpy as np
from PIL import Image
from typing import Optional

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


# Platform sikistirma profilleri
PLATFORM_PROFILES = {
    "original": {"label": "Orijinal", "quality": 100, "max_size": None, "icon": "📷"},
    "twitter": {"label": "Twitter/X", "quality": 80, "max_size": 4096, "icon": "🐦"},
    "tiktok": {"label": "TikTok", "quality": 72, "max_size": 1080, "icon": "🎵"},
}


def estimate_jpeg_quality(image) -> dict:
    """
    JPEG kalite tahmini — DCT katsayi varyans analizi + blockiness.

    Returns:
        dict: {
            estimated_quality: int (0-100),
            blockiness_score: float (0-1, yuksek = daha cok blok artefakti),
            noise_level: float,
            reliability: str ('high' / 'medium' / 'low'),
            reliability_pct: int (0-100),
            warning: str veya None,
        }
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    img = image.convert("RGB")

    # Yontem 1: JPEG yeniden sikistirma karsilastirmasi
    est_quality = _estimate_quality_by_recompression(img)

    # Yontem 2: Blockiness skoru (8x8 DCT blok siniri sureksizligi)
    blockiness = _measure_blockiness(img)

    # Yontem 3: Gurultu seviyesi tahmini
    noise_level = _estimate_noise_level(img)

    # Guvenilirlik puani
    if est_quality >= 85:
        reliability = "high"
        reliability_pct = min(95, 70 + est_quality // 5)
        warning = None
    elif est_quality >= 70:
        reliability = "medium"
        reliability_pct = max(40, est_quality - 20)
        warning = "Orta duzey sikistirma tespit edildi. Sonuclar etkilenebilir."
    else:
        reliability = "low"
        reliability_pct = max(20, est_quality - 30)
        warning = "Yuksek sikistirma tespit edildi! Ozellikle frekans analizi (DWT/GradCAM) sonuclari guvenilir olmayabilir."

    return {
        "estimated_quality": est_quality,
        "blockiness_score": round(blockiness, 4),
        "noise_level": round(noise_level, 4),
        "reliability": reliability,
        "reliability_pct": reliability_pct,
        "warning": warning,
    }


def _estimate_quality_by_recompression(img: Image.Image) -> int:
    """
    JPEG kalite tahmini: orijinal goruntu ile farkli Q seviyeleri arasindaki
    SSIM/MSE farki en az olan Q = tahmini kalite.
    """
    img_np = np.array(img).astype(np.float64)
    best_q = 95
    best_diff = float("inf")

    for q in range(30, 100, 5):
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=q)
        buf.seek(0)
        recompressed = np.array(Image.open(buf)).astype(np.float64)

        if recompressed.shape != img_np.shape:
            continue

        mse = np.mean((img_np - recompressed) ** 2)
        if mse < best_diff:
            best_diff = mse
            best_q = q

    return best_q


def _measure_blockiness(img: Image.Image) -> float:
    """
    8x8 blok siniri sureksizligi olcer.
    Yuksek blockiness = yuksek JPEG sikistirma.
    """
    gray = np.array(img.convert("L")).astype(np.float64)
    h, w = gray.shape

    if h < 16 or w < 16:
        return 0.0

    # Yatay 8-piksel sinirlarindaki fark
    h_edges = []
    for x in range(8, w - 1, 8):
        diff = np.abs(gray[:, x] - gray[:, x - 1])
        h_edges.append(np.mean(diff))

    # Dikey 8-piksel sinirlarindaki fark
    v_edges = []
    for y in range(8, h - 1, 8):
        diff = np.abs(gray[y, :] - gray[y - 1, :])
        v_edges.append(np.mean(diff))

    if not h_edges or not v_edges:
        return 0.0

    # Blok sinirlarindaki ortalama fark / genel ortalama fark
    block_diff = (np.mean(h_edges) + np.mean(v_edges)) / 2

    # Normalize: 0-1 araligina cek
    # Tipik degerler: orijinal=2-5, agir sikistirma=10-25
    return min(1.0, block_diff / 30.0)


def _estimate_noise_level(img: Image.Image) -> float:
    """Laplacian varyans ile gurultu seviyesi tahmini."""
    if not HAS_CV2:
        return 0.0
    gray = np.array(img.convert("L"))
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var()) / 1000.0


def compression_robustness_sweep(
    image, predictor, qualities=None
) -> dict:
    """
    Sikistirma robustness sweep — farkli JPEG kalite seviyelerinde
    model tahmininin nasil degistigini analiz et.

    Args:
        image: PIL Image
        predictor: DeepfakePredictor instance
        qualities: Test edilecek kalite seviyeleri

    Returns:
        dict: {
            qualities: list[int],
            fake_probs: list[float],
            verdicts: list[str],
            confidences: list[float],
            decision_flip_quality: int veya None,
            original_verdict: str,
        }
    """
    if qualities is None:
        qualities = [95, 90, 85, 80, 75, 70, 65, 60, 50, 40, 30, 20]

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    img = image.convert("RGB")

    # Orijinal tahmin
    orig_result = predictor.predict(np.array(img))
    original_verdict = orig_result["label"]

    fake_probs = []
    verdicts = []
    confidences = []
    flip_quality = None

    for q in qualities:
        compressed = _compress_jpeg(img, q)
        result = predictor.predict(np.array(compressed))

        fake_probs.append(result["fake_prob"])
        verdicts.append(result["label"])
        confidences.append(result["confidence"])

        # Karar degisimi tespit
        if flip_quality is None and result["label"] != original_verdict:
            flip_quality = q

    return {
        "qualities": qualities,
        "fake_probs": fake_probs,
        "verdicts": verdicts,
        "confidences": confidences,
        "decision_flip_quality": flip_quality,
        "original_verdict": original_verdict,
    }


def simulate_platform_compression(image, platform: str) -> dict:
    """
    Belirli bir platformun sikistirmasini simule et.

    Returns:
        dict: {
            platform: str,
            compressed_image: PIL Image,
            quality_used: int,
            original_size: tuple,
            compressed_size: tuple,
            file_size_kb: float,
        }
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    img = image.convert("RGB")

    profile = PLATFORM_PROFILES.get(platform, PLATFORM_PROFILES["original"])
    quality = profile["quality"]
    max_size = profile["max_size"]

    # Resize (platform max boyutu)
    original_size = img.size
    if max_size and max(img.size) > max_size:
        ratio = max_size / max(img.size)
        new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
        img = img.resize(new_size, Image.LANCZOS)

    # JPEG sikistirma
    if quality < 100:
        compressed = _compress_jpeg(img, quality)
    else:
        compressed = img

    # Dosya boyutu
    buf = io.BytesIO()
    compressed.save(buf, format="JPEG", quality=quality)
    file_size_kb = len(buf.getvalue()) / 1024

    return {
        "platform": platform,
        "platform_label": profile["label"],
        "compressed_image": compressed,
        "quality_used": quality,
        "original_size": original_size,
        "compressed_size": compressed.size,
        "file_size_kb": round(file_size_kb, 1),
    }


def get_reliability_rating(quality: int) -> dict:
    """Kaliteye gore guvenilirlik degerlendirmesi."""
    if quality >= 90:
        return {"level": "high", "color": "#22C55E", "text": "Yuksek Guvenilirlik",
                "detail": "Goruntu kalitesi yeterli, tum analiz kanallari guvenilir."}
    elif quality >= 75:
        return {"level": "medium", "color": "#F59E0B", "text": "Orta Guvenilirlik",
                "detail": "Frekans analizi (DWT) kismi etkilenmis olabilir. Geometri ve semantik analizler guvenilir."}
    elif quality >= 55:
        return {"level": "low", "color": "#EF4444", "text": "Dusuk Guvenilirlik",
                "detail": "Agir sikistirma. GradCAM/DWT sonuclari yaniltici olabilir. Sadece yuz geometrisi analizi guvenilir."}
    else:
        return {"level": "critical", "color": "#991B1B", "text": "Cok Dusuk Guvenilirlik",
                "detail": "Asiri sikistirma. Sonuclara temkinli yaklasmaniz onerilir."}


def _compress_jpeg(img: Image.Image, quality: int) -> Image.Image:
    """Gorseli belirli JPEG kalitesinde yeniden sikistir."""
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")
