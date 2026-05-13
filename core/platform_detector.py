"""
Sosyal Medya Platform Tespiti — Forensik Parmak Izi Analizi
JPEG quantization tablosu, EXIF metadata, cozunurluk ve blockiness
desenlerinden gorselin hangi platformdan geldigini otomatik tespit eder.
"""
import io
import struct
import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS
from typing import Optional

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


# ═══════════════════════════════════════════════════════════
# PLATFORM PARMAK IZI PROFILLERI
# ═══════════════════════════════════════════════════════════

PLATFORM_FINGERPRINTS = {
    "twitter": {
        "label": "Twitter/X",
        "icon": "🐦",
        "max_dim": 4096,
        "typical_quality": (78, 88),
        "strips_exif": True,
        "typical_ratios": [16/9, 2/1, 3/2],
        "software_hints": [],
        "description": "Max 4096px, Q=78-88, genis format tercih",
    },
    "tiktok": {
        "label": "TikTok",
        "icon": "🎵",
        "max_dim": 1080,
        "typical_quality": (65, 78),
        "strips_exif": True,
        "typical_ratios": [9/16, 3/4],  # dikey video
        "software_hints": [],
        "description": "Max 1080px, Q=65-78, dikey format (9:16)",
    },
}


# ═══════════════════════════════════════════════════════════
# JPEG QUANTIZATION TABLE CIKARIMI
# ═══════════════════════════════════════════════════════════

def extract_jpeg_qtables(image) -> Optional[list]:
    """JPEG quantization tablolarini cikar (PIL veya bytes)."""
    try:
        if isinstance(image, Image.Image):
            buf = io.BytesIO()
            image.save(buf, format="JPEG", quality=95)
            buf.seek(0)
            data = buf.getvalue()
        elif isinstance(image, bytes):
            data = image
        elif isinstance(image, np.ndarray):
            img = Image.fromarray(image)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=95)
            buf.seek(0)
            data = buf.getvalue()
        else:
            return None

        # PIL quantization — orijinal dosyadan
        if isinstance(image, Image.Image) and hasattr(image, "quantization"):
            qtables = image.quantization
            if qtables:
                return [list(qtables[k]) for k in sorted(qtables.keys())]

        return None
    except Exception:
        return None


def estimate_quality_from_qtable(qtable: list) -> int:
    """Quantization tablosundan JPEG kalite tahmini."""
    if not qtable or len(qtable) < 1:
        return 85

    # Standart JPEG luminance tablosu (Q=50)
    std_lum = [
        16, 11, 10, 16, 24, 40, 51, 61,
        12, 12, 14, 19, 26, 58, 60, 55,
        14, 13, 16, 24, 40, 57, 69, 56,
        14, 17, 22, 29, 51, 87, 80, 62,
        18, 22, 37, 56, 68, 109, 103, 77,
        24, 35, 55, 64, 81, 104, 113, 92,
        49, 64, 78, 87, 103, 121, 120, 101,
        72, 92, 95, 98, 112, 100, 103, 99,
    ]

    table = qtable[0] if isinstance(qtable[0], list) else qtable
    if len(table) < 64:
        return 85

    # Q tahmini: scale factor hesapla
    total_ratio = sum(t / s for t, s in zip(table[:64], std_lum) if s > 0) / 64
    if total_ratio < 0.5:
        quality = min(100, int(100 - total_ratio * 50))
    else:
        quality = max(1, int(50 / total_ratio))

    return min(100, max(1, quality))


# ═══════════════════════════════════════════════════════════
# EXIF METADATA ANALIZI
# ═══════════════════════════════════════════════════════════

def analyze_exif(image) -> dict:
    """EXIF metadata'dan platform ipuclari cikar."""
    result = {
        "has_exif": False,
        "software": None,
        "make": None,
        "model": None,
        "datetime": None,
        "gps": False,
        "orientation": None,
        "exif_count": 0,
        "platform_hints": [],
    }

    try:
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        exif_data = image.getexif()
        if not exif_data:
            return result

        result["has_exif"] = True
        result["exif_count"] = len(exif_data)

        tag_map = {}
        for tag_id, value in exif_data.items():
            tag_name = TAGS.get(tag_id, str(tag_id))
            tag_map[tag_name] = value

        result["software"] = str(tag_map.get("Software", ""))
        result["make"] = str(tag_map.get("Make", ""))
        result["model"] = str(tag_map.get("Model", ""))
        result["datetime"] = str(tag_map.get("DateTime", ""))
        result["orientation"] = tag_map.get("Orientation")

        # GPS kontrolu
        if 34853 in exif_data:  # GPSInfo tag
            result["gps"] = True

        # Platform ipuclari
        sw = result["software"].lower()
        if "snapchat" in sw:
            result["platform_hints"].append("snapchat")

    except Exception:
        pass

    return result


# ═══════════════════════════════════════════════════════════
# BLOCKINESS OLCUMU
# ═══════════════════════════════════════════════════════════

def measure_blockiness(image) -> float:
    """8x8 DCT blok siniri sureksizligi — platform parmak izi."""
    try:
        if isinstance(image, np.ndarray):
            gray = image if len(image.shape) == 2 else np.mean(image, axis=2)
        else:
            gray = np.array(image.convert("L")).astype(np.float64)

        h, w = gray.shape
        if h < 16 or w < 16:
            return 0.0

        # Yatay ve dikey 8-piksel sinirlarindaki fark
        h_diffs, v_diffs = [], []
        for x in range(8, w - 1, 8):
            h_diffs.append(np.mean(np.abs(gray[:, x] - gray[:, x - 1])))
        for y in range(8, h - 1, 8):
            v_diffs.append(np.mean(np.abs(gray[y, :] - gray[y - 1, :])))

        if not h_diffs or not v_diffs:
            return 0.0

        return (np.mean(h_diffs) + np.mean(v_diffs)) / 2.0
    except Exception:
        return 0.0


# ═══════════════════════════════════════════════════════════
# COZUNURLUK & EN-BOY ORANI ANALIZI
# ═══════════════════════════════════════════════════════════

def analyze_dimensions(image) -> dict:
    """Cozunurluk ve en-boy orani profili."""
    if isinstance(image, np.ndarray):
        h, w = image.shape[:2]
    else:
        w, h = image.size

    ratio = w / h if h > 0 else 1.0
    max_dim = max(w, h)

    return {
        "width": w,
        "height": h,
        "max_dim": max_dim,
        "aspect_ratio": round(ratio, 3),
        "is_square": abs(ratio - 1.0) < 0.05,
        "is_portrait": ratio < 0.9,
        "is_landscape": ratio > 1.1,
        "is_vertical_video": abs(ratio - 9/16) < 0.05,
    }


# ═══════════════════════════════════════════════════════════
# ANA TESPIT MOTORU
# ═══════════════════════════════════════════════════════════

def detect_platform(image) -> dict:
    """
    Gorselin hangi sosyal medya platformundan geldigini tespit et.

    Analiz katmanlari:
        1. EXIF metadata (software tag, veri kaybi)
        2. JPEG quantization table → kalite tahmini
        3. Cozunurluk & en-boy orani profili
        4. Blockiness (8x8 DCT artefakt) deseni
        5. Dosya boyutu / piksel orani

    Returns:
        dict: {
            detected_platform: str (key),
            platform_label: str,
            platform_icon: str,
            confidence: float (0-1),
            all_scores: dict[str, float],
            evidence: list[str],
            quality_estimate: int,
            exif_info: dict,
            dimensions: dict,
        }
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    evidence = []
    scores = {k: 0.0 for k in PLATFORM_FINGERPRINTS}

    # ── 1. EXIF Analizi ──
    exif = analyze_exif(image)

    if exif["platform_hints"]:
        for hint in exif["platform_hints"]:
            if hint in scores:
                scores[hint] += 35.0
                evidence.append(f"EXIF software tag: '{exif['software']}' → {hint}")

    if not exif["has_exif"] or exif["exif_count"] < 3:
        # EXIF yok veya cok az — twitter, tiktok siler
        for p in ["twitter", "tiktok"]:
            scores[p] += 5.0
        evidence.append("EXIF verisi yok veya minimal — platform muhtemelen silmis")
    else:
        # Zengin EXIF — muhtemelen orijinal
        for p in ["tiktok"]:
            scores[p] -= 3.0
        evidence.append(f"EXIF mevcut ({exif['exif_count']} tag) — orijinal")

    # ── 2. JPEG Kalite Tahmini ──
    from core.compression import estimate_jpeg_quality
    quality_info = estimate_jpeg_quality(image)
    quality = quality_info["estimated_quality"]

    for key, fp in PLATFORM_FINGERPRINTS.items():
        q_min, q_max = fp["typical_quality"]
        if q_min <= quality <= q_max:
            scores[key] += 15.0
            evidence.append(f"JPEG Q={quality} → {fp['label']} araliginda ({q_min}-{q_max})")
        elif abs(quality - q_min) <= 5 or abs(quality - q_max) <= 5:
            scores[key] += 5.0  # yakin

    # ── 3. Cozunurluk Profili ──
    dims = analyze_dimensions(image)
    max_dim = dims["max_dim"]
    ratio = dims["aspect_ratio"]

    for key, fp in PLATFORM_FINGERPRINTS.items():
        fp_max = fp["max_dim"]

        # Max boyut eslesmesi
        if abs(max_dim - fp_max) <= 20:
            scores[key] += 12.0
            evidence.append(f"Max boyut {max_dim}px ≈ {fp['label']} limiti ({fp_max}px)")
        elif max_dim <= fp_max:
            scores[key] += 3.0

        # En-boy orani eslesmesi
        for typical_ratio in fp["typical_ratios"]:
            if abs(ratio - typical_ratio) < 0.08:
                scores[key] += 8.0
                evidence.append(
                    f"En-boy orani {ratio:.2f} ≈ {fp['label']} tipik orani ({typical_ratio:.2f})"
                )
                break

    # TikTok ozel: 1080px dikey
    if dims["is_vertical_video"] and max_dim <= 1080:
        scores["tiktok"] += 12.0
        evidence.append("Dikey format (9:16) + 1080px → TikTok")

    # ── 4. Blockiness Deseni ──
    blockiness = measure_blockiness(image)

    if blockiness > 8.0:
        # Agir sikistirma — TikTok
        scores["tiktok"] += 6.0
        evidence.append(f"Yuksek blockiness ({blockiness:.1f}) → agir JPEG sikistirma")
    elif blockiness < 3.0:
        # Dusuk sikistirma — orijinal
        evidence.append(f"Dusuk blockiness ({blockiness:.1f}) → yuksek kalite (orijinal)")

    # ── 5. Dosya Boyutu Analizi ──
    try:
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=quality)
        file_kb = len(buf.getvalue()) / 1024
        pixels = dims["width"] * dims["height"]
        bytes_per_pixel = (file_kb * 1024) / max(pixels, 1)

        if bytes_per_pixel < 0.3:
            scores["tiktok"] += 3.0
            evidence.append(f"Dusuk byte/piksel ({bytes_per_pixel:.2f}) → agresif sikistirma")
        elif bytes_per_pixel > 0.8:
            evidence.append(f"Yuksek byte/piksel ({bytes_per_pixel:.2f}) → kaliteli kaynak")
    except Exception:
        pass

    # ── Sonuc Hesapla ──
    # Negatif skorlari sifirla
    scores = {k: max(0, v) for k, v in scores.items()}

    total = sum(scores.values())
    if total > 0:
        normalized = {k: v / total for k, v in scores.items()}
    else:
        normalized = {k: 1.0 / len(scores) for k in scores}

    # En yuksek skoru bul
    best_platform = max(normalized, key=normalized.get)
    best_confidence = normalized[best_platform]

    # Cok dusuk guven → "orijinal/bilinmiyor"
    if best_confidence < 0.20 or total < 10:
        detected = "original"
        label = "Orijinal / Bilinmiyor"
        icon = "📷"
        confidence = 1.0 - best_confidence
    else:
        fp = PLATFORM_FINGERPRINTS[best_platform]
        detected = best_platform
        label = fp["label"]
        icon = fp["icon"]
        confidence = best_confidence

    return {
        "detected_platform": detected,
        "platform_label": label,
        "platform_icon": icon,
        "confidence": round(confidence, 3),
        "all_scores": {k: round(v, 3) for k, v in normalized.items()},
        "evidence": evidence,
        "quality_estimate": quality,
        "blockiness": round(blockiness, 2),
        "exif_info": exif,
        "dimensions": dims,
    }


def format_detection_result(result: dict) -> str:
    """Tespit sonucunu okunabilir Markdown formatinda dondur."""
    icon = result["platform_icon"]
    label = result["platform_label"]
    conf = result["confidence"]
    q = result["quality_estimate"]
    dims = result["dimensions"]

    lines = [
        f"### {icon} Platform Tespiti: **{label}**",
        f"**Guven:** %{conf*100:.0f} | **JPEG Q:** {q} | "
        f"**Boyut:** {dims['width']}×{dims['height']}",
        "",
        "| Platform | Skor |",
        "|----------|-----:|",
    ]

    sorted_scores = sorted(result["all_scores"].items(), key=lambda x: -x[1])
    for key, score in sorted_scores:
        fp = PLATFORM_FINGERPRINTS.get(key, {})
        p_icon = fp.get("icon", "")
        p_label = fp.get("label", key)
        bar = "█" * int(score * 20)
        lines.append(f"| {p_icon} {p_label} | {score:.1%} {bar} |")

    lines.append("")
    lines.append("**Kanıtlar:**")
    for ev in result["evidence"][:8]:
        lines.append(f"- {ev}")

    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        img = Image.open(sys.argv[1])
        result = detect_platform(img)
        print(format_detection_result(result))
    else:
        print("Kullanim: python platform_detector.py <gorsel_yolu>")
