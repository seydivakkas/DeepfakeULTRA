"""
Platform Detection — JPEG Forensik Analizi ile Sosyal Medya Tespiti.

Görselin hangi platformdan geçtiğini JPEG artefaktlarından tespit eder:
  1. JPEG Quantization Table fingerprinting
  2. Boyut profili analizi (max_dim eşleştirme)
  3. EXIF metadata analizi (strip/modify deseni)
  4. Kalite tahmini (Q-factor)
  5. Çift JPEG sıkıştırma tespiti
"""

import io
import struct
import numpy as np
from PIL import Image, ExifTags
from pathlib import Path
from typing import Optional


# ═══════════════════════════════════════════════════════════
# PLATFORM PROFİLLERİ
# ═══════════════════════════════════════════════════════════
PLATFORM_PROFILES = {
    "twitter": {
        "label": "Twitter/X",
        "icon": "🐦",
        "max_dims": [(4096, 4096), (1280, 1280)],
        "quality_range": (78, 90),
        "strips_exif": True,
        "color": "#1DA1F2",
    },
    "tiktok": {
        "label": "TikTok",
        "icon": "🎵",
        "max_dims": [(1080, 1920), (720, 1280)],
        "quality_range": (62, 78),
        "strips_exif": True,
        "color": "#69C9D0",
    },
}


# ═══════════════════════════════════════════════════════════
# JPEG KALİTE TAHMİNİ
# ═══════════════════════════════════════════════════════════
# Standart JPEG luminans quantization tablosu (Q=50 baseline)
STANDARD_LUMINANCE_QT = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99],
], dtype=np.float64)


def estimate_jpeg_quality(image_path: str = None, image: Image.Image = None) -> dict:
    """
    JPEG görselin tahmini kalite faktörünü (Q) hesaplar.
    Quantization tablosundan reverse-engineering ile Q değeri bulur.

    Returns:
        dict: quality (int), has_qt (bool), qt_luminance (np.array)
    """
    result = {"quality": None, "has_qt": False, "qt_luminance": None}

    try:
        if image_path:
            img = Image.open(image_path)
        elif image:
            img = image
        else:
            return result

        # JPEG quantization tablosunu oku
        qt = getattr(img, 'quantization', None)
        if qt is None:
            return result

        result["has_qt"] = True

        # İlk tablo (luminans)
        if 0 in qt:
            qt_lum = np.array(qt[0], dtype=np.float64).reshape(8, 8)
            result["qt_luminance"] = qt_lum

            # Q-factor tahmini: standart tablo ile karşılaştırma
            # Q = 50 → standart tablo, Q > 50 → küçük değerler, Q < 50 → büyük değerler
            ratios = qt_lum / STANDARD_LUMINANCE_QT
            avg_ratio = np.mean(ratios)

            if avg_ratio < 1.0:
                # Q > 50 durumu
                quality = int(50 + 50 * (1 - avg_ratio))
            else:
                # Q ≤ 50 durumu
                quality = int(50 / avg_ratio)

            result["quality"] = max(1, min(100, quality))

    except Exception:
        pass

    return result


# ═══════════════════════════════════════════════════════════
# EXIF ANALİZİ
# ═══════════════════════════════════════════════════════════
def analyze_exif(image: Image.Image) -> dict:
    """
    EXIF metadata analizi — hangi bilgiler mevcut, hangileri silinmiş?

    Returns:
        dict: has_exif, camera_make, camera_model, software, gps,
              exif_tags_count, stripped_indicators
    """
    result = {
        "has_exif": False,
        "camera_make": None,
        "camera_model": None,
        "software": None,
        "has_gps": False,
        "exif_tags_count": 0,
        "stripped_indicators": [],
    }

    try:
        exif_data = image.getexif()
        if not exif_data:
            result["stripped_indicators"].append("EXIF tamamen silinmiş")
            return result

        result["has_exif"] = True
        result["exif_tags_count"] = len(exif_data)

        tag_map = {v: k for k, v in ExifTags.TAGS.items()}

        # Temel bilgiler
        make_tag = tag_map.get("Make")
        model_tag = tag_map.get("Model")
        software_tag = tag_map.get("Software")

        if make_tag and make_tag in exif_data:
            result["camera_make"] = str(exif_data[make_tag])
        if model_tag and model_tag in exif_data:
            result["camera_model"] = str(exif_data[model_tag])
        if software_tag and software_tag in exif_data:
            result["software"] = str(exif_data[software_tag])

        # GPS kontrolü
        gps_tag = tag_map.get("GPSInfo")
        if gps_tag and gps_tag in exif_data:
            result["has_gps"] = True

        # Strip göstergeleri
        if result["exif_tags_count"] < 5:
            result["stripped_indicators"].append("Çok az EXIF etiketi (kısmi silme)")
        if not result["camera_make"] and not result["camera_model"]:
            result["stripped_indicators"].append("Kamera bilgisi yok")

    except Exception:
        result["stripped_indicators"].append("EXIF okunamadı")

    return result


# ═══════════════════════════════════════════════════════════
# ÇİFT JPEG SIKIŞTIRMA TESPİTİ
# ═══════════════════════════════════════════════════════════
def detect_double_jpeg(image: Image.Image) -> dict:
    """
    Çift JPEG sıkıştırma tespiti — DCT katsayılarının histogram
    analizi ile görselin birden fazla kez sıkıştırılıp sıkıştırılmadığını belirler.

    Returns:
        dict: is_double_compressed (bool), confidence (float), analysis (str)
    """
    result = {
        "is_double_compressed": False,
        "confidence": 0.0,
        "analysis": "",
    }

    try:
        img_np = np.array(image.convert("L"), dtype=np.float64)

        # 8x8 blok DCT katsayı histogramı
        h, w = img_np.shape
        block_h, block_w = h // 8, w // 8
        dct_coeffs = []

        for i in range(min(block_h, 50)):  # İlk 50 blok yeterli
            for j in range(min(block_w, 50)):
                block = img_np[i*8:(i+1)*8, j*8:(j+1)*8]
                if block.shape == (8, 8):
                    # Basit DCT yaklaşımı (fark tabanlı)
                    dct_coeffs.extend(block.flatten() - block.mean())

        if not dct_coeffs:
            return result

        coeffs = np.array(dct_coeffs)
        hist, bin_edges = np.histogram(coeffs, bins=100, range=(-128, 128))

        # Çift sıkıştırma belirtisi: histogram periyodikliği
        # Tek sıkıştırmada düzgün dağılım, çift sıkıştırmada periyodik tepeler
        hist_normalized = hist / (hist.sum() + 1e-8)
        fft_hist = np.abs(np.fft.fft(hist_normalized))
        # DC bileşeni hariç
        fft_magnitudes = fft_hist[1:len(fft_hist)//2]

        if len(fft_magnitudes) > 0:
            peak_ratio = np.max(fft_magnitudes) / (np.mean(fft_magnitudes) + 1e-8)

            if peak_ratio > 5.0:
                result["is_double_compressed"] = True
                result["confidence"] = min(1.0, (peak_ratio - 5.0) / 10.0)
                result["analysis"] = f"Çift JPEG sıkıştırma tespit edildi (güven: {result['confidence']:.0%})"
            else:
                result["analysis"] = "Tek sıkıştırma (normal)"

    except Exception as e:
        result["analysis"] = f"Analiz hatası: {e}"

    return result


# ═══════════════════════════════════════════════════════════
# ANA FONKSİYON — Platform Tespiti
# ═══════════════════════════════════════════════════════════
def detect_platform(
    image: Image.Image,
    image_path: str = None,
) -> dict:
    """
    Görselin hangi sosyal medya platformundan geçtiğini tahmin eder.

    Analiz yöntemleri:
        1. JPEG kalite tahmini → platform kalite aralığı eşleştirmesi
        2. Boyut profili → platform max_dim eşleştirmesi
        3. EXIF strip durumu → platform EXIF politikası
        4. Çift JPEG sıkıştırma → paylaşım geçmişi

    Returns:
        dict: platform, confidence, scores, quality_info, exif_info, double_jpeg
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    image = image.convert("RGB")

    w, h = image.size

    # 1. JPEG Kalite Analizi
    quality_info = estimate_jpeg_quality(image_path=image_path, image=image)
    estimated_q = quality_info.get("quality")

    # 2. EXIF Analizi
    exif_info = analyze_exif(image)

    # 3. Çift Sıkıştırma
    double_jpeg = detect_double_jpeg(image)

    # 4. Platform skorlama
    scores = {}
    for platform_id, profile in PLATFORM_PROFILES.items():
        score = 0.0
        reasons = []

        # Kalite eşleştirme (en ağırlıklı sinyal)
        if estimated_q is not None:
            q_min, q_max = profile["quality_range"]
            if q_min <= estimated_q <= q_max:
                # Aralık içindeyse yüksek skor
                center = (q_min + q_max) / 2
                distance = abs(estimated_q - center) / ((q_max - q_min) / 2 + 1e-8)
                q_score = max(0, 1 - distance) * 0.4
                score += q_score
                reasons.append(f"Q={estimated_q} ✓ ({q_min}-{q_max})")
            else:
                # Aralık dışında ama yakınsa az skor
                if estimated_q < q_min:
                    dist = q_min - estimated_q
                else:
                    dist = estimated_q - q_max
                if dist < 10:
                    score += 0.1
                reasons.append(f"Q={estimated_q} ✗")

        # Boyut eşleştirme
        max_dim = max(w, h)
        for mw, mh in profile["max_dims"]:
            target_max = max(mw, mh)
            if abs(max_dim - target_max) < 50:
                score += 0.25
                reasons.append(f"Boyut ~{target_max}px ✓")
                break
            elif max_dim <= target_max:
                score += 0.05
                break

        # EXIF eşleştirme
        if profile["strips_exif"] and not exif_info["has_exif"]:
            score += 0.2
            reasons.append("EXIF silinmiş ✓")
        elif not profile["strips_exif"] and exif_info["has_exif"]:
            score += 0.15
            reasons.append("EXIF mevcut ✓")

        scores[platform_id] = {
            "score": score,
            "label": profile["label"],
            "icon": profile["icon"],
            "color": profile["color"],
            "reasons": reasons,
        }

    # En yüksek skorlu platform
    best_platform = max(scores, key=lambda k: scores[k]["score"])
    best_score = scores[best_platform]["score"]

    # Güven seviyesi belirleme
    if best_score >= 0.6:
        confidence = "Yüksek"
    elif best_score >= 0.35:
        confidence = "Orta"
    elif best_score >= 0.15:
        confidence = "Düşük"
    else:
        best_platform = "original"
        confidence = "Tespit edilemedi"

    return {
        "platform": best_platform,
        "platform_label": scores.get(best_platform, {}).get("label", "Orijinal"),
        "platform_icon": scores.get(best_platform, {}).get("icon", "📷"),
        "confidence": confidence,
        "best_score": best_score,
        "scores": scores,
        "quality_info": {
            "estimated_quality": estimated_q,
            "has_quantization_table": quality_info["has_qt"],
        },
        "exif_info": exif_info,
        "double_jpeg": double_jpeg,
        "image_dimensions": {"width": w, "height": h},
    }


def format_platform_report(result: dict) -> str:
    """Platform tespiti sonucunu markdown rapor olarak formatla."""
    md = "### 📱 Platform Tespit Raporu\n\n"

    # Ana sonuç
    icon = result["platform_icon"]
    label = result["platform_label"]
    conf = result["confidence"]
    md += f"**Tespit Edilen Platform:** {icon} **{label}** ({conf} güven)\n\n"

    # JPEG Kalite
    q = result["quality_info"]["estimated_quality"]
    if q:
        md += f"**JPEG Kalite:** Q={q}\n\n"

    # Boyut
    dims = result["image_dimensions"]
    md += f"**Boyut:** {dims['width']}×{dims['height']}px\n\n"

    # Çift sıkıştırma
    dj = result["double_jpeg"]
    if dj["is_double_compressed"]:
        md += f"**⚠️ Çift Sıkıştırma:** {dj['analysis']}\n\n"

    # EXIF
    exif = result["exif_info"]
    if exif["has_exif"]:
        md += "**EXIF:** Mevcut"
        if exif["camera_make"]:
            md += f" — {exif['camera_make']}"
        if exif["camera_model"]:
            md += f" {exif['camera_model']}"
        md += "\n\n"
    else:
        md += "**EXIF:** Silinmiş (platform paylaşımı göstergesi)\n\n"

    # Platform skorları tablosu
    md += "| Platform | Skor | Eşleşme |\n|---|---|---|\n"
    sorted_scores = sorted(result["scores"].items(), key=lambda x: x[1]["score"], reverse=True)
    for pid, data in sorted_scores:
        bar_len = int(data["score"] * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        reasons = ", ".join(data["reasons"]) if data["reasons"] else "—"
        md += f"| {data['icon']} {data['label']} | {bar} {data['score']:.2f} | {reasons} |\n"

    return md
