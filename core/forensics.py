"""
Klasik Goruntu Forensik Analiz Modulu — ELA + Noise Analysis.

ELA (Error Level Analysis):
    Manipule edilmis bolgeler farkli JPEG kayip desenleri gosterir.
    Uniform ELA = orijinal, yuksek ELA bolgesi = olasi manipulasyon.

Noise Analysis:
    Farkli kaynaktan gelen bolgeler (face swap vb.) farkli gurultu
    dagilimi gosterir. Tutarsiz bolgeler kirmizi ile isaretlenir.
"""
import io
import numpy as np
from PIL import Image, ImageFilter

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


def generate_ela(image, quality: int = 90, amplification: int = 18):
    """
    Error Level Analysis (ELA) haritasi olustur.

    Args:
        image: PIL Image (RGB)
        quality: JPEG yeniden kaydetme kalitesi (70-95 arasi ideal)
        amplification: Fark amplifikasyon katsayisi (15-20 arasi ideal)

    Returns:
        PIL Image — ELA heatmap (renk kodlu)
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    image = image.convert("RGB")

    # 1. Belirli JPEG kalitesinde yeniden kaydet
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    recompressed = Image.open(buffer).convert("RGB")

    # 2. Orijinal ile yeniden kaydedilmis arasindaki farki hesapla
    original_np = np.array(image, dtype=np.float32)
    recomp_np = np.array(recompressed, dtype=np.float32)
    diff = np.abs(original_np - recomp_np)

    # 3. Farki amplify et
    amplified = np.clip(diff * amplification, 0, 255).astype(np.uint8)

    # 4. Renk kodlu heatmap olustur
    if HAS_CV2:
        gray = cv2.cvtColor(amplified, cv2.COLOR_RGB2GRAY)
        heatmap = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    else:
        # Fallback: matplotlib colormap
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            gray = np.mean(amplified, axis=2).astype(np.uint8)
            heatmap = (plt.cm.jet(gray / 255.0)[:, :, :3] * 255).astype(np.uint8)
        except ImportError:
            heatmap = amplified

    return Image.fromarray(heatmap)


def generate_noise_map(image, blur_kernel: int = 5, block_size: int = 32):
    """
    Gurultu tutarlilik haritasi olustur.

    Farkli kaynaklardan gelen bolgeler farkli gurultu seviyeleri
    gosterir. Face swap manipulasyonlarinda belirgin tutarsizlik olusur.

    Args:
        image: PIL Image (RGB)
        blur_kernel: Gaussian blur cekirdek boyutu (tek sayi)
        block_size: Bolge bazli analiz icin blok boyutu (piksel)

    Returns:
        PIL Image — Gurultu tutarlilik haritasi (renk kodlu)
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    image = image.convert("RGB")
    img_np = np.array(image, dtype=np.float32)

    # 1. Gaussian blur ile duzlestirmis versiyon
    if HAS_CV2:
        k = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
        smoothed = cv2.GaussianBlur(img_np, (k, k), 0)
    else:
        smoothed = np.array(
            image.filter(ImageFilter.GaussianBlur(radius=blur_kernel // 2)),
            dtype=np.float32,
        )

    # 2. Gurultu haritasi: orijinal - blur
    noise = img_np - smoothed

    # 3. Bolge bazli gurultu std sapma hesapla
    h, w = noise.shape[:2]
    # Gri tonlama gurultu buyuklugu
    noise_magnitude = np.sqrt(np.sum(noise ** 2, axis=2))

    # Blok bazli std sapma haritasi
    block_h = max(1, h // block_size)
    block_w = max(1, w // block_size)
    std_map = np.zeros((block_h, block_w), dtype=np.float32)

    for by in range(block_h):
        for bx in range(block_w):
            y0, y1 = by * block_size, min((by + 1) * block_size, h)
            x0, x1 = bx * block_size, min((bx + 1) * block_size, w)
            block = noise_magnitude[y0:y1, x0:x1]
            std_map[by, bx] = np.std(block) if block.size > 0 else 0

    # 4. Normalize et (0-255 arasi)
    if std_map.max() > std_map.min():
        norm_map = ((std_map - std_map.min()) / (std_map.max() - std_map.min()) * 255)
    else:
        norm_map = np.zeros_like(std_map)
    norm_map = norm_map.astype(np.uint8)

    # Tam boyuta geri olceklendir
    if HAS_CV2:
        full_map = cv2.resize(norm_map, (w, h), interpolation=cv2.INTER_LINEAR)
        heatmap = cv2.applyColorMap(full_map, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    else:
        # PIL resize
        resized = Image.fromarray(norm_map).resize((w, h), Image.BILINEAR)
        full_map = np.array(resized)
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            heatmap = (plt.cm.jet(full_map / 255.0)[:, :, :3] * 255).astype(np.uint8)
        except ImportError:
            heatmap = np.stack([full_map] * 3, axis=2)

    return Image.fromarray(heatmap)


def analyze_forensics(image):
    """
    Tam forensik analiz — ELA + Noise haritalarini birlikte dondurur.

    Args:
        image: PIL Image (RGB)

    Returns:
        dict: {
            "ela_map": PIL Image,
            "noise_map": PIL Image,
            "ela_score": float (0-1 arasi, yuksek = suphe),
            "noise_score": float (0-1 arasi, yuksek = tutarsizlik),
        }
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    ela_map = generate_ela(image)
    noise_map = generate_noise_map(image)

    # ELA skor: ortalama ELA yogunlugu (normalize)
    ela_np = np.array(ela_map, dtype=np.float32)
    ela_score = float(np.mean(ela_np) / 255.0)

    # Noise skor: gurultu varyans tutarsizligi
    noise_np = np.array(noise_map, dtype=np.float32)
    noise_score = float(np.std(noise_np) / 128.0)  # Normalize (0-1)
    noise_score = min(noise_score, 1.0)

    return {
        "ela_map": ela_map,
        "noise_map": noise_map,
        "ela_score": round(ela_score, 4),
        "noise_score": round(noise_score, 4),
    }


# ═══════════════════════════════════════════════════════════
# FORENSIK KONSENSÜS SKORU — Model + ELA + Noise Birleşimi
# ═══════════════════════════════════════════════════════════

# ELA ve Noise için ampirik eşikler (283K dataset bazlı kalibrasyon)
ELA_THRESHOLDS = {
    "clean":     0.15,   # ELA < 0.15 → temiz görsel
    "suspect":   0.25,   # 0.15-0.25 → şüpheli bölge
    "tampered":  0.40,   # > 0.40 → yüksek manipülasyon
}

NOISE_THRESHOLDS = {
    "consistent":   0.30,  # < 0.30 → tutarlı gürültü
    "suspect":      0.45,  # 0.30-0.45 → kısmi tutarsızlık
    "inconsistent": 0.60,  # > 0.60 → ciddi tutarsızlık
}


def compute_forensic_consensus(
    model_fake_prob: float,
    ela_score: float,
    noise_score: float,
    model_weight: float = 0.6,
    ela_weight: float = 0.25,
    noise_weight: float = 0.15,
) -> dict:
    """
    Model tahmini + ELA + Noise skorlarını birleştirerek
    tek bir forensik konsensüs kararı üretir.

    Ağırlıklar:
        Model: %60 (ana karar verici)
        ELA:   %25 (manipülasyon kanıtı)
        Noise: %15 (kaynak tutarlılığı)

    Returns:
        dict: consensus_score, verdict, confidence, consistency, explanation
    """
    # ELA skorunu 0-1 FAKE olasılığına dönüştür
    if ela_score < ELA_THRESHOLDS["clean"]:
        ela_fake_signal = ela_score / ELA_THRESHOLDS["clean"] * 0.3
    elif ela_score < ELA_THRESHOLDS["suspect"]:
        ela_fake_signal = 0.3 + (ela_score - ELA_THRESHOLDS["clean"]) / \
            (ELA_THRESHOLDS["suspect"] - ELA_THRESHOLDS["clean"]) * 0.3
    else:
        ela_fake_signal = 0.6 + min(0.4, (ela_score - ELA_THRESHOLDS["suspect"]) / 0.3 * 0.4)

    # Noise skorunu 0-1 FAKE olasılığına dönüştür
    if noise_score < NOISE_THRESHOLDS["consistent"]:
        noise_fake_signal = noise_score / NOISE_THRESHOLDS["consistent"] * 0.3
    elif noise_score < NOISE_THRESHOLDS["suspect"]:
        noise_fake_signal = 0.3 + (noise_score - NOISE_THRESHOLDS["consistent"]) / \
            (NOISE_THRESHOLDS["suspect"] - NOISE_THRESHOLDS["consistent"]) * 0.3
    else:
        noise_fake_signal = 0.6 + min(0.4, (noise_score - NOISE_THRESHOLDS["suspect"]) / 0.3 * 0.4)

    # Ağırlıklı birleşim
    consensus = (
        model_weight * model_fake_prob +
        ela_weight * ela_fake_signal +
        noise_weight * noise_fake_signal
    )
    consensus = max(0.0, min(1.0, consensus))

    # Tutarlılık kontrolü — model ile forensik arasındaki uyum
    model_says_fake = model_fake_prob > 0.5
    ela_says_fake = ela_score > ELA_THRESHOLDS["suspect"]
    noise_says_fake = noise_score > NOISE_THRESHOLDS["suspect"]

    forensic_votes = sum([model_says_fake, ela_says_fake, noise_says_fake])

    if forensic_votes == 3:
        consistency = "full"
        consistency_label = "✅ Tam Uyum"
        consistency_detail = "Model, ELA ve Noise aynı yönde: FAKE"
    elif forensic_votes == 0:
        consistency = "full"
        consistency_label = "✅ Tam Uyum"
        consistency_detail = "Model, ELA ve Noise aynı yönde: REAL"
    elif forensic_votes == 2:
        consistency = "partial"
        # Hangi sinyal farklı?
        dissenter = []
        if not model_says_fake: dissenter.append("Model")
        if not ela_says_fake: dissenter.append("ELA")
        if not noise_says_fake: dissenter.append("Noise")
        consistency_label = f"⚠️ Kısmi Uyum ({', '.join(dissenter)} farklı)"
        consistency_detail = f"2/3 sinyal FAKE, ancak {', '.join(dissenter)} REAL diyor"
    else:  # forensic_votes == 1
        consistency = "partial"
        supporter = []
        if model_says_fake: supporter.append("Model")
        if ela_says_fake: supporter.append("ELA")
        if noise_says_fake: supporter.append("Noise")
        consistency_label = f"⚠️ Kısmi Uyum (sadece {', '.join(supporter)} FAKE)"
        consistency_detail = f"2/3 sinyal REAL, ancak {', '.join(supporter)} FAKE diyor"

    # Karar ve güven
    if consensus >= 0.65:
        verdict = "FAKE"
        confidence = "Yüksek" if consensus >= 0.80 else "Orta"
    elif consensus <= 0.35:
        verdict = "REAL"
        confidence = "Yüksek" if consensus <= 0.20 else "Orta"
    else:
        verdict = "UNCERTAIN"
        confidence = "Düşük"

    return {
        "consensus_score": round(consensus, 4),
        "verdict": verdict,
        "confidence": confidence,
        "consistency": consistency,
        "consistency_label": consistency_label,
        "consistency_detail": consistency_detail,
        "signals": {
            "model": {"value": round(model_fake_prob, 4), "says_fake": model_says_fake},
            "ela":   {"value": round(ela_fake_signal, 4), "raw": round(ela_score, 4), "says_fake": ela_says_fake},
            "noise": {"value": round(noise_fake_signal, 4), "raw": round(noise_score, 4), "says_fake": noise_says_fake},
        },
        "weights": {"model": model_weight, "ela": ela_weight, "noise": noise_weight},
    }


def format_consensus_report(consensus: dict) -> str:
    """Forensik konsensüs sonucunu markdown rapor olarak formatla."""
    c = consensus
    s = c["signals"]

    md = "### 🔬 Forensik Konsensüs Raporu\n\n"

    # Ana karar
    icon = {"FAKE": "🔴", "REAL": "🟢", "UNCERTAIN": "🟡"}.get(c["verdict"], "⚪")
    md += f"**{icon} Konsensüs Kararı:** {c['verdict']} "
    md += f"(Skor: {c['consensus_score']:.4f}, Güven: {c['confidence']})\n\n"

    # Tutarlılık
    md += f"**{c['consistency_label']}**\n"
    md += f"> {c['consistency_detail']}\n\n"

    # Sinyal tablosu
    md += "| Sinyal | Ağırlık | Skor | Ham Değer | Karar |\n|---|---|---|---|---|\n"

    m = s["model"]
    model_level = "🔴 Yüksek" if m["value"] > 0.65 else "🟡 Orta" if m["value"] > 0.35 else "🟢 Düşük"
    md += f"| 🧠 Model (DualPath) | {c['weights']['model']:.0%} | "
    md += f"{m['value']:.4f} | {m['value']:.4f} ({model_level}) | {'FAKE' if m['says_fake'] else 'REAL'} |\n"

    e = s["ela"]
    ela_level = "🔴 Yüksek" if e["raw"] > 0.25 else "🟡 Orta" if e["raw"] > 0.15 else "🟢 Düşük"
    md += f"| 🔍 ELA | {c['weights']['ela']:.0%} | "
    md += f"{e['value']:.4f} | {e['raw']:.4f} ({ela_level}) | {'FAKE' if e['says_fake'] else 'REAL'} |\n"

    n = s["noise"]
    noise_level = "🔴 Tutarsız" if n["raw"] > 0.45 else "🟡 Kısmi" if n["raw"] > 0.30 else "🟢 Tutarlı"
    md += f"| 📊 Noise | {c['weights']['noise']:.0%} | "
    md += f"{n['value']:.4f} | {n['raw']:.4f} ({noise_level}) | {'FAKE' if n['says_fake'] else 'REAL'} |\n"

    return md

