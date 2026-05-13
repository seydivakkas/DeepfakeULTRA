"""
Deepfake Detection System v3.0 — Geliştirilmiş PDF Rapor Üretimi
Heatmap grid, DWT haritası, TTA dağılımı, model özeti dahil.
"""
import os
import io
import tempfile
from pathlib import Path
from datetime import datetime
from config import paths, VERSION

try:
    from fpdf import FPDF
    HAS_FPDF = True
except ImportError:
    HAS_FPDF = False

try:
    from PIL import Image
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


TRANSLATIONS = {
    "tr": {
        "title": "Deepfake Analiz Raporu",
        "verdict": "Karar",
        "fake_prob": "Sahte Olasılığı",
        "real_prob": "Gerçek Olasılığı",
        "confidence": "Güven Skoru",
        "tta_count": "TTA Augmentasyon",
        "tta_std": "TTA Std Sapma",
        "xai_title": "XAI Açıklanabilirlik Haritaları",
        "dwt_title": "DWT Frekans Haritası",
        "model_title": "Model Mimari Özeti",
        "generated": "Bu rapor otomatik olarak oluşturulmuştur",
    },
    "en": {
        "title": "Deepfake Analysis Report",
        "verdict": "Verdict",
        "fake_prob": "Fake Probability",
        "real_prob": "Real Probability",
        "confidence": "Confidence Score",
        "tta_count": "TTA Augmentations",
        "tta_std": "TTA Std Dev",
        "xai_title": "XAI Explainability Maps",
        "dwt_title": "DWT Frequency Map",
        "model_title": "Model Architecture Summary",
        "generated": "This report was generated automatically",
    },
}


def generate_pdf_report(
    analysis_result: dict,
    output_path: str = None,
    include_heatmaps: bool = True,
    language: str = "tr",
) -> str:
    """
    Geliştirilmiş PDF rapor oluştur.

    Rapor içeriği:
    - Başlık + tarih/saat
    - Analiz edilen görsel (watermarklı)
    - Karar + güven skoru
    - Metrik tablosu
    - XAI heatmap görselleri (2×2 grid)
    - DWT frekans haritası
    - TTA dağılım grafiği
    - Model mimari özeti
    """
    if not HAS_FPDF:
        print("⚠️ fpdf2 yüklü değil: pip install fpdf2")
        return None

    paths.ensure_dirs()
    t = TRANSLATIONS.get(language, TRANSLATIONS["tr"])

    if output_path is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = str(paths.REPORTS_DIR / f"deepfake_report_{ts}.pdf")

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # ── Başlık ──
    pdf.set_font("Helvetica", "B", 20)
    pdf.cell(0, 15, t["title"], ln=True, align="C")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 8, f"v{VERSION} | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align="C")
    pdf.ln(8)

    # ── Verdict kutusu ──
    verdict = analysis_result.get("verdict", "UNKNOWN")
    color = (231, 76, 60) if verdict == "FAKE" else (46, 204, 113)
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_fill_color(*color)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 14, f"  {t['verdict']}: {verdict}", ln=True, fill=True)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(5)

    # ── Analiz edilen görsel ──
    _add_pil_image(pdf, analysis_result.get("watermarked_image") or analysis_result.get("original_image"), w=70)

    # ── Metrik tablosu ──
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 10, "Metrics", ln=True)
    pdf.set_font("Helvetica", "", 10)

    metrics = [
        (t["fake_prob"], f"{analysis_result.get('fake_prob', 0):.4f}"),
        (t["real_prob"], f"{analysis_result.get('real_prob', 0):.4f}"),
        (t["confidence"], f"{analysis_result.get('confidence', 0):.4f}"),
        ("GradCAM++ Score", f"{analysis_result.get('gradcam_score', 0):.4f}"),
        ("Counterfactual Prob", f"{analysis_result.get('counterfactual_prob', 0):.4f}"),
        (t["tta_count"], str(len(analysis_result.get("tta_individual", [])))),
        (t["tta_std"], f"{analysis_result.get('tta_std', 0):.4f}"),
    ]
    for key, val in metrics:
        pdf.cell(80, 7, key, border=1)
        pdf.cell(0, 7, val, border=1, ln=True)

    pdf.ln(5)

    # ── XAI Heatmap Grid (2×2) ──
    if include_heatmaps:
        heatmaps = analysis_result.get("heatmaps", {})
        if heatmaps:
            pdf.set_font("Helvetica", "B", 12)
            pdf.cell(0, 10, t["xai_title"], ln=True)
            pdf.ln(2)

            hm_items = list(heatmaps.items())
            for i in range(0, len(hm_items), 2):
                for j in range(2):
                    if i + j < len(hm_items):
                        name, img = hm_items[i + j]
                        pdf.set_font("Helvetica", "", 8)
                        pdf.cell(90, 6, name.upper(), ln=False)
                pdf.ln(6)
                for j in range(2):
                    if i + j < len(hm_items):
                        _, img = hm_items[i + j]
                        x_pos = pdf.get_x() + (j * 95)
                        _add_pil_image(pdf, img, w=80, x=10 + (j * 95))
                pdf.ln(2)

    # ── DWT frekans haritası ──
    dwt_map = analysis_result.get("dwt_map")
    if dwt_map:
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 10, t["dwt_title"], ln=True)
        _add_pil_image(pdf, dwt_map, w=100)

        # Füzyon ağırlıkları
        fw = analysis_result.get("fusion_weights", {})
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 8, f"RGB: {fw.get('rgb', 0):.2f}%  |  Freq: {fw.get('freq', 0):.2f}%  |  Geo: {fw.get('geo', 0):.2f}%", ln=True)

    # ── TTA Dağılım Grafiği ──
    tta_ind = analysis_result.get("tta_individual", [])
    if tta_ind and HAS_MPL:
        pdf.ln(5)
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 10, "TTA Distribution", ln=True)

        fig, ax = plt.subplots(figsize=(5, 2.5), dpi=100)
        ax.bar(range(len(tta_ind)), tta_ind, color="#06B6D4", alpha=0.8)
        ax.axhline(y=0.5, color="red", linestyle="--", linewidth=1, label="Threshold")
        ax.set_xlabel("Augmentation")
        ax.set_ylabel("Fake Prob")
        ax.legend()
        fig.tight_layout()

        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        fig.savefig(tmp.name)
        plt.close(fig)
        pdf.image(tmp.name, w=120)
        os.unlink(tmp.name)

    # ── Model Mimari Özeti ──
    pdf.ln(5)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 10, t["model_title"], ln=True)
    pdf.set_font("Helvetica", "", 9)
    pdf.multi_cell(0, 6,
        "DualPathDeepfakeDetector\n"
        "RGB: MobileNetV3-Large (pretrained)\n"
        "Freq: MobileNetV3-Large (DWT 12-ch)\n"
        "Mesh: FaceMeshMLP (468x3 landmarks)\n"
        "Fusion: SE-based LearnableFusion\n"
        "Temporal: Stacked BiLSTM + Multi-Head Attention\n"
        "Classifier: FC → ReLU → Dropout → FC → Sigmoid"
    )

    # ── Footer ──
    pdf.ln(10)
    pdf.set_font("Helvetica", "I", 8)
    pdf.cell(0, 6, t["generated"], ln=True, align="C")

    pdf.output(output_path)
    return output_path


def _add_pil_image(pdf, img, w=80, x=None):
    """PIL Image'ı PDF'e ekle."""
    if img is None:
        return
    try:
        if isinstance(img, Image.Image):
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            img.save(tmp.name)
            if x is not None:
                pdf.image(tmp.name, x=x, w=w)
            else:
                pdf.image(tmp.name, w=w)
            os.unlink(tmp.name)
    except Exception:
        pass
