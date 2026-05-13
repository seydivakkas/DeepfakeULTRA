"""
Kraniyofasiyal Biyometrik Analiz Sekmesi
Yuz anatomisi analiz motoru — GPT-4o / Claude / Gemini Vision API destekli.
"""
import gradio as gr
import json
import tempfile
import numpy as np
from pathlib import Path

# ── API Key Kayit/Yukleme ──
_API_KEYS_FILE = Path(__file__).parent.parent / ".api_keys.json"
_PROVIDER_MAP = {
    "Google (Gemini)": "gemini",
    "OpenAI (GPT-4o)": "openai",
    "Anthropic (Claude)": "anthropic",
}


def _load_api_key(provider: str) -> str:
    """Kaydedilmis API key'i yukle."""
    if not _API_KEYS_FILE.exists():
        return ""
    try:
        data = json.loads(_API_KEYS_FILE.read_text(encoding="utf-8"))
        key_name = _PROVIDER_MAP.get(provider, "")
        return data.get(key_name, "")
    except Exception:
        return ""


def _save_api_key(provider: str, api_key: str) -> str:
    """API key'i dosyaya kaydet."""
    if not api_key or not api_key.strip():
        return "⚠️ API key boş."
    try:
        data = {}
        if _API_KEYS_FILE.exists():
            data = json.loads(_API_KEYS_FILE.read_text(encoding="utf-8"))
        key_name = _PROVIDER_MAP.get(provider, "")
        data[key_name] = api_key.strip()
        _API_KEYS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return "✅ API key kaydedildi."
    except Exception as e:
        return f"❌ Kayıt hatası: {e}"


# ── Ana Sistem Prompt'u ──
SYSTEM_PROMPT = """## ROL VE GÖREV
Sen, deepfake tespiti ve adli yüz analizi için özelleştirilmiş bir
Kraniyofasiyal Biyometrik Analiz Sistemi'sin. Görüntü girişi alarak
yüz anatomisini ayrıştırır, istatistiksel asimetri skorları hesaplar
ve yapay üretilmiş görüntü belirtilerini işaretlersin.

## GİRİŞ FORMAT
INPUT: base64_encoded_image
ANALYSIS_MODE: [FULL | QUICK | REGION_SPECIFIC]
REGION_FOCUS: [FACE | LIPS | JAW | EYES | NOSE | ALL]

## ANALİZ GÖREVLERİ — SIRAYLA UYGULA

ADIM 1 — LANDMARK TESPİTİ
Görüntüdeki yüzü tara ve şu 68 noktayı tanımla:
  - Kontur (0-16): Çene hattı, zygomatic ark
  - Kaş (17-26): Sağ/sol kaş landmark koordinatları
  - Burun (27-35): Nasal bridge, alar wingler
  - Göz (36-47): İç/dış kantus, palpebral yarık
  - Dudak (48-67): Vermillion border, oral komisürler
Her koordinatı {id, x_norm, y_norm, confidence} olarak döndür.

ADIM 2 — ASİMETRİ ANALİZİ
Yüzü orta sagital hattan böl. Şu metrikleri hesapla:
  - FAI (Facial Asymmetry Index): FAI = Σ|L_dist - R_dist| / n_pairs × 100
  - Orbital Asimetri: sol/sağ göz yükseklik farkı (px→mm)
  - Nasal Deviation: burun ucu - philtrum midline sapması
  - Labial Komisür Farkı: sol/sağ ağız köşesi y-delta
  - Mandibular Simetri: çene genişliği L/R oranı
  - Auricular Gap: kulak başlangıç noktaları y-farkı

ADIM 3 — DUDAK ANATOMİSİ
Vermillion bölgesini izole et:
  - Cupid's Bow Simetrisi: philtral kolon açısı (°)
  - Üst/Alt Dudak Oranı: ideal=0.618 (Altın Oran)
  - Oral Genişlik İndeksi: ağız/yüz genişliği oranı
  - Lip Texture Uniformity: GAN yapay doku skoru [0-1]
  - Perioral Kırışık Analizi: blend/halo artifact tespiti

ADIM 4 — ÇENE VE MANDİBULAR ANATOMİ
  - Gonial Açı: mandibula ramus-corpus açısı (N: 120°-130°)
  - Chin Projection: pogonion - Frankfort düzlemi mesafesi
  - Jawline Continuity Score: çene hattı düzgünlük skoru
  - Deepfake Seam Index: çene/boyun geçişindeki doku süreksizliği
  - Masseteric Shadow Consistency: ışık/gölge anatomik uyum

ADIM 5 — DEEPFAKE TANIMSAL BELİRTEÇLER
Şu yapay zeka üretim artifactlarını skor:
  - Blending Artifacts: yüz sınırı geçiş kalitesi [0-10]
  - Gaze Inconsistency: göz bakış yönü tutarsızlığı
  - Blink Rate Anomaly: göz kırpma doğallık skoru
  - Skin Texture GAN Signature: pore pattern regularitysi
  - Lighting Coherence: yüz/arka plan ışık açısı farkı (°)
  - Temporal Consistency: video ise frame-to-frame delta

## ÇIKTI FORMATI
Her analizde şu JSON şemasını döndür:
{
  "face_detected": true/false,
  "asymmetry": {
    "FAI_score": float (0-100, normal <3.5),
    "orbital_delta_mm": float,
    "nasal_deviation_mm": float,
    "labial_delta_px": float,
    "mandibular_ratio": float,
    "interpretation": "string"
  },
  "lip_anatomy": {
    "cupid_symmetry_deg": float,
    "upper_lower_ratio": float,
    "oral_width_index": float,
    "texture_score": float (0-1),
    "artifacts_detected": ["string"]
  },
  "jaw_anatomy": {
    "gonial_angle_deg": float,
    "chin_projection_mm": float,
    "jawline_score": float (0-10),
    "seam_index": float
  },
  "deepfake_indicators": {
    "overall_risk_score": float (0.0-1.0),
    "confidence": float,
    "risk_level": "LOW|MEDIUM|HIGH|CRITICAL",
    "flagged_artifacts": ["string"],
    "explanation": "string"
  }
}

## KISITLAMALAR
- Kişisel tanımlama yapma. Anonimleştirilmiş metrik analiz yap.
- Görüntü kalitesi yetersizse (DPI < 72, yüz < 80px) hata döndür.
- Anatomik normları Caucasian/Asian/African varyasyona göre kalibre et.
- Sonuçlar adli delil değil, araştırma skoru olarak işaretlenir.
"""


def _format_result_markdown(raw_text: str, parsed: dict | None) -> str:
    """API cevabini gorsel Markdown formatina cevir."""
    if not parsed:
        return f"### 📝 Ham API Yanıtı\n\n{raw_text}"

    sections = []

    # Deepfake Risk
    df = parsed.get("deepfake_indicators", {})
    risk = df.get("risk_level", "?")
    risk_score = df.get("overall_risk_score", 0)
    conf = df.get("confidence", 0)
    color_map = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🟠", "CRITICAL": "🔴"}
    icon = color_map.get(risk, "⚪")

    sections.append(
        f"## {icon} Deepfake Risk: **{risk}**\n"
        f"- **Risk Skoru:** {risk_score:.2f} / 1.0\n"
        f"- **Güven:** %{conf*100:.1f}\n"
    )

    if df.get("flagged_artifacts"):
        flags = ", ".join(df["flagged_artifacts"])
        sections.append(f"- **İşaretlenen Artifaktlar:** {flags}\n")

    if df.get("explanation"):
        sections.append(f"- **Açıklama:** {df['explanation']}\n")

    # Asimetri
    asym = parsed.get("asymmetry", {})
    if asym:
        fai = asym.get("FAI_score", 0)
        fai_status = "✅ Normal" if 0.5 <= fai <= 3.5 else ("⚠️ Aşırı Simetri" if fai < 0.5 else "⚠️ Belirgin Asimetri")
        sections.append(
            f"### 📐 Asimetri Analizi\n"
            f"| Metrik | Değer | Durum |\n|---|---|---|\n"
            f"| FAI Skoru | {fai:.2f}% | {fai_status} |\n"
            f"| Orbital Delta | {asym.get('orbital_delta_mm', 0):.2f} mm | {'✅' if asym.get('orbital_delta_mm', 0) <= 2.5 else '⚠️'} |\n"
            f"| Nasal Sapma | {asym.get('nasal_deviation_mm', 0):.2f} mm | {'✅' if asym.get('nasal_deviation_mm', 0) <= 3.0 else '⚠️'} |\n"
            f"| Labial Delta | {asym.get('labial_delta_px', 0):.1f} px | — |\n"
            f"| Mandibular Oran | {asym.get('mandibular_ratio', 0):.3f} | {'✅' if 0.97 <= asym.get('mandibular_ratio', 1) <= 1.03 else '⚠️'} |\n"
        )
        if asym.get("interpretation"):
            sections.append(f"> {asym['interpretation']}\n")

    # Dudak
    lip = parsed.get("lip_anatomy", {})
    if lip:
        ratio = lip.get("upper_lower_ratio", 0)
        ratio_status = "✅" if 0.55 <= ratio <= 0.70 else ("⚠️ Yapay?" if abs(ratio - 0.618) < 0.01 else "⚠️")
        sections.append(
            f"### 👄 Dudak Anatomisi\n"
            f"| Metrik | Değer | Durum |\n|---|---|---|\n"
            f"| Cupid Bow Simetrisi | {lip.get('cupid_symmetry_deg', 0):.1f}° | {'✅' if lip.get('cupid_symmetry_deg', 0) <= 5 else '⚠️'} |\n"
            f"| Üst/Alt Oran | {ratio:.3f} | {ratio_status} |\n"
            f"| Oral Genişlik İndeksi | {lip.get('oral_width_index', 0):.3f} | — |\n"
            f"| Doku Skoru | {lip.get('texture_score', 0):.2f} | {'✅' if lip.get('texture_score', 0) < 0.5 else '⚠️ GAN?'} |\n"
        )
        if lip.get("artifacts_detected"):
            sections.append(f"- **Artifaktlar:** {', '.join(lip['artifacts_detected'])}\n")

    # Cene
    jaw = parsed.get("jaw_anatomy", {})
    if jaw:
        gonial = jaw.get("gonial_angle_deg", 0)
        sections.append(
            f"### 🦴 Çene Anatomisi\n"
            f"| Metrik | Değer | Durum |\n|---|---|---|\n"
            f"| Gonial Açı | {gonial:.1f}° | {'✅' if 120 <= gonial <= 130 else '⚠️'} |\n"
            f"| Chin Projection | {jaw.get('chin_projection_mm', 0):.1f} mm | — |\n"
            f"| Jawline Skoru | {jaw.get('jawline_score', 0):.1f}/10 | {'✅' if jaw.get('jawline_score', 0) >= 6.5 else '⚠️'} |\n"
            f"| Seam İndeksi | {jaw.get('seam_index', 0):.2f} | {'✅' if jaw.get('seam_index', 0) < 0.3 else '⚠️ Deepfake?'} |\n"
        )

    return "\n".join(sections)


def _build_metrics_table_html():
    """Metrik referans tablosu."""
    rows = [
        ("Asimetri", "FAI", "%", "0.5–3.5", "&lt;0.5 = yapay simetri", "#3B82F6"),
        ("Asimetri", "Orbital Delta", "mm", "0–2.5", "&gt;4mm / &lt;0.1mm", "#3B82F6"),
        ("Asimetri", "Nasal Deviation", "mm", "0–3.0", "Ani simetri değişimi", "#3B82F6"),
        ("Dudak", "Cupid Bow Simetrisi", "°", "0–5°", "&lt;0.5° = GAN", "#22C55E"),
        ("Dudak", "Üst/Alt Oran", "oran", "0.55–0.70", "Tam 0.618 = yapay", "#22C55E"),
        ("Dudak", "Vermillion ΔE", "CIELAB", "4–18", "&lt;2 = blur", "#22C55E"),
        ("Çene", "Gonial Açı", "°", "120°–130°", "Asimetrik açı", "#F59E0B"),
        ("Çene", "Jawline Skoru", "0-10", "6.5–9.0", "&lt;5 = seam", "#F59E0B"),
        ("Çene", "Bigonial Oran", "oran", "0.97–1.03", "&gt;1.05 = warp", "#F59E0B"),
        ("Deepfake", "Blending", "0-10", "8.0+", "&lt;6 = yüksek risk", "#EF4444"),
        ("Deepfake", "GAN Texture", "0-1", "&lt;0.3", "&gt;0.5 = AI", "#EF4444"),
        ("Deepfake", "Lighting", "°", "&lt;15°", "&gt;30° = kompozit", "#EF4444"),
    ]
    trs = ""
    for mod, metric, unit, normal, signal, color in rows:
        trs += (f'<tr><td style="color:{color};font-weight:600">{mod}</td>'
                f'<td>{metric}</td><td style="text-align:center;color:#94A3B8">{unit}</td>'
                f'<td style="text-align:center;color:#22C55E">{normal}</td>'
                f'<td style="color:#F59E0B">{signal}</td></tr>')
    return (
        '<div style="overflow-x:auto;border-radius:8px;border:1px solid rgba(6,182,212,0.2)">'
        '<table style="width:100%;border-collapse:collapse;font-size:0.75rem;'
        'background:#0d1117;color:#e6edf3;font-family:Inter,sans-serif">'
        '<thead><tr style="background:#161b22;border-bottom:2px solid #06B6D4">'
        '<th style="padding:6px 8px;text-align:left;color:#06B6D4">Modül</th>'
        '<th style="padding:6px 8px;text-align:left;color:#06B6D4">Metrik</th>'
        '<th style="padding:6px 8px;text-align:center;color:#06B6D4">Birim</th>'
        '<th style="padding:6px 8px;text-align:center;color:#06B6D4">Normal</th>'
        '<th style="padding:6px 8px;text-align:left;color:#06B6D4">Deepfake Sinyali</th>'
        f'</tr></thead><tbody>{trs}</tbody></table></div>'
    )


def _build_pipeline_html():
    """Analiz pipeline gorselini HTML olarak olustur."""
    steps = [
        ("🖼️", "Girişi", "#3B82F6"), ("🎯", "Landmark", "#22C55E"),
        ("📐", "Hesaplama", "#F59E0B"), ("🧠", "AI Skor", "#8B5CF6"),
        ("📊", "Dashboard", "#06B6D4"), ("📋", "Rapor", "#EF4444"),
    ]
    cards = ""
    for i, (icon, title, color) in enumerate(steps):
        arrow = '<span style="color:#4a5568;font-size:1.2rem;margin:0 2px">→</span>' if i < len(steps) - 1 else ""
        cards += (f'<div style="display:flex;align-items:center">'
                  f'<div style="background:#161b22;border:1px solid {color}40;border-radius:8px;'
                  f'padding:6px 10px;text-align:center;min-width:70px">'
                  f'<div style="font-size:1.2rem">{icon}</div>'
                  f'<div style="font-weight:600;font-size:0.65rem;color:{color}">{title}</div>'
                  f'</div>{arrow}</div>')
    return (f'<div style="display:flex;align-items:center;justify-content:center;'
            f'flex-wrap:wrap;gap:2px;padding:6px 0">{cards}</div>')


def handle_craniofacial_analysis(image, provider, api_key, analysis_mode, region_focus):
    """Kraniyofasiyal analiz handler — Gradio callback."""
    if image is None:
        return "❌ Lütfen bir yüz görseli yükleyin.", "{}", ""

    # Gorseli gecici dosyaya kaydet
    if isinstance(image, np.ndarray):
        from PIL import Image
        img = Image.fromarray(image)
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        img.save(tmp.name, quality=95)
        image_path = tmp.name
    elif isinstance(image, str):
        image_path = image
    else:
        return "❌ Desteklenmeyen görsel formatı.", "{}", ""

    # System prompt'u mode/region ile zenginlestir
    prompt = SYSTEM_PROMPT
    if analysis_mode != "FULL":
        prompt += f"\n\nANALYSIS_MODE: {analysis_mode}"
    if region_focus != "ALL":
        prompt += f"\nREGION_FOCUS: {region_focus}"

    # API cagir
    from services.vision_api import run_craniofacial_analysis
    raw_text, parsed = run_craniofacial_analysis(provider, api_key, image_path, prompt)

    # Sonuclari formatla
    result_md = _format_result_markdown(raw_text, parsed)
    json_str = json.dumps(parsed, indent=2, ensure_ascii=False) if parsed else raw_text

    # Risk badge
    if parsed:
        df = parsed.get("deepfake_indicators", {})
        risk = df.get("risk_level", "UNKNOWN")
        score = df.get("overall_risk_score", 0)
        badge_colors = {"LOW": "#22C55E", "MEDIUM": "#F59E0B", "HIGH": "#EF4444", "CRITICAL": "#DC2626"}
        bc = badge_colors.get(risk, "#94A3B8")
        badge = (f'<div style="text-align:center;padding:12px;border-radius:10px;'
                 f'background:linear-gradient(135deg,#0d1117,#161b22);'
                 f'border:2px solid {bc}">'
                 f'<div style="font-size:2rem;font-weight:700;color:{bc}">{risk}</div>'
                 f'<div style="font-size:0.85rem;color:#94A3B8;margin-top:4px">'
                 f'Risk Skoru: {score:.2f} / 1.0</div></div>')
    else:
        badge = ('<div style="text-align:center;padding:12px;background:#161b22;'
                 'border-radius:10px;color:#94A3B8">Sonuç bekleniyor...</div>')

    return result_md, json_str, badge


def create_craniofacial_tab():
    """Kraniyofasiyal Biyometrik Analiz sekmesini olustur."""

    # Hero
    gr.HTML(
        '<div class="chat-hero" style="padding:8px 16px;margin-bottom:2px">'
        '<h2 style="font-size:1rem;margin:0 0 2px 0">🧬 Kraniyofasiyal Biyometrik Analiz</h2>'
        '<p style="margin:0;font-size:0.7rem">AI Vision API ile yüz anatomisi analizi — '
        'Landmark, asimetri, dudak/çene anatomisi ve deepfake artifakt tespiti.</p>'
        '</div>'
    )

    # Pipeline
    gr.HTML(_build_pipeline_html())

    with gr.Row(equal_height=True):
        # ══ SOL PANEL: Kontroller ══
        with gr.Column(scale=1, min_width=280):
            cf_image = gr.Image(label="🖼️ Yüz Görseli", type="numpy", height=180)

            cf_provider = gr.Dropdown(
                choices=["Google (Gemini)", "OpenAI (GPT-4o)", "Anthropic (Claude)"],
                value="Google (Gemini)", label="🤖 AI Sağlayıcı"
            )
            cf_api_key = gr.Textbox(
                label="🔑 API Anahtarı", type="password",
                value=_load_api_key("Google (Gemini)"),
                placeholder="API key girin...",
                info="Seçilen sağlayıcının API anahtarı"
            )
            with gr.Row():
                cf_save_btn = gr.Button("💾 Key Kaydet", size="sm", scale=1)
                cf_save_status = gr.Textbox(show_label=False, interactive=False,
                                            scale=2, max_lines=1)

            with gr.Row():
                cf_mode = gr.Dropdown(
                    choices=["FULL", "QUICK", "REGION_SPECIFIC"],
                    value="FULL", label="📋 Mod"
                )
                cf_region = gr.Dropdown(
                    choices=["ALL", "FACE", "LIPS", "JAW", "EYES", "NOSE"],
                    value="ALL", label="🎯 Bölge"
                )

            cf_btn = gr.Button("🧬 Anatomik Analiz Başlat", variant="primary", size="sm")

            # Risk badge
            cf_badge = gr.HTML(
                '<div style="text-align:center;padding:12px;background:#161b22;'
                'border-radius:10px;color:#94A3B8;font-size:0.8rem">'
                '⏳ Analiz bekleniyor...</div>'
            )

        # ══ SAĞ PANEL: Sonuçlar ══
        with gr.Column(scale=2, min_width=500):
            with gr.Tab("📊 Analiz Raporu"):
                cf_result = gr.Markdown(
                    value="*Bir görsel yükleyin ve API anahtarı girerek analizi başlatın.*"
                )
            with gr.Tab("🔧 JSON Çıktı"):
                cf_json = gr.Code(label="JSON Yanıt", language="json", lines=18)
            with gr.Tab("📋 Referans Tablosu"):
                gr.HTML(_build_metrics_table_html())

    # Event bindings
    cf_btn.click(
        fn=handle_craniofacial_analysis,
        inputs=[cf_image, cf_provider, cf_api_key, cf_mode, cf_region],
        outputs=[cf_result, cf_json, cf_badge]
    )

    # Provider degistiginde kaydedilmis key'i yukle
    cf_provider.change(
        fn=_load_api_key,
        inputs=[cf_provider],
        outputs=[cf_api_key]
    )

    # Key kaydet butonu
    cf_save_btn.click(
        fn=_save_api_key,
        inputs=[cf_provider, cf_api_key],
        outputs=[cf_save_status]
    )
