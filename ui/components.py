"""
Deepfake Detection System v3.0 — UI Yardımcı Fonksiyonlar
Sekme handler fonksiyonları ve Plotly grafik oluşturucuları.
"""
import numpy as np
from PIL import Image
from ui.translations import t

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# ── Lazy predictor ──
_predictor = None
def lazy_predictor():
    global _predictor
    if _predictor is None:
        from inference.predictor import DeepfakePredictor
        _predictor = DeepfakePredictor()
    return _predictor

# ── Lazy LLM ──
_llm = None
def get_llm(api_key=None):
    global _llm
    if _llm is None:
        from services.llm_module import DeepfakeAnalysisAssistant
        _llm = DeepfakeAnalysisAssistant(gemini_api_key=api_key)
    return _llm

# Son analiz bağlamı (chatbot için)
_last_analysis = {}
_last_image = None

def set_last_analysis(result, image=None):
    global _last_analysis, _last_image
    _last_analysis = result
    if image is not None:
        _last_image = image

def get_last_analysis():
    return _last_analysis

def get_last_image():
    return _last_image

# ================================================================
# PLOTLY GRAFİKLERİ
# ================================================================
PLOT_LAYOUT = dict(
    paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
    font=dict(color="#e6edf3", family="Inter"),
    margin=dict(l=40, r=20, t=40, b=40),
)

def create_probability_bar(fake_prob, real_prob):
    if not HAS_PLOTLY:
        return None
    fig = go.Figure(data=[
        go.Bar(x=["FAKE", "REAL"], y=[fake_prob, real_prob],
               marker_color=["#EF4444", "#22C55E"], text=[f"{fake_prob:.3f}", f"{real_prob:.3f}"],
               textposition="outside")
    ])
    fig.update_layout(**PLOT_LAYOUT, title="Sahte/Gerçek Olasılık", height=200,
                      yaxis=dict(range=[0, 1], gridcolor="#1e293b"))
    return fig

def create_tta_chart(tta_individual):
    if not HAS_PLOTLY or not tta_individual:
        return None
    fig = go.Figure()
    fig.add_trace(go.Bar(x=list(range(len(tta_individual))), y=tta_individual,
                         marker_color="#06B6D4", name="Fake Prob"))
    fig.add_hline(y=0.5, line_dash="dash", line_color="red", annotation_text="Threshold")
    fig.update_layout(**PLOT_LAYOUT, title="TTA Augmentasyon Sonuçları", height=200,
                      xaxis_title="Aug #", yaxis_title="Fake Prob",
                      yaxis=dict(range=[0, 1], gridcolor="#1e293b"))
    return fig


def create_compression_sweep_chart(sweep_result):
    """Compression robustness sweep Plotly grafigi."""
    if not HAS_PLOTLY or not sweep_result:
        return None
    qualities = sweep_result["qualities"]
    fake_probs = sweep_result["fake_probs"]
    verdicts = sweep_result["verdicts"]
    colors = ["#EF4444" if v == "FAKE" else "#22C55E" for v in verdicts]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=qualities, y=fake_probs, mode="lines+markers",
        line=dict(color="#06B6D4", width=2),
        marker=dict(color=colors, size=10, line=dict(width=1, color="white")),
        name="Fake Prob", text=[f"Q={q} | {v}" for q, v in zip(qualities, verdicts)],
        hovertemplate="Quality: %{x}<br>Fake Prob: %{y:.4f}<br>%{text}",
    ))
    fig.add_hline(y=0.5, line_dash="dash", line_color="#EF4444",
                  annotation_text="Karar Siniri (0.5)")

    flip = sweep_result.get("decision_flip_quality")
    if flip:
        fig.add_vline(x=flip, line_dash="dot", line_color="#F59E0B",
                      annotation_text=f"Verdict Flip Q={flip}")

    # Platform isaretleri
    from core.compression import PLATFORM_PROFILES
    for key, p in PLATFORM_PROFILES.items():
        if key != "original" and p["quality"] in range(min(qualities), max(qualities)+1):
            fig.add_vline(x=p["quality"], line_dash="dot", line_color="#94A3B8",
                          annotation_text=p["label"], annotation_position="top")

    fig.update_layout(**PLOT_LAYOUT, title="Compression Robustness Sweep", height=220,
                      xaxis_title="JPEG Quality", yaxis_title="Fake Probability",
                      xaxis=dict(autorange="reversed", gridcolor="#1e293b"),
                      yaxis=dict(range=[0, 1], gridcolor="#1e293b"))
    return fig



def create_analytics_charts(analytics):
    """Dashboard için 3 grafik döndür (XAI kaldırıldı)."""
    if not HAS_PLOTLY:
        return None, None, None
    daily = analytics.get("daily", [])
    sources = analytics.get("sources", {})

    # Günlük analiz
    fig1 = go.Figure(go.Bar(
        x=[d["day"] for d in daily], y=[d["cnt"] for d in daily],
        marker_color="#3B82F6"))
    fig1.update_layout(**PLOT_LAYOUT, title="Günlük Analiz Sayısı", height=200)
    fig1.update_layout(margin=dict(l=35, r=10, t=30, b=30))

    # FAKE vs REAL
    fig2 = go.Figure(data=[
        go.Bar(name="FAKE", x=[d["day"] for d in daily],
               y=[d.get("fake_cnt", 0) for d in daily], marker_color="#EF4444"),
        go.Bar(name="REAL", x=[d["day"] for d in daily],
               y=[d.get("real_cnt", 0) for d in daily], marker_color="#22C55E"),
    ])
    fig2.update_layout(**PLOT_LAYOUT, title="FAKE vs REAL", height=200, barmode="stack")
    fig2.update_layout(margin=dict(l=35, r=10, t=30, b=30))

    # Kaynak dağılımı
    fig3 = go.Figure(go.Bar(
        x=list(sources.values()), y=list(sources.keys()),
        orientation="h", marker_color="#8B5CF6"))
    fig3.update_layout(**PLOT_LAYOUT, title="Kaynak Tipi", height=200)
    fig3.update_layout(margin=dict(l=35, r=10, t=30, b=30))

    return fig1, fig2, fig3

def create_trend_chart(daily):
    if not HAS_PLOTLY or not daily:
        return None
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=[d["day"] for d in daily], y=[d["cnt"] for d in daily],
                         name="Analiz Sayısı", marker_color="#3B82F6", opacity=0.6), secondary_y=False)
    fig.add_trace(go.Scatter(x=[d["day"] for d in daily],
                             y=[d.get("avg_fake", 0.5) for d in daily],
                             name="Ort. Fake Prob", line=dict(color="#EF4444", width=2),
                             mode="lines+markers"), secondary_y=True)
    fig.add_hline(y=0.5, line_dash="dash", line_color="#F59E0B", secondary_y=True)
    fig.update_layout(**PLOT_LAYOUT, title="Model Doğruluk Trendi", height=200)
    fig.update_layout(margin=dict(l=35, r=35, t=30, b=30))
    fig.update_yaxes(title_text="Analiz Sayısı", secondary_y=False, gridcolor="#1e293b")
    fig.update_yaxes(title_text="Ort. Fake Prob", secondary_y=True, range=[0, 1])
    return fig

# ================================================================
# SEKME HANDLER FONKSİYONLARI
# ================================================================

def build_context_html(r: dict) -> str:
    """Son analiz sonucunu Analiz Bağlamı kartına dönüştür."""
    if not r:
        return ('<div class="ctx-card">'
                '<strong>⏳ Henüz analiz yapılmadı</strong><br>'
                'Single Image sekmesinden bir görsel analiz edin, '
                'sonuçlar otomatik olarak buraya yansıyacaktır.'
                '</div>')

    v       = r.get('verdict', '?')
    fake_p  = r.get('fake_prob', 0)
    real_p  = r.get('real_prob', 0)
    tta_std = r.get('tta_std', 0)
    gcam    = r.get('gradcam_score', 0)
    cf_prob = r.get('counterfactual_prob', 0)
    n_tta   = len(r.get('tta_individual', []))

    # Verdict'e gore guven ve renk
    if v == 'FAKE':
        badge_color, badge_icon = '#EF4444', '🚨'
        conf = fake_p
    elif v == 'REAL':
        badge_color, badge_icon = '#22C55E', '✅'
        conf = real_p
    else:  # UNCERTAIN
        badge_color, badge_icon = '#F59E0B', '⚠️'
        conf = max(fake_p, real_p)
    conf_pct    = conf * 100
    fake_pct    = fake_p * 100
    real_pct    = real_p * 100

    # Güven rengi
    if conf_pct >= 80:
        conf_color = '#22C55E'
    elif conf_pct >= 60:
        conf_color = '#F59E0B'
    else:
        conf_color = '#EF4444'

    # TTA güvenilirlik
    if tta_std < 0.03:
        tta_label, tta_color = 'Kararlı', '#22C55E'
    elif tta_std < 0.08:
        tta_label, tta_color = 'Belirsiz', '#F59E0B'
    else:
        tta_label, tta_color = 'Tutarsız', '#EF4444'

    # Görsel kalite
    q    = r.get('image_quality', {})
    q_val = q.get('estimated_quality', '?') if q else '?'
    q_rel = q.get('reliability_pct', '?') if q else '?'

    bar_style = ('height:6px;border-radius:3px;margin-top:3px;'
                 'background:linear-gradient(90deg,{color} {pct}%,#1e293b {pct}%)')

    html = f"""
<div style="font-family:'Inter',sans-serif;font-size:0.82rem;line-height:1.55;color:#e6edf3">

  <!-- Karar Badge -->
  <div style="display:flex;align-items:center;gap:8px;margin-bottom:10px">
    <span style="background:{badge_color}22;color:{badge_color};border:1px solid {badge_color};
                 border-radius:6px;padding:3px 10px;font-weight:700;font-size:0.9rem;">
      {badge_icon} {v}
    </span>
    <span style="color:{conf_color};font-weight:600">%{conf_pct:.1f} güven</span>
  </div>

  <!-- Olasılık Barları -->
  <div style="margin-bottom:8px">
    <div style="display:flex;justify-content:space-between">
      <span style="color:#EF4444">🔴 FAKE</span>
      <span style="color:#EF4444;font-weight:600">{fake_pct:.1f}%</span>
    </div>
    <div style="{bar_style.format(color='#EF4444', pct=fake_pct)}"></div>

    <div style="display:flex;justify-content:space-between;margin-top:5px">
      <span style="color:#22C55E">🟢 REAL</span>
      <span style="color:#22C55E;font-weight:600">{real_pct:.1f}%</span>
    </div>
    <div style="{bar_style.format(color='#22C55E', pct=real_pct)}"></div>
  </div>

  <hr style="border:none;border-top:1px solid #1e293b;margin:8px 0">

  <!-- TTA & XAI -->
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:5px;margin-bottom:6px">
    <div>
      <span style="color:#94A3B8">TTA ({n_tta} aug)</span><br>
      <span style="color:{tta_color};font-weight:600">{tta_label}</span>
      <span style="color:#64748B;font-size:0.75rem"> (σ={tta_std:.3f})</span>
    </div>
    <div>
      <span style="color:#94A3B8">GradCAM++</span><br>
      <span style="color:#06B6D4;font-weight:600">{gcam:.4f}</span>
    </div>
    <div>
      <span style="color:#94A3B8">Counterfactual</span><br>
      <span style="color:#8B5CF6;font-weight:600">{cf_prob:.4f}</span>
    </div>
    <div>
      <span style="color:#94A3B8">Görsel Kalite</span><br>
      <span style="color:#F59E0B;font-weight:600">Q={q_val} (%{q_rel})</span>
    </div>
  </div>

</div>"""
    return html


def handle_single_analysis(image, tta_count, source_platform="original"):
    """Sekme 1: Single Image Analysis + compression quality + platform tespiti."""
    if image is None:
        return [None]*16
    from inference.analyze_engine import analyze_image

    # Kaynak platformuna gore on isleme
    analysis_image = image
    source_note = ""
    if source_platform and source_platform not in ("original", "bilinmiyor"):
        try:
            from core.compression import simulate_platform_compression, PLATFORM_PROFILES
            sim = simulate_platform_compression(image, source_platform)
            analysis_image = sim["compressed_image"]
            p = PLATFORM_PROFILES.get(source_platform, {})
            source_note = (f"\n\n> **{p.get('icon','')} {p.get('label', source_platform)} modu aktif** "
                          f"| Simulasyon Q={sim['quality_used']} | "
                          f"Boyut: {sim['original_size']} -> {sim['compressed_size']}")
        except Exception:
            pass

    r = analyze_image(analysis_image, tta_count=int(tta_count), source="ui")
    set_last_analysis(r, image=image)  # Gorseli de sakla

    # Platform tespiti — kaynak seçimine göre davranış
    platform_md = ""
    if source_platform == "original":
        # Orijinal görsel → JPEG forensik analizi atla
        platform_md = (
            "### 📱 Platform Tespit Raporu\n\n"
            "> ✅ **Görsel orijinal olarak yüklendi.**\n"
            "> Platform sıkıştırması uygulanmadı, JPEG forensik analizi atlandı.\n\n"
            "_Kaynağı bilinmeyen görseller için 'bilinmiyor' seçeneğini kullanın._"
        )
    elif source_platform == "bilinmiyor":
        # Kaynak bilinmiyor → tam JPEG forensik platform tespiti çalıştır
        try:
            from core.platform_detect import detect_platform, format_platform_report
            pd_result = detect_platform(image)
            platform_md = format_platform_report(pd_result)
            r["platform_detection"] = pd_result
        except Exception as e:
            platform_md = f"Platform tespiti hatasi: {e}"
    else:
        # Twitter/TikTok seçili → simülasyon bilgisi + platform tespiti
        try:
            from core.platform_detect import detect_platform, format_platform_report
            pd_result = detect_platform(image)
            platform_md = format_platform_report(pd_result)
            r["platform_detection"] = pd_result
        except Exception as e:
            platform_md = f"Platform tespiti hatasi: {e}"

    # Kalite tahmini
    quality_info = ""
    try:
        from core.compression import estimate_jpeg_quality, get_reliability_rating
        q = estimate_jpeg_quality(image)
        rating = get_reliability_rating(q["estimated_quality"])
        r["image_quality"] = q
        quality_info = (f"| Image Quality | Q={q['estimated_quality']} ({rating['text']}) |\n"
                        f"| Blockiness | {q['blockiness_score']:.4f} |\n"
                        f"| Reliability | {q['reliability_pct']}% |\n")
        if q["warning"]:
            quality_info += f"| **Uyari** | {q['warning']} |\n"
    except Exception:
        pass

    # Verdict icon: FAKE=kirmizi, REAL=yesil, NON-PHOTO=turuncu
    v = r['verdict']
    if v == "FAKE":
        v_icon = "\U0001f6a8"
        display_prob = r['fake_prob']
    elif v == "REAL":
        v_icon = "\u2705"
        display_prob = r['real_prob']
    elif v == "NON-PHOTO":
        v_icon = "\U0001f5bc\ufe0f"
        display_prob = r.get('confidence', 0)
    else:
        v_icon = "\u26a0\ufe0f"
        display_prob = max(r['fake_prob'], r['real_prob'])

    if v == "NON-PHOTO":
        warning_msg = r.get('warning', 'Bu gorsel bir fotograf degil.')
        verdict_text = (f"### {v_icon} FOTOGRAF DEGIL\n"
                        f"> ⚠️ {warning_msg}\n\n"
                        f"Deepfake analizi yalnizca gercek fotograflar icin gecerlidir.")
    else:
        verdict_text = (f"### {v_icon} {v} ({display_prob:.4f})\n"
                        f"Guven: %{display_prob*100:.1f}{source_note}")

    # Foto filtre bilgisi
    pf = r.get("photo_filter", {})
    pf_rows = []
    if pf:
        pf_rows.append("| **--- Fotograf Filtresi ---** |  |")
        pf_rows.append(f"| Method | {pf.get('method', '-')} |")
        pf_rows.append(f"| Photo Score (stat) | {pf.get('photo_score', '-')} |")
        pf_rows.append(f"| Color Ratio | {pf.get('color_ratio', '-')} |")
        pf_rows.append(f"| Sharp Ratio | {pf.get('sharp_ratio', '-')} |")
        pf_rows.append(f"| Noise Std | {pf.get('noise_std', '-')} |")
        pf_rows.append(f"| Flat Ratio | {pf.get('flat_ratio', '-')} |")
        if "clip_score" in pf:
            pf_rows.append(f"| CLIP Score | {pf.get('clip_score', '-')} |")
            pf_rows.append(f"| CLIP Label | {pf.get('clip_label', '-')} |")
            pf_rows.append(f"| Combined Score | {pf.get('combined_score', '-')} |")

    m_rows = [
        "| Metrik | Deger |", "|---|---|",
        f"| Verdict | {r['verdict']} |",
        f"| Fake Probability | {r['fake_prob']:.4f} |",
        f"| Real Probability | {r['real_prob']:.4f} |",
        "| Calibrated | No |",
    ]
    if quality_info:
        for line in quality_info.strip().split("\n"):
            if line.strip():
                m_rows.append(line)
    m_rows.extend([
        f"| GradCAM++ Score | {r['gradcam_score']:.4f} |",
        f"| Counterfactual Prob | {r['counterfactual_prob']:.4f} |",
        f"| TTA Augmentations | {len(r.get('tta_individual',[]))} |",
        f"| TTA Std Dev | {r['tta_std']:.4f} |",
        f"| TTA Individual | {r.get('tta_individual',[])} |",
    ])
    m_rows.extend(pf_rows)
    metrics = "\n".join(m_rows)

    # Quality warning banner
    quality_warning = ""
    try:
        from core.compression import estimate_jpeg_quality, get_reliability_rating
        q = r.get("image_quality", {})
        if q and q.get("warning"):
            rating = get_reliability_rating(q["estimated_quality"])
            quality_warning = (f'<div style="background:{rating["color"]}22; border-left:4px solid {rating["color"]};'
                               f' padding:8px 12px; border-radius:4px; margin:4px 0; font-size:0.85em;">'
                               f'<b>{rating["text"]}</b> (Q={q["estimated_quality"]}) — {q["warning"]}</div>')
    except Exception:
        pass

    # Yuz kutulari
    face_img = None
    try:
        from core.face_detector import draw_face_boxes
        face_img = draw_face_boxes(image, r.get("face_boxes"))
    except Exception:
        pass

    hm = r.get("heatmaps", {})
    prob_chart = create_probability_bar(r["fake_prob"], r["real_prob"])
    tta_chart = create_tta_chart(r.get("tta_individual", []))

    wm_text = "Watermark uygulandi." if r.get("watermarked_image") else ""

    return [
        r.get("watermarked_image", image),
        verdict_text,
        metrics,
        face_img,
        hm.get("gradcam"),
        hm.get("eigencam"),
        hm.get("fastcam"),
        hm.get("lime"),
        prob_chart,
        tta_chart,
        wm_text,
        r.get("analysis_id", ""),
        f"Analiz #{r.get('analysis_id','')} kaydedildi.",
        quality_warning,
        platform_md,
        build_context_html(r),   # 16. çıktı: Analiz Bağlamı kartı
    ]

def handle_dwt_analysis(image):
    """Mimari Ic Isleyisi — RGB + DWT + Mesh gorselleri."""
    if image is None:
        return None, None, None
    try:
        from core.frequency import (
            generate_dwt_visualization, generate_rgb_visualization,
            generate_mesh_visualization
        )
        from inference.analyze_engine import _add_watermark

        rgb_img = _add_watermark(generate_rgb_visualization(image))
        dwt_img = _add_watermark(generate_dwt_visualization(image))
        mesh_img = _add_watermark(generate_mesh_visualization(image))
        return rgb_img, dwt_img, mesh_img
    except Exception as e:
        return None, None, None

def handle_feedback(analysis_id, label):
    """Geri bildirim kaydet + gorseli diske kaydet."""
    if not analysis_id:
        return "Once bir analiz yapin."
    try:
        from db.database import get_db
        from core.fine_tuner import save_feedback_image, check_readiness

        db = get_db()

        # Son analiz gorselini feedback dizinine kaydet
        image_path = ""
        last_img = get_last_image()
        if last_img is not None:
            image_path = save_feedback_image(last_img, label)

        db.save_feedback(int(analysis_id), label, image_path=image_path)
        readiness = check_readiness()
        ready_msg = "\U0001f7e2 Fine-tune icin hazir!" if readiness["ready"] else f"\U0001f7e1 Minimum {readiness['min_required']} gorsel gerekli."
        return (f"\u2705 Geri bildirim kaydedildi!\n"
                f"{readiness['message']}\n{ready_msg}")
    except Exception as e:
        return f"Hata: {e}"


def handle_finetune():
    """Fine-tune tetikle."""
    try:
        from core.fine_tuner import start_finetune, check_readiness
        readiness = check_readiness()
        if not readiness["ready"]:
            return f"\u274c {readiness['message']} — Minimum {readiness['min_required']} gorsel gerekli."
        model = lazy_predictor().model
        result = start_finetune(model)
        return f"\U0001f9e0 {result}"
    except Exception as e:
        return f"Hata: {e}"


def handle_finetune_status():
    """Fine-tune ilerleme durumu."""
    try:
        from core.fine_tuner import get_finetune_status
        s = get_finetune_status()
        if s["error"]:
            return f"\u274c Hata: {s['error']}"
        if s["running"]:
            return f"\u23f3 {s['progress']}"
        if s["completed"]:
            return f"\u2705 {s['progress']}"
        return "Henuz fine-tune baslatilmadi."
    except Exception as e:
        return f"Hata: {e}"


def handle_rollback():
    """Orijinal modele geri don."""
    try:
        from core.fine_tuner import rollback_model
        model = lazy_predictor().model
        result = rollback_model(model)
        return f"\u21a9\ufe0f {result}"
    except Exception as e:
        return f"Hata: {e}"


def handle_pool_status():
    """Feedback havuz durumunu dondur."""
    try:
        from core.fine_tuner import check_readiness
        r = check_readiness()
        status = r['message']
        if r['has_finetuned']:
            status += " | \U0001f9e0 Fine-tuned model aktif"
        return status
    except Exception as e:
        return f"Hata: {e}"

def handle_pdf_download(lang="tr"):
    """PDF rapor oluştur ve yolunu döndür."""
    r = get_last_analysis()
    if not r:
        return None
    try:
        from services.pdf_report import generate_pdf_report
        path = generate_pdf_report(r, language=lang)
        return path
    except Exception as e:
        print(f"PDF hatası: {e}")
        return None


def handle_branch_knockout(image):
    """Branch Knockout testi handler."""
    if image is None:
        return "Görsel yükleyin"
    try:
        from core.adversarial import branch_knockout_test
        model = lazy_predictor().model
        results = branch_knockout_test(image, model)

        md = f"### 🔬 Branch Knockout Sonuçları\n\n"
        md += f"**En Kritik Dal:** {results['_critical_branch']} "
        md += f"(Δ={results['_max_shift']:.4f})\n\n"
        md += "| Kombinasyon | Karar | Fake Prob |\n|---|---|---|\n"
        for key, val in results.items():
            if key.startswith("_"):
                continue
            emoji = "✅" if val["verdict"] == results["all"]["verdict"] else "⚠️"
            md += f"| {emoji} {val['label']} | {val['verdict']} | {val['fake_prob']:.4f} |\n"
        return md
    except Exception as e:
        return f"Hata: {e}"

def handle_freq_ablation(image):
    """Frekans Band Ablasyonu handler."""
    if image is None:
        return "Görsel yükleyin"
    try:
        from core.adversarial import frequency_band_ablation
        model = lazy_predictor().model
        results = frequency_band_ablation(image, model)

        base = results["baseline"]["fake_prob"]
        md = f"### 📡 Frekans Band Ablasyonu\n\n"
        md += f"**Baseline:** Fake Prob = {base:.4f}\n\n"
        md += "| Band | Fake Prob | Kayma (Δ) | Etki |\n|---|---|---|---|\n"
        for key, val in results.items():
            if key == "baseline":
                continue
            shift = val.get("shift", 0)
            impact = "🔴 Yüksek" if abs(shift) > 0.1 else "🟡 Orta" if abs(shift) > 0.03 else "🟢 Düşük"
            md += f"| {val['label']} | {val['fake_prob']:.4f} | {shift:+.4f} | {impact} |\n"
        return md
    except Exception as e:
        return f"Hata: {e}"

def handle_resolution_test(image):
    """Çözünürlük dayanıklılığı handler."""
    if image is None:
        return "Görsel yükleyin", None
    try:
        from core.adversarial import resolution_robustness
        model = lazy_predictor().model
        results = resolution_robustness(image, model)

        flip = results["decision_flip_resolution"]
        md = f"### 📐 Çözünürlük Dayanıklılığı\n\n"
        if flip:
            md += f"**⚠️ Karar Değişimi:** {flip}px altında model yanılıyor!\n\n"
        else:
            md += "**✅ Tüm çözünürlüklerde tutarlı karar.**\n\n"
        md += "| Çözünürlük | Fake Prob | Karar |\n|---|---|---|\n"
        for res, fp, v in zip(results["resolutions"], results["fake_probs"], results["verdicts"]):
            marker = "→" if res == 224 else " "
            md += f"| {marker} {res}px | {fp:.4f} | {v} |\n"

        # Grafik
        try:
            import plotly.graph_objects as go
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=results["resolutions"], y=results["fake_probs"],
                mode="lines+markers", name="Fake Prob",
                line=dict(color="#ff6b6b", width=2),
                marker=dict(size=8),
            ))
            fig.add_hline(y=0.5, line_dash="dash", line_color="gray",
                          annotation_text="Karar Eşiği (0.5)")
            fig.update_layout(
                title="Çözünürlük vs Fake Olasılığı",
                xaxis_title="Çözünürlük (px)",
                yaxis_title="P(FAKE)",
                yaxis_range=[0, 1],
                template="plotly_dark",
                height=260,
                margin=dict(l=40, r=20, t=35, b=35),
            )
            return md, fig
        except ImportError:
            return md, None
    except Exception as e:
        return f"Hata: {e}", None

def handle_double_compression(image):
    """Çift sıkıştırma testi handler."""
    if image is None:
        return "Görsel yükleyin"
    try:
        from core.adversarial import double_compression_test
        model = lazy_predictor().model
        results = double_compression_test(image, model)

        orig = results["original"]
        md = f"### 🔄 Çift Sıkıştırma Testi\n\n"
        md += f"**Orijinal Karar:** {orig['verdict']} (Fake={orig['fake_prob']:.4f})\n\n"
        md += "| Zincir | Adımlar | Fake Prob | Kayma | Karar |\n|---|---|---|---|---|\n"
        for key, val in results.items():
            if key == "original":
                continue
            shift = val.get("shift", 0)
            emoji = "✅" if val["verdict"] == orig["verdict"] else "⚠️"
            md += (f"| {emoji} {val['label']} | {val['steps']} | "
                   f"{val['fake_prob']:.4f} | {shift:+.4f} | {val['verdict']} |\n")
        return md
    except Exception as e:
        return f"Hata: {e}"

def handle_compression_sweep(image):
    """Robustness sekmesi: Compression sweep."""
    if image is None:
        return "Gorsel yukleyin", None
    try:
        from core.compression import compression_robustness_sweep, PLATFORM_PROFILES
        p = lazy_predictor()
        result = compression_robustness_sweep(image, p)

        flip = result.get("decision_flip_quality")
        summary = f"**Orijinal Karar:** {result['original_verdict']}\n\n"
        if flip:
            summary += f"⚠️ **Karar Değişimi:** Q={flip} altında model kararı değişiyor!\n\n"
            affected = []
            for key, prof in PLATFORM_PROFILES.items():
                if key != "original" and prof["quality"] <= flip:
                    affected.append(f"{prof['icon']} {prof['label']} (Q={prof['quality']})")
            if affected:
                summary += "**Etkilenen Platformlar:** " + ", ".join(affected) + "\n\n"
        else:
            summary += "✅ Model tüm sıkıştırma seviyelerinde tutarlı karar veriyor.\n\n"

        # Kompakt istatistik
        fps = result["fake_probs"]
        summary += (f"**Fake Prob Aralığı:** {min(fps):.4f} — {max(fps):.4f} | "
                    f"**Δ:** {max(fps)-min(fps):.4f}")

        chart = create_compression_sweep_chart(result)
        return summary, chart
    except Exception as e:
        return f"Hata: {e}", None

def handle_analytics(days):
    """Sekme 5: Dashboard verileri (XAI kaldırıldı)."""
    try:
        from db.database import get_db
        db = get_db()
        a = db.get_analytics(int(days))
        fig1, fig2, fig3 = create_analytics_charts(a)
        trend = create_trend_chart(a.get("daily", []))
        return (str(a["total"]), str(a["fake"]), str(a["real"]),
                f"{a['fake_rate']}%", fig1, fig2, fig3, trend)
    except Exception as e:
        return ("0", "0", "0", "0%", None, None, None, None)



def handle_history(limit):
    """Sekme 7: Geçmiş tablosu."""
    try:
        from db.database import get_db
        db = get_db()
        rows = db.get_history(int(limit))
        if not rows:
            return "Henüz kayıt yok.", ""
        table = "| ID | Tarih | Dosya | Karar | Güven | Fake% | Model | Kaynak |\n|---|---|---|---|---|---|---|---|\n"
        for r in rows:
            table += (f"| {r['id']} | {r['timestamp'][:19]} | {r['filename'][:20]} | "
                     f"{r['verdict']} | {r['confidence']:.3f} | {r['fake_prob']*100:.1f} | "
                     f"{r.get('model_name','—') or '—'} | {r.get('source','ui')} |\n")
        return table, f"{len(rows)} kayıt gösteriliyor."
    except Exception as e:
        return f"Hata: {e}", ""

def handle_clear_history():
    try:
        from db.database import get_db
        get_db().clear_history()
        return "✅ Geçmiş temizlendi.", ""
    except Exception as e:
        return f"Hata: {e}", ""

def handle_chat(message, history, api_key):
    """Sekme 8: Chat."""
    if not message or not message.strip():
        return "", history
    llm = get_llm(api_key if api_key and api_key.strip() else None)
    ctx = get_last_analysis()
    if ctx:
        llm.set_analysis_context(ctx)
    resp = llm.chat(message)
    history = history or []
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": resp})
    return "", history


# ================================================================
# FAZ 1: FORENSİK ANALİZ HANDLER'LARI
# ================================================================

def handle_forensics(image):
    """Forensik analiz: ELA + Noise + Model Konsensüs."""
    if image is None:
        return None, None, "Gorsel yukleyin."
    try:
        from core.forensics import (
            analyze_forensics, compute_forensic_consensus, format_consensus_report
        )
        result = analyze_forensics(image)

        # Model tahminini al (konsensüs için)
        model_fake_prob = 0.5  # varsayılan
        try:
            p = lazy_predictor()
            pred = p.predict(image)
            model_fake_prob = pred.get("fake_prob", 0.5)
        except Exception:
            pass

        # Konsensüs hesapla
        consensus = compute_forensic_consensus(
            model_fake_prob=model_fake_prob,
            ela_score=result["ela_score"],
            noise_score=result["noise_score"],
        )
        consensus_md = format_consensus_report(consensus)

        # Klasik özet
        ela_level = "🔴 Yüksek" if result['ela_score'] > 0.25 else "🟡 Orta" if result['ela_score'] > 0.15 else "🟢 Düşük"
        noise_level = "🔴 Tutarsız" if result['noise_score'] > 0.45 else "🟡 Kısmi" if result['noise_score'] > 0.30 else "🟢 Tutarlı"

        summary = consensus_md + "\n---\n\n"
        summary += (
            f"**ELA Skoru:** {result['ela_score']:.4f} ({ela_level})\n\n"
            f"**Gürültü Tutarlılığı:** {result['noise_score']:.4f} ({noise_level})\n\n"
            f"> ELA: Manipüle bölgeler yüksek error level gösterir.\n"
            f"> Noise: Farklı kaynaklardan gelen bölgeler farklı gürültü dağılımı gösterir."
        )
        # Watermark ekle
        from inference.analyze_engine import _add_watermark
        ela_img = _add_watermark(result["ela_map"])
        noise_img = _add_watermark(result["noise_map"])
        return ela_img, noise_img, summary
    except Exception as e:
        return None, None, f"Forensik analiz hatasi: {e}"


# ================================================================
# FAZ 2: EMBEDDING GÖRSELLEŞTİRME HANDLER'LARI
# ================================================================

def handle_embedding_viz(method="t-SNE"):
    """Embedding uzay gorsellestirmesi — t-SNE veya UMAP.
    Havuz boşsa jury_test setinden otomatik batch embedding cikarir."""
    try:
        from core.embedding_viz import (
            get_pool_size, generate_tsne_plot, generate_umap_plot,
            add_to_pool, extract_embedding, MIN_POINTS_FOR_VIZ,
        )

        pool_size = get_pool_size()

        # Havuz yetersizse jury_test'ten otomatik yükle
        if pool_size < MIN_POINTS_FOR_VIZ:
            loaded = _auto_fill_embedding_pool()
            pool_size = get_pool_size()
            if pool_size < MIN_POINTS_FOR_VIZ:
                return None, (
                    f"⚠️ Yeterli veri yok: **{pool_size}/{MIN_POINTS_FOR_VIZ}** nokta.\n\n"
                    f"Daha fazla görsel analiz edin veya `jury_test/` setinin mevcut olduğundan emin olun."
                )

        if method == "UMAP":
            fig = generate_umap_plot()
        else:
            fig = generate_tsne_plot()

        if fig is None:
            return None, f"⚠️ {method} görselleştirme başarısız — sklearn/umap kurulu mu?"

        # İstatistikler
        from core.embedding_viz import _embedding_pool
        labels = _embedding_pool["labels"]
        n_real = sum(1 for l in labels if l == "REAL")
        n_fake = sum(1 for l in labels if l == "FAKE")

        status = (
            f"✅ **{pool_size}** nokta ile **{method}** görselleştirmesi oluşturuldu.\n\n"
            f"| | Sayı |\n|---|---|\n"
            f"| 🟢 REAL | {n_real} |\n"
            f"| 🔴 FAKE | {n_fake} |\n"
        )
        return fig, status
    except Exception as e:
        import traceback
        return None, f"Embedding görselleştirme hatası: {e}\n```\n{traceback.format_exc()}\n```"


def _auto_fill_embedding_pool(max_per_class=25):
    """Jury test setinden otomatik embedding cikar ve havuza ekle."""
    loaded = 0
    try:
        from core.embedding_viz import add_to_pool, extract_embedding, get_pool_size
        from inference.predictor import get_predictor
        from PIL import Image
        import os, random

        predictor = get_predictor()
        jury_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dataset", "jury_test")

        if not os.path.exists(jury_dir):
            return 0

        for label_name in ["real", "fake"]:
            label_dir = os.path.join(jury_dir, label_name)
            if not os.path.exists(label_dir):
                continue

            files = [f for f in os.listdir(label_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            random.seed(42)
            selected = random.sample(files, min(max_per_class, len(files)))

            for fname in selected:
                try:
                    fpath = os.path.join(label_dir, fname)
                    result = predictor.predict(fpath)
                    # Embedding zaten analyze_engine üzerinden havuza ekleniyor
                    # Ama doğrudan predictor ile tahmin yaptığımızda eklenmez
                    # Burda elle extract_embedding yapalım
                    img = Image.open(fpath).convert("RGB")
                    from core.preprocess import preprocess_image
                    rgb, freq, mesh = preprocess_image(img)
                    embedding = extract_embedding(predictor.model, rgb, freq, mesh)
                    verdict = "FAKE" if result["fake_prob"] > 0.5 else "REAL"
                    add_to_pool(embedding, verdict, fname, result["fake_prob"])
                    loaded += 1
                except Exception:
                    continue
    except Exception:
        pass
    return loaded


# ================================================================
# FAZ 3: MODEL METRİKLERİ HANDLER'LARI
# ================================================================

def handle_compute_metrics():
    """Model performans metriklerini hesapla — ROC, CM, metrik tablosu.
    Feedback havuzu yetersizse jury_test setinden otomatik hesaplar."""
    try:
        from core.model_metrics import (
            compute_metrics_from_history,
            generate_roc_plot,
            generate_confusion_matrix_plot,
            generate_metrics_summary,
        )

        data = compute_metrics_from_history()

        # Feedback havuzu 20'den az ise jury_test'e düş (daha güvenilir)
        if not data["ready"] or data["count"] < 20:
            jury_data = _compute_metrics_from_jury()
            if jury_data["ready"]:
                data = jury_data

        if not data["ready"]:
            return (
                None, None,
                f"> ⚠️ Yeterli etiketli veri yok ({data['count']} görsel).\n\n"
                f"Feedback havuzu veya `jury_test/` seti bulunamadı.\n"
                f"Sekme 1'deki geri bildirim butonlarını kullanarak veri toplayın."
            )

        source = data.get("source", "feedback")
        roc_fig = generate_roc_plot(data["labels"], data["probs"])
        cm_fig = generate_confusion_matrix_plot(data["labels"], data["preds"])
        summary = generate_metrics_summary(data["labels"], data["preds"], data["probs"])

        source_note = ""
        if source == "jury_test":
            source_note = (
                "\n\n> ℹ️ **Veri Kaynağı:** Jury test seti kullanıldı "
                f"({data['count']} görsel). Feedback havuzu dolduğunda otomatik olarak "
                "feedback verisi kullanılacaktır."
            )
        summary += source_note

        return roc_fig, cm_fig, summary
    except Exception as e:
        import traceback
        return None, None, f"Metrik hesaplama hatası: {e}\n```\n{traceback.format_exc()}\n```"


def _compute_metrics_from_jury(max_per_class=30):
    """Jury test setinden metrik hesapla (feedback yetersizse fallback)."""
    try:
        from inference.predictor import get_predictor
        import os, random

        predictor = get_predictor()
        jury_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dataset", "jury_test")

        if not os.path.exists(jury_dir):
            return {"labels": [], "preds": [], "probs": [], "count": 0, "ready": False}

        labels = []
        preds = []
        probs = []

        for label_name, label_id in [("real", 0), ("fake", 1)]:
            label_dir = os.path.join(jury_dir, label_name)
            if not os.path.exists(label_dir):
                continue

            files = [f for f in os.listdir(label_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            random.seed(42)
            selected = random.sample(files, min(max_per_class, len(files)))

            for fname in selected:
                try:
                    fpath = os.path.join(label_dir, fname)
                    result = predictor.predict(fpath)
                    labels.append(label_id)
                    preds.append(1 if result["label"] == "FAKE" else 0)
                    probs.append(result["fake_prob"])
                except Exception:
                    continue

        return {
            "labels": labels,
            "preds": preds,
            "probs": probs,
            "count": len(labels),
            "ready": len(labels) >= 4,
            "source": "jury_test",
        }
    except Exception:
        return {"labels": [], "preds": [], "probs": [], "count": 0, "ready": False}


# ================================================================
# FAZ 4: MODEL PROFİLİ SEKMESİ
# ================================================================

def handle_model_profile():
    """Model mimarisi, eğitim konfigürasyonu, checkpoint ve
    TEST split üzerinden canlı metrikleri döndür."""
    try:
        import torch, json, os
        from config import model_cfg, DEVICE, paths

        arch_md = _build_architecture_md()
        train_md = _build_training_config_md()
        ckpt_md = _build_checkpoint_md()
        dataset_md = _build_dataset_md()

        # Eğitim test split'inden metrik hesapla
        roc_fig, cm_fig, metrics_md = _compute_test_split_metrics()

        return arch_md, train_md, ckpt_md, dataset_md, roc_fig, cm_fig, metrics_md

    except Exception as e:
        import traceback
        err = f"Model profili hatası: {e}\n```\n{traceback.format_exc()}\n```"
        return err, err, err, err, None, None, err


def _compute_test_split_metrics(max_per_class=50):
    """Eğitim pipeline'ının test split'i (faces_split/test) üzerinden
    ROC, CM ve metrik hesapla. Bu gerçek eğitim performansını yansıtır."""
    try:
        from inference.predictor import get_predictor
        from core.model_metrics import (
            generate_roc_plot, generate_confusion_matrix_plot,
            generate_metrics_summary,
        )
        import os, random

        predictor = get_predictor()
        test_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                "dataset", "faces_split", "test")

        if not os.path.exists(test_dir):
            return None, None, "> ⚠️ Test split bulunamadı: `dataset/faces_split/test/`"

        labels = []
        preds = []
        probs = []

        for label_name, label_id in [("real", 0), ("fake", 1)]:
            label_dir = os.path.join(test_dir, label_name)
            if not os.path.exists(label_dir):
                continue

            files = [f for f in os.listdir(label_dir)
                     if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            random.seed(42)
            selected = random.sample(files, min(max_per_class, len(files)))

            for fname in selected:
                try:
                    fpath = os.path.join(label_dir, fname)
                    result = predictor.predict(fpath)
                    labels.append(label_id)
                    preds.append(1 if result["label"] == "FAKE" else 0)
                    probs.append(result["fake_prob"])
                except Exception:
                    continue

        if len(labels) < 4:
            return None, None, f"> ⚠️ Yetersiz test verisi ({len(labels)} görsel)."

        roc_fig = generate_roc_plot(labels, probs)
        cm_fig = generate_confusion_matrix_plot(labels, preds)
        summary = generate_metrics_summary(labels, preds, probs)
        summary += (
            f"\n\n> ℹ️ **Kaynak:** Eğitim test split'i (`faces_split/test/`) — "
            f"{len(labels)} görsel (random sample, seed=42)"
        )
        return roc_fig, cm_fig, summary
    except Exception as e:
        import traceback
        return None, None, f"Test split metrik hatası: {e}\n```\n{traceback.format_exc()}\n```"


def handle_cross_validation(n_folds=5, max_per_class=100):
    """K-Fold Cross Validation — test split üzerinde."""
    try:
        from inference.predictor import get_predictor
        from core.model_metrics import generate_roc_plot
        import os, random
        import numpy as np

        predictor = get_predictor()
        test_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                "dataset", "faces_split", "test")

        if not os.path.exists(test_dir):
            return None, "> ⚠️ Test split bulunamadı."

        # Veri topla
        all_items = []
        for label_name, label_id in [("real", 0), ("fake", 1)]:
            label_dir = os.path.join(test_dir, label_name)
            if not os.path.exists(label_dir):
                continue
            files = [f for f in os.listdir(label_dir)
                     if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            random.seed(42)
            selected = random.sample(files, min(max_per_class, len(files)))
            for fname in selected:
                all_items.append((os.path.join(label_dir, fname), label_id))

        if len(all_items) < n_folds * 4:
            return None, f"> ⚠️ Yetersiz veri ({len(all_items)}) — en az {n_folds*4} görsel gerekli."

        random.seed(42)
        random.shuffle(all_items)
        fold_size = len(all_items) // n_folds

        from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

        fold_results = []
        for fold in range(n_folds):
            start = fold * fold_size
            end = start + fold_size if fold < n_folds - 1 else len(all_items)
            fold_items = all_items[start:end]

            labels, preds_list, probs_list = [], [], []
            for fpath, label_id in fold_items:
                try:
                    result = predictor.predict(fpath)
                    labels.append(label_id)
                    preds_list.append(1 if result["label"] == "FAKE" else 0)
                    probs_list.append(result["fake_prob"])
                except Exception:
                    continue

            if len(labels) < 4:
                continue

            try:
                auc = roc_auc_score(labels, probs_list)
            except ValueError:
                auc = 0.5
            acc = accuracy_score(labels, preds_list)
            f1 = f1_score(labels, preds_list, zero_division=0)

            fold_results.append({
                "fold": fold + 1, "n": len(labels),
                "auc": auc, "acc": acc, "f1": f1,
            })

        if not fold_results:
            return None, "> ⚠️ Cross-validation başarısız."

        # Özet tablo
        aucs = [r["auc"] for r in fold_results]
        accs = [r["acc"] for r in fold_results]
        f1s = [r["f1"] for r in fold_results]

        md = (
            f"### {n_folds}-Fold Cross Validation\n\n"
            "| Fold | N | AUC | Accuracy | F1 |\n|---|---|---|---|---|\n"
        )
        for r in fold_results:
            md += f"| {r['fold']} | {r['n']} | {r['auc']:.4f} | {r['acc']*100:.1f}% | {r['f1']:.4f} |\n"

        md += (
            f"| **Ortalama** | | **{np.mean(aucs):.4f}** | **{np.mean(accs)*100:.1f}%** | **{np.mean(f1s):.4f}** |\n"
            f"| **Std** | | ±{np.std(aucs):.4f} | ±{np.std(accs)*100:.1f}% | ±{np.std(f1s):.4f} |\n"
        )

        # ROC grafiği (tüm fold birleşik)
        all_labels = [l for r in fold_results for l in ([0]*r["n"]//2 + [1]*r["n"]//2)]
        # Basit birleşik ROC
        all_l, all_p = [], []
        for fpath, label_id in all_items:
            try:
                result = predictor.predict(fpath)
                all_l.append(label_id)
                all_p.append(result["fake_prob"])
            except Exception:
                continue
        roc_fig = generate_roc_plot(all_l, all_p) if len(all_l) >= 4 else None

        return roc_fig, md
    except Exception as e:
        import traceback
        return None, f"Cross-validation hatası: {e}\n```\n{traceback.format_exc()}\n```"


def handle_test_new_dataset(folder_path):
    """Kullanıcının yüklediği yeni veri seti klasörüyle model test et.
    Klasör yapısı: folder/real/*.jpg + folder/fake/*.jpg beklenir."""
    try:
        from inference.predictor import get_predictor
        from core.model_metrics import (
            generate_roc_plot, generate_confusion_matrix_plot,
            generate_metrics_summary,
        )
        import os

        if not folder_path or not os.path.exists(folder_path):
            return None, None, "> ⚠️ Geçerli bir klasör yolu girin."

        predictor = get_predictor()
        labels, preds, probs = [], [], []

        for label_name, label_id in [("real", 0), ("fake", 1)]:
            label_dir = os.path.join(folder_path, label_name)
            if not os.path.exists(label_dir):
                continue
            files = [f for f in os.listdir(label_dir)
                     if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            for fname in files:
                try:
                    result = predictor.predict(os.path.join(label_dir, fname))
                    labels.append(label_id)
                    preds.append(1 if result["label"] == "FAKE" else 0)
                    probs.append(result["fake_prob"])
                except Exception:
                    continue

        if len(labels) < 4:
            return None, None, (
                f"> ⚠️ Yetersiz veri ({len(labels)} görsel).\n\n"
                "Klasör yapısı: `klasör/real/*.jpg` + `klasör/fake/*.jpg`"
            )

        roc = generate_roc_plot(labels, probs)
        cm = generate_confusion_matrix_plot(labels, preds)
        summary = generate_metrics_summary(labels, preds, probs)
        summary += f"\n\n> ℹ️ **Kaynak:** `{folder_path}` — {len(labels)} görsel"
        return roc, cm, summary
    except Exception as e:
        import traceback
        return None, None, f"Test hatası: {e}\n```\n{traceback.format_exc()}\n```"




def _build_architecture_md():
    """Model mimarisi detaylarını Markdown olarak üret."""
    from config import model_cfg
    return (
        "## 🏗️ DualPathDeepfakeDetector v5\n\n"
        "**Tri-Branch + CrossBranchTransformer Füzyon Mimarisi**\n\n"
        "```\n"
        "┌─────────────────────────────────────────────────────────────┐\n"
        "│                    GİRİŞ KATMANI                          │\n"
        "│  RGB (224×224×3)  │  Frekans (224×224×18)  │  Mesh (1404)  │\n"
        "└────────┬──────────┴──────────┬──────────────┴──────┬───────┘\n"
        "         ▼                     ▼                     ▼        \n"
        "┌────────────────┐  ┌────────────────────┐  ┌──────────────┐\n"
        "│ MobileNetV3-L  │  │  MobileNetV3-L     │  │  FaceMeshMLP │\n"
        "│ (pretrained)   │  │  (18-ch adapted)   │  │  1404→256    │\n"
        "│ → 960-dim      │  │  → 960-dim         │  │  →128→960    │\n"
        "└────────┬───────┘  └──────────┬──────────┘  └──────┬───────┘\n"
        "         └──────────────┬──────┘                     │        \n"
        "                       ▼                             │        \n"
        "              ┌────────────────────────────────────┐ │        \n"
        "              │   CrossBranchTransformer           │←┘        \n"
        "              │   2-Layer, 4-Head Self-Attention   │          \n"
        "              │   FF_MULT=2, Dropout=0.1           │          \n"
        "              └────────────────┬───────────────────┘          \n"
        "                               ▼                             \n"
        "                    ┌──────────────────┐                      \n"
        "                    │    Mean Pool      │                     \n"
        "                    │    → 960-dim      │                     \n"
        "                    └────────┬─────────┘                      \n"
        "                             ▼                                \n"
        "                  ┌──────────────────────┐                    \n"
        "                  │  Classifier Head     │                    \n"
        "                  │  960→256→2           │                    \n"
        "                  │  Dropout=0.5         │                    \n"
        "                  └──────────────────────┘                    \n"
        "                    REAL (0) / FAKE (1)                       \n"
        "```\n\n"
        "### Dallar\n\n"
        "| Dal | Backbone | Girdi | Çıktı | Açıklama |\n"
        "|-----|----------|-------|-------|----------|\n"
        f"| 🖼️ **RGB** | {model_cfg.RGB_BACKBONE} | 224×224×3 | {model_cfg.FUSION_DIM}-dim | Görsel özellikler (pretrained ImageNet) |\n"
        f"| 📡 **Frekans** | {model_cfg.FREQ_BACKBONE} | 224×224×{model_cfg.DWT_CHANNELS} | {model_cfg.FUSION_DIM}-dim | DWT+DCT+Phase hibrit frekans |\n"
        f"| 🔺 **Mesh** | FaceMeshMLP | {model_cfg.MESH_INPUT_DIM} | {model_cfg.FUSION_DIM}-dim | 468 yüz landmark koordinatları |\n\n"
        "### Frekans Kanalları (18-ch Hibrit)\n\n"
        f"| Kaynak | Wavelet | Kanal Sayısı |\n|---|---|---|\n"
        f"| DWT | {', '.join(model_cfg.DWT_WAVELETS)} | 12 (4 alt-band × 3 wavelet) |\n"
        "| DCT | Discrete Cosine | 3 |\n"
        "| Phase | Phase Spectrum | 3 |\n"
        f"| **Toplam** | | **{model_cfg.DWT_CHANNELS}** |\n\n"
        "### Füzyon\n\n"
        f"| Parametre | Değer |\n|---|---|\n"
        f"| Transformer Heads | {model_cfg.XBRANCH_HEADS} |\n"
        f"| Transformer Layers | {model_cfg.XBRANCH_LAYERS} |\n"
        f"| Attention Dropout | {model_cfg.XBRANCH_DROPOUT} |\n"
        f"| FF Multiplier | {model_cfg.XBRANCH_FF_MULT}× |\n"
        f"| Fusion Dimension | {model_cfg.FUSION_DIM} |\n"
        f"| Classifier Dropout | {model_cfg.CLASSIFIER_DROPOUT} |\n"
    )


def _build_training_config_md():
    """Eğitim konfigürasyonu ve yöntemlerini Markdown olarak üret."""
    from config import model_cfg

    return (
        "## ⚙️ Eğitim Konfigürasyonu\n\n"
        "### Temel Hiperparametreler\n\n"
        "| Parametre | Değer | Açıklama |\n|---|---|---|\n"
        f"| Learning Rate | {model_cfg.LEARNING_RATE} | OneCycleLR benzeri cosine |\n"
        f"| Weight Decay | {model_cfg.WEIGHT_DECAY} | L2 regularizasyon |\n"
        f"| Batch Size | {model_cfg.BATCH_SIZE} | GPU başına |\n"
        f"| Grad. Accumulation | {model_cfg.GRADIENT_ACCUMULATION_STEPS}× | Efektif batch = {model_cfg.BATCH_SIZE * model_cfg.GRADIENT_ACCUMULATION_STEPS} |\n"
        f"| Epochs | {model_cfg.EPOCHS} | Toplam eğitim süresi |\n"
        f"| Early Stopping | {model_cfg.EARLY_STOPPING_PATIENCE} epoch | Sabır süresi |\n"
        f"| Mixed Precision | {'✅ FP16' if model_cfg.USE_MIXED_PRECISION else '❌'} | VRAM tasarrufu |\n"
        f"| Gradient Clipping | {model_cfg.GRADIENT_CLIP_MAX_NORM} | Patlama önleme |\n"
        f"| Grad. Checkpointing | {'✅' if model_cfg.USE_GRADIENT_CHECKPOINTING else '❌'} | Bellek optimizasyonu |\n\n"
        "### Kayıp Fonksiyonları\n\n"
        "| Yöntem | Parametre | Ağırlık |\n|---|---|---|\n"
        f"| **Focal Loss** | γ={model_cfg.FOCAL_GAMMA}, α={model_cfg.FOCAL_ALPHA} | 0.8 |\n"
        f"| **Triplet Loss** | margin={model_cfg.CONTRASTIVE_MARGIN}, {model_cfg.CONTRASTIVE_DISTANCE} | {model_cfg.CONTRASTIVE_WEIGHT} |\n"
        f"| **Label Smoothing** | ε={model_cfg.LABEL_SMOOTHING} | — |\n\n"
        "### Veri Artırma (Augmentation)\n\n"
        "| Teknik | Parametre | Açıklama |\n|---|---|---|\n"
        f"| **MixUp** | α={model_cfg.MIXUP_ALPHA} | {'✅ Aktif' if model_cfg.USE_MIXUP else '❌ Devre dışı'} |\n"
        f"| **CutMix** | α={model_cfg.CUTMIX_ALPHA}, ratio={model_cfg.CUTMIX_RATIO} | %{int(model_cfg.CUTMIX_RATIO*100)} CutMix, %{int((1-model_cfg.CUTMIX_RATIO)*100)} MixUp |\n"
        f"| **FGSM Adversarial** | ε=[{model_cfg.FGSM_EPSILON_MIN}, {model_cfg.FGSM_EPSILON_MAX}] | {'✅ Her ' + str(model_cfg.FGSM_EVERY_N_STEPS) + ' adımda' if model_cfg.USE_FGSM_TRAINING else '❌'} |\n"
        f"| **SBI Augmentation** | — | Self-Blended Images |\n\n"
        "### Curriculum Learning\n\n"
        f"{'✅ **Aktif**' if model_cfg.USE_CURRICULUM else '❌ Devre Dışı'}\n\n"
        "| Epoch Aralığı | Hard-Real Oranı | Açıklama |\n|---|---|---|\n"
        + "\n".join(
            f"| {p['start']}–{min(p['end'], model_cfg.EPOCHS)} | %{int(p['hard_real_ratio']*100)} | "
            + ("Temel öğrenme" if p['hard_real_ratio'] == 0 else
               "Hafif zorlaştırma" if p['hard_real_ratio'] <= 0.15 else
               "Orta zorluk" if p['hard_real_ratio'] <= 0.30 else
               "Tam zorluk") + " |"
            for p in model_cfg.CURRICULUM_PHASES[:4]
        ) + "\n\n"
        "### Scheduler & Optimizasyon\n\n"
        "| Parametre | Değer |\n|---|---|\n"
        f"| Scheduler | CosineAnnealing (T_max={model_cfg.COSINE_T_MAX}) |\n"
        f"| Minimum LR | {model_cfg.COSINE_ETA_MIN} |\n"
        f"| Warmup | {model_cfg.WARMUP_EPOCHS} epoch |\n"
        f"| Backbone LR Factor | {model_cfg.BACKBONE_LR_FACTOR}× |\n"
        f"| Unfreeze Epoch | {model_cfg.UNFREEZE_EPOCH} |\n"
        f"| Plateau Factor | {model_cfg.PLATEAU_FACTOR} |\n"
        f"| Plateau Patience | {model_cfg.PLATEAU_PATIENCE} epoch |\n\n"
        "### Karar Eşikleri\n\n"
        "| Bölge | Koşul | Karar |\n|---|---|---|\n"
        f"| 🔴 FAKE | fake_prob ≥ {model_cfg.FAKE_THRESHOLD} | Sahte |\n"
        f"| 🟡 UNCERTAIN | {model_cfg.REAL_THRESHOLD} < fake_prob < {model_cfg.FAKE_THRESHOLD} | Belirsiz |\n"
        f"| 🟢 REAL | fake_prob ≤ {model_cfg.REAL_THRESHOLD} | Gerçek |\n"
    )


def _build_checkpoint_md():
    """Aktif checkpoint bilgilerini Markdown olarak üret."""
    import torch, os
    from config import paths

    ckpt_path = paths.BASE_DIR / "models" / "best_run5_forensic.pth"
    if not ckpt_path.exists():
        ckpt_path = paths.BASE_DIR / "models" / "best_model.pth"

    if not ckpt_path.exists():
        return "> ⚠️ Checkpoint dosyası bulunamadı."

    try:
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        size_mb = os.path.getsize(str(ckpt_path)) / (1024 * 1024)

        epoch = ckpt.get("epoch", "?")
        val_auc = ckpt.get("val_auc", ckpt.get("best_auc", "?"))
        val_acc = ckpt.get("val_acc", "?")
        val_f1 = ckpt.get("val_macro_f1", "?")
        run_name = ckpt.get("run", "?")
        freq_ch = ckpt.get("freq_channels", "?")
        hybrid = ckpt.get("use_hybrid_freq", "?")

        # Parametre sayısı
        try:
            param_count = sum(p.numel() for p in ckpt["model_state_dict"].values())
            param_count_str = f"{param_count:,} ({param_count/1e6:.1f}M)"
        except Exception:
            param_count_str = "?"

        val_auc_str = f"{val_auc:.4f}" if isinstance(val_auc, float) else str(val_auc)
        val_acc_str = f"{val_acc:.4f} ({val_acc*100:.1f}%)" if isinstance(val_acc, float) else str(val_acc)
        val_f1_str = f"{val_f1:.4f}" if isinstance(val_f1, float) else str(val_f1)

        return (
            "## 📦 Aktif Checkpoint\n\n"
            f"**Dosya:** `{ckpt_path.name}`\n\n"
            "| Bilgi | Değer |\n|---|---|\n"
            f"| Run | {run_name} |\n"
            f"| Epoch | {epoch} |\n"
            f"| **Val AUC** | **{val_auc_str}** |\n"
            f"| Val Accuracy | {val_acc_str} |\n"
            f"| Val Macro-F1 | {val_f1_str} |\n"
            f"| Frekans Kanalları | {freq_ch} |\n"
            f"| Hibrit Frekans | {'✅' if hybrid else '❌'} |\n"
            f"| Parametre Sayısı | {param_count_str} |\n"
            f"| Dosya Boyutu | {size_mb:.1f} MB |\n"
        )
    except Exception as e:
        return f"> ⚠️ Checkpoint okuma hatası: {e}"


def _build_dataset_md():
    """Veri seti özet bilgilerini Markdown olarak üret."""
    from config import paths

    base = paths.BASE_DIR / "dataset"
    splits_dir = base / "faces_split"
    jury_dir = base / "jury_test"

    def count_images(d):
        if not d.exists():
            return 0
        c = 0
        for ext in ["*.jpg", "*.png", "*.jpeg"]:
            c += len(list(d.glob(ext)))
        return c

    rows = []
    total = 0

    if splits_dir.exists():
        for split in ["train", "val", "test"]:
            for label in ["real", "fake"]:
                d = splits_dir / split / label
                n = count_images(d)
                total += n
                rows.append(f"| {split} | {label} | {n:,} |")

    jury_real = count_images(jury_dir / "real") if jury_dir.exists() else 0
    jury_fake = count_images(jury_dir / "fake") if jury_dir.exists() else 0

    md = (
        "## 📊 Veri Seti Özeti\n\n"
        "| Split | Sınıf | Görsel Sayısı |\n|---|---|---|\n"
        + "\n".join(rows) + "\n"
        f"| jury | real | {jury_real:,} |\n"
        f"| jury | fake | {jury_fake:,} |\n"
        f"| **TOPLAM** | | **{total + jury_real + jury_fake:,}** |\n\n"
        "### Veri Bölme Stratejisi\n\n"
        "| Parametre | Değer |\n|---|---|\n"
        "| Bölme Oranı | 70% Train / 15% Val / 15% Test |\n"
        "| Strateji | Kalite-bilinçli, identity-separated |\n"
        "| Leakage Kontrolü | pHash + MD5 çapraz doğrulama |\n"
        "| Jury Seti | Bağımsız (eğitimde kullanılmaz) |\n"
    )
    return md


