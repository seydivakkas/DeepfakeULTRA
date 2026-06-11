"""
Deepfake Detection System v3.0 — 8 Sekmeli Ana UI
python app.py → http://localhost:7860
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gradio as gr
from config import VERSION, SYSTEM_NAME

from ui.components import (
    handle_single_analysis, handle_dwt_analysis, handle_feedback, handle_pdf_download,
    handle_branch_knockout, handle_freq_ablation,
    handle_resolution_test, handle_double_compression,
    handle_analytics, handle_history, handle_clear_history, handle_chat,
    handle_finetune, handle_finetune_status, handle_rollback, handle_pool_status,
    handle_compression_sweep,
    handle_forensics,
    handle_model_profile,
    handle_cross_dataset_benchmark,
    handle_domain_ft_discover, handle_domain_ft_start,
    handle_domain_ft_status, handle_domain_ft_rollback,
    handle_model_switch,
    handle_jury_gallery, handle_jury_single_analysis, handle_jury_batch_test,
)
from ui.craniofacial_tab import create_craniofacial_tab

CSS = """
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Inter:wght@300;400;500;600;700&display=swap');
.gradio-container { font-family: 'Inter', system-ui, sans-serif !important; }
.app-title {
    text-align: center; font-family: 'JetBrains Mono', monospace;
    font-size: 2rem; font-weight: 700; letter-spacing: 2px;
    background: linear-gradient(135deg, #06B6D4, #22C55E);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    text-shadow: 0 0 30px rgba(6,182,212,0.3); margin-bottom: 4px;
}
.app-subtitle {
    text-align: center; font-size: 0.85rem; color: #94A3B8;
    font-family: 'JetBrains Mono', monospace; margin-bottom: 16px;
}
.watermark-banner {
    background: linear-gradient(135deg, #065F46, #047857); color: #D1FAE5;
    padding: 10px 16px; border-radius: 8px; font-size: 0.85rem; margin: 8px 0;
}
.verdict-fake { color: #EF4444; font-size: 1.4em; font-weight: 700; }
.verdict-real { color: #22C55E; font-size: 1.4em; font-weight: 700; }
.metric-card {
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    border: 1px solid rgba(6,182,212,0.3); border-radius: 12px;
    padding: 16px; text-align: center; color: #eaeaea;
}
.stat-value { font-size: 2em; font-weight: 700; color: #06B6D4; font-family: 'JetBrains Mono', monospace; }
.stat-label { font-size: 0.8em; color: #94A3B8; margin-top: 4px; }
.chat-hero {
    background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #0d1117 100%);
    border: 1px solid rgba(6,182,212,0.15); border-radius: 16px;
    padding: 28px 32px; margin-bottom: 16px; position: relative; overflow: hidden;
}
.chat-hero::before {
    content: ''; position: absolute; top: -50%; left: -50%; width: 200%; height: 200%;
    background: radial-gradient(circle at 30% 50%, rgba(6,182,212,0.06) 0%, transparent 50%);
}
.chat-hero h2 {
    font-family: 'JetBrains Mono', monospace; font-size: 1.4rem; font-weight: 700;
    background: linear-gradient(135deg, #06B6D4, #22C55E);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0 0 6px 0;
}
.chat-hero p { color: #8B949E; font-size: 0.85rem; margin: 0; line-height: 1.5; }
.chat-status {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 4px 12px; border-radius: 20px; font-size: 0.75rem;
    font-family: 'JetBrains Mono', monospace; margin-top: 10px;
}
.chat-status.online { background: rgba(34,197,94,0.1); color: #22C55E; border: 1px solid rgba(34,197,94,0.3); }
.chat-status.offline { background: rgba(239,68,68,0.1); color: #EF4444; border: 1px solid rgba(239,68,68,0.3); }
.chat-status .dot { width: 6px; height: 6px; border-radius: 50%; animation: pulse-dot 2s infinite; }
.chat-status.online .dot { background: #22C55E; }
.chat-status.offline .dot { background: #EF4444; }
@keyframes pulse-dot { 0%,100% { opacity: 1; } 50% { opacity: 0.4; } }
.quick-q-btn {
    background: linear-gradient(135deg, #161b22, #1c2333) !important;
    border: 1px solid rgba(6,182,212,0.2) !important; border-radius: 10px !important;
    color: #c9d1d9 !important; font-size: 0.82rem !important; padding: 10px 14px !important;
    transition: all 0.2s ease !important; cursor: pointer !important; text-align: left !important;
}
.quick-q-btn:hover {
    border-color: rgba(6,182,212,0.5) !important;
    background: linear-gradient(135deg, #1c2333, #22293a) !important;
    transform: translateY(-1px) !important; box-shadow: 0 4px 12px rgba(6,182,212,0.1) !important;
}
.ctx-card {
    background: linear-gradient(135deg, #0d1117, #131a24); border: 1px solid rgba(6,182,212,0.12);
    border-radius: 12px; padding: 14px 18px; font-size: 0.82rem; color: #8B949E;
}
.ctx-card strong { color: #c9d1d9; }
.ctx-card .label-fake { color: #EF4444; font-weight: 700; }
.ctx-card .label-real { color: #22C55E; font-weight: 700; }
/* Kompakt layout */
.gradio-accordion { margin-bottom: 4px !important; }
.gradio-accordion > .label-wrap { padding: 8px 12px !important; }
.gradio-accordion > .content { padding: 6px 10px !important; }
.prose table { font-size: 0.78rem !important; }
.prose table td, .prose table th { padding: 3px 8px !important; }
.prose h3 { margin-top: 4px !important; margin-bottom: 6px !important; font-size: 0.95rem !important; }
.prose p { margin-bottom: 4px !important; }
.plot-container { min-height: unset !important; }
.gradio-container .contain { gap: 4px !important; }
.gradio-container .gap { gap: 4px !important; }
.gradio-container .block { margin-top: 0 !important; }
"""

def create_app():
    with gr.Blocks(title=f"{SYSTEM_NAME} v{VERSION}") as demo:

        gr.HTML(f'<div class="app-title">🔍 {SYSTEM_NAME} v{VERSION}</div>')


        # Gizli state
        analysis_id_state = gr.State("")

        # SEKME 1: SINGLE IMAGE ANALYSIS
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        with gr.Tab("🔍 Single Image Analysis"):
            # Hero (kompakt)
            gr.HTML(
                '<div class="chat-hero" style="padding:12px 20px;margin-bottom:6px">'
                '<h2 style="font-size:1.1rem;margin:0 0 2px 0">🔍 Tek Görsel Deepfake Analizi</h2>'
                '<p style="margin:0;font-size:0.75rem">GradCAM++, EigenCAM, FastCAM, LIME ile XAI haritaları, '
                'frekans analizi, yüz geometrisi ve forensik tarama tek tıkla.</p>'
                '<div class="chat-status online"><span class="dot"></span> Model Hazır</div>'
                '</div>'
            )

            with gr.Row(equal_height=True):
                # ══ SOL PANEL: Girdi + Kontroller + Mini Grafikler ══
                with gr.Column(scale=1, min_width=200):
                    img_in = gr.Image(type="pil", label="📤 Görsel Yükle", height=160, format="png")
                    source_dd = gr.Dropdown(
                        choices=["original", "bilinmiyor", "twitter", "tiktok"],
                        value="original",
                        label="📱 Görsel Kaynağı",
                        info="sıkıştırma simülasyonu için",
                    )
                    tta_sl = gr.Slider(
                        5, 15, 8, step=1,
                        label="🎯 TTA Sayısı",
                        info="Yüksek → Daha güvenilir"
                    )
                    model_selector = gr.Dropdown(
                        choices=[
                            "🏆 Orijinal (AUC 0.982)",
                            "🌐 Generalized (Cross-Dataset)",
                        ],
                        value="🏆 Orijinal (AUC 0.982)",
                        label="🧠 Model Seçimi",
                        info="Orijinal = eğitim verisi uzmanı, Generalized = cross-dataset",
                    )
                    model_status = gr.Markdown(value="")
                    model_selector.change(
                        fn=handle_model_switch,
                        inputs=[model_selector],
                        outputs=[model_status],
                    )
                    btn_analyze = gr.Button("🔬 Analiz Et", variant="primary", size="lg")
                    watermark_md = gr.Markdown()
                    prob_plot = gr.Plot(label="📊 Olasılık Dağılımı")
                    tta_plot = gr.Plot(label="📈 TTA Dağılımı")

                # ══ ORTA PANEL: Karar + Detaylar + Platform + Forensik ══
                with gr.Column(scale=2, min_width=280):
                    verdict_md = gr.Markdown()
                    quality_warning_html = gr.HTML(value="")
                    with gr.Accordion("👤 Yüz Algılama Sonuçları", open=False):
                        face_img = gr.Image(label="Yüz Kutuları", height=100, format="png", buttons=['download', 'share', 'fullscreen'])
                    analysis_md = gr.Markdown()
                    with gr.Accordion("📱 Platform Tespiti — JPEG Forensik", open=True):
                        platform_detection_md = gr.Markdown(value="_Görsel yükleyin ve Analiz Et'e basın._")
                    with gr.Accordion("🔬 Forensik Analiz — ELA + Noise", open=True):
                        forensic_summary = gr.Markdown()
                    with gr.Accordion("🎯 Guided LIME — Sadece Yüz", open=True):
                        lime_out = gr.Image(label="Guided LIME", height=140, format="png", buttons=['download', 'share', 'fullscreen'])

                # ══ SAĞ PANEL: XAI + Yol Analizi 2x3 Grid ══
                with gr.Column(scale=2, min_width=320):
                    with gr.Row(equal_height=True):
                        with gr.Column(scale=1, min_width=140):
                            with gr.Accordion("🔥 GradCAM++", open=True):
                                gcam_out = gr.Image(label="GradCAM++", height=140, format="png", buttons=['download', 'share', 'fullscreen'])
                        with gr.Column(scale=1, min_width=140):
                            with gr.Accordion("🏗️ RGB Yolu", open=True):
                                rgb_path_img = gr.Image(label="🟢 RGB", height=140, format="png", buttons=['download', 'share', 'fullscreen'])
                    with gr.Row(equal_height=True):
                        with gr.Column(scale=1, min_width=140):
                            with gr.Accordion("🔷 EigenCAM", open=True):
                                ecam_out = gr.Image(label="EigenCAM", height=140, format="png", buttons=['download', 'share', 'fullscreen'])
                        with gr.Column(scale=1, min_width=140):
                            with gr.Accordion("🔵 Frekans Yolu", open=True):
                                dwt_img = gr.Image(label="Frekans (DWT)", height=140, format="png", buttons=['download', 'share', 'fullscreen'])
                    with gr.Row(equal_height=True):
                        with gr.Column(scale=1, min_width=140):
                            with gr.Accordion("⚡ FastCAM", open=True):
                                fcam_out = gr.Image(label="FastCAM", height=140, format="png", buttons=['download', 'share', 'fullscreen'])
                        with gr.Column(scale=1, min_width=140):
                            with gr.Accordion("🟡 Geometri Yolu", open=True):
                                mesh_path_img = gr.Image(label="Face Mesh", height=140, format="png", buttons=['download', 'share', 'fullscreen'])
                    with gr.Row(equal_height=True):
                        ela_out = gr.Image(label="🗺️ ELA", height=120, format="png", buttons=['download', 'share', 'fullscreen'])
                        noise_out = gr.Image(label="🔊 Gürültü", height=120, format="png", buttons=['download', 'share', 'fullscreen'])

            # ── Alt bar: PDF + Geri Bildirim + Active Learning ──
            with gr.Row():
                btn_pdf = gr.DownloadButton("📄 PDF İndir", variant="secondary")
                btn_pdf.click(fn=handle_pdf_download, outputs=[btn_pdf])
                with gr.Accordion("💬 Geri Bildirim", open=False):
                    with gr.Row():
                        btn_fb_real = gr.Button("✅ GERÇEK", variant="secondary", scale=1)
                        btn_fb_fake = gr.Button("🚨 SAHTE", variant="stop", scale=1)
                    fb_status = gr.Markdown()
                    hidden_id = gr.Textbox(visible=False)
                with gr.Accordion("🧠 Active Learning", open=False):
                    pool_status_md = gr.Markdown("_Havuz durumu_")
                    with gr.Row():
                        btn_finetune = gr.Button("🧠 Güncelle", variant="primary")
                        btn_rollback = gr.Button("↩️ Geri Al", variant="stop")
                        btn_refresh_pool = gr.Button("🔄 Havuz", variant="secondary")
                        btn_refresh_status = gr.Button("🔄 Durum", variant="secondary")
                    finetune_status_md = gr.Markdown()
                    btn_finetune.click(fn=handle_finetune, outputs=[finetune_status_md])
                    btn_rollback.click(fn=handle_rollback, outputs=[finetune_status_md])
                    btn_refresh_pool.click(fn=handle_pool_status, outputs=[pool_status_md])
                    btn_refresh_status.click(fn=handle_finetune_status, outputs=[finetune_status_md])





        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # SEKME 2: ROBUSTNESS TEST
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        with gr.Tab("🛡️ Robustness Test"):
            # Hero (kompakt)
            gr.HTML(
                '<div class="chat-hero" style="padding:14px 20px;margin-bottom:8px">'
                '<h2 style="font-size:1.1rem;margin:0 0 2px 0">🛡️ Robustness Test</h2>'
                '<p style="margin:0;font-size:0.75rem">Sıkıştırma, dal bağımlılığı, frekans ve çözünürlük dayanıklılığını tek tıkla test edin.</p>'
                '</div>'
            )

            # ── Buton + Durum ──
            with gr.Row(equal_height=True):
                rob_run_all = gr.Button("🚀 Tüm Testleri Çalıştır", variant="primary", size="lg")
                rob_status = gr.Markdown("⏳ Single Image Analysis'ten görsel yükleyin ve butona tıklayın…")

            # ── Compression Sweep ──
            with gr.Row(equal_height=True):
                comp_summary = gr.Markdown()
                comp_chart = gr.Plot(label="Compression Robustness")

            # ── 4 Test Bölümü TEK SATIRDA ──
            with gr.Row(equal_height=True):
                with gr.Column(scale=1, min_width=180):
                    with gr.Accordion("🔬 Knockout", open=True):
                        ko_result = gr.Markdown()
                with gr.Column(scale=1, min_width=180):
                    with gr.Accordion("📡 Frekans", open=True):
                        freq_abl_result = gr.Markdown()
                with gr.Column(scale=1, min_width=180):
                    with gr.Accordion("📐 Çözünürlük", open=True):
                        res_result = gr.Markdown()
                with gr.Column(scale=1, min_width=180):
                    with gr.Accordion("🔄 Çift Sıkıştırma", open=True):
                        dc_result = gr.Markdown()

            # Çözünürlük grafiği gizli (artık tablo yeterli)
            res_chart = gr.Plot(visible=False)

            # ── Tek butonla tüm testleri çalıştır ──
            def run_all_robustness(image):
                if image is None:
                    empty = "Görsel yükleyin"
                    return empty, None, empty, empty, empty, None, empty, "❌ Görsel yok"
                status_parts = []
                try:
                    c_sum, c_chart = handle_compression_sweep(image)
                    status_parts.append("✅ Compression")
                except Exception as e:
                    c_sum, c_chart = f"Hata: {e}", None
                    status_parts.append("❌ Compression")
                try:
                    ko = handle_branch_knockout(image)
                    status_parts.append("✅ Knockout")
                except Exception as e:
                    ko = f"Hata: {e}"
                    status_parts.append("❌ Knockout")
                try:
                    fa = handle_freq_ablation(image)
                    status_parts.append("✅ Frekans")
                except Exception as e:
                    fa = f"Hata: {e}"
                    status_parts.append("❌ Frekans")
                try:
                    rr, rc = handle_resolution_test(image)
                    status_parts.append("✅ Çözünürlük")
                except Exception as e:
                    rr, rc = f"Hata: {e}", None
                    status_parts.append("❌ Çözünürlük")
                try:
                    dc = handle_double_compression(image)
                    status_parts.append("✅ Çift Sıkıştırma")
                except Exception as e:
                    dc = f"Hata: {e}"
                    status_parts.append("❌ Çift Sıkıştırma")

                status = "### 🏁 Test Sonuçları\n" + " | ".join(status_parts)
                return c_sum, c_chart, ko, fa, rr, rc, dc, status

            rob_run_all.click(
                fn=run_all_robustness,
                inputs=[img_in],
                outputs=[comp_summary, comp_chart, ko_result, freq_abl_result,
                         res_result, res_chart, dc_result, rob_status]
            )

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # SEKME 3: ANALYTICS DASHBOARD
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        with gr.Tab("📊 Analytics Dashboard"):
            # ── Kontroller + İstatistikler: TEK SATIR ──
            with gr.Row(equal_height=True):
                dash_days = gr.Slider(7, 90, 30, step=1, label="📅 Son N Gün")
                dash_btn = gr.Button("🔄 Yenile", variant="primary", size="sm")
                stat_total = gr.Textbox(label="📊 Toplam", interactive=False)
                stat_fake = gr.Textbox(label="🔴 Sahte", interactive=False)
                stat_real = gr.Textbox(label="🟢 Gerçek", interactive=False)
                stat_rate = gr.Textbox(label="📈 Oran", interactive=False)

            # ── 2x2 Grafik Grid (tam genişlik) ──
            with gr.Row(equal_height=True):
                dash_daily = gr.Plot(label="📅 Günlük Analiz")
                dash_dist = gr.Plot(label="🎯 FAKE vs REAL")
            with gr.Row(equal_height=True):
                dash_source = gr.Plot(label="📱 Kaynak Tipi")
                dash_trend = gr.Plot(label="📊 Doğruluk Trendi")

            def run_full_dashboard(days):
                try:
                    analytics_out = handle_analytics(days)
                except Exception:
                    analytics_out = ("0", "0", "0", "0%", None, None, None, None)
                return analytics_out

            dash_btn.click(
                fn=run_full_dashboard,
                inputs=[dash_days],
                outputs=[
                    stat_total, stat_fake, stat_real, stat_rate,
                    dash_daily, dash_dist, dash_source, dash_trend,
                ]
            )

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # SEKME 4: MODEL PROFİLİ & METRİKLER
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        with gr.Tab("🧬 Model Profili"):
            gr.HTML(
                '<div class="chat-hero" style="padding:12px 20px;margin-bottom:6px">'
                '<h2 style="font-size:1.1rem;margin:0 0 2px 0">🧬 Model Profili & Performans</h2>'
                '<p style="margin:0;font-size:0.75rem">DualPathDeepfakeDetector v5 — Mimari, eğitim, checkpoint ve canlı metrikler.</p>'
                '</div>'
            )

            mp_btn = gr.Button("🔄 Model Profilini Yükle", variant="primary", size="sm")

            # ── Üst: Test Split ROC + CM (ana metrikler) ──
            gr.Markdown("##### 📈 Eğitim Test Split Performansı (`faces_split_v2/test/`)")
            with gr.Row(equal_height=True):
                mp_roc = gr.Plot(label="📈 ROC (Test Split)")
                mp_cm = gr.Plot(label="🎯 CM (Test Split)")
            mp_metrics = gr.Markdown()

            # ── Detaylar: kapalı accordion'lar ──
            with gr.Row(equal_height=True):
                with gr.Column(scale=1, min_width=250):
                    with gr.Accordion("📦 Checkpoint & Veri Seti", open=False):
                        mp_ckpt = gr.Markdown("_Yüklemek için butona tıklayın…_")
                        mp_dataset = gr.Markdown("")
                with gr.Column(scale=1, min_width=250):
                    with gr.Accordion("🏗️ Model Mimarisi", open=False):
                        mp_arch = gr.Markdown("_Yükleniyor…_")

            with gr.Accordion("⚙️ Eğitim Konfigürasyonu", open=False):
                mp_train = gr.Markdown("_Yükleniyor…_")

            mp_btn.click(
                fn=handle_model_profile,
                outputs=[mp_arch, mp_train, mp_ckpt, mp_dataset, mp_roc, mp_cm, mp_metrics]
            )

            with gr.Accordion("🌐 Cross-Dataset Benchmark", open=False):
                gr.Markdown(
                    "Tüm harici veri setleri (`external_tests/`, `jury_test/`, `faces_split_v2/test/`) "
                    "üzerinden otomatik benchmark. Karşılaştırmalı AUC, F1, ROC overlay."
                )
                with gr.Row():
                    bench_samples = gr.Slider(
                        50, 500, 200, step=50,
                        label="Sınıf Başına Maks. Örnek",
                        info="Düşük = hızlı, yüksek = doğru",
                    )
                    bench_btn = gr.Button(
                        "🚀 Benchmark Başlat", variant="primary", size="sm",
                    )
                bench_progress = gr.Markdown("_Butona tıklayarak başlatın…_")
                bench_table = gr.Markdown()
                with gr.Row(equal_height=True):
                    bench_radar = gr.Plot(label="📊 Radar Chart")
                    bench_roc = gr.Plot(label="📈 ROC Overlay")
                bench_summary = gr.Markdown()

                bench_btn.click(
                    fn=handle_cross_dataset_benchmark,
                    inputs=[bench_samples],
                    outputs=[bench_progress, bench_table, bench_radar, bench_roc, bench_summary],
                )

            with gr.Accordion("🧬 Domain Fine-Tune", open=False):
                gr.Markdown(
                    "Harici veri setleri üzerinden modeli fine-tune ederek **domain generalization** "
                    "yeteneğini artırın. Tam tri-branch pipeline (RGB+Freq+Mesh) kullanılır."
                )
                with gr.Row():
                    dft_info = gr.Markdown("_Dataset'leri keşfetmek için butona tıklayın._")
                dft_discover_btn = gr.Button("🔍 Veri Setlerini Keşfet", size="sm")
                dft_datasets = gr.CheckboxGroup(
                    choices=[], label="Fine-Tune İçin Veri Setleri Seçin",
                    info="50/50 dengeli örnekleme uygulanır",
                )
                with gr.Row():
                    dft_epochs = gr.Slider(
                        3, 20, 8, step=1, label="Epoch",
                        info="Önerilen: 5-10",
                    )
                    dft_lr = gr.Slider(
                        1e-6, 1e-4, 5e-5, step=1e-6, label="Learning Rate",
                        info="Düşük = güvenli, yüksek = hızlı",
                    )
                with gr.Row():
                    dft_start_btn = gr.Button("🚀 Fine-Tune Başlat", variant="primary", size="sm")
                    dft_status_btn = gr.Button("📊 Durum", size="sm")
                    dft_rollback_btn = gr.Button("↩️ Rollback", variant="stop", size="sm")
                dft_result = gr.Markdown()

                def _discover_and_update():
                    choices, info = handle_domain_ft_discover()
                    return gr.update(choices=choices, value=[]), info

                dft_discover_btn.click(
                    fn=_discover_and_update,
                    outputs=[dft_datasets, dft_info],
                )
                dft_start_btn.click(
                    fn=handle_domain_ft_start,
                    inputs=[dft_datasets, dft_epochs, dft_lr],
                    outputs=[dft_result],
                )
                dft_status_btn.click(
                    fn=handle_domain_ft_status,
                    outputs=[dft_result],
                )
                dft_rollback_btn.click(
                    fn=handle_domain_ft_rollback,
                    outputs=[dft_result],
                )

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # SEKME 5: ANALİZ GEÇMİŞİ
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        with gr.Tab("🕒 Analiz Geçmişi"):
            gr.Markdown("### Analiz Geçmişi\nSQLite tabanlı kalıcı geçmiş — uygulama yeniden başlatılsa bile korunur.")
            with gr.Row():
                hist_limit = gr.Slider(10, 200, 200, step=10, label="Gösterilecek Kayıt")
                hist_btn = gr.Button("🔄 Yenile", variant="primary")
                hist_clear = gr.Button("🗑️ Geçmişi Temizle", variant="stop")
            hist_table = gr.Markdown()
            hist_info = gr.Markdown()
            hist_btn.click(fn=handle_history, inputs=[hist_limit], outputs=[hist_table, hist_info])
            hist_clear.click(fn=handle_clear_history, outputs=[hist_table, hist_info])

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # SEKME 6: KRANİYOFASİYAL BİYOMETRİK ANALİZ
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        with gr.Tab("🧬 Yüz Anatomisi"):
            create_craniofacial_tab()

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # SEKME 7: JÜRI GALERİSİ
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        with gr.Tab("🏆 Jüri Galerisi"):
            gr.HTML(
                '<div class="chat-hero" style="padding:14px 20px;margin-bottom:8px">'
                '<h2 style="font-size:1.1rem;margin:0 0 2px 0">🏆 Jüri Demo V2 — Görsel Galeri</h2>'
                '<p style="margin:0;font-size:0.75rem">100 Real + 100 Fake yüksek güvenli görsel. '
                'Görsele tıklayarak detaylı analiz yapın veya batch test çalıştırın.</p>'
                '<div class="chat-status online"><span class="dot"></span> Galeri Hazır</div>'
                '</div>'
            )

            with gr.Row(equal_height=True):
                jury_load_btn = gr.Button("📂 Galeriyi Yükle", variant="primary", size="sm")
                jury_model_selector = gr.Dropdown(
                    choices=[
                        "🏆 Orijinal (AUC 0.982)",
                        "🌐 Generalized (Cross-Dataset)",
                    ],
                    value="🏆 Orijinal (AUC 0.982)",
                    label="🧠 Model Seçimi",
                    info="Analiz ve batch test için kullanılacak model",
                    scale=2,
                )
                jury_batch_btn = gr.Button("🧪 Tüm Görselleri Test Et", variant="secondary", size="sm")
            jury_model_status = gr.Markdown(value="")
            jury_info = gr.Markdown("_Galeriyi yüklemek için butona tıklayın._")

            # Model seçimi değiştiğinde — jüri sekmesinden de model değiştirebilsin
            jury_model_selector.change(
                fn=handle_model_switch,
                inputs=[jury_model_selector],
                outputs=[jury_model_status],
            )

            with gr.Row(equal_height=True):
                with gr.Column(scale=3, min_width=400):
                    jury_gallery = gr.Gallery(
                        label="📸 Jüri Demo V2",
                        columns=5,
                        rows=4,
                        height=500,
                        object_fit="cover",
                        allow_preview=True,
                    )
                with gr.Column(scale=1, min_width=250):
                    jury_selected_img = gr.Image(
                        label="Seçilen Görsel", height=200, format="png",
                        interactive=False,
                    )
                    jury_analyze_btn = gr.Button("🔬 Seçili Görseli Analiz Et", variant="primary", size="sm")
                    jury_result_md = gr.Markdown("_Bir görsel seçin ve analiz edin._")

            # Batch test sonuçları
            with gr.Row(equal_height=True):
                jury_batch_report = gr.Markdown()
            with gr.Row(equal_height=True):
                jury_batch_chart = gr.Plot(label="📊 Accuracy")
                jury_cm_chart = gr.Plot(label="🎯 Confusion Matrix")

            # Event bindings
            jury_load_btn.click(
                fn=handle_jury_gallery,
                outputs=[jury_gallery, jury_info],
            )

            def _gallery_select(evt: gr.SelectData):
                """Galeriden görsel seçildiğinde."""
                if evt and evt.value:
                    img_data = evt.value
                    if isinstance(img_data, dict):
                        return img_data.get("image", {}).get("path", None)
                    elif isinstance(img_data, str):
                        return img_data
                return None

            jury_gallery.select(
                fn=_gallery_select,
                outputs=[jury_selected_img],
            )

            jury_analyze_btn.click(
                fn=handle_jury_single_analysis,
                inputs=[jury_selected_img],
                outputs=[jury_result_md],
            )

            jury_batch_btn.click(
                fn=handle_jury_batch_test,
                outputs=[jury_batch_report, jury_batch_chart, jury_cm_chart],
            )

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # SEKME 8: ANALİZ ASİSTANI
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        with gr.Tab("💬 Analiz Asistanı"):
            # Hero başlık
            gr.HTML(
                '<div class="chat-hero">'
                '<h2>🧠 DeepfakeULTRA Analiz Asistanı</h2>'
                '<p>Görsel analiz sonuçlarını yorumlayın, XAI haritalarını anlatın, '
                'model mimarisini keşfedin. Gemini 3.0 Pro destekli.</p>'
                '<div class="chat-status online"><span class="dot"></span> Asistan Hazır</div>'
                '</div>'
            )

            with gr.Row():
                # Sol: Chat alanı
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(
                        label="💬 Analiz Asistanı",
                        height=480,
                        placeholder="🔍 Bir görsel analiz yapın, sonra bana sorun...",
                    )
                    with gr.Row():
                        chat_msg = gr.Textbox(
                            label="Mesajınız",
                            placeholder="Örn: Model neden buna Fake dedi? GradCAM haritasini yorumla...",
                            scale=5, lines=1, max_lines=3,
                        )
                        chat_send = gr.Button("➤ Gönder", variant="primary", scale=1, min_width=100)
                    with gr.Row():
                        chat_clear = gr.Button("🗑️ Sohbeti Temizle", variant="stop", size="sm")

                # Sağ: Panel
                with gr.Column(scale=1, min_width=280):
                    # API Ayarları
                    with gr.Accordion("⚙️ API Yapılandırması", open=False):
                        api_key_in = gr.Textbox(
                            label="Gemini API Key", type="password",
                            placeholder="AI Studio API anahtarınız",
                            info="Girilmezse yerel bilgi tabanı + Ollama kullanılır",
                        )

                    # Hızlı Sorular
                    gr.Markdown("#### ⚡ Hızlı Sorular")
                    q1 = gr.Button("🔍 Model neden FAKE dedi?", elem_classes="quick-q-btn")
                    q2 = gr.Button("🌡️ GradCAM haritasini yorumla", elem_classes="quick-q-btn")
                    q3 = gr.Button("🌊 DWT frekans analizi nedir?", elem_classes="quick-q-btn")
                    q4 = gr.Button("🏗️ Model mimarisi nasil calisir?", elem_classes="quick-q-btn")
                    q5 = gr.Button("📱 Platform sikistirma etkisi?", elem_classes="quick-q-btn")
                    q6 = gr.Button("🧠 Active Learning nasil calisir?", elem_classes="quick-q-btn")

                    # Bağlam Kartı
                    gr.Markdown("#### 📊 Analiz Bağlamı")
                    ctx_display = gr.HTML(
                        '<div class="ctx-card">'
                        '<strong>⏳ Henüz analiz yapılmadı</strong><br>'
                        'Single Image sekmesinden bir görsel analiz edin, '
                        'sonuçlar otomatik olarak buraya yansıyacaktır.'
                        '</div>'
                    )

            # Event bindings
            def send_quick(question, history, api_key):
                return handle_chat(question, history, api_key)

            # --- btn_analyze burada baglaniyor (ctx_display tanimlandiktan sonra) ---
            btn_analyze.click(
                fn=handle_single_analysis, inputs=[img_in, tta_sl, source_dd],
                outputs=[img_in, verdict_md, analysis_md, face_img,
                         gcam_out, ecam_out, fcam_out, lime_out,
                         prob_plot, tta_plot, watermark_md, hidden_id, fb_status,
                         quality_warning_html, platform_detection_md, ctx_display]
            ).then(
                fn=handle_dwt_analysis, inputs=[img_in],
                outputs=[rgb_path_img, dwt_img, mesh_path_img]
            ).then(
                fn=handle_forensics, inputs=[img_in],
                outputs=[ela_out, noise_out, forensic_summary]
            )
            btn_fb_real.click(fn=lambda aid: handle_feedback(aid, "REAL"), inputs=[hidden_id], outputs=[fb_status])
            btn_fb_fake.click(fn=lambda aid: handle_feedback(aid, "FAKE"), inputs=[hidden_id], outputs=[fb_status])

            chat_send.click(fn=handle_chat, inputs=[chat_msg, chatbot, api_key_in],
                            outputs=[chat_msg, chatbot])
            chat_msg.submit(fn=handle_chat, inputs=[chat_msg, chatbot, api_key_in],
                            outputs=[chat_msg, chatbot])
            chat_clear.click(fn=lambda: ([], ""), outputs=[chatbot, chat_msg])

            # Hızlı soru butonları — direkt handle_chat (tek round-trip)
            for btn, q_text in [
                (q1, "Neden FAKE dedi? Analiz sonuclarini yorumla."),
                (q2, "GradCAM++ haritasini yorumla."),
                (q3, "DWT frekans analizi nedir ve nasil calisir?"),
                (q4, "Model mimarisi nasil calisir? DualPath yapisi nedir?"),
                (q5, "Platform sikistirma deepfake tespitini nasil etkiler?"),
                (q6, "Active Learning ve geri bildirim sistemi nasil calisir?"),
            ]:
                btn.click(
                    fn=lambda history, api_key, q=q_text: handle_chat(q, history, api_key),
                    inputs=[chatbot, api_key_in],
                    outputs=[chat_msg, chatbot],
                )

    return demo


if __name__ == "__main__":
    demo = create_app()
    demo.launch(
        server_name="0.0.0.0", server_port=7860, share=False,
        css=CSS, theme=gr.themes.Base(
            primary_hue="cyan",
            neutral_hue="gray",
        ),
    )
