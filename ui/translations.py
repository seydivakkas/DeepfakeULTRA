"""
Deepfake Detection System v3.0 — Çok Dillilik (TR/EN)
"""

TRANSLATIONS = {
    "tr": {
        # Genel
        "app_title": "🔍 Deepfake Detection System V3.0",
        "app_subtitle": "AI-powered deepfake analysis — GradCAM++ · XAI · TTA · Batch · Adversarial · Federated",
        "language": "🌐 Dil / Language",
        "dark_mode": "🌙 Dark mod",
        "analyze": "🔬 Analiz Et",
        "refresh": "🔄 Yenile",
        "download": "📄 İndir",
        "clear": "Temizle",
        "send": "Gönder",
        "upload": "Yüklemek için tıkla",
        "drag_drop": "Resmi buraya sürükle",
        "loading": "Yükleniyor...",

        # Sekmeler
        "tab_single": "🔍 Single Image Analysis",
        "tab_compare": "👥 Compare Two Images",
        "tab_video": "🎥 Video Analysis",
        "tab_robustness": "🛡️ Robustness Test",
        "tab_analytics": "📊 Analytics Dashboard",
        "tab_batch": "🔥 Batch Analysis",
        "tab_history": "🕒 Analiz Geçmişi",
        "tab_assistant": "💬 Analiz Asistanı",

        # Sonuçlar
        "verdict": "Karar",
        "fake_prob": "Sahte Olasılığı",
        "real_prob": "Gerçek Olasılığı",
        "confidence": "Güven",
        "calibrated": "Kalibre",
        "tta_augmentations": "TTA Augmentasyon Sayısı",
        "tta_std": "TTA Standart Sapma",
        "tta_individual": "TTA Bireysel",
        "gradcam_score": "GradCAM++ Skoru",
        "counterfactual": "Counterfactual Prob",

        # XAI
        "xai_gradcam": "ⓘ GradCAM++ (Gradient-based)",
        "xai_eigencam": "ⓘ EigenCAM (SVD-based)",
        "xai_fastcam": "ⓘ FastCAM (SMOE-based)",
        "xai_lime": "🎯 Güdümlü LIME (Sadece Yüz)",
        "face_boxes": "Tespit Edilen Yüz Kutuları",

        # Mimari
        "architecture_title": "🏗️ Mimari İç İşleyişi (Dual-Stream)",
        "run_dwt": "🏃 Run DWT Analysis (Frekans Gözü)",
        "dwt_title": "ⓘ Multi-Scale Frequency Map (DWT)",
        "fusion_weights": "Öğrenilebilir Füzyon Ağırlıkları",
        "rgb_path": "RGB Yolu",
        "freq_path": "Frekans Yolu",
        "geo_path": "Geometri Yolu",
        "fusion_desc": "Bu kısım, modelin karar verirken hangi 'yol'dan gelen bilgiye ne kadar güvendiğini gösterir.",

        # Watermark
        "watermark_banner": "✅ Tüm analiz edilen görüntülere güvenlik amacıyla görünür watermark eklenmiştir.",

        # Feedback
        "feedback_title": "Geri Bildirim (Sürekli Öğrenme)",
        "feedback_real": "✅ Bu resim GERÇEKTİ",
        "feedback_fake": "🚨 Bu resim SAHTEYDİ",
        "feedback_pool": "Havuz Durumu",

        # PDF
        "download_pdf": "📄 PDF Raporu İndir",

        # Adversarial
        "adv_title": "Adversarial Robustness Testing",
        "adv_desc": "Modelin saldırılara karşı dayanıklılığını test edin.",
        "attack_type": "Saldırı Türü",
        "epsilon": "Epsilon (ε)",
        "epsilon_desc": "Pertürbasyon büyüklüğü — küçük=görünmez, büyük=güçlü",
        "run_attack": "✓ Run Attack",
        "epsilon_sweep": "📈 Epsilon Sweep",
        "attack_result": "Saldırı Sonucu",

        # Analytics
        "total_analyses": "Toplam Analiz",
        "fake_count": "Sahte",
        "real_count": "Gerçek",
        "fake_rate": "Fake Oranı",
        "last_n_days": "Son N Gün",
        "daily_chart": "Günlük Analiz Sayısı",
        "distribution_chart": "FAKE vs REAL Dağılımı",
        "histogram_chart": "Fake Olasılık Histogramı",
        "source_chart": "Kaynak Tipi Dağılımı",
        "trend_chart": "Model Doğruluk Trendi",
        "xai_usage_chart": "En Çok Kullanılan XAI Yöntemleri",

        # Batch
        "batch_title": "🔥 Toplu Görüntü Analizi",
        "batch_desc": "ZIP dosyası veya birden fazla görsel yükleyin, tüm sonuçları CSV/Excel/PDF olarak indirin.",
        "batch_start": "▶ Toplu Analizi Başlat",
        "export_format": "Dışa Aktarım Formatı",
        "batch_status": "Durum",
        "batch_download": "↓ İndir",
        "batch_pdf": "📄 Toplu PDF Rapor",

        # Geçmiş
        "history_title": "Analiz Geçmişi",
        "history_desc": "SQLite tabanlı kalıcı geçmiş — uygulama yeniden başlatılsa bile korunur.",
        "records_to_show": "Gösterilecek Kayıt",
        "clear_history": "🗑️ Geçmişi Temizle",
        "showing_records": "kayıt gösteriliyor",

        # Asistan
        "assistant_title": "Analiz Sohbet Asistanı",
        "assistant_warning": "Görsel analiz yapıldıktan sonra model çıktıları hakkında asistana soru sorabilirsiniz.",
        "gemini_key": "Gemini API Key",
        "ollama_fallback": "Ollama fallback: API key girilmezse yerel Ollama (qwen2.5:7b) kullanılır.",
        "question_placeholder": "Örn: Model neden buna Fake dedi?",
        "clear_chat": "Sohbeti Temizle",
    },
    "en": {
        "app_title": "🔍 Deepfake Detection System V3.0",
        "app_subtitle": "AI-powered deepfake analysis — GradCAM++ · XAI · TTA · Batch · Adversarial · Federated",
        "language": "🌐 Language",
        "dark_mode": "🌙 Dark mode",
        "analyze": "🔬 Analyze",
        "refresh": "🔄 Refresh",
        "download": "📄 Download",
        "clear": "Clear",
        "send": "Send",
        "upload": "Click to upload",
        "drag_drop": "Drag image here",
        "loading": "Loading...",

        "tab_single": "🔍 Single Image Analysis",
        "tab_compare": "👥 Compare Two Images",
        "tab_video": "🎥 Video Analysis",
        "tab_robustness": "🛡️ Robustness Test",
        "tab_analytics": "📊 Analytics Dashboard",
        "tab_batch": "🔥 Batch Analysis",
        "tab_history": "🕒 Analysis History",
        "tab_assistant": "💬 Analysis Assistant",

        "verdict": "Verdict",
        "fake_prob": "Fake Probability",
        "real_prob": "Real Probability",
        "confidence": "Confidence",
        "calibrated": "Calibrated",
        "tta_augmentations": "TTA Augmentations",
        "tta_std": "TTA Std Dev",
        "tta_individual": "TTA Individual",
        "gradcam_score": "GradCAM++ Score",
        "counterfactual": "Counterfactual Prob",

        "xai_gradcam": "ⓘ GradCAM++ (Gradient-based)",
        "xai_eigencam": "ⓘ EigenCAM (SVD-based)",
        "xai_fastcam": "ⓘ FastCAM (SMOE-based)",
        "xai_lime": "🎯 Guided LIME (Face Only)",
        "face_boxes": "Detected Face Bounding Boxes",

        "architecture_title": "🏗️ Architecture Internals (Dual-Stream)",
        "run_dwt": "🏃 Run DWT Analysis (Frequency Eye)",
        "dwt_title": "ⓘ Multi-Scale Frequency Map (DWT)",
        "fusion_weights": "Learnable Fusion Weights",
        "rgb_path": "RGB Path",
        "freq_path": "Frequency Path",
        "geo_path": "Geometry Path",
        "fusion_desc": "Shows how much the model trusts each pathway when making decisions.",

        "watermark_banner": "✅ All analyzed images have been watermarked for security purposes.",

        "feedback_title": "Feedback (Continuous Learning)",
        "feedback_real": "✅ This image was REAL",
        "feedback_fake": "🚨 This image was FAKE",
        "feedback_pool": "Pool Status",

        "download_pdf": "📄 Download PDF Report",

        "adv_title": "Adversarial Robustness Testing",
        "adv_desc": "Test model robustness against adversarial attacks.",
        "attack_type": "Attack Type",
        "epsilon": "Epsilon (ε)",
        "epsilon_desc": "Perturbation magnitude — small=invisible, large=powerful",
        "run_attack": "✓ Run Attack",
        "epsilon_sweep": "📈 Epsilon Sweep",
        "attack_result": "Attack Result",

        "total_analyses": "Total Analyses",
        "fake_count": "Fake",
        "real_count": "Real",
        "fake_rate": "Fake Rate",
        "last_n_days": "Last N Days",
        "daily_chart": "Daily Analysis Count",
        "distribution_chart": "FAKE vs REAL Distribution",
        "histogram_chart": "Fake Probability Histogram",
        "source_chart": "Source Type Distribution",
        "trend_chart": "Model Accuracy Trend",
        "xai_usage_chart": "Most Used XAI Methods",

        "batch_title": "🔥 Batch Image Analysis",
        "batch_desc": "Upload a ZIP file or multiple images, download all results as CSV/Excel/PDF.",
        "batch_start": "▶ Start Batch Analysis",
        "export_format": "Export Format",
        "batch_status": "Status",
        "batch_download": "↓ Download",
        "batch_pdf": "📄 Batch PDF Report",

        "history_title": "Analysis History",
        "history_desc": "SQLite-based persistent history — preserved even after app restart.",
        "records_to_show": "Records to Show",
        "clear_history": "🗑️ Clear History",
        "showing_records": "records shown",

        "assistant_title": "Analysis Chat Assistant",
        "assistant_warning": "After analyzing an image, you can ask the assistant about model outputs.",
        "gemini_key": "Gemini API Key",
        "ollama_fallback": "Ollama fallback: If no API key, local Ollama (qwen2.5:7b) is used.",
        "question_placeholder": "E.g.: Why did the model say this is Fake?",
        "clear_chat": "Clear Chat",
    },
}


def t(key: str, lang: str = "tr") -> str:
    """Çeviri helper fonksiyonu."""
    return TRANSLATIONS.get(lang, TRANSLATIONS["tr"]).get(key, key)
