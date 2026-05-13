"""
Deepfake Detection System v3 — LLM Asistan + Yerel NLP Bilgi Tabani
Gemini (birincil) + Ollama (fallback) + Yerel Bilgi Tabani (offline).
"""
import os
import re
from config import llm_cfg

SYSTEM_PROMPT = """Sen bir deepfake ve yuz tespit uzmanisin. Kullanicinin yukledigi goruntulerin
analiz sonuclarini yorumla. GradCAM++, EigenCAM, Counterfactual XAI haritalarini acikla.
Yuz algilama, landmark tespiti, frekans analizi, mesh geometri ve manipulasyon teknikleri
hakkinda derin bilgi sahibisin. Turkce yanit ver. Teknik ama anlasilir ol."""

# ── Yerel NLP Bilgi Tabani ──
KNOWLEDGE_BASE = {
    "gradcam": {
        "keywords": ["gradcam", "grad-cam", "grad cam", "isı haritası", "isi haritasi", "heatmap", "saliency"],
        "answer": (
            "**GradCAM++ (Gradient-weighted Class Activation Mapping)**\n\n"
            "Modelin karar verirken goruntunun hangi bolgesine odaklandigini gosteren bir XAI yontemidir.\n\n"
            "**Nasil calisir:**\n"
            "1. Modelin son konvolusyon katmanindan gradyanlar alinir\n"
            "2. Bu gradyanlar global average pooling ile agirlik haline getirilir\n"
            "3. Ozellik haritalariyla carpilarak isi haritasi olusturulur\n\n"
            "**Yorum:**\n"
            "- 🔴 Kirmizi bolgeler = Model buralara odaklaniyor (suphe edilen alan)\n"
            "- 🔵 Mavi bolgeler = Model bu alanlari gormezden geliyor\n"
            "- Deepfake'lerde genellikle yuz sinirlari, goz cevresi ve agiz bolgesi vurgulanir"
        )
    },
    "eigencam": {
        "keywords": ["eigencam", "eigen cam", "eigen"],
        "answer": (
            "**EigenCAM**\n\n"
            "Birinci temel bilesen (PCA) kullanarak ozellik haritalarindan en onemli deseni cikarir.\n\n"
            "**GradCAM'den farki:**\n"
            "- Gradyan gerektirmez — sadece ozellik haritalarinin SVD ayristirmasini kullanir\n"
            "- Daha hizlidir ve modelden bagimsizdir\n"
            "- Genellikle daha genellestirilmis vurgu alanlari gosterir\n\n"
            "**Deepfake baglami:** Manipule edilmis bolgelerdeki yapisal bozulmalari yakalar."
        )
    },
    "counterfactual": {
        "keywords": ["counterfactual", "karşı olgusal", "karsi olgusal", "ne olsaydı", "ne olsaydi"],
        "answer": (
            "**Counterfactual XAI**\n\n"
            "\"Bu gorsel REAL olsaydi, model nereye farkli bakardi?\" sorusunu cevaplar.\n\n"
            "**Calisma prensibi:**\n"
            "1. Gorseli FAKE olarak siniflandiran ozellikleri belirler\n"
            "2. Bu ozellikleri tersine cevirerek REAL kararina giden yolu gosterir\n"
            "3. Fark haritasi, manipulasyonun en belirgin oldugu alanlari isaret eder\n\n"
            "**Kullanim:** Modelin FAKE demesinin asil sebebini anlamak icin en etkili XAI yontemidir."
        )
    },
    "dwt": {
        "keywords": ["dwt", "wavelet", "frekans", "frequency", "dalgacık", "dalgacik", "spektral"],
        "answer": (
            "**DWT (Discrete Wavelet Transform) — Frekans Analizi**\n\n"
            "Gorseli frekans bilesenlerine ayirarak GAN/Deepfake izlerini tespit eder.\n\n"
            "**4 Alt-Bant:**\n"
            "- **LL** (Dusuk-Dusuk): Orijinal gorselin kucultulmusu\n"
            "- **LH** (Dusuk-Yuksek): Yatay kenarlar\n"
            "- **HL** (Yuksek-Dusuk): Dikey kenarlar\n"
            "- **HH** (Yuksek-Yuksek): Capraz detaylar / gurultu\n\n"
            "**Deepfake tespiti:** GAN uretimi sirasinda yuksek frekanslarda belirgin izler (spectral artifacts) "
            "birakir. HH bandinda anormal dusuk enerji = GAN manipulasyonu supesi."
        )
    },
    "mesh": {
        "keywords": ["mesh", "landmark", "face mesh", "geometri", "468", "yüz noktası", "yuz noktasi"],
        "answer": (
            "**Face Mesh — Geometri Yolu**\n\n"
            "MediaPipe FaceLandmarker ile yuzdeki 468 3D noktayi (landmark) cikarir.\n\n"
            "**Bolgeler:**\n"
            "- 🟡 Sarı: Goz bolgeleri (iris, goz kapagi)\n"
            "- 🔴 Kirmizi: Dudak bolgeleri\n"
            "- 🟣 Mor: Yuz konturu (cene hatti)\n"
            "- 🔵 Cyan: Tesselation (tum baglanti aglari)\n\n"
            "**Deepfake tespiti:**\n"
            "- Asimetrik landmark dagilimi → Manipulasyon supesi\n"
            "- Goz kirpma frekansi anormalligi → Video deepfake\n"
            "- Dudak-cene uyumsuzlugu → Lip-sync deepfake"
        )
    },
    "model": {
        "keywords": ["model", "mimari", "architecture", "nasıl çalışıyor", "nasil calisiyor", "dual path", "dualpath"],
        "answer": (
            "**DualPathDeepfakeDetector — Model Mimarisi**\n\n"
            "```\n"
            "Gorsel → 3 Paralel Yol:\n"
            "  ├── RGB Branch (MobileNetV3-Large) → 960-dim\n"
            "  ├── Freq Branch (DWT → MobileNetV3) → 960-dim\n"
            "  └── Mesh Branch (468 landmarks → MLP) → 128→960-dim\n"
            "        ↓\n"
            "  LearnableFusion (SE-Net tabanli agirlikli birlestirme)\n"
            "        ↓\n"
            "  Stacked BiLSTM → TemporalAttention\n"
            "        ↓\n"
            "  Classifier → REAL / FAKE\n"
            "```\n\n"
            "Her yol farkli sinyal arar: RGB → gorsel bozulma, Frekans → spektral iz, Geometri → yapisal tutarsizlik."
        )
    },
    "tta": {
        "keywords": ["tta", "test time", "augmentation", "augmentasyon"],
        "answer": (
            "**TTA (Test-Time Augmentation)**\n\n"
            "Ayni gorselin farkli donusturulmus versiyonlari uzerinde tahmin yaparak guvenilirligi arttirir.\n\n"
            "**Ugulanan donusumler:** Yatay cevir, hafif rotasyon, parlaklik/kontrast, kirpma\n\n"
            "**Sonuc:** N tahmin ortalamalanir. Standart sapma dusukse → Model emin, yuksekse → Belirsiz."
        )
    },
    "deepfake": {
        "keywords": ["deepfake nedir", "deepfake ne", "deepfake türleri", "deepfake turleri", "sahte yüz",
                     "sahte yuz", "face swap", "faceswap"],
        "answer": (
            "**Deepfake Turleri**\n\n"
            "| Tur | Yontem | Tespit Ipucu |\n"
            "|---|---|---|\n"
            "| **Face Swap** | Bir yuzun baskasinin yerine konmasi | Yuz sinirlari, aydinlanma uyumsuzlugu |\n"
            "| **Face Reenactment** | Mimik/ifade transferi | Dudak-ses senkronu, goz hareketi |\n"
            "| **AI Generated** | GAN/Diffusion ile sifirdan uretim | Frekans izleri, simetri bozukluklari |\n"
            "| **Lip Sync** | Dudak hareketini farkli sese uyarlama | Dudak-cene geometri tutarsizligi |\n"
            "| **Physical Spoof** | Yazici cikti/ekran fotografi | Moire deseni, doku kaybi |"
        )
    },
    "compression": {
        "keywords": ["sıkıştırma", "sikistirma", "jpeg", "tiktok", "twitter", "platform",
                     "kalite", "quality"],
        "answer": (
            "**Platform Sikistirma ve Deepfake Tespiti**\n\n"
            "Sosyal medya platformlari gorselleri JPEG sikistirmasina tabi tutar:\n\n"
            "| Platform | Kalite | Etki |\n"
            "|---|---|---|\n"
            "| Twitter/X | Q≈78-88 | Orta, EXIF silinir |\n"
            "| TikTok | Q≈65-78 | Video frame kaybi, dikey format |\n\n"
            "**Robustness modumuz** bu sikistirmalari simule ederek modelin gercek dunya performansini test eder."
        )
    },
    "xai": {
        "keywords": ["xai", "açıklanabilir", "aciklanabilir", "explainable", "lime", "fastcam", "smoe"],
        "answer": (
            "**XAI (Aciklanabilir Yapay Zeka) Yontemlerimiz**\n\n"
            "| Yontem | Aciklama | Guc |\n"
            "|---|---|---|\n"
            "| **GradCAM++** | Gradyan tabanli odak haritasi | En yaygin, guvenilir |\n"
            "| **EigenCAM** | PCA tabanli ozellik vurgusu | Hizli, gradyan gerektirmez |\n"
            "| **FastCAM/SMOE** | Istatistiksel ozellik haritasi | Cok hizli |\n"
            "| **Counterfactual** | Ters senaryo analizi | Nedensellik gosterir |\n"
            "| **Guided LIME** | Yerel dogrusal aciklama | Piksel bazinda onemi gosterir |"
        )
    },
    "active_learning": {
        "keywords": ["active learning", "sürekli öğrenme", "surekli ogrenme", "fine-tune", "finetune",
                     "geri bildirim", "feedback"],
        "answer": (
            "**Active Learning / Surekli Ogrenme Pipeline'i**\n\n"
            "1. Analiz yap → GERCEK/SAHTE butonuna tikla\n"
            "2. Gorsel `feedback_images/REAL/` veya `/FAKE/` dizinine kaydedilir\n"
            "3. 10+ gorsel birikince 'Modeli Guncelle' ile classifier head fine-tune baslar\n\n"
            "**Guvenlik:** Sadece classifier (son 2 FC katman) egitilir, backbone dondurulur.\n"
            "Early stopping ile overfitting onlenir. Rollback ile orijinale donulebilir."
        )
    },
}


def _find_local_answer(message: str) -> str:
    """Anahtar kelime eslestirmesiyle yerel bilgi tabanindan cevap bul."""
    msg_lower = message.lower()

    best_match = None
    best_score = 0

    for topic, data in KNOWLEDGE_BASE.items():
        score = sum(1 for kw in data["keywords"] if kw in msg_lower)
        if score > best_score:
            best_score = score
            best_match = topic

    if best_match and best_score > 0:
        return KNOWLEDGE_BASE[best_match]["answer"]
    return ""


def _format_analysis_context(result: dict) -> str:
    """Son analiz sonucunu zengin bağlam metnine cevir."""
    if not result:
        return "Henuz analiz yapilmamis."

    label = result.get("label", "?")
    fake_p = result.get("fake_prob", 0)
    real_p = result.get("real_prob", 0)
    conf = result.get("confidence", 0)
    tta_std = result.get("tta_std", 0)
    gcam = result.get("gradcam_score", 0)
    cf = result.get("counterfactual_prob", 0)

    lines = [
        f"**Son Analiz Sonucu:** {label}",
        f"- Fake Olasilik: {fake_p:.4f} | Real Olasilik: {real_p:.4f}",
        f"- Guven: %{conf*100:.1f}",
        f"- TTA Std: {tta_std:.4f} ({'Dusuk → Model emin' if tta_std < 0.05 else 'Yuksek → Belirsiz'})",
        f"- GradCAM++ Skoru: {gcam:.4f}",
        f"- Counterfactual Prob: {cf:.4f}",
    ]

    q = result.get("image_quality", {})
    if q:
        lines.append(f"- Gorsel Kalite: Q={q.get('estimated_quality', '?')}")

    return "\n".join(lines)


def _generate_smart_response(message: str, context: dict) -> str:
    """LLM olmadan akilli yerel cevap uret."""
    msg_lower = message.lower()

    # Yerel bilgi tabani
    local = _find_local_answer(message)

    # Analiz yorumu
    if context and any(w in msg_lower for w in ["neden", "niye", "sebebi", "yorumla", "analiz",
                                                  "sonuc", "sonuç", "acikla", "açıkla"]):
        analysis_text = _format_analysis_context(context)
        label = context.get("label", "?")
        fake_p = context.get("fake_prob", 0)

        if label == "FAKE":
            interpretation = (
                f"\n\n**Yorum:** Model bu gorseli **%{fake_p*100:.1f}** olasilikla SAHTE olarak degerlendirdi.\n"
                "Olasi nedenler:\n"
                "- GradCAM++ haritasinda yuz sinirlari veya goz cevresinde yogun aktivasyon\n"
                "- Frekans analizinde yuksek bantlarda anormal enerji dagilimi\n"
                "- Geometrik landmark dagilimdaki asimetri"
            )
        else:
            interpretation = (
                f"\n\n**Yorum:** Model bu gorseli **%{(1-fake_p)*100:.1f}** olasilikla GERCEK olarak degerlendirdi.\n"
                "Gozlemler:\n"
                "- Frekans spektrumunda dogal gurultu dagilimu\n"
                "- Yuz landmark'lari simetrik ve tutarli\n"
                "- GradCAM haritasinda odakli anormal bolge yok"
            )

        return f"{analysis_text}{interpretation}"

    # Yerel cevap varsa dondur
    if local:
        ctx_note = ""
        if context:
            ctx_note = f"\n\n---\n📊 **Mevcut Analiz:** {context.get('label', '?')} (%{context.get('confidence',0)*100:.0f} guven)"
        return local + ctx_note

    # Genel yardim
    if any(w in msg_lower for w in ["yardım", "yardim", "help", "ne yapabilirsin", "merhaba", "selam"]):
        return (
            "**DeepfakeULTRA Analiz Asistani** 🤖\n\n"
            "Bana su konularda soru sorabilirsiniz:\n\n"
            "- 🔍 **GradCAM++, EigenCAM, Counterfactual** → XAI yontemlerini acikla\n"
            "- 🌊 **DWT / Frekans analizi** → Spektral tespit nasil calisir\n"
            "- 🔷 **Face Mesh / Landmark** → Geometri yolu ne yapar\n"
            "- 🏗️ **Model mimarisi** → DualPath nasil calisir\n"
            "- 🎯 **Deepfake turleri** → Face swap, reenactment, AI generated\n"
            "- 📱 **Platform sikistirma** → TikTok/Twitter etkisi\n"
            "- 🧠 **Active Learning** → Geri bildirim ile model guncelleme\n"
            "- 📊 **Son analiz yorumu** → \"Neden FAKE dedi?\" gibi sorular\n\n"
            "_Bir gorsel analiz yaptiktan sonra \"Neden FAKE dedi?\" diye sorarsaniz, "
            "analiz sonuclarini detaylica yorumlarim._"
        )

    return ""


class DeepfakeAnalysisAssistant:
    """LLM + Yerel NLP tabanli analiz asistani."""

    def __init__(self, gemini_api_key=None):
        self.api_key = gemini_api_key or llm_cfg.GEMINI_API_KEY
        self.context = {}
        self.history = []
        self.backend = None
        self._init_backend()

    def _init_backend(self):
        if self.api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel(
                    model_name=llm_cfg.GEMINI_MODEL,
                    generation_config={"temperature": llm_cfg.GEMINI_TEMPERATURE,
                                       "max_output_tokens": llm_cfg.GEMINI_MAX_TOKENS})
                self.chat_session = self.model.start_chat(history=[])
                self.backend = "gemini"
                return
            except Exception as e:
                print(f"Gemini baslatilamadi: {e}")

        try:
            import httpx
            r = httpx.get(f"{llm_cfg.OLLAMA_BASE_URL}/api/tags", timeout=3)
            if r.status_code == 200:
                self.backend = "ollama"
                return
        except Exception:
            pass

        self.backend = "local"

    def set_analysis_context(self, result: dict):
        self.context = result

    def chat(self, message: str) -> str:
        # Oncelikle yerel NLP dene
        local_response = _generate_smart_response(message, self.context)

        # Yerel cevap varsa — LLM'e gitme, aninda don
        if local_response:
            return local_response

        # Yerel cevap yok — LLM'e sor
        if self.backend == "gemini":
            try:
                ctx_text = _format_analysis_context(self.context) if self.context else "Analiz yok."
                enriched = f"{SYSTEM_PROMPT}\n\nBaglam:\n{ctx_text}\n\nSoru: {message}"
                response = self.chat_session.send_message(enriched)
                return response.text
            except Exception as e:
                return f"Gemini hatasi: {e}"

        elif self.backend == "ollama":
            try:
                import httpx
                ctx_text = _format_analysis_context(self.context) if self.context else "Analiz yok."
                enriched = f"{SYSTEM_PROMPT}\n\nBaglam:\n{ctx_text}\n\nSoru: {message}"
                r = httpx.post(f"{llm_cfg.OLLAMA_BASE_URL}/api/generate",
                              json={"model": llm_cfg.OLLAMA_MODEL, "prompt": enriched,
                                    "stream": False}, timeout=60)
                return r.json().get("response", "Yanit alinamadi")
            except Exception as e:
                return f"Ollama hatasi: {e}"

        # LLM yok — yerel cevap
        if local_response:
            return local_response

        return (
            "Bu konuda henuz yerel bilgim yok. Daha spesifik sorabilirsiniz:\n"
            "- \"GradCAM nedir?\"\n"
            "- \"Model nasil calisiyor?\"\n"
            "- \"Neden FAKE dedi?\"\n"
            "- \"DWT frekans analizi ne?\"\n\n"
            "_Gemini API key girerseniz sinirsiz soru sorabilirsiniz._"
        )

    def clear_history(self):
        self.history = []
        if self.backend == "gemini":
            self.chat_session = self.model.start_chat(history=[])
