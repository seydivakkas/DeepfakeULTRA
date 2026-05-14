<div align="center">

# 🔍 DeepfakeULTRA

### AI-Powered Forensic Deepfake Detection System

[![Python 3.14+](https://img.shields.io/badge/Python-3.14+-3776AB?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![CUDA](https://img.shields.io/badge/CUDA-12.x-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![Gradio](https://img.shields.io/badge/Gradio-4.0+-F97316?logo=gradio&logoColor=white)](https://gradio.app)
[![Val AUC](https://img.shields.io/badge/Val_AUC-0.9839-brightgreen)](/)
[![Cross‑Dataset](https://img.shields.io/badge/Cross--Dataset_AUC-0.7527-blue)](/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Dual-Path CNN mimarisi ile tek görsellik deepfake / face swap / AI-generated yüz tespiti.**
Domain Generalization · GradCAM++ · EigenCAM · DWT Frekans · Kraniyofasiyal Biyometrik · TTA

</div>

---

## 📋 İçindekiler

- [Genel Bakış](#-genel-bakış)
- [Performans & Benchmark](#-performans--benchmark-sonuçları)
- [Arayüz Sekmeleri](#️-arayüz-sekmeleri)
- [Mimari](#️-mimari)
- [Domain Generalization](#-domain-generalization)
- [Kurulum](#-kurulum)
- [Kullanım](#️-kullanım)
- [Proje Yapısı](#-proje-yapısı)
- [Eğitim Pipeline'ı](#-eğitim-pipelineı)
- [Teknoloji Yığını](#️-teknoloji-yığını)

---

## 🎯 Genel Bakış

DeepfakeULTRA, **tek görsel üzerinden** deepfake, face swap, AI-generated yüz ve fiziksel spoof tespiti yapan end-to-end bir forensik analiz sistemidir. 7 sekmeli Gradio web arayüzü ile kapsamlı analiz, test ve raporlama sunar.

**Domain Generalization** stratejisi ile sadece eğitim verisinde değil, **hiç görmediği 5 farklı harici datasette** de yüksek doğruluk sağlar. JPEG compression, çözünürlük değişimi, renk kayması gibi domain augmentasyonlar ve curriculum fine-tuning ile cross-dataset genelleme başarısı **%40 artırılmıştır**.

### Ne Yapar?

| Yetenek | Açıklama |
|---|---|
| **Deepfake Tespiti** | REAL / FAKE / UNCERTAIN üçlü karar sistemi |
| **Domain Generalization** | 5 farklı harici datasette doğrulanmış cross-dataset genelleme |
| **XAI Haritaları** | GradCAM++, EigenCAM, FastCAM, Guided LIME ile modelin neye baktığını görselleştirir |
| **Frekans Analizi** | DWT + DCT + Phase ile GAN/Diffusion izlerini spektral düzlemde yakalar |
| **Yüz Geometrisi** | MediaPipe 468 landmark ile yapısal tutarsızlıkları tespit eder |
| **Forensik Tarama** | ELA + Noise analizi ile manipülasyon bölgelerini belirler |
| **TTA (Test-Time Aug.)** | GPU-native 8 augmentasyon ile çıkarım güvenilirliğini artırır |
| **Kraniyofasiyal Analiz** | AI Vision API (GPT-4o / Claude / Gemini) ile yüz anatomisi biyometrik analizi |
| **Platform Tespiti** | JPEG artefaktlarından kaynak sosyal medya platformunu otomatik tespit eder |
| **Adversarial Test** | FGSM / PGD / CW saldırıları ile model dayanıklılığını ölçer |
| **Active Learning** | Kullanıcı geri bildirimleriyle classifier head'i fine-tune eder |

---

## 🏆 Performans & Benchmark Sonuçları

### İç Test Seti

| Metrik | Değer |
|---|---|
| **Val AUC** | **0.9839** |
| **Val Accuracy** | 93.1% |
| **Val F1-Score** | 0.931 |
| **Mimari** | DualPath (RGB + DWT + Mesh) |
| **Backbone** | MobileNetV3-Large (×2) |
| **Parametre** | ~15M |
| **Çıkarım Süresi** | ~200ms / görsel (GPU) |
| **VRAM Kullanımı** | ~2GB (inference) / ~6GB (eğitim, FP16) |

### Cross-Dataset Benchmark (5 Harici Dataset)

Model hiç eğitilmediği, tamamen **görülmemiş** datasetler üzerinde test edilmiştir:

| Dataset | Baseline AUC | Final AUC | Artış |
|---------|-------------|-----------|-------|
| **Deepfake20K** | 0.514 | **0.956** | 🔥 +85.9% |
| **DFDC** | 0.546 | **0.821** | 🔥 +50.3% |
| **DeepfakeFace** | 0.583 | **0.777** | 🟢 +33.3% |
| **CelebDF v2** | 0.541 | **0.708** | 🟢 +30.8% |
| **FaceForensics++** | 0.502 | 0.500 | — |
| **Ortalama** | **0.537** | **0.752** | **+40.0%** |

> **Combined Score:** `0.8683` (val_auc=0.9839, ext_mean_auc=0.7527)

---

## 🖥️ Arayüz Sekmeleri

DeepfakeULTRA 7 sekmeli bir Gradio web arayüzü sunar:

### 🔍 Sekme 1 — Single Image Analysis

Ana forensik analiz sekmesi. Görsel yükle, tek tıkla kapsamlı sonuç al.

- **Üçlü Karar Sistemi:** FAKE / UNCERTAIN / REAL (Youden J-statistic bazlı eşikler)
- **TTA (Test-Time Augmentation):** 5–15 augmentasyon ile güvenilirlik artırımı
- **4 XAI Haritası:** GradCAM++, EigenCAM, FastCAM, Guided LIME — yan yana görselleştirme
- **3 Analiz Yolu:** RGB görsel / DWT frekans haritası / Face Mesh geometri
- **📱 Platform Tespiti:** JPEG forensik ile WhatsApp, Instagram, TikTok, Telegram otomatik tanıma
- **🔬 Forensik Konsensüs:** Model (%60) + ELA (%25) + Noise (%15) ağırlıklı birleşim
- **Tutarlılık Kontrolü:** Model-ELA-Noise çelişki tespiti ve raporlama
- **PDF Rapor:** Tek tıkla indirilebilir profesyonel rapor
- **Geri Bildirim:** GERÇEK / SAHTE butonları ile Active Learning havuzu

### 🛡️ Sekme 2 — Robustness Test

Modelin saldırı ve sıkıştırma dayanıklılığını test eder.

- **Adversarial Attack:** FGSM, PGD, CW saldırı türleri
- **Branch Knockout:** RGB / Frekans / Geometri dallarını izole test
- **Compression Sweep:** Q=10→100 JPEG sıkıştırma direnci
- **Resolution Scaling:** Farklı çözünürlüklerde performans testi
- **Double Compression:** İkili sıkıştırma artefakt analizi

### 📊 Sekme 3 — Analytics Dashboard

Tüm analizlerin istatistiksel takip paneli. Scroll-free kompakt tasarım.

- **4 Metrik Kartı:** Toplam analiz, FAKE sayısı, REAL sayısı, ortalama güven
- **Günlük Dağılım:** Zaman serisi trend grafikleri
- **Güven Dağılımı:** Histogram ile model güven aralıkları
- **Sonuç Dağılımı:** FAKE vs REAL oranları
- **Anlık Yenileme:** Slider ile kayıt limiti ayarı

### 🔧 Sekme 4 — Model Profili

Model performansını test eden ve doğrulayan araçlar.

- **Cross-Validation:** K-Fold çapraz doğrulama ve ROC eğrileri
- **Yeni Dataset Testi:** Harici veri setleri (Celeb-DF v2 vb.) üzerinde performans
- **Confusion Matrix:** Detaylı sınıflandırma matrisi
- **Fine-Tuning:** Active Learning havuzundan model güncelleme
- **Rollback:** Model ağırlıklarını önceki versiyona geri alma

### 🕒 Sekme 5 — Analiz Geçmişi

SQLite tabanlı kalıcı analiz kaydı — uygulama yeniden başlatılsa bile korunur.

- **200'e kadar kayıt** görüntüleme
- **Geçmiş temizleme** butonu
- **Tarih, sonuç, güven skoru** detaylı tablo

### 🧬 Sekme 6 — Yüz Anatomisi (Kraniyofasiyal Biyometrik Analiz)

AI Vision API destekli yüz anatomisi analiz motoru. **Bu sekme yeni eklenmiştir.**

- **Multi-Provider Desteği:** Google Gemini, OpenAI GPT-4o, Anthropic Claude
- **API Key Kaydetme:** Bir kere gir, otomatik hatırla (`.api_keys.json`)
- **Provider Geçişi:** Dropdown'dan provider değiştirince kaydedilmiş key otomatik yüklenir
- **Analiz Modları:** FULL / QUICK / REGION_SPECIFIC
- **Bölge Odaklı:** ALL / FACE / LIPS / JAW / EYES / NOSE

#### Analiz Pipeline:
```
🖼️ Görüntü Girişi → 🎯 Landmark Tespiti (68-pt) → 📐 Anatomik Hesaplama
→ 🧠 AI Skor Motoru → 📊 Dashboard → 📋 Adli Rapor
```

#### Ölçülen Metrikler:

| Modül | Metrik | Normal Aralık | Deepfake Sinyali |
|---|---|---|---|
| **Asimetri** | FAI (Facial Asymmetry Index) | 0.5 – 3.5% | <0.5 = yapay aşırı simetri |
| **Asimetri** | Orbital Delta | 0 – 2.5 mm | >4mm = anomali |
| **Dudak** | Cupid Bow Simetrisi | 0 – 5° | <0.5° = GAN imzası |
| **Dudak** | Üst/Alt Dudak Oranı | 0.55 – 0.70 | Tam 0.618 = yapay |
| **Çene** | Gonial Açı | 120° – 130° | Asimetrik gonial açı |
| **Çene** | Jawline Continuity | 6.5 – 9.0 | <5 = seam artifact |
| **Deepfake** | Blending Artifact | 8.0+ | <6 = yüksek risk |
| **Deepfake** | Skin GAN Texture | <0.3 | >0.5 = AI doku |
| **Deepfake** | Lighting Coherence | <15° | >30° = kompozit ışık |

#### Çıktı Formatı:
- **📊 Analiz Raporu** — Tablolu Markdown rapor (asimetri, dudak, çene, deepfake risk)
- **🔧 JSON Çıktı** — Yapılandırılmış JSON yanıt
- **📋 Referans Tablosu** — 12 metrik referans değerleri

#### Model Fallback Zinciri (Gemini):
```
gemini-2.5-flash → gemini-2.0-flash-lite → gemini-1.5-flash
```
Her modelde 3 retry (5s, 10s, 15s exponential backoff).

### 💬 Sekme 7 — Analiz Asistanı

LLM destekli sohbet asistanı — analiz sonuçlarını yorumlayın.

- **Gemini 3.0 Pro:** Birincil LLM backend
- **Ollama Fallback:** Yerel qwen2.5:7b ile çevrimdışı çalışma
- **Yerel Bilgi Tabanı:** 12 konu başlığında (GradCAM, DWT, Mesh, TTA vb.) offline cevaplama
- **Hızlı Sorular:** 6 adet tek tıkla sık sorulan sorular
- **Bağlam Kartı:** Son analiz sonuçları otomatik yansır

---

## 🏗️ Mimari

### DualPathDeepfakeDetector

```
Görsel (224×224)
    │
    ├── 🟢 RGB Branch ─────────── MobileNetV3-Large ──── 960-dim
    │
    ├── 🔵 Frequency Branch ────── DWT+DCT+Phase → MobileNetV3 ── 960-dim
    │        (Haar + DB2 + Coif1 → 18 kanal hibrit frekans)
    │
    └── 🟡 Geometry Branch ─────── MediaPipe 468 Landmarks → MLP ── 960-dim
              (1404-dim → 256 → 128 → 960-dim)
                    │
         ┌──────────┼──────────┐
         │          │          │
      [token₁]  [token₂]  [token₃]   ← 3 branch = 3 token
         │          │          │
         └──────────┼──────────┘
                    │
         CrossBranchTransformer (2-Layer, 4-Head Self-Attention)
              + Branch Embedding + Pre-LayerNorm + GELU FFN
                    │
              Mean Pool → 960-dim
                    │
              Classifier (960 → 256 → ReLU → Dropout(0.5) → 2)
                    │
              P(REAL), P(FAKE)
```

### Branch Detayları

| Branch | Backbone | Giriş | Ne Yakalar |
|---|---|---|---|
| **RGB** | MobileNetV3-Large (ImageNet) | `(B, 3, 224, 224)` | Blending artifact, renk tutarsızlığı, doku anomalisi |
| **Frekans** | MobileNetV3-Large (scratch) | `(B, 18, 224, 224)` | GAN/Diffusion spektral izleri, JPEG ghost |
| **Geometri** | MLP | `(B, 1404)` | Landmark asimetri, yapısal tutarsızlık |

### HybridFrequencyExtractor — 18 Kanal

```
Kaynak Görsel (H×W×3)
    ├── DWT (12 kanal): Haar + DB2 + Coif1 → cA, cH, cV, cD × 3
    ├── DCT (3 kanal): Low/Mid/High frequency bands
    └── Phase Spectrum (3 kanal): R/G/B channel FFT phase
```

### Loss Fonksiyonu

```
L_total = 0.80 × Focal Loss (γ=2.0, α=0.75) + 0.20 × Triplet Loss (margin=1.0)
```

### Domain Augmentation Pipeline

Eğitim sırasında domain-invariant özellik öğrenimi için uygulanan augmentasyonlar:

```
Kaynak Görsel (224×224)
    │
    ├── JPEG Compression (Q=30–95, p=0.5)
    ├── Downscale-Upscale (0.25x–0.75x → orijinal boyut, p=0.4)
    ├── Gaussian Noise (σ=0.01–0.03, p=0.3)
    ├── Gamma Correction (γ=0.7–1.3, p=0.3)
    └── Color Channel Shift (±15, p=0.3)
         │
         └── Domain-Robust Eğitim Görseli
```

### TTA (Test-Time Augmentation) Inference

```
Girdi Görseli
    │
    ├── Orijinal ─────────── P₁
    ├── Horizontal Flip ──── P₂
    ├── Gaussian Blur ────── P₃
    ├── Resize 0.9x ──────── P₄
    ├── Resize 1.1x ──────── P₅
    ├── Gaussian Noise ───── P₆
    ├── Brightness +0.1 ──── P₇
    └── Brightness -0.1 ──── P₈
         │
         └── Mean(P₁..P₈) → Final Prediction
```

---

## 🧬 Domain Generalization

Deepfake tespit modellerinin en büyük sorunu **domain-specific overfitting** — model eğitildiği veri setinde mükemmel çalışırken, farklı kaynaklardan gelen görsellerde başarısız olur.

### Problem

| Sorun | Açıklama |
|---|---|
| **JPEG Kalitesi** | Her platform farklı sıkıştırma kullanır (WhatsApp Q=60, Instagram Q=85) |
| **Çözünürlük** | Datasetler arası 128px–1024px arası değişkenlik |
| **Renk Profili** | Kamera sensörü, ISP, ekran kaydı farklılıkları |
| **Deepfake Yöntemi** | Autoencoder vs GAN vs Diffusion — farklı artifact türleri |

### Çözüm: 3 Aşamalı Domain Generalization

```
Aşama 1: TTA (Test-Time Augmentation)
├── GPU-native 8 augmentasyon → mean prediction
├── Sonuç: +1–2 puan hafif iyileşme
└── Kök sorunu çözmez, ama güvenilirliği artırır

Aşama 2: Domain Augmentation
├── JPEG, resolution, noise, color, gamma augmentasyonları
├── Eğitim sırasında domain çeşitliliği simüle eder
└── Model domain-specific artifact'lara bağımlılığı azaltır

Aşama 3: Curriculum Fine-Tuning
├── Epoch 1-3: Backbone FROZEN (sadece classifier adapte olur)
├── Epoch 4-8: Backbone UNFROZEN (derin feature refinement)
├── Combined Score = 0.5 × val_auc + 0.5 × ext_mean_auc
└── Model seçimi iç + harici performans dengesine göre yapılır
```

### Eğitim İlerleme Tablosu

| Epoch | Durum | Train Loss | Val AUC | Ext Mean AUC |
|-------|-------|-----------|---------|-------------|
| 1 | FROZEN | 0.1004 | 0.9816 | — |
| 2 | FROZEN | 0.1000 | 0.9811 | — |
| 3 | FROZEN | 0.0998 | 0.9818 | 0.7347 |
| 4 | **UNFROZEN** | 0.0990 | 0.9828 | — |
| 5 | UNFROZEN | 0.0983 | 0.9832 | — |
| 6 | UNFROZEN | 0.0980 | 0.9829 | — |
| 7 | UNFROZEN | 0.0977 | 0.9835 | — |
| **8** | **UNFROZEN** | **0.0975** | **0.9839** | **0.7527** |

---

## 🚀 Kurulum

### Gereksinimler

- Python 3.14+
- NVIDIA GPU (CUDA destekli) — önerilir
- 8GB+ VRAM (RTX 4070 önerilir)

### Adımlar

```bash
# 1. Repoyu klonla
git clone https://github.com/seydivakkas/DeepfakeULTRA.git
cd DeepfakeULTRA

# 2. Sanal ortam oluştur
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/macOS

# 3. Bağımlılıkları yükle
pip install -r requirements.txt

# 4. Model dosyasını yerleştir
# models/best_model.pth dosyasını models/ dizinine koy

# 5. Çalıştır
python app.py
```

Uygulama **http://localhost:7860** adresinde açılacaktır.

### Docker

```bash
docker-compose up --build
```

### Ortam Değişkenleri (Opsiyonel)

```env
GEMINI_API_KEY=your-google-ai-studio-key   # Analiz Asistanı için
JWT_SECRET=your-secret-key                  # API güvenliği
```

> **Not:** Kraniyofasiyal Analiz sekmesi için API key'ler arayüzden girilebilir ve kaydedilebilir.

---

## 🖥️ Kullanım

### Gradio Web Arayüzü

```bash
python app.py
# → http://localhost:7860
```

### FastAPI REST API

```bash
python main.py
# → http://localhost:8000/docs (Swagger UI)
```

### CLI Araçları

```bash
# Harici datasetlerde benchmark (TTA opsiyonel)
python scripts/evaluate_model.py --tta

# Domain generalization fine-tuning
python scripts/finetune_generalization.py
```

### API Endpoint'leri

| Method | Endpoint | Açıklama |
|---|---|---|
| `POST` | `/api/analyze` | Tek görsel analizi |
| `GET` | `/api/health` | Sistem sağlık kontrolü |
| `GET` | `/api/analytics` | İstatistik verileri |
| `GET` | `/api/history` | Analiz geçmişi |

---

## 📁 Proje Yapısı

```
DeepfakeULTRA/
├── app.py                          # Gradio ana arayüz (7 sekme)
├── main.py                         # FastAPI + Gradio birleşik başlatma
├── config.py                       # Merkezi konfigürasyon (dataclass)
├── requirements.txt                # Python bağımlılıkları
│
├── core/                           # Çekirdek modüller
│   ├── dual_mobilenetv3.py         # DualPathDeepfakeDetector modeli
│   ├── frequency.py                # DWT frekans görselleştirme
│   ├── frequency_v2.py             # Hibrit frekans çıkarıcı (DWT+DCT+Phase)
│   ├── data_pipeline.py            # Veri yükleme ve augmentasyon
│   ├── domain_augmentation.py      # 🆕 Domain-robust augmentasyon modülü
│   ├── tta_inference.py            # 🆕 GPU-native Test-Time Augmentation
│   ├── trainer.py                  # Eğitim döngüsü
│   ├── adversarial.py              # FGSM / PGD / CW saldırıları
│   ├── compression.py              # Platform sıkıştırma simülasyonu
│   ├── forensics.py                # ELA + Noise analizi
│   ├── face_detector.py            # Yüz algılama + bounding box
│   ├── platform_detector.py        # JPEG forensik platform tespiti
│   ├── fine_tuner.py               # Active Learning fine-tuning
│   ├── model_metrics.py            # ROC, CM, F1 hesaplama
│   ├── sbi_augmentation.py         # Self-Blended Image augmentasyonu
│   ├── hard_real_augmentation.py   # Hard-real veri üretimi
│   ├── contrastive_loss.py         # Triplet Loss implementasyonu
│   ├── loss_utils.py               # Focal Loss + Label Smoothing
│   ├── evaluation.py               # Model değerlendirme
│   └── calibration.py              # Temperature Scaling kalibrasyon
│
├── inference/                      # Çıkarım modülleri
│   ├── predictor.py                # Ana tahmin sınıfı
│   ├── analyze_engine.py           # Tam analiz pipeline'ı
│   ├── tta_inference.py            # Test-Time Augmentation
│   ├── xai_module.py               # GradCAM++ / EigenCAM / FastCAM
│   ├── hybrid_xai.py               # Birleşik XAI raporu
│   ├── model_ensemble.py           # Multi-model ensemble
│   └── subtype_classifier.py       # Hiyerarşik alt-tip sınıflandırma
│
├── ui/                             # Arayüz bileşenleri
│   ├── components.py               # Tüm Gradio handler fonksiyonları
│   └── craniofacial_tab.py         # 🧬 Kraniyofasiyal Biyometrik Analiz sekmesi
│
├── services/                       # Harici servisler
│   ├── llm_module.py               # Gemini / Ollama chat asistanı
│   ├── vision_api.py               # Multi-provider Vision API (GPT-4o/Claude/Gemini)
│   └── pdf_report.py               # PDF rapor oluşturma
│
├── api/                            # REST API katmanı
│   └── server.py                   # FastAPI endpoint'leri
│
├── db/                             # Veritabanı
│   └── database.py                 # SQLite analiz geçmişi + feedback
│
├── scripts/                        # Eğitim & veri işleme scriptleri
│   ├── 01_extract_faces.py         # Yüz çıkarma pipeline'ı
│   ├── 06_smart_split.py           # Kalite-bilinçli akıllı bölme
│   ├── finetune_generalization.py  # 🆕 Domain generalization fine-tuning
│   ├── evaluate_model.py           # Tam model değerlendirme + harici benchmark
│   ├── generate_hard_real.py       # Hard-real veri üretimi
│   ├── generate_sbi_data.py        # SBI veri üretimi
│   ├── find_threshold.py           # Youden J-statistic eşik optimizasyonu
│   ├── jury_evaluation.py          # Jury test seti değerlendirme
│   ├── leakage_checker.py          # Veri sızıntısı kontrolü
│   └── weekly_scheduler.py         # Haftalık izleme pipeline'ı
│
├── models/                         # Model ağırlıkları (.gitignore)
│   ├── best_model.pth              # Temel eğitilmiş model (AUC=0.9825)
│   └── best_model_generalized.pth  # 🆕 Domain-generalized model (Combined=0.8683)
│
├── evaluation/                     # Değerlendirme sonuçları
│   ├── metrics.json                # İç test metrikleri
│   ├── roc_curve.png               # ROC eğrisi
│   └── external/                   # 5 harici dataset benchmark sonuçları
│       ├── celeb_df_v2/
│       ├── dfdc/
│       ├── deepfake20k/
│       ├── deepfakeface/
│       └── faceforensics/
│
├── dataset/                        # Veri seti (.gitignore)
├── sunum_gorselleri/               # Demo görselleri (fake/real)
├── tests/                          # Test suite
├── Dockerfile                      # Docker imajı
├── docker-compose.yml              # Docker Compose
└── LICENSE                         # MIT License
```

---

## 🎓 Eğitim Pipeline'ı

### Eğitim Stratejisi

| Bileşen | Değer | Açıklama |
|---|---|---|
| **Loss** | 0.8 × Focal + 0.2 × Triplet | Hard-negative odaklı |
| **Optimizer** | AdamW (lr=3e-4, wd=1e-4) | Backbone lr×0.1 |
| **Augmentation** | MixUp + CutMix + SBI + SocialCompress | Class-aware |
| **Scheduler** | Cosine Annealing + 1 epoch Warmup | — |
| **Precision** | FP16 Mixed Precision | AMP + TF32 |
| **Batch** | 20 × 8 accumulation = 160 efektif | — |
| **Unfreeze** | Epoch 3'te backbone açılır | Erken fine-tune |
| **Curriculum** | 4 fazlı hard-real artırımı | 0% → 40% |

### Veri Seti Kaynakları

| Kaynak | Tür | Kullanım |
|---|---|---|
| **FF++** | Face Swap (DeepFakes, FaceSwap, Face2Face, NeuralTextures) | Eğitim |
| **DF40** | 40 farklı deepfake yöntemi | Eğitim + Test |
| **CelebA-HQ** | Yüksek kalite gerçek yüzler | Eğitim |
| **FFHQ** | Yüksek çözünürlük gerçek yüzler | Eğitim |
| **UTKFace** | Demografik çeşitlilik | Jury Test |
| **SBI** | Self-Blended Image | Fine-tuning |

### Domain Generalization Fine-Tuning

| Bileşen | Değer | Açıklama |
|---|---|---|
| **Domain Aug.** | JPEG + Resize + Noise + Gamma + Color | Domain çeşitliliği simülasyonu |
| **Frozen Epochs** | 1–3 | Backbone dondurulmuş, classifier adaptasyonu |
| **Unfrozen Epochs** | 4–8 | Backbone açık, derin feature refinement |
| **Model Seçimi** | Combined Score | `0.5 × val_auc + 0.5 × ext_mean_auc` |
| **Harici Benchmark** | 5 dataset | CelebDF, FF++, DFDC, Deepfake20K, DeepfakeFace |

### Hızlı Başlangıç

```bash
# Yüzleri çıkar
python scripts/01_extract_faces.py

# Akıllı bölme
python scripts/06_smart_split.py

# Eğitim
python run_training.py

# Değerlendirme (TTA ile)
python scripts/evaluate_model.py --tta

# Domain generalization fine-tuning
python scripts/finetune_generalization.py
```

---

## 🛠️ Teknoloji Yığını

| Katman | Teknoloji |
|---|---|
| **Derin Öğrenme** | PyTorch 2.0+, timm, torchvision |
| **Yüz Algılama** | MediaPipe FaceLandmarker (468 3D landmark) |
| **Frekans Analizi** | PyWavelets (DWT), NumPy (DCT, Phase) |
| **XAI** | Captum (GradCAM++, EigenCAM), LIME |
| **Kraniyofasiyal** | Google Gemini, OpenAI GPT-4o, Anthropic Claude Vision API |
| **Web UI** | Gradio 4.0+ |
| **REST API** | FastAPI + Uvicorn |
| **Veritabanı** | SQLite (analiz geçmişi + feedback) |
| **LLM** | Google Gemini 3.0 Pro + Ollama |
| **PDF Rapor** | fpdf2 |
| **Görselleştirme** | Plotly, Matplotlib |
| **Konteyner** | Docker + Docker Compose |

---

## 📄 Lisans

Bu proje [MIT License](LICENSE) altında lisanslanmıştır.

---

<div align="center">

**DeepfakeULTRA** — AI-Powered Forensic Deepfake Detection System

*seydivakkas © 2026*

</div>
