<div align="center">

# 🔍 DeepfakeULTRA V5

### AI-Powered Forensic Deepfake Detection System

[![Python 3.14+](https://img.shields.io/badge/Python-3.14+-3776AB?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![Gradio](https://img.shields.io/badge/Gradio-4.0+-F97316?logo=gradio&logoColor=white)](https://gradio.app)
[![License](https://img.shields.io/badge/License-All%20Rights%20Reserved-red)]()

**Dual-Path mimarisi ile tek görsellik deepfake / face swap / AI-generated yüz tespiti.**
GradCAM++ · EigenCAM · FastCAM · LIME · DWT Frekans · Face Mesh · Forensik Analiz

</div>

---

## 📋 İçindekiler

- [Genel Bakış](#-genel-bakış)
- [Mimari](#-mimari)
- [Özellikler](#-özellikler)
- [Kurulum](#-kurulum)
- [Kullanım](#-kullanım)
- [Proje Yapısı](#-proje-yapısı)
- [Eğitim Pipeline'ı](#-eğitim-pipelineı)
- [Production-Ready Pipeline](#-production-ready-pipeline)
- [Konfigürasyon](#-konfigürasyon)
- [Teknoloji Yığını](#-teknoloji-yığını)

---

## 🎯 Genel Bakış

DeepfakeULTRA, **tek görsel üzerinden** deepfake, face swap, AI-generated yüz ve fiziksel spoof tespiti yapan end-to-end bir forensik analiz sistemidir.

### Ne Yapar?

| Yetenek | Açıklama |
|---|---|
| **Deepfake Tespiti** | REAL / FAKE / UNCERTAIN üçlü karar sistemi |
| **XAI Haritaları** | GradCAM++, EigenCAM, FastCAM, Guided LIME ile modelin neye baktığını görselleştirir |
| **Frekans Analizi** | DWT + DCT + Phase ile GAN/Diffusion izlerini spektral düzlemde yakalar |
| **Yüz Geometrisi** | MediaPipe 468 landmark ile yapısal tutarsızlıkları tespit eder |
| **Forensik Tarama** | ELA + Noise analizi ile manipülasyon bölgelerini belirler |
| **Platform Tespiti** | JPEG artefaktlarından kaynak sosyal medya platformunu otomatik tespit eder |
| **Adversarial Test** | FGSM / PGD / CW saldırıları ile model dayanıklılığını ölçer |
| **Active Learning** | Kullanıcı geri bildirimleriyle classifier head'i fine-tune eder |

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

---

### 🟢 Branch 1: RGB — Görsel Bozulma Analizi

Görselin piksel düzeyindeki bozulma izlerini (blending artifact, renk tutarsızlığı, doku anomalisi) yakalayan ana dal.

| Parametre | Değer |
|---|---|
| **Backbone** | MobileNetV3-Large (ImageNet pretrained) |
| **Giriş** | `(batch, 3, 224, 224)` — RGB |
| **Çıkış** | `(batch, 960)` — AdaptiveAvgPool2d → Flatten |
| **Projeksiyon** | Identity (zaten 960-dim) |

**Kademeli Unfreeze:** İlk 5 epoch boyunca backbone dondurulur (`requires_grad=False`), epoch 5'ten itibaren tüm parametreler açılır. Bu, classifier head'in rastgele başlatılmış ağırlıklarının stabilize olmasını sağlar.

---

### 🔵 Branch 2: Frekans — Spektral İz Analizi

GAN/Diffusion modellerinin ürettiği görsellerdeki frekans domenindeki izleri (periodic artifact, spectral peak, JPEG ghost) yakalayan dal.

| Parametre | Değer |
|---|---|
| **Backbone** | MobileNetV3-Large (scratch, 18-ch input) |
| **Giriş** | `(batch, 18, 224, 224)` — Hibrit frekans haritası |
| **Çıkış** | `(batch, 960)` — AdaptiveAvgPool2d → Flatten |
| **Projeksiyon** | Identity (zaten 960-dim) |

#### HybridFrequencyExtractor — 18 Kanal Detay

```
Kaynak Görsel (H×W×3)
    │
    ├── DWT (12 kanal)
    │   ├── Haar wavelet  → cA, cH, cV, cD (4 sub-band)
    │   ├── DB2 wavelet   → cA, cH, cV, cD (4 sub-band)
    │   └── Coif1 wavelet → cA, cH, cV, cD (4 sub-band)
    │        cA = Approximation (düşük frekans)
    │        cH = Horizontal detail (yatay kenar)
    │        cV = Vertical detail (dikey kenar)
    │        cD = Diagonal detail (çapraz kenar)
    │
    ├── DCT (3 kanal) — 8×8 Block JPEG fingerprint
    │   ├── Low freq  (0-5)   → DC + genel parlaklık
    │   ├── Mid freq  (6-20)  → Kenar ve texture bilgisi
    │   └── High freq (21+)   → Noise ve artifact sinyali
    │
    └── Phase Spectrum (3 kanal) — FFT faz bileşeni
        ├── R channel phase  → Yapısal sınır bilgisi
        ├── G channel phase  → Blending kenar tespiti
        └── B channel phase  → GAN tutarsızlık deseni
```

**Neden 3 farklı frekans yöntemi?**

| Yöntem | Ne Yakalar | Zayıf Noktası |
|---|---|---|
| **DWT** | Spatial-frequency lokalizasyon (nerede hangi frekans) | Sadece ayrık bantlar |
| **DCT** | JPEG 8×8 block artifact, sıkıştırma izi | Spatial bilgi kaybı |
| **Phase** | Blending sınırı, GAN tutarsız phase pattern | Noise'a duyarlı |

Üçünün birleşimi, her birinin zayıf noktasını diğeri ile kapatır.

---

### 🟡 Branch 3: Geometri — Yapısal Tutarsızlık Analizi

Yüz geometrisindeki doğal olmayan oranları, asimetrileri ve landmark tutarsızlıklarını tespit eden dal.

| Parametre | Değer |
|---|---|
| **Ön İşlem** | MediaPipe FaceLandmarker — 468 adet 3D landmark |
| **Giriş** | `(batch, 1404)` — 468 × 3 (x, y, z) |
| **MLP** | Linear(1404→256) → BN → ReLU → Dropout(0.3) → Linear(256→256) → BN → ReLU → Dropout(0.2) → Linear(256→128) → BN → ReLU |
| **Çıkış** | `(batch, 128)` |
| **Projeksiyon** | Linear(128→960) → ReLU → `(batch, 960)` |

---

### 🔀 CrossBranchTransformer — Dallar Arası Dikkat Mekanizması

3 branch çıktısını **Transformer Self-Attention** ile birleştirir. Her branch bir **token** olarak ele alınır (`seq_len=3`). Self-Attention, hangi branch'in hangi durumda daha bilgilendirici olduğunu öğrenir.

| Parametre | Değer |
|---|---|
| **Katman Sayısı** | 2 (Transformer Encoder) |
| **Heads** | 4 |
| **Head Dim** | 240 (960 / 4) |
| **FFN Boyut** | 1920 (960 × 2) |
| **Aktivasyon** | GELU |
| **Dropout** | 0.1 |
| **Norm** | Pre-LayerNorm (daha stabil eğitim) |
| **Pooling** | Mean Pool (3 token → 1 vektör) |

```
[RGB_feat, Freq_feat, Mesh_feat]     # Her biri (batch, 960)
           │
    Stack → (batch, 3, 960)          ← 3 token sekansı
           │
    + Branch Embedding               ← Öğrenilebilir modalite kimliği
      (RGB=embed₀, Freq=embed₁, Mesh=embed₂)
           │
    Transformer Encoder Layer 1:
      Pre-LayerNorm → Multi-Head Self-Attention(4-head)
      → Residual → Pre-LayerNorm → FFN(960→1920→960, GELU)
      → Residual
           │
    Transformer Encoder Layer 2:
      (aynı yapı)
           │
    Final LayerNorm
           │
    Mean Pool → (batch, 960)
```

**Neden BiLSTM Değil?**

| Özellik | BiLSTM | CrossBranchTransformer |
|---|---|---|
| **Tasarım amacı** | Sıralı/temporal veri | Cross-modal füzyon |
| **Tek görsel** | seq_len=1 → boş çalışır | 3 token → anlamlı attention |
| **Branch ilişkisi** | Dolaylı (sıralı işlem) | Doğrudan (self-attention) |
| **Paralellik** | Sıralı hesaplama | Tam paralel |
| **Modalite farkındalığı** | Yok | Branch embedding ile var |

---

### 🧮 Classifier Head

```
Linear(960 → 256) → ReLU → Dropout(0.5) → Linear(256 → 2)
                                                    │
                                            [logit_REAL, logit_FAKE]
                                                    │
                                            softmax → P(REAL), P(FAKE)
```

**Karar Eşikleri (3 Katmanlı — Eğitim Sonrası Optimize Edilecek):**

| Karar | Koşul (Varsayılan) | Anlamı |
|---|---|---|
| 🔴 **FAKE** | P(FAKE) ≥ threshold + margin | Yüksek güvenle sahte |
| 🟡 **UNCERTAIN** | threshold ± margin arası | Belirsiz bölge |
| 🟢 **REAL** | P(FAKE) ≤ threshold - margin | Yüksek güvenle gerçek |

> **⚠️ Not:** Eşik değerleri eğitim sonrası `scripts/find_threshold.py` ile **Youden J-statistic** kullanılarak validation seti üzerinden otomatik hesaplanır ve `models/optimal_threshold.txt`'e kaydedilir. Sabit eşikler (0.40/0.70) yerine veri bazlı optimal eşik kullanılacaktır.

---

### 📐 Loss Fonksiyonu — İkili Bileşik

```
L_total = 0.80 × L_focal + 0.20 × L_triplet
```

#### Focal Loss (γ=2.0, α=0.5)

```
FL(p_t) = -α_t × (1 - p_t)^γ × log(p_t)
```

Kolay örneklerde `(1-p_t)^γ` → 0'a yaklaşır, zor örneklerde loss yükselir. Label smoothing (0.05) ile overconfidence önlenir.

#### Triplet Contrastive Loss (margin=1.0)

```
L_triplet = max(0, d(anchor, positive) - d(anchor, negative) + margin)
```

Hard Negative Mining ile en zor örneklere odaklanır. Cosine distance kullanılır. REAL ve FAKE embedding'lerini 256-dim uzayda birbirinden uzaklaştırır.

---

### 🔬 XAI (Açıklanabilirlik) Yöntemleri

| Yöntem | Teknik | Ne Gösterir |
|---|---|---|
| **GradCAM++** | Gradient-weighted Class Activation | Modelin kararını en çok etkileyen bölgeleri ısı haritası olarak gösterir |
| **EigenCAM** | SVD tabanlı aktivasyon ayrıştırma | Birinci temel bileşeni görselleştirerek en dominant özellik bölgesini belirler |
| **FastCAM** | SMOE (Spatial Mixture of Experts) | Hızlı yaklaşım — birden fazla katmanın birleşik aktivasyonu |
| **Guided LIME** | Yüz bölgesi kısıtlı süperpiksel | Sadece yüz alanında hangi bölgelerin FAKE kararını tetiklediğini gösterir |

---

### 📊 Inference Pipeline

```
Görsel Yükleme
    │
    ├── 1. Yüz Algılama (MediaPipe) → Bounding box + 468 3D landmark
    │
    ├── 2. Ön İşleme
    │       ├── RGB: Resize(224) → Normalize(ImageNet)
    │       ├── Freq: HybridFrequencyExtractor → 18-ch tensor
    │       └── Mesh: 468 landmarks → flatten(1404)
    │
    ├── 3. TTA (Test-Time Augmentation)
    │       8× farklı augmentasyon ile tekrar tahmin → ortalaması alınır
    │
    ├── 4. Model Forward
    │       DualPathDeepfakeDetector(rgb, freq, mesh) → logits
    │
    ├── 5. Post-Processing
    │       ├── Softmax → P(REAL), P(FAKE)
    │       ├── Eşik kontrolü → FAKE / UNCERTAIN / REAL
    │       └── Güven skoru hesaplama
    │
    └── 6. XAI Haritaları (paralel)
            ├── GradCAM++ overlay
            ├── EigenCAM overlay
            ├── FastCAM overlay
            └── Guided LIME overlay
```

### Eğitim Stratejisi — 283K Dengeli Dataset İçin Optimize

| Bileşen | Detay | Neden |
|---|---|---|
| **Loss** | 0.8 × Focal(γ=2.0) + 0.2 × Triplet(m=1.0) | KD kaldırıldı → Focal ağırlığı artırıldı |
| **Optimizer** | AdamW (lr=3e-4, wd=1e-4, backbone lr×0.1) | Agresif LR — teacher yok, hızlı yakınsama |
| **Augmentation** | MixUp(α=0.2) + CutMix(60/40, α=1.0) + SBI + SocialCompress | Platform sıkıştırma simülasyonu dahil |
| **Scheduler** | Cosine Annealing (T_max=15, η_min=1e-6) + Warmup (1 epoch) | Kısa warmup + hızlı azalma |
| **Precision** | FP16 Mixed Precision (AMP) + TF32 (RTX 40xx) | Ada Lovelace optimizasyonu |
| **Accumulation** | 8 step → efektif batch = 128 | Daha stabil gradient |
| **Unfreeze** | Epoch 3'te backbone açılır (lr×0.1) | Erken fine-tune başlangıcı |
| **Label Smoothing** | 0.1 | Generalizasyon artırımı |
| **Early Stopping** | Patience=4 (val_AUC bazlı) | Hızlı karar |

---

## ✨ Özellikler

### 🔍 1. Single Image Analysis

Ana analiz sekmesi — görsel yükle, tek tıkla kapsamlı forensik sonuç al.

- **Üçlü Karar:** Eğitim sonrası optimize edilecek (Youden J-statistic bazlı)
- **TTA:** 5–15 augmentasyon ile güvenilirlik artırımı
- **4 XAI Haritası:** GradCAM++, EigenCAM, FastCAM, Guided LIME
- **Dual-Stream Yol Analizi:** RGB / DWT Frekans / Face Mesh ayrı ayrı
- **📱 Platform Tespiti:** JPEG forensik ile WhatsApp, Instagram, TikTok, Telegram otomatik tanıma
- **🔬 Forensik Konsensüs:** Model (%60) + ELA (%25) + Noise (%15) ağırlıklı birleşim
- **Tutarlılık Kontrolü:** Model-ELA-Noise arasındaki çelişkileri tespit ve raporlama
- **PDF Rapor:** Tek tıkla indirilebilir profesyonel rapor
- **Geri Bildirim:** GERÇEK / SAHTE butonları ile Active Learning havuzu

### 🛡️ 2. Robustness Test

Modelin saldırı ve sıkıştırma dayanıklılığını test eder.

- **Adversarial Attack:** FGSM, PGD, CW saldırı türleri
- **Epsilon Sweep:** ε=0.001→0.3 aralığında karar değişim analizi
- **Compression Sweep:** Q=10→100 JPEG sıkıştırma direnci
- **Platform Simülasyonu:** WhatsApp/Instagram/TikTok/Telegram sıkıştırma profilleri

### 📊 3. Analytics Dashboard

Tüm analizlerin istatistiksel takip paneli.

- **İstatistik Kartları:** Toplam analiz, FAKE/REAL oranları
- **Günlük Dağılım:** Zaman serisi grafikleri
- **XAI Kullanımı:** Hangi XAI yönteminin ne kadar kullanıldığı
- **Embedding Space:** t-SNE / UMAP ile özellik uzayı görselleştirme
- **Model Metrikleri:** ROC eğrisi, Confusion Matrix, AUC, F1-Score

### 🕒 4. Analiz Geçmişi

SQLite tabanlı kalıcı analiz kaydı — uygulama yeniden başlasa bile korunur.

### 💬 5. Analiz Asistanı

LLM destekli sohbet asistanı — analiz sonuçlarını yorumlayın.

- **Gemini 3.0 Pro:** Birincil LLM backend
- **Ollama Fallback:** Yerel qwen2.5:7b ile çevrimdışı çalışma
- **Yerel Bilgi Tabanı:** 12 konu başlığında offline cevaplama
- **Hızlı Sorular:** Tek tıkla sık sorulan konulara erişim

---

## 🚀 Kurulum

### Gereksinimler

- Python 3.14+
- NVIDIA GPU (CUDA destekli) — önerilir
- 8GB+ VRAM (RTX 4070 önerilir)

### Adımlar

```bash
# 1. Repoyu klonla
git clone https://github.com/your-username/DeepfakeULTRA.git
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

Uygulama **<http://localhost:7860>** adresinde açılacaktır.

### Docker ile Çalıştırma

```bash
docker-compose up --build
```

### Ortam Değişkenleri (Opsiyonel)

```env
GEMINI_API_KEY=your-google-ai-studio-key   # Chat asistanı için
JWT_SECRET=your-secret-key                  # API güvenliği
SLACK_WEBHOOK_URL=...                       # Bildirimler
```

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
├── app.py                          # Gradio ana arayüz (5 sekme)
├── main.py                         # FastAPI + Gradio birleşik başlatma
├── config.py                       # Merkezi konfigürasyon (dataclass)
├── requirements.txt                # Python bağımlılıkları
│
├── core/                           # Çekirdek modüller
│   ├── dual_mobilenetv3.py         # DualPathDeepfakeDetector modeli
│   ├── frequency.py                # DWT frekans görselleştirme
│   ├── frequency_v2.py             # Hibrit frekans çıkarıcı (DWT+DCT+Phase)
│   ├── data_pipeline.py            # Veri yükleme ve augmentasyon
│   ├── trainer.py                  # Eğitim döngüsü
│   ├── adversarial.py              # FGSM / PGD / CW saldırıları
│   ├── compression.py              # Platform sıkıştırma simülasyonu
│   ├── forensics.py                # ELA + Noise analizi
│   ├── face_detector.py            # Yüz algılama + bounding box
│   ├── platform_detector.py        # JPEG forensik platform tespiti
│   ├── fine_tuner.py               # Active Learning fine-tuning
│   ├── embedding_viz.py            # t-SNE / UMAP görselleştirme
│   ├── model_metrics.py            # ROC, CM, F1 hesaplama
│   ├── watermark.py                # Görünmez watermark
│   ├── sbi_augmentation.py         # Self-Blended Image augmentasyonu
│   ├── contrastive_loss.py         # Triplet Loss implementasyonu
│   ├── loss_utils.py               # Focal Loss + Label Smoothing
│   ├── evaluation.py               # Model değerlendirme
│   └── efficientnet_teacher.py     # Knowledge Distillation öğretmen model
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
│   └── components.py               # Tüm Gradio handler fonksiyonları
│
├── services/                       # Harici servisler
│   ├── llm_module.py               # Gemini / Ollama chat asistanı
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
│   ├── 05_split_dataset.py         # Temel veri bölme
│   ├── 06_smart_split.py           # Kalite-bilinçli akıllı bölme
│   ├── 28_quality_pipeline.py      # Kalite değerlendirme
│   ├── augment_dataset.py          # Augmentasyon pipeline'ı
│   ├── generate_sbi_data.py        # SBI veri üretimi
│   ├── find_threshold.py           # Youden J-statistic eşik optimizasyonu
│   ├── jury_evaluation.py          # Jury test seti değerlendirme
│   ├── test_evaluate.py            # Temel değerlendirme
│   ├── test_evaluate_advanced.py   # Gelişmiş metrik değerlendirme
│   └── train_monitor.py            # Eğitim izleme dashboard'u
│
├── models/                         # Model ağırlıkları
│   └── best_model.pth              # Eğitilmiş model checkpoint
│
├── dataset/                        # Veri seti dizini
│   └── faces/                      # Yüz verileri (FFPP, DF40, CelebA-HQ, FFHQ, UTKFace...)
│
├── configs/                        # Ek konfigürasyon dosyaları
├── tests/                          # Test suite
├── reports/                        # Analiz raporları
├── feedback_images/                # Active Learning görsel havuzu
├── Dockerfile                      # Docker imajı
└── docker-compose.yml              # Docker Compose
```

---

## 🎓 Eğitim Pipeline'ı

### Veri Hazırlama

```bash
# 1. Yüzleri çıkar (MTCNN/MediaPipe)
python scripts/01_extract_faces.py

# 2. Kalite-bilinçli akıllı bölme (train/val/test)
python scripts/06_smart_split.py

# 3. SBI augmentasyon verisi üret
python scripts/generate_sbi_data.py
```

### Model Eğitimi

```bash
# Ana eğitim (config.py parametreleri ile)
python -c "from core.trainer import Trainer; Trainer().train()"
```

### Değerlendirme

```bash
# Jury test seti değerlendirme
python scripts/jury_evaluation.py

# Eşik optimizasyonu (Youden J-statistic)
python scripts/find_threshold.py
```

### Veri Seti Kaynakları

| Kaynak | Tür | Kullanım |
|---|---|---|
| **FF++** | Face Swap (DeepFakes, FaceSwap, Face2Face, NeuralTextures) | Eğitim |
| **DF40** | 40 farklı deepfake yöntemi | Eğitim + Test |
| **CelebA-HQ** | Yüksek kalite gerçek yüzler | Eğitim |
| **FFHQ** | Yüksek çözünürlük gerçek yüzler | Eğitim |
| **UTKFace** | Demografik çeşitlilik (yaş, etnisite) | Jury Test |
| **SBI** | Self-Blended Image (sentetik augmentasyon) | Fine-tuning |

---

## 🚀 Production-Ready Pipeline

### Neden Gerekli?

Standart bir eğitim pipeline'ı **laboratuvar ortamında** yüksek metrikler üretebilir, ancak **gerçek dünya dağıtımında** ciddi performans düşüşleri yaşanır. Bu bölüm, modeli lab'dan production'a taşırken karşılaşılan 6 kritik riski ve çözümlerini açıklar.

### 🔴 Risk 1 — Veri Sızıntısı (Data Leakage)

**Sorun:** Eğitim setindeki bir görselin resize/crop/recompress edilmiş versiyonu test (Jury) setinde bulunursa, model ezberlediği için metrikler **sahte yüksek** çıkar. Gerçek dağıtımda AUC: %95 → %65 düşüşü yaşanır.

**Çözüm — 3 Katmanlı Leakage Prevention (`scripts/leakage_checker.py`):**

| Katman | Yöntem | Ne Yakalar |
|--------|--------|------------|
| **Katman 1** | MD5 Hash | Byte-level birebir kopya |
| **Katman 2** | pHash (Hamming < 10) | Resize, crop, recompress edilmiş near-duplicate |
| **Katman 3** | FaceNet Embedding (cosine > 0.85) | Farklı açı/ışık ama aynı çekim oturumu |

```bash
# Eğitim seti index'ini oluştur
python scripts/leakage_checker.py --build-index dataset/faces_split/train

# Jury setini kontrol et
python scripts/leakage_checker.py --check dataset/faces_split/train dataset/jury_test
```

### 🔴 Risk 2 — Kimlik Sızıntısı (Identity Leakage)

**Sorun:** Aynı kişinin farklı fotoğrafları hem eğitim hem test setinde varsa, model **manipülasyon artefaktını değil, kişinin yüzünü** tanıyarak karar verir. Yeni kişilerde çöker.

**Çözüm — Identity-Level Separation:**

- FaceNet 512-d embedding ile her yüzü vektörize et
- Cosine similarity > 0.70 → **aynı kimlik** → Jury'den engelle
- FAKE görsellerde **stratified sampling** ile 5 saldırı kategorisinden eşit temsil:

| Kategori | Kaynak | Saldırı Türü |
|----------|--------|-------------|
| GAN | DF40 | StyleGAN, ProGAN |
| Diffusion | GenImage | Stable Diffusion, DALL-E, Midjourney |
| FaceSwap | FF++ | DeepFaceLab, FaceSwap |
| Audio-driven | Custom | Yüz canlandırma (reenactment) |
| Hybrid | SIDSet | Tampered / composite |

```bash
# Jury setini identity-safe olarak genişlet (hedef: 5000)
python scripts/extend_jury.py
```

### 🟡 Risk 3 — Frekans İmzası Bozulması

**Sorun:** Standart augmentasyon (JPEG sıkıştırma, Gaussian blur, noise) FAKE görsellere de uygulandığında, GAN/Diffusion'ın bıraktığı **frekans domain parmak izi silinir**. Model bu kritik imzayı öğrenemez.

**Çözüm — Class-Aware Augmentation:**

| Hedef | İzin Verilen Augmentasyonlar | Yasaklanan |
|-------|------------------------------|------------|
| **REAL** görseller | Beauty filter, HDR, JPEG Q=30-50, Gaussian noise, DCT quantization noise, bilateral smooth | — |
| **FAKE** görseller | Horizontal flip, rotation (±10°), hafif brightness/contrast (±5%), DCT frequency mask | ❌ JPEG recompress, ❌ Blur, ❌ Noise, ❌ Heavy color jitter |

Ayrıca `WeightedRandomSampler` düzeltmesi:
- `replacement=True` modunda epoch başına ~%33 benzersiz örnek hiç çekilmez
- **Düzeltme:** `num_samples × 1.3` + `EPOCHS: 15 → 20` ile tam veri kapsama

### 🟡 Risk 4 — Eğitim Başında Zor Örneklerle Çökme

**Sorun:** Hard-real (ekran fotoğrafı, sosyal medya artefaktı) görseller eğitimin başından itibaren verilirse, model temel REAL/FAKE ayrımını öğrenemeden gradient çakışması yaşar → kararsız loss.

**Çözüm — Curriculum Learning:**

```
Epoch 0-4:   hard_real_ratio = 0.00  →  %100 temiz veri ile temel öğrenme
Epoch 5-9:   hard_real_ratio = 0.15  →  %15 hafif zor örnekler
Epoch 10-15: hard_real_ratio = 0.30  →  %30 orta zorluk
Epoch 16+:   hard_real_ratio = 0.40  →  %40 tam zorluk (production profili)
```

Screen recapture simülasyonu (moiré paterni + renk uzayı kayması + ekran vignette) hard-real üretimine dahildir.

### 🟡 Risk 5 — Kalibre Edilmemiş Olasılıklar

**Sorun:** Model "%90 FAKE" dediğinde gerçekte sadece %70 FAKE olabilir. Kalibre edilmemiş olasılıklar güvenilir karar vermeyi engeller.

**Çözüm — Temperature Scaling + Deployment Metrikleri:**

| Metrik | Hedef | Açıklama |
|--------|-------|----------|
| **ECE** | < %5 | Expected Calibration Error — olasılıklar gerçek doğruluğu yansıtmalı |
| **Brier Score** | < 0.15 | (1/N) Σ(p_i - y_i)² — düşük = iyi kalibrasyon |
| **Temperature** | Otomatik | Validation NLL minimize → optimal T |
| **Reliability Diagram** | Diagonal'e yakın | Güven vs. doğruluk görselleştirmesi |

```bash
# Tam değerlendirme (Brier + Temperature Scaling + ONNX + Reliability Diagram)
python scripts/evaluate_model.py

# Çıktılar:
# evaluation/metrics.json        — Tüm metrikler
# evaluation/reliability_diagram.png — Kalibrasyon görselleştirmesi
# models/calibration_weights.json — Deployment için T değeri
```

### 🟢 Risk 6 — Statik Model, Adaptif Değil

**Sorun:** Tek seferlik eğitim sonrası model eskir. Yeni deepfake teknolojilerine (SORA, Runway vb.) karşı adaptif olamaz. Hangi tür FAKE'leri kaçırdığı bilinmez.

**Çözüm — Hard-Negative Feedback Loop:**

```
┌────────────────────────────────────────────────────┐
│  evaluate_model.py → FP/FN tespit                  │
│         ↓                                          │
│  error_analysis.py → hard-negative mining           │
│         ↓                                          │
│  metadata/hard_negatives.json (replay buffer, FIFO) │
│         ↓                                          │
│  Epoch 10+ sonrası: %5 oranında eğitime karıştır   │
│         ↓                                          │
│  reports/weekly_{tarih}.md → trend izleme           │
└────────────────────────────────────────────────────┘
```

```bash
# Tam haftalık pipeline (değerlendirme + analiz + rapor)
python scripts/weekly_scheduler.py

# Windows Task Scheduler kurulumu (haftalık Pazartesi 09:00)
python scripts/weekly_scheduler.py --setup-task
```

### Büyük Resim

```
ÖNCE (standart pipeline):
  Lab AUC = 0.95 → Gerçek dünya AUC = ~0.65-0.75
  Neden: leakage, frekans bozulması, kalibrasyon eksikliği

SONRA (production-ready pipeline):
  Lab AUC = 0.90 (daha düşük ama GERÇEK)
  Gerçek dünya AUC = 0.85-0.90 (lab ile tutarlı = güvenilir)
  Kalibre edilmiş olasılıklar (ECE < %5)
  ONNX ile <50ms inference
  Haftalık otomatik izleme + adaptif öğrenme
```

### Yol Haritası

```bash
# 1. Bağımlılıkları yükle
pip install imagehash facenet-pytorch

# 2. Hard-real veri üret (screen recapture dahil)
python scripts/generate_hard_real.py

# 3. Jury setini genişlet (identity-safe, stratified, 5000 hedef)
python scripts/extend_jury.py

# 4. Yeni eğitim başlat (curriculum learning + class-aware augmentation aktif)
python -c "from core.trainer import Trainer; Trainer().train()"

# 5. Deployment metriklerini kontrol et
python scripts/evaluate_model.py

# 6. Haftalık izleme kur
python scripts/weekly_scheduler.py --setup-task
```

---

## ⚙️ Konfigürasyon

Tüm ayarlar `config.py` dosyasında merkezi olarak yönetilir:

| Konfigürasyon | Sınıf | Açıklama |
|---|---|---|
| `PathConfig` | Yol yapısı | Veri seti, model, log dizinleri |
| `ModelConfig` | Model parametreleri | Mimari, hiperparametre, eşikler |
| `APIConfig` | API ayarları | Port, JWT, CORS, rate limiting |
| `LLMConfig` | LLM ayarları | Gemini API, Ollama, model seçimi |

### Önemli Parametreler

```python
# Karar Eşikleri
FAKE_THRESHOLD = 0.70     # Bu değer üzerinde → FAKE
REAL_THRESHOLD = 0.40     # Bu değer altında → REAL
# Arada → UNCERTAIN

# Model Mimarisi
RGB_BACKBONE = "mobilenet_v3_large"
FREQ_BACKBONE = "mobilenet_v3_large"
TEACHER_BACKBONE = "efficientnet_b5"

# Eğitim
BATCH_SIZE = 16            # × 4 accumulation = 64 efektif
EPOCHS = 30
LEARNING_RATE = 1e-4
```

---

## 🛠️ Teknoloji Yığını

| Katman | Teknoloji |
|---|---|
| **Derin Öğrenme** | PyTorch 2.0+, timm, torchvision |
| **Yüz Algılama** | MediaPipe FaceLandmarker (468 3D landmark) |
| **Frekans Analizi** | PyWavelets (DWT), NumPy (DCT, Phase) |
| **XAI** | Captum (GradCAM++, EigenCAM), LIME |
| **Web UI** | Gradio 4.0+ |
| **REST API** | FastAPI + Uvicorn |
| **Veritabanı** | SQLite (analiz geçmişi + feedback) |
| **LLM** | Google Gemini 3.0 Pro + Ollama |
| **PDF Rapor** | fpdf2 |
| **Görselleştirme** | Plotly, Matplotlib |
| **Deney Takibi** | MLflow |
| **Konteyner** | Docker + Docker Compose |

---

## 📊 Performans

| Metrik | Değer |
|---|---|
| **Mimari** | DualPath (RGB + DWT + Mesh) |
| **Backbone** | MobileNetV3-Large (×2) |
| **Parametre** | ~15M |
| **Çıkarım Süresi** | ~200ms / görsel (GPU) |
| **VRAM Kullanımı** | ~2GB (inference) / ~6GB (eğitim, FP16) |

---

<div align="center">

**DeepfakeULTRA** — Forensik Deepfake Tespit Sistemi

Seydi Eryılmaz • 2026

*Tüm hakları saklıdır.*

</div>
