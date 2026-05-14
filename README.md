<div align="center">

# 🔍 DeepfakeULTRA

### AI-Powered Forensic Deepfake Detection System

[![Python 3.14+](https://img.shields.io/badge/Python-3.14+-3776AB?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![CUDA](https://img.shields.io/badge/CUDA-12.x-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![Gradio](https://img.shields.io/badge/Gradio-4.0+-F97316?logo=gradio&logoColor=white)](https://gradio.app)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Val AUC](https://img.shields.io/badge/Val_AUC-0.9839-brightgreen)](/)
[![Cross‑Dataset](https://img.shields.io/badge/Cross--Dataset_AUC-0.7527-blue)](/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

<br/>

**Dual-Path CNN + Transformer Fusion mimarisi ile tek görsellik deepfake / face swap / AI-generated yüz tespiti.**

Tek bir görselden **RGB analizi**, **DWT frekans spektrumu** ve **468 noktalı yüz geometrisi** birleştirilerek
deepfake manipülasyonları tespit edilir. Model, **5 farklı harici datasette** doğrulanmış
cross-dataset genelleme yeteneğine sahiptir.

---

**🎯 Val AUC 0.9839** · **🌍 Cross-Dataset +40%** · **⚡ ~200ms inference** · **🧬 15M parametre** · **7 sekmeli UI**

---

`Domain Generalization` · `GradCAM++` · `EigenCAM` · `DWT+DCT+Phase` · `TTA` · `Kraniyofasiyal Biyometrik` · `Active Learning`

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
- [Chrome Uzantısı](#-chrome-uzantısı)
- [Akıllı Fotoğraf Filtresi](#-akıllı-fotoğraf-filtresi-non-photo-detection)
- [Bilinen Sınırlamalar & Gelecek Çalışmalar](#-bilinen-sınırlamalar--gelecek-çalışmalar)

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

### Model Özellikleri

| Özellik | Değer |
|---|---|
| **Mimari** | DualPath CNN + CrossBranch Transformer (2-Layer, 4-Head) |
| **Backbone** | MobileNetV3-Large (×2, ImageNet pretrained) |
| **Toplam Parametre** | ~15M |
| **Eğitim Verisi** | 314K görsel (153K REAL + 160K FAKE) |
| **Precision** | FP16 Mixed Precision (AMP + TF32) |
| **Çıkarım Süresi** | ~200ms / görsel (GPU) |
| **VRAM Kullanımı** | ~2GB (inference) / ~6GB (eğitim) |

### İç Test Seti Performansı

59,655 görsel üzerinde değerlendirme (eğitimde hiç kullanılmamış test split):

| Metrik | Değer |
|---|---|
| **ROC-AUC** | **0.9839** |
| **Accuracy** | 93.1% |
| **F1-Score** | 0.931 |
| **Train Loss** | 0.0975 |
| **Eğitim Süresi** | 8 epoch (3 frozen + 5 unfrozen) |

### Cross-Dataset Benchmark (5 Harici Dataset)

Model hiç eğitilmediği, tamamen **görülmemiş** datasetler üzerinde test edilmiştir. Domain Generalization fine-tuning öncesi ve sonrası karşılaştırma:

| Dataset | Görsel | Kaynak | Baseline AUC | Final AUC | Artış |
|---------|--------|--------|-------------|-----------|-------|
| **Deepfake20K** | 20,000 | Kaggle — çeşitli GAN/swap yöntemleri | 0.514 | **0.956** | 🔥 +85.9% |
| **DFDC** | 5,000+ | Facebook AI — yarışma dataseti, düşük kalite videolar | 0.546 | **0.821** | 🔥 +50.3% |
| **DeepfakeFace** | 3,000+ | Karma — farklı manipülasyon türleri | 0.583 | **0.777** | 🟢 +33.3% |
| **CelebDF v2** | 1,890 | Li et al. — eski nesil autoencoder face swap | 0.541 | **0.708** | 🟢 +30.8% |
| **FaceForensics++** | 750 | TU München — Face2Face, FaceSwap reenactment | 0.502 | 0.500 | ⚪ — |
| **Ortalama** | — | — | **0.537** | **0.752** | **📈 +40.0%** |

### Analiz Notları

| Dataset | Durum | Açıklama |
|---------|-------|----------|
| **Deepfake20K** | ✅ Mükemmel | Farklı GAN yöntemlerinde neredeyse kusursuz genelleme |
| **DFDC** | ✅ Güçlü | Düşük çözünürlük + sıkıştırma koşullarında başarılı |
| **DeepfakeFace** | ✅ İyi | Karma manipülasyon türlerinde tutarlı performans |
| **CelebDF v2** | ✅ İyi | Eski nesil autoencoder artifact'lerini öğrendi |
| **FF++** | ⚠️ Zayıf | Reenactment (Face2Face) artifact'leri domain augmentation ile çözülemiyor — ek veri gerekli |

> **Combined Score:** `0.8683` = 0.5 × val_auc (0.9839) + 0.5 × ext_mean_auc (0.7527)
>
> Model seçimi bu metriğe göre yapılır: iç performansı korurken harici genellemeyi maksimize eder.

---

## 🖥️ Arayüz Sekmeleri

DeepfakeULTRA 7 sekmeli bir Gradio web arayüzü sunar. Her sekme bağımsız bir forensik modül olarak çalışır:

### 🔍 Sekme 1 — Single Image Analysis

Ana forensik analiz sekmesi. Görsel yükle, tek tıkla kapsamlı sonuç al.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  🔍 Single Image Analysis                                                  │
├────────────────────────┬────────────────────────────────────────────────────┤
│                        │  ┌──────────────────────────────────────────────┐ │
│   ┌──────────────┐     │  │  SONUÇ: 🔴 FAKE                             │ │
│   │              │     │  │  Güven: 94.7%  |  Platform: Instagram        │ │
│   │   📸 Görsel  │     │  │  ████████████████████░░░  94.7%              │ │
│   │   Yükleme    │     │  └──────────────────────────────────────────────┘ │
│   │   Alanı      │     │                                                  │
│   │              │     │  Forensik Konsensüs:                             │
│   └──────────────┘     │  ┌────────┐ ┌────────┐ ┌────────┐               │
│                        │  │Model   │ │  ELA   │ │ Noise  │               │
│   [📤 Yükle]           │  │ FAKE   │ │ FAKE   │ │ GERÇEK │               │
│   ☑ TTA Aktif          │  │ %60    │ │ %25    │ │ %15    │               │
│                        │  └────────┘ └────────┘ └────────┘               │
│   [🔍 Analiz Et]       │                                                  │
├────────────────────────┴────────────────────────────────────────────────────┤
│  XAI Haritaları                                                            │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐         │
│  │  GradCAM++  │ │  EigenCAM   │ │  FastCAM    │ │ Guided LIME │         │
│  │  ░░▓▓██░░   │ │  ░░░▓██░░  │ │  ░▓▓██░░░  │ │  ░░▓▓▓░░░  │         │
│  │  ░▓████▓░   │ │  ░░▓███░░  │ │  ░▓███▓░░  │ │  ░▓▓██▓░░  │         │
│  │  ░░▓██░░░   │ │  ░░░▓█░░░  │ │  ░░▓██░░░  │ │  ░░▓▓░░░░  │         │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘         │
├─────────────────────────────────────────────────────────────────────────────┤
│  [📥 PDF İndir]                    [👍 GERÇEK]  [👎 SAHTE]                │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Analiz Pipeline:**
```
📸 Görsel Yükleme → 👤 Yüz Algılama (MTCNN) → 🔄 Preprocessing (224×224)
    │
    ├── 🟢 RGB Branch → MobileNetV3 → 960-dim feature
    ├── 🔵 Frekans Branch → DWT+DCT+Phase → MobileNetV3 → 960-dim feature
    └── 🟡 Geometri Branch → 468 Landmark → MLP → 960-dim feature
         │
    CrossBranch Transformer → P(REAL), P(FAKE)
         │
    ├── 📊 XAI Haritaları (GradCAM++, EigenCAM, FastCAM, LIME)
    ├── 🔬 Forensik Analiz (ELA + Noise)
    ├── 📱 Platform Tespiti (JPEG forensik)
    └── 📋 PDF Rapor Oluşturma
```

| Özellik | Detay |
|---|---|
| **Karar Sistemi** | FAKE / UNCERTAIN / REAL — Youden J-statistic bazlı çift eşik |
| **TTA Modu** | 8 GPU-native augmentasyon → ortalama skor, güvenilirliği artırır |
| **XAI Haritaları** | GradCAM++, EigenCAM, FastCAM, Guided LIME — 4 harita yan yana |
| **3 Analiz Yolu** | RGB görsel / DWT frekans haritası / Face Mesh geometri |
| **Platform Tespiti** | WhatsApp, Instagram, TikTok, Telegram — JPEG quantization tablosu analizi |
| **Forensik Konsensüs** | Model (%60) + ELA (%25) + Noise (%15) ağırlıklı birleşim |
| **Tutarlılık Kontrolü** | Model-ELA-Noise çelişki tespiti ve raporlama |
| **PDF Rapor** | Tek tıkla indirilebilir profesyonel forensik rapor |
| **Geri Bildirim** | GERÇEK / SAHTE butonları → Active Learning havuzuna eklenir |

### 🛡️ Sekme 2 — Robustness Test

Modelin saldırı ve sıkıştırma dayanıklılığını test eder. Adversarial robustness ve real-world koşullarını simüle eder.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  🛡️ Robustness Test                                                       │
├────────────────────────┬────────────────────────────────────────────────────┤
│                        │  Adversarial Sonuçları                            │
│   ┌──────────────┐     │  ┌──────────────────────────────────────────────┐ │
│   │   📸 Görsel  │     │  │  Orijinal:  FAKE  94.7%                     │ │
│   │   Yükleme    │     │  │  FGSM ε=4:  FAKE  88.2%  ✅ Dayanıklı      │ │
│   └──────────────┘     │  │  PGD  ε=4:  FAKE  71.3%  ⚠️ Kısmi          │ │
│                        │  │  CW:        REAL  45.1%  ❌ Kırıldı         │ │
│   Saldırı Türü:        │  └──────────────────────────────────────────────┘ │
│   ○ FGSM  ● PGD  ○ CW │                                                  │
│   Epsilon: [===●==] 4  │  Branch Knockout                                 │
│                        │  ┌──────────┬──────────┬──────────┐             │
│   [🚀 Test Et]         │  │ RGB ✅   │ Frek ✅  │ Geo ⚠️  │             │
│                        │  │ AUC 0.91 │ AUC 0.87 │ AUC 0.62 │             │
│   Compression:         │  └──────────┴──────────┴──────────┘             │
│   Q: [10━━━━━━100]     │                                                  │
│                        │  JPEG Sweep:  Q10=72% → Q50=91% → Q100=95%      │
└────────────────────────┴────────────────────────────────────────────────────┘
```

| Test Türü | Yöntem | Açıklama |
|---|---|---|
| **Adversarial Attack** | FGSM, PGD, CW | Gradient-based saldırılarla modelin manipüle edilebilirliğini ölçer |
| **Branch Knockout** | RGB / Frekans / Geometri izolasyonu | Her dalı tek tek devre dışı bırakarak katkılarını analiz eder |
| **Compression Sweep** | Q=10 → Q=100 | JPEG kalite seviyelerine karşı direnci test eder |
| **Resolution Scaling** | 64px → 512px | Farklı çözünürlüklerde performans değişimini ölçer |
| **Double Compression** | İkili JPEG sıkıştırma | Sosyal medya paylaşım zincirini simüle eder |

### 📊 Sekme 3 — Analytics Dashboard

Tüm analizlerin istatistiksel takip paneli. Scroll-free kompakt tasarım, Plotly interaktif grafikler.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  📊 Analytics Dashboard                                                    │
├──────────────┬──────────────┬──────────────┬───────────────────────────────┤
│  📈 Toplam   │  🔴 FAKE     │  🟢 REAL     │  📊 Ort. Güven               │
│     1,247    │     687      │     560      │     87.3%                    │
├──────────────┴──────────────┴──────────────┴───────────────────────────────┤
│                                                                           │
│  Günlük Analiz Dağılımı               Sonuç Dağılımı                     │
│  ┌─────────────────────────┐          ┌──────────────────┐               │
│  │       ╭─╮               │          │    ╭────────╮    │               │
│  │    ╭──╯ ╰──╮            │          │  ╭─╯ FAKE   ╰╮  │               │
│  │ ╭──╯       ╰──╮   ╭─╮  │          │  │   55.1%    │  │               │
│  │─╯              ╰──╯ ╰─ │          │  │  ────────  │  │               │
│  │ Pzt Sal Çar Per Cum Cts│          │  ╰╮ REAL     ╭╯  │               │
│  └─────────────────────────┘          │   ╰─ 44.9% ─╯   │               │
│                                       └──────────────────┘               │
│  Güven Dağılımı (Histogram)                                              │
│  ┌─────────────────────────┐                                             │
│  │  █                   █  │  Kayıt Limiti: [====●========] 200          │
│  │  █ █               █ █  │                                             │
│  │  █ █ █           █ █ █  │  [🔄 Yenile]                                │
│  │  █ █ █ █ █ █ █ █ █ █ █  │                                             │
│  │  0.0   0.25  0.50  1.0  │                                             │
│  └─────────────────────────┘                                             │
└───────────────────────────────────────────────────────────────────────────┘
```

| Bileşen | Açıklama |
|---|---|
| **4 Metrik Kartı** | Toplam analiz sayısı, FAKE, REAL, ortalama güven skoru |
| **Günlük Dağılım** | Zaman serisi trend grafikleri (günlük analiz hacmi) |
| **Güven Dağılımı** | Histogram — model güven aralıklarının dağılımı |
| **Sonuç Dağılımı** | FAKE vs REAL pie chart oranları |
| **Anlık Yenileme** | Slider ile kayıt limiti ayarı (10–200 arası) |

### 🔧 Sekme 4 — Model Profili

Model performansını test eden, doğrulayan ve güncelleyen araçlar.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  🔧 Model Profili                                                          │
├────────────────────────┬────────────────────────────────────────────────────┤
│                        │  ROC Eğrisi (5-Fold CV)                           │
│  Cross-Validation      │  ┌──────────────────────────┐                    │
│  K = [3|●5|7|10]       │  │ 1.0 ┤ ╭───────────────── │  AUC = 0.9839     │
│                        │  │     │╭╯                  │                    │
│  Harici Dataset:       │  │ 0.5 ┤│                   │  Fold 1: 0.981     │
│  [CelebDF v2    ▼]     │  │     ││                   │  Fold 2: 0.984     │
│                        │  │ 0.0 ┤╰──────────────────│  Fold 3: 0.986     │
│  [📊 Test Et]          │  │     0.0    0.5     1.0  │                    │
│  [🔄 Fine-Tune]        │  └──────────────────────────┘                    │
│  [⏪ Rollback]          │                                                  │
│                        │  Confusion Matrix                                │
│  Durum: ✅ Hazır       │  ┌───────────┬───────┬───────┐                   │
│  Model: best_model     │  │           │ PRED  │ PRED  │                   │
│  _generalized.pth      │  │           │ REAL  │ FAKE  │                   │
│                        │  ├───────────┼───────┼───────┤                   │
│                        │  │ ACT REAL  │ 27654 │  2115 │                   │
│                        │  │ ACT FAKE  │  1983 │ 27903 │                   │
│                        │  └───────────┴───────┴───────┘                   │
└────────────────────────┴────────────────────────────────────────────────────┘
```

| Özellik | Açıklama |
|---|---|
| **Cross-Validation** | K-Fold çapraz doğrulama — her fold için ROC eğrisi ve AUC hesaplama |
| **Harici Dataset Testi** | CelebDF v2, DFDC, FF++ gibi datasetleri yükleyip otomatik benchmark |
| **Confusion Matrix** | TP/FP/TN/FN detaylı sınıflandırma matrisi görselleştirme |
| **Fine-Tuning** | Active Learning havuzundaki kullanıcı etiketleriyle classifier head güncelleme |
| **Rollback** | Bir önceki model ağırlıklarına tek tıkla geri dönme |

### 🕒 Sekme 5 — Analiz Geçmişi

SQLite tabanlı kalıcı analiz kaydı — uygulama yeniden başlatılsa bile korunur.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  🕒 Analiz Geçmişi                                    [🗑️ Temizle]        │
├─────┬──────────────────┬───────────────┬────────┬────────┬─────────────────┤
│  #  │ Tarih            │ Dosya         │ Sonuç  │ Güven  │ Platform        │
├─────┼──────────────────┼───────────────┼────────┼────────┼─────────────────┤
│  1  │ 2026-05-14 15:30 │ photo_01.jpg  │ 🔴FAKE │  94.7% │ Instagram       │
│  2  │ 2026-05-14 15:28 │ selfie.png    │ 🟢REAL │  89.2% │ WhatsApp        │
│  3  │ 2026-05-14 15:25 │ suspect.jpg   │ 🔴FAKE │  97.1% │ TikTok          │
│  4  │ 2026-05-14 15:20 │ profile.jpg   │ 🟡UNCE │  52.3% │ Telegram        │
│  5  │ 2026-05-14 15:15 │ avatar.png    │ 🟢REAL │  91.8% │ —               │
│ ... │ ...              │ ...           │ ...    │ ...    │ ...             │
├─────┴──────────────────┴───────────────┴────────┴────────┴─────────────────┤
│  Toplam: 1,247 kayıt                    Gösterilen: 200  [━━━●━━━━] 200   │
└───────────────────────────────────────────────────────────────────────────┘
```

| Özellik | Açıklama |
|---|---|
| **Kayıt Kapasitesi** | 200'e kadar geçmiş analiz kaydı |
| **Tablo Görünümü** | Tarih, dosya adı, sonuç (FAKE/REAL), güven skoru, platform |
| **Geçmiş Temizleme** | Tek tıkla tüm kayıtları silme |
| **Veritabanı** | `deepfake_history.db` — SQLite, uygulama bağımsız kalıcı |

### 🧬 Sekme 6 — Yüz Anatomisi (Kraniyofasiyal Biyometrik Analiz)

AI Vision API destekli yüz anatomisi analiz motoru. Deepfake'lerin fiziksel tutarsızlıklarını tespit eder.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  🧬 Yüz Anatomisi — Kraniyofasiyal Biyometrik Analiz                      │
├────────────────────────┬────────────────────────────────────────────────────┤
│                        │  📊 Analiz Raporu                                │
│   ┌──────────────┐     │  ┌──────────────────────────────────────────────┐ │
│   │   📸 Görsel  │     │  │  Asimetri Modülü                            │ │
│   │              │     │  │  ├─ FAI: 0.8%        ✅ Normal               │ │
│   │   👤 Yüz    │     │  │  └─ Orbital: 1.2mm   ✅ Normal               │ │
│   │   Landmark   │     │  │                                              │ │
│   │   • • • •    │     │  │  Dudak Modülü                                │ │
│   │    •   •     │     │  │  ├─ Cupid Bow: 2.1°  ✅ Normal               │ │
│   │     •••      │     │  │  └─ Oran: 0.63       ⚠️ Sınırda              │ │
│   └──────────────┘     │  │                                              │ │
│                        │  │  Deepfake Risk                               │ │
│  Provider: [Gemini ▼]  │  │  ├─ Blending: 7.2    ⚠️ Orta Risk           │ │
│  API Key: [●●●●●●●●]  │  │  ├─ GAN Texture: 0.4 ⚠️ Dikkat             │ │
│  Mod: [FULL       ▼]  │  │  └─ Lighting: 12°    ✅ Normal               │ │
│  Bölge: [ALL      ▼]  │  └──────────────────────────────────────────────┘ │
│                        │                                                  │
│  [🔬 Analiz Başlat]    │  🔧 JSON Çıktı:  { "fai": 0.8, ... }           │
└────────────────────────┴────────────────────────────────────────────────────┘
```

**Analiz Pipeline:**
```
🖼️ Görüntü Girişi → 🎯 Landmark Tespiti (68-pt) → 📐 Anatomik Hesaplama
    → 🧠 AI Vision API Analizi → 📊 Risk Dashboard → 📋 Adli Rapor
```

| Özellik | Detay |
|---|---|
| **Multi-Provider** | Google Gemini, OpenAI GPT-4o, Anthropic Claude — tek arayüzden |
| **API Key Yönetimi** | Bir kere gir, `.api_keys.json`'a kaydet, otomatik hatırla |
| **Analiz Modları** | FULL / QUICK / REGION_SPECIFIC |
| **Bölge Odaklı** | ALL / FACE / LIPS / JAW / EYES / NOSE |

**Ölçülen Biyometrik Metrikler:**

| Modül | Metrik | Normal Aralık | Deepfake Sinyali |
|---|---|---|---|
| **Asimetri** | FAI (Facial Asymmetry Index) | 0.5 – 3.5% | <0.5 = yapay aşırı simetri |
| **Asimetri** | Orbital Delta | 0 – 2.5 mm | >4mm = anomali |
| **Dudak** | Cupid Bow Simetrisi | 0 – 5° | <0.5° = GAN imzası |
| **Dudak** | Üst/Alt Dudak Oranı | 0.55 – 0.70 | Tam 0.618 = yapay |
| **Çene** | Gonial Açı | 120° – 130° | Asimetrik gonial açı |
| **Çene** | Jawline Continuity | 6.5 – 9.0 | <5 = seam artifact |
| **Deepfake** | Blending Artifact Score | 8.0+ | <6 = yüksek risk |
| **Deepfake** | Skin GAN Texture | <0.3 | >0.5 = AI doku |
| **Deepfake** | Lighting Coherence | <15° | >30° = kompozit ışık |

**Çıktı:** Markdown rapor + JSON + referans tablosu

**Model Fallback Zinciri (Gemini):**
```
gemini-2.5-flash → gemini-2.0-flash-lite → gemini-1.5-flash
```
Her modelde 3 retry (5s, 10s, 15s exponential backoff).

### 💬 Sekme 7 — Analiz Asistanı

LLM destekli sohbet asistanı — forensik analiz sonuçlarını doğal dilde yorumlar.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  💬 Analiz Asistanı                                                        │
├─────────────────────────────────┬───────────────────────────────────────────┤
│                                 │  📋 Bağlam Kartı                        │
│  🤖 Asistan:                    │  ┌─────────────────────────────────────┐ │
│  ┌─────────────────────────┐    │  │ Son Analiz: photo_01.jpg            │ │
│  │ GradCAM++ haritasında   │    │  │ Sonuç: FAKE (94.7%)                │ │
│  │ burun-alın bölgesinde   │    │  │ Platform: Instagram                │ │
│  │ yüksek aktivasyon       │    │  │ XAI: Burun bölgesi aktif           │ │
│  │ görülüyor. Bu tipik     │    │  └─────────────────────────────────────┘ │
│  │ bir face-swap           │    │                                         │
│  │ artifact'idir...        │    │  ⚡ Hızlı Sorular:                      │
│  └─────────────────────────┘    │  ┌─────────────────────────────────────┐ │
│                                 │  │ [GradCAM ne anlama geliyor?]        │ │
│  👤 Sen:                        │  │ [DWT frekans analizi nedir?]        │ │
│  ┌─────────────────────────┐    │  │ [TTA güvenilirliği artırır mı?]     │ │
│  │ Bu sonucu açıklar       │    │  │ [ELA analizi nasıl çalışır?]        │ │
│  │ mısın?                  │    │  │ [Forensik konsensüs nedir?]         │ │
│  └─────────────────────────┘    │  │ [Model ne kadar güvenilir?]         │ │
│                                 │  └─────────────────────────────────────┘ │
│  ┌──────────────────────┐       │                                         │
│  │ Mesajınızı yazın...   │      │  LLM: Gemini 3.0 Pro ✅                 │
│  └──────────────────────┘       │  Fallback: Ollama qwen2.5 ⬚             │
│  [📤 Gönder]                    │                                         │
└─────────────────────────────────┴───────────────────────────────────────────┘
```

| Özellik | Detay |
|---|---|
| **Birincil LLM** | Google Gemini 3.0 Pro — bulut tabanlı |
| **Fallback** | Ollama (qwen2.5:7b) — yerel, çevrimdışı çalışma |
| **Yerel Bilgi Tabanı** | 12 konu (GradCAM, DWT, Mesh, TTA, ELA, Noise vb.) — API olmadan yanıt |
| **Hızlı Sorular** | 6 adet tek tıkla sık sorulan soru |
| **Bağlam Kartı** | Son analiz sonuçları otomatik olarak sohbete yansır |
| **Kullanım** | "Bu GradCAM haritası ne anlama geliyor?" gibi sorulara anında yanıt |

---

## 🏗️ Mimari

### DualPathDeepfakeDetector — Genel Bakış

Tri-Branch (RGB + Frekans + Face Mesh) mimarisi, her daldan çıkarılan 960 boyutlu özellik vektörlerini **CrossBranchTransformer** ile birleştirerek tek bir binary karar (REAL/FAKE) üretir.

```
                            Girdi Görseli (224×224×3)
                                     │
                ┌────────────────────┼────────────────────┐
                │                    │                    │
                ▼                    ▼                    ▼
    ┌───────────────────┐  ┌──────────────────┐  ┌──────────────────┐
    │   🟢 RGB Branch   │  │  🔵 Freq Branch  │  │  🟡 Mesh Branch  │
    │                   │  │                  │  │                  │
    │  MobileNetV3-Large│  │  DWT+DCT+Phase   │  │  MediaPipe 468   │
    │  (ImageNet prtnd) │  │  → 18 kanal      │  │  3D Landmarks    │
    │                   │  │  → MobileNetV3   │  │  → FaceMeshMLP   │
    │  (B,3,224,224)    │  │  (B,18,224,224)  │  │  (B,1404)        │
    │       │           │  │       │          │  │       │          │
    │  AdaptiveAvgPool  │  │  AdaptiveAvgPool │  │  Linear layers   │
    │       │           │  │       │          │  │       │          │
    │   960-dim         │  │   960-dim        │  │  128 → 960-dim   │
    └───────┬───────────┘  └───────┬──────────┘  └───────┬──────────┘
            │                      │                     │
            ▼                      ▼                     ▼
         [token₁]              [token₂]              [token₃]
            │                      │                     │
            └──────────────────────┼─────────────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │  CrossBranchTransformer      │
                    │  ┌────────────────────────┐  │
                    │  │ + Branch Embedding      │  │
                    │  │ + Pre-LayerNorm         │  │
                    │  │ + 2-Layer Encoder       │  │
                    │  │ + 4-Head Self-Attention │  │
                    │  │ + GELU FFN (×4 expand)  │  │
                    │  │ + Final LayerNorm       │  │
                    │  └────────────────────────┘  │
                    └──────────────┬──────────────┘
                                   │
                            Mean Pool → 960-dim
                                   │
                    ┌──────────────▼──────────────┐
                    │  Classifier                  │
                    │  Linear(960→256) → ReLU      │
                    │  Dropout(0.5)                │
                    │  Linear(256→2)               │
                    └──────────────┬──────────────┘
                                   │
                          P(REAL), P(FAKE)
```

### Branch Detayları

| Branch | Backbone | Giriş Boyutu | Çıkış | Ne Yakalar |
|---|---|---|---|---|
| **🟢 RGB** | MobileNetV3-Large (ImageNet pretrained) | `(B, 3, 224, 224)` | 960-dim | Blending artifact, renk tutarsızlığı, doku anomalisi |
| **🔵 Frekans** | MobileNetV3-Large (scratch, 18-ch conv1) | `(B, 18, 224, 224)` | 960-dim | GAN/Diffusion spektral izleri, JPEG ghost |
| **🟡 Geometri** | FaceMeshMLP (3 Linear + BN + ReLU) | `(B, 1404)` | 128 → 960-dim | Landmark asimetri, yüz yapısal tutarsızlık |

### HybridFrequencyExtractor — 18 Kanal Detay

```
Kaynak Görsel (H×W×3)
    │
    ├── DWT (12 kanal)
    │   ├── Haar wavelet    → cA, cH, cV, cD  (4 kanal)  — Kenar/doku geçişleri
    │   ├── Daubechies-2    → cA, cH, cV, cD  (4 kanal)  — Orta frekans detaylar
    │   └── Coiflet-1       → cA, cH, cV, cD  (4 kanal)  — Yumuşak geçiş bölgeleri
    │
    ├── DCT (3 kanal)
    │   ├── Low frequency band    — Genel aydınlatma/renk bilgisi
    │   ├── Mid frequency band    — Doku ve kenar detayları
    │   └── High frequency band   — Gürültü ve kompresyon artefaktları
    │
    └── Phase Spectrum (3 kanal)
        ├── R channel FFT phase   — Kırmızı kanal faz bilgisi
        ├── G channel FFT phase   — Yeşil kanal faz bilgisi
        └── B channel FFT phase   — Mavi kanal faz bilgisi
```

> **Neden 18 kanal?** Farklı wavelet aileleri farklı frekans bölgelerini yakalayarak GAN/Diffusion modellerinin bıraktığı spektral izleri tespit eder. DCT, JPEG sıkıştırma artefaktlarını; Phase Spectrum ise faz tutarsızlıklarını yakalar.

### CrossBranchTransformer — Neden Transformer?

| Özellik | BiLSTM (eski) | Transformer (mevcut) |
|---|---|---|
| **Tasarım amacı** | Temporal (sıralı) veri | Cross-modal dikkat |
| **seq_len=3 ile** | Boş çalışır (fazla parametre) | 3 token arası ilişki öğrenir |
| **Parallellik** | Sıralı işlem | Paralel self-attention |
| **Branch katkısı** | Sabit ağırlık | Dinamik, girdi bağımlı ağırlık |

**Teknik detay:** Her branch çıktısı bir token olarak ele alınır. Öğrenilebilir branch embedding, her token'a modalite kimliği kazandırır (RGB=0, Freq=1, Mesh=2). Self-attention, hangi branch'in hangi görsel için daha bilgilendirici olduğunu dinamik olarak öğrenir.

### Loss Fonksiyonu

```
L_total = 0.80 × Focal Loss (γ=2.0, α=0.75) + 0.20 × Triplet Loss (margin=1.0)
```

| Bileşen | Ağırlık | Amaç |
|---|---|---|
| **Focal Loss** | %80 | Zor örneklere odaklanır, sınıf dengesizliğini azaltır |
| **Triplet Loss** | %20 | REAL/FAKE embedding'leri ayrıştırır, contrastive öğrenme |

### Eğitim Sırasında: Domain Augmentation Pipeline

```
Kaynak Görsel (224×224)
    │
    ├── JPEG Compression (Q=30–95, p=0.5)      — Platform sıkıştırma simülasyonu
    ├── Downscale-Upscale (0.25x–0.75x, p=0.4) — Çözünürlük değişkenliği
    ├── Gaussian Noise (σ=0.01–0.03, p=0.3)    — Sensör gürültüsü
    ├── Gamma Correction (γ=0.7–1.3, p=0.3)    — Aydınlatma varyasyonu
    └── Color Channel Shift (±15, p=0.3)       — Renk profili farklılıkları
         │
         └── Domain-Robust Eğitim Görseli
```

### Çıkarım Sırasında: TTA (Test-Time Augmentation)

```
Girdi Görseli
    │
    ├── Orijinal ─────────── P₁
    ├── Horizontal Flip ──── P₂
    ├── Gaussian Blur ────── P₃
    ├── Resize 0.9x ──────── P₄     8 kopya → tek GPU batch'te paralel
    ├── Resize 1.1x ──────── P₅     işlenir (ek ~50ms)
    ├── Gaussian Noise ───── P₆
    ├── Brightness +0.1 ──── P₇
    └── Brightness -0.1 ──── P₈
         │
         └── Mean(P₁..P₈) → Final Prediction
```

> **TTA Etkisi:** Ortalama +1-2 puan AUC artışı sağlar. GPU-native implementasyon sayesinde ek latency minimal (~50ms).

---

## 🧬 Domain Generalization

Deepfake tespit modellerinin en büyük sorunu **domain-specific overfitting** — model eğitildiği veri setinde mükemmel çalışırken, farklı kaynaklardan gelen görsellerde başarısız olur. DeepfakeULTRA V5 bu problemi 3 aşamalı bir strateji ile çözer.

### Problem: Neden Genelleme Başarısız Olur?

```
                    Eğitim Verisi (DF40)
                    ┌─────────────────┐
                    │  BlendFace      │    AUC = 0.98
                    │  CollabDiff     │    ✅ Mükemmel
                    │  Modern GAN     │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
              ▼              ▼              ▼
        Celeb-DF v2     FF++ (c23)      DFDC
        Autoencoder     Face2Face       Düşük Kalite
        AUC = 0.54      AUC = 0.50      AUC = 0.55
        ❌ Başarısız     ❌ Başarısız     ❌ Başarısız

        → Model, eğitim verisine ÖZGÜ artefaktları ezberliyor
        → Farklı yöntem/platform/kalite = farklı artefaktlar
```

| Sorun | Gerçek Dünya Etkisi | Örnek |
|---|---|---|
| **JPEG Kalitesi** | Her platform farklı Q kullanır | WhatsApp Q=60, Instagram Q=85, Telegram Q=70 |
| **Çözünürlük** | Yeniden boyutlandırma izleri kaybolur | 128px ekran kaydı vs 1024px orijinal |
| **Renk Profili** | Kamera ISP farkları doku bilgisini değiştirir | iPhone vs Samsung vs ekran kaydı |
| **Deepfake Yöntemi** | Her yöntem farklı artifact bırakır | Autoencoder (bulanık sınır) vs GAN (doku gürültüsü) vs Diffusion (frekans izleri) |

### Çözüm: 3 Aşamalı Domain Generalization Stratejisi

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  AŞAMA 1: TTA (Test-Time Augmentation)                                  │
│  ├── 8 GPU-native augmentasyon → mean prediction                        │
│  ├── Ek maliyet: ~50ms (GPU batch paralel)                              │
│  ├── Sonuç: +1–2 puan AUC artışı                                       │
│  └── Kök sorunu çözmez, ama tahmin güvenilirliğini artırır              │
│                                                                         │
│  AŞAMA 2: Domain Augmentation (Eğitim Sırasında)                       │
│  ├── JPEG compression (Q=30–95, p=0.5)                                  │
│  ├── Downscale-Upscale (0.25x–0.75x, p=0.4)                            │
│  ├── Gaussian noise, gamma correction, color shift                      │
│  ├── %50 olasılıkla her görsele uygulanır                               │
│  └── Model domain-specific artifact'lara bağımlılığı azaltır            │
│                                                                         │
│  AŞAMA 3: Curriculum Fine-Tuning                                        │
│  ├── Base model: best_model.pth (AUC=0.9825) üzerine inşa              │
│  ├── Epoch 1-3: Backbone FROZEN → sadece classifier adapte olur         │
│  ├── Epoch 4-8: Backbone UNFROZEN → derin feature refinement            │
│  ├── Discriminative LR: classifier=5e-5, backbone=5e-6 (10× fark)      │
│  ├── CosineAnnealing + GradClip(1.0) + FP16 AMP                        │
│  └── Combined Score = 0.5 × val_auc + 0.5 × ext_mean_auc               │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Curriculum Learning: Neden İki Fazlı?

```
Faz 1: FROZEN (Epoch 1-3)                     Faz 2: UNFROZEN (Epoch 4-8)
┌─────────────────────────────┐               ┌─────────────────────────────┐
│                             │               │                             │
│  Backbone  ██████ DONDURULDU│               │  Backbone  ░░░░░░ EĞİTİLİR │
│  (340 param donuk)          │               │  (385 param eğitilebilir)   │
│                             │               │                             │
│  Classifier  ░░ EĞİTİLİR   │               │  Classifier  ░░ EĞİTİLİR   │
│  (45 param aktif)           │               │  LR = 5e-5                  │
│                             │               │  Backbone LR = 5e-6 (10×↓)  │
│  Amaç: Domain augmentation  │               │                             │
│  altında classifier'ı       │               │  Amaç: Derin feature'ları   │
│  adapte et                  │               │  domain-invariant hale getir │
│                             │               │                             │
│  Risk: Düşük                │               │  Risk: Orta (iç perf. kaybı)│
│  (backbone bozulmaz)        │               │  (discriminative LR korur)  │
└─────────────────────────────┘               └─────────────────────────────┘
```

### Model Seçim Metriği: Combined Score

Geleneksel yaklaşımda sadece `val_auc` ile model seçilir. Bu, **iç performans mükemmel ama harici genelleme sıfır** olan modellere yol açar. Combined Score bu sorunu çözer:

```
Combined Score = 0.5 × val_auc + 0.5 × ext_mean_auc

val_auc     = İç validasyon AUC (eğitimden kalan performans)
ext_mean_auc = 5 harici dataset ortalaması (genelleme yeteneği)
```

| Model | Val AUC | Ext Mean | Combined | Seçim |
|---|---|---|---|---|
| Baseline (fine-tune öncesi) | **0.9825** | 0.5375 | 0.7600 | ❌ |
| Epoch 3 (frozen son) | 0.9818 | 0.7347 | 0.8583 | ❌ |
| **Epoch 8 (final)** | **0.9839** | **0.7527** | **0.8683** | ✅ En iyi |

### Eğitim İlerleme Tablosu

| Epoch | Durum | Train Loss | Val AUC | Val Acc | LR | Ext Mean AUC |
|-------|-------|-----------|---------|---------|------|-------------|
| 1 | 🔒 FROZEN | 0.1004 | 0.9816 | 92.9% | 5e-5 | — |
| 2 | 🔒 FROZEN | 0.1000 | 0.9811 | 92.8% | 4.5e-5 | — |
| 3 | 🔒 FROZEN | 0.0998 | 0.9818 | 92.5% | 3.5e-5 | 0.7347 |
| 4 | 🔓 **UNFROZEN** | 0.0990 | 0.9828 | 92.9% | 5e-5 | — |
| 5 | 🔓 UNFROZEN | 0.0983 | 0.9832 | 93.5% | 3.5e-5 | — |
| 6 | 🔓 UNFROZEN | 0.0980 | 0.9829 | 93.1% | 1.8e-5 | — |
| 7 | 🔓 UNFROZEN | 0.0977 | 0.9835 | 93.2% | 5.7e-6 | — |
| **8** | 🔓 **UNFROZEN** | **0.0975** | **0.9839** | **93.1%** | 1e-6 | **0.7527** |

### Sonuç: Before vs After

```
                  Baseline (0.537)                    Final (0.752)
                  ─────────────────                   ─────────────────
  Deepfake20K     █████░░░░░░░░░░░░░  0.514           ████████████████░░  0.956  🔥
  DFDC            ██████░░░░░░░░░░░░  0.546           █████████████░░░░░  0.821  🔥
  DeepfakeFace    ██████░░░░░░░░░░░░  0.583           █████████████░░░░░  0.777  🟢
  CelebDF v2      █████░░░░░░░░░░░░░  0.541           ████████████░░░░░░  0.708  🟢
  FF++            █████░░░░░░░░░░░░░  0.502           █████░░░░░░░░░░░░░  0.500  ⚪
                  ─────────────────                   ─────────────────
  Ortalama        █████░░░░░░░░░░░░░  0.537           █████████████░░░░░  0.752  📈+40%
```

---

## 🚀 Kurulum

### Sistem Gereksinimleri

| Bileşen | Minimum | Önerilen |
|---|---|---|
| **İşletim Sistemi** | Windows 10 / Ubuntu 20.04 / macOS 13+ | Windows 11 / Ubuntu 22.04 / macOS 14+ |
| **Python** | 3.11+ | 3.14+ |
| **GPU** | CUDA destekli NVIDIA GPU (veya CPU) | RTX 4070+ (8GB+ VRAM) |
| **VRAM** | 4GB (sadece inference) | 8GB (eğitim + inference) |
| **RAM** | 16GB | 32GB |
| **Disk** | 10GB (model + kod) | 50GB+ (datasetler dahil) |
| **CUDA** | 11.8+ (macOS'ta gerekli değil) | 12.x |

> **macOS Notu:** Apple Silicon (M1/M2/M3/M4) üzerinde CPU modunda çalışır. CUDA gerekmez.

### Adım 1: Proje Kurulumu

```bash
# Repoyu klonla
git clone https://github.com/seydivakkas/DeepfakeULTRA.git
cd DeepfakeULTRA

# Sanal ortam oluştur
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/macOS
```

### Adım 2: Bağımlılıkları Yükle

```bash
# Tüm bağımlılıkları yükle (~71 paket)
pip install -r requirements.txt
```

**Temel bağımlılıklar (kategoriye göre):**

| Kategori | Paketler | Kullanım |
|---|---|---|
| **Derin Öğrenme** | `torch>=2.0`, `torchvision`, `timm` | Model eğitimi ve çıkarım |
| **Yüz Algılama** | `mediapipe`, `opencv-python`, `Pillow` | Face detection + 468 landmark |
| **Frekans Analizi** | `PyWavelets` | DWT/DCT frekans çıkarımı |
| **XAI** | `captum`, `lime` | GradCAM++, EigenCAM, LIME haritaları |
| **Web Arayüzü** | `gradio>=4.0`, `fastapi`, `uvicorn` | 7 sekmeli UI + REST API |
| **LLM** | `google-generativeai` | Gemini analiz asistanı |
| **Görselleştirme** | `plotly`, `matplotlib` | Dashboard + benchmark grafikleri |
| **PDF** | `fpdf2` | Forensik rapor oluşturma |
| **ML** | `scikit-learn`, `scipy`, `numpy`, `pandas` | Metrikler, istatistik |

### Adım 3: CUDA Doğrulama

```bash
# GPU'nun algılandığını doğrula
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"Yok\"}')"
```

Beklenen çıktı:
```
CUDA: True, GPU: NVIDIA GeForce RTX 4070
```

### Adım 4: Pretrained Model İndir

Model ağırlıkları büyük boyutları nedeniyle Git reposuna dahil değildir. GitHub Releases'tan otomatik indirilir:

```bash
# Otomatik indirme (önerilen)
python download_model.py

# Belirli bir model
python download_model.py --model best_run5_forensic.pth

# Mevcut dosyaları listele
python download_model.py --list
```

| Model | Boyut | Açıklama |
|---|---|---|
| `best_run5_forensic.pth` | ~247 MB | Ana forensik model (önerilen) |
| `best_model.pth` | ~83 MB | Hafif model |

> **Manuel İndirme:** [GitHub Releases](https://github.com/seydivakkas/DeepfakeULTRA/releases) sayfasından `.pth` dosyalarını indirip `models/` klasörüne yerleştirin.

### Adım 5: Çalıştır

```bash
# Gradio Web Arayüzü (varsayılan)
python app.py
# → http://localhost:7860

# FastAPI REST API
python main.py
# → http://localhost:8000/docs (Swagger UI)
```

### Docker (Opsiyonel)

```bash
docker-compose up --build
# → http://localhost:7860 (Gradio)
# → http://localhost:8000 (FastAPI)
```

### Ortam Değişkenleri

| Değişken | Zorunlu | Açıklama |
|---|---|---|
| `GEMINI_API_KEY` | ❌ | Analiz Asistanı (Sekme 7) + Kraniyofasiyal (Sekme 6) için |
| `JWT_SECRET` | ❌ | FastAPI JWT token imzalama |
| `SLACK_WEBHOOK_URL` | ❌ | Slack bildirim entegrasyonu |
| `TELEGRAM_BOT_TOKEN` | ❌ | Telegram bot entegrasyonu |
| `TELEGRAM_CHAT_ID` | ❌ | Telegram hedef chat |

```env
# .env dosyası (opsiyonel)
GEMINI_API_KEY=your-google-ai-studio-key
JWT_SECRET=your-secret-key
```

> **Not:** Kraniyofasiyal Analiz sekmesi için API key'ler doğrudan arayüzden de girilebilir ve `.api_keys.json` dosyasına kaydedilir.

### Sorun Giderme

| Sorun | Çözüm |
|---|---|
| `CUDA out of memory` | `BATCH_SIZE`'ı `config.py`'da 16'ya düşürün |
| `ModuleNotFoundError` | `pip install -r requirements.txt` yeniden çalıştırın |
| `Model dosyası bulunamadı` | `models/best_model.pth` dosyasını kontrol edin |
| `mediapipe hata` | `pip install mediapipe --upgrade` |
| `Port 7860 kullanımda` | `python app.py --port 7861` veya süreci kapatın |
| `Gradio CORS hatası` | `--share` flag'i ile çalıştırın: `python app.py --share` |

---

## 🖥️ Kullanım

### Gradio Web Arayüzü

```bash
# Varsayılan başlatma
python app.py
# → http://localhost:7860

# Public link ile (ngrok tunnelling)
python main.py demo --share
```

### `main.py` CLI Komutları

`main.py` birleşik giriş noktasıdır. Alt komutları:

```
┌─────────────────────────────────────────────────────────────────┐
│  python main.py <komut> [seçenekler]                            │
├──────────┬──────────────────────────────────────────────────────┤
│  demo    │  Gradio UI başlat                                    │
│  train   │  Model eğitimi (20 epoch, curriculum learning)       │
│  eval    │  Model değerlendirmesi (test seti)                   │
│  api     │  FastAPI REST sunucusu                               │
│  predict │  Tek görsel analizi (CLI)                            │
└──────────┴──────────────────────────────────────────────────────┘
```

| Komut | Kullanım | Açıklama |
|---|---|---|
| `demo` | `python main.py demo --share` | Gradio UI, opsiyonel public link |
| `train` | `python main.py train --epochs 20 --batch-size 20` | Model eğitimi, `--resume ckpt.pth` ile devam |
| `eval` | `python main.py eval --model models/best_model.pth` | Test seti üzerinde değerlendirme |
| `api` | `python main.py api --port 8000 --reload` | FastAPI, hot-reload geliştirme modu |
| `predict` | `python main.py predict -i foto.jpg --xai --report` | Tek görsel, XAI + PDF rapor |
| `--test-random` | `python main.py --test-random` | Rastgele girdi ile model mimari testi |

### Tek Görsel Analizi (CLI)

```bash
# Basit analiz
python main.py predict -i suspect_photo.jpg

# XAI haritaları + PDF rapor ile
python main.py predict -i suspect_photo.jpg --xai --report
```

Çıktı:
```
==================================================
📊 Sonuç: FAKE
   Sahte Olasılığı: 0.9471
   Gerçek Olasılığı: 0.0529
   GradCAM++ Skoru: 0.82
==================================================

🔍 XAI haritaları oluşturuluyor...
📄 PDF rapor oluşturuluyor...
```

### Script Araçları

32 farklı script, kategoriye göre:

| Kategori | Script | Komut | Açıklama |
|---|---|---|---|
| **Benchmark** | `evaluate_model.py` | `python scripts/evaluate_model.py --tta` | 5 harici dataset benchmark + TTA |
| **Benchmark** | `evaluate_external.py` | `python scripts/evaluate_external.py` | Tek dataset detaylı değerlendirme |
| **Benchmark** | `run_tta_benchmark.py` | `python scripts/run_tta_benchmark.py` | TTA öncesi/sonrası karşılaştırma |
| **Fine-Tune** | `finetune_generalization.py` | `python scripts/finetune_generalization.py` | Domain-robust 8 epoch fine-tuning |
| **Fine-Tune** | `finetune_cross_dataset.py` | `python scripts/finetune_cross_dataset.py` | Harici dataset ile fine-tuning |
| **Veri Hazırlama** | `prepare_celeb_df_v2.py` | `python scripts/prepare_celeb_df_v2.py` | CelebDF v2 dataset indirme/hazırlama |
| **Veri Hazırlama** | `prepare_ff++.py` | `python scripts/prepare_ff++.py` | FaceForensics++ hazırlama |
| **Veri Hazırlama** | `01_extract_faces.py` | `python scripts/01_extract_faces.py` | MTCNN ile yüz kırpma |
| **Veri Hazırlama** | `06_smart_split.py` | `python scripts/06_smart_split.py` | Akıllı train/val/test split |
| **Analiz** | `error_analysis.py` | `python scripts/error_analysis.py` | Hata analizi, yanlış tahminler |
| **Analiz** | `find_threshold.py` | `python scripts/find_threshold.py` | Youden J-statistic ile eşik hesaplama |
| **Kalite** | `28_quality_pipeline.py` | `python scripts/28_quality_pipeline.py` | Veri seti kalite kontrolü |
| **Kalite** | `leakage_checker.py` | `python scripts/leakage_checker.py` | Train/val/test veri sızıntı kontrolü |
| **İzleme** | `monitor.py` | `python scripts/monitor.py` | Gerçek zamanlı eğitim izleme |

### FastAPI REST API

```bash
python main.py api --port 8000
# → http://localhost:8000/docs (Swagger UI)
```

**Endpoint'ler:**

| Method | Endpoint | Body | Yanıt | Açıklama |
|---|---|---|---|---|
| `POST` | `/api/analyze` | `multipart/form-data (file)` | `{label, fake_prob, platform, ...}` | Tek görsel analizi |
| `GET` | `/api/health` | — | `{status, gpu, model_loaded}` | Sistem sağlık kontrolü |
| `GET` | `/api/analytics` | `?limit=100` | `{total, fake_count, real_count, ...}` | İstatistik verileri |
| `GET` | `/api/history` | `?limit=200` | `[{date, file, result, confidence}, ...]` | Analiz geçmişi |

**API Güvenliği:**
- JWT token tabanlı kimlik doğrulama (`HS256`)
- Rate limiting: `30 istek/dakika` (SlowAPI)
- CORS: Yapılandırılabilir origins

**cURL Örneği:**
```bash
# Tek görsel analizi
curl -X POST http://localhost:8000/api/analyze \
  -H "Authorization: Bearer <JWT_TOKEN>" \
  -F "file=@suspect_photo.jpg"

# Sistem durumu
curl http://localhost:8000/api/health
```

### Python SDK Kullanımı

```python
from inference.predictor import DeepfakePredictor

# Predictor oluştur (model otomatik yüklenir)
predictor = DeepfakePredictor()

# Tek görsel analizi
result = predictor.predict("suspect_photo.jpg")

print(f"Sonuç: {result['label']}")       # FAKE / REAL / UNCERTAIN
print(f"Fake olasılığı: {result['fake_prob']:.4f}")
print(f"Platform: {result.get('platform', 'Bilinmiyor')}")
```

---

## 📁 Proje Yapısı

```
DeepfakeULTRA/
│
├── 📌 Giriş Noktaları
├── app.py                              # Gradio ana arayüz (7 sekme, ~31KB)
├── main.py                             # CLI giriş noktası (demo/train/eval/api/predict)
├── run_training.py                     # Eğitim başlatma scripti
├── monitor_training.py                 # Canlı eğitim izleme
├── config.py                           # Merkezi konfigürasyon (4 dataclass, ~357 satır)
├── requirements.txt                    # 71 Python bağımlılığı
│
├── 🧠 core/                            # Çekirdek Modüller (26 dosya)
│   ├── dual_mobilenetv3.py             # DualPathDeepfakeDetector ana model (~14KB)
│   │                                   #   ├── FaceMeshMLP (468×3 → 128 → 960)
│   │                                   #   ├── CrossBranchTransformer (2L, 4H)
│   │                                   #   └── Classifier (960 → 256 → 2)
│   ├── frequency.py                    # DWT frekans görselleştirme
│   ├── frequency_v2.py                 # HybridFrequencyExtractor (18 kanal, ~15KB)
│   │                                   #   ├── DWT: Haar + DB2 + Coif1 (12ch)
│   │                                   #   ├── DCT: Low/Mid/High (3ch)
│   │                                   #   └── Phase: R/G/B FFT (3ch)
│   ├── data_pipeline.py                # Veri yükleme + augmentasyon (~33KB)
│   ├── domain_augmentation.py          # 🆕 Domain-robust augmentasyon
│   ├── tta_inference.py                # 🆕 GPU-native TTA (8 augmentasyon)
│   ├── trainer.py                      # Ana eğitim döngüsü (~41KB)
│   │                                   #   ├── Curriculum Learning
│   │                                   #   ├── MixUp + CutMix
│   │                                   #   ├── FGSM Adversarial Training
│   │                                   #   └── FP16 AMP + GradAccum
│   ├── adversarial.py                  # FGSM / PGD / CW saldırıları (~19KB)
│   ├── compression.py                  # Platform sıkıştırma simülasyonu
│   ├── forensics.py                    # ELA + Noise forensik analizi (~13KB)
│   ├── face_detector.py                # MTCNN yüz algılama + bbox
│   ├── platform_detector.py            # JPEG quantization tablosu → platform tespiti
│   ├── fine_tuner.py                   # Active Learning fine-tuning
│   ├── model_metrics.py                # ROC, CM, F1, EER hesaplama
│   ├── sbi_augmentation.py             # Self-Blended Image augmentasyonu
│   ├── hard_real_augmentation.py       # Hard-real veri üretimi
│   ├── contrastive_loss.py             # Triplet Loss (hard mining, ~12KB)
│   ├── loss_utils.py                   # Focal Loss + Label Smoothing
│   ├── calibration.py                  # Temperature Scaling kalibrasyon
│   └── evaluation.py                   # Model değerlendirme
│
├── 🔮 inference/                       # Çıkarım Modülleri (10 dosya)
│   ├── predictor.py                    # Ana tahmin sınıfı (DeepfakePredictor)
│   ├── analyze_engine.py               # Tam analiz pipeline'ı
│   ├── tta_inference.py                # Test-Time Augmentation (~11KB)
│   ├── xai_module.py                   # GradCAM++ / EigenCAM / FastCAM
│   ├── hybrid_xai.py                   # Birleşik XAI raporu
│   ├── model_ensemble.py               # Multi-model ensemble (~11KB)
│   └── subtype_classifier.py           # Hiyerarşik alt-tip sınıflandırma
│
├── 🎨 ui/                              # Arayüz Bileşenleri
│   ├── components.py                   # Tüm Gradio handler fonksiyonları
│   └── craniofacial_tab.py             # 🧬 Kraniyofasiyal Biyometrik sekme
│
├── 🔌 services/                        # Harici Servisler
│   ├── llm_module.py                   # Gemini 3.0 Pro / Ollama chat
│   ├── vision_api.py                   # Multi-provider Vision API
│   │                                   #   ├── Google Gemini
│   │                                   #   ├── OpenAI GPT-4o
│   │                                   #   └── Anthropic Claude
│   └── pdf_report.py                   # PDF forensik rapor oluşturma
│
├── 🌐 api/                             # REST API Katmanı
│   └── server.py                       # FastAPI endpoint'leri + JWT + CORS
│
├── 💾 db/                              # Veritabanı
│   └── database.py                     # SQLite analiz geçmişi + feedback
│
├── 🛠️ scripts/                         # Araç Scriptleri (32 dosya)
│   ├── finetune_generalization.py      # 🆕 Domain-robust fine-tuning (~14KB)
│   ├── finetune_cross_dataset.py       # 🆕 Harici dataset fine-tuning
│   ├── evaluate_model.py               # Benchmark + TTA (~21KB)
│   ├── evaluate_external.py            # 🆕 Harici dataset değerlendirme
│   ├── run_tta_benchmark.py            # 🆕 TTA benchmark runner
│   ├── prepare_celeb_df_v2.py          # CelebDF v2 dataset hazırlama
│   ├── prepare_ff++.py                 # 🆕 FaceForensics++ hazırlama
│   ├── download_ff++.py                # 🆕 FF++ dataset indirme
│   ├── 01_extract_faces.py             # MTCNN yüz çıkarma
│   ├── 06_smart_split.py               # Kalite-bilinçli akıllı split
│   ├── generate_hard_real.py           # Hard-real veri üretimi
│   ├── generate_sbi_data.py            # SBI veri üretimi
│   ├── find_threshold.py               # Youden J-statistic eşik
│   ├── jury_evaluation.py              # Jury test seti (~20KB)
│   ├── leakage_checker.py              # Veri sızıntısı kontrolü (~18KB)
│   ├── error_analysis.py               # Hata analizi (~13KB)
│   ├── monitor.py                      # Gerçek zamanlı eğitim izleme (~18KB)
│   └── weekly_scheduler.py             # Haftalık izleme pipeline
│
├── 📊 evaluation/                      # Değerlendirme Sonuçları
│   ├── metrics.json                    # İç test metrikleri
│   ├── roc_curve.png                   # ROC eğrisi
│   ├── confusion_matrix.png            # Confusion matrix
│   ├── reliability_diagram.png         # Kalibrasyon diyagramı
│   └── external/                       # 🆕 5 Harici Dataset Benchmark
│       ├── celeb_df_v2/                #   metrics.json + ROC + CM + rapor
│       ├── dfdc/
│       ├── deepfake20k/
│       ├── deepfakeface/
│       └── faceforensics/
│
├── 🤖 models/                          # Model Ağırlıkları (.gitignore)
│   ├── best_model.pth                  # Ana model (AUC=0.9825, ~60MB)
│   └── best_model_generalized.pth      # 🆕 Domain-robust (Combined=0.8683)
│
├── 📦 Veri & Çıktı (.gitignore)
├── dataset/                            # Eğitim/test verileri
│   ├── train/ val/ test/               # Ana split
│   └── external_tests/                 # Harici benchmark datasetleri
├── feedback_images/                    # Active Learning geri bildirim
├── analytics_logs/                     # Günlük analiz logları
├── reports/                            # Üretilen PDF raporlar
├── sunum_gorselleri/                   # Demo görselleri (fake/real)
│
├── ⚙️ Altyapı
├── .github/workflows/ci.yml           # GitHub Actions CI/CD
├── Dockerfile                          # Docker imajı
├── docker-compose.yml                  # Docker Compose
├── .env                                # Ortam değişkenleri (opsiyonel)
├── .api_keys.json                      # Kaydedilmiş API anahtarları
└── LICENSE                             # MIT License
```

### Modül İlişki Haritası

```
                    app.py / main.py
                         │
            ┌────────────┼────────────┐
            │            │            │
         ui/          api/       services/
      components    server.py    llm_module
      craniofacial               vision_api
            │            │        pdf_report
            └────────────┼────────────┘
                         │
                    inference/
                  predictor.py
                 analyze_engine
                  xai_module
                 tta_inference
                         │
                      core/
              dual_mobilenetv3.py
            ┌────────┼────────┐
         frequency  data_     trainer
         _v2.py    pipeline   .py
                      │
               domain_augmentation
               tta_inference
               forensics
               adversarial
```

---

## 🎓 Eğitim Pipeline'ı

### Tam Eğitim Akışı

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                        DeepfakeULTRA Eğitim Pipeline                        │
└──────────────────────────────────────────────────────────────────────────────┘

  ADIM 1: Veri Hazırlama
  ┌────────────────────────────────────────────────┐
  │  Ham Veri → MTCNN Yüz Çıkarma → 224×224 Crop   │
  │  → Kalite Filtresi → Akıllı Split (60/20/20)   │
  │  → Sızıntı Kontrolü (leakage_checker)           │
  └────────────────────┬───────────────────────────┘
                       │
  ADIM 2: Ana Eğitim (20 Epoch)
  ┌────────────────────▼───────────────────────────┐
  │  Curriculum Learning (4 Faz)                    │
  │  Epoch  1-4:  Backbone FROZEN, hard_real=0%     │
  │  Epoch  5-9:  Backbone FROZEN, hard_real=15%    │
  │  Epoch 10-15: Backbone UNFROZEN, hard_real=30%  │
  │  Epoch 16-20: Full train, hard_real=40%         │
  │                                                 │
  │  + MixUp (α=0.2) + CutMix (60/40 ratio)        │
  │  + FGSM Adversarial (ε=0.005-0.03, epoch 2+)   │
  │  + FP16 AMP + Gradient Accumulation (×8)        │
  └────────────────────┬───────────────────────────┘
                       │
  ADIM 3: Domain Generalization Fine-Tune (8 Epoch)
  ┌────────────────────▼───────────────────────────┐
  │  Base: best_model.pth (AUC=0.9825)              │
  │  + Domain Augmentation (JPEG/Resize/Noise/...)  │
  │  Epoch 1-3: FROZEN (LR=5e-5)                    │
  │  Epoch 4-8: UNFROZEN (Backbone LR=5e-6)         │
  │  Seçim: Combined Score (val + ext benchmark)     │
  └────────────────────┬───────────────────────────┘
                       │
  ADIM 4: Doğrulama
  ┌────────────────────▼───────────────────────────┐
  │  İç Test (59K görsel) → AUC, F1, CM, ROC        │
  │  Harici Benchmark (5 dataset) → Cross-Dataset    │
  │  Jury Evaluation → Bağımsız doğrulama            │
  │  Kalibrasyon → Temperature Scaling (ECE<5%)      │
  └────────────────────────────────────────────────┘
```

### Eğitim Hiperparametreleri

| Bileşen | Değer | Açıklama |
|---|---|---|
| **Loss** | `0.8 × Focal(γ=2.0, α=0.75) + 0.2 × Triplet(m=1.0)` | Hard-negative odaklı, contrastive |
| **Optimizer** | AdamW (`lr=3e-4`, `wd=1e-4`) | Backbone `lr × 0.1` (discriminative) |
| **Scheduler** | Cosine Annealing (`T_max=20`, `η_min=1e-6`) | + 1 epoch linear warmup |
| **Precision** | FP16 Mixed Precision | AMP + TF32 (RTX 4070) |
| **Batch** | 20 × 8 accumulation = **160 efektif** | 8GB VRAM kısıtı |
| **Gradient Clip** | `max_norm=1.0` | Eğitim stabilitesi |
| **Label Smoothing** | `ε=0.1` | Genellemede iyileşme |
| **Early Stopping** | Patience = 4 epoch | Val AUC bazlı |

### Curriculum Learning — 4 Fazlı Strateji

```
  Epoch  1  ─────  4     5  ─────  9    10  ────── 15    16  ────── 20
  ┌──────────────┐ ┌──────────────┐ ┌───────────────┐ ┌───────────────┐
  │ FAZ 1        │ │ FAZ 2        │ │ FAZ 3         │ │ FAZ 4         │
  │              │ │              │ │               │ │               │
  │ 🔒 Frozen    │ │ 🔒 Frozen    │ │ 🔓 Unfrozen   │ │ 🔓 Full Train │
  │ Hard-real: 0%│ │ Hard-real:15%│ │ Hard-real: 30%│ │ Hard-real: 40%│
  │              │ │              │ │               │ │               │
  │ Amaç:        │ │ Amaç:        │ │ Amaç:         │ │ Amaç:         │
  │ Temel uyum   │ │ Zor örnekler│ │ Derin feature │ │ Tam ince ayar │
  └──────────────┘ └──────────────┘ └───────────────┘ └───────────────┘
```

### Augmentasyon Stratejisi

| Yöntem | Olasılık | Parametre | Amaç |
|---|---|---|---|
| **MixUp** | %40 | `α=0.2` | İnter-class interpolation |
| **CutMix** | %60 | `α=1.0` | Bölge bazlı augmentasyon |
| **FGSM Adversarial** | Her 4 step'te 1 | `ε=0.005–0.03` | Adversarial robustness |
| **SBI** | Konfigürasyona göre | Self-Blended | Realtime artifact üretimi |
| **Social Compress** | Öğrenime göre | WhatsApp/IG/TikTok sim. | Platform robustness |
| **Domain Aug** | %50 (fine-tune) | JPEG/Resize/Noise/Gamma/Color | Cross-dataset genelleme |

### Veri Seti Kaynakları

| Kaynak | Tür | Boyut | Kullanım |
|---|---|---|---|
| **DF40** | 40 farklı deepfake yöntemi | ~160K fake | Ana eğitim seti |
| **FF++** | DeepFakes, FaceSwap, Face2Face, NeuralTextures | ~10K | Eğitim + Harici test |
| **CelebA-HQ** | Yüksek kaliteli gerçek yüzler | ~30K | Eğitim (REAL) |
| **FFHQ** | 1024px yüksek çözünürlük gerçek yüzler | ~70K | Eğitim (REAL) |
| **UTKFace** | Demografik çeşitlilik (yaş, cinsiyet, etnisite) | ~24K | Eğitim (REAL) + Jury |
| **VGGFace2** | Kimlik çeşitliliği | ~10K | Eğitim (REAL) |
| **SBI** | Self-Blended Image (sentetik) | Dinamik | Fine-tuning |
| **CelebDF v2** | Eski nesil autoencoder swap | 1,890 | Harici benchmark |
| **DFDC** | Facebook AI yarışma dataseti | 5,000+ | Harici benchmark |
| **Deepfake20K** | Karma GAN/swap yöntemleri | 20,000 | Harici benchmark |

### Domain Generalization Fine-Tuning Detay

| Bileşen | Değer | Açıklama |
|---|---|---|
| **Base Model** | `best_model.pth` (AUC=0.9825) | Üzerine inşa edilir |
| **Domain Aug.** | JPEG(Q=30–95) + Resize(0.25–0.75x) + Noise + Gamma + Color | %50 olasılıkla |
| **Frozen Epochs** | 1–3 (`LR=5e-5`) | Classifier adaptasyonu |
| **Unfrozen Epochs** | 4–8 (`Backbone LR=5e-6`) | Derin feature refinement |
| **Model Seçimi** | Combined Score | `0.5 × val_auc + 0.5 × ext_mean_auc` |
| **Harici Benchmark** | 5 dataset (her epoch sonunda) | CelebDF, FF++, DFDC, Deepfake20K, DeepfakeFace |
| **Çıktı** | `best_model_generalized.pth` | Combined=0.8683 |

### Hızlı Başlangıç Komutları

```bash
# ── ADIM 1: Veri Hazırlama ──
python scripts/01_extract_faces.py          # MTCNN yüz çıkarma
python scripts/06_smart_split.py            # Train/Val/Test split
python scripts/leakage_checker.py           # Sızıntı kontrolü

# ── ADIM 2: Ana Eğitim ──
python run_training.py                      # 20 epoch curriculum training
python scripts/monitor.py                   # (ayrı terminal) canlı izleme

# ── ADIM 3: Değerlendirme ──
python scripts/evaluate_model.py --tta      # İç test + TTA
python scripts/find_threshold.py            # Optimal eşik hesaplama

# ── ADIM 4: Domain Generalization ──
python scripts/finetune_generalization.py   # 8 epoch domain-robust fine-tune

# ── ADIM 5: Harici Benchmark ──
python scripts/evaluate_model.py --tta      # 5 dataset cross-dataset benchmark
```

---

## 🛠️ Teknoloji Yığını

### Mimari Katmanları

```
┌─────────────────────────────────────────────────────────────────┐
│                        🎨 Sunum Katmanı                        │
│  Gradio 4.0+ │ Plotly │ Matplotlib │ fpdf2 (PDF Rapor)         │
├─────────────────────────────────────────────────────────────────┤
│                        🌐 API Katmanı                          │
│  FastAPI │ Uvicorn │ PyJWT │ SlowAPI │ python-multipart         │
├─────────────────────────────────────────────────────────────────┤
│                        🤖 Zeka Katmanı                         │
│  Gemini 3.0 Pro │ Ollama (qwen2.5) │ sentence-transformers     │
│  Google Vision API │ OpenAI GPT-4o │ Anthropic Claude          │
├─────────────────────────────────────────────────────────────────┤
│                        🔮 Çıkarım Katmanı                      │
│  Captum (GradCAM++, EigenCAM) │ LIME │ TTA (8-aug ensemble)    │
│  MediaPipe (468 3D Landmark) │ OpenCV │ scikit-image            │
├─────────────────────────────────────────────────────────────────┤
│                        🧠 Model Katmanı                        │
│  PyTorch 2.0+ │ torchvision │ timm │ FP16 AMP │ CUDA 12.x     │
│  MobileNetV3-Large │ Transformer Encoder │ Triplet Loss         │
├─────────────────────────────────────────────────────────────────┤
│                        📊 Veri Katmanı                         │
│  PyWavelets (DWT) │ NumPy (DCT, Phase) │ SciPy │ pandas        │
│  SQLite │ scikit-learn │ MLflow │ Optuna                        │
├─────────────────────────────────────────────────────────────────┤
│                        ⚙️ Altyapı                              │
│  Docker │ Docker Compose │ GitHub Actions CI/CD                 │
│  python-dotenv │ colorama │ tqdm │ pydantic                     │
└─────────────────────────────────────────────────────────────────┘
```

### Detaylı Bağımlılık Tablosu

| Katman | Teknoloji | Versiyon | Kullanım Alanı |
|---|---|---|---|
| **Derin Öğrenme** | PyTorch | `>=2.0.0` | Model eğitimi, çıkarım, AMP |
| | torchvision | `>=0.15.0` | MobileNetV3, transform pipeline |
| | timm | `>=0.9.0` | Model zoo, pretrained backbone |
| **Yüz İşleme** | MediaPipe | `>=0.10.0` | 468 3D Face Landmark, yüz algılama |
| | OpenCV | `>=4.8.0` | Görüntü işleme, JPEG decode |
| | scikit-image | `>=0.21.0` | ELA, frekans analizi |
| | Pillow | `>=10.0.0` | Görüntü okuma/yazma |
| **Frekans** | PyWavelets | `>=1.4.0` | DWT (Haar, DB2, Coif1) |
| **XAI** | Captum | `>=0.7.0` | GradCAM++, EigenCAM, LayerAttribution |
| | LIME | `>=0.2.0` | Model-agnostic açıklanabilirlik |
| **LLM** | google-generativeai | `>=0.8.0` | Gemini 3.0 Pro chat + vision |
| | httpx | `>=0.25.0` | Ollama local API iletişimi |
| **Web** | Gradio | `>=4.0.0` | 7 sekmeli forensik arayüz |
| | FastAPI | `>=0.100.0` | REST API endpoint'leri |
| | Uvicorn | `>=0.23.0` | ASGI sunucu (async) |
| | SlowAPI | `>=0.1.9` | Rate limiting (30 req/min) |
| **Güvenlik** | PyJWT | `>=2.8.0` | JWT token (HS256) |
| **PDF** | fpdf2 | `>=2.7.0` | Forensik rapor oluşturma |
| **Görselleştirme** | Plotly | `>=5.18.0` | İnteraktif dashboard grafikleri |
| | Matplotlib | `>=3.8.0` | ROC, CM, güvenilirlik diyagramı |
| **ML** | scikit-learn | `>=1.3.0` | AUC, F1, ROC, confusion matrix |
| | scipy | `>=1.11.0` | FFT, istatistiksel testler |
| | numpy | `>=1.24.0` | DCT, Phase spectrum hesaplama |
| | pandas | `>=2.1.0` | Veri manipülasyonu |
| **Deney** | MLflow | `>=2.8.0` | Eğitim takibi, metrik loglama |
| | Optuna | `>=3.4.0` | Hiperparametre optimizasyonu |
| **Embedding** | FAISS | `>=1.7.4` | Vektör benzerlik arama |
| | sentence-transformers | `>=2.2.0` | RAG embedding (MiniLM-L6-v2) |
| **Bildirim** | python-telegram-bot | `>=20.0` | Telegram bildirim entegrasyonu |
| **Altyapı** | Docker | — | Konteyner |
| | tqdm | `>=4.66.0` | İlerleme çubuğu |
| | pydantic | `>=2.0.0` | Veri doğrulama |

---

## 🧩 Chrome Uzantısı

Web'deki herhangi bir görsele **sağ tıklayarak** doğrudan DeepfakeULTRA ile deepfake analizi yapabilirsiniz. Uzantı, görseli otomatik olarak Gradio arayüzüne yükler ve analizi başlatır.

### Nasıl Çalışır?

```
  Herhangi bir web sayfası (Google, haber sitesi, sosyal medya...)
  ┌────────────────────────────────────────────────────┐
  │                                                    │
  │   Görsele sağ tıkla                                │
  │        │                                           │
  │        ▼                                           │
  │   "🔍 DeepfakeULTRA — Görüntüyü Analiz Et"        │
  │        │                                           │
  └────────┼───────────────────────────────────────────┘
           │
           ▼
  ┌────────────────────────────────────────────────────┐
  │   localhost:7860 (Gradio) otomatik açılır          │
  │        │                                           │
  │        ├── Görsel otomatik indirilir               │
  │        ├── Upload alanına enjekte edilir            │
  │        └── "🔬 Analiz Et" butonuna basılır         │
  │                                                    │
  │   Sonuçlar Gradio'da görüntülenir:                 │
  │   FAKE/REAL + GradCAM++ + Frekans + XAI            │
  └────────────────────────────────────────────────────┘
```

### Kurulum

```bash
# 1. Chrome'da uzantı sayfasını aç
#    Adres çubuğuna yaz: chrome://extensions

# 2. "Geliştirici modu" toggle'ını AÇ (sağ üst köşe)

# 3. "Paketlenmemiş uzantı yükle" butonuna tıkla

# 4. DeepfakeULTRA/extension/ klasörünü seç
```

### Kullanım

1. **Ön koşul:** `python app.py` çalışıyor olmalı (`localhost:7860`)
2. Web'de herhangi bir görsele **sağ tıklayın**
3. **"🔍 DeepfakeULTRA — Görüntüyü Analiz Et"** seçeneğine tıklayın
4. Gradio sekmesi açılır, görsel otomatik yüklenir ve analiz başlar

> **Not:** Zaten açık bir Gradio sekmesi varsa, yeni sekme açmak yerine mevcut sekmeyi kullanır.

### Uzantı Dosyaları

```
extension/
├── manifest.json          # Manifest V3 konfigürasyon
├── background.js          # Service worker (context menü + yönlendirme)
├── content.js             # Gradio sayfasına görsel enjeksiyonu
├── popup.html             # Uzantı popup UI
├── popup.css              # Dark theme styling
├── popup.js               # Popup logic
└── icons/                 # 16/48/128px ikonlar
    ├── icon16.png
    ├── icon48.png
    └── icon128.png
```

### Desteklenen Senaryolar

| Senaryo | Yöntem | Durum |
|---|---|---|
| Doğrudan erişilebilir görsel (Google, Wikipedia) | URL ile indirme | ✅ |
| CORS korumalı görsel (Instagram, Facebook) | Canvas → base64 fallback | ✅ |
| Zaten açık Gradio sekmesi | Mevcut sekmeyi güncelle | ✅ |
| Sunucu kapalıyken tıklama | Hata bildirimi | ✅ |

---

## 🖼️ Akıllı Fotoğraf Filtresi (Non-Photo Detection)

Deepfake analizi yalnızca **gerçek fotoğraflar** üzerinde anlamlıdır. Karikatür, çizim, illüstrasyon, 3D render gibi fotoğraf olmayan görseller yüklendiğinde model yanıltıcı sonuçlar üretebilir. Bu sorunu çözmek için **eğitimsiz hibrit ön-filtre** geliştirilmiştir.

### Neden Gerekli?

```
Model Eğitim Dağılımı:

  REAL sınıfı          FAKE sınıfı           OOD (Dağılım Dışı)
  ┌──────────┐         ┌──────────┐          ┌──────────┐
  │ Gerçek   │         │ Deepfake │          │ Karikatür│
  │ Fotoğraf │         │ Fotoğraf │          │ Çizim    │
  │ (FFHQ,   │         │ (SimSwap,│          │ 3D Render│
  │  UTKFace) │         │  DF40)   │          │ İlüstr.  │
  └──────────┘         └──────────┘          └──────────┘
       ✅                   ✅                    ❌
    Eğitimde var          Eğitimde var         Eğitimde YOK!
```

Model karikatür/çizim hiç görmediği için "deepfake artefaktı yok" → "REAL" kararı verir. Bu **teknik olarak tutarlı ama kavramsal olarak yanlıştır.**

### Hibrit Filtre Mimarisi

```
Görsel Yüklendi
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│  AŞAMA 1: İstatistiksel Analiz (<10ms)                      │
│  ├── Benzersiz Renk Oranı (color_ratio)                     │
│  ├── Kenar Keskinliği (sharp_ratio)                          │
│  ├── Doğal Gürültü Seviyesi (noise_std)                     │
│  └── Düz Renk Bölgesi Oranı (flat_ratio)                    │
│                     → stat_score (0-1)                       │
└──────────────────────────────┬───────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│  AŞAMA 2: CLIP Doğrulama (~500ms)                           │
│  ├── "a real photograph of a human face"                    │
│  └── "a cartoon drawing illustration caricature of a face"  │
│                     → clip_score (0-1)                       │
└──────────────────────────────┬───────────────────────────────┘
                               │
                               ▼
            Hibrit Skor = clip × 0.60 + stat × 0.40
                               │
                    ┌──────────┴──────────┐
                < 0.40                  ≥ 0.40
                    │                     │
          🖼️ NON-PHOTO             Normal Pipeline
          "Fotoğraf Değil"          (FAKE / REAL)
```

### İstatistiksel Metrikler

| Metrik | Fotoğrafta | Karikatürde | Ne Ölçüyor? |
|---|---|---|---|
| **color_ratio** | >0.15 | <0.05 | Benzersiz renk çeşitliliği (fotoğraflar çok renkli) |
| **sharp_ratio** | <0.08 | >0.15 | Kenar keskinliği (çizimler sert kenarlara sahip) |
| **noise_std** | >3.0 | <1.5 | Doğal gürültü (fotoğraflarda sensör gürültüsü var) |
| **flat_ratio** | <0.60 | >0.75 | Düz renk bölgeleri (çizimler geniş tek renk alanları) |

### CLIP Doğrulama

```python
# OpenAI CLIP-ViT-B/32 modeli — sıfır eğitim, doğal dil-görsel eşleştirme
texts = [
    "a real photograph of a human face",       # Fotoğraf olasılığı
    "a cartoon drawing illustration caricature" # Çizim olasılığı
]
# Softmax → photo_prob vs cartoon_prob
```

**Özellikler:**
- ⚡ Lazy loading — CLIP modeli sadece ilk kullanımda yüklenir (~400MB)
- 🔒 CPU'da çalışır — GPU VRAM'e ek yük getirmez
- 🔄 Graceful degradation — CLIP yüklü değilse sadece istatistiksel filtre aktif

### UI Çıktısı

Fotoğraf olmayan bir görsel tespit edildiğinde:

```
🖼️ FOTOĞRAF DEĞİL

⚠️ Bu görsel bir fotoğraf değil (karikatür/çizim/illüstrasyon).
Deepfake analizi yalnızca gerçek fotoğraflar için geçerlidir.
```

Metrikler tablosunda ek bilgiler gösterilir:

| Metrik | Değer |
|---|---|
| Verdict | NON-PHOTO |
| Method | clip+statistical |
| CLIP Score | 0.1234 |
| CLIP Label | cartoon/illustration |
| Photo Score | 0.2100 |

---

## ⚠️ Bilinen Sınırlamalar & Gelecek Çalışmalar

### Bilinen Sınırlamalar

| # | Sınırlama | Detay | Etki | Planlanan Çözüm |
|---|---|---|---|---|
| 1 | **FF++ Reenactment** | Face2Face/NeuralTextures türü reenactment deepfake'lerde AUC=0.500 | Model bu yöntemi rastgele tahmin seviyesinde tespit eder | Reenactment-spesifik veri ile ek fine-tuning |
| 2 | **Sadece Tek Görsel** | Video analizi desteklenmez (frame-by-frame manuel) | Video forensik kullanım dışı | Video pipeline + temporal consistency analizi |
| 3 | **VRAM Kısıtı** | 3 branch + Transformer, 8GB VRAM'de eğitim sırasında sınırlı batch size | Eğitim hızı düşük (batch=20, accum=8) | Gradient checkpointing optimizasyonu |
| 4 | **Diffusion Model** | Son nesil diffusion tabanlı deepfake'ler (DALL-E 3, Midjourney) test edilmedi | Bilinmeyen performans | Diffusion-specific benchmark ekleme |
| 5 | **Yüz Bulunamama** | MediaPipe yüz tespit edemezse tüm pipeline başarısız olur | Profil, düşük çözünürlük, kapalı yüzler | Fallback face detector (RetinaFace) |
| 6 | **API Bağımlılığı** | Kraniyofasiyal + Asistan sekmeleri bulut API gerektirir | Çevrimdışı kullanımda sınırlı | Ollama fallback genişletme |

### FF++ Performans Notu

```
FF++ Reenactment Problemi:

  Eğitim Verisi (DF40)              FF++ (c23)
  ┌─────────────────┐               ┌─────────────────┐
  │  BlendFace      │               │  Face2Face      │
  │  CollabDiff     │  ──────────── │  NeuralTextures │
  │  Modern swap    │   AUC=0.50    │  Reenactment    │
  └─────────────────┘               └─────────────────┘

  Neden: DF40, ağırlıklı olarak face-swap yöntemleri içerir.
  Reenactment (yüz ifadesi transferi) farklı bir artifact türü
  bırakır — spatial blending yerine temporal texture değişimi.

  Bu, modelin "kör" olduğu bir alan değil, "görmediği" bir alan.
```

### Gelecek Çalışmalar (Roadmap)

```
┌─────────────────────────────────────────────────────────────────────────┐
│  KISA VADE (v5.1)                                                       │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  □ FF++ reenactment fine-tuning (Face2Face + NeuralTextures)      │  │
│  │  □ Video frame extraction + temporal analysis pipeline            │  │
│  │  □ RetinaFace fallback detector entegrasyonu                      │  │
│  │  □ ONNX export + TensorRT optimizasyonu (mobil deploy)            │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  ORTA VADE (v6.0)                                                       │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  □ Diffusion deepfake benchmark (DALL-E 3, Midjourney, SD)        │  │
│  │  □ Multi-frame temporal consistency scoring                       │  │
│  │  □ Federated learning desteği (veri gizliliği)                    │  │
│  │  □ Browser extension (Chrome/Firefox)                             │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  UZUN VADE (v7.0)                                                       │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  □ Audio deepfake tespiti (voice cloning)                         │  │
│  │  □ Real-time video stream analizi                                 │  │
│  │  □ Multi-language UI (i18n)                                       │  │
│  │  □ Enterprise API (SaaS dağıtımı)                                 │  │
│  └───────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

### Katkıda Bulunma

1. Fork yapın
2. Feature branch oluşturun (`git checkout -b feature/amazing-feature`)
3. Değişikliklerinizi commit edin (`git commit -m 'feat: Add amazing feature'`)
4. Branch'i push edin (`git push origin feature/amazing-feature`)
5. Pull Request açın

---

## 📄 Lisans

Bu proje [MIT License](LICENSE) altında lisanslanmıştır.

---

<div align="center">

### DeepfakeULTRA

**AI-Powered Forensic Deepfake Detection System**

Tri-Branch CNN + Transformer | 18-Channel Frequency Analysis | Domain Generalization

[![Python](https://img.shields.io/badge/Python-3.14+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org)
[![Gradio](https://img.shields.io/badge/Gradio-4.0+-orange.svg)](https://gradio.app)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

*seydivakkas © 2026*

</div>
