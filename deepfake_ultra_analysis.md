# DeepfakeULTRA — Awesome-Skills Kapsamlı Proje Analizi

> **Uygulanan Skills:** `pytorch-expert` · `debug-diagnose` · `architecture-review` · `pandas-expert`
> **Tarih:** 2026-05-28 · **Analiz Edilen Commit:** mevcut çalışma dizini

---

## 📊 VERİ SETİ DURUM TABLOSU (Gerçek Sayımlar)

| Dizin | Train Real | Train Fake | Val Real | Val Fake | Test Real | Test Fake | Denge |
|-------|-----------|-----------|---------|---------|---------|---------|-------|
| `faces_split` | 198,138 | 197,958 | 42,276 | 42,380 | 42,356 | 42,432 | ✅ ~50/50 |
| `faces_split_v2` | 198,864 | 198,864 | 42,612 | 42,612 | 42,616 | 42,616 | ✅ Tam 50/50 |

### Ham faces/ Kaynakları

| Kaynak | Görsel Sayısı | Tür | Durum |
|--------|--------------|-----|-------|
| `df40` | 827,703 | FAKE | ✅ Ana FAKE kaynağı |
| `sidset` | 210,001 | REAL+FAKE | ⚠️ 57,656 yüzsüz silindi |
| `vggface2` | 50,000 | REAL | ✅ |
| `genimage` | 49,581 | FAKE(diffusion) | ⚠️ Split'lere dahil edilmemiş! |
| `ffpp` | 54,729 | REAL+FAKE | ✅ |
| `celeba_hq` | 30,000 | REAL | ✅ |
| `ffhq_256` | 32,424 | REAL | ✅ |
| `utkface` | 7,725 | REAL | ✅ |
| `hard_real` | 8,001 | REAL | ⚠️ Split'lere dahil edilmemiş! |
| `custom_team` | 10,790 | ? | ⚠️ Kullanım belirsiz |

**Toplam:** 1,281,954 ham görsel → faces_split_v2'de ~568,184 kullanılıyor (%44.3)

---

## 🔴 KRİTİK SORUNLAR (debug-diagnose perspektifi)

### Sorun 1 — faces_split vs faces_split_v2: İki Split Çakışması

```
Dosyalar: core/data_pipeline.py (L578-581), scripts/06_smart_split.py, scripts/dataset_restructure.py

Friction: Pipeline önce faces_split'i dener, bulamazsa faces_split_v2'ye düşer.
İki split arasında ~726 fark var (faces_split_v2 tam dengeli).

Hipotezler (Ranked):
  H1: 06_smart_split.py → faces_split üretir (hash-split, hafif dengesiz)
  H2: dataset_restructure.py → faces_split_v2 üretir (tam 50/50)
  H3: Eğitim hangi split'i kullandığını loglamıyor
```

> **Somut Etki:** `train_loader` bazen 395,916 bazen 397,728 örnek yükler.  
> Bu, her eğitim run'ında farklı data karşılaştırmayı imkânsız kılar.

**Çözüm:**
```python
# data_pipeline.py — L578 civarı
# Sadece faces_split_v2 kullan, fallback'i kaldır
split_dir = paths.DATASET_DIR / "faces_split_v2" / split
assert split_dir.exists(), f"faces_split_v2/{split} bulunamadı! Önce dataset_restructure.py --execute çalıştırın."
```

---

### Sorun 2 — genimage (49,581) Split'lere Hiç Dahil Edilmemiş

```
Dosyalar: scripts/06_smart_split.py (L250-333)

Friction: 06_smart_split.py'deki real_sources ve fake_sources tanımları
genimage/ klasörünü listeliyor (L324-332) AMA faces/genimage/ var.
dataset_restructure.py ise genimage'ı hiç taramıyor.

Etki: 49,581 diffusion-model fake görsel eğitimde YOK.
GenImage, en zor fake tipi (Stable Diffusion, DALL-E) → kritik boşluk.
```

**Çözüm:** `dataset_restructure.py`'nin `collect_all_files()` fonksiyonuna genimage ekle:
```python
# L172-183 — collect_all_files() içine ekle
genimage_dir = paths.FACES_DIR / "genimage"
if genimage_dir.exists():
    for f in genimage_dir.rglob("*"):
        if f.is_file() and f.suffix.lower() in SUPPORTED_FORMATS:
            files["fake"].append(str(f))
```

---

### Sorun 3 — Config Versiyonu Tutarsızlığı (YAML vs Python)

```
Dosyalar: configs/run3_config.yaml vs config.py

run3_config.yaml diyor:
  num_classes: 3       (REAL, FAKE, SPOOF)
  kd_alpha: 0.5        (KD aktif)
  num_workers: 12      (Windows'ta OOM/deadlock riski)
  lstm_hidden: 128     (model yok!)

config.py diyor:
  NUM_CLASSES: 2       (sadece REAL, FAKE)
  KD_ALPHA: 0.0        (KD kapalı)
  NUM_WORKERS: 4       (güvenli)

⚠️ YAML dosyası KULLANILMIYOR — trainer.py doğrudan config.py import ediyor.
YAML bir "tarihsel belge" haline gelmiş, yanıltıcı.
```

**Çözüm:** YAML'ı ya sil, ya da `configs/` klasörüne `DEPRECATED.md` ekle.

---

### Sorun 4 — Data Leakage Riski: faces_split'te Symlink + faces_split_v2'de Copy

```
Dosyalar: scripts/dataset_restructure.py (L361-368), dataset/faces_split_v2/

Windows'ta os.symlink yerine shutil.copy2 kullanıyor (doğru).
AMA faces_split'te orijinal dosyalar var, faces_split_v2'de kopyalar var.
Leakage checker sadece MD5 bazlı — aynı görsel farklı isimle listelenirse geçer.

jury_leakage_report.csv: 159,999 byte — içeriği kontrol gerekiyor.
leakage_report.csv: Sadece başlık satırı var → HIÇ leakage testi yapılmamış!
```

**Çözüm:**
```bash
python scripts/leakage_checker.py --check dataset/faces_split_v2/train dataset/faces_split_v2/test --no-embedding
```

---

### Sorun 5 — Frekans Cache'i Devre Dışı: Her Epoch'ta 283K DWT Hesabı

```
Dosyalar: core/data_pipeline.py (L517-519)

_get_freq_cached() metodunun adı yanıltıcı:
    def _get_freq_cached(self, img_path, img_np):
        return self.dwt(img_np)  # Cache HİÇ KULLANILMIYOR!

"444GB disk kaplamasını önle" yorumuyla cache kapatılmış.
AMA her epoch'ta 283K × 3 wavelet × 4 band = ~3.4M DWT hesabı yapılıyor.
RTX 4070'te bu, GPU'dan çok CPU'yu meşgul ediyor (DataLoader bottleneck).
```

**Çözüm — Seçenek A (Küçük):** `--pin_memory=True` + `NUM_WORKERS=4` yeterli mi profil et:
```python
# trainer.py'de DataLoader profili
from torch.profiler import profile, record_function, ProfilerActivity
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    for i, batch in enumerate(train_loader):
        if i == 10: break
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
```

**Çözüm — Seçenek B (Kalıcı):** Frekans map'lerini `.npy` olarak precache et (mevcut `precache_mesh.py` var!):
```bash
python scripts/precache_mesh.py  # var, ama freq için değil
# precache_freq.py oluştur
```

---

## 🟡 MİMARİ SORUNLAR (architecture-review perspektifi)

### Candidate A: Çift Split Yönetimi — Derin Modül Eksikliği

```
Files: scripts/06_smart_split.py, scripts/dataset_restructure.py, core/data_pipeline.py

Friction: Split oluşturma mantığı 3 farklı script'e dağılmış:
  - 06_smart_split.py → faces_split (kalite+duplicate filtreli)
  - dataset_restructure.py → faces_split_v2 (kalite filtreli)
  - data_pipeline.py → hangisinin kullanılacağını runtime'da karar veriyor

Benefit: Tek "DatasetBuilder" modülü = tek kaynak of truth
Risk: Mevcut split'leri geçersiz kılabilir
Effort: M
```

**Önerilen Mimari:**
```
scripts/
  build_dataset.py     ← TEK giriş noktası
    --source: faces/
    --output: faces_split_final/
    --strategy: smart | restructure
    --balance: 50-50 | proportional
    --verify: leakage check dahil
```

---

### Candidate B: DeepfakeDataset.__getitem__ Çok Şey Yapıyor (Shallow Separation)

```
Şu an __getitem__ içinde:
  1. PIL ile görsel yükleme
  2. Class-aware augmentation (HardReal / FakeAware)
  3. DWT frekans hesabı (444GB engeli için cache kapalı)
  4. MediaPipe Face Mesh (cache-first, lazy-init)
  5. RGB transform pipeline
  6. 5-tuple return

Bir item için potansiyel işlem süresi: 50-500ms (mesh varsa)
Bu, DataLoader'ı boğuyor → GPU boşta bekliyor
```

**Çözüm:**
```python
# Mesh ve freq'i ayrı Dataset wrapper'larına taşı:
class FreqAugDataset(Dataset):
    def __getitem__(self, idx):
        rgb, label, src = self.base[idx]
        freq = self.freq_extractor(rgb)
        return rgb, freq, label, src

# Mesh'i tamamen precache yap, hiç runtime hesaplama
```

---

### Candidate C: run3_config.yaml — Ölü Kod Riski

```
Modül Derinlik Skoru (1-5):
  Interface: 1 — YAML okunmuyor, trainer doğrudan config.py'yi import ediyor
  Behavior: 1 — Davranış sıfır
  Testability: 1 — Test edilemiyor (kullanılmıyor)
  Locality: 1 — num_classes=3 config.py'deki 2 ile çelişiyor
  Stability: 1 — Hiç güncellenmeden kalmış

Ortalama: 1.0/5 → Güçlü silme/arşivleme adayı
```

---

## 🔵 PyTorch Expert Bulguları

### Bulgu 1: zero_grad() Sırası Yanlış (trainer.py L216)

```python
# MEVCUT (L216 civarı):
optimizer.zero_grad()  # ← Loop başında
for step, batch in enumerate(pbar):
    ...
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()  # ← Accumulation sonunda DA var

# PyTorch Best Practice (§4.2):
optimizer.zero_grad(set_to_none=True)  # Faster + more memory-efficient
```

**Etki:** Minör ama tutarsız — iki `zero_grad` çağrısı var (L216 ve L290). `set_to_none=True` ile memory %5-10 tasarruf.

---

### Bulgu 2: Validation'da AMP dtype Tutarsızlığı

```python
# trainer.py L368 — MEVCUT:
with autocast(device_type="cuda", enabled=use_amp):
    logits = model(rgb, freq, mesh)
    losses = criterion(logits, labels)

# Train'de L262:
with autocast(device_type="cuda"):  # enabled parametresi YOK
```

`enabled=use_amp` parametresi train'de eksik → CPU modunda bile `autocast` aktif olabilir. Bu NaN riskine yol açar.

**Çözüm:**
```python
# Her iki yerde de:
with autocast(device_type="cuda", enabled=use_amp, dtype=torch.float16):
```

---

### Bulgu 3: FGSM Sonrası `model.zero_grad()` Yanlış

```python
# trainer.py L256-258:
adv_loss.backward()
if rgb_adv.grad is not None:
    rgb = (rgb + eps * rgb_adv.grad.sign()).detach()
model.zero_grad()  # ← SORUN!
```

`adv_loss.backward()` modelin tüm gradyanlarını kirletiyor. `model.zero_grad()` çağrısı bu gradyanları temizliyor AMA bu `optimizer.zero_grad()` ile senkronize değil. Accumulation adımlarında gradyan birikimi bozulabilir.

**Çözüm:**
```python
# rgb_adv'nin gradyanını izole et:
with torch.enable_grad():
    rgb_adv = rgb.detach().clone().requires_grad_(True)
    with autocast(device_type="cuda", enabled=use_amp):
        adv_logits = model(rgb_adv, freq.detach(), mesh.detach())
        adv_loss = F.cross_entropy(adv_logits, labels_a)
    adv_loss.backward(inputs=[rgb_adv])  # Sadece rgb_adv gradyanı
# model.zero_grad() KALDIR
```

---

### Bulgu 4: Curriculum Phase Kontrolü Off-by-One

```python
# config.py L216-219:
CURRICULUM_PHASES = [
    {"start": 0, "end": 4, "hard_real_ratio": 0.0},
    {"start": 5, "end": 9, "hard_real_ratio": 0.15},
    ...
]

# trainer.py L612:
if phase["start"] <= epoch <= phase["end"]:
```

`epoch=4` ve `epoch=5` aynı anda iki phase'e girebilir (end=4, start=5 — aralarında boşluk yok). Bu OK görünür ama `epoch=4` eğer `end < 4` ise hiçbir phase'e girmez → `hard_real_ratio=0` kalır (sessiz hata).

---

## 📐 Pandas Expert: Veri Analizi Eksiklikleri

### Eksik: Dataset Istatistik Raporu (Pandera / DataFrame bazlı)

Mevcut `dataset_report.py` var ama şunlar eksik:

```python
import pandas as pd

# Önerilen: faces_split_v2 üzerinde hızlı analiz
def generate_dataset_stats(split_dir):
    records = []
    for split in ["train", "val", "test"]:
        for cls in ["real", "fake"]:
            p = Path(split_dir) / split / cls
            files = list(p.rglob("*")) if p.exists() else []
            for f in files:
                records.append({
                    "split": split, "label": cls,
                    "source": detect_source(f.name),
                    "size_kb": f.stat().st_size / 1024,
                    "ext": f.suffix.lower(),
                })

    df = pd.DataFrame(records)

    # Kaynak çeşitliliği analizi
    pivot = df.groupby(["split", "label", "source"]).size().unstack(fill_value=0)
    print(pivot)

    # Boyut dağılımı
    print(df.groupby("label")["size_kb"].describe())

    # Uzantı dağılımı (bozuk dosya göstergesi)
    print(df.groupby(["label", "ext"]).size())
```

---

## ✅ EYLEM PLANI (Öncelik Sırasına Göre)

| # | Öncelik | Sorun | Dosya | Süre |
|---|---------|-------|-------|------|
| 1 | 🔴 KRİTİK | `genimage` (49,581 görsel) split'e dahil et | `dataset_restructure.py` | 2 saat |
| 2 | 🔴 KRİTİK | `data_pipeline.py`'de `faces_split_v2` sabit kullan | `data_pipeline.py` L578-581 | 15 dk |
| 3 | 🔴 KRİTİK | FGSM `adv_loss.backward()` gradyan izolasyonu | `trainer.py` L251-258 | 30 dk |
| 4 | 🟡 ORTA | `autocast(enabled=use_amp)` her yerde tutarlı yap | `trainer.py` L262, L368 | 15 dk |
| 5 | 🟡 ORTA | `zero_grad(set_to_none=True)` → memory tasarrufu | `trainer.py` L216, L291 | 10 dk |
| 6 | 🟡 ORTA | `run3_config.yaml` → arşivle veya sil | `configs/` | 5 dk |
| 7 | 🟢 DÜŞÜK | Frekans map precache (`precache_freq.py`) | yeni script | 1 gün |
| 8 | 🟢 DÜŞÜK | `generate_dataset_stats()` pandas raporu | yeni script | 2 saat |
| 9 | 🟢 DÜŞÜK | `leakage_checker.py` çalıştır (mevcut CSV boş) | CLI | 1 saat |

---

## 🔬 Somut Test Protokolü (debug-diagnose §2)

```bash
# Phase 1: Feedback Loop — veri seti sayımını doğrula
python -c "
from core.data_pipeline import get_dataloaders
train, val, test = get_dataloaders()
print(f'Train: {len(train.dataset):,}')
print(f'Val: {len(val.dataset):,}')
print(f'Test: {len(test.dataset):,}')
# Hangi split kullanıldığını logla
"

# Phase 2: FGSM gradyan izolasyon testi
python -c "
import torch
from core.dual_mobilenetv3 import DualPathDeepfakeDetector
model = DualPathDeepfakeDetector()
rgb = torch.randn(2, 3, 224, 224).requires_grad_(True)
freq = torch.randn(2, 18, 224, 224)
mesh = torch.randn(2, 1404)
logits, _ = model.forward_with_embeddings(rgb, freq, mesh)
loss = logits.sum()
loss.backward(inputs=[rgb])  # Sadece rgb gradyanı
assert all(p.grad is None for p in model.parameters()), 'Model gradyanları kirli!'
print('FGSM izolasyon: OK')
"

# Phase 3: DataLoader bottleneck profili
python -c "
import time
from core.data_pipeline import get_dataloaders
train, _, _ = get_dataloaders()
t0 = time.time()
for i, batch in enumerate(train):
    if i == 50: break
print(f'50 batch: {time.time()-t0:.1f}s → {50/(time.time()-t0):.1f} batch/s')
"
```

---

*Bu rapor `pytorch-expert`, `debug-diagnose`, `architecture-review` ve `pandas-expert` skill'leri kullanılarak üretilmiştir.*
