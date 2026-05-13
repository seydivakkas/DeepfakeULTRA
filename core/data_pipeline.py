"""
Deepfake Detection System v4 — Veri İşleme Pipeline
Binary sınıflandırma: REAL (0) / FAKE (1)
Tüm kaynaklar (FF++, AntiSpoof, DF40) birleşik dataset.

Kullanım:
    train_loader, val_loader, test_loader = get_dataloaders()
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
from pathlib import Path
from PIL import Image
from typing import Tuple, Optional, Dict, List
from config import model_cfg, paths, DEVICE, NUM_WORKERS

# Opsiyonel bağımlılıklar — graceful degradation
try:
    import pywt
    HAS_PYWT = True
except ImportError:
    HAS_PYWT = False
    print("PyWavelets yuklu degil. DWT devre disi.")

try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False
    print("MediaPipe yuklu degil. Face Mesh devre disi.")

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

# Hibrit frekans ve SBI augmentation (Run 5)
try:
    from core.frequency_v2 import HybridFrequencyExtractor
    HAS_HYBRID_FREQ = True
except ImportError:
    HAS_HYBRID_FREQ = False

try:
    from core.sbi_augmentation import SBITransform
    HAS_SBI = True
except ImportError:
    HAS_SBI = False

# Class-aware augmentation (G3)
try:
    from core.hard_real_augmentation import HardRealAugmentation
    from core.fake_augmentation import FakeAwareAugmentation
    HAS_CLASS_AWARE_AUG = True
except ImportError:
    HAS_CLASS_AWARE_AUG = False


# ═══════════════════════════════════════════════════════════
# MULTI-SCALE DWT FREKANS ANALİZİ
# ═══════════════════════════════════════════════════════════
class MultiScaleDWT:
    """
    Haar + db2 + coif1 wavelet füzyonu.
    3 wavelet × 4 alt bant (LL, LH, HL, HH) = 12 kanallı frekans haritası.
    """

    def __init__(self, wavelets: list = None, size: int = model_cfg.IMG_SIZE):
        self.wavelets = wavelets or model_cfg.DWT_WAVELETS
        self.size = size

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if not HAS_PYWT:
            return np.zeros((model_cfg.DWT_CHANNELS, self.size, self.size), dtype=np.float32)

        if len(image.shape) == 3:
            gray = np.mean(image, axis=2).astype(np.float32)
        else:
            gray = image.astype(np.float32)

        if gray.shape[0] != self.size or gray.shape[1] != self.size:
            from PIL import Image as PILImage
            gray = np.array(PILImage.fromarray(gray.astype(np.uint8)).resize(
                (self.size, self.size), PILImage.BILINEAR
            )).astype(np.float32)

        gray = gray / 255.0 if gray.max() > 1.0 else gray

        channels = []
        for wavelet in self.wavelets:
            try:
                coeffs = pywt.dwt2(gray, wavelet)
                cA, (cH, cV, cD) = coeffs
                for band in [cA, cH, cV, cD]:
                    resized = np.array(
                        Image.fromarray(band.astype(np.float32)).resize(
                            (self.size, self.size), Image.BILINEAR
                        )
                    )
                    channels.append(resized)
            except Exception:
                for _ in range(4):
                    channels.append(np.zeros((self.size, self.size), dtype=np.float32))

        freq_map = np.stack(channels, axis=0)  # (12, H, W)
        return freq_map


# ═══════════════════════════════════════════════════════════
# FACE MESH LANDMARK ÇIKARIMI
# ═══════════════════════════════════════════════════════════
class FaceMeshExtractor:
    """MediaPipe Face Mesh ile 468 3D yüz landmark noktası çıkarır."""

    def __init__(self):
        self.face_mesh = None
        if HAS_MEDIAPIPE:
            try:
                self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=True,
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.5,
                )
            except (AttributeError, Exception):
                self.face_mesh = None

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if not HAS_MEDIAPIPE or self.face_mesh is None:
            return np.zeros(model_cfg.MESH_INPUT_DIM, dtype=np.float32)

        try:
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            results = self.face_mesh.process(image)
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0]
                coords = []
                for lm in landmarks.landmark:
                    coords.extend([lm.x, lm.y, lm.z])
                return np.array(coords, dtype=np.float32)
        except Exception:
            pass

        return np.zeros(model_cfg.MESH_INPUT_DIM, dtype=np.float32)

    def __del__(self):
        if self.face_mesh:
            self.face_mesh.close()


# ═══════════════════════════════════════════════════════════
# BINARY DEEPFAKE DATASET (REAL=0, FAKE=1)
# ═══════════════════════════════════════════════════════════

# Dizin adı → binary etiket
LABEL_MAP = {
    "real": 0,      # REAL sınıfı
    "fake": 1,      # FAKE sınıfı (dijital + AI)
}

# Veri kaynağı → hiyerarşik inference alt-tipi (eğitimde kullanılmaz)
SOURCE_SUBTYPE_MAP = {
    "ffpp": "digital",
    "df40_face_swap": "digital",
    "df40_face_reenact": "digital",
    "df40_entire_face": "ai_generated",
    "df40_face_edit": "ai_generated",
    "sidset_synthetic": "ai_generated",
    "sidset_tampered": "digital",
    "utkface": "real_portrait",
    "ffhq_1024": "real_portrait",
}


# ═══════════════════════════════════════════════════════════
# SOSYAL MEDYA SIKIŞTIRMA AUGMENTASYONU
# ═══════════════════════════════════════════════════════════

SOCIAL_MEDIA_PLATFORMS = {
    "twitter": {"max_dim": 1280, "quality": (80, 88), "weight": 0.55},
    "tiktok": {"max_dim": 1080, "quality": (65, 75), "weight": 0.45},
}


class SocialMediaCompress:
    """
    Sosyal medya platform sikistirma simulasyonu (epoch-aware).

    Epoch-bazli dinamik sikistirma orani:
        Epoch  1-5:  %70 sikistirilmis (sikistirmayi ogren)
        Epoch  6-15: %50 sikistirilmis (dengeli ogrenme)
        Epoch 16-30: %60 sikistirilmis (gercek dunya odak)
    """

    def __init__(self, platforms: dict = None, epoch: int = 1):
        self.platforms = platforms or SOCIAL_MEDIA_PLATFORMS
        self.epoch = epoch
        names = list(self.platforms.keys())
        weights = [self.platforms[n]["weight"] for n in names]
        total = sum(weights)
        self._names = names
        self._probs = [w / total for w in weights]

    def get_compress_probability(self) -> float:
        """Epoch'a gore sikistirma olasiligi."""
        if self.epoch <= 5:
            return 0.70
        elif self.epoch <= 15:
            return 0.50
        else:
            return 0.60

    def set_epoch(self, epoch: int):
        """Epoch guncelle — trainer tarafindan cagirilir."""
        self.epoch = epoch

    def __call__(self, img):
        import io
        import random

        name = random.choices(self._names, weights=self._probs, k=1)[0]
        cfg = self.platforms[name]

        # Boyut sinirla
        w, h = img.size
        max_dim = cfg["max_dim"]
        if max(w, h) > max_dim:
            ratio = max_dim / max(w, h)
            img = img.resize((int(w * ratio), int(h * ratio)), Image.BILINEAR)

        # JPEG sikistirma
        q_min, q_max = cfg["quality"]
        quality = random.randint(q_min, q_max)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        img = Image.open(buffer).convert("RGB")

        return img

    def __repr__(self):
        return (f"SocialMediaCompress(epoch={self.epoch}, "
                f"p={self.get_compress_probability():.0%})")


class _SocialCompressWrapper:
    """Picklable wrapper for SocialMediaCompress (Python 3.14 uyumlu)."""
    def __init__(self, compress_instance):
        self.compress = compress_instance

    def __call__(self, img):
        return self.compress(img)


class DeepfakeDataset(Dataset):
    """
    Binary Deepfake veri seti: REAL (0) / FAKE (1).

    Dizin yapısı:
        root/
        ├── real/   → label=0
        └── fake/   → label=1

    DF40 alt-dizinlerini de otomatik tarar:
        root/fake/face_swap/simswap/img001.png → label=1

    Her öğe döndürür:
        (rgb_tensor, freq_tensor, mesh_tensor, label, source_tag)
    """

    SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

    def __init__(
        self,
        root_dir: str,
        transform: transforms.Compose = None,
        split: str = "train",
        source_tag: str = "ffpp",
        recursive: bool = False,
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.source_tag = source_tag
        self.recursive = recursive
        self.transform = transform or self._default_transform(split)

        # Frekans ekstraktörü: Run 5 hibrit (DWT+DCT+Phase=18ch) veya eski DWT (12ch)
        if HAS_HYBRID_FREQ and getattr(model_cfg, 'USE_HYBRID_FREQ', False):
            self.dwt = HybridFrequencyExtractor(
                wavelets=model_cfg.DWT_WAVELETS,
                size=model_cfg.IMG_SIZE,
                include_dwt=True,
                include_dct=True,
                include_phase=True,
            )
        else:
            self.dwt = MultiScaleDWT()

        self.mesh_extractor = FaceMeshExtractor()

        # Class-aware augmentation (G3: REAL ve FAKE için ayrı pipeline)
        if split == "train" and HAS_CLASS_AWARE_AUG:
            self.hard_real_aug = HardRealAugmentation(prob=model_cfg.HARD_REAL_AUG_PROB)
            self.fake_aug = FakeAwareAugmentation(prob=0.5)
        else:
            self.hard_real_aug = None
            self.fake_aug = None

        # Dosyaları topla: (path, label, source_tag)
        self.samples: List[Tuple[str, int, str]] = []
        self._load_samples()

    def _load_samples(self):
        """real/, fake/, spoof/, live/ dizinlerinden dosya listesi oluştur."""
        for label_name, label_id in LABEL_MAP.items():
            label_dir = self.root_dir / label_name
            if not label_dir.exists():
                continue

            # Her zaman recursive tara — alt klasörler (sbi_generated vb.) dahil
            for f in sorted(label_dir.rglob("*")):
                if f.is_file() and f.suffix.lower() in self.SUPPORTED_FORMATS:
                    # .cache klasörlerini atla
                    if ".cache" in str(f):
                        continue
                    self.samples.append((str(f), label_id, self.source_tag))

        if not self.samples:
            print(f"  [UYARI] {self.root_dir} dizininde veri bulunamadi!")

    def _default_transform(self, split: str) -> transforms.Compose:
        size = model_cfg.IMG_SIZE
        if split == "train":
            # Epoch-aware sıkıştırma instance'ı
            self._social_compress = SocialMediaCompress(epoch=1)

            # SBI augmentation (Run 5) — REAL görsellere sahte deepfake artefaktı ekler
            sbi_transforms = []
            if HAS_SBI and getattr(model_cfg, 'USE_HYBRID_FREQ', False):
                sbi_transforms = [transforms.RandomApply([SBITransform()], p=0.15)]

            return transforms.Compose([
                transforms.Resize((size, size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.RandomAffine(
                    degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)
                ),
                # Epoch-aware sosyal medya sıkıştırma (picklable wrapper)
                transforms.RandomApply([
                    _SocialCompressWrapper(self._social_compress),
                ], p=self._social_compress.get_compress_probability()),
                # SBI augmentation (Run 5 — blending artifact ogren)
                *sbi_transforms,
                transforms.ColorJitter(
                    brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1
                ),
                transforms.RandomGrayscale(p=0.1),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
                # Ekstra agresif blur
                transforms.RandomApply([
                    transforms.GaussianBlur(kernel_size=5, sigma=(1.0, 3.0)),
                ], p=0.2),
                transforms.RandomPerspective(distortion_scale=0.1, p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                transforms.RandomErasing(p=0.15, scale=(0.02, 0.2)),
            ])
        else:
            return transforms.Compose([
                transforms.Resize((size, size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, str]:
        img_path, label, source_tag = self.samples[idx]

        image = Image.open(img_path).convert("RGB")

        # Class-aware augmentation (G3): label'a göre farklı pipeline
        if self.split == "train":
            if label == 0 and self.hard_real_aug:  # REAL
                image = self.hard_real_aug(image, label)
            elif label == 1 and self.fake_aug:     # FAKE
                image = self.fake_aug(image)

        img_np = np.array(image)

        # Frekans haritası — augmentation SONRASI hesapla
        freq_map = self._get_freq_cached(img_path, img_np)
        freq_tensor = torch.from_numpy(freq_map).float()

        # Face Mesh landmarks — disk cache
        mesh = self._get_mesh_cached(img_path, img_np)
        mesh_tensor = torch.from_numpy(mesh).float()

        # RGB transform (ortak: resize + normalize + flip/rotation)
        rgb_tensor = self.transform(image)

        return rgb_tensor, freq_tensor, mesh_tensor, label, source_tag

    def _get_cache_path(self, img_path: str, suffix: str) -> Path:
        """Cache dosya yolunu oluştur — orijinal dosyanın yanına .freq.npy / .mesh.npy"""
        p = Path(img_path)
        cache_dir = p.parent / ".cache"
        cache_dir.mkdir(exist_ok=True)
        return cache_dir / f"{p.stem}_{suffix}.npy"

    def _get_freq_cached(self, img_path: str, img_np: np.ndarray) -> np.ndarray:
        """Frekans haritasını hesapla (cache devre dışı — 444GB disk kaplamasını önle)."""
        return self.dwt(img_np)

    def _get_mesh_cached(self, img_path: str, img_np: np.ndarray) -> np.ndarray:
        """Face mesh vektörünü hesapla."""
        return self.mesh_extractor(img_np)

    def get_class_distribution(self) -> Dict[str, int]:
        dist = {name: 0 for name in model_cfg.CLASS_NAMES}
        for _, label_id, _ in self.samples:
            if label_id < len(model_cfg.CLASS_NAMES):
                dist[model_cfg.CLASS_NAMES[label_id]] += 1
        return dist


# ═══════════════════════════════════════════════════════════
# BİRLEŞİK DATASET FABRİKASI (V5: FF++ + DF40 + CelebA + FFHQ-1024 + UTKFace + SID-Set)
# ═══════════════════════════════════════════════════════════
def create_unified_dataset(split: str = "train") -> Dataset:
    """
    Tüm veri kaynaklarını birleştiren unified binary dataset (V5).

    Öncelik sırası:
        1. faces_split/{split}/ → fiziksel split (varsa)
        2. faces/ kaynaklarından ayrı ayrı yükle ve ConcatDataset

    Her öğe 5'li tuple: (rgb, freq, mesh, label, source_tag)
    """
    datasets = []

    # --- ONCELIK: Fiziksel split dizini ---
    split_dir = paths.DATASET_DIR / "faces_split" / split
    if split_dir.exists() and (split_dir / "real").exists():
        ds = DeepfakeDataset(str(split_dir), split=split, source_tag="unified")
        if len(ds) > 0:
            dist = ds.get_class_distribution()
            print(f"  [faces_split/{split}] {dist} (birlesik fiziksel split)")

            # Train ise: sikistirilmis veriyi de ekle
            if split == "train":
                compressed_base = paths.DATASET_DIR / "faces_split" / "train_compressed"
                if compressed_base.exists():
                    compressed_datasets = [ds]
                    for platform_dir in sorted(compressed_base.iterdir()):
                        if platform_dir.is_dir() and (platform_dir / "real").exists():
                            c_ds = DeepfakeDataset(
                                str(platform_dir), split=split,
                                source_tag=f"compressed_{platform_dir.name}"
                            )
                            if len(c_ds) > 0:
                                compressed_datasets.append(c_ds)
                                c_dist = c_ds.get_class_distribution()
                                print(f"  [compressed/{platform_dir.name}] {c_dist}")
                    if len(compressed_datasets) > 1:
                        unified = ConcatDataset(compressed_datasets)
                        print(f"  [OK] Orijinal + {len(compressed_datasets)-1} platform = "
                              f"{len(unified):,} toplam")
                        return unified

            return ds

    # ─── FALLBACK: Her kaynaktan ayrı yükle ───

    # 1. FF++ yüzler (dataset/faces/ffpp/)
    ffpp_dir = paths.FACES_FFPP_DIR
    if ffpp_dir.exists():
        ds = DeepfakeDataset(str(ffpp_dir), split=split, source_tag="ffpp")
        if len(ds) > 0:
            datasets.append(ds)
            dist = ds.get_class_distribution()
            print(f"  [FF++] {split}: {dist}")

    # 2. DF40 yüzler (dataset/faces/df40/) — recursive tarama
    df40_dir = paths.FACES_DF40_DIR
    if df40_dir.exists():
        ds = DeepfakeDataset(str(df40_dir), split=split, source_tag="df40", recursive=True)
        if len(ds) > 0:
            datasets.append(ds)
            dist = ds.get_class_distribution()
            print(f"  [DF40] {split}: {dist}")

    # 3. CelebA-HQ (dataset/faces/celeba_hq/) — sadece REAL
    celeba_dir = paths.FACES_CELEBA_DIR
    if celeba_dir.exists():
        ds = DeepfakeDataset(str(celeba_dir), split=split, source_tag="celeba_hq")
        if len(ds) > 0:
            datasets.append(ds)
            dist = ds.get_class_distribution()
            print(f"  [CelebA-HQ] {split}: {dist}")

    # 4. FFHQ 1024px filtered (dataset/faces/ffhq_1024_filtered/) — sadece REAL
    ffhq_1024_dir = paths.FACES_FFHQ_1024_DIR
    if ffhq_1024_dir.exists():
        ds = DeepfakeDataset(str(ffhq_1024_dir), split=split, source_tag="ffhq_1024")
        if len(ds) > 0:
            datasets.append(ds)
            dist = ds.get_class_distribution()
            print(f"  [FFHQ-1024] {split}: {dist}")

    # 5. UTKFace (dataset/faces/utkface/) — sadece REAL
    utkface_dir = paths.FACES_UTKFACE_DIR
    if utkface_dir.exists():
        ds = DeepfakeDataset(str(utkface_dir), split=split, source_tag="utkface")
        if len(ds) > 0:
            datasets.append(ds)
            dist = ds.get_class_distribution()
            print(f"  [UTKFace] {split}: {dist}")

    # 6. SID-Set (dataset/faces/sidset/) — REAL + FAKE
    sidset_dir = paths.FACES_SIDSET_DIR
    if sidset_dir.exists():
        ds = DeepfakeDataset(str(sidset_dir), split=split, source_tag="sidset", recursive=True)
        if len(ds) > 0:
            datasets.append(ds)
            dist = ds.get_class_distribution()
            print(f"  [SID-Set] {split}: {dist}")

    # 7. FFHQ eski (geriye uyumluluk, varsa)
    ffhq_dir = paths.FACES_FFHQ_DIR
    if ffhq_dir.exists():
        ds = DeepfakeDataset(str(ffhq_dir), split=split, source_tag="ffhq")
        if len(ds) > 0:
            datasets.append(ds)
            dist = ds.get_class_distribution()
            print(f"  [FFHQ] {split}: {dist}")

    # Eski dizin yapısı fallback
    if not datasets:
        old_ffpp = paths.FFPP_DIR / split
        if old_ffpp.exists():
            ds = DeepfakeDataset(str(old_ffpp), split=split, source_tag="ffpp")
            if len(ds) > 0:
                datasets.append(ds)
                print(f"  [FF++ eski] {split}: {len(ds)} ornek")

        old_dir = paths.DATASET_DIR / split
        if not datasets and old_dir.exists():
            ds = DeepfakeDataset(str(old_dir), split=split, source_tag="ffpp")
            if len(ds) > 0:
                datasets.append(ds)
                print(f"  [Eski yapi] {split}: {len(ds)} ornek")

    if not datasets:
        print(f"  [UYARI] {split} icin veri bulunamadi!")
        return DeepfakeDataset(str(paths.TRAIN_DIR), split=split)

    if len(datasets) == 1:
        return datasets[0]

    unified = ConcatDataset(datasets)
    print(f"  [OK] Birlesik {split}: {len(unified)} toplam ornek")
    return unified


# ═══════════════════════════════════════════════════════════
# SINIF DENGELEYİCİ YARDIMCILAR
# ═══════════════════════════════════════════════════════════
def _collect_all_labels(dataset: Dataset) -> List[int]:
    """ConcatDataset veya DeepfakeDataset'ten tüm etiketleri topla."""
    labels = []
    if isinstance(dataset, ConcatDataset):
        for ds in dataset.datasets:
            if hasattr(ds, "samples"):
                labels.extend([label for _, label, *_ in ds.samples])
    elif hasattr(dataset, "samples"):
        labels.extend([label for _, label, *_ in dataset.samples])
    return labels


def compute_sample_weights(dataset: Dataset) -> torch.Tensor:
    """
    Her örnek için örnekleme ağırlığı hesapla.
    Ağırlık = 1 / sınıf_frekansı → azınlık sınıfları daha sık örneklenir.
    """
    labels = _collect_all_labels(dataset)
    if not labels:
        return torch.ones(len(dataset))

    from collections import Counter
    class_counts = Counter(labels)
    total = len(labels)

    num_classes = len(class_counts)
    class_weight = {
        cls: total / (num_classes * count)
        for cls, count in class_counts.items()
    }

    print(f"  Sinif Dengeleme (WeightedRandomSampler):")
    for cls_id in sorted(class_counts.keys()):
        name = model_cfg.CLASS_NAMES[cls_id] if cls_id < len(model_cfg.CLASS_NAMES) else f"class_{cls_id}"
        count = class_counts[cls_id]
        weight = class_weight[cls_id]
        print(f"     {name}: {count:,} ornek -> agirlik {weight:.2f}")

    sample_weights = torch.tensor(
        [class_weight[label] for label in labels],
        dtype=torch.float64
    )
    return sample_weights


def compute_class_weights(dataset: Dataset) -> List[float]:
    """
    Loss fonksiyonu için sınıf ağırlıklarını hesapla.
    sqrt dampening ile normalize eder.
    """
    labels = _collect_all_labels(dataset)
    if not labels:
        return model_cfg.CLASS_WEIGHTS

    from collections import Counter
    import math
    class_counts = Counter(labels)
    total = len(labels)
    num_classes = model_cfg.NUM_CLASSES

    raw_weights = []
    for i in range(num_classes):
        count = class_counts.get(i, 0)
        if count > 0:
            raw_weights.append(total / (num_classes * count))
        else:
            raw_weights.append(1.0)

    dampened = [math.sqrt(w) for w in raw_weights]
    min_w = min(dampened) if min(dampened) > 0 else 1.0
    weights = [w / min_w for w in dampened]

    return weights


# ═══════════════════════════════════════════════════════════
# CUTMIX AUGMENTASYON
# ═══════════════════════════════════════════════════════════
def cutmix_data(rgb, freq, mesh, labels, alpha=1.0):
    """CutMix: bir bölgeyi başka örnekten kes-yapıştır."""
    lam = np.random.beta(alpha, alpha)
    batch_size = rgb.size(0)
    index = torch.randperm(batch_size, device=rgb.device)

    _, _, H, W = rgb.shape
    cut_ratio = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_ratio)
    cut_h = int(H * cut_ratio)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    x1 = np.clip(cx - cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y2 = np.clip(cy + cut_h // 2, 0, H)

    mixed_rgb = rgb.clone()
    mixed_rgb[:, :, y1:y2, x1:x2] = rgb[index, :, y1:y2, x1:x2]
    mixed_freq = freq.clone()
    mixed_freq[:, :, y1:y2, x1:x2] = freq[index, :, y1:y2, x1:x2]
    mixed_mesh = lam * mesh + (1 - lam) * mesh[index]

    lam = 1 - ((x2 - x1) * (y2 - y1) / (W * H))

    return mixed_rgb, mixed_freq, mixed_mesh, labels, labels[index], lam


# ═══════════════════════════════════════════════════════════
# DATALOADER FABRİKASI
# ═══════════════════════════════════════════════════════════
def get_dataloaders(
    batch_size: int = model_cfg.BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Birleşik train/val/test DataLoader'larını döndür.
    Binary sınıflandırma: REAL (0) / FAKE (1).
    WeightedRandomSampler ile sınıf dengeleme.
    """
    from torch.utils.data import WeightedRandomSampler

    print("Veri setleri yukleniyor (binary: REAL/FAKE)...")

    train_dataset = create_unified_dataset("train")
    val_dataset = create_unified_dataset("val")
    test_dataset = create_unified_dataset("test")

    # WeightedRandomSampler
    sample_weights = compute_sample_weights(train_dataset)
    # G3: %30 fazla örnekleme → replacement=True modunda ~%33 unique kayıp telafisi
    effective_samples = int(len(train_dataset) * 1.3)
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=effective_samples,
        replacement=True,
    )
    print(f"  ⚡ WeightedRandomSampler: num_samples={effective_samples} "
          f"(x1.3, orijinal={len(train_dataset)})")

    pin = DEVICE.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin,
        drop_last=True,
        prefetch_factor=4 if num_workers > 0 else None,  # ↑ 2→4
        persistent_workers=num_workers > 0,  # ↑ True: worker recycle yok
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
        prefetch_factor=4 if num_workers > 0 else None,  # ↑ 2→4
        persistent_workers=num_workers > 0,  # ↑ True: worker recycle yok
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
        prefetch_factor=4 if num_workers > 0 else None,  # ↑ 2→4
        persistent_workers=num_workers > 0,  # ↑ True: worker recycle yok
    )

    print(f"Veri seti: Train={len(train_dataset)}, "
          f"Val={len(val_dataset)}, Test={len(test_dataset)}")

    return train_loader, val_loader, test_loader


# ═══════════════════════════════════════════════════════════
# KAYNAK BAZLI DATALOADERS (geriye uyumluluk)
# ═══════════════════════════════════════════════════════════
def _build_loaders(train_ds, val_ds, test_ds, batch_size, num_workers):
    """WeightedRandomSampler ile DataLoader'lar oluştur."""
    from torch.utils.data import WeightedRandomSampler
    pin = DEVICE.type == "cuda"

    sample_weights = compute_sample_weights(train_ds)
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(train_ds), replacement=True
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler,
        num_workers=num_workers, pin_memory=pin, drop_last=True,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=False,
    ) if val_ds and len(val_ds) > 0 else None
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=False,
    ) if test_ds and len(test_ds) > 0 else None

    return train_loader, val_loader, test_loader


def get_ffpp_dataloaders(
    batch_size: int = model_cfg.BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Sadece FF++ veri seti — REAL/FAKE binary."""
    print("FF++ veri seti yukleniyor (REAL/FAKE)...")

    ffpp_dir = paths.FACES_FFPP_DIR
    if not ffpp_dir.exists():
        ffpp_dir = paths.FFPP_DIR  # eski yol fallback

    loaders_data = []
    for split in ["train", "val", "test"]:
        split_dir = ffpp_dir / split if (ffpp_dir / split).exists() else ffpp_dir
        ds = DeepfakeDataset(str(split_dir), split=split, source_tag="ffpp")
        loaders_data.append(ds if len(ds) > 0 else None)

    return _build_loaders(loaders_data[0], loaders_data[1], loaders_data[2], batch_size, num_workers)


# ═══════════════════════════════════════════════════════════
# VERİ SETİ OLUŞTURMA YARDIMCISI
# ═══════════════════════════════════════════════════════════
def create_dummy_dataset(n_samples: int = 20):
    """Test amaçlı binary sahte veri seti oluştur."""
    for split in ["train", "val", "test"]:
        for label in ["real", "fake"]:
            d = paths.FACES_FFPP_DIR / split / label
            d.mkdir(parents=True, exist_ok=True)
            for i in range(n_samples):
                img = Image.fromarray(
                    np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                )
                img.save(d / f"{label}_{i:04d}.jpg")

    print(f"Dummy binary veri seti olusturuldu: {paths.FACES_FFPP_DIR}")


if __name__ == "__main__":
    print("Veri pipeline testi...")

    dwt = MultiScaleDWT()
    dummy_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    freq = dwt(dummy_img)
    print(f"DWT: {dummy_img.shape} -> {freq.shape}")

    mesh_ext = FaceMeshExtractor()
    landmarks = mesh_ext(dummy_img)
    print(f"Face Mesh: {dummy_img.shape} -> {landmarks.shape}")

    print(f"\nEtiket Eslemesi: {LABEL_MAP}")
    print(f"Siniflar: {model_cfg.CLASS_NAMES}")
    print(f"NUM_CLASSES: {model_cfg.NUM_CLASSES}")
