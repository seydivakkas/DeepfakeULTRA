"""
Deepfake Detection System v3 — Merkezi Konfigürasyon
Tüm sabitler, yollar ve model parametreleri burada tanımlanır.
"""

import os
import torch
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any


# ═══════════════════════════════════════════════════════════
# CİHAZ SEÇİMİ
# ═══════════════════════════════════════════════════════════
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = min(12, os.cpu_count() or 1)  # ↑ 6→12 (i7-14700HX 20 thread)


# ═══════════════════════════════════════════════════════════
# PROJE YOL YAPISI
# ═══════════════════════════════════════════════════════════
@dataclass
class PathConfig:
    """Proje genelinde kullanılan dosya/dizin yolları."""
    BASE_DIR: Path = field(default_factory=lambda: Path(__file__).parent)

    # Veri seti (eski uyumluluk)
    DATASET_DIR: Path = field(default=None)
    TRAIN_DIR: Path = field(default=None)
    VAL_DIR: Path = field(default=None)
    TEST_DIR: Path = field(default=None)

    # Yeni veri seti yapısı (FF++ + CASIA + DF40)
    FFPP_DIR: Path = field(default=None)
    CASIA_DIR: Path = field(default=None)
    BENCHMARK_DIR: Path = field(default=None)

    # V5 veri seti yolları (faces/ tabanlı)
    FACES_DIR: Path = field(default=None)
    FACES_FFPP_DIR: Path = field(default=None)
    FACES_DF40_DIR: Path = field(default=None)
    FACES_CELEBA_DIR: Path = field(default=None)
    FACES_FFHQ_DIR: Path = field(default=None)

    # V5 yeni kaynaklar
    FACES_FFHQ_1024_DIR: Path = field(default=None)
    FACES_UTKFACE_DIR: Path = field(default=None)
    FACES_SIDSET_DIR: Path = field(default=None)
    FACES_RFW_DIR: Path = field(default=None)
    FACES_VGGFACE2_DIR: Path = field(default=None)

    # Model çıktıları
    MODEL_DIR: Path = field(default=None)
    BEST_MODEL_PATH: Path = field(default=None)
    ONNX_MODEL_PATH: Path = field(default=None)
    QUANTIZED_MODEL_PATH: Path = field(default=None)

    # Loglar ve raporlar
    MLRUNS_DIR: Path = field(default=None)
    ANALYTICS_DIR: Path = field(default=None)
    REPORTS_DIR: Path = field(default=None)
    AUDIT_LOG_DIR: Path = field(default=None)

    # Statik dosyalar
    STATIC_DIR: Path = field(default=None)
    REFERENCE_FEATURES_PATH: Path = field(default=None)

    def __post_init__(self):
        """Türetilmiş yolları hesapla."""
        b = self.BASE_DIR
        self.DATASET_DIR = self.DATASET_DIR or b / "dataset"
        self.TRAIN_DIR = self.TRAIN_DIR or self.DATASET_DIR / "train"
        self.VAL_DIR = self.VAL_DIR or self.DATASET_DIR / "val"
        self.TEST_DIR = self.TEST_DIR or self.DATASET_DIR / "test"

        # Eski veri seti yolları (geriye uyumluluk)
        self.FFPP_DIR = self.FFPP_DIR or self.DATASET_DIR / "deepfake" / "ff++"
        self.CASIA_DIR = self.CASIA_DIR or self.DATASET_DIR / "liveness" / "casia-fasd"
        self.BENCHMARK_DIR = self.BENCHMARK_DIR or self.DATASET_DIR / "benchmark"

        # V5 faces/ tabanlı yollar
        self.FACES_DIR = self.FACES_DIR or self.DATASET_DIR / "faces"
        self.FACES_FFPP_DIR = self.FACES_FFPP_DIR or self.FACES_DIR / "ffpp"
        self.FACES_DF40_DIR = self.FACES_DF40_DIR or self.FACES_DIR / "df40"
        self.FACES_CELEBA_DIR = self.FACES_CELEBA_DIR or self.FACES_DIR / "celeba_hq"
        self.FACES_FFHQ_DIR = self.FACES_FFHQ_DIR or self.FACES_DIR / "ffhq"

        # V5 yeni kaynak yolları
        self.FACES_FFHQ_1024_DIR = self.FACES_FFHQ_1024_DIR or self.FACES_DIR / "ffhq_1024_filtered"
        self.FACES_UTKFACE_DIR = self.FACES_UTKFACE_DIR or self.FACES_DIR / "utkface"
        self.FACES_SIDSET_DIR = self.FACES_SIDSET_DIR or self.FACES_DIR / "sidset"
        self.FACES_RFW_DIR = self.FACES_RFW_DIR or self.FACES_DIR / "rfw_caucasian"
        self.FACES_VGGFACE2_DIR = self.FACES_VGGFACE2_DIR or self.FACES_DIR / "vggface2"

        self.MODEL_DIR = self.MODEL_DIR or b / "models"
        self.BEST_MODEL_PATH = self.BEST_MODEL_PATH or self.MODEL_DIR / "best_model.pth"
        self.ONNX_MODEL_PATH = self.ONNX_MODEL_PATH or self.MODEL_DIR / "model.onnx"
        self.QUANTIZED_MODEL_PATH = self.QUANTIZED_MODEL_PATH or self.MODEL_DIR / "model_quantized.pth"

        self.MLRUNS_DIR = self.MLRUNS_DIR or b / "mlruns"
        self.ANALYTICS_DIR = self.ANALYTICS_DIR or b / "analytics_logs"
        self.REPORTS_DIR = self.REPORTS_DIR or b / "reports"
        self.AUDIT_LOG_DIR = self.AUDIT_LOG_DIR or b / "audit_logs"

        self.STATIC_DIR = self.STATIC_DIR or b / "static"
        self.REFERENCE_FEATURES_PATH = self.REFERENCE_FEATURES_PATH or self.MODEL_DIR / "reference_features.npy"

    def ensure_dirs(self):
        """Gerekli dizinleri oluştur."""
        for d in [
            self.MODEL_DIR, self.MLRUNS_DIR, self.ANALYTICS_DIR,
            self.REPORTS_DIR, self.AUDIT_LOG_DIR, self.STATIC_DIR,
        ]:
            d.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════
# MODEL KONFİGÜRASYONU
# ═══════════════════════════════════════════════════════════
@dataclass
class ModelConfig:
    """DualPathDeepfakeDetector model parametreleri."""

    # Görüntü boyutu
    IMG_SIZE: int = 224
    NUM_CLASSES: int = 2  # 0=REAL, 1=FAKE (binary)

    # Sınıf isimleri
    CLASS_NAMES: List[str] = field(default_factory=lambda: ["REAL", "FAKE"])

    # Sınıf ağırlıkları (50-50 fiziksel denge — ek ağırlık gereksiz)
    CLASS_WEIGHTS: List[float] = field(default_factory=lambda: [1.0, 1.0])

    # Hiyerarşik inference alt-tipleri (eğitimde kullanılmaz)
    FAKE_SUBTYPES: List[str] = field(default_factory=lambda: ["digital", "physical", "ai_generated"])

    # ═══════════════════════════════════════════════════════════
    # HIZLI EĞİTİM STRATEJİSİ — 283K Dengeli Dataset
    # ═══════════════════════════════════════════════════════════
    # Felsefe: 283K görsel, 50/50 dengeli → agresif LR, erken unfreeze,
    # kısa epoch, OneCycleLR ile hızlı yakınsama.
    # ═══════════════════════════════════════════════════════════

    # Temel hiperparametreler
    LEARNING_RATE: float = 3e-4       # ↑ 1e-4 → 3e-4 (KD yok, daha agresif)
    WEIGHT_DECAY: float = 1e-4
    BATCH_SIZE: int = 20              # 20 — RTX 4070 8GB güvenli
    EPOCHS: int = 20                  # ↑ 15 → 20 (WeightedRandomSampler unique coverage)
    EARLY_STOPPING_PATIENCE: int = 4  # ↓ 5 → 4 (hızlı karar)
    GRADIENT_CLIP_MAX_NORM: float = 1.0

    # Gradient Accumulation (efektif batch = 128)
    GRADIENT_ACCUMULATION_STEPS: int = 8  # ↑ 4 → 8 (daha stabil gradient)

    # Mixed Precision (FP16 — %40 VRAM tasarrufu)
    USE_MIXED_PRECISION: bool = True

    # Focal Loss — dengeli veri için optimize
    FOCAL_GAMMA: float = 2.0
    FOCAL_ALPHA: float = 0.75   # ↑ 0.5 → 0.75 (Hard-negative ağırlıklı)
    LABEL_SMOOTHING: float = 0.1  # ↑ 0.05 → 0.1 (generalizasyon)

    # Contrastive Learning (Triplet Loss)
    USE_CONTRASTIVE: bool = True
    CONTRASTIVE_WEIGHT: float = 0.2   # L = 0.8×focal + 0.2×triplet
    CONTRASTIVE_MARGIN: float = 1.0
    CONTRASTIVE_DISTANCE: str = "cosine"
    CONTRASTIVE_MINING: str = "hard"

    # Knowledge Distillation — DEVRE DIŞI
    KD_ALPHA: float = 0.0
    KD_TEMPERATURE: float = 3.0

    # Scheduler — OneCycleLR benzeri agresif cosine
    COSINE_T_MAX: int = 20           # 20 epoch ile eşleşir
    COSINE_ETA_MIN: float = 1e-6

    # Warmup — kısa ve agresif
    WARMUP_EPOCHS: int = 1           # ↓ 2 → 1 (dengeli veri, hızlı başlangıç)
    BACKBONE_LR_FACTOR: float = 0.1  # Backbone için LR çarpanı

    # Kademeli Unfreeze — erken açma
    UNFREEZE_EPOCH: int = 3          # ↓ 5 → 3 (hızlı fine-tune başlasın)

    # Mixup Augmentation
    MIXUP_ALPHA: float = 0.2
    USE_MIXUP: bool = True

    # CutMix Augmentation
    CUTMIX_RATIO: float = 0.6  # %60 CutMix, %40 MixUp
    CUTMIX_ALPHA: float = 1.0

    # ReduceLROnPlateau (cosine scheduler sonrası yedek)
    PLATEAU_FACTOR: float = 0.5
    PLATEAU_PATIENCE: int = 2

    # Gradient Checkpointing (VRAM optimizasyonu)
    USE_GRADIENT_CHECKPOINTING: bool = True

    # FGSM Adversarial Training (hafif augmentation)
    USE_FGSM_TRAINING: bool = True
    FGSM_EPSILON_MIN: float = 0.005   # Minimum pertürbasyon
    FGSM_EPSILON_MAX: float = 0.03    # Maksimum pertürbasyon
    FGSM_EVERY_N_STEPS: int = 4       # Her N step'te bir FGSM uygula
    FGSM_START_EPOCH: int = 2         # Warmup sonrası başla

    # Curriculum Learning
    USE_CURRICULUM: bool = True
    CURRICULUM_PHASES: list = None  # Epoch aralıkları ve hard-real oranları
    HARD_REAL_AUG_PROB: float = 0.3   # Hard-real augmentation olasılığı

    def __post_init__(self):
        if self.CURRICULUM_PHASES is None:
            self.CURRICULUM_PHASES = [
                {"start": 0, "end": 4, "hard_real_ratio": 0.0},
                {"start": 5, "end": 9, "hard_real_ratio": 0.15},
                {"start": 10, "end": 15, "hard_real_ratio": 0.30},
                {"start": 16, "end": 999, "hard_real_ratio": 0.40},
            ]

    # Model mimarisi
    RGB_BACKBONE: str = "mobilenet_v3_large"
    FREQ_BACKBONE: str = "mobilenet_v3_large"
    MESH_INPUT_DIM: int = 1404  # 468 landmarks × 3 (x, y, z)
    MESH_HIDDEN_DIM: int = 256
    MESH_OUTPUT_DIM: int = 128

    # Cross-Branch Transformer Encoder (BiLSTM yerine — tek görsel analizine optimize)
    XBRANCH_HEADS: int = 4       # Self-attention head sayısı
    XBRANCH_LAYERS: int = 2      # Transformer encoder katman sayısı
    XBRANCH_DROPOUT: float = 0.1 # Attention dropout
    XBRANCH_FF_MULT: int = 2     # Feed-forward boyut çarpanı (dim × mult)

    # Fusion
    FUSION_DIM: int = 960  # MobileNetV3-Large son katman çıktı boyutu

    # Dropout
    CLASSIFIER_DROPOUT: float = 0.5

    # DWT + Hibrit Frekans Analizi (Run 5: DWT+DCT+Phase)
    DWT_WAVELETS: List[str] = field(default_factory=lambda: ["haar", "db2", "coif1"])
    DWT_CHANNELS: int = 18  # 12 DWT + 3 DCT + 3 Phase = 18 kanal (Run 5)
    USE_HYBRID_FREQ: bool = True  # True=HybridFrequencyExtractor, False=MultiScaleDWT

    # TTA
    TTA_AUGMENTATIONS: int = 15

    # Karar Esikleri (3 katmanli)
    # FAKE: fake_prob >= FAKE_THRESHOLD
    # UNCERTAIN: REAL_THRESHOLD < fake_prob < FAKE_THRESHOLD
    # REAL: fake_prob <= REAL_THRESHOLD
    FAKE_THRESHOLD: float = 0.70    # Bu deger uzerinde FAKE
    REAL_THRESHOLD: float = 0.40    # Bu deger altinda REAL
    # Aradaki bolge: UNCERTAIN (belirsiz)

    # MC Dropout
    MC_DROPOUT_PASSES: int = 30

    # Deployment & Kalibrasyon (G5)
    CALIBRATION_TARGET_ECE: float = 0.05  # < %5 hedef
    ONNX_OPSET_VERSION: int = 17
    USE_FP16_EXPORT: bool = True


# ═══════════════════════════════════════════════════════════
# API KONFİGÜRASYONU
# ═══════════════════════════════════════════════════════════
@dataclass
class APIConfig:
    """FastAPI ve servis konfigürasyonu."""
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    GRADIO_PORT: int = 7860

    # JWT
    JWT_SECRET: str = field(
        default_factory=lambda: os.environ.get("JWT_SECRET", "deepfake-system-v3-secret-2026")
    )
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRATION_HOURS: int = 24

    # Rate Limiting
    RATE_LIMIT: str = "30/minute"

    # CORS
    CORS_ORIGINS: List[str] = field(default_factory=lambda: ["*"])


# ═══════════════════════════════════════════════════════════
# LLM KONFİGÜRASYONU
# ═══════════════════════════════════════════════════════════
@dataclass
class LLMConfig:
    """Gemini / Ollama konfigürasyonu."""

    # Gemini 3.0 Pro
    GEMINI_API_KEY: str = field(
        default_factory=lambda: os.environ.get("GEMINI_API_KEY", "")
    )
    GEMINI_MODEL: str = "gemini-3.0-pro"
    GEMINI_TEMPERATURE: float = 0.2
    GEMINI_MAX_TOKENS: int = 8192

    # Ollama (fallback)
    OLLAMA_MODEL: str = "qwen2.5:7b"
    OLLAMA_BASE_URL: str = "http://localhost:11434"

    # RAG
    RAG_EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    RAG_TOP_K: int = 5


# ═══════════════════════════════════════════════════════════
# BİLDİRİM KONFİGÜRASYONU
# ═══════════════════════════════════════════════════════════
@dataclass
class NotificationConfig:
    """Harici bildirim servisleri."""
    SLACK_WEBHOOK_URL: str = field(
        default_factory=lambda: os.environ.get("SLACK_WEBHOOK_URL", "")
    )
    TELEGRAM_BOT_TOKEN: str = field(
        default_factory=lambda: os.environ.get("TELEGRAM_BOT_TOKEN", "")
    )
    TELEGRAM_CHAT_ID: str = field(
        default_factory=lambda: os.environ.get("TELEGRAM_CHAT_ID", "")
    )


# ═══════════════════════════════════════════════════════════
# GLOBAL KONFİGÜRASYON INSTANCE'LARI
# ═══════════════════════════════════════════════════════════
paths = PathConfig()
model_cfg = ModelConfig()
api_cfg = APIConfig()
llm_cfg = LLMConfig()
notif_cfg = NotificationConfig()

# Dizinleri oluştur
paths.ensure_dirs()

# Versiyonlama
VERSION = "3.0.0"
SYSTEM_NAME = "Deepfake Detection System"


if __name__ == "__main__":
    print(f"🔧 {SYSTEM_NAME} v{VERSION}")
    print(f"📱 Cihaz: {DEVICE}")
    print(f"📂 Proje: {paths.BASE_DIR}")
    print(f"🤖 Model: {model_cfg.RGB_BACKBONE}")
    print(f"🧠 LSTM: BiLSTM={model_cfg.LSTM_BIDIRECTIONAL}, "
          f"Layers={model_cfg.LSTM_LAYERS}, Hidden={model_cfg.LSTM_HIDDEN}")
    print(f"🎓 Öğretmen: {model_cfg.TEACHER_BACKBONE}")
    print(f"🔑 Gemini: {'✅ API Key mevcut' if llm_cfg.GEMINI_API_KEY else '❌ API Key yok'}")
