"""
Deepfake Detection System v4 — Hiyerarşik Inference Pipeline
Aşama 1: Binary model → REAL / FAKE
Aşama 2: FAKE ise → alt-tip (Digital / Physical / AI-Generated)
"""
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms
from config import model_cfg, paths, DEVICE
from core.dual_mobilenetv3 import DualPathDeepfakeDetector
from core.data_pipeline import MultiScaleDWT, FaceMeshExtractor
from inference.subtype_classifier import SubtypeClassifier

# HybridFrequencyExtractor — 18 kanal (DWT 12 + DCT 3 + Phase 3)
try:
    from core.frequency_v2 import HybridFrequencyExtractor
    HAS_HYBRID_FREQ = True
except ImportError:
    HAS_HYBRID_FREQ = False


def _classify_verdict(fake_prob: float) -> str:
    """Binary karar: fake_prob > 0.5 ise FAKE, degilse REAL."""
    if fake_prob > 0.5:
        return "FAKE"
    return "REAL"


def is_likely_photo(image: np.ndarray) -> tuple:
    """
    Hibrit on-filtre: gorsel bir fotograf mi, yoksa cizim/karikatur/ilustrasyon mu?
    Asama 1: Istatistiksel analiz (<10ms, sifir bagimlilik)
    Asama 2: CLIP dogrulama (~500ms, yuksek dogruluk)

    Returns:
        (is_photo: bool, score: float, details: dict)
    """
    try:
        # Asama 1: Istatistiksel on-filtre
        stat_photo, stat_score, stat_details = _statistical_photo_check(image)

        # Asama 2: CLIP dogrulama (istatistiksel filtre supheliyse)
        clip_score = None
        clip_label = None
        try:
            clip_photo, clip_score, clip_label = _clip_photo_check(image)
            stat_details["clip_score"] = round(clip_score, 4)
            stat_details["clip_label"] = clip_label

            # Hibrit skor: CLIP %60, istatistiksel %40
            combined = clip_score * 0.60 + stat_score * 0.40
            stat_details["combined_score"] = round(combined, 4)
            stat_details["method"] = "clip+statistical"

            is_photo = combined > 0.40
            return is_photo, combined, stat_details

        except Exception:
            # CLIP yuklu degilse sadece istatistiksel filtre
            stat_details["method"] = "statistical_only"
            return stat_photo, stat_score, stat_details

    except Exception:
        return True, 1.0, {}


def _statistical_photo_check(image: np.ndarray) -> tuple:
    """Istatistiksel fotograf/cizim ayrimi. <10ms."""
    h, w = image.shape[:2]
    total_pixels = h * w

    # 1. Benzersiz renk orani
    pixels = image.reshape(-1, 3)
    quantized = (pixels // 16) * 16
    unique_colors = len(np.unique(quantized, axis=0))
    color_ratio = unique_colors / total_pixels

    # 2. Kenar keskinligi
    gray = np.mean(image, axis=2).astype(np.float32)
    grad_x = np.abs(np.diff(gray, axis=1))
    grad_y = np.abs(np.diff(gray, axis=0))
    sharp_edges = np.mean(grad_x > 40) + np.mean(grad_y > 40)
    sharp_ratio = sharp_edges / 2

    # 3. Gurultu seviyesi
    block_size = 3
    noise_vals = []
    for i in range(0, h - block_size, block_size * 2):
        for j in range(0, w - block_size, block_size * 2):
            block = gray[i:i+block_size, j:j+block_size]
            noise_vals.append(np.std(block))
    noise_std = np.mean(noise_vals) if noise_vals else 0

    # 4. Duz renk bolgesi orani — grad_x (h,w-1) ve grad_y (h-1,w) boyutlarini esle
    min_h = min(grad_x.shape[0], grad_y.shape[0])
    min_w = min(grad_x.shape[1], grad_y.shape[1])
    flat_mask = (grad_x[:min_h, :min_w] < 3) & (grad_y[:min_h, :min_w] < 3)
    flat_ratio = np.mean(flat_mask)

    score = (
        min(color_ratio / 0.15, 1.0) * 0.30 +
        max(1.0 - sharp_ratio / 0.15, 0) * 0.20 +
        min(noise_std / 4.0, 1.0) * 0.25 +
        max(1.0 - flat_ratio / 0.8, 0) * 0.25
    )

    details = {
        "color_ratio": round(color_ratio, 4),
        "sharp_ratio": round(sharp_ratio, 4),
        "noise_std": round(float(noise_std), 2),
        "flat_ratio": round(float(flat_ratio), 4),
        "photo_score": round(score, 4),
    }

    return score > 0.35, score, details


# CLIP modeli — lazy load (ilk kullanimda yuklenir)
_clip_model = None
_clip_processor = None

def _clip_photo_check(image: np.ndarray) -> tuple:
    """
    CLIP tabanli fotograf/cizim ayrimi.
    'a photograph of a person' vs 'a cartoon drawing illustration' karsilastirir.
    ~500ms, yuksek dogruluk.
    """
    global _clip_model, _clip_processor
    from transformers import CLIPProcessor, CLIPModel

    if _clip_model is None:
        _clip_model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32"
        ).to("cpu").eval()
        _clip_processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32"
        )

    pil_image = Image.fromarray(image)

    texts = [
        "a real photograph of a human face",
        "a cartoon drawing illustration caricature of a face",
    ]

    inputs = _clip_processor(
        text=texts, images=pil_image, return_tensors="pt",
        padding=True, truncation=True,
    )
    with torch.no_grad():
        outputs = _clip_model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1)[0]

    photo_prob = float(probs[0])
    cartoon_prob = float(probs[1])

    is_photo = photo_prob > 0.5
    label = "photograph" if is_photo else "cartoon/illustration"

    return is_photo, photo_prob, label

# MC Dropout opsiyonel
try:
    from ml_extensions.mc_dropout import MCDropoutPredictor, MCDropoutConfig
    HAS_MC = True
except ImportError:
    HAS_MC = False

# Temperature Scaling opsiyonel
try:
    from ml_extensions.temperature_scaling import TemperatureScaler
    HAS_TEMP = True
except ImportError:
    HAS_TEMP = False


class DeepfakePredictor:
    """
    Hiyerarşik tahmin sınıfı.
    Aşama 1: Binary REAL/FAKE sınıflandırma
    Aşama 2: FAKE tespiti sonrası alt-tip belirleme
    """

    def __init__(self, model_path=None, device=DEVICE):
        self.device = device
        self.model = DualPathDeepfakeDetector().to(device)

        # Frekans ekstraktörü: eğitim ile aynı seçim mantığı
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
        self.subtype_classifier = SubtypeClassifier()

        # Transform
        self.transform = transforms.Compose([
            transforms.Resize((model_cfg.IMG_SIZE, model_cfg.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        # Model yukle
        model_path = model_path or str(paths.BEST_MODEL_PATH)
        if Path(model_path).exists():
            ckpt = torch.load(model_path, map_location=device, weights_only=False)
            state = ckpt.get("model_state_dict", ckpt)
            self.model.load_state_dict(state)
            print(f"Model yuklendi: {model_path}")
        else:
            print("Model dosyasi bulunamadi, rastgele agirliklar")

        self.model.eval()

    def preprocess(self, image_input):
        """Goruntuy RGB + DWT + Mesh tensorlerine donustur."""
        if isinstance(image_input, (str, Path)):
            image = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, np.ndarray):
            image = Image.fromarray(image_input)
        else:
            image = image_input

        img_np = np.array(image.resize((model_cfg.IMG_SIZE, model_cfg.IMG_SIZE)))
        rgb_tensor = self.transform(image).unsqueeze(0)
        freq_tensor = torch.from_numpy(self.dwt(img_np)).unsqueeze(0).float()
        mesh_tensor = torch.from_numpy(self.mesh_extractor(img_np)).unsqueeze(0).float()
        return rgb_tensor.to(self.device), freq_tensor.to(self.device), mesh_tensor.to(self.device), img_np

    def predict(self, image_input, source_hint=None, uncertainty=False, n_mc_passes=20) -> dict:
        """
        Hiyerarşik tahmin: Binary → Alt-Tip.

        Args:
            image_input: Dosya yolu, numpy array veya PIL Image
            source_hint: 'upload', 'webcam', 'browser_extension', 'live'
                        Alt-tip belirleme icin kullanilir
            uncertainty: True ise MC Dropout

        Returns:
            dict: {
                "label": "REAL" | "FAKE",
                "real_prob": float,
                "fake_prob": float,
                "confidence": float,
                "fake_subtype": str | None,  — "digital" | "physical" | "ai_generated"
                "subtype_confidence": float | None,
                "subtype_method": str | None,
                "source_hint": str | None,
            }
        """
        rgb, freq, mesh, img_np = self.preprocess(image_input)

        # Asama 0: Fotograf on-filtresi (karikatur/cizim tespiti)
        is_photo, photo_score, photo_details = is_likely_photo(img_np)
        if not is_photo:
            return {
                "label": "NON-PHOTO",
                "verdict": "NON-PHOTO",
                "real_prob": 0.0,
                "fake_prob": 0.0,
                "confidence": round(1.0 - photo_score, 4),
                "source_hint": source_hint,
                "logits": [0.0, 0.0],
                "fake_subtype": None,
                "subtype_confidence": None,
                "subtype_method": None,
                "photo_filter": photo_details,
                "warning": "Bu gorsel bir fotograf degil (karikatur/cizim/ilustrasyon). "
                           "Deepfake analizi sadece fotograflar icin gecerlidir.",
            }

        # Asama 1: Binary siniflandirma
        with torch.no_grad():
            logits = self.model(rgb, freq, mesh)
            probs = torch.softmax(logits, dim=1)[0]

        real_prob = float(probs[0])
        fake_prob = float(probs[1])
        label = _classify_verdict(fake_prob)
        # Confidence: verdict'in kendi olasiligi
        if label == "FAKE":
            confidence = fake_prob
        elif label == "REAL":
            confidence = real_prob
        else:  # UNCERTAIN
            confidence = max(real_prob, fake_prob)

        result = {
            "label": label,
            "real_prob": real_prob,
            "fake_prob": fake_prob,
            "confidence": confidence,
            "source_hint": source_hint,
            "logits": logits[0].cpu().numpy().tolist(),
            # Alt-tip default
            "fake_subtype": None,
            "subtype_confidence": None,
            "subtype_method": None,
            "photo_filter": photo_details,
        }

        # Asama 2: FAKE ise alt-tip siniflandirma
        if label == "FAKE":
            subtype_result = self.subtype_classifier.classify(img_np, source_hint=source_hint)
            result["fake_subtype"] = subtype_result["subtype"]
            result["subtype_confidence"] = subtype_result["confidence"]
            result["subtype_method"] = subtype_result["method"]

        # MC Dropout uncertainty
        if uncertainty and HAS_MC:
            try:
                mc = MCDropoutPredictor(self.model, MCDropoutConfig(n_passes=n_mc_passes))
                mc_result = mc.predict_with_uncertainty(rgb, freq, mesh)
                result["mc_uncertainty"] = float(mc_result.get("predictive_entropy", 0))
                result["mc_std"] = float(mc_result.get("std_fake", 0))
                result["epistemic_uncertainty"] = float(mc_result.get("epistemic_uncertainty", 0))
            except Exception:
                pass

        return result

    def predict_batch(self, image_paths: list, source_hint=None, uncertainty=False) -> list:
        """Toplu tahmin."""
        return [self.predict(p, source_hint=source_hint, uncertainty=uncertainty) for p in image_paths]


# Singleton predictor
_predictor = None

def get_predictor() -> DeepfakePredictor:
    global _predictor
    if _predictor is None:
        _predictor = DeepfakePredictor()
    return _predictor


if __name__ == "__main__":
    predictor = DeepfakePredictor()
    # Rastgele giris testi
    dummy = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

    result = predictor.predict(dummy, source_hint="upload")
    print(f"Sonuc: {result['label']} (fake={result['fake_prob']:.3f})")
    if result["fake_subtype"]:
        print(f"Alt-tip: {result['fake_subtype']} (conf={result['subtype_confidence']:.2f})")
