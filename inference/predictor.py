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
    """Binary karar: fake_prob >= FAKE_THRESHOLD ise FAKE, degilse REAL."""
    if fake_prob >= model_cfg.FAKE_THRESHOLD:
        return "FAKE"
    return "REAL"

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

        # Asama 1: Binary siniflandirma
        with torch.no_grad():
            logits = self.model(rgb, freq, mesh)
            probs = torch.softmax(logits, dim=1)[0]

        real_prob = float(probs[0])
        fake_prob = float(probs[1])
        label = _classify_verdict(fake_prob)
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
