"""Deepfake v3 — Faz 5 Test: Çıkarım & XAI."""
import sys, os
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import model_cfg, DEVICE

def test_inference_pipeline():
    from inference.predictor import DeepfakePredictor
    predictor = DeepfakePredictor()
    dummy = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    result = predictor.predict(dummy)
    assert result["label"] in ("FAKE", "REAL")
    print(f"  ✓ Inference: {result['label']} ({result['confidence']:.3f})")

def test_preprocess_shapes():
    from inference.predictor import DeepfakePredictor
    predictor = DeepfakePredictor()
    rgb, freq, mesh = predictor.preprocess(np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8))
    assert rgb.shape == (1, 3, 224, 224)
    print(f"  ✓ Preprocess: rgb={rgb.shape}")

def test_gradcam():
    from inference.predictor import DeepfakePredictor
    from inference.xai_module import GradCAMPlusPlus
    predictor = DeepfakePredictor()
    rgb, freq, mesh = predictor.preprocess(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    cam = GradCAMPlusPlus(predictor.model).generate(rgb, freq, mesh)
    assert cam.shape == (model_cfg.IMG_SIZE, model_cfg.IMG_SIZE)
    print(f"  ✓ GradCAM++: {cam.shape}")

def test_tta():
    from inference.predictor import DeepfakePredictor
    from inference.tta_inference import TTAPredictor
    predictor = DeepfakePredictor()
    tta = TTAPredictor(predictor, n_augmentations=3)
    result = tta.predict_tta(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    assert result["label"] in ("FAKE", "REAL")
    print(f"  ✓ TTA: {result['label']} (std={result['std']:.4f})")

def test_xai_quality():
    from inference.explainability_report import compute_faithfulness_score, saliency_quality_metrics
    score = compute_faithfulness_score(0.7, 0.8, True)
    assert 0 <= score <= 1
    print(f"  ✓ XAI quality: faithfulness={score:.3f}")

if __name__ == "__main__":
    print("=== Faz 5: Çıkarım & XAI ===")
    for fn in [test_inference_pipeline, test_preprocess_shapes, test_gradcam, test_tta, test_xai_quality]:
        try: fn()
        except Exception as e:
            if "mediapipe" in str(e).lower():
                print(f"  ⚠ {fn.__name__}: MediaPipe uyumsuz (graceful degradation)")
            else:
                print(f"  ✗ {fn.__name__}: {e}"); sys.exit(1)
    print("✅ Faz 5 tamamlandı")
