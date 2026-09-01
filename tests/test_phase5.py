"""Phase 5 tests for inference and explainability contracts."""

import numpy as np

from config import model_cfg


def _dummy_image(size=224):
    return np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)


def test_inference_pipeline():
    from inference.predictor import DeepfakePredictor

    predictor = DeepfakePredictor()
    result = predictor.predict(_dummy_image())
    assert result["label"] in ("FAKE", "REAL", "NON-PHOTO")
    assert 0.0 <= result["confidence"] <= 1.0


def test_preprocess_shapes():
    from inference.predictor import DeepfakePredictor

    predictor = DeepfakePredictor()
    rgb, freq, mesh, image_np = predictor.preprocess(_dummy_image(300))

    assert rgb.shape == (1, 3, model_cfg.IMG_SIZE, model_cfg.IMG_SIZE)
    assert freq.shape[0] == 1
    assert mesh.shape[0] == 1
    assert image_np.shape == (model_cfg.IMG_SIZE, model_cfg.IMG_SIZE, 3)


def test_gradcam():
    from inference.predictor import DeepfakePredictor
    from inference.xai_module import GradCAMPlusPlus

    predictor = DeepfakePredictor()
    rgb, freq, mesh, _ = predictor.preprocess(_dummy_image())
    cam = GradCAMPlusPlus(predictor.model).generate(rgb, freq, mesh)

    assert cam.shape == (model_cfg.IMG_SIZE, model_cfg.IMG_SIZE)


def test_tta():
    from inference.predictor import DeepfakePredictor
    from inference.tta_inference import TTAPredictor

    predictor = DeepfakePredictor()
    tta = TTAPredictor(predictor, n_augmentations=3)
    result = tta.predict_tta(_dummy_image())

    assert result["label"] in ("FAKE", "REAL", "NON-PHOTO")
    assert result["std"] >= 0.0


def test_xai_quality():
    from inference.explainability_report import compute_faithfulness_score

    score = compute_faithfulness_score(0.7, 0.8, True)
    assert 0 <= score <= 1
