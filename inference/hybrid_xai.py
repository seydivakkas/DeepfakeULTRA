"""
Deepfake Detection System v3 — Hybrid XAI
FastCAM (SMOE), Guided LIME, Combined Weighted Saliency.
"""
import numpy as np
from PIL import Image
from config import model_cfg, DEVICE
import torch

try:
    from lime import lime_image
    HAS_LIME = True
except ImportError:
    HAS_LIME = False


class FastCAM:
    """FastCAM — SMOE (Sparse Mixture of Experts) saliency."""
    def __init__(self, model, target_layer=None):
        self.model = model
        self.target_layer = target_layer or model.rgb_features[-1]
        self.activations = None
        self.target_layer.register_forward_hook(
            lambda m, i, o: setattr(self, 'activations', o.detach()))

    def generate(self, rgb, freq, mesh):
        self.model.eval()
        with torch.no_grad():
            self.model(rgb, freq, mesh)
        acts = self.activations[0]
        # SMOE: kanal bazlı sparse attention
        channel_importance = acts.mean(dim=(1, 2))
        top_k = max(1, int(len(channel_importance) * 0.3))
        _, top_indices = channel_importance.topk(top_k)
        selected = acts[top_indices]
        cam = selected.mean(dim=0).cpu().numpy()
        cam = np.maximum(cam, 0)
        cam = np.array(Image.fromarray(cam).resize((model_cfg.IMG_SIZE, model_cfg.IMG_SIZE)))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


class GuidedLIME:
    """Guided LIME — superpixel tabanlı açıklama."""
    def __init__(self, predictor):
        self.predictor = predictor

    def generate(self, image_np, num_samples=100):
        if not HAS_LIME:
            return np.zeros((model_cfg.IMG_SIZE, model_cfg.IMG_SIZE))
        explainer = lime_image.LimeImageExplainer()

        def predict_fn(images):
            results = []
            for img in images:
                r = self.predictor.predict(img.astype(np.uint8))
                results.append([r["real_prob"], r["fake_prob"]])
            return np.array(results)

        img_resized = np.array(Image.fromarray(image_np).resize(
            (model_cfg.IMG_SIZE, model_cfg.IMG_SIZE)))
        explanation = explainer.explain_instance(
            img_resized, predict_fn, top_labels=2, num_samples=num_samples)
        _, mask = explanation.get_image_and_mask(
            explanation.top_labels[0], positive_only=True, num_features=5)
        return mask.astype(np.float32)


class CombinedWeightedSaliency:
    """Çoklu XAI haritalarını ağırlıklı birleştir."""
    @staticmethod
    def combine(maps: dict, weights: dict = None) -> np.ndarray:
        if not maps:
            return np.zeros((model_cfg.IMG_SIZE, model_cfg.IMG_SIZE))
        if weights is None:
            weights = {k: 1.0 / len(maps) for k in maps}
        result = np.zeros_like(list(maps.values())[0], dtype=np.float32)
        total_w = 0
        for name, smap in maps.items():
            w = weights.get(name, 0)
            result += w * smap.astype(np.float32)
            total_w += w
        if total_w > 0:
            result /= total_w
        return (result - result.min()) / (result.max() - result.min() + 1e-8)
