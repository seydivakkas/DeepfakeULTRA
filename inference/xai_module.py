"""
Deepfake Detection System v3 — XAI Modülü
GradCAM++, EigenCAM, CounterfactualXAI, TemporalIntegratedGradients.
"""
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from config import model_cfg, DEVICE


class GradCAMPlusPlus:
    """GradCAM++ — gradient-weighted class activation mapping."""

    def __init__(self, model, target_layer=None):
        self.model = model
        self.target_layer = target_layer or model.rgb_features[-1]
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, rgb, freq, mesh, target_class=None):
        """GradCAM++ haritasi uret."""
        # NOT: model.train() analyze_engine tarafindan cagrilir (BiLSTM cudnn backward icin)
        rgb = rgb.detach().requires_grad_(True)
        logits = self.model(rgb, freq, mesh)

        if target_class is None:
            target_class = logits.argmax(dim=1)

        self.model.zero_grad()
        one_hot = torch.zeros_like(logits)
        one_hot[0, target_class] = 1
        logits.backward(gradient=one_hot, retain_graph=True)

        grads = self.gradients
        acts = self.activations

        # GradCAM++ ağırlıkları
        grads_power_2 = grads ** 2
        grads_power_3 = grads ** 3
        sum_acts = acts.sum(dim=(2, 3), keepdim=True) + 1e-7
        alpha = grads_power_2 / (2 * grads_power_2 + sum_acts * grads_power_3 + 1e-7)
        weights = (alpha * F.relu(grads)).sum(dim=(2, 3), keepdim=True)

        cam = (weights * acts).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(model_cfg.IMG_SIZE, model_cfg.IMG_SIZE),
                           mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


class EigenCAM:
    """EigenCAM — PCA tabanlı activation map."""

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
        acts = self.activations[0]  # (C, H, W)
        C, H, W = acts.shape
        reshaped = acts.reshape(C, H * W).cpu().numpy()
        U, S, Vt = np.linalg.svd(reshaped, full_matrices=False)
        cam = Vt[0].reshape(H, W)
        cam = np.maximum(cam, 0)
        cam = np.array(Image.fromarray(cam).resize((model_cfg.IMG_SIZE, model_cfg.IMG_SIZE)))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


class CounterfactualXAI:
    """Counterfactual analizi — minimal pertürbasyon ile karar değiştirme."""

    def __init__(self, model, steps=50, lr=0.01):
        self.model = model
        self.steps = steps
        self.lr = lr

    def generate(self, rgb, freq, mesh):
        """Counterfactual perturbasyon haritasi."""
        # NOT: model.train() analyze_engine tarafindan cagrilir (BiLSTM cudnn backward icin)
        with torch.no_grad():
            orig_logits = self.model(rgb, freq, mesh)
            orig_pred = orig_logits.argmax(dim=1).item()

        target_class = 1 - orig_pred
        perturbation = torch.zeros_like(rgb, requires_grad=True)
        optimizer = torch.optim.Adam([perturbation], lr=self.lr)

        flipped = False
        for step in range(self.steps):
            optimizer.zero_grad()
            perturbed = rgb + perturbation
            logits = self.model(perturbed, freq, mesh)
            target_loss = -logits[0, target_class]
            reg_loss = 0.01 * perturbation.abs().mean()
            loss = target_loss + reg_loss
            loss.backward()
            optimizer.step()

            pred = logits.argmax(dim=1).item()
            if pred == target_class:
                flipped = True
                break

        pert_map = perturbation.detach().abs().sum(dim=1).squeeze().cpu().numpy()
        pert_map = (pert_map - pert_map.min()) / (pert_map.max() - pert_map.min() + 1e-8)
        return pert_map, flipped


def generate_xai_maps(image_input, result, model=None):
    """Tüm XAI haritalarını üret."""
    from inference.predictor import get_predictor
    predictor = get_predictor()
    if model is None:
        model = predictor.model
    rgb, freq, mesh = predictor.preprocess(image_input)

    maps = {}
    try:
        gcam = GradCAMPlusPlus(model)
        maps["gradcam_pp"] = gcam.generate(rgb, freq, mesh)
    except Exception as e:
        print(f"⚠️ GradCAM++ hatası: {e}")

    try:
        eigen = EigenCAM(model)
        maps["eigen_cam"] = eigen.generate(rgb, freq, mesh)
    except Exception as e:
        print(f"⚠️ EigenCAM hatası: {e}")

    try:
        cf = CounterfactualXAI(model)
        cf_map, flipped = cf.generate(rgb, freq, mesh)
        maps["counterfactual"] = cf_map
        maps["cf_flipped"] = flipped
    except Exception as e:
        print(f"⚠️ Counterfactual hatası: {e}")

    return maps
