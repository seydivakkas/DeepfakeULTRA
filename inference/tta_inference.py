"""
Deepfake Detection System v4 — Test-Time Augmentation (TTA)
8 deterministik augmentasyon ile ensemble tahmin.

Tekli ve batch modu destekler:
  - predict_tta(image)           : tek goruntu
  - predict_batch_tta(loader)    : DataLoader uzerinde batch TTA
"""
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as TF
from config import model_cfg, DEVICE


# ================================================================
# DETERMINISTiK TTA AUGMENTASYONLARI
# ================================================================
class DeterministicTTA:
    """
    15 deterministik augmentasyon — her calistirmada ayni sonuc.
    Sadece RGB'ye uygulanir. DWT ve Mesh tensorlari sabittir.
    Slider 5-15: kac augmentasyon kullanilacagini kontrol eder.
    """

    def __init__(self, img_size: int = model_cfg.IMG_SIZE):
        self.img_size = img_size
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def get_augmentations(self):
        """15 deterministik augmentasyon fonksiyonu dondur."""
        return [
            self._original,            # 1. Orijinal
            self._hflip,               # 2. Yatay aynalama
            self._rotate_pos5,         # 3. +5 derece rotasyon
            self._rotate_neg5,         # 4. -5 derece rotasyon
            self._brightness_up,       # 5. Parlaklik +20%
            self._contrast_up,         # 6. Kontrast +20%
            self._gaussian_blur,       # 7. Gaussian blur (sigma=1)
            self._scale_crop,          # 8. %110 olcek + krop
            self._rotate_pos10,        # 9. +10 derece rotasyon
            self._rotate_neg10,        # 10. -10 derece rotasyon
            self._brightness_down,     # 11. Parlaklik -20%
            self._contrast_down,       # 12. Kontrast -20%
            self._saturation_up,       # 13. Doygunluk +30%
            self._sharpness_up,        # 14. Keskinlik +50%
            self._color_jitter,        # 15. Renk bozulmasi
        ]

    def _to_tensor(self, img: Image.Image) -> torch.Tensor:
        """PIL Image -> normalized tensor."""
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        t = TF.to_tensor(img)
        return self.normalize(t)

    def _original(self, img: Image.Image) -> torch.Tensor:
        return self._to_tensor(img)

    def _hflip(self, img: Image.Image) -> torch.Tensor:
        return self._to_tensor(TF.hflip(img))

    def _rotate_pos5(self, img: Image.Image) -> torch.Tensor:
        return self._to_tensor(TF.rotate(img, 5))

    def _rotate_neg5(self, img: Image.Image) -> torch.Tensor:
        return self._to_tensor(TF.rotate(img, -5))

    def _brightness_up(self, img: Image.Image) -> torch.Tensor:
        return self._to_tensor(TF.adjust_brightness(img, 1.2))

    def _contrast_up(self, img: Image.Image) -> torch.Tensor:
        return self._to_tensor(TF.adjust_contrast(img, 1.2))

    def _gaussian_blur(self, img: Image.Image) -> torch.Tensor:
        img_resized = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        blurred = TF.gaussian_blur(TF.to_tensor(img_resized), kernel_size=3, sigma=1.0)
        return self.normalize(blurred)

    def _scale_crop(self, img: Image.Image) -> torch.Tensor:
        scale = int(self.img_size * 1.1)
        img_scaled = img.resize((scale, scale), Image.BILINEAR)
        img_cropped = TF.center_crop(img_scaled, self.img_size)
        return self._to_tensor(img_cropped)

    # ── 9-15: Genisletilmis augmentasyonlar ──

    def _rotate_pos10(self, img: Image.Image) -> torch.Tensor:
        return self._to_tensor(TF.rotate(img, 10))

    def _rotate_neg10(self, img: Image.Image) -> torch.Tensor:
        return self._to_tensor(TF.rotate(img, -10))

    def _brightness_down(self, img: Image.Image) -> torch.Tensor:
        return self._to_tensor(TF.adjust_brightness(img, 0.8))

    def _contrast_down(self, img: Image.Image) -> torch.Tensor:
        return self._to_tensor(TF.adjust_contrast(img, 0.8))

    def _saturation_up(self, img: Image.Image) -> torch.Tensor:
        return self._to_tensor(TF.adjust_saturation(img, 1.3))

    def _sharpness_up(self, img: Image.Image) -> torch.Tensor:
        return self._to_tensor(TF.adjust_sharpness(img, 1.5))

    def _color_jitter(self, img: Image.Image) -> torch.Tensor:
        """Hafif renk perturbasyonu (deterministik)."""
        img = TF.adjust_hue(img, 0.02)
        img = TF.adjust_gamma(img, 1.1)
        return self._to_tensor(img)


# ================================================================
# TTA PREDICTOR (tekli goruntu)
# ================================================================
class TTAPredictor:
    """TTA ile guclendirilmis tahmin — tekli goruntu modu."""

    def __init__(self, predictor=None, n_augmentations: int = None):
        self.predictor = predictor
        self.n_aug = n_augmentations or model_cfg.TTA_AUGMENTATIONS
        self.tta = DeterministicTTA()
        self.augmentations = self.tta.get_augmentations()

    def predict_tta(self, image_input) -> dict:
        """TTA ensemble tahmini — tekli goruntu."""
        if self.predictor is None:
            from inference.predictor import DeepfakePredictor
            self.predictor = DeepfakePredictor()

        if isinstance(image_input, str):
            image = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, np.ndarray):
            image = Image.fromarray(image_input)
        else:
            image = image_input

        img_np = np.array(image.resize((model_cfg.IMG_SIZE, model_cfg.IMG_SIZE)))

        # Predictor'ın kendi dwt'sini kullan (HybridFreq veya MultiScaleDWT — eğitimle aynı)
        from core.data_pipeline import FaceMeshExtractor
        freq = torch.from_numpy(self.predictor.dwt(img_np)).unsqueeze(0).float().to(DEVICE)
        mesh_ext = FaceMeshExtractor()
        mesh = torch.from_numpy(mesh_ext(img_np)).unsqueeze(0).float().to(DEVICE)

        all_probs = []
        n = min(self.n_aug, len(self.augmentations))
        model = self.predictor.model
        model.eval()

        with torch.no_grad():
            for aug_fn in self.augmentations[:n]:
                rgb = aug_fn(image).unsqueeze(0).to(DEVICE)
                logits = model(rgb, freq, mesh)
                probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
                all_probs.append(probs)

        all_probs = np.array(all_probs)
        mean_probs = all_probs.mean(axis=0)
        std_probs = all_probs.std(axis=0)

        fake_prob = float(mean_probs[1])
        from inference.predictor import _classify_verdict
        return {
            "label": _classify_verdict(fake_prob),
            "fake_prob": fake_prob,
            "real_prob": float(mean_probs[0]),
            "confidence": float(max(mean_probs)),
            "std": float(std_probs[1]),
            "n_augmentations": n,
            "individual_predictions": all_probs.tolist(),
        }


# ================================================================
# BATCH TTA (DataLoader uzerinde)
# ================================================================
def batch_tta_evaluate(model, test_loader, device=DEVICE, n_augmentations: int = 8):
    """
    DataLoader uzerinde batch TTA degerlendirmesi.

    Her batch icin:
    1. DWT ve Mesh tensorlari sabittir (augmentasyon uygulanmaz)
    2. RGB tensorune 8 deterministik augmentasyon uygula
    3. 8 olasilik ortalamasi = final tahmin

    Args:
        model: DualPathDeepfakeDetector (eval modunda)
        test_loader: DataLoader (5'li tuple: rgb, freq, mesh, labels, tags)
        device: torch.device
        n_augmentations: Kac augmentasyon kullanilacak (maks 8)

    Returns:
        dict: {labels, predictions, probabilities, per_sample_std}
    """
    tta = DeterministicTTA()
    aug_fns = tta.get_augmentations()[:n_augmentations]

    model.eval()
    all_labels = []
    all_mean_probs = []
    all_stds = []

    with torch.no_grad():
        for batch in test_loader:
            rgb, freq, mesh, labels, source_tags = batch
            freq = freq.to(device)
            mesh = mesh.to(device)
            labels_np = labels.numpy()
            batch_size = rgb.size(0)

            # Her augmentasyondan olasilik topla
            batch_aug_probs = []  # shape sonra: (n_aug, batch, n_classes)

            for aug_fn in aug_fns:
                # RGB tensorunu CPU'da denormalize → PIL → augmentasyon → tekrar tensor
                aug_rgb = _apply_augmentation_to_batch(rgb, aug_fn, device)
                logits = model(aug_rgb, freq, mesh)
                probs = torch.softmax(logits, dim=1).cpu().numpy()  # (batch, n_classes)
                batch_aug_probs.append(probs)

            # (n_aug, batch, n_classes) -> mean/std uzerinden
            stacked = np.stack(batch_aug_probs, axis=0)  # (n_aug, batch, n_classes)
            mean_probs = stacked.mean(axis=0)  # (batch, n_classes)
            std_probs = stacked.std(axis=0)    # (batch, n_classes)

            all_labels.extend(labels_np)
            all_mean_probs.extend(mean_probs)
            all_stds.extend(std_probs[:, 1])  # FAKE olasiligin std'si

    all_labels = np.array(all_labels)
    all_mean_probs = np.array(all_mean_probs)
    all_preds = all_mean_probs.argmax(axis=1)
    all_stds = np.array(all_stds)

    return {
        "labels": all_labels,
        "predictions": all_preds,
        "probabilities": all_mean_probs,
        "per_sample_std": all_stds,
    }


def _apply_augmentation_to_batch(rgb_batch: torch.Tensor, aug_fn, device) -> torch.Tensor:
    """
    Batch RGB tensorune tek bir TTA augmentasyonu uygula.

    Akis:
    1. Normalize edilmis tensoru denormalize et
    2. PIL Image'a donustur
    3. aug_fn uygula (icerisinde normalize var)
    4. GPU'ya gonder

    Args:
        rgb_batch: (B, 3, H, W) — normalized tensor
        aug_fn: DeterministicTTA metodu
        device: torch.device

    Returns:
        (B, 3, H, W) — augmented + normalized tensor
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    # Denormalize (CPU'da)
    denorm = rgb_batch * std + mean
    denorm = denorm.clamp(0, 1)

    augmented = []
    for i in range(denorm.size(0)):
        # Tensor → PIL Image
        img_np = (denorm[i].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_np)
        # Augmentasyon uygula (aug_fn normalize eder)
        aug_tensor = aug_fn(pil_img)
        augmented.append(aug_tensor)

    return torch.stack(augmented).to(device)
