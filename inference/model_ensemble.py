"""
Deepfake Detection System v4 — Checkpoint Ensemble
Birden fazla checkpoint'u VRAM-guvenli sekilde birlestir.

Modlar:
  - predict()              : tum modelleri bellekte tut (kucuk modeller)
  - predict_sequential()   : sirayla yukle-tahmin yap-sil (VRAM guvenli)
  - evaluate_sequential()  : DataLoader uzerinde sirayla degerlendirme
"""
import torch
import gc
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict
from config import DEVICE, model_cfg, paths
from core.dual_mobilenetv3 import DualPathDeepfakeDetector


# ================================================================
# CHECKPOINT ENSEMBLE — BELLEKTE TUTMA MODU (kucuk modeller icin)
# ================================================================
class ModelEnsemble:
    """Birden fazla checkpoint'i ensemble olarak kullan."""

    def __init__(self, model_paths: list, weights: Optional[List[float]] = None):
        """
        Args:
            model_paths: Checkpoint dosya yollari
            weights: Her model icin agirlik (varsayilan: esit)
        """
        self.model_paths = [Path(p) for p in model_paths]
        self.weights = weights

        # Agirliklari normalize et
        if self.weights:
            total = sum(self.weights)
            self.weights = [w / total for w in self.weights]
        else:
            n = len(model_paths)
            self.weights = [1.0 / n] * n

        # Modelleri yukle
        self.models = []
        for p in self.model_paths:
            m = DualPathDeepfakeDetector().to(DEVICE)
            ckpt = torch.load(str(p), map_location=DEVICE, weights_only=False)
            m.load_state_dict(ckpt.get("model_state_dict", ckpt))
            m.eval()
            self.models.append(m)

    def predict(self, rgb, freq, mesh) -> torch.Tensor:
        """Agirlikli olasilik ortalamasi (tum modeller bellekte)."""
        all_probs = []
        with torch.no_grad():
            for m in self.models:
                logits = m(rgb, freq, mesh)
                probs = torch.softmax(logits, dim=1)
                all_probs.append(probs)

        # Agirlikli ortalama
        weighted = torch.zeros_like(all_probs[0])
        for probs, w in zip(all_probs, self.weights):
            weighted += w * probs

        return weighted


# ================================================================
# SIRASAL ENSEMBLE — VRAM GUVENLI MOD
# ================================================================
class SequentialEnsemble:
    """
    Checkpoint'lari sirayla yukle → tahmin yap → bellekten sil.
    8 GB VRAM sinirinda guvenle calisir.
    """

    def __init__(
        self,
        model_paths: List[str],
        weights: Optional[List[float]] = None,
        device=DEVICE
    ):
        self.model_paths = [Path(p) for p in model_paths]
        self.device = device

        # Agirliklari normalize et
        if weights:
            total = sum(weights)
            self.weights = [w / total for w in weights]
        else:
            n = len(model_paths)
            self.weights = [1.0 / n] * n

        # Checkpoint bilgileri
        self.checkpoint_info = []
        for p in self.model_paths:
            info = {"path": str(p), "name": p.stem}
            if p.exists():
                ckpt = torch.load(str(p), map_location="cpu", weights_only=False)
                info["epoch"] = ckpt.get("epoch", "?")
                info["val_auc"] = ckpt.get("val_auc", "?")
            self.checkpoint_info.append(info)

    def _load_model(self, path: Path) -> DualPathDeepfakeDetector:
        """Tek bir modeli GPU'ya yukle."""
        model = DualPathDeepfakeDetector().to(self.device)
        ckpt = torch.load(str(path), map_location=self.device, weights_only=False)
        model.load_state_dict(ckpt.get("model_state_dict", ckpt))
        model.eval()
        return model

    def _unload_model(self, model):
        """Modeli bellekten sil ve GPU'yu temizle."""
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def evaluate_sequential(self, test_loader, verbose: bool = True) -> Dict:
        """
        Test DataLoader uzerinde sirasal ensemble degerlendirmesi.

        Akis:
        1. Her checkpoint icin:
           a. Modeli GPU'ya yukle
           b. Tum test setini forward-pass yap
           c. Olasiliklari kaydet
           d. Modeli bellekten sil
        2. Tum checkpoint olasiliklerinin agirlikli ortalamasini al
        3. Final tahminleri dondur

        Returns:
            dict: {labels, predictions, probabilities, per_model_probs}
        """
        n_models = len(self.model_paths)
        all_labels = None
        per_model_probs = []  # her model icin: (N, n_classes)

        for idx, (path, weight) in enumerate(zip(self.model_paths, self.weights)):
            if verbose:
                info = self.checkpoint_info[idx]
                val_auc = info.get('val_auc', '?')
                auc_str = f"{val_auc:.4f}" if isinstance(val_auc, float) else "?"
                print(f"    [{idx+1}/{n_models}] {info['name']} "
                      f"(epoch={info.get('epoch','?')}, "
                      f"val_auc={auc_str}, w={weight:.3f})")

            model = self._load_model(path)
            model_probs = []
            model_labels = []

            with torch.no_grad():
                for batch in test_loader:
                    rgb, freq, mesh, labels, source_tags = batch
                    rgb = rgb.to(self.device)
                    freq = freq.to(self.device)
                    mesh = mesh.to(self.device)

                    logits = model(rgb, freq, mesh)
                    probs = torch.softmax(logits, dim=1).cpu().numpy()
                    model_probs.append(probs)
                    model_labels.append(labels.numpy())

            model_probs = np.concatenate(model_probs, axis=0)
            per_model_probs.append(model_probs)

            if all_labels is None:
                all_labels = np.concatenate(model_labels, axis=0)

            self._unload_model(model)

        # Agirlikli ortalama
        weighted_probs = np.zeros_like(per_model_probs[0])
        for probs, weight in zip(per_model_probs, self.weights):
            weighted_probs += weight * probs

        predictions = weighted_probs.argmax(axis=1)

        return {
            "labels": all_labels,
            "predictions": predictions,
            "probabilities": weighted_probs,
            "per_model_probs": per_model_probs,
        }


# ================================================================
# TTA + ENSEMBLE KOMBINE MOD
# ================================================================
class TTAEnsemble:
    """
    TTA + Checkpoint Ensemble kombinasyonu.
    Her checkpoint icin TTA uygula, sonra agirlikli birlestir.
    VRAM guvenli: sirasal yukle-tahmin-sil.
    """

    def __init__(
        self,
        model_paths: List[str],
        weights: Optional[List[float]] = None,
        n_augmentations: int = 8,
        device=DEVICE
    ):
        self.model_paths = [Path(p) for p in model_paths]
        self.device = device
        self.n_aug = n_augmentations

        if weights:
            total = sum(weights)
            self.weights = [w / total for w in weights]
        else:
            n = len(model_paths)
            self.weights = [1.0 / n] * n

        self.checkpoint_info = []
        for p in self.model_paths:
            info = {"path": str(p), "name": p.stem}
            if p.exists():
                ckpt = torch.load(str(p), map_location="cpu", weights_only=False)
                info["epoch"] = ckpt.get("epoch", "?")
                info["val_auc"] = ckpt.get("val_auc", "?")
            self.checkpoint_info.append(info)

    def evaluate_tta_ensemble(self, test_loader, verbose: bool = True) -> Dict:
        """
        TTA + Ensemble degerlendirmesi.

        Her checkpoint icin:
          1. Modeli yukle
          2. Batch TTA uygula (8 augmentasyon)
          3. TTA olasilik ortalamasini kaydet
          4. Modeli sil

        Sonra: tum checkpoint TTA ortalamalarini agirlikli birlestir.

        Returns:
            dict: {labels, predictions, probabilities, per_model_tta_probs}
        """
        from inference.tta_inference import batch_tta_evaluate

        n_models = len(self.model_paths)
        all_labels = None
        per_model_tta_probs = []

        for idx, (path, weight) in enumerate(zip(self.model_paths, self.weights)):
            if verbose:
                info = self.checkpoint_info[idx]
                print(f"    [{idx+1}/{n_models}] TTA({self.n_aug}) x {info['name']} "
                      f"(w={weight:.3f})")

            model = DualPathDeepfakeDetector().to(self.device)
            ckpt = torch.load(str(path), map_location=self.device, weights_only=False)
            model.load_state_dict(ckpt.get("model_state_dict", ckpt))
            model.eval()

            tta_result = batch_tta_evaluate(
                model, test_loader, self.device, self.n_aug
            )

            per_model_tta_probs.append(tta_result["probabilities"])
            if all_labels is None:
                all_labels = tta_result["labels"]

            # Bellegi temizle
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Agirlikli ortalama
        weighted_probs = np.zeros_like(per_model_tta_probs[0])
        for probs, weight in zip(per_model_tta_probs, self.weights):
            weighted_probs += weight * probs

        predictions = weighted_probs.argmax(axis=1)

        return {
            "labels": all_labels,
            "predictions": predictions,
            "probabilities": weighted_probs,
            "per_model_tta_probs": per_model_tta_probs,
        }


# ================================================================
# VARSAYILAN ENSEMBLE KONFIGURASYONU
# ================================================================
def get_default_ensemble_config() -> Dict:
    """
    Run #4 Binary checkpoint'larindan varsayilan ensemble yapilandirmasi.

    Strateji: En yakin 3 checkpoint (E25 best + E30 + E20).
    Erken epoch'lar (E10, E15) dahil edilmiyor cunku karar sinirlari
    farkli ve ensemble'in performansini dusuruyor.

    Agirliklar: Egitimden bilinen val AUC degerleri.
    """
    models_dir = paths.MODEL_DIR

    # En iyi 3 checkpoint — yakin epoch'lar daha guvenilir ensemble uretir
    checkpoints = [
        {"path": models_dir / "best_run4_binary.pth", "val_auc": 0.9972},
        {"path": models_dir / "checkpoint_epoch30.pth", "val_auc": 0.9971},
        {"path": models_dir / "checkpoint_epoch20.pth", "val_auc": 0.9963},
    ]

    # Sadece mevcut dosyalari filtrele
    valid = [c for c in checkpoints if c["path"].exists()]

    paths_list = [str(c["path"]) for c in valid]
    weights = [c["val_auc"] for c in valid]

    return {
        "paths": paths_list,
        "weights": weights,
        "n_models": len(valid),
    }

