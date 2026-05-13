"""
temperature_scaling.py — Temperature Scaling Kalibrasyonu
Deepfake Detection System v3

Model softmax çıktılarını güvenilir olasılıklara dönüştürür.
Post-hoc kalibrasyon: model ağırlıkları değişmez, yalnızca T parametresi öğrenilir.
ECE (Expected Calibration Error) hesaplama ve MLflow loglama destekli.
EU AI Act Madde 13 (Şeffaflık) uyumluluğu için kritik.

Referans: Guo et al. (2017) "On Calibration of Modern Neural Networks"
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import mlflow
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ECE Hesaplayıcı
# ---------------------------------------------------------------------------

def expected_calibration_error(
    confidences: np.ndarray,
    correctness: np.ndarray,
    n_bins: int = 15,
) -> float:
    """
    Expected Calibration Error (ECE) hesaplar.

    ECE = Σ (|B_m| / n) × |acc(B_m) − conf(B_m)|

    Args:
        confidences: Model'in max olasılık değerleri, shape (N,)
        correctness: 1.0 doğru, 0.0 yanlış, shape (N,)
        n_bins: Güven aralığı sayısı
    Returns:
        ECE skoru [0, 1] — 0 mükemmel kalibrasyon
    """
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(confidences)

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        bin_acc = correctness[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)

    return float(ece)


def maximum_calibration_error(
    confidences: np.ndarray,
    correctness: np.ndarray,
    n_bins: int = 15,
) -> float:
    """MCE: En kötü bin'deki kalibrasyon hatası."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    mce = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        bin_acc = correctness[mask].mean()
        bin_conf = confidences[mask].mean()
        mce = max(mce, abs(bin_acc - bin_conf))
    return float(mce)


# ---------------------------------------------------------------------------
# Temperature Scaling Wrapper
# ---------------------------------------------------------------------------

class TemperatureScaling(nn.Module):
    """
    Modelin logit çıktısını T ile bölerek softmax kalibrasyonu yapar.

    model_with_ts = TemperatureScaling(base_model)
    model_with_ts.calibrate(val_loader)
    probs = model_with_ts(image_tensor)  # kalibre edilmiş olasılıklar
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        # T > 1 → daha yumuşak olasılıklar (under-confident modeller için düşürür)
        # T < 1 → daha keskin olasılıklar
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.model(x)
        return logits / self.temperature.clamp(min=0.05)

    @property
    def T(self) -> float:
        return self.temperature.item()

    # ------------------------------------------------------------------
    # Kalibrasyon
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _collect_logits_labels(
        self, dataloader: DataLoader, device: torch.device
    ):
        """Validation seti üzerinde logit ve label toplar."""
        self.model.eval()
        all_logits, all_labels = [], []
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                inputs, labels = batch[0].to(device), batch[1].to(device)
            else:
                raise ValueError("DataLoader (input, label) batch'i döndürmelidir.")
            logits = self.model(inputs)
            all_logits.append(logits)
            all_labels.append(labels)
        return torch.cat(all_logits), torch.cat(all_labels)

    def calibrate(
        self,
        val_loader: DataLoader,
        device: Optional[torch.device] = None,
        max_iter: int = 1000,
        lr: float = 0.01,
        log_mlflow: bool = True,
    ) -> dict[str, float]:
        """
        Validation seti üzerinde T parametresini optimize eder.
        NLL (Negative Log-Likelihood) minimizasyonu.

        Returns:
            Kalibrasyon sonuçları: {T, ece_before, ece_after, mce_after}
        """
        device = device or next(self.model.parameters()).device
        self.to(device)

        logits, labels = self._collect_logits_labels(val_loader, device)

        # Kalibrasyon öncesi ECE
        probs_before = torch.softmax(logits, dim=1).cpu().numpy()
        confs_before = probs_before.max(axis=1)
        correct_before = (
            (logits.argmax(dim=1) == labels).cpu().numpy().astype(float)
        )
        ece_before = expected_calibration_error(confs_before, correct_before)

        # T'yi optimize et (NLL minimize)
        nll_criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS(
            [self.temperature], lr=lr, max_iter=max_iter
        )

        def _eval():
            optimizer.zero_grad()
            scaled_logits = logits / self.temperature.clamp(min=0.05)
            loss = nll_criterion(scaled_logits, labels)
            loss.backward()
            return loss

        optimizer.step(_eval)
        T_opt = self.T

        # Kalibrasyon sonrası ECE
        with torch.no_grad():
            scaled_logits = logits / self.temperature.clamp(min=0.05)
            probs_after = torch.softmax(scaled_logits, dim=1).cpu().numpy()
        confs_after = probs_after.max(axis=1)
        correct_after = correct_before.copy()
        ece_after = expected_calibration_error(confs_after, correct_after)
        mce_after = maximum_calibration_error(confs_after, correct_after)

        reduction_pct = (ece_before - ece_after) / (ece_before + 1e-10) * 100

        logger.info(
            "Kalibrasyon tamamlandı | T=%.4f | ECE %.4f → %.4f (Δ=%.1f%%)",
            T_opt,
            ece_before,
            ece_after,
            reduction_pct,
        )

        result = {
            "T": T_opt,
            "ece_before": ece_before,
            "ece_after": ece_after,
            "mce_after": mce_after,
            "ece_reduction_pct": reduction_pct,
        }

        if log_mlflow:
            try:
                mlflow.log_metrics(
                    {
                        "calibration/T": T_opt,
                        "calibration/ece_before": ece_before,
                        "calibration/ece_after": ece_after,
                        "calibration/mce_after": mce_after,
                        "calibration/ece_reduction_pct": reduction_pct,
                    }
                )
            except Exception as e:
                logger.debug("MLflow loglama atlandı: %s", e)

        return result

    # ------------------------------------------------------------------
    # Kaydetme / Yükleme
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        torch.save({"temperature": self.temperature}, path)
        logger.info("Kalibrasyon T değeri kaydedildi: %s", path)

    def load(self, path: str | Path) -> None:
        ckpt = torch.load(path, map_location="cpu")
        self.temperature = nn.Parameter(ckpt["temperature"])
        logger.info("Kalibrasyon T değeri yüklendi: T=%.4f", self.T)


# ---------------------------------------------------------------------------
# Convenience: Kalibre Edilmiş Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def calibrated_predict(
    model_with_ts: TemperatureScaling,
    image_tensor: torch.Tensor,
    device: torch.device,
) -> dict[str, float]:
    """
    Tek görüntü için kalibre edilmiş olasılık tahmini.

    Returns:
        {"fake_prob": float, "real_prob": float, "T": float}
    """
    model_with_ts.eval()
    x = image_tensor.unsqueeze(0).to(device)
    logits = model_with_ts(x)
    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    return {
        "real_prob": float(probs[0]),
        "fake_prob": float(probs[1]),
        "T": model_with_ts.T,
    }
