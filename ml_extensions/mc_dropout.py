"""
mc_dropout.py — MC Dropout Belirsizlik Tahmini (Genişletilmiş)
Deepfake Detection System v3

Inference sırasında Dropout katmanları aktif tutularak N forward pass çalıştırır.
Epistemic (model) belirsizliği aleatorik belirsizlikten ayrıştırır.
Mevcut inference.py modülünü genişletir — drop-in replacement.

Referans: Gal & Ghahramani (2016) "Dropout as a Bayesian Approximation"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class UncertaintyResult:
    """MC Dropout çıktısı için konteyner."""
    mean_probs: np.ndarray          # shape: (n_classes,) — ortalama olasılık
    std_probs: np.ndarray           # shape: (n_classes,) — standart sapma
    epistemic: float                # Model belirsizliği (bilgi eksikliği)
    aleatoric: float                # Veri belirsizliği (gürültü)
    total_uncertainty: float
    n_passes: int
    extra: dict = field(default_factory=dict)

    @property
    def prediction(self) -> int:
        return int(np.argmax(self.mean_probs))

    @property
    def confidence(self) -> float:
        return float(self.mean_probs.max())

    @property
    def is_uncertain(self, threshold: float = 0.1) -> bool:
        return self.epistemic > threshold


# ---------------------------------------------------------------------------
# MC Dropout Etkinleştirme
# ---------------------------------------------------------------------------

def enable_dropout_only(model: nn.Module) -> None:
    """
    Modeli eval moduna alır ancak Dropout katmanlarını train() konumunda bırakır.
    BatchNorm katmanları eval modunda kalır (running statistics kullanır).
    """
    model.eval()
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.AlphaDropout)):
            module.train()


# ---------------------------------------------------------------------------
# Belirsizlik Ayrıştırma
# ---------------------------------------------------------------------------

def decompose_uncertainty(sample_probs: np.ndarray) -> tuple[float, float]:
    """
    Epistemic ve aleatorik belirsizliği ayrıştırır.

    sample_probs: shape (N_passes, n_classes)

    Toplam Belirsizlik  = H[E_θ[p(y|x, θ)]]   (ortalama üzerinden entropi)
    Aleatorik           = E_θ[H[p(y|x, θ)]]    (entropi'nin beklentisi)
    Epistemic           = Toplam - Aleatorik    (mutual information)
    """
    eps = 1e-10
    mean_p = sample_probs.mean(axis=0)                    # (n_classes,)

    # Toplam belirsizlik: ortalama olasılığın entropisi
    H_mean = -np.sum(mean_p * np.log2(mean_p + eps))

    # Aleatorik belirsizlik: her pass'in entropisi'nin ortalaması
    H_samples = -np.sum(sample_probs * np.log2(sample_probs + eps), axis=1)  # (N,)
    E_H = H_samples.mean()

    epistemic = max(0.0, H_mean - E_H)  # mutual information
    aleatoric = E_H

    return float(epistemic), float(aleatoric)


# ---------------------------------------------------------------------------
# Ana MC Dropout Sınıfı
# ---------------------------------------------------------------------------

class MCDropoutPredictor:
    """
    MC Dropout tabanlı Bayesianesque belirsizlik tahmini.

    Mevcut inference.py'deki DeepfakeInference sınıfını genişletir;
    bağımsız olarak veya wrapper olarak kullanılabilir.

    Kullanım:
        predictor = MCDropoutPredictor(model, n_passes=30)
        result = predictor.predict(image_tensor)
        print(f"Epistemic: {result.epistemic:.4f}")
        print(f"Aleatoric: {result.aleatoric:.4f}")
    """

    def __init__(
        self,
        model: nn.Module,
        n_passes: int = 30,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model
        self.n_passes = n_passes
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)

    @torch.no_grad()
    def predict(self, image_tensor: torch.Tensor) -> UncertaintyResult:
        """
        Tek görüntü için N forward pass çalıştırır ve belirsizliği hesaplar.

        Args:
            image_tensor: shape (C, H, W) veya (1, C, H, W)
        Returns:
            UncertaintyResult
        """
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor.to(self.device)

        enable_dropout_only(self.model)

        all_probs = []
        for _ in range(self.n_passes):
            logits = self.model(image_tensor)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            all_probs.append(probs)

        sample_probs = np.stack(all_probs, axis=0)  # (N_passes, n_classes)
        mean_probs = sample_probs.mean(axis=0)
        std_probs = sample_probs.std(axis=0)
        epistemic, aleatoric = decompose_uncertainty(sample_probs)

        return UncertaintyResult(
            mean_probs=mean_probs,
            std_probs=std_probs,
            epistemic=epistemic,
            aleatoric=aleatoric,
            total_uncertainty=epistemic + aleatoric,
            n_passes=self.n_passes,
            extra={
                "sample_probs": sample_probs,
                "var_probs": std_probs ** 2,
            },
        )

    @torch.no_grad()
    def predict_batch(
        self, images: torch.Tensor
    ) -> list[UncertaintyResult]:
        """
        Batch görüntüler için paralel MC Dropout tahmini.
        Her görüntü için ayrı UncertaintyResult döndürür.
        """
        if images.dim() == 3:
            images = images.unsqueeze(0)
        B = images.shape[0]
        images = images.to(self.device)

        enable_dropout_only(self.model)

        all_passes = []  # (N_passes, B, n_classes)
        for _ in range(self.n_passes):
            logits = self.model(images)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            all_passes.append(probs)

        stacked = np.stack(all_passes, axis=0)  # (N_passes, B, n_classes)

        results = []
        for b in range(B):
            sample_probs = stacked[:, b, :]  # (N_passes, n_classes)
            mean_p = sample_probs.mean(axis=0)
            std_p = sample_probs.std(axis=0)
            epi, ale = decompose_uncertainty(sample_probs)
            results.append(
                UncertaintyResult(
                    mean_probs=mean_p,
                    std_probs=std_p,
                    epistemic=epi,
                    aleatoric=ale,
                    total_uncertainty=epi + ale,
                    n_passes=self.n_passes,
                )
            )
        return results

    # ------------------------------------------------------------------
    # inference.py ile Uyumlu Sarmalayıcı
    # ------------------------------------------------------------------

    def predict_compatible(self, image_tensor: torch.Tensor) -> dict:
        """
        inference.py'nin döndürdüğü dict formatıyla uyumlu çıktı.
        Mevcut koda zarar vermeden drop-in replacement olarak kullanılabilir.
        """
        result = self.predict(image_tensor)
        label = "FAKE" if result.prediction == 1 else "REAL"
        return {
            "label": label,
            "fake_probability": float(result.mean_probs[1]),
            "real_probability": float(result.mean_probs[0]),
            "epistemic_uncertainty": result.epistemic,
            "aleatoric_uncertainty": result.aleatoric,
            "total_uncertainty": result.total_uncertainty,
            "confidence_std": float(result.std_probs.max()),
            "mc_passes": result.n_passes,
        }
