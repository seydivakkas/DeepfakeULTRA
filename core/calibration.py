"""
GÖREV 5: Post-hoc Model Kalibrasyon
Temperature Scaling + Brier Score + Reliability Diagram.

Kullanım:
    from core.calibration import TemperatureScaling
    calibrator = TemperatureScaling()
    calibrator.fit(val_loader, model, device)
    calibrated_probs = calibrator.calibrate(logits)
"""
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Tuple, Optional

from config import model_cfg, paths, DEVICE


class TemperatureScaling(nn.Module):
    """
    Post-hoc kalibrasyon: validation set üzerinde optimal T öğrenir.
    NLL minimize ederek tek bir temperature parametresi optimize edilir.

    Referans: Guo et al., "On Calibration of Modern Neural Networks" (2017)
    """

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Logit'leri temperature ile ölçekle."""
        return logits / self.temperature

    def calibrate(self, logits: torch.Tensor) -> torch.Tensor:
        """Kalibre edilmiş olasılıklar döndür."""
        with torch.no_grad():
            scaled = self.forward(logits)
            return F.softmax(scaled, dim=1)

    def fit(
        self,
        val_loader,
        model: nn.Module,
        device: torch.device = DEVICE,
        max_iter: int = 50,
        lr: float = 0.01,
    ) -> float:
        """
        Validation set üzerinde optimal temperature öğren.

        Args:
            val_loader: Validation DataLoader
            model: Eğitilmiş model (eval modunda)
            device: Hesaplama cihazı
            max_iter: Optimizasyon adımı
            lr: Öğrenme oranı

        Returns:
            Optimal temperature değeri
        """
        model.eval()

        # Tüm validation logit'lerini ve etiketlerini topla
        all_logits = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                rgb, freq, mesh, labels, _ = batch
                rgb = rgb.to(device)
                freq = freq.to(device)
                mesh = mesh.to(device)
                labels = labels.to(device)

                logits = model(rgb, freq, mesh)
                all_logits.append(logits)
                all_labels.append(labels)

        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        print(f"  📊 Kalibrasyon verileri: {len(all_labels)} örnek")

        # Temperature optimize et
        self.temperature = nn.Parameter(torch.ones(1, device=device) * 1.5)
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)
        nll_criterion = nn.CrossEntropyLoss()

        def closure():
            optimizer.zero_grad()
            scaled_logits = all_logits / self.temperature
            loss = nll_criterion(scaled_logits, all_labels)
            loss.backward()
            return loss

        optimizer.step(closure)

        optimal_t = self.temperature.item()
        print(f"  🌡️ Optimal Temperature: {optimal_t:.4f}")
        return optimal_t

    def save(self, path: Optional[Path] = None):
        """Kalibrasyon ağırlıklarını kaydet."""
        path = path or paths.MODEL_DIR / "calibration_weights.json"
        data = {"temperature": self.temperature.item()}
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  💾 Kalibrasyon kaydedildi: {path}")

    def load(self, path: Optional[Path] = None):
        """Kayıtlı kalibrasyon ağırlıklarını yükle."""
        path = path or paths.MODEL_DIR / "calibration_weights.json"
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            self.temperature = nn.Parameter(torch.tensor([data["temperature"]]))
            print(f"  📂 Kalibrasyon yüklendi: T={data['temperature']:.4f}")
            return True
        return False


# ═══════════════════════════════════════════════════════════
# METRİK YARDIMCILARI
# ═══════════════════════════════════════════════════════════
def compute_brier_score(y_true: np.ndarray, y_probs: np.ndarray) -> float:
    """
    Brier Score hesapla.
    BS = (1/N) Σ(p_i - y_i)²
    Daha düşük = daha iyi kalibrasyon.
    """
    return float(np.mean((y_probs - y_true) ** 2))


def compute_ece(y_true: np.ndarray, y_probs: np.ndarray, n_bins: int = 15) -> float:
    """
    Expected Calibration Error hesapla.
    ECE = Σ (|B_m|/N) × |acc(B_m) - conf(B_m)|
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(y_true)

    for i in range(n_bins):
        mask = (y_probs >= bin_boundaries[i]) & (y_probs < bin_boundaries[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = y_true[mask].mean()
        bin_conf = y_probs[mask].mean()
        ece += (mask.sum() / total) * abs(bin_acc - bin_conf)

    return float(ece)


def generate_reliability_diagram(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    n_bins: int = 15,
    output_path: Optional[Path] = None,
) -> Optional[Path]:
    """
    Reliability Diagram oluştur.
    Diagonal'e yakınlık = iyi kalibrasyon.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  ⚠️ matplotlib yüklü değil, reliability diagram atlanıyor")
        return None

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_accs = []
    bin_confs = []
    bin_counts = []

    for i in range(n_bins):
        mask = (y_probs >= bin_boundaries[i]) & (y_probs < bin_boundaries[i + 1])
        if mask.sum() == 0:
            bin_accs.append(0)
            bin_confs.append(0)
            bin_counts.append(0)
        else:
            bin_accs.append(float(y_true[mask].mean()))
            bin_confs.append(float(y_probs[mask].mean()))
            bin_counts.append(int(mask.sum()))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10), gridspec_kw={"height_ratios": [3, 1]})

    # Reliability diagram
    bin_centers = [(bin_boundaries[i] + bin_boundaries[i + 1]) / 2 for i in range(n_bins)]
    ax1.bar(bin_centers, bin_accs, width=1.0 / n_bins, alpha=0.7,
            edgecolor="black", label="Model Accuracy")
    ax1.plot([0, 1], [0, 1], "r--", linewidth=2, label="Perfect Calibration")
    ax1.set_xlabel("Confidence", fontsize=12)
    ax1.set_ylabel("Accuracy", fontsize=12)
    ax1.set_title("Reliability Diagram", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)

    # Histogram
    ax2.bar(bin_centers, bin_counts, width=1.0 / n_bins, alpha=0.7,
            edgecolor="black", color="steelblue")
    ax2.set_xlabel("Confidence", fontsize=12)
    ax2.set_ylabel("Count", fontsize=12)
    ax2.set_title("Confidence Distribution", fontsize=12)
    ax2.set_xlim(0, 1)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = output_path or paths.BASE_DIR / "evaluation" / "reliability_diagram.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊 Reliability Diagram: {output_path}")
    return output_path


def test_onnx_export(
    model: nn.Module,
    output_path: Optional[Path] = None,
) -> dict:
    """
    ONNX export + doğrulama testi.
    """
    output_path = output_path or paths.ONNX_MODEL_PATH
    results = {"success": False, "path": str(output_path)}

    try:
        model.eval()
        dummy_rgb = torch.randn(1, 3, 224, 224).to(DEVICE)
        dummy_freq = torch.randn(1, model_cfg.DWT_CHANNELS, 224, 224).to(DEVICE)
        dummy_mesh = torch.randn(1, model_cfg.MESH_INPUT_DIM).to(DEVICE)

        torch.onnx.export(
            model,
            (dummy_rgb, dummy_freq, dummy_mesh),
            str(output_path),
            export_params=True,
            opset_version=model_cfg.ONNX_OPSET_VERSION,
            do_constant_folding=True,
            input_names=["rgb", "freq", "mesh"],
            output_names=["logits"],
            dynamic_axes={
                "rgb": {0: "batch"},
                "freq": {0: "batch"},
                "mesh": {0: "batch"},
                "logits": {0: "batch"},
            },
        )
        results["success"] = True
        results["file_size_mb"] = round(output_path.stat().st_size / (1024 * 1024), 2)
        print(f"  ✅ ONNX export: {results['file_size_mb']}MB → {output_path}")

    except Exception as e:
        results["error"] = str(e)
        print(f"  ❌ ONNX export hatası: {e}")

    return results
