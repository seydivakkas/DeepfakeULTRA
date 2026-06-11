"""
GÖREV 5: Post-hoc Model Kalibrasyon
Temperature Scaling + Platt Scaling + Vector Scaling + Auto Calibrate.
Brier Score + Reliability Diagram.

Kullanım:
    from core.calibration import auto_calibrate
    best_method, calibrator = auto_calibrate(val_loader, model, device)
    calibrated_probs = calibrator.calibrate(logits)

    # Veya tek yöntem:
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
from typing import Tuple, Optional, Dict, List

from config import model_cfg, paths, DEVICE


# ═══════════════════════════════════════════════════════════
# TEMPERATURE SCALING (Guo et al., 2017)
# ═══════════════════════════════════════════════════════════
class TemperatureScaling(nn.Module):
    """
    Post-hoc kalibrasyon: validation set üzerinde optimal T öğrenir.
    NLL minimize ederek tek bir temperature parametresi optimize edilir.

    Referans: Guo et al., "On Calibration of Modern Neural Networks" (2017)

    KRİTİK: Temperature Scaling LOGIT'ler üzerinde çalışır.
    calibrated_prob = softmax(logit / T) — olasılıklara uygulanmaz.
    """

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
        self.method_name = "temperature_scaling"

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Logit'leri temperature ile ölçekle."""
        t = self.temperature.to(logits.device)
        return logits / t

    def calibrate(self, logits: torch.Tensor) -> torch.Tensor:
        """Kalibre edilmiş olasılıklar döndür (logit girişi zorunlu)."""
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
        print(f"  🌡️ [TemperatureScaling] Optimal T: {optimal_t:.4f}")
        return optimal_t

    def save(self, path: Optional[Path] = None):
        """Kalibrasyon ağırlıklarını kaydet."""
        path = path or paths.MODEL_DIR / "calibration_weights.json"
        data = {
            "method": self.method_name,
            "temperature": self.temperature.item(),
        }
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
# PLATT SCALING (Platt, 1999)
# ═══════════════════════════════════════════════════════════
class PlattScaling(nn.Module):
    """
    Platt Scaling — underconfident modeller için ideal.
    calibrated_prob = σ(a × logit + b)

    İki öğrenilebilir parametre (a, b) ile logit'leri sigmoid üzerinden
    olasılıklara dönüştürür. Temperature Scaling'den daha esnek çünkü
    bias terimi de öğrenir.

    Referans: Platt, "Probabilistic Outputs for SVMs" (1999)
    """

    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.ones(1))
        self.b = nn.Parameter(torch.zeros(1))
        self.method_name = "platt_scaling"

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Binary: FAKE sınıfı logit'ini ölçekle."""
        # logits: (batch, 2) — [:, 1] = fake logit
        fake_logit = logits[:, 1] - logits[:, 0]  # Log-odds (FAKE vs REAL)
        a = self.a.to(logits.device)
        b = self.b.to(logits.device)
        scaled = a * fake_logit + b
        return scaled

    def calibrate(self, logits: torch.Tensor) -> torch.Tensor:
        """Kalibre edilmiş olasılıklar döndür — (batch, 2)."""
        with torch.no_grad():
            scaled = self.forward(logits)
            fake_prob = torch.sigmoid(scaled)
            real_prob = 1.0 - fake_prob
            return torch.stack([real_prob, fake_prob], dim=1)

    def fit(
        self,
        val_loader,
        model: nn.Module,
        device: torch.device = DEVICE,
        max_iter: int = 200,
        lr: float = 0.01,
    ) -> Tuple[float, float]:
        """
        Validation set üzerinde (a, b) parametrelerini öğren.
        Binary Cross-Entropy (BCE) minimize eder.
        """
        model.eval()

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
        all_labels = torch.cat(all_labels, dim=0).float()

        print(f"  📊 Platt Scaling verileri: {len(all_labels)} örnek")

        # Parametreleri optimize et
        self.a = nn.Parameter(torch.ones(1, device=device))
        self.b = nn.Parameter(torch.zeros(1, device=device))

        optimizer = torch.optim.LBFGS([self.a, self.b], lr=lr, max_iter=max_iter)
        bce_criterion = nn.BCEWithLogitsLoss()

        def closure():
            optimizer.zero_grad()
            fake_logit = all_logits[:, 1] - all_logits[:, 0]
            scaled = self.a * fake_logit + self.b
            loss = bce_criterion(scaled, all_labels)
            loss.backward()
            return loss

        optimizer.step(closure)

        a_val = self.a.item()
        b_val = self.b.item()
        print(f"  📐 [PlattScaling] a={a_val:.4f}, b={b_val:.4f}")
        return a_val, b_val

    def save(self, path: Optional[Path] = None):
        """Kalibrasyon ağırlıklarını kaydet."""
        path = path or paths.MODEL_DIR / "calibration_weights.json"
        data = {
            "method": self.method_name,
            "a": self.a.item(),
            "b": self.b.item(),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  💾 Platt kalibrasyon kaydedildi: {path}")

    def load(self, path: Optional[Path] = None):
        """Kayıtlı kalibrasyon ağırlıklarını yükle."""
        path = path or paths.MODEL_DIR / "calibration_weights.json"
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            if data.get("method") == self.method_name:
                self.a = nn.Parameter(torch.tensor([data["a"]]))
                self.b = nn.Parameter(torch.tensor([data["b"]]))
                print(f"  📂 Platt yüklendi: a={data['a']:.4f}, b={data['b']:.4f}")
                return True
        return False


# ═══════════════════════════════════════════════════════════
# VECTOR SCALING (Guo et al., 2017 genişletmesi)
# ═══════════════════════════════════════════════════════════
class VectorScaling(nn.Module):
    """
    Vector Scaling — sınıf bazlı temperature + bias.
    calibrated = softmax(W ⊙ logit + b)

    Her sınıf için ayrı ölçekleme ve kaydırma parametresi.
    Temperature Scaling'in genelleştirilmiş hali.
    """

    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.num_classes = num_classes
        self.W = nn.Parameter(torch.ones(num_classes))
        self.b = nn.Parameter(torch.zeros(num_classes))
        self.method_name = "vector_scaling"

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Sınıf bazlı ölçekleme ve kaydırma."""
        W = self.W.to(logits.device)
        b = self.b.to(logits.device)
        return logits * W + b

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
        max_iter: int = 100,
        lr: float = 0.01,
    ) -> Tuple[List[float], List[float]]:
        """Validation set üzerinde W ve b parametrelerini öğren."""
        model.eval()

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

        print(f"  📊 Vector Scaling verileri: {len(all_labels)} örnek")

        self.W = nn.Parameter(torch.ones(self.num_classes, device=device))
        self.b = nn.Parameter(torch.zeros(self.num_classes, device=device))

        optimizer = torch.optim.LBFGS([self.W, self.b], lr=lr, max_iter=max_iter)
        nll_criterion = nn.CrossEntropyLoss()

        def closure():
            optimizer.zero_grad()
            scaled = all_logits * self.W + self.b
            loss = nll_criterion(scaled, all_labels)
            loss.backward()
            return loss

        optimizer.step(closure)

        w_vals = self.W.detach().cpu().tolist()
        b_vals = self.b.detach().cpu().tolist()
        print(f"  📐 [VectorScaling] W={[f'{v:.4f}' for v in w_vals]}, "
              f"b={[f'{v:.4f}' for v in b_vals]}")
        return w_vals, b_vals

    def save(self, path: Optional[Path] = None):
        """Kalibrasyon ağırlıklarını kaydet."""
        path = path or paths.MODEL_DIR / "calibration_weights.json"
        data = {
            "method": self.method_name,
            "W": self.W.detach().cpu().tolist(),
            "b": self.b.detach().cpu().tolist(),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  💾 Vector kalibrasyon kaydedildi: {path}")

    def load(self, path: Optional[Path] = None):
        """Kayıtlı kalibrasyon ağırlıklarını yükle."""
        path = path or paths.MODEL_DIR / "calibration_weights.json"
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            if data.get("method") == self.method_name:
                self.W = nn.Parameter(torch.tensor(data["W"]))
                self.b = nn.Parameter(torch.tensor(data["b"]))
                print(f"  📂 Vector yüklendi: W={data['W']}, b={data['b']}")
                return True
        return False


# ═══════════════════════════════════════════════════════════
# OTOMATİK KALİBRASYON SEÇİCİ
# ═══════════════════════════════════════════════════════════
def auto_calibrate(
    val_loader,
    model: nn.Module,
    device: torch.device = DEVICE,
) -> Tuple[str, nn.Module, Dict]:
    """
    Tüm kalibrasyon yöntemlerini dener → en düşük ECE'yi veren yöntemi seçer.

    Returns:
        (method_name, calibrator, results_dict)
    """
    print("\n  🔄 Auto-Calibrate: Tüm yöntemler deneniyor...")

    model.eval()

    # Logit'leri ve etiketleri bir kere topla (tekrar kullanmak için)
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for batch in val_loader:
            rgb, freq, mesh, labels, _ = batch
            rgb, freq, mesh = rgb.to(device), freq.to(device), mesh.to(device)
            labels = labels.to(device)
            logits = model(rgb, freq, mesh)
            all_logits.append(logits)
            all_labels.append(labels)

    all_logits_t = torch.cat(all_logits, dim=0)
    all_labels_t = torch.cat(all_labels, dim=0)
    y_true = all_labels_t.cpu().numpy()

    # Ham (kalibre edilmemiş) ECE
    uncal_probs = F.softmax(all_logits_t, dim=1).cpu().numpy()[:, 1]
    uncal_ece = compute_ece(y_true, uncal_probs)
    uncal_brier = compute_brier_score(y_true, uncal_probs)

    results = {
        "uncalibrated": {
            "ece": uncal_ece,
            "brier": uncal_brier,
            "calibrator": None,
        }
    }
    print(f"    [Kalibre edilmemiş] ECE={uncal_ece:.4f}, Brier={uncal_brier:.4f}")

    # --- Yöntem 1: Temperature Scaling ---
    try:
        ts = TemperatureScaling()
        ts.fit(val_loader, model, device)
        cal_probs = ts.calibrate(all_logits_t).cpu().numpy()[:, 1]
        ts_ece = compute_ece(y_true, cal_probs)
        ts_brier = compute_brier_score(y_true, cal_probs)
        results["temperature_scaling"] = {
            "ece": ts_ece,
            "brier": ts_brier,
            "calibrator": ts,
            "temperature": ts.temperature.item(),
        }
        print(f"    [TemperatureScaling] ECE={ts_ece:.4f}, Brier={ts_brier:.4f}, "
              f"T={ts.temperature.item():.4f}")
    except Exception as e:
        print(f"    [TemperatureScaling] Hata: {e}")

    # --- Yöntem 2: Platt Scaling ---
    try:
        ps = PlattScaling()
        ps.fit(val_loader, model, device)
        cal_probs = ps.calibrate(all_logits_t).cpu().numpy()[:, 1]
        ps_ece = compute_ece(y_true, cal_probs)
        ps_brier = compute_brier_score(y_true, cal_probs)
        results["platt_scaling"] = {
            "ece": ps_ece,
            "brier": ps_brier,
            "calibrator": ps,
            "a": ps.a.item(),
            "b": ps.b.item(),
        }
        print(f"    [PlattScaling] ECE={ps_ece:.4f}, Brier={ps_brier:.4f}, "
              f"a={ps.a.item():.4f}, b={ps.b.item():.4f}")
    except Exception as e:
        print(f"    [PlattScaling] Hata: {e}")

    # --- Yöntem 3: Vector Scaling ---
    try:
        vs = VectorScaling(num_classes=model_cfg.NUM_CLASSES)
        vs.fit(val_loader, model, device)
        cal_probs = vs.calibrate(all_logits_t).cpu().numpy()[:, 1]
        vs_ece = compute_ece(y_true, cal_probs)
        vs_brier = compute_brier_score(y_true, cal_probs)
        results["vector_scaling"] = {
            "ece": vs_ece,
            "brier": vs_brier,
            "calibrator": vs,
            "W": vs.W.detach().cpu().tolist(),
            "b": vs.b.detach().cpu().tolist(),
        }
        print(f"    [VectorScaling] ECE={vs_ece:.4f}, Brier={vs_brier:.4f}")
    except Exception as e:
        print(f"    [VectorScaling] Hata: {e}")

    # En iyi yöntemi seç (en düşük ECE)
    best_method = "uncalibrated"
    best_ece = uncal_ece
    best_calibrator = None

    for method, r in results.items():
        if method == "uncalibrated":
            continue
        if r["ece"] < best_ece:
            best_ece = r["ece"]
            best_method = method
            best_calibrator = r["calibrator"]

    print(f"\n  ✅ En iyi yöntem: {best_method} (ECE={best_ece:.4f})")

    # Karşılaştırma tablosu
    print(f"\n  {'Yöntem':<25} {'ECE':>8} {'Brier':>8} {'İyileşme':>10}")
    print(f"  {'-'*51}")
    for method, r in results.items():
        improvement = ((uncal_ece - r["ece"]) / uncal_ece * 100) if uncal_ece > 0 else 0
        marker = " ← EN İYİ" if method == best_method else ""
        print(f"  {method:<25} {r['ece']:>8.4f} {r['brier']:>8.4f} "
              f"{improvement:>+9.1f}%{marker}")

    return best_method, best_calibrator, results


def load_calibrator(path: Optional[Path] = None) -> Optional[nn.Module]:
    """Kayıtlı kalibrasyon ağırlıklarını yükle — yöntemden bağımsız."""
    path = path or paths.MODEL_DIR / "calibration_weights.json"
    if not path.exists():
        return None

    with open(path) as f:
        data = json.load(f)

    method = data.get("method", "temperature_scaling")

    if method == "platt_scaling":
        cal = PlattScaling()
        cal.a = nn.Parameter(torch.tensor([data["a"]]))
        cal.b = nn.Parameter(torch.tensor([data["b"]]))
        print(f"  📂 Platt yüklendi: a={data['a']:.4f}, b={data['b']:.4f}")
        return cal
    elif method == "vector_scaling":
        cal = VectorScaling(num_classes=len(data["W"]))
        cal.W = nn.Parameter(torch.tensor(data["W"]))
        cal.b = nn.Parameter(torch.tensor(data["b"]))
        print(f"  📂 Vector yüklendi: W={data['W']}, b={data['b']}")
        return cal
    else:
        cal = TemperatureScaling()
        cal.temperature = nn.Parameter(torch.tensor([data.get("temperature", 1.0)]))
        print(f"  📂 Temperature yüklendi: T={data.get('temperature', 1.0):.4f}")
        return cal


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
    title: str = "Reliability Diagram",
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
    ax1.set_title(title, fontsize=14, fontweight="bold")
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
    onnxscript yoksa eski API ile dener, başarısız olursa graceful skip.
    """
    output_path = output_path or paths.ONNX_MODEL_PATH
    results = {"success": False, "path": str(output_path)}

    try:
        model.eval()
        dummy_rgb = torch.randn(1, 3, 224, 224).to(DEVICE)
        dummy_freq = torch.randn(1, model_cfg.DWT_CHANNELS, 224, 224).to(DEVICE)
        dummy_mesh = torch.randn(1, model_cfg.MESH_INPUT_DIM).to(DEVICE)

        # Opset sürümünü ayarla — onnxscript yoksa düşük sürüm dene
        opset = model_cfg.ONNX_OPSET_VERSION
        try:
            import onnxscript  # noqa: F401
        except ImportError:
            opset = min(opset, 14)  # onnxscript olmadan eski opset kullan
            print(f"  ⚠️ onnxscript bulunamadı, opset {opset} ile deneniyor")

        torch.onnx.export(
            model,
            (dummy_rgb, dummy_freq, dummy_mesh),
            str(output_path),
            export_params=True,
            opset_version=opset,
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
        results["opset_version"] = opset
        results["file_size_mb"] = round(output_path.stat().st_size / (1024 * 1024), 2)
        print(f"  ✅ ONNX export: {results['file_size_mb']}MB → {output_path}")

    except Exception as e:
        error_msg = str(e)
        results["error"] = error_msg

        # Kullanıcı dostu hata mesajı
        if "onnxscript" in error_msg.lower() or "onnx" in error_msg.lower():
            print(f"  ⚠️ ONNX export atlandı: pip install onnxscript onnx gerekli")
        else:
            print(f"  ❌ ONNX export hatası: {error_msg}")

    return results
