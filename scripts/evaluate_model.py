"""
GÖREV 5: Çok Katmanlı Model Değerlendirme
ROC-AUC, EER, ECE, FPR@95TPR, per-source accuracy, latency ölçümü.

Kullanım:
    python scripts/evaluate_model.py
    python scripts/evaluate_model.py --checkpoint models/best_run6.pth
    python scripts/evaluate_model.py --jury-only
"""
import os
import sys
import json
import time
import shutil
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import model_cfg, paths, DEVICE

try:
    from sklearn.metrics import (
        roc_auc_score, roc_curve, precision_recall_curve,
        f1_score, classification_report, confusion_matrix
    )
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

OUTPUT_DIR = Path(__file__).parent.parent / "evaluation"

# Kalibrasyon modülü (G5)
try:
    from core.calibration import (
        TemperatureScaling, compute_brier_score,
        generate_reliability_diagram, test_onnx_export,
        compute_ece as compute_ece_v2,
    )
    HAS_CALIBRATION = True
except ImportError:
    HAS_CALIBRATION = False


def load_model(checkpoint_path=None):
    """Model yükle."""
    from core.dual_mobilenetv3 import DualPathDeepfakeDetector
    model = DualPathDeepfakeDetector()
    cp = checkpoint_path or str(paths.BEST_MODEL_PATH)
    if Path(cp).exists():
        state = torch.load(cp, map_location=DEVICE, weights_only=False)
        if "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"], strict=False)
        else:
            model.load_state_dict(state, strict=False)
        print(f"  ✅ Model yüklendi: {cp}")
    else:
        print(f"  ⚠️ Checkpoint bulunamadı: {cp}")
    model.to(DEVICE).eval()
    return model


# Lazy-init paylaşılan extractor instance'ları (her çağrıda yeniden oluşturmayı önle)
_shared_dwt = None
_shared_mesh = None


def _get_extractors():
    """Paylaşılan DWT ve FaceMesh extractor'larını döndür."""
    global _shared_dwt, _shared_mesh
    if _shared_dwt is None:
        from core.data_pipeline import FaceMeshExtractor
        # Eğitimle aynı frekans extractor'ı kullan
        if getattr(model_cfg, 'USE_HYBRID_FREQ', False):
            try:
                from core.frequency_v2 import HybridFrequencyExtractor
                _shared_dwt = HybridFrequencyExtractor(
                    wavelets=model_cfg.DWT_WAVELETS,
                    size=model_cfg.IMG_SIZE,
                    include_dwt=True, include_dct=True, include_phase=True,
                )
            except ImportError:
                from core.data_pipeline import MultiScaleDWT
                _shared_dwt = MultiScaleDWT()
        else:
            from core.data_pipeline import MultiScaleDWT
            _shared_dwt = MultiScaleDWT()
        _shared_mesh = FaceMeshExtractor()
    return _shared_dwt, _shared_mesh


def predict_image(model, image_path: Path, transform) -> dict:
    """Tek görsel için tahmin — eğitim pipeline ile aynı ön işleme."""
    try:
        img = Image.open(image_path).convert("RGB")
        img_np = np.array(img)  # DWT ve FaceMesh numpy bekler

        rgb_tensor = transform(img).unsqueeze(0).to(DEVICE)

        dwt, mesh_ext = _get_extractors()

        # Frekans haritası (numpy array girdi)
        freq_map = dwt(img_np)
        if freq_map is not None:
            freq_tensor = torch.from_numpy(freq_map).float().unsqueeze(0).to(DEVICE)
        else:
            freq_tensor = torch.zeros(1, model_cfg.DWT_CHANNELS, 224, 224).to(DEVICE)

        # Face mesh (numpy array girdi)
        mesh = mesh_ext(img_np)
        if mesh is not None:
            mesh_tensor = torch.from_numpy(mesh).float().unsqueeze(0).to(DEVICE)
        else:
            mesh_tensor = torch.zeros(1, model_cfg.MESH_INPUT_DIM).to(DEVICE)

        with torch.no_grad():
            logits = model(rgb_tensor, freq_tensor, mesh_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        return {
            "fake_prob": float(probs[1]),
            "real_prob": float(probs[0]),
            "pred": int(probs[1] > 0.5),
        }
    except Exception as e:
        return {"fake_prob": 0.5, "real_prob": 0.5, "pred": 0, "error": str(e)}


def compute_eer(y_true, y_scores):
    """Equal Error Rate hesapla."""
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    # FPR == FNR noktasını bul
    idx = np.argmin(np.abs(fpr - fnr))
    eer = (fpr[idx] + fnr[idx]) / 2
    return float(eer), float(thresholds[idx])


def compute_ece(y_true, y_probs, n_bins=15):
    """Expected Calibration Error hesapla."""
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


def compute_fpr_at_tpr(y_true, y_scores, target_tpr=0.95):
    """Belirli TPR'da FPR hesapla."""
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    idx = np.argmin(np.abs(tpr - target_tpr))
    return float(fpr[idx])


def evaluate_dataset(model, data_dir: Path, transform, label: int, source_name: str = "") -> list:
    """Bir dizindeki tüm görselleri değerlendir."""
    results = []
    files = list(data_dir.glob("*.jpg")) + list(data_dir.glob("*.png"))

    for f in files:
        pred = predict_image(model, f, transform)
        pred["true_label"] = label
        pred["file"] = f.name
        pred["source"] = source_name or data_dir.parent.name
        results.append(pred)

    return results


def measure_latency(model, transform, n_runs=50):
    """Inference latency ölçümü."""
    dummy_img = Image.new("RGB", (224, 224), color=(128, 128, 128))
    rgb = transform(dummy_img).unsqueeze(0).to(DEVICE)
    freq = torch.randn(1, model_cfg.DWT_CHANNELS, 224, 224).to(DEVICE)
    mesh = torch.randn(1, model_cfg.MESH_INPUT_DIM).to(DEVICE)

    # Warmup
    for _ in range(5):
        with torch.no_grad():
            model(rgb, freq, mesh)

    # GPU sync
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        with torch.no_grad():
            model(rgb, freq, mesh)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - start)

    return {
        "mean_ms": float(np.mean(times) * 1000),
        "median_ms": float(np.median(times) * 1000),
        "p95_ms": float(np.percentile(times, 95) * 1000),
        "device": str(DEVICE),
        "n_runs": n_runs,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Model değerlendirme")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--jury-only", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("GÖREV 5: Çok Katmanlı Model Değerlendirme")
    print("=" * 60)

    if not HAS_SKLEARN:
        print("❌ scikit-learn gerekli: pip install scikit-learn")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fp_dir = OUTPUT_DIR / "fp_analysis"
    fn_dir = OUTPUT_DIR / "fn_analysis"
    fp_dir.mkdir(exist_ok=True)
    fn_dir.mkdir(exist_ok=True)

    # Model yükle
    print("\n📦 Model yükleniyor...")
    model = load_model(args.checkpoint)

    # Transform
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    all_results = []
    dataset_dir = Path(__file__).parent.parent / "dataset"

    # Test seti
    if not args.jury_only:
        print("\n📊 Test seti değerlendiriliyor...")
        test_dir = dataset_dir / "faces_split" / "test"
        if test_dir.exists():
            for label_name, label_id in [("real", 0), ("fake", 1)]:
                label_dir = test_dir / label_name
                if label_dir.exists():
                    results = evaluate_dataset(model, label_dir, transform, label_id, "test")
                    all_results.extend(results)
                    print(f"  {label_name}: {len(results)} görsel")

    # Jury seti
    print("\n📊 Jury seti değerlendiriliyor...")
    jury_dir = dataset_dir / "jury_test"
    if jury_dir.exists():
        jury_results = []
        for label_name, label_id in [("real", 0), ("fake", 1)]:
            label_dir = jury_dir / label_name
            if label_dir.exists():
                results = evaluate_dataset(model, label_dir, transform, label_id, "jury")
                jury_results.extend(results)
                all_results.extend(results)
                print(f"  jury/{label_name}: {len(results)} görsel")

    if not all_results:
        print("❌ Değerlendirilecek veri bulunamadı!")
        return

    # Metrikleri hesapla
    print("\n📈 Metrikler hesaplanıyor...")
    y_true = np.array([r["true_label"] for r in all_results])
    y_scores = np.array([r["fake_prob"] for r in all_results])
    y_pred = np.array([r["pred"] for r in all_results])

    auc = roc_auc_score(y_true, y_scores)
    eer, eer_threshold = compute_eer(y_true, y_scores)
    ece = compute_ece(y_true, y_scores)
    fpr_95 = compute_fpr_at_tpr(y_true, y_scores, 0.95)
    f1 = f1_score(y_true, y_pred, average="macro")
    cm = confusion_matrix(y_true, y_pred).tolist()

    # Per-source accuracy
    source_metrics = defaultdict(lambda: {"correct": 0, "total": 0, "fp": 0, "fn": 0})
    for r in all_results:
        src = r["source"]
        source_metrics[src]["total"] += 1
        if r["pred"] == r["true_label"]:
            source_metrics[src]["correct"] += 1
        elif r["true_label"] == 0 and r["pred"] == 1:
            source_metrics[src]["fp"] += 1
            # FP görselini kopyala
            src_file = Path(__file__).parent.parent / "dataset" / "faces_split" / "test" / "real" / r["file"]
            if src_file.exists():
                shutil.copy2(src_file, fp_dir / r["file"])
        elif r["true_label"] == 1 and r["pred"] == 0:
            source_metrics[src]["fn"] += 1

    per_source = {}
    for src, m in source_metrics.items():
        per_source[src] = {
            "accuracy": m["correct"] / max(m["total"], 1),
            "fp_count": m["fp"],
            "fn_count": m["fn"],
            "total": m["total"],
        }

    # Latency
    print("\n⏱️ Latency ölçülüyor...")
    latency = measure_latency(model, transform)

    # Sonuçları kaydet
    metrics = {
        "roc_auc": round(auc, 4),
        "eer": round(eer, 4),
        "eer_threshold": round(eer_threshold, 4),
        "ece": round(ece, 4),
        "fpr_at_95tpr": round(fpr_95, 4),
        "macro_f1": round(f1, 4),
        "confusion_matrix": cm,
        "per_source": per_source,
        "latency": latency,
        "total_evaluated": len(all_results),
        "fp_total": sum(m["fp"] for m in source_metrics.values()),
        "fn_total": sum(m["fn"] for m in source_metrics.values()),
    }

    # Brier Score & Temperature Scaling (G5)
    if HAS_CALIBRATION:
        brier = compute_brier_score(y_true, y_scores)
        metrics["brier_score"] = round(brier, 4)
        print(f"  Brier Score: {brier:.4f}")

        # Reliability Diagram
        rel_path = generate_reliability_diagram(
            y_true, y_scores,
            output_path=OUTPUT_DIR / "reliability_diagram.png"
        )
        if rel_path:
            metrics["reliability_diagram"] = str(rel_path)

        # Temperature Scaling (validation set üzerinde)
        try:
            from core.data_pipeline import get_dataloaders
            _, val_loader, _ = get_dataloaders(batch_size=model_cfg.BATCH_SIZE)
            calibrator = TemperatureScaling()
            optimal_t = calibrator.fit(val_loader, model, DEVICE)
            metrics["temperature"] = round(optimal_t, 4)

            # Kalibre edilmiş ECE
            calibrated_probs = torch.softmax(
                torch.tensor(np.column_stack([1 - y_scores, y_scores])) / optimal_t, dim=1
            ).numpy()[:, 1]
            calibrated_ece = compute_ece_v2(y_true, calibrated_probs)
            metrics["calibrated_ece"] = round(calibrated_ece, 4)
            print(f"  Calibrated ECE: {calibrated_ece:.4f} (T={optimal_t:.4f})")

            # Kalibrasyon ağırlıklarını kaydet
            calibrator.save()
        except Exception as e:
            print(f"  ⚠️ Temperature Scaling hatası: {e}")

        # ONNX Export Testi
        try:
            onnx_results = test_onnx_export(model)
            metrics["onnx_export"] = onnx_results
        except Exception as e:
            print(f"  ⚠️ ONNX export hatası: {e}")

    metrics_path = OUTPUT_DIR / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # Rapor yazdır
    print(f"\n{'=' * 60}")
    print(f"📊 DEĞERLENDİRME SONUÇLARI")
    print(f"{'=' * 60}")
    print(f"  ROC-AUC:       {auc:.4f}")
    print(f"  EER:           {eer:.4f} (threshold={eer_threshold:.4f})")
    print(f"  ECE:           {ece:.4f}")
    print(f"  FPR@95TPR:     {fpr_95:.4f}")
    print(f"  Macro F1:      {f1:.4f}")
    print(f"  False Positive: {metrics['fp_total']}")
    print(f"  False Negative: {metrics['fn_total']}")
    print(f"  Latency:       {latency['mean_ms']:.1f}ms ({latency['device']})")
    if 'brier_score' in metrics:
        print(f"  Brier Score:   {metrics['brier_score']:.4f}")
    if 'temperature' in metrics:
        print(f"  Temperature:   {metrics['temperature']:.4f}")
    if 'calibrated_ece' in metrics:
        print(f"  Calibrated ECE:{metrics['calibrated_ece']:.4f} "
              f"({'✅ < 5%' if metrics['calibrated_ece'] < 0.05 else '⚠️ > 5%'})")
    print(f"\n  Per-Source:")
    for src, m in sorted(per_source.items()):
        print(f"    {src}: acc={m['accuracy']:.3f} FP={m['fp_count']} FN={m['fn_count']} (n={m['total']})")
    print(f"\n  📂 Sonuçlar: {metrics_path}")
    print(f"  📂 FP analiz: {fp_dir}")
    if 'reliability_diagram' in metrics:
        print(f"  📂 Reliability Diagram: {metrics['reliability_diagram']}")
    print(f"{'=' * 60}")
    print("✅ GÖREV_5_TAMAMLANDI")


if __name__ == "__main__":
    main()
