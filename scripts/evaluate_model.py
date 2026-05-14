"""
Model Degerlendirme  DataLoader Batch Inference + Youden's J Threshold
ROC-AUC, EER, ECE, FPR@95TPR, per-source accuracy, latency olcumu.

Kullanim:
    python scripts/evaluate_model.py
    python scripts/evaluate_model.py --checkpoint models/best_run5_forensic.pth
    python scripts/evaluate_model.py --jury-only
    python scripts/evaluate_model.py --external dataset/external_tests/celeb_df_v2
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
from torch.utils.data import DataLoader

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

# Kalibrasyon modulu (G5)
try:
    from core.calibration import (
        TemperatureScaling, compute_brier_score,
        generate_reliability_diagram, test_onnx_export,
        compute_ece as compute_ece_v2,
    )
    HAS_CALIBRATION = True
except ImportError:
    HAS_CALIBRATION = False

# MLflow
try:
    import mlflow
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False


# 
# MODEL YUKLEME
# 
def load_model(checkpoint_path=None):
    """Model yukle."""
    from core.dual_mobilenetv3 import DualPathDeepfakeDetector
    model = DualPathDeepfakeDetector()
    cp = checkpoint_path or str(paths.BEST_MODEL_PATH)
    if Path(cp).exists():
        state = torch.load(cp, map_location=DEVICE, weights_only=False)
        if "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"], strict=False)
        else:
            model.load_state_dict(state, strict=False)
        print(f"   Model yuklendi: {cp}")
    else:
        print(f"   Checkpoint bulunamadi: {cp}")
    model.to(DEVICE).eval()
    return model


# 
# DATALOADER TABANLI BATCH INFERENCE
# 
def batch_evaluate(model, data_dir: Path, split: str = "val", source_name: str = "test",
                   use_tta: bool = False, tta_n: int = 10):
    """DataLoader ile batch inference  egitimle ayni pipeline."""
    from core.data_pipeline import DeepfakeDataset
    from tqdm import tqdm

    # TTA predictor (opsiyonel)
    tta_predictor = None
    if use_tta:
        try:
            from core.tta_inference import TTAPredictor
            tta_predictor = TTAPredictor(model, n_aug=tta_n, device=str(DEVICE))
            print(f"  [TTA] aktif: {tta_n} augmentasyon")
        except ImportError:
            print("  [UYARI] TTA modulu bulunamadi, normal inference kullaniliyor")

    ds = DeepfakeDataset(str(data_dir), split=split, source_tag=source_name)
    if len(ds) == 0:
        print(f"   {data_dir} dizininde veri bulunamadi!")
        return [], [], []

    # TTA modunda batch size kucult (N kopya bellekte)
    bs = 32 if use_tta else 64
    loader = DataLoader(
        ds, batch_size=bs, shuffle=False,
        num_workers=4, pin_memory=DEVICE.type == "cuda",
        persistent_workers=True,
    )

    all_probs = []
    all_labels = []
    all_sources = []

    total_batches = len(loader)
    desc = f"[{source_name}" + (" TTA]" if use_tta else "]")
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc, total=total_batches):
            rgb, freq, mesh, labels, source_tags = batch
            rgb = rgb.to(DEVICE)
            freq = freq.to(DEVICE)
            mesh = mesh.to(DEVICE)

            if tta_predictor:
                probs = tta_predictor.predict_batch(rgb, freq, mesh).cpu().numpy()
            else:
                logits = model(rgb, freq, mesh)
                probs = torch.softmax(logits, dim=1).cpu().numpy()

            all_probs.extend(probs)
            all_labels.extend(labels.numpy())
            all_sources.extend([source_name] * labels.size(0))

    return np.array(all_labels), np.array(all_probs), all_sources


# 
# METRIK HESAPLAMA
# 
def compute_eer(y_true, y_scores):
    """Equal Error Rate hesapla."""
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    idx = np.argmin(np.abs(fpr - fnr))
    eer = (fpr[idx] + fnr[idx]) / 2
    return float(eer), float(thresholds[idx])


def compute_optimal_threshold(y_true, y_scores):
    """Youden's J-statistic ile optimal threshold bul."""
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    return float(thresholds[best_idx]), float(j_scores[best_idx])


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
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    idx = np.argmin(np.abs(tpr - target_tpr))
    return float(fpr[idx]), float(thresholds[idx])


# 
# LATENCY OLCUMU
# 
def measure_latency(model, n_runs=50):
    """Inference latency olcumu."""
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dummy_img = Image.new("RGB", (224, 224), color=(128, 128, 128))
    rgb = transform(dummy_img).unsqueeze(0).to(DEVICE)
    freq = torch.randn(1, model_cfg.DWT_CHANNELS, 224, 224).to(DEVICE)
    mesh = torch.randn(1, model_cfg.MESH_INPUT_DIM).to(DEVICE)

    # Warmup
    for _ in range(5):
        with torch.no_grad():
            model(rgb, freq, mesh)

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


# 
# ROC CURVE GORSELLESTIME
# 
def save_roc_curve(y_true, y_scores, output_path, title="ROC Curve"):
    """ROC egrisini PNG olarak kaydet."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fpr, tpr, _ = roc_curve(y_true, y_scores)
        auc = roc_auc_score(y_true, y_scores)

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.plot(fpr, tpr, color="#2196F3", lw=2, label=f"AUC = {auc:.4f}")
        ax.plot([0, 1], [0, 1], color="#ccc", lw=1, linestyle="--")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(title)
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(str(output_path), dpi=150)
        plt.close(fig)
        return str(output_path)
    except ImportError:
        return None


def save_confusion_matrix_plot(cm, output_path, class_names=None):
    """Confusion matrix gorselini kaydet."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        names = class_names or model_cfg.CLASS_NAMES
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        ax.set_title("Confusion Matrix")
        fig.colorbar(im, ax=ax)
        ax.set_xticks(range(len(names)))
        ax.set_yticks(range(len(names)))
        ax.set_xticklabels(names)
        ax.set_yticklabels(names)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

        # Hucre degerleri
        for i in range(len(names)):
            for j in range(len(names)):
                color = "white" if cm[i, j] > cm.max() / 2 else "black"
                ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", color=color, fontsize=12)

        fig.tight_layout()
        fig.savefig(str(output_path), dpi=150)
        plt.close(fig)
        return str(output_path)
    except ImportError:
        return None


# 
# ANA DEGERLENDIRME
# 
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Model degerlendirme")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--external", type=str, default=None,
                        help="Harici dataset dizini (cross-dataset benchmark)")
    parser.add_argument("--tta", action="store_true",
                        help="Test-Time Augmentation aktif et")
    parser.add_argument("--tta-n", type=int, default=10,
                        help="TTA augmentasyon sayisi (varsayilan: 10)")
    args = parser.parse_args()

    print("=" * 60)
    tta_label = " + TTA" if args.tta else ""
    print(f"Model Degerlendirme  DataLoader Batch Inference{tta_label}")
    print("=" * 60)

    if not HAS_SKLEARN:
        print("scikit-learn gerekli: pip install scikit-learn")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fp_dir = OUTPUT_DIR / "fp_analysis"
    fn_dir = OUTPUT_DIR / "fn_analysis"
    fp_dir.mkdir(exist_ok=True)
    fn_dir.mkdir(exist_ok=True)

    # Model yukle
    print("\nModel yukleniyor...")
    model = load_model(args.checkpoint)

    all_labels = []
    all_probs = []
    all_sources = []
    dataset_dir = Path(__file__).parent.parent / "dataset"

    #  Harici dataset benchmark 
    if args.external:
        ext_dir = Path(args.external)
        if not ext_dir.exists():
            print(f"Harici dataset bulunamadi: {ext_dir}")
            return

        print(f"\nHarici dataset degerlendiriliyor: {ext_dir}")
        labels, probs, sources = batch_evaluate(
            model, ext_dir, split="val", source_name=ext_dir.name,
            use_tta=args.tta, tta_n=args.tta_n
        )
        if len(labels) > 0:
            all_labels = labels
            all_probs = probs
            all_sources = sources
            print(f"  {ext_dir.name}: {len(labels)} gorsel")
    else:
        #  Test seti 
        print("\nTest seti degerlendiriliyor (DataLoader batch)...")
        test_dir = dataset_dir / "faces_split" / "test"
        if test_dir.exists():
            labels, probs, sources = batch_evaluate(
                model, test_dir, split="val", source_name="test",
                use_tta=args.tta, tta_n=args.tta_n
            )
            if len(labels) > 0:
                all_labels.extend(labels)
                all_probs.extend(probs)
                all_sources.extend(sources)
                n_real = (labels == 0).sum()
                n_fake = (labels == 1).sum()
                print(f"  test: {n_real} REAL + {n_fake} FAKE = {len(labels)}")

    if len(all_labels) == 0:
        print(" Degerlendirilecek veri bulunamadi!")
        return

    all_labels = np.array(all_labels) if not isinstance(all_labels, np.ndarray) else all_labels
    all_probs = np.array(all_probs) if not isinstance(all_probs, np.ndarray) else all_probs

    # FAKE olasiliklari
    y_true = all_labels
    y_scores = all_probs[:, 1]

    #  Metrikleri hesapla 
    print("\n Metrikler hesaplaniyor...")

    auc = roc_auc_score(y_true, y_scores)

    # Optimal threshold (Youden's J)
    optimal_threshold, j_score = compute_optimal_threshold(y_true, y_scores)
    y_pred_optimal = (y_scores >= optimal_threshold).astype(int)

    # Sabit 0.5 threshold (karsilastirma)
    y_pred_fixed = (y_scores >= 0.5).astype(int)

    # EER
    eer, eer_threshold = compute_eer(y_true, y_scores)

    # ECE
    ece = compute_ece(y_true, y_scores)

    # FPR@95TPR
    fpr_95, fpr_95_threshold = compute_fpr_at_tpr(y_true, y_scores, 0.95)

    # F1 (optimal threshold ile)
    f1_optimal = f1_score(y_true, y_pred_optimal, average="macro")
    f1_fixed = f1_score(y_true, y_pred_fixed, average="macro")

    # Confusion matrix
    cm_optimal = confusion_matrix(y_true, y_pred_optimal).tolist()
    cm_fixed = confusion_matrix(y_true, y_pred_fixed).tolist()

    # Per-source accuracy (optimal threshold)
    source_metrics = defaultdict(lambda: {"correct": 0, "total": 0, "fp": 0, "fn": 0})
    for i in range(len(all_labels)):
        src = all_sources[i] if i < len(all_sources) else "unknown"
        source_metrics[src]["total"] += 1
        if y_pred_optimal[i] == y_true[i]:
            source_metrics[src]["correct"] += 1
        elif y_true[i] == 0 and y_pred_optimal[i] == 1:
            source_metrics[src]["fp"] += 1
        elif y_true[i] == 1 and y_pred_optimal[i] == 0:
            source_metrics[src]["fn"] += 1

    per_source = {}
    for src, m in source_metrics.items():
        per_source[src] = {
            "accuracy": m["correct"] / max(m["total"], 1),
            "fp_count": m["fp"],
            "fn_count": m["fn"],
            "total": m["total"],
        }

    # Olasilik dagilimi
    real_mask = y_true == 0
    fake_mask = y_true == 1
    prob_distribution = {
        "real_mean": float(y_scores[real_mask].mean()) if real_mask.any() else 0,
        "real_std": float(y_scores[real_mask].std()) if real_mask.any() else 0,
        "fake_mean": float(y_scores[fake_mask].mean()) if fake_mask.any() else 0,
        "fake_std": float(y_scores[fake_mask].std()) if fake_mask.any() else 0,
    }

    # Latency
    print("\n Latency olculuyor...")
    latency = measure_latency(model)

    #  Sonuclari kaydet 
    # Cikti dizini (external vs normal)
    if args.external:
        ext_name = Path(args.external).name
        out_dir = OUTPUT_DIR / "external" / ext_name
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = OUTPUT_DIR

    metrics = {
        "roc_auc": round(auc, 4),
        "eer": round(eer, 4),
        "eer_threshold": round(eer_threshold, 4),
        "optimal_threshold": round(optimal_threshold, 4),
        "youden_j": round(j_score, 4),
        "ece": round(ece, 4),
        "fpr_at_95tpr": round(fpr_95, 4),
        "fpr_at_95tpr_threshold": round(fpr_95_threshold, 4),
        "accuracy_optimal": round((y_pred_optimal == y_true).mean(), 4),
        "accuracy_fixed_05": round((y_pred_fixed == y_true).mean(), 4),
        "macro_f1_optimal": round(f1_optimal, 4),
        "macro_f1_fixed_05": round(f1_fixed, 4),
        "confusion_matrix_optimal": cm_optimal,
        "confusion_matrix_fixed_05": cm_fixed,
        "per_source": per_source,
        "probability_distribution": prob_distribution,
        "latency": latency,
        "total_evaluated": len(all_labels),
        "n_real": int(real_mask.sum()),
        "n_fake": int(fake_mask.sum()),
        "fp_total": sum(m["fp"] for m in source_metrics.values()),
        "fn_total": sum(m["fn"] for m in source_metrics.values()),
    }

    # Brier Score & Temperature Scaling (G5)  sadece normal evaluation icin
    if HAS_CALIBRATION and not args.external:
        brier = compute_brier_score(y_true, y_scores)
        metrics["brier_score"] = round(brier, 4)
        print(f"  Brier Score: {brier:.4f}")

        rel_path = generate_reliability_diagram(
            y_true, y_scores,
            output_path=out_dir / "reliability_diagram.png"
        )
        if rel_path:
            metrics["reliability_diagram"] = str(rel_path)

        try:
            from core.data_pipeline import get_dataloaders
            _, val_loader, _ = get_dataloaders(batch_size=model_cfg.BATCH_SIZE)
            calibrator = TemperatureScaling()
            optimal_t = calibrator.fit(val_loader, model, DEVICE)
            metrics["temperature"] = round(optimal_t, 4)

            calibrated_probs = torch.softmax(
                torch.tensor(np.column_stack([1 - y_scores, y_scores])) / optimal_t, dim=1
            ).numpy()[:, 1]
            calibrated_ece = compute_ece_v2(y_true, calibrated_probs)
            metrics["calibrated_ece"] = round(calibrated_ece, 4)
            print(f"  Calibrated ECE: {calibrated_ece:.4f} (T={optimal_t:.4f})")
            calibrator.save()
        except Exception as e:
            print(f"   Temperature Scaling hatasi: {e}")

        try:
            onnx_results = test_onnx_export(model)
            metrics["onnx_export"] = onnx_results
        except Exception as e:
            print(f"   ONNX export hatasi: {e}")

    # JSON kaydet
    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # ROC Curve gorsel
    roc_path = save_roc_curve(
        y_true, y_scores, out_dir / "roc_curve.png",
        title=f"ROC Curve (AUC={auc:.4f})"
    )
    if roc_path:
        metrics["roc_curve"] = roc_path

    # Confusion matrix gorsel
    cm_arr = np.array(cm_optimal)
    cm_path = save_confusion_matrix_plot(cm_arr, out_dir / "confusion_matrix.png")
    if cm_path:
        metrics["confusion_matrix_plot"] = cm_path

    # MLflow loglama
    if HAS_MLFLOW and not args.external:
        try:
            mlflow.set_tracking_uri(f"file:{paths.MLRUNS_DIR}")
            mlflow.set_experiment("deepfake-v5-evaluation")
            with mlflow.start_run(run_name="full_evaluation"):
                mlflow.log_metrics({
                    "eval_auc": auc,
                    "eval_eer": eer,
                    "eval_ece": ece,
                    "eval_f1_optimal": f1_optimal,
                    "eval_accuracy_optimal": float((y_pred_optimal == y_true).mean()),
                    "eval_optimal_threshold": optimal_threshold,
                })
                if metrics_path.exists():
                    mlflow.log_artifact(str(metrics_path))
                if roc_path and Path(roc_path).exists():
                    mlflow.log_artifact(roc_path)
        except Exception as e:
            print(f"   MLflow hatasi: {e}")

    #  Rapor yazdir 
    print(f"\n{'=' * 60}")
    print(f" DEGERLENDIRME SONUCLARI")
    print(f"{'=' * 60}")
    print(f"  ROC-AUC:           {auc:.4f}")
    print(f"  EER:               {eer:.4f} (threshold={eer_threshold:.4f})")
    print(f"  ECE:               {ece:.4f}")
    print(f"  FPR@95TPR:         {fpr_95:.4f} (threshold={fpr_95_threshold:.4f})")
    print(f"")
    print(f"  Optimal Threshold: {optimal_threshold:.4f} (Youden J={j_score:.4f})")
    print(f"  Accuracy (opt):    {(y_pred_optimal == y_true).mean():.4f}")
    print(f"  Macro F1 (opt):    {f1_optimal:.4f}")
    print(f"  Accuracy (0.5):    {(y_pred_fixed == y_true).mean():.4f}")
    print(f"  Macro F1 (0.5):    {f1_fixed:.4f}")
    print(f"")
    print(f"  REAL: {int(real_mask.sum()):,} gorsel | FAKE: {int(fake_mask.sum()):,} gorsel")
    print(f"  FP: {metrics['fp_total']:,} | FN: {metrics['fn_total']:,}")
    print(f"  Latency: {latency['mean_ms']:.1f}ms ({latency['device']})")
    print(f"")
    print(f"  Olasilik Dagilimi:")
    print(f"    REAL: {prob_distribution['real_mean']:.4f}  {prob_distribution['real_std']:.4f}")
    print(f"    FAKE: {prob_distribution['fake_mean']:.4f}  {prob_distribution['fake_std']:.4f}")

    if "brier_score" in metrics:
        print(f"  Brier Score:       {metrics['brier_score']:.4f}")
    if "temperature" in metrics:
        print(f"  Temperature:       {metrics['temperature']:.4f}")
    if "calibrated_ece" in metrics:
        status = " < 5%" if metrics["calibrated_ece"] < 0.05 else " > 5%"
        print(f"  Calibrated ECE:    {metrics['calibrated_ece']:.4f} ({status})")

    print(f"\n  Per-Source:")
    for src, m in sorted(per_source.items()):
        print(f"    {src}: acc={m['accuracy']:.3f} FP={m['fp_count']} FN={m['fn_count']} (n={m['total']})")

    print(f"\n   Sonuclar: {metrics_path}")
    if roc_path:
        print(f"   ROC Curve: {roc_path}")
    if cm_path:
        print(f"   Confusion Matrix: {cm_path}")
    print(f"{'=' * 60}")
    print(" DEGERLENDIRME_TAMAMLANDI")


if __name__ == "__main__":
    main()
