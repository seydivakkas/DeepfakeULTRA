"""
Deepfake Detection System v3 — Değerlendirme
Accuracy, AUC, EER, F1, Confusion Matrix, ROC Curve.
"""
import torch
import numpy as np
from pathlib import Path
from config import model_cfg, paths, DEVICE
from core.dual_mobilenetv3 import DualPathDeepfakeDetector
from core.data_pipeline import get_dataloaders

try:
    from sklearn.metrics import (roc_auc_score, accuracy_score, f1_score,
                                 precision_score, recall_score, confusion_matrix,
                                 roc_curve)
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


def compute_eer(labels, scores):
    """Equal Error Rate hesapla."""
    if not HAS_SKLEARN:
        return 0.0
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    return float(fpr[idx])


def compute_metrics(labels, predictions, probabilities=None):
    """
    3 sınıf (REAL/FAKE/SPOOF) metrik sözlüğü döndür.
    probabilities: (N, 3) numpy array veya None
    """
    metrics = {
        "accuracy": float(accuracy_score(labels, predictions)) if HAS_SKLEARN else 0.0,
        "f1_macro": float(f1_score(labels, predictions, average="macro", zero_division=0)) if HAS_SKLEARN else 0.0,
        "precision_macro": float(precision_score(labels, predictions, average="macro", zero_division=0)) if HAS_SKLEARN else 0.0,
        "recall_macro": float(recall_score(labels, predictions, average="macro", zero_division=0)) if HAS_SKLEARN else 0.0,
    }

    if HAS_SKLEARN:
        # Per-class F1
        class_names = model_cfg.CLASS_NAMES  # ["REAL", "FAKE", "SPOOF"]
        per_class_f1 = f1_score(labels, predictions, average=None, zero_division=0)
        for i, name in enumerate(class_names):
            if i < len(per_class_f1):
                metrics[f"f1_{name.lower()}"] = float(per_class_f1[i])

        # Multi-class AUC (One-vs-Rest)
        if probabilities is not None and len(set(labels)) > 1:
            try:
                metrics["auc_macro"] = float(roc_auc_score(
                    labels, probabilities, multi_class="ovr", average="macro"
                ))
            except Exception:
                metrics["auc_macro"] = 0.5

        # Confusion Matrix (3×3)
        metrics["confusion_matrix"] = confusion_matrix(
            labels, predictions, labels=[0, 1, 2]
        ).tolist()

        # Liveness metrikleri (ISO 30107): SPOOF vs REAL
        labels_arr = np.array(labels)
        preds_arr = np.array(predictions)
        # APCER: Spoof örneklerin yanlışlıkla REAL olarak sınıflandırılma oranı
        spoof_mask = labels_arr == 2
        if spoof_mask.sum() > 0:
            metrics["apcer"] = float((preds_arr[spoof_mask] == 0).sum() / spoof_mask.sum())
        # BPCER: Gerçek örneklerin yanlışlıkla SPOOF olarak sınıflandırılma oranı
        real_mask = labels_arr == 0
        if real_mask.sum() > 0:
            metrics["bpcer"] = float((preds_arr[real_mask] == 2).sum() / real_mask.sum())
        # ACER: (APCER + BPCER) / 2
        if "apcer" in metrics and "bpcer" in metrics:
            metrics["acer"] = (metrics["apcer"] + metrics["bpcer"]) / 2.0

    return metrics


def run_evaluation(model_path=None):
    """Model değerlendirmesi."""
    model_path = model_path or str(paths.BEST_MODEL_PATH)
    print(f"📊 Değerlendirme başlıyor — {model_path}")

    model = DualPathDeepfakeDetector().to(DEVICE)
    if Path(model_path).exists():
        ckpt = torch.load(model_path, map_location=DEVICE, weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state)
        print("✅ Model ağırlıkları yüklendi")
    else:
        print("⚠️ Model dosyası bulunamadı, rastgele ağırlıklar kullanılıyor")

    _, _, test_loader = get_dataloaders()
    model.eval()

    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for rgb, freq, mesh, labels in test_loader:
            rgb, freq = rgb.to(DEVICE), freq.to(DEVICE)
            mesh, labels = mesh.to(DEVICE), labels.to(DEVICE)
            logits = model(rgb, freq, mesh)
            probs = torch.softmax(logits, dim=1)  # (batch, 3)
            preds = logits.argmax(dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    if not all_labels:
        print("⚠️ Test verisi bulunamadı")
        return {}

    all_probs_np = np.array(all_probs)  # (N, 3)
    metrics = compute_metrics(all_labels, all_preds, all_probs_np)
    print(f"\n{'='*40}")
    print(f"📊 Sonuçlar:")
    for k, v in metrics.items():
        if k != "confusion_matrix":
            print(f"   {k}: {v:.4f}" if isinstance(v, float) else f"   {k}: {v}")
    print(f"{'='*40}")

    # MLflow loglama
    try:
        import mlflow
        mlflow.set_tracking_uri(f"file:{paths.MLRUNS_DIR}")
        with mlflow.start_run(run_name="evaluation"):
            for k, v in metrics.items():
                if isinstance(v, float):
                    mlflow.log_metric(f"test_{k}", v)
    except Exception:
        pass

    return metrics


if __name__ == "__main__":
    run_evaluation()
