"""
Run #4 Binary — Test Seti Kapsamlı Değerlendirme
Best model'i test setine karşı çalıştırır.
Çıktılar: AUC, Accuracy, F1, Precision, Recall, EER, Confusion Matrix, ROC Curve.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from pathlib import Path
from collections import Counter
from config import model_cfg, paths, DEVICE
from core.dual_mobilenetv3 import DualPathDeepfakeDetector
from core.data_pipeline import get_dataloaders
from tqdm import tqdm

try:
    from sklearn.metrics import (
        roc_auc_score, accuracy_score, f1_score,
        precision_score, recall_score, confusion_matrix,
        classification_report, roc_curve
    )
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("sklearn yüklü değil!")
    sys.exit(1)


def compute_eer(labels, scores):
    """Equal Error Rate hesapla."""
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = float((fpr[idx] + fnr[idx]) / 2)
    eer_threshold = float(thresholds[idx])
    return eer, eer_threshold


def run_test_evaluation(model_path=None):
    """Best model'i test setine karşı kapsamlı değerlendir."""
    # Model yolu
    if model_path:
        model_path = Path(model_path)
    else:
        # best_run4_binary.pth öncelikli, yoksa best_model.pth
        best_run4 = paths.MODEL_DIR / "best_run4_binary.pth"
        if best_run4.exists():
            model_path = best_run4
        else:
            model_path = paths.BEST_MODEL_PATH

    print(f"\n{'='*60}")
    print(f"  📊 TEST SETİ DEĞERLENDİRMESİ — Run #4 Binary")
    print(f"{'='*60}")
    print(f"  Model: {model_path}")
    print(f"  Cihaz: {DEVICE}")
    print(f"  Sınıflar: {model_cfg.CLASS_NAMES} ({model_cfg.NUM_CLASSES} sınıf)")

    # Model yükle
    model = DualPathDeepfakeDetector().to(DEVICE)
    if model_path.exists():
        ckpt = torch.load(str(model_path), map_location=DEVICE, weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state)
        train_auc = ckpt.get("val_auc", "?")
        train_epoch = ckpt.get("epoch", "?")
        print(f"  ✅ Model yüklendi — Eğitim Best AUC: {train_auc}, Epoch: {train_epoch}")
    else:
        print(f"  ❌ Model dosyası bulunamadı: {model_path}")
        return

    # Test DataLoader
    print(f"\n  Veri seti yükleniyor...")
    _, _, test_loader = get_dataloaders(batch_size=32)
    test_size = len(test_loader.dataset)
    print(f"  Test seti: {test_size} örnek")

    # Inference
    model.eval()
    all_labels, all_preds, all_probs = [], [], []

    print(f"\n  🔄 Test inference başlıyor...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="  Test", leave=True):
            rgb, freq, mesh, labels, source_tags = batch
            rgb, freq = rgb.to(DEVICE), freq.to(DEVICE)
            mesh, labels = mesh.to(DEVICE), labels.to(DEVICE)

            logits = model(rgb, freq, mesh)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    # ═══════════════════════════════════════════════════════════
    # METRİKLER
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"  📊 TEST SONUÇLARI")
    print(f"{'='*60}")

    # Genel metrikler
    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    macro_precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    macro_recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)

    # Binary AUC (FAKE olasılığı üzerinden)
    auc = roc_auc_score(all_labels, all_probs[:, 1])

    # EER
    eer, eer_threshold = compute_eer(all_labels, all_probs[:, 1])

    print(f"\n  ┌─────────────────────────────────────────┐")
    print(f"  │  ANA METRİKLER                          │")
    print(f"  ├─────────────────────────────────────────┤")
    print(f"  │  AUC-ROC       : {auc:.6f}              │")
    print(f"  │  Accuracy      : {accuracy:.6f} ({accuracy*100:.2f}%)    │")
    print(f"  │  F1 (macro)    : {macro_f1:.6f}              │")
    print(f"  │  Precision (m) : {macro_precision:.6f}              │")
    print(f"  │  Recall (m)    : {macro_recall:.6f}              │")
    print(f"  │  EER           : {eer:.6f} (t={eer_threshold:.4f}) │")
    print(f"  │  Test Örnekleri: {test_size:,}                   │")
    print(f"  └─────────────────────────────────────────┘")

    # Per-class rapor
    print(f"\n  📋 Per-Class Rapor:")
    print(f"  {'─'*50}")
    report = classification_report(
        all_labels, all_preds,
        labels=[0, 1],
        target_names=model_cfg.CLASS_NAMES,
        digits=4
    )
    # Indent
    for line in report.split("\n"):
        print(f"  {line}")

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    print(f"\n  🔢 Confusion Matrix:")
    print(f"  {'─'*40}")
    print(f"              Predicted")
    print(f"              REAL    FAKE")
    print(f"  Actual REAL  {cm[0][0]:>6}  {cm[0][1]:>6}")
    print(f"  Actual FAKE  {cm[1][0]:>6}  {cm[1][1]:>6}")

    # Türetilmiş metrikler
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr_val = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr_val = fn / (fn + tp) if (fn + tp) > 0 else 0

    print(f"\n  📈 Türetilmiş Metrikler:")
    print(f"  {'─'*40}")
    print(f"  Sensitivity (FAKE Recall) : {sensitivity:.4f}")
    print(f"  Specificity (REAL Recall) : {specificity:.4f}")
    print(f"  False Positive Rate       : {fpr_val:.4f}")
    print(f"  False Negative Rate       : {fnr_val:.4f}")
    print(f"  True Positives (FAKE→FAKE): {tp:,}")
    print(f"  True Negatives (REAL→REAL): {tn:,}")
    print(f"  False Positives (REAL→FAKE): {fp:,}")
    print(f"  False Negatives (FAKE→REAL): {fn:,}")

    # Sınıf dağılımı
    print(f"\n  📊 Test Seti Sınıf Dağılımı:")
    print(f"  {'─'*40}")
    label_counts = Counter(all_labels)
    for cls_id in sorted(label_counts.keys()):
        name = model_cfg.CLASS_NAMES[int(cls_id)]
        count = label_counts[cls_id]
        pct = count / len(all_labels) * 100
        print(f"  {name}: {count:,} ({pct:.1f}%)")

    # Sonuçları dosyaya kaydet
    results_dir = paths.BASE_DIR / "logs" / "run4_binary"
    results_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "model_path": str(model_path),
        "test_size": test_size,
        "auc": float(auc),
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "eer": float(eer),
        "eer_threshold": float(eer_threshold),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "fpr": float(fpr_val),
        "fnr": float(fnr_val),
        "confusion_matrix": cm.tolist(),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }

    import json
    results_path = results_dir / "test_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  💾 Sonuçlar kaydedildi: {results_path}")

    # Confusion matrix numpy
    np.save(results_dir / "test_confusion_matrix.npy", cm)

    # ROC Curve verileri kaydet
    fpr_roc, tpr_roc, thresholds_roc = roc_curve(all_labels, all_probs[:, 1])
    np.savez(
        results_dir / "test_roc_curve.npz",
        fpr=fpr_roc, tpr=tpr_roc, thresholds=thresholds_roc
    )
    print(f"  💾 ROC curve verileri kaydedildi: {results_dir / 'test_roc_curve.npz'}")

    # MLflow loglama
    try:
        import mlflow
        mlflow.set_tracking_uri(f"file:{paths.MLRUNS_DIR}")
        mlflow.set_experiment("deepfake-v4-binary-50-50")
        with mlflow.start_run(run_name="test_evaluation"):
            for k, v in results.items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(f"test_{k}", v)
            mlflow.log_artifact(str(results_path))
    except Exception as e:
        print(f"  ⚠️ MLflow loglama atlandı: {e}")

    print(f"\n{'='*60}")
    print(f"  ✅ Test değerlendirmesi tamamlandı!")
    print(f"  🏆 AUC: {auc:.4f} | Acc: {accuracy:.4f} | F1: {macro_f1:.4f} | EER: {eer:.4f}")
    print(f"{'='*60}\n")

    return results


if __name__ == "__main__":
    model_path = sys.argv[1] if len(sys.argv) > 1 else None
    run_test_evaluation(model_path)
