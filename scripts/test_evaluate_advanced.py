"""
Run #4 Binary — Gelismis Test Degerlendirmesi
4 mod: Baseline, TTA Only, Ensemble Only, TTA + Ensemble

Kullanim:
  python scripts/test_evaluate_advanced.py                # Tum modlar
  python scripts/test_evaluate_advanced.py --mode baseline
  python scripts/test_evaluate_advanced.py --mode tta
  python scripts/test_evaluate_advanced.py --mode ensemble
  python scripts/test_evaluate_advanced.py --mode tta_ensemble
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import json
import time
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
except ImportError:
    print("sklearn yuklu degil!")
    sys.exit(1)


# ================================================================
# YARDIMCI FONKSIYONLAR
# ================================================================
def compute_eer(labels, scores):
    """Equal Error Rate hesapla."""
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = float((fpr[idx] + fnr[idx]) / 2)
    eer_threshold = float(thresholds[idx])
    return eer, eer_threshold


def compute_all_metrics(labels, predictions, probabilities):
    """Tum metrikleri hesapla."""
    accuracy = accuracy_score(labels, predictions)
    macro_f1 = f1_score(labels, predictions, average="macro", zero_division=0)
    macro_precision = precision_score(labels, predictions, average="macro", zero_division=0)
    macro_recall = recall_score(labels, predictions, average="macro", zero_division=0)
    auc = roc_auc_score(labels, probabilities[:, 1])
    eer, eer_threshold = compute_eer(labels, probabilities[:, 1])
    cm = confusion_matrix(labels, predictions, labels=[0, 1])

    return {
        "auc": float(auc),
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "eer": float(eer),
        "eer_threshold": float(eer_threshold),
        "confusion_matrix": cm.tolist(),
    }


def print_metrics(name: str, metrics: dict, elapsed: float):
    """Metrikleri guzel formatla yazdir."""
    print(f"\n  {'='*55}")
    print(f"  {name}")
    print(f"  {'='*55}")
    print(f"  AUC-ROC       : {metrics['auc']:.6f}")
    print(f"  Accuracy      : {metrics['accuracy']:.6f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  F1 (macro)    : {metrics['macro_f1']:.6f}")
    print(f"  Precision (m) : {metrics['macro_precision']:.6f}")
    print(f"  Recall (m)    : {metrics['macro_recall']:.6f}")
    print(f"  EER           : {metrics['eer']:.6f}")
    cm = metrics["confusion_matrix"]
    print(f"  CM: REAL [{cm[0][0]:>5}, {cm[0][1]:>5}]  FAKE [{cm[1][0]:>5}, {cm[1][1]:>5}]")
    print(f"  Sure          : {elapsed:.1f} saniye")
    print(f"  {'='*55}")


# ================================================================
# MOD 1: BASELINE — tek model, augmentasyon yok
# ================================================================
def evaluate_baseline(model_path: str, test_loader, device=DEVICE):
    """Baseline degerlendirme — tek model, TTA yok."""
    print("\n  >> Mod 1: BASELINE (tek model, TTA yok)")

    model = DualPathDeepfakeDetector().to(device)
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    model.eval()

    all_labels, all_probs = [], []
    t0 = time.time()

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="    Baseline", leave=True):
            rgb, freq, mesh, labels, _ = batch
            rgb, freq, mesh = rgb.to(device), freq.to(device), mesh.to(device)
            logits = model(rgb, freq, mesh)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_labels.extend(labels.numpy())
            all_probs.extend(probs)

    elapsed = time.time() - t0
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_preds = all_probs.argmax(axis=1)

    metrics = compute_all_metrics(all_labels, all_preds, all_probs)
    metrics["elapsed_seconds"] = elapsed
    print_metrics("BASELINE", metrics, elapsed)

    # GPU temizle
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return metrics


# ================================================================
# MOD 2: TTA ONLY — tek model, 8 augmentasyon
# ================================================================
def evaluate_tta_only(model_path: str, test_loader, device=DEVICE, n_aug: int = 8):
    """TTA degerlendirme — tek model, 8 augmentasyon."""
    print(f"\n  >> Mod 2: TTA ONLY (tek model, {n_aug} augmentasyon)")

    model = DualPathDeepfakeDetector().to(device)
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    model.eval()

    from inference.tta_inference import batch_tta_evaluate
    t0 = time.time()

    result = batch_tta_evaluate(model, tqdm(test_loader, desc="    TTA", leave=True),
                                device, n_aug)

    elapsed = time.time() - t0
    metrics = compute_all_metrics(result["labels"], result["predictions"],
                                  result["probabilities"])
    metrics["elapsed_seconds"] = elapsed
    metrics["mean_std"] = float(result["per_sample_std"].mean())
    print_metrics(f"TTA (n={n_aug})", metrics, elapsed)

    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return metrics


# ================================================================
# MOD 3: ENSEMBLE ONLY — 5 checkpoint, TTA yok
# ================================================================
def evaluate_ensemble_only(test_loader, device=DEVICE):
    """Checkpoint ensemble — 5 model, TTA yok."""
    from inference.model_ensemble import SequentialEnsemble, get_default_ensemble_config

    config = get_default_ensemble_config()
    print(f"\n  >> Mod 3: ENSEMBLE ONLY ({config['n_models']} checkpoint, TTA yok)")

    ensemble = SequentialEnsemble(
        model_paths=config["paths"],
        weights=config["weights"],
        device=device,
    )

    t0 = time.time()
    result = ensemble.evaluate_sequential(test_loader, verbose=True)
    elapsed = time.time() - t0

    metrics = compute_all_metrics(result["labels"], result["predictions"],
                                  result["probabilities"])
    metrics["elapsed_seconds"] = elapsed
    metrics["n_models"] = config["n_models"]
    print_metrics(f"ENSEMBLE ({config['n_models']} model)", metrics, elapsed)

    return metrics


# ================================================================
# MOD 4: TTA + ENSEMBLE — 5 checkpoint x 8 augmentasyon
# ================================================================
def evaluate_tta_ensemble(test_loader, device=DEVICE, n_aug: int = 8):
    """TTA + Ensemble kombine degerlendirme."""
    from inference.model_ensemble import TTAEnsemble, get_default_ensemble_config

    config = get_default_ensemble_config()
    n_models = config["n_models"]
    total = n_models * n_aug
    print(f"\n  >> Mod 4: TTA + ENSEMBLE ({n_models} model x {n_aug} TTA = {total} tahmin/ornek)")

    tta_ens = TTAEnsemble(
        model_paths=config["paths"],
        weights=config["weights"],
        n_augmentations=n_aug,
        device=device,
    )

    t0 = time.time()
    result = tta_ens.evaluate_tta_ensemble(test_loader, verbose=True)
    elapsed = time.time() - t0

    metrics = compute_all_metrics(result["labels"], result["predictions"],
                                  result["probabilities"])
    metrics["elapsed_seconds"] = elapsed
    metrics["n_models"] = n_models
    metrics["n_augmentations"] = n_aug
    metrics["total_predictions_per_sample"] = total
    print_metrics(f"TTA+ENSEMBLE ({total} pred/sample)", metrics, elapsed)

    return metrics


# ================================================================
# ANA FONKSIYON
# ================================================================
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Gelismis Test Degerlendirmesi")
    parser.add_argument("--mode", type=str, default="all",
                        choices=["all", "baseline", "tta", "ensemble", "tta_ensemble"],
                        help="Degerlendirme modu")
    parser.add_argument("--n-aug", type=int, default=8,
                        help="TTA augmentasyon sayisi")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size")
    args = parser.parse_args()

    print(f"\n{'#'*60}")
    print(f"  GELISMIS TEST DEGERLENDIRMESI — Run #4 Binary")
    print(f"  Cihaz: {DEVICE}")
    print(f"  Mod: {args.mode}")
    print(f"{'#'*60}")

    # Best model yolu
    best_model = str(paths.MODEL_DIR / "best_run4_binary.pth")
    if not Path(best_model).exists():
        best_model = str(paths.BEST_MODEL_PATH)

    # Test DataLoader (bir kez yukle)
    print("\n  Veri seti yukleniyor...")
    _, _, test_loader = get_dataloaders(batch_size=args.batch_size)
    test_size = len(test_loader.dataset)
    print(f"  Test seti: {test_size:,} ornek\n")

    results = {}

    # ── MOD CALISTIRMA ──
    if args.mode in ("all", "baseline"):
        results["baseline"] = evaluate_baseline(best_model, test_loader)

    if args.mode in ("all", "tta"):
        results["tta"] = evaluate_tta_only(best_model, test_loader, n_aug=args.n_aug)

    if args.mode in ("all", "ensemble"):
        results["ensemble"] = evaluate_ensemble_only(test_loader)

    if args.mode in ("all", "tta_ensemble"):
        results["tta_ensemble"] = evaluate_tta_ensemble(test_loader, n_aug=args.n_aug)

    # ── KARSILASTIRMA TABLOSU ──
    if len(results) > 1:
        print(f"\n\n{'='*70}")
        print(f"  KARSILASTIRMA TABLOSU")
        print(f"{'='*70}")
        print(f"  {'Mod':<20} {'AUC':>10} {'Acc':>10} {'F1':>10} {'EER':>10} {'Sure(s)':>10}")
        print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

        for name, m in results.items():
            auc_str = f"{m['auc']:.6f}"
            acc_str = f"{m['accuracy']*100:.2f}%"
            f1_str = f"{m['macro_f1']:.6f}"
            eer_str = f"{m['eer']:.6f}"
            time_str = f"{m.get('elapsed_seconds', 0):.1f}"
            print(f"  {name:<20} {auc_str:>10} {acc_str:>10} {f1_str:>10} {eer_str:>10} {time_str:>10}")

        # En iyi mod
        best_mode = max(results.keys(), key=lambda k: results[k]["auc"])
        best_auc = results[best_mode]["auc"]
        baseline_auc = results.get("baseline", {}).get("auc", 0)
        improvement = best_auc - baseline_auc if baseline_auc > 0 else 0

        print(f"\n  En iyi: {best_mode} (AUC: {best_auc:.6f})")
        if improvement > 0:
            print(f"  Baseline'a gore iyilesme: +{improvement:.6f} AUC")
        print(f"{'='*70}\n")

    # ── SONUCLARI KAYDET ──
    results_dir = paths.BASE_DIR / "logs" / "run4_binary"
    results_dir.mkdir(parents=True, exist_ok=True)

    # JSON'a donusturulemeyen tipleri temizle
    clean_results = {}
    for mode_name, mode_metrics in results.items():
        clean = {}
        for k, v in mode_metrics.items():
            if isinstance(v, (int, float, str, bool, list)):
                clean[k] = v
            elif isinstance(v, np.floating):
                clean[k] = float(v)
            elif isinstance(v, np.integer):
                clean[k] = int(v)
            elif isinstance(v, np.ndarray):
                clean[k] = v.tolist()
        clean_results[mode_name] = clean

    results_path = results_dir / "test_advanced_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(clean_results, f, indent=2, ensure_ascii=False)
    print(f"  Sonuclar kaydedildi: {results_path}")

    return results


if __name__ == "__main__":
    main()
