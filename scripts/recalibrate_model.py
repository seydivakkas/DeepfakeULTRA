"""
DeepfakeULTRA — Bağımsız Model Recalibrasyon Scripti
Eğitim yapmadan mevcut modeli yeniden kalibre eder.

3 yöntem dener: Temperature Scaling, Platt Scaling, Vector Scaling.
En iyi yöntemi otomatik seçer ve kaydeder.

Kullanım:
    # Otomatik en iyi yöntemi seç
    python scripts/recalibrate_model.py

    # Belirli checkpoint ile
    python scripts/recalibrate_model.py --checkpoint models/best_model.pth

    # Belirli yöntem ile
    python scripts/recalibrate_model.py --method platt_scaling

    # Sonuçları test setinde doğrula
    python scripts/recalibrate_model.py --verify
"""
import sys
import json
import numpy as np
import torch
from pathlib import Path

# Windows cp1254 encoding sorunu: emoji karakterleri icin UTF-8 zorla
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import model_cfg, paths, DEVICE


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
        print(f"  ❌ Checkpoint bulunamadı: {cp}")
        sys.exit(1)
    model.to(DEVICE).eval()
    return model


def collect_logits_and_labels(model, loader, desc="Logitler toplanıyor"):
    """DataLoader'dan tüm logit ve etiketleri topla."""
    from tqdm import tqdm
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=desc):
            rgb, freq, mesh, labels, _ = batch
            rgb, freq, mesh = rgb.to(DEVICE), freq.to(DEVICE), mesh.to(DEVICE)
            logits = model(rgb, freq, mesh)
            all_logits.append(logits.cpu())
            all_labels.append(labels)

    return torch.cat(all_logits, dim=0), torch.cat(all_labels, dim=0)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Model Recalibrasyon")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Model checkpoint yolu")
    parser.add_argument("--method", type=str, default="auto",
                        choices=["auto", "temperature_scaling", "platt_scaling", "vector_scaling"],
                        help="Kalibrasyon yöntemi (varsayılan: auto)")
    parser.add_argument("--verify", action="store_true",
                        help="Test setinde sonuçları doğrula")
    args = parser.parse_args()

    print("=" * 60)
    print("  🔄 DeepfakeULTRA — Model Recalibrasyon")
    print("=" * 60)

    from core.calibration import (
        TemperatureScaling, PlattScaling, VectorScaling,
        auto_calibrate, compute_ece, compute_brier_score,
        generate_reliability_diagram,
    )
    from core.data_pipeline import get_dataloaders

    # Model yükle
    model = load_model(args.checkpoint)

    # DataLoader'lar
    print("\n  📂 Veri setleri yükleniyor...")
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=model_cfg.BATCH_SIZE
    )

    # === KALİBRASYON ===
    if args.method == "auto":
        print("\n  🔄 Auto-Calibrate: Tüm yöntemler deneniyor...")
        best_method, calibrator, cal_results = auto_calibrate(
            val_loader, model, DEVICE
        )
    else:
        print(f"\n  🔧 Yöntem: {args.method}")
        if args.method == "temperature_scaling":
            calibrator = TemperatureScaling()
            calibrator.fit(val_loader, model, DEVICE)
        elif args.method == "platt_scaling":
            calibrator = PlattScaling()
            calibrator.fit(val_loader, model, DEVICE)
        elif args.method == "vector_scaling":
            calibrator = VectorScaling(num_classes=model_cfg.NUM_CLASSES)
            calibrator.fit(val_loader, model, DEVICE)
        best_method = args.method
        cal_results = None

    if calibrator is None:
        print("  ❌ Kalibrasyon başarısız — hiçbir yöntem iyileşme sağlamadı")
        return

    # Kaydet
    calibrator.save()
    print(f"\n  💾 Kalibrasyon kaydedildi: {paths.MODEL_DIR / 'calibration_weights.json'}")

    # === DOĞRULAMA (Test Seti) ===
    if args.verify:
        print("\n" + "=" * 60)
        print("  📊 Test Seti Doğrulaması")
        print("=" * 60)

        test_logits, test_labels = collect_logits_and_labels(
            model, test_loader, desc="Test logitleri"
        )
        y_true = test_labels.numpy()

        # Kalibre edilmemiş
        uncal_probs = torch.softmax(test_logits, dim=1).numpy()[:, 1]
        uncal_ece = compute_ece(y_true, uncal_probs)
        uncal_brier = compute_brier_score(y_true, uncal_probs)

        # Kalibre edilmiş
        cal_probs = calibrator.calibrate(test_logits).numpy()[:, 1]
        cal_ece = compute_ece(y_true, cal_probs)
        cal_brier = compute_brier_score(y_true, cal_probs)

        improvement = (uncal_ece - cal_ece) / uncal_ece * 100 if uncal_ece > 0 else 0

        print(f"\n  {'Metrik':<20} {'Önce':>10} {'Sonra':>10} {'İyileşme':>10}")
        print(f"  {'-' * 50}")
        print(f"  {'ECE':<20} {uncal_ece:>10.4f} {cal_ece:>10.4f} {improvement:>+9.1f}%")
        print(f"  {'Brier Score':<20} {uncal_brier:>10.4f} {cal_brier:>10.4f} "
              f"{(uncal_brier - cal_brier) / uncal_brier * 100:>+9.1f}%")

        # Olasılık dağılımı karşılaştırması
        real_mask = y_true == 0
        fake_mask = y_true == 1

        print(f"\n  Olasılık Dağılımı (FAKE prob):")
        print(f"  {'Sınıf':<10} {'Önce (mean±std)':<25} {'Sonra (mean±std)':<25}")
        print(f"  {'-' * 60}")
        print(f"  {'REAL':<10} {uncal_probs[real_mask].mean():.4f}±{uncal_probs[real_mask].std():.4f}"
              f"{'':>10}"
              f"{cal_probs[real_mask].mean():.4f}±{cal_probs[real_mask].std():.4f}")
        print(f"  {'FAKE':<10} {uncal_probs[fake_mask].mean():.4f}±{uncal_probs[fake_mask].std():.4f}"
              f"{'':>10}"
              f"{cal_probs[fake_mask].mean():.4f}±{cal_probs[fake_mask].std():.4f}")

        # Eşik optimizasyonu (kalibre edilmiş olasılıklar ile)
        from sklearn.metrics import roc_curve, roc_auc_score
        fpr, tpr, thresholds = roc_curve(y_true, cal_probs)
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        cal_threshold = thresholds[best_idx]
        cal_auc = roc_auc_score(y_true, cal_probs)

        y_pred_cal = (cal_probs >= cal_threshold).astype(int)
        cal_acc = (y_pred_cal == y_true).mean()
        fp_cal = ((y_pred_cal == 1) & (y_true == 0)).sum()
        fn_cal = ((y_pred_cal == 0) & (y_true == 1)).sum()

        print(f"\n  Kalibre Edilmiş Karar Eşiği:")
        print(f"    Optimal Threshold: {cal_threshold:.4f}")
        print(f"    AUC: {cal_auc:.4f}")
        print(f"    Accuracy: {cal_acc:.4f}")
        print(f"    FP: {fp_cal} | FN: {fn_cal}")

        # Reliability diagram (öncesi ve sonrası)
        eval_dir = paths.BASE_DIR / "evaluation"
        eval_dir.mkdir(parents=True, exist_ok=True)

        generate_reliability_diagram(
            y_true, uncal_probs,
            output_path=eval_dir / "reliability_diagram.png",
            title="Reliability Diagram (Uncalibrated)",
        )
        generate_reliability_diagram(
            y_true, cal_probs,
            output_path=eval_dir / "reliability_diagram_calibrated.png",
            title=f"Reliability Diagram ({best_method})",
        )

        # Güncellenmiş metrikleri kaydet
        metrics_path = eval_dir / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path, encoding="utf-8") as f:
                metrics = json.load(f)
        else:
            metrics = {}

        metrics.update({
            "calibration_method": best_method,
            "calibrated_ece": round(cal_ece, 4),
            "calibrated_brier": round(cal_brier, 4),
            "calibrated_threshold": round(float(cal_threshold), 4),
            "calibrated_accuracy": round(float(cal_acc), 4),
            "calibrated_fp": int(fp_cal),
            "calibrated_fn": int(fn_cal),
        })

        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

        print(f"\n  📄 Metrikler güncellendi: {metrics_path}")

        # Başarı değerlendirmesi
        target_ece = model_cfg.CALIBRATION_TARGET_ECE
        if cal_ece < target_ece:
            print(f"\n  ✅ Kalibrasyon BAŞARILI! ECE={cal_ece:.4f} < hedef {target_ece}")
        else:
            print(f"\n  ⚠️ Kalibrasyon iyileşti ama hedefin ({target_ece}) üstünde: "
                  f"ECE={cal_ece:.4f}")

    print(f"\n{'=' * 60}")
    print(f"  ✅ Recalibrasyon tamamlandı!")
    print(f"  Yöntem: {best_method}")
    print(f"  Dosya: {paths.MODEL_DIR / 'calibration_weights.json'}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
