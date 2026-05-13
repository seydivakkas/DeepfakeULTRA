"""
Faz 2 — Jüri Test Seti: Custom Team Evaluation
Egitim setinden TAMAMEN bagimsiz, el secme zorlu gorseller ile modeli degerlendirme.

Jüri Test Seti Yapisi:
    dataset/jury_test/
    ├── real/
    │   ├── celeb_df/        → CelebDF gercek yuzler
    │   ├── custom_team/     → Ogrenci/hoca gercek yuzler
    │   ├── wild/            → Internetten rastgele yuzler
    │   └── compressed/      → WhatsApp/Instagram sikistirilmis
    └── fake/
        ├── celeb_df/        → CelebDF deepfake'ler
        ├── dfdc/            → DFDC deepfake'ler
        ├── custom_swap/     → Kendi face swap'lerimiz
        └── ai_generated/    → SD/DALL-E/MJ uretimi

Kullanim:
    python scripts/jury_evaluation.py
    python scripts/jury_evaluation.py --model models/best_run4_binary.pth
    python scripts/jury_evaluation.py --create-structure   # Bos dizin yapis olustur
"""

import sys
import os
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# Proje root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from PIL import Image
from torchvision import transforms

from config import model_cfg, paths, DEVICE

try:
    from sklearn.metrics import (
        roc_auc_score, accuracy_score, f1_score,
        precision_score, recall_score, confusion_matrix, roc_curve
    )
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# ═══════════════════════════════════════════════════════════
# JURY TEST SET YAPISI
# ═══════════════════════════════════════════════════════════

JURY_DIR = paths.DATASET_DIR / "jury_test"

# Beklenen dizin yapisi ve aciklamalari
JURY_STRUCTURE = {
    "real": {
        "celeb_df": "CelebDF gercek yuzler (egitimde gormedigi)",
        "custom_team": "Kendi ekibimizin gercek yuzleri",
        "wild": "Internetten rastgele indirilmis yuzler",
        "compressed": "Agir sikistirma gecmis yuzler (WhatsApp, TikTok)",
    },
    "fake": {
        "celeb_df": "CelebDF deepfake gorseller",
        "dfdc": "DeepFake Detection Challenge gorselleri",
        "custom_swap": "Kendi olusturdugumuz face swap'ler",
        "ai_generated": "Stable Diffusion / DALL-E / Midjourney uretimi",
    },
}

SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def create_jury_structure():
    """Juri test seti dizin yapisini olustur."""
    print("📂 Jüri Test Seti dizin yapısı oluşturuluyor...\n")
    for label, sources in JURY_STRUCTURE.items():
        for source, desc in sources.items():
            d = JURY_DIR / label / source
            d.mkdir(parents=True, exist_ok=True)
            print(f"  ✅ {d.relative_to(paths.DATASET_DIR)}/  — {desc}")

    print(f"\n📍 Konum: {JURY_DIR}")
    print("ℹ️  Bu dizinlere görsel ekleyip evaluation çalıştırın.")


def scan_jury_dataset() -> dict:
    """Juri veri setini tara, dosya sayilarini dondur."""
    stats = {}
    for label in ["real", "fake"]:
        label_dir = JURY_DIR / label
        if not label_dir.exists():
            continue
        for source_dir in sorted(label_dir.iterdir()):
            if not source_dir.is_dir():
                continue
            count = sum(
                1 for f in source_dir.rglob("*")
                if f.is_file() and f.suffix.lower() in SUPPORTED_FORMATS
            )
            key = f"{label}/{source_dir.name}"
            stats[key] = count

    return stats


# ═══════════════════════════════════════════════════════════
# MODEL YUKLEME VE INFERENCE
# ═══════════════════════════════════════════════════════════

def load_model(model_path: str = None):
    """En iyi modeli yukle."""
    from core.dual_mobilenetv3 import DualPathDeepfakeDetector

    model_path = model_path or str(paths.BEST_MODEL_PATH)
    model = DualPathDeepfakeDetector().to(DEVICE)

    if Path(model_path).exists():
        ckpt = torch.load(model_path, map_location=DEVICE, weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state, strict=False)
        epoch = ckpt.get("epoch", "?")
        auc = ckpt.get("val_auc", "?")
        print(f"✅ Model yüklendi: {Path(model_path).name} (epoch={epoch}, AUC={auc})")
    else:
        print(f"⚠️ Model bulunamadı: {model_path}")

    model.eval()
    return model


def predict_single(model, image_path: str, dwt_extractor, mesh_extractor, transform, threshold=0.5) -> dict:
    """Tek gorsel icin tahmin yap."""
    try:
        image = Image.open(image_path).convert("RGB")
        img_np = np.array(image.resize((224, 224)))

        # DWT frekans
        freq_map = dwt_extractor(img_np)
        freq_tensor = torch.from_numpy(freq_map).float().unsqueeze(0).to(DEVICE)

        # Face Mesh
        mesh = mesh_extractor(img_np)
        mesh_tensor = torch.from_numpy(mesh).float().unsqueeze(0).to(DEVICE)

        # RGB transform
        rgb_tensor = transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = model(rgb_tensor, freq_tensor, mesh_tensor)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

        # Kalibrasyon: optimal threshold kullan
        p_fake = float(probs[1])
        prediction = 1 if p_fake >= threshold else 0

        return {
            "prediction": prediction,
            "prob_real": float(probs[0]),
            "prob_fake": p_fake,
            "confidence": float(max(probs)),
        }
    except Exception as e:
        return {
            "prediction": -1,
            "prob_real": 0.5,
            "prob_fake": 0.5,
            "confidence": 0.0,
            "error": str(e),
        }


# ═══════════════════════════════════════════════════════════
# ANA EVALUATION FONKSİYONU
# ═══════════════════════════════════════════════════════════

def run_jury_evaluation(model_path: str = None, verbose: bool = True) -> dict:
    """
    Juri test seti uzerinde tam evaluation.

    Returns:
        dict: {
            'overall': {accuracy, auc, f1, precision, recall, eer},
            'per_source': {source_key: {accuracy, auc, f1, count}},
            'confusion_matrix': [[TN, FP], [FN, TP]],
            'predictions': [{path, label, pred, prob_fake, source}],
        }
    """
    # Veri seti kontrolu
    stats = scan_jury_dataset()
    total_images = sum(stats.values())

    if total_images == 0:
        print("❌ Jüri test setinde görsel bulunamadı!")
        print(f"   Konum: {JURY_DIR}")
        print("   Yapıyı oluşturmak için: python scripts/jury_evaluation.py --create-structure")
        return {}

    if verbose:
        print(f"\n{'='*60}")
        print(f"  📋 JÜRI TEST SETİ EVALUATION")
        print(f"{'='*60}")
        print(f"\n📊 Veri seti dağılımı:")
        for source, count in sorted(stats.items()):
            emoji = "🟢" if source.startswith("real") else "🔴"
            print(f"  {emoji} {source}: {count} görsel")
        print(f"  ─────────────────────")
        print(f"  📎 Toplam: {total_images} görsel\n")

    # Model yukle
    model = load_model(model_path)

    # Frekans extractor — config'e göre 12ch veya 18ch
    from core.data_pipeline import MultiScaleDWT, FaceMeshExtractor
    if getattr(model_cfg, 'USE_HYBRID_FREQ', False):
        try:
            from core.frequency_v2 import HybridFrequencyExtractor
            dwt_extractor = HybridFrequencyExtractor(
                wavelets=model_cfg.DWT_WAVELETS,
                size=model_cfg.IMG_SIZE,
            )
        except ImportError:
            dwt_extractor = MultiScaleDWT()
    else:
        dwt_extractor = MultiScaleDWT()
    mesh_extractor = FaceMeshExtractor()

    # Transform (val/test ile ayni)
    transform = transforms.Compose([
        transforms.Resize((model_cfg.IMG_SIZE, model_cfg.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Optimal threshold yükle (varsa)
    threshold = 0.5
    thresh_path = paths.MODEL_DIR / "optimal_threshold.txt"
    if thresh_path.exists():
        try:
            threshold = float(thresh_path.read_text().strip())
            if verbose:
                print(f"  📐 Optimal threshold yüklendi: {threshold:.4f}")
        except Exception:
            pass

    # Tahmin toplama
    all_labels = []
    all_preds = []
    all_probs = []
    predictions_log = []
    source_results = defaultdict(lambda: {"labels": [], "preds": [], "probs": []})

    # Tum gorselleri tara
    for label_name, label_id in [("real", 0), ("fake", 1)]:
        label_dir = JURY_DIR / label_name
        if not label_dir.exists():
            continue

        for source_dir in sorted(label_dir.iterdir()):
            if not source_dir.is_dir() or source_dir.name == ".cache":
                continue

            source_key = f"{label_name}/{source_dir.name}"
            images = [
                f for f in sorted(source_dir.rglob("*"))
                if f.is_file() and f.suffix.lower() in SUPPORTED_FORMATS
            ]

            if verbose and images:
                print(f"  ⏳ {source_key} — {len(images)} görsel işleniyor...")

            for img_path in images:
                result = predict_single(
                    model, str(img_path), dwt_extractor, mesh_extractor, transform,
                    threshold=threshold
                )

                if result.get("prediction", -1) == -1:
                    continue  # Hatali gorsel, atla

                all_labels.append(label_id)
                all_preds.append(result["prediction"])
                all_probs.append(result["prob_fake"])

                source_results[source_key]["labels"].append(label_id)
                source_results[source_key]["preds"].append(result["prediction"])
                source_results[source_key]["probs"].append(result["prob_fake"])

                predictions_log.append({
                    "path": str(img_path.relative_to(JURY_DIR)),
                    "label": label_name.upper(),
                    "prediction": "FAKE" if result["prediction"] == 1 else "REAL",
                    "prob_fake": round(result["prob_fake"], 4),
                    "confidence": round(result["confidence"], 4),
                    "source": source_key,
                    "correct": result["prediction"] == label_id,
                })

    if not all_labels:
        print("❌ Hiç geçerli tahmin üretilemedi!")
        return {}

    # ═══════════════════════════════════════════════════════
    # METRİK HESAPLAMA
    # ═══════════════════════════════════════════════════════
    results = {"timestamp": datetime.now().isoformat()}

    if HAS_SKLEARN:
        # Overall metrikler
        results["overall"] = {
            "accuracy": float(accuracy_score(all_labels, all_preds)),
            "f1": float(f1_score(all_labels, all_preds, average="binary", pos_label=1)),
            "precision": float(precision_score(all_labels, all_preds, pos_label=1, zero_division=0)),
            "recall": float(recall_score(all_labels, all_preds, pos_label=1, zero_division=0)),
        }

        # AUC
        try:
            results["overall"]["auc"] = float(roc_auc_score(all_labels, all_probs))
        except Exception:
            results["overall"]["auc"] = 0.5

        # EER
        try:
            fpr, tpr, _ = roc_curve(all_labels, all_probs)
            fnr = 1 - tpr
            idx = np.nanargmin(np.abs(fpr - fnr))
            results["overall"]["eer"] = float(fpr[idx])
        except Exception:
            results["overall"]["eer"] = 0.5

        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
        results["confusion_matrix"] = cm.tolist()

        # Per-source metrikler
        results["per_source"] = {}
        for source_key, data in sorted(source_results.items()):
            source_metrics = {
                "count": len(data["labels"]),
                "accuracy": float(accuracy_score(data["labels"], data["preds"])),
            }
            if len(set(data["labels"])) > 1:
                try:
                    source_metrics["auc"] = float(roc_auc_score(data["labels"], data["probs"]))
                except Exception:
                    pass
            try:
                source_metrics["f1"] = float(
                    f1_score(data["labels"], data["preds"], average="binary",
                             pos_label=1, zero_division=0)
                )
            except Exception:
                pass
            results["per_source"][source_key] = source_metrics

    # Hatali tahminler
    errors = [p for p in predictions_log if not p["correct"]]
    results["error_count"] = len(errors)
    results["total_evaluated"] = len(predictions_log)

    # ═══════════════════════════════════════════════════════
    # RAPORLAMA
    # ═══════════════════════════════════════════════════════
    if verbose:
        print(f"\n{'='*60}")
        print(f"  📊 JÜRI DEĞERLENDİRME SONUÇLARI")
        print(f"{'='*60}")

        ov = results.get("overall", {})
        print(f"\n  ▸ Accuracy:  {ov.get('accuracy', 0):.4f}")
        print(f"  ▸ AUC:       {ov.get('auc', 0):.4f}")
        print(f"  ▸ F1:        {ov.get('f1', 0):.4f}")
        print(f"  ▸ Precision: {ov.get('precision', 0):.4f}")
        print(f"  ▸ Recall:    {ov.get('recall', 0):.4f}")
        print(f"  ▸ EER:       {ov.get('eer', 0):.4f}")

        cm = results.get("confusion_matrix", [[0, 0], [0, 0]])
        print(f"\n  Confusion Matrix:")
        print(f"                  Pred REAL  Pred FAKE")
        print(f"    Gerçek REAL   {cm[0][0]:>8}  {cm[0][1]:>8}")
        print(f"    Gerçek FAKE   {cm[1][0]:>8}  {cm[1][1]:>8}")

        print(f"\n  📋 Kaynak Bazlı Sonuçlar:")
        for source, metrics in sorted(results.get("per_source", {}).items()):
            emoji = "🟢" if source.startswith("real") else "🔴"
            auc_str = f"AUC={metrics.get('auc', '-'):.3f}" if 'auc' in metrics else ""
            print(f"    {emoji} {source:30s} | Acc={metrics['accuracy']:.3f} "
                  f"F1={metrics.get('f1', 0):.3f} {auc_str} (n={metrics['count']})")

        if errors:
            print(f"\n  ❌ Yanlış tahminler: {len(errors)}/{len(predictions_log)}")
            for err in errors[:10]:
                print(f"    • {err['path']} — Gerçek: {err['label']}, "
                      f"Tahmin: {err['prediction']} (p_fake={err['prob_fake']:.3f})")
            if len(errors) > 10:
                print(f"    ... ve {len(errors) - 10} daha")

    # Sonuclari kaydet
    output_dir = paths.REPORTS_DIR / "jury"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"jury_evaluation_{timestamp}.json"

    # Detayli predictions log'u ayri kaydet
    results_with_log = {**results, "predictions": predictions_log}
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_with_log, f, indent=2, ensure_ascii=False)

    # En son sonucu da latest olarak kaydet
    latest_path = output_dir / "jury_latest.json"
    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump(results_with_log, f, indent=2, ensure_ascii=False)

    if verbose:
        print(f"\n  💾 Rapor kaydedildi: {json_path.relative_to(paths.BASE_DIR)}")
        print(f"  💾 Son rapor: {latest_path.relative_to(paths.BASE_DIR)}")

    return results


# ═══════════════════════════════════════════════════════════
# GRADIO UI ENTEGRASYONU
# ═══════════════════════════════════════════════════════════

def handle_jury_evaluation_ui() -> tuple:
    """Gradio UI'dan cagirilabilecek evaluation handler.

    Returns:
        (summary_markdown, per_source_markdown)
    """
    try:
        results = run_jury_evaluation(verbose=False)
    except Exception as e:
        return (f"❌ Hata: {e}", "")

    if not results:
        return (
            "❌ Jüri test setinde görsel bulunamadı!\n\n"
            f"📍 Konum: `{JURY_DIR}`\n\n"
            "Yapıyı oluşturmak için:\n"
            "```\npython scripts/jury_evaluation.py --create-structure\n```",
            ""
        )

    ov = results.get("overall", {})
    cm = results.get("confusion_matrix", [[0, 0], [0, 0]])

    summary = f"""## 📊 Jüri Değerlendirme Sonuçları

| Metrik | Değer |
|--------|-------|
| **Accuracy** | {ov.get('accuracy', 0):.4f} |
| **AUC** | {ov.get('auc', 0):.4f} |
| **F1** | {ov.get('f1', 0):.4f} |
| **Precision** | {ov.get('precision', 0):.4f} |
| **Recall** | {ov.get('recall', 0):.4f} |
| **EER** | {ov.get('eer', 0):.4f} |

### Confusion Matrix
|  | Pred REAL | Pred FAKE |
|--|----------|-----------|
| **Gerçek REAL** | {cm[0][0]} | {cm[0][1]} |
| **Gerçek FAKE** | {cm[1][0]} | {cm[1][1]} |

**Toplam:** {results.get('total_evaluated', 0)} görsel, {results.get('error_count', 0)} hatalı tahmin
"""

    # Per-source tablo
    per_source_lines = ["## 📋 Kaynak Bazlı Sonuçlar\n",
                        "| Kaynak | Adet | Accuracy | F1 | AUC |",
                        "|--------|------|----------|-----|-----|"]
    for source, metrics in sorted(results.get("per_source", {}).items()):
        emoji = "🟢" if source.startswith("real") else "🔴"
        auc_val = f"{metrics.get('auc', '-'):.3f}" if 'auc' in metrics else "-"
        per_source_lines.append(
            f"| {emoji} {source} | {metrics['count']} | "
            f"{metrics['accuracy']:.3f} | {metrics.get('f1', 0):.3f} | {auc_val} |"
        )
    per_source = "\n".join(per_source_lines)

    return (summary, per_source)


# ═══════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Jüri Test Seti Evaluation")
    parser.add_argument("--model", type=str, default=None, help="Model dosya yolu")
    parser.add_argument("--create-structure", action="store_true",
                        help="Jüri dizin yapısını oluştur")
    args = parser.parse_args()

    if args.create_structure:
        create_jury_structure()
    else:
        run_jury_evaluation(model_path=args.model)
