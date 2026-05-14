"""
Celeb-DF v2 Cross-Dataset Benchmark
====================================
Harici veriseti uzerinde DeepfakeULTRA modelini test eder.
Sonuclari evaluation/external/celeb_df_v2/ altina kaydeder.

Kullanim:
    python scripts/evaluate_external.py --dataset celeb_df_v2
    python scripts/evaluate_external.py --dataset celeb_df_v2 --checkpoint models/best_run5_forensic.pth
    python scripts/evaluate_external.py --list  (mevcut harici datasetleri listele)
"""
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import paths

EXTERNAL_DIR = paths.DATASET_DIR / "external_tests"
EVAL_DIR = Path(__file__).parent.parent / "evaluation" / "external"
REPORTS_DIR = Path(__file__).parent.parent / "reports"


def list_external_datasets():
    """Mevcut harici datasetleri listele."""
    if not EXTERNAL_DIR.exists():
        print("Harici dataset dizini bulunamadi!")
        return

    print(f"\nMevcut harici datasetler ({EXTERNAL_DIR}):")
    for d in sorted(EXTERNAL_DIR.iterdir()):
        if d.is_dir():
            real_dir = d / "real"
            fake_dir = d / "fake"
            n_real = len(list(real_dir.glob("*"))) if real_dir.exists() else 0
            n_fake = len(list(fake_dir.glob("*"))) if fake_dir.exists() else 0
            status = "OK" if n_real > 0 and n_fake > 0 else "BOS"
            print(f"  [{status}] {d.name}: {n_real} REAL + {n_fake} FAKE")


def generate_markdown_report(metrics, dataset_name, output_path):
    """Cross-dataset benchmark Markdown raporu olustur."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    report = f"""# Cross-Dataset Benchmark Raporu

**Tarih:** {timestamp}
**Model:** DeepfakeULTRA V5 (best_model.pth)

---

## Benchmark: {dataset_name}

| Metrik | Deger |
|--------|-------|
| **ROC-AUC** | {metrics.get('roc_auc', 'N/A')} |
| **EER** | {metrics.get('eer', 'N/A')} (threshold={metrics.get('eer_threshold', 'N/A')}) |
| **Optimal Threshold** | {metrics.get('optimal_threshold', 'N/A')} (Youden J={metrics.get('youden_j', 'N/A')}) |
| **Accuracy (optimal)** | {metrics.get('accuracy_optimal', 'N/A')} |
| **Macro F1 (optimal)** | {metrics.get('macro_f1_optimal', 'N/A')} |
| **FPR@95TPR** | {metrics.get('fpr_at_95tpr', 'N/A')} |
| **ECE** | {metrics.get('ece', 'N/A')} |

### Veri Dagilimi

| Sinif | Sayi |
|-------|------|
| REAL | {metrics.get('n_real', 'N/A')} |
| FAKE | {metrics.get('n_fake', 'N/A')} |
| **Toplam** | {metrics.get('total_evaluated', 'N/A')} |

### Olasilik Dagilimi

"""
    prob_dist = metrics.get("probability_distribution", {})
    if prob_dist:
        report += f"""| Sinif | Ortalama | Std |
|-------|----------|-----|
| REAL | {prob_dist.get('real_mean', 'N/A')} | {prob_dist.get('real_std', 'N/A')} |
| FAKE | {prob_dist.get('fake_mean', 'N/A')} | {prob_dist.get('fake_std', 'N/A')} |
"""

    # Ic test sonuclari ile karsilastirma
    internal_metrics_path = Path(__file__).parent.parent / "evaluation" / "metrics.json"
    if internal_metrics_path.exists():
        with open(internal_metrics_path, "r") as f:
            internal = json.load(f)

        report += f"""
---

## Karsilastirma: Ic Test vs {dataset_name}

| Metrik | Ic Test | {dataset_name} | Fark |
|--------|---------|{'-' * len(dataset_name)}--|------|
| AUC | {internal.get('roc_auc', 'N/A')} | {metrics.get('roc_auc', 'N/A')} | {_diff(internal.get('roc_auc'), metrics.get('roc_auc'))} |
| EER | {internal.get('eer', 'N/A')} | {metrics.get('eer', 'N/A')} | {_diff(metrics.get('eer'), internal.get('eer'), lower_better=True)} |
| F1 | {internal.get('macro_f1_optimal', 'N/A')} | {metrics.get('macro_f1_optimal', 'N/A')} | {_diff(internal.get('macro_f1_optimal'), metrics.get('macro_f1_optimal'))} |
"""

    report += f"""
---

## Dosyalar

- Metrikler: `evaluation/external/{dataset_name}/metrics.json`
- ROC Curve: `evaluation/external/{dataset_name}/roc_curve.png`
- Confusion Matrix: `evaluation/external/{dataset_name}/confusion_matrix.png`
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  Markdown rapor: {output_path}")


def _diff(a, b, lower_better=False):
    """Iki metrik arasindaki farki formatla."""
    if a is None or b is None:
        return "N/A"
    diff = float(b) - float(a)
    if lower_better:
        diff = -diff
    sign = "+" if diff > 0 else ""
    return f"{sign}{diff:.4f}"


def main():
    parser = argparse.ArgumentParser(description="Harici dataset benchmark")
    parser.add_argument("--dataset", type=str, default="celeb_df_v2",
                        help="Dataset adi (external_tests/ altinda)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Model checkpoint yolu")
    parser.add_argument("--list", action="store_true",
                        help="Mevcut datasetleri listele")
    args = parser.parse_args()

    if args.list:
        list_external_datasets()
        return

    # Dataset yolu
    dataset_path = EXTERNAL_DIR / args.dataset
    if not dataset_path.exists():
        print(f"Dataset bulunamadi: {dataset_path}")
        list_external_datasets()
        return

    # Gorsel sayisi kontrol
    real_dir = dataset_path / "real"
    fake_dir = dataset_path / "fake"
    n_real = len(list(real_dir.glob("*"))) if real_dir.exists() else 0
    n_fake = len(list(fake_dir.glob("*"))) if fake_dir.exists() else 0

    if n_real == 0 or n_fake == 0:
        print(f"Dataset bos: {n_real} REAL + {n_fake} FAKE")
        return

    if n_real < 50 or n_fake < 50:
        print(f"UYARI: Yetersiz veri ({n_real} REAL + {n_fake} FAKE). "
              f"Guvenilir benchmark icin en az 500+ gorsel/sinif oneriliyor.")

    print(f"\nCross-Dataset Benchmark: {args.dataset}")
    print(f"  REAL: {n_real}, FAKE: {n_fake}")

    # evaluate_model.py --external ile calistir
    import subprocess
    cmd = [
        sys.executable, "scripts/evaluate_model.py",
        "--external", str(dataset_path),
    ]
    if args.checkpoint:
        cmd.extend(["--checkpoint", args.checkpoint])

    env = dict(__import__("os").environ, PYTHONIOENCODING="utf-8")
    result = subprocess.run(cmd, capture_output=True, text=True, env=env,
                            cwd=str(Path(__file__).parent.parent))
    print(result.stdout)
    if result.returncode != 0:
        print(f"HATA: {result.stderr}")
        return

    # Markdown rapor olustur
    metrics_path = EVAL_DIR / args.dataset / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path, "r") as f:
            metrics = json.load(f)

        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        report_path = REPORTS_DIR / "cross_dataset_benchmark.md"
        generate_markdown_report(metrics, args.dataset, report_path)
        print(f"\nBenchmark tamamlandi!")
    else:
        print(f"Metrik dosyasi bulunamadi: {metrics_path}")


if __name__ == "__main__":
    main()
