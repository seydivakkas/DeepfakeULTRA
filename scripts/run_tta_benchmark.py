"""
TTA Benchmark — Tum harici dataset'lerde TTA ile degerlendirme.
Normal vs TTA sonuclarini karsilastirir.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from config import paths

ext_base = paths.BASE_DIR / "dataset" / "external_tests"
datasets = ["celeb_df_v2", "faceforensics", "dfdc", "deepfake20k", "deepfakeface"]

print("=" * 70)
print("  TTA BENCHMARK — Cross-Dataset Karsilastirma")
print("=" * 70)

for ds in datasets:
    ds_path = ext_base / ds
    if ds_path.exists():
        print(f"\n{'='*60}")
        print(f"  {ds} — TTA=10")
        print(f"{'='*60}")
        os.system(
            f'python -u "{paths.BASE_DIR / "scripts" / "evaluate_model.py"}" '
            f'--external "{ds_path}" --tta --tta-n 10'
        )

print("\n" + "=" * 70)
print("  TUM TTA BENCHMARKLARI TAMAMLANDI!")
print("=" * 70)
