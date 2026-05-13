"""
Train/Val/Test Fiziksel Split — Tum kaynaklari birlesik split yapisina kopyala.

Cikti yapisi:
    dataset/faces_split/
    ├── train/
    │   ├── real/    (tum REAL kaynaklar)
    │   └── fake/    (tum FAKE kaynaklar)
    ├── val/
    │   ├── real/
    │   └── fake/
    └── test/
        ├── real/
        └── fake/

Kullanim:
    python scripts/05_split_dataset.py
    python scripts/05_split_dataset.py --dry-run
"""

import sys
import os
import shutil
import hashlib
import argparse
from pathlib import Path
from collections import defaultdict, Counter
import random

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
FACES_DIR = PROJECT_ROOT / "dataset" / "faces"
OUTPUT_DIR = PROJECT_ROOT / "dataset" / "faces_split"

# Split oranlari
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

SEED = 42


def file_hash_split(filepath: str, seed: int = SEED) -> str:
    """Dosya yolunun hash'ine gore split belirle (deterministik)."""
    h = hashlib.md5(f"{seed}_{filepath}".encode()).hexdigest()
    val = int(h[:8], 16) / 0xFFFFFFFF

    if val < TRAIN_RATIO:
        return "train"
    elif val < TRAIN_RATIO + VAL_RATIO:
        return "val"
    else:
        return "test"


def video_id_from_filename(filename: str) -> str:
    """FF++ dosya adindaki video ID'yi cikar. Ornek: 003_0001_frame5.jpg -> 003_0001"""
    parts = filename.split("_frame")
    if len(parts) >= 2:
        return parts[0]
    parts = filename.split("_f")
    if len(parts) >= 2:
        return parts[0]
    return filename


def collect_source_files(source_dir: Path, label_name: str, recursive: bool = False):
    """Belirli bir label dizininden dosyalari topla."""
    label_dir = source_dir / label_name
    if not label_dir.exists():
        return []

    files = []
    if recursive:
        for f in sorted(label_dir.rglob("*")):
            if f.is_file() and f.suffix.lower() in SUPPORTED_FORMATS:
                files.append(f)
    else:
        for f in sorted(label_dir.iterdir()):
            if f.is_file() and f.suffix.lower() in SUPPORTED_FORMATS:
                files.append(f)
    return files


def split_by_video_id(files: list, seed: int = SEED) -> dict:
    """Video ID bazli split (ayni video tamamen ayni split'te kalir)."""
    # Video ID'leri grupla
    video_groups = defaultdict(list)
    for f in files:
        vid_id = video_id_from_filename(f.stem)
        video_groups[vid_id].append(f)

    # Video ID'leri karistir
    video_ids = sorted(video_groups.keys())
    rng = random.Random(seed)
    rng.shuffle(video_ids)

    n = len(video_ids)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    splits = {"train": [], "val": [], "test": []}
    for i, vid_id in enumerate(video_ids):
        if i < n_train:
            splits["train"].extend(video_groups[vid_id])
        elif i < n_train + n_val:
            splits["val"].extend(video_groups[vid_id])
        else:
            splits["test"].extend(video_groups[vid_id])

    return splits


def split_by_hash(files: list, seed: int = SEED) -> dict:
    """Dosya hash bazli rastgele split."""
    splits = {"train": [], "val": [], "test": []}
    for f in files:
        s = file_hash_split(str(f), seed)
        splits[s].append(f)
    return splits


def copy_files(file_list: list, dest_dir: Path, source_name: str, dry_run: bool = False):
    """Dosyalari hedef dizine kopyala (prefix ile cakisma engelle)."""
    if not file_list:
        return 0

    dest_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    total = len(file_list)

    for i, src in enumerate(file_list):
        # Cakisma engellemek icin kaynak_prefix ekle
        dest_name = f"{source_name}_{src.name}"
        dest_path = dest_dir / dest_name

        if dest_path.exists():
            dest_name = f"{source_name}_{i:06d}_{src.name}"
            dest_path = dest_dir / dest_name

        if not dry_run:
            shutil.copy2(str(src), str(dest_path))

        count += 1

        if (i + 1) % 10000 == 0 or (i + 1) == total:
            pct = (i + 1) / total * 100
            print(f"    {source_name}: {i+1}/{total} ({pct:.0f}%)")

    return count


def process_source(name: str, source_dir: Path, label_map: dict,
                   use_video_split: bool = False, recursive: bool = False,
                   dry_run: bool = False) -> dict:
    """Tek bir veri kaynagini isle ve split et."""
    print(f"\n  [{name}] Isleniyor...")

    stats = defaultdict(lambda: defaultdict(int))

    for orig_label, binary_label in label_map.items():
        files = collect_source_files(source_dir, orig_label, recursive=recursive)
        if not files:
            print(f"    {orig_label}: bos veya yok")
            continue

        # Split yontemi sec
        if use_video_split:
            splits = split_by_video_id(files)
        else:
            splits = split_by_hash(files)

        print(f"    {orig_label} ({binary_label}): {len(files):,} dosya -> "
              f"train={len(splits['train']):,}, "
              f"val={len(splits['val']):,}, "
              f"test={len(splits['test']):,}")

        # Kopyala
        for split_name, split_files in splits.items():
            dest = OUTPUT_DIR / split_name / binary_label
            copied = copy_files(split_files, dest, name, dry_run=dry_run)
            stats[split_name][binary_label] += copied

    return dict(stats)


def main(dry_run: bool = False):
    print("=" * 60)
    print("  FIZIKSEL TRAIN/VAL/TEST SPLIT")
    print(f"  Oranlar: {TRAIN_RATIO:.0%} / {VAL_RATIO:.0%} / {TEST_RATIO:.0%}")
    print(f"  Cikti: {OUTPUT_DIR}")
    if dry_run:
        print("  [DRY-RUN] Dosya kopyalanmayacak")
    print("=" * 60)

    # Hedef dizini temizle (eski split varsa)
    if OUTPUT_DIR.exists() and not dry_run:
        print(f"\n  Eski split siliniyor: {OUTPUT_DIR}")
        shutil.rmtree(str(OUTPUT_DIR))

    all_stats = defaultdict(lambda: defaultdict(int))

    # ─── 1. FF++ (video ID bazli split) ───
    ffpp_stats = process_source(
        "ffpp", FACES_DIR / "ffpp",
        {"real": "real", "fake": "fake"},
        use_video_split=True, recursive=True, dry_run=dry_run
    )
    for s in ffpp_stats:
        for l in ffpp_stats[s]:
            all_stats[s][l] += ffpp_stats[s][l]

    # ─── 2. AntiSpoof (kucuk set → tumunu train'e al) ───
    antispoof_dir = FACES_DIR / "antispoof"
    if antispoof_dir.exists():
        print(f"\n  [antispoof] Isleniyor...")
        for orig_label, binary_label in {"live": "real", "spoof": "fake"}.items():
            files = collect_source_files(antispoof_dir, orig_label, recursive=True)
            if files:
                # Cok kucuk set: tumunu train'e koy
                dest = OUTPUT_DIR / "train" / binary_label
                copied = copy_files(files, dest, "antispoof", dry_run=dry_run)
                all_stats["train"][binary_label] += copied
                print(f"    {orig_label} ({binary_label}): {len(files)} -> tumunu train'e")

    # ─── 3. DF40 (hash bazli split, recursive) ───
    df40_stats = process_source(
        "df40", FACES_DIR / "df40",
        {"fake": "fake"},
        use_video_split=False, recursive=True, dry_run=dry_run
    )
    for s in df40_stats:
        for l in df40_stats[s]:
            all_stats[s][l] += df40_stats[s][l]

    # ─── 4. CelebA-HQ (hash bazli split) ───
    celeba_stats = process_source(
        "celeba", FACES_DIR / "celeba_hq",
        {"real": "real"},
        use_video_split=False, dry_run=dry_run
    )
    for s in celeba_stats:
        for l in celeba_stats[s]:
            all_stats[s][l] += celeba_stats[s][l]

    # ─── 5. FFHQ (hash bazli split) ───
    ffhq_stats = process_source(
        "ffhq", FACES_DIR / "ffhq",
        {"real": "real"},
        use_video_split=False, dry_run=dry_run
    )
    for s in ffhq_stats:
        for l in ffhq_stats[s]:
            all_stats[s][l] += ffhq_stats[s][l]

    # ─── OZET ───
    print("\n" + "=" * 60)
    print("  SPLIT OZETI")
    print("=" * 60)
    print(f"\n  {'Split':8s} {'REAL':>10s} {'FAKE':>10s} {'TOPLAM':>10s} {'REAL%':>7s}")
    print(f"  {'-'*47}")

    grand_total = 0
    for split_name in ["train", "val", "test"]:
        r = all_stats[split_name].get("real", 0)
        f = all_stats[split_name].get("fake", 0)
        t = r + f
        pct = r / t * 100 if t > 0 else 0
        grand_total += t
        print(f"  {split_name:8s} {r:>10,} {f:>10,} {t:>10,} {pct:>6.1f}%")

    print(f"  {'-'*47}")
    total_r = sum(all_stats[s].get("real", 0) for s in all_stats)
    total_f = sum(all_stats[s].get("fake", 0) for s in all_stats)
    print(f"  {'TOPLAM':8s} {total_r:>10,} {total_f:>10,} {grand_total:>10,}")
    print(f"\n  WeightedRandomSampler -> her epoch 50-50 dengeleme")
    print(f"  Cikti dizini: {OUTPUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fiziksel Train/Val/Test Split")
    parser.add_argument("--dry-run", action="store_true", help="Sadece rapor")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
