"""
Kalite-Bazli Akilli FAKE Undersampling + 50-50 Fiziksel Split

3 Asamali Filtre:
  1. Kalite filtresi: Bulanik/karanlik/bozuk goruntuleri ele
  2. Duplicate filtresi: Perceptual hash ile benzer kareleri cikar
  3. Esit ornekleme: Her yontemden esit payli rastgele sec

Cikti: dataset/faces_split/{train,val,test}/{real,fake}/

Kullanim:
    python scripts/06_smart_split.py
    python scripts/06_smart_split.py --dry-run
    python scripts/06_smart_split.py --workers 8
"""

import sys
import os
import shutil
import hashlib
import argparse
from pathlib import Path
from collections import defaultdict, Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import random
import time

import numpy as np
from PIL import Image
import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
FACES_DIR = PROJECT_ROOT / "dataset" / "faces"
OUTPUT_DIR = PROJECT_ROOT / "dataset" / "faces_split"

# Split oranlari
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
SEED = 42

# Kalite esik degerleri
BLUR_THRESHOLD = 35.0       # Laplacian variance < bu = bulanik
MIN_BRIGHTNESS = 25         # Ortalama piksel < bu = cok karanlik
MAX_BRIGHTNESS = 240        # Ortalama piksel > bu = overexposed
MIN_IMAGE_SIZE = 48         # Piksel < bu = cok kucuk
HASH_DISTANCE_THRESHOLD = 5 # Hamming distance < bu = duplicate


# ═══════════════════════════════════════════════════════════
# KALITE ANALIZI
# ═══════════════════════════════════════════════════════════

def compute_dhash(img_gray, hash_size=8):
    """Difference Hash — hizli perceptual hash."""
    resized = cv2.resize(img_gray, (hash_size + 1, hash_size))
    diff = resized[:, 1:] > resized[:, :-1]
    return sum(2 ** i for i, v in enumerate(diff.flatten()) if v)


def hamming_distance(h1, h2):
    """Iki hash arasindaki Hamming mesafesi."""
    return bin(h1 ^ h2).count("1")


def analyze_single_image(filepath_str):
    """Tek goruntu icin kalite metrikleri hesapla."""
    filepath = Path(filepath_str)
    try:
        img = cv2.imread(str(filepath))
        if img is None:
            return None

        h, w = img.shape[:2]
        if h < MIN_IMAGE_SIZE or w < MIN_IMAGE_SIZE:
            return None

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Laplacian variance (bulaniklik)
        # Kucuk boyutta hesapla (hiz icin)
        small = cv2.resize(gray, (128, 128))
        blur_score = cv2.Laplacian(small, cv2.CV_64F).var()

        # Ortalama parlaklik
        brightness = np.mean(small)

        # dHash
        dhash = compute_dhash(gray)

        return {
            "path": filepath_str,
            "blur": blur_score,
            "brightness": brightness,
            "dhash": dhash,
            "size": (w, h),
        }
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════
# YONTEM BAZLI ISLEM
# ═══════════════════════════════════════════════════════════

def collect_method_files(source_dir, label_name, recursive=False):
    """Kaynak dizinden goruntuleri topla, yontem bazli grupla."""
    label_dir = source_dir / label_name
    if not label_dir.exists():
        return {}

    methods = {}

    if recursive:
        # DF40 yapisi: fake/category/method/images
        for cat_dir in sorted(label_dir.iterdir()):
            if not cat_dir.is_dir():
                continue
            for method_dir in sorted(cat_dir.iterdir()):
                if not method_dir.is_dir():
                    continue
                files = [f for f in method_dir.rglob("*")
                         if f.is_file() and f.suffix.lower() in SUPPORTED]
                if files:
                    methods[f"df40_{method_dir.name}"] = files
    else:
        # Duz yapi: label/images
        files = [f for f in label_dir.rglob("*")
                 if f.is_file() and f.suffix.lower() in SUPPORTED]
        if files:
            methods[source_dir.name] = files

    return methods


def filter_method_quality(method_name, files, max_workers=4):
    """Tek yontem icin kalite filtresi + duplicate temizligi."""
    total = len(files)
    print(f"    {method_name}: {total:,} goruntu analiz ediliyor...")

    # Paralel kalite analizi
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(analyze_single_image, str(f)): f for f in files}
        done = 0
        for future in as_completed(futures):
            done += 1
            result = future.result()
            if result is not None:
                results.append(result)
            if done % 5000 == 0:
                print(f"      analiz: {done}/{total}")

    # Faz 1: Kalite filtresi
    quality_passed = []
    rejected_blur = 0
    rejected_bright = 0
    rejected_dark = 0

    for r in results:
        if r["blur"] < BLUR_THRESHOLD:
            rejected_blur += 1
            continue
        if r["brightness"] < MIN_BRIGHTNESS:
            rejected_dark += 1
            continue
        if r["brightness"] > MAX_BRIGHTNESS:
            rejected_bright += 1
            continue
        quality_passed.append(r)

    print(f"      kalite: {len(quality_passed):,} gecti "
          f"(bulanik={rejected_blur}, karanlik={rejected_dark}, parlak={rejected_bright})")

    # Faz 2: Duplicate temizligi (greedy)
    if not quality_passed:
        return []

    # Hash bazli gruplama — hizli duplicate eleme
    quality_passed.sort(key=lambda x: x["blur"], reverse=True)  # En net oncelikli

    unique = []
    seen_hashes = []

    for item in quality_passed:
        is_dup = False
        for seen_hash in seen_hashes:
            if hamming_distance(item["dhash"], seen_hash) < HASH_DISTANCE_THRESHOLD:
                is_dup = True
                break

        if not is_dup:
            unique.append(item)
            seen_hashes.append(item["dhash"])

            # Performans: hash listesi cok buyurse sadece son N'i kontrol et
            if len(seen_hashes) > 2000:
                seen_hashes = seen_hashes[-2000:]

    dup_removed = len(quality_passed) - len(unique)
    print(f"      duplicate: {dup_removed:,} cikarildi -> {len(unique):,} unique")

    return unique


# ═══════════════════════════════════════════════════════════
# ANA ISLEM
# ═══════════════════════════════════════════════════════════

def main(dry_run=False, max_workers=4):
    t0 = time.time()

    print("=" * 65)
    print("  KALITE-BAZLI AKILLI SPLIT")
    print(f"  Hedef: 50-50 REAL/FAKE, {TRAIN_RATIO:.0%}/{VAL_RATIO:.0%}/{TEST_RATIO:.0%}")
    if dry_run:
        print("  [DRY-RUN] Dosya kopyalanmayacak")
    print("=" * 65)

    # --- REAL verisi sayimi (V6 — VGGFace2 eklendi, FFHQ 256 guncellendi) ---
    real_sources = {
        "ffpp":           (FACES_DIR / "ffpp",                "real",  False),
        "celeba":         (FACES_DIR / "celeba_hq",           "real",  False),
        "ffhq_256":       (FACES_DIR / "ffhq_256",            "real",  False),
        "utkface":        (FACES_DIR / "utkface",             "real",  False),
        "sidset_real":    (FACES_DIR / "sidset",              "real",  False),
        "vggface2":       (FACES_DIR / "vggface2",            "real",  False),
        "custom_team":   (FACES_DIR / "custom_team",          "real",  False),
    }

    all_real_files = []
    print("\n REAL kaynaklar:")
    for name, (src, label, rec) in real_sources.items():
        ldir = src / label
        if ldir.exists():
            files = [f for f in ldir.rglob("*")
                     if f.is_file() and f.suffix.lower() in SUPPORTED]
            all_real_files.extend([(f, name) for f in files])
            print(f"  {name:12s}: {len(files):>8,}")

    total_real = len(all_real_files)
    print(f"  {'TOPLAM':12s}: {total_real:>8,}")
    target_fake = total_real  # 50-50

    # --- FAKE veri analizi ve filtresi ---
    print(f"\n FAKE kaynaklar (hedef: {target_fake:,}):")

    fake_sources = {}

    # FF++ fake
    ffpp_fake = FACES_DIR / "ffpp" / "fake"
    if ffpp_fake.exists():
        files = [f for f in ffpp_fake.rglob("*")
                 if f.is_file() and f.suffix.lower() in SUPPORTED]
        if files:
            fake_sources["ffpp_fake"] = files

    # SID-Set fake (synthetic + tampered)
    sidset_fake = FACES_DIR / "sidset" / "fake"
    if sidset_fake.exists():
        files = [f for f in sidset_fake.rglob("*")
                 if f.is_file() and f.suffix.lower() in SUPPORTED]
        if files:
            fake_sources["sidset_fake"] = files

    # DF40 yontemleri
    df40_fake = FACES_DIR / "df40" / "fake"
    if df40_fake.exists():
        for cat_dir in sorted(df40_fake.iterdir()):
            if not cat_dir.is_dir():
                continue
            for method_dir in sorted(cat_dir.iterdir()):
                if not method_dir.is_dir():
                    continue
                files = [f for f in method_dir.rglob("*")
                         if f.is_file() and f.suffix.lower() in SUPPORTED]
                if files:
                    fake_sources[f"df40_{method_dir.name}"] = files

    # GenImage (diffusion) yontemleri
    genimage_fake = FACES_DIR / "genimage" / "fake"
    if genimage_fake.exists():
        for method_dir in sorted(genimage_fake.iterdir()):
            if not method_dir.is_dir():
                continue
            files = [f for f in method_dir.rglob("*")
                     if f.is_file() and f.suffix.lower() in SUPPORTED]
            if files:
                fake_sources[f"genimage_{method_dir.name}"] = files

    print(f"  {len(fake_sources)} yontem bulundu:")
    for name, files in sorted(fake_sources.items()):
        print(f"    {name:25s}: {len(files):>8,}")
    total_fake_raw = sum(len(f) for f in fake_sources.values())
    print(f"  {'TOPLAM':25s}: {total_fake_raw:>8,}")

    # --- Kalite filtresi uygula ---
    print(f"\n Kalite filtresi basliyor ({max_workers} worker)...")

    filtered_methods = {}
    for name, files in sorted(fake_sources.items()):
        filtered = filter_method_quality(name, files, max_workers=max_workers)
        filtered_methods[name] = filtered

    total_after_filter = sum(len(v) for v in filtered_methods.values())
    print(f"\n  Filtre sonrasi toplam FAKE: {total_after_filter:,} "
          f"(elenen: {total_fake_raw - total_after_filter:,})")

    # --- Esit ornekleme ---
    print(f"\n Esit ornekleme (hedef: {target_fake:,})...")

    n_methods = len(filtered_methods)
    # Nadir yontemleri tanimla (quota'dan az olanlari)
    base_quota = target_fake // n_methods

    # Ilk gecis: nadir yontemleri belirle
    budget = target_fake
    locked = {}
    flexible = {}

    for name, items in filtered_methods.items():
        if len(items) <= base_quota:
            # Tum olanlari al
            locked[name] = items
            budget -= len(items)
        else:
            flexible[name] = items

    # Kalan budgeti esit dagit
    if flexible:
        per_method = budget // len(flexible)
        remainder = budget - per_method * len(flexible)
    else:
        per_method = 0
        remainder = 0

    rng = random.Random(SEED)
    all_fake_selected = []

    print(f"\n  Yontem bazli dagilim:")
    for name in sorted(filtered_methods.keys()):
        if name in locked:
            selected = locked[name]
        else:
            pool = flexible[name]
            # En yuksek kalite oncelikli sec (blur score yuksek = net)
            pool.sort(key=lambda x: x["blur"], reverse=True)
            quota = per_method + (1 if remainder > 0 else 0)
            if remainder > 0:
                remainder -= 1
            selected = pool[:quota]

        all_fake_selected.extend([(item["path"], name) for item in selected])
        pct = len(selected) / len(filtered_methods[name]) * 100 if filtered_methods[name] else 0
        print(f"    {name:25s}: {len(selected):>6,} / {len(filtered_methods[name]):>6,} ({pct:.0f}%)")

    total_fake_selected = len(all_fake_selected)
    print(f"\n  Secilen FAKE: {total_fake_selected:,}")
    print(f"  REAL:         {total_real:,}")
    print(f"  Oran:         1:{total_fake_selected/max(total_real,1):.2f}")

    # --- Split ---
    print(f"\n Split: {TRAIN_RATIO:.0%} / {VAL_RATIO:.0%} / {TEST_RATIO:.0%}")

    def hash_split(filepath, seed=SEED):
        h = hashlib.md5(f"{seed}_{filepath}".encode()).hexdigest()
        val = int(h[:8], 16) / 0xFFFFFFFF
        if val < TRAIN_RATIO:
            return "train"
        elif val < TRAIN_RATIO + VAL_RATIO:
            return "val"
        return "test"

    # REAL split
    real_splits = {"train": [], "val": [], "test": []}
    for fpath, src in all_real_files:
        s = hash_split(str(fpath))
        real_splits[s].append((fpath, src))

    # FAKE split
    fake_splits = {"train": [], "val": [], "test": []}
    for fpath_str, src in all_fake_selected:
        s = hash_split(fpath_str)
        fake_splits[s].append((fpath_str, src))

    # --- Kopyala ---
    if not dry_run:
        print(f"\n Kopyalama basliyor...")
        if OUTPUT_DIR.exists():
            print(f"  Eski split siliniyor...")
            shutil.rmtree(str(OUTPUT_DIR))

    for split_name in ["train", "val", "test"]:
        # REAL
        real_dest = OUTPUT_DIR / split_name / "real"
        if not dry_run:
            real_dest.mkdir(parents=True, exist_ok=True)
        for i, (fpath, src) in enumerate(real_splits[split_name]):
            dst = real_dest / f"{src}_{i:06d}_{fpath.name}"
            if not dry_run:
                shutil.copy2(str(fpath), str(dst))
            if (i + 1) % 10000 == 0:
                print(f"    {split_name}/real: {i+1}/{len(real_splits[split_name])}")

        # FAKE
        fake_dest = OUTPUT_DIR / split_name / "fake"
        if not dry_run:
            fake_dest.mkdir(parents=True, exist_ok=True)
        for i, (fpath_str, src) in enumerate(fake_splits[split_name]):
            fpath = Path(fpath_str)
            dst = fake_dest / f"{src}_{i:06d}_{fpath.name}"
            if not dry_run:
                shutil.copy2(fpath_str, str(dst))
            if (i + 1) % 10000 == 0:
                print(f"    {split_name}/fake: {i+1}/{len(fake_splits[split_name])}")

    # --- OZET ---
    elapsed = time.time() - t0
    print(f"\n{'=' * 65}")
    print(f"  SONUC OZETI (sure: {elapsed/60:.1f} dk)")
    print(f"{'=' * 65}")
    print(f"\n  {'Split':8s} {'REAL':>10s} {'FAKE':>10s} {'TOPLAM':>10s} {'REAL%':>7s}")
    print(f"  {'-'*47}")

    for split_name in ["train", "val", "test"]:
        r = len(real_splits[split_name])
        f = len(fake_splits[split_name])
        t = r + f
        pct = r / t * 100 if t > 0 else 0
        print(f"  {split_name:8s} {r:>10,} {f:>10,} {t:>10,} {pct:>6.1f}%")

    total_r = sum(len(v) for v in real_splits.values())
    total_f = sum(len(v) for v in fake_splits.values())
    print(f"  {'-'*47}")
    print(f"  {'TOPLAM':8s} {total_r:>10,} {total_f:>10,} {total_r+total_f:>10,}")

    print(f"\n  Kalite filtresi: {total_fake_raw - total_after_filter:,} dusuk kalite elendi")
    print(f"  Duplicate:       benzer kareler cikarildi")
    print(f"  Yontem dengesi:  {n_methods} yontem esit dagitildi")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kalite-bazli akilli split")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    main(dry_run=args.dry_run, max_workers=args.workers)
