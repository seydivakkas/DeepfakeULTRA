"""
Veri Seti Yeniden Yapilandirma — Kalite Filtreleme + Dengeleme + Re-Split

Adimlar:
    1. Tum gorselleri tara → cozunurluk, bulaniklik, butunluk kontrolu
    2. Dusuk kalitedekileri ele
    3. Sinif dengeleme: REAL == FAKE (50/50)
    4. Yeni split: 75/15/15 (train/val/test)
    5. faces_split_v2/ olustur (symlink veya hardlink)

Kullanim:
    python scripts/dataset_restructure.py                    # Sadece tarama (dry-run)
    python scripts/dataset_restructure.py --execute          # Gercek islem
    python scripts/dataset_restructure.py --execute --copy   # Symlink yerine kopya
"""

import os
import sys
import json
import time
import shutil
import argparse
from pathlib import Path
from collections import defaultdict
from multiprocessing import Pool, cpu_count

import numpy as np
from PIL import Image, ImageFile

# Truncated dosyalari yuklemeye izin ver (kontrol icin)
ImageFile.LOAD_TRUNCATED_IMAGES = True

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import paths

# ═══════════════════════════════════════════════════════════
# SABITLER
# ═══════════════════════════════════════════════════════════

SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

# Kalite esikleri
MIN_RESOLUTION = 128          # min(w, h) < 128 → sil
MIN_LAPLACIAN_VAR = 15.0      # Bulaniklik esigi (dusuk = bulanik)
MIN_FILE_SIZE_KB = 2          # < 2KB → bozuk

# Split oranlari (70/15/15 = %100)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Kaynak tespiti
def detect_source(filename: str) -> str:
    name = filename.lower()
    if name.startswith("df40_"):
        parts = name.split("_")
        return "df40_" + parts[1] if len(parts) >= 2 else "df40_other"
    elif name.startswith("celeba_"):
        return "celeba_hq"
    elif name.startswith("ffpp_") or name.startswith("faceforensics"):
        return "ffpp"
    elif name.startswith("ffhq"):
        return "ffhq"
    elif name.startswith("utk") or "utkface" in name:
        return "utkface"
    elif name.startswith("sidset") or name.startswith("sid_"):
        return "sidset"
    elif name.startswith("sbi_"):
        return "sbi_generated"
    else:
        return "diger"


# ═══════════════════════════════════════════════════════════
# KALITE DEGERLENDIRME
# ═══════════════════════════════════════════════════════════

def compute_laplacian_variance(img_path: str) -> float:
    """Laplacian varyansini hesapla (bulaniklik olcusu)."""
    try:
        import cv2
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return 0.0
        laplacian = cv2.Laplacian(img, cv2.CV_64F)
        return float(laplacian.var())
    except Exception:
        return 0.0


def assess_quality(file_path: str) -> dict:
    """
    Tek bir gorselin kalite degerini olc.
    
    Returns:
        dict: {
            'path': str,
            'valid': bool,
            'reason': str or None,
            'width': int,
            'height': int,
            'laplacian': float,
            'file_size_kb': float,
            'source': str
        }
    """
    result = {
        "path": file_path,
        "valid": False,
        "reason": None,
        "width": 0,
        "height": 0,
        "laplacian": 0.0,
        "file_size_kb": 0.0,
        "source": detect_source(Path(file_path).name),
    }

    p = Path(file_path)

    # 1. Dosya boyutu kontrolu
    try:
        size_kb = p.stat().st_size / 1024
        result["file_size_kb"] = size_kb
        if size_kb < MIN_FILE_SIZE_KB:
            result["reason"] = "tiny_file"
            return result
    except OSError:
        result["reason"] = "file_error"
        return result

    # 2. Gorsel acma + cozunurluk kontrolu
    try:
        img = Image.open(file_path)
        img.verify()  # Butunluk kontrolu
        # verify() sonrasi tekrar ac
        img = Image.open(file_path)
        w, h = img.size
        result["width"] = w
        result["height"] = h

        if min(w, h) < MIN_RESOLUTION:
            result["reason"] = "low_resolution"
            return result
    except Exception:
        result["reason"] = "corrupted"
        return result

    # 3. Bulaniklik kontrolu
    lap_var = compute_laplacian_variance(file_path)
    result["laplacian"] = lap_var
    if lap_var < MIN_LAPLACIAN_VAR:
        result["reason"] = "blurry"
        return result

    # Tum kontrollerden gecti
    result["valid"] = True
    return result


# ═══════════════════════════════════════════════════════════
# TARAMA
# ═══════════════════════════════════════════════════════════

def collect_all_files() -> dict:
    """
    Tum gorsel dosyalarini topla.
    
    Tarama sirasi (oncelik: ilk bulunan kazanir, duplicate path kontrol edilir):
        1. faces_split_v2/{train,val,test}/{real,fake}/
        2. faces_split_v2_old/{train,val,test}/{real,fake}/ (rename sonrasi fallback)
        3. faces/{kaynak}/{real,fake}/  (orijinal kaynaklar: df40, celeba_hq, ffhq_256...)
        4. faces/genimage/fake/  (diffusion-model fake)
        5. faces/hard_real/      (hard negative REAL)

    Returns:
        {'real': [path, ...], 'fake': [path, ...]}
    """
    files = {"real": [], "fake": []}
    seen_names = set()  # dosya adi bazli duplicate engeli

    def _add_files(directory, cls):
        """Bir dizindeki dosyalari ekle (duplicate kontrol ile)."""
        count = 0
        if not directory.exists():
            return 0
        for f in directory.rglob("*"):
            if f.is_file() and f.suffix.lower() in SUPPORTED_FORMATS and ".cache" not in str(f):
                if f.name not in seen_names:
                    files[cls].append(str(f))
                    seen_names.add(f.name)
                    count += 1
        return count

    # 1. Mevcut split dizinlerinden topla (v2 + v2_old)
    for split_base_name in ["faces_split_v2", "faces_split_v2_old"]:
        split_base = paths.DATASET_DIR / split_base_name
        if not split_base.exists():
            continue
        for split in ["train", "val", "test"]:
            for cls in ["real", "fake"]:
                cls_dir = split_base / split / cls
                n = _add_files(cls_dir, cls)

    # 2. Orijinal faces/ kaynak dizinlerinden topla (split'e dahil edilmemis olanlar)
    faces_base = paths.DATASET_DIR / "faces"
    if faces_base.exists():
        for source_dir in sorted(faces_base.iterdir()):
            if not source_dir.is_dir():
                continue
            # genimage ve hard_real ayri handle ediliyor (asagida)
            if source_dir.name in ("genimage", "hard_real"):
                continue
            for cls in ["real", "fake"]:
                cls_dir = source_dir / cls
                n = _add_files(cls_dir, cls)

    # 3. GenImage (diffusion fake'ler: Stable Diffusion, DALL-E vb.)
    genimage_dir = faces_base / "genimage"
    if genimage_dir.exists():
        genimage_count = _add_files(genimage_dir, "fake")
        if genimage_count > 0:
            print(f"  [GenImage] {genimage_count:,} diffusion fake eklendi")

    # 4. Hard Real (zor negatif gorseller)
    hard_real_dir = faces_base / "hard_real"
    if hard_real_dir.exists():
        hard_real_count = _add_files(hard_real_dir, "real")
        if hard_real_count > 0:
            print(f"  [HardReal] {hard_real_count:,} hard negative REAL eklendi")

    return files


def scan_quality(files: list, num_workers: int = None) -> list:
    """
    Gorsel listesini paralel olarak tara.
    
    Returns:
        list of quality assessment dicts
    """
    if num_workers is None:
        num_workers = min(cpu_count(), 8)

    total = len(files)
    print(f"  Taranacak: {total:,} gorsel ({num_workers} worker)")

    t0 = time.time()

    if num_workers <= 1:
        results = []
        for i, f in enumerate(files):
            results.append(assess_quality(f))
            if (i + 1) % 5000 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (total - i - 1) / rate
                print(f"    [{i+1:,}/{total:,}] {rate:.0f} img/s, ETA: {eta:.0f}s")
        return results

    # Paralel islem
    results = []
    with Pool(num_workers) as pool:
        for i, r in enumerate(pool.imap_unordered(assess_quality, files, chunksize=100)):
            results.append(r)
            if (i + 1) % 10000 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (total - i - 1) / rate
                print(f"    [{i+1:,}/{total:,}] {rate:.0f} img/s, ETA: {eta:.0f}s")

    elapsed = time.time() - t0
    print(f"  Tarama tamamlandi: {elapsed:.0f}s ({total/elapsed:.0f} img/s)")
    return results


# ═══════════════════════════════════════════════════════════
# DENGELEME + SPLIT
# ═══════════════════════════════════════════════════════════

def balance_and_split(
    real_valid: list,
    fake_valid: list,
    seed: int = 42
) -> dict:
    """
    50/50 dengele + 75/15/15 split.
    
    FAKE downsample stratejisi:
        - Kaynak bazli esit kes (DF40 cesitliligi korunsun)
        - Kalite skoruna gore (laplacian desc) siralayip en iyileri sec
    
    Returns:
        {
            'train': {'real': [...], 'fake': [...]},
            'val': {'real': [...], 'fake': [...]},
            'test': {'real': [...], 'fake': [...]}
        }
    """
    rng = np.random.RandomState(seed)

    # Kalite skoruna gore sirala (en iyi -> en kotu)
    real_sorted = sorted(real_valid, key=lambda x: x["laplacian"], reverse=True)
    fake_sorted = sorted(fake_valid, key=lambda x: x["laplacian"], reverse=True)

    # Hedef: min(real, fake)
    target_per_class = min(len(real_sorted), len(fake_sorted))
    print(f"  Hedef per class: {target_per_class:,}")

    # REAL: en kaliteli target_per_class'i sec
    real_selected = real_sorted[:target_per_class]

    # FAKE: kaynak cesitliligini koruyarak sec
    fake_by_source = defaultdict(list)
    for item in fake_sorted:
        fake_by_source[item["source"]].append(item)

    n_sources = len(fake_by_source)
    per_source = target_per_class // n_sources
    remainder = target_per_class % n_sources

    fake_selected = []
    overflow = []

    for i, (src, items) in enumerate(sorted(fake_by_source.items())):
        quota = per_source + (1 if i < remainder else 0)
        if len(items) >= quota:
            fake_selected.extend(items[:quota])
        else:
            fake_selected.extend(items)
            overflow.append(quota - len(items))

    # Eksik varsa diger kaynaklardan tamamla
    if len(fake_selected) < target_per_class:
        remaining = target_per_class - len(fake_selected)
        selected_paths = {item["path"] for item in fake_selected}
        extras = [item for item in fake_sorted if item["path"] not in selected_paths]
        fake_selected.extend(extras[:remaining])

    print(f"  Secilen: REAL={len(real_selected):,}, FAKE={len(fake_selected):,}")

    # Shuffle
    rng.shuffle(real_selected)
    rng.shuffle(fake_selected)

    # Split
    n = len(real_selected)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)
    # n_test = gerisi

    result = {
        "train": {
            "real": [item["path"] for item in real_selected[:n_train]],
            "fake": [item["path"] for item in fake_selected[:n_train]],
        },
        "val": {
            "real": [item["path"] for item in real_selected[n_train:n_train + n_val]],
            "fake": [item["path"] for item in fake_selected[n_train:n_train + n_val]],
        },
        "test": {
            "real": [item["path"] for item in real_selected[n_train + n_val:]],
            "fake": [item["path"] for item in fake_selected[n_train + n_val:]],
        },
    }

    for split_name, split_data in result.items():
        print(f"    {split_name}: REAL={len(split_data['real']):,}, FAKE={len(split_data['fake']):,}")

    return result


# ═══════════════════════════════════════════════════════════
# DIZIN OLUSTURMA
# ═══════════════════════════════════════════════════════════

def create_split_directory(split_plan: dict, use_copy: bool = False):
    """
    faces_split_v2/ dizinini olustur.
    
    Symlink (varsayilan) veya dosya kopyasi kullanir.
    
    KRITIK: Mevcut dizin dogrudan silinmez — kaynak dosya yollari
    bu dizinden gelebilir. Once _old'a tasir, yeni dizin olustur,
    sonra eski dizini sil.
    """
    base_dir = paths.DATASET_DIR / "faces_split_v2"
    old_dir = paths.DATASET_DIR / "faces_split_v2_old"

    # Eski yedek varsa temizle
    if old_dir.exists():
        print(f"  [TEMIZLIK] Eski yedek siliniyor: {old_dir}")
        shutil.rmtree(old_dir)

    # Mevcut dizini rename et (silme DEGIL — kaynak dosyalar burada!)
    if base_dir.exists():
        print(f"  [RENAME] {base_dir.name} -> {old_dir.name} (kaynak korunuyor)")
        base_dir.rename(old_dir)

    total_created = 0
    total_errors = 0

    for split_name, split_data in split_plan.items():
        for cls, file_list in split_data.items():
            target_dir = base_dir / split_name / cls
            target_dir.mkdir(parents=True, exist_ok=True)

            for src_path in file_list:
                src = Path(src_path)

                # Kaynak dosya eski dizine tasinmis olabilir — yolu guncelle
                if not src.exists() and old_dir.exists():
                    # faces_split_v2/train/real/x.jpg → faces_split_v2_old/train/real/x.jpg
                    try:
                        rel = src.relative_to(base_dir)
                        alt_src = old_dir / rel
                        if alt_src.exists():
                            src = alt_src
                    except ValueError:
                        pass

                dst = target_dir / src.name

                # Isim catismasi kontrolu
                if dst.exists():
                    # Benzersiz isim olustur
                    dst = target_dir / f"{src.stem}_{hash(src_path) % 100000}{src.suffix}"

                try:
                    if use_copy:
                        shutil.copy2(src, dst)
                    else:
                        # Symlink dene, basarisizsa hardlink, o da olmazsa copy
                        try:
                            os.symlink(src.resolve(), dst)
                        except OSError:
                            try:
                                os.link(src, dst)
                            except OSError:
                                shutil.copy2(src, dst)
                    total_created += 1
                except Exception as e:
                    total_errors += 1
                    if total_errors <= 10:
                        print(f"    [HATA] {src.name}: {e}")

            print(f"  {split_name}/{cls}: {len(file_list):,} dosya ({total_created:,} basarili)")

    print(f"  Toplam: {total_created:,} olusturuldu, {total_errors} hata")

    # Basarili olusturma sonrasi eski dizini temizle
    if old_dir.exists() and total_errors < total_created * 0.01:
        print(f"  [TEMIZLIK] Eski dizin siliniyor: {old_dir.name}")
        shutil.rmtree(old_dir)
    elif old_dir.exists():
        print(f"  [UYARI] Cok fazla hata ({total_errors}), eski dizin korundu: {old_dir}")

    return base_dir


# ═══════════════════════════════════════════════════════════
# DOGRULAMA
# ═══════════════════════════════════════════════════════════

def verify_split(base_dir: Path):
    """Olusturulan split'i dogrula."""
    print("\n=== DOGRULAMA ===")
    total = 0
    for split in ["train", "val", "test"]:
        split_dir = base_dir / split
        for cls in ["real", "fake"]:
            cls_dir = split_dir / cls
            if cls_dir.exists():
                count = sum(1 for f in cls_dir.iterdir() if f.is_file() or f.is_symlink())
                total += count
                print(f"  {split}/{cls}: {count:,}")
            else:
                print(f"  {split}/{cls}: YOK!")

    print(f"  Genel toplam: {total:,}")

    # 10 rastgele dosya ac
    import random
    all_files = list((base_dir / "train" / "real").iterdir())[:10]
    ok = 0
    for f in all_files:
        try:
            target = f.resolve() if f.is_symlink() else f
            img = Image.open(target)
            img.verify()
            ok += 1
        except Exception as e:
            print(f"    [HATA] {f.name}: {e}")
    print(f"  Butunluk testi: {ok}/{len(all_files)} basarili")


# ═══════════════════════════════════════════════════════════
# ANA AKIS
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Veri seti yeniden yapilandirma")
    parser.add_argument("--execute", action="store_true", help="Gercek islem yap (dry-run degilse)")
    parser.add_argument("--copy", action="store_true", help="Symlink yerine kopya kullan")
    parser.add_argument("--workers", type=int, default=None, help="Paralel worker sayisi")
    parser.add_argument("--skip-scan", action="store_true", help="Taramayi atla (onceki sonuclari kullan)")
    args = parser.parse_args()

    print("=" * 60)
    print("DEEPFAKE ULTRA — VERI SETI YENIDEN YAPILANDIRMA")
    print("=" * 60)

    report_path = paths.BASE_DIR / "evaluation" / "quality_report.json"
    scan_cache_path = paths.BASE_DIR / "evaluation" / "quality_scan_cache.json"

    # ─── ADIM 1: Gorsel toplama ───
    print("\n[1/5] Gorsel dosyalari toplaniyor...")
    t0 = time.time()
    files = collect_all_files()
    print(f"  REAL: {len(files['real']):,}")
    print(f"  FAKE: {len(files['fake']):,}")
    print(f"  Toplam: {len(files['real']) + len(files['fake']):,}")

    # ─── ADIM 2: Kalite taramasi ───
    if args.skip_scan and scan_cache_path.exists():
        print("\n[2/5] Onceki tarama sonuclari yukleniyor...")
        with open(scan_cache_path, "r", encoding="utf-8") as f:
            cache = json.load(f)
        real_results = cache["real"]
        fake_results = cache["fake"]
    else:
        print("\n[2/5] Kalite taramasi basliyor...")
        print("  --- REAL ---")
        real_results = scan_quality(files["real"], args.workers)
        print("  --- FAKE ---")
        fake_results = scan_quality(files["fake"], args.workers)

        # Cache kaydet
        scan_cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(scan_cache_path, "w", encoding="utf-8") as f:
            json.dump({"real": real_results, "fake": fake_results}, f, ensure_ascii=False)
        print(f"  Cache kaydedildi: {scan_cache_path}")

    # ─── ADIM 3: Kalite raporu ───
    print("\n[3/5] Kalite raporu olusturuluyor...")

    for cls_name, results in [("REAL", real_results), ("FAKE", fake_results)]:
        total = len(results)
        valid = sum(1 for r in results if r["valid"])
        invalid = total - valid

        reasons = defaultdict(int)
        for r in results:
            if not r["valid"]:
                reasons[r["reason"]] += 1

        print(f"\n  {cls_name}:")
        print(f"    Toplam:  {total:,}")
        print(f"    Gecerli: {valid:,} ({valid/total*100:.1f}%)")
        print(f"    Elenen:  {invalid:,} ({invalid/total*100:.1f}%)")
        if reasons:
            print(f"    Eleme nedenleri:")
            for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
                print(f"      {reason:20s}: {count:,}")

    real_valid = [r for r in real_results if r["valid"]]
    fake_valid = [r for r in fake_results if r["valid"]]
    print(f"\n  Gecerli toplam: REAL={len(real_valid):,}, FAKE={len(fake_valid):,}")

    # ─── ADIM 4: Dengeleme + Split ───
    print("\n[4/5] Dengeleme + Split (75/15/15)...")
    split_plan = balance_and_split(real_valid, fake_valid)

    # Kaynak cesitliligi raporu
    print("\n  Kaynak cesitliligi (train/fake):")
    fake_sources = defaultdict(int)
    for path in split_plan["train"]["fake"]:
        src = detect_source(Path(path).name)
        fake_sources[src] += 1
    for src, count in sorted(fake_sources.items(), key=lambda x: -x[1])[:10]:
        print(f"    {src:25s}: {count:,}")
    print(f"    ... toplam {len(fake_sources)} kaynak")

    # Rapor kaydet
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "original": {
            "real": len(files["real"]),
            "fake": len(files["fake"]),
        },
        "after_quality_filter": {
            "real": len(real_valid),
            "fake": len(fake_valid),
        },
        "after_balance": {
            split_name: {
                cls: len(file_list)
                for cls, file_list in split_data.items()
            }
            for split_name, split_data in split_plan.items()
        },
        "quality_thresholds": {
            "min_resolution": MIN_RESOLUTION,
            "min_laplacian_var": MIN_LAPLACIAN_VAR,
            "min_file_size_kb": MIN_FILE_SIZE_KB,
        },
        "split_ratios": {
            "train": TRAIN_RATIO,
            "val": VAL_RATIO,
            "test": TEST_RATIO,
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n  Rapor: {report_path}")

    # ─── ADIM 5: Dizin olusturma ───
    if args.execute:
        print("\n[5/5] faces_split_v2/ olusturuluyor...")
        base_dir = create_split_directory(split_plan, use_copy=args.copy)
        verify_split(base_dir)
    else:
        print("\n[5/5] DRY-RUN modu — dizin olusturulmadi.")
        print("       Gercek islem icin: --execute bayragi ekleyin")

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"TAMAMLANDI ({elapsed:.0f}s)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
