"""
Frekans Haritasi Onbellek — DWT+DCT+Phase Precache

Her epoch'ta tekrar tekrar hesaplanan frekans haritalarini diske onbelekler.
283K gorsel × 18 kanal × 224×224 = buyuk islem yuku.
Bu script, ilk calistirmada tum haritalari .npy olarak kaydeder.

Kullanim:
    python scripts/precache_freq.py                          # faces_split_v2
    python scripts/precache_freq.py --source faces_split     # eski split
    python scripts/precache_freq.py --workers 8              # paralel islem
    python scripts/precache_freq.py --force                  # mevcut cache'i yeniden olustur

Disk kullanimi tahmini:
    Her gorsel: 18 × 224 × 224 × 4 byte (float32) = ~3.5 MB
    283K gorsel × 3.5 MB = ~990 GB (CIDDEN BUYUK!)
    
    Optimizasyon: float16 ile kaydet → ~495 GB
    Veya: sadece train split'i onbellekle → ~330 GB
"""

import sys
import time
import argparse
from pathlib import Path
from multiprocessing import Pool, cpu_count

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import paths, model_cfg

# Frekans ekstraktor
try:
    from core.frequency_v2 import HybridFrequencyExtractor
    HAS_HYBRID = True
except ImportError:
    HAS_HYBRID = False

try:
    from core.data_pipeline import MultiScaleDWT
    HAS_DWT = True
except ImportError:
    HAS_DWT = False

SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


def get_extractor():
    """Konfigurasyona gore uygun frekans ekstraktorunu dondur."""
    if HAS_HYBRID and getattr(model_cfg, 'USE_HYBRID_FREQ', False):
        return HybridFrequencyExtractor(
            wavelets=model_cfg.DWT_WAVELETS,
            size=model_cfg.IMG_SIZE,
            include_dwt=True,
            include_dct=True,
            include_phase=True,
        )
    elif HAS_DWT:
        return MultiScaleDWT()
    else:
        raise RuntimeError("Ne HybridFrequencyExtractor ne de MultiScaleDWT yuklenebildi!")


def process_single(args):
    """Tek gorsel icin frekans haritasi hesapla ve kaydet."""
    img_path, force, use_fp16 = args

    p = Path(img_path)
    cache_dir = p.parent / ".cache"
    suffix = "freq_hybrid" if HAS_HYBRID else "freq_dwt"
    cache_path = cache_dir / f"{p.stem}_{suffix}.npy"

    # Zaten varsa atla
    if cache_path.exists() and not force:
        return "skip"

    try:
        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img)

        extractor = get_extractor()
        freq_map = extractor(img_np)

        # Disk tasarrufu: float16 olarak kaydet
        if use_fp16:
            freq_map = freq_map.astype(np.float16)

        cache_dir.mkdir(exist_ok=True)
        np.save(str(cache_path), freq_map)
        return "ok"
    except Exception as e:
        return f"err:{e}"


def main():
    parser = argparse.ArgumentParser(description="Frekans haritasi precache")
    parser.add_argument("--source", type=str, default="faces_split_v2",
                        help="Kaynak dizin (faces_split_v2 veya faces_split)")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"],
                        help="Hangi split'ler")
    parser.add_argument("--workers", type=int, default=None,
                        help="Paralel worker sayisi")
    parser.add_argument("--force", action="store_true",
                        help="Mevcut cache'i yeniden olustur")
    parser.add_argument("--fp16", action="store_true", default=True,
                        help="float16 ile kaydet (disk tasarrufu)")
    parser.add_argument("--fp32", action="store_true",
                        help="float32 ile kaydet (tam hassasiyet)")
    args = parser.parse_args()

    use_fp16 = not args.fp32
    num_workers = args.workers or min(cpu_count(), 6)

    print("=" * 60)
    print("DEEPFAKE ULTRA — FREKANS HARITASI PRECACHE")
    print("=" * 60)
    print(f"  Kaynak: {args.source}")
    print(f"  Split'ler: {args.splits}")
    print(f"  Workers: {num_workers}")
    print(f"  Format: {'float16' if use_fp16 else 'float32'}")
    print(f"  Force: {args.force}")
    print()

    # Tum dosyalari topla
    all_files = []
    base_dir = paths.DATASET_DIR / args.source

    for split in args.splits:
        for cls in ["real", "fake"]:
            cls_dir = base_dir / split / cls
            if not cls_dir.exists():
                print(f"  [ATLA] {cls_dir} mevcut degil")
                continue
            files = [
                str(f) for f in cls_dir.rglob("*")
                if f.is_file() and f.suffix.lower() in SUPPORTED
                and ".cache" not in str(f)
            ]
            print(f"  {split}/{cls}: {len(files):,} gorsel")
            all_files.extend(files)

    total = len(all_files)
    print(f"\n  Toplam: {total:,} gorsel")

    if total == 0:
        print("  Islenecek gorsel yok!")
        return

    # Disk tahmini
    channels = model_cfg.DWT_CHANNELS  # 18
    size = model_cfg.IMG_SIZE  # 224
    bytes_per = channels * size * size * (2 if use_fp16 else 4)
    total_gb = (total * bytes_per) / (1024**3)
    print(f"  Tahmini disk: {total_gb:.1f} GB")

    confirm = input(f"\n  {total:,} gorsel icin precache baslatilsin mi? [y/N]: ")
    if confirm.lower() != "y":
        print("  Iptal edildi.")
        return

    # Paralel islem
    t0 = time.time()
    task_args = [(f, args.force, use_fp16) for f in all_files]

    ok, skip, err = 0, 0, 0
    with Pool(num_workers) as pool:
        for i, result in enumerate(pool.imap_unordered(process_single, task_args, chunksize=50)):
            if result == "ok":
                ok += 1
            elif result == "skip":
                skip += 1
            else:
                err += 1

            if (i + 1) % 5000 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (total - i - 1) / rate
                print(f"  [{i+1:,}/{total:,}] {rate:.0f} img/s, "
                      f"OK={ok:,} Skip={skip:,} Err={err}, ETA: {eta:.0f}s")

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"  TAMAMLANDI ({elapsed/60:.1f} dk)")
    print(f"  Yeni: {ok:,} | Atlanan: {skip:,} | Hata: {err}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
