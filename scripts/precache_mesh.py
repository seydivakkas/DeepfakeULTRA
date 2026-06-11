"""
FaceMesh Pre-Cache — Paralel, Thread-Budgeted, Chunk-Based

Sorun:
    TFLite her worker icinde kendi thread havuzunu acar.
    N worker x M thread = N*M toplam thread → cpu_count'u asarsa thrashing.

Cozum:
    1. Thread Butcesi: physical_cores / N_workers = TFLite thread/worker
    2. Chunk Mimarisi: 284K gorsel / chunk_size = daha az scheduling overhead
    3. spawn context: TFLite internal state fork'ta kopyalanmaz
    4. Profiling modu: Optimal worker sayisini otomatik bul

Kullanim:
    # Profil al (en iyi worker sayisini bul)
    python scripts/precache_mesh.py profile

    # Optimal config ile cache olustur
    python scripts/precache_mesh.py build --num-workers 4

    # Tek worker (guvenli mod)
    python scripts/precache_mesh.py build --num-workers 1
"""

import os
import sys
import time
import argparse
import multiprocessing as mp_proc
from pathlib import Path
from typing import List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import paths, model_cfg

SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
MESH_DIM = model_cfg.MESH_INPUT_DIM  # 1404


# ═══════════════════════════════════════════════════════════
# THREAD BUTCESI
# ═══════════════════════════════════════════════════════════

def get_physical_cores() -> int:
    """Fiziksel cekirdek sayisi (HT yok sayilir)."""
    try:
        import psutil
        return psutil.cpu_count(logical=False) or (os.cpu_count() // 2)
    except ImportError:
        # HT tahmini: logical / 2
        return max(1, (os.cpu_count() or 4) // 2)


def compute_thread_budget(num_workers: int) -> int:
    """Her worker icin TFLite thread sayisi."""
    physical = get_physical_cores()
    threads_per_worker = max(1, physical // num_workers)
    return threads_per_worker


# ═══════════════════════════════════════════════════════════
# WORKER INIT + CHUNK ISLEYICI
# ═══════════════════════════════════════════════════════════

# Global worker state
_worker_extractor = None
_worker_id = None


def worker_init(tflite_threads: int, worker_id: int = 0):
    """
    Her worker process'te bir kez cagrilir.
    TFLite thread sayisini sinirla ve FaceLandmarker olustur.
    """
    global _worker_extractor, _worker_id
    _worker_id = worker_id

    # TFLite/OpenMP/MKL thread limitlerini ayarla
    thread_str = str(tflite_threads)
    os.environ["OMP_NUM_THREADS"] = thread_str
    os.environ["MKL_NUM_THREADS"] = thread_str
    os.environ["OPENBLAS_NUM_THREADS"] = thread_str
    os.environ["TF_NUM_INTEROP_THREADS"] = "1"
    os.environ["TF_NUM_INTRAOP_THREADS"] = thread_str

    # FaceLandmarker olustur (thread-limited)
    try:
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        model_path = _find_model()
        if model_path:
            base_options = mp_python.BaseOptions(
                model_asset_path=str(model_path),
                # NOT: MediaPipe BaseOptions destekliyorsa num_threads buraya
            )
            options = mp_vision.FaceLandmarkerOptions(
                base_options=base_options,
                output_face_blendshapes=False,
                num_faces=1,
            )
            _worker_extractor = mp_vision.FaceLandmarker.create_from_options(options)
    except Exception as e:
        print(f"  [Worker-{worker_id}] FaceLandmarker olusturulamadi: {e}")
        _worker_extractor = None


def _find_model() -> Path:
    """face_landmarker.task dosyasini bul."""
    candidates = [
        paths.MODEL_DIR / "face_landmarker.task",
        paths.BASE_DIR / "face_landmarker.task",
        Path("face_landmarker.task"),
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def get_cache_path(img_path: str) -> Path:
    """Cache yolunu hesapla."""
    p = Path(img_path)
    cache_dir = p.parent / ".cache"
    return cache_dir / f"{p.stem}_mesh.npy"


def process_chunk(chunk: List[str]) -> dict:
    """
    Bir chunk (gorsel listesi) isler.
    TFLite JIT isinmasi chunk icinde amortize edilir.
    """
    global _worker_extractor

    ok = 0
    err = 0
    skip = 0
    zero_mesh = 0

    import mediapipe as mp_lib

    for img_path in chunk:
        cache_path = get_cache_path(img_path)

        # Zaten cache'li mi?
        if cache_path.exists():
            skip += 1
            continue

        try:
            from PIL import Image
            image = Image.open(img_path).convert("RGB")
            img_np = np.array(image)

            if _worker_extractor is not None:
                mp_image = mp_lib.Image(
                    image_format=mp_lib.ImageFormat.SRGB,
                    data=img_np
                )
                results = _worker_extractor.detect(mp_image)

                if results.face_landmarks and len(results.face_landmarks) > 0:
                    landmarks = results.face_landmarks[0]
                    coords = []
                    for lm in landmarks:
                        coords.extend([lm.x, lm.y, lm.z])
                    mesh = np.array(coords, dtype=np.float32)
                else:
                    mesh = np.zeros(MESH_DIM, dtype=np.float32)
                    zero_mesh += 1
            else:
                mesh = np.zeros(MESH_DIM, dtype=np.float32)
                zero_mesh += 1

            # Cache dizinini olustur ve kaydet
            cache_path.parent.mkdir(exist_ok=True)
            np.save(str(cache_path), mesh)
            ok += 1

        except Exception:
            # Hata → sifir vektoru kaydet
            try:
                cache_path.parent.mkdir(exist_ok=True)
                np.save(str(cache_path), np.zeros(MESH_DIM, dtype=np.float32))
            except Exception:
                pass
            err += 1

    return {"ok": ok, "err": err, "skip": skip, "zero": zero_mesh}


# ═══════════════════════════════════════════════════════════
# DOSYA TOPLAMA
# ═══════════════════════════════════════════════════════════

def collect_files() -> List[str]:
    """faces_split_v2 (veya v1) altindaki tum gorselleri topla."""
    base_dir = paths.DATASET_DIR / "faces_split_v2"
    if not base_dir.exists():
        base_dir = paths.DATASET_DIR / "faces_split"

    files = []
    for split in ["train", "val", "test"]:
        for cls in ["real", "fake"]:
            cls_dir = base_dir / split / cls
            if not cls_dir.exists():
                continue
            for f in sorted(cls_dir.rglob("*")):
                if f.is_file() and f.suffix.lower() in SUPPORTED:
                    real_path = f.resolve() if f.is_symlink() else f
                    files.append(str(real_path))
    return files


def make_chunks(files: List[str], chunk_size: int) -> List[List[str]]:
    """Dosyalari chunk'lara bol."""
    return [files[i:i + chunk_size] for i in range(0, len(files), chunk_size)]


# ═══════════════════════════════════════════════════════════
# PROFILING
# ═══════════════════════════════════════════════════════════

def run_profile(sample_size: int = 200):
    """
    Farkli worker sayilariyla fps olcerek optimal config bul.
    """
    print("=" * 60)
    print("MESH PRE-CACHE — PROFILING")
    print("=" * 60)

    physical_cores = get_physical_cores()
    print(f"  Fiziksel cekirdek: {physical_cores}")
    print(f"  Mantiksal cekirdek: {os.cpu_count()}")

    files = collect_files()
    sample = files[:sample_size]
    print(f"  Profil orneklemi: {len(sample)} gorsel\n")

    # Test konfigurasyonlari
    configs = [1, 2, 4]
    if physical_cores >= 8:
        configs.append(8)
    if physical_cores >= 16:
        configs.append(physical_cores // 2)

    results = []

    print(f"  {'Workers':>8}  {'TFLite-T':>10}  {'FPS':>8}  {'Sure(s)':>10}")
    print(f"  {'-'*8}  {'-'*10}  {'-'*8}  {'-'*10}")

    for n_workers in configs:
        tflite_threads = compute_thread_budget(n_workers)
        chunks = make_chunks(sample, max(8, len(sample) // (n_workers * 2)))

        t0 = time.time()

        if n_workers == 1:
            worker_init(tflite_threads, 0)
            total_ok = 0
            for chunk in chunks:
                r = process_chunk(chunk)
                total_ok += r["ok"] + r["skip"]
        else:
            ctx = mp_proc.get_context("spawn")
            with ctx.Pool(
                n_workers,
                initializer=worker_init,
                initargs=(tflite_threads, 0),
            ) as pool:
                total_ok = 0
                for r in pool.imap_unordered(process_chunk, chunks):
                    total_ok += r["ok"] + r["skip"]

        elapsed = time.time() - t0
        fps = len(sample) / elapsed if elapsed > 0 else 0
        results.append((n_workers, tflite_threads, fps, elapsed))

        marker = ""
        if len(results) > 1 and fps < results[-2][2] * 0.9:
            marker = " ← thrashing"

        print(f"  {n_workers:>8}  {tflite_threads:>10}  {fps:>8.1f}  {elapsed:>10.1f}{marker}")

        # Cache'leri temizle (profil icin)
        for f in sample:
            cp = get_cache_path(f)
            if cp.exists():
                try:
                    cp.unlink()
                except Exception:
                    pass

    # En iyi config
    best = max(results, key=lambda x: x[2])
    print(f"\n  ✅ OPTIMAL: {best[0]} worker, {best[1]} TFLite thread/worker, {best[2]:.1f} fps")
    print(f"\n  Komut:")
    print(f"    python scripts/precache_mesh.py build --num-workers {best[0]}")

    return best


# ═══════════════════════════════════════════════════════════
# BUILD (TAM CACHE)
# ═══════════════════════════════════════════════════════════

def run_build(num_workers: int, chunk_size: int = 64, force: bool = False):
    """Tum gorsellerin mesh cache'ini olustur."""
    print("=" * 60)
    print("MESH PRE-CACHE — BUILD")
    print("=" * 60)

    tflite_threads = compute_thread_budget(num_workers)
    physical = get_physical_cores()

    print(f"  Fiziksel cekirdek: {physical}")
    print(f"  Workers: {num_workers}")
    print(f"  TFLite thread/worker: {tflite_threads}")
    print(f"  Toplam thread: {num_workers * tflite_threads} / {physical}")
    print(f"  Chunk size: {chunk_size}")

    files = collect_files()
    total = len(files)
    print(f"  Toplam gorsel: {total:,}")

    if force:
        print("  --force: Mevcut cache'ler yeniden hesaplanacak")
        cleaned = 0
        for f in files:
            cp = get_cache_path(f)
            if cp.exists():
                try:
                    cp.unlink()
                    cleaned += 1
                except Exception:
                    pass
        print(f"  {cleaned:,} cache silindi")

    # Kac tanesi zaten cache'li?
    already = sum(1 for f in files if get_cache_path(f).exists())
    to_process = total - already
    print(f"  Cache'li: {already:,}")
    print(f"  Islenecek: {to_process:,}")

    if to_process == 0:
        print("\n  Tum gorseller zaten cache'li!")
        return

    chunks = make_chunks(files, chunk_size)
    n_chunks = len(chunks)
    print(f"  Chunk sayisi: {n_chunks:,}\n")

    t0 = time.time()
    total_ok = 0
    total_err = 0
    total_skip = 0
    total_zero = 0
    processed_chunks = 0

    if num_workers == 1:
        # Single-worker (guvenli mod)
        worker_init(tflite_threads, 0)
        for chunk in chunks:
            r = process_chunk(chunk)
            total_ok += r["ok"]
            total_err += r["err"]
            total_skip += r["skip"]
            total_zero += r["zero"]
            processed_chunks += 1

            done = total_ok + total_err + total_skip
            if processed_chunks % 50 == 0:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                eta = (total - done) / rate if rate > 0 else 0
                print(f"  [{done:,}/{total:,}] {rate:.0f} img/s, "
                      f"ETA: {eta/60:.1f}m | ok={total_ok} err={total_err} zero={total_zero}")
    else:
        # Multi-worker (spawn context)
        ctx = mp_proc.get_context("spawn")
        with ctx.Pool(
            num_workers,
            initializer=worker_init,
            initargs=(tflite_threads, 0),
        ) as pool:
            for r in pool.imap_unordered(process_chunk, chunks):
                total_ok += r["ok"]
                total_err += r["err"]
                total_skip += r["skip"]
                total_zero += r["zero"]
                processed_chunks += 1

                done = total_ok + total_err + total_skip
                if processed_chunks % 50 == 0:
                    elapsed = time.time() - t0
                    rate = done / elapsed if elapsed > 0 else 0
                    eta = (total - done) / rate if rate > 0 else 0
                    print(f"  [{done:,}/{total:,}] {rate:.0f} img/s, "
                          f"ETA: {eta/60:.1f}m | ok={total_ok} err={total_err} zero={total_zero}")

    elapsed = time.time() - t0
    total_done = total_ok + total_err + total_skip
    cache_mb = total_ok * MESH_DIM * 4 / 1024 / 1024

    print(f"\n{'='*60}")
    print(f"  TAMAMLANDI: {elapsed/60:.1f} dakika ({total_done/elapsed:.0f} img/s)")
    print(f"  OK:    {total_ok:,}")
    print(f"  Hata:  {total_err:,}")
    print(f"  Skip:  {total_skip:,} (zaten cache'li)")
    print(f"  Zero:  {total_zero:,} (yuz bulunamadi)")
    print(f"  Cache: ~{cache_mb:.0f} MB")
    print(f"{'='*60}")


# ═══════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="FaceMesh Pre-Cache — Thread-Budgeted Paralel",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", help="Komut")

    # profile
    p_profile = sub.add_parser("profile", help="Optimal worker sayisini bul")
    p_profile.add_argument("--sample-size", type=int, default=200,
                           help="Profil orneklem buyuklugu")

    # build
    p_build = sub.add_parser("build", help="Tam cache olustur")
    p_build.add_argument("--num-workers", type=int, default=None,
                         help="Worker sayisi (None=otomatik)")
    p_build.add_argument("--chunk-size", type=int, default=64,
                         help="Chunk buyuklugu")
    p_build.add_argument("--force", action="store_true",
                         help="Mevcut cache'leri yeniden hesapla")

    args = parser.parse_args()

    if args.command == "profile":
        run_profile(args.sample_size)

    elif args.command == "build":
        num_workers = args.num_workers
        if num_workers is None:
            # Otomatik: fiziksel cekirdek / 2 (guvenli basla)
            num_workers = max(1, get_physical_cores() // 2)
            print(f"  Otomatik worker: {num_workers}")
        run_build(num_workers, args.chunk_size, args.force)

    else:
        parser.print_help()
        print("\n  Ornek:")
        print("    python scripts/precache_mesh.py profile")
        print("    python scripts/precache_mesh.py build --num-workers 4")


if __name__ == "__main__":
    main()
