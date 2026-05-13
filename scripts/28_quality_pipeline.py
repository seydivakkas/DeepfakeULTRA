"""
Entegrasyon Oncesi Kalite Pipeline

Adimlar:
  1. EXIF/metadata temizleme
  2. BRISQUE kalite skoru hesaplama + dusuk kalite filtreleme
  3. Near-duplicate tespit + silme (pHash)
  4. Identity outlier tespiti (cosine similarity)

Kullanim:
    python scripts/28_quality_pipeline.py
    python scripts/28_quality_pipeline.py --dry-run
    python scripts/28_quality_pipeline.py --skip-outlier  (sadece 1-3)
"""
import sys, os, shutil, json, time, argparse, hashlib
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

CUSTOM_DIR = PROJECT_ROOT / "dataset" / "faces" / "custom_team"
QUARANTINE = CUSTOM_DIR / "_quarantine"
PERSONS = ["beyza", "seydi", "vasif_nabiyev", "mustafa_ulutas"]


# ═══ ADIM 1: EXIF Temizleme ═══════════════════════════════════

def strip_exif(img_path):
    """EXIF verisini sil, sadece pixel data tut."""
    try:
        img = Image.open(img_path)
        data = list(img.getdata())
        clean = Image.new(img.mode, img.size)
        clean.putdata(data)
        clean.save(str(img_path), "JPEG", quality=95)
        return True
    except Exception:
        return False


def step1_exif(all_files, dry_run=False):
    print("\n  [ADIM 1] EXIF/Metadata Temizleme")
    print(f"  {len(all_files)} dosya")
    if dry_run:
        print("  [DRY-RUN] Atlanıyor")
        return
    cleaned = 0
    for f in all_files:
        if strip_exif(f):
            cleaned += 1
    print(f"  -> {cleaned} dosya temizlendi")


# ═══ ADIM 2: Kalite Skoru (Laplacian Variance) ════════════════

def quality_score(img_path):
    """Laplacian variance ile bulanıklık/kalite skoru."""
    try:
        import cv2
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return 0.0
        return cv2.Laplacian(img, cv2.CV_64F).var()
    except Exception:
        return 0.0


def step2_quality(person_files, dry_run=False, min_score=15.0):
    print(f"\n  [ADIM 2] Kalite Skoru (min: {min_score})")
    total_removed = 0
    for person, files in person_files.items():
        scores = [(f, quality_score(f)) for f in files]
        low = [(f, s) for f, s in scores if s < min_score]
        good = [s for _, s in scores if s >= min_score]

        avg = np.mean(good) if good else 0
        print(f"    {person:20s}: {len(files):>4} dosya | "
              f"ort={avg:.1f} | dusuk={len(low)}")

        if low and not dry_run:
            q_dir = QUARANTINE / "low_quality" / person
            q_dir.mkdir(parents=True, exist_ok=True)
            for f, s in low:
                shutil.move(str(f), str(q_dir / f.name))
                # real klasorundan de sil
                real_copy = f.parent.parent.parent / "real" / person / f.name
                if real_copy.exists():
                    real_copy.unlink()
            total_removed += len(low)

    print(f"  -> {total_removed} dusuk kaliteli dosya karantinaya alindi")
    return total_removed


# ═══ ADIM 3: Near-Duplicate Tespit (pHash) ════════════════════

def perceptual_hash(img_path, hash_size=8):
    """Basit pHash: DCT tabanlı perceptual hash."""
    try:
        img = Image.open(img_path).convert("L").resize((hash_size + 1, hash_size), Image.LANCZOS)
        arr = np.array(img, dtype=np.float64)
        # Yatay gradient: komsular arasi fark
        diff = arr[:, 1:] > arr[:, :-1]
        return diff.flatten().tobytes()
    except Exception:
        return b""


def hamming(h1, h2):
    """Iki hash arasi Hamming mesafesi."""
    if len(h1) != len(h2):
        return 999
    return sum(a != b for a, b in zip(h1, h2))


def step3_dedup(person_files, dry_run=False, threshold=5):
    print(f"\n  [ADIM 3] Near-Duplicate Tespit (hamming < {threshold})")
    total_removed = 0

    for person, files in person_files.items():
        hashes = []
        for f in files:
            h = perceptual_hash(f)
            if h:
                hashes.append((f, h))

        # Her goruntu ciftini karsilastir
        duplicates = set()
        for i in range(len(hashes)):
            if hashes[i][0] in duplicates:
                continue
            for j in range(i + 1, len(hashes)):
                if hashes[j][0] in duplicates:
                    continue
                dist = hamming(hashes[i][1], hashes[j][1])
                if dist < threshold:
                    # Augmented olani sil, orijinali tut
                    if "_aug_" in hashes[j][0].name:
                        duplicates.add(hashes[j][0])
                    elif "_aug_" in hashes[i][0].name:
                        duplicates.add(hashes[i][0])
                    else:
                        duplicates.add(hashes[j][0])

        print(f"    {person:20s}: {len(hashes):>4} hash | "
              f"{len(duplicates)} duplicate")

        if duplicates and not dry_run:
            q_dir = QUARANTINE / "duplicates" / person
            q_dir.mkdir(parents=True, exist_ok=True)
            for f in duplicates:
                if f.exists():
                    shutil.move(str(f), str(q_dir / f.name))
                    real_copy = f.parent.parent.parent / "real" / person / f.name
                    if real_copy.exists():
                        real_copy.unlink()
            total_removed += len(duplicates)

    print(f"  -> {total_removed} duplicate dosya karantinaya alindi")
    return total_removed


# ═══ ADIM 4: Identity Outlier Tespiti ═════════════════════════

def step4_outlier(person_files, dry_run=False, threshold=0.45):
    """InsightFace embedding ile outlier tespit."""
    print(f"\n  [ADIM 4] Identity Outlier Tespiti (cosine < {threshold})")
    try:
        import insightface
        from numpy.linalg import norm
    except ImportError:
        print("  insightface bulunamadi, atlaniyor")
        return 0

    # Model yukle
    app = insightface.app.FaceAnalysis(name="buffalo_l",
                                        providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(256, 256))

    total_removed = 0

    for person, files in person_files.items():
        embeddings = []
        valid_files = []

        for f in files[:50]:  # Ilk 50 ile centroid hesapla (hiz)
            try:
                import cv2
                img = cv2.imread(str(f))
                faces = app.get(img)
                if faces:
                    embeddings.append(faces[0].embedding)
                    valid_files.append(f)
            except Exception:
                pass

        if len(embeddings) < 5:
            print(f"    {person:20s}: yeterli embedding yok ({len(embeddings)})")
            continue

        # Centroid hesapla
        emb_arr = np.array(embeddings)
        centroid = emb_arr.mean(axis=0)
        centroid = centroid / norm(centroid)

        # Tum dosyalari kontrol et
        outliers = []
        for f in files:
            try:
                import cv2
                img = cv2.imread(str(f))
                faces = app.get(img)
                if not faces:
                    outliers.append((f, 0.0))
                    continue
                emb = faces[0].embedding
                emb = emb / norm(emb)
                sim = np.dot(centroid, emb)
                if sim < threshold:
                    outliers.append((f, sim))
            except Exception:
                pass

        print(f"    {person:20s}: {len(files):>4} dosya | "
              f"{len(outliers)} outlier")

        if outliers and not dry_run:
            q_dir = QUARANTINE / "outliers" / person
            q_dir.mkdir(parents=True, exist_ok=True)
            for f, sim in outliers:
                if f.exists():
                    shutil.move(str(f), str(q_dir / f.name))
                    real_copy = f.parent.parent.parent / "real" / person / f.name
                    if real_copy.exists():
                        real_copy.unlink()
            total_removed += len(outliers)

    print(f"  -> {total_removed} outlier dosya karantinaya alindi")
    return total_removed


# ═══ ANA ═══════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-outlier", action="store_true",
                        help="Identity outlier adimini atla")
    args = parser.parse_args()

    print("=" * 55)
    print("  Entegrasyon Oncesi Kalite Pipeline")
    if args.dry_run:
        print("  [DRY-RUN]")
    print("=" * 55)

    # Dosyalari topla
    person_files = {}
    all_files = []
    for person in PERSONS:
        d = CUSTOM_DIR / "faces" / person
        if not d.exists():
            continue
        files = sorted(d.glob("*.jpg"))
        person_files[person] = files
        all_files.extend(files)

    print(f"\n  Toplam: {len(all_files)} dosya, {len(person_files)} kisi")
    for p, f in person_files.items():
        orig = len([x for x in f if "_aug_" not in x.name])
        aug  = len(f) - orig
        print(f"    {p:20s}: {len(f):>4} ({orig} orijinal + {aug} augmented)")

    t0 = time.time()

    # Adim 1: EXIF
    step1_exif(all_files, dry_run=args.dry_run)

    # Dosyalari yeniden oku (adim 2 icin guncel liste)
    person_files_fresh = {}
    for person in PERSONS:
        d = CUSTOM_DIR / "faces" / person
        if d.exists():
            person_files_fresh[person] = sorted(d.glob("*.jpg"))

    # Adim 2: Kalite
    step2_quality(person_files_fresh, dry_run=args.dry_run)

    # Dosyalari yeniden oku
    person_files_fresh2 = {}
    for person in PERSONS:
        d = CUSTOM_DIR / "faces" / person
        if d.exists():
            person_files_fresh2[person] = sorted(d.glob("*.jpg"))

    # Adim 3: Dedup
    step3_dedup(person_files_fresh2, dry_run=args.dry_run)

    # Adim 4: Outlier
    if not args.skip_outlier:
        person_files_fresh3 = {}
        for person in PERSONS:
            d = CUSTOM_DIR / "faces" / person
            if d.exists():
                person_files_fresh3[person] = sorted(d.glob("*.jpg"))
        step4_outlier(person_files_fresh3, dry_run=args.dry_run)
    else:
        print("\n  [ADIM 4] Identity outlier atlandi (--skip-outlier)")

    elapsed = time.time() - t0

    # Final sayim
    print(f"\n{'='*55}")
    print(f"  PIPELINE TAMAMLANDI ({elapsed:.1f}s)")
    print(f"{'='*55}")
    for person in PERSONS:
        d = CUSTOM_DIR / "faces" / person
        n = len(list(d.glob("*.jpg"))) if d.exists() else 0
        rd = CUSTOM_DIR / "real" / person
        rn = len(list(rd.glob("*.jpg"))) if rd.exists() else 0
        print(f"    {person:20s}: faces={n:>4} | real={rn:>4}")

    # Karantina ozeti
    if QUARANTINE.exists():
        print(f"\n  Karantina ({QUARANTINE}):")
        for reason in sorted(QUARANTINE.iterdir()):
            if reason.is_dir():
                total_q = sum(1 for _ in reason.rglob("*.jpg"))
                print(f"    {reason.name:20s}: {total_q}")


if __name__ == "__main__":
    main()
