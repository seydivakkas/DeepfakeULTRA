"""
Egitim/val/test setindeki sidset_real_* dosyalardan
yuz odakli olmayanlari siler.

Haar Cascade ile yuz algilama yapar.
Yuz yoksa veya yuz cok kucukse (<%3) -> sil.
"""
import os
import sys
import cv2
import json
import time
from pathlib import Path
from datetime import datetime

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPLIT_DIR = PROJECT_ROOT / "dataset" / "faces_split_v2"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MIN_FACE_AREA_RATIO = 0.03  # %3

# Haar cascade yolunu ayarla
SCRIPT_DIR = Path(__file__).resolve().parent
CASCADE_PATH = str(SCRIPT_DIR / "cv2_data" / "haarcascade_frontalface_default.xml")
PROFILE_PATH = str(SCRIPT_DIR / "cv2_data" / "haarcascade_profileface.xml")


def load_cascades():
    """Haar cascade yukle."""
    if not os.path.exists(CASCADE_PATH):
        print(f"[HATA] Cascade bulunamadi: {CASCADE_PATH}")
        sys.exit(1)

    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    if face_cascade.empty():
        print("[HATA] Haar cascade yuklenemedi!")
        sys.exit(1)

    profile_cascade = None
    if os.path.exists(PROFILE_PATH):
        profile_cascade = cv2.CascadeClassifier(PROFILE_PATH)

    return face_cascade, profile_cascade


def detect_face(img_path, face_cascade, profile_cascade):
    """Resimde yuz var mi ve buyuk mu kontrol et."""
    img = cv2.imread(str(img_path))
    if img is None:
        return False, 0.0

    h, w = img.shape[:2]
    img_area = h * w

    # Buyuk resimleri kucult
    max_dim = 800
    scale = 1.0
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, None, fx=scale, fy=scale)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    # Frontal
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=4,
        minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Profil (frontal bulunamadiysa)
    if len(faces) == 0 and profile_cascade is not None:
        faces = profile_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30)
        )
        if len(faces) == 0:
            flipped = cv2.flip(gray, 1)
            faces = profile_cascade.detectMultiScale(
                flipped, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30)
            )

    if len(faces) == 0:
        return False, 0.0

    max_face_area = max((fw / scale) * (fh / scale) for (_, _, fw, fh) in faces)
    return True, max_face_area / img_area


def clean_split(split_name, face_cascade, profile_cascade):
    """Tek bir split (train/val/test) icindeki sidset dosyalarini temizle."""
    real_dir = SPLIT_DIR / split_name / "real"
    if not real_dir.exists():
        print(f"  [{split_name}] Dizin bulunamadi, atlaniyor.")
        return 0, 0, 0

    # sidset_real_* dosyalarini bul
    sidset_files = sorted([
        f for f in real_dir.iterdir()
        if f.is_file() and f.name.startswith("sidset_real_")
        and f.suffix.lower() in IMAGE_EXTS
    ])

    total = len(sidset_files)
    if total == 0:
        print(f"  [{split_name}] Sidset dosyasi bulunamadi.")
        return 0, 0, 0

    print(f"  [{split_name}] {total:,} sidset dosyasi taraniyor...")

    deleted_no_face = 0
    deleted_small_face = 0
    kept = 0
    batch = max(1, total // 10)

    for i, img_path in enumerate(sidset_files):
        if (i + 1) % batch == 0:
            pct = (i + 1) / total * 100
            print(f"    [{i+1}/{total}] ({pct:.0f}%) silinen: {deleted_no_face + deleted_small_face}")

        try:
            has_face, ratio = detect_face(img_path, face_cascade, profile_cascade)

            if not has_face:
                img_path.unlink()
                deleted_no_face += 1
            elif ratio < MIN_FACE_AREA_RATIO:
                img_path.unlink()
                deleted_small_face += 1
            else:
                kept += 1
        except Exception:
            # Hata durumunda koru
            kept += 1

    print(f"    -> {split_name}: silinen={deleted_no_face + deleted_small_face} "
          f"(yuz_yok={deleted_no_face}, kucuk_yuz={deleted_small_face}), kalan={kept}")

    return deleted_no_face, deleted_small_face, kept


def main():
    t0 = time.time()
    print("=" * 60)
    print("Egitim Seti Sidset Non-Face Temizleme")
    print("=" * 60)

    face_cascade, profile_cascade = load_cascades()
    print("Haar Cascade yuklendi.\n")

    report = {"timestamp": datetime.now().isoformat(), "splits": {}}
    total_deleted = 0
    total_kept = 0

    for split in ["train", "val", "test"]:
        no_face, small_face, kept = clean_split(split, face_cascade, profile_cascade)
        deleted = no_face + small_face
        total_deleted += deleted
        total_kept += kept
        report["splits"][split] = {
            "no_face_deleted": no_face,
            "small_face_deleted": small_face,
            "kept": kept
        }

    elapsed = time.time() - t0

    # Sonuc
    print(f"\n{'=' * 60}")
    print("TAMAMLANDI!")
    print(f"{'=' * 60}")
    print(f"  Toplam silinen:  {total_deleted:,}")
    print(f"  Toplam kalan:    {total_kept:,}")
    print(f"  Sure:            {elapsed/60:.1f} dk")

    # Guncel dosya sayilari
    print(f"\n  Guncel dosya sayilari:")
    for split in ["train", "val", "test"]:
        for label in ["real", "fake"]:
            d = SPLIT_DIR / split / label
            if d.exists():
                count = sum(1 for f in d.iterdir() if f.is_file())
                print(f"    {split}/{label}: {count:,}")

    # Rapor kaydet
    report["total_deleted"] = total_deleted
    report["total_kept"] = total_kept
    report["elapsed_seconds"] = elapsed

    report_path = PROJECT_ROOT / "dataset" / "sidset_train_cleanup_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n  Rapor: {report_path}")


if __name__ == "__main__":
    main()
