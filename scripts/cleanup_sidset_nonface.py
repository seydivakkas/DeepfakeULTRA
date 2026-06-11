"""
sidset/real dizinindeki yuz odakli olmayan resimleri tespit edip siler.
OpenCV DNN face detector ile hizli ve dogru yuz algilama yapar.

Kriter: Resimde en az 1 yuz OLMALI ve en buyuk yuz, resim alaninin
en az %3'unu kaplamalıdir (yakin cekim yuz portresi kriteri).
"""
import os
import sys
import cv2
import numpy as np
import json
import shutil
from pathlib import Path
from datetime import datetime

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Yollar
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SIDSET_REAL = PROJECT_ROOT / "dataset" / "faces" / "sidset" / "real"
QUARANTINE_DIR = PROJECT_ROOT / "dataset" / "faces" / "sidset" / "_non_face_quarantine"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Yuz algilama icin minimum yuz alani orani (resim alanina gore)
MIN_FACE_AREA_RATIO = 0.03  # %3 - yakin cekim yuz portresi


def create_face_detector():
    """OpenCV Haar cascade face detector olustur."""
    # Proje dizinindeki kopyalari kullan (Turkce karakter encoding sorunu icin)
    script_dir = Path(__file__).resolve().parent
    cascade_path = str(script_dir / "cv2_data" / "haarcascade_frontalface_default.xml")
    profile_path = str(script_dir / "cv2_data" / "haarcascade_profileface.xml")
    
    if not os.path.exists(cascade_path):
        # Fallback: OpenCV varsayilan yol
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        print(f"[HATA] Haar cascade yuklenemedi: {cascade_path}")
        sys.exit(1)
    
    profile_cascade = None
    if os.path.exists(profile_path):
        profile_cascade = cv2.CascadeClassifier(profile_path)
    
    return face_cascade, profile_cascade


def detect_faces(img_path, face_cascade, profile_cascade=None):
    """
    Resimde yuz algila. 
    Returns: (yuz_var, en_buyuk_yuz_orani, yuz_sayisi)
    """
    img = cv2.imread(str(img_path))
    if img is None:
        return False, 0.0, 0
    
    h, w = img.shape[:2]
    img_area = h * w
    
    # Performans icin buyuk resimleri kucult
    max_dim = 800
    scale = 1.0
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, None, fx=scale, fy=scale)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    
    # Frontal yuz algilama
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    # Profil yuz algilama (frontal bulunamadiysa)
    if len(faces) == 0 and profile_cascade is not None:
        faces = profile_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(30, 30)
        )
        # Yatik profil de dene
        if len(faces) == 0:
            flipped = cv2.flip(gray, 1)
            faces = profile_cascade.detectMultiScale(
                flipped,
                scaleFactor=1.1,
                minNeighbors=4,
                minSize=(30, 30)
            )
    
    if len(faces) == 0:
        return False, 0.0, 0
    
    # En buyuk yuzu bul (scale'i geri al)
    max_face_area = 0
    for (x, y, fw, fh) in faces:
        face_area = (fw / scale) * (fh / scale)
        max_face_area = max(max_face_area, face_area)
    
    face_ratio = max_face_area / img_area
    
    return True, face_ratio, len(faces)


def main():
    print("=" * 60)
    print("Sidset Non-Face Temizleme Scripti")
    print("=" * 60)
    
    if not SIDSET_REAL.exists():
        print(f"[HATA] Dizin bulunamadi: {SIDSET_REAL}")
        return
    
    # Karantina dizini olustur
    QUARANTINE_DIR.mkdir(parents=True, exist_ok=True)
    
    # Yuz algilayici olustur
    print("\n[1/3] Yuz algilayici yukleniyor...")
    face_cascade, profile_cascade = create_face_detector()
    print("  -> Haar Cascade (frontal + profil) yuklendi.")
    
    # Tum resimleri listele
    print("\n[2/3] Resimler taraniyor...")
    all_images = sorted([
        f for f in SIDSET_REAL.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTS
    ])
    total = len(all_images)
    print(f"  -> {total} resim bulundu.")
    
    # Yuz algilama
    print("\n[3/3] Yuz algilama ve temizleme basliyor...")
    
    no_face = []        # Hic yuz yok
    small_face = []     # Yuz var ama cok kucuk (yakin cekim degil)
    face_ok = []        # Yuz odakli
    errors = []         # Okunamayan
    
    batch_size = 1000
    for i, img_path in enumerate(all_images):
        if (i + 1) % batch_size == 0 or i == 0:
            pct = ((i + 1) / total) * 100
            print(f"  [{i+1}/{total}] ({pct:.1f}%) - Silinen: {len(no_face) + len(small_face)}")
        
        try:
            has_face, ratio, count = detect_faces(img_path, face_cascade, profile_cascade)
            
            if not has_face:
                no_face.append(img_path)
                # Karantinaya tasi
                shutil.move(str(img_path), str(QUARANTINE_DIR / img_path.name))
            elif ratio < MIN_FACE_AREA_RATIO:
                small_face.append((img_path, ratio))
                # Cok kucuk yuz - karantinaya tasi
                shutil.move(str(img_path), str(QUARANTINE_DIR / img_path.name))
            else:
                face_ok.append(img_path)
        except Exception as e:
            errors.append((img_path, str(e)))
    
    # Sonuc
    removed = len(no_face) + len(small_face)
    print(f"\n{'=' * 60}")
    print("TAMAMLANDI!")
    print(f"{'=' * 60}")
    print(f"\nSonuclar:")
    print(f"  Toplam taranan:        {total}")
    print(f"  Yuz odakli (kalan):    {len(face_ok)}")
    print(f"  Yuz yok (silinen):     {len(no_face)}")
    print(f"  Kucuk yuz (silinen):   {len(small_face)}")
    print(f"  Hata:                  {len(errors)}")
    print(f"  Toplam karantinaya:    {removed}")
    print(f"\nKarantina dizini: {QUARANTINE_DIR}")
    
    # Rapor kaydet
    report = {
        "timestamp": datetime.now().isoformat(),
        "source": str(SIDSET_REAL),
        "total_scanned": total,
        "face_ok": len(face_ok),
        "no_face_removed": len(no_face),
        "small_face_removed": len(small_face),
        "errors": len(errors),
        "min_face_area_ratio": MIN_FACE_AREA_RATIO,
        "no_face_files": [f.name for f in no_face],
        "small_face_files": [(str(f.name), r) for f, r in small_face],
        "error_files": [(str(f.name), e) for f, e in errors],
    }
    
    report_path = QUARANTINE_DIR / "cleanup_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"  Rapor: {report_path}")


if __name__ == "__main__":
    main()
