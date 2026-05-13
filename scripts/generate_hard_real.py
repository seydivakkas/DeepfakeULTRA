"""
GÖREV 1: Hard-Real Veri Üretimi
Mevcut REAL görsellerden beauty filter, HDR, düşük çözünürlük simülasyonu ile
5000+ hard-real görsel üretir.

Kullanım:
    python scripts/generate_hard_real.py
    python scripts/generate_hard_real.py --validate
"""
import os
import sys
import hashlib
import random
import csv
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

sys.path.insert(0, str(Path(__file__).parent.parent))

# 3-Katmanlı Leakage Kontrolü (G1)
try:
    from scripts.leakage_checker import (
        compute_md5, compute_phash, build_reference_index,
        check_leakage as check_leakage_3layer, ReferenceIndex,
    )
    HAS_LEAKAGE_CHECKER = True
except ImportError:
    HAS_LEAKAGE_CHECKER = False
    print("⚠️ leakage_checker modülü yüklenemedi, MD5-only mod aktif")

# Kaynak dizinler
BASE = Path(__file__).parent.parent / "dataset" / "faces"
OUTPUT = BASE / "hard_real"
SOURCES = {
    "ffhq_256": BASE / "ffhq_256" / "real",
    "celeba_hq": BASE / "celeba_hq" / "real",
    "vggface2": BASE / "vggface2" / "real",
    "utkface": BASE / "utkface" / "real",
}

# Hedef sayılar (toplam: 7000)
TARGETS = {
    "beauty_filter": 1500,
    "hdr_edited": 1500,
    "low_quality": 1500,
    "heavy_makeup": 1000,
    "profile_angle": 500,
    "screen_recapture": 1000,  # G1: Ekran fotoğrafı + moiré simülasyonu
}

# Hızlı MD5 hash
def file_hash(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def collect_source_images(sources: dict, limit_per_source: int = 10000) -> list:
    """Tüm kaynaklardan REAL görselleri topla."""
    images = []
    for name, src_dir in sources.items():
        if not src_dir.exists():
            print(f"  ⚠️ {name} bulunamadı: {src_dir}")
            continue
        files = list(src_dir.glob("*.jpg")) + list(src_dir.glob("*.png"))
        random.shuffle(files)
        files = files[:limit_per_source]
        for f in files:
            images.append({"path": f, "source": name})
    random.shuffle(images)
    print(f"  📂 {len(images)} kaynak görsel toplandı")
    return images


# ═══════════════════════════════════════════════════════════
# AUGMENTATION FONKSİYONLARI
# ═══════════════════════════════════════════════════════════

def apply_beauty_filter(img: Image.Image) -> Image.Image:
    """Instagram/TikTok beauty filter simülasyonu."""
    img_np = np.array(img, dtype=np.float32)

    # Bilateral filter (skin smoothing)
    if HAS_CV2:
        smoothed = cv2.bilateralFilter(
            img_np.astype(np.uint8), d=9,
            sigmaColor=random.randint(50, 85),
            sigmaSpace=random.randint(50, 85)
        )
        img_np = smoothed.astype(np.float32)
    else:
        img = img.filter(ImageFilter.SMOOTH_MORE)
        img_np = np.array(img, dtype=np.float32)

    # Saturation boost
    img = Image.fromarray(img_np.astype(np.uint8))
    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(random.uniform(1.1, 1.35))

    # Hafif sharpening
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(random.uniform(0.7, 1.1))

    # Brightness
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(random.uniform(1.02, 1.12))

    return img


def apply_hdr_edit(img: Image.Image) -> Image.Image:
    """HDR/Photoshop düzenleme simülasyonu."""
    img_np = np.array(img, dtype=np.uint8)

    if HAS_CV2:
        # CLAHE
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(
            clipLimit=random.uniform(2.0, 4.0),
            tileGridSize=(8, 8)
        )
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        img_np = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    img = Image.fromarray(img_np)

    # Gamma correction
    gamma = random.uniform(0.7, 1.4)
    img_np = np.array(img, dtype=np.float32) / 255.0
    img_np = np.clip(np.power(img_np, gamma) * 255, 0, 255).astype(np.uint8)
    img = Image.fromarray(img_np)

    # Color grading (warm/cool tint)
    img_np = np.array(img, dtype=np.float32)
    tint = random.choice(["warm", "cool", "vintage"])
    if tint == "warm":
        img_np[:, :, 0] = np.clip(img_np[:, :, 0] * random.uniform(1.02, 1.08), 0, 255)
        img_np[:, :, 2] = np.clip(img_np[:, :, 2] * random.uniform(0.92, 0.98), 0, 255)
    elif tint == "cool":
        img_np[:, :, 2] = np.clip(img_np[:, :, 2] * random.uniform(1.02, 1.08), 0, 255)
        img_np[:, :, 0] = np.clip(img_np[:, :, 0] * random.uniform(0.92, 0.98), 0, 255)
    else:  # vintage
        img_np[:, :, 1] = np.clip(img_np[:, :, 1] * random.uniform(0.95, 1.0), 0, 255)

    # Contrast
    img = Image.fromarray(img_np.astype(np.uint8))
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(random.uniform(1.1, 1.4))

    return img


def apply_low_quality(img: Image.Image) -> Image.Image:
    """Düşük kalite/eski telefon simülasyonu."""
    import io

    # Downscale → upscale
    target_res = random.choice([48, 64, 96, 128])
    w, h = img.size
    img = img.resize((target_res, target_res), Image.BILINEAR)
    img = img.resize((w, h), Image.BILINEAR)

    # Gaussian noise
    img_np = np.array(img, dtype=np.float32)
    sigma = random.uniform(3, 12)
    noise = np.random.normal(0, sigma, img_np.shape)
    img_np = np.clip(img_np + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(img_np)

    # Ağır JPEG sıkıştırma
    quality = random.randint(20, 50)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    img = Image.open(buf).convert("RGB")

    return img


def apply_heavy_makeup(img: Image.Image) -> Image.Image:
    """Ağır makyaj simülasyonu."""
    # Color channel shift (lip/eye)
    img_np = np.array(img, dtype=np.float32)

    # Saturation boost (makyaj etkisi)
    img = Image.fromarray(img_np.astype(np.uint8))
    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(random.uniform(1.25, 1.55))

    # Contrast artışı
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(random.uniform(1.1, 1.3))

    # Skin smoothing
    if HAS_CV2:
        img_np = np.array(img, dtype=np.uint8)
        smoothed = cv2.bilateralFilter(img_np, d=7, sigmaColor=60, sigmaSpace=60)
        img = Image.fromarray(smoothed)

    # Brightness
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(random.uniform(1.05, 1.15))

    return img


def apply_profile_angle(img: Image.Image) -> Image.Image:
    """Profil/açılı yüz simülasyonu."""
    w, h = img.size
    img_np = np.array(img, dtype=np.uint8)

    if HAS_CV2:
        # Affine rotation
        angle = random.uniform(-30, 30)
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img_np, M, (w, h),
                                  borderMode=cv2.BORDER_REFLECT)

        # Perspective transform
        pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        dx = random.randint(10, 40)
        dy = random.randint(5, 20)
        pts2 = np.float32([
            [dx, dy], [w - dx, -dy],
            [-dx, h - dy], [w + dx, h + dy]
        ])
        M2 = cv2.getPerspectiveTransform(pts1, pts2)
        result = cv2.warpPerspective(rotated, M2, (w, h),
                                      borderMode=cv2.BORDER_REFLECT)
        img = Image.fromarray(result)
    else:
        img = img.rotate(random.uniform(-25, 25), expand=False, fillcolor=(128, 128, 128))

    return img


AUGMENT_FNS = {
    "beauty_filter": apply_beauty_filter,
    "hdr_edited": apply_hdr_edit,
    "low_quality": apply_low_quality,
    "heavy_makeup": apply_heavy_makeup,
    "profile_angle": apply_profile_angle,
    "screen_recapture": apply_screen_recapture,
}


def apply_screen_recapture(img: Image.Image) -> Image.Image:
    """
    Screen recapture simülasyonu (G1).
    Ekran fotoğrafı çekerken oluşan moiré paterni, renk uzayı kayması
    ve ekran parıltısını simüle eder.
    """
    import io
    img_np = np.array(img, dtype=np.float32)
    h, w = img_np.shape[:2]

    # Moiré pattern (sinusoidal grid overlay)
    freq_x = random.uniform(0.05, 0.15)
    freq_y = random.uniform(0.05, 0.15)
    x_grid = np.arange(w)
    y_grid = np.arange(h)
    xx, yy = np.meshgrid(x_grid, y_grid)
    moire = np.sin(2 * np.pi * freq_x * xx) * np.sin(2 * np.pi * freq_y * yy)
    moire_strength = random.uniform(5, 20)
    for ch in range(3):
        img_np[:, :, ch] += moire * moire_strength

    # Renk uzayı kayması (ekran beyaz dengesi farkı)
    color_shift = np.array([
        random.uniform(-15, 15),
        random.uniform(-10, 10),
        random.uniform(-20, 20),
    ])
    img_np += color_shift

    # Ekran parıltısı (vignette + brightness gradient)
    center_x, center_y = w // 2, h // 2
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)
    max_dist = np.sqrt(center_x ** 2 + center_y ** 2)
    vignette = 1.0 - (dist / max_dist) * random.uniform(0.1, 0.3)
    for ch in range(3):
        img_np[:, :, ch] *= vignette

    # Hafif gamma kayması (ekran gamma)
    gamma = random.uniform(0.85, 1.15)
    img_np = np.clip(img_np, 0, 255)
    img_np = np.power(img_np / 255.0, gamma) * 255

    img_np = np.clip(img_np, 0, 255).astype(np.uint8)
    img = Image.fromarray(img_np)

    # Hafif JPEG (kameradan çekim)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=random.randint(60, 80))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def process_single(args):
    """Tek bir görsel için augmentation uygula."""
    src_path, category, out_dir, idx = args
    try:
        img = Image.open(src_path).convert("RGB")
        # 224x224 resize (model input boyutu)
        img = img.resize((224, 224), Image.LANCZOS)
        # Augmentation uygula
        augmented = AUGMENT_FNS[category](img)
        # Kaydet
        out_name = f"{category}_{idx:05d}.jpg"
        out_path = out_dir / out_name
        augmented.save(out_path, quality=95)
        return {
            "file": out_name,
            "source": str(src_path.name),
            "category": category,
            "hash": file_hash(out_path),
            "status": "ok",
        }
    except Exception as e:
        return {"file": str(src_path), "category": category, "status": f"error: {e}"}


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Hard-Real veri üretimi")
    parser.add_argument("--validate", action="store_true", help="Sadece doğrulama yap")
    args = parser.parse_args()

    if args.validate:
        validate()
        return

    print("=" * 60)
    print("GÖREV 1: Hard-Real Veri Üretimi")
    print("=" * 60)

    # Kaynakları topla
    print("\n📂 Kaynak görseller toplanıyor...")
    all_images = collect_source_images(SOURCES)
    if not all_images:
        print("❌ Kaynak görsel bulunamadı!")
        return

    # Çıktı dizinlerini oluştur
    OUTPUT.mkdir(parents=True, exist_ok=True)
    metadata_rows = []
    total_generated = 0

    for category, target_count in TARGETS.items():
        cat_dir = OUTPUT / category
        cat_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n🔧 {category}: {target_count} görsel üretiliyor...")

        # Kaynaktan örnekle
        sampled = random.sample(all_images, min(target_count, len(all_images)))
        tasks = [(s["path"], category, cat_dir, i) for i, s in enumerate(sampled)]

        # Paralel işleme
        results = []
        with ProcessPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(process_single, t): t for t in tasks}
            for future in as_completed(futures):
                r = future.result()
                results.append(r)

        ok_count = sum(1 for r in results if r["status"] == "ok")
        metadata_rows.extend([r for r in results if r["status"] == "ok"])
        total_generated += ok_count
        print(f"  ✅ {ok_count}/{target_count} üretildi")

    # Metadata kaydet
    meta_path = OUTPUT / "metadata.csv"
    with open(meta_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "source", "category", "hash", "status"])
        writer.writeheader()
        writer.writerows(metadata_rows)

    # Duplicate kontrolü
    hashes = [r["hash"] for r in metadata_rows]
    unique = len(set(hashes))
    dupes = len(hashes) - unique

    print(f"\n{'=' * 60}")
    print(f"📊 SONUÇ:")
    print(f"  Toplam üretilen: {total_generated}")
    print(f"  Unique: {unique}")
    print(f"  Duplicate: {dupes}")
    print(f"  Metadata: {meta_path}")
    print(f"{'=' * 60}")

    if dupes > 0:
        print(f"⚠️ {dupes} duplicate bulundu — temizleme gerekli")

    # 3-Katmanlı leakage kontrolü (G1)
    if HAS_LEAKAGE_CHECKER:
        print(f"\n🔐 3-Katmanlı Leakage Kontrolü...")
        # Eğitim seti index'i oluştur
        split_dir = Path(__file__).parent.parent / "dataset" / "faces_split" / "train"
        if split_dir.exists():
            ref_index = build_reference_index(split_dir, use_embedding=False)
            leaked = 0
            for row in metadata_rows:
                path = OUTPUT / row["category"] / row["file"]
                if path.exists():
                    result = check_leakage_3layer(path, ref_index)
                    if result.is_leaked:
                        leaked += 1
                        print(f"  ⚠️ Leakage: {row['file']} ({result.leakage_type})")
            print(f"  {'\u2705' if leaked == 0 else '\u274c'} Leakage kontrol: {leaked} tespit")

    print("✅ GÖREV_1_TAMAMLANDI")


def validate():
    """Üretilen verileri doğrula."""
    print("🔍 Hard-Real doğrulama...")

    if not OUTPUT.exists():
        print("❌ hard_real/ dizini bulunamadı. Önce üretimi çalıştırın.")
        return

    for category, target in TARGETS.items():
        cat_dir = OUTPUT / category
        if not cat_dir.exists():
            print(f"❌ {category}/ bulunamadı")
            continue
        count = len(list(cat_dir.glob("*.jpg")))
        status = "✅" if count >= target * 0.9 else "⚠️"
        print(f"  {status} {category}: {count}/{target}")

    # Metadata kontrol
    meta_path = OUTPUT / "metadata.csv"
    if meta_path.exists():
        import csv as csv_mod
        with open(meta_path, "r", encoding="utf-8") as f:
            rows = list(csv_mod.DictReader(f))
        print(f"\n  📋 Metadata: {len(rows)} kayıt")
        # Duplicate kontrol
        hashes = [r["hash"] for r in rows]
        unique = len(set(hashes))
        print(f"  📋 Unique hash: {unique}/{len(hashes)}")
    else:
        print("  ⚠️ metadata.csv bulunamadı")


if __name__ == "__main__":
    main()
