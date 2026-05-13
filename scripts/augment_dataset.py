"""
DeepFake Detection System — Run #3 Offline Veri Artırımı
SPOOF 5×, REAL(CASIA) 8×, REAL(FF++) 2× çarpanlarla veri seti genişletme.
Sınıf-özel augmentasyonlar: JPEG compression, moiré pattern, webcam noise vb.

Kullanım:
    python scripts/augment_dataset.py
    python scripts/augment_dataset.py --dry-run    # Sadece rapor, dosya üretmez
"""

import sys
import os
import json
import hashlib
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
from concurrent.futures import ProcessPoolExecutor, as_completed

# Proje kök dizinini sys.path'e ekle
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import paths

# ═══════════════════════════════════════════════════════════
# AUGMENTASYON TEKNİKLERİ
# ═══════════════════════════════════════════════════════════

def horizontal_flip(img: Image.Image) -> Image.Image:
    return img.transpose(Image.FLIP_LEFT_RIGHT)


def random_rotation(img: Image.Image, max_deg: int = 20) -> Image.Image:
    angle = np.random.uniform(-max_deg, max_deg)
    return img.rotate(angle, resample=Image.BILINEAR, fillcolor=(0, 0, 0))


def random_affine(img: Image.Image) -> Image.Image:
    """Shear + scale dönüşümü."""
    w, h = img.size
    shear_x = np.random.uniform(-0.05, 0.05)
    shear_y = np.random.uniform(-0.05, 0.05)
    scale = np.random.uniform(0.92, 1.08)
    # Affine matris: (a, b, c, d, e, f)
    coeffs = (scale, shear_x, -shear_x * w / 2,
              shear_y, scale, -shear_y * h / 2)
    return img.transform((w, h), Image.AFFINE, coeffs, resample=Image.BILINEAR)


def color_jitter(img: Image.Image) -> Image.Image:
    """Brightness, contrast, saturation, hue varyasyonu."""
    enhancers = [
        (ImageEnhance.Brightness, np.random.uniform(0.7, 1.3)),
        (ImageEnhance.Contrast, np.random.uniform(0.7, 1.3)),
        (ImageEnhance.Color, np.random.uniform(0.8, 1.2)),
    ]
    np.random.shuffle(enhancers)
    for enhancer_cls, factor in enhancers:
        img = enhancer_cls(img).enhance(factor)
    return img


def gaussian_blur(img: Image.Image) -> Image.Image:
    radius = np.random.uniform(0.5, 2.5)
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def random_perspective(img: Image.Image) -> Image.Image:
    """Hafif perspektif bozulma."""
    w, h = img.size
    offset = int(min(w, h) * 0.04)
    coeffs = []
    for _ in range(8):
        coeffs.append(np.random.uniform(-offset, offset))
    # 4 köşe noktasını kaydır
    pts1 = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    pts2 = pts1 + np.float32([
        [coeffs[0], coeffs[1]],
        [coeffs[2], coeffs[3]],
        [coeffs[4], coeffs[5]],
        [coeffs[6], coeffs[7]],
    ])
    # PIL perspective transform coefficients
    try:
        from PIL.ImageTransform import QuadTransform
        return img.transform(
            (w, h), Image.QUAD,
            pts2.flatten().tolist(),
            resample=Image.BILINEAR
        )
    except Exception:
        return img


def random_grayscale(img: Image.Image, p: float = 0.15) -> Image.Image:
    if np.random.random() < p:
        return ImageOps.grayscale(img).convert("RGB")
    return img


def clahe_equalize(img: Image.Image) -> Image.Image:
    """CLAHE histogram eşitleme — PIL tabanlı basit versiyon."""
    return ImageOps.autocontrast(img, cutoff=2)


def additive_noise(img: Image.Image) -> Image.Image:
    """Gaussian gürültü enjeksiyonu."""
    arr = np.array(img, dtype=np.float32)
    std = np.random.uniform(3.0, 12.0)
    noise = np.random.normal(0, std, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def random_crop_resize(img: Image.Image) -> Image.Image:
    """Rastgele kırpma + orijinal boyuta yeniden boyutlandırma."""
    w, h = img.size
    crop_ratio = np.random.uniform(0.85, 0.95)
    cw, ch = int(w * crop_ratio), int(h * crop_ratio)
    x = np.random.randint(0, w - cw + 1)
    y = np.random.randint(0, h - ch + 1)
    return img.crop((x, y, x + cw, y + ch)).resize((w, h), Image.BILINEAR)


def elastic_deformation(img: Image.Image) -> Image.Image:
    """Hafif elastik deformasyon (PIL tabanlı basit versiyon)."""
    # Basitleştirilmiş: küçük mesh warp
    w, h = img.size
    grid_size = 4
    mesh = []
    dw, dh = w // grid_size, h // grid_size
    for i in range(grid_size):
        for j in range(grid_size):
            x0, y0 = j * dw, i * dh
            x1, y1 = x0 + dw, y0 + dh
            dx = np.random.randint(-3, 4)
            dy = np.random.randint(-3, 4)
            mesh.append((
                (x0, y0, x1, y1),
                (x0 + dx, y0 + dy, x0 + dw + dx, y0, x1 + dx, y1 + dy, x0 + dx, y0 + dh + dy)
            ))
    try:
        return img.transform((w, h), Image.MESH, mesh, resample=Image.BILINEAR)
    except Exception:
        return img


# ═══════════════════════════════════════════════════════════
# SINIF-ÖZEL AUGMENTASYONLAR
# ═══════════════════════════════════════════════════════════

def jpeg_compression_artifact(img: Image.Image, quality_range=(30, 60)) -> Image.Image:
    """JPEG sıkıştırma artefaktları — spoof ve print efekti simülasyonu."""
    import io
    quality = np.random.randint(*quality_range)
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def moire_pattern(img: Image.Image) -> Image.Image:
    """Moiré desen simülasyonu — ekran fotoğrafı artefaktı."""
    arr = np.array(img, dtype=np.float32)
    h, w = arr.shape[:2]
    freq = np.random.uniform(0.05, 0.15)
    amplitude = np.random.uniform(8, 20)
    x = np.arange(w)
    y = np.arange(h)
    xx, yy = np.meshgrid(x, y)
    pattern = amplitude * np.sin(2 * np.pi * freq * (xx + yy))
    pattern = np.stack([pattern] * 3, axis=-1)
    arr = np.clip(arr + pattern, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def print_effect(img: Image.Image) -> Image.Image:
    """Baskı / mat kağıt efekti — kontrast düşürme + hafif blur."""
    img = ImageEnhance.Contrast(img).enhance(np.random.uniform(0.6, 0.8))
    img = ImageEnhance.Sharpness(img).enhance(np.random.uniform(0.5, 0.8))
    img = img.filter(ImageFilter.GaussianBlur(radius=np.random.uniform(0.3, 1.0)))
    return img


def specular_highlight(img: Image.Image) -> Image.Image:
    """Ekran yansıma gürültüsü — parlak noktalar ekleme."""
    arr = np.array(img, dtype=np.float32)
    h, w = arr.shape[:2]
    n_spots = np.random.randint(1, 4)
    for _ in range(n_spots):
        cx, cy = np.random.randint(0, w), np.random.randint(0, h)
        radius = np.random.randint(10, 40)
        yy, xx = np.ogrid[:h, :w]
        mask = ((xx - cx) ** 2 + (yy - cy) ** 2) < radius ** 2
        intensity = np.random.uniform(30, 80)
        arr[mask] = np.clip(arr[mask] + intensity, 0, 255)
    return Image.fromarray(arr.astype(np.uint8))


def webcam_noise(img: Image.Image) -> Image.Image:
    """Webcam sensör gürültüsü — Poisson noise."""
    arr = np.array(img, dtype=np.float64)
    # Poisson noise scale
    scale = np.random.uniform(0.3, 0.8)
    noisy = np.random.poisson(arr * scale) / scale
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy)


def exposure_variation(img: Image.Image) -> Image.Image:
    """Pozlama / ISO varyasyonu."""
    factor = np.random.uniform(0.6, 1.4)
    return ImageEnhance.Brightness(img).enhance(factor)


def low_light_simulation(img: Image.Image) -> Image.Image:
    """Düşük ışık simülasyonu — gamma düzeltme."""
    arr = np.array(img, dtype=np.float32) / 255.0
    gamma = np.random.uniform(0.4, 0.7)
    arr = np.power(arr, 1.0 / gamma)
    arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def color_temperature_shift(img: Image.Image) -> Image.Image:
    """Beyaz dengesi kayması — soğuk/sıcak renk tonu."""
    arr = np.array(img, dtype=np.float32)
    # Sıcak vs soğuk
    if np.random.random() > 0.5:
        # Sıcak: R+, B-
        arr[:, :, 0] = np.clip(arr[:, :, 0] + np.random.uniform(5, 15), 0, 255)
        arr[:, :, 2] = np.clip(arr[:, :, 2] - np.random.uniform(5, 15), 0, 255)
    else:
        # Soğuk: R-, B+
        arr[:, :, 0] = np.clip(arr[:, :, 0] - np.random.uniform(5, 15), 0, 255)
        arr[:, :, 2] = np.clip(arr[:, :, 2] + np.random.uniform(5, 15), 0, 255)
    return Image.fromarray(arr.astype(np.uint8))


def lens_distortion(img: Image.Image) -> Image.Image:
    """Lens distorsiyonu — barrel/pincushion efekti (basit)."""
    w, h = img.size
    # Küçük ölçekte kırp + yeniden boyutlandır
    scale = np.random.uniform(0.96, 1.04)
    nw, nh = int(w * scale), int(h * scale)
    resized = img.resize((nw, nh), Image.BILINEAR)
    # Ortalayarak kırp
    left = (nw - w) // 2
    top = (nh - h) // 2
    return resized.crop((max(0, left), max(0, top),
                         max(0, left) + w, max(0, top) + h)).resize((w, h), Image.BILINEAR)


# ═══════════════════════════════════════════════════════════
# AUGMENTASYON PIPELINE'LARI
# ═══════════════════════════════════════════════════════════

# Genel teknikler (tüm sınıflar)
GENERAL_TRANSFORMS = [
    horizontal_flip,
    random_rotation,
    random_affine,
    color_jitter,
    gaussian_blur,
    random_perspective,
    random_grayscale,
    clahe_equalize,
    additive_noise,
    random_crop_resize,
    elastic_deformation,
]

# Sınıf-özel teknikler
SPOOF_TRANSFORMS = [
    jpeg_compression_artifact,
    moire_pattern,
    print_effect,
    specular_highlight,
]

REAL_CASIA_TRANSFORMS = [
    webcam_noise,
    exposure_variation,
    low_light_simulation,
]

REAL_FFPP_TRANSFORMS = [
    color_temperature_shift,
    lens_distortion,
]


def generate_augmented_image(img: Image.Image, class_type: str) -> Image.Image:
    """Tek bir augmented görüntü üret."""
    result = img.copy()

    # Genel dönüşümlerden 3-5 tanesini rastgele seç
    n_general = np.random.randint(3, 6)
    selected = np.random.choice(len(GENERAL_TRANSFORMS), size=min(n_general, len(GENERAL_TRANSFORMS)), replace=False)
    for idx in selected:
        try:
            result = GENERAL_TRANSFORMS[idx](result)
        except Exception:
            pass

    # Sınıf-özel dönüşümlerden 1-2 tanesini uygula
    special_transforms = {
        "spoof": SPOOF_TRANSFORMS,
        "real_casia": REAL_CASIA_TRANSFORMS,
        "real_ffpp": REAL_FFPP_TRANSFORMS,
    }
    extras = special_transforms.get(class_type, [])
    if extras:
        n_extra = np.random.randint(1, min(3, len(extras) + 1))
        selected_extra = np.random.choice(len(extras), size=n_extra, replace=False)
        for idx in selected_extra:
            try:
                result = extras[idx](result)
            except Exception:
                pass

    return result


# ═══════════════════════════════════════════════════════════
# DOSYA İŞLEME
# ═══════════════════════════════════════════════════════════

SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


def collect_files(directory: Path) -> list:
    """Dizindeki tüm desteklenen görüntü dosyalarını topla."""
    files = []
    if not directory.exists():
        return files
    for f in sorted(directory.rglob("*")):
        if f.suffix.lower() in SUPPORTED_FORMATS and f.is_file():
            files.append(f)
    return files


def sha256_hash(filepath: Path) -> str:
    """Dosyanın SHA-256 hash'ini hesapla."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def augment_single_file(args):
    """Tek bir dosyayı augment et (multiprocessing uyumlu)."""
    src_path, output_dir, multiplier, class_type, file_idx = args
    results = []
    try:
        img = Image.open(src_path).convert("RGB")
    except Exception as e:
        return results

    for aug_idx in range(multiplier - 1):  # -1 çünkü orijinal zaten var
        try:
            augmented = generate_augmented_image(img, class_type)
            stem = Path(src_path).stem
            out_name = f"{stem}_aug{aug_idx:03d}.jpg"
            out_path = output_dir / out_name
            augmented.save(out_path, "JPEG", quality=85)
            results.append({
                "source": str(src_path),
                "output": str(out_path),
                "class_type": class_type,
                "aug_index": aug_idx,
            })
        except Exception:
            pass
    return results


# ═══════════════════════════════════════════════════════════
# ANA AUGMENTASYON MOTORU
# ═══════════════════════════════════════════════════════════

class DataAugmentationEngine:
    """Offline data augmentation motoru."""

    def __init__(self, dry_run: bool = False, max_workers: int = 4):
        self.dry_run = dry_run
        self.max_workers = max_workers
        self.manifest = []
        self.stats = defaultdict(int)

        # Hedef çarpanlar
        self.multipliers = {
            "spoof": 5,          # SPOOF: 15,606 → ~78,030
            "real_casia": 8,     # REAL(CASIA): 5,016 → ~40,128
            "real_ffpp": 2,      # REAL(FF++): 46,020 → ~92,040
            "fake": 1,           # FAKE: dokunma
        }

        # Kaynak dizinler
        self.sources = {
            "spoof": paths.CASIA_DIR / "train" / "spoof",
            "real_casia": paths.CASIA_DIR / "train" / "live",
            "real_ffpp": paths.FFPP_DIR / "train" / "real",
        }

        # Çıktı dizini
        self.output_base = paths.DATASET_DIR / "augmented_v3"

    def _get_output_dir(self, class_type: str) -> Path:
        """Augmented dosyaların kaydedileceği dizin."""
        label_map = {
            "spoof": "spoof",
            "real_casia": "live",
            "real_ffpp": "real",
        }
        label = label_map.get(class_type, class_type)
        source_map = {
            "spoof": "casia-fasd",
            "real_casia": "casia-fasd",
            "real_ffpp": "ff++",
        }
        source = source_map.get(class_type, "unknown")
        return self.output_base / source / "train" / label

    def process_class(self, class_type: str):
        """Belirli bir sınıfı augment et."""
        src_dir = self.sources.get(class_type)
        if not src_dir or not src_dir.exists():
            print(f"  ⚠️ {class_type}: kaynak dizin bulunamadı ({src_dir})")
            return

        multiplier = self.multipliers[class_type]
        if multiplier <= 1:
            print(f"  ⏭️ {class_type}: çarpan=1, atlıyorum")
            return

        files = collect_files(src_dir)
        if not files:
            print(f"  ⚠️ {class_type}: dosya bulunamadı")
            return

        output_dir = self._get_output_dir(class_type)
        total_expected = len(files) * (multiplier - 1)

        print(f"  📂 {class_type}: {len(files)} kaynak → {total_expected} yeni dosya üretilecek")
        print(f"     Çıktı: {output_dir}")

        if self.dry_run:
            self.stats[class_type] = total_expected
            return

        # Çıktı dizinini oluştur
        output_dir.mkdir(parents=True, exist_ok=True)

        # Argüman listesi
        tasks = [
            (str(f), output_dir, multiplier, class_type, i)
            for i, f in enumerate(files)
        ]

        # Paralel işleme
        produced = 0
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(augment_single_file, t): t for t in tasks}
            total = len(futures)
            done_count = 0
            for future in as_completed(futures):
                done_count += 1
                try:
                    results = future.result()
                    produced += len(results)
                    self.manifest.extend(results)
                except Exception as e:
                    pass

                if done_count % 500 == 0 or done_count == total:
                    pct = done_count / total * 100
                    print(f"     İlerleme: {done_count}/{total} ({pct:.1f}%) — {produced} dosya üretildi")

        self.stats[class_type] = produced
        print(f"  ✅ {class_type}: {produced} dosya üretildi")

    def run(self):
        """Tüm sınıfları augment et."""
        print("=" * 60)
        print("🔧 Run #3 Offline Veri Artırımı Başlıyor")
        print("=" * 60)

        for class_type in ["spoof", "real_casia", "real_ffpp"]:
            print(f"\n🔄 İşleniyor: {class_type}")
            self.process_class(class_type)

        self._print_report()
        self._save_manifest()

    def _print_report(self):
        """Sonuç raporu."""
        print("\n" + "=" * 60)
        print("📊 AUGMENTASYON RAPORU")
        print("=" * 60)

        # Orijinal sayılar
        originals = {
            "spoof": len(collect_files(self.sources.get("spoof", Path("/nonexistent")))),
            "real_casia": len(collect_files(self.sources.get("real_casia", Path("/nonexistent")))),
            "real_ffpp": len(collect_files(self.sources.get("real_ffpp", Path("/nonexistent")))),
            "fake": len(collect_files(paths.FFPP_DIR / "train" / "fake")),
        }

        total_original = sum(originals.values())
        total_augmented = sum(self.stats.values())
        total_final = total_original + total_augmented

        print(f"\n{'Sınıf':<15} {'Orijinal':>10} {'Augmented':>10} {'Toplam':>10} {'Oran':>8}")
        print("-" * 55)

        for ct in ["real_ffpp", "real_casia", "spoof", "fake"]:
            orig = originals[ct]
            aug = self.stats.get(ct, 0)
            total = orig + aug
            pct = total / total_final * 100 if total_final > 0 else 0
            print(f"  {ct:<13} {orig:>10,} {aug:>10,} {total:>10,} {pct:>7.1f}%")

        print("-" * 55)
        real_total = originals["real_ffpp"] + self.stats.get("real_ffpp", 0) + \
                     originals["real_casia"] + self.stats.get("real_casia", 0)
        fake_total = originals["fake"]
        spoof_total = originals["spoof"] + self.stats.get("spoof", 0)

        print(f"\n  REAL (birleşik):  {real_total:>10,}  ({real_total/total_final*100:.1f}%)")
        print(f"  FAKE:             {fake_total:>10,}  ({fake_total/total_final*100:.1f}%)")
        print(f"  SPOOF:            {spoof_total:>10,}  ({spoof_total/total_final*100:.1f}%)")
        print(f"  TOPLAM:           {total_final:>10,}")

    def _save_manifest(self):
        """Manifest dosyasını kaydet."""
        if self.dry_run:
            print("\n⏭️ Dry-run modu — manifest kaydedilmedi")
            return

        manifest_dir = self.output_base
        manifest_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        manifest_path = manifest_dir / f"manifest_v3_{timestamp}.json"

        manifest_data = {
            "version": "v3",
            "timestamp": datetime.now().isoformat(),
            "multipliers": self.multipliers,
            "stats": dict(self.stats),
            "total_produced": sum(self.stats.values()),
            "entries_count": len(self.manifest),
        }

        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, ensure_ascii=False, indent=2)

        print(f"\n💾 Manifest: {manifest_path}")


# ═══════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run #3 Offline Veri Artırımı")
    parser.add_argument("--dry-run", action="store_true",
                        help="Sadece rapor üret, dosya oluşturma")
    parser.add_argument("--workers", type=int, default=4,
                        help="Paralel işçi sayısı (varsayılan: 4)")
    parser.add_argument("--spoof-mult", type=int, default=5,
                        help="SPOOF çarpanı (varsayılan: 5)")
    parser.add_argument("--real-casia-mult", type=int, default=8,
                        help="REAL(CASIA) çarpanı (varsayılan: 8)")
    parser.add_argument("--real-ffpp-mult", type=int, default=2,
                        help="REAL(FF++) çarpanı (varsayılan: 2)")
    args = parser.parse_args()

    engine = DataAugmentationEngine(
        dry_run=args.dry_run,
        max_workers=args.workers,
    )

    # Özel çarpanlar
    engine.multipliers["spoof"] = args.spoof_mult
    engine.multipliers["real_casia"] = args.real_casia_mult
    engine.multipliers["real_ffpp"] = args.real_ffpp_mult

    engine.run()
