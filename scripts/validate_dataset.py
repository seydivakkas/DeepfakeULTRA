"""
Veri Seti Doğrulama Aracı
Pipeline çıktısını kontrol eder: dizin yapısı, dosya sayıları, bozuk dosyalar.

Kullanım:
    python scripts/validate_dataset.py
    python scripts/validate_dataset.py --path dataset
    python scripts/validate_dataset.py --fix  (bozuk dosyaları sil)
"""

import argparse
import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Beklenen dizin yapısı (V5)
EXPECTED_STRUCTURE = {
    "faces/ffpp": {
        "splits": [],
        "labels": ["real", "fake"],
    },
    "faces/df40": {
        "splits": [],
        "labels": ["real", "fake"],
    },
    "faces/celeba_hq": {
        "splits": [],
        "labels": ["real"],
    },
    "faces/ffhq_1024_filtered": {
        "splits": [],
        "labels": ["real"],
    },
    "faces/utkface": {
        "splits": [],
        "labels": ["real"],
    },
    "faces/sidset": {
        "splits": [],
        "labels": ["real", "fake"],
    },
    "faces_split/train": {
        "splits": [],
        "labels": ["real", "fake"],
    },
    "faces_split/val": {
        "splits": [],
        "labels": ["real", "fake"],
    },
}


def check_directory_structure(dataset_dir: Path) -> Dict:
    """Dizin yapısının doğruluğunu kontrol et."""
    results = {"status": "OK", "issues": [], "structure": {}}

    for ds_name, config in EXPECTED_STRUCTURE.items():
        ds_path = dataset_dir / ds_name
        ds_info = {"exists": ds_path.exists(), "splits": {}}

        if not ds_path.exists():
            results["issues"].append(f"[X] Dizin bulunamadı: {ds_name}")
            results["status"] = "WARNING"
        else:
            for split in config["splits"]:
                split_path = ds_path / split
                split_info = {"exists": split_path.exists(), "labels": {}}

                if not split_path.exists():
                    results["issues"].append(f"⚠️ Split dizini yok: {ds_name}/{split}")
                else:
                    for label in config["labels"]:
                        label_path = split_path / label
                        if label_path.exists():
                            files = [f for f in label_path.iterdir()
                                     if f.suffix.lower() in IMAGE_EXTS]
                            split_info["labels"][label] = len(files)
                        else:
                            split_info["labels"][label] = 0
                            results["issues"].append(
                                f"⚠️ Etiket dizini yok: {ds_name}/{split}/{label}"
                            )

                ds_info["splits"][split] = split_info

        results["structure"][ds_name] = ds_info

    return results


def check_image_integrity(
    dataset_dir: Path,
    fix: bool = False,
) -> Dict:
    """Görüntü dosyalarının bütünlüğünü kontrol et."""
    results = {
        "total_checked": 0,
        "valid": 0,
        "corrupted": [],
        "too_small": [],
        "size_stats": {},
    }

    if not HAS_PIL:
        print("  ⚠️ PIL yüklü değil, bütünlük kontrolü atlanıyor")
        return results

    all_images = []
    for ext in IMAGE_EXTS:
        all_images.extend(dataset_dir.rglob(f"*{ext}"))

    # _raw_frames ve _cropped_faces gibi ara dizinleri atla
    all_images = [f for f in all_images if not any(
        p.startswith("_") for p in f.relative_to(dataset_dir).parts
    )]

    if not all_images:
        return results

    iterator = all_images
    if HAS_TQDM:
        iterator = tqdm(all_images, desc="  Bütünlük kontrolü")

    for img_path in iterator:
        results["total_checked"] += 1

        try:
            with Image.open(img_path) as img:
                img.verify()

            # Boyut kontrolü
            with Image.open(img_path) as img:
                w, h = img.size
                if w < 32 or h < 32:
                    results["too_small"].append(str(img_path))
                    if fix:
                        img_path.unlink()
                else:
                    results["valid"] += 1
                    
                    # Boyut istatistikleri
                    size_key = f"{w}x{h}"
                    results["size_stats"][size_key] = results["size_stats"].get(size_key, 0) + 1

        except Exception:
            results["corrupted"].append(str(img_path))
            if fix:
                img_path.unlink()

    return results


def get_class_distribution(dataset_dir: Path) -> Dict:
    """Sınıf dağılımını hesapla."""
    dist = {}

    for ds_name, config in EXPECTED_STRUCTURE.items():
        ds_path = dataset_dir / ds_name
        if not ds_path.exists():
            continue

        ds_dist = {}
        for split in config["splits"]:
            split_dist = {}
            for label in config["labels"]:
                label_path = ds_path / split / label
                if label_path.exists():
                    count = len([f for f in label_path.iterdir()
                                 if f.suffix.lower() in IMAGE_EXTS])
                    split_dist[label] = count
                else:
                    split_dist[label] = 0
            ds_dist[split] = split_dist

        dist[ds_name] = ds_dist

    return dist


def check_pipeline_compatibility(dataset_dir: Path) -> Dict:
    """data_pipeline.py ile uyumluluğu kontrol et."""
    results = {"compatible": True, "issues": []}

    # config.py'deki yolları kontrol et
    try:
        sys.path.insert(0, str(dataset_dir.parent))
        from config import paths, model_cfg

        # FF++ yolu
        if not paths.FFPP_DIR.exists():
            results["issues"].append(f"FF++ dizini yok: {paths.FFPP_DIR}")
        else:
            for split in ["train", "val", "test"]:
                split_dir = paths.FFPP_DIR / split
                if split_dir.exists():
                    labels = [d.name for d in split_dir.iterdir() if d.is_dir()]
                    if "real" not in labels:
                        results["issues"].append(f"FF++/{split}/real dizini yok")
                    if "fake" not in labels:
                        results["issues"].append(f"FF++/{split}/fake dizini yok")

        # CASIA yolu
        if not paths.CASIA_DIR.exists():
            results["issues"].append(f"CASIA dizini yok: {paths.CASIA_DIR}")
        else:
            for split in ["train", "val", "test"]:
                split_dir = paths.CASIA_DIR / split
                if split_dir.exists():
                    labels = [d.name for d in split_dir.iterdir() if d.is_dir()]
                    if "live" not in labels:
                        results["issues"].append(f"CASIA/{split}/live dizini yok")
                    if "spoof" not in labels:
                        results["issues"].append(f"CASIA/{split}/spoof dizini yok")

        # Sınıf sayısı kontrolü
        if model_cfg.NUM_CLASSES != 3:
            results["issues"].append(f"NUM_CLASSES={model_cfg.NUM_CLASSES}, beklenen: 3")

    except ImportError:
        results["issues"].append("config.py import edilemedi")
    except Exception as e:
        results["issues"].append(f"Uyumluluk kontrolü hatası: {e}")

    results["compatible"] = len(results["issues"]) == 0
    return results


def print_report(
    structure: Dict,
    integrity: Dict,
    distribution: Dict,
    compatibility: Dict,
):
    """Doğrulama raporunu yazdır."""
    print(f"\n{'='*60}")
    print(f" Veri Seti Doğrulama Raporu")
    print(f"{'='*60}")

    # Dizin yapısı
    print(f"\n Dizin Yapısı")
    print(f"   Durum: {structure['status']}")
    for issue in structure["issues"]:
        print(f"   {issue}")

    # Sınıf dağılımı
    print(f"\n Sınıf Dağılımı")
    for ds_name, splits in distribution.items():
        print(f"    {ds_name}:")
        for split, labels in splits.items():
            total = sum(labels.values())
            label_str = " | ".join(f"{k}={v}" for k, v in labels.items())
            print(f"      {split:6s}: {label_str} (toplam: {total})")

    # Bütünlük
    print(f"\n Dosya Bütünlüğü")
    print(f"   Kontrol edilen: {integrity['total_checked']}")
    print(f"   Geçerli: {integrity['valid']}")
    print(f"   Bozuk: {len(integrity['corrupted'])}")
    print(f"   Çok küçük: {len(integrity['too_small'])}")
    if integrity["size_stats"]:
        top_sizes = sorted(integrity["size_stats"].items(), key=lambda x: -x[1])[:3]
        print(f"   En yaygın boyutlar: {', '.join(f'{s}({c})' for s, c in top_sizes)}")

    # Uyumluluk
    print(f"\n Pipeline Uyumluluğu")
    print(f"   Uyumlu: {'[OK]' if compatibility['compatible'] else '[X]'}")
    for issue in compatibility["issues"]:
        print(f"   [!] {issue}")

    # Genel sonuç
    total_issues = (
        len(structure["issues"]) +
        len(integrity["corrupted"]) +
        len(compatibility["issues"])
    )
    print(f"\n{'='*60}")
    if total_issues == 0:
        print(f"[OK] TÜM KONTROLLER BAŞARILI — Veri seti eğitime hazır!")
    else:
        print(f"[!] {total_issues} sorun tespit edildi")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Veri Seti Doğrulama Aracı",
    )

    parser.add_argument(
        "--path", type=str, default="dataset",
        help="Veri seti kök dizini (varsayılan: dataset)",
    )
    parser.add_argument(
        "--fix", action="store_true",
        help="Bozuk ve çok küçük dosyaları otomatik sil",
    )
    parser.add_argument(
        "--json", type=str, default=None,
        help="Raporu JSON olarak kaydet",
    )

    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    dataset_dir = project_root / args.path

    print(f"\n{'='*60}")
    print(f" Veri Seti Doğrulama")
    print(f"{'='*60}")
    print(f"   Yol: {dataset_dir}")
    if args.fix:
        print(f"   Otomatik düzeltme: AÇIK")
    print()

    # Kontroller
    structure = check_directory_structure(dataset_dir)
    integrity = check_image_integrity(dataset_dir, fix=args.fix)
    distribution = get_class_distribution(dataset_dir)
    compatibility = check_pipeline_compatibility(dataset_dir)

    # Rapor
    print_report(structure, integrity, distribution, compatibility)

    # JSON çıktı
    if args.json:
        report = {
            "structure": structure,
            "integrity": {
                "total_checked": integrity["total_checked"],
                "valid": integrity["valid"],
                "corrupted_count": len(integrity["corrupted"]),
                "too_small_count": len(integrity["too_small"]),
            },
            "distribution": distribution,
            "compatibility": compatibility,
        }
        json_path = project_root / args.json
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\n   JSON rapor: {json_path}")


if __name__ == "__main__":
    main()
