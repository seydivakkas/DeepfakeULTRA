"""
Train / Val / Test Stratifiye Bölme Aracı
Kırpılmış yüz verilerini 70% / 15% / 15% oranında böler.

FF++ yapısı için:
    input: _cropped_faces/ffpp/original_sequences/... + manipulated_sequences/...
    output: dataset/deepfake/ff++/{train,val,test}/{real,fake}/

Anti-Spoofing yapısı için:
    input: _cropped_faces/antispoof/{live,spoof}/
    output: dataset/liveness/casia-fasd/{train,val,test}/{live,spoof}/

Kullanım:
    python scripts/split_dataset.py --input dataset/_cropped_faces --output dataset --mode ffpp
    python scripts/split_dataset.py --input dataset/_cropped_faces/antispoof --output dataset --mode antispoof
"""

import argparse
import os
import sys
import json
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_images(directory: Path) -> List[Path]:
    """Dizindeki tüm görüntü dosyalarını bul."""
    images = []
    for ext in IMAGE_EXTS:
        images.extend(directory.rglob(f"*{ext}"))
    return sorted(images)


def stratified_split(
    files: List[Path],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[Path], List[Path], List[Path]]:
    """Dosya listesini stratifiye olarak böl."""
    random.seed(seed)
    shuffled = list(files)
    random.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train = shuffled[:n_train]
    val = shuffled[n_train:n_train + n_val]
    test = shuffled[n_train + n_val:]

    return train, val, test


def copy_files(
    files: List[Path],
    target_dir: Path,
    desc: str = "",
):
    """Dosyaları hedef dizine kopyala."""
    target_dir.mkdir(parents=True, exist_ok=True)

    iterator = files
    if HAS_TQDM:
        iterator = tqdm(files, desc=f"  {desc}", leave=False)

    for f in iterator:
        target = target_dir / f.name
        # İsim çakışmasını önle
        if target.exists():
            stem = f.stem
            suffix = f.suffix
            counter = 1
            while target.exists():
                target = target_dir / f"{stem}_{counter}{suffix}"
                counter += 1
        shutil.copy2(str(f), str(target))


def classify_ffpp_files(input_dir: Path) -> Dict[str, List[Path]]:
    """
    FF++ dizin yapısındaki dosyaları real/fake olarak sınıflandır.

    FF++ indirme yapısı:
        original_sequences/youtube/c23/videos/*.mp4 → REAL
        manipulated_sequences/*/c23/videos/*.mp4 → FAKE
    
    Frame çıkarma sonrası:
        original_sequences/... → REAL
        manipulated_sequences/... → FAKE
    """
    classified = {"real": [], "fake": []}
    all_images = find_images(input_dir)

    for img in all_images:
        # Dosya yolundan sınıf belirle
        path_str = str(img).lower().replace("\\", "/")

        if "original" in path_str:
            classified["real"].append(img)
        elif any(m in path_str for m in [
            "deepfake", "face2face", "faceswap", "faceshifter",
            "neuraltexture", "manipulated",
        ]):
            classified["fake"].append(img)
        else:
            # Dizin adından tahmin et
            parts = [p.lower() for p in img.parts]
            if "real" in parts or "live" in parts or "original" in parts:
                classified["real"].append(img)
            elif "fake" in parts or "spoof" in parts:
                classified["fake"].append(img)

    return classified


def classify_antispoof_files(input_dir: Path) -> Dict[str, List[Path]]:
    """Anti-spoofing dosyalarını live/spoof olarak sınıflandır."""
    classified = {"live": [], "spoof": []}
    all_images = find_images(input_dir)

    for img in all_images:
        parts = [p.lower() for p in img.parts]
        path_str = str(img).lower()

        if any(k in parts or k in path_str for k in ["live", "real", "genuine", "client"]):
            classified["live"].append(img)
        elif any(k in parts or k in path_str for k in ["spoof", "fake", "attack", "imposter"]):
            classified["spoof"].append(img)

    return classified


def split_ffpp(
    input_dir: Path,
    output_base: Path,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
):
    """FF++ verisini train/val/test/real/fake yapısına böl."""
    print("  🔍 FF++ dosyaları sınıflandırılıyor...")
    classified = classify_ffpp_files(input_dir)

    print(f"     REAL: {len(classified['real'])} dosya")
    print(f"     FAKE: {len(classified['fake'])} dosya")

    if not classified["real"] and not classified["fake"]:
        print("  ❌ Sınıflandırılacak dosya bulunamadı!")
        return

    ffpp_dir = output_base / "deepfake" / "ff++"
    split_info = {}

    for label, files in classified.items():
        if not files:
            continue

        train, val, test = stratified_split(files, train_ratio, val_ratio, seed)

        copy_files(train, ffpp_dir / "train" / label, f"train/{label}")
        copy_files(val, ffpp_dir / "val" / label, f"val/{label}")
        copy_files(test, ffpp_dir / "test" / label, f"test/{label}")

        split_info[label] = {
            "total": len(files),
            "train": len(train),
            "val": len(val),
            "test": len(test),
        }

    # Bölme bilgisini kaydet
    info_path = ffpp_dir / "_split_info.json"
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(
            {"ratios": {"train": train_ratio, "val": val_ratio, "test": 1 - train_ratio - val_ratio},
             "seed": seed, "classes": split_info},
            f, indent=2, ensure_ascii=False,
        )

    print(f"\n  📊 FF++ Bölme Sonucu:")
    for label, info in split_info.items():
        print(f"     {label.upper()}: {info['train']}t / {info['val']}v / {info['test']}te = {info['total']}")
    print(f"  📝 Bölme bilgisi: {info_path}")


def split_antispoof(
    input_dir: Path,
    output_base: Path,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
):
    """Anti-spoofing verisini train/val/test/live/spoof yapısına böl."""
    print("  🔍 Anti-spoofing dosyaları sınıflandırılıyor...")
    classified = classify_antispoof_files(input_dir)

    print(f"     LIVE: {len(classified['live'])} dosya")
    print(f"     SPOOF: {len(classified['spoof'])} dosya")

    if not classified["live"] and not classified["spoof"]:
        print("  ❌ Sınıflandırılacak dosya bulunamadı!")
        return

    casia_dir = output_base / "liveness" / "casia-fasd"
    split_info = {}

    for label, files in classified.items():
        if not files:
            continue

        train, val, test = stratified_split(files, train_ratio, val_ratio, seed)

        copy_files(train, casia_dir / "train" / label, f"train/{label}")
        copy_files(val, casia_dir / "val" / label, f"val/{label}")
        copy_files(test, casia_dir / "test" / label, f"test/{label}")

        split_info[label] = {
            "total": len(files),
            "train": len(train),
            "val": len(val),
            "test": len(test),
        }

    # Bölme bilgisini kaydet
    info_path = casia_dir / "_split_info.json"
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(
            {"ratios": {"train": train_ratio, "val": val_ratio, "test": 1 - train_ratio - val_ratio},
             "seed": seed, "classes": split_info},
            f, indent=2, ensure_ascii=False,
        )

    print(f"\n  📊 Anti-Spoofing Bölme Sonucu:")
    for label, info in split_info.items():
        print(f"     {label.upper()}: {info['train']}t / {info['val']}v / {info['test']}te = {info['total']}")
    print(f"  📝 Bölme bilgisi: {info_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Train/Val/Test Stratifiye Bölme Aracı (70/15/15)",
    )

    parser.add_argument("--input", type=str, required=True, help="Kırpılmış yüz kaynak dizini")
    parser.add_argument("--output", type=str, default="dataset", help="Hedef dizin (varsayılan: dataset)")
    parser.add_argument(
        "--mode", type=str, required=True, choices=["ffpp", "antispoof", "both"],
        help="Bölme modu: ffpp, antispoof veya both",
    )
    parser.add_argument("--train-ratio", type=float, default=0.70, help="Eğitim oranı (varsayılan: 0.70)")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Doğrulama oranı (varsayılan: 0.15)")
    parser.add_argument("--seed", type=int, default=42, help="Rastgele tohum (varsayılan: 42)")

    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    input_dir = project_root / args.input
    output_base = project_root / args.output

    print(f"\n{'='*60}")
    print(f"📂 Train / Val / Test Bölme")
    print(f"{'='*60}")
    print(f"  📂 Girdi: {input_dir}")
    print(f"  📂 Çıktı: {output_base}")
    print(f"  📊 Oranlar: {args.train_ratio}/{args.val_ratio}/{1-args.train_ratio-args.val_ratio:.2f}")
    print(f"  🎲 Seed: {args.seed}")
    print()

    if args.mode in ("ffpp", "both"):
        ffpp_input = input_dir / "ffpp" if (input_dir / "ffpp").exists() else input_dir
        split_ffpp(ffpp_input, output_base, args.train_ratio, args.val_ratio, args.seed)

    if args.mode in ("antispoof", "both"):
        as_input = input_dir / "antispoof" if (input_dir / "antispoof").exists() else input_dir
        split_antispoof(as_input, output_base, args.train_ratio, args.val_ratio, args.seed)


if __name__ == "__main__":
    main()
