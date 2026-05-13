"""
Faz 1 — Ana Orkestratör: Veri Hazırlama Pipeline
Tüm aşamaları sırasıyla çalıştırır:
  1. FF++ videoları → Frame çıkarma → Yüz kırpma → Split
  2. Anti-Spoofing verisi → (opsiyonel frame çıkarma) → Yüz kırpma → Split
  3. Doğrulama raporu

Kullanım:
    # FF++ videolarını işle (download sonrası)
    python scripts/prepare_data.py --ffpp-path dataset/raw/ffpp

    # Anti-spoofing verisini işle
    python scripts/prepare_data.py --antispoof-path dataset/raw/antispoof

    # Her ikisini birden
    python scripts/prepare_data.py --ffpp-path dataset/raw/ffpp --antispoof-path dataset/raw/antispoof

    # Sadece doğrulama
    python scripts/prepare_data.py --validate-only
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Proje kökünü path'e ekle
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


def step_header(step_num: int, title: str):
    """Aşama başlığını yazdır."""
    print(f"\n{'='*60}")
    print(f"  Aşama {step_num}: {title}")
    print(f"{'='*60}")


def run_ffpp_pipeline(
    raw_path: Path,
    frames_per_video: int = 30,
    target_size: int = 224,
    margin: float = 0.3,
    workers: int = 4,
):
    """FF++ tam pipeline: frame çıkarma → yüz kırpma → split."""
    from extract_frames import extract_all_frames, find_videos
    from crop_faces import crop_all_faces
    from split_dataset import split_ffpp

    dataset_dir = PROJECT_ROOT / "dataset"
    frames_dir = dataset_dir / "_raw_frames" / "ffpp"
    cropped_dir = dataset_dir / "_cropped_faces" / "ffpp"

    # Aşama 1: Video var mı kontrol et
    videos = find_videos(raw_path)
    images_in_raw = list(raw_path.rglob("*.jpg")) + list(raw_path.rglob("*.png"))

    if videos:
        step_header(1, f"FF++ Frame Çıkarma ({len(videos)} video)")
        extract_all_frames(
            input_dir=raw_path,
            output_dir=frames_dir,
            frames_per_video=frames_per_video,
            max_workers=workers,
            preserve_structure=True,
        )
        face_input = frames_dir
    elif images_in_raw:
        print(f"\n  ℹ️ Video bulunamadı ama {len(images_in_raw)} görüntü var → Frame çıkarma atlanıyor")
        face_input = raw_path
    else:
        print(f"\n  ❌ {raw_path} dizininde video veya görüntü bulunamadı!")
        return

    # Aşama 2: Yüz kırpma
    step_header(2, "FF++ Yüz Kırpma")
    crop_all_faces(
        input_dir=face_input,
        output_dir=cropped_dir,
        target_size=target_size,
        margin=margin,
        max_workers=workers,
        preserve_structure=True,
    )

    # Aşama 3: Train/Val/Test bölme
    step_header(3, "FF++ Train/Val/Test Bölme (70/15/15)")
    split_ffpp(
        input_dir=cropped_dir,
        output_base=dataset_dir,
        train_ratio=0.70,
        val_ratio=0.15,
        seed=42,
    )


def run_antispoof_pipeline(
    raw_path: Path,
    frames_per_video: int = 50,
    target_size: int = 224,
    margin: float = 0.3,
    workers: int = 4,
):
    """Anti-spoofing pipeline: (frame çıkarma →) yüz kırpma → split."""
    from extract_frames import extract_all_frames, find_videos
    from crop_faces import crop_all_faces
    from split_dataset import split_antispoof

    dataset_dir = PROJECT_ROOT / "dataset"
    frames_dir = dataset_dir / "_raw_frames" / "antispoof"
    cropped_dir = dataset_dir / "_cropped_faces" / "antispoof"

    # Video mu yoksa hazır frame mi?
    videos = find_videos(raw_path)
    images_in_raw = list(raw_path.rglob("*.jpg")) + list(raw_path.rglob("*.png"))

    if videos:
        step_header(1, f"Anti-Spoofing Frame Çıkarma ({len(videos)} video)")
        extract_all_frames(
            input_dir=raw_path,
            output_dir=frames_dir,
            frames_per_video=frames_per_video,
            max_workers=workers,
            preserve_structure=True,
        )
        face_input = frames_dir
    elif images_in_raw:
        print(f"\n  ℹ️ Hazır görüntüler bulundu ({len(images_in_raw)} adet) → Frame çıkarma atlanıyor")
        face_input = raw_path
    else:
        print(f"\n  ❌ {raw_path} dizininde video veya görüntü bulunamadı!")
        return

    # Aşama 2: Yüz kırpma
    step_header(2, "Anti-Spoofing Yüz Kırpma")
    crop_all_faces(
        input_dir=face_input,
        output_dir=cropped_dir,
        target_size=target_size,
        margin=margin,
        max_workers=workers,
        preserve_structure=True,
    )

    # Aşama 3: Train/Val/Test bölme
    step_header(3, "Anti-Spoofing Train/Val/Test Bölme (70/15/15)")
    split_antispoof(
        input_dir=cropped_dir,
        output_base=dataset_dir,
        train_ratio=0.70,
        val_ratio=0.15,
        seed=42,
    )


def run_validation():
    """Doğrulama raporu oluştur."""
    from validate_dataset import (
        check_directory_structure,
        check_image_integrity,
        get_class_distribution,
        check_pipeline_compatibility,
        print_report,
    )

    dataset_dir = PROJECT_ROOT / "dataset"

    step_header(4, "Doğrulama Raporu")

    structure = check_directory_structure(dataset_dir)
    integrity = check_image_integrity(dataset_dir)
    distribution = get_class_distribution(dataset_dir)
    compatibility = check_pipeline_compatibility(dataset_dir)

    print_report(structure, integrity, distribution, compatibility)


def main():
    parser = argparse.ArgumentParser(
        description="Faz 1 — Veri Hazırlama Pipeline Orkestratörü",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Örnekler:
  # FF++ indirdikten sonra tam pipeline
  python scripts/prepare_data.py --ffpp-path dataset/raw/ffpp

  # Anti-spoofing verisini işle
  python scripts/prepare_data.py --antispoof-path dataset/raw/antispoof

  # Tümü
  python scripts/prepare_data.py --ffpp-path dataset/raw/ffpp --antispoof-path dataset/raw/antispoof

  # Sadece doğrulama
  python scripts/prepare_data.py --validate-only
        """,
    )

    parser.add_argument("--ffpp-path", type=str, default=None, help="FF++ ham veri dizini")
    parser.add_argument("--antispoof-path", type=str, default=None, help="Anti-spoofing ham veri dizini")
    parser.add_argument("--frames-per-video", type=int, default=30, help="FF++ video başına frame (varsayılan: 30)")
    parser.add_argument("--size", type=int, default=224, help="Çıktı boyutu (varsayılan: 224)")
    parser.add_argument("--margin", type=float, default=0.3, help="Yüz margin (varsayılan: 0.3)")
    parser.add_argument("--workers", type=int, default=4, help="Paralel işlem sayısı")
    parser.add_argument("--validate-only", action="store_true", help="Sadece doğrulama yap")
    parser.add_argument("--skip-validation", action="store_true", help="Doğrulamayı atla")

    args = parser.parse_args()

    start_time = time.time()

    print(f"\n{'='*60}")
    print(f"🚀 Faz 1 — Veri Hazırlama Pipeline")
    print(f"{'='*60}")

    if args.validate_only:
        run_validation()
        return

    if not args.ffpp_path and not args.antispoof_path:
        print("  ❌ En az bir veri kaynağı belirtmelisiniz:")
        print("     --ffpp-path veya --antispoof-path")
        parser.print_help()
        sys.exit(1)

    # FF++ pipeline
    if args.ffpp_path:
        ffpp_path = PROJECT_ROOT / args.ffpp_path
        print(f"\n  📁 FF++ Kaynak: {ffpp_path}")
        run_ffpp_pipeline(
            raw_path=ffpp_path,
            frames_per_video=args.frames_per_video,
            target_size=args.size,
            margin=args.margin,
            workers=args.workers,
        )

    # Anti-spoofing pipeline
    if args.antispoof_path:
        antispoof_path = PROJECT_ROOT / args.antispoof_path
        print(f"\n  📁 Anti-Spoofing Kaynak: {antispoof_path}")
        run_antispoof_pipeline(
            raw_path=antispoof_path,
            frames_per_video=50,
            target_size=args.size,
            margin=args.margin,
            workers=args.workers,
        )

    # Doğrulama
    if not args.skip_validation:
        run_validation()

    elapsed = time.time() - start_time
    print(f"\n⏱️ Toplam süre: {elapsed:.1f} saniye ({elapsed/60:.1f} dakika)")


if __name__ == "__main__":
    main()
