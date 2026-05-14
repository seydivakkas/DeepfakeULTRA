"""
FaceForensics++ Hazirlik Scripti
=================================
FF++ videolarindan center-crop ile yuz gorselleri cikarir.
real/ ve fake/ klasorlerine yerlestirir.

Kullanim:
  python scripts/prepare_ff++.py
"""
import os
import sys
import cv2
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE, "dataset", "external_tests", "faceforensics_raw")
OUT_DIR = os.path.join(BASE, "dataset", "external_tests", "faceforensics")
REAL_DIR = os.path.join(OUT_DIR, "real")
FAKE_DIR = os.path.join(OUT_DIR, "fake")

FRAMES_PER_VIDEO = 5


def extract_frames(video_path, output_dir, prefix, max_frames=5):
    """Videodan center-crop frame cikar."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 1:
        cap.release()
        return 0

    step = max(1, total // max_frames)
    indices = list(range(0, total, step))[:max_frames]
    saved = 0

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue

        h, w = frame.shape[:2]
        crop_size = min(h, w)
        cy, cx = h // 2, w // 2
        cropped = frame[cy - crop_size // 2:cy + crop_size // 2,
                        cx - crop_size // 2:cx + crop_size // 2]

        resized = cv2.resize(cropped, (224, 224))
        fname = f"{prefix}_frame{idx:05d}.jpg"
        cv2.imwrite(os.path.join(output_dir, fname), resized)
        saved += 1

    cap.release()
    return saved


def process_folder(video_dir, output_dir, label_prefix):
    """Bir klasordeki tum videolari isle."""
    if not os.path.exists(video_dir):
        print(f"  Klasor bulunamadi: {video_dir}")
        return 0

    videos = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
    print(f"  {len(videos)} video bulundu: {video_dir}")

    total_faces = 0
    for i, vfile in enumerate(sorted(videos)):
        vpath = os.path.join(video_dir, vfile)
        vname = os.path.splitext(vfile)[0]
        prefix = f"ff_{label_prefix}_{vname}"
        n = extract_frames(vpath, output_dir, prefix, FRAMES_PER_VIDEO)
        total_faces += n
        if (i + 1) % 10 == 0:
            print(f"    [{i+1}/{len(videos)}] {total_faces} frame")

    return total_faces


def main():
    os.makedirs(REAL_DIR, exist_ok=True)
    os.makedirs(FAKE_DIR, exist_ok=True)

    print("FaceForensics++ Hazirlik")
    print("=" * 50)

    # REAL: original_sequences/youtube/c23/videos/
    real_dir = os.path.join(RAW_DIR, "original_sequences", "youtube", "c23", "videos")
    print(f"\n[REAL] Original videolar:")
    real_count = process_folder(real_dir, REAL_DIR, "real")
    print(f"  -> {real_count} gorsel kaydedildi")

    # FAKE: manipulated_sequences/*/c23/videos/
    fake_methods = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures", "FaceShifter"]
    total_fake = 0

    for method in fake_methods:
        fake_dir = os.path.join(RAW_DIR, "manipulated_sequences", method, "c23", "videos")
        if os.path.exists(fake_dir):
            print(f"\n[FAKE] {method}:")
            n = process_folder(fake_dir, FAKE_DIR, method.lower())
            total_fake += n
            print(f"  -> {n} gorsel kaydedildi")

    # Ozet
    real_final = len([f for f in os.listdir(REAL_DIR) if f.endswith('.jpg')])
    fake_final = len([f for f in os.listdir(FAKE_DIR) if f.endswith('.jpg')])
    print(f"\n{'=' * 50}")
    print(f"FF++ Test Seti Hazir!")
    print(f"  REAL: {real_final} gorsel")
    print(f"  FAKE: {fake_final} gorsel")
    print(f"  Konum: {OUT_DIR}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
