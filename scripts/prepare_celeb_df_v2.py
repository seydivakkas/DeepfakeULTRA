"""
Celeb-DF v2 Hazrlk Scripti
=============================
Celeb-DF v2 veri setinden yz grselleri karr ve test klasrne yerletirir.

Kullanm:
  python scripts/prepare_celeb_df_v2.py --source C:/path/to/Celeb-DF-v2

Beklenen Celeb-DF v2 klasr yaps:
  Celeb-DF-v2/
   Celeb-real/           Gerek nl videolar
   Celeb-synthesis/      Deepfake videolar
   YouTube-real/         Gerek YouTube videolar
   List_of_testing_videos.txt   Resmi test split

kt:
  dataset/external_tests/celeb_df_v2/
   real/    Gerek yz grselleri
   fake/    Deepfake yz grselleri
"""
import argparse
import os
import sys
import random
import cv2

# Proje kk dizinini PATH'e ekle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def extract_faces_from_video(video_path, output_dir, max_frames=5, prefix=""):
    """Videodan esit aralikla kare cikar ve center-crop yap.
    
    Celeb-DF videolari zaten yuz odakli oldugu icin center-crop yeterlidir.
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return 0

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < 1:
            cap.release()
            return 0

        step = max(1, total_frames // max_frames)
        frame_indices = list(range(0, total_frames, step))[:max_frames]

        saved = 0
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue

            h, w = frame.shape[:2]
            # Center crop (kare)
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
    except Exception as e:
        print(f"  Video hatasi: {video_path} -- {e}")
        return 0


def parse_test_list(source_dir):
    """List_of_testing_videos.txt dosyasn parse et."""
    list_path = os.path.join(source_dir, "List_of_testing_videos.txt")
    test_videos = {"real": [], "fake": []}

    if os.path.exists(list_path):
        with open(list_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    label = int(parts[0])  # 0=real, 1=fake
                    video_rel = parts[1]
                    video_path = os.path.join(source_dir, video_rel)
                    if label == 1:
                        test_videos["fake"].append(video_path)
                    else:
                        test_videos["real"].append(video_path)
        print(f" Test listesi: {len(test_videos['real'])} real, {len(test_videos['fake'])} fake")
    else:
        print(" List_of_testing_videos.txt bulunamad  tm videolar kullanlacak.")
        # Tm videolar kullan
        for folder, label in [
            ("Celeb-real", "real"), ("YouTube-real", "real"),
            ("Celeb-synthesis", "fake"),
        ]:
            folder_path = os.path.join(source_dir, folder)
            if os.path.exists(folder_path):
                for f in os.listdir(folder_path):
                    if f.endswith(".mp4"):
                        test_videos[label].append(os.path.join(folder_path, f))
        print(f" Bulunan: {len(test_videos['real'])} real, {len(test_videos['fake'])} fake")

    return test_videos


def main():
    parser = argparse.ArgumentParser(description="Celeb-DF v2 hazrlk scripti")
    parser.add_argument("--source", required=True, help="Celeb-DF v2 kaynak klasr")
    parser.add_argument("--max-videos", type=int, default=200,
                        help="Sinif basina max video (varsayilan: 200)")
    parser.add_argument("--frames-per-video", type=int, default=5,
                        help="Video basina kare sayisi (varsayilan: 5)")
    args = parser.parse_args()

    if not os.path.exists(args.source):
        print(f" Kaynak klasr bulunamad: {args.source}")
        sys.exit(1)

    # kt dizini
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_base = os.path.join(base_dir, "dataset", "external_tests", "celeb_df_v2")
    real_dir = os.path.join(output_base, "real")
    fake_dir = os.path.join(output_base, "fake")
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(fake_dir, exist_ok=True)

    # Test listesini parse et
    test_videos = parse_test_list(args.source)

    # Video se ve yz kar
    random.seed(42)
    for label, out_dir in [("real", real_dir), ("fake", fake_dir)]:
        videos = test_videos[label]
        selected = random.sample(videos, min(args.max_videos, len(videos)))
        print(f"\n{'' if label == 'real' else ''} {label.upper()}: {len(selected)} video ileniyor...")

        total_faces = 0
        for i, vpath in enumerate(selected):
            vname = os.path.splitext(os.path.basename(vpath))[0]
            prefix = f"celebdf_{label}_{vname}"
            n = extract_faces_from_video(vpath, out_dir, args.frames_per_video, prefix)
            total_faces += n
            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{len(selected)}] {total_faces} yz karld")

        print(f"   {label.upper()}: {total_faces} yz grseli kaydedildi  {out_dir}")

    # zet
    real_count = len([f for f in os.listdir(real_dir) if f.endswith('.jpg')])
    fake_count = len([f for f in os.listdir(fake_dir) if f.endswith('.jpg')])
    print(f"\n{'='*50}")
    print(f" Celeb-DF v2 Test Seti Hazr!")
    print(f"    REAL: {real_count} grsel")
    print(f"    FAKE: {fake_count} grsel")
    print(f"    Konum: {output_base}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
