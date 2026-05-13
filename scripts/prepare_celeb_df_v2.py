"""
Celeb-DF v2 Hazırlık Scripti
=============================
Celeb-DF v2 veri setinden yüz görselleri çıkarır ve test klasörüne yerleştirir.

Kullanım:
  python scripts/prepare_celeb_df_v2.py --source C:/path/to/Celeb-DF-v2

Beklenen Celeb-DF v2 klasör yapısı:
  Celeb-DF-v2/
  ├── Celeb-real/          ← Gerçek ünlü videoları
  ├── Celeb-synthesis/     ← Deepfake videoları
  ├── YouTube-real/        ← Gerçek YouTube videoları
  └── List_of_testing_videos.txt  ← Resmi test split

Çıktı:
  dataset/external_tests/celeb_df_v2/
  ├── real/   ← Gerçek yüz görselleri
  └── fake/   ← Deepfake yüz görselleri
"""
import argparse
import os
import sys
import random
import cv2

# Proje kök dizinini PATH'e ekle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def extract_faces_from_video(video_path, output_dir, max_frames=5, prefix=""):
    """Videodan eşit aralıklarla kare çıkar ve yüz kırp."""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return 0

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < 1:
            cap.release()
            return 0

        # Eşit aralıklarla kare seç
        step = max(1, total_frames // max_frames)
        frame_indices = list(range(0, total_frames, step))[:max_frames]

        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        saved = 0
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            if len(faces) == 0:
                # Yüz bulunamadı — tüm kareyi 224x224 olarak kaydet
                resized = cv2.resize(frame, (224, 224))
                fname = f"{prefix}_frame{idx:05d}.jpg"
                cv2.imwrite(os.path.join(output_dir, fname), resized)
                saved += 1
            else:
                # En büyük yüzü kırp
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                # Yüz etrafına margin ekle
                margin = int(max(w, h) * 0.3)
                x1 = max(0, x - margin)
                y1 = max(0, y - margin)
                x2 = min(frame.shape[1], x + w + margin)
                y2 = min(frame.shape[0], y + h + margin)

                face_crop = frame[y1:y2, x1:x2]
                if face_crop.size == 0:
                    continue

                resized = cv2.resize(face_crop, (224, 224))
                fname = f"{prefix}_frame{idx:05d}.jpg"
                cv2.imwrite(os.path.join(output_dir, fname), resized)
                saved += 1

        cap.release()
        return saved
    except Exception as e:
        print(f"  ⚠️ Video hatası: {video_path} — {e}")
        return 0


def parse_test_list(source_dir):
    """List_of_testing_videos.txt dosyasını parse et."""
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
        print(f"📋 Test listesi: {len(test_videos['real'])} real, {len(test_videos['fake'])} fake")
    else:
        print("⚠️ List_of_testing_videos.txt bulunamadı — tüm videolar kullanılacak.")
        # Tüm videoları kullan
        for folder, label in [
            ("Celeb-real", "real"), ("YouTube-real", "real"),
            ("Celeb-synthesis", "fake"),
        ]:
            folder_path = os.path.join(source_dir, folder)
            if os.path.exists(folder_path):
                for f in os.listdir(folder_path):
                    if f.endswith(".mp4"):
                        test_videos[label].append(os.path.join(folder_path, f))
        print(f"📂 Bulunan: {len(test_videos['real'])} real, {len(test_videos['fake'])} fake")

    return test_videos


def main():
    parser = argparse.ArgumentParser(description="Celeb-DF v2 hazırlık scripti")
    parser.add_argument("--source", required=True, help="Celeb-DF v2 kaynak klasörü")
    parser.add_argument("--max-videos", type=int, default=30,
                        help="Sınıf başına max video (varsayılan: 30)")
    parser.add_argument("--frames-per-video", type=int, default=3,
                        help="Video başına kare sayısı (varsayılan: 3)")
    args = parser.parse_args()

    if not os.path.exists(args.source):
        print(f"❌ Kaynak klasör bulunamadı: {args.source}")
        sys.exit(1)

    # Çıktı dizini
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_base = os.path.join(base_dir, "dataset", "external_tests", "celeb_df_v2")
    real_dir = os.path.join(output_base, "real")
    fake_dir = os.path.join(output_base, "fake")
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(fake_dir, exist_ok=True)

    # Test listesini parse et
    test_videos = parse_test_list(args.source)

    # Video seç ve yüz çıkar
    random.seed(42)
    for label, out_dir in [("real", real_dir), ("fake", fake_dir)]:
        videos = test_videos[label]
        selected = random.sample(videos, min(args.max_videos, len(videos)))
        print(f"\n{'🟢' if label == 'real' else '🔴'} {label.upper()}: {len(selected)} video işleniyor...")

        total_faces = 0
        for i, vpath in enumerate(selected):
            vname = os.path.splitext(os.path.basename(vpath))[0]
            prefix = f"celebdf_{label}_{vname}"
            n = extract_faces_from_video(vpath, out_dir, args.frames_per_video, prefix)
            total_faces += n
            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{len(selected)}] {total_faces} yüz çıkarıldı")

        print(f"  ✅ {label.upper()}: {total_faces} yüz görseli kaydedildi → {out_dir}")

    # Özet
    real_count = len([f for f in os.listdir(real_dir) if f.endswith('.jpg')])
    fake_count = len([f for f in os.listdir(fake_dir) if f.endswith('.jpg')])
    print(f"\n{'='*50}")
    print(f"📊 Celeb-DF v2 Test Seti Hazır!")
    print(f"   🟢 REAL: {real_count} görsel")
    print(f"   🔴 FAKE: {fake_count} görsel")
    print(f"   📂 Konum: {output_base}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
