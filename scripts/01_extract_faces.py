"""
Birleşik Yüz Çıkarım Aracı — §3 GÖREV 3.1

Rehber kuralları:
  - MediaPipe FaceDetection (fallback: Haar Cascade)
  - FF++ her videodan max 30 kare, perceptual hash farkı < 10 olanları atla
  - Anti-Spoof statik resimler doğrudan işlenir
  - Yüz padding: %20 (bounding box etrafına)
  - Çıktı: 224×224 PNG

Kullanım:
  python scripts/01_extract_faces.py --dataset ffpp
  python scripts/01_extract_faces.py --dataset antispoof
  python scripts/01_extract_faces.py --dataset all
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import cv2
except ImportError:
    print("❌ OpenCV yüklü değil: pip install opencv-python")
    sys.exit(1)

try:
    import numpy as np
except ImportError:
    print("❌ NumPy yüklü değil: pip install numpy")
    sys.exit(1)

try:
    import mediapipe as mp_lib
except ImportError:
    mp_lib = None

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# ═══════════════════════════════════════════════════════════
# SABİTLER
# ═══════════════════════════════════════════════════════════
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MAX_FRAMES_PER_VIDEO = 30
PHASH_THRESHOLD = 10          # Bu değerin altındaki fark → skip
FACE_PADDING = 0.20           # %20 bounding box padding
OUTPUT_SIZE = 224              # 224×224 PNG
MIN_FACE_SIZE = 40             # Minimum yüz boyutu (piksel)
MIN_CONFIDENCE = 0.5


# ═══════════════════════════════════════════════════════════
# PERCEPTUAL HASH
# ═══════════════════════════════════════════════════════════
def compute_phash(image: np.ndarray, hash_size: int = 8) -> int:
    """DCT tabanlı perceptual hash hesapla."""
    resized = cv2.resize(image, (hash_size * 4, hash_size * 4), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) if len(resized.shape) == 3 else resized
    gray = np.float32(gray)
    dct = cv2.dct(gray)
    dct_low = dct[:hash_size, :hash_size]
    median = np.median(dct_low)
    bits = (dct_low > median).flatten()
    hash_val = 0
    for bit in bits:
        hash_val = (hash_val << 1) | int(bit)
    return hash_val


def hamming_distance(hash1: int, hash2: int) -> int:
    """İki hash arasındaki Hamming mesafesi."""
    return bin(hash1 ^ hash2).count("1")


def is_similar_to_any(new_hash: int, existing_hashes: List[int], threshold: int) -> bool:
    """Yeni hash, mevcut hash'lerden herhangi birine çok benzer mi?"""
    for h in existing_hashes:
        if hamming_distance(new_hash, h) < threshold:
            return True
    return False


# ═══════════════════════════════════════════════════════════
# YÜZ ALGILAMA
# ═══════════════════════════════════════════════════════════
class FaceDetector:
    """MediaPipe + Haar Cascade fallback yüz algılayıcı."""

    def __init__(self, min_confidence: float = MIN_CONFIDENCE):
        self.mp_detector = None
        self.haar_cascade = None

        # MediaPipe
        if mp_lib is not None:
            try:
                self.mp_detector = mp_lib.solutions.face_detection.FaceDetection(
                    model_selection=1,
                    min_detection_confidence=min_confidence,
                )
            except Exception:
                self.mp_detector = None

        # Haar Cascade fallback
        try:
            import shutil
            original = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            local = Path(__file__).parent / ".cache" / "haarcascade_frontalface_default.xml"
            local.parent.mkdir(parents=True, exist_ok=True)
            if not local.exists() and os.path.exists(original):
                shutil.copy2(original, str(local))
            if local.exists():
                cascade = cv2.CascadeClassifier(str(local))
                if not cascade.empty():
                    self.haar_cascade = cascade
        except Exception:
            pass

        detector_name = "MediaPipe" if self.mp_detector else ("Haar" if self.haar_cascade else "YOK")
        print(f"  🔍 Algılayıcı: {detector_name}")

    def detect(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Yüz algıla → (x, y, w, h) veya None."""
        result = self._detect_mediapipe(image)
        if result is not None:
            return result
        return self._detect_haar(image)

    def _detect_mediapipe(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        if self.mp_detector is None:
            return None
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.mp_detector.process(rgb)
        if not results.detections:
            return None
        det = results.detections[0]
        bbox = det.location_data.relative_bounding_box
        h, w = image.shape[:2]
        return (int(bbox.xmin * w), int(bbox.ymin * h), int(bbox.width * w), int(bbox.height * h))

    def _detect_haar(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        if self.haar_cascade is None:
            return None
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.haar_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
        if len(faces) == 0:
            return None
        areas = [w * h for (_, _, w, h) in faces]
        best = max(range(len(areas)), key=lambda i: areas[i])
        return tuple(faces[best])

    def close(self):
        if self.mp_detector:
            self.mp_detector.close()


# ═══════════════════════════════════════════════════════════
# YÜZ KIRPMA
# ═══════════════════════════════════════════════════════════
def crop_face(image: np.ndarray, bbox: Tuple[int, int, int, int],
              padding: float = FACE_PADDING, output_size: int = OUTPUT_SIZE) -> Optional[np.ndarray]:
    """Yüzü padding ile kırp ve 224×224'e yeniden boyutlandır."""
    x, y, w, h = bbox
    if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
        return None

    img_h, img_w = image.shape[:2]
    margin_w = int(w * padding)
    margin_h = int(h * padding)
    x1 = max(0, x - margin_w)
    y1 = max(0, y - margin_h)
    x2 = min(img_w, x + w + margin_w)
    y2 = min(img_h, y + h + margin_h)

    face = image[y1:y2, x1:x2]
    if face.size == 0:
        return None

    return cv2.resize(face, (output_size, output_size), interpolation=cv2.INTER_AREA)


# ═══════════════════════════════════════════════════════════
# VİDEO İŞLEME
# ═══════════════════════════════════════════════════════════
def process_video(
    video_path: Path,
    output_dir: Path,
    detector: FaceDetector,
    max_frames: int = MAX_FRAMES_PER_VIDEO,
    phash_threshold: int = PHASH_THRESHOLD,
) -> Dict:
    """Tek videodan yüz çıkar — rehber kurallarına uygun."""
    stats = {"video": str(video_path.name), "total_sampled": 0,
             "accepted": 0, "no_face": 0, "too_similar": 0, "too_small": 0}

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        stats["error"] = "video açılamadı"
        return stats

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        stats["error"] = "boş video"
        return stats

    # Eşit aralıklı kare indeksleri
    actual_count = min(max_frames, total_frames)
    indices = [int(i * total_frames / actual_count) for i in range(actual_count)]

    output_dir.mkdir(parents=True, exist_ok=True)
    accepted_hashes: List[int] = []
    video_stem = video_path.stem

    for frame_idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        stats["total_sampled"] += 1

        # Yüz algıla
        bbox = detector.detect(frame)
        if bbox is None:
            stats["no_face"] += 1
            continue

        # Yüz kırp
        face = crop_face(frame, bbox)
        if face is None:
            stats["too_small"] += 1
            continue

        # pHash benzerlik filtresi
        face_hash = compute_phash(face)
        if is_similar_to_any(face_hash, accepted_hashes, phash_threshold):
            stats["too_similar"] += 1
            continue

        # Kabul — kaydet
        accepted_hashes.append(face_hash)
        out_path = output_dir / f"{video_stem}_f{frame_idx:06d}.png"
        cv2.imwrite(str(out_path), face)
        stats["accepted"] += 1

    cap.release()
    return stats


def process_image(
    image_path: Path,
    output_dir: Path,
    detector: FaceDetector,
) -> Dict:
    """Tek statik görüntüden yüz çıkar (anti-spoof statik resimler)."""
    stats = {"file": str(image_path.name), "accepted": 0, "no_face": 0, "too_small": 0}

    image = cv2.imread(str(image_path))
    if image is None:
        stats["error"] = "okunamadı"
        return stats

    bbox = detector.detect(image)
    if bbox is None:
        stats["no_face"] = 1
        return stats

    face = crop_face(image, bbox)
    if face is None:
        stats["too_small"] = 1
        return stats

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{image_path.stem}.png"
    cv2.imwrite(str(out_path), face)
    stats["accepted"] = 1
    return stats


# ═══════════════════════════════════════════════════════════
# FF++ PIPELINE
# ═══════════════════════════════════════════════════════════
def run_ffpp(project_root: Path, detector: FaceDetector,
             max_frames: int = MAX_FRAMES_PER_VIDEO, phash_threshold: int = PHASH_THRESHOLD):
    """FF++ videolarından yüz çıkar."""
    raw_dir = project_root / "dataset" / "raw" / "ffpp"
    out_base = project_root / "dataset" / "faces" / "ffpp"

    # REAL: original_sequences
    real_video_dir = raw_dir / "original_sequences" / "youtube" / "c23" / "videos"
    real_out = out_base / "real"

    # FAKE: tüm manipulated yöntemler
    manip_dir = raw_dir / "manipulated_sequences"

    all_stats = []

    # --- REAL ---
    print(f"\n{'='*60}")
    print("📸 FF++ REAL (original_sequences)")
    print(f"{'='*60}")

    real_videos = sorted([f for f in real_video_dir.iterdir() if f.suffix.lower() in VIDEO_EXTS]) if real_video_dir.exists() else []
    print(f"  📹 {len(real_videos)} video bulundu")

    iterator = tqdm(real_videos, desc="  REAL") if tqdm else real_videos
    for video in iterator:
        video_out = real_out / video.stem
        s = process_video(video, video_out, detector, max_frames, phash_threshold)
        s["class"] = "REAL"
        all_stats.append(s)

    # --- FAKE ---
    print(f"\n{'='*60}")
    print("📸 FF++ FAKE (manipulated_sequences)")
    print(f"{'='*60}")

    if manip_dir.exists():
        methods = sorted([d for d in manip_dir.iterdir() if d.is_dir()])
        for method_dir in methods:
            video_dir = method_dir / "c23" / "videos"
            if not video_dir.exists():
                print(f"  ⚠️ {method_dir.name}/c23/videos bulunamadı, atlanıyor")
                continue

            fake_videos = sorted([f for f in video_dir.iterdir() if f.suffix.lower() in VIDEO_EXTS])
            print(f"  📹 {method_dir.name}: {len(fake_videos)} video")

            fake_out = out_base / "fake"
            desc = f"  FAKE/{method_dir.name}"
            iterator = tqdm(fake_videos, desc=desc) if tqdm else fake_videos
            for video in iterator:
                # Yöntem adını prefix olarak ekle (çakışma önleme)
                video_out = fake_out / f"{method_dir.name}_{video.stem}"
                s = process_video(video, video_out, detector, max_frames, phash_threshold)
                s["class"] = "FAKE"
                s["method"] = method_dir.name
                all_stats.append(s)

    return all_stats


# ═══════════════════════════════════════════════════════════
# ANTI-SPOOF PIPELINE
# ═══════════════════════════════════════════════════════════
def run_antispoof(project_root: Path, detector: FaceDetector,
                  max_frames: int = MAX_FRAMES_PER_VIDEO, phash_threshold: int = PHASH_THRESHOLD):
    """Anti-Spoof videolarından ve resimlerinden yüz çıkar."""
    raw_dir = project_root / "dataset" / "raw" / "antispoof" / "_raw"
    out_base = project_root / "dataset" / "faces" / "antispoof"

    # Sınıf eşlemesi: dizin adı → hedef sınıf
    class_mapping = {
        "live_selfie": "live",
        "live_video": "live",
        "cut-out printouts": "spoof",
        "printouts": "spoof",
        "replay": "spoof",
    }

    all_stats = []

    for source_name, target_class in class_mapping.items():
        source_dir = raw_dir / source_name
        if not source_dir.exists():
            print(f"  ⚠️ {source_name} bulunamadı, atlanıyor")
            continue

        out_dir = out_base / target_class

        # Dosyaları bul (video + resim)
        files = sorted(source_dir.iterdir())
        videos = [f for f in files if f.suffix.lower() in VIDEO_EXTS]
        images = [f for f in files if f.suffix.lower() in IMAGE_EXTS]

        print(f"\n  📂 {source_name} → {target_class.upper()}: {len(videos)} video, {len(images)} resim")

        # Videoları işle
        for video in videos:
            video_out = out_dir / f"{source_name}_{video.stem}"
            s = process_video(video, video_out, detector, max_frames, phash_threshold)
            s["class"] = target_class.upper()
            s["source"] = source_name
            all_stats.append(s)

        # Resimleri işle
        for img in images:
            s = process_image(img, out_dir / f"{source_name}_static", detector)
            s["class"] = target_class.upper()
            s["source"] = source_name
            all_stats.append(s)

    return all_stats


# ═══════════════════════════════════════════════════════════
# RAPOR
# ═══════════════════════════════════════════════════════════
def print_report(stats: List[Dict], dataset_name: str, project_root: Path):
    """İstatistik raporu yazdır ve JSON olarak kaydet."""
    total_accepted = sum(s.get("accepted", 0) for s in stats)
    total_sampled = sum(s.get("total_sampled", 0) for s in stats)
    total_no_face = sum(s.get("no_face", 0) for s in stats)
    total_similar = sum(s.get("too_similar", 0) for s in stats)
    total_small = sum(s.get("too_small", 0) for s in stats)

    print(f"\n{'='*60}")
    print(f"📊 {dataset_name} RAPOR")
    print(f"{'='*60}")
    print(f"  📹 İşlenen kaynak  : {len(stats)}")
    print(f"  📸 Örneklenen kare : {total_sampled}")
    print(f"  ✅ Kabul edilen    : {total_accepted}")
    print(f"  ❌ Yüz bulunamadı  : {total_no_face}")
    print(f"  🔄 Benzer (skip)   : {total_similar}")
    print(f"  📏 Küçük yüz       : {total_small}")

    if total_sampled > 0:
        accept_rate = total_accepted / total_sampled * 100
        print(f"  📈 Kabul oranı     : {accept_rate:.1f}%")

    # Sınıf bazlı dağılım
    class_counts = {}
    for s in stats:
        cls = s.get("class", "?")
        class_counts[cls] = class_counts.get(cls, 0) + s.get("accepted", 0)

    print(f"\n  Sınıf Dağılımı:")
    for cls, count in sorted(class_counts.items()):
        print(f"    {cls}: {count}")

    # JSON rapor kaydet
    report_dir = project_root / "dataset" / "faces"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"{dataset_name}_extraction_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "dataset": dataset_name,
            "summary": {
                "total_sources": len(stats),
                "total_sampled": total_sampled,
                "total_accepted": total_accepted,
                "no_face": total_no_face,
                "too_similar": total_similar,
                "too_small": total_small,
                "class_distribution": class_counts,
            },
            "details": stats[:100],  # İlk 100 detay
        }, f, indent=2, ensure_ascii=False)
    print(f"\n  📝 Rapor: {report_path}")


# ═══════════════════════════════════════════════════════════
# ANA GİRİŞ NOKTASI
# ═══════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Birleşik Yüz Çıkarım Aracı — §3 GÖREV 3.1",
    )
    parser.add_argument(
        "--dataset", type=str, required=True,
        choices=["ffpp", "antispoof", "all"],
        help="İşlenecek veri seti",
    )
    parser.add_argument(
        "--max-frames", type=int, default=MAX_FRAMES_PER_VIDEO,
        help=f"Video başına max kare (varsayılan: {MAX_FRAMES_PER_VIDEO})",
    )
    parser.add_argument(
        "--phash-threshold", type=int, default=PHASH_THRESHOLD,
        help=f"pHash benzerlik eşiği (varsayılan: {PHASH_THRESHOLD})",
    )

    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    print(f"\n{'='*60}")
    print("🧠 Birleşik Yüz Çıkarım — §3 GÖREV 3.1")
    print(f"{'='*60}")
    print(f"  📂 Proje: {project_root}")
    print(f"  🎯 Dataset: {args.dataset}")
    print(f"  🎞️ Max kare/video: {args.max_frames}")
    print(f"  🔗 pHash eşiği: {args.phash_threshold}")

    max_frames = args.max_frames
    phash_thresh = args.phash_threshold

    detector = FaceDetector()
    start_time = time.time()

    if args.dataset in ("ffpp", "all"):
        stats = run_ffpp(project_root, detector, max_frames, phash_thresh)
        print_report(stats, "ffpp", project_root)

    if args.dataset in ("antispoof", "all"):
        stats = run_antispoof(project_root, detector, max_frames, phash_thresh)
        print_report(stats, "antispoof", project_root)

    elapsed = time.time() - start_time
    print(f"\n⏱️ Toplam süre: {elapsed/60:.1f} dakika")

    detector.close()


if __name__ == "__main__":
    main()
