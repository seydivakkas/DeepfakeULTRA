"""
Yüz Algılama + Kırpma Aracı
Frame'lerden yüz bölgelerini algılayıp kırpar ve 224×224 boyutuna yeniden boyutlandırır.

Algılama yöntemleri (öncelik sırasıyla):
  1. MediaPipe Face Detection (tercih edilen — hızlı ve doğru)
  2. OpenCV Haar Cascade (fallback)

Kullanım:
    python scripts/crop_faces.py --input dataset/_raw_frames/ffpp --output dataset/_cropped_faces/ffpp
    python scripts/crop_faces.py --input dataset/_raw_frames/ffpp --output dataset/_cropped_faces/ffpp --margin 0.3 --size 224
"""

import argparse
import os
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Tuple, Optional, List
import json

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class FaceCropper:
    """Yüz algılama ve kırpma sınıfı."""

    def __init__(
        self,
        target_size: int = 224,
        margin: float = 0.3,
        min_confidence: float = 0.5,
    ):
        self.target_size = target_size
        self.margin = margin
        self.min_confidence = min_confidence

        # MediaPipe başlat
        self.mp_detector = None
        if HAS_MEDIAPIPE:
            try:
                self.mp_detector = mp.solutions.face_detection.FaceDetection(
                    model_selection=1,  # 0=yakın, 1=uzak
                    min_detection_confidence=min_confidence,
                )
            except Exception:
                self.mp_detector = None

        # Haar Cascade fallback
        # Windows'ta Türkçe karakterli kullanıcı adları OpenCV'de sorun yaratıyor.
        # Cascade dosyasını proje dizinine kopyalayarak çözüyoruz.
        self.haar_cascade = None
        if HAS_CV2:
            try:
                import shutil
                original_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                local_path = Path(__file__).parent / ".cache" / "haarcascade_frontalface_default.xml"
                local_path.parent.mkdir(parents=True, exist_ok=True)

                if not local_path.exists() and os.path.exists(original_path):
                    shutil.copy2(original_path, str(local_path))

                if local_path.exists():
                    cascade = cv2.CascadeClassifier(str(local_path))
                    if not cascade.empty():
                        self.haar_cascade = cascade
            except Exception:
                self.haar_cascade = None

    def detect_face_mediapipe(
        self, image: "np.ndarray"
    ) -> Optional[Tuple[int, int, int, int]]:
        """MediaPipe ile yüz algıla → (x, y, w, h) döndür."""
        if self.mp_detector is None:
            return None

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.mp_detector.process(rgb)

        if not results.detections:
            return None

        detection = results.detections[0]
        bbox = detection.location_data.relative_bounding_box
        h, w = image.shape[:2]

        x = int(bbox.xmin * w)
        y = int(bbox.ymin * h)
        bw = int(bbox.width * w)
        bh = int(bbox.height * h)

        return (x, y, bw, bh)

    def detect_face_haar(
        self, image: "np.ndarray"
    ) -> Optional[Tuple[int, int, int, int]]:
        """Haar Cascade ile yüz algıla → (x, y, w, h) döndür."""
        if self.haar_cascade is None:
            return None

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.haar_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )

        if len(faces) == 0:
            return None

        # En büyük yüzü seç
        areas = [w * h for (_, _, w, h) in faces]
        best_idx = max(range(len(areas)), key=lambda i: areas[i])
        return tuple(faces[best_idx])

    def detect_face(
        self, image: "np.ndarray"
    ) -> Optional[Tuple[int, int, int, int]]:
        """Yüz algıla (önce MediaPipe, fallback Haar)."""
        result = self.detect_face_mediapipe(image)
        if result is not None:
            return result
        return self.detect_face_haar(image)

    def crop_face(
        self, image: "np.ndarray", bbox: Tuple[int, int, int, int]
    ) -> "np.ndarray":
        """Yüzü margin ile kırp ve hedef boyuta yeniden boyutlandır."""
        x, y, w, h = bbox
        img_h, img_w = image.shape[:2]

        # Margin ekle
        margin_w = int(w * self.margin)
        margin_h = int(h * self.margin)

        x1 = max(0, x - margin_w)
        y1 = max(0, y - margin_h)
        x2 = min(img_w, x + w + margin_w)
        y2 = min(img_h, y + h + margin_h)

        face = image[y1:y2, x1:x2]

        # Boyutlandır
        face = cv2.resize(
            face, (self.target_size, self.target_size), interpolation=cv2.INTER_AREA
        )
        return face

    def process_image(self, input_path: Path, output_path: Path) -> bool:
        """Tek bir görüntüyü işle: yüz algıla → kırp → kaydet."""
        image = cv2.imread(str(input_path))
        if image is None:
            return False

        bbox = self.detect_face(image)
        if bbox is None:
            return False

        face = self.crop_face(image, bbox)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), face, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return True

    def close(self):
        if self.mp_detector:
            self.mp_detector.close()


def process_image_task(args_tuple) -> Tuple[str, bool]:
    """Multiprocessing görevi — her worker kendi FaceCropper'ını oluşturur."""
    input_path, output_path, target_size, margin, min_confidence = args_tuple

    try:
        cropper = FaceCropper(
            target_size=target_size, margin=margin, min_confidence=min_confidence
        )
        success = cropper.process_image(Path(input_path), Path(output_path))
        cropper.close()
        return str(input_path), success
    except Exception:
        return str(input_path), False


def find_images(input_dir: Path) -> List[Path]:
    """Dizindeki tüm görüntü dosyalarını bul."""
    images = []
    for ext in IMAGE_EXTS:
        images.extend(input_dir.rglob(f"*{ext}"))
    return sorted(images)


def crop_all_faces(
    input_dir: Path,
    output_dir: Path,
    target_size: int = 224,
    margin: float = 0.3,
    min_confidence: float = 0.5,
    max_workers: int = 4,
    preserve_structure: bool = True,
):
    """
    Tüm framelerdeki yüzleri algılayıp kırp.

    Args:
        input_dir: Frame kaynak dizini
        output_dir: Kırpılmış yüz hedef dizini
        target_size: Çıktı boyutu (kare)
        margin: Yüz çevresindeki ek boşluk oranı
        min_confidence: Minimum algılama güveni
        max_workers: Paralel işlem sayısı
        preserve_structure: Dizin yapısını koru
    """
    images = find_images(input_dir)
    if not images:
        print(f"  ⚠️ {input_dir} dizininde görüntü bulunamadı!")
        return

    print(f"  📸 {len(images)} görüntü bulundu")

    # İşlem argümanlarını hazırla
    tasks = []
    for img_path in images:
        if preserve_structure:
            relative = img_path.relative_to(input_dir)
            out_path = output_dir / relative
        else:
            out_path = output_dir / img_path.name

        tasks.append(
            (str(img_path), str(out_path), target_size, margin, min_confidence)
        )

    success_count = 0
    failed_count = 0
    failed_files = []

    # Tek process ile çalıştır (MediaPipe multiprocess uyumsuz)
    cropper = FaceCropper(
        target_size=target_size, margin=margin, min_confidence=min_confidence
    )

    iterator = tasks
    if HAS_TQDM:
        iterator = tqdm(tasks, desc="  Yüz kırpma")

    for task in iterator:
        input_path, output_path = task[0], task[1]
        try:
            success = cropper.process_image(Path(input_path), Path(output_path))
        except Exception:
            success = False

        if success:
            success_count += 1
        else:
            failed_count += 1
            failed_files.append(input_path)

    cropper.close()

    print(f"\n  ✅ Başarılı: {success_count}")
    print(f"  ❌ Başarısız (yüz bulunamadı): {failed_count}")

    # Başarısız dosyaları logla
    if failed_files:
        log_path = output_dir / "_failed_crops.json"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "total_failed": failed_count,
                    "failed_rate": f"{failed_count / (success_count + failed_count) * 100:.1f}%",
                    "files": failed_files[:100],  # İlk 100
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        print(f"  📝 Başarısız dosya logu: {log_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Yüz Algılama + Kırpma Aracı",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--input", type=str, required=True, help="Frame kaynak dizini")
    parser.add_argument("--output", type=str, required=True, help="Kırpılmış yüz hedef dizini")
    parser.add_argument("--size", type=int, default=224, help="Çıktı boyutu (varsayılan: 224)")
    parser.add_argument("--margin", type=float, default=0.3, help="Yüz margin oranı (varsayılan: 0.3)")
    parser.add_argument("--confidence", type=float, default=0.5, help="Min algılama güveni (varsayılan: 0.5)")
    parser.add_argument("--workers", type=int, default=4, help="Paralel işlem sayısı")

    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    input_dir = project_root / args.input
    output_dir = project_root / args.output

    print(f"\n{'='*60}")
    print(f"✂️ Yüz Algılama + Kırpma")
    print(f"{'='*60}")
    print(f"  📂 Girdi: {input_dir}")
    print(f"  📂 Çıktı: {output_dir}")
    print(f"  📐 Boyut: {args.size}×{args.size}")
    print(f"  📏 Margin: {args.margin}")
    detector = "MediaPipe" if HAS_MEDIAPIPE else ("Haar Cascade" if HAS_CV2 else "YOK")
    print(f"  🔍 Algılayıcı: {detector}")
    print()

    crop_all_faces(
        input_dir=input_dir,
        output_dir=output_dir,
        target_size=args.size,
        margin=args.margin,
        min_confidence=args.confidence,
        max_workers=args.workers,
    )


if __name__ == "__main__":
    main()
