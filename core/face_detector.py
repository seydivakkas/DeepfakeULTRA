"""
Deepfake Detection System v3.0 — Yuz Tespit Modulu
MediaPipe Tasks API veya OpenCV ile yuz bounding box tespiti.
"""
import os
import numpy as np
from PIL import Image, ImageDraw

try:
    import mediapipe as mp
    # Yeni API kontrolu (v0.10.x: mp.tasks)
    HAS_MP_TASKS = hasattr(mp, 'tasks')
    # Eski API kontrolu (v0.9.x: mp.solutions)
    HAS_MP_SOLUTIONS = hasattr(mp, 'solutions')
    HAS_MP = HAS_MP_TASKS or HAS_MP_SOLUTIONS
except ImportError:
    HAS_MP = False
    HAS_MP_TASKS = False
    HAS_MP_SOLUTIONS = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


def detect_faces(image) -> list:
    """Yuz bounding box tespiti."""
    if isinstance(image, Image.Image):
        img_np = np.array(image.convert("RGB"))
    elif isinstance(image, str):
        img_np = np.array(Image.open(image).convert("RGB"))
    else:
        img_np = image.copy()

    # MediaPipe Tasks API (v0.10+)
    if HAS_MP_TASKS:
        try:
            result = _detect_mediapipe_tasks(img_np)
            if result:
                return result
        except Exception as e:
            _log(f"MediaPipe Tasks hatasi: {e}")

    # MediaPipe Solutions API (v0.9.x fallback)
    if HAS_MP_SOLUTIONS:
        try:
            result = _detect_mediapipe_solutions(img_np)
            if result:
                return result
        except Exception as e:
            _log(f"MediaPipe Solutions hatasi: {e}")

    # OpenCV Haar Cascade
    if HAS_CV2:
        try:
            result = _detect_opencv_haar(img_np)
            if result:
                return result
        except Exception as e:
            _log(f"OpenCV Haar hatasi: {e}")

    return []


def _detect_mediapipe_tasks(img_np: np.ndarray) -> list:
    """MediaPipe Tasks API (v0.10+) ile yuz tespiti."""
    BaseOptions = mp.tasks.BaseOptions
    FaceDetector = mp.tasks.vision.FaceDetector
    FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    # Blaze Face model dosyasi
    model_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    model_path = os.path.join(model_dir, "blaze_face_short_range.tflite")

    # Model dosyasi yoksa indirmeyi dene
    if not os.path.exists(model_path):
        try:
            _download_blaze_face_model(model_path)
        except Exception:
            pass

    if not os.path.exists(model_path):
        return []

    options = FaceDetectorOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.IMAGE,
        min_detection_confidence=0.3,
    )

    h, w = img_np.shape[:2]
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(img_np))
    faces = []

    with FaceDetector.create_from_options(options) as detector:
        result = detector.detect(mp_image)
        for detection in result.detections:
            bbox = detection.bounding_box
            fx = max(0, bbox.origin_x)
            fy = max(0, bbox.origin_y)
            fw = min(bbox.width, w - fx)
            fh = min(bbox.height, h - fy)
            score = detection.categories[0].score if detection.categories else 0.5
            if fw > 10 and fh > 10:
                faces.append({
                    "x": fx, "y": fy, "w": fw, "h": fh,
                    "confidence": round(score, 3),
                })

    return faces


def _detect_mediapipe_solutions(img_np: np.ndarray) -> list:
    """MediaPipe Solutions API (v0.9.x) ile yuz tespiti."""
    mp_face = mp.solutions.face_detection
    h, w = img_np.shape[:2]
    faces = []
    img_rgb = np.ascontiguousarray(img_np[:, :, :3])

    with mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.3) as detector:
        results = detector.process(img_rgb)
        if results and results.detections:
            for det in results.detections:
                bbox = det.location_data.relative_bounding_box
                fx = max(0, int(bbox.xmin * w))
                fy = max(0, int(bbox.ymin * h))
                fw = min(int(bbox.width * w), w - fx)
                fh = min(int(bbox.height * h), h - fy)
                if fw > 10 and fh > 10:
                    faces.append({
                        "x": fx, "y": fy, "w": fw, "h": fh,
                        "confidence": round(det.score[0], 3),
                    })
    return faces


def _detect_opencv_haar(img_np: np.ndarray) -> list:
    """OpenCV Haar Cascade fallback."""
    cascade_file = None
    try:
        default = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        if os.path.exists(default):
            cascade_file = default
    except Exception:
        pass

    if cascade_file is None:
        local = os.path.join(os.path.dirname(__file__), "..", "models", "haarcascade_frontalface_default.xml")
        if os.path.exists(local):
            cascade_file = local

    if cascade_file is None:
        return []

    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    cascade = cv2.CascadeClassifier(cascade_file)
    if cascade.empty():
        return []

    rects = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(rects) == 0:
        return []

    return [
        {"x": int(x), "y": int(y), "w": int(w), "h": int(h), "confidence": 0.85}
        for (x, y, w, h) in rects
    ]


def _download_blaze_face_model(save_path: str):
    """BlazeFace modelini indir."""
    import urllib.request
    url = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    urllib.request.urlretrieve(url, save_path)
    _log(f"BlazeFace model indirildi: {save_path}")


def draw_face_boxes(image, faces: list = None) -> Image.Image:
    """Yuz kutularini gorsel uzerine ciz."""
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    img = image.copy().convert("RGB")

    if faces is None:
        faces = detect_faces(img)

    if not faces:
        draw = ImageDraw.Draw(img)
        draw.text((10, 10), "No face detected", fill="#FF6B6B")
        return img

    draw = ImageDraw.Draw(img)
    for i, f in enumerate(faces):
        x, y, w, h = f["x"], f["y"], f["w"], f["h"]
        conf = f.get("confidence", 0)

        # Neon yesil dikdortgen (kalin)
        for offset in range(4):
            draw.rectangle(
                [x - offset, y - offset, x + w + offset, y + h + offset],
                outline="#00FF88",
            )

        # Label
        label = f"Face #{i+1} ({conf:.0%})"
        label_y = max(0, y - 22)
        draw.rectangle([x, label_y, x + len(label) * 8, label_y + 18], fill="#00FF88")
        draw.text((x + 4, label_y + 2), label, fill="#000000")

    return img


def _log(msg: str):
    try:
        print(f"[FaceDetector] {msg}")
    except UnicodeEncodeError:
        print(f"[FaceDetector] {msg.encode('ascii', 'replace').decode()}")
