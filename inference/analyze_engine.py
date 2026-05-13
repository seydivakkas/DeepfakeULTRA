"""
Deepfake Detection System v3.0 — Birlesik Analiz Motoru
Tum pipeline'i tek fonksiyonda orkestre eder:
  Predictor -> TTA -> XAI -> DWT -> Face Detection -> Watermark -> DB kayit
"""
import numpy as np
from PIL import Image
from typing import Optional
from config import model_cfg, DEVICE


def analyze_image(
    image,
    tta_count: int = 5,
    xai_methods: list = None,
    run_dwt: bool = True,
    apply_watermark: bool = True,
    device: str = "auto",
    source: str = "ui",
    filename: str = "uploaded_image",
) -> dict:
    if xai_methods is None:
        xai_methods = ["gradcam", "eigencam", "fastcam", "lime"]

    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image).convert("RGB")
    else:
        image = image.convert("RGB")

    img_np = np.array(image)

    # 1. Temel tahmin
    from inference.predictor import get_predictor
    predictor = get_predictor()
    base_result = predictor.predict(img_np)

    result = {
        "verdict": base_result["label"],
        "fake_prob": base_result["fake_prob"],
        "real_prob": base_result["real_prob"],
        "confidence": base_result["confidence"],
        "calibrated": False,
        "gradcam_score": 0.0,
        "counterfactual_prob": 0.0,
        "tta_std": 0.0,
        "tta_individual": [],
        "face_boxes": [],
        "heatmaps": {},
        "dwt_map": None,
        "fusion_weights": {"rgb": 50.0, "freq": 49.0, "geo": 1.0},
        "watermarked_image": None,
        "original_image": image,
    }

    # 2. TTA
    try:
        from inference.tta_inference import TTAPredictor
        tta = TTAPredictor(predictor, n_augmentations=tta_count)
        tta_result = tta.predict_tta(img_np)
        result["verdict"] = tta_result["label"]
        result["fake_prob"] = tta_result["fake_prob"]
        result["real_prob"] = tta_result["real_prob"]
        result["confidence"] = tta_result["confidence"]
        result["tta_std"] = tta_result["std"]
        result["tta_individual"] = [
            round(p[1], 4) for p in tta_result["individual_predictions"]
        ]
    except Exception as e:
        _log(f"TTA hatasi: {e}")

    # 3. Yuz tespiti
    try:
        from core.face_detector import detect_faces
        result["face_boxes"] = detect_faces(image)
    except Exception as e:
        _log(f"Yuz tespiti hatasi: {e}")

    # 4. XAI Heatmaps
    rgb, freq, mesh, _ = predictor.preprocess(img_np)

    if "gradcam" in xai_methods:
        try:
            from inference.xai_module import GradCAMPlusPlus
            _enable_lstm_training(predictor.model)
            gcam = GradCAMPlusPlus(predictor.model)
            cam = gcam.generate(rgb, freq, mesh)
            _disable_lstm_training(predictor.model)
            result["heatmaps"]["gradcam"] = _cam_to_image(img_np, cam, name="GradCAM")
            result["gradcam_score"] = round(float(cam.mean()), 4)
        except Exception as e:
            _disable_lstm_training(predictor.model)
            _log(f"GradCAM++ hatasi: {e}")

    if "eigencam" in xai_methods:
        try:
            from inference.xai_module import EigenCAM
            ecam = EigenCAM(predictor.model)
            cam = ecam.generate(rgb, freq, mesh)
            result["heatmaps"]["eigencam"] = _cam_to_image(img_np, cam, name="EigenCAM")
        except Exception as e:
            _log(f"EigenCAM hatasi: {e}")

    if "fastcam" in xai_methods:
        try:
            from inference.hybrid_xai import FastCAM
            fcam = FastCAM(predictor.model)
            cam = fcam.generate(rgb, freq, mesh)
            result["heatmaps"]["fastcam"] = _cam_to_image(img_np, cam, name="FastCAM")
        except Exception as e:
            _log(f"FastCAM hatasi: {e}")

    if "lime" in xai_methods:
        try:
            from inference.hybrid_xai import GuidedLIME
            lime_gen = GuidedLIME(predictor)
            lime_mask = lime_gen.generate(img_np, num_samples=50)
            if lime_mask.max() > 0:
                result["heatmaps"]["lime"] = _cam_to_image(img_np, lime_mask, alpha=0.5, name="LIME")
        except Exception as e:
            _log(f"LIME hatasi: {e}")

    # 5. Counterfactual
    try:
        from inference.xai_module import CounterfactualXAI
        _enable_lstm_training(predictor.model)
        cf = CounterfactualXAI(predictor.model)
        cf_map, flipped = cf.generate(rgb, freq, mesh)
        _disable_lstm_training(predictor.model)
        result["counterfactual_prob"] = round(float(cf_map.mean()), 4)
    except Exception as e:
        _disable_lstm_training(predictor.model)
        _log(f"Counterfactual hatasi: {e}")

    # 6. DWT Frekans Haritasi
    if run_dwt:
        try:
            from core.frequency import generate_dwt_visualization, get_fusion_weights
            result["dwt_map"] = generate_dwt_visualization(image)
            result["fusion_weights"] = get_fusion_weights(predictor.model)
        except Exception as e:
            _log(f"DWT hatasi: {e}")

    # 7. Watermark
    if apply_watermark:
        try:
            from core.watermark import apply_invisible_watermark
            result["watermarked_image"] = apply_invisible_watermark(image)
        except Exception as e:
            result["watermarked_image"] = image
            _log(f"Watermark hatasi: {e}")

    # 8. DB kayit
    try:
        from db.database import get_db
        db = get_db()
        analysis_id = db.save_analysis(
            filename=filename,
            verdict=result["verdict"],
            confidence=result["confidence"],
            fake_prob=result["fake_prob"],
            real_prob=result["real_prob"],
            source=source,
            xai_methods=xai_methods,
            tta_count=tta_count,
            extra_data={
                "tta_std": result["tta_std"],
                "gradcam_score": result["gradcam_score"],
                "counterfactual_prob": result["counterfactual_prob"],
                "face_count": len(result["face_boxes"]),
            },
        )
        result["analysis_id"] = analysis_id
    except Exception as e:
        result["analysis_id"] = None
        _log(f"DB kayit hatasi: {e}")

    # 9. Embedding havuzuna ekle (t-SNE/UMAP icin)
    try:
        from core.embedding_viz import extract_embedding, add_to_pool
        embedding = extract_embedding(predictor.model, rgb, freq, mesh)
        add_to_pool(
            embedding=embedding,
            label=result["verdict"],
            filename=filename,
            fake_prob=result["fake_prob"],
        )
    except Exception as e:
        _log(f"Embedding kayit hatasi: {e}")

    return result

def _add_watermark(img: Image.Image, text: str = "DeepfakeULTRA") -> Image.Image:
    """Görsele yarı saydam watermark ekle."""
    try:
        from PIL import ImageDraw, ImageFont
        img = img.convert("RGBA")
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        w, h = img.size
        font_size = max(10, w // 16)
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except OSError:
            font = ImageFont.load_default()
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x = w - tw - 6
        y = h - th - 6
        draw.rectangle([x - 3, y - 2, x + tw + 3, y + th + 2], fill=(0, 0, 0, 100))
        draw.text((x, y), text, fill=(255, 255, 255, 180), font=font)
        result = Image.alpha_composite(img, overlay)
        return result.convert("RGB")
    except Exception:
        return img.convert("RGB") if img.mode == "RGBA" else img


def _cam_to_image(
    img_np: np.ndarray, cam: np.ndarray, alpha: float = 0.5, name: str = "XAI"
) -> Image.Image:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        heatmap = (plt.cm.jet(cam)[:, :, :3] * 255).astype(np.uint8)
        img_resized = np.array(
            Image.fromarray(img_np).resize((model_cfg.IMG_SIZE, model_cfg.IMG_SIZE))
        )
        blended = (img_resized * (1 - alpha) + heatmap * alpha).astype(np.uint8)
        result = Image.fromarray(blended)
        result = _add_watermark(result)
        return result
    except Exception:
        return Image.fromarray(img_np)



def _log(msg: str):
    """Windows cp1254 uyumlu log."""
    try:
        print(f"[WARN] {msg}")
    except UnicodeEncodeError:
        print(f"[WARN] {msg.encode('ascii', 'replace').decode()}")


def _enable_lstm_training(model):
    """
    Sadece LSTM modullerini training moduna al.
    BatchNorm'lar eval'da kalir (batch_size=1 hatasi onlenir).
    cudnn RNN backward sadece training modda calisir.
    """
    import torch.nn as nn
    for name, module in model.named_modules():
        if isinstance(module, (nn.LSTM, nn.GRU)):
            module.train()


def _disable_lstm_training(model):
    """Tum modeli eval moduna dondur."""
    model.eval()
