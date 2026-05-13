"""Deepfake v3 — FastCAM Demo Betiği."""
import numpy as np, torch
from PIL import Image
from config import model_cfg, DEVICE

def run_fastcam_demo(image_path=None):
    """FastCAM tek görüntü demo."""
    from inference.predictor import DeepfakePredictor
    from inference.hybrid_xai import FastCAM

    predictor = DeepfakePredictor()
    if image_path:
        img = np.array(Image.open(image_path).convert("RGB"))
    else:
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        print("Rastgele goruntu kullaniliyor")

    result = predictor.predict(img)
    print(f"Tahmin: {result['label']} (fake={result['fake_prob']:.3f})")

    rgb, freq, mesh = predictor.preprocess(img)
    cam = FastCAM(predictor.model)
    saliency = cam.generate(rgb, freq, mesh)
    print(f"FastCAM shape: {saliency.shape}, max={saliency.max():.3f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(img)
        axes[0].set_title("Orijinal")
        axes[1].imshow(saliency, cmap="jet")
        axes[1].set_title("FastCAM (SMOE)")
        for ax in axes: ax.axis("off")
        plt.savefig("fastcam_output.png", dpi=150, bbox_inches="tight")
        print("Kaydedildi: fastcam_output.png")
    except Exception as e:
        print(f"Gorsellestime hatasi: {e}")

if __name__ == "__main__":
    import sys
    run_fastcam_demo(sys.argv[1] if len(sys.argv) > 1 else None)
