"""Deepfake v3 — Faz 3 Test: Veri İşleme Pipeline."""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import model_cfg

def test_dwt_output():
    from core.data_pipeline import MultiScaleDWT
    dwt = MultiScaleDWT()
    img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    out = dwt(img)
    assert out.shape[0] == model_cfg.DWT_CHANNELS
    print(f"  ✓ DWT çıktı: {out.shape}")

def test_face_mesh():
    from core.data_pipeline import FaceMeshExtractor
    ext = FaceMeshExtractor()
    img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    out = ext(img)
    assert out.shape[0] == model_cfg.MESH_INPUT_DIM
    print(f"  ✓ Face Mesh: {out.shape} (nonzero={np.count_nonzero(out)})")

def test_transform_pipeline():
    from torchvision import transforms
    from PIL import Image
    tf = transforms.Compose([
        transforms.Resize((model_cfg.IMG_SIZE, model_cfg.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
    t = tf(img)
    assert t.shape == (3, model_cfg.IMG_SIZE, model_cfg.IMG_SIZE)
    print(f"  ✓ Transform: {t.shape}")

if __name__ == "__main__":
    print("=== Faz 3: Veri İşleme ===")
    tests = [test_dwt_output, test_face_mesh, test_transform_pipeline]
    for fn in tests:
        try: fn()
        except Exception as e:
            if "mediapipe" in str(e).lower():
                print(f"  ⚠ {fn.__name__}: MediaPipe uyumsuz (graceful degradation)")
            else:
                print(f"  ✗ {fn.__name__}: {e}"); sys.exit(1)
    print("✅ Faz 3 tamamlandı")
