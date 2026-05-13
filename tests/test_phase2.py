"""Deepfake v3 — Faz 2 Test: Model Mimarisi."""
import sys, os, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import model_cfg, DEVICE

def test_forward_pass():
    from core.dual_mobilenetv3 import DualPathDeepfakeDetector
    model = DualPathDeepfakeDetector().to(DEVICE).eval()
    s = model_cfg.IMG_SIZE
    rgb = torch.randn(2, 3, s, s).to(DEVICE)
    freq = torch.randn(2, model_cfg.DWT_CHANNELS, s, s).to(DEVICE)
    mesh = torch.randn(2, model_cfg.MESH_INPUT_DIM).to(DEVICE)
    with torch.no_grad():
        out = model(rgb, freq, mesh)
    assert out.shape == (2, 2)
    print(f"  ✓ Forward pass: {out.shape}")

def test_parameter_count():
    from core.dual_mobilenetv3 import DualPathDeepfakeDetector
    model = DualPathDeepfakeDetector()
    total = sum(p.numel() for p in model.parameters())
    assert total > 5_000_000
    print(f"  ✓ Parametre: {total:,}")

def test_gradient_flow():
    from core.dual_mobilenetv3 import DualPathDeepfakeDetector
    model = DualPathDeepfakeDetector().to(DEVICE).train()
    s = model_cfg.IMG_SIZE
    rgb = torch.randn(2, 3, s, s, requires_grad=True).to(DEVICE)
    freq = torch.randn(2, model_cfg.DWT_CHANNELS, s, s).to(DEVICE)
    mesh = torch.randn(2, model_cfg.MESH_INPUT_DIM).to(DEVICE)
    out = model(rgb, freq, mesh)
    out.sum().backward()
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    assert has_grad
    print("  ✓ Gradient akışı")

def test_teacher_model():
    from core.efficientnet_teacher import EfficientNetTeacher
    teacher = EfficientNetTeacher().to(DEVICE).eval()
    with torch.no_grad():
        out = teacher(torch.randn(2, 3, 224, 224).to(DEVICE))
    assert out.shape == (2, 2)
    print(f"  ✓ Teacher: {out.shape}")

def test_forward_branches():
    from core.dual_mobilenetv3 import DualPathDeepfakeDetector
    model = DualPathDeepfakeDetector().to(DEVICE).eval()
    s = model_cfg.IMG_SIZE
    rgb = torch.randn(2, 3, s, s).to(DEVICE)
    freq = torch.randn(2, model_cfg.DWT_CHANNELS, s, s).to(DEVICE)
    mesh = torch.randn(2, model_cfg.MESH_INPUT_DIM).to(DEVICE)
    with torch.no_grad():
        branches = model.forward_branches(rgb, freq, mesh)
    assert "rgb_logits" in branches
    print(f"  ✓ Branch çıktıları: {list(branches.keys())}")

if __name__ == "__main__":
    print("=== Faz 2: Model Mimarisi ===")
    for fn in [test_forward_pass, test_parameter_count, test_gradient_flow,
               test_teacher_model, test_forward_branches]:
        try: fn()
        except Exception as e: print(f"  ✗ {fn.__name__}: {e}"); sys.exit(1)
    print("✅ Faz 2 tamamlandı")
