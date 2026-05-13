"""Deepfake v3 — Faz 4 Test: Eğitim & Loss."""
import sys, os, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import DEVICE

def test_focal_loss():
    from core.loss_utils import FocalLoss
    loss = FocalLoss()(torch.randn(4, 2).to(DEVICE), torch.randint(0, 2, (4,)).to(DEVICE))
    assert loss.item() > 0
    print(f"  ✓ FocalLoss: {loss.item():.4f}")

def test_kd_loss():
    from core.loss_utils import KnowledgeDistillationLoss
    loss = KnowledgeDistillationLoss()(torch.randn(4, 2), torch.randn(4, 2))
    assert loss.item() > 0
    print(f"  ✓ KD Loss: {loss.item():.4f}")

def test_combined_loss():
    from core.loss_utils import CombinedLoss
    result = CombinedLoss()(torch.randn(4, 2), torch.randint(0, 2, (4,)), torch.randn(4, 2))
    assert "total_loss" in result
    print(f"  ✓ CombinedLoss: {result['total_loss'].item():.4f}")

def test_gradient_backward():
    from core.loss_utils import FocalLoss
    p = torch.randn(4, 2, requires_grad=True)
    FocalLoss()(p, torch.randint(0, 2, (4,))).backward()
    assert p.grad is not None
    print("  ✓ Gradient backward")

if __name__ == "__main__":
    print("=== Faz 4: Eğitim & Loss ===")
    for fn in [test_focal_loss, test_kd_loss, test_combined_loss, test_gradient_backward]:
        try: fn()
        except Exception as e: print(f"  ✗ {fn.__name__}: {e}"); sys.exit(1)
    print("✅ Faz 4 tamamlandı")
