"""Deepfake v3 — Faz 1 Test: Config & Altyapı."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

def test_config_load():
    from config import model_cfg, paths, DEVICE, VERSION
    assert VERSION == "3.0.0", f"Version mismatch: {VERSION}"
    assert model_cfg.IMG_SIZE == 224
    assert model_cfg.RGB_BACKBONE == "mobilenet_v3_large"
    assert model_cfg.LSTM_BIDIRECTIONAL is True
    print("  ✓ Config yükleme")

def test_paths():
    from config import paths
    paths.ensure_dirs()
    assert paths.MODEL_DIR.exists()
    assert paths.REPORTS_DIR.exists()
    print("  ✓ Path doğrulama")

def test_device():
    from config import DEVICE
    assert DEVICE.type in ("cpu", "cuda")
    print(f"  ✓ Cihaz: {DEVICE}")

def test_packages_exist():
    pkgs = ["core", "inference", "api", "ml_extensions", "training", "services", "security", "deploy", "utils"]
    for p in pkgs:
        assert os.path.isdir(os.path.join(os.path.dirname(os.path.dirname(__file__)), p)), f"{p}/ yok"
    print(f"  ✓ {len(pkgs)} paket mevcut")

if __name__ == "__main__":
    print("=== Faz 1: Config & Altyapı ===")
    for fn in [test_config_load, test_paths, test_device, test_packages_exist]:
        try: fn()
        except Exception as e: print(f"  ✗ {fn.__name__}: {e}"); sys.exit(1)
    print("✅ Faz 1 tamamlandı")
