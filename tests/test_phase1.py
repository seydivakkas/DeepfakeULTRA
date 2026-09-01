"""Phase 1 tests for current configuration and repository layout."""

from pathlib import Path

from config import DEVICE, VERSION, model_cfg, paths


ROOT = Path(__file__).resolve().parents[1]


def test_config_load():
    """Validate the active configuration contract without pinning a stale release."""
    version_parts = VERSION.split(".")
    assert len(version_parts) == 3
    assert all(part.isdigit() for part in version_parts)
    assert model_cfg.IMG_SIZE == 224
    assert model_cfg.RGB_BACKBONE == "mobilenet_v3_large"
    assert model_cfg.XBRANCH_HEADS == 4
    assert model_cfg.XBRANCH_LAYERS == 2
    assert model_cfg.USE_HYBRID_FREQ is True


def test_paths():
    """Ensure runtime output directories can be materialized."""
    paths.ensure_dirs()
    assert paths.MODEL_DIR.exists()
    assert paths.REPORTS_DIR.exists()


def test_device():
    """The runtime must select a supported Torch device."""
    assert DEVICE.type in ("cpu", "cuda")


def test_runtime_packages_exist():
    """Track the current package layout after training moved into core/trainer.py."""
    packages = [
        "core",
        "inference",
        "api",
        "ml_extensions",
        "services",
        "security",
        "deploy",
        "utils",
    ]
    for package in packages:
        assert (ROOT / package).is_dir(), f"{package}/ missing"

    assert (ROOT / "core" / "trainer.py").is_file(), "core/trainer.py missing"
