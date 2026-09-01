"""Smoke-test the modules that make up the current runtime architecture."""

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MODULES = [
    # Core model and training runtime
    "core.dual_mobilenetv3",
    "core.efficientnet_teacher",
    "core.loss_utils",
    "core.trainer",
    "core.evaluation",
    "core.data_pipeline",
    # Inference and explainability
    "inference.predictor",
    "inference.tta_inference",
    "inference.model_ensemble",
    "inference.xai_module",
    "inference.hybrid_xai",
    "inference.explainability_report",
    "inference.fastcam_demo",
    # API and active ML extensions
    "api.server",
    "ml_extensions.mc_dropout",
    "ml_extensions.temperature_scaling",
    # Services used by the current repository
    "services.llm_module",
    "services.pdf_report",
    # Root configuration
    "config",
]


def run_tests():
    """Import every active runtime module and return the number of failures."""
    passed = []
    failed = []

    for module_name in MODULES:
        try:
            importlib.import_module(module_name)
            print(f"  ✓ {module_name}")
            passed.append(module_name)
        except Exception as exc:  # noqa: BLE001 - smoke test must report any import-time failure
            print(f"  ✗ {module_name}: {exc}")
            failed.append(module_name)

    print(f"\n{'=' * 50}")
    print(f"✅ Geçen: {len(passed)}/{len(MODULES)}")
    if failed:
        print(f"❌ Başarısız: {failed}")
    print("=" * 50)
    return len(failed)


if __name__ == "__main__":
    sys.exit(run_tests())
