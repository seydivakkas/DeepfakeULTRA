"""Deepfake v3 — Smoke Test (52 modül, paket mimarisi)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

MODULES = [
    # Core
    "core.dual_mobilenetv3", "core.efficientnet_teacher", "core.vit_backbone",
    "core.loss_utils", "core.trainer", "core.evaluation", "core.data_pipeline",
    # Inference
    "inference.predictor", "inference.tta_inference", "inference.model_ensemble",
    "inference.xai_module", "inference.hybrid_xai",
    "inference.explainability_report", "inference.fastcam_demo",
    # API
    "api.server", "api.demo_ui",
    # ML Extensions
    "ml_extensions.active_learning", "ml_extensions.optuna_hpo",
    "ml_extensions.pcgrad", "ml_extensions.mc_dropout",
    "ml_extensions.temperature_scaling", "ml_extensions.data_drift_monitor",
    "ml_extensions.xai_faithfulness", "ml_extensions.curriculum_scheduler",
    "ml_extensions.cross_dataset_benchmark",
    "ml_extensions.adversarial_robustness_suite",
    "ml_extensions.synthetic_attack_augmentation",
    # Training
    "training.adversarial_train", "training.adversarial_testing",
    "training.continual_learning", "training.online_trainer",
    "training.pretrain_contrastive", "training.dino_pretrain",
    # Services
    "services.llm_module", "services.rag_module", "services.pdf_report",
    "services.telegram_bot", "services.webhook_notifications",
    "services.analytics_dashboard", "services.multi_language",
    # Security
    "security.rbac", "security.audit_log", "security.model_watermark",
    # Deploy
    "deploy.export_onnx", "deploy.quantize",
    "deploy.benchmark_suite", "deploy.ood_detector",
    # Utils
    "utils.batch_processor", "utils.multi_face_detector", "utils.dfdc_prepare",
    # Config (root)
    "config",
]

def run_tests():
    passed, failed = [], []
    for m in MODULES:
        try:
            __import__(m)
            short = m.split(".")[-1]
            print(f"  ✓ {m}")
            passed.append(m)
        except Exception as e:
            print(f"  ✗ {m}: {e}")
            failed.append(m)

    print(f"\n{'='*50}")
    print(f"✅ Geçen: {len(passed)}/{len(MODULES)}")
    if failed:
        print(f"❌ Başarısız: {failed}")
    print(f"{'='*50}")
    return len(failed)

if __name__ == "__main__":
    sys.exit(run_tests())
