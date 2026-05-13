"""
Deepfake Detection System v3 — XAI Açıklanabilirlik Raporu
XAI kalite skoru, EU AI Act Madde 13 uyumluluk.
"""
import numpy as np


def compute_faithfulness_score(pixel_deletion_auc, pixel_insertion_auc,
                                sanity_check_passed=True):
    """XAI faithfulness skoru hesapla."""
    bonus = 0.2 if sanity_check_passed else 0.0
    score = (pixel_deletion_auc + pixel_insertion_auc) / 2 + bonus * 0.5
    return min(score, 1.0)


def check_eu_ai_act_compliance(faithfulness_score, ece_score=None):
    """EU AI Act Madde 13 şeffaflık kontrolü."""
    checks = {
        "xai_faithful": faithfulness_score > 0.6,
        "calibration_ok": ece_score is not None and ece_score < 0.10,
    }
    compliant = all(checks.values())
    return {"compliant": compliant, "checks": checks, "score": faithfulness_score}


def saliency_quality_metrics(saliency_map):
    """Saliency haritası kalite metrikleri."""
    if saliency_map is None:
        return {}
    flat = saliency_map.flatten()
    return {
        "coverage": float((flat > 0.1).mean()),
        "sparsity": float((flat > 0.5).mean()),
        "max_activation": float(flat.max()),
        "mean_activation": float(flat.mean()),
        "entropy": float(-np.sum(flat * np.log2(flat + 1e-10)) / len(flat)),
    }
