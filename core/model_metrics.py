"""
Model Performans Metrikleri — ROC/AUC, Confusion Matrix, F1, EER.

Feedback havuzundaki etiketli gorseller uzerinden modeli yeniden
degerlendirerek nicel metrikler uretir.
"""
import numpy as np
from typing import Optional

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

PLOT_LAYOUT = dict(
    paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
    font=dict(color="#e6edf3", family="Inter"),
    margin=dict(l=40, r=20, t=50, b=40),
)


def compute_metrics_from_history() -> dict:
    """
    Feedback havuzundaki etiketli gorseller uzerinden modeli
    yeniden degerlendirip metrik hesapla.

    Returns:
        dict: {
            "labels": list[int],     # 0=REAL, 1=FAKE (ground truth)
            "preds": list[int],      # 0=REAL, 1=FAKE (model tahmini)
            "probs": list[float],    # fake_prob (model ciktisi)
            "count": int,
            "ready": bool,
        }
    """
    from core.fine_tuner import FEEDBACK_DIR
    from pathlib import Path

    image_paths = []
    ground_truths = []  # 0=REAL, 1=FAKE

    for label_name, label_id in [("REAL", 0), ("FAKE", 1)]:
        label_dir = FEEDBACK_DIR / label_name
        if not label_dir.exists():
            continue
        for ext in ["*.jpg", "*.png", "*.jpeg"]:
            for fpath in label_dir.glob(ext):
                image_paths.append(str(fpath))
                ground_truths.append(label_id)

    if len(image_paths) < 4:
        return {"labels": [], "preds": [], "probs": [],
                "count": len(image_paths), "ready": False}

    # Model ile yeniden degerlendirme
    try:
        from inference.predictor import get_predictor
        predictor = get_predictor()
    except Exception:
        return {"labels": [], "preds": [], "probs": [],
                "count": len(image_paths), "ready": False}

    preds = []
    probs = []
    valid_labels = []

    for i, path in enumerate(image_paths):
        try:
            result = predictor.predict(path)
            preds.append(1 if result["label"] == "FAKE" else 0)
            probs.append(result["fake_prob"])
            valid_labels.append(ground_truths[i])
        except Exception:
            continue

    return {
        "labels": valid_labels,
        "preds": preds,
        "probs": probs,
        "count": len(valid_labels),
        "ready": len(valid_labels) >= 4,
    }


def generate_roc_plot(labels: list, probs: list) -> Optional[object]:
    """
    ROC egrisi + AUC degeri.

    Args:
        labels: Ground truth listesi (0/1)
        probs: Model fake_prob listesi

    Returns:
        plotly.Figure
    """
    if not HAS_PLOTLY or len(labels) < 4:
        return None

    try:
        from sklearn.metrics import roc_curve, auc
    except ImportError:
        return None

    fpr, tpr, thresholds = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)

    fig = go.Figure()

    # ROC egrisi
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode="lines",
        line=dict(color="#06B6D4", width=2.5),
        name=f"ROC (AUC = {roc_auc:.4f})",
        fill="tozeroy",
        fillcolor="rgba(6, 182, 212, 0.1)",
    ))

    # Rastgele tahmin cizgisi
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        line=dict(color="#94A3B8", width=1, dash="dash"),
        name="Random (AUC = 0.5)",
        showlegend=True,
    ))

    # EER noktasi (FPR = 1 - TPR olan nokta)
    eer_idx = np.argmin(np.abs(fpr - (1 - tpr)))
    eer = float(fpr[eer_idx])
    fig.add_trace(go.Scatter(
        x=[fpr[eer_idx]], y=[tpr[eer_idx]],
        mode="markers+text",
        marker=dict(color="#F59E0B", size=12, symbol="diamond"),
        text=[f"EER={eer:.3f}"],
        textposition="top right",
        textfont=dict(color="#F59E0B"),
        name=f"EER = {eer:.4f}",
    ))

    fig.update_layout(
        **PLOT_LAYOUT,
        title=f"ROC Egrisi (AUC = {roc_auc:.4f})",
        height=320,
        xaxis=dict(title="False Positive Rate (FPR)", range=[0, 1],
                   gridcolor="#1e293b"),
        yaxis=dict(title="True Positive Rate (TPR)", range=[0, 1.02],
                   gridcolor="#1e293b"),
        legend=dict(bgcolor="rgba(0,0,0,0.4)"),
    )

    return fig


def generate_confusion_matrix_plot(labels: list, preds: list) -> Optional[object]:
    """
    2x2 Confusion Matrix heatmap (REAL/FAKE).

    Args:
        labels: Ground truth (0=REAL, 1=FAKE)
        preds: Model tahminleri (0=REAL, 1=FAKE)

    Returns:
        plotly.Figure
    """
    if not HAS_PLOTLY or len(labels) < 4:
        return None

    try:
        from sklearn.metrics import confusion_matrix
    except ImportError:
        return None

    cm = confusion_matrix(labels, preds, labels=[0, 1])
    class_names = ["REAL", "FAKE"]

    # Annotasyon metinleri
    annotations = []
    for i in range(2):
        for j in range(2):
            annotations.append(
                dict(
                    x=class_names[j], y=class_names[i],
                    text=str(cm[i][j]),
                    font=dict(size=24, color="white"),
                    showarrow=False,
                )
            )

    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=class_names,
        y=class_names,
        colorscale=[
            [0, "#161b22"],
            [0.5, "#1e40af"],
            [1, "#06B6D4"],
        ],
        showscale=True,
        colorbar=dict(title="Sayi"),
        hovertemplate="Gercek: %{y}<br>Tahmin: %{x}<br>Sayi: %{z}<extra></extra>",
    ))

    fig.update_layout(
        **PLOT_LAYOUT,
        title="Confusion Matrix",
        height=300,
        xaxis=dict(title="Tahmin Edilen", side="bottom"),
        yaxis=dict(title="Gercek Etiket", autorange="reversed"),
        annotations=annotations,
    )

    return fig


def generate_metrics_summary(labels: list, preds: list, probs: list) -> str:
    """
    Markdown tablosu: Accuracy, F1, Precision, Recall, AUC, EER.

    Returns:
        str — Markdown formatlı metrik ozeti
    """
    if len(labels) < 4:
        return "> Yeterli veri yok (minimum 4 etiketli gorsel gerekli)."

    try:
        from sklearn.metrics import (
            accuracy_score, f1_score, precision_score,
            recall_score, roc_auc_score, roc_curve,
        )
    except ImportError:
        return "> sklearn yuklu degil."

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, zero_division=0)
    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)

    # AUC
    try:
        auc_val = roc_auc_score(labels, probs)
    except ValueError:
        auc_val = 0.0

    # EER
    try:
        fpr, tpr, _ = roc_curve(labels, probs)
        eer_idx = np.argmin(np.abs(fpr - (1 - tpr)))
        eer = float(fpr[eer_idx])
    except Exception:
        eer = 0.0

    n_real = sum(1 for l in labels if l == 0)
    n_fake = sum(1 for l in labels if l == 1)

    return (
        f"### Model Performans Metrikleri\n\n"
        f"| Metrik | Deger |\n|---|---|\n"
        f"| **Accuracy** | {acc:.4f} ({acc*100:.1f}%) |\n"
        f"| **F1 Score** | {f1:.4f} |\n"
        f"| **Precision** | {precision:.4f} |\n"
        f"| **Recall (Sensitivity)** | {recall:.4f} |\n"
        f"| **AUC** | {auc_val:.4f} |\n"
        f"| **EER** | {eer:.4f} |\n"
        f"| **Veri Sayisi** | {len(labels)} (REAL={n_real}, FAKE={n_fake}) |\n"
    )
