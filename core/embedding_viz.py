"""
Embedding Gorsellestirme — t-SNE / UMAP ile ozellik uzayi haritalama.

Modelin fusion katmanindan cikarilan 960-dim vektorleri 2D'ye indirger.
REAL (yesil) ve FAKE (kirmizi) kumelerin ayrisimini gorsellestirir.
"""
import numpy as np
from typing import Optional

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

PLOT_LAYOUT = dict(
    paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
    font=dict(color="#e6edf3", family="Inter"),
    margin=dict(l=40, r=20, t=50, b=40),
)

# ── Embedding havuzu (bellek ici — analiz gecmisinden birikir) ──
_embedding_pool = {
    "embeddings": [],   # list of np.ndarray (960-dim)
    "labels": [],       # list of str ("REAL" / "FAKE")
    "filenames": [],    # list of str
    "probs": [],        # list of float (fake_prob)
}

MIN_POINTS_FOR_VIZ = 20


def clear_pool():
    """Embedding havuzunu temizle."""
    _embedding_pool["embeddings"].clear()
    _embedding_pool["labels"].clear()
    _embedding_pool["filenames"].clear()
    _embedding_pool["probs"].clear()


def get_pool_size() -> int:
    return len(_embedding_pool["embeddings"])


def add_to_pool(embedding: np.ndarray, label: str,
                filename: str = "", fake_prob: float = 0.0):
    """Havuza yeni embedding ekle."""
    _embedding_pool["embeddings"].append(embedding.flatten())
    _embedding_pool["labels"].append(label)
    _embedding_pool["filenames"].append(filename)
    _embedding_pool["probs"].append(fake_prob)


def extract_embedding(model, rgb, freq, mesh) -> np.ndarray:
    """
    Modelin fusion katmanindan embedding vektoru cikar.

    Hook kullanmadan dogrudan extract_features + fusion cagrilir.
    Sonuc: 960-dim vektor (FUSION_DIM).

    Args:
        model: DualPathDeepfakeDetector instance
        rgb: (1, 3, 224, 224) tensor
        freq: (1, 12, 224, 224) tensor
        mesh: (1, 1404) tensor

    Returns:
        np.ndarray shape (960,)
    """
    import torch

    model.eval()
    with torch.no_grad():
        rgb_feat, freq_feat, mesh_feat = model.extract_features(rgb, freq, mesh)
        fused = model.fusion(rgb_feat, freq_feat, mesh_feat)

    return fused[0].cpu().numpy()


def generate_tsne_plot(
    embeddings: Optional[np.ndarray] = None,
    labels: Optional[list] = None,
    filenames: Optional[list] = None,
    probs: Optional[list] = None,
    perplexity: int = 15,
) -> Optional[object]:
    """
    t-SNE ile 2D gorsellestirme olustur.

    Args:
        embeddings: (N, D) array — None ise havuzdan alinir
        labels: N uzunlugunda etiket listesi
        filenames: N uzunlugunda dosya adi listesi
        probs: N uzunlugunda fake_prob listesi
        perplexity: t-SNE perplexity (nokta sayisina gore ayarla)

    Returns:
        plotly.Figure veya None (yetersiz veri / kutuphane eksik)
    """
    if not HAS_PLOTLY:
        return None

    # Havuzdan al
    if embeddings is None:
        if get_pool_size() < MIN_POINTS_FOR_VIZ:
            return None
        embeddings = np.array(_embedding_pool["embeddings"])
        labels = _embedding_pool["labels"]
        filenames = _embedding_pool["filenames"]
        probs = _embedding_pool["probs"]

    n = len(embeddings)
    if n < MIN_POINTS_FOR_VIZ:
        return None

    try:
        from sklearn.manifold import TSNE
    except ImportError:
        return None

    # Perplexity, nokta sayisindan kucuk olmali
    effective_perplexity = min(perplexity, max(2, n // 3))

    tsne = TSNE(n_components=2, perplexity=effective_perplexity,
                random_state=42, max_iter=1000, learning_rate="auto")
    try:
        coords = tsne.fit_transform(embeddings)
    except TypeError:
        # Eski sklearn: n_iter yerine max_iter desteklenmeyebilir
        tsne = TSNE(n_components=2, perplexity=effective_perplexity,
                    random_state=42, learning_rate="auto")
        coords = tsne.fit_transform(embeddings)

    return _create_scatter(coords, labels, filenames, probs, "t-SNE")


def generate_umap_plot(
    embeddings: Optional[np.ndarray] = None,
    labels: Optional[list] = None,
    filenames: Optional[list] = None,
    probs: Optional[list] = None,
) -> Optional[object]:
    """
    UMAP ile 2D gorsellestirme olustur (t-SNE'den hizli).

    Returns:
        plotly.Figure veya None
    """
    if not HAS_PLOTLY:
        return None

    if embeddings is None:
        if get_pool_size() < MIN_POINTS_FOR_VIZ:
            return None
        embeddings = np.array(_embedding_pool["embeddings"])
        labels = _embedding_pool["labels"]
        filenames = _embedding_pool["filenames"]
        probs = _embedding_pool["probs"]

    n = len(embeddings)
    if n < MIN_POINTS_FOR_VIZ:
        return None

    try:
        from umap import UMAP
    except ImportError:
        # umap-learn yuklu degil — t-SNE'ye geri don
        return generate_tsne_plot(embeddings, labels, filenames, probs)

    n_neighbors = min(15, max(2, n - 1))
    reducer = UMAP(n_components=2, n_neighbors=n_neighbors,
                   min_dist=0.3, random_state=42)
    coords = reducer.fit_transform(embeddings)

    return _create_scatter(coords, labels, filenames, probs, "UMAP")


def _create_scatter(coords, labels, filenames, probs, method_name):
    """Ortak Plotly scatter olusturucu."""
    colors = ["#EF4444" if l == "FAKE" else "#22C55E" for l in labels]

    hover_texts = []
    for i in range(len(labels)):
        fn = filenames[i] if filenames and i < len(filenames) else f"#{i}"
        fp = probs[i] if probs and i < len(probs) else 0
        hover_texts.append(f"{fn}<br>{labels[i]} (fake={fp:.3f})")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=coords[:, 0], y=coords[:, 1],
        mode="markers",
        marker=dict(
            color=colors, size=8, opacity=0.85,
            line=dict(width=0.5, color="white"),
        ),
        text=hover_texts,
        hoverinfo="text",
        name="Analizler",
    ))

    fig.update_layout(
        **PLOT_LAYOUT,
        title=f"{method_name} Embedding Space ({len(labels)} nokta)",
        height=450,
        xaxis=dict(title=f"{method_name}-1", gridcolor="#1e293b"),
        yaxis=dict(title=f"{method_name}-2", gridcolor="#1e293b"),
        showlegend=False,
    )

    # Legend proxy: REAL + FAKE isaretleri
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(color="#22C55E", size=10),
        name="REAL", showlegend=True,
    ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(color="#EF4444", size=10),
        name="FAKE", showlegend=True,
    ))
    fig.update_layout(showlegend=True, legend=dict(
        bgcolor="rgba(0,0,0,0.4)", font=dict(color="#e6edf3"),
    ))

    return fig
