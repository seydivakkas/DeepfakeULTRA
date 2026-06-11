"""
Cross-Dataset Benchmark — Harici veri setleri uzerinden model generalizasyon testi.

Otomatik kesif → paralel test → karsilastirmali tablo + radar chart + ROC overlay.

Kullanim:
    from evaluation.cross_dataset_benchmark import CrossDatasetBenchmark
    bench = CrossDatasetBenchmark()
    results = bench.run_full_benchmark(max_samples=200)
"""

import os
import random
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Callable, Tuple

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

try:
    from sklearn.metrics import (
        roc_auc_score, accuracy_score, f1_score,
        precision_score, recall_score, roc_curve,
    )
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# Plotly ortak tema
PLOT_LAYOUT = dict(
    paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
    font=dict(color="#e6edf3", family="Inter"),
    margin=dict(l=50, r=30, t=50, b=50),
)

# Dataset renk paleti (10 renk)
DATASET_COLORS = [
    "#06B6D4",  # cyan
    "#22C55E",  # yesil
    "#EF4444",  # kirmizi
    "#F59E0B",  # turuncu
    "#8B5CF6",  # mor
    "#EC4899",  # pembe
    "#14B8A6",  # teal
    "#F97316",  # koyu turuncu
    "#6366F1",  # indigo
    "#84CC16",  # lime
]

# Bilinen harici dataset metadata
KNOWN_DATASETS = {
    "celeb_df_v2": {"label": "Celeb-DF v2", "icon": "🎬", "type": "face_swap"},
    "dfdc": {"label": "DFDC", "icon": "📹", "type": "mixed"},
    "deepfake20k": {"label": "Deepfake 20K", "icon": "🖼️", "type": "mixed"},
    "deepfakeface": {"label": "DeepFakeFace", "icon": "👤", "type": "face_swap"},
    "faceforensics": {"label": "FaceForensics++", "icon": "🔬", "type": "multi_method"},
    "jury_test": {"label": "Jury Test", "icon": "⚖️", "type": "curated"},
    "test_split": {"label": "Eğitim Test Split", "icon": "📊", "type": "in_distribution"},
}


class CrossDatasetBenchmark:
    """Harici veri setleri uzerinden cross-dataset benchmark testi."""

    SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(self, base_dir: str = None):
        if base_dir is None:
            base_dir = str(Path(__file__).parent.parent)
        self.base_dir = Path(base_dir)
        self.dataset_dir = self.base_dir / "dataset"
        self.results: Dict[str, dict] = {}

    def discover_datasets(self) -> List[Dict]:
        """
        Otomatik olarak test edilebilir tum datasetleri kesfet.

        Returns:
            List[Dict]: Her biri {name, path, label, icon, real_count, fake_count} iceren liste.
        """
        datasets = []

        # 1. external_tests/ altindaki tum alt klasorler
        ext_dir = self.dataset_dir / "external_tests"
        if ext_dir.exists():
            for d in sorted(ext_dir.iterdir()):
                if d.is_dir() and (d / "real").exists() and (d / "fake").exists():
                    real_count = self._count_images(d / "real")
                    fake_count = self._count_images(d / "fake")
                    # Minimum 10 gorsel gereksiz datasetleri atla
                    if real_count + fake_count < 10:
                        continue
                    meta = KNOWN_DATASETS.get(d.name, {})
                    datasets.append({
                        "name": d.name,
                        "path": str(d),
                        "label": meta.get("label", d.name),
                        "icon": meta.get("icon", "📁"),
                        "type": meta.get("type", "unknown"),
                        "real_count": real_count,
                        "fake_count": fake_count,
                    })

        # 2. jury_test/
        jury_dir = self.dataset_dir / "jury_test"
        if jury_dir.exists() and (jury_dir / "real").exists():
            real_count = self._count_images(jury_dir / "real")
            fake_count = self._count_images(jury_dir / "fake") if (jury_dir / "fake").exists() else 0
            if real_count + fake_count >= 10:
                meta = KNOWN_DATASETS.get("jury_test", {})
                datasets.append({
                    "name": "jury_test",
                    "path": str(jury_dir),
                    "label": meta.get("label", "Jury Test"),
                    "icon": meta.get("icon", "⚖️"),
                    "type": meta.get("type", "curated"),
                    "real_count": real_count,
                    "fake_count": fake_count,
                })

        # 3. faces_split/test/ (egitim test split)
        test_split = self.dataset_dir / "faces_split_v2" / "test"
        if test_split.exists() and (test_split / "real").exists():
            real_count = self._count_images(test_split / "real")
            fake_count = self._count_images(test_split / "fake") if (test_split / "fake").exists() else 0
            if real_count + fake_count >= 10:
                meta = KNOWN_DATASETS.get("test_split", {})
                datasets.append({
                    "name": "test_split",
                    "path": str(test_split),
                    "label": meta.get("label", "Test Split"),
                    "icon": meta.get("icon", "📊"),
                    "type": meta.get("type", "in_distribution"),
                    "real_count": real_count,
                    "fake_count": fake_count,
                })

        return datasets

    def run_single_dataset(
        self,
        dataset_path: str,
        dataset_name: str,
        max_samples: int = 200,
        progress_fn: Optional[Callable] = None,
    ) -> Dict:
        """
        Tek bir veri seti uzerinde benchmark testi calistir.
        Her zaman 50/50 dengeli ornekleme yapar.

        Args:
            dataset_path: real/ ve fake/ alt klasorleri iceren dizin yolu.
            dataset_name: Gosterim adi.
            max_samples: Sinif basina maksimum ornek sayisi.
            progress_fn: Ilerleme callback (mevcut, toplam, mesaj).

        Returns:
            Dict: Metrikler ve ham veriler.
        """
        from inference.predictor import get_predictor
        predictor = get_predictor()

        # Oncelikle her iki sinifin dosyalarini topla
        class_files = {}
        for label_name in ["real", "fake"]:
            label_dir = os.path.join(dataset_path, label_name)
            if not os.path.exists(label_dir):
                class_files[label_name] = []
                continue
            class_files[label_name] = [
                os.path.join(label_dir, f)
                for f in os.listdir(label_dir)
                if os.path.splitext(f)[1].lower() in self.SUPPORTED_FORMATS
            ]

        real_count = len(class_files.get("real", []))
        fake_count = len(class_files.get("fake", []))

        # 50/50 dengeli ornekleme: min(real, fake, max_samples)
        balanced_n = min(real_count, fake_count, max_samples)

        if balanced_n < 2:
            return {
                "name": dataset_name,
                "error": f"Yetersiz veri (REAL={real_count}, FAKE={fake_count})",
                "count": 0,
            }

        labels = []
        preds = []
        probs = []
        errors = 0
        total_to_process = balanced_n * 2  # REAL + FAKE

        for label_name, label_id in [("real", 0), ("fake", 1)]:
            files = class_files.get(label_name, [])
            if not files:
                continue

            # Rastgele ornekle (tekrarlanabilir, dengeli)
            random.seed(42)
            selected = random.sample(files, balanced_n)

            for i, fpath in enumerate(selected):
                try:
                    result = predictor.predict(fpath)
                    labels.append(label_id)
                    preds.append(1 if result["label"] == "FAKE" else 0)
                    probs.append(result["fake_prob"])
                except Exception:
                    errors += 1
                    continue

                if progress_fn and (i + 1) % 10 == 0:
                    progress_fn(i + 1, balanced_n, f"{dataset_name}: {label_name}")

        if len(labels) < 4:
            return {
                "name": dataset_name,
                "error": f"Yetersiz veri ({len(labels)} gorsel)",
                "count": len(labels),
            }

        # Metrikleri hesapla
        metrics = self._compute_metrics(labels, preds, probs)
        metrics["name"] = dataset_name
        metrics["count"] = len(labels)
        metrics["balanced_n"] = balanced_n
        metrics["n_real"] = sum(1 for l in labels if l == 0)
        metrics["n_fake"] = sum(1 for l in labels if l == 1)
        metrics["errors"] = errors
        metrics["labels"] = labels
        metrics["preds"] = preds
        metrics["probs"] = probs

        return metrics

    def run_full_benchmark(
        self,
        max_samples: int = 200,
        progress_fn: Optional[Callable] = None,
    ) -> Dict[str, dict]:
        """
        Tum harici veri setleri uzerinden benchmark.

        Args:
            max_samples: Her datasetten sinif basina maksimum ornek.
            progress_fn: (dataset_idx, total_datasets, dataset_name, pct) callback.

        Returns:
            Dict[str, dict]: Dataset adi → metrik sozlugu.
        """
        datasets = self.discover_datasets()
        if not datasets:
            return {"_error": "Harici veri seti bulunamadi."}

        self.results = {}
        start_time = time.time()

        for idx, ds in enumerate(datasets):
            ds_start = time.time()

            if progress_fn:
                progress_fn(idx, len(datasets), ds["label"], 0)

            result = self.run_single_dataset(
                dataset_path=ds["path"],
                dataset_name=ds["label"],
                max_samples=max_samples,
            )

            result["icon"] = ds["icon"]
            result["type"] = ds["type"]
            result["real_count"] = ds["real_count"]
            result["fake_count"] = ds["fake_count"]
            result["elapsed"] = time.time() - ds_start

            self.results[ds["name"]] = result

            if progress_fn:
                progress_fn(idx + 1, len(datasets), ds["label"], 100)

        self.results["_meta"] = {
            "total_time": time.time() - start_time,
            "max_samples": max_samples,
            "n_datasets": len(datasets),
        }

        return self.results

    def generate_comparison_table(self, results: Dict = None) -> str:
        """Karsilastirmali metrik tablosu (Markdown)."""
        results = results or self.results
        if not results:
            return "> Benchmark henuz calistirilmadi."

        rows = []
        valid_results = {
            k: v for k, v in results.items()
            if k != "_meta" and "error" not in v
        }

        if not valid_results:
            return "> Gecerli sonuc bulunamadi."

        # Siralamalar (en iyi → AUC'ye gore)
        sorted_ds = sorted(
            valid_results.items(),
            key=lambda x: x[1].get("auc", 0),
            reverse=True,
        )

        for rank, (name, r) in enumerate(sorted_ds, 1):
            icon = r.get("icon", "📁")
            ds_name = r.get("name", name)
            auc = r.get("auc", 0)
            acc = r.get("accuracy", 0)
            f1 = r.get("f1", 0)
            prec = r.get("precision", 0)
            rec = r.get("recall", 0)
            eer = r.get("eer", 0)
            count = r.get("count", 0)

            # Performans badge
            if auc >= 0.95:
                badge = "🟢"
            elif auc >= 0.85:
                badge = "🟡"
            elif auc >= 0.70:
                badge = "🟠"
            else:
                badge = "🔴"

            n_real = r.get("n_real", count // 2)
            n_fake = r.get("n_fake", count // 2)

            rows.append(
                f"| {rank} | {badge} {icon} **{ds_name}** | "
                f"{auc:.4f} | {acc*100:.1f}% | {f1:.4f} | "
                f"{prec:.4f} | {rec:.4f} | {eer:.4f} | {n_real}+{n_fake}={count} |"
            )

        # Ortalama hesapla
        aucs = [r.get("auc", 0) for r in valid_results.values()]
        accs = [r.get("accuracy", 0) for r in valid_results.values()]
        f1s = [r.get("f1", 0) for r in valid_results.values()]

        avg_auc = np.mean(aucs) if aucs else 0
        avg_acc = np.mean(accs) if accs else 0
        avg_f1 = np.mean(f1s) if f1s else 0
        std_auc = np.std(aucs) if aucs else 0

        # Generalizasyon skoru
        gen_score = avg_auc * (1 - std_auc)

        header = (
            "### 📊 Cross-Dataset Benchmark Sonuçları\n\n"
            "> ⚖️ **Dengeli örnekleme:** Her dataset 50/50 REAL/FAKE oranıyla test edildi.\n\n"
            "| # | Dataset | AUC | Accuracy | F1 | Precision | Recall | EER | R+F |"
            "\n|---|---------|-----|----------|----|-----------|--------|-----|------|\n"
        )
        footer = (
            f"| | **Ortalama** | **{avg_auc:.4f}** | "
            f"**{avg_acc*100:.1f}%** | **{avg_f1:.4f}** | | | | |\n"
            f"| | **Std** | ±{std_auc:.4f} | "
            f"±{np.std(accs)*100:.1f}% | ±{np.std(f1s):.4f} | | | | |\n\n"
            f"> **Generalizasyon Skoru:** {gen_score:.4f} "
            f"(AUC_avg × (1 - AUC_std) — yüksek = tutarlı performans)\n"
        )

        meta = results.get("_meta", {})
        if meta:
            elapsed = meta.get("total_time", 0)
            footer += (
                f"\n> ⏱️ Toplam süre: {elapsed:.0f}s | "
                f"Sınıf başına max: {meta.get('max_samples', '?')} örnek\n"
            )

        return header + "\n".join(rows) + "\n" + footer

    def generate_radar_chart(self, results: Dict = None) -> Optional[object]:
        """Radar/spider chart — her dataset icin AUC, Acc, F1, Precision, Recall."""
        if not HAS_PLOTLY:
            return None

        results = results or self.results
        valid = {
            k: v for k, v in results.items()
            if k != "_meta" and "error" not in v
        }
        if not valid:
            return None

        categories = ["AUC", "Accuracy", "F1", "Precision", "Recall"]

        fig = go.Figure()

        for idx, (name, r) in enumerate(valid.items()):
            values = [
                r.get("auc", 0),
                r.get("accuracy", 0),
                r.get("f1", 0),
                r.get("precision", 0),
                r.get("recall", 0),
            ]
            # Poligonu kapatmak icin ilk degeri tekrarla
            values_closed = values + [values[0]]
            cats_closed = categories + [categories[0]]

            color = DATASET_COLORS[idx % len(DATASET_COLORS)]
            ds_name = r.get("name", name)

            fig.add_trace(go.Scatterpolar(
                r=values_closed,
                theta=cats_closed,
                fill="toself",
                fillcolor=f"rgba({int(color[1:3], 16)},{int(color[3:5], 16)},{int(color[5:7], 16)},0.08)",
                line=dict(color=color, width=2),
                name=f"{r.get('icon', '')} {ds_name}",
                hovertemplate="%{theta}: %{r:.4f}<extra>%{fullData.name}</extra>",
            ))

        fig.update_layout(
            **PLOT_LAYOUT,
            title="Cross-Dataset Performans Karşılaştırması",
            height=420,
            polar=dict(
                bgcolor="#161b22",
                radialaxis=dict(
                    visible=True, range=[0, 1],
                    gridcolor="#1e293b", linecolor="#1e293b",
                    tickfont=dict(size=10, color="#94A3B8"),
                ),
                angularaxis=dict(
                    gridcolor="#1e293b", linecolor="#1e293b",
                    tickfont=dict(size=11, color="#e6edf3"),
                ),
            ),
            legend=dict(
                bgcolor="rgba(0,0,0,0.4)",
                font=dict(size=10),
                orientation="h",
                yanchor="bottom", y=-0.25,
                xanchor="center", x=0.5,
            ),
        )

        return fig

    def generate_roc_overlay(self, results: Dict = None) -> Optional[object]:
        """Tum datasetlerin ROC egrisini tek grafik uzerinde ciz."""
        if not HAS_PLOTLY or not HAS_SKLEARN:
            return None

        results = results or self.results
        valid = {
            k: v for k, v in results.items()
            if k != "_meta" and "error" not in v and "labels" in v
        }
        if not valid:
            return None

        fig = go.Figure()

        for idx, (name, r) in enumerate(valid.items()):
            labels = r["labels"]
            probs_list = r["probs"]

            if len(set(labels)) < 2:
                continue

            fpr, tpr, _ = roc_curve(labels, probs_list)
            auc_val = roc_auc_score(labels, probs_list)
            color = DATASET_COLORS[idx % len(DATASET_COLORS)]
            ds_name = r.get("name", name)

            fig.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode="lines",
                line=dict(color=color, width=2),
                name=f"{r.get('icon', '')} {ds_name} (AUC={auc_val:.3f})",
                hovertemplate="FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra>%{fullData.name}</extra>",
            ))

        # Rastgele sinir
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode="lines",
            line=dict(color="#94A3B8", width=1, dash="dash"),
            name="Random (AUC=0.5)",
            showlegend=True,
        ))

        fig.update_layout(
            **PLOT_LAYOUT,
            title="Cross-Dataset ROC Karşılaştırması",
            height=400,
            xaxis=dict(
                title="False Positive Rate",
                range=[0, 1], gridcolor="#1e293b",
            ),
            yaxis=dict(
                title="True Positive Rate",
                range=[0, 1.02], gridcolor="#1e293b",
            ),
            legend=dict(
                bgcolor="rgba(0,0,0,0.5)",
                font=dict(size=9),
                yanchor="bottom", y=0.02,
                xanchor="right", x=0.98,
            ),
        )

        return fig

    def generate_summary_report(self, results: Dict = None) -> str:
        """Benchmark ozet raporu — en guclu/zayif dataset, oneriler."""
        results = results or self.results
        valid = {
            k: v for k, v in results.items()
            if k != "_meta" and "error" not in v
        }

        if not valid:
            return "> Gecerli sonuc bulunamadi."

        # En iyi / en kotu
        by_auc = sorted(valid.items(), key=lambda x: x[1].get("auc", 0), reverse=True)
        best_name, best = by_auc[0]
        worst_name, worst = by_auc[-1]

        # Ortalamalar
        aucs = [r.get("auc", 0) for r in valid.values()]
        avg_auc = np.mean(aucs)
        std_auc = np.std(aucs)
        gen_score = avg_auc * (1 - std_auc)

        # Performans degerlendirmesi
        if gen_score >= 0.90:
            verdict = "🟢 **Mükemmel** — Model tüm datasetlerde yüksek ve tutarlı performans gösteriyor."
        elif gen_score >= 0.80:
            verdict = "🟡 **İyi** — Genel performans güçlü, bazı datasetlerde iyileştirme potansiyeli var."
        elif gen_score >= 0.65:
            verdict = "🟠 **Orta** — Bazı datasetlerde belirgin performans düşüşü mevcut."
        else:
            verdict = "🔴 **Zayıf** — Model generalizasyonu yetersiz, ek eğitim gerekli."

        # In-distribution vs out-of-distribution karsilastirmasi
        in_dist = [r for r in valid.values() if r.get("type") == "in_distribution"]
        out_dist = [r for r in valid.values() if r.get("type") != "in_distribution"]

        id_ood_note = ""
        if in_dist and out_dist:
            id_auc = np.mean([r.get("auc", 0) for r in in_dist])
            ood_auc = np.mean([r.get("auc", 0) for r in out_dist])
            gap = id_auc - ood_auc
            if gap > 0.05:
                id_ood_note = (
                    f"\n\n> ⚠️ **In-Distribution / Out-of-Distribution Farkı:** "
                    f"ID AUC={id_auc:.4f} vs OOD AUC={ood_auc:.4f} (Δ={gap:.4f}) — "
                    f"{'Overfitting riski var!' if gap > 0.10 else 'Kabul edilebilir fark.'}"
                )
            else:
                id_ood_note = (
                    f"\n\n> ✅ **ID/OOD Tutarlılık:** "
                    f"ID AUC={id_auc:.4f} ≈ OOD AUC={ood_auc:.4f} — Generalizasyon güçlü."
                )

        report = (
            "### 📋 Benchmark Özet Raporu\n\n"
            f"**Generalizasyon Skoru:** {gen_score:.4f}\n\n"
            f"{verdict}\n\n"
            f"| Metrik | Değer |\n|---|---|\n"
            f"| Ortalama AUC | {avg_auc:.4f} |\n"
            f"| AUC Std | ±{std_auc:.4f} |\n"
            f"| Test Edilen Dataset | {len(valid)} |\n"
            f"| 🏆 En Güçlü | {best.get('icon', '')} {best.get('name', best_name)} (AUC={best.get('auc', 0):.4f}) |\n"
            f"| 📉 En Zayıf | {worst.get('icon', '')} {worst.get('name', worst_name)} (AUC={worst.get('auc', 0):.4f}) |\n"
            f"{id_ood_note}"
        )

        # Hata olan datasetler
        errors = {k: v for k, v in results.items() if k != "_meta" and "error" in v}
        if errors:
            report += "\n\n**⚠️ Atlanan Datasetler:**\n"
            for k, v in errors.items():
                report += f"- {k}: {v['error']}\n"

        return report

    # ─── Yardımcılar ───

    def _count_images(self, directory: Path) -> int:
        """Bir dizindeki gorsel dosya sayisini say (recursive)."""
        count = 0
        if not directory.exists():
            return 0
        for f in directory.rglob("*"):
            if f.is_file() and f.suffix.lower() in self.SUPPORTED_FORMATS:
                count += 1
        return count

    def _compute_metrics(self, labels: list, preds: list, probs: list) -> Dict:
        """Metrik hesapla."""
        if not HAS_SKLEARN or len(labels) < 4:
            return {}

        metrics = {
            "accuracy": float(accuracy_score(labels, preds)),
            "f1": float(f1_score(labels, preds, zero_division=0)),
            "precision": float(precision_score(labels, preds, zero_division=0)),
            "recall": float(recall_score(labels, preds, zero_division=0)),
        }

        # AUC
        try:
            metrics["auc"] = float(roc_auc_score(labels, probs))
        except ValueError:
            metrics["auc"] = 0.5

        # EER
        try:
            fpr, tpr, _ = roc_curve(labels, probs)
            fnr = 1 - tpr
            eer_idx = np.argmin(np.abs(fpr - fnr))
            metrics["eer"] = float(fpr[eer_idx])
        except Exception:
            metrics["eer"] = 0.0

        return metrics


if __name__ == "__main__":
    bench = CrossDatasetBenchmark()
    datasets = bench.discover_datasets()
    print(f"Bulunan veri setleri: {len(datasets)}")
    for ds in datasets:
        print(f"  {ds['icon']} {ds['label']}: {ds['real_count']} REAL + {ds['fake_count']} FAKE")
