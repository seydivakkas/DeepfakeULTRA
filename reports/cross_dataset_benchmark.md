# DeepfakeULTRA — Cross-Dataset Benchmark Report

> **Canonical status:** The current portfolio summary must be derived from the machine-readable artifacts under `evaluation/`. Historical experiment snapshots are retained below for engineering traceability, but they are not the authoritative source for current claims.

## Current artifact-backed snapshot

The current repository README and committed evaluation artifacts report:

| Evaluation | ROC-AUC |
|---|---:|
| Internal evaluation | **0.9820** |
| Deepfake20K | **0.9617** |
| DFDC | **0.8057** |
| DeepfakeFace | **0.7614** |
| Celeb-DF v2 | **0.6670** |
| FaceForensics++ | **0.5068** |
| External mean | **0.7405** |

### Interpretation

- Internal performance remains strong, but external-domain behavior varies substantially.
- FaceForensics++ is intentionally kept visible because the current artifact-backed result is close to random discrimination on that domain.
- Portfolio claims should therefore emphasize **cross-dataset evaluation, domain shift and failure disclosure**, not only the best internal score.
- When documentation and historical reports disagree, the committed machine-readable artifacts are authoritative.

### Canonical evidence

- `evaluation/metrics.json`
- `evaluation/external/`
- `docs/evidence/README.md`
- `KNOWN_LIMITATIONS.md`
- root `README.md` artifact-backed evaluation snapshot

---

# Historical V5 snapshot — 2026-05-13

The section below is preserved as a historical experiment record. These values must **not** be used as the current portfolio benchmark when they differ from the canonical artifact-backed snapshot above.

**Model:** DeepfakeULTRA V5 (Dual-MobileNetV3 + Transformer)  
**Training data:** DF40 Dataset  
**Historical report date:** 2026-05-13

## 1. Historical overview

DeepfakeULTRA V5 was evaluated on six distributions:

1. **Internal test (DF40)** — same distribution as training data
2. **Celeb-DF v2** — autoencoder face-swap
3. **FaceForensics++ (c23)** — Deepfakes + Face2Face
4. **DFDC** — Deepfake Detection Challenge
5. **Deepfake-vs-Real-20K** — mixed modern deepfake methods
6. **DeepFakeFace (DFF)** — diffusion-generated content

## 2. Historical comparative table

| Metric | Internal | Celeb-DF v2 | FF++ | DFDC | Deepfake20K | DFF |
|---|---:|---:|---:|---:|---:|---:|
| Images | 59,655 | 1,890 | 750 | 2,000 | 2,000 | 2,000 |
| ROC-AUC | **0.9606** | 0.5439 | 0.5121 | 0.5361 | **0.6231** | 0.4745 |
| EER | 0.1036 | 0.4651 | 0.4920 | 0.4745 | 0.3990 | 0.5205 |
| Accuracy | 0.8990 | 0.5566 | 0.4640 | 0.5315 | **0.6290** | 0.5065 |
| Macro F1 | 0.8989 | 0.5162 | 0.4638 | 0.5294 | **0.6164** | 0.3662 |
| Threshold | 0.4423 | 0.3209 | 0.4076 | 0.2930 | 0.2904 | 0.2564 |
| ECE | 0.2705 | 0.1694 | 0.2911 | 0.1811 | 0.1900 | 0.1909 |

## 3. Historical finding

This older snapshot showed a large generalization gap between the DF40 in-distribution test and external datasets. That conclusion remains useful as engineering history: **single-domain performance is not sufficient evidence of generalization**.

The repository now uses the artifact-backed snapshot at the top of this report for current claims and keeps this historical table only to preserve experiment provenance.
