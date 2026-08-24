# Evidence Index

This directory is the human-readable index for DeepfakeULTRA engineering evidence. Numeric claims in the portfolio README should be traceable to repository artifacts rather than copied from memory or screenshots.

## Current evaluation artifacts

- `../../evaluation/metrics.json` — internal evaluation, calibration and latency snapshot.
- `../../evaluation/external/deepfake20k/metrics.json` — Deepfake20K external evaluation.
- `../../evaluation/external/dfdc/metrics.json` — DFDC external evaluation.
- `../../evaluation/external/deepfakeface/metrics.json` — DeepfakeFace external evaluation.
- `../../evaluation/external/celeb_df_v2/metrics.json` — Celeb-DF v2 external evaluation.
- `../../evaluation/external/faceforensics/metrics.json` — FaceForensics++ external evaluation.

Each external evaluation directory also contains benchmark reports and/or ROC/confusion-matrix artifacts where available.

## Claim discipline

1. Prefer machine-readable metric artifacts over prose-only values.
2. Keep internal and external evaluation clearly separated.
3. Do not hide failed domains behind an average score.
4. Treat latency as hardware/configuration-specific evidence, not a universal guarantee.
5. Recompute portfolio summary values when evaluation artifacts change.

## Current artifact-backed snapshot

At the time this index was created, the repository artifacts report:

- Internal ROC-AUC: **0.9820**
- Deepfake20K ROC-AUC: **0.9617**
- DFDC ROC-AUC: **0.8057**
- DeepfakeFace ROC-AUC: **0.7614**
- Celeb-DF v2 ROC-AUC: **0.6670**
- FaceForensics++ ROC-AUC: **0.5068**
- Unweighted external mean ROC-AUC: **0.7405**

See `../../KNOWN_LIMITATIONS.md` for interpretation boundaries.
