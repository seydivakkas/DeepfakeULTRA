# Architecture Index

DeepfakeULTRA uses a multi-evidence forensic architecture:

```text
Input face
  ├─ RGB branch ──────────────┐
  ├─ Frequency branch ────────┼─ Cross-Branch Transformer ─ Classifier
  └─ Facial-geometry branch ──┘
```

## Evidence streams

- **RGB:** spatial appearance and blending/texture cues.
- **Frequency:** DWT, DCT and phase-spectrum representations.
- **Geometry:** facial landmark structure represented through an MLP branch.

## System layers

- `core/` — model, training, frequency extraction, calibration and robustness primitives.
- `inference/` — prediction, XAI and analysis composition.
- `api/` — FastAPI service layer.
- `ui/` — application-facing Gradio components.
- `evaluation/` — internal and external machine-readable evaluation artifacts.

For the original detailed architecture walkthrough, see `../README_FULL.md`. For current evidence, see `../evidence/README.md`.
