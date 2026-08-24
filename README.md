<div align="center">

# DeepfakeULTRA

### Reliability-Aware Multi-Evidence Deepfake Forensics

![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-12.x-76B900?style=flat-square&logo=nvidia&logoColor=white)
[![CI](https://github.com/seydivakkas/DeepfakeULTRA/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/seydivakkas/DeepfakeULTRA/actions/workflows/ci.yml)
![License](https://img.shields.io/badge/License-All%20Rights%20Reserved-red?style=flat-square)

**A forensic deepfake detection system that fuses spatial appearance, frequency-domain evidence and facial geometry, then evaluates the model beyond its training distribution.**

**0.9820 internal ROC-AUC** · **0.7405 mean external ROC-AUC** · **5 external datasets** · **artifact-backed evaluation**

`Deepfake Detection` · `Computer Vision` · `XAI` · `Domain Generalization` · `Frequency Analysis` · `Transformers`

</div>

---

## Research question

Deepfake detectors often look strong on an internal validation split and then degrade sharply when the manipulation method, compression pipeline or source domain changes.

DeepfakeULTRA is built around a stricter question:

> **Can multiple forensic evidence streams improve detection while making failure modes, domain shift and model uncertainty inspectable?**

The project therefore treats cross-dataset evaluation and failure analysis as first-class engineering requirements rather than optional benchmark additions.

---

## Minimum reproducible run

```bash
git clone https://github.com/seydivakkas/DeepfakeULTRA.git
cd DeepfakeULTRA
python -m venv .venv
# activate .venv for your shell
pip install -r requirements.txt
python main.py --test-random
python app.py
```

Model weights and large datasets are intentionally not assumed to be present in a fresh clone. Full model-backed inference requires the documented model artifacts; the smoke path above validates the software surface first.

---

## Multi-evidence architecture

```text
Input Face
   ├── RGB Evidence ────────────────┐
   │   MobileNetV3                  │
   │                                │
   ├── Frequency Evidence ──────────┤
   │   DWT + DCT + Phase            │
   │   MobileNetV3                  ├── Cross-Branch Transformer
   │                                │          ↓
   └── Facial Geometry ─────────────┤      Classifier
       468 landmarks + MLP          │          ↓
                                    ┘   REAL / FAKE / UNCERTAIN
```

The frequency branch combines multiple wavelet families, DCT bands and phase information; the geometry branch adds structural facial evidence instead of relying only on RGB texture cues.

---

## Artifact-backed evaluation snapshot

The values below are taken from the machine-readable files currently committed under `evaluation/`.

| Evaluation | ROC-AUC |
|---|---:|
| Internal evaluation | **0.9820** |
| Deepfake20K | **0.9617** |
| DFDC | **0.8057** |
| DeepfakeFace | **0.7614** |
| Celeb-DF v2 | **0.6670** |
| FaceForensics++ | **0.5068** |
| External mean | **0.7405** |

The FaceForensics++ result is intentionally visible: the current repository artifacts show a near-random failure on this domain. Historical experiment values may appear in legacy technical documentation; for the current portfolio summary, machine-readable evaluation artifacts are authoritative.

### Evidence

- [Evidence index](docs/evidence/README.md)
- [Internal metrics](evaluation/metrics.json)
- [External evaluation folders](evaluation/external/)
- [Known limitations](KNOWN_LIMITATIONS.md)

---

## Reliability & forensic tooling

| Capability | Purpose |
|---|---|
| Domain augmentation | Reduce dependence on source-specific JPEG, resolution and color artifacts |
| Test-time augmentation | Check prediction stability under controlled transformations |
| Branch knockout | Measure contribution of RGB, frequency and geometry evidence |
| Adversarial tests | FGSM / PGD / CW stress testing |
| XAI | GradCAM++, EigenCAM, FastCAM and Guided LIME |
| Forensic analysis | ELA and noise-oriented supporting evidence |
| Calibration | Threshold and confidence analysis for non-binary decision handling |
| Error analysis | Make failed domains and manipulation families explicit |

---

## Domain-generalization strategy

```text
Internal Model
    ↓
Domain Augmentation
  JPEG · Resize · Noise · Gamma · Color Shift
    ↓
Curriculum Fine-Tuning
  Frozen → Unfrozen
    ↓
Internal Validation
        +
External Benchmark Suite
    ↓
Evidence-Aware Model Selection
```

Model selection is intended to consider both familiar validation performance and external-domain behavior, reducing the incentive to optimize only for the training distribution.

---

## Product surfaces

- Single-image forensic analysis
- Robustness testing
- Analytics dashboard
- Model evaluation / profiling
- Persistent analysis history
- Craniofacial analysis module
- LLM-assisted result interpretation
- FastAPI inference layer
- Gradio interface

---

## Technology stack

**Machine Learning**  
`PyTorch` · `MobileNetV3` · `Transformer Encoder` · `Mixed Precision` · `CUDA`

**Forensics & Vision**  
`OpenCV` · `MediaPipe` · `DWT` · `DCT` · `Phase Spectrum` · `ELA`

**Explainability & Evaluation**  
`GradCAM++` · `EigenCAM` · `LIME` · `ROC-AUC` · `Calibration` · `Cross-Dataset Benchmarking`

**Application**  
`Gradio` · `FastAPI` · `SQLite` · `Docker`

---

## Engineering principles

1. **External evaluation before generalization claims**
2. **Failure disclosure before benchmark marketing**
3. **Multiple evidence streams before single-cue dependence**
4. **Calibration / uncertainty before forced confidence**
5. **Explainability as an inspection tool, not proof of correctness**
6. **Reproducible evidence before architectural novelty claims**

---

## Research & documentation

- [Architecture index](docs/architecture/README.md)
- [Evidence index](docs/evidence/README.md)
- [Known limitations](KNOWN_LIMITATIONS.md)
- [Citation metadata](CITATION.cff)
- [Full technical documentation](docs/README_FULL.md)

The repository is source-available under the terms in `LICENSE`; it is **not MIT-licensed**.

---

<div align="center">

**Detect · explain · stress-test · disclose failure modes**

[GitHub Profile](https://github.com/seydivakkas) · [Evidence](docs/evidence/README.md) · [Limitations](KNOWN_LIMITATIONS.md)

</div>
