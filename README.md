<div align="center">

# DeepfakeULTRA

### Reliability-Aware Multi-Evidence Deepfake Forensics

![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-12.x-76B900?style=flat-square&logo=nvidia&logoColor=white)
![Val AUC](https://img.shields.io/badge/Val%20AUC-0.9839-2EA44F?style=flat-square)
![Cross Dataset](https://img.shields.io/badge/Cross--Dataset%20AUC-0.7527-0A66C2?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-2EA44F?style=flat-square)

**A forensic deepfake detection system that fuses spatial appearance, frequency-domain evidence and facial geometry, then evaluates the model beyond its training distribution.**

**0.9839 validation AUC** · **0.7527 mean cross-dataset AUC** · **5 external datasets** · **~200 ms GPU inference** · **~15M parameters**

`Deepfake Detection` · `Computer Vision` · `XAI` · `Domain Generalization` · `Frequency Analysis` · `Transformers`

</div>

---

## Research question

Deepfake detectors often look strong on an internal validation split and then degrade sharply when the manipulation method, compression pipeline or source domain changes.

DeepfakeULTRA is built around a stricter question:

> **Can multiple forensic evidence streams improve detection while making failure modes, domain shift and model uncertainty inspectable?**

The project therefore treats cross-dataset evaluation and failure analysis as first-class engineering requirements rather than optional benchmark additions.

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

## Evaluation snapshot

| Evaluation | Result |
|---|---:|
| Internal ROC-AUC | **0.9839** |
| Internal accuracy | **93.1%** |
| Internal F1 | **0.931** |
| Mean external AUC | **0.7527** |
| External datasets | **5** |
| Approx. GPU inference | **~200 ms / image** |

### Cross-dataset results

| Dataset | Final AUC |
|---|---:|
| Deepfake20K | **0.956** |
| DFDC | **0.821** |
| DeepfakeFace | **0.777** |
| CelebDF v2 | **0.708** |
| FaceForensics++ | **0.500** |

The FaceForensics++ result is intentionally visible here: the project documents **where the current system does not generalize**, rather than hiding a random-level failure behind an average score.

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
Combined Model Selection
```

Model selection considers both internal validation performance and external-domain performance, reducing the incentive to optimize only for the familiar validation distribution.

---

## Product surfaces

- **Single-image forensic analysis**
- **Robustness testing**
- **Analytics dashboard**
- **Model evaluation / profiling**
- **Persistent analysis history**
- **Craniofacial analysis module**
- **LLM-assisted result interpretation**
- **FastAPI inference layer**
- **Gradio interface**

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

## Documentation

The root README is intentionally concise and research-portfolio oriented. The original full system documentation is preserved here:

### **[Full Technical Documentation →](docs/README_FULL.md)**

The full document contains architecture details, training pipeline, benchmarks, UI walkthroughs, domain-generalization experiments, API usage, robustness tooling, limitations and roadmap.

---

## Known limitation

The current model performs at approximately random level on the documented FaceForensics++ reenactment benchmark (`AUC ≈ 0.50`). This remains an open failure mode and is part of the project roadmap rather than an omitted result.

---

<div align="center">

**Detect · explain · stress-test · disclose failure modes**

[GitHub Profile](https://github.com/seydivakkas) · [Full Documentation](docs/README_FULL.md)

</div>
