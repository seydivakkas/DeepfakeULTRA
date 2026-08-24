# Known Limitations

DeepfakeULTRA is an experimental forensic decision-support system. The following limitations are intentionally documented so that repository claims remain auditable.

## 1. Reenactment generalization

The current FaceForensics++ evaluation artifact reports ROC-AUC close to random performance (~0.51). Reenactment-style manipulations remain a documented failure mode and should not be represented as a solved domain.

## 2. Cross-dataset variance

Performance varies materially across external datasets. Strong results on one manipulation family do not imply equivalent performance on unseen generation, reenactment, compression or capture pipelines.

## 3. Single-image emphasis

The primary system is image-oriented. It does not yet provide a validated temporal-consistency model for video-level forensic decisions.

## 4. Distribution shift and OOD inputs

Cartoons, illustrations, extreme compression, unusual face poses, heavy occlusion or image domains outside the training/evaluation contract can invalidate confidence assumptions. OOD filtering reduces risk but does not prove in-distribution validity.

## 5. Explainability boundary

GradCAM, EigenCAM, LIME and related maps are diagnostic inspection tools. They do not prove causal correctness or establish that a prediction is forensically true.

## 6. Calibration scope

Calibration artifacts are tied to specific evaluation conditions. Thresholds and confidence interpretation should be revalidated after material model, preprocessing or deployment-domain changes.

## 7. Not a legal or identity verdict

A model result is not a legal determination, identity-authentication guarantee or standalone forensic certification. High-impact decisions require independent evidence and qualified human review.

## Evidence policy

Repository-level claims should point to files under `evaluation/` or `docs/evidence/`. When a historical README value and a current evaluation artifact differ, the artifact-backed value is authoritative for the current portfolio summary.
