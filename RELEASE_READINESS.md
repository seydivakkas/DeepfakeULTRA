# Release Readiness

## Recommended policy

Do **not** publish a `v1.0.0` tag until the release contract below is green. DeepfakeULTRA historical documentation uses V5-style nomenclature, so semantic-version naming should be chosen deliberately rather than creating a conflicting version story.

## Gate

- [x] Portfolio README uses current machine-readable evaluation artifacts.
- [x] Known limitations are explicit.
- [x] Academic citation metadata exists (`CITATION.cff`).
- [x] Current license is explicit and root README no longer claims MIT.
- [x] Legacy README is separated from the current source of truth.
- [x] `main` branch contains the current release candidate content.
- [ ] GitHub default branch switched to `main`.
- [ ] `main` CI run confirmed green on the intended release commit.
- [ ] Release commit SHA recorded in release notes.
- [ ] Model artifact availability/distribution policy finalized.
- [ ] Final release tag/version naming reconciled with historical V5 terminology.

## Release notes must include

- artifact-backed internal/external evaluation snapshot,
- FaceForensics++ failure disclosure,
- installation / smoke path,
- model-weight availability,
- license terms,
- citation instructions,
- hardware/context for latency claims.

A release is an evidence snapshot, not a claim that all deepfake domains are solved.
