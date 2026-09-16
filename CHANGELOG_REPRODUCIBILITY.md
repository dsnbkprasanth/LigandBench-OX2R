# Reproducibility release changes

- Added exact pinned software environment (`requirements.txt`, `environment.yml`).
- Added `CITATION.cff`.
- Added explicit separation of software licensing and ChEMBL-derived data provenance (`DATA_LICENSE.md`).
- Added frozen 6,415-compound OX2R model input under `data/`.
- Added reproducible similarity-stratified validation with 2,000 bootstrap resamples.
- Added reproducible cross-class-proximal analysis.
- Added reproducible 120-cluster K-means / five-fold grouped validation.
- Added one-command reproducibility entry point.
- Added backward-compatible `--target-label` CLI argument so the documented manuscript command executes.
- Removed Python bytecode/cache files from the release package.
- Added machine-readable reproducibility configuration.
- Added manuscript-to-repository reproducibility map.
