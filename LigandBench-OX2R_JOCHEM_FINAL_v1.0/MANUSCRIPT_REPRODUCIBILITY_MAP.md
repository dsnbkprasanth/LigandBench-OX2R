# Manuscript-to-repository reproducibility map

| Manuscript analysis | Repository location |
|---|---|
| ChEMBL OX2R source/provenance | `DATA_PROVENANCE.md`, `data/chembl34_ox2r_query.sql` |
| Frozen curated model input (6,415 compounds) | `data/ox2r_curated_6415.csv` |
| Primary 30-model benchmark | `scripts/benchmark_frozen.py`, `ox2_agvsan_v2/` |
| Scaffold-grouped split and grouped CV | `ligandbench/splitting.py` |
| Molecular representations | `ligandbench/features.py` |
| Classifier benchmark | `ligandbench/benchmark.py` |
| Y-randomization | `ligandbench/validation.py`, `reviewer_extra.py`, `ox2_extra/yrand.json` |
| SHAP interpretation | `ligandbench/validation.py`, `ligandbench/reporting.py`, `ox2_agvsan_v2/` |
| Applicability domain | `ligandbench/applicability.py`, `ox2_agvsan_v2/` |
| Similarity-stratified analysis | `scripts/similarity_stratified.py` |
| Cross-class-proximal subset | `scripts/similarity_stratified.py` |
| 120-cluster chemical-space validation | `scripts/cluster_split.py` |
| Exact analysis configuration | `config/reproducibility.yaml` |
| Exact software environment | `requirements.txt`, `environment.yml` |
| Automated repository verification | `tests/test_reproducibility.py`, `.github/workflows/ci.yml` |
