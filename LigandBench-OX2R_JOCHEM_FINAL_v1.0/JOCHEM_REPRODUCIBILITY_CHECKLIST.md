# Journal of Cheminformatics reproducibility checklist

This checklist maps the repository to the journal's reproducibility expectations.

| Requirement | Repository evidence |
|---|---|
| Public source code | Full Python source under `ligandbench/`, `scripts/`, and root scripts |
| Complete analysis workflow | `scripts/reproduce_main.py`, `scripts/benchmark_frozen.py`, `scripts/similarity_stratified.py`, `scripts/cluster_split.py` |
| Frozen data used by model | `data/ox2r_curated_6415.csv` |
| Data provenance | `DATA_PROVENANCE.md`, `DATA_LICENSE.md`, `data/README.md` |
| Source-release identification | ChEMBL 34; DOI 10.6019/CHEMBL.database.34 |
| Chemical-space similarity analysis | `scripts/similarity_stratified.py` |
| Cross-class proximal analysis | `scripts/similarity_stratified.py` |
| 120-cluster validation | `scripts/cluster_split.py` |
| Exact software versions | `requirements.txt`, `environment.yml` |
| Fixed random seed | `config/reproducibility.yaml` and scripts |
| Installation instructions | `README.md` |
| Automated repository tests | `tests/test_reproducibility.py` |
| Continuous integration | `.github/workflows/ci.yml` |
| Software licence | `LICENSE` |
| Data licence/provenance distinction | `DATA_LICENSE.md` |
| Citation metadata | `CITATION.cff` |
| Permanent archival guidance | `README.md`, `REPRODUCIBILITY.md` |
| No compiled artefacts | No `__pycache__` or `.pyc` files in release package |

## Important manuscript-side action

After uploading this repository, replace any placeholder repository wording in the manuscript with the actual public GitHub URL. If a Zenodo/Figshare DOI is created, add that DOI only after it has been minted.

The manuscript's Data Availability Statement should point to the public repository rather than stating that all data are available only in the supplementary workbook.
