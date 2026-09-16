# LigandBench-OX2R

**Scaffold-aware, ligand-based machine-learning benchmark for separating human OX2R agonists from antagonists.**

This repository contains the scientific analysis code, frozen model input, precomputed outputs, chemical-space robustness analyses, and reproducibility metadata accompanying:

> **Ligand-based machine learning separates orexin receptor type 2 agonists from antagonists: scaffold-aware benchmarking reveals chemical-space limits**

The package is prepared for reproducibility of the computational results reported in the manuscript and is designed to support third-party inspection and rerunning of the main analyses.

## Repository contents

```text
LigandBench-OX2R/
├── README.md
├── LICENSE
├── CITATION.cff
├── DATA_LICENSE.md
├── DATA_PROVENANCE.md
├── REPRODUCIBILITY.md
├── CHANGELOG_REPRODUCIBILITY.md
├── JOCHEM_REPRODUCIBILITY_CHECKLIST.md
├── requirements.txt
├── environment.yml
├── pyproject.toml
├── config/reproducibility.yaml
├── data/
│   ├── README.md
│   ├── ox2r_curated_6415.csv
│   └── chembl34_ox2r_query.sql
├── ligandbench/
├── scripts/
│   ├── benchmark_frozen.py
│   ├── similarity_stratified.py
│   ├── cluster_split.py
│   └── reproduce_main.py
├── ox2_agvsan_v2/       # precomputed primary outputs
├── ox2_extra/            # precomputed robustness outputs
├── sample_data/          # offline synthetic demo
├── tests/
├── run_cli.py
├── reviewer_extra.py
├── score_external.py
├── run.sh
└── app.py
```

No manuscript DOCX files are included.

## Data provenance and licensing

The study used human OX2R (`CHEMBL4792`) bioactivity data from **ChEMBL version 34**. The manuscript reports an upstream retrieval of 30,116 activity records in August 2026, followed by hierarchical pharmacological-direction labeling and structure curation.

The **frozen model input** supplied here contains 6,415 compounds: **747 agonists and 5,668 antagonists**, with the canonical SMILES, compound identifier, binary label, label class, sample weight, and median pActivity used by the benchmark. This frozen file is the direct input for the reproducible benchmark and therefore avoids dependence on a mutable live ChEMBL endpoint.

The upstream ChEMBL release is identified by DOI **10.6019/CHEMBL.database.34**. A SQL query specification for retrieving the OX2R activity records from a local ChEMBL 34 database is provided in `data/chembl34_ox2r_query.sql`. The repository does not claim to redistribute the complete ChEMBL database. ChEMBL-derived material remains subject to the applicable ChEMBL licence and attribution requirements; see `DATA_LICENSE.md`.

## Exact software environment

The manuscript environment is pinned in both `requirements.txt` and `environment.yml`:

- Python 3.11.15
- RDKit 2026.03.5
- scikit-learn 1.8.0
- XGBoost 3.2.0
- LightGBM 4.7.0
- imbalanced-learn 0.14.2
- SHAP 0.51.0
- NumPy 2.4.4
- pandas 3.0.2
- Matplotlib 3.10.8
- requests 2.32.5
- urllib3 2.6.0

Randomized analyses use **seed 42**, with the exact analysis settings recorded in `config/reproducibility.yaml`.

## Installation

Recommended Conda environment:

```bash
conda env create -f environment.yml
conda activate ligandbench-ox2r
```

Alternatively, with Python 3.11:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Reproduce the main manuscript analyses

### 1. Primary scaffold-aware benchmark

```bash
python scripts/benchmark_frozen.py
```

This reproduces the benchmark from the frozen 6,415-compound input using the fixed seed, approximately 20% scaffold-grouped held-out test split, five-fold grouped cross-validation, five molecular representations, six classifier variants, and OOF-MCC model selection.

### 2. Similarity-stratified validation

```bash
python scripts/similarity_stratified.py --bootstrap 2000
```

For each held-out compound, maximum ECFP4 Tanimoto similarity to the development set is calculated. MCC is then reported in the prespecified similarity bins with bootstrap 95% confidence intervals. The script also reports the held-out compounds with similarity ≥0.5 to a development-set compound of the opposite class.

### 3. Stricter 120-cluster chemical-space validation

```bash
python scripts/cluster_split.py --clusters 120 --folds 5 --n-init 10
```

Compounds are clustered with K-means in 2048-bit ECFP4 space using 120 clusters, random state 42 and `n_init=10`. Five-fold `GroupKFold` uses the cluster IDs as groups, preventing members of the same learned chemical neighbourhood from being split between training and validation folds. Model selection remains based on OOF MCC.

### 4. One-command workflow

```bash
python scripts/reproduce_main.py
```

This runs the primary benchmark, similarity-stratified analysis, and 120-cluster validation. Results are written under `results/reproduction/`.

### 5. Additional robustness analyses

```bash
python reviewer_extra.py
```

This reproduces the high-confidence action-type subset, 300-permutation Y-randomization, and ECFP4 bit visualisation using the supplied frozen data.

## Precomputed outputs

`ox2_agvsan_v2/` contains the primary benchmark outputs used in the manuscript, including the curated dataset, complete 30-model benchmark table, SHAP importances, summary information, and scientific figures/tables.

`ox2_extra/` contains the high-confidence subset benchmark, Y-randomization results, and fingerprint-bit visualisation.

The dedicated `scripts/` directory additionally provides executable code for the manuscript's similarity-stratified and 120-cluster chemical-space analyses rather than relying only on precomputed numbers.

## Live ChEMBL retrieval

The generic GUI/CLI retains functionality for retrieving current ChEMBL records. Because a live database can change independently of the cited ChEMBL 34 release, **do not use live retrieval to claim exact reproduction of the manuscript benchmark**. For exact reruns of the reported computational analyses, use the frozen `data/ox2r_curated_6415.csv` input.

The optional CLI compatibility argument `--target-label` is accepted but does not alter the scientific computation.

## Verification

Run the lightweight repository tests with:

```bash
pytest -q tests
```

The tests verify the frozen dataset counts, required reproducibility configuration, and presence of the dedicated analysis scripts.

## Archival and versioning

For the manuscript, cite the public GitHub repository. For stronger long-term preservation, create a versioned GitHub release and archive that release in Zenodo or another persistent repository. **Do not add a DOI to the manuscript until the DOI has actually been minted.**

## Citation

Citation metadata are provided in `CITATION.cff`.

## Licences

- Original software code: **MIT License** (`LICENSE`)
- ChEMBL-derived data: subject to the applicable ChEMBL licence and attribution requirements (`DATA_LICENSE.md`)
- Third-party Python packages: retain their own licences
