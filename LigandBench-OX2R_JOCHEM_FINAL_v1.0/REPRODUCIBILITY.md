# Reproducibility record for the OX2R manuscript

This repository is designed so that the reported main computational findings can be reproduced from a frozen model input rather than from a mutable live database endpoint.

## Frozen model input

`data/ox2r_curated_6415.csv` is the exact curated compound-level model input used for the reported benchmark: 6,415 compounds, 747 agonists and 5,668 antagonists. Its SHA-256 checksum is documented in `data/README.md`.

The manuscript reports that the upstream retrieval contained 30,116 activity records from human OX2R (`CHEMBL4792`) from ChEMBL release 34. The release is identified by DOI 10.6019/CHEMBL.database.34. The repository supplies the direct frozen model input plus a SQL query specification for reconstructing the upstream target query from a local ChEMBL 34 database.

## Primary benchmark

```bash
python scripts/benchmark_frozen.py
```

## Similarity-stratified analysis

```bash
python scripts/similarity_stratified.py --bootstrap 2000
```

## Stricter chemical-cluster analysis

```bash
python scripts/cluster_split.py --clusters 120 --folds 5 --n-init 10
```

Compounds are clustered with K-means on 2048-bit ECFP4 vectors (120 clusters, `random_state=42`, `n_init=10`) and cluster IDs are used as GroupKFold groups. Model selection is by OOF MCC.

## One-command workflow

```bash
python scripts/reproduce_main.py
```

## Live ChEMBL retrieval

The generic GUI/CLI can retrieve current ChEMBL activity records. Because current database content can change, exact manuscript reproduction should use the frozen curated input.
