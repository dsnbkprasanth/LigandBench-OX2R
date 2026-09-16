# Data provenance and licensing

## ChEMBL-derived data

The OX2R bioactivity data in this repository were obtained from **ChEMBL version 34** for target **CHEMBL4792**. The repository contains curated/processed derivatives of ChEMBL records used for the reported computational analyses.

ChEMBL is a third-party resource. Its data are not relicensed by the MIT license included in this repository. Users who reuse or redistribute the ChEMBL-derived data should consult and comply with the current ChEMBL terms, attribution requirements, and database licensing information.

ChEMBL: https://www.ebi.ac.uk/chembl/

## Software

The analysis and reproducibility scripts in this repository are released under the MIT License. Third-party Python packages retain their own licenses.

## Scope of included files

- `ligandbench/`, `app.py`, `run_cli.py`, `reviewer_extra.py`, `score_external.py`, and `sample_data/make_synthetic.py` are scientific analysis/reproducibility code.
- `ox2_agvsan_v2/` and `ox2_extra/` contain data, precomputed results, tables, and figures associated with the OX2R analysis.
- No manuscript-generation script or manuscript document is included in this repository.
