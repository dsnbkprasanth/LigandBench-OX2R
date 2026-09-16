# Data provenance and licensing

## ChEMBL-derived data

The study uses human orexin receptor 2 (OX2R; **CHEMBL4792**) bioactivity data from **ChEMBL version 34**. The manuscript reports retrieval of 30,116 activity records in August 2026, followed by pharmacological-direction labelling, compound-level consensus labelling, structure standardisation, and curation.

ChEMBL 34 was released in March 2024 and is identified by DOI **10.6019/CHEMBL.database.34**.

The final frozen model input is:

- `data/ox2r_curated_6415.csv`
- 6,415 compounds
- 747 agonists
- 5,668 antagonists
- SHA-256: `0d0c75da92dffa13f8f46c4e8648189c381512c911b71e4cdd686e45eb57a7f0`

The frozen file is included specifically to prevent changes in the live ChEMBL API from changing the published benchmark. The upstream retrieval query specification is provided in `data/chembl34_ox2r_query.sql`.

## Licensing

ChEMBL is a third-party resource. ChEMBL-derived data are not relicensed by the MIT licence in this repository. Users reusing or redistributing ChEMBL-derived material should consult the applicable ChEMBL release terms and attribution requirements.

ChEMBL: https://www.ebi.ac.uk/chembl/

## Software

The original analysis and reproducibility code in this repository is released under the MIT License. Third-party Python packages retain their own licences.

## Scope of included files

- `ligandbench/`, `scripts/`, `app.py`, `run_cli.py`, `reviewer_extra.py`, `score_external.py`, and `sample_data/make_synthetic.py` are analysis/reproducibility code.
- `data/` contains the frozen model input and provenance/query specification.
- `ox2_agvsan_v2/` and `ox2_extra/` contain precomputed OX2R results, tables, and figures.
- No manuscript-generation script or manuscript document is included.
