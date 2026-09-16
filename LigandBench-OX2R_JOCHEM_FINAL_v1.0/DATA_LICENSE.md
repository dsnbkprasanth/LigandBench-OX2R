# Data licensing and provenance

## ChEMBL-derived data

The OX2R bioactivity data used in this project were obtained from **ChEMBL version 34**, target **CHEMBL4792**. ChEMBL 34 was released in March 2024 and is identified by DOI **10.6019/CHEMBL.database.34**.

ChEMBL is a third-party resource and is distributed under the Creative Commons Attribution-ShareAlike 3.0 Unported (CC BY-SA 3.0) licence. ChEMBL-derived files in this repository remain subject to the applicable ChEMBL terms and attribution requirements. The MIT licence below applies to the original software code authored for this repository, not to ChEMBL-derived data.

ChEMBL: https://www.ebi.ac.uk/chembl/

## Included frozen dataset

`data/ox2r_curated_6415.csv` is the frozen, curated compound-level dataset used as the model input for the reported benchmark: 6,415 compounds (747 agonists and 5,668 antagonists). It is provided to make the main modelling results independently reproducible without relying on the live ChEMBL API.

The repository does not claim that this frozen derivative replaces the ChEMBL source database. Users who redistribute or modify ChEMBL-derived material should consult the ChEMBL release terms and provide appropriate attribution.

## Software

Original analysis and reproducibility code in this repository is released under the MIT License. Third-party packages retain their own licences.
