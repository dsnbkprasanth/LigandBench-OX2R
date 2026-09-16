# Frozen OX2R model input

`ox2r_curated_6415.csv` is the frozen curated compound-level dataset used as the model input for the reported scaffold-aware benchmark.

- Target: human OX2R / ChEMBL `CHEMBL4792`
- Source release: ChEMBL 34
- Curated compounds: 6,415
- Agonists: 747
- Antagonists: 5,668
- Fixed analysis seed: 42
- SHA-256: `0d0c75da92dffa13f8f46c4e8648189c381512c911b71e4cdd686e45eb57a7f0`

The file contains the final canonical SMILES, compound identifiers, binary labels, label class, sample weights, and median pActivity used by the benchmark.

The manuscript describes the upstream retrieval as 30,116 ChEMBL 34 activity records. The frozen curated file is supplied because it is the direct model input and allows the main computational results to be reproduced without depending on a mutable live API. The SQL query specification for retrieving the upstream ChEMBL 34 records is in `chembl34_ox2r_query.sql`.
