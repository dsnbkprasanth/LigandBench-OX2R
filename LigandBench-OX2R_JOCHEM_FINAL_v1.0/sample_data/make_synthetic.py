"""Generate a synthetic ChEMBL-shaped activity CSV for offline testing/demo.

Builds ~600 valid, drug-like molecules with diverse scaffolds by decorating a
pool of cores with random substituents. Compounds that contain a
sulfonamide-like motif are made *potent* (active), the rest weak (inactive),
plus measurement noise and some multi-measurement compounds -- so the pipeline
has genuine, learnable signal and a realistic label mix.
"""
import random
import numpy as np
import pandas as pd
from rdkit import Chem

random.seed(7)
np.random.seed(7)

CORES = [
    "c1ccccc1", "c1ccncc1", "c1ccc2ccccc2c1", "C1CCCCC1", "c1ccc(cc1)C(=O)",
    "c1cc2ccccc2cc1", "C1CCNCC1", "c1ccoc1", "c1ccsc1", "c1cnc2ccccc2n1",
    "C1CCC(CC1)N", "c1ccc(cc1)O", "c1ccc(cc1)N", "C1CCOC1", "c1ccc2[nH]ccc2c1",
]
SUBS = ["C", "CC", "CCC", "CCN", "CO", "CF", "CCl", "CBr", "C(=O)O", "CN(C)C",
        "CS(=O)(=O)N", "Cc1ccccc1", "COC", "CC(C)C", "CCO", "CS(=O)(=O)N"]

rows = []
seen = set()
i = 0
attempts = 0
while len(seen) < 600 and attempts < 6000:
    attempts += 1
    core = random.choice(CORES)
    n_sub = random.randint(1, 3)
    smi = core
    for _ in range(n_sub):
        smi = smi + random.choice(SUBS)
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        continue
    can = Chem.MolToSmiles(mol)
    if can in seen:
        continue
    seen.add(can)
    i += 1

    # "active" if it carries a sulfonamide motif -> potent; else weak
    is_active_truth = mol.HasSubstructMatch(Chem.MolFromSmarts("S(=O)(=O)N"))
    if is_active_truth:
        p_center = 7.2
    else:
        p_center = 4.6
    # some compounds get multiple measurements with noise
    n_meas = random.choice([1, 1, 1, 2, 3])
    for _ in range(n_meas):
        p = float(np.clip(np.random.normal(p_center, 0.4), 2.0, 11.0))
        molar = 10 ** (-p)
        rows.append({
            "molecule_chembl_id": f"CHEMBLSYN{i:05d}",
            "canonical_smiles": can,
            "target_chembl_id": "CHEMBL_SYNTH",
            "assay_chembl_id": f"A{i%50}",
            "assay_type": "B",
            "assay_description": "synthetic binding assay",
            "document_chembl_id": f"DOC{i%20}",
            "standard_type": "IC50",
            "standard_relation": "=",
            "standard_value": molar * 1e9,  # nM
            "standard_units": "nM",
            "pchembl_value": round(p, 2),
        })

df = pd.DataFrame(rows)
df.to_csv("/root/LigandBench/sample_data/synthetic_activities.csv", index=False)
print(f"Wrote {len(df)} rows, {df['molecule_chembl_id'].nunique()} compounds")
print(df.head(3).to_string())
