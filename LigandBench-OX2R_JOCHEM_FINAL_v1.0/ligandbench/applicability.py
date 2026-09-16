"""
Stage 08 - Applicability domain.

A simple, standard AD measure for fingerprint models: for each test compound,
the maximum Tanimoto similarity (ECFP4) to any training compound. Compounds far
from the training set (low max-similarity) are outside the reliable domain. We
report the distribution and the fraction of test compounds above a threshold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def _fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return _gen.GetFingerprint(mol) if mol is not None else None


def applicability_domain(train_smiles, test_smiles, threshold: float = 0.3):
    train_fps = [fp for fp in (_fp(s) for s in train_smiles) if fp is not None]
    if not train_fps:
        return pd.DataFrame(), {"n_test": 0}

    max_sims = []
    for s in test_smiles:
        fp = _fp(s)
        if fp is None:
            max_sims.append(np.nan)
            continue
        sims = DataStructs.BulkTanimotoSimilarity(fp, train_fps)
        max_sims.append(max(sims) if sims else 0.0)

    df = pd.DataFrame({"smiles": list(test_smiles), "max_train_similarity": max_sims})
    valid = df["max_train_similarity"].dropna()
    summary = {
        "n_test": int(len(df)),
        "mean_max_similarity": float(valid.mean()) if len(valid) else np.nan,
        "median_max_similarity": float(valid.median()) if len(valid) else np.nan,
        "fraction_in_domain": float((valid >= threshold).mean()) if len(valid) else np.nan,
        "threshold": threshold,
    }
    return df, summary
