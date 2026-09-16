"""
Stage 03 - Molecular representations / feature sets.

Five feature sets, exactly as in the reference paper:

    1. ECFP4                 (Morgan radius 2, 2048 bits)
    2. ECFP6                 (Morgan radius 3, 2048 bits)
    3. RDKit descriptors     (physicochemical panel, variance/NaN filtered)
    4. ECFP4 + RDKit descriptors
    5. ECFP6 + RDKit descriptors

Fingerprints use chirality encoding. Descriptors are median-imputed and, for
the descriptor-containing sets, standardized inside the modeling pipeline
(handled in benchmark.py, not here, to avoid train/test leakage).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from rdkit.Chem import rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")

_FP_SIZE = 2048
_ecfp4_gen = rdFingerprintGenerator.GetMorganGenerator(
    radius=2, fpSize=_FP_SIZE, includeChirality=True
)
_ecfp6_gen = rdFingerprintGenerator.GetMorganGenerator(
    radius=3, fpSize=_FP_SIZE, includeChirality=True
)

# A broad but standard RDKit descriptor panel. We compute all available
# descriptors and let the variance/NaN filter trim them.
_DESCRIPTOR_NAMES = [name for name, _ in Descriptors._descList]
_DESCRIPTOR_FUNCS = {name: fn for name, fn in Descriptors._descList}


def _fp_to_array(fp) -> np.ndarray:
    arr = np.zeros((_FP_SIZE,), dtype=np.int8)
    from rdkit import DataStructs

    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def morgan_matrix(smiles_list, radius: int = 2) -> np.ndarray:
    gen = _ecfp4_gen if radius == 2 else _ecfp6_gen
    rows = []
    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            rows.append(np.zeros((_FP_SIZE,), dtype=np.int8))
        else:
            rows.append(_fp_to_array(gen.GetFingerprint(mol)))
    return np.vstack(rows)


def descriptor_matrix(smiles_list):
    """Return (matrix, kept_descriptor_names). NaN cells are left as NaN here;
    imputation happens in the modeling pipeline."""
    rows = []
    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            rows.append([np.nan] * len(_DESCRIPTOR_NAMES))
            continue
        vals = []
        for name in _DESCRIPTOR_NAMES:
            try:
                vals.append(float(_DESCRIPTOR_FUNCS[name](mol)))
            except Exception:
                vals.append(np.nan)
        rows.append(vals)
    mat = np.array(rows, dtype=float)

    # drop descriptors that are entirely NaN or (near) zero variance
    keep = []
    names = []
    for j, name in enumerate(_DESCRIPTOR_NAMES):
        col = mat[:, j]
        finite = col[np.isfinite(col)]
        if finite.size == 0:
            continue
        if np.nanstd(col) < 1e-8:
            continue
        keep.append(j)
        names.append(name)
    return mat[:, keep], names


def build_feature_sets(curated: pd.DataFrame):
    """Compute all five feature sets once and return a dict:

        {name: (X, feature_names)}

    plus the descriptor block separately so hybrids reuse it.
    """
    smiles = curated["canonical_smiles"].tolist()

    ecfp4 = morgan_matrix(smiles, radius=2)
    ecfp6 = morgan_matrix(smiles, radius=3)
    desc, desc_names = descriptor_matrix(smiles)

    ecfp4_names = [f"ecfp4_{i}" for i in range(ecfp4.shape[1])]
    ecfp6_names = [f"ecfp6_{i}" for i in range(ecfp6.shape[1])]

    feature_sets = {
        "ECFP4": (ecfp4, ecfp4_names),
        "ECFP6": (ecfp6, ecfp6_names),
        "RDKit_desc": (desc, list(desc_names)),
        "ECFP4+RDKit": (
            np.hstack([ecfp4, desc]),
            ecfp4_names + list(desc_names),
        ),
        "ECFP6+RDKit": (
            np.hstack([ecfp6, desc]),
            ecfp6_names + list(desc_names),
        ),
    }
    # mark which feature sets contain a (dense) descriptor block that needs
    # imputation + scaling in the pipeline
    needs_scaling = {
        "ECFP4": False,
        "ECFP6": False,
        "RDKit_desc": True,
        "ECFP4+RDKit": True,
        "ECFP6+RDKit": True,
    }
    return feature_sets, needs_scaling, desc_names
