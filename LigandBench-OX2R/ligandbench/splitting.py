"""
Stage 04a - Scaffold-aware splitting.

Bemis-Murcko scaffolds are used as grouping variables so that compounds sharing
a scaffold never appear in both training and test. We create:

  * a scaffold-grouped held-out test set (default ~20% of compounds), and
  * grouped k-fold cross-validation on the remaining development set,

guaranteeing zero scaffold overlap between train and test, and between CV folds.
This is the key methodological choice from the reference paper: it estimates
generalization to *unseen chemotypes* rather than to memorized analogues.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import GroupKFold


def bemis_murcko_scaffold(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles  # fall back to the SMILES itself as its own group
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        s = Chem.MolToSmiles(scaffold)
        return s if s else smiles
    except Exception:
        return smiles


def add_scaffolds(curated: pd.DataFrame) -> pd.DataFrame:
    df = curated.copy()
    df["scaffold"] = [bemis_murcko_scaffold(s) for s in df["canonical_smiles"]]
    return df


def scaffold_grouped_test_split(
    df: pd.DataFrame, test_fraction: float = 0.2, seed: int = 42
):
    """Assign whole scaffolds to a held-out test set until ~test_fraction of
    compounds are covered. Larger scaffold groups are placed first (deterministic
    given seed) so the split is stable. Returns (dev_df, test_df).
    """
    if "scaffold" not in df.columns:
        df = add_scaffolds(df)

    rng = np.random.default_rng(seed)
    groups = df.groupby("scaffold")
    sizes = groups.size().sort_values(ascending=False)

    # shuffle within equal sizes for a fair but reproducible assignment
    order = list(sizes.index)
    rng.shuffle(order)

    n_total = len(df)
    n_target = int(round(test_fraction * n_total))
    test_scaffolds = set()
    n_test = 0
    for scaf in order:
        if n_test >= n_target:
            break
        test_scaffolds.add(scaf)
        n_test += int(sizes[scaf])

    test_mask = df["scaffold"].isin(test_scaffolds)
    dev_df = df[~test_mask].reset_index(drop=True)
    test_df = df[test_mask].reset_index(drop=True)
    return dev_df, test_df


def grouped_cv_indices(dev_df: pd.DataFrame, n_splits: int = 5):
    """Yield (train_idx, val_idx) for grouped k-fold on the development set,
    grouping by scaffold. n_splits is capped by the number of scaffolds."""
    groups = dev_df["scaffold"].to_numpy()
    n_groups = len(np.unique(groups))
    n_splits = max(2, min(n_splits, n_groups))
    gkf = GroupKFold(n_splits=n_splits)
    X_dummy = np.zeros((len(dev_df), 1))
    y = dev_df["label"].to_numpy()
    return list(gkf.split(X_dummy, y, groups)), n_splits


def split_report(dev_df, test_df) -> dict:
    dev_scaf = set(dev_df["scaffold"]) if "scaffold" in dev_df else set()
    test_scaf = set(test_df["scaffold"]) if "scaffold" in test_df else set()
    return {
        "n_dev": int(len(dev_df)),
        "n_test": int(len(test_df)),
        "n_dev_scaffolds": len(dev_scaf),
        "n_test_scaffolds": len(test_scaf),
        "scaffold_overlap": len(dev_scaf & test_scaf),
        "dev_active_fraction": float(dev_df["label"].mean()) if len(dev_df) else 0.0,
        "test_active_fraction": float(test_df["label"].mean()) if len(test_df) else 0.0,
    }
