"""
Stage 02 - Structure curation.

Standardize SMILES with RDKit and apply the same "chemical sanity" filters used
in the reference paper, then remove duplicates and near-duplicates:

  * canonicalize / sanitize; drop molecules that fail to parse
  * keep the largest organic fragment (strip salts/solvents)
  * sanity window: MW 100-900 Da, 8-70 heavy atoms, TPSA <= 250, rotatable
    bonds <= 20, no metal atoms
  * collapse exact duplicates by InChIKey (prefer the higher-confidence label)
  * remove near-duplicates: any control (label 0) within Tanimoto >= 0.90 of a
    training active is dropped to avoid leaking near-identical structures across
    the class boundary.

All thresholds are arguments so the GUI can expose them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem import rdFingerprintGenerator
from rdkit import DataStructs

RDLogger.DisableLog("rdApp.*")

_METALS = {
    "Li", "Be", "Na", "Mg", "Al", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn",
    "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Rb", "Sr", "Y", "Zr", "Nb", "Mo",
    "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Cs", "Ba", "La",
    "Ce", "Pt", "Au", "Hg", "Tl", "Pb", "Bi",
}

_normalizer = rdMolStandardize.Normalizer()
_chooser = rdMolStandardize.LargestFragmentChooser()
_uncharger = rdMolStandardize.Uncharger()
_ecfp4_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def standardize_smiles(smiles: str):
    """Return (canonical_smiles, rdkit_mol) or (None, None) on failure."""
    if not isinstance(smiles, str) or not smiles:
        return None, None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None
    try:
        mol = _chooser.choose(mol)
        mol = _normalizer.normalize(mol)
        mol = _uncharger.uncharge(mol)
        Chem.SanitizeMol(mol)
    except Exception:
        return None, None
    return Chem.MolToSmiles(mol), mol


def _has_metal(mol) -> bool:
    return any(a.GetSymbol() in _METALS for a in mol.GetAtoms())


def _passes_sanity(mol, mw_min, mw_max, ha_min, ha_max, tpsa_max, rotb_max) -> bool:
    mw = Descriptors.MolWt(mol)
    if not (mw_min <= mw <= mw_max):
        return False
    ha = mol.GetNumHeavyAtoms()
    if not (ha_min <= ha <= ha_max):
        return False
    if rdMolDescriptors.CalcTPSA(mol) > tpsa_max:
        return False
    if rdMolDescriptors.CalcNumRotatableBonds(mol) > rotb_max:
        return False
    if _has_metal(mol):
        return False
    return True


def curate(
    labeled: pd.DataFrame,
    mw_min: float = 100.0,
    mw_max: float = 900.0,
    ha_min: int = 8,
    ha_max: int = 70,
    tpsa_max: float = 250.0,
    rotb_max: int = 20,
    dedup_tanimoto: float = 0.90,
    progress=None,
) -> pd.DataFrame:
    """Curate a labeled compound table. Returns curated compounds with columns:
    molecule_chembl_id, canonical_smiles (standardized), inchikey, label,
    sample_weight, plus an rdkit mol cached in the frame is NOT kept (recomputed
    downstream to keep the frame picklable).
    """
    if labeled.empty:
        return labeled

    rows = []
    n = len(labeled)
    for i, (_, r) in enumerate(labeled.iterrows()):
        std_smiles, mol = standardize_smiles(r["canonical_smiles"])
        if mol is None:
            continue
        if not _passes_sanity(mol, mw_min, mw_max, ha_min, ha_max, tpsa_max, rotb_max):
            continue
        inchikey = Chem.MolToInchiKey(mol)
        rows.append(
            {
                "molecule_chembl_id": r["molecule_chembl_id"],
                "canonical_smiles": std_smiles,
                "inchikey": inchikey,
                "label": int(r["label"]),
                "label_class": r.get("label_class", "na"),
                "sample_weight": float(r.get("sample_weight", 1.0)),
                "median_pactivity": r.get("median_pactivity", np.nan),
            }
        )
        if progress and (i % 200 == 0):
            progress(i, n)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # 1) collapse exact duplicates by InChIKey; prefer higher-confidence label
    df = df.sort_values("sample_weight", ascending=False)
    df = df.drop_duplicates(subset="inchikey", keep="first").reset_index(drop=True)

    # 2) near-duplicate removal across the class boundary
    df = _remove_near_duplicates(df, dedup_tanimoto)
    return df.reset_index(drop=True)


def _ecfp4(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return _ecfp4_gen.GetFingerprint(mol)


def _remove_near_duplicates(df: pd.DataFrame, cutoff: float) -> pd.DataFrame:
    """Drop control (label 0) compounds that are Tanimoto >= cutoff to any
    active (label 1), preventing near-identical structures spanning the classes.
    """
    if cutoff >= 1.0:
        return df
    actives = df[df["label"] == 1]
    if actives.empty:
        return df
    active_fps = [fp for fp in (_ecfp4(s) for s in actives["canonical_smiles"]) if fp]
    if not active_fps:
        return df

    keep = []
    for _, r in df.iterrows():
        if r["label"] == 1:
            keep.append(True)
            continue
        fp = _ecfp4(r["canonical_smiles"])
        if fp is None:
            keep.append(False)
            continue
        sims = DataStructs.BulkTanimotoSimilarity(fp, active_fps)
        keep.append(max(sims) < cutoff)
    return df[pd.Series(keep, index=df.index)].copy()


def curation_summary(curated: pd.DataFrame) -> dict:
    if curated.empty:
        return {"n": 0}
    return {
        "n": int(len(curated)),
        "n_active": int((curated["label"] == 1).sum()),
        "n_inactive": int((curated["label"] == 0).sum()),
        "active_fraction": float((curated["label"] == 1).mean()),
    }
