"""
Stage 01 - Standardization & consensus labeling.

The reference paper assigned a binary label per compound using a two-level
consensus over all of its measurements. That logic is target-specific (it
inferred *agonist* assay direction from assay text). Here we generalize it to a
potency-threshold consensus that works for any target and any activity type,
while keeping the same spirit:

  * Convert each measurement to a pActivity scale  p = -log10(C[M]).
    (pChEMBL is used directly when present.)
  * Group rows by compound.
  * A compound is ACTIVE (label 1) if its median pActivity >= active_cut with
    enough agreement, INACTIVE (label 0) if its median pActivity <= inactive_cut,
    and AMBIGUOUS otherwise (dropped, or down-weighted).
  * A per-compound sample weight encodes label confidence (more concordant
    measurements -> higher weight), mirroring the paper's confidence weighting.

The active/inactive cut-offs and the required agreement are exposed so the user
can tune them per target from the GUI.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Units we know how to convert to molar. Everything else is dropped unless a
# pChEMBL value is already present.
_UNIT_TO_MOLAR = {
    "M": 1.0,
    "mM": 1e-3,
    "uM": 1e-6,
    "µM": 1e-6,
    "nM": 1e-9,
    "pM": 1e-12,
    "fM": 1e-15,
}


def _to_pactivity(row) -> float:
    """Return pActivity for one measurement, or NaN if not usable."""
    pchembl = row.get("pchembl_value")
    if pchembl is not None and not pd.isna(pchembl):
        try:
            return float(pchembl)
        except (TypeError, ValueError):
            pass
    val = row.get("standard_value")
    units = row.get("standard_units")
    if val is None or pd.isna(val) or units not in _UNIT_TO_MOLAR:
        return np.nan
    try:
        molar = float(val) * _UNIT_TO_MOLAR[units]
    except (TypeError, ValueError):
        return np.nan
    if molar <= 0:
        return np.nan
    return float(-np.log10(molar))


def add_pactivity(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["pactivity"] = df.apply(_to_pactivity, axis=1)
    # keep only rows with a valid SMILES and a usable measurement
    df = df[df["canonical_smiles"].notna() & df["pactivity"].notna()].copy()
    return df


def consensus_labels(
    df: pd.DataFrame,
    active_cut: float = 6.0,
    inactive_cut: float = 5.0,
    min_agreement: float = 0.67,
    keep_ambiguous: bool = False,
    ambiguous_weight: float = 0.5,
) -> pd.DataFrame:
    """Collapse per-measurement rows to one labeled row per compound.

    Returns a DataFrame with one row per compound:
        molecule_chembl_id, canonical_smiles, n_measurements,
        median_pactivity, label (1/0), sample_weight, label_class
    Ambiguous compounds are dropped unless keep_ambiguous is True (then labeled
    by the sign of the median and given a reduced weight).
    """
    df = add_pactivity(df)
    if df.empty:
        return pd.DataFrame()

    records = []
    for mol_id, g in df.groupby("molecule_chembl_id"):
        p = g["pactivity"].to_numpy()
        smiles = g["canonical_smiles"].dropna().iloc[0]
        med = float(np.median(p))
        n = len(p)

        frac_active = float(np.mean(p >= active_cut))
        frac_inactive = float(np.mean(p <= inactive_cut))

        label = None
        cls = "ambiguous"
        weight = ambiguous_weight
        if med >= active_cut and frac_active >= min_agreement:
            label, cls, weight = 1, "active", 1.0
        elif med <= inactive_cut and frac_inactive >= min_agreement:
            label, cls, weight = 0, "inactive", 1.0
        else:
            if keep_ambiguous:
                label = int(med >= (active_cut + inactive_cut) / 2.0)
                cls = "ambiguous"
                weight = ambiguous_weight
            else:
                continue

        # single-measurement compounds are less certain -> lower weight
        if n == 1 and cls != "ambiguous":
            weight = min(weight, 0.6)

        records.append(
            {
                "molecule_chembl_id": mol_id,
                "canonical_smiles": smiles,
                "n_measurements": n,
                "median_pactivity": med,
                "frac_active": frac_active,
                "frac_inactive": frac_inactive,
                "label": label,
                "label_class": cls,
                "sample_weight": weight,
            }
        )

    out = pd.DataFrame(records)
    return out


def direction_labels(
    df: pd.DataFrame,
    potency_floor: Optional[float] = None,
    min_agreement: float = 0.6,
) -> pd.DataFrame:
    """Label compounds by pharmacological DIRECTION: agonist (1) vs antagonist (0).

    This is the fix for targets (e.g. orexin OX2R) whose agonist assays report
    almost only potent compounds, leaving no usable inactive class. Instead of
    active-vs-inactive potency, we classify each compound as an agonist or an
    antagonist from its assay directions, giving two well-populated classes.

    A compound is class 1 (agonist) if a majority of its direction-bearing
    measurements are agonist, class 0 (antagonist) if the majority are
    antagonist; ties and compounds with no agonist/antagonist evidence are
    dropped. An optional potency_floor keeps only compounds whose median
    pActivity is >= the floor, so we compare genuine agonists with genuine
    antagonists rather than weak noise.
    """
    from .chembl_data import classify_direction

    df = add_pactivity(df)
    if df.empty:
        return pd.DataFrame()
    df = df.copy()
    df["direction"] = df.apply(classify_direction, axis=1)

    records = []
    for mol_id, g in df.groupby("molecule_chembl_id"):
        smiles = g["canonical_smiles"].dropna().iloc[0]
        med = float(np.median(g["pactivity"].to_numpy()))
        dirs = g["direction"]
        n_ag = int((dirs == "agonist").sum())
        n_an = int((dirs == "antagonist").sum())
        if n_ag == 0 and n_an == 0:
            continue
        total_dir = n_ag + n_an
        if n_ag == n_an:
            continue  # tie -> ambiguous, drop
        label = 1 if n_ag > n_an else 0
        agree = max(n_ag, n_an) / total_dir
        if agree < min_agreement:
            continue
        if potency_floor is not None and med < potency_floor:
            continue
        records.append(
            {
                "molecule_chembl_id": mol_id,
                "canonical_smiles": smiles,
                "n_measurements": int(len(g)),
                "median_pactivity": med,
                "frac_active": float(n_ag / total_dir),
                "frac_inactive": float(n_an / total_dir),
                "label": label,
                "label_class": "agonist" if label == 1 else "antagonist",
                "sample_weight": 1.0,
            }
        )
    return pd.DataFrame(records)


def label_summary(labeled: pd.DataFrame) -> dict:
    if labeled.empty:
        return {"n": 0}
    return {
        "n": int(len(labeled)),
        "n_active": int((labeled["label"] == 1).sum()),
        "n_inactive": int((labeled["label"] == 0).sum()),
        "n_ambiguous": int((labeled["label_class"] == "ambiguous").sum()),
        "active_fraction": float((labeled["label"] == 1).mean()),
    }
