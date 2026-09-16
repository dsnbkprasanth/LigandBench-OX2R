"""
Stage 00 - Data acquisition.

Retrieve bioactivity records for any target directly from the ChEMBL REST API
(https://www.ebi.ac.uk/chembl/api/data), or load them from a user-supplied CSV.

We deliberately talk to the REST endpoint with ``requests`` rather than the
``chembl_webresource_client`` package, because that client performs a network
call at *import* time (fetching its SPORE description), which makes an app fail
to even start when offline. Direct REST calls are lazy, paginated, and easy to
cache.

The two public entry points are:

    search_targets(query)          -> list of candidate targets (for the GUI picker)
    fetch_activities(target_id)    -> tidy pandas DataFrame of raw activity rows

All network use is confined to this module.
"""

from __future__ import annotations

import time
from typing import Callable, Optional

import pandas as pd
import requests
from requests.adapters import HTTPAdapter

try:
    from urllib3.util.retry import Retry
except Exception:  # pragma: no cover
    Retry = None

CHEMBL_BASE = "https://www.ebi.ac.uk/chembl/api/data"

_SESSION = None


def _session() -> requests.Session:
    """A shared session with automatic retries for transient failures
    (dropped SSL connections, 429/5xx, read timeouts)."""
    global _SESSION
    if _SESSION is None:
        s = requests.Session()
        if Retry is not None:
            retry = Retry(
                total=6, connect=6, read=6, status=6,
                backoff_factor=1.5,  # 0, 1.5, 3, 6, 12, 24s between tries
                status_forcelist=[429, 500, 502, 503, 504],
                allowed_methods=frozenset(["GET"]),
                raise_on_status=False,
            )
            adapter = HTTPAdapter(max_retries=retry)
            s.mount("https://", adapter)
            s.mount("http://", adapter)
        _SESSION = s
    return _SESSION

# Columns we keep from the ChEMBL activity endpoint. These mirror the fields the
# reference paper retained: identifiers, structure, assay/target keys, and the
# standardized measurement (type, relation, value, units, pChEMBL).
ACTIVITY_FIELDS = [
    "molecule_chembl_id",
    "canonical_smiles",
    "target_chembl_id",
    "assay_chembl_id",
    "assay_type",
    "assay_description",
    "document_chembl_id",
    "standard_type",
    "standard_relation",
    "standard_value",
    "standard_units",
    "pchembl_value",
    "action_type",   # ChEMBL-curated AGONIST/ANTAGONIST when available (authoritative)
]


def _get_json(url: str, params: dict, timeout: int = 90, attempts: int = 4) -> dict:
    """GET JSON with the retrying session, plus a manual outer retry loop so a
    stubborn transient error (e.g. SSL EOF) doesn't abort a long paginated fetch.
    """
    headers = {"Accept": "application/json"}
    last_err = None
    for i in range(attempts):
        try:
            resp = _session().get(url, params=params, headers=headers, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:  # noqa: BLE001 - retry any transient failure
            last_err = e
            time.sleep(2.0 * (i + 1))
    raise last_err


def search_targets(query: str, limit: int = 25) -> pd.DataFrame:
    """Search ChEMBL targets by free text (name, gene, synonym) or ChEMBL ID.

    Returns a DataFrame with target_chembl_id, pref_name, target_type,
    organism, and the ChEMBL record count so the user can pick sensibly.
    """
    query = (query or "").strip()
    if not query:
        return pd.DataFrame()

    # If it already looks like a target ChEMBL ID, resolve it directly.
    if query.upper().startswith("CHEMBL"):
        try:
            data = _get_json(
                f"{CHEMBL_BASE}/target/{query.upper()}.json", params={}
            )
            return pd.DataFrame(
                [
                    {
                        "target_chembl_id": data.get("target_chembl_id"),
                        "pref_name": data.get("pref_name"),
                        "target_type": data.get("target_type"),
                        "organism": data.get("organism"),
                    }
                ]
            )
        except Exception:
            pass  # fall through to text search

    data = _get_json(
        f"{CHEMBL_BASE}/target/search.json",
        params={"q": query, "limit": limit},
    )
    rows = []
    for t in data.get("targets", []):
        rows.append(
            {
                "target_chembl_id": t.get("target_chembl_id"),
                "pref_name": t.get("pref_name"),
                "target_type": t.get("target_type"),
                "organism": t.get("organism"),
                "score": t.get("score"),
            }
        )
    return pd.DataFrame(rows)


def fetch_activities(
    target_id: str,
    standard_types: Optional[list] = None,
    page_size: int = 1000,
    max_records: Optional[int] = None,
    progress: Optional[Callable[[int, str], None]] = None,
) -> pd.DataFrame:
    """Fetch all activity rows for a target from ChEMBL, with pagination.

    Parameters
    ----------
    target_id : str
        A target ChEMBL ID, e.g. "CHEMBL4630".
    standard_types : list of str, optional
        Restrict to these measurement types (e.g. ["EC50", "IC50", "Ki"]).
        None keeps everything and lets the labeling stage decide.
    max_records : int, optional
        Safety cap. None fetches all available rows.
    progress : callable(count, message), optional
        Progress callback for the GUI.
    """
    target_id = target_id.strip().upper()
    params = {
        "target_chembl_id": target_id,
        "limit": page_size,
        "offset": 0,
        "only": ",".join(ACTIVITY_FIELDS),
    }
    if standard_types:
        params["standard_type__in"] = ",".join(standard_types)

    rows: list = []
    url = f"{CHEMBL_BASE}/activity.json"
    partial = False
    while True:
        try:
            data = _get_json(url, params=params)
        except Exception as e:
            # Network gave out mid-fetch. Keep whatever we already have rather
            # than discarding thousands of downloaded rows; the caller decides
            # if the partial set is enough.
            if rows:
                partial = True
                if progress:
                    progress(len(rows), f"Network error after {len(rows)} rows; "
                                        f"using partial data ({e.__class__.__name__}).")
                break
            raise
        page = data.get("activities", [])
        rows.extend(page)
        if progress:
            progress(len(rows), f"Fetched {len(rows)} activity records ...")

        if max_records and len(rows) >= max_records:
            rows = rows[:max_records]
            break

        meta = data.get("page_meta", {})
        nxt = meta.get("next")
        if not nxt:
            break
        # page_meta.next is a relative path incl. query string
        params = None  # type: ignore
        url = "https://www.ebi.ac.uk" + nxt
        time.sleep(0.05)  # be polite to the API

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    for col in ACTIVITY_FIELDS:
        if col not in df.columns:
            df[col] = None
    df = df[ACTIVITY_FIELDS].copy()
    df.attrs["partial"] = partial
    return df


def load_activity_csv(path_or_buffer) -> pd.DataFrame:
    """Load a user-supplied CSV of activities.

    The only required columns are a SMILES column and a measurement column.
    We normalize common column names so downstream stages see the canonical
    schema. Recognized aliases:

        smiles / canonical_smiles / SMILES
        molecule_chembl_id / compound_id / id
        standard_type / activity_type
        standard_value / value
        standard_units / units
        standard_relation / relation
        pchembl_value / pchembl
        assay_type, assay_description
    """
    df = pd.read_csv(path_or_buffer)
    lower = {c.lower(): c for c in df.columns}

    def pick(*names):
        for n in names:
            if n in lower:
                return lower[n]
        return None

    mapping = {
        "canonical_smiles": pick("canonical_smiles", "smiles"),
        "molecule_chembl_id": pick("molecule_chembl_id", "compound_id", "id", "mol_id"),
        "standard_type": pick("standard_type", "activity_type", "type"),
        "standard_value": pick("standard_value", "value"),
        "standard_units": pick("standard_units", "units"),
        "standard_relation": pick("standard_relation", "relation"),
        "pchembl_value": pick("pchembl_value", "pchembl", "pactivity"),
        "assay_type": pick("assay_type"),
        "assay_description": pick("assay_description"),
        "assay_chembl_id": pick("assay_chembl_id"),
        "target_chembl_id": pick("target_chembl_id"),
        "document_chembl_id": pick("document_chembl_id"),
    }
    out = pd.DataFrame()
    for canon, src in mapping.items():
        out[canon] = df[src] if src is not None else None

    if out["canonical_smiles"].isna().all():
        raise ValueError("CSV must contain a SMILES column (e.g. 'smiles').")
    if out["molecule_chembl_id"].isna().all():
        out["molecule_chembl_id"] = [f"MOL{i:06d}" for i in range(len(out))]
    return out


# --------------------------------------------------------------------------- #
# Functional-direction filter                                                 #
# --------------------------------------------------------------------------- #
# The generic potency labeling cannot tell an agonist from an antagonist. For a
# target whose ChEMBL data is dominated by antagonists/binding (e.g. orexin
# OX2R), that would put potent antagonists into the "active" class. This filter
# restricts the raw rows to a functional direction BEFORE labeling, so the
# active class reflects real agonists (or antagonists / binders, if chosen).
#
# It is a transparent, heuristic classifier over the measurement type and the
# assay-description text. It is meant as a strong first pass; users should spot
# check the retained assays for their target.

_AGONIST_KW = [
    "agonist", "agonism", "agonistic", "camp", "cyclic amp", "cyclic-amp",
    "calcium", "ca2+", "ca(2+)", "ca2 +", "gtpgammas", "gtpγs", "gtp gamma s",
    "[35s]gtp", "35s-gtp", "mobil", "stimulat", "activation", "beta-arrestin",
    "β-arrestin", "arrestin", "ip1", "ip3", "inositol phosphate", "efficacy",
    "receptor activation", "functional response",
]
_ANTAGONIST_KW = [
    "antagonist", "antagonism", "antagonistic", "inhibition", "inhibitor",
    "inhibit", "blockade", "block of", "schild", "reversal", "suppress",
]
_BINDING_KW = [
    "binding", "displacement", "radioligand", "affinity", "[3h]", "3h-",
    "[125i]", "competition binding", "saturation binding",
]

_AGONIST_TYPES = {"ec50", "pec50", "potency", "activity", "efficacy", "emax"}
_ANTAGONIST_TYPES = {"ic50", "kb", "pa2", "ke"}
_BINDING_TYPES = {"ki", "kd", "pki", "pkd"}


# Phrases that mention the REFERENCE agonist (e.g. orexin-A used to stimulate the
# receptor in an antagonist FLIPR assay). These must NOT make the test compound an
# agonist — a validation against real ChEMBL assay text showed this was the main
# source of false-agonist labels (IC50 inhibition assays that say "orexin-A as the
# agonist"). We strip them before keyword matching.
_REF_AGONIST_PHRASES = [
    "orexin-a as the agonist", "orexin-a as an agonist", "orexin a as the agonist",
    "as the agonist", "as an agonist", "the agonist", "agonist ala", "full agonist",
    "induced by", "reference agonist", "endogenous agonist", "peak induced",
    "response to", "agonist-induced", "orexin b", "orexin-b",
]
# Strict agonist keywords that indicate the TEST compound's own agonism.
_AGONIST_KW_STRICT = [
    "agonist activity", "agonistic activity", "receptor agonist", "agonist activity of",
    "intrinsic activity", "emax", "efficacy of the test", "partial agonist activity",
]


def classify_direction(row) -> str:
    """Return 'agonist' | 'antagonist' | 'binding' | 'unknown' for one row.

    Priority (highest first), reflecting a validation against real ChEMBL assay
    descriptions that revealed keyword-only rules mislabel IC50 inhibition assays
    as agonist:
      1. ChEMBL-curated ``action_type`` (AGONIST / ANTAGONIST) when present.
      2. Measurement type: EC50/pEC50/Emax/AC50/Potency -> agonist;
         IC50/Kb/pA2/Ke -> antagonist; Ki/Kd -> binding.
      3. Assay-description keywords, AFTER removing reference-agonist phrases.
    """
    # 1) curated action_type is authoritative
    at = str(row.get("action_type") or "").upper()
    if at:
        if "ANTAGONIST" in at or "INHIBITOR" in at or "BLOCKER" in at:
            return "antagonist"
        if "AGONIST" in at:  # includes PARTIAL AGONIST, POSITIVE ALLOSTERIC (treat as agonist-direction)
            return "agonist"

    stype = str(row.get("standard_type") or "").lower()
    # 2) measurement type is a strong, reliable prior
    if stype in ("ec50", "pec50", "emax", "ac50", "potency", "efficacy"):
        return "agonist"
    if stype in ("ic50", "kb", "pa2", "ke", "pic50"):
        return "antagonist"
    if stype in ("ki", "kd", "pki", "pkd"):
        return "binding"

    # 3) fall back to description keywords, guarding reference-agonist mentions
    desc = str(row.get("assay_description") or "").lower()
    for ph in _REF_AGONIST_PHRASES:
        desc = desc.replace(ph, " ")
    agon = any(k in desc for k in _AGONIST_KW_STRICT)
    antag = any(k in desc for k in _ANTAGONIST_KW)
    binding = any(k in desc for k in _BINDING_KW)
    if antag and not agon:
        return "antagonist"
    if agon and not antag:
        return "agonist"
    if binding:
        return "binding"
    return "unknown"


def filter_activity_direction(df: pd.DataFrame, mode: str = "all") -> pd.DataFrame:
    """Keep only rows matching a functional direction.

    mode: 'all' (no filtering), 'agonist', 'antagonist', or 'binding'.
    The returned frame carries df.attrs['direction_counts'] with the tally.
    """
    if df.empty or mode == "all":
        if not df.empty:
            df = df.copy()
            df.attrs["direction_counts"] = (
                df.apply(classify_direction, axis=1).value_counts().to_dict()
            )
        return df

    directions = df.apply(classify_direction, axis=1)
    counts = directions.value_counts().to_dict()
    keep = df[directions == mode].copy()
    keep.attrs["direction_counts"] = counts
    keep.attrs["direction_mode"] = mode
    return keep
