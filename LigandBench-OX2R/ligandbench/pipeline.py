"""
End-to-end orchestrator.

``run_pipeline`` runs the full flow from a labeled/curated set of compounds to a
benchmark table, best-model selection, Y-randomization and applicability domain.
It is GUI-agnostic: the Streamlit app calls the individual stage functions so it
can show intermediate tables, but this module is handy for scripted/batch runs
and for testing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from . import (
    chembl_data, labeling, curation, features, splitting, benchmark,
    validation, applicability,
)


@dataclass
class PipelineConfig:
    activity_mode: str = "all"  # all | agonist | antagonist | binding | agonist_vs_antagonist
    direction_potency_floor: float = None  # optional pActivity floor for direction mode
    active_cut: float = 6.0
    inactive_cut: float = 5.0
    min_agreement: float = 0.67
    keep_ambiguous: bool = False
    mw_min: float = 100.0
    mw_max: float = 900.0
    ha_min: int = 8
    ha_max: int = 70
    tpsa_max: float = 250.0
    rotb_max: int = 20
    dedup_tanimoto: float = 0.90
    test_fraction: float = 0.2
    n_splits: int = 5
    n_permutations: int = 20
    use_smote: bool = True
    seed: int = 42
    feature_sets: Optional[list] = None  # None = all five


@dataclass
class PipelineResult:
    labeled: pd.DataFrame = field(default_factory=pd.DataFrame)
    curated: pd.DataFrame = field(default_factory=pd.DataFrame)
    dev_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    test_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    split_report: dict = field(default_factory=dict)
    results: pd.DataFrame = field(default_factory=pd.DataFrame)
    best_key: Optional[tuple] = None
    yrand: dict = field(default_factory=dict)
    ad_summary: dict = field(default_factory=dict)
    ad_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    best_test: dict = field(default_factory=dict)  # y_true, y_prob on held-out test
    shap_imp: pd.DataFrame = field(default_factory=pd.DataFrame)
    label_summary: dict = field(default_factory=dict)
    curation_summary: dict = field(default_factory=dict)
    direction_counts: dict = field(default_factory=dict)
    n_raw_rows: int = 0
    config: Optional["PipelineConfig"] = None
    target_label: str = ""


def run_pipeline(raw_activities: pd.DataFrame, config: PipelineConfig, log=print) -> PipelineResult:
    res = PipelineResult()
    res.config = config
    res.n_raw_rows = int(len(raw_activities))

    if config.activity_mode == "agonist_vs_antagonist":
        # Direction classification: positive = agonists, negative = antagonists.
        # Fixes targets whose agonist assays lack a usable inactive class.
        log("Direction labeling (agonist vs antagonist) ...")
        res.direction_counts = dict(
            chembl_data.filter_activity_direction(raw_activities, "all")
            .attrs.get("direction_counts", {})
        )
        res.labeled = labeling.direction_labels(
            raw_activities, potency_floor=config.direction_potency_floor
        )
        log(f"  direction tally: {res.direction_counts}")
        if res.labeled.empty:
            log("No agonist/antagonist-labelable compounds.")
            return res
    else:
        if config.activity_mode and config.activity_mode != "all":
            log(f"Filtering to {config.activity_mode} assays ...")
            raw_activities = chembl_data.filter_activity_direction(
                raw_activities, config.activity_mode
            )
            res.direction_counts = dict(raw_activities.attrs.get("direction_counts", {}))
            log(f"  kept {len(raw_activities)} {config.activity_mode} rows "
                f"(direction tally: {res.direction_counts})")
            if raw_activities.empty:
                log("No rows matched that direction; try 'all' or a different mode.")
                return res

        log("Consensus labeling ...")
        res.labeled = labeling.consensus_labels(
            raw_activities,
            active_cut=config.active_cut,
            inactive_cut=config.inactive_cut,
            min_agreement=config.min_agreement,
            keep_ambiguous=config.keep_ambiguous,
        )
        if res.labeled.empty:
            log("No labelable compounds.")
            return res

    log("Structure curation ...")
    res.curated = curation.curate(
        res.labeled,
        mw_min=config.mw_min, mw_max=config.mw_max,
        ha_min=config.ha_min, ha_max=config.ha_max,
        tpsa_max=config.tpsa_max, rotb_max=config.rotb_max,
        dedup_tanimoto=config.dedup_tanimoto,
    )
    if len(res.curated) < 40 or res.curated["label"].nunique() < 2:
        log("Too few compounds or only one class after curation.")
        return res

    log("Scaffold-aware split ...")
    curated = splitting.add_scaffolds(res.curated).reset_index(drop=True)
    curated["_row"] = range(len(curated))
    res.dev_df, res.test_df = splitting.scaffold_grouped_test_split(
        curated, test_fraction=config.test_fraction, seed=config.seed
    )
    res.split_report = splitting.split_report(res.dev_df, res.test_df)

    log("Featurizing ...")
    # Featurize the whole curated set ONCE, then slice by row so dev and test
    # share identical feature columns (the descriptor variance filter must see
    # the same molecules for both).
    fs_all, needs_scaling, _ = features.build_feature_sets(curated)
    dev_rows = res.dev_df["_row"].to_numpy()
    test_rows = res.test_df["_row"].to_numpy()
    fs_dev = {k: (X[dev_rows], names) for k, (X, names) in fs_all.items()}
    fs_test = {k: (X[test_rows], names) for k, (X, names) in fs_all.items()}
    if config.feature_sets:
        fs_dev = {k: v for k, v in fs_dev.items() if k in config.feature_sets}
        fs_test = {k: v for k, v in fs_test.items() if k in config.feature_sets}

    log("Benchmarking ...")
    cv_indices, _ = splitting.grouped_cv_indices(res.dev_df, n_splits=config.n_splits)
    models = benchmark.available_models(use_smote=config.use_smote)
    res.results, oof_store, res.best_key = benchmark.run_benchmark(
        res.dev_df, res.test_df, fs_dev, fs_test, needs_scaling,
        cv_indices, models=models, seed=config.seed,
    )

    res.label_summary = labeling.label_summary(res.labeled)
    res.curation_summary = curation.curation_summary(res.curated)

    if res.best_key is not None:
        best_fs, best_model = res.best_key
        needs = needs_scaling.get(best_fs, False)
        w_dev = res.dev_df["sample_weight"].to_numpy() if "sample_weight" in res.dev_df else None

        # refit best model on full dev, capture held-out test predictions
        best_pipe = benchmark.fit_final_model(
            fs_dev[best_fs][0], res.dev_df["label"].to_numpy(),
            best_model, needs, sample_weight=w_dev, seed=config.seed,
        )
        if len(res.test_df):
            y_prob = best_pipe.predict_proba(fs_test[best_fs][0])[:, 1]
            res.best_test = {
                "y_true": res.test_df["label"].to_numpy().tolist(),
                "y_prob": y_prob.tolist(),
            }

        log(f"Y-randomization on best model ({best_fs} / {best_model}) ...")
        res.yrand = validation.y_randomization(
            fs_dev[best_fs][0], res.dev_df["label"].to_numpy(), cv_indices,
            best_model, needs,
            n_permutations=config.n_permutations, seed=config.seed,
        )

        log("SHAP interpretation ...")
        try:
            imp = validation.shap_importance(
                best_pipe, fs_dev[best_fs][0], fs_dev[best_fs][1], max_samples=300,
            )
            if imp is not None:
                res.shap_imp = imp
        except Exception as e:  # SHAP is best-effort
            log(f"SHAP skipped: {e}")

        log("Applicability domain ...")
        res.ad_df, res.ad_summary = applicability.applicability_domain(
            res.dev_df["canonical_smiles"], res.test_df["canonical_smiles"]
        )

    log("Done.")
    return res
