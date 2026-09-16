"""
LigandBench - Streamlit GUI
===========================

A target-agnostic, scaffold-aware ligand-based ML benchmarking app that mirrors
the workflow of Drewe (J. Cheminformatics, 2026) for ADRB3 agonists.

Run:  streamlit run app.py
"""

import io
import json
import os
import sys
import tempfile
import zipfile

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

# Guard against being launched with `python app.py` instead of `streamlit run`.
# In "bare mode" there is no script-run context, st.stop() does not halt, and the
# app crashes later with confusing AttributeErrors. Fail fast with a clear hint.
try:
    from streamlit.runtime.scriptrunner import get_script_run_ctx
    if get_script_run_ctx() is None:
        sys.stderr.write(
            "\n[LigandBench] This is a Streamlit app. Start it with:\n"
            "    streamlit run app.py\n"
            "(You ran it as a plain script, e.g. `python app.py`.)\n\n"
        )
        sys.exit(0)
except Exception:
    pass

from ligandbench import chembl_data, reporting
from ligandbench.pipeline import run_pipeline, PipelineConfig

st.set_page_config(page_title="LigandBench", page_icon="🧪", layout="wide")

# ---------------------------------------------------------------- helpers ----

ALL_FEATURE_SETS = ["ECFP4", "ECFP6", "RDKit_desc", "ECFP4+RDKit", "ECFP6+RDKit"]


def _init_state():
    for k, v in {
        "raw": None,
        "result": None,
        "target_choices": None,
        "target_label": "",
        "source_note": "",
    }.items():
        st.session_state.setdefault(k, v)


_init_state()


def confusion_fig(y_true, y_prob):
    from sklearn.metrics import confusion_matrix
    y_pred = (np.array(y_prob) >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    im = ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center",
                color="white" if v > cm.max() / 2 else "black", fontsize=12)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Inactive", "Active"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Inactive", "Active"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Observed")
    ax.set_title("Confusion matrix (test)")
    fig.tight_layout()
    return fig


def roc_pr_fig(y_true, y_prob):
    from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score
    y_true = np.array(y_true); y_prob = np.array(y_prob)
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.2))
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    axes[0].plot(fpr, tpr, lw=2)
    axes[0].plot([0, 1], [0, 1], "--", color="gray", lw=1)
    axes[0].set_title(f"ROC (AUC={roc_auc_score(y_true, y_prob):.3f})")
    axes[0].set_xlabel("False positive rate"); axes[0].set_ylabel("True positive rate")
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    axes[1].plot(rec, prec, lw=2, color="#c0392b")
    axes[1].set_title(f"Precision-Recall (AP={average_precision_score(y_true, y_prob):.3f})")
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    fig.tight_layout()
    return fig


def yrand_fig(yrand):
    perm = np.array(yrand.get("permuted_aucs", []))
    real = yrand.get("real_auc", np.nan)
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    if perm.size:
        ax.hist(perm, bins=min(20, max(5, perm.size)), color="#5b8def", alpha=0.85,
                label="Permuted runs")
    ax.axvline(real, color="#c0392b", ls="--", lw=2, label=f"Real ROC-AUC = {real:.3f}")
    ax.set_xlabel("ROC-AUC"); ax.set_ylabel("Count")
    ax.set_title("Y-randomization")
    ax.legend()
    fig.tight_layout()
    return fig


def shap_fig(imp, top=20):
    d = imp.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.5, max(3.0, 0.28 * len(d))))
    ax.barh(d["feature"], d["mean_abs_shap"], color="#2e86c1")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"SHAP feature importance (top {len(d)})")
    fig.tight_layout()
    return fig


def ad_fig(ad_df, threshold):
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    vals = ad_df["max_train_similarity"].dropna()
    ax.hist(vals, bins=25, color="#48c9b0", alpha=0.9)
    ax.axvline(threshold, color="#c0392b", ls="--", lw=2,
               label=f"in-domain threshold = {threshold}")
    ax.set_xlabel("Max Tanimoto similarity to training set")
    ax.set_ylabel("Test compounds")
    ax.set_title("Applicability domain")
    ax.legend()
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------ sidebar --

st.sidebar.title("🧪 LigandBench")
st.sidebar.caption("Scaffold-aware ligand-based ML benchmark for **any** target")

source = st.sidebar.radio(
    "1 · Data source",
    ["ChEMBL target", "Upload CSV", "Synthetic demo"],
    help="Pull activities from ChEMBL, load your own CSV, or try a built-in demo set.",
)

if source == "ChEMBL target":
    q = st.sidebar.text_input("Target name or ChEMBL ID", value="",
                              placeholder="e.g. dopamine D2 or CHEMBL217")
    if st.sidebar.button("Search targets"):
        try:
            st.session_state.target_choices = chembl_data.search_targets(q)
        except Exception as e:
            st.sidebar.error(f"Search failed (network?): {e}")
    tc = st.session_state.target_choices
    if tc is not None and not tc.empty:
        labels = [
            f"{r.target_chembl_id} · {r.pref_name} ({r.organism})"
            for r in tc.itertuples()
        ]
        choice = st.sidebar.selectbox("Pick a target", labels)
        target_id = choice.split(" · ")[0]
        max_records = st.sidebar.number_input(
            "Max records to fetch (0 = all)", min_value=0, value=0, step=1000)
        if st.sidebar.button("Fetch activities"):
            try:
                with st.spinner("Fetching from ChEMBL ..."):
                    raw = chembl_data.fetch_activities(
                        target_id,
                        max_records=(max_records or None),
                    )
                st.session_state.raw = raw
                st.session_state.target_label = choice.split(" (")[0].split(" · ", 1)[-1] or target_id
                note = f"ChEMBL target {target_id} ({len(raw)} rows)"
                if getattr(raw, "attrs", {}).get("partial"):
                    note += " — ⚠ partial (network dropped mid-fetch; re-fetch for the full set)"
                st.session_state.source_note = note
            except Exception as e:
                st.sidebar.error(f"Fetch failed (network?): {e}")

elif source == "Upload CSV":
    up = st.sidebar.file_uploader("Activity CSV (needs a SMILES + value column)",
                                  type=["csv"])
    if up is not None:
        try:
            st.session_state.raw = chembl_data.load_activity_csv(up)
            st.session_state.source_note = f"Uploaded CSV ({len(st.session_state.raw)} rows)"
            st.session_state.target_label = "User dataset"
        except Exception as e:
            st.sidebar.error(f"Could not read CSV: {e}")

else:  # Synthetic demo
    if st.sidebar.button("Load synthetic demo set"):
        try:
            st.session_state.raw = chembl_data.load_activity_csv(
                "sample_data/synthetic_activities.csv")
            st.session_state.source_note = "Synthetic demo (600 compounds)"
            st.session_state.target_label = "Synthetic demo"
        except Exception as e:
            st.sidebar.error(f"Demo data missing: {e}")

st.sidebar.divider()
st.sidebar.subheader("2 · Configuration")

activity_mode = st.sidebar.selectbox(
    "Activity direction",
    ["all", "agonist", "antagonist", "binding", "agonist_vs_antagonist"],
    index=0,
    help="How to define the classes. 'all' = potent vs weak (potency). "
         "'agonist'/'antagonist'/'binding' = keep one direction, then potency-"
         "label it. 'agonist_vs_antagonist' = classify agonists (positive) vs "
         "antagonists (negative) — use this when agonist data lacks a usable "
         "inactive class (e.g. orexin OX2R), giving two balanced classes.",
)

with st.sidebar.expander("Labeling (potency consensus)", expanded=False):
    active_cut = st.slider("Active cut-off (pActivity ≥)", 4.0, 9.0, 6.0, 0.1)
    inactive_cut = st.slider("Inactive cut-off (pActivity ≤)", 3.0, 8.0, 5.0, 0.1)
    min_agreement = st.slider("Required measurement agreement", 0.5, 1.0, 0.67, 0.01)
    keep_ambiguous = st.checkbox("Keep ambiguous (down-weighted)", value=False)

with st.sidebar.expander("Curation (sanity filters)", expanded=False):
    mw_min, mw_max = st.slider("Molecular weight (Da)", 50, 1200, (100, 900), 10)
    ha_min, ha_max = st.slider("Heavy atoms", 3, 100, (8, 70), 1)
    tpsa_max = st.slider("Max TPSA (Å²)", 50, 400, 250, 10)
    rotb_max = st.slider("Max rotatable bonds", 5, 40, 20, 1)
    dedup_tanimoto = st.slider("Near-duplicate Tanimoto cut", 0.7, 1.0, 0.90, 0.01)

with st.sidebar.expander("Split & benchmark", expanded=False):
    test_fraction = st.slider("Held-out test fraction", 0.1, 0.4, 0.2, 0.05)
    n_splits = st.slider("CV folds", 3, 10, 5, 1)
    feature_sets = st.multiselect("Feature sets", ALL_FEATURE_SETS,
                                  default=["ECFP4", "ECFP6", "RDKit_desc"])
    use_smote = st.checkbox("Include SMOTE variants", value=True)
    n_permutations = st.slider("Y-randomization permutations", 5, 100, 20, 5)
    seed = st.number_input("Random seed", value=42, step=1)

run = st.sidebar.button("▶ Run pipeline", type="primary",
                        disabled=st.session_state.raw is None)


# -------------------------------------------------------------------- main ----

st.title("LigandBench — reproducible ligand-based ML for any target")
st.markdown(
    "This reproduces the **scaffold-aware benchmark** from *Drewe, J. "
    "Cheminformatics (2026)* — ChEMBL → consensus labels → RDKit curation → "
    "five feature sets → six classifiers under grouped scaffold CV → "
    "Y-randomization + SHAP — generalized to work for **any target**."
)

if st.session_state.raw is None:
    st.info("⬅ Choose a data source in the sidebar to begin. "
            "The **Synthetic demo** runs fully offline in ~2 minutes.")
    st.stop()

st.success(f"Data loaded — {st.session_state.source_note}")

if run:
    cfg = PipelineConfig(
        activity_mode=activity_mode,
        active_cut=active_cut, inactive_cut=inactive_cut, min_agreement=min_agreement,
        keep_ambiguous=keep_ambiguous, mw_min=mw_min, mw_max=mw_max,
        ha_min=ha_min, ha_max=ha_max, tpsa_max=tpsa_max, rotb_max=rotb_max,
        dedup_tanimoto=dedup_tanimoto, test_fraction=test_fraction, n_splits=n_splits,
        n_permutations=n_permutations, use_smote=use_smote, seed=int(seed),
        feature_sets=feature_sets or None,
    )
    prog = st.progress(0.0, text="Starting ...")
    steps = {"Consensus labeling ...": 0.1, "Structure curation ...": 0.25,
             "Scaffold-aware split ...": 0.35, "Featurizing ...": 0.45,
             "Benchmarking ...": 0.6, "Y-randomization": 0.8,
             "SHAP interpretation ...": 0.9, "Applicability domain ...": 0.95,
             "Done.": 1.0}

    def logger(msg):
        for key, frac in steps.items():
            if msg.startswith(key.split(" ")[0]):
                prog.progress(frac, text=msg)
                return
        prog.progress(0.5, text=msg)

    with st.spinner("Running the full pipeline — this can take a couple of minutes ..."):
        try:
            st.session_state.result = run_pipeline(st.session_state.raw, cfg, log=logger)
            st.session_state.result.target_label = st.session_state.get("target_label", "")
        except Exception as e:
            st.error(f"Pipeline error: {e}")
            st.stop()
    prog.progress(1.0, text="Done.")

res = st.session_state.result
if res is None:
    st.info("Configure options in the sidebar, then click **▶ Run pipeline**.")
    st.stop()

if getattr(res, "results", None) is None or res.results.empty:
    st.warning(
        "The pipeline stopped early — usually too few labelable/curated compounds "
        "or only one class. Try loosening the labeling cut-offs or curation filters."
    )
    if not res.labeled.empty:
        st.write("Label summary:", res.label_summary)
    st.stop()

# ---- tabs ----
tabs = st.tabs(["📊 Overview", "🧬 Data & labels", "🧹 Curation & split",
                "🏆 Benchmark", "🥇 Best model", "🔬 Validation", "📥 Export"])

with tabs[0]:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Curated compounds", res.curation_summary.get("n", 0))
    c2.metric("Active fraction", f"{res.curation_summary.get('active_fraction', 0):.2f}")
    c3.metric("Dev / Test", f"{res.split_report.get('n_dev',0)} / {res.split_report.get('n_test',0)}")
    c4.metric("Scaffold overlap", res.split_report.get("scaffold_overlap", 0))
    best_fs, best_model = res.best_key
    st.subheader(f"Best model: {best_model} on {best_fs}")
    best = res.results.iloc[0]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("OOF ROC-AUC", f"{best.get('OOF_ROC-AUC', float('nan')):.3f}")
    m2.metric("OOF MCC", f"{best.get('OOF_MCC', float('nan')):.3f}")
    if "Test_ROC-AUC" in best:
        m3.metric("Test ROC-AUC", f"{best.get('Test_ROC-AUC', float('nan')):.3f}")
        m4.metric("Test MCC", f"{best.get('Test_MCC', float('nan')):.3f}")
    if res.yrand:
        st.caption(
            f"Y-randomization: real ROC-AUC {res.yrand['real_auc']:.3f} vs permuted "
            f"mean {res.yrand['permuted_mean']:.3f} (p = {res.yrand['p_value']:.3f}, "
            f"{res.yrand['n_permutations']} permutations)."
        )

with tabs[1]:
    st.subheader("Raw activity records")
    st.dataframe(st.session_state.raw.head(500), use_container_width=True)
    st.subheader("Consensus label summary")
    st.write(res.label_summary)
    if not res.labeled.empty:
        fig, ax = plt.subplots(figsize=(6, 2.6))
        res.labeled["median_pactivity"].hist(bins=30, ax=ax, color="#5b8def")
        ax.axvline(active_cut, color="green", ls="--", label="active cut")
        ax.axvline(inactive_cut, color="red", ls="--", label="inactive cut")
        ax.set_xlabel("Median pActivity per compound"); ax.legend()
        st.pyplot(fig)

with tabs[2]:
    st.subheader("Curation summary")
    st.write(res.curation_summary)
    st.subheader("Scaffold-aware split")
    st.json(res.split_report)
    st.caption("Scaffold overlap of 0 confirms no chemotype leaks between "
               "development and held-out test sets.")

with tabs[3]:
    st.subheader("Benchmark — all feature-set × classifier combinations")
    show_cols = [c for c in res.results.columns
                 if c in ("feature_set", "model") or c.startswith("OOF_") or c.startswith("Test_")]
    styled = res.results[show_cols].copy()
    num_cols = [c for c in styled.columns if c not in ("feature_set", "model")]
    st.dataframe(styled.style.format({c: "{:.3f}" for c in num_cols})
                 .background_gradient(subset=["OOF_MCC"], cmap="Greens"),
                 use_container_width=True, height=520)
    st.caption("Sorted by out-of-fold MCC (primary selection metric).")
    fig, ax = plt.subplots(figsize=(8, 3.5))
    top = res.results.head(12).iloc[::-1]
    ax.barh([f"{r.feature_set}/{r.model}" for r in top.itertuples()],
            top["OOF_MCC"], color="#27ae60")
    ax.set_xlabel("OOF MCC"); ax.set_title("Top combinations")
    fig.tight_layout(); st.pyplot(fig)

with tabs[4]:
    best_fs, best_model = res.best_key
    st.subheader(f"Held-out test performance — {best_model} / {best_fs}")
    best = res.results.iloc[0]
    test_metrics = {k.replace("Test_", ""): v for k, v in best.items() if k.startswith("Test_")}
    oof_metrics = {k.replace("OOF_", ""): v for k, v in best.items() if k.startswith("OOF_")}
    comp = pd.DataFrame({"OOF (dev)": oof_metrics, "Held-out test": test_metrics})
    st.dataframe(comp.style.format("{:.3f}"), use_container_width=True)
    if res.best_test:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.pyplot(roc_pr_fig(res.best_test["y_true"], res.best_test["y_prob"]))
        with col2:
            st.pyplot(confusion_fig(res.best_test["y_true"], res.best_test["y_prob"]))

with tabs[5]:
    st.subheader("Y-randomization")
    if res.yrand:
        st.pyplot(yrand_fig(res.yrand))
        st.caption(
            f"Empirical one-sided p = {res.yrand['p_value']:.3f}. A real ROC-AUC far "
            "above the permuted null argues against chance correlation."
        )
    st.subheader("SHAP feature importance (best model)")
    if res.shap_imp is not None and not res.shap_imp.empty:
        st.pyplot(shap_fig(res.shap_imp))
    else:
        st.info("SHAP not available for this model/feature set (tree models only).")
    st.subheader("Applicability domain")
    if res.ad_df is not None and not res.ad_df.empty:
        st.pyplot(ad_fig(res.ad_df, res.ad_summary.get("threshold", 0.3)))
        st.write(res.ad_summary)

with tabs[6]:
    st.subheader("Download results")
    st.download_button("Benchmark table (CSV)",
                       res.results.to_csv(index=False).encode(),
                       "benchmark_results.csv", "text/csv")
    if not res.curated.empty:
        st.download_button("Curated dataset (CSV)",
                           res.curated.to_csv(index=False).encode(),
                           "curated_dataset.csv", "text/csv")
    if res.shap_imp is not None and not res.shap_imp.empty:
        st.download_button("SHAP importances (CSV)",
                           res.shap_imp.to_csv(index=False).encode(),
                           "shap_importances.csv", "text/csv")
    summary = {
        "source": st.session_state.source_note,
        "best_model": list(res.best_key) if res.best_key else None,
        "split_report": res.split_report,
        "label_summary": res.label_summary,
        "curation_summary": res.curation_summary,
        "y_randomization": {k: v for k, v in res.yrand.items() if k != "permuted_aucs"},
        "applicability_domain": res.ad_summary,
    }
    st.download_button("Run summary (JSON)",
                       json.dumps(summary, indent=2, default=str).encode(),
                       "run_summary.json", "application/json")
    st.caption("Everything here is reproducible from the same seed and config.")


