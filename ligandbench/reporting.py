"""
Reporting - publication figures and tables.

Given a completed ``PipelineResult``, write every figure and table an article
would need to ``outdir`` (figures/ and tables/ subfolders) at 300 dpi, and
return the paths plus the table DataFrames for reproducible reporting
them. Nothing here invents numbers: every value comes from the pipeline run.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 300, "font.size": 10})


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_data_flow(res, path):
    stages = ["Raw\nrecords", "Labeled\ncompounds", "Curated\ncompounds",
              "Dev\nset", "Test\nset"]
    vals = [
        res.n_raw_rows,
        res.label_summary.get("n", 0),
        res.curation_summary.get("n", 0),
        res.split_report.get("n_dev", 0),
        res.split_report.get("n_test", 0),
    ]
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    bars = ax.bar(stages, vals, color="#4c72b0")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,}", ha="center",
                va="bottom", fontsize=9)
    ax.set_ylabel("Count")
    ax.set_title("Dataset construction flow")
    ax.margins(y=0.15)
    return _save(fig, path)


def fig_label_dist(res, path):
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    if not res.labeled.empty and "median_pactivity" in res.labeled:
        ax.hist(res.labeled["median_pactivity"].dropna(), bins=30, color="#5b8def")
    cfg = res.config
    if cfg is not None:
        ax.axvline(cfg.active_cut, color="green", ls="--", label=f"active ≥ {cfg.active_cut}")
        ax.axvline(cfg.inactive_cut, color="red", ls="--", label=f"inactive ≤ {cfg.inactive_cut}")
        ax.legend()
    ax.set_xlabel("Median pActivity per compound")
    ax.set_ylabel("Compounds")
    ax.set_title("Consensus potency distribution")
    return _save(fig, path)


def fig_benchmark_mcc(res, path, top=12):
    d = res.results.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.0, max(3.2, 0.32 * len(d))))
    ax.barh([f"{r.feature_set} / {r.model}" for r in d.itertuples()],
            d["OOF_MCC"], color="#27ae60")
    ax.set_xlabel("Out-of-fold MCC")
    ax.set_title("Benchmark — feature set × classifier (top combinations)")
    return _save(fig, path)


def fig_best_roc_pr_cm(res, path):
    from sklearn.metrics import (roc_curve, precision_recall_curve, roc_auc_score,
                                 average_precision_score, confusion_matrix)
    if not res.best_test:
        return None
    y = np.array(res.best_test["y_true"]); p = np.array(res.best_test["y_prob"])
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4))
    fpr, tpr, _ = roc_curve(y, p)
    axes[0].plot(fpr, tpr, lw=2); axes[0].plot([0, 1], [0, 1], "--", color="gray")
    axes[0].set_title(f"ROC (AUC = {roc_auc_score(y, p):.3f})")
    axes[0].set_xlabel("False positive rate"); axes[0].set_ylabel("True positive rate")
    prec, rec, _ = precision_recall_curve(y, p)
    axes[1].plot(rec, prec, lw=2, color="#c0392b")
    axes[1].set_title(f"Precision–Recall (AP = {average_precision_score(y, p):.3f})")
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    cm = confusion_matrix(y, (p >= 0.5).astype(int), labels=[0, 1])
    im = axes[2].imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        axes[2].text(j, i, str(v), ha="center", va="center",
                     color="white" if v > cm.max() / 2 else "black")
    axes[2].set_xticks([0, 1]); axes[2].set_xticklabels(["Inactive", "Active"])
    axes[2].set_yticks([0, 1]); axes[2].set_yticklabels(["Inactive", "Active"])
    axes[2].set_xlabel("Predicted"); axes[2].set_ylabel("Observed")
    axes[2].set_title("Confusion matrix (test)")
    return _save(fig, path)


def fig_yrand(res, path):
    if not res.yrand:
        return None
    perm = np.array(res.yrand.get("permuted_aucs", []))
    real = res.yrand.get("real_auc", np.nan)
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    if perm.size:
        ax.hist(perm, bins=min(20, max(5, perm.size)), color="#5b8def",
                alpha=0.85, label="Permuted runs")
    ax.axvline(real, color="#c0392b", ls="--", lw=2, label=f"Real ROC-AUC = {real:.3f}")
    ax.set_xlabel("ROC-AUC"); ax.set_ylabel("Count")
    ax.set_title("Y-randomization"); ax.legend()
    return _save(fig, path)


def fig_shap(res, path, top=20):
    if res.shap_imp is None or res.shap_imp.empty:
        return None
    d = res.shap_imp.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.5, max(3.2, 0.30 * len(d))))
    ax.barh(d["feature"], d["mean_abs_shap"], color="#2e86c1")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"SHAP feature importance (top {len(d)})")
    return _save(fig, path)


def fig_applicability(res, path):
    if res.ad_df is None or res.ad_df.empty:
        return None
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    vals = res.ad_df["max_train_similarity"].dropna()
    ax.hist(vals, bins=25, color="#48c9b0", alpha=0.9)
    thr = res.ad_summary.get("threshold", 0.3)
    ax.axvline(thr, color="#c0392b", ls="--", lw=2, label=f"threshold = {thr}")
    ax.set_xlabel("Max Tanimoto similarity to training set")
    ax.set_ylabel("Test compounds")
    ax.set_title("Applicability domain"); ax.legend()
    return _save(fig, path)


def _fmt(df):
    df = df.copy()
    for c in df.columns:
        if df[c].dtype.kind in "fc":
            df[c] = df[c].round(3)
    return df


def generate_report(res, outdir: str) -> dict:
    """Write all figures and tables. Returns a dict of paths + table DataFrames."""
    figdir = os.path.join(outdir, "figures")
    tabdir = os.path.join(outdir, "tables")
    os.makedirs(figdir, exist_ok=True)
    os.makedirs(tabdir, exist_ok=True)

    figs = {
        "data_flow": fig_data_flow(res, os.path.join(figdir, "fig1_data_flow.png")),
        "label_dist": fig_label_dist(res, os.path.join(figdir, "fig2_label_distribution.png")),
        "benchmark": fig_benchmark_mcc(res, os.path.join(figdir, "fig3_benchmark_mcc.png")),
        "best_model": fig_best_roc_pr_cm(res, os.path.join(figdir, "fig4_best_model_test.png")),
        "yrand": fig_yrand(res, os.path.join(figdir, "fig5_yrandomization.png")),
        "shap": fig_shap(res, os.path.join(figdir, "fig6_shap_importance.png")),
        "applicability": fig_applicability(res, os.path.join(figdir, "fig7_applicability_domain.png")),
    }
    figs = {k: v for k, v in figs.items() if v}

    # Table 1 — full benchmark grid
    t1 = _fmt(res.results)
    t1_path = os.path.join(tabdir, "table1_benchmark.csv")
    t1.to_csv(t1_path, index=False)

    # Table 2 — best model OOF vs held-out test
    best = res.results.iloc[0]
    oof = {k.replace("OOF_", ""): v for k, v in best.items() if k.startswith("OOF_")}
    test = {k.replace("Test_", ""): v for k, v in best.items() if k.startswith("Test_")}
    t2 = pd.DataFrame({"Metric": list(oof.keys()),
                       "OOF_development": list(oof.values()),
                       "Held_out_test": [test.get(k, np.nan) for k in oof.keys()]})
    t2 = _fmt(t2)
    t2_path = os.path.join(tabdir, "table2_best_model.csv")
    t2.to_csv(t2_path, index=False)

    # Table 3 — Y-randomization summary
    yr = res.yrand or {}
    t3 = pd.DataFrame([{
        "real_ROC_AUC": round(yr.get("real_auc", float("nan")), 3),
        "permuted_mean_ROC_AUC": round(yr.get("permuted_mean", float("nan")), 3),
        "n_permutations": yr.get("n_permutations", 0),
        "empirical_p_value": round(yr.get("p_value", float("nan")), 4),
    }])
    t3_path = os.path.join(tabdir, "table3_yrandomization.csv")
    t3.to_csv(t3_path, index=False)

    # curated dataset + SHAP importances for completeness
    if not res.curated.empty:
        res.curated.to_csv(os.path.join(tabdir, "curated_dataset.csv"), index=False)
    if res.shap_imp is not None and not res.shap_imp.empty:
        res.shap_imp.to_csv(os.path.join(tabdir, "shap_importances.csv"), index=False)

    return {
        "figures": figs,
        "tables": {"table1": t1_path, "table2": t2_path, "table3": t3_path},
        "table_dfs": {"table1": t1, "table2": t2, "table3": t3},
    }
