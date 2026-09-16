"""
Stage 07 - Y-randomization and SHAP interpretation.

Y-randomization: permute the labels and repeat the grouped-CV OOF procedure a
number of times. If the real model's ROC-AUC sits far above the permuted
distribution, the signal is unlikely to be a chance correlation. We report an
empirical one-sided permutation p-value.

SHAP: explain the selected best model with TreeExplainer (tree models) on a
sampled subset, returning per-feature mean |SHAP| for a feature-importance plot.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .benchmark import _make_pipeline, _clone, _is_smote


def y_randomization(
    X, y, cv_indices, model_name, needs_scaling,
    n_permutations: int = 20, seed: int = 42, progress=None,
):
    """Return dict with real_auc, permuted_aucs (list), and p_value."""
    rng = np.random.default_rng(seed)

    def oof_auc(labels):
        oof = np.full(len(labels), np.nan)
        pipe = _make_pipeline(model_name, needs_scaling, seed)
        for tr, va in cv_indices:
            p = _clone(pipe)
            p.fit(X[tr], labels[tr])
            oof[va] = p.predict_proba(X[va])[:, 1]
        m = ~np.isnan(oof)
        if len(np.unique(labels[m])) < 2:
            return np.nan
        return roc_auc_score(labels[m], oof[m])

    real_auc = oof_auc(y)
    permuted = []
    for i in range(n_permutations):
        yp = y.copy()
        rng.shuffle(yp)
        permuted.append(oof_auc(yp))
        if progress:
            progress(i + 1, n_permutations)

    permuted = np.array([a for a in permuted if not np.isnan(a)])
    n_ge = int(np.sum(permuted >= real_auc))
    p_value = (n_ge + 1) / (len(permuted) + 1)
    return {
        "real_auc": float(real_auc),
        "permuted_aucs": permuted.tolist(),
        "permuted_mean": float(np.mean(permuted)) if permuted.size else np.nan,
        "p_value": float(p_value),
        "n_permutations": int(len(permuted)),
    }


def shap_importance(
    fitted_pipeline, X, feature_names, max_samples: int = 300, seed: int = 42,
):
    """Compute mean |SHAP| per feature for a fitted pipeline whose final step is
    a tree model. Returns a DataFrame sorted by importance, or None if SHAP is
    unavailable / the model is not tree-based.
    """
    try:
        import shap
    except Exception:
        return None

    # locate the final estimator and any preprocessing
    steps = getattr(fitted_pipeline, "steps", None)
    if steps is None:
        return None
    if _is_smote(fitted_pipeline):
        # SMOTE only affects training; transform X through non-smote steps
        pass
    estimator = steps[-1][1]

    # apply preprocessing (impute/scale) that precedes the classifier
    Xt = X
    for name, step in steps[:-1]:
        if name == "smote":
            continue
        Xt = step.transform(Xt)

    rng = np.random.default_rng(seed)
    if Xt.shape[0] > max_samples:
        idx = rng.choice(Xt.shape[0], max_samples, replace=False)
        Xs = Xt[idx]
    else:
        Xs = Xt

    try:
        explainer = shap.TreeExplainer(estimator)
        sv = explainer.shap_values(Xs)
        # binary classifiers may return a list [class0, class1]
        if isinstance(sv, list):
            sv = sv[1]
        # newer shap returns (n, features, classes)
        if sv.ndim == 3:
            sv = sv[:, :, 1]
        mean_abs = np.abs(sv).mean(axis=0)
    except Exception:
        return None

    imp = pd.DataFrame(
        {"feature": feature_names, "mean_abs_shap": mean_abs}
    ).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    return imp
