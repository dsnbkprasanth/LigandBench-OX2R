"""
Stage 04b - Model benchmarking.

Six classifier variants, matching the reference paper:

    RF, SVM-RBF, XGBoost, LightGBM, RF+SMOTE, XGBoost+SMOTE

For each (feature set x classifier) combination we:
  * generate out-of-fold (OOF) predictions with scaffold-grouped k-fold CV on
    the development set, and
  * fit a final model on the whole development set and score it once on the
    held-out scaffold-grouped test set.

Descriptor-containing feature sets get median imputation + standardization
inside the CV pipeline (fit on train folds only, so there is no leakage).
SMOTE, when used, is applied strictly inside the training folds.

Metrics: ROC-AUC, PR-AUC, Brier, balanced accuracy, MCC, sensitivity,
specificity, precision, F1. The best model is chosen by OOF MCC.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    balanced_accuracy_score,
    matthews_corrcoef,
    recall_score,
    precision_score,
    f1_score,
    confusion_matrix,
)

warnings.filterwarnings("ignore")

# optional deps
try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False
try:
    from lightgbm import LGBMClassifier
    _HAS_LGBM = True
except Exception:
    _HAS_LGBM = False
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    _HAS_SMOTE = True
except Exception:
    _HAS_SMOTE = False


def _base_estimators(seed: int = 42):
    est = {}
    est["RandomForest"] = RandomForestClassifier(
        n_estimators=500, max_features="sqrt", class_weight="balanced",
        n_jobs=-1, random_state=seed,
    )
    est["SVM-RBF"] = SVC(
        kernel="rbf", C=1.0, class_weight="balanced",
        probability=True, random_state=seed,
    )
    if _HAS_XGB:
        est["XGBoost"] = XGBClassifier(
            n_estimators=500, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
            n_jobs=-1, random_state=seed, tree_method="hist",
        )
    if _HAS_LGBM:
        est["LightGBM"] = LGBMClassifier(
            n_estimators=500, learning_rate=0.05, max_depth=6,
            num_leaves=31, subsample=0.8, colsample_bytree=0.8,
            n_jobs=-1, random_state=seed, verbose=-1,
        )
    return est


def available_models(use_smote: bool = True):
    names = list(_base_estimators().keys())
    if use_smote and _HAS_SMOTE:
        if "RandomForest" in names:
            names.append("RF+SMOTE")
        if "XGBoost" in names:
            names.append("XGBoost+SMOTE")
    return names


def _make_pipeline(model_name: str, needs_scaling: bool, seed: int = 42):
    est_map = _base_estimators(seed)
    use_smote = model_name.endswith("+SMOTE")
    base_name = model_name.replace("+SMOTE", "").replace("RF", "RandomForest") \
        if model_name.startswith("RF+") else model_name.replace("+SMOTE", "")
    if base_name not in est_map:
        return None
    estimator = est_map[base_name]

    steps = []
    if needs_scaling:
        steps.append(("impute", SimpleImputer(strategy="median")))
        steps.append(("scale", StandardScaler()))
    if use_smote and _HAS_SMOTE:
        steps.append(("smote", SMOTE(random_state=seed)))
        steps.append(("clf", estimator))
        return ImbPipeline(steps)
    steps.append(("clf", estimator))
    return Pipeline(steps)


def _metrics(y_true, y_prob, weights=None) -> dict:
    y_pred = (y_prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "ROC-AUC": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan,
        "PR-AUC": average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan,
        "Brier": brier_score_loss(y_true, y_prob),
        "Bal_acc": balanced_accuracy_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "Sens": sens,
        "Spec": spec,
        "Prec": precision_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
    }


def _oof_predictions(pipeline, X, y, cv_indices, sample_weight=None):
    """Return OOF probabilities aligned to X's row order."""
    oof = np.full(len(y), np.nan)
    for train_idx, val_idx in cv_indices:
        pipe = _clone(pipeline)
        fit_kwargs = {}
        # pass sample weights to the final estimator when not using SMOTE
        if sample_weight is not None and not _is_smote(pipe):
            fit_kwargs["clf__sample_weight"] = sample_weight[train_idx]
        try:
            pipe.fit(X[train_idx], y[train_idx], **fit_kwargs)
        except Exception:
            pipe.fit(X[train_idx], y[train_idx])
        oof[val_idx] = pipe.predict_proba(X[val_idx])[:, 1]
    return oof


def _clone(pipeline):
    from sklearn.base import clone
    return clone(pipeline)


def _is_smote(pipe) -> bool:
    return any(name == "smote" for name, _ in getattr(pipe, "steps", []))


def run_benchmark(
    dev_df,
    test_df,
    feature_sets_dev,
    feature_sets_test,
    needs_scaling_map,
    cv_indices,
    models=None,
    seed: int = 42,
    progress=None,
):
    """Run the full benchmark grid.

    feature_sets_dev / feature_sets_test : {name: (X, feature_names)}
    Returns (results_df, oof_store, best_key) where oof_store[(fs, model)] holds
    OOF probabilities for later Y-randomization / SHAP reuse.
    """
    y_dev = dev_df["label"].to_numpy()
    y_test = test_df["label"].to_numpy() if len(test_df) else None
    w_dev = dev_df["sample_weight"].to_numpy() if "sample_weight" in dev_df else None

    if models is None:
        models = available_models(use_smote=True)

    rows = []
    oof_store = {}
    combos = [(fs, m) for fs in feature_sets_dev for m in models]
    for k, (fs, model_name) in enumerate(combos):
        needs_scaling = needs_scaling_map.get(fs, False)
        pipe = _make_pipeline(model_name, needs_scaling, seed)
        if pipe is None:
            continue
        X_dev = feature_sets_dev[fs][0]

        oof = _oof_predictions(pipe, X_dev, y_dev, cv_indices, w_dev)
        mask = ~np.isnan(oof)
        oof_m = _metrics(y_dev[mask], oof[mask])
        oof_store[(fs, model_name)] = oof

        row = {"feature_set": fs, "model": model_name}
        row.update({f"OOF_{k2}": v for k2, v in oof_m.items()})

        # final fit on full dev, score on held-out test
        if y_test is not None and len(test_df):
            final = _clone(pipe)
            fit_kwargs = {}
            if w_dev is not None and not _is_smote(final):
                fit_kwargs["clf__sample_weight"] = w_dev
            try:
                final.fit(X_dev, y_dev, **fit_kwargs)
            except Exception:
                final.fit(X_dev, y_dev)
            X_test = feature_sets_test[fs][0]
            test_prob = final.predict_proba(X_test)[:, 1]
            test_m = _metrics(y_test, test_prob)
            row.update({f"Test_{k2}": v for k2, v in test_m.items()})

        rows.append(row)
        if progress:
            progress(k + 1, len(combos), f"{fs} / {model_name}")

    results = pd.DataFrame(rows)
    if results.empty:
        return results, oof_store, None

    results = results.sort_values("OOF_MCC", ascending=False).reset_index(drop=True)
    best_row = results.iloc[0]
    best_key = (best_row["feature_set"], best_row["model"])
    return results, oof_store, best_key


def fit_final_model(feature_set_X, y, model_name, needs_scaling, sample_weight=None, seed=42):
    """Fit a single pipeline on all provided data (used for SHAP on the best)."""
    pipe = _make_pipeline(model_name, needs_scaling, seed)
    fit_kwargs = {}
    if sample_weight is not None and not _is_smote(pipe):
        fit_kwargs["clf__sample_weight"] = sample_weight
    try:
        pipe.fit(feature_set_X, y, **fit_kwargs)
    except Exception:
        pipe.fit(feature_set_X, y)
    return pipe
