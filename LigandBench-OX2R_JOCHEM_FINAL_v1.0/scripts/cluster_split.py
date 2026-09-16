"""Reproduce the manuscript's stricter 120-cluster chemical-space validation.

Compounds are clustered by K-means in ECFP4 space (120 clusters, seed 42).
Five-fold GroupKFold uses cluster IDs as groups, so an entire chemical
neighbourhood is held out from each validation fold. The full 5x6 benchmark is
run and the best model is selected by OOF MCC, matching the primary benchmark's
model-selection rule.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from ligandbench import benchmark, features
from sklearn.model_selection import GroupKFold

SEED = 42


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', default='data/ox2r_curated_6415.csv')
    ap.add_argument('--out', default='results/cluster_split')
    ap.add_argument('--clusters', type=int, default=120)
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--n-init', type=int, default=10)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input).reset_index(drop=True)
    fs, needs, _ = features.build_feature_sets(df)
    X4 = fs['ECFP4'][0]
    km = KMeans(n_clusters=args.clusters, random_state=SEED, n_init=args.n_init)
    groups = km.fit_predict(X4)
    df['cluster_id'] = groups

    gkf = GroupKFold(n_splits=args.folds)
    cv = list(gkf.split(X4, df.label.to_numpy(), groups))
    models = benchmark.available_models(use_smote=True)
    rows = []
    for fs_name, (X, names) in fs.items():
        for model_name in models:
            pipe = benchmark._make_pipeline(model_name, needs[fs_name], SEED)
            oof = np.full(len(df), np.nan)
            for tr, va in cv:
                from sklearn.base import clone
                p = clone(pipe)
                w = df['sample_weight'].to_numpy()[tr] if 'sample_weight' in df and not benchmark._is_smote(p) else None
                try:
                    p.fit(X[tr], df.label.to_numpy()[tr], **({'clf__sample_weight': w} if w is not None else {}))
                except Exception:
                    p.fit(X[tr], df.label.to_numpy()[tr])
                oof[va] = p.predict_proba(X[va])[:,1]
            m = benchmark._metrics(df.label.to_numpy(), oof)
            rows.append({'feature_set': fs_name, 'model': model_name, **{f'OOF_{k}': v for k,v in m.items()}})
    result = pd.DataFrame(rows).sort_values('OOF_MCC', ascending=False).reset_index(drop=True)
    result.to_csv(out/'cluster_benchmark_results.csv', index=False)
    report = {
        'seed': SEED,
        'n_compounds': int(len(df)),
        'n_clusters': int(args.clusters),
        'folds': int(args.folds),
        'n_init': int(args.n_init),
        'group_overlap_check': [len(set(cv[i][1]).intersection(set(cv[j][1]))) for i in range(len(cv)) for j in range(i+1,len(cv))],
        'best_model': result.iloc[0][['feature_set','model']].to_dict(),
        'best_OOF_MCC': float(result.iloc[0]['OOF_MCC']),
        'best_OOF_ROC_AUC': float(result.iloc[0]['OOF_ROC-AUC']),
    }
    (out/'cluster_split_summary.json').write_text(json.dumps(report, indent=2))
    df[['molecule_chembl_id','label','cluster_id']].to_csv(out/'cluster_assignments.csv', index=False)
    print(result.head(10).to_string(index=False))
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
