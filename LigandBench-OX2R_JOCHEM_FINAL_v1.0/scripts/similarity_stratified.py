"""Reproduce the manuscript's chemical-similarity stratification.

Uses the fixed scaffold-grouped split (seed 42), the selected ECFP4+RDKit
LightGBM model, maximum ECFP4 Tanimoto similarity to the development set,
and bootstrap 95% CIs for MCC in prespecified similarity bins.
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
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from sklearn.metrics import matthews_corrcoef

from ligandbench import benchmark, features, splitting

SEED = 42
BINS = [0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0000001]
FP_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=True)


def fps(smiles):
    return [FP_GEN.GetFingerprint(Chem.MolFromSmiles(s)) for s in smiles]


def max_similarity(train_smiles, query_smiles):
    train = fps(train_smiles)
    out = []
    for s in query_smiles:
        fp = FP_GEN.GetFingerprint(Chem.MolFromSmiles(s))
        sims = DataStructs.BulkTanimotoSimilarity(fp, train)
        out.append(float(max(sims)) if sims else np.nan)
    return np.asarray(out)


def bootstrap_mcc(y, p, n=2000, seed=SEED):
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    p = np.asarray(p)
    vals = np.empty(n, dtype=float)
    for i in range(n):
        idx = rng.integers(0, len(y), len(y))
        yp = (p[idx] >= 0.5).astype(int)
        vals[i] = matthews_corrcoef(y[idx], yp)
    return float(np.nanmedian(vals)), float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', default='data/ox2r_curated_6415.csv')
    ap.add_argument('--out', default='results/similarity_stratified')
    ap.add_argument('--bootstrap', type=int, default=2000)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    df = splitting.add_scaffolds(df).reset_index(drop=True)
    df['_row'] = np.arange(len(df))
    dev, test = splitting.scaffold_grouped_test_split(df, test_fraction=0.20, seed=SEED)

    fs_all, needs, _ = features.build_feature_sets(df)
    tr = dev['_row'].to_numpy(); te = test['_row'].to_numpy()
    Xtr = fs_all['ECFP4+RDKit'][0][tr]
    Xte = fs_all['ECFP4+RDKit'][0][te]
    model = benchmark.fit_final_model(Xtr, dev['label'].to_numpy(), 'LightGBM', True, seed=SEED)
    prob = model.predict_proba(Xte)[:, 1]
    y = test['label'].to_numpy()
    sim = max_similarity(dev['canonical_smiles'].tolist(), test['canonical_smiles'].tolist())

    rows = []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        mask = (sim >= lo) & (sim < hi)
        if not mask.any():
            continue
        med, lo_ci, hi_ci = bootstrap_mcc(y[mask], prob[mask], n=args.bootstrap, seed=SEED)
        rows.append({
            'similarity_bin': f'{lo:.1f}-{min(hi,1.0):.1f}',
            'lower': lo,
            'upper': min(hi,1.0),
            'n_compounds': int(mask.sum()),
            'n_agonists': int(y[mask].sum()),
            'mcc': float(matthews_corrcoef(y[mask], (prob[mask] >= .5).astype(int))),
            'bootstrap_median_mcc': med,
            'ci95_lower': lo_ci,
            'ci95_upper': hi_ci,
        })
    result = pd.DataFrame(rows)
    result.to_csv(out/'similarity_bins.csv', index=False)

    # Cross-class-proximal subset described in the manuscript.
    dev_ag = fps(dev.loc[dev.label == 1, 'canonical_smiles'].tolist())
    dev_an = fps(dev.loc[dev.label == 0, 'canonical_smiles'].tolist())
    proximal = []
    for s, lab in zip(test.canonical_smiles, y):
        fp = FP_GEN.GetFingerprint(Chem.MolFromSmiles(s))
        opposite = dev_an if lab == 1 else dev_ag
        mx = max(DataStructs.BulkTanimotoSimilarity(fp, opposite)) if opposite else 0.0
        proximal.append(mx >= 0.5)
    proximal = np.asarray(proximal)
    prox = {
        'threshold': 0.5,
        'n_compounds': int(proximal.sum()),
        'n_correct': int(((prob[proximal] >= .5).astype(int) == y[proximal]).sum()),
    }
    (out/'cross_class_proximal.json').write_text(json.dumps(prox, indent=2))

    meta = {
        'seed': SEED,
        'bootstrap': args.bootstrap,
        'split': 'Bemis-Murcko scaffold grouped, test_fraction=0.20',
        'model': 'ECFP4+RDKit / LightGBM',
        'similarity': 'maximum ECFP4 Tanimoto to development set',
        'n_test': int(len(test)),
    }
    (out/'metadata.json').write_text(json.dumps(meta, indent=2))
    print(result.to_string(index=False))
    print(json.dumps(prox, indent=2))

if __name__ == '__main__':
    main()
