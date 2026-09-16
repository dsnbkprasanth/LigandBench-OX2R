"""Run the primary benchmark directly from the frozen curated OX2R dataset."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from ligandbench import benchmark, features, splitting

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input',default='data/ox2r_curated_6415.csv')
    ap.add_argument('--out',default='results/frozen_benchmark')
    args=ap.parse_args(); out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    df=splitting.add_scaffolds(pd.read_csv(args.input)).reset_index(drop=True); df['_row']=np.arange(len(df))
    dev,test=splitting.scaffold_grouped_test_split(df,0.2,42)
    fs,needs,_=features.build_feature_sets(df)
    tr=dev['_row'].to_numpy(); te=test['_row'].to_numpy()
    fdev={k:(X[tr],n) for k,(X,n) in fs.items()}; ftest={k:(X[te],n) for k,(X,n) in fs.items()}
    cv,_=splitting.grouped_cv_indices(dev,5)
    res,oof,best=benchmark.run_benchmark(dev,test,fdev,ftest,needs,cv,models=benchmark.available_models(True),seed=42)
    res.to_csv(out/'benchmark_results.csv',index=False)
    summary={'best_model':list(best),'n_dev':len(dev),'n_test':len(test),'seed':42,'split_report':splitting.split_report(dev,test)}
    (out/'summary.json').write_text(json.dumps(summary,indent=2,default=str))
    print(res.head(10).to_string(index=False)); print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
