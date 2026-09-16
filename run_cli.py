"""
Command-line runner (no GUI) for scripted / batch use.

Examples
--------
# From a ChEMBL target ID (requires internet):
python run_cli.py --target CHEMBL217 --out results/

# From your own CSV:
python run_cli.py --csv my_activities.csv --out results/

# The offline synthetic demo:
python run_cli.py --csv sample_data/synthetic_activities.csv --out results/
"""

import argparse
import json
import os

from ligandbench import chembl_data
from ligandbench.pipeline import run_pipeline, PipelineConfig


def main():
    ap = argparse.ArgumentParser(description="LigandBench CLI")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--target", help="ChEMBL target ID, e.g. CHEMBL217")
    src.add_argument("--csv", help="Path to an activity CSV (SMILES + value)")
    ap.add_argument("--out", default="ligandbench_out", help="Output directory")
    ap.add_argument("--active-cut", type=float, default=6.0)
    ap.add_argument("--inactive-cut", type=float, default=5.0)
    ap.add_argument("--test-fraction", type=float, default=0.2)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--permutations", type=int, default=20)
    ap.add_argument("--no-smote", action="store_true")
    ap.add_argument("--max-records", type=int, default=0, help="0 = all")
    ap.add_argument("--mode", default="all",
                    choices=["all", "agonist", "antagonist", "binding",
                             "agonist_vs_antagonist"],
                    help="Class definition. 'agonist_vs_antagonist' classifies "
                         "agonists vs antagonists (balanced) — use for OX2R.")
    ap.add_argument("--report", action="store_true",
                    help="Also generate publication figures and tables (no manuscript generation)")
    ap.add_argument("--target-label", default=None, help="Optional display label for the target; accepted for manuscript reproducibility commands.")
    args = ap.parse_args()

    if args.target:
        print(f"Fetching {args.target} from ChEMBL ...")
        raw = chembl_data.fetch_activities(
            args.target, max_records=(args.max_records or None),
            progress=lambda n, m: print(f"  {m}"),
        )
    else:
        raw = chembl_data.load_activity_csv(args.csv)
    print(f"Loaded {len(raw)} activity rows.")

    cfg = PipelineConfig(
        activity_mode=args.mode,
        active_cut=args.active_cut, inactive_cut=args.inactive_cut,
        test_fraction=args.test_fraction, n_splits=args.folds,
        n_permutations=args.permutations, use_smote=not args.no_smote,
    )
    res = run_pipeline(raw, cfg)

    os.makedirs(args.out, exist_ok=True)
    if not res.results.empty:
        res.results.to_csv(os.path.join(args.out, "benchmark_results.csv"), index=False)
    if not res.curated.empty:
        res.curated.to_csv(os.path.join(args.out, "curated_dataset.csv"), index=False)
    if res.shap_imp is not None and not res.shap_imp.empty:
        res.shap_imp.to_csv(os.path.join(args.out, "shap_importances.csv"), index=False)

    summary = {
        "best_model": list(res.best_key) if res.best_key else None,
        "split_report": res.split_report,
        "label_summary": res.label_summary,
        "curation_summary": res.curation_summary,
        "y_randomization": {k: v for k, v in res.yrand.items() if k != "permuted_aucs"},
        "applicability_domain": res.ad_summary,
    }
    with open(os.path.join(args.out, "run_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    if args.report and not res.results.empty:
        from ligandbench import reporting
        print("Generating scientific figures and tables ...")
        reporting.generate_report(res, args.out)
        print(f"  figures/ + tables/ written to {args.out}/")

    print("\n=== Best model ===", res.best_key)
    if not res.results.empty:
        cols = [c for c in ["feature_set", "model", "OOF_ROC-AUC", "OOF_MCC",
                            "Test_ROC-AUC", "Test_MCC"] if c in res.results.columns]
        print(res.results[cols].head(6).to_string(index=False))
    print(f"\nWrote outputs to {args.out}/")


if __name__ == "__main__":
    main()
