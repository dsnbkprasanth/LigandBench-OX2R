# LigandBench

**A target-agnostic, scaffold-aware ligand-based machine-learning benchmark with reproducible analysis scripts and a GUI.**

LigandBench reproduces the workflow of Drewe, *"Morgan fingerprints outperform
physicochemical descriptors for scaffold-aware machine learning classification
of β3-adrenergic receptor agonists"* (Journal of Cheminformatics, 2026), and
generalizes it so the **same flow runs for any target**. Type a target name,
click Run, and get the full benchmark — curated dataset, five feature sets, six
classifiers under scaffold-grouped cross-validation, a held-out scaffold test
set, Y-randomization, SHAP, and an applicability-domain check.

---

## The pipeline (same nine-stage flow as the paper)

| Stage | Module | What it does |
|------|--------|--------------|
| 00 Data | `chembl_data.py` | Fetch activities from the ChEMBL REST API for any target, or load your own CSV |
| 01 Labeling | `labeling.py` | Convert to pActivity, assign per-compound **consensus** active/inactive labels with confidence weights |
| 02 Curation | `curation.py` | RDKit standardization, sanity filters (MW, heavy atoms, TPSA, rotatable bonds, no metals), dedup + near-duplicate removal |
| 03 Features | `features.py` | ECFP4, ECFP6, RDKit descriptor panel, and the two hybrids (2048-bit, chirality on) |
| 04 Split + benchmark | `splitting.py`, `benchmark.py` | Bemis–Murcko **scaffold-grouped** test split + grouped k-fold CV; RF, SVM-RBF, XGBoost, LightGBM, RF+SMOTE, XGBoost+SMOTE |
| 07 Validation | `validation.py` | Y-randomization (permutation null) + SHAP feature importance on the best model |
| 08 Applicability | `applicability.py` | Max-Tanimoto-to-training applicability domain |
| — Orchestrator | `pipeline.py` | Runs the whole thing end to end |
| — GUI | `app.py` | Streamlit front end tying it all together |

Metrics reported per model: ROC-AUC, PR-AUC, Brier, balanced accuracy, MCC,
sensitivity, specificity, precision, F1. **Best model is selected by
out-of-fold MCC** (as in the paper).

---

## Install

```bash
python -m venv venv && source venv/bin/activate      # optional but recommended
pip install -r requirements.txt
```

RDKit installs from PyPI wheels on modern Python (3.9–3.12). If it fails, use
conda: `conda install -c conda-forge rdkit`.

## Run the GUI

```bash
streamlit run app.py
```

Then in the browser:

1. **Pick a data source** — ChEMBL target search, upload a CSV, or the built-in
   *Synthetic demo* (runs fully offline).
2. **Tune the config** (labeling cut-offs, curation filters, feature sets,
   classifiers, CV folds, permutations) — sensible defaults match the paper.
3. **Run pipeline** and read the results across the tabs: Overview, Data &
   labels, Curation & split, Benchmark, Best model, Validation, Export.

### Try it without internet

Choose **Synthetic demo** in the sidebar (or `--csv sample_data/synthetic_activities.csv`
on the CLI). It's a 600-compound toy set with a real, learnable structural
signal, so every stage — including SHAP and Y-randomization — has something to
show.

## Run headless / batch (no GUI)

```bash
# any ChEMBL target (needs internet)
python run_cli.py --target CHEMBL217 --out results/

# your own data
python run_cli.py --csv my_activities.csv --out results/

# agonist-only model + scientific figures/tables
python run_cli.py --target CHEMBL4792 --mode agonist --report --out results/
```

Outputs: `benchmark_results.csv`, `curated_dataset.csv`, `shap_importances.csv`,
`run_summary.json`; with `--report` also scientific `figures/` and `tables/`.

## Activity-direction filter (agonist vs antagonist vs binding)

The generic potency labeling cannot tell an agonist from an antagonist. For a
target whose ChEMBL data is dominated by antagonists/binding (e.g. **orexin
OX2R, CHEMBL4792**), a default run would put potent *antagonists* into the
"active" class. Use the **Activity direction** control (sidebar) or `--mode`
(CLI) to keep only `agonist` (or `antagonist` / `binding`) records *before*
labeling, so the active class reflects the pharmacology you intend. Direction is
inferred from measurement type and assay-description text; check the per-mode
tally reported in the run.

## Bring your own data (CSV format)

The only required column is a **SMILES** column plus a **measurement**. Common
column names are auto-detected:

| Concept | Accepted column names |
|---------|----------------------|
| Structure | `smiles`, `canonical_smiles` |
| Compound id | `molecule_chembl_id`, `compound_id`, `id` |
| Activity type | `standard_type`, `activity_type` |
| Value | `standard_value`, `value` |
| Units | `standard_units`, `units` (M/mM/uM/nM/pM) |
| Potency (optional) | `pchembl_value`, `pchembl`, `pactivity` |

If a `pchembl_value` is present it's used directly; otherwise value+units are
converted to a pActivity scale.

---

## Notes & caveats

- **Labeling is generalized** to a potency-threshold consensus (active vs
  inactive). The paper used agonist-specific assay-direction inference; if your
  target needs functional-direction labels (agonist vs antagonist), encode that
  in your input CSV or extend `labeling.py`.
- Network use is confined to `chembl_data.py`. Everything else runs offline.
- Results are reproducible from a fixed seed + config.
- Not a substitute for domain review — curation thresholds and cut-offs should
  be justified per target.

## License

Provided as-is for research use. RDKit, scikit-learn, XGBoost, LightGBM, SHAP,
and imbalanced-learn retain their own licenses.

---

## Reproduce the OX2R agonist-vs-antagonist findings (this paper)

Frozen data and precomputed results are included:

- `ox2_agvsan_v2/` — main run: `tables/curated_dataset.csv` (6,415 compounds: 747
  agonists, 5,668 antagonists), `benchmark_results.csv` (all 30 model/representation
  combinations), `shap_importances.csv`, `run_summary.json`, and `figures/`.
- `ox2_extra/` — robustness checks: `highconf_benchmark.csv` /
  `highconf_summary.json` (high-confidence subset labelled solely by ChEMBL
  `action_type`, n=501), `yrand.json` (Y-randomization, 300 permutations), and
  `top_ecfp4_bits.png` (substructures of the top SHAP fingerprint bits).

Regenerate everything end to end from ChEMBL (needs internet; seed fixed at 42):

```bash
python run_cli.py --target CHEMBL4792 --mode agonist_vs_antagonist \
    --permutations 300 --report --target-label "OX2R" --out ox2_repro
```

Robustness checks on the frozen curated dataset:

```bash
python reviewer_extra.py          # high-confidence subset, Y-randomization x300, bit images
python score_external.py external.csv   # optional external validation (smiles,label)
```

All randomized steps use a fixed random seed of 42. Source bioactivity data are
from ChEMBL version 34 (target CHEMBL4792).
