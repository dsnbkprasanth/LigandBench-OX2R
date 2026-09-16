# Run from the LigandBench folder in the `ligandbench` conda env:
#     python reviewer_extra.py
# Produces ox2_extra/ with: top ECFP4 bit images, high-confidence-subset
# benchmark, and Y-randomisation (with LIVE progress + incremental save).
#
# NOTE ON SPEED: the Y-randomisation step is the slow one. It fits the model
# n_permutations x n_folds times. With the defaults below (300 perms x 3 folds
# = 900 fits) it takes a while but PRINTS PROGRESS every few permutations and
# writes ox2_extra/yrand.json as it goes, so you always know it is alive.
# Override from the shell if you like:
#     YRAND_PERMS=500 YRAND_FOLDS=5 python reviewer_extra.py   # thorough
#     YRAND_PERMS=100 YRAND_FOLDS=3 python reviewer_extra.py   # quick
import os, json, time, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import AllChem, Draw
RDLogger.DisableLog("rdApp.*")
from ligandbench import chembl_data, curation, features, splitting, benchmark, validation
from sklearn.metrics import matthews_corrcoef, roc_auc_score

OUT = "ox2_extra"; os.makedirs(OUT, exist_ok=True); SEED = 42
CUR = "ox2_agvsan_v2/tables/curated_dataset.csv"   # existing main curated set

PERMS = int(os.environ.get("YRAND_PERMS", "300"))
FOLDS = int(os.environ.get("YRAND_FOLDS", "3"))

# Featurise the main set once; every step below reuses it.
print("[setup] loading + featurising main curated set ...", flush=True)
cur = pd.read_csv(CUR); cur = splitting.add_scaffolds(cur).reset_index(drop=True)
cur["_row"] = range(len(cur))
dev, _ = splitting.scaffold_grouped_test_split(cur, test_fraction=0.2, seed=SEED)
fs, needs, _ = features.build_feature_sets(cur)
print(f"   curated {len(cur)} | dev {len(dev)} | features ready", flush=True)

# ---------- 1) Top ECFP4 bit substructure images (FAST — runs first) ----------
print("[1/3] Drawing top ECFP4 bits ...", flush=True)
try:
    shap_bits = [b for b in pd.read_csv("ox2_agvsan_v2/tables/shap_importances.csv")["feature"]
                 if str(b).startswith("ecfp4_")][:6]
    bits = [int(b.split("_")[1]) for b in shap_bits]
    exmpl = []
    all_smiles = cur[cur.label == 1]["canonical_smiles"].tolist() + \
                 cur[cur.label == 0]["canonical_smiles"].tolist()
    for bit in bits:
        for smi in all_smiles:
            m = Chem.MolFromSmiles(smi); bi = {}
            if m is None: continue
            AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048, bitInfo=bi)
            if bit in bi:
                exmpl.append((m, bit, bi)); break
    if exmpl:
        img = Draw.DrawMorganBits(exmpl, molsPerRow=3,
                                  legends=[f"ecfp4_{b}" for (_, b, _) in exmpl],
                                  subImgSize=(260, 220))
        img.save(f"{OUT}/top_ecfp4_bits.png")
        print("   saved ox2_extra/top_ecfp4_bits.png", flush=True)
except Exception as e:
    print("   bit drawing skipped:", e, flush=True)

# ---------- 2) High-confidence subset (ChEMBL action_type only) ----------
print("[2/3] High-confidence subset (re-fetching CHEMBL4792) ...", flush=True)
raw = chembl_data.fetch_activities("CHEMBL4792")
raw["at"] = raw["action_type"].astype(str).str.upper()
raw = raw[raw["at"].str.contains("AGONIST|ANTAGONIST", na=False)].copy()
raw["dir"] = np.where(raw["at"].str.contains("ANTAGONIST"), "antagonist", "agonist")
rows = []
for mid, g in raw.groupby("molecule_chembl_id"):
    na = (g["dir"] == "agonist").sum(); nn = (g["dir"] == "antagonist").sum()
    if na == nn: continue
    rows.append({"molecule_chembl_id": mid,
                 "canonical_smiles": g["canonical_smiles"].dropna().iloc[0],
                 "label": 1 if na > nn else 0,
                 "label_class": "agonist" if na > nn else "antagonist",
                 "sample_weight": 1.0, "median_pactivity": np.nan})
hc = pd.DataFrame(rows)
print("   high-confidence labelled:", len(hc),
      "| agonists", int((hc.label == 1).sum()),
      "antagonists", int((hc.label == 0).sum()), flush=True)
hc = curation.curate(hc)
res = {"n_curated": int(len(hc)), "n_agonist": int((hc.label == 1).sum()),
       "n_antagonist": int((hc.label == 0).sum())}
if len(hc) >= 100 and hc["label"].nunique() == 2:
    hc = splitting.add_scaffolds(hc).reset_index(drop=True); hc["_row"] = range(len(hc))
    d2, t2 = splitting.scaffold_grouped_test_split(hc, test_fraction=0.2, seed=SEED)
    fsa, nd, _ = features.build_feature_sets(hc)
    fdev = {k: (v[0][d2["_row"].to_numpy()], v[1]) for k, v in fsa.items()}
    ftst = {k: (v[0][t2["_row"].to_numpy()], v[1]) for k, v in fsa.items()}
    cvi, _ = splitting.grouped_cv_indices(d2, n_splits=5)
    tab, _, best = benchmark.run_benchmark(d2, t2, fdev, ftst, nd, cvi,
                                           models=benchmark.available_models(True), seed=SEED)
    tab.to_csv(f"{OUT}/highconf_benchmark.csv", index=False)
    hc.to_csv(f"{OUT}/highconf_curated.csv", index=False)
    br = tab.sort_values("OOF_MCC", ascending=False).iloc[0]
    res.update({"best": [br["feature_set"], br["model"]],
                "OOF_MCC": round(float(br["OOF_MCC"]), 3),
                "Test_MCC": round(float(br.get("Test_MCC", float('nan'))), 3)})
    print("   high-confidence best OOF MCC", res["OOF_MCC"],
          "Test MCC", res.get("Test_MCC"), flush=True)
else:
    print("   too few high-confidence compounds for a benchmark; counts saved.", flush=True)
json.dump(res, open(f"{OUT}/highconf_summary.json", "w"), indent=2)

# ---------- 3) Y-randomisation (SLOW — runs last, with live progress) ----------
print(f"[3/3] Y-randomisation: {PERMS} permutations x {FOLDS} folds "
      f"= {PERMS*FOLDS} fits. Progress below.", flush=True)
X = fs["ECFP4+RDKit"][0][dev["_row"].to_numpy()]; y = dev["label"].to_numpy()
cv, _ = splitting.grouped_cv_indices(dev, n_splits=FOLDS)

_t0 = time.time()
def _progress(i, total):
    # called once per permutation; print a heartbeat so you know it is alive
    if i == 1 or i % 5 == 0 or i == total:
        el = time.time() - _t0
        eta = el / i * (total - i)
        print(f"   perm {i}/{total}  elapsed={el:.0f}s  eta~{eta:.0f}s", flush=True)

yr = validation.y_randomization(X, y, cv, "LightGBM", needs["ECFP4+RDKit"],
                                n_permutations=PERMS, seed=SEED, progress=_progress)
json.dump({k: v for k, v in yr.items()}, open(f"{OUT}/yrand.json", "w"))
print("   real", round(yr["real_auc"], 3),
      "permuted_mean", round(yr["permuted_mean"], 3),
      "p", round(yr["p_value"], 5), flush=True)

print("\nDONE. Outputs in ox2_extra/. Tell Claude it is done.")
