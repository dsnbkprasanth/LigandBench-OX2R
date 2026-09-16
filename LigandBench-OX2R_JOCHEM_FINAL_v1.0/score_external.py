# OPTIONAL external validation. You build external.csv by hand from RECENT OX2R papers
# NOT in ChEMBL training: two columns  smiles,label   (label = agonist or antagonist).
# Then run:  python score_external.py external.csv
import sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")
from ligandbench import features, splitting, benchmark, curation
from sklearn.metrics import matthews_corrcoef, roc_auc_score
SEED=42
cur=pd.read_csv("ox2_agvsan_v2/tables/curated_dataset.csv")
ext=pd.read_csv(sys.argv[1]); ext["label"]=(ext["label"].str.lower().str.startswith("ago")).astype(int)
ext=ext.rename(columns={"smiles":"canonical_smiles"})
ext["molecule_chembl_id"]=[f"EXT{i}" for i in range(len(ext))]; ext["label_class"]=np.where(ext.label==1,"agonist","antagonist")
ext["sample_weight"]=1.0; ext["median_pactivity"]=np.nan
alld=pd.concat([cur.assign(_src="train"),ext.assign(_src="ext")],ignore_index=True)
fs,needs,_=features.build_feature_sets(alld)
tr=alld._src=="train"; te=alld._src=="ext"
X,_=fs["ECFP4+RDKit"]
pipe=benchmark.fit_final_model(X[tr.values],alld.loc[tr,"label"].to_numpy(),"LightGBM",needs["ECFP4+RDKit"],seed=SEED)
p=pipe.predict_proba(X[te.values])[:,1]; yt=alld.loc[te,"label"].to_numpy()
print("External n=",len(yt),"MCC",round(matthews_corrcoef(yt,(p>=.5).astype(int)),3),"ROC-AUC",round(roc_auc_score(yt,p),3))
