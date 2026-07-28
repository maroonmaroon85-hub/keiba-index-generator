import os, json, sys
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score
sys.path.insert(0,"ml")
import features as F
q=float(sys.argv[1]) if len(sys.argv)>1 else 0.3
d=F.to_model(F.load_files())
f=F.build_features(d)
keep=f["n_prior"]>=1
d,f=d[keep].reset_index(drop=True),f[keep].reset_index(drop=True)
y=(d["finish"]==1).astype(int)
cut=d["date"].quantile(q)
tr,te=(d["date"]<cut).values,(d["date"]>=cut).values
fx,maps=F.encode_categoricals(f); cols=list(fx.columns)
print(f"分割 {q:.0%}: train {tr.sum():,} / test {te.sum():,}  分割日 {cut.date()}")
m=lgb.LGBMClassifier(n_estimators=400,learning_rate=0.03,num_leaves=31,subsample=0.8,
                     colsample_bytree=0.8,min_child_samples=100,verbose=-1)
m.fit(fx[tr],y[tr],categorical_feature=F.CAT_COLS)
d=d.copy(); d["pred"]=m.predict_proba(fx)[:,1]
auc=roc_auc_score(y[te],d["pred"][te])
dte=d[te].copy(); dte["prob"]=dte["pred"]/dte.groupby("raceid")["pred"].transform("sum")
top=dte.loc[dte.groupby("raceid")["prob"].idxmax()]
print(f"  AUC {auc:.4f}  top複勝 {(top['finish']<=3).mean()*100:.1f}%")
dte[["raceid","umaban","prob","finish","odds"]].to_csv("out/ml_test_pred_early.csv",index=False)
print("  → out/ml_test_pred_early.csv")
