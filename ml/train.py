"""
本番学習パイプライン（再利用可能）。
DS*.CSV 全部で LightGBM を学習し、モデル＋カテゴリマップを ml/model/ に保存。
アウトオブサンプル(日付分割)の指標も出す。データが増えたら `python3 ml/train.py` を再実行するだけ。

出力:
  ml/model/model.txt        学習済みLightGBM
  ml/model/cat_maps.json    カテゴリ→整数コードの対応
  ml/model/feature_cols.json 特徴量の列順
  out/ml_test_pred.csv      test期間の予測（raceid,umaban,pred,finish,odds）
"""
import os, json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import features as F

MODEL_DIR = "ml/model"

def main():
    d = F.to_model(F.load_files())
    f = F.build_features(d)
    keep = f["n_prior"] >= 1
    d, f = d[keep].reset_index(drop=True), f[keep].reset_index(drop=True)
    y = (d["finish"] == 1).astype(int)

    cut = d["date"].quantile(0.7)
    tr, te = (d["date"] < cut).values, (d["date"] >= cut).values
    fx, cat_maps = F.encode_categoricals(f)
    cols = list(fx.columns)
    print(f"train {tr.sum()} / test {te.sum()}  分割日 {cut.date()}  勝率(base) {y.mean():.3f}")

    model = lgb.LGBMClassifier(
        n_estimators=400, learning_rate=0.03, num_leaves=31,
        subsample=0.8, colsample_bytree=0.8, min_child_samples=100, verbose=-1,
    )
    model.fit(fx[tr], y[tr], categorical_feature=F.CAT_COLS)

    d = d.copy()
    d["pred"] = model.predict_proba(fx)[:, 1]
    auc = roc_auc_score(y[te], d["pred"][te])
    dte = d[te].copy()
    dte["prob"] = dte["pred"] / dte.groupby("raceid")["pred"].transform("sum")
    top = dte.loc[dte.groupby("raceid")["prob"].idxmax()]
    ev = dte[(dte["odds"] > 0) & (dte["prob"] >= 1 / dte["fieldsize"])].copy()
    ev["ev"] = ev["prob"] * ev["odds"]
    bet = ev[ev["ev"] >= 1.0]
    roi = (bet.loc[bet["finish"] == 1, "odds"].sum() * 100) / (len(bet) * 100) * 100 if len(bet) else 0

    print("\n===== アウトオブサンプル =====")
    print(f"AUC                 : {auc:.4f}")
    print(f"top-pick 複勝率     : {(top['finish']<=3).mean()*100:.1f}%")
    print(f"top-pick 勝率       : {(top['finish']==1).mean()*100:.1f}%")
    print(f"単勝EV≥1 回収率     : {roi:.1f}%  ({len(bet)}点)")

    os.makedirs(MODEL_DIR, exist_ok=True)
    model.booster_.save_model(f"{MODEL_DIR}/model.txt")
    json.dump(cat_maps, open(f"{MODEL_DIR}/cat_maps.json", "w"), ensure_ascii=False)
    json.dump(cols, open(f"{MODEL_DIR}/feature_cols.json", "w"), ensure_ascii=False)
    os.makedirs("out", exist_ok=True)
    dte[["raceid", "umaban", "prob", "finish", "odds"]].to_csv("out/ml_test_pred.csv", index=False)
    print(f"\n保存: {MODEL_DIR}/model.txt, cat_maps.json, feature_cols.json / out/ml_test_pred.csv")

if __name__ == "__main__":
    main()
