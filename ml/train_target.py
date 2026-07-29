"""
軸①の本命: **学習目標を買い方に合わせる**。

現行モデルの目標は「1着になる確率」だが、実際の買い方が要求するのは1着ではない:
  - 三連単 一軸マルチ×紐n … 軸に必要なのは **3着以内**（1着でなくてよいのがマルチの定義）
  - 馬連 軸流し×紐n      … 軸に必要なのは **2着以内**
＝**買い方と学習目標がずれている**。1着の精度を上げても、マルチでは1着であることの旨みは使われない。

そこで目標を win / top2 / top3 に差し替えて学習し、同じ買い方で連系の回収率を比べる。
特徴量・分割・ハイパーパラメータは一切変えず、**目標だけ**を変える（差分を目標の効果として切り出すため）。

出力: out/ml_pred_{win,top2,top3}.csv （himo_sweep.py / filter_search.py がそのまま読める形式）
実行: python3 ml/train_target.py [分割位置(既定0.3)]
"""
import sys

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "ml")
import features as F

TARGETS = {"win": 1, "top2": 2, "top3": 3}


def main():
    q = float(sys.argv[1]) if len(sys.argv) > 1 else 0.3
    d = F.to_model(F.load_files())
    f = F.build_features(d)
    keep = f["n_prior"] >= 1
    d, f = d[keep].reset_index(drop=True), f[keep].reset_index(drop=True)

    cut = d["date"].quantile(q)
    tr, te = (d["date"] < cut).to_numpy(), (d["date"] >= cut).to_numpy()
    fx, _ = F.encode_categoricals(f)
    print(f"分割 {q:.0%}: train {tr.sum():,} / test {te.sum():,}  分割日 {cut.date()}")

    common = dict(n_estimators=400, learning_rate=0.03, num_leaves=31, subsample=0.8,
                  colsample_bytree=0.8, min_child_samples=100, verbose=-1)

    for name, k in TARGETS.items():
        y = (d["finish"] <= k).astype(int).to_numpy()
        m = lgb.LGBMClassifier(**common)
        m.fit(fx[tr], y[tr], categorical_feature=F.CAT_COLS)
        p = m.predict_proba(fx)[:, 1]
        auc = roc_auc_score(y[te], p[te])

        out = d.loc[te, ["raceid", "umaban", "finish", "odds"]].copy()
        out["prob"] = p[te]
        # レース内で合計1に正規化（順位は不変。gap等の指標を既存スクリプトと揃えるため）
        out["prob"] = out["prob"] / out.groupby("raceid")["prob"].transform("sum")
        path = f"out/ml_pred_{name}.csv"
        out[["raceid", "umaban", "prob", "finish", "odds"]].to_csv(path, index=False)

        # 参考: このモデルで各レース1位に選ばれた馬が、実際に k着以内に入る率
        top = out.loc[out.groupby("raceid")["prob"].idxmax()]
        print(f"  目標={name:<5} (≤{k}着)  OOS AUC {auc:.4f}   "
              f"1位指名馬の≤{k}着率 {(top['finish'] <= k).mean() * 100:.1f}%   → {path}")


if __name__ == "__main__":
    main()
