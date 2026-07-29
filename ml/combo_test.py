"""
(33)で事前指定した5軸のうち、ノイズを超えた効果を**重ねる**検定。

(33)の結果、単体でノイズ幅を超えたのは 多頭数(単調)／継続騎乗(前後半一致)／短距離／3勝-OP の4つ。
そのうち最も頑健な2つ（16頭〜・継続騎乗）を土台に、**3通りだけ**を事前宣言して測る。
組み合わせを総当たりすると(28)以前の探索に逆戻りするため、通り数を絞ることが規律の中身。

事前宣言（2026-07-29・事後に追加しない）:
  A: 16頭〜 & 継続騎乗
  B: A & 〜1200m
  C: A & 3勝-OP

判定は(30)(32)の基準どおり: シード幅を並記し、**前後半の両方で100%超**でなければ採用しない。
標本が小さくなるほどノイズ床が上がる（(30): 945Rで±20pt超）ので、R数と幅を必ず見ること。

実行: python3 ml/combo_test.py [シード数(既定3)]
"""
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, "ml")
import features as F
from himo_sweep import POINTS, KEY, hits
from pocket_eval import load_payout_a

TNAME, NHIMO = "三連単マルチ", 7
SPLIT_YEAR = 2022


def conds(df):
    """事前宣言した条件。ここに後から足さないこと。"""
    big = df["fieldsize"] >= 16
    keep_j = ~df["jockey_changed"].astype(bool)
    sprint = df["distance"] <= 1200
    upper = (df["raceclass"] >= 4) & (df["raceclass"] <= 5)  # 3勝-OP
    a = big & keep_j
    return {
        "全体": pd.Series(True, index=df.index),
        "単体: 16頭〜": big,
        "単体: 継続騎乗": keep_j,
        "A: 16頭〜&継続": a,
        "B: A&〜1200m": a & sprint,
        "C: A&3勝-OP": a & upper,
    }


def main():
    n_seed = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    d = F.to_model(F.load_files())
    f = F.build_features(d)
    keep = f["n_prior"] >= 1
    d, f = d[keep].reset_index(drop=True), f[keep].reset_index(drop=True)
    d["prev_jockey"] = d.groupby("horse")["jockey"].shift(1)
    d["jockey_changed"] = (d["prev_jockey"].notna()) & (d["jockey"] != d["prev_jockey"])

    y = (d["finish"] == 1).astype(int).to_numpy()
    cut = d["date"].quantile(0.3)
    tr, te = (d["date"] < cut).to_numpy(), (d["date"] >= cut).to_numpy()
    fx, _ = F.encode_categoricals(f)
    pays = load_payout_a("data/payout/a.csv")

    meta = d.loc[te, ["raceid", "umaban", "jockey_changed"]].copy()
    for c in ["fieldsize", "raceclass", "distance"]:
        meta[c] = f.loc[te, c].to_numpy()
    print(f"train {tr.sum():,} / test {te.sum():,}  シード{n_seed}本  戦略={TNAME}×紐{NHIMO}")

    pts = POINTS[TNAME](NHIMO) * 100
    res = defaultdict(list)
    cnts = {}
    for seed in range(n_seed):
        m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=31, subsample=0.8,
                               colsample_bytree=0.8, min_child_samples=100, verbose=-1,
                               random_state=seed)
        m.fit(fx[tr], y[tr], categorical_feature=F.CAT_COLS)
        pr = meta.copy()
        pr["prob"] = m.predict_proba(fx[te])[:, 1]
        rows = []
        for rid, g in pr.groupby("raceid", sort=False):
            p = pays.get(rid)
            if p is None or not p["sanrentan"] or len(g) <= NHIMO:
                continue
            g = g.sort_values("prob", ascending=False, kind="mergesort")
            nums = g["umaban"].astype(int).tolist()
            a = g.iloc[0]
            rows.append({"year": p["date"].year, "pay": hits(p[KEY[TNAME]], nums[0], set(nums[1:NHIMO + 1])),
                         "fieldsize": a["fieldsize"], "raceclass": a["raceclass"],
                         "distance": a["distance"], "jockey_changed": a["jockey_changed"]})
        df = pd.DataFrame(rows)

        def roi(sub):
            return sub["pay"].sum() / (len(sub) * pts) * 100 if len(sub) else float("nan")

        for name, mask in conds(df).items():
            sub = df[mask.to_numpy()]
            cnts[name] = len(sub)
            res[(name, "all")].append(roi(sub))
            res[(name, "1st")].append(roi(sub[sub["year"] < SPLIT_YEAR]))
            res[(name, "2nd")].append(roi(sub[sub["year"] >= SPLIT_YEAR]))
        print(f"  seed {seed} 完了")

    print("\n" + "=" * 78)
    print(f"★事前宣言した組み合わせの検定  {TNAME}×紐{NHIMO}")
    print("=" * 78)
    print(f"{'条件':<20}{'R':>7}{'ROI':>8}{'シード幅':>9}{'前半':>9}{'後半':>9}   判定")
    for name in conds(pd.DataFrame({"fieldsize": [], "jockey_changed": [], "distance": [], "raceclass": []})):
        if name not in cnts:
            continue
        a = np.array(res[(name, "all")], float)
        r1 = np.array(res[(name, "1st")], float)
        r2 = np.array(res[(name, "2nd")], float)
        w = np.ptp(a)
        if name == "全体":
            v = "基準"
        elif r1.mean() > 100 and r2.mean() > 100:
            v = "★両期間100%超"
        elif a.mean() > 100:
            v = "全体100%超だが片期間のみ"
        else:
            v = "100%未満"
        print(f"{name:<20}{cnts[name]:>7}{a.mean():>7.1f}%{w:>8.1f}{r1.mean():>8.1f}%{r2.mean():>8.1f}%   {v}")
    print("\n※標本が小さいほどノイズ床は上がる（(30): 945Rで±20pt超）。R数とシード幅を必ず併せて読むこと。")


if __name__ == "__main__":
    main()
