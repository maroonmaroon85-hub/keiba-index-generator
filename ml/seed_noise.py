"""
⚠本プロジェクトで最も重要な計測: **学習の乱数シードだけで部分集合ROIがどれだけ動くか**。

(29)で「道悪ダートは両期間で再現する本物」と判定したが、**同じ目標・同じ特徴・同じパラメータのまま
シードを変えて学習し直しただけで消えた**。つまり期間を分ける対策（(28)(29)）だけでは不十分で、
モデル自体が乱数の1つの引きである以上、その揺れ幅を知らないと「本物」の判定ができない。

そこで win 目標のモデルを複数シードで学習し、同一の部分集合について回収率の散らばりを測る。
ここで得られる幅が**ノイズの床**であり、これより小さい差は何であれ意味を持たない。

実行: python3 ml/seed_noise.py [シード数(既定5)]
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

STRAT = ("三連単マルチ", 8)
SPLIT_YEAR = 2022


def main():
    n_seed = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    d = F.to_model(F.load_files())
    f = F.build_features(d)
    keep = f["n_prior"] >= 1
    d, f = d[keep].reset_index(drop=True), f[keep].reset_index(drop=True)
    y = (d["finish"] == 1).astype(int).to_numpy()
    cut = d["date"].quantile(0.3)
    tr, te = (d["date"] < cut).to_numpy(), (d["date"] >= cut).to_numpy()
    fx, _ = F.encode_categoricals(f)
    print(f"train {tr.sum():,} / test {te.sum():,}  分割日 {cut.date()}  シード{n_seed}本")

    pays = load_payout_a("data/payout/a.csv")
    meta = d.loc[te, ["raceid", "umaban", "cond"]].copy()

    tname, n = STRAT
    results = defaultdict(list)
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
            if p is None or not p["sanrentan"] or len(g) < n + 1:
                continue
            g = g.sort_values("prob", ascending=False)
            nums = g["umaban"].astype(int).tolist()
            rows.append({"year": p["date"].year, "cs": p["surface"] + g["cond"].iloc[0],
                         "pay": hits(p[KEY[tname]], nums[0], set(nums[1 : n + 1]))})
        df = pd.DataFrame(rows)
        pts = POINTS[tname](n) * 100

        def roi(sub):
            return sub["pay"].sum() / (len(sub) * pts) * 100 if len(sub) else float("nan")

        results["全体"].append(roi(df))
        for cs in ["ダ稍", "ダ重", "ダ不", "ダ良", "芝良"]:
            results[cs].append(roi(df[df["cs"] == cs]))
        va = df[df["year"] >= SPLIT_YEAR]
        for cs in ["ダ稍", "ダ重", "ダ不"]:
            results[f"{cs}(検証期のみ)"].append(roi(va[va["cs"] == cs]))
        print(f"  seed {seed}: 全体 {results['全体'][-1]:.1f}%  "
              + "  ".join(f"{cs} {results[cs][-1]:.1f}%" for cs in ["ダ稍", "ダ重", "ダ不"]))

    print(f"\n===== {tname}×紐{n}  シード{n_seed}本の散らばり =====")
    print(f"{'部分集合':<18}{'最小':>8}{'最大':>8}{'幅':>8}{'平均':>8}{'標準偏差':>9}")
    for k, v in results.items():
        a = np.array(v, float)
        print(f"{k:<18}{a.min():>7.1f}%{a.max():>7.1f}%{a.max() - a.min():>7.1f}{a.mean():>7.1f}%{a.std():>8.1f}")
    print("\n※ここに出る『幅』がノイズの床。これより小さい差は、期間を分けて再現しても意味を持たない。")


if __name__ == "__main__":
    main()
