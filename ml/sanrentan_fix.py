"""
三連単で軸を1着/2着/3着に固定した場合を比較する（マルチとの対比）。

軸は(49)より**8割が1番人気**。1番人気が勝つのは最も予想されている結果＝最も買われすぎているので、
1着に固定するのは過剰人気を正面から買うことになる。逆に2着・3着に固定すると
「1番人気が取りこぼす」側に賭けることになり、配当が上がる。
(41)の「モデルの優位は市場の期待と逆側にある」が正しければ、固定位置を後ろにするほどROIは上がるはず。

順位付けは win目標 / top3目標 の両方で出す（1着固定はwin目標が向くはず、という仮説の確認）。
点数は 固定=n(n-1)、マルチ=3n(n-1)。

実行: python3 ml/sanrentan_fix.py [シード数(既定3)]
"""
import itertools
import sys
import warnings

import numpy as np
import pandas as pd
import lightgbm as lgb

warnings.filterwarnings("ignore")
sys.path.insert(0, "ml")
import features as F
from himo_sweep import KEY
from place_wide import boot, PARAMS
from pocket_eval import load_payout_a

KINDS = ["1着固定", "2着固定", "3着固定", "マルチ"]


def build(kind, axis, himo):
    P = list(itertools.permutations(himo, 2))
    if kind == "1着固定":
        return [(axis, a, b) for a, b in P]
    if kind == "2着固定":
        return [(a, axis, b) for a, b in P]
    if kind == "3着固定":
        return [(a, b, axis) for a, b in P]
    return [t for a, b in P for t in ((axis, a, b), (a, axis, b), (a, b, axis))]


def main():
    n_seed = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    d = F.to_model(F.load_files())
    f = F.build_features(d)
    keep = (f["n_prior"] >= 1) & d["odds"].notna() & (d["odds"] > 0)
    d, f = d[keep].reset_index(drop=True), f[keep].reset_index(drop=True)
    win = (d["finish"] == 1).astype(int).to_numpy()
    top3 = (d["finish"] <= 3).astype(int).to_numpy()
    odds = d["odds"].to_numpy(float)
    inv = 1.0 / odds
    mkt = inv / pd.Series(inv).groupby(d["raceid"]).transform("sum").to_numpy()
    fx, _ = F.encode_categoricals(f)
    fx["log_odds"] = np.log(odds)
    fx["mkt_prob"] = mkt
    cut = d["date"].quantile(0.3)
    tr, te = (d["date"] < cut).to_numpy(), (d["date"] >= cut).to_numpy()
    pays = load_payout_a("data/payout/a.csv")
    K = KEY["三連単マルチ"]

    pr = {}
    for tag, y in [("win", win), ("top3", top3)]:
        ps = []
        for s in range(n_seed):
            m = lgb.LGBMClassifier(random_state=s, **PARAMS)
            m.fit(fx[tr], y[tr], categorical_feature=F.CAT_COLS)
            ps.append(m.predict_proba(fx[te])[:, 1])
        pr[tag] = np.mean(ps, axis=0)
        print(f"  {tag} 学習完了")

    sub = d.loc[te, ["raceid", "umaban"]].copy()
    sub["year"] = d.loc[te, "date"].dt.year.to_numpy()
    for k, v in pr.items():
        sub[k] = v
    rng = np.random.default_rng(0)

    print(f"\n{'買い方':<14}{'紐':>3}{'順位':<6}{'点':>4}{'コスト':>8}{'的中率':>8}"
          f"{'的中時配当':>11}{'ROI':>8}{'95%CI':>14}{'年別':>13}")
    for n in [3, 4]:
        for kind in KINDS:
            for tag in ["win", "top3"]:
                rows = []
                for rid, g in sub.groupby("raceid", sort=False):
                    p = pays.get(rid)
                    if p is None or not p["sanrentan"] or len(g) < n + 1:
                        continue
                    t = g.sort_values(tag, ascending=False, kind="mergesort")["umaban"].astype(int).tolist()
                    cs = build(kind, t[0], t[1:n + 1])
                    rows.append({"y": g["year"].iloc[0],
                                 "pay": sum(p[K].get(c, 0) for c in cs), "np": len(cs)})
                df = pd.DataFrame(rows)
                pts = df["np"].iloc[0] * 100
                x = (df["pay"] / pts).to_numpy(float)
                lo, hi = boot(x, rng, 2000)
                nz = df.loc[df["pay"] > 0, "pay"]
                ys = (df["pay"] / pts).groupby(df["y"]).mean() * 100
                print(f"{kind if tag=='win' else '':<14}{n if tag=='win' else '':>3}{tag:<6}"
                      f"{df['np'].iloc[0]:>4}{pts:>7,}円{(x>0).mean()*100:>7.2f}%{nz.mean():>10,.0f}円"
                      f"{x.mean()*100:>7.1f}%{f'[{lo:.0f},{hi:.0f}]':>14}"
                      f"{f'{ys.min():.0f}〜{ys.max():.0f}%':>13}")
            print()


if __name__ == "__main__":
    main()
