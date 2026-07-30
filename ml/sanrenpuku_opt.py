"""
三連複の買い方を総当たりで最適化する。(48)の推奨「軸1×紐3」は未最適化だった。

(48)では紐3と紐5しか比べておらず、紐2・紐4、およびBOX買いや軸の取り方（モデル2位/3位を軸に）
を測っていなかった。(42)で三連単の紐数を最適化したら結論が変わった（紐7→紐5）ので、同じ確認をする。

同一レース集合（9頭立て以上）で12通りを比較する。順位付けは top3目標のオッズ入りモデル((45))。

実行: python3 ml/sanrenpuku_opt.py [シード数(既定3)]
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
from place_wide import boot, PARAMS
from pocket_eval import load_payout_a


def plans():
    p = {}
    for n in [2, 3, 4, 5, 6]:
        p[f"軸1位×紐{n}"] = ("axis", 0, n)
    for n in [3, 4, 5]:
        p[f"BOX上位{n}"] = ("box", None, n)
    for a in [1, 2]:
        for n in [3, 4]:
            p[f"軸{a+1}位×紐{n}"] = ("axis", a, n)
    return p


def combos(kind, ai, n, top):
    if kind == "axis":
        others = [x for i, x in enumerate(top) if i != ai][:n]
        return [tuple(sorted((top[ai], a, b))) for a, b in itertools.combinations(others, 2)]
    return [tuple(sorted(c)) for c in itertools.combinations(top[:n], 3)]


def main():
    n_seed = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    d = F.to_model(F.load_files())
    f = F.build_features(d)
    keep = (f["n_prior"] >= 1) & d["odds"].notna() & (d["odds"] > 0)
    d, f = d[keep].reset_index(drop=True), f[keep].reset_index(drop=True)
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

    ps = []
    for s in range(n_seed):
        m = lgb.LGBMClassifier(random_state=s, **PARAMS)
        m.fit(fx[tr], top3[tr], categorical_feature=F.CAT_COLS)
        ps.append(m.predict_proba(fx[te])[:, 1])
        print(f"  seed {s} 完了")
    sub = d.loc[te, ["raceid", "umaban"]].copy()
    sub["p3"] = np.mean(ps, axis=0)
    sub["year"] = d.loc[te, "date"].dt.year.to_numpy()

    P = plans()
    rows = {k: [] for k in P}
    for rid, g in sub.groupby("raceid", sort=False):
        p = pays.get(rid)
        if p is None or not p["sanrenpuku"] or len(g) < 9:
            continue
        t = g.sort_values("p3", ascending=False, kind="mergesort")["umaban"].astype(int).tolist()
        yr = g["year"].iloc[0]
        for k, (kind, ai, n) in P.items():
            cs = combos(kind, ai, n, t)
            rows[k].append({"y": yr, "pay": sum(p["sanrenpuku"].get(c, 0) for c in cs), "k": len(cs)})

    rng = np.random.default_rng(0)
    print(f"\n{'買い方':<22}{'点':>4}{'コスト':>8}{'的中率':>8}{'的中時配当':>11}"
          f"{'ROI':>8}{'95%CI':>14}{'年別':>13}")
    for k in P:
        df = pd.DataFrame(rows[k])
        pts = df["k"].iloc[0] * 100
        x = (df["pay"] / pts).to_numpy(float)
        lo, hi = boot(x, rng, 2000)
        nz = df.loc[df["pay"] > 0, "pay"]
        ys = (df["pay"] / pts).groupby(df["y"]).mean() * 100
        print(f"{k:<22}{df['k'].iloc[0]:>4}{pts:>7,}円{(x>0).mean()*100:>7.2f}%{nz.mean():>10,.0f}円"
              f"{x.mean()*100:>7.1f}%{f'[{lo:.0f},{hi:.0f}]':>14}"
              f"{f'{ys.min():.0f}〜{ys.max():.0f}%':>13}")


if __name__ == "__main__":
    main()
