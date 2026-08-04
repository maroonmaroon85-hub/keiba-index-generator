"""
(76) 小倉の優位は「場」なのか「そこで行われるレースの構成」なのか。

(73)で小倉だけが4つの判定（帰無分布・前後半・年別・配当キャップ）を通った。
だが**場が違えば行われるレースも違う**——小倉は芝1200が多く、頭数も他場と違う。
「小倉が特別」ではなく「小倉でよく行われる条件が特別」なら、それは場の話ではない。

やること: **標準化（direct standardization）**。
  1. レースを (芝ダ, 距離帯, 頭数帯, クラス帯) の層に切る
  2. 各層で**小倉以外**の 差(モデル−人気順) の平均を出す
  3. 小倉の層構成でそれを重み付け平均 → 「小倉が平均的な場だったら出るはずの差」＝期待値
  4. 小倉の実測差 − 期待値 = **場そのものに帰属する分**
期待値が実測差の大半を説明するなら、小倉の優位は**構成の話**であって場の話ではない。

同じ計算を全場で出す。判定は★判定基準どおり、層内シャッフルの帰無分布と並べる。

実行: python3 ml/kokura_check.py [シード数(既定3)] [シャッフル回数(既定2000)]
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
from market_baseline import load, wakuren_cs
from place_wide import PARAMS, boot
from pocket_eval import load_payout_a
from waku_umatan import load_wu

PAYOUT = "data/payout/a.csv"


def build(sub, wu, pa):
    rows = []
    for rid, g in sub.groupby("raceid", sort=False):
        n = int(g["fieldsize"].iloc[0])
        w, s3 = wu.get(rid), pa.get(rid)
        nums = {"model": g.sort_values("p", ascending=False, kind="mergesort")
                          ["umaban"].astype(int).tolist(),
                "pop": g.sort_values("odds", ascending=True, kind="mergesort")
                        ["umaban"].astype(int).tolist()}
        r = {"track": g["course"].iloc[0], "fieldsize": n, "year": g["year"].iloc[0],
             "surface": int(g["surface"].iloc[0]), "distance": float(g["distance"].iloc[0]),
             "raceclass": float(g["raceclass"].iloc[0])}
        if w and w["wakuren"] and len(g) >= 3:
            for k, nm in nums.items():
                cs = wakuren_cs(nm, n)
                r[f"waku_{k}"] = sum(w["wakuren"].get(c, 0) for c in cs) / (len(cs) * 100.0)
        if s3 and s3["sanrenpuku"] and len(g) >= 9:
            for k, nm in nums.items():
                cs = [tuple(sorted(c)) for c in itertools.combinations(nm[:4], 3)]
                r[f"trio_{k}"] = sum(s3["sanrenpuku"].get(c, 0) for c in cs) / 400.0
        rows.append(r)
    return pd.DataFrame(rows)


def strata(d):
    """(芝ダ, 距離帯, 頭数帯, クラス帯) の層ラベル。層が細かすぎると空になるので4×4程度に抑える。"""
    dist = pd.cut(d["distance"], [0, 1400, 1800, 2200, 9999], labels=False)
    fs = pd.cut(d["fieldsize"], [0, 10, 13, 15, 99], labels=False)
    cls = pd.cut(d["raceclass"], [-1, 1, 3, 5, 9], labels=False)
    return (d["surface"].astype(str) + "|" + dist.astype(str) + "|"
            + fs.astype(str) + "|" + cls.astype(str))


def standardize(d, mcol, pcol):
    """各場について「平均的な場だったら出るはずの差」を層構成から作る。"""
    d = d[d[mcol].notna()].copy()
    d["_s"] = strata(d)
    d["_diff"] = d[mcol] - d[pcol]
    out = []
    for t, g in d.groupby("track"):
        others = d[d["track"] != t]
        base = others.groupby("_s")["_diff"].mean()
        wgt = g["_s"].value_counts(normalize=True)
        common = wgt.index.intersection(base.index)
        cover = float(wgt[common].sum())
        exp = float((wgt[common] * base[common]).sum() / max(cover, 1e-9))
        obs = float(g["_diff"].mean())
        out.append({"track": t, "n": len(g), "obs": obs * 100, "exp": exp * 100,
                    "resid": (obs - exp) * 100, "cover": cover * 100})
    return pd.DataFrame(out).sort_values("obs", ascending=False)


def compo(d, mcol):
    """小倉と他場の構成の違い（何が特別なのか）を素で見る。"""
    d = d[d[mcol].notna()].copy()
    print(f"\n{'場':<6}{'R数':>8}{'ダート率':>10}{'平均距離':>10}{'平均頭数':>10}"
          f"{'1200m以下率':>13}{'新馬未勝利率':>14}")
    for t, g in d.groupby("track"):
        print(f"{t:<6}{len(g):>8,}{g['surface'].mean()*100:>9.1f}%"
              f"{g['distance'].mean():>9.0f}m{g['fieldsize'].mean():>9.1f}頭"
              f"{(g['distance']<=1200).mean()*100:>12.1f}%"
              f"{(g['raceclass']<=1).mean()*100:>13.1f}%")


def market_quality(d, te, odds, y, p_model, course):
    """★なぜその場でモデルが勝つのか: **市場のオッズの情報量が場によって違うのでは**という仮説。

    市場含意の3着以内確率（Harville）を場ごとにAUCで評価する。
    薄い市場（＝オッズが情報を持たない場）ほどモデルが勝ちやすい、という筋が通るなら、
    小倉の市場AUCは最低のはず。モデルのAUCと並べて差も出す。
    """
    from model_line import auc, market_top3
    sub = d.loc[te, ["raceid"]].copy()
    mt3 = market_top3(d.loc[te].reset_index(drop=True), odds[te])
    print(f"\n=== なぜその場なのか: 場ごとの「単勝オッズだけの情報量」 ===")
    print(f"{'場':<6}{'頭数':>8}{'市場AUC':>10}{'モデルAUC':>11}{'モデル−市場':>13}"
          f"{'市場含意の平均':>15}{'実際のtop3率':>14}")
    rows = []
    for t in sorted(set(course)):
        m = (course == t)
        if m.sum() < 2000:
            continue
        a_m, a_p = auc(y[te][m], mt3[m]), auc(y[te][m], p_model[m])
        rows.append((t, int(m.sum()), a_m, a_p, a_p - a_m,
                     float(mt3[m].mean()), float(y[te][m].mean())))
    for t, n, a_m, a_p, gap, mp, ap in sorted(rows, key=lambda r: r[2]):
        print(f"{t:<6}{n:>8,}{a_m:>10.4f}{a_p:>11.4f}{gap:>+12.4f}"
              f"{mp:>14.4f}{ap:>13.4f}")
    print("  ※市場AUCの低い順。**低い場＝オッズが当てにならない場**。"
          "そこでモデルが人気順に勝ちやすいなら、(73)の場の差の正体はこれで説明できる。")


def main():
    n_seed = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    n_shuf = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
    d, fx, odds = load()
    y = (d["finish"] <= 3).astype(int).to_numpy()
    cut = d["date"].quantile(0.3)
    tr, te = (d["date"] < cut).to_numpy(), (d["date"] >= cut).to_numpy()
    print(f"train {tr.sum():,} / test {te.sum():,}（分割 {cut.date()}）  シード{n_seed}本")

    wu, pa = load_wu(PAYOUT), load_payout_a(PAYOUT)
    sub = d.loc[te, ["raceid", "umaban", "fieldsize", "course", "surface", "distance",
                     "raceclass"]].copy()
    sub["odds"] = odds[te]
    sub["year"] = d.loc[te, "date"].dt.year.to_numpy()

    ps = []
    for s in range(n_seed):
        m = lgb.LGBMClassifier(random_state=s, **PARAMS).fit(fx[tr], y[tr],
                                                             categorical_feature=F.CAT_COLS)
        ps.append(m.predict_proba(fx[te])[:, 1])
        print(f"  seed {s} 完了")
    sub["p"] = np.mean(ps, axis=0)
    df = build(sub, wu, pa)

    compo(df, "waku_model")
    market_quality(d, te, odds, y, sub["p"].to_numpy(float),
                   d.loc[te, "course"].to_numpy())

    rng = np.random.default_rng(0)
    for title, mcol, pcol in [("枠連 軸枠×紐枠2", "waku_model", "waku_pop"),
                              ("三連複 BOX上位4", "trio_model", "trio_pop")]:
        st = standardize(df, mcol, pcol)
        print(f"\n=== {title}: 差(モデル−人気順) を層構成で標準化 ===")
        print(f"{'場':<6}{'R数':>8}{'実測の差':>11}{'構成から期待される差':>22}"
              f"{'場に帰属する分':>17}{'層のカバー率':>14}")
        for _, r in st.iterrows():
            print(f"{r['track']:<6}{int(r['n']):>8,}{r['obs']:>+10.2f}pt"
                  f"{r['exp']:>+21.2f}pt{r['resid']:>+16.2f}pt{r['cover']:>13.1f}%")
        print("  ※「構成から期待される差」= 同じ(芝ダ,距離帯,頭数帯,クラス帯)のレースを"
              "**他の9場で**買ったときの差を、その場の層構成で重み付けした値。")
        print("  ※「場に帰属する分」が実測とほぼ同じなら場そのものの性質、"
              "0に近いなら**構成の違いで説明できる**。")

        # ★帰無分布: 場ラベルを層内でシャッフルしたときの「場に帰属する分」の最大値
        dd = df[df[mcol].notna()].copy()
        dd["_s"] = strata(dd)
        dd["_diff"] = (dd[mcol] - dd[pcol]).to_numpy()
        codes, uniq = pd.factorize(dd["track"])
        sc, _ = pd.factorize(dd["_s"])
        v = dd["_diff"].to_numpy(float)
        idx_by_s = [np.flatnonzero(sc == i) for i in range(sc.max() + 1)]
        obs_max = st["resid"].abs().max()
        null = np.empty(n_shuf)
        for i in range(n_shuf):
            perm = codes.copy()
            for ix in idx_by_s:
                perm[ix] = rng.permutation(perm[ix])
            s_ = np.bincount(perm, weights=v, minlength=len(uniq))
            c_ = np.bincount(perm, minlength=len(uniq))
            mns = s_ / np.maximum(c_, 1)
            null[i] = np.abs(mns - v.mean()).max() * 100
        p = float((null >= obs_max).mean())
        print(f"  ★層内シャッフル{n_shuf:,}回: 最大の|場に帰属する分| 観測 {obs_max:.2f}pt / "
              f"偶然の中央値 {np.percentile(null,50):.2f}pt・95%点 {np.percentile(null,95):.2f}pt "
              f"→ p={p:.3f}" + ("  ⇒ **偶然と区別できない**" if p >= 0.05 else "  ⇒ 偶然では説明しにくい"))


if __name__ == "__main__":
    main()
