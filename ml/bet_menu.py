"""
買い方の横断比較。「回収率」だけでなく「配当の大きさ」と「1レースのコスト」を並べる。

(47)で複勝の最良が95.8%と分かったが、その正体は**的中時の配当が107円**＝
100円賭けて7円勝つ勝負で、実質的な旨みがない。
つまり評価軸は回収率だけでは足りず、**回収率 × 配当の大きさ**のトレードオフで見る必要がある。

そこで(45)のオッズ入りモデルを使い、配当Aに入っている全券種を同じ土俵で並べる:
    複勝 / ワイド / 馬連 / 三連複 / 三連単
順位付けは2通り（どちらが効くかは券種で違うため両方出す）:
    win目標モデル  … 1着になる確率
    top3目標モデル … 3着以内に入る確率（(47)で複勝/ワイドに効いた）

出力は 的中率・的中時の平均配当・1レースのコスト・ROI・95%CI・年別の範囲。
「どの回収率を、どのくらいの配当と引き換えに得られるか」を一覧で選べるようにするのが目的。

実行: python3 ml/bet_menu.py [シード数(既定3)]
"""
import itertools
import sys

import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, "ml")
import features as F
from place_wide import load_place_wide, boot, PARAMS
from pocket_eval import load_payout_a


def combos(kind, top, n):
    """買い目の組を返す。top はモデル順に並んだ馬番リスト。"""
    axis, himo = top[0], top[1:n + 1]
    if kind == "fuku":
        return [(h,) for h in top[:n]]                                  # 複勝を上位n頭
    if kind == "wide_box":
        return [tuple(sorted(c)) for c in itertools.combinations(top[:n], 2)]
    if kind == "wide_axis":
        return [tuple(sorted((axis, h))) for h in himo]
    if kind == "umaren":
        return [tuple(sorted((axis, h))) for h in himo]
    if kind == "sanrenpuku":
        return [tuple(sorted((axis, a, b))) for a, b in itertools.combinations(himo, 2)]
    if kind == "sanrentan":
        return [tuple(p) for a, b in itertools.permutations(himo, 2)
                for p in ([axis, a, b], [a, axis, b], [a, b, axis])]
    raise ValueError(kind)


# (表示名, 払戻の種類, combos の kind, n)
MENU = [
    ("複勝 1点",            "fuku",       "fuku",        1),
    ("複勝 上位2頭",         "fuku",       "fuku",        2),
    ("複勝 上位3頭",         "fuku",       "fuku",        3),
    ("ワイド 上位2頭1点",     "wide",       "wide_box",    2),
    ("ワイド 上位3頭BOX",     "wide",       "wide_box",    3),
    ("ワイド 軸1×紐3",       "wide",       "wide_axis",   3),
    ("馬連 軸1×紐2",        "umaren",     "umaren",      2),
    ("馬連 軸1×紐4",        "umaren",     "umaren",      4),
    ("三連複 軸1×紐3",       "sanrenpuku", "sanrenpuku",  3),
    ("三連複 軸1×紐5",       "sanrenpuku", "sanrenpuku",  5),
    ("三連単 マルチ×紐4",     "sanrentan",  "sanrentan",   4),
]


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
    print(f"train {tr.sum():,} / test {te.sum():,}  シード{n_seed}本（(45)のオッズ入りモデル）")

    pa = load_payout_a("data/payout/a.csv")
    pw = load_place_wide("data/payout/a.csv")

    preds = {}
    for tag, tgt in [("win", win), ("top3", top3)]:
        ps = []
        for seed in range(n_seed):
            m = lgb.LGBMClassifier(random_state=seed, **PARAMS)
            m.fit(fx[tr], tgt[tr], categorical_feature=F.CAT_COLS)
            ps.append(m.predict_proba(fx[te])[:, 1])
            print(f"  {tag} seed {seed} 完了")
        preds[tag] = np.mean(ps, axis=0)

    sub = d.loc[te, ["raceid", "umaban"]].copy()
    sub["year"] = d.loc[te, "date"].dt.year.to_numpy()
    for tag in preds:
        sub[tag] = preds[tag]

    rng = np.random.default_rng(0)
    print("\n" + "=" * 104)
    print("★買い方の横断比較（(45)オッズ入りモデル・順位付けは win / top3 の2通り）")
    print("=" * 104)
    print(f"{'買い方':<20}{'順位':<6}{'R':>7}{'点/R':>5}{'コスト':>8}{'的中率':>8}"
          f"{'的中時配当':>10}{'ROI':>8}{'95%CI':>15}{'年別':>14}")

    for name, paykey, kind, n in MENU:
        for tag in ["win", "top3"]:
            rets, years, hits_pay = [], [], []
            for rid, g in sub.groupby("raceid", sort=False):
                src = pw.get(rid) if paykey in ("fuku", "wide") else pa.get(rid)
                if src is None:
                    continue
                table = src[paykey] if paykey in ("fuku", "wide") else src[paykey]
                if not table or len(g) < n + 1:
                    continue
                gg = g.sort_values(tag, ascending=False, kind="mergesort")
                top = gg["umaban"].astype(int).tolist()
                cs = combos(kind, top, n)
                pay = sum(table.get(c[0] if paykey == "fuku" else c, 0) for c in cs)
                rets.append(pay / (len(cs) * 100.0))
                years.append(g["year"].iloc[0])
                if pay > 0:
                    hits_pay.append(pay)
            if len(rets) < 500:
                continue
            x = np.array(rets, float)
            yr = np.array(years)
            lo, hi = boot(x, rng, 2000)
            ys = pd.Series(x).groupby(yr).mean() * 100
            pts = len(combos(kind, list(range(1, n + 3)), n))
            print(f"{name if tag=='win' else '':<20}{tag:<6}{len(x):>7,}{pts:>5}{pts*100:>7,}円"
                  f"{(x>0).mean()*100:>7.1f}%{np.mean(hits_pay):>9,.0f}円{x.mean()*100:>7.1f}%"
                  f"{f'[{lo:.0f},{hi:.0f}]':>15}{f'{ys.min():.0f}〜{ys.max():.0f}%':>14}")
        print()


if __name__ == "__main__":
    main()
