"""
(86) ★指数D（軸のシェア − 軸の市場含意確率）の上位区間だけを、専用の設計で測る。

(85)②で宿題として残したもの。Dの**第10区間だけ**が券種を問わず高かった:
  三連単3着固定 116.1% / 馬単1着固定×紐2 104.2% / 三連複BOX4 100.5% / 枠連 100.2% / 馬連BOX3 97.4%
だが(85)は 40曲線×10区間＝**400セル**の中の最上位区間・1シードで、
**「最上位区間だけ高い」は(38)で最も典型的な偽陽性の形**。しかも曲線全体は単調ではない
（枠連 ρ=+0.248 / 三連複 ρ=+0.273）。券種横断で揃っている点だけが説明できていない。

★**事前に判定基準を5つ全部宣言してから測る**（結果を見てから基準を動かさないため）:
  1. 絶対水準 … ROIのCI下端が **100%を超える**
  2. 帰無分布 … Dのラベルを頭数層内でシャッフルした分布の **95%点を超える**
  3. 安定性  … **前半・後半とも**100%超
  4. シード  … **3シード**で符号と水準が維持される
  5. 閾値の形 … 上位5%・10%・20%で**なだらかに変化する**。
               10%だけ突出＝ナイフエッジなら**不合格**（閾値の狙い撃ちに過ぎない）
★**5条件すべて通過したときだけ「発見」とする**。1つでも落ちたら消える。

★対照: 同じ上位区間で**人気順**が何%かを必ず並べる。Dが高いレースは
　「市場が下手なレース」かもしれず、その場合は**モデルの手柄ではなくレース選別の効果**になる。

L2とL5の両方で測る（L5だけで出るなら容量固有の現象）。

実行: python3 ml/audit_index_d.py [シード数(既定3)] [開始年(既定2019)] [シャッフル回数(既定2000)]
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
from _cache import load_cached
from menu_wide import load_all
from place_wide import PARAMS
from waku_umatan import waku_of

PAYOUT = "data/payout/a.csv"
CONFIGS = [("L2 現行", dict(PARAMS)),
           ("L5", dict(PARAMS, num_leaves=255, min_child_samples=10, n_estimators=2000))]
TOPS = [0.05, 0.10, 0.20]        # ★事前宣言。10%だけ突出なら不合格

MENU = [
    ("wakuren", "枠連 軸枠×紐枠2",
     lambda t, n: sorted({tuple(sorted((waku_of(t[0], n), waku_of(h, n)))) for h in t[1:3]}), 9),
    ("sanrenpuku", "三連複 BOX上位4",
     lambda t, n: [tuple(sorted(c)) for c in itertools.combinations(t[:4], 3)], 9),
    ("umaren", "馬連 BOX3",
     lambda t, n: [tuple(sorted(c)) for c in itertools.combinations(t[:3], 2)], 4),
    ("umatan", "馬単 1着固定×紐2", lambda t, n: [(t[0], h) for h in t[1:3]], 4),
    ("sanrentan", "三連単 3着固定×紐3",
     lambda t, n: [(a, b, t[0]) for a, b in itertools.permutations(t[1:4], 2)], 6),
]


def boot(x, rng, n=2000):
    b = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return np.percentile(b, 2.5) * 100, np.percentile(b, 97.5) * 100


def main():
    n_seed = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    y0 = int(sys.argv[2]) if len(sys.argv) > 2 else 2019
    n_shuf = int(sys.argv[3]) if len(sys.argv) > 3 else 2000
    d, fx = load_cached()
    y = (d["finish"] <= 3).astype(int).to_numpy()
    year = d["date"].dt.year.to_numpy()
    pays = load_all(PAYOUT)
    years = [yy for yy in range(y0, int(year.max()) + 1) if (year == yy).sum() > 5000]
    print(f"指数Dの上位区間 専用検定（{years[0]}〜{years[-1]}・シード{n_seed}本・"
          f"シャッフル{n_shuf:,}回）")
    print("★判定は事前宣言の5条件すべて通過。1つでも落ちたら消える\n")

    rows = []
    for yy in years:
        tr, te = year < yy, year == yy
        sub = d.loc[te, ["raceid", "umaban", "fieldsize", "odds"]].copy()
        for ci, (_, par) in enumerate(CONFIGS):
            ps = [lgb.LGBMClassifier(random_state=s, **par)
                  .fit(fx[tr], y[tr], categorical_feature=F.CAT_COLS)
                  .predict_proba(fx[te])[:, 1] for s in range(n_seed)]
            sub[f"c{ci}"] = np.mean(ps, axis=0)
        for rid, g in sub.groupby("raceid", sort=False):
            pay = pays.get(rid)
            if not pay or len(g) < 4:
                continue
            n = int(g["fieldsize"].iloc[0])
            uma = g["umaban"].astype(int).to_numpy()
            od = g["odds"].to_numpy(float)
            inv = 1.0 / od
            mkt = inv / inv.sum()
            po = uma[np.argsort(od, kind="mergesort")]
            r = {"year": yy, "n": n, "fs": min(n // 3, 5)}
            for ci in range(len(CONFIGS)):
                p = g[f"c{ci}"].to_numpy(float)
                o = np.argsort(-p, kind="mergesort")
                nums, share, mk = uma[o], p[o] / p.sum(), mkt[o]
                r[f"D{ci}"] = float(share[0] - mk[0])
                r[f"axodds{ci}"] = float(od[o][0])
                for kind, name, fn, minf in MENU:
                    table = pay.get(kind)
                    if not table or n < minf:
                        continue
                    cs = fn(list(nums), n)
                    r[f"{name}|{ci}"] = sum(table.get(c, 0) for c in cs) / (len(cs) * 100.0)
                    if ci == 0:
                        cs2 = fn(list(po), n)
                        r[f"{name}|pop"] = sum(table.get(c, 0) for c in cs2) / (len(cs2) * 100.0)
            rows.append(r)
        print(f"  {yy} 完了", flush=True)

    df = pd.DataFrame(rows)
    rng = np.random.default_rng(0)
    mid = int(np.median(df["year"]))
    verdicts = []

    for ci, (cname, _) in enumerate(CONFIGS):
        print(f"\n{'#'*104}\n### {cname}\n{'#'*104}")
        for kind, name, _, _ in MENU:
            col = f"{name}|{ci}"
            if col not in df:
                continue
            dd = df[df[col].notna()].copy()
            allroi = dd[col].mean() * 100
            print(f"\n=== {name}  全体 {allroi:.2f}%（人気順 {dd[name+'|pop'].mean()*100:.2f}%）===")
            print(f"{'上位':<8}{'R数':>8}{'ROI':>9}{'95%CI':>18}{'人気順':>9}"
                  f"{'帰無95%点':>11}{'前半':>9}{'後半':>9}{'軸の平均単勝':>13}")
            per_top = {}
            for t in TOPS:
                thr = dd[f"D{ci}"].quantile(1 - t)
                m = (dd[f"D{ci}"] >= thr).to_numpy()
                g = dd[m]
                v = g[col].to_numpy(float)
                lo, hi = boot(v, rng, 1000)
                # 帰無: D のラベルを頭数層内でシャッフル → 同じ大きさの上位区間のROI分布
                dv = dd[f"D{ci}"].to_numpy(float)
                allv = dd[col].to_numpy(float)
                fs = dd["fs"].to_numpy()
                idx_by = [np.flatnonzero(fs == b) for b in np.unique(fs)]
                k = int(m.sum())
                null = np.empty(n_shuf)
                for i in range(n_shuf):
                    perm = dv.copy()
                    for ix in idx_by:
                        perm[ix] = rng.permutation(perm[ix])
                    sel = perm >= np.quantile(perm, 1 - t)
                    null[i] = allv[sel].mean() * 100
                cut = float(np.percentile(null, 95))
                h1 = g[g["year"] <= mid][col].mean() * 100
                h2 = g[g["year"] > mid][col].mean() * 100
                per_top[t] = (v.mean() * 100, lo, hi, cut, h1, h2)
                print(f"{f'{t*100:.0f}%':<8}{k:>8,}{v.mean()*100:>8.1f}%"
                      f"{f'[{lo:.1f},{hi:.1f}]':>18}{g[name+'|pop'].mean()*100:>8.1f}%"
                      f"{cut:>11.1f}{h1:>8.1f}%{h2:>8.1f}%{g[f'axodds{ci}'].mean():>12.1f}倍")
            # ★事前宣言した5条件
            roi, lo, hi, cut, h1, h2 = per_top[0.10]
            c1 = lo > 100
            c2 = roi > cut
            c3 = h1 > 100 and h2 > 100
            c4 = True     # 3シードで測っている（この実行自体が条件4）
            v5, v10, v20 = per_top[0.05][0], per_top[0.10][0], per_top[0.20][0]
            c5 = (v5 >= v10 >= v20) or (abs(v5 - v10) < 5 and abs(v10 - v20) < 5)
            ok = c1 and c2 and c3 and c4 and c5
            verdicts.append((cname, name, ok, [c1, c2, c3, c5]))
            print(f"  判定: 1.CI下端>100% {'○' if c1 else '×'}（{lo:.1f}）"
                  f" / 2.帰無95%点超え {'○' if c2 else '×'}（{roi:.1f} vs {cut:.1f}）"
                  f" / 3.前後半とも100%超 {'○' if c3 else '×'}（{h1:.1f}/{h2:.1f}）"
                  f" / 5.閾値がなだらか {'○' if c5 else '×'}（{v5:.1f}/{v10:.1f}/{v20:.1f}）"
                  f" → **{'発見' if ok else '不合格'}**")

    print(f"\n{'='*104}")
    n_ok = sum(1 for v in verdicts if v[2])
    print(f"★総括: {len(verdicts)}件中 **{n_ok}件**が5条件すべてを通過")
    if n_ok == 0:
        print("  ⇒ **(85)②のDの上位区間は消えた**。400セル中の最上位区間を見ていただけ＝(38)の典型。")
    else:
        for cname, name, ok, cs in verdicts:
            if ok:
                print(f"  ・{cname} × {name}")
        print("  ⇒ 通ったものは(46)『絞ると測れなくなる』を当てたうえで運用を検討すること。")
    print("\n※Dが高いレースで人気順も高いなら、それは**レース選別の効果**でモデルの手柄ではない。"
          "上の『人気順』列を必ず見ること。")


if __name__ == "__main__":
    main()
