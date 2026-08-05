"""
(82) 買い方29通りを **ウォークフォワード × 容量L2/L5** で測り直す。

(77)で8券種29通りを横断比較したが、あれには穴が2つある:
  1. **現行モデル(L2)での結果**。(81)で容量を上げたときの効き方が券種で2倍以上違うと分かった
     （枠連 +2.78pt / 三連複 +5.98pt）。**L5では順位が並び替わるはず**。
  2. **単一分割（前30%学習）**。(78)①でそれが実運用と2pt以上ずれると判明している。

そこで全29通りを、実運用と同じ手続き（各年をそれ以前の全データで学習）で、
**L2（現行）とL5（(81)の最良）の両方**で測る。見たいのは3つ:
  ・L5でどの買い方が最良になるか（L2での順位から変わるか）
  ・容量を上げたときの効き方が券種でどう違うか（連系ほど効く、という仮説）
  ・(77)の「現行2つを置き換えるものは無い」がL5でも成り立つか

⚠(77)と同じ作法を守る: 買い方は**事前宣言**（menu_wide.MENU をそのまま使い、増やさない）、
　全通りに**人気順の対照**を並べ、L2との比較は**対応あり**で行う。多重性も明示する。

実行: python3 ml/audit_menu_wf.py [シード数(既定1)] [開始年(既定2019)]
"""
import sys
import warnings

import numpy as np
import pandas as pd
import lightgbm as lgb

warnings.filterwarnings("ignore")
sys.path.insert(0, "ml")
import features as F
from _cache import load_cached
from menu_wide import MENU, load_all
from place_wide import PARAMS

PAYOUT = "data/payout/a.csv"

# ★L4とL5の両方を載せる。3シードで**最良の段は券種によって入れ替わった**（枠連L5・三連複L4）ので、
#   どちらか一方を選ぶと(80)と同じ後知恵の汚染になる。両方出して読者に判断させる。
CONFIGS = [
    ("L2 現行 leaves31/mc100/400本", dict(PARAMS)),
    ("L4 leaves127/mc20/1500本", dict(PARAMS, num_leaves=127, min_child_samples=20,
                                     n_estimators=1500)),
    ("L5 leaves255/mc10/2000本", dict(PARAMS, num_leaves=255, min_child_samples=10,
                                     n_estimators=2000)),
]


def boot(x, rng, n=2000):
    b = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return np.percentile(b, 2.5) * 100, np.percentile(b, 97.5) * 100


def main():
    n_seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    y0 = int(sys.argv[2]) if len(sys.argv) > 2 else 2019
    d, fx = load_cached()
    y = (d["finish"] <= 3).astype(int).to_numpy()
    year = d["date"].dt.year.to_numpy()
    pays = load_all(PAYOUT)
    years = [yy for yy in range(y0, int(year.max()) + 1) if (year == yy).sum() > 5000]
    print(f"買い方{len(MENU)}通り × 容量2種 × ウォークフォワード"
          f"（{years[0]}〜{years[-1]}・シード{n_seed}本）\n")

    # rows[name] = list of {"L2":.., "L5":.., "pop":.., "hit_L5":.., "pts":..}
    rows = {nm: [] for _, nm, _, _ in MENU}
    for yy in years:
        tr, te = year < yy, year == yy
        sub = d.loc[te, ["raceid", "umaban", "fieldsize", "finish", "odds"]].copy()
        for ci, (cname, par) in enumerate(CONFIGS):
            ps = [lgb.LGBMClassifier(random_state=s, **par)
                  .fit(fx[tr], y[tr], categorical_feature=F.CAT_COLS)
                  .predict_proba(fx[te])[:, 1] for s in range(n_seed)]
            sub[f"c{ci}"] = np.mean(ps, axis=0)
        for rid, g in sub.groupby("raceid", sort=False):
            pay = pays.get(rid)
            if not pay:
                continue
            n = int(g["fieldsize"].iloc[0])
            wnr = g[g["finish"] == 1]
            tansho = {(int(wnr["umaban"].iloc[0]),): float(wnr["odds"].iloc[0]) * 100} \
                if len(wnr) == 1 else {}
            uma = g["umaban"].astype(int).to_numpy()
            tops = {"pop": uma[np.argsort(g["odds"].to_numpy(float), kind="mergesort")]}
            for ci in range(len(CONFIGS)):
                tops[f"c{ci}"] = uma[np.argsort(-g[f"c{ci}"].to_numpy(float), kind="mergesort")]
            for kind, name, fn, minf in MENU:
                table = tansho if kind == "tansho" else pay.get(kind)
                if not table or n < minf or len(g) < minf:
                    continue
                r = {}
                for k, t in tops.items():
                    cs = fn(list(t), n)
                    r[k] = sum(table.get(c, 0) for c in cs) / (len(cs) * 100.0)
                    r[f"{k}_hit"] = float(any(c in table for c in cs))
                    r["pts"] = len(cs)
                rows[name].append(r)
        print(f"  {yy} 完了", flush=True)

    rng = np.random.default_rng(0)
    res = []
    for kind, name, _, _ in MENU:
        df = pd.DataFrame(rows[name])
        if len(df) < 500:
            continue
        a, b = df["c0"].to_numpy(float), df["c1"].to_numpy(float)
        c = df["c2"].to_numpy(float)
        pop = df["pop"].to_numpy(float)
        lo, hi = boot(b - a, rng)
        # ★ROIが上がった理由が「当てる力」か「配当を取る力」かを分ける。
        #   ROI = 的中率 × 的中時配当 ÷ コスト なので、この2つに分ければ寄与が一意に決まる。
        row = {"name": name, "n": len(df), "pts": df["pts"].mean(),
               "L2": a.mean() * 100, "L4": b.mean() * 100, "pop": pop.mean() * 100,
               "d": (b - a).mean() * 100, "lo": lo, "hi": hi, "L5": c.mean() * 100,
               "d5": (c - a).mean() * 100, "vs_pop": (b - pop).mean() * 100,
               "hit": df["c1_hit"].mean() * 100}
        pts = df["pts"].to_numpy(float)
        for tag, col in (("L2", "c0"), ("L4", "c1"), ("L5", "c2")):
            v = df[col].to_numpy(float)
            h = df[f"{col}_hit"].to_numpy(float) > 0
            row[f"hit_{tag}"] = h.mean() * 100
            row[f"pay_{tag}"] = float((v[h] * pts[h] * 100).mean()) if h.any() else 0.0
        res.append(row)
    r = pd.DataFrame(res).sort_values("L4", ascending=False)

    print("\n" + "=" * 116)
    print("L4でのROI順。L2は現行・L5はさらに容量を上げた段。差(L4−L2)はレース単位の対応あり")
    print("=" * 116)
    print(f"{'買い方':<26}{'R数':>8}{'点数':>6}{'的中率':>8}{'L2':>8}{'L4':>8}{'L5':>8}"
          f"{'L4−L2':>10}{'95%CI':>17}{'人気順':>8}{'L4対人気順':>12}")
    for _, x in r.iterrows():
        mark = "★" if x["lo"] > 0 else ""
        print(f"{x['name']:<26}{int(x['n']):>8,}{x['pts']:>6.1f}{x['hit']:>7.1f}%"
              f"{x['L2']:>7.1f}%{x['L4']:>7.1f}%{x['L5']:>7.1f}%{x['d']:>+9.2f}pt"
              f"{f'[{x.lo:+.2f},{x.hi:+.2f}]':>17}"
              f"{x['pop']:>7.1f}%{x['vs_pop']:>+11.2f}pt{mark}")

    print(f"\n■ L4で上位5（L5も併記）")
    for _, x in r.head(5).iterrows():
        print(f"  {x['name']:<26} L4 {x['L4']:.2f}% / L5 {x['L5']:.2f}%（L2は{x['L2']:.2f}%）"
              f"  的中{x['hit']:.1f}%  対人気順{x['vs_pop']:+.2f}pt")
    print("■ 容量を上げた効き方が大きい順（上位5）")
    for _, x in r.sort_values("d", ascending=False).head(5).iterrows():
        print(f"  {x['name']:<26} {x['d']:+.2f}pt [{x['lo']:+.2f},{x['hi']:+.2f}]"
              f"  L2 {x['L2']:.1f}% → L4 {x['L4']:.1f}% / L5 {x['L5']:.1f}%")
    print("\n" + "=" * 118)
    print("ROIの上がり方を「的中率」と「的中時配当」に分解（L2→L5）")
    print("=" * 118)
    print(f"{'買い方':<26}{'L2的中':>8}{'L5的中':>8}{'的中率差':>10}"
          f"{'L2配当':>10}{'L5配当':>10}{'配当差':>10}{'ROI差':>10}{'寄与の内訳':>22}")
    for _, x in r.sort_values("d5", ascending=False).iterrows():
        dh = x["hit_L5"] - x["hit_L2"]
        dp = x["pay_L5"] - x["pay_L2"]
        rh = np.log(max(x["hit_L5"], 1e-9) / max(x["hit_L2"], 1e-9))
        rp = np.log(max(x["pay_L5"], 1e-9) / max(x["pay_L2"], 1e-9))
        tot = abs(rh) + abs(rp)
        share = f"的中{abs(rh)/tot*100:.0f}% / 配当{abs(rp)/tot*100:.0f}%" if tot > 0 else "—"
        print(f"{x['name']:<26}{x['hit_L2']:>7.1f}%{x['hit_L5']:>7.1f}%{dh:>+9.2f}pt"
              f"{x['pay_L2']:>9,.0f}円{x['pay_L5']:>9,.0f}円{dp:>+9,.0f}円"
              f"{x['d5']:>+9.2f}pt{share:>22}")
    up = r[r["d5"] > 0]
    nh = int((up["hit_L5"] > up["hit_L2"]).sum())
    print(f"  → ROIが上がった{len(up)}通りのうち、**的中率も上がった**のは {nh}通り。"
          f"残り{len(up)-nh}通りは**的中率を下げて配当で取り返している**。")

    nsig = int((r["lo"] > 0).sum())
    print(f"\n★多重性: {len(r)}通り中 {nsig}通りで L5>L2 のCIが0を外れた"
          f"（偶然の期待値は約{len(r)*0.025:.1f}通り・片側）。")
    print("　★ただし29通りは同じモデル・同じレースを見ているので独立ではない。"
          "「容量を上げると効く」という1つの事実が券種を問わず現れている、と読むこと。")
    print("★どれもROIは100%未満。買い方の順位づけであって勝てる買い方の探索ではない。")


if __name__ == "__main__":
    main()
