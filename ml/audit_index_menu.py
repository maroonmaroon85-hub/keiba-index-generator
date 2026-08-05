"""
(85) 指数の区間分けを**券種横断**でやる。(84)の枠連・三連複だけでは足りないため。

⚠**多重性の設計がこの検証の本体**。29通り×5指数×10区間＝1,450セルを素で並べると
(38)の罠に直撃する（総当たり6,206セルでは**偶然だけで34.4セルが通過**した）。
そこで3点で守る:
  (a) **券種ごとに代表1つ＝8通りに絞る**（29通り全部だと145曲線になりρの検定でも多重性が効く）
  (b) 判定を**「最良の区間」から「曲線の単調性(ρ)」に変える**
      … 「10区間のどれかが高い」は偶然でも起きるが、
        「10区間が**指数の順に並ぶ**」は偶然では起きにくい
  (c) **指数ラベルを頭数層内でシャッフルした帰無分布**とρを比べる（(73)と同じ手口）

指数は(84)と同じ5つ（事前宣言・結果を見てから足さない）:
  A 軸のシェア / B 軸と2位の差 / C 枠連スコア / D 軸のシェア−市場含意 / E 買い目の市場含意

★判定: |ρ| が帰無分布の95%点を超えた曲線だけを「傾向あり」とする。
　40曲線あるので、偶然でも2本は通る。**通った本数が2本前後なら何も見つかっていない**のと同じ。

実行: python3 ml/audit_index_menu.py [シード数(既定1)] [開始年(既定2019)] [シャッフル回数(既定500)]
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
PAR_L5 = dict(PARAMS, num_leaves=255, min_child_samples=10, n_estimators=2000)

# ★券種ごとに代表1つ。事前宣言（結果を見てから足さない）
MENU = [
    ("tansho", "単勝 top1", lambda t, n: [(t[0],)], 4),
    ("fuku", "複勝 top1", lambda t, n: [(t[0],)], 4),
    ("wide", "ワイド BOX3",
     lambda t, n: [tuple(sorted(c)) for c in itertools.combinations(t[:3], 2)], 4),
    ("umaren", "馬連 BOX3",
     lambda t, n: [tuple(sorted(c)) for c in itertools.combinations(t[:3], 2)], 4),
    ("wakuren", "枠連 軸枠×紐枠2",
     lambda t, n: sorted({tuple(sorted((waku_of(t[0], n), waku_of(h, n)))) for h in t[1:3]}), 9),
    ("umatan", "馬単 1着固定×紐2", lambda t, n: [(t[0], h) for h in t[1:3]], 4),
    ("sanrenpuku", "三連複 BOX上位4",
     lambda t, n: [tuple(sorted(c)) for c in itertools.combinations(t[:4], 3)], 9),
    ("sanrentan", "三連単 3着固定×紐3",
     lambda t, n: [(a, b, t[0]) for a, b in itertools.permutations(t[1:4], 2)], 6),
]
IDX = [("A", "軸のシェア"), ("B", "軸と2位の差"), ("C", "枠連スコア"),
       ("D", "軸のシェア−市場含意"), ("E", "買い目の市場含意")]


def wakuren_cs(nums, n):
    wa = waku_of(nums[0], n)
    return sorted({tuple(sorted((wa, waku_of(h, n)))) for h in nums[1:3]})


def bracket_prob(nums, share, n):
    bp = {}
    for u, s in zip(nums, share):
        w = waku_of(u, n)
        bp[w] = bp.get(w, 0.0) + float(s)
    return bp


def wscore(pairs, bp):
    return sum(bp.get(a, 0.) ** 2 if a == b else 2 * bp.get(a, 0.) * bp.get(b, 0.)
               for a, b in pairs)


def rho_of(bins, vals):
    """区間番号とROIの順位相関。区間ごとの平均を取ってから相関を見る。"""
    s = pd.Series(vals).groupby(bins).mean()
    if len(s) < 4:
        return float("nan")
    return float(pd.Series(s.index, dtype=float).corr(pd.Series(s.values), method="spearman"))


def main():
    n_seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    y0 = int(sys.argv[2]) if len(sys.argv) > 2 else 2019
    n_shuf = int(sys.argv[3]) if len(sys.argv) > 3 else 500
    d, fx = load_cached()
    y = (d["finish"] <= 3).astype(int).to_numpy()
    year = d["date"].dt.year.to_numpy()
    pays = load_all(PAYOUT)
    years = [yy for yy in range(y0, int(year.max()) + 1) if (year == yy).sum() > 5000]
    print(f"指数の区間別ROI × 券種横断（{years[0]}〜{years[-1]}・L5・シード{n_seed}本"
          f"／シャッフル{n_shuf}回）")
    print(f"{len(MENU)}券種 × {len(IDX)}指数 = {len(MENU)*len(IDX)}曲線。"
          f"有意水準5%なら**偶然でも約{len(MENU)*len(IDX)*0.05:.0f}本**が通る\n")

    rows = []
    for yy in years:
        tr, te = year < yy, year == yy
        sub = d.loc[te, ["raceid", "umaban", "fieldsize", "finish", "odds"]].copy()
        ps = [lgb.LGBMClassifier(random_state=s, **PAR_L5)
              .fit(fx[tr], y[tr], categorical_feature=F.CAT_COLS)
              .predict_proba(fx[te])[:, 1] for s in range(n_seed)]
        sub["p"] = np.mean(ps, axis=0)
        for rid, g in sub.groupby("raceid", sort=False):
            pay = pays.get(rid)
            if not pay:
                continue
            n = int(g["fieldsize"].iloc[0])
            uma = g["umaban"].astype(int).to_numpy()
            od = g["odds"].to_numpy(float)
            inv = 1.0 / od
            mkt = inv / inv.sum()
            p = g["p"].to_numpy(float)
            o = np.argsort(-p, kind="mergesort")
            nums, share, mk = uma[o], p[o] / p.sum(), mkt[o]
            po = uma[np.argsort(od, kind="mergesort")]
            pairs = wakuren_cs(nums, n)
            r = {"n": n, "fs": min(n // 3, 5)}
            r["A"] = float(share[0])
            r["B"] = float(share[0] - share[1])
            r["C"] = float(wscore(pairs, bracket_prob(nums, share, n)))
            r["D"] = float(share[0] - mk[0])
            r["E"] = float(wscore(pairs, bracket_prob(nums, mk, n)))
            wnr = g[g["finish"] == 1]
            tansho = {(int(wnr["umaban"].iloc[0]),): float(wnr["odds"].iloc[0]) * 100} \
                if len(wnr) == 1 else {}
            for kind, name, fn, minf in MENU:
                table = tansho if kind == "tansho" else pay.get(kind)
                if not table or n < minf:
                    continue
                cs = fn(list(nums), n)
                r[name] = sum(table.get(c, 0) for c in cs) / (len(cs) * 100.0)
                cs2 = fn(list(po), n)
                r[name + "|pop"] = sum(table.get(c, 0) for c in cs2) / (len(cs2) * 100.0)
            rows.append(r)
        print(f"  {yy} 完了", flush=True)

    df = pd.DataFrame(rows)
    rng = np.random.default_rng(0)
    hits = []
    for kind, name, _, _ in MENU:
        if name not in df:
            continue
        dd = df[df[name].notna()]
        print(f"\n{'='*104}\n=== {name}  {len(dd):,}R  "
              f"全体 {dd[name].mean()*100:.2f}%（人気順 {dd[name+'|pop'].mean()*100:.2f}%）===")
        print(f"{'指数':<22}{'ρ':>8}{'帰無の95%点':>13}{'判定':>8}"
              f"{'第1区間':>9}{'第5区間':>9}{'第10区間':>9}{'最良区間':>10}")
        for code, iname in IDX:
            q = pd.qcut(dd[code], 10, labels=False, duplicates="drop").to_numpy()
            v = dd[name].to_numpy(float)
            rho = rho_of(q, v)
            # 帰無分布: 指数の区間ラベルを**頭数層内で**シャッフル
            fs = dd["fs"].to_numpy()
            idx_by = [np.flatnonzero(fs == b) for b in np.unique(fs)]
            null = np.empty(n_shuf)
            for i in range(n_shuf):
                perm = q.copy()
                for ix in idx_by:
                    perm[ix] = rng.permutation(perm[ix])
                null[i] = abs(rho_of(perm, v))
            cut = float(np.percentile(null, 95))
            ok = abs(rho) > cut
            hits.append((name, code, rho, cut, ok))
            gs = pd.Series(v).groupby(q).mean() * 100
            best = int(gs.idxmax()) + 1
            print(f"{code+' '+iname:<22}{rho:>+8.3f}{cut:>13.3f}{('★' if ok else ''):>8}"
                  f"{gs.iloc[0]:>8.1f}%{gs.iloc[min(4,len(gs)-1)]:>8.1f}%"
                  f"{gs.iloc[-1]:>8.1f}%{f'第{best}':>10}")

    n_ok = sum(1 for h in hits if h[4])
    print(f"\n{'='*104}")
    print(f"★多重性の判定: {len(hits)}曲線中 **{n_ok}本**が帰無分布の95%点を超えた"
          f"（偶然の期待値は約{len(hits)*0.05:.0f}本）")
    if n_ok <= len(hits) * 0.05 * 1.5:
        print("  ⇒ **期待値と同程度＝何も見つかっていない**のと同じ。指数で区間を切る意味は無い。")
    else:
        print("  ⇒ 期待値より多い。ただし通った曲線ごとに")
        print("     ★判定基準1(シード)・5(絞ると測れない)を当て直すこと。曲線が単調でも")
        print("     (46)より『上位区間だけ買う』は運用として検証しきれない。")
        for name, code, rho, cut, ok in hits:
            if ok:
                print(f"     ・{name} × {code}: ρ={rho:+.3f}（帰無95%点 {cut:.3f}）")
    print("\n※(62)ではL2で『上位に絞っても回収率は上がらない／下位20%除外だけが使える』が結論だった。")


if __name__ == "__main__":
    main()
