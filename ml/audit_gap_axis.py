"""(171) ★★ユーザー案: **軸を「モデルと市場の差が最大の馬」にして三連複で広く流す**

★★問い（2026-09-05・ユーザー）——
　「オッズ1,2番人気を推奨しても回収率は上がらなかった記憶。**モデルと市場が食い違う馬**が
　　いるはずで、**下まで見にいくとキリがないからある程度のところまで見て**、
　　**その馬を軸に三連複などで広く買う**とどうなる？」

★**この形では測っていない**。**(156)(157)はレースを切り取る話**で、**軸の選び方を変える話ではない**。
　**(48)(77)は買い方（軸1×紐3・BOX4等）を比べたが、軸は常にモデル確率1位**だった。

⚠⚠**先に強い逆風を置く（判定基準として）**——**(88)③④の実測**:
| オッズ帯 | 実測ROI | 「較正が完璧」の線=79.5%との差 |
|---|---|---|
| **1.3-1.4倍** | 88.9% | ★**+9.4pt**（**極端な本命の過小評価だけが本物**） |
| 10-30倍 | 82.3% | +2.9pt |
| 100-200倍 | 54.9% | **−24.6pt** |
| 200倍超 | 37.1% | **−42.4pt** |
★**市場の誤りは最大でも+10pt、埋めるべき控除率は25pt**（三連複）。
⚠**「モデルが市場より高く見ている馬」はたいてい穴寄り**なので、**この案はこの逆風を越える必要がある**。

★★★事前登録（**この8アームだけ。後から増やさない**）
　**軸の選び方4通り**:
　　**現行** … モデル確率1位
　　★**K=3** … **人気上位3頭の中で「差（モデルのシェア − 市場シェア）」が最大の馬**
　　★**K=5** … 同・上位5頭の中で
　　★**K=全** … 同・全頭の中で（＝「下まで見にいく」版）
　**紐の広さ2通り**: **紐4**（軸＋モデル上位4頭から2頭＝6点）／**紐6**（15点）
　**券種は三連複**（ユーザーの案どおり）。**1点100円**。

判定
　⚠**ゲート（判定基準32）**: **現行の三連複BOX4が既知84.5%を±2.5ptで再現しなければ読まない**。
　★**主判定**: **現行（同じ紐の広さ）との対応差（円・全レース）**。**Bonferroni α=0.01/6**。
　★★**採用の追加条件（判定基準39/40/41）**: **有意なだけでは足りない**。
　　**K=3 → K=5 → K=全 で単調でなければ「機構がある」と見なさない**——
　　**8マスのどれかが跳ねるのは偶然でも起きる**。
　★**判定基準8**: **円で比べる**（点数が違うので**ROIだけだと読み間違える**）。
　★**(88)の逆風と突き合わせる**: **軸の平均オッズも出す**。**穴寄りに寄るなら、それが理由**。

⚠**予想**: ★**(88)から言えるのは「穴寄りに寄るほど悪くなるはず」**。
　**K を広げるほど悪化するなら(88)の再現**であり、**新しい情報ではない**。
　**K=3 あたりだけ良いなら、それは(112)の裾の言い換えの可能性**（(119)(134)で2回踏んだ形）。

★★★実測（2026-09-05・26,791レース・ゲート: 現行BOX4 84.1% vs 既知84.5%＝−0.4pt → 立った）
| 軸の選び方 | 紐 | 点数 | ROI | 1R損益 | 軸の平均オッズ | 軸が確率1位 |
|---|---|---|---|---|---|---|
| 現行(確率1位) | 4 | 6 | 79.2% | −125.1円 | 2.9倍 | 100.0% |
| 現行(確率1位) | 6 | 15 | 77.7% | −334.0円 | 2.9倍 | 100.0% |
| ★差最大 K=3 | 4 | 6 | 79.3% | −124.4円 | 6.0倍 | 15.4% |
| ★差最大 K=5 | 4 | 6 | **77.9%** | −132.8円 | 9.8倍 | 6.4% |
| ★差最大 K=全 | 4 | 6 | 81.2% | −112.9円 | **18.6倍** | 3.1% |
（紐6は K=3 77.8 / K=5 78.1 / K=全 77.8%）
★**主判定: 6比較すべて有意差なし**（最大 K=全・紐4 の +12.2円・CI[−25.7,+50.0]）。
⚠**事前登録した単調性を満たさない**（79.3 → 77.9 → 81.2）。→ **機構があるとは見なさない**。

★★★**もっと重要な副産物: 「広く流す」形そのものが現行より悪い**
| 買い方 | 点数 | ROI | 1R損益 |
|---|---|---|---|
| ★**三連複BOX4（現行）** | 4 | ★**84.1%** | **−63円** |
| 軸1×紐4 | 6 | 79.2% | −125円 |
| 軸1×紐6 | **15** | 77.7% | **−334円** |
★**点数を増やすほどROIが下がり損失は5倍**。**(77)の「ROI<100%では『たくさん買う』は常に悪手」の再現**。

★**(88)の逆風がそのまま見えた**: **軸の平均オッズが 2.9 → 6.0 → 9.8 → 18.6倍**と穴側に寄る。
　**差が大きい馬＝市場が安く見ている馬＝穴**なので構造的にそうなる。
→ ★**結論: 「差が大きい馬を軸にする」は三連複ではROIを上げない**。**運用は変えない**。

実行: python3 ml/audit_gap_axis.py
"""
import math
import sys
from itertools import combinations

import numpy as np

sys.path.insert(0, "ml")
import features as F
from audit_crosspool import load_races, payoff, zq
from audit_crosspool2 import realized
from train_prod import CAPACITY, add_odds_features, fit_seeds

KNOWN, TOL, NCMP = 84.5, 2.5, 6
COST = 100.0
AXES = [("現行(確率1位)", 0), ("★差最大 K=3", 3), ("★差最大 K=5", 5), ("★差最大 K=全", 99)]
WIDTHS = [4, 6]


def main():
    MODEL_DIR, PAR = CAPACITY["l2"]
    d = F.to_model(F.load_files())
    f = F.build_features(d)
    keep = (f["n_prior"] >= 1) & d["odds"].notna() & (d["odds"] > 0)
    d, f = d[keep].reset_index(drop=True), f[keep].reset_index(drop=True)
    y = (d["finish"] <= 3).astype(int).to_numpy()
    fx, _ = F.encode_categoricals(f)
    fx = add_odds_features(fx, d["odds"].to_numpy(float), d["raceid"].to_numpy())
    cut = d["date"].quantile(0.3)
    tr, te = (d["date"] < cut).to_numpy(), (d["date"] >= cut).to_numpy()
    print("(171) ★軸を「モデルと市場の差が最大の馬」にして三連複で広く流す")
    print(f"　学習 {tr.sum():,} / 検証 {te.sum():,}・分割 {cut.date()}")
    print("⚠**(88): 市場の誤りは最大+10pt・穴側は単調に悪い**。**この逆風を越える必要がある**\n")
    ms = fit_seeds(fx[tr], y[tr], 3, PAR)
    p = np.mean([m.predict_proba(fx[te])[:, 1] for m in ms], axis=0)
    sub = d.loc[te, ["raceid", "umaban", "odds"]].copy()
    sub["p"] = p

    races = {r["rid"]: r for r in load_races()}
    rows = []
    for rid, g in sub.groupby("raceid"):
        r = races.get(str(rid))
        if r is None:
            continue
        rl = realized(r)
        if rl is None:
            continue
        a, b, c = rl
        nums = [u for u, _, _ in r["horses"]]
        if a not in nums or b not in nums or c not in nums:
            continue
        gg = g.sort_values("p", ascending=False, kind="mergesort")
        order = np.array([int(u) for u in gg["umaban"]])
        pv = gg["p"].to_numpy(float)
        od = gg["odds"].to_numpy(float)
        if len(order) < 7 or not np.isfinite(od).all() or (od <= 0).any():
            continue
        v = payoff(r, "三連複", sorted([a, b, c]))
        if v is None:
            continue
        share = pv / pv.sum()
        mk = (1.0 / od) / (1.0 / od).sum()
        gap = share - mk
        pop_rank = np.argsort(np.argsort(od))          # 0=1番人気
        real = tuple(sorted((a, b, c)))
        rec = {"v": v, "real": real, "order": order, "od": od}
        for nm, K in AXES:
            if K == 0:
                ax = int(order[0])
            else:
                m = pop_rank < K
                if not m.any():
                    ax = int(order[0])
                else:
                    idx = np.where(m)[0]
                    ax = int(order[idx[np.argmax(gap[idx])]])
            rec[f"ax{K}"] = ax
            rec[f"axod{K}"] = float(od[list(order).index(ax)])
        rows.append(rec)

    N = len(rows)
    print(f"★突き合わせ {N:,}レース\n")
    z = zq(0.01 / NCMP)

    def run(K, w):
        prof = np.zeros(N)
        cost = np.zeros(N)
        for i, rec in enumerate(rows):
            ax = rec[f"ax{K}"]
            partners = [int(u) for u in rec["order"] if int(u) != ax][:w]
            combos = [tuple(sorted((ax,) + pr)) for pr in combinations(partners, 2)]
            cost[i] = COST * len(combos)
            prof[i] = (rec["v"] if rec["real"] in combos else 0.0) - cost[i]
        return prof, cost

    # ゲート: 現行の三連複BOX4（軸＋上位3頭の4頭BOX）
    box = np.zeros(N); cb = np.zeros(N)
    for i, rec in enumerate(rows):
        top4 = sorted(int(u) for u in rec["order"][:4])
        combos = list(combinations(top4, 3))
        cb[i] = COST * len(combos)
        box[i] = (rec["v"] if rec["real"] in combos else 0.0) - cb[i]
    g0 = 100.0 * (box.sum() + cb.sum()) / cb.sum()
    ok = abs(g0 - KNOWN) <= TOL
    print(f"⚠ゲート: 現行の三連複BOX4 **{g0:.1f}%** vs 既知 {KNOWN}%　差 {g0-KNOWN:+.1f}pt"
          f" → **{'★立った' if ok else '⚠⚠落ちた'}**")
    if not ok:
        print("⚠⚠落ちた。読まない。"); return

    res = {}
    print(f"\n{'軸の選び方':<16}{'紐':>4}{'点数':>6}{'ROI':>9}{'1R損益':>10}"
          f"{'軸の平均オッズ':>14}{'軸が確率1位':>12}")
    for nm, K in AXES:
        for w in WIDTHS:
            pr, co = run(K, w)
            roi = 100.0 * (pr.sum() + co.sum()) / co.sum()
            axod = np.mean([rec[f"axod{K}"] for rec in rows])
            same = np.mean([rec[f"ax{K}"] == int(rec["order"][0]) for rec in rows])
            res[(K, w)] = (pr, co, roi)
            print(f"{nm:<16}{w:>4}{int(co[0]/COST):>6}{roi:>8.1f}%{pr.mean():>+9.1f}円"
                  f"{axod:>13.1f}倍{100*same:>11.1f}%")

    print(f"\n■ ★★主判定: 現行との対応差（**同じ紐の広さ・全レース**）")
    print(f"{'比較':<24}{'1R損益差':>11}{'99%CI(Bonf)':>21}{'判定':>16}")
    for w in WIDTHS:
        base = res[(0, w)][0]
        for nm, K in AXES[1:]:
            dd = res[(K, w)][0] - base
            md, sd = dd.mean(), dd.std(ddof=1) / math.sqrt(N)
            v = "★差がある" if abs(md) > z * sd else "⚠差は検出できない"
            print(f"{nm+f' 紐{w} − 現行':<24}{md:>+10.1f}円"
                  f"{f'[{md-z*sd:+.1f},{md+z*sd:+.1f}]':>21}{v:>16}")

    print("\n" + "=" * 92)
    print("★読み方（**事前登録のとおり**）")
    print("  ★**採用には「有意」だけでなく「K=3→5→全 で単調」が要る**（判定基準39/40/41）。")
    print("  ⚠**Kを広げるほど悪化するなら(88)の再現**——**新しい情報ではない**。")
    print("  ⚠**K=3あたりだけ良いなら(112)の裾の言い換えを疑う**（(119)(134)で2回踏んだ形）。")


if __name__ == "__main__":
    main()
