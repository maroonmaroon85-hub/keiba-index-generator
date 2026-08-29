"""(167) ★★★表に残った最後の空白＝**単勝**を厳密経路で測る。**ハードルが全券種で最も低い**

★★なぜ今か——**(161)で構造仮説（不人気なプールに住む）が死に、跡地に別の形が残った**。
| 券種 | 強い経路のROI | 払戻率 | 利益に要る比 |
|---|---|---|---|
| ★**枠連** | **112.5〜117.3%** | 77.5% | 1.290 |
| 複勝 | 89.0% | 80.0% | 1.250 |
| 馬連 | 42.8〜43.4% | 77.5% | 1.290 |
| 馬単 | 38.7% | 75.0% | 1.333 |
| 三連複 | 28.3% | 75.0% | 1.333 |
| ★★**単勝** | ❌**未測定** | **80.0%** | ★**1.250＝最も低い** |

★**`馬単 → 単勝` は厳密**（`q(i) = Σ_j q_馬単(i,j)`。順序を潰すだけ・Harville不使用）。
　**板は両方とも手元にある**。**一度もやっていない**。

★★★**事前登録した見立てと、それを殺す観測（判定基準39）**
　◇**見立て（post-hoc・未検定）**: **馬連・馬単・三連複・三連単は同じ客層**（高配当を追う層）
　　**が値を付けているので、互いに集約しても偏りを移し替えるだけ**（実測28〜43%＝控除率以下）。
　　**枠連だけが別の客層**（枠で買う古い商品）なので、馬の粒度の合意が新しい情報になる。
　★**予言**: **単勝も古くて素朴なプールなので、複勝(89%)と枠連(112%)のあいだに落ちるはず**。
　⚠⚠**これを殺す観測**: ★**単勝が60%台なら見立ては死ぬ**。**そのときは見立てを捨てる**。
　⚠**「当たったら仮説が正しい」とは言わない**——**1点で法則を書くのが(161)で死んだ形**。

測るもの（**この2つだけ。後から増やさない**）
　★**A 馬単→単勝**（厳密）**本命**
　★**B 三連単→複勝**（厳密。`q(i)=Σ_{iを含む三つ組}`・Σq=3）
　　 ← **(146)の複勝89.0%は三連複由来。別の独立な源で再現するかの対照**

判定
　⚠**ゲート1（判定基準38）**: ★**単勝の払戻率を板から実測する**。**0.800±1.0ptで無ければ読まない**。
　　 **(162)で馬単の定数が2.6pt間違っていた**ので、**単勝も信用しない**。
　⚠**ゲート2（判定基準37）**: **板に比例して買うROIが実測払戻率±1.0ptに乗ること**。
　⚠**ゲート3（判定基準32）**: **Bが(146)の89.0%を±5ptで再現すること**。
　★**主判定**: **Aの閾値1/R で 99%CI下端 > 100%**・**年91%割れ ≤2**・**2021-も100%超**。
　　 **Bonferroni α=0.01/2**。
　★**(160)の対照4つをAに当てる**（比例買い／オッズ層別／両側の裾／単調性）。

⚠⚠**陽性でも確定オッズのオラクル**。**張れる時点は(148)＝時系列オッズ待ちのまま**。

★★★実測（2026-08-29）
　ゲート1: 単勝の払戻率 実測 **79.47%**（定数80.0%どおり）→ 立った
　ゲート3: B 三連単→複勝 **89.8%** vs (146)の89.0%（**+0.8pt**）→ 立った
　★**A 馬単→単勝 主閾値1.250 で 69.8% [59.0,80.7]**・年91%割れ11・2021- 62.8%
　⚠⚠**予言「複勝89%と枠連112%のあいだ」は大外れ。事前に書いた通り見立ては死んだ**。
　★**単勝は自分の払戻率80%すら下回る**（−10.2pt）。**比の裾で選ぶとむしろ悪くなる**。

★**6マス埋まった表（強い経路・確定オッズ）**
| 券種 | ROI | 払戻率 | 差 |
|---|---|---|---|
| ★**枠連** | **112.5〜117.3%** | 77.5% | ★**+38.5pt** |
| 複勝 | 89.0 / **89.8%** | 80.0% | +9.8pt |
| **単勝** | **69.8%** | 80.0% | −10.2pt |
| 馬連 | 42.8〜43.4% | 77.5% | −34.5pt |
| 馬単 | 38.7% | 75.0% | −36.3pt |
| 三連複 | 28.3% | 75.0% | −46.7pt |
★★★**枠連は孤立した事実で、この表に説明変数は無い**。→ **判定基準40**。

実行: python3 ml/audit_tansho_tail.py [開始年(既定2015)]
"""
import math
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, "ml")
from audit_crosspool import load_races, payoff, zq
from audit_crosspool2 import realized
from audit_overlay_all import load_board

NCMP = 2
ODDS_EDGES = [0, 3, 6, 15, 50, 1e9]      # 単勝向けに事前に決め打ち（データを見ていない）
NDEC = 10
KNOWN_FUKU, TOL_F = 89.0, 5.0


def roi(rows, mask_fn, stake_fn=None):
    cost = ret = 0.0
    nb = 0
    prof, yl, cl = [], [], []
    for yy, rat, odds, win, pays in rows:
        m = mask_fn(rat, odds)
        if not m.any():
            continue
        st = np.full(int(m.sum()), 100.0) if stake_fn is None else stake_fn(odds[m])
        c = float(st.sum())
        g = float((st * (pays[m] / 100.0))[win[m]].sum()) if (m & win).any() else 0.0
        cost += c; ret += g; nb += int(m.sum())
        prof.append(g - c); yl.append(yy); cl.append(c)
    if cost <= 0 or len(prof) < 2:
        return None
    p = np.array(prof)
    se = p.std(ddof=1) / math.sqrt(len(p)) * len(p) / cost * 100.0
    return dict(roi=100.0 * ret / cost, nr=len(p), nb=nb, se=se, z=zq(0.01 / NCMP),
                prof=p, yy=np.array(yl), costs=np.array(cl), cost=cost)


def ybad(r):
    bad = 0
    for y in sorted(set(r["yy"].tolist())):
        m = r["yy"] == y
        c = float(r["costs"][m].sum())
        if c > 0 and 100.0 * (r["prof"][m].sum() + c) / c < 91.0:
            bad += 1
    return bad


def table(name, rows, rate):
    th = 1.0 / rate
    print(f"\n■ {name}（払戻率 {rate:.3f} → 利益に要る比 {th:.3f}・{len(rows):,}レース）")
    print(f"{'閾値':>9}{'買ったR':>9}{'点数':>10}{'ROI':>9}{'99%CI(Bonf)':>21}"
          f"{'年91%割れ':>11}{'2021-':>9}")
    main = None
    for t in [1.00, 1.10, None, 1.50, 2.00]:
        tt = th if t is None else t
        r = roi(rows, lambda a, o, tt=tt: a >= tt)
        if r is None:
            continue
        r21 = roi([x for x in rows if x[0] >= 2021], lambda a, o, tt=tt: a >= tt)
        ci = "[%.1f,%.1f]" % (r["roi"] - r["z"] * r["se"], r["roi"] + r["z"] * r["se"])
        s21 = ("%.1f%%" % r21["roi"]) if r21 else "—"
        print(f"{tt:>8.3f}{'★' if t is None else ' '}{r['nr']:>9,}{r['nb']:>10,}"
              f"{r['roi']:>8.1f}%{ci:>21}{ybad(r):>11}{s21:>9}")
        if t is None:
            main = r
    return main


def main():
    y0 = int(sys.argv[1]) if len(sys.argv) > 1 else 2015
    print(f"(167) ★表に残った空白＝**単勝**を厳密経路で測る（{y0}年以降）")
    print("　★A: q = **馬単の板 → 単勝（厳密）** / q_pool = **単勝オッズ**")
    print("　★B: q = **三連単の板 → 複勝（厳密）** / q_pool = **複勝の板**（(146)の89.0%の対照）")
    tb = load_board(6, 4)
    fb = load_board(2, 2)
    races = load_races()
    rowsA, rowsB = [], []
    from nk_odds_bulk import iter_records
    from nk_parse import nk_raceid
    rmap = {r["rid"]: r for r in races if r["year"] >= y0}

    # ── A: 馬単 → 単勝 ──
    for rid, r in rmap.items():
        T = tb.get(rid)
        if not T:
            continue
        rl = realized(r)
        if rl is None:
            continue
        a = rl[0]
        hs = r["horses"]
        nums = [u for u, _, _ in hs]
        od = {u: o for u, o, _ in hs if o and o > 0}
        if a not in od:
            continue
        w = defaultdict(float)
        for k, o in T.items():
            if len(k) != 4 or not k.isdigit() or o <= 0:
                continue
            i, j = int(k[:2]), int(k[2:])
            if i in od and j in od and i != j:
                w[i] += 1.0 / o
        keys = [u for u in sorted(w) if u in od]
        if a not in keys or len(keys) < 5:
            continue
        v = payoff(r, "単勝", [a])
        if not v or v <= 0:
            continue
        q = np.array([w[u] for u in keys]); q /= q.sum()
        oo = np.array([od[u] for u in keys])
        inv = 1.0 / oo; qp = inv / inv.sum()
        rowsA.append((r["year"], q / qp, oo, np.array([u == a for u in keys]),
                      np.array([v if u == a else 0.0 for u in keys])))

    # ── B: 三連単 → 複勝（板が巨大なのでストリーミング） ──
    for rec in iter_records(8):
        rid = nk_raceid(rec["race_id"])
        r = rmap.get(rid) if rid else None
        F = fb.get(rid) if r else None
        if r is None or not F:
            continue
        rl = realized(r)
        if rl is None:
            continue
        nums = {u for u, _, _ in r["horses"]}
        if not set(rl) <= nums:
            continue
        w = defaultdict(float)
        for k, v0 in rec["odds"].items():
            o = v0[0] if isinstance(v0, (list, tuple)) else v0
            if len(k) != 6 or not k.isdigit() or not o or float(o) <= 0:
                continue
            i, j, m3 = int(k[:2]), int(k[2:4]), int(k[4:])
            if len({i, j, m3}) < 3 or not {i, j, m3} <= nums:
                continue
            iw = 1.0 / float(o)
            w[i] += iw; w[j] += iw; w[m3] += iw
        fo = {}
        for k, v0 in F.items():
            if len(k) == 2 and k.isdigit():
                lo = v0[0] if isinstance(v0, (list, tuple)) else v0
                if lo and float(lo) > 0:
                    fo[int(k)] = float(lo)
        keys = [u for u in sorted(w) if u in fo]
        if len(keys) < 5:
            continue
        win = np.array([u in set(rl) for u in keys])
        if not win.any():
            continue
        pays = np.array([(payoff(r, "複勝", [u]) or 0.0) if u in set(rl) else 0.0
                         for u in keys])
        if (pays[win] <= 0).any():
            continue
        q = np.array([w[u] for u in keys]); q /= q.sum()
        oo = np.array([fo[u] for u in keys])
        inv = 1.0 / oo; qp = inv / inv.sum()
        rowsB.append((r["year"], q / qp, oo, win, pays))

    print(f"\n　A {len(rowsA):,}レース / B {len(rowsB):,}レース")
    if not rowsA:
        sys.exit("Aが作れない")

    # ★ゲート1: 単勝の払戻率を板から実測する（判定基準38）
    ors = np.array([float((1.0 / r[2]).sum()) for r in rowsA])
    Rm = 1.0 / float(np.median(ors))
    prop = roi(rowsA, lambda a, o: np.ones_like(a, dtype=bool),
               stake_fn=lambda o: 100.0 / o)
    print(f"\n■ ⚠ゲート1（判定基準38）: **単勝の払戻率を板から実測する**")
    print(f"　Σ1/odds 中央値 {np.median(ors):.4f} → **R_implied {100*Rm:.2f}%**"
          f"／比例買い **{prop['roi']:.2f}%**　（コードの定数 80.0%）")
    ok1 = abs(prop["roi"] - 80.0) <= 1.0
    print(f"　→ **{'★立った（定数どおり）' if ok1 else '⚠⚠定数と食い違う。馬単と同じ形'}**")
    R = 0.800 if ok1 else Rm
    if not ok1:
        print(f"　★**実測値 {100*R:.2f}% を採用して続ける**（(162)と同じ扱い）")

    mainA = table("★A 馬単→単勝（厳密）", rowsA, R)

    if rowsB:
        mB = table("B 三連単→複勝（厳密）＝(146)の対照", rowsB, 0.800)
        if mB:
            d = mB["roi"] - KNOWN_FUKU
            print(f"\n　⚠ゲート3: 複勝 **{mB['roi']:.1f}%** vs (146)の {KNOWN_FUKU}%"
                  f"　差 {d:+.1f}pt → **{'★立った' if abs(d) <= TOL_F else '⚠⚠立っていない'}**")

    # ★(160)の対照4つをAに当てる
    print("\n■ ★(160)の対照を A に当てる")
    flat = roi(rowsA, lambda a, o: np.ones_like(a, dtype=bool))
    print(f"　①比例買い {prop['roi']:.2f}%（払戻率 {100*R:.1f}%）／均等買い {flat['roi']:.1f}%")
    th = 1.0 / R
    print(f"　★②オッズ層別")
    print(f"{'オッズ層':>16}{'全部買う':>11}{f'比≥{th:.3f}':>11}{'差(pt)':>10}{'99%CI(差)':>21}")
    for lo, hi in zip(ODDS_EDGES[:-1], ODDS_EDGES[1:]):
        b = roi(rowsA, lambda a, o, lo=lo, hi=hi: (o >= lo) & (o < hi))
        s = roi(rowsA, lambda a, o, lo=lo, hi=hi: (o >= lo) & (o < hi) & (a >= th))
        nm = f"{lo:g}〜{hi:g}倍" if hi < 1e8 else f"{lo:g}倍〜"
        if b is None or s is None:
            print(f"{nm:>16}  ——"); continue
        dd = s["roi"] - b["roi"]; sd = math.hypot(b["se"], s["se"]); z = b["z"]
        print(f"{nm:>16}{b['roi']:>10.1f}%{s['roi']:>10.1f}%{dd:>+10.1f}"
              f"{f'[{dd-z*sd:+.1f},{dd+z*sd:+.1f}]':>21}")
    hi_ = roi(rowsA, lambda a, o: a >= th)
    lo_ = roi(rowsA, lambda a, o: a <= R)
    print(f"　★③両側の裾  高い側 {hi_['roi']:.1f}% ／ 低い側 {lo_['roi']:.1f}%")
    allrat = np.concatenate([r[1] for r in rowsA])
    ed = np.quantile(allrat, np.linspace(0, 1, NDEC + 1)); ed[0], ed[-1] = -np.inf, np.inf
    prev, mono, vals = None, 0, []
    for i in range(NDEC):
        rr = roi(rowsA, lambda a, o, l=ed[i], h=ed[i + 1]: (a >= l) & (a < h))
        if rr is None:
            continue
        vals.append(f"{rr['roi']:.0f}")
        if prev is not None and rr["roi"] > prev:
            mono += 1
        prev = rr["roi"]
    print(f"　★④単調性 {' → '.join(vals)}%　上がった回数 **{mono}/{len(vals)-1}**")
    print("　⚠**④は半分自動**（ROI ∝ p_true/q_pool）。**100%超の証拠ではない**")

    print("\n" + "=" * 92)
    print("★★事前登録した見立てとの突き合わせ（**当たっても外れても消さない**）")
    if mainA:
        print(f"　予言「単勝は複勝89%と枠連112%のあいだ」→ **実測 {mainA['roi']:.1f}%**")
        print(f"　⚠⚠**60%台なら見立ては死ぬ** → "
              f"**{'見立ては死んだ' if mainA['roi'] < 70 else '殺されはしなかった'}**")
        print("　⚠**当たっても「法則が正しい」とは言わない**（(161)で1点法則が死んだ形）。")
    print("⚠⚠**確定オッズのオラクル。張れる時点は(148)待ちのまま**。")


if __name__ == "__main__":
    main()
