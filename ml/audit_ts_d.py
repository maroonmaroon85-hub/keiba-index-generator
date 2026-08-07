"""(109) ★★★時系列オッズの筋を決着させる — 361レースで**実際にDを測る**。

★(108)で分かったこと
　**儲かるサイズの効果(+0.2367)なら10レースで検出できる**。手元には既に361レースある。
　＝**測れば決着する**。(92)⑤(94)④は40頭程度の狭い帯のROIしか見ておらず、
　　**361レース全部でDを測ったことは一度も無い**。ここを埋める。

★問い（1つに絞る）
　**「発走前オッズの動きは、確定オッズに無い情報を持つか」**
　持たないなら (89)④の上界がそのまま効き、**この道は完全に閉じる**。

★測り方
　q_final … 確定オッズのHarville（λ補正つき）＝(96)の最良形。これが基準。
　q_move  … 確定オッズを**動きで補正**したもの。補正の形は2つ用意する:
　　 A `drift`  … 「前日22時→確定」でオッズが**下がった馬**（＝売れた馬）を強める。
　　 　　 q ∝ p_final · (p_final/p_prev)^γ   γ>0 なら「直前に売れた馬をさらに買う」
　　 B `revert` … その逆（γ<0）。**どちらの向きが正しいかは事前に決めない**ので、
　　 　　 γ をグリッドで動かして**対数尤度が最大になる向きを見る**。
　　 ★γは**レースをまたいで1つ**。361レースしか無いので自由度を1に絞る。

★★事前登録（測る前に宣言）
　1. **予想**: 文献では「直前に売れた馬は勝ちやすい（smart money）」なので **γ>0** と予想する。
　2. **判定**: `D(q_move) − D(q_final)` の99%CIが0を除外して正なら「動きは情報を持つ」。
　　 ★**γを同じデータで当てはめるので、これは上界の推定になる**（実運用の値ではない）。
　　 　 上界ですら小さいなら、この道は閉じる。**わざと有利にして潰す**という(101)と同じ作法。
　3. **標本が361レースしか無い**ので、CIは±0.03程度になる見込み。
　　 → **+0.2367（儲かるのに必要な残り）は決定的に判定できる**が、
　　 　 **+0.0024（モデル混合と同等）は判定できない**。これは事前に承知の上。
　4. **前日22時が取れないレースは除く**。カバー率を必ず出す。

実行: python3 ml/audit_ts_d.py
"""
import math
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "ml")
from audit_crosspool import PAYBACK, load_races, payoff, probs, zq
from audit_crosspool2 import PARTS, PAYKEY, realized
from audit_lbs import build_matrix, fit_lambda, q_of_lbs
from odds_ts import load_dir, odds_at

GAMMA = np.round(np.arange(-1.0, 1.01, 0.05), 3)     # ★向きは事前に決めない


def mci(x, alpha=0.01):
    x = np.asarray(x, float)
    m = x.mean()
    se = x.std(ddof=1) / math.sqrt(len(x))
    z = zq(alpha)
    return m, m - z * se, m + z * se


def norm(o):
    inv = 1.0 / np.maximum(o, 1e-9)
    return inv / inv.sum()


def main():
    recs = load_dir()
    print(f"(109) 時系列オッズの筋を決着させる（{len(recs)}レース）")
    print("★問い: 発走前オッズの動きは、確定オッズに無い情報を持つか\n")

    races = {r["rid"]: r for r in load_races()}
    P, i1, i2, i3, yrs = build_matrix(list(races.values()), 2015)
    lam = {}
    for yy in sorted(set(yrs.tolist())):
        tr = yrs < yy
        if tr.sum() < 3000:
            continue
        ok3 = tr & (i3 >= 0)
        lam[yy] = (fit_lambda(P[tr], i1[tr], i2[tr]),
                   fit_lambda(P[ok3], i1[ok3], i2[ok3], stage3=True, ic=i3[ok3]))
    lam_last = lam[max(lam)]

    # ── 各レースの (確定確率, 過去時点の確率, 勝ち馬index, race) を集める ──
    pts = [("前日22時", ("prev", 22, 0)), ("当日9時", ("day", 9, 0)),
           ("発走30分前", ("before", 30)), ("発走10分前", ("before", 10))]
    data = {lab: [] for lab, _ in pts}
    n_tot = n_ok = 0
    for rid, rec in recs.items():
        r = races.get(rid)
        n_tot += 1
        if r is None or realized(r) is None:
            continue
        a, b, c = realized(r)
        num2k = {num: k for k, (num, _, _) in enumerate(r["horses"])}
        if a not in num2k or b not in num2k:
            continue
        of = odds_at(rec, ("final",))
        if of is None or len(of) != len(r["horses"]):
            continue
        n_ok += 1
        pf = norm(np.asarray(of, float))
        for lab, when in pts:
            op = odds_at(rec, when)
            if op is None or len(op) != len(pf):
                continue
            arr = np.asarray(op, float)
            if (arr <= 0).any():
                continue
            data[lab].append((pf, norm(arr), r))
    print(f"  照合できたレース: {n_ok}/{n_tot}")
    for lab, _ in pts:
        print(f"    {lab}: {len(data[lab])}レース")

    print(f"\n{'='*100}")
    print("【1】γ（動きの効かせ方）を対数尤度で選ぶ — ★向きは事前に決めていない")
    print("=" * 100)
    print(f"{'時点':<12}{'R数':>7}{'最適γ':>9}{'解釈':<34}")
    best_g = {}
    for lab, _ in pts:
        rows = data[lab]
        if len(rows) < 100:
            continue
        def ll(g):
            s = 0.0
            for pf, pp, r in rows:
                q = pf * (pf / pp) ** g
                q = q / q.sum()
                k = {num: i for i, (num, _, _) in enumerate(r["horses"])}[realized(r)[0]]
                s += math.log(max(q[k], 1e-12))
            return s
        g = float(max(GAMMA, key=ll))
        best_g[lab] = g
        note = "直前に売れた馬をさらに買う" if g > 0 else \
               ("直前に売れた馬を割り引く" if g < 0 else "動きを使わないのが最適")
        print(f"{lab:<12}{len(rows):>7}{g:>9.2f}  {note:<34}")

    print(f"\n{'='*100}")
    print("【2】★D の比較 — 動きを使うと情報は増えるか（γは同じデータで当てはめた＝上界）")
    print("=" * 100)
    for lab, _ in pts:
        rows = data[lab]
        if len(rows) < 100 or lab not in best_g:
            continue
        g = best_g[lab]
        res = {k: [] for k in PARTS}
        for pf, pp, r in rows:
            a, b, c = realized(r)
            num2k = {num: k for k, (num, _, _) in enumerate(r["horses"])}
            if c is not None and c not in num2k:
                c = None
            qm = pf * (pf / pp) ** g
            qm = qm / qm.sum()
            l2, l3 = lam.get(r["year"], lam_last)
            for kind, key in PARTS.items():
                if not r[key]:
                    continue
                q1, combo = q_of_lbs(kind, r, pf, l2, l3, num2k, a, b, c)
                q2, _ = q_of_lbs(kind, r, qm, l2, l3, num2k, a, b, c)
                if q1 <= 0 or q2 <= 0 or combo is None:
                    continue
                v = payoff(r, PAYKEY[kind], combo)
                if not v or v <= 0:
                    continue
                lp = math.log((v + 5) / 100.0) - math.log(PAYBACK[kind])
                res[kind].append((math.log(q2) + lp, math.log(q1) + lp))
        print(f"\n■ {lab}（γ={g:+.2f}）")
        print(f"{'券種':<8}{'R数':>7}{'D(動きあり)':>13}{'D(確定のみ)':>13}"
              f"{'利得':>11}{'利得の99%CI':>24}{'必要量の残り':>14}")
        for kind in PARTS:
            v = res[kind]
            if len(v) < 100:
                continue
            dm = np.array([x[0] for x in v])
            dk = np.array([x[1] for x in v])
            gg, lo, hi = mci(dm - dk)
            need = -math.log(PAYBACK[kind]) - dk.mean()
            print(f"{kind:<8}{len(v):>7}{dm.mean():>+13.4f}{dk.mean():>+13.4f}{gg:>+11.4f}"
                  f"{f'[{lo:+.4f},{hi:+.4f}]':>24}{need:>14.4f}")

    print(f"\n{'='*100}")
    print("★読み方")
    print("  ・γを同じデータで当てはめているので、**この利得は上界**（実運用ではこれ以下）。")
    print("    上界ですら『必要量の残り』に遠いなら、**この道は閉じる**。")
    print("  ・(108)より361レースのCIは±0.03程度。**+0.2367は決定的に判定できる**が、")
    print("    **+0.0024（モデル混合と同等）は判定できない**。それは事前に承知の上。")
    print("  ・γの符号が時点で一貫しないなら、それは**推定ノイズ**であって smart money ではない。")


if __name__ == "__main__":
    main()
