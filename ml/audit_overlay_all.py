"""(150) ★★★★馬連と三連複を**比の裾**で当て直す — 判定基準29の宿題（2026-08-13）

★なぜやるか
　(89)④は5券種をDで閉じた: 枠連 +0.0145 / **馬連 −0.0173 / 馬単 −0.0596 / 三連複 −0.0665** / 三連単 −0.0888。
　だが**(141)でその論法が誤りだと分かった**——`log(払戻率)+D` は上界ではなく下界で、
　**利益の条件は「Dが必要量に届くこと」ではなく「∃組: q/q_pool ≥ 1/払戻率」＝比の裾**。
　→ **判定基準29**（基準を足したら過去の結論に遡って当て直す）。
　**枠連は(141)、複勝は(146)でやった。馬連と三連複が残っている**。**手元の板で測れる**。

★★★ここが肝: **三連複は組が816ある**（馬連153・枠連36）。
　**比の分布の裾は組数が多いほど厚い**。**Dが平均で負でも、816組の裾に1.333超が居る可能性は
　枠連の36組より高い**。→ ★**Dが一番悪かった券種ほど、裾では有望かもしれない**。**測る価値がある**。

⚠★★正直に書く限界（先に書く。これが一番大事）
　比 `q/q_pool` に意味があるのは **q と q_pool が別の市場から来ているとき**だけ。
　枠連では **馬連プール→枠**（Harville不使用）が使えたが、
　**馬連と三連複には「それを厳密に集約できるより細かい板」が手元に無い**
　（三連単の板 type=8 を集めれば作れるが、未収集）。
　→ ★**ここで使える q は λ補正Harville(単勝)しかない**。**これは弱いルートだと実測済み**:
　　 (141)で 枠連の比の裾は **馬連ルート117.3% / λHarvilleルート93.9%**。
　→ ⚠**だから陰性が出ても「比の裾で閉じた」とは書けない**。書けるのは
　　 **「単勝由来のqでは取れない」**まで。**三連単の板を集めれば別の答えになりうる**。
　　 **この読み分けを測る前に決めておく**（判定基準25で4回踏んだ形の予防）。

★★事前登録
　1. **券種は 馬連(払戻率0.775→閾値1.290) と 三連複(0.75→1.333)**。**後から増やさない**。
　2. **閾値の梯子 1.00 / 1.10 / 1/払戻率 / 1.50 / 2.00**。
　3. **賭け方**: 比が閾値以上の**全組を1点100円**。該当無しは見送り。**選ばない**。
　4. **判定は race単位の収支**。**運用判定: ROIの99%CI下端 > 100%** かつ
　　 **年別で「91%を割る年が2つ以下」**（ユーザーの許容水準・2026-08-13）。
　5. **時間分割（2021-2026）も本判定に入れる**（(142)の教訓）。
　6. **道具の検算**: 的中組で「板×100 と実配当」を突き合わせる。(127)は0.07%だった。
　7. **プラセボ**: 無作為に同数買う。**解析的に出す**（判定基準23）。
　8. **予想**: ★**当てにしてよい予想は持っていない**（類推はこの3日で4連敗）。

実行: python3 ml/audit_overlay_all.py [開始年(既定2015)]
"""
import math
import sys
from itertools import combinations

import numpy as np

sys.path.insert(0, "ml")
from audit_crosspool import PAYBACK, load_races, payoff, probs, zq
from audit_crosspool2 import PAYKEY, realized
from audit_lbs import build_matrix, fit_lambda

NPLA_NOTE = "解析的（判定基準23）"


def load_board(t, keylen):
    from nk_odds_bulk import iter_records
    from nk_parse import nk_raceid
    out = {}
    for rec in iter_records(t):
        r8 = nk_raceid(rec["race_id"])
        if not r8:
            continue
        d = {}
        for k, v in rec["odds"].items():
            if len(k) != keylen or not k.isdigit():
                continue
            o = v[0] if isinstance(v, (list, tuple)) else v
            if o and float(o) > 0:
                d[k] = float(o)
        if d:
            out[r8] = d
    return out


def harville_pair(p, l2):
    """λ補正Harvilleの馬連確率（1着2着の順不同）。O(n^2)。"""
    n = len(p)
    w = p ** l2
    out = {}
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            den = w.sum() - w[i]
            if den <= 0:
                continue
            out[(i, j)] = p[i] * w[j] / den
    return {(i, j): out.get((i, j), 0.0) + out.get((j, i), 0.0)
            for i, j in combinations(range(n), 2)}


def harville_trio(p, l2, l3):
    """λ補正Harvilleの三連複確率（上位3頭の集合）。O(n^3)。"""
    n = len(p)
    w2, w3 = p ** l2, p ** l3
    S2, S3 = w2.sum(), w3.sum()
    out = {}
    for a, b, c in combinations(range(n), 3):
        tot = 0.0
        for i, j, k in ((a, b, c), (a, c, b), (b, a, c), (b, c, a), (c, a, b), (c, b, a)):
            d2 = S2 - w2[i]
            d3 = S3 - w3[i] - w3[j]
            if d2 > 0 and d3 > 0:
                tot += p[i] * (w2[j] / d2) * (w3[k] / d3)
        out[(a, b, c)] = tot
    return out


def main():
    y0 = int(sys.argv[1]) if len(sys.argv) > 1 else 2015
    races = load_races()
    ub, tb = load_board(4, 4), load_board(7, 6)
    if not ub or not tb:
        sys.exit("馬連(--type 4)と三連複(--type 7)の板が要る。")
    P, i1, i2, i3, yrs = build_matrix(races, y0)
    lam = {}
    for yy in sorted(set(yrs.tolist())):
        tr = yrs < yy
        if tr.sum() < 3000:
            lam[yy] = None
            continue
        ok3 = tr & (i3 >= 0)
        lam[yy] = (fit_lambda(P[tr], i1[tr], i2[tr]),
                   fit_lambda(P[ok3], i1[ok3], i2[ok3], stage3=True, ic=i3[ok3]))

    print("(150) 馬連と三連複を比の裾で当て直す（判定基準29の宿題）")
    print("⚠**qは λ補正Harville(単勝)しか無い**。(141)でこれは弱いルートと実測済み")
    print("　（枠連の比の裾: 馬連ルート117.3% / λHarvilleルート93.9%）。")
    print("　→ **陰性でも『比の裾で閉じた』とは書けない。書けるのは『単勝由来のqでは取れない』まで**\n")

    for kind, board, keylen, comb in (("馬連", ub, 4, 2), ("三連複", tb, 6, 3)):
        R = PAYBACK[kind]
        THS = [1.00, 1.10, 1.0 / R, 1.50, 2.00]
        rows, bad, hits = [], 0, 0
        for r in races:
            yy = r["year"]
            if yy < y0 or not lam.get(yy):
                continue
            B = board.get(r["rid"])
            if not B:
                continue
            rl = realized(r)
            if rl is None:
                continue
            a, b, c = rl
            nums = [u for u, _, _ in r["horses"]]
            idx = {u: i for i, u in enumerate(nums)}
            need = (a, b) if comb == 2 else (a, b, c)
            if any(x is None or x not in idx for x in need):
                continue
            key = tuple(sorted(idx[x] for x in need))
            p = probs(r["horses"])
            l2, l3 = lam[yy]
            md = harville_pair(p, l2) if comb == 2 else harville_trio(p, l2, l3)
            bd = {}
            for k, o in B.items():
                t = tuple(sorted(int(k[2 * m:2 * m + 2]) for m in range(comb)))
                if all(x in idx for x in t):
                    bd[tuple(sorted(idx[x] for x in t))] = o
            keys = [k for k in sorted(md) if k in bd]
            if key not in keys or len(keys) < 5:
                continue
            v = payoff(r, PAYKEY[kind], [nums[i] for i in key])
            if not v or v <= 0:
                continue
            hits += 1
            if abs(bd[key] * 100 - v) > max(10.0, v * 0.01):
                bad += 1
            inv = np.array([1.0 / bd[k] for k in keys])
            qp = inv / inv.sum()
            qq = np.array([md[k] for k in keys])
            qq /= qq.sum()
            odds = np.array([bd[k] for k in keys])
            win = np.array([k == key for k in keys])
            rows.append((yy, qq / qp, odds, win))

        if not rows:
            print(f"■ {kind}: 突き合わせできたレースが無い\n")
            continue
        ys = np.array([x[0] for x in rows])
        print(f"■ {kind}（払戻率 {R:.3f} → 利益に要る比 {1/R:.3f}・{len(rows):,}レース）")
        print(f"　★道具の検算: 的中 {hits:,} 件のうち板×100と実配当のずれ {bad}"
              f"（{bad/max(hits,1):.2%}）  ※(127)は0.07%")
        print(f"{'閾値':>7}{'買ったR':>9}{'点数':>10}{'的中':>7}{'ROI':>9}{'99%CI':>21}"
              f"{'年91%割れ':>10}{'2021-':>9}{'プラセボ':>9}")
        for t in THS:
            prof, cost, ret, nb, hit, yl = [], 0.0, 0.0, 0, 0, []
            plr, plc = 0.0, 0.0
            r21, c21 = 0.0, 0.0
            for (yy, rat, odds, win) in rows:
                sel = rat >= t
                if not sel.any():
                    continue
                cc = 100.0 * sel.sum()
                vv = float(odds[sel & win].sum() * 100.0)
                prof.append(vv - cc)
                yl.append(yy)
                cost += cc
                ret += vv
                nb += int(sel.sum())
                hit += int((sel & win).sum())
                plr += (100.0 * odds[win].sum() / len(odds)) * int(sel.sum())
                plc += cc
                if yy >= 2021:
                    r21 += vv
                    c21 += cc
            if not prof:
                print(f"{t:>7.3f}   該当なし")
                continue
            pr = np.array(prof)
            mc = cost / len(pr)
            se = pr.std(ddof=1) / math.sqrt(len(pr))
            z = zq(0.01)
            lo, hi = 1 + (pr.mean() - z * se) / mc, 1 + (pr.mean() + z * se) / mc
            ya = np.array(yl)
            bad_y = sum(1 for u in sorted(set(ya.tolist()))
                        if (ya == u).sum() >= 30
                        and 1 + pr[ya == u].mean() / (100.0 * nb / max(len(pr), 1)) < 0.91)
            mark = "★★買える" if lo > 1.0 else ""
            print(f"{t:>7.3f}{len(pr):>9,}{nb:>10,}{hit:>7,}{100*ret/cost:>8.1f}%"
                  f"{'[' + format(100*lo, '.1f') + ',' + format(100*hi, '.1f') + ']':>21}"
                  f"{bad_y:>10}{100*r21/max(c21,1):>8.1f}%{100*plr/max(plc,1):>8.1f}% {mark}")
        print()

    print("=" * 100)
    print("★読み方（事前登録のとおり）")
    print("  ・**ROIの99%CI下端>100% かつ 年91%割れが2つ以下 かつ 2021-2026でも100%超**なら★。")
    print("  ⚠**陰性でも『比の裾で閉じた』とは書かない**。qが λHarville(単勝) しか無いため。")
    print("    書けるのは **『単勝由来のqでは取れない』** まで。")
    print("    ★**三連単の板(type=8)を集めれば、三連単→三連複／三連単→馬連 が厳密に作れる**。")
    print("      (127)で枠連に効いたのと同じ形。**そこが次の分岐**。")


if __name__ == "__main__":
    main()
