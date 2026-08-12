"""(125) ★★★枠連プールは単勝プールの情報を取り込みきれているか — (113)(B)の枠連版

★なぜ枠連でやるか（三連複より意味が大きい）
　(113)(B)で **三連複プールは単勝プールの情報を取り込みきれていない**と分かった
　（`q ∝ q_pool^(1−w)·q_λ^w` のウォークフォワードで +0.0026・99%CI[+0.0017,+0.0035]・
　　選ばれるwは13年すべて0.20〜0.25）。だが**三連複はそもそも閉じた券種**で運用に使わない。
　**枠連は我々の最良プール**（D=+0.0207／2%裾で+0.0394）で、**実際に買っている券種**。
　同じ非効率が枠連にあるなら、**現行のqにそのまま足せる**。

★板が無いとできない理由（(113)(B)と同じ）
　`q ∝ q_pool^(1−w)·q_λ^w` は**Σを1にするために全枠組のオッズが要る**。
　実配当は的中した組の分しか無いので、板が無ければ正規化できず混合が作れない。

★★事前登録（測る前に宣言）
　1. w は **0/0.05/0.1/0.2/0.3/0.5/0.7/1.0** の8点（(113)(B)と同じ）。後から増やさない。
　2. **ウォークフォワードでwを選ぶ**（その年より前の年だけで決める）。これが主結果。
　3. **年分割で符号が揃うか**を必ず見る。(113)(B)は11/13年で正だった。
　4. **道具の検算**: 的中組の「板のオッズ×100」と実配当を突き合わせる。
　　 (113)では 5/3,681（0.14%）しかずれなかった。同水準でなければ**先に道具を疑う**。
　5. **予想**: **+0.002〜+0.004 程度は出る**（三連複と同オーダー）。
　　 理由: 枠連は三連複より薄いプール（組が少ない）なので、単勝からの情報の残り方は
　　 　　　同程度かやや大きいと見る。**ただし必要量 0.2549 の 1〜2% にすぎない**。
　　 ⚠(117)(122)(123)で予想を3回外している。この予想も外れる前提で読むこと。
　6. **★運用が変わる条件を先に書く**: ウォークフォワードのDが **+0.005を超え**、かつ
　　 **年分割で10/13年以上**正なら、**現行のqに枠連プールを混ぜる**価値がある。
　　 それ未満なら記述にとどめ、**運用は変えない**。

実行（板を集めたあと）: python3 ml/audit_waku_board.py [開始年(既定2015)]
　　★先にMacで: python3 ml/nk_odds_bulk.py --type 3   （枠連は9頭以上でしか発売されない）
"""
import math
import sys

import numpy as np

sys.path.insert(0, "ml")
from audit_crosspool import PAYBACK, load_races, payoff, probs, zq
from audit_crosspool2 import PAYKEY, realized
from audit_capacity_d import mkt_waku_dist
from audit_lbs import build_matrix, fit_lambda, q_of_lbs
from nk_parse import nk_raceid

NEED = -math.log(PAYBACK["枠連"])          # 0.2549
WS = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0)


def mci(x, alpha=0.01):
    x = np.asarray(x, float)
    if len(x) < 2:
        return float("nan"), float("nan"), float("nan")
    m = x.mean()
    se = x.std(ddof=1) / math.sqrt(len(x))
    return m, m - zq(alpha) * se, m + zq(alpha) * se


def load_boards():
    """race_id(8桁) → {(枠a,枠b): オッズ}。枠連の板キーは '0102' のような枠2桁ずつ。"""
    from nk_odds_bulk import iter_records
    out = {}
    for rec in iter_records(3):
        rid8 = nk_raceid(rec["race_id"])
        if not rid8:
            continue
        d = {}
        for k, v in rec["odds"].items():
            if len(k) != 4 or not k.isdigit():
                continue
            o = v[0] if isinstance(v, (list, tuple)) else v
            if o and o > 0:
                d[(int(k[:2]), int(k[2:]))] = float(o)
        if d:
            out[rid8] = d
    return out


def main():
    y0 = int(sys.argv[1]) if len(sys.argv) > 1 else 2015
    boards = load_boards()
    if not boards:
        sys.exit("枠連の板が無い。Macで `python3 ml/nk_odds_bulk.py --type 3` を回して push すること。")
    races = load_races()
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

    prep, yrs_, bad, hits = [], [], 0, 0
    for r in races:
        yy = r["year"]
        if yy < y0 or not lam.get(yy) or not r["wakuren"]:
            continue
        bd = boards.get(r["rid"])
        if not bd:
            continue
        rl = realized(r)
        if rl is None:
            continue
        a, b, c = rl
        num2k = {u: k for k, (u, _, _) in enumerate(r["horses"])}
        if a not in num2k or b not in num2k:
            continue
        p = probs(r["horses"])
        l2, l3 = lam[yy]
        q, combo = q_of_lbs("枠連", r, p, l2, l3, num2k, a, b, c)
        if q <= 0 or combo is None:
            continue
        v = payoff(r, PAYKEY["枠連"], combo)
        if not v or v <= 0:
            continue
        key = tuple(sorted(combo))
        if key not in bd:
            continue
        hits += 1
        if abs(bd[key] * 100 - v) > max(10.0, v * 0.01):
            bad += 1
        md = mkt_waku_dist(r, p, l2)                    # λ補正Harvilleの枠組分布
        if not md or key not in md:
            continue
        keys = [t for t in sorted(md) if t in bd]
        if key not in keys or len(keys) < 3:
            continue
        s = sum(1.0 / bd[t] for t in keys)
        lp = np.array([math.log((1.0 / bd[t]) / s) for t in keys])   # log q_pool（板）
        lh = np.array([math.log(md[t]) for t in keys])               # log q_λ（単勝由来）
        prep.append((lp, lh, keys.index(key)))
        yrs_.append(yy)

    yrs_ = np.array(yrs_)
    print(f"(125) 枠連プールに単勝プールを混ぜる（{y0}年以降・{len(prep):,}レース）")
    print(f"★道具の検算: 的中 {hits:,} 件のうち 板×100 と実配当がずれたもの **{bad}**"
          f"（{bad/max(hits,1):.2%}）  ※(113)は0.14%")
    if hits and bad / hits > 0.02:
        print("  ⚠ずれが多い。**先に道具を疑うこと**（事前登録4）。")
    print(f"  必要量 {NEED:.4f}\n")

    dtab = np.empty((len(WS), len(prep)))
    for j, w in enumerate(WS):
        for i, (lp, lh, k) in enumerate(prep):
            z = (1 - w) * lp + w * lh
            z -= z.max()
            dtab[j, i] = (z[k] - math.log(np.exp(z).sum())) - lp[k]

    print(f"{'w':>6}{'R数':>8}{'E[d]':>10}{'99%CI下':>10}{'上':>9}")
    for j, w in enumerate(WS):
        m, lo, hi = mci(dtab[j])
        print(f"{w:>6.2f}{len(prep):>8,}{m:>+10.4f}{lo:>+10.4f}{hi:>+9.4f}")
    print("  ★w=0 は板そのもの＝定義上0。0を有意に超えるwがあれば"
          "**枠連プールが単勝プールの情報を取り込みきれていない**")

    # ★ウォークフォワード（wを前年までで決める）
    out, picked = [], []
    for yy in sorted(set(yrs_.tolist())):
        tr, te = yrs_ < yy, yrs_ == yy
        if tr.sum() < 1000:
            continue
        j = int(np.argmax(dtab[:, tr].mean(axis=1)))
        m, lo, hi = mci(dtab[j, te])
        picked.append((yy, WS[j], int(te.sum()), m, lo))
        out.extend(dtab[j, te].tolist())
    if len(out) >= 100:
        m, lo, hi = mci(np.array(out))
        print(f"\n★ウォークフォワード: {len(out):,}件  E[d] = {m:+.4f}  "
              f"99%CI [{lo:+.4f}, {hi:+.4f}]  必要量の {m/NEED:.1%}")
        pos = sum(1 for _, _, _, mm, _ in picked if mm > 0)
        for yy, w, n, mm, lo2 in picked:
            print(f"   {yy}  w={w:<5g} {n:>5}本  {mm:+.4f}" + ("  ★" if lo2 > 0 else ""))
        print(f"   → {pos}/{len(picked)} 年で正")
        print("\n★運用の判定（事前登録6のとおり）")
        ok = m > 0.005 and pos >= 10
        print(f"   D>+0.005 かつ 10年以上で正 → {'**満たした。混ぜる価値がある**' if ok else '満たさない。**運用は変えない**'}")


if __name__ == "__main__":
    main()
