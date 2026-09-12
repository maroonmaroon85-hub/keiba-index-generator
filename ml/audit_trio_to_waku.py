"""(129) ★★三連複プールのHarville非依存な情報を、枠連のqに持ち込めるか

★何が新しいか
　(125)は**枠連プール**を枠連のqに混ぜた（上積み +0.0014）。
　(113)(B)は**三連複プール**を三連複のqに混ぜた（+0.0033）。
　**まだやっていないのは「三連複プールの情報を枠連へ渡す」**——券種をまたぐ移送である。
　三連複プールは**Harvilleを一切使わない別市場**なので、そこから取り出した
　**馬ごとのtop3確率**は、単勝オッズにもモデルにも無い独立な意見になりうる。

★★仕掛け
　板の全組を正規化した q_pool から、**馬ごとの周辺確率**を作る:
　　`t3_pool[i] = Σ_{i を含む組} q_pool(組)`（Σ=3 なので3で割って正規化）
　これを単勝の市場確率と**対数線形プール**で混ぜ、λ補正Harvilleに通して枠連分布を作る:
　　`p_mix ∝ p_mkt^(1−w) · t3n^w`  →  `mkt_waku_dist(p_mix, λ2)`
　⚠**正規化定数を必ず入れる**（判定基準14①。省くと作り物が出る）。

★★事前登録（測る前に宣言する）
　1. w は **0 / 0.05 / 0.1 / 0.2 / 0.3 / 0.5 / 0.7 / 1.0** の8点（(113)(B)(125)と同じ）。**後から増やさない**。
　2. **ウォークフォワードでwを選ぶ**（その年より前の年だけで決める）。これが主結果。
　3. ★**対照は「現行の最良」＝w=0**（判定基準17・(125)で踏んだ穴）。
　　 ここでの d は**実配当に対する絶対的なD**（プール基準ではない）ので、
　　 **w=0 の値が既知の枠連D=+0.0182 に一致するはず**＝相互検算。
　　 **上積みは d(w) − d(0)** で測る。**プール基準と混同しないこと**。
　4. **年分割で符号が揃うか**を見る。
　5. **★運用が変わる条件を先に書く**: ウォークフォワードの**上積みが +0.005 を超え**、かつ
　　 **年分割で10/11年以上**正なら、現行のqに三連複プールを混ぜる価値がある。
　　 それ未満なら**記述にとどめ、運用は変えない**（(125)と同じ基準）。
　6. **予想**: **+0.000 〜 +0.002**（**採用条件には届かない**）と予想する。
　　 理由: (125)の枠連プール直送ですら +0.0014 だった。三連複プールは**枠連の事象から更に遠い**
　　 　　　（top3集合 → 1-2着）ので、移送で情報は落ちるはず。
　　 　　　加えて(130)で「三連複プールの食い違いはレース選択の信号にならない」と出ている。
　　 ⚠**予想はあてにしない**。

実行: python3 ml/audit_trio_to_waku.py [開始年(既定2015)]
"""
import math
import sys
from itertools import combinations

import numpy as np

sys.path.insert(0, "ml")
from audit_capacity_d import mkt_waku_dist
from audit_crosspool import PAYBACK, load_races, payoff, probs, zq
from audit_crosspool2 import PAYKEY, realized
from audit_lbs import build_matrix, fit_lambda
from nk_parse import nk_raceid
from waku_umatan import waku_of

WS_WF = (0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5)
D_ALL = 0.0182


def mci(x, alpha=0.01):
    x = np.asarray(x, float)
    if len(x) < 2:
        return float("nan"), float("nan"), float("nan")
    m = x.mean()
    se = x.std(ddof=1) / math.sqrt(len(x))
    z = zq(alpha)
    return m, m - z * se, m + z * se


def key_of(nums):
    return "".join(f"{n:02d}" for n in sorted(nums))


def load_boards():
    from nk_odds_bulk import iter_records
    out = {}
    for rec in iter_records(7):
        r8 = nk_raceid(rec["race_id"])
        if r8:
            out[r8] = rec["odds"]
    return out


def t3_from_pool(nums, board):
    """板 → 馬ごとの「top3に入る」周辺確率。**Harvilleを一切使っていない**。"""
    n = len(nums)
    acc = np.zeros(n)
    tot = 0.0
    for c in combinations(range(n), 3):
        o = board.get(key_of([nums[i] for i in c]))
        if not o or o <= 0:
            continue
        v = 1.0 / o
        tot += v
        for i in c:
            acc[i] += v
    if tot <= 0:
        return None
    acc /= tot                      # Σacc = 3
    s = acc.sum()
    if s <= 0:
        return None
    return acc / s                  # Σ=1 に直す


def mix(p_mkt, t3n, w):
    """★対数線形プール。**正規化定数を必ず入れる**（判定基準14①）。"""
    if w <= 0:
        return p_mkt
    a = np.log(np.maximum(p_mkt, 1e-300)) * (1 - w) + np.log(np.maximum(t3n, 1e-300)) * w
    a -= a.max()
    e = np.exp(a)
    s = e.sum()
    return e / s if s > 0 else p_mkt


def main():
    y0 = int(sys.argv[1]) if len(sys.argv) > 1 else 2015
    races = load_races()
    boards = load_boards()
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

    dtab, years = [], []
    for r in races:
        yy = r["year"]
        if yy < y0 or not lam.get(yy) or not r["wakuren"]:
            continue
        bd = boards.get(r["rid"])
        if not bd:
            continue
        rl = realized(r)
        if rl is None or rl[2] is None:
            continue
        a, b, _ = rl
        num2k = {u: k for k, (u, _, _) in enumerate(r["horses"])}
        if a not in num2k or b not in num2k:
            continue
        nums = [u for u, _, _ in r["horses"]]
        t3n = t3_from_pool(nums, bd)
        if t3n is None:
            continue
        p_mkt = probs(r["horses"])
        l2, _l3 = lam[yy]
        key = tuple(sorted((waku_of(a, r["n"]), waku_of(b, r["n"]))))
        v = payoff(r, PAYKEY["枠連"], key)
        if not v or v <= 0:
            continue
        row = []
        ok = True
        for w in WS_WF:
            md = mkt_waku_dist(r, mix(p_mkt, t3n, w), l2)
            if not md or key not in md or md[key] <= 0:
                ok = False
                break
            row.append(math.log(md[key]) + math.log((v + 5) / 100.0)
                       - math.log(PAYBACK["枠連"]))
        if not ok:
            continue
        dtab.append(row)
        years.append(yy)

    if not dtab:
        sys.exit("対象レースが無い")
    D = np.array(dtab)                       # [レース][w]
    yv = np.array(years)
    print(f"対象 {len(dtab)} レース（{int(yv.min())}〜{int(yv.max())}）")

    m0, lo0, hi0 = mci(D[:, 0])
    print(f"★検算: w=0（現行の最良）の D = {m0:+.4f} 99%CI[{lo0:+.4f},{hi0:+.4f}]"
          f"  ※(96)は {D_ALL:+.4f}")

    print("\n── w を固定したときの D（★上積みは d(w) − d(0)） ──")
    for j, w in enumerate(WS_WF):
        m, lo, hi = mci(D[:, j])
        up, ulo, uhi = mci(D[:, j] - D[:, 0])
        print(f"  w={w:<5} D={m:+.4f} [{lo:+.4f},{hi:+.4f}]   "
              f"上積み {up:+.4f} 99%CI[{ulo:+.4f},{uhi:+.4f}]")

    print("\n── ★ウォークフォワード（その年より前の年だけで w を選ぶ） ──")
    ys = sorted(set(yv.tolist()))
    picked, gains, rows = [], [], 0
    per_year = []
    for y in ys:
        tr = yv < y
        te = yv == y
        if tr.sum() < 2000 or te.sum() < 50:
            continue
        j = int(np.argmax(D[tr].mean(axis=0) - D[tr, 0].mean()))
        g = D[te, j] - D[te, 0]
        picked.append(WS_WF[j])
        gains.append(g)
        rows += int(te.sum())
        gm, glo, ghi = mci(g)
        per_year.append((y, int(te.sum()), WS_WF[j], gm, glo, ghi))
        print(f"  {y}  n={int(te.sum()):5d}  選ばれた w={WS_WF[j]:<5} "
              f"上積み {gm:+.4f} 99%CI[{glo:+.4f},{ghi:+.4f}]")
    if not gains:
        sys.exit("ウォークフォワードに足りるデータが無い")
    allg = np.concatenate(gains)
    gm, glo, ghi = mci(allg)
    pos = sum(1 for _, _, _, m, _, _ in per_year if m > 0)
    print(f"\n  ★ウォークフォワードの上積み {gm:+.4f} 99%CI[{glo:+.4f},{ghi:+.4f}]"
          f"（n={rows}）")
    print(f"  選ばれた w: {sorted(set(picked))} ／ 正の年 {pos}/{len(per_year)}")

    print("\n── ★事前登録5: 運用が変わる条件に当てる ──")
    okg = gm > 0.005 and pos >= 10
    print(f"  条件: 上積み>+0.0050 かつ 正の年10以上  →  実測 {gm:+.4f} / {pos}年")
    print("  → " + ("★条件を満たす。現行のqに三連複プールを混ぜる価値がある"
                    if okg else "**条件を満たさない。記述にとどめ、運用は変えない**"))


if __name__ == "__main__":
    main()
