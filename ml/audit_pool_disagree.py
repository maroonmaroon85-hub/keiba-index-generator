"""(130) ★★プール間の不整合は「どのレースを買うか」の新しい変数になるか

★なぜこれを測るか
　(112)で**探索の方向が変わった**: qの作り込み（モデル混合の全成果 +0.0024）に対し、
　**レース選択だけで +0.0212**＝**10倍効いた**。以来「どのレースを買うか」が主戦場である。
　ところが**使えている選択変数は1本だけ**——(112)の「軸の複勝の期待払戻E」で、
　(117)の枠連スコア除外も**(119)で同じ信号だと判明**している。★**実質1変数しか無い**。
　(114)で上限も見えた（正味+0.0218＝必要量の8.5%）が、それは**その1変数での上限**であって、
　**別の情報源から作った変数**は試されていない。

★★新しい情報源＝**2つの市場の食い違い**
　三連複プール(type=7)は**Harvilleを一切使わない別市場の意見**である。
　単勝由来のλ補正Harvilleと三連複プールが**大きく食い違うレース**は、
　　(仮説A) 市場の意見が割れている＝**甘い**（買うべき）
　　(仮説B) 三連複プールが我々のqに無い情報を持っている＝**我々が間違っている**（避けるべき）
　どちらでもありうる。**向きも含めて事前に宣言する**（(113)(A)で「効かない」と「逆に効く」を
　区別できず半分外した教訓）。

★★事前登録（測る前に宣言する）
　1. **変数は1本だけ**: `KL(q_pool^三連複 ‖ q_λ^三連複)`。**発走前に計算できる**（結果を使わない）。
　　 頭数で大きさが変わるので、**(頭数, 年)の層内で順位→パーセンタイル**に直す。**後から増やさない**。
　2. **測る対象は枠連のd**（＝実際に買っている券種・現行の最良）。
　　 `d = log q_λ^枠 + log((払戻+5)/100) − log(0.775)`。λは**ウォークフォワード**で当てる。
　3. **判定は単調性**（十分位のSpearman ρ）。**最良のビンでは判定しない**（(106)(111)の教訓）。
　4. **プラセボ**: 選択変数を(頭数, 年)層内でシャッフル。**30回引いて平均する**（判定基準13）。
　　 **効果量は必ず「実測−プラセボ」**（判定基準2・(110)⑤）。
　5. **裾は先に宣言**: 高い側・低い側とも 2% / 5% / 10% / 25% / 50%。(112)と同じ刻み。
　6. **★対照は「現行の最良」**（判定基準17）: (112)の裾2%は **+0.0394**。
　　 **全体の+0.0182ではなく、これと比べる**。さらに**(112)の裾の中で追加の効きがあるか**も見る
　　 ——(119)で「(112)と(117)は同じ信号だった」となった前例があるため。
　7. **★運用が変わる条件を先に書く**: 裾2%で **d ≥ +0.0394 かつ 99%CI下端が +0.0182 を超え**、
　　 **単調性が出て**、**(112)の裾の中でも追加で効く**——**全部満たしたときだけ**運用を検討する。
　　 それ未満なら**記述にとどめ、運用は変えない**。
　8. **予想**: ★**仮説B側（負）または無効果**と予想する。ρ は 0 〜 −0.6 と見る。
　　 理由: 三連複プールは(113)(B)で**単勝に無い情報を実際に持っていた**（w≈0.2で+0.0033）。
　　 　　　食い違いが大きい所は**その情報が効いている所**＝我々のqが劣る所のはず。
　　 　　　⚠**仮説Aが正しくて正に出る可能性も残す**。そのときは「市場の不一致＝甘さ」となり、
　　 　　　**(112)とは別系統の変数**なので価値が大きい。
　　 ⚠**予想はあてにしない**。2026-08-11 だけで4回外している。

実行: python3 ml/audit_pool_disagree.py [開始年(既定2015)] [プラセボ反復(既定30)]
"""
import math
import sys
from collections import defaultdict
from itertools import combinations

import numpy as np

sys.path.insert(0, "ml")
from audit_capacity_d import mkt_waku_dist
from audit_crosspool import PAYBACK, load_races, payoff, probs, zq
from audit_crosspool2 import PAYKEY, realized
from audit_lbs import build_matrix, fit_lambda
from nk_parse import nk_raceid
import soft_axis as SA

RNG = np.random.default_rng(20260812)
NEED = -math.log(PAYBACK["枠連"])          # 0.2549
TAILS = (0.02, 0.05, 0.10, 0.25, 0.50)     # ★先に宣言した裾。後から増やさない
D_112_TAIL = 0.0394                        # (112) 裾2% の現行最良
D_ALL = 0.0182                             # (96) 枠連の全体D
E_112_THR = 86.0                           # (112) 軸の複勝Eの閾値[円]（裾2%）


def mci(x, alpha=0.01):
    x = np.asarray(x, float)
    if len(x) < 2:
        return float("nan"), float("nan"), float("nan")
    m = x.mean()
    se = x.std(ddof=1) / math.sqrt(len(x))
    z = zq(alpha)
    return m, m - z * se, m + z * se


def spearman(xs, ys):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for pos, i in enumerate(order):
            r[i] = pos
        return r
    a, b = rank(xs), rank(ys)
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    den = math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))
    return num / den if den else float("nan")


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


def disagreement(r, p, board):
    """★選択変数: KL(q_pool ‖ q_λ) を三連複の空間で。**発走前に計算できる**。"""
    nums = [n for n, _, _ in r["horses"]]
    pool, lam = [], []
    for c in combinations(range(len(nums)), 3):
        o = board.get(key_of([nums[i] for i in c]))
        if not o or o <= 0:
            continue
        pool.append(1.0 / o)
        lam.append(max(SA.trio_prob(p, list(c)), 1e-300))
    if len(pool) < 10:
        return None
    pool = np.asarray(pool, float)
    lam = np.asarray(lam, float)
    sp, sl = pool.sum(), lam.sum()
    if sp <= 0 or sl <= 0:
        return None
    pool /= sp
    lam /= sl                          # ★同じ支持集合の上で正規化する（判定基準14②）
    return float((pool * (np.log(pool) - np.log(lam))).sum())


def build(y0):
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

    rows = []
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
        p = probs(r["horses"])
        l2, _l3 = lam[yy]
        md = mkt_waku_dist(r, p, l2)                 # λ補正Harvilleの枠組分布（正規化済）
        if not md:
            continue
        from waku_umatan import waku_of
        key = tuple(sorted((waku_of(a, r["n"]), waku_of(b, r["n"]))))
        if key not in md:
            continue
        v = payoff(r, PAYKEY["枠連"], key)
        if not v or v <= 0:
            continue
        kl = disagreement(r, p, bd)
        if kl is None:
            continue
        # ★測る対象: 枠連のd（現行の最良のqで）
        d = math.log(md[key]) + math.log((v + 5) / 100.0) - math.log(PAYBACK["枠連"])
        odds = [o for _, o, _ in r["horses"]]
        _k, e_axis, _q = SA.axis_expect(odds)        # (112)の変数（対照用）
        rows.append((d, kl, r["n"], yy, e_axis if e_axis else float("nan")))
    return rows


def pctile_in_strata(kl, strata):
    """(頭数,年)の層内で順位→[0,1]のパーセンタイルに直す。"""
    out = np.zeros(len(kl))
    idx = defaultdict(list)
    for i, s in enumerate(strata):
        idx[s].append(i)
    for s, ii in idx.items():
        ii = np.array(ii)
        v = kl[ii]
        order = np.argsort(np.argsort(v))
        out[ii] = order / max(len(ii) - 1, 1)
    return out


def tail_means(d, pct, tails):
    """高い側・低い側の裾それぞれの平均d。"""
    hi, lo = [], []
    for t in tails:
        mh = pct >= (1.0 - t)
        ml = pct <= t
        hi.append(mci(d[mh]) if mh.sum() >= 20 else (float("nan"),) * 3)
        lo.append(mci(d[ml]) if ml.sum() >= 20 else (float("nan"),) * 3)
    return hi, lo


def main():
    y0 = int(sys.argv[1]) if len(sys.argv) > 1 else 2015
    reps = int(sys.argv[2]) if len(sys.argv) > 2 else 30

    rows = build(y0)
    if not rows:
        sys.exit("対象レースが無い")
    arr = np.array(rows, float)
    d, kl, nn, yy, ea = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 4]
    strata = list(zip(nn.astype(int).tolist(), yy.astype(int).tolist()))
    pct = pctile_in_strata(kl, strata)

    m_all, lo_all, hi_all = mci(d)
    print(f"対象 {len(rows)} レース（{int(yy.min())}〜{int(yy.max())}）")
    print(f"★検算: 枠連の全体D = {m_all:+.4f} 99%CI[{lo_all:+.4f},{hi_all:+.4f}]"
          f"  ※(96)は +0.0182。ずれるなら先に道具を疑う")
    print(f"　必要量 {NEED:.4f} ／ (112)裾2%の現行最良 {D_112_TAIL:+.4f}")

    # ── 十分位（判定の本体: 単調性） ──
    print("\n── 十分位（★判定は単調性・最良のビンでは判定しない） ──")
    dec = np.minimum((pct * 10).astype(int), 9)
    xs, ys = [], []
    for g in range(10):
        m = dec == g
        if m.sum() < 20:
            continue
        a, lo2, hi2 = mci(d[m])
        xs.append(g)
        ys.append(a)
        print(f"  第{g + 1:2d}十分位（不一致{'小' if g < 5 else '大'}）"
              f" n={int(m.sum()):5d}  d={a:+.4f}  99%CI[{lo2:+.4f},{hi2:+.4f}]")
    rho = spearman(xs, ys)
    print(f"  → Spearman ρ = {rho:+.3f}")

    # ── プラセボ（30回平均） ──
    print(f"\n── 裾（★効果量は「実測−プラセボ」・プラセボ{reps}回平均） ──")
    pl_hi = np.zeros(len(TAILS))
    pl_lo = np.zeros(len(TAILS))
    for _ in range(reps):
        sh = np.zeros(len(pct))
        idx = defaultdict(list)
        for i, s in enumerate(strata):
            idx[s].append(i)
        for s, ii in idx.items():
            ii = np.array(ii)
            sh[ii] = pct[ii][RNG.permutation(len(ii))]
        h, l = tail_means(d, sh, TAILS)
        pl_hi += np.array([x[0] for x in h])
        pl_lo += np.array([x[0] for x in l])
    pl_hi /= reps
    pl_lo /= reps

    hi_t, lo_t = tail_means(d, pct, TAILS)
    print("  【不一致が大きい側】")
    print("   裾      n      実測d      99%CI            プラセボ    実測−プラセボ")
    for j, t in enumerate(TAILS):
        m = pct >= (1.0 - t)
        a, l2, h2 = hi_t[j]
        print(f"   {int(t*100):3d}%  {int(m.sum()):5d}  {a:+.4f}  "
              f"[{l2:+.4f},{h2:+.4f}]  {pl_hi[j]:+.4f}   {a - pl_hi[j]:+.4f}")
    print("  【不一致が小さい側】")
    for j, t in enumerate(TAILS):
        m = pct <= t
        a, l2, h2 = lo_t[j]
        print(f"   {int(t*100):3d}%  {int(m.sum()):5d}  {a:+.4f}  "
              f"[{l2:+.4f},{h2:+.4f}]  {pl_lo[j]:+.4f}   {a - pl_lo[j]:+.4f}")

    # ── 事前登録6: (112)の裾の中でも効くか ──
    print("\n── ★事前登録6: (112)の裾2%（軸の複勝E≤86円）の中で追加の効きがあるか ──")
    in112 = ea <= E_112_THR
    a112, l112, h112 = mci(d[in112])
    print(f"  (112)の裾 n={int(in112.sum())}  d={a112:+.4f} 99%CI[{l112:+.4f},{h112:+.4f}]"
          f"  ※(112)報告値 {D_112_TAIL:+.4f}")
    if in112.sum() >= 60:
        sub = pct[in112]
        dd = d[in112]
        for t in (0.25, 0.50):
            mh = sub >= (1.0 - t)
            ml = sub < (1.0 - t)
            if mh.sum() >= 20 and ml.sum() >= 20:
                ah, _, _ = mci(dd[mh])
                al, _, _ = mci(dd[ml])
                # ★互いに素な「上位 vs 残り」の2標本で差を見る（判定基準13）。
                # 入れ子の部分集合どうしのCIを見比べても有意性は判定できない。
                se = math.sqrt(dd[mh].std(ddof=1) ** 2 / mh.sum()
                               + dd[ml].std(ddof=1) ** 2 / ml.sum())
                z = zq(0.01)
                print(f"   上位{int(t*100)}% {ah:+.4f}(n={int(mh.sum())}) vs "
                      f"残り {al:+.4f}(n={int(ml.sum())})  差 {ah - al:+.4f} "
                      f"99%CI[{ah - al - z * se:+.4f},{ah - al + z * se:+.4f}]")

    # ── 年別 ──
    print("\n── 年別（不一致が大きい側 25% の d） ──")
    pos = 0
    ys_ = sorted(set(int(x) for x in yy))
    for y in ys_:
        m = (yy == y) & (pct >= 0.75)
        if m.sum() < 20:
            continue
        a, l2, h2 = mci(d[m])
        pos += a > 0
        print(f"  {y}  n={int(m.sum()):4d}  d={a:+.4f}  99%CI[{l2:+.4f},{h2:+.4f}]")
    print(f"  → 正の年 {pos}/{len(ys_)}")

    # ── 事前登録7 の判定 ──
    print("\n── ★事前登録7: 運用が変わる条件に当てる ──")
    a2, l2, _ = hi_t[0]
    a2l, l2l, _ = lo_t[0]
    best = max((a2, "不一致大"), (a2l, "不一致小"))
    print(f"  裾2%の良いほう: {best[1]} d={best[0]:+.4f}"
          f"（条件は d≥{D_112_TAIL:+.4f} かつ CI下端>{D_ALL:+.4f}）")
    ok = (best[0] >= D_112_TAIL) and ((l2 if best[1] == "不一致大" else l2l) > D_ALL)
    print(f"  単調性 ρ={rho:+.3f}")
    print("  → " + ("★条件を満たす。(112)の裾の中での追加効きも確認すること"
                    if ok else "**条件を満たさない。記述にとどめ、運用は変えない**"))


if __name__ == "__main__":
    main()
