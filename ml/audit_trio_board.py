"""(113) 三連複のオッズ板を使った2つの検証。**(113事前登録)のとおりに実行する**。

★(A) 運用の問い（ユーザー発案）
　「買う1点（人気上位3頭）の三連複オッズが安いレースは見送るべきか」。
　**閾値は先に宣言してある**: なし / ≤3 / ≤5 / ≤10 / ≤20 倍。
　**判定は単調性（Spearman ρ）で行い、最良のビンでは判定しない**（(106)(111)の教訓）。
　**プラセボ**（オッズをレース間でシャッフル）を必ず対で出し、効果量は「実測−プラセボ」で書く。

★(B) 板でなければできないこと ＝ **全組にわたる正規化**
　`q ∝ q_pool^(1−w) · q_λ^w` のような混合はΣを1にする必要があり、**外れた組のオッズが要る**。
　実配当しか無いとこれができなかった。→ **三連複プールという別の市場の意見**を初めて混合できる。
　(102)(107)は単勝プール由来の専門家しか混ぜられていない。**新しい情報源はここ数十項目で初**。

　⚠**訂正済みの誤解**: 「板が無いと真の市場qが分からない」は誤り。Dは実現した組の1点でしか
　　評価せず、`d = log q + log(払戻/100) − log(払戻率)` に q_pool は実配当から既に入っていた。
　　**(112)の −0.0213 は板があっても変わらない**。板の価値は正規化にある。

★判定の基準
　三連複の必要量は **0.2877**（= |log 0.750|）。E[d|S] の99%CI下端がこれを超えなければ、
　**その群でどんな買い方をしても儲からない**（ケリーが上限）。

実行: python3 ml/audit_trio_board.py [開始年(既定2013)]
　　　板は data/nk_odds/type7_*.jsonl.gz（`nk_odds_bulk.py` が集める）。
　　　★**集まっているぶんだけで走る**。634本の2%裾が先に集まる設計なので、途中でも意味がある。
"""
import math
import sys

import numpy as np

sys.path.insert(0, "ml")
from audit_crosspool import PAYBACK, load_races, payoff, zq
from audit_crosspool2 import realized
from nk_parse import nk_raceid
import soft_axis as SA

NEED = -math.log(PAYBACK["三連複"])          # 0.2877
CUTS = [None, 3.0, 5.0, 10.0, 20.0]          # ★先に宣言した閾値。後から増やさない
RNG = np.random.default_rng(20260808)


def mci(x, alpha=0.01):
    x = np.asarray(x, float)
    if len(x) < 2:
        return float("nan"), float("nan"), float("nan")
    m = x.mean()
    se = x.std(ddof=1) / math.sqrt(len(x))
    z = zq(alpha)
    return m, m - z * se, m + z * se


def spearman(xs, ys):
    """順位相関。ビン数が5しかないので、点推定の向きを見るためだけに使う。"""
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
    """race_id(8桁) → {組キー: オッズ}。集まっているぶんだけ返す。"""
    from nk_odds_bulk import iter_records
    out = {}
    for rec in iter_records(7):
        rid8 = nk_raceid(rec["race_id"])
        if rid8:
            out[rid8] = rec["odds"]
    return out


def build(races, boards):
    """板のあるレースだけを、検証に必要な形にそろえる。"""
    rows = []
    for r in races:
        board = boards.get(r["rid"])
        if not board:
            continue
        rl = realized(r)
        if rl is None or rl[2] is None:
            continue
        hs = r["horses"]
        nums = [n for n, _, _ in hs]
        odds = [o for _, o, _ in hs]
        p = SA.win_probs(odds)
        order = sorted(range(len(hs)), key=lambda i: odds[i])
        pick_idx = order[:3]
        pick = [nums[i] for i in pick_idx]
        o_pick = board.get(key_of(pick))
        if not o_pick or o_pick <= 0:
            continue                      # 板に無い＝取消などが絡む。落とす
        # 実現した組
        real = key_of(rl)
        o_real = board.get(real)
        if not o_real or o_real <= 0:
            continue
        # 板そのものが含む確率（全組で正規化する。**これが板でしかできないこと**）
        inv = {k: 1.0 / v for k, v in board.items() if v > 0}
        s = sum(inv.values())
        if s <= 0:
            continue
        pay = payoff(r, "三連複", list(rl)) or 0.0
        if pay <= 0:                  # 実配当が無いと d が測れない（(99)と同じ理由）
            continue
        rows.append(dict(
            rid=r["rid"], year=r["year"], n=r["n"], p=p, nums=nums,
            pick=pick, pick_idx=pick_idx, o_pick=o_pick,
            hit=(key_of(pick) == real), real=rl, real_idx=[nums.index(x) for x in rl],
            o_real=o_real, inv_sum=s, pay=pay, board=board))
    return rows


# ───────────────────────── (A) 安い三連複を外すか ─────────────────────────
def part_a(rows, shuffle=False):
    o = np.array([x["o_pick"] for x in rows], float)
    if shuffle:
        o = o[RNG.permutation(len(o))]      # ★プラセボ: 選択変数だけ壊す
    # d は実現した組で測る（(112)と同じ定義。q は市場のλ補正Harville）
    d = np.array([
        math.log(max(SA.trio_prob(x["p"], x["real_idx"]), 1e-300))
        + math.log(x["pay"] / 100.0) - math.log(PAYBACK["三連複"])
        for x in rows], float)
    ret = np.array([(x["o_pick"] * 100.0 if x["hit"] else 0.0) for x in rows], float)
    out = []
    for c in CUTS:
        m = np.ones(len(rows), bool) if c is None else (o > c)
        if m.sum() < 20:
            out.append((c, int(m.sum()), float("nan"), float("nan"),
                        float("nan"), float("nan")))
            continue
        dm, lo, hi = mci(d[m])
        roi = ret[m].mean()
        out.append((c, int(m.sum()), dm, lo, hi, roi))
    return out


# ───────────────────────── (B) 板と混ぜる ─────────────────────────
def combos(n):
    from itertools import combinations
    return list(combinations(range(n), 3))


def prep_b(x):
    """1レース分の (log q_pool, log q_λ, 実現組の位置) を**1回だけ**作る（wごとに作り直さない）。"""
    nums, board, p = x["nums"], x["board"], x["p"]
    ks, lp, lh = [], [], []
    for c in combos(len(nums)):
        k = key_of([nums[i] for i in c])
        o = board.get(k)
        if not o or o <= 0:
            continue
        ks.append(k)
        lp.append(-math.log(o) - math.log(x["inv_sum"]))            # log q_pool
        lh.append(math.log(max(SA.trio_prob(p, list(c)), 1e-300)))  # log q_λ
    real = key_of(x["real"])
    if real not in ks:
        return None
    return np.array(lp), np.array(lh), ks.index(real)


def part_b(rows, ws=(0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0)):
    prepped = [v for v in (prep_b(x) for x in rows) if v is not None]
    out = []
    for w in ws:
        d = []
        for lp, lh, i in prepped:
            z = (1 - w) * lp + w * lh
            z -= z.max()
            d.append(float((z[i] - math.log(np.exp(z).sum())) - lp[i]))
        if len(d) < 20:
            out.append((w, len(d), float("nan"), float("nan"), float("nan")))
            continue
        m, lo, hi = mci(np.array(d))
        out.append((w, len(d), m, lo, hi))
    return out


def part_b_wf(rows, ws=(0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5)):
    """★ウォークフォワード。**wを同じデータで選んだら有利になりすぎる**ので、
    　 その年より**前**の年だけでwを決め、その年に当てる。実運用でできる手続きと同じ。"""
    prep, yrs = [], []
    for x in rows:
        v = prep_b(x)
        if v is not None:
            prep.append(v)
            yrs.append(x["year"])
    yrs = np.array(yrs)
    dtab = np.empty((len(ws), len(prep)))       # [w][レース] の d を先に作る
    for j, w in enumerate(ws):
        for i, (lp, lh, k) in enumerate(prep):
            z = (1 - w) * lp + w * lh
            z -= z.max()
            dtab[j, i] = (z[k] - math.log(np.exp(z).sum())) - lp[k]
    out, picked = [], []
    for yy in sorted(set(yrs.tolist())):
        tr, te = yrs < yy, yrs == yy
        if tr.sum() < 1000:
            continue
        j = int(np.argmax(dtab[:, tr].mean(axis=1)))
        m, lo, hi = mci(dtab[j, te])
        picked.append((yy, ws[j], int(te.sum()), m, lo, hi))
        out.extend(dtab[j, te].tolist())
    return np.array(out), picked


def main():
    y0 = int(sys.argv[1]) if len(sys.argv) > 1 else 2013
    boards = load_boards()
    if not boards:
        sys.exit("板がまだ無い。Macで `nk_odds_bulk.py` を回して data/nk_odds を push すること。")
    races = [r for r in load_races() if r["year"] >= y0]
    rows = build(races, boards)
    print(f"板のあるレース {len(boards)} / 突き合わせできた {len(rows)}")
    if len(rows) < 50:
        sys.exit("まだ少なすぎる。収集が進んでから回すこと。")

    # 板と実配当の整合（★道具の検算。ここが合わなければ以降は全部信用できない）
    hits = [x for x in rows if x["hit"]]
    bad = [x for x in hits if abs(x["o_pick"] * 100 - x["pay"]) > max(10, x["pay"] * 0.01)]
    print(f"  的中 {len(hits)} 件のうち 板×100 と実配当がずれたもの: {len(bad)}")
    if bad:
        for x in bad[:5]:
            print(f"    {x['rid']} 板{x['o_pick']}×100={x['o_pick']*100:.0f} vs 実配当{x['pay']:.0f}")

    eff = np.mean([1.0 / x["inv_sum"] for x in rows])
    print(f"  板から出した三連複の実効払戻率: {eff:.4f}（公示 0.750）")

    print(f"\n── (A) 「買う1点のオッズが安いレースを外す」 ── 必要量 {NEED:+.4f}")
    obs, pla = part_a(rows), part_a(rows, shuffle=True)
    print(f"{'閾値':>8}{'R数':>7}{'E[d|S]':>10}{'99%CI下':>10}{'上':>9}"
          f"{'ROI':>8}   プラセボE[d]")
    for (c, n, m, lo, hi, roi), (_, _, pm, _, _, _) in zip(obs, pla):
        lab = "なし" if c is None else f">{c:g}倍"
        print(f"{lab:>8}{n:>7}{m:>10.4f}{lo:>10.4f}{hi:>9.4f}"
              f"{roi:>8.1f}{pm:>13.4f}")
    xs = [i for i, (c, n, m, *_) in enumerate(obs) if not math.isnan(m)]
    if len(xs) >= 3:
        rho = spearman(xs, [obs[i][2] for i in xs])
        rho_p = spearman(xs, [pla[i][2] for i in xs])
        print(f"  単調性 Spearman ρ = {rho:+.3f}（プラセボ {rho_p:+.3f}）"
              "  ★ρが0近傍なら「安いのを外す」は効いていない")

    print(f"\n── (B) 板と混ぜる q ∝ q_pool^(1−w)·q_λ^w ── 必要量 {NEED:+.4f}")
    print(f"{'w':>6}{'R数':>7}{'E[d]':>10}{'99%CI下':>10}{'上':>9}")
    for w, n, m, lo, hi in part_b(rows):
        print(f"{w:>6.2f}{n:>7}{m:>10.4f}{lo:>10.4f}{hi:>9.4f}")
    print("  ★w=0 は板そのもの＝定義上 0.0000。0を有意に超えるwがあれば"
          "**三連複プールが単勝プールの情報を取り込みきれていない**ということ。")

    d, picked = part_b_wf(rows)
    if len(d) >= 100:
        m, lo, hi = mci(d)
        print(f"\n★ウォークフォワード（wを前年までで決める）: {len(d):,}件  "
              f"E[d] = {m:+.4f}  99%CI [{lo:+.4f}, {hi:+.4f}]  "
              f"必要量の {m / NEED:.1%}")
        # ★年分割（判定基準）。**特定の年が引っ張っているだけ**でないかを見る
        print(f"\n  {'年':>6}{'選ばれたw':>10}{'R数':>7}{'E[d]':>10}{'99%CI下':>10}{'上':>9}"
              f"   符号")
        pos = 0
        for y, w, n, m, lo, hi in picked:
            pos += m > 0
            print(f"  {y:>6}{w:>10.2f}{n:>7}{m:>+10.4f}{lo:>+10.4f}{hi:>+9.4f}"
                  f"   {'＋' if m > 0 else '−'}{'★' if lo > 0 else ''}")
        print(f"  → {pos}/{len(picked)} 年で正。★は99%CIが0を除外した年"
              "（1年3,000件では検出力が足りないので、★が少なくても弱点ではない）")


if __name__ == "__main__":
    main()
