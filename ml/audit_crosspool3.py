"""(89c) 券種間の整合性 第3パス — 枠連の結果を潰しにいく＋上界の定式化。

第2パスで **枠連だけ D=+0.0105 (CI[+0.0070,+0.0141])**＝単勝プール由来のHarvilleの方が
枠連プールより鋭い、と出た。★これは(1-b)が探していた「市場が市場自身と矛盾する箇所」に当たる。
報告する前に、潰しにいく:

【潰しにいく点（測る前に宣言）】
 1. ★控除率の regime: JRAは2014年6月に払戻率を改定している。2013〜2014前半に
    現行の77.5%を当てると D が**上振れする**。年ごとに t* を出せば改定は目で見える。
    → **2015年以降に限定**して測り直す。これが本命の反証仮説。
 2. ★プラセボのバグ修正: 第2パスの枠連プラセボは馬番だけを並べ替えていたが、
    枠連のHarvilleは馬番から枠を引き直すのでシャッフルが効いていなかった（差が厳密に0.0000）。
    **確率ベクトル p の側を並べ替える**形に直す。
 3. ゾロ目(同枠)の寄与を分離（第2パスで D=−0.078 と逆向きだった）。
 4. 場・頭数・クラスで割っても符号が保つか。

【★上界の定式化 — これが本当の成果物】
 対数スコア差 D = E[log q_model] − E[log q_pool] は、そのまま**ケリー最適の対数成長率**になる:
   成長率 = log(払戻率) + D
 （プールの値付けに対して確率 q_model でケリーを打ったときの1レースあたり対数成長）
 → **どんな買い方をしても** この値を超えられない（ケリーが上限だから）。
 → 枠連なら log(0.775) = −0.2549。D=+0.0105 は控除率の **4.1%** しか埋めない。
 ★これは「買い方の探索」を一切せずに、単勝オッズを使う全戦略を同時に否定する。

実行: python3 ml/audit_crosspool3.py
"""
import math
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, "ml")
from audit_crosspool import (PAYBACK, h_pair_ordered, h_pair_unordered,  # noqa: E402
                             h_tri_ordered, h_tri_unordered, load_races, payoff,
                             probs, zq)
from audit_crosspool2 import PARTS, PAYKEY, realized  # noqa: E402
from waku_umatan import waku_of  # noqa: E402

RNG = np.random.default_rng(20260805)


def frame_members(r):
    wk = defaultdict(list)
    for k, (num, _, _) in enumerate(r["horses"]):
        wk[waku_of(num, r["n"])].append(k)
    return wk


def q_of(kind, r, p, num2k, a, b, c):
    """p はインデックス k（= r['horses'] の並び）に対応する確率。"""
    if kind == "枠連":
        wa, wb = sorted((waku_of(a, r["n"]), waku_of(b, r["n"])))
        wk = frame_members(r)
        q = 0.0
        if wa == wb:
            mem = wk[wa]
            for x in range(len(mem)):
                for y in range(x + 1, len(mem)):
                    q += h_pair_unordered(p, mem[x], mem[y])
        else:
            for x in wk[wa]:
                for y in wk[wb]:
                    q += h_pair_unordered(p, x, y)
        return q, (wa, wb)
    if kind == "馬連":
        return h_pair_unordered(p, num2k[a], num2k[b]), tuple(sorted((a, b)))
    if kind == "馬単":
        return h_pair_ordered(p, num2k[a], num2k[b]), (a, b)
    if c is None:
        return 0.0, None
    if kind == "三連複":
        return h_tri_unordered(p, (num2k[a], num2k[b], num2k[c])), tuple(sorted((a, b, c)))
    return h_tri_ordered(p, num2k[a], num2k[b], num2k[c]), (a, b, c)


def collect(races, shuffle_p=False):
    """★プラセボは確率ベクトル p を馬に対して並べ替える（枠にも効く正しい形）。"""
    out = {k: [] for k in PARTS}
    for r in races:
        rl = realized(r)
        if rl is None:
            continue
        a, b, c = rl
        hs = r["horses"]
        p = probs(hs)
        if shuffle_p:
            p = p[RNG.permutation(len(p))]
        num2k = {num: k for k, (num, _, _) in enumerate(hs)}
        if a not in num2k or b not in num2k or (c is not None and c not in num2k):
            continue
        for kind, key in PARTS.items():
            if not r[key]:
                continue
            q, combo = q_of(kind, r, p, num2k, a, b, c)
            if q <= 0 or combo is None:
                continue
            v = payoff(r, PAYKEY[kind], combo)
            if not v or v <= 0:
                continue
            d = math.log(q) + math.log((v + 5) / 100.0) - math.log(PAYBACK[kind])
            out[kind].append((d, r["year"], r["n"], int(combo[0] == combo[1])
                              if kind == "枠連" else 0, r["rid"]))
    return out


def mci(x, alpha):
    x = np.asarray(x, float)
    m = x.mean()
    se = x.std(ddof=1) / math.sqrt(len(x))
    z = zq(alpha)
    return m, m - z * se, m + z * se


def main():
    races = load_races()
    obs = collect(races)

    # ── 1. 年ごとの t*（控除率改定が見えるはず） ──
    print(f"{'='*104}")
    print("【1】年ごとの t*（＝Harvilleと引き分けになる払戻率）。控除率の改定は目で見えるはず")
    print("=" * 104)
    print(f"{'券種':<8}" + "".join(f"{y:>7}" for y in range(2013, 2027)))
    for kind in PARTS:
        yr = defaultdict(list)
        for d, y, *_ in obs[kind]:
            yr[y].append(d)
        line = f"{kind:<8}"
        for y in range(2013, 2027):
            if y in yr and len(yr[y]) > 100:
                line += f"{100*PAYBACK[kind]*math.exp(np.mean(yr[y])):>6.1f}%"
            else:
                line += f"{'—':>7}"
        print(line)
    print("　※ t* は『実際の払戻率』を仮に当てて逆算したもの。改定前後で段差が出れば regime の証拠。")

    # ── 2. 2015年以降に限定して測り直す（本命の反証） ──
    print(f"\n{'='*104}")
    print("【2】★2015年以降に限定（控除率 regime を揃える）— これが本命の反証仮説")
    print("=" * 104)
    alpha = 0.05 / 5
    z = zq(alpha)
    print(f"{'券種':<8}{'R数':>8}{'D':>11}{'99%CI':>22}{'t*':>9}{'実際':>8}"
          f"{'判定':>26}{'控除率のうち埋まる割合':>16}")
    print("-" * 104)
    res15 = {}
    for kind in PARTS:
        x = [d for d, y, *_ in obs[kind] if y >= 2015]
        if len(x) < 500:
            continue
        m, lo, hi = mci(x, alpha)
        pb = PAYBACK[kind]
        res15[kind] = (m, lo, hi, len(x))
        v = "単勝プールの方が鋭い" if lo > 0 else ("そのプールの方が鋭い" if hi < 0 else "確定しない")
        print(f"{kind:<8}{len(x):>8,}{m:>+11.4f} [{lo:>+8.4f},{hi:>+8.4f}]"
              f"{100*pb*math.exp(m):>8.1f}%{100*pb:>7.1f}%{v:>24}"
              f"{100*max(m,0)/abs(math.log(pb)):>14.1f}%")

    # ── 3. プラセボ（修正版） ──
    print(f"\n{'='*104}")
    print("【3】プラセボ（修正版）— 確率ベクトル p をレース内で並べ替え。枠にも効く形にした")
    print("=" * 104)
    pl = collect(races, shuffle_p=True)
    print(f"{'券種':<8}{'本物 D':>12}{'プラセボ D':>14}{'差':>12}")
    for kind in PARTS:
        if not pl[kind] or kind not in res15:
            continue
        pm = float(np.mean([d for d, y, *_ in pl[kind] if y >= 2015]))
        print(f"{kind:<8}{res15[kind][0]:>+12.4f}{pm:>+14.4f}{res15[kind][0]-pm:>+12.4f}")

    # ── 4. 枠連の内訳（2015年以降） ──
    print(f"\n{'='*104}")
    print("【4】枠連の内訳（2015年以降・丸め補正済み）")
    print("=" * 104)
    arr = np.array([(d, y, n, z0) for d, y, n, z0, _ in obs["枠連"] if y >= 2015])
    z6 = zq(0.05 / 6)
    print(f"{'区分':<16}{'R数':>9}{'D':>11}{'CI':>24}{'t*':>9}")
    groups = [("全体(2015+)", np.ones(len(arr), bool)),
              ("ゾロ目を除く", arr[:, 3] == 0), ("ゾロ目のみ", arr[:, 3] == 1)]
    for lo, hi in [(9, 12), (13, 15), (16, 18)]:
        groups.append((f"頭数{lo}-{hi}", (arr[:, 2] >= lo) & (arr[:, 2] <= hi)))
    for nm, m in groups:
        x = arr[m, 0]
        if len(x) < 200:
            continue
        mu, l, h = mci(x, 0.05 / 6)
        print(f"{nm:<16}{len(x):>9,}{mu:>+11.4f} [{l:>+9.4f},{h:>+9.4f}]"
              f"{100*PAYBACK['枠連']*math.exp(mu):>8.1f}%")

    # ── 5. ★上界 ──
    print(f"\n{'='*104}")
    print("【5】★上界 — 単勝オッズを使う『あらゆる買い方』の1レースあたり対数成長率")
    print("　　成長率 = log(払戻率) + D 　（ケリー最適。買い方の探索では超えられない）")
    print("=" * 104)
    print(f"{'券種':<8}{'log(払戻率)':>13}{'D':>10}{'成長率':>11}{'100%まで足りない分':>20}"
          f"{'埋まった割合':>14}")
    print("-" * 104)
    for kind in PARTS:
        if kind not in res15:
            continue
        m = res15[kind][0]
        lp = math.log(PAYBACK[kind])
        g = lp + m
        print(f"{kind:<8}{lp:>+13.4f}{m:>+10.4f}{g:>+11.4f}"
              f"{100*(math.exp(g)-1):>18.1f}%{100*max(m,0)/abs(lp):>13.1f}%")
    print("\n★読み方: 成長率が負なら、その券種は単勝オッズ由来のどんな戦略でも長期的に破産する。")
    print("　『埋まった割合』= 市場間の不整合で控除率の何%を回収できるか。")


if __name__ == "__main__":
    main()
