"""(89b) 券種間の整合性 第2パス — 対数スコア差に誤差をつける。

第1パス(`audit_crosspool.py`)で:
  ・第1部: 複勝/枠連/馬連 は払戻率の線をBonferroni補正CIで超えた（+3.2〜+5.5pt・14年安定）
  ・第2部: **枠連だけ t*=77.8% > 実際77.5%**＝Harville(単勝プール由来)の方が鋭い
    ——ただし差は0.3ptしかない。誤差をつけないと(83)と同じ轍を踏む。

★本パスで確かめること（測る前に宣言）
  1. 対数スコア差 D = log q_Harville − log q_pool の**レース単位CI**（券種5つでBonferroni）
     D>0 ⟺ t* > 払戻率 ⟺ 単勝プールの方が鋭い
  2. ★丸めの補正: 払戻は10円単位で切り捨てられる。素の値を使うと log(払戻) が
     下振れし、プール側が不当に有利になる。**払戻+5円**（丸め区間の中点）で再計算する。
     枠連の差は0.3ptしかないので、この補正だけで符号が変わりうる。
  3. 年ごとの符号一貫性（14年中何年でD>0か）
  4. 頭数別（枠連は9頭以上でしか発売されない。頭数で構造が変わるか）
  5. ★ゾロ目(同枠)を除く感度 — 枠連特有の構造なので分けて見る
  6. ★プラセボ: 馬番をレース内でシャッフルした偽Harvilleでも同じ差が出ないか
     （(79)の作法。中身が無くても差が出るなら意味が無い）

実行: python3 ml/audit_crosspool2.py
"""
import math
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, "ml")
from audit_crosspool import (PAYBACK, h_pair_ordered, h_pair_unordered,  # noqa: E402
                             h_tri_ordered, h_tri_unordered, load_races, payoff,
                             probs, zq)
from waku_umatan import waku_of  # noqa: E402

NPART = 5
ALPHA = 0.05 / NPART
RNG = np.random.default_rng(20260805)


def realized(r):
    fin = {num: f for num, _, f in r["horses"]}
    first = [n for n in fin if fin[n] == 1]
    second = [n for n in fin if fin[n] == 2]
    third = [n for n in fin if fin[n] == 3]
    if len(first) != 1 or len(second) != 1:
        return None
    return first[0], second[0], (third[0] if len(third) == 1 else None)


def harville_of(kind, r, p, num2k, a, b, c):
    if kind == "枠連":
        wa, wb = sorted((waku_of(a, r["n"]), waku_of(b, r["n"])))
        wk = defaultdict(list)
        for k, (num, _, _) in enumerate(r["horses"]):
            wk[waku_of(num, r["n"])].append(k)
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
    if kind == "三連複":
        if c is None:
            return 0.0, None
        return h_tri_unordered(p, (num2k[a], num2k[b], num2k[c])), tuple(sorted((a, b, c)))
    if c is None:
        return 0.0, None
    return h_tri_ordered(p, num2k[a], num2k[b], num2k[c]), (a, b, c)


PARTS = {"枠連": "wakuren", "馬連": "umaren", "馬単": "umatan",
         "三連複": "puku", "三連単": "tan3"}
PAYKEY = {"枠連": "枠連(人気順)", "馬連": "馬連", "馬単": "馬単",
          "三連複": "三連複", "三連単": "三連単"}


def collect(races, shuffle=False):
    """券種 → レースごとの (D_raw, D_round, year, n, zorome)。D = log q_H − log q_pool。"""
    out = {k: [] for k in PARTS}
    for r in races:
        rl = realized(r)
        if rl is None:
            continue
        a, b, c = rl
        hs = r["horses"]
        p = probs(hs)
        if shuffle:
            # 馬番だけをレース内で入れ替える＝確率の中身は同じで対応関係だけ壊す
            perm = RNG.permutation(len(hs))
            nums = [hs[k][0] for k in perm]
        else:
            nums = [h[0] for h in hs]
        num2k = {num: k for k, num in enumerate(nums)}
        if a not in num2k or b not in num2k or (c is not None and c not in num2k):
            continue
        for kind, key in PARTS.items():
            if not r[key]:
                continue
            q, combo = harville_of(kind, r, p, num2k, a, b, c)
            if q <= 0 or combo is None:
                continue
            v = payoff(r, PAYKEY[kind], combo)
            if not v or v <= 0:
                continue
            pb = PAYBACK[kind]
            d_raw = math.log(q) + math.log(v / 100.0) - math.log(pb)
            d_rnd = math.log(q) + math.log((v + 5) / 100.0) - math.log(pb)
            zoro = (kind == "枠連" and combo[0] == combo[1])
            out[kind].append((d_raw, d_rnd, r["year"], r["n"], zoro))
    return out


def summarize(vals, alpha, label):
    d = np.array([v[0] for v in vals])
    dr = np.array([v[1] for v in vals])
    z = zq(alpha)
    res = []
    for arr, nm in ((d, "素"), (dr, "丸め補正")):
        m = arr.mean()
        se = arr.std(ddof=1) / math.sqrt(len(arr))
        res.append((m, m - z * se, m + z * se, nm))
    return res


def main():
    races = load_races()
    print(f"配当A: {len(races):,}レース")
    obs = collect(races)
    z = zq(ALPHA)

    print(f"\n{'='*112}")
    print("【1〜3】対数スコア差 D = log q(Harville) − log q(そのプール)。D>0 なら単勝プールの方が鋭い")
    print(f"　券種5つでBonferroni α=0.05/5={ALPHA:.3f}（z={z:.2f}）/ 丸め補正=払戻に+5円（10円切り捨ての中点）")
    print("=" * 112)
    print(f"{'券種':<8}{'R数':>8}{'D(素)':>10}{'99%CI':>20}{'t*(素)':>9}"
          f"{'D(丸め補正)':>13}{'99%CI':>20}{'t*(補正)':>10}{'実際':>7}{'D>0の年':>9}")
    print("-" * 112)
    keep = {}
    for kind in PARTS:
        vals = obs[kind]
        if len(vals) < 500:
            continue
        (m0, l0, h0, _), (m1, l1, h1, _) = summarize(vals, ALPHA, kind)
        pb = PAYBACK[kind]
        yr = defaultdict(list)
        for v in vals:
            yr[v[2]].append(v[1])
        nyr = sum(1 for y in yr if np.mean(yr[y]) > 0)
        keep[kind] = (m1, l1, h1)
        print(f"{kind:<8}{len(vals):>8,}{m0:>+10.4f} [{l0:>+7.4f},{h0:>+7.4f}]"
              f"{100*pb*math.exp(m0):>8.1f}%{m1:>+13.4f} [{l1:>+7.4f},{h1:>+7.4f}]"
              f"{100*pb*math.exp(m1):>9.1f}%{100*pb:>6.1f}%{nyr:>6}/{len(yr)}")
    sig = [k for k, v in keep.items() if v[1] > 0]
    print(f"\n  ★丸め補正後にCI下端>0（＝単勝プールの方が鋭いと確定）: **{len(sig)}件** {sig}")
    neg = [k for k, v in keep.items() if v[2] < 0]
    print(f"  ★CI上端<0（＝そのプールの方が鋭いと確定）: {len(neg)}件 {neg}")

    # 4-5 枠連の内訳
    print(f"\n{'='*112}")
    print("【4-5】枠連の内訳（頭数別 / ゾロ目の有無）— 丸め補正後のD")
    print("=" * 112)
    vals = obs["枠連"]
    arr = np.array([(v[1], v[3], v[4]) for v in vals])
    z4 = zq(0.05 / 6)
    print(f"{'区分':<16}{'R数':>9}{'D':>11}{'CI':>22}{'t*':>9}")
    groups = [("全体", np.ones(len(arr), bool))]
    for lo, hi in [(9, 12), (13, 15), (16, 18)]:
        groups.append((f"頭数{lo}-{hi}", (arr[:, 1] >= lo) & (arr[:, 1] <= hi)))
    groups.append(("ゾロ目のみ", arr[:, 2] == 1))
    groups.append(("ゾロ目を除く", arr[:, 2] == 0))
    for nm, m in groups:
        x = arr[m, 0]
        if len(x) < 200:
            continue
        mu = x.mean()
        se = x.std(ddof=1) / math.sqrt(len(x))
        print(f"{nm:<16}{len(x):>9,}{mu:>+11.4f} [{mu-z4*se:>+8.4f},{mu+z4*se:>+8.4f}]"
              f"{100*PAYBACK['枠連']*math.exp(mu):>8.1f}%")

    # 6 プラセボ
    print(f"\n{'='*112}")
    print("【6】プラセボ — レース内で馬番をシャッフルした偽Harville（中身の無い対応関係）")
    print("　本物と同じ差が出るなら、この差は『確率の中身』ではなく手続きの産物")
    print("=" * 112)
    pl = collect(races, shuffle=True)
    print(f"{'券種':<8}{'本物 D(補正)':>14}{'プラセボ D(補正)':>18}{'差':>12}")
    for kind in PARTS:
        if kind not in keep or len(pl[kind]) < 500:
            continue
        pm = float(np.mean([v[1] for v in pl[kind]]))
        print(f"{kind:<8}{keep[kind][0]:>+14.4f}{pm:>+18.4f}{keep[kind][0]-pm:>+12.4f}")

    print("\n★読み方: プラセボは『市場の値付けは実現結果と相関するが、Harville側の対応を壊した』状態。")
    print("　本物 − プラセボ が、単勝オッズが実際に持っている情報量。")


if __name__ == "__main__":
    main()
