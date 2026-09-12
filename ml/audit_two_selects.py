"""(119) ★★(112)の選別と(117)の選別は**同じものか、重ねられるか**（2026-08-11）

★なぜこれをやるか（(86)の測り直しより価値が高いと判断して差し替えた）
　選別で D が上がる、という結果が**2つ独立に出ている**:
　　・(112) **軸のλ補正top3**（市場オッズ由来）で裾を切る → 枠連 +0.0182 → +0.0394
　　・(117) **枠連スコア**（モデル由来）で下位を外す → 枠連 +0.0184 → +0.0235（40%除外）
　どちらも「断然人気がいるレースを選ぶ」に帰着している疑いが濃い。
　**同じなら統合して1つの基準にすべきで、別物なら重ねれば足し算になる**。運用が変わりうる。

★★事前登録（測る前に宣言）
　1. 2軸を**それぞれ独立に**中央で切って 2×2 にする。水準は後から増やさない。
　　 ・軸E: `soft_axis.axis_expect` の期待払戻（**小さいほど良い**）の下位20%
　　 ・枠連スコア: 下位20%を外す（＝現行運用）
　2. **判定**: 片方で切ったあと、**もう片方がまだ効くか**。
　　 ・効かない → **同じ信号**。統合する（片方だけ使えばよい）
　　 ・効く → **別の信号**。重ねる価値がある
　3. **相関も出す**（Spearman）。0.7以上なら実質同じと見なす。
　4. **予想**: **ほぼ同じ信号**（相関0.5以上）で、重ねても片方単独＋0.005未満。
　　 理由: どちらも「1番人気が強いレース」に高い値を付ける構造。
　　 ⚠(117)で予想を外している（「ほぼ動かない」と書いて実際は10/10年で正だった）ので、
　　 　 この予想も外れる前提で読むこと。
　5. **どの水準でも必要量 0.2549 には遠いはず**。運用が変わるとしたら「どちらを使うか」だけ。

実行: python3 ml/audit_two_selects.py [開始年(既定2015)]
"""
import glob
import math
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "ml")
from audit_crosspool import PAYBACK, load_races, payoff, probs, zq
from audit_crosspool2 import PAYKEY, realized
from audit_lbs import build_matrix, fit_lambda, q_of_lbs
from waku_umatan import bracket_probs, waku_score, wakuren_buy
import soft_axis as SA

MODEL_CACHE = "data/cache/exp_L2-top3_2015"
RNG = np.random.default_rng(20260811)


def mci(x, alpha=0.01):
    x = np.asarray(x, float)
    if len(x) < 2:
        return float("nan"), float("nan"), float("nan")
    m = x.mean()
    se = x.std(ddof=1) / math.sqrt(len(x))
    return m, m - zq(alpha) * se, m + zq(alpha) * se


def spearman_rank(a, b):
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    y0 = int(sys.argv[1]) if len(sys.argv) > 1 else 2015
    fs = sorted(glob.glob(f"{MODEL_CACHE}/*.csv"))
    if not fs:
        sys.exit(f"{MODEL_CACHE} が無い")
    mp = {}
    for f in fs:
        for rid, u, p in pd.read_csv(f)[["raceid", "umaban", "p"]].itertuples(index=False):
            mp.setdefault(str(rid), {})[int(u)] = float(p)

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

    rows = []
    for r in races:
        yy = r["year"]
        if yy < y0 or not lam.get(yy) or not r["wakuren"]:
            continue
        pm = mp.get(r["rid"])
        if not pm:
            continue
        rl = realized(r)
        if rl is None:
            continue
        a, b, c = rl
        hs = r["horses"]
        nums = [u for u, _, _ in hs]
        if any(u not in pm for u in nums):
            continue
        num2k = {u: k for k, (u, _, _) in enumerate(hs)}
        if a not in num2k or b not in num2k:
            continue
        _, e_axis, _ = SA.axis_expect([o for _, o, _ in hs])
        if e_axis is None:
            continue
        order = sorted(nums, key=lambda u: -pm[u])
        sc = waku_score(wakuren_buy(order, r["n"], 2),
                        bracket_probs(nums, [pm[u] for u in nums], r["n"]))
        p = probs(hs)
        l2, l3 = lam[yy]
        q, combo = q_of_lbs("枠連", r, p, l2, l3, num2k, a, b, c)
        if q <= 0 or combo is None:
            continue
        v = payoff(r, PAYKEY["枠連"], combo)
        if not v or v <= 0:
            continue
        rows.append(dict(year=yy, e=e_axis, sc=sc,
                         d=math.log(q) + math.log((v + 5) / 100.0)
                         - math.log(PAYBACK["枠連"])))

    df = pd.DataFrame(rows)
    d = df["d"].to_numpy()
    print(f"(119) (112)の選別と(117)の選別は同じものか（{y0}年以降・{len(df):,}レース）\n")

    # ★相関。軸Eは小さいほど良いので符号を反転して「良さ」に揃える
    rho = spearman_rank(-df["e"].to_numpy(), df["sc"].to_numpy())
    print(f"★2つの選別変数の順位相関: **{rho:+.3f}**"
          "（0.7以上なら実質同じ／0.3未満ならほぼ別物）\n")

    e_th = np.quantile(df["e"], 0.20)          # 軸Eの下位20%＝甘い軸
    s_th = np.quantile(df["sc"], 0.20)         # 枠連スコアの下位20%＝現行の除外
    good_e = df["e"].to_numpy() <= e_th
    good_s = df["sc"].to_numpy() > s_th
    base = d.mean()
    print(f"全体 E[d] = {base:+.4f}（必要量 0.2549）\n")
    print(f"{'区分':<34}{'R数':>8}{'E[d|S]':>10}{'99%CI':>22}{'全体との差':>12}")
    for lab, m in (("① 両方とも良い（甘い軸＋高スコア）", good_e & good_s),
                   ("② 甘い軸だけ（スコアは下位）", good_e & ~good_s),
                   ("③ 高スコアだけ（軸は甘くない）", ~good_e & good_s),
                   ("④ どちらでもない", ~good_e & ~good_s)):
        if m.sum() < 100:
            continue
        mm, lo, hi = mci(d[m])
        print(f"{lab:<34}{int(m.sum()):>8,}{mm:>+10.4f}"
              f"{f'[{lo:+.4f},{hi:+.4f}]':>22}{mm-base:>+12.4f}")

    # ★★本題: 片方で切ったあと、もう片方がまだ効くか（互いに素の2標本で見る）
    print("\n★★片方で切ったあと、もう片方がまだ効くか（これが判定の本体）")
    for lab, sub, m1, m0 in (
            ("甘い軸の中で 枠連スコア上位80% vs 下位20%", good_e,
             good_e & good_s, good_e & ~good_s),
            ("スコア上位80%の中で 甘い軸 vs そうでない", good_s,
             good_s & good_e, good_s & ~good_e)):
        x, y = d[m1], d[m0]
        if len(x) < 100 or len(y) < 100:
            continue
        diff = x.mean() - y.mean()
        se = math.sqrt(x.var(ddof=1) / len(x) + y.var(ddof=1) / len(y))
        z = zq(0.01)
        print(f"  {lab}")
        print(f"    {x.mean():+.4f}（{len(x):,}本） vs {y.mean():+.4f}（{len(y):,}本）"
              f"  差 {diff:+.4f}  99%CI [{diff-z*se:+.4f},{diff+z*se:+.4f}]"
              f"{'  ★まだ効く' if diff - z*se > 0 else '  効かない'}")

    print("\n★読み方（事前登録のとおり）")
    print("  ・両方とも『まだ効く』なら **別の信号**。重ねる価値がある＝運用を変える余地。")
    print("  ・片方だけなら、効かないほうは**もう片方の代理**だったということ。")
    print("  ・どちらも効かないなら、①の利得は単に**部分集合が小さいことの偶然**。")


if __name__ == "__main__":
    main()
