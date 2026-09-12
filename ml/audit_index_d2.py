"""(120) ★(86)「指数Dの上位区間」を**選別変数としてDで測り直す**（2026-08-11）

★なぜやり直すか
　(86)はROIで「5条件中0件通過。消えた」と結論した。**陰性の結論**なので安全に見えるが、
　今日(117)で「(62)のROI陰性は偽陰性だった」と実証された。同じ疑いがかかる。
　しかも(86)が見ていたのは**ROIが100%を超えるか**であって、
　**E[d|S]が上がるか**（＝選別として機能するか）は測っていない。この2つは別の問い。

★★なぜ「第3の信号」の可能性があるか
　(119)で「(112)の軸E」と「(117)の枠連スコア」は**同じ信号**（相関+0.572・包含関係）と分かった。
　どちらも「**断然人気がいるレース**」を選んでいる。
　**指数D = 軸のモデルシェア − 軸の市場含意確率**は性質が違う——
　「モデルが市場より高く買っている」量で、**人気の強さとは独立でありうる**。
　⚠ただし(75)で「残差だけで順位付けするとAUC0.3605＝逆相関」と出ている。
　　残差系の量は**人気薄を拾う方向**に働きやすい。そこは警戒する。

★★事前登録（測る前に宣言）
　1. 上位 **5/10/20/40%** の4水準（(86)の 5/10/20 に 40 を足した。後から増やさない）。
　2. **プラセボは200回引いて平均**（(117)で1回引きが偽の単調を作ると分かったため）。
　3. **判定の本体は「上位 vs 残り」の2標本検定**（互いに素）。
　4. **枠連スコアとの相関**も出す。0.7以上なら(117)の代理に過ぎない。
　5. **予想**: **効かない**。理由は(75)——モデル−市場の残差は人気薄を拾う方向に働く。
　　 効くとしても(119)と同じ信号（相関が高く出る）と予想する。
　　 ⚠(117)で予想を外しているので、この予想も外れる前提で読むこと。

実行: python3 ml/audit_index_d2.py [開始年(既定2015)]
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

CUTS = [0.05, 0.10, 0.20, 0.40]          # ★先に宣言
MODEL_CACHE = "data/cache/exp_L2-top3_2015"
RNG = np.random.default_rng(20260811)


def mci(x, alpha=0.01):
    x = np.asarray(x, float)
    if len(x) < 2:
        return float("nan"), float("nan"), float("nan")
    m = x.mean()
    se = x.std(ddof=1) / math.sqrt(len(x))
    return m, m - zq(alpha) * se, m + zq(alpha) * se


def main():
    y0 = int(sys.argv[1]) if len(sys.argv) > 1 else 2015
    mp = {}
    for f in sorted(glob.glob(f"{MODEL_CACHE}/*.csv")):
        for rid, u, p in pd.read_csv(f)[["raceid", "umaban", "p"]].itertuples(index=False):
            mp.setdefault(str(rid), {})[int(u)] = float(p)
    if not mp:
        sys.exit(f"{MODEL_CACHE} が無い")

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
        p_mkt = probs(hs)
        tot = sum(pm[u] for u in nums)
        if tot <= 0:
            continue
        share = {u: pm[u] / tot for u in nums}
        axis = max(nums, key=lambda u: share[u])         # モデルの1位＝軸
        # ★指数D = 軸のモデルシェア − 軸の市場含意確率（(85)(86)の定義）
        idx_d = share[axis] - p_mkt[num2k[axis]]
        order = sorted(nums, key=lambda u: -share[u])
        sc = waku_score(wakuren_buy(order, r["n"], 2),
                        bracket_probs(nums, [pm[u] for u in nums], r["n"]))
        l2, l3 = lam[yy]
        q, combo = q_of_lbs("枠連", r, p_mkt, l2, l3, num2k, a, b, c)
        if q <= 0 or combo is None:
            continue
        v = payoff(r, PAYKEY["枠連"], combo)
        if not v or v <= 0:
            continue
        pairs = wakuren_buy(order, r["n"], 2)
        ret = sum(payoff(r, PAYKEY["枠連"], list(pr)) or 0.0 for pr in pairs)
        rows.append(dict(year=yy, idx=idx_d, sc=sc,
                         d=math.log(q) + math.log((v + 5) / 100.0)
                         - math.log(PAYBACK["枠連"]),
                         bet=len(pairs) * 100, ret=ret))

    df = pd.DataFrame(rows)
    d = df["d"].to_numpy()
    idx = df["idx"].to_numpy()
    print(f"(120) 指数D（軸のモデルシェア − 軸の市場含意確率）を選別変数としてDで測る"
          f"（{y0}年以降・{len(df):,}レース）\n")

    rho = float(np.corrcoef(pd.Series(idx).rank(), pd.Series(df["sc"]).rank())[0, 1])
    print(f"★枠連スコアとの順位相関: **{rho:+.3f}**"
          "（0.7以上なら(117)の代理に過ぎない／0.3未満なら別の信号）\n")

    base = d.mean()
    print(f"全体 E[d] = {base:+.4f}（必要量 {-math.log(PAYBACK['枠連']):.4f}）\n")
    print(f"{'上位':>6}{'R数':>8}{'E[d|S]':>10}{'99%CI':>22}{'ROI':>8}"
          f"{'プラセボ':>10}{'実測−プラセボ':>14}")
    for cpt in CUTS:
        th = np.quantile(idx, 1 - cpt)
        m = idx >= th
        n = int(m.sum())
        mm, lo, hi = mci(d[m])
        roi = df.loc[m, "ret"].sum() / df.loc[m, "bet"].sum() * 100
        pl = float(np.mean([d[RNG.choice(len(d), size=n, replace=False)].mean()
                            for _ in range(200)]))
        print(f"{cpt:>6.0%}{n:>8,}{mm:>+10.4f}{f'[{lo:+.4f},{hi:+.4f}]':>22}"
              f"{roi:>8.1f}{pl:>+10.4f}{mm-pl:>+14.4f}")

    print("\n★上位 vs 残り（互いに素・判定の本体）")
    print(f"{'上位':>6}{'上位D':>10}{'残りD':>10}{'差':>10}{'99%CI':>22}{'判定':>7}")
    for cpt in CUTS:
        th = np.quantile(idx, 1 - cpt)
        x, y = d[idx >= th], d[idx < th]
        diff = x.mean() - y.mean()
        se = math.sqrt(x.var(ddof=1) / len(x) + y.var(ddof=1) / len(y))
        z = zq(0.01)
        print(f"{cpt:>6.0%}{x.mean():>+10.4f}{y.mean():>+10.4f}{diff:>+10.4f}"
              f"{f'[{diff-z*se:+.4f},{diff+z*se:+.4f}]':>22}"
              f"{'★' if diff - z*se > 0 else '':>7}")

    print("\n★読み方（事前登録のとおり）")
    print("  ・★が付き、枠連スコアとの相関が低ければ **第3の信号**＝重ねる価値がある。")
    print("  ・★が付かなければ (86)の陰性は D でも陰性＝**偽陰性ではなかった**と確定する。")
    print("  ・符号が負なら (75)のとおり **残差は人気薄を拾う**方向に働いたということ。")


if __name__ == "__main__":
    main()
