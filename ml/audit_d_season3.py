"""(122) ★★場・季節の層別を**モデル混合まで入れた今のq**でやる（2026-08-11）

★経緯（ここが大事）
　(91) 素の市場Harville で層別 → 「厚いプールほどDが大きい」
　(116) **λ補正**した市場qで測り直し → **3券種で差が消えた**。(91)②は半分が近似誤差だった
　(122) **さらにモデル混合まで入れる** ← いま
　★(116)の事前登録には「**モデル混合は入れない。混合は一様に効くため**」と書いた。
　　**その理由は同じ日に(117)で否定された**——モデル由来の枠連スコアで切るとDが動く
　　（全水準★・10/10年で正）。**モデルの寄与はレースによって一様ではない**。
　　→ 「結果を見てから基準を動かす」のではなく「**宣言した根拠が実測で崩れた**」ので測り直す。

★★事前登録（測る前に宣言）
　1. 混合比は **w=0.14 固定**（(102)の「市場を土台にモデルを14%混ぜる」）。**探索しない**。
　　 探索すると「どの層で最適wが違うか」という別の問いになり、必ず偽陽性が出る。
　2. 対象は **枠連のみ**（(112)(114)(117)で唯一利得が出ている券種。他は測る意味が薄い）。
　3. 軸は **季節 / 場 / 頭数帯**。(91)(116)と同じ。判定は **α=0.05/18**（同じ基準を使う）。
　4. **予想**: 全体のDは上がる（+0.0183 → +0.020前後）が、**層のパターンは(116)から動かない**。
　　 理由: (117)で効いたのは「人気が強いレースを選ぶ」型で、これは(119)より市場側の
　　 　　　軸Eとほぼ同じ信号。**市場qの層別で既に織り込まれている**はず。
　5. ⚠(117)(121)で予想を外している。この予想も外れる前提で読むこと。

★道具の注意（(121)で2回間違えた点）
　・モデル由来の枠連分布は **全枠組で正規化する**
　・**1頭しかいない枠にゾロ目を作らない**（成立しない組に確率を捨てるとモデル間で歪む）

実行: python3 ml/audit_d_season3.py [開始年(既定2015)]
"""
import glob
import math
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "ml")
from audit_capacity_d import mkt_waku_dist, waku_dist
from audit_crosspool import PAYBACK, load_races, payoff, probs, zq
from audit_crosspool2 import PAYKEY, realized
from audit_d_season import LOCAL, PLACE, SEASON, month_map
from audit_lbs import build_matrix, fit_lambda, q_of_lbs

W = 0.14                                  # ★(102)の混合比。探索しない
MODEL_CACHE = "data/cache/exp_L2-top3_2015"


def mci(x, alpha):
    x = np.asarray(x, float)
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
    mmap = month_map()
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
        l2, l3 = lam[yy]
        q, combo = q_of_lbs("枠連", r, p_mkt, l2, l3, num2k, a, b, c)
        if q <= 0 or combo is None:
            continue
        v = payoff(r, PAYKEY["枠連"], combo)
        if not v or v <= 0:
            continue
        key = tuple(sorted(combo))
        md = mkt_waku_dist(r, p_mkt, l2)
        dd = waku_dist(nums, pm, r["n"])
        if not md or not dd or set(dd) != set(md) or md.get(key, 0) <= 0 or dd.get(key, 0) <= 0:
            continue
        keys = sorted(md)
        lm = np.array([math.log(md[t]) for t in keys])
        lk = np.array([math.log(dd[t]) for t in keys])
        j = keys.index(key)
        z = (1 - W) * lm + W * lk
        z -= z.max()
        lmix = float(z[j] - math.log(np.exp(z).sum()))
        lg = math.log((v + 5) / 100.0) - math.log(PAYBACK["枠連"])
        mon = mmap.get(r["rid"])
        place = PLACE.get(r["rid"][:2], "?")
        rows.append({"year": yy, "n": r["n"], "place": place,
                     "local": place in LOCAL,
                     "season": SEASON.get(mon, "?") if mon else "?",
                     "d_lbs": lm[j] + lg, "d_mix": lmix + lg})

    df = pd.DataFrame(rows)
    df["fsbin"] = pd.cut(df["n"], [0, 11, 13, 15, 99],
                         labels=["〜11頭", "12-13頭", "14-15頭", "16頭〜"])
    alpha = 0.05 / 18
    m_l, _, _ = mci(df["d_lbs"], alpha)
    m_m, lo, hi = mci(df["d_mix"], alpha)
    print(f"(122) 場・季節の層別を**モデル混合(w={W})まで入れたq**で（{y0}年以降・{len(df):,}レース）")
    print(f"★全体 枠連 D: λ補正のみ {m_l:+.4f} → **混合 {m_m:+.4f}** "
          f"[{lo:+.4f},{hi:+.4f}]  必要量 0.2549\n")

    for axis, label in (("season", "季節"), ("place", "場"),
                        ("fsbin", "頭数帯"), ("local", "ローカル開催か")):
        print(f"■ {label}")
        print(f"{'区分':<12}{'件数':>9}{'λのみ':>10}{'混合':>10}"
              f"{'全体との差':>12}{'99.7%CI':>24}{'判定':>7}")
        for v, g in df.groupby(axis, observed=True):
            if len(g) < 300:
                continue
            mm, l2_, h2 = mci(g["d_mix"], alpha)
            mark = "★上" if l2_ > m_m else ("★下" if h2 < m_m else "")
            print(f"{str(v):<12}{len(g):>9,}{g['d_lbs'].mean():>+10.4f}{mm:>+10.4f}"
                  f"{mm-m_m:>+12.4f}{f'[{l2_:+.4f},{h2:+.4f}]':>24}{mark:>7}")
        print()

    print("=" * 96)
    print("★読み方（事前登録のとおり）")
    print("  ・層のパターンが(116)から動かなければ、**場・季節はこれで完全に閉じる**。")
    print("  ・動くなら、**モデルの寄与が層によって違う**ということ＝(117)の続きがある。")
    print("  ・どの層でも必要量0.2549には遠いはずで、運用は変わらない。")


if __name__ == "__main__":
    main()
