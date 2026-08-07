"""(108) ★時系列オッズは何開催ぶん集めれば判定できるか — **収集の前に必要量を決める**。

★なぜ先に計算するのか
　残った道は「単勝オッズの外にある情報」だけで、その筆頭が**オッズの時系列変化**。
　だが収集は手元Macでの手作業（(65)の手順・クラウドからは取得できない）なので、
　**「何開催ぶん必要か」を知らずに始めると、足りないまま結論を出すか、無駄に集めすぎる**。
　(92)⑤で36頭・(94)④で41頭しか無く判定できなかったのは、この計算をしていなかったから。

★測る対象（**先に問いを1つに絞る**）
　問い: **「発走前オッズの動きは、確定オッズに無い情報を持つか」**
　これは D で答えられる。q に「確定オッズのHarville」を入れた場合と、
　「確定オッズ＋動きの特徴」を入れた場合で D を比べればよい。
　→ **必要なのは「D の差を検出するのに何レース要るか」**。

★計算の中身（手元の361レースから**分散だけ**を推定する）
　1. 各レースの d のばらつき σ を、既存の361レースで実測する（**効果の大きさは推定しない**）。
　2. 検出したい効果 Δ に対して、必要レース数 n = (z_{α/2}+z_β)² σ² / Δ² を出す。
　3. Δ の候補は**実際に意味のある値**にする:
　　 ・Δ=+0.0024 … (102)のモデル混合と同じ大きさ（**足せたら同等**）
　　 ・Δ=+0.0100 … 必要量の4%（**モデル混合の4倍。ここまで出れば大きい**）
　　 ・Δ=+0.2367 … **枠連が儲かるのに必要な残り**（0.2549−0.0182）。参考として出す
　4. **1開催日あたりのレース数**（実測）で割って**開催日数**に直す。これが収集計画になる。

★★事前に言っておくこと
　・これは**効果があるかどうかの検定ではない**。**設計の計算**。
　・σ は券種で違う（三連単は配当の裾が厚い）。**券種ごとに出す**。
　・(89)④より、どれだけ集めても**上界を超えることはない**。ここで決めるのは
　　「**この道に賭ける価値があるか**」であって「儲かるか」ではない。

実行: python3 ml/audit_ts_power.py [開始年(既定2015)]
"""
import math
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "ml")
from audit_crosspool import PAYBACK, load_races, payoff, probs, zq
from audit_crosspool2 import PARTS, PAYKEY, realized
from audit_lbs import build_matrix, fit_lambda, q_of_lbs

DELTAS = [("(102)のモデル混合と同等", 0.0024), ("その4倍", 0.0100),
          ("必要量の半分", 0.1274), ("★儲かるのに必要な残り", 0.2367)]


def main():
    y0 = int(sys.argv[1]) if len(sys.argv) > 1 else 2015
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
        if yy < y0 or not lam.get(yy) or realized(r) is None:
            continue
        a, b, c = realized(r)
        num2k = {num: k for k, (num, _, _) in enumerate(r["horses"])}
        if a not in num2k or b not in num2k or (c is not None and c not in num2k):
            continue
        p = probs(r["horses"])
        l2, l3 = lam[yy]
        for kind, key in PARTS.items():
            if not r[key]:
                continue
            q, combo = q_of_lbs(kind, r, p, l2, l3, num2k, a, b, c)
            if q <= 0 or combo is None:
                continue
            v = payoff(r, PAYKEY[kind], combo)
            if not v or v <= 0:
                continue
            d = math.log(q) + math.log((v + 5) / 100.0) - math.log(PAYBACK[kind])
            rows.append({"kind": kind, "d": d, "rid": r["rid"]})
    df = pd.DataFrame(rows)

    # 1開催日あたりのレース数（raceid の先頭6桁＝場＋年＋回次日次で数える）
    per_day = df.groupby("kind")["rid"].nunique() / df["rid"].str[:6].nunique()

    print(f"(108) 時系列オッズの収集計画（{y0}年以降の実測σを使う）")
    print("★これは効果の検定ではなく**設計の計算**。集める前に必要量を決めるためのもの\n")
    print(f"{'券種':<8}{'σ(d)':>9}{'必要量':>9}" +
          "".join(f"{lab:>22}" for lab, _ in DELTAS))
    z = zq(0.01) + zq(0.20 * 2)          # α=0.01（両側）・検出力80%
    for kind in PARTS:
        g = df[df["kind"] == kind]
        if len(g) < 2000:
            continue
        s = g["d"].std()
        need = -math.log(PAYBACK[kind])
        cells = []
        for _, delta in DELTAS:
            n = (z ** 2) * (s ** 2) / (delta ** 2)
            cells.append(f"{n:>13,.0f}R")
        print(f"{kind:<8}{s:>9.3f}{need:>9.4f}" + "".join(f"{c:>22}" for c in cells))

    print(f"\n{'='*104}")
    print("【開催日数に直す】1開催日あたりのレース数（実測）で割る")
    print("=" * 104)
    print(f"{'券種':<8}{'1日あたり':>10}" + "".join(f"{lab:>22}" for lab, _ in DELTAS))
    for kind in PARTS:
        g = df[df["kind"] == kind]
        if len(g) < 2000:
            continue
        s = g["d"].std()
        pd_ = per_day.get(kind, 36.0)
        cells = []
        for _, delta in DELTAS:
            n = (z ** 2) * (s ** 2) / (delta ** 2)
            cells.append(f"{n/pd_:>13,.0f}日")
        print(f"{kind:<8}{pd_:>10.1f}" + "".join(f"{c:>22}" for c in cells))

    print(f"\n{'='*104}")
    print("★読み方")
    print("  ・α=0.01（両側）・検出力80% で計算した。**判定基準を緩めれば必要量は減る**が、")
    print("    このプロジェクトは偽陽性を何度も出しているので緩めないこと。")
    print("  ・手元にあるのは **361レース＝約10開催日**。表の日数と比べれば足りるかが分かる。")
    print("  ・★**「必要量の残り(0.2367)」の列に届く日数なら、集める価値がある**。")
    print("    その列が非現実的な日数なら、この道でも100%には届かないと**集める前に**分かる。")
    print("  ・σ が小さい券種ほど少ない標本で判定できる。**枠連が最も測りやすい**はず。")


if __name__ == "__main__":
    main()
