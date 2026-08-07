"""(98) λ補正は**買い目の選択**を変えるか — (61)「枠確率での集約は改善せず」を測り直す。

★revisit する理由（「否定済みに戻らない」の例外にあたる理由を先に書く）
　(61)は「枠連はHarvilleで枠ペアを集計して最大の組を買っても改善しない」と結論した。
　だが(96)で **その集計に使ったHarville自体に系統誤差がある**と分かった。
　Harvilleは人気馬の2着確率を過大評価するので、**人気馬が単独で入っている枠を過大に選ぶ**方向に歪む。
　＝(61)は**歪んだ道具で測った結論**。道具を直したら答えが変わりうる。
　★これは「同じことをもう一度やる」ではなく「**前提が変わったのでやり直す**」に当たる。

★★事前登録（測る前に宣言）
　1. **比較する3つ**（全て1点・100円）:
　　 a 人気順  … 単勝上位2頭の枠（現行の考え方に最も近い・モデル不要）
　　 b Harville … 枠ペアの確率が最大の組（(61)が測ったもの）
　　 c **λ補正** … 同じことを λ2 補正した確率でやる（λは各年それ以前の年だけで推定）
　2. **判定量は円**（(80)の判定基準8）。1レース期待損失で並べる。ROIは参考。
　3. **判定**: c が b より**対応ありで**期待損失が小さく、CIが0を除外すること。
　　 ★同時に「c と b が同じ組を選ぶ割合」を出す。**9割方同じなら差が出ようがない**ので、
　　 　まず一致率を見る。一致率が高いのに差が大きく出たら、それは少数の外れ値＝疑う。
　4. **★上界は動かない**: (89)④より枠連の成長率は log(0.775)+D を超えない。
　　 D=+0.0182 なので**どの選び方でも1レース −21%**。ここで測っているのは
　　 「その上界にどれだけ近づけるか」であって、勝てる買い方の探索ではない。

実行: python3 ml/audit_lbs_bet.py [開始年(既定2015)]
"""
import math
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, "ml")
from audit_crosspool import PAYBACK, load_races, payoff, probs, zq
from audit_crosspool2 import realized
from audit_lbs import build_matrix, fit_lambda, g_pair_unordered, stage_w
from waku_umatan import waku_of


def best_frame_pair(r, p, lam2):
    """枠ペアの確率が最大になる組を返す。lam2=1.0 で(61)と同じHarville。"""
    hs = r["horses"]
    n = r["n"]
    wk = defaultdict(list)
    for k, (num, _, _) in enumerate(hs):
        wk[waku_of(num, n)].append(k)
    w2 = stage_w(p, lam2)
    W2 = w2.sum()
    best, bv = None, -1.0
    frames = sorted(wk)
    for i in range(len(frames)):
        for j in range(i, len(frames)):
            fa, fb = frames[i], frames[j]
            v = 0.0
            if fa == fb:
                mem = wk[fa]
                for x in range(len(mem)):
                    for y in range(x + 1, len(mem)):
                        v += g_pair_unordered(p, w2, W2, mem[x], mem[y])
            else:
                for x in wk[fa]:
                    for y in wk[fb]:
                        v += g_pair_unordered(p, w2, W2, x, y)
            if v > bv:
                best, bv = (fa, fb), v
    return best


def mci(x, alpha=0.01):
    x = np.asarray(x, float)
    m = x.mean()
    se = x.std(ddof=1) / math.sqrt(len(x))
    z = zq(alpha)
    return m, m - z * se, m + z * se


def main():
    y0 = int(sys.argv[1]) if len(sys.argv) > 1 else 2015
    races = load_races()
    P, i1, i2, i3, yrs = build_matrix(races, y0)
    lam = {}
    for yy in sorted(set(yrs.tolist())):
        tr = yrs < yy
        lam[yy] = fit_lambda(P[tr], i1[tr], i2[tr]) if tr.sum() >= 3000 else None
    got = {k: v for k, v in lam.items() if v}
    print(f"(98) λ補正は買い目の選択を変えるか（{y0}年以降）")
    print(f"  λ2 のウォークフォワード推定: {min(got.values()):.3f}〜{max(got.values()):.3f}")
    print("  ★判定量は円（1レース期待損失）。ROIは参考。\n")

    rows = []
    for r in races:
        yy = r["year"]
        if yy < y0 or not got.get(yy) or not r["wakuren"]:
            continue
        rl = realized(r)
        if rl is None:
            continue
        a, b, _ = rl
        hs = r["horses"]
        p = probs(hs)
        nums = [hs[k][0] for k in np.argsort(-p)]
        hit = tuple(sorted((waku_of(a, r["n"]), waku_of(b, r["n"]))))
        pick = {
            "a 人気順": tuple(sorted((waku_of(nums[0], r["n"]), waku_of(nums[1], r["n"])))),
            "b Harville": best_frame_pair(r, p, 1.0),
            "c λ補正": best_frame_pair(r, p, got[yy]),
        }
        row = {"rid": r["rid"], "year": yy}
        for k, v in pick.items():
            pay = payoff(r, "枠連(人気順)", v) if v == hit else 0.0
            row[k] = (pay or 0.0) - 100.0        # 100円買って払戻。損益（円）
            row[k + "_hit"] = 1 if v == hit else 0
        row["同じ組か(b=c)"] = 1 if pick["b Harville"] == pick["c λ補正"] else 0
        row["同じ組か(a=c)"] = 1 if pick["a 人気順"] == pick["c λ補正"] else 0
        rows.append(row)

    import pandas as pd
    df = pd.DataFrame(rows)
    print(f"対象 {len(df):,}レース\n")
    print("=" * 88)
    print("【1】そもそも選ぶ組は違うのか")
    print("=" * 88)
    print(f"  b(Harville) と c(λ補正) が同じ組: {df['同じ組か(b=c)'].mean()*100:.1f}%")
    print(f"  a(人気順)   と c(λ補正) が同じ組: {df['同じ組か(a=c)'].mean()*100:.1f}%")
    if df["同じ組か(b=c)"].mean() > 0.95:
        print("  ⚠**ほぼ同じ組を選んでいる**。差が出るとしたら少数の外れレースなので、"
              "有意でも実質的な意味は小さい。")

    print("\n" + "=" * 88)
    print("【2】1レース期待損失（円）— これが判定量")
    print("=" * 88)
    print(f"{'選び方':<14}{'的中率':>9}{'ROI':>9}{'1R損益':>11}{'99%CI':>24}")
    for k in ("a 人気順", "b Harville", "c λ補正"):
        m, lo, hi = mci(df[k])
        roi = (df[k].mean() + 100) / 100 * 100
        print(f"{k:<14}{df[k+'_hit'].mean()*100:>8.1f}%{roi:>8.1f}%{m:>+11.2f}円"
              f"{f'[{lo:+.2f},{hi:+.2f}]':>24}")

    print("\n" + "=" * 88)
    print("【3】対応ありの差（同じレースでの引き算）")
    print("=" * 88)
    for x, y in (("c λ補正", "b Harville"), ("c λ補正", "a 人気順"), ("b Harville", "a 人気順")):
        m, lo, hi = mci(df[x] - df[y])
        v = "★有利" if lo > 0 else ("★不利" if hi < 0 else "区別できない")
        print(f"  {x} − {y}: {m:+.2f}円 [{lo:+.2f},{hi:+.2f}]  {v}")

    print("\n" + "=" * 88)
    print("★読み方")
    print("  ・(61)は『枠確率での集約は改善せず』としたが、それは歪んだ変換で測った結論だった。")
    print("  ・それでも(89)④より枠連の1レース成長率は −21% を超えない。**儲かる話にはならない**。")
    print("  ・ここで見ているのは『上界にどれだけ近づけるか』であって、勝てる買い方の探索ではない。")


if __name__ == "__main__":
    main()
