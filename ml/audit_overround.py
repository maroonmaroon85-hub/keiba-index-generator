"""(103) ★実効控除率のばらつきを使えるか — **情報ではなく「安く買えるレース」を探す**別筋。

★これまでと発想が違う点
　(89)以降の話は全部「**情報でDを稼ぐ**」だった。だが成長率は `log(払戻率) + D` なので、
　**払戻率そのものが高いレースがあれば、Dを1ナットも増やさずに得をする**。
　JRAの控除率は制度上一定（単勝80%）だが、**表示オッズは10円刻みに丸められる**ので、
　実際に買える条件はレースごとに違う。オーバーラウンド `Σ(1/オッズ)` はその実測値で、
　　　実効払戻率 ≈ 1 / Σ(1/オッズ)
　★**これは発走前に分かる**。予測能力を一切要求しない。ここが他の全ての試みと決定的に違う。

★★事前登録
　1. **予想**: ばらつきはあるが小さい（1.25±0.01程度）。頭数が多いほど丸めが効いて不利になるはず。
　2. **判定**: オーバーラウンドの十分位で、**1番人気の単勝ROIが単調に動くか**。
　　 動くなら「安いレースだけ買う」が成立する。★ただし
　　 **ばらつきが1%程度なら、20%の控除率は埋まらない**ので運用は変わらない。
　3. **★交絡**: オーバーラウンドは**頭数と強く相関する**はず。頭数を固定しても効くかを必ず見る。
　　 頭数で説明が付くなら、それは「少頭数レースを買え」という既知の話（(69)で否定済み）に戻る。
　4. **単調性で判定する**（(84)の教訓）。最良の十分位を選んで判定しない。

実行: python3 ml/audit_overround.py [開始年(既定2015)]
"""
import math
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "ml")
from audit_crosspool import load_races, payoff, probs, zq
from audit_crosspool2 import realized


def mci(x, alpha=0.01):
    x = np.asarray(x, float)
    m = x.mean()
    se = x.std(ddof=1) / math.sqrt(len(x))
    z = zq(alpha)
    return m, m - z * se, m + z * se


def main():
    y0 = int(sys.argv[1]) if len(sys.argv) > 1 else 2015
    rows = []
    for r in load_races():
        if r["year"] < y0:
            continue
        rl = realized(r)
        if rl is None:
            continue
        hs = r["horses"]
        inv = sum(1.0 / o for _, o, _ in hs if o and o > 0)
        if inv <= 0:
            continue
        p = probs(hs)
        k = int(np.argmax(p))
        fav, fodds = hs[k][0], hs[k][1]
        pay = payoff(r, "単勝", (fav,)) or 0.0
        pay3 = payoff(r, "複勝", (fav,)) or 0.0
        rows.append({"rid": r["rid"], "year": r["year"], "n": r["n"],
                     "inv": inv, "back": 1.0 / inv,
                     "tan": pay - 100.0, "fuku": pay3 - 100.0,
                     "fodds": fodds, "hit": 1 if pay > 0 else 0})
    df = pd.DataFrame(rows)
    print(f"(103) 実効控除率のばらつき（{y0}年以降・{len(df):,}レース）")
    print("★これは発走前に分かる量。予測能力を一切要求しない\n")

    print("=" * 96)
    print("【1】オーバーラウンド Σ(1/オッズ) の分布")
    print("=" * 96)
    q = df["inv"].quantile([0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99])
    print("  " + " / ".join(f"{int(k*100)}%点 {v:.4f}" for k, v in q.items()))
    print(f"  平均 {df['inv'].mean():.4f} / 標準偏差 {df['inv'].std():.4f}")
    print(f"  → 実効払戻率にすると 平均 {df['back'].mean()*100:.2f}% / "
          f"1%点 {1/q[0.99]*100:.2f}% 〜 99%点 {1/q[0.01]*100:.2f}%")
    print("  ※制度上の払戻率は80%。ここからのズレは**表示オッズの丸め**による。")

    print("\n" + "=" * 96)
    print("【2】頭数との関係（交絡の確認）")
    print("=" * 96)
    print(f"  相関 r = {df['inv'].corr(df['n']):+.3f}")
    print(f"{'頭数':<8}{'R数':>9}{'平均オーバーラウンド':>22}{'実効払戻率':>12}")
    for b, g in df.groupby(pd.cut(df["n"], [0, 9, 11, 13, 15, 99],
                                  labels=["〜9頭", "10-11頭", "12-13頭", "14-15頭", "16頭〜"]),
                           observed=True):
        print(f"{str(b):<8}{len(g):>9,}{g['inv'].mean():>22.4f}{1/g['inv'].mean()*100:>11.2f}%")

    print("\n" + "=" * 96)
    print("【3】★オーバーラウンドの十分位別 — 1番人気を買ったときのROI")
    print("=" * 96)
    df["dec"] = pd.qcut(df["inv"], 10, labels=False, duplicates="drop")
    print(f"{'十分位':<8}{'R数':>9}{'平均OR':>10}{'実効払戻率':>12}"
          f"{'単勝ROI':>10}{'99%CI':>22}{'複勝ROI':>10}{'平均頭数':>10}")
    for b, g in df.groupby("dec", observed=True):
        m, lo, hi = mci(g["tan"])
        roi = (m + 100) / 100 * 100
        froi = (g["fuku"].mean() + 100) / 100 * 100
        print(f"{'第'+str(int(b)+1):<8}{len(g):>9,}{g['inv'].mean():>10.4f}"
              f"{1/g['inv'].mean()*100:>11.2f}%{roi:>9.1f}%"
              f"{f'[{(lo+100):.1f},{(hi+100):.1f}]':>22}{froi:>9.1f}%{g['n'].mean():>10.1f}")
    rho = df.groupby("dec")["tan"].mean().reset_index()
    print(f"  単調性 ρ = {rho['dec'].corr(rho['tan'], method='spearman'):+.3f}"
          "（負なら『オーバーラウンドが低いほど得』＝仮説どおり）")

    print("\n" + "=" * 96)
    print("【4】★頭数を固定してもオーバーラウンドが効くか（交絡の除去）")
    print("=" * 96)
    print(f"{'頭数帯':<10}{'OR下位半分のROI':>18}{'OR上位半分のROI':>18}{'差':>10}{'99%CI':>22}")
    for b, g in df.groupby(pd.cut(df["n"], [0, 11, 13, 15, 99],
                                  labels=["〜11頭", "12-13頭", "14-15頭", "16頭〜"]),
                           observed=True):
        if len(g) < 1000:
            continue
        med = g["inv"].median()
        lo_g, hi_g = g[g["inv"] <= med], g[g["inv"] > med]
        d = lo_g["tan"].mean() - hi_g["tan"].mean()
        se = math.sqrt(lo_g["tan"].var(ddof=1) / len(lo_g) + hi_g["tan"].var(ddof=1) / len(hi_g))
        z = zq(0.01)
        print(f"{str(b):<10}{(lo_g['tan'].mean()+100):>17.1f}%{(hi_g['tan'].mean()+100):>17.1f}%"
              f"{d:>+10.2f}{f'[{d-z*se:+.2f},{d+z*se:+.2f}]':>22}")

    print("\n" + "=" * 96)
    print("★読み方")
    print("  ・実効払戻率の幅が数%あり、単調性が出て、頭数を固定しても残るなら、")
    print("    **情報を一切使わずに『安いレースだけ買う』が成立する**。")
    print("  ・ただし幅が1%程度なら、控除率20%は埋まらない。**運用は変わらない**。")
    print("  ・頭数で説明が付くなら(69)の『少頭数を買え』に戻るだけなので、そこは必ず分離する。")


if __name__ == "__main__":
    main()
