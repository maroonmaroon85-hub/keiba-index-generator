"""
(92) ★複勝プールの D を測る — 必要量が最小(0.2231)で、唯一「線」を14/14年超えている券種。

(89)②で複勝だけが特異だった:
| 券種 | 人気順ROI | 払戻率の線 | 差 | 線超えの年 |
|---|---|---|---|---|
| **複勝** | **84.1%** | 80.0% | **+4.07pt** | **14/14** |
| 単勝 | 78.8% | 80.0% | −1.17pt | 3/14 |
**払戻率も「1頭を選ぶ」構造も同じなのに、単勝は線を下回り複勝は超える**。
複勝プール固有の何かがある。しかも**必要量 |log(0.8)|=0.2231 は全券種で最小**。

★事前に立てた仮説（測る前に宣言）
　**複勝には最低配当100円の保証がある**。圧倒的1番人気で本来90円が妥当な場面でも100円返る。
　これは買い手に有利な非対称で、**払戻率80%の線を上に押し上げる**。
　単勝は下限が効く場面がほとんど無い（オッズ1.0倍は稀）が、**複勝は人気馬で日常的に100円配当が出る**。
　→ **100円配当を除いた瞬間にDが消えるなら、複勝の優位は「最低保証」という制度の産物**。
　　 そしてそれは**実在する優位**——制度で保証されているので次の10年でも消えない。
　→ 消えなければ、複勝プールの値付けそのものが甘いことになる。**そちらの方が大きい話**。

★対数スコアの扱い（(89)③と同じ枠組みが使える理由）
　複勝は「3頭が同時に的中する」ので一見(89)の枠組み（排他的な組）に乗らないが、
　**「無作為に選んだ入着枠をどの馬が占めるか」という実験**と見れば、
　着内3頭それぞれが1標本になり、確率は q_i/3（Harvilleの3着以内確率）と p_i/3（プール由来）。
　1/3は差では消えるので、**(89)と同じ式がそのまま使える**:
　　　d = log q + log(払戻/100) − log(払戻率)

⚠**丸めの扱い**: (89)は10円切り捨てを+5円で補正している。だが**配当100円は切り捨てではなく下限**なので、
　+5すると過大評価になる。**100円の行を含む版と除く版の両方**を出して、どちらでも結論が立つか見る。

実行: python3 ml/audit_fuku_d.py [開始年(既定2015)]
"""
import math
import sys

import numpy as np

sys.path.insert(0, "ml")
from audit_crosspool import PAYBACK, load_races, probs, zq
from model_line import harville_top3

NEED = -math.log(PAYBACK["複勝"])          # 0.2231


def mci(x, alpha=0.01):
    x = np.asarray(x, float)
    m = x.mean()
    se = x.std(ddof=1) / math.sqrt(len(x))
    z = zq(alpha)
    return m, m - z * se, m + z * se


def collect(races, y0):
    rows = []
    for r in races:
        if r["year"] < y0 or not r["fuku"]:
            continue
        hs = r["horses"]
        p = probs(hs)
        t3 = harville_top3(p)          # 各馬の3着以内確率（Harville）
        num2k = {num: k for k, (num, _, _) in enumerate(hs)}
        for key, v in r["fuku"].items():
            num = key[0] if isinstance(key, tuple) else key
            k = num2k.get(int(num))
            if k is None or not v or v <= 0:
                continue
            q = float(t3[k])
            if q <= 0 or q >= 1:
                continue
            odds = hs[k][1]
            for tag, val in (("with", v + 5), ("raw", v)):
                rows.append({"tag": tag, "d": math.log(q) + math.log(val / 100.0)
                             - math.log(PAYBACK["複勝"]),
                             "pay": v, "odds": odds, "year": r["year"],
                             "floor": v == 100, "n": r["n"], "q": q})
    return rows


def main():
    y0 = int(sys.argv[1]) if len(sys.argv) > 1 else 2015
    races = load_races()
    import pandas as pd
    df = pd.DataFrame(collect(races, y0))
    if df.empty:
        sys.exit("複勝データが取れなかった")
    w = df[df["tag"] == "with"]
    print(f"複勝プールの D（{y0}年以降・{len(w):,}件・{w['year'].nunique()}年）")
    print(f"★必要量 |log(0.8)| = {NEED:.4f}（全券種で最小）\n")

    print("=" * 96)
    print("【1】全体")
    print("=" * 96)
    for tag, lab in (("with", "10円切り捨てを+5円補正（(89)と同じ）"), ("raw", "補正なし")):
        x = df[df["tag"] == tag]["d"]
        m, lo, hi = mci(x)
        print(f"  {lab:<34} D={m:+.4f} [{lo:+.4f},{hi:+.4f}]"
              f"  必要量の{max(m,0)/NEED*100:.1f}%")

    print("\n" + "=" * 96)
    print("【2】★仮説の検定 — 最低配当100円の保証が正体か")
    print("=" * 96)
    fl = w["floor"].mean() * 100
    print(f"  配当がちょうど100円の割合: {fl:.1f}%")
    for lab, g in (("100円を含む（全体）", w), ("100円を除く", w[~w["floor"]]),
                   ("100円だけ", w[w["floor"]])):
        if len(g) < 300:
            continue
        m, lo, hi = mci(g["d"])
        print(f"  {lab:<20}{len(g):>9,}件  D={m:+.4f} [{lo:+.4f},{hi:+.4f}]"
              f"  必要量の{max(m,0)/NEED*100:.1f}%")
    print("  ※100円を除いてDが消えるなら**制度（最低保証）の産物**。"
          "残るなら**プールの値付けそのものが甘い**。")

    print("\n" + "=" * 96)
    print("【3】単勝オッズ帯別（どの馬で甘いのか）")
    print("=" * 96)
    w = w.copy()
    w["band"] = pd.cut(w["odds"], [0, 2, 4, 8, 16, 40, 1e9],
                       labels=["〜2倍", "2-4倍", "4-8倍", "8-16倍", "16-40倍", "40倍〜"])
    print(f"{'オッズ帯':<10}{'件数':>9}{'100円率':>9}{'平均配当':>10}{'D':>11}{'99%CI':>22}")
    for b, g in w.groupby("band", observed=True):
        if len(g) < 300:
            continue
        m, lo, hi = mci(g["d"])
        print(f"{str(b):<10}{len(g):>9,}{g['floor'].mean()*100:>8.1f}%"
              f"{g['pay'].mean():>9.0f}円{m:>+11.4f}{f'[{lo:+.4f},{hi:+.4f}]':>22}")

    print("\n" + "=" * 96)
    print("【4】年別（14/14年で線を超えていたのはここでも再現するか）")
    print("=" * 96)
    ys = w.groupby("year")["d"].mean()
    print("  " + " ".join(f"{y}:{v:+.3f}" for y, v in ys.items()))
    print(f"  → D>0 の年 {int((ys > 0).sum())}/{len(ys)}")

    print("\n" + "=" * 96)
    print("★読み方")
    print(f"  ・D が必要量 {NEED:.4f} を超えたときだけ儲かる（ケリーが上限）。")
    print("  ・超えなくても、**100円を除いても残るD**は『市場の値付けの甘さ』として実在する。")
    print("  ・100円だけで説明が付くなら、それは**制度の非対称**であって予測の話ではない。")
    print("    その場合の運用は『人気馬の複勝を買う』だけになるが、**元本保証に近く増えもしない**。")


if __name__ == "__main__":
    main()
