"""(99) ★★複勝プールの D を **λ補正**で測り直す — (92)①は歪んだ変換で測っていた。

★なぜここが一番効くか
　(92)は複勝プールの D を **Harville の3着以内確率**で測って「全体でほぼゼロ（−0.0021）」とした。
　だが(96)で **Harville は人気馬の2・3着確率を系統的に過大評価する**（λ2≈0.85 / λ3≈0.72）と確定した。
　複勝は**まさにその3着以内確率そのもの**を使う券種なので、**(92)①の値は最も歪みを受けている**。
　しかも:
　　・**必要量が全券種で最小**（|log(0.8)| = 0.2231）
　　・(92)②(94)③で**短いオッズ帯のROIが実際に線を超えている**（1.0-1.6倍で93〜95%）
　→ **λ補正で複勝のDがどこまで動くか**が、このプロジェクトに残った最も意味のある数字。

★測るもの（(96)(97)と同じ3パラメータ、市場側だけ）
　　1着: p^τ を正規化 ／ 2着: p^λ2 ／ 3着: p^λ3 （τ=λ2=λ3=1 が Harville）
　　d = log q_top3 + log(払戻/100) − log(0.8)
　各年のパラメータは**それ以前の年だけ**で最尤推定する。

★★事前登録（測る前に宣言）
　1. **予想する向き**: λ3<1 なので**人気馬の3着以内確率は下方修正**される。
　　 複勝の的中は人気馬に偏るので、**Dは下がる（より負になる）可能性が高い**。
　　 ★つまりこれは**自分に不利な方向を測る検査**。都合の良い方向を探しているのではない。
　2. **判定**: 補正後のDが必要量 0.2231 を超えたら儲かる。**超えないなら (92)①の結論は変わらない**
　　 （複勝プール全体としては甘くない）。
　3. **オッズ帯別も出す**。(92)②で甘さは短いオッズ帯に集中していた。
　　 **帯別のDが必要量を超える帯があるか**が本題。★ただし帯の選択は事後になるので、
　　 　**(94)④と同じく「事前に使える人気順位」でも切って両方出す**。
　4. **上界の意味**: 複勝のDが 0.2231 を超える帯があれば、**その帯ではケリーで増やせる**。
　　 (94)③の「ROIの天井は98%」と食い違ったら、**どちらかが間違っている**ので必ず突き合わせる。
　　 ★(94)は実測ROI、こちらは対数スコア。**同じ帯で両方を並べて出す**。

実行: python3 ml/audit_fuku_lbs.py [開始年(既定2015)]
"""
import math
import sys

import numpy as np

sys.path.insert(0, "ml")
from audit_crosspool import PAYBACK, load_races, probs, zq
from audit_crosspool2 import realized
from audit_lbs_model import fit_walk, build

NEED = -math.log(PAYBACK["複勝"])          # 0.2231


def top3_probs(p, tau, l2, l3):
    """各馬の3着以内確率。τ=λ2=λ3=1 で Harville に一致する。O(n^2)。"""
    w1 = p ** tau
    w1 = w1 / w1.sum()
    w2, w3 = p ** l2, p ** l3
    W2, W3 = w2.sum(), w3.sum()
    d1 = W2 - w2                                   # 各馬が1着のときの2着の分母
    ok1 = d1 > 1e-12
    # 2着: q2_i = w2_i * Σ_{x≠i} w1_x / (W2 - w2_x)
    ratio = np.where(ok1, w1 / np.where(ok1, d1, 1.0), 0.0)
    q2 = w2 * (ratio.sum() - ratio)
    # 3着: A[x,y] = w1_x * w2_y/(W2-w2_x) を x≠y で作り、分母 W3-w3_x-w3_y で割る
    A = np.where(ok1, w1 / np.where(ok1, d1, 1.0), 0.0)[:, None] * w2[None, :]
    np.fill_diagonal(A, 0.0)
    den = W3 - w3[:, None] - w3[None, :]
    C = np.where(den > 1e-12, A / np.where(den > 1e-12, den, 1.0), 0.0)
    np.fill_diagonal(C, 0.0)
    q3 = w3 * (C.sum() - C.sum(axis=0) - C.sum(axis=1))
    return np.clip(w1 + q2 + q3, 1e-12, 1 - 1e-12)


def mci(x, alpha=0.01):
    x = np.asarray(x, float)
    m = x.mean()
    se = x.std(ddof=1) / math.sqrt(len(x))
    z = zq(alpha)
    return m, m - z * se, m + z * se


def main():
    y0 = int(sys.argv[1]) if len(sys.argv) > 1 else 2015
    races = load_races()
    P, i1, i2, i3, yrs = build(races, y0, None)
    par = fit_walk(P, i1, i2, i3, yrs)
    got = {k: v for k, v in par.items() if v}
    arr = np.array(list(got.values()))
    print(f"(99) 複勝プールの D を λ補正で測り直す（{y0}年以降）")
    print(f"  τ {arr[:,0].min():.3f}〜{arr[:,0].max():.3f} / "
          f"λ2 {arr[:,1].min():.3f}〜{arr[:,1].max():.3f} / "
          f"λ3 {arr[:,2].min():.3f}〜{arr[:,2].max():.3f}（各年それ以前の年だけで推定）")
    print(f"★必要量 |log(0.8)| = {NEED:.4f}（全券種で最小）\n")

    rows = []
    for r in races:
        yy = r["year"]
        if yy < y0 or yy not in got or not r["fuku"]:
            continue
        if realized(r) is None:
            continue
        hs = r["horses"]
        p = probs(hs)
        tau, l2, l3 = got[yy]
        th = top3_probs(p, 1.0, 1.0, 1.0)         # Harville
        tl = top3_probs(p, tau, l2, l3)           # λ補正
        num2k = {num: k for k, (num, _, _) in enumerate(hs)}
        pop = {num: i + 1 for i, num in
               enumerate([hs[k][0] for k in np.argsort([o for _, o, _ in hs])])}
        for key, v in r["fuku"].items():
            num = key[0] if isinstance(key, tuple) else key
            k = num2k.get(int(num))
            if k is None or not v or v <= 0:
                continue
            lp = math.log((v + 5) / 100.0) - math.log(PAYBACK["複勝"])
            rows.append({"dh": math.log(th[k]) + lp, "dl": math.log(tl[k]) + lp,
                         "odds": hs[k][1], "pop": pop[hs[k][0]], "year": yy,
                         "qh": th[k], "ql": tl[k]})

    import pandas as pd
    df = pd.DataFrame(rows)
    print("=" * 96)
    print(f"【1】全体（{len(df):,}件）")
    print("=" * 96)
    for col, lab in (("dh", "Harville（(92)①の測り方）"), ("dl", "★λ補正")):
        m, lo, hi = mci(df[col])
        print(f"  {lab:<28} D={m:+.4f} [{lo:+.4f},{hi:+.4f}]  必要量の{max(m,0)/NEED*100:.1f}%")
    g, lo, hi = mci(df["dl"] - df["dh"])
    print(f"  補正による変化: {g:+.4f} [{lo:+.4f},{hi:+.4f}]")
    print("  ※事前に『下がる可能性が高い』と宣言している。上がったならHarvilleは複勝を過小評価していた。")

    print("\n" + "=" * 96)
    print("【2】⚠**無効**: オッズ帯別（**上方バイアスがある**。理由は最後に書いた）")
    print("=" * 96)
    df["band"] = pd.cut(df["odds"], [0, 1.6, 2.0, 3.0, 5.0, 10, 30, 1e9],
                        labels=["〜1.6倍", "1.6-2倍", "2-3倍", "3-5倍", "5-10倍", "10-30倍", "30倍〜"])
    print(f"{'オッズ帯':<10}{'件数':>9}{'D(Harville)':>14}{'D(λ補正)':>12}{'99%CI':>22}"
          f"{'必要量の':>10}{'的中時平均':>11}")
    for b, gg in df.groupby("band", observed=True):
        if len(gg) < 300:
            continue
        m, lo, hi = mci(gg["dl"])
        print(f"{str(b):<10}{len(gg):>9,}{gg['dh'].mean():>+14.4f}{m:>+12.4f}"
              f"{f'[{lo:+.4f},{hi:+.4f}]':>22}{max(m,0)/NEED*100:>9.1f}%"
              f"{gg['ql'].mean()*100:>10.1f}%")
    print("  ※これは**的中した馬だけ**の集計なので、帯の選択は事後。次の【3】が事前に使える形。")

    print("\n" + "=" * 96)
    print("【3】⚠**無効**: 人気順位別（【2】と同じ理由。数字は残すが読まないこと）")
    print("=" * 96)
    print(f"{'人気':<8}{'件数':>9}{'D(Harville)':>14}{'D(λ補正)':>12}{'99%CI':>22}{'必要量の':>10}")
    for pp in range(1, 7):
        gg = df[df["pop"] == pp]
        if len(gg) < 300:
            continue
        m, lo, hi = mci(gg["dl"])
        print(f"{pp}番人気{'':<2}{len(gg):>9,}{gg['dh'].mean():>+14.4f}{m:>+12.4f}"
              f"{f'[{lo:+.4f},{hi:+.4f}]':>22}{max(m,0)/NEED*100:>9.1f}%")

    print("\n" + "=" * 96)
    print("★★結論 — 事前に宣言した『矛盾したら詰める』を実行した結果")
    print("=" * 96)
    print(f"  【2】は〜1.6倍で D=+0.2395＝必要量の107%＝**儲かる**と出た。")
    print("  だが(94)③の**同じ帯の実測ROIは93〜95%**。両立しない。**【2】【3】が誤り**。")
    print("  ★原因: **複勝の払戻は着内に来た馬にしか公表されない**。")
    print("    つまりこの d は『その馬が3着以内に来たレース』でしか作れない。")
    print("    全体（【1】）では『各レースの着内3頭』を数えているので実現値そのもので偏りは無いが、")
    print("    **馬の属性（オッズ帯・人気順）で層別した瞬間、来なかった場合が抜け落ちる**。")
    print("    1番人気は約63%しか来ないので、来た37%…ではなく来た63%だけを見れば当然Dは跳ね上がる。")
    print("  ★**構造的な限界**: 層別した複勝プールのDは、**過去の複勝オッズが無い限り計算できない**。")
    print("    払戻データだけでは『来なかった馬の p_pool』が観測できないため。")
    print("    → 層別で複勝プールを見たいときは**Dではなく実測ROI**を使うこと（(92)②(94)③の方法）。")
    print(f"  ★有効なのは【1】だけ: **λ補正で複勝プール全体の D は {'':s}"
          f"−0.0015 → +0.0048**（必要量の2.1%）。")
    print("    (92)①の『複勝プール全体のDはゼロ』は**符号としては上向きに訂正**されるが、")
    print("    必要量 0.2231 には遠く、**運用は変わらない**。")


if __name__ == "__main__":
    main()
