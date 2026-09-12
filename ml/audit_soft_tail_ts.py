"""(152) ★★★★(106)の「甘い裾」は**張れる時点でも同じ裾か** — **最良の数字の未検査部分**

★★なぜ穴が残っていたか
　**(106)の複勝2%裾 ROI 96.8% [93.9,99.6] はこのプロジェクトで最良の数字**。
　判定基準31は「**順位で買う戦略は時点に頑健／比で買う戦略は時点に脆い**」と分けた。
　⚠**(106)はどちらでもない**——**`期待払戻 = 0.8/q × 100` という連続量の閾値で
　　「どのレースを買うか」を選ぶ**。**レース内の順位ではない**。
　→ ★**qは単勝オッズ全体の関数なので、締切に向けて動く**。**閾値をまたぐレースは入れ替わる**。
　　 **(66)の「当日朝以降なら順位は劣化しない」は、この選択には効かない**（順位の話だから）。
　★**一度も測っていない**。**判定基準29（基準を足したら遡って当て直す）の未処理分**。

★★事前登録（**測る前に書いている**）
　1. **時点は 前日21時 / 当日9時 / 30分前 / ★10分前 / 確定**。**後から増やさない**。
　2. **裾は(106)と同じ 20% / 10% / 5% / 2% / 1%**。**後から増やさない**。
　3. ★**閾値は「全レース(load_races)の確定オッズ」で決めた絶対値を使う**。
　　 **この396レースの中の分位ではない**（標本が小さいので分位が暴れる）。
　　 **同じ絶対閾値を両時点に当てる**ので、**選ばれる本数が変わること自体が結果**。
　4. ★★**陽性対照（判定基準32）**: **(144)が同じデータで「10分前の単勝は確定より
　　 0.0832 nat 悪い」と実測している**。**それを再現するか先に見る**。
　　 → **±0.02 nat 以内で再現しなければ、以下の表は読まない**。
　5. **主判定**: **10分前で選ぶ集合が、確定で選ぶ集合をどれだけ復元するか（再現率）**。
　　 ★**再現率が高ければ(106)は張れる**。**低ければ(106)の96.8%は確定オッズのオラクル**。
　6. ⚠**検出力を先に書く**: **396レースしかない**。**2%裾なら8レース級**。
　　 → **ROIでは判定できない**。**だからROIは出さない**。**再現率と順位相関だけで読む**。
　　 ★**その代わり「期待払戻そのものの動き」を全レースで測る**（396本あるので精度が出る）。
　7. ★**予想**: **当てにしてよい予想は持っていない**（類推はこの4日で5連敗）。
　　 **恒等式から言えることだけ**: **裾を詰めるほど選ばれる本数が減る**ので、
　　 **同じ絶対量の揺れでも入れ替わりの割合は大きくなる**。**向きだけは分かっている**。

★★★実行済みの結果（2026-08-16・360レース）**(106)はオラクルではなかった**
　★①陽性対照: (144)の +0.0832 を **+0.0756** で再現（差 −0.0076・±0.02以内）→ **立った**。

■②確定で選ぶ裾を、その時点でどれだけ復元できるか（**主判定＝★10分前**）
| 時点 | 20%裾 再現/適合 | 10% | 5% | 2% | 1% |
|---|---|---|---|---|---|
| 前日21時 | 54.1/62.5 | 64.1/56.8 | 76.2/50.0 | 61.5/**36.4** | 70.0/43.8 |
| 当日9時 | 54.8/88.5 | 55.8/85.7 | 70.8/81.0 | 78.6/91.7 | 63.6/100 |
| 30分前 | 65.5/84.6 | 65.1/80.0 | 79.2/67.9 | 92.9/76.5 | 63.6/87.5 |
| ★**10分前** | 72.6/88.4 | 76.7/80.5 | 87.5/75.0 | ★**85.7/75.0** | 63.6/100 |

■③期待払戻そのものの動き（比＝確定/その時点）
| 時点 | n | 比の中央値 | \|変化\|>10% | 順位相関ρ |
|---|---|---|---|---|
| 前日21時 | 328 | 0.923 | 60.1% | 0.617 |
| 当日9時 | 360 | 0.895 | 56.4% | 0.806 |
| 30分前 | 360 | 0.933 | 43.1% | 0.892 |
| ★10分前 | 360 | **0.970** | **22.2%** | ★**0.919** |

★★**結論: (106)の96.8%は確定オッズのオラクルではない**。**10分前で裾の86%が見えている**
　（ρ=0.919・期待払戻の中央値のずれ3%）。**判定基準31の「比で買う戦略は時点に脆い」は
　(106)には当てはまらなかった**——**閾値による選択だが、閾値をまたぐ量が10分前でほぼ固まっている**。

⚠★**残る留保（薄めない）**
　・**適合率75%＝買う4本に1本は確定では裾に入らない本**。**その入れ替わりのROI影響は未測定**。
　　**入れ替わるのは必ず境界のレース**なので、**中心の本より甘くない可能性がある**。
　・**2%裾は確定で14本しかない**。**再現率85.7%は12/14**。**±1本で6pt動く**。
　・**ROIは出していない**（事前登録6。396レースでは判定不能）。**「張れる」と「儲かる」は別**。
　★**前日21時は適合率36.4%＝運用メモの「前日22時は誤買いが突出」を独立に再現した**。

実行: python3 ml/audit_soft_tail_ts.py
"""
import math
import sys

import numpy as np

sys.path.insert(0, "ml")
from audit_crosspool import load_races, probs, zq
from audit_crosspool2 import realized
from audit_fuku_lbs import top3_probs
from audit_lbs import build_matrix, fit_lambda
from odds_ts import load_dir, odds_at

PAYBACK = 0.80
TAILS = [0.20, 0.10, 0.05, 0.02, 0.01]
WHEN = [("前日21時", ("prev", 21, 0)), ("当日9時", ("day", 9, 0)),
        ("30分前", ("before", 30)), ("★10分前", ("before", 10)), ("確定", ("final",))]
KNOWN_GAP = 0.0832       # ★(144)の実測。陽性対照の真値


def exp_pay(od, l2, l3):
    """単勝オッズベクトル → (軸の馬index, 期待払戻)。(106)と同じ定義。"""
    od = np.asarray(od, float)
    ok = np.isfinite(od) & (od > 0)
    if ok.sum() < 3:
        return None
    inv = np.where(ok, 1.0 / np.where(ok, od, 1.0), 0.0)
    p = inv / inv.sum()
    q3 = np.asarray(top3_probs(p, 1.0, l2, l3), float)
    j = int(np.argmax(p))
    if not (0 < q3[j] < 1):
        return None
    return j, PAYBACK / q3[j] * 100.0


def main():
    ts = load_dir()
    if not ts:
        sys.exit("data/odds_ts が無い。")
    races = {r["rid"]: r for r in load_races()}
    P, i1, i2, i3, yrs = build_matrix(list(races.values()), 2015)
    ok3 = i3 >= 0
    l2 = fit_lambda(P, i1, i2)
    l3 = fit_lambda(P[ok3], i1[ok3], i2[ok3], stage3=True, ic=i3[ok3])

    # ★事前登録3: 閾値は全レース(確定オッズ)で決めた絶対値
    allv = []
    for r in races.values():
        if realized(r) is None:
            continue
        e = exp_pay([o for _, o, _ in r["horses"]], l2, l3)
        if e:
            allv.append(e[1])
    allv = np.array(allv)
    cuts = {t: float(np.quantile(allv, t)) for t in TAILS}

    ids = sorted(set(ts) & set(races))
    per, gaps = {lab: {} for lab, _ in WHEN}, {lab: [] for lab, _ in WHEN}
    for rid in ids:
        r = races[rid]
        rl = realized(r)
        if rl is None:
            continue
        nums = [u for u, _, _ in r["horses"]]
        win = rl[0]
        if win not in nums:
            continue
        wk = nums.index(win)
        for lab, w in WHEN:
            od = odds_at(ts[rid], w)
            if od is None or len(od) != len(nums):
                continue
            e = exp_pay(od, l2, l3)
            if not e:
                continue
            per[lab][rid] = e[1]
            o = np.asarray(od, float)
            ok = np.isfinite(o) & (o > 0)
            if ok.sum() < 3 or not ok[wk]:
                continue
            inv = np.where(ok, 1.0 / np.where(ok, o, 1.0), 0.0)
            gaps[lab].append(math.log(inv[wk] / inv.sum()))

    z = zq(0.01)
    print(f"(152) (106)の甘い裾は張れる時点でも同じ裾か（{len(per['確定']):,}レース）")
    print(f"　閾値は**全{len(allv):,}レースの確定オッズ**で決めた絶対値: "
          + " / ".join(f"{int(t*100)}%={cuts[t]:.1f}円" for t in TAILS) + "\n")

    print("■ ★★①陽性対照（判定基準32）— (144)の『10分前の単勝は確定より0.0832 nat 悪い』")
    base = np.array(gaps["確定"])
    got = None
    for lab, _ in WHEN:
        v = np.array(gaps[lab])
        if lab == "確定" or len(v) < 2:
            continue
        common = min(len(v), len(base))
        d = base[:common] - v[:common]
        if lab == "★10分前":
            got = float(d.mean())
        print(f"　{lab:>8}  確定との対数スコア差 {d.mean():+.4f}  "
              f"99%CI [{d.mean()-z*d.std(ddof=1)/math.sqrt(len(d)):+.4f},"
              f"{d.mean()+z*d.std(ddof=1)/math.sqrt(len(d)):+.4f}]  n={len(d)}")
    ok = got is not None and abs(got - KNOWN_GAP) <= 0.02
    print(f"　★(144)の実測 {KNOWN_GAP:+.4f} vs 再現 {got:+.4f}　差 {got-KNOWN_GAP:+.4f}")
    print(f"　→ ★**①は立ったか（±0.02 nat以内）: {'★立った' if ok else '⚠立っていない'}**")
    if not ok:
        print("　⚠**立っていない**。**以下の表を読まないこと**（判定基準32）。")

    print("\n■ ★★②主判定 — 確定で選ぶ裾を、その時点でどれだけ復元できるか")
    fin = per["確定"]
    print(f"{'時点':>10}{'裾':>6}{'確定で選ぶ':>10}{'時点で選ぶ':>10}{'両方':>7}"
          f"{'再現率':>9}{'適合率':>9}")
    for lab, _ in WHEN:
        if lab == "確定":
            continue
        cur = per[lab]
        both_ids = set(cur) & set(fin)
        for t in TAILS:
            c = cuts[t]
            A = {i for i in both_ids if fin[i] <= c}
            B = {i for i in both_ids if cur[i] <= c}
            if not A and not B:
                continue
            print(f"{lab:>10}{int(t*100):>5}%{len(A):>10}{len(B):>10}{len(A & B):>7}"
                  f"{len(A & B)/max(len(A),1):>8.1%}{len(A & B)/max(len(B),1):>9.1%}")

    print("\n■ ③期待払戻そのものの動き（396本あるので精度が出る・事前登録6）")
    print(f"{'時点':>10}{'n':>6}{'確定との比の中央値':>20}{'|変化|>10%':>11}{'順位相関ρ':>11}")
    for lab, _ in WHEN:
        if lab == "確定":
            continue
        ids2 = sorted(set(per[lab]) & set(fin))
        if len(ids2) < 5:
            continue
        a = np.array([per[lab][i] for i in ids2])
        b = np.array([fin[i] for i in ids2])
        ra = np.argsort(np.argsort(a)).astype(float)
        rb = np.argsort(np.argsort(b)).astype(float)
        rho = float(np.corrcoef(ra, rb)[0, 1])
        rel = b / a
        print(f"{lab:>10}{len(ids2):>6}{np.median(rel):>20.4f}"
              f"{np.mean(np.abs(rel - 1) > 0.10):>10.1%}{rho:>11.3f}")

    print("\n" + "=" * 92)
    print("★読み方（事前登録のとおり。**後から足していない**）")
    print("  ・**①が立たなければ②③を読まない**（判定基準32）。")
    print("  ・★**主判定は②の「★10分前」の再現率**。**高ければ(106)は張れる**。")
    print("    **低ければ(106)の96.8%は確定オッズのオラクル**＝**運用の根拠にならない**。")
    print("  ⚠**ROIは出していない**。396レースでは2%裾が8本級で判定不能だから（事前登録6）。")
    print("  ⚠**裾を詰めるほど入れ替わりの割合は必ず大きくなる**（本数が減るので）。**向きは既知**。")


if __name__ == "__main__":
    main()
