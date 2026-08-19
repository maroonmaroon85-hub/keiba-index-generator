"""(106) ★★★(105)の裾を詰める — 「期待払戻が小さい組ほど甘い」が単調なら、**極端まで行けばどこまで行くか**。

★(105)で確定したこと
　「その組の期待払戻が小さいほどROIが線を上回る」が**7券種すべてで単調**（ρ −1.000〜−0.371）。
　6分位の最下位区分で:
　　三連複 上位3頭（期待払戻400円）**88.7%・線+13.71pt**
　　複勝 1番人気（期待払戻92円）**91.2%・線+11.16pt**
　単調なら**もっと極端な裾ではもっと甘い**はず。6分位で止める理由が無い。

★★なぜ三連複が本命か
　1. **複勝には天井がある**（(94)③）。最低配当100円に張り付くので `的中率×配当` が100%を超えられない。
　2. **三連複にその天井は無い**。期待払戻400円は下限から遠く、床に当たらない。
　3. しかも(105)で**複勝と同じ規模（+13.71pt）の甘さ**が出ている。控除率が高い(25%)のに、である。
　→ **上振れの余地が最も大きいのは三連複**。ここが現時点で最も未踏。

★★事前登録（測る前に宣言）
　1. **切り方は発走前に分かる量だけ**: λ補正Harvilleによる組の確率 q（(105)【4】と同じ）。
　　 これは**単勝オッズだけから計算できる**ので事後選択にならない。
　2. **裾の水準**: 上位 20% / 10% / 5% / 2% / 1% の5段階。**後から段を増やさない**。
　3. **判定**: ①線との差が単調に増えること ②最も極端な段でROIが100%を超えるか
　　 ③**前半年代と後半年代で符号が保つこと**（(94)続で最良帯が年代で崩れた前例がある）
　　 ④CIの下端が線を上回ること（点推定で判定しない）
　4. **★多重性**: 7券種 × 5段 = 35セル。**券種は(105)で事前に絞ってある**（三連複と複勝が本命）ので、
　　 判定は**本命2つ**で行い、残りは記述として並べる。
　5. **予想**: 三連複は極端な裾で95%前後まで行くが**100%は超えない**。
　　 複勝は(94)③の天井どおり97〜98%で頭打ち。★外れたらそう書く。
　6. ⚠**点数とコストを必ず出す**。裾を詰めるほど対象レースが減るので、
　　 「年に何レース買えるのか」を出さないと運用の話にならない。

実行: python3 ml/audit_soft_tail.py [開始年(既定2015)]
"""
import math
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "ml")
from audit_crosspool import load_races, payoff, probs, zq
from audit_crosspool2 import realized
from audit_lbs import build_matrix, fit_lambda
from audit_soft_combo import combo_q

LINE = {"複勝": 0.800, "ワイド": 0.775, "馬連": 0.775, "馬単": 0.750,   # ★馬単は(162)で訂正
        "三連複": 0.750, "三連単": 0.725}
TAILS = [0.20, 0.10, 0.05, 0.02, 0.01]        # ★事前登録。後から増やさない
MAIN = ("三連複", "複勝")                       # ★本命はこの2つ（(105)で事前に絞った）


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
        hs = r["horses"]
        p = probs(hs)
        order = np.argsort(-p)
        nums = [hs[k][0] for k in order]
        if len(nums) < 3:
            continue
        num2k = {num: k for k, (num, _, _) in enumerate(hs)}
        l2, l3 = lam[yy]
        cb = {"複勝": (nums[0],), "ワイド": tuple(sorted(nums[:2])),
              "馬連": tuple(sorted(nums[:2])), "馬単": (nums[0], nums[1]),
              "三連複": tuple(sorted(nums[:3])), "三連単": (nums[0], nums[1], nums[2])}
        for kind, c in cb.items():
            v = payoff(r, kind, c)
            if v is None:
                continue
            try:
                q = combo_q(kind, r, p, l2, l3, c, num2k)
            except Exception:
                q = None
            if not q or not (0 < q < 1):
                continue
            rows.append({"kind": kind, "year": yy, "q": q,
                         "exp_pay": LINE[kind] / q * 100,
                         "ret": (v or 0.0) - 100.0, "hit": 1 if (v or 0) > 0 else 0})
    df = pd.DataFrame(rows)
    n_years = df["year"].nunique()
    print(f"(106) 甘い裾をどこまで詰められるか（{y0}年以降・{n_years}年）")
    print("★切り方は単勝オッズだけから計算できる量。事後選択ではない\n")

    for kind in list(MAIN) + [k for k in LINE if k not in MAIN]:
        g = df[df["kind"] == kind]
        if len(g) < 3000:
            continue
        line = LINE[kind] * 100
        star = "★本命" if kind in MAIN else "参考"
        print("=" * 108)
        print(f"{star}【{kind}】線 {line:.1f}%")
        print("=" * 108)
        print(f"{'裾':>6}{'R数':>8}{'年間R数':>9}{'平均期待払戻':>13}{'的中率':>8}"
              f"{'ROI':>8}{'線との差':>10}{'99%CI':>22}{'前半':>8}{'後半':>8}")
        prev = None
        for t in TAILS:
            thr = g["exp_pay"].quantile(t)
            s = g[g["exp_pay"] <= thr]
            if len(s) < 150:
                continue
            m, lo, hi = mci(s["ret"])
            half = s["year"].median()
            a = s[s["year"] <= half]["ret"].mean() + 100
            b = s[s["year"] > half]["ret"].mean() + 100
            mark = ""
            if prev is not None and (m + 100 - line) > prev:
                mark = ""
            prev = m + 100 - line
            print(f"{int(t*100):>5}%{len(s):>8,}{len(s)/n_years:>9.0f}"
                  f"{s['exp_pay'].mean():>12.0f}円{s['hit'].mean()*100:>7.1f}%"
                  f"{(m+100):>7.1f}%{(m+100)-line:>+9.2f}pt"
                  f"{f'[{lo+100:.1f},{hi+100:.1f}]':>22}{a:>7.1f}%{b:>7.1f}%{mark}")
        # 単調性
        vals = []
        for i, t in enumerate(TAILS):
            thr = g["exp_pay"].quantile(t)
            s = g[g["exp_pay"] <= thr]
            if len(s) >= 150:
                vals.append((i, s["ret"].mean() + 100 - line))
        if len(vals) >= 4:
            a = pd.DataFrame(vals, columns=["i", "d"])
            print(f"   裾を詰めるほど甘くなるか: ρ={a['i'].corr(a['d'], method='spearman'):+.3f}"
                  "（**正なら詰めるほど甘い**）")

    print("\n" + "=" * 108)
    print("★読み方")
    print("  ・**CIの下端が100%を超えて初めて『勝てる』**。点推定が100%を超えただけでは足りない。")
    print("  ・前半/後半で符号が変われば(94)続と同じ『選んだから高いだけ』。**必ず両方見る**。")
    print("  ・年間R数が小さすぎると、実際には資金を回せない。**運用可能性はここで決まる**。")
    print("  ・複勝は(94)③の天井（最低配当100円）で97〜98%が上限のはず。三連複にその天井は無い。")


if __name__ == "__main__":
    main()
