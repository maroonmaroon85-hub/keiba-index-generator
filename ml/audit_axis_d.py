"""(112) ★★★(111)を **D で測り直す** — 「年を待つ」以外の道。待たずに決着させる。

★(111)が判定不能だった理由
　ROIで測ったから。ROIは**配当の裾に振り回される**ので、656レースでは
　三連複 軸→上位3頭 の99%CIが **[76.0, 116.0]** になり何も区別できなかった。
　→ **同じ問いを D で測れば桁違いに精密**。Dは1レース1標本で、しかも `log(払戻)` なので
　　 裾が圧縮される。(89)③(53)(80)で繰り返し確認してきたとおり。

★★Dで測ると「買い方の探索」が不要になる（これが決定的）
　部分集合 S だけ買うときの1レースあたり対数成長率は **log(払戻率) + E[d | S]**。
　**ケリーが上限**なので、**E[d|S] < 必要量なら、その S の中でどんな買い方をしても儲からない**。
　→ (111)は「三連複の流しが良いのでは」を問うていたが、**そもそも問う必要が無くなる**。
　　 「軸の複勝が甘いレース群」の D を測れば、**その群で可能な最良が一撃で出る**。

★測るもの
　・レース選択: (111)と同じ **1番人気の複勝の期待払戻 E**（λ補正Harville・発走前に計算できる）
　　 裾 20/10/5/2%（★(111)と同じ水準。後から増やさない）
　・各裾で **券種ごとの E[d|S]** を出し、**必要量**（複勝0.2231/枠連0.2549/三連複0.2877 …）と比べる
　・q は(96)の最良形（λ補正した市場のHarville）

★★事前登録（測る前に宣言）
　1. **判定**: E[d|S] の99%CI**下端**が必要量を超えたら「この群は儲かる」。
　　 超えなければ **(111)の三連複96.0%は配当の裾を引いただけ**と確定し、**この筋は閉じる**。
　2. **予想**: どの裾・どの券種でも必要量に**遠く届かない**（E[d|S] は 0.02〜0.05 程度と予想）。
　　 理由: (101)で「レース選択は予測相関0.000〜0.008で閉じている」と出ており、
　　 　　　軸の複勝Eも事前情報の1つに過ぎないから。
　3. **★ただし全体(+0.0182)より上がるかは別問題**。上がるなら「甘さ」は実在する。
　　 (105)(106)(110)(111)がROIで見てきたものを、**Dという精密な物差しで裏取りする**意味がある。
　4. **★プラセボ**: 選択基準 E をレース間でシャッフルし、**同じ裾の大きさで**同じ手続きを踏む。
　　 (110)⑤の教訓どおり、**効果量は「実測−プラセボ」で書く**。
　5. **単調性**も見る（裾を詰めるほど E[d|S] が上がるか）。(111)②で三連複が単調でなかったので、
　　 **Dでも単調でなければ「最良のビンの罠」だったと確定する**。

実行: python3 ml/audit_axis_d.py [開始年(既定2015)]
"""
import math
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "ml")
from audit_crosspool import PAYBACK, load_races, payoff, probs, zq
from audit_crosspool2 import PARTS, PAYKEY, realized
from audit_fuku_lbs import top3_probs
from audit_lbs import build_matrix, fit_lambda, q_of_lbs

TAILS = [1.00, 0.20, 0.10, 0.05, 0.02]      # ★(111)と同じ水準＋全体
RNG = np.random.default_rng(20260807)


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
        a, b, c = realized(r)
        hs = r["horses"]
        num2k = {num: k for k, (num, _, _) in enumerate(hs)}
        if a not in num2k or b not in num2k or (c is not None and c not in num2k):
            continue
        p = probs(hs)
        l2, l3 = lam[yy]
        t3 = top3_probs(p, 1.0, l2, l3)
        k_top = int(np.argmax(p))
        q_axis = float(t3[k_top])
        if not (0 < q_axis < 1):
            continue
        e_axis = PAYBACK["複勝"] / q_axis * 100        # ★選択基準（発走前に分かる）
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
            rows.append({"kind": kind, "rid": r["rid"], "year": yy,
                         "d": d, "e_axis": e_axis})
        # 複勝も測る（(99)の注意: 全体集計なら偏りは無い。★層別は軸1頭に固定しているので
        # 「軸が来なかった場合」が抜ける。だから複勝は**参考**にとどめる）
    df = pd.DataFrame(rows)
    n_years = df["year"].nunique()
    print(f"(112) (111)をDで測り直す（{y0}年以降・{df['rid'].nunique():,}レース・{n_years}年）")
    print("★E[d|S] < 必要量 なら、その群では**どんな買い方をしても儲からない**（ケリーが上限）\n")

    print("=" * 108)
    print("【1】軸の複勝の甘さで選んだ群の D")
    print("=" * 108)
    print(f"{'券種':<8}{'裾':>6}{'R数':>8}{'年間R':>7}{'E[d|S]':>10}{'99%CI':>22}"
          f"{'必要量':>9}{'必要量に対して':>15}{'不足':>10}")
    curves = {}
    for kind in PARTS:
        g0 = df[df["kind"] == kind]
        if len(g0) < 2000:
            continue
        need = -math.log(PAYBACK[kind])
        vals = []
        for t in TAILS:
            thr = g0["e_axis"].quantile(t)
            s = g0[g0["e_axis"] <= thr]
            if len(s) < 150:
                continue
            m, lo, hi = mci(s["d"])
            vals.append((t, m))
            mark = "★届く" if lo > need else ""
            print(f"{kind:<8}{int(t*100):>5}%{len(s):>8,}{len(s)/n_years:>7.0f}{m:>+10.4f}"
                  f"{f'[{lo:+.4f},{hi:+.4f}]':>22}{need:>9.4f}"
                  f"{max(m,0)/need*100:>14.1f}%{need-m:>10.4f}{mark}")
        curves[kind] = vals
        print()

    print("=" * 108)
    print("【2】単調性 — 裾を詰めるほど D は上がるか")
    print("=" * 108)
    for kind, vals in curves.items():
        if len(vals) < 4:
            continue
        a_ = pd.DataFrame(vals, columns=["t", "d"])
        rho = a_["t"].corr(a_["d"], method="spearman")
        print(f"  {kind:<8} ρ={rho:+.3f}"
              f"（**負なら裾を詰めるほど甘い**）  " +
              " → ".join(f"{v:+.4f}" for _, v in vals))

    print(f"\n{'='*108}")
    print("【3】★プラセボ — 選択基準Eをレース間でシャッフル×20回（効果量は実測−プラセボ）")
    print("=" * 108)
    print(f"{'券種':<8}{'裾':>6}{'実測 E[d|S]':>13}{'プラセボ':>11}{'実測−プラセボ':>15}")
    for kind in PARTS:
        g0 = df[df["kind"] == kind].copy()
        if len(g0) < 2000:
            continue
        for t in (0.10, 0.02):
            thr = g0["e_axis"].quantile(t)
            real = g0[g0["e_axis"] <= thr]["d"].mean()
            pl = []
            for _ in range(20):
                e = RNG.permutation(g0["e_axis"].to_numpy())
                pl.append(g0["d"].to_numpy()[e <= thr].mean())
            pv = float(np.mean(pl))
            print(f"{kind:<8}{int(t*100):>5}%{real:>+13.4f}{pv:>+11.4f}{real-pv:>+15.4f}")

    print(f"\n{'='*108}")
    print("★読み方")
    print("  ・**CIの下端が必要量を超えて初めて『この群は儲かる』**。点推定では判定しない。")
    print("  ・超えなければ、(111)の三連複96.0%は**配当の裾を引いただけ**と確定し、この筋は閉じる。")
    print("    ★Dで測れば656レースでもCIは±0.03程度なので、**年を待たずに決着する**。")
    print("  ・単調でなければ(111)②と同じ『最良のビンの罠』。**Dでも同じ形が出るかを見る**。")
    print("  ・全体(+0.0182)より上がっていれば『甘さ』自体は実在する。それは記述として残る。")


if __name__ == "__main__":
    main()
