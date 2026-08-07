"""(111) ★★「複勝で甘いと判定した馬を**軸にして他の券種へ流す**」を測る。

★これまでと何が違うか（**同じことの繰り返しではない**理由を先に書く）
　(105)(106)(110) は「**その組そのものの期待払戻が小さい組**」を **1点買い**で測っていた。
　ここで測るのは違う形:
　　**① レースを「複勝の甘さ」で選び ② その馬を軸に固定して ③ 他の券種へ流す**
　・レース選択の基準が「軸馬の複勝の期待払戻」＝**組ではなく馬**で選ぶ
　・買うのは**1点ではなく流し**（軸×紐複数）＝的中率と配当の組み合わせが変わる
　→ **(110)の走査は「各券種で最小Eの組」を選んでおり、軸を固定していない**。ここは未測定。

★なぜ筋が通るか
　(106)で複勝の天井（96.8%）は**最低配当100円の床**で決まっていた。
　`ROI = 的中率 × 払戻` で、床の中では払戻が100円に固定されるので**上限は的中率そのもの**。
　→ **床から出れば払戻は増える**。同じ「甘い馬」を、床の無い券種（ワイド・馬連・三連複）で
　　 買えば、**的中率を少し落とす代わりに払戻の伸びしろを取れる**可能性がある。
　★(110)④で**枠連は床から遠いのに1%裾でROI 93.0%（線+15.5pt）**と出ているので、
　　「床の外にも甘さは実在する」ことは確認済み。**軸を固定した流しは、その延長線上**。

★★事前登録（測る前に宣言）
　1. **レース選択**: 1番人気の**複勝の期待払戻 E**（λ補正Harvilleの3着以内確率から計算・
　　 **発走前に分かる**）が小さい順に 20/10/5/2% の4水準。★水準を後から増やさない。
　2. **買い方のメニュー**（すべて軸＝1番人気に固定）:
　　 複勝 軸（基準・(106)で96.8%）／ ワイド 軸×2番人気 ／ ワイド 軸→上位3頭(2点)
　　 馬連 軸×2番人気 ／ 馬連 軸→上位3頭(2点) ／ 三連複 軸→上位3頭(1点)
　　 三連複 軸→上位4頭(3点) ／ 枠連 軸枠×2番人気の枠
　3. **判定**: ①ROIが100%を超えるか（**CIの下端**で見る）②裾を詰めるほど上がるか（単調性）
　　 ③**★プラセボ**（払戻を(頭数,年)層内でシャッフル）を必ず並べ、
　　 　 **効果量は「実測−プラセボ」で書く**（(110)⑤の教訓。線との差をそのまま書くと約2倍過大）
　4. **予想**: **どのメニューも100%を超えない**。理由は(89)④の上界——
　　 このレース群の D は 0.02 程度で、必要量 0.2549 には遠い。買い方を変えても上界は動かない。
　　 ★**ただし「複勝より良いメニューがあるか」は別問題**で、そこは開いている。
　　 　 予想としては **ワイド 軸×2番人気が複勝に近づくが超えない**。
　5. **年間R数を必ず出す**。裾を詰めるほど買えるレースが減るので、そこで運用可能性が決まる。

実行: python3 ml/audit_axis_flow.py [開始年(既定2015)]
"""
import itertools
import math
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, "ml")
from audit_crosspool import PAYBACK, load_races, payoff, probs, zq
from audit_crosspool2 import realized
from audit_fuku_lbs import top3_probs
from audit_lbs import build_matrix, fit_lambda
from waku_umatan import waku_of

TAILS = [0.20, 0.10, 0.05, 0.02]          # ★事前登録。後から増やさない
LINE = {"複勝": 0.800, "ワイド": 0.775, "馬連": 0.775, "枠連(人気順)": 0.775,
        "三連複": 0.750}


def mci(x, alpha=0.01):
    x = np.asarray(x, float)
    m = x.mean()
    se = x.std(ddof=1) / math.sqrt(len(x))
    z = zq(alpha)
    return m, m - z * se, m + z * se


def menu_of(r, nums, n):
    """軸＝nums[0] に固定した買い方。値は (払戻キー, [組...])。点数＝len(組)。"""
    w = lambda x: waku_of(x, n)
    return {
        "複勝 軸": ("複勝", [(nums[0],)]),
        "ワイド 軸×2人気": ("ワイド", [tuple(sorted(nums[:2]))]),
        "ワイド 軸→上位3頭": ("ワイド", [tuple(sorted((nums[0], nums[1]))),
                                        tuple(sorted((nums[0], nums[2])))]),
        "馬連 軸×2人気": ("馬連", [tuple(sorted(nums[:2]))]),
        "馬連 軸→上位3頭": ("馬連", [tuple(sorted((nums[0], nums[1]))),
                                    tuple(sorted((nums[0], nums[2])))]),
        "三連複 軸→上位3頭": ("三連複", [tuple(sorted(nums[:3]))]),
        "三連複 軸→上位4頭": ("三連複", [tuple(sorted((nums[0], a, b)))
                                        for a, b in itertools.combinations(nums[1:4], 2)]),
        "枠連 軸枠×2人気枠": ("枠連(人気順)", [tuple(sorted((w(nums[0]), w(nums[1]))))]),
    }


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
        if len(hs) < 5:
            continue
        p = probs(hs)
        order = np.argsort(-p)
        nums = [hs[k][0] for k in order]
        num2k = {num: k for k, (num, _, _) in enumerate(hs)}
        l2, l3 = lam[yy]
        t3 = top3_probs(p, 1.0, l2, l3)
        q_axis = float(t3[num2k[nums[0]]])
        if not (0 < q_axis < 1):
            continue
        # ★レース選択の基準: 軸の複勝の期待払戻（発走前に計算できる）
        e_axis = PAYBACK["複勝"] / q_axis * 100
        row = {"rid": r["rid"], "year": yy, "n": r["n"], "e_axis": e_axis}
        for lab, (key, combos) in menu_of(r, nums, r["n"]).items():
            got = [payoff(r, key, c) for c in combos]
            if any(g is None for g in got):
                row[lab] = np.nan
                continue
            row[lab] = sum(g or 0.0 for g in got) - 100.0 * len(combos)
            row[lab + "|cost"] = 100.0 * len(combos)
            row[lab + "|hit"] = 1 if any((g or 0) > 0 for g in got) else 0
        rows.append(row)
    df = pd.DataFrame(rows)
    n_years = df["year"].nunique()
    print(f"(111) 複勝で甘い馬を軸に他の券種へ流す（{y0}年以降・{len(df):,}レース・{n_years}年）")
    print("★レース選択の基準は「軸の複勝の期待払戻」＝発走前に計算できる量\n")

    labs = [l for l in menu_of(races[0], [1, 2, 3, 4], 16)]
    key_of = {l: k for l, (k, _) in menu_of(races[0], [1, 2, 3, 4], 16).items()}

    for t in TAILS:
        thr = df["e_axis"].quantile(t)
        s = df[df["e_axis"] <= thr]
        print("=" * 112)
        print(f"■ 裾 {int(t*100)}%（軸の複勝期待払戻 ≤ {thr:.0f}円・{len(s):,}R・"
              f"年間{len(s)/n_years:.0f}R）")
        print("=" * 112)
        print(f"{'買い方':<20}{'点数':>5}{'的中率':>8}{'ROI':>8}{'線':>7}"
              f"{'線との差':>10}{'99%CI':>22}")
        for lab in labs:
            g = s[s[lab].notna()]
            if len(g) < 150:
                continue
            cost = g[lab + "|cost"].iloc[0]
            m, lo, hi = mci(g[lab] / cost * 100.0)      # 100円あたりに正規化
            line = LINE[key_of[lab]] * 100
            print(f"{lab:<20}{int(cost/100):>5}{g[lab+'|hit'].mean()*100:>7.1f}%"
                  f"{(m+100):>7.1f}%{line:>6.1f}%{(m+100)-line:>+9.2f}pt"
                  f"{f'[{lo+100:.1f},{hi+100:.1f}]':>22}")
        print()

    # ───────── ★プラセボ（(110)⑤の教訓：効果量は実測−プラセボで書く） ─────────
    print("=" * 112)
    print("★プラセボ — 払戻を(頭数,年)層内でシャッフル×20回。**効果量は「実測−プラセボ」**")
    print("=" * 112)
    rng = np.random.default_rng(0)
    print(f"{'買い方':<20}{'裾':>6}{'実測ROI':>9}{'プラセボ':>10}{'実測−プラセボ':>15}")
    for lab in labs:
        base = df[df[lab].notna()].copy()
        if len(base) < 3000:
            continue
        cost = base[lab + "|cost"].iloc[0]
        for t in (0.10, 0.02):
            thr = base["e_axis"].quantile(t)
            real = base[base["e_axis"] <= thr][lab].mean() / cost * 100.0 + 100
            pl = []
            for _ in range(20):
                g = base.copy()
                g[lab] = g.groupby(["n", "year"])[lab].transform(
                    lambda x: rng.permutation(x.to_numpy()))
                pl.append(g[g["e_axis"] <= thr][lab].mean() / cost * 100.0 + 100)
            pv = float(np.mean(pl))
            print(f"{lab:<20}{int(t*100):>5}%{real:>9.1f}{pv:>10.1f}{real-pv:>+15.2f}pt")

    print("\n" + "=" * 112)
    print("★読み方")
    print("  ・**CIの下端が100%を超えて初めて『勝てる』**。点推定では判定しない。")
    print("  ・(110)⑤より、線との差には『人気を買うこと自体の優位』が混ざる。")
    print("    **裾固有の効果は「実測−プラセボ」**。こちらで比較すること。")
    print("  ・複勝軸(96.8%)を超えるメニューがあるかが本題。無ければ複勝が最良のまま。")
    print("  ・(89)④の上界より、このレース群でも D は0.02程度。**買い方を変えても上界は動かない**。")


if __name__ == "__main__":
    main()
