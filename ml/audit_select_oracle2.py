"""(114) ★★★(101)と(112)の食い違いを決着させる — **矛盾なのか、測り方の違いなのか**。

★何が食い違っているか
　(101): 事前情報から d を予測する相関は **0.000〜0.008**。上位10%を選んでも必要量に届かない。
　　　　 → 「**レース選択という手段は閉じている**」と書いた。
　(112): 軸の複勝の期待払戻 E で **2%裾**を切ると、枠連の E[d|S] が
　　　　 全体 +0.0182 → **+0.0394**（単調・プラセボ通過・年分割で安定）。
　　　　 → 「**レース選択が本プロジェクト最大の利得**」と書いた。
　**両方を放置していた**。どちらかが間違っているか、両立する説明があるかを決める。

★★仮説（測る前に宣言する）
　**矛盾していない。原因は2つで、どちらも(101)側の測り方の問題**と予想する。
　　**H1 薄まり**: (101)は**上位10%**で選んでいる。(112)の効果は**2%に集中**している。
　　　　10%まで広げれば、効果の無い8%で薄まって消える。
　　**H2 特徴の欠落**: (101)の特徴に**軸のλ補正3着以内確率**が無い。p1/ent/hhi はあるが、
　　　　これは頭数依存の非線形量で、木が p1 から再構成できるとは限らない。
　→ **両方を足せば(101)の枠組みでも(112)の裾が見えるはず**。見えれば「(101)の結論は
　　 『10%では選べない』であって『選べない』ではなかった」と書き直せる。

★★事前登録
　1. 選ぶ深さは **10% / 5% / 2%** の3つ。**(112)と同じ2%を必ず含める**（後から足さない）。
　2. 特徴は(101)のものに **axis_top3（軸のλ補正3着以内確率）と axis_E（その期待払戻）** を足す。
　3. **比較対象を3つ並べる**:
　　 ・**ルール**: (112)そのもの＝ axis_E の小さい順に選ぶ（学習しない・発走前に決まる）
　　 ・**学習**: ウォークフォワードで d を予測して上位を選ぶ（実運用で可能な最善）
　　 ・**完全オラクル**: d そのものの上位（到達不可能な天井）
　4. **プラセボ**: 予測値をシャッフルして同じ手続き。効果量は「実測−プラセボ」で書く。
　5. **判定**: どれかが必要量（枠連0.2549 / 複勝0.2231 …）を超えれば道は開く。
　　 **予想は「1つも超えない」**。(112)の+0.0394は必要量の15.5%でしかなく、
　　 H1H2を直しても桁が足りないと予想する。**この実験は"開く"ためでなく"整合させる"ため**。

実行: python3 ml/audit_select_oracle2.py [開始年(既定2015)]
"""
import math
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, "ml")
from audit_crosspool import PAYBACK, load_races, payoff, probs, zq
from audit_crosspool2 import PARTS, PAYKEY, realized
from audit_lbs import build_matrix, fit_lambda, q_of_lbs
from audit_select_oracle import pre_race_features
import soft_axis as SA

FRACS = [0.10, 0.05, 0.02]          # ★先に宣言。(112)と同じ2%を必ず含める
RNG = np.random.default_rng(20260808)


def mci(x, alpha=0.01):
    x = np.asarray(x, float)
    if len(x) < 2:
        return float("nan"), float("nan"), float("nan")
    m = x.mean()
    se = x.std(ddof=1) / math.sqrt(len(x))
    z = zq(alpha)
    return m, m - z * se, m + z * se


def axis_features(r):
    """★(101)に**足りなかった**特徴。(112)の選択基準そのもの。発走前に計算できる。"""
    odds = [o for _, o, _ in r["horses"]]
    k, e, q = SA.axis_expect(odds)
    if k is None:
        return None
    return {"axis_top3": q, "axis_E": e}


def build(y0):
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
        if yy < y0 or not lam.get(yy):
            continue
        rl = realized(r)
        if rl is None:
            continue
        a, b, c = rl
        num2k = {num: k for k, (num, _, _) in enumerate(r["horses"])}
        if a not in num2k or b not in num2k or (c is not None and c not in num2k):
            continue
        ax = axis_features(r)
        if ax is None:
            continue
        p = probs(r["horses"])
        l2, l3 = lam[yy]
        f = dict(pre_race_features(r, p), **ax)
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
            rows.append(dict(f, kind=kind, d=d))
    return pd.DataFrame(rows)


def topmean(y, score, frac):
    """score の**大きい順**に frac だけ選んだときの y の平均と99%CI。"""
    k = max(int(len(y) * frac), 1)
    idx = np.argsort(-score)[:k]
    return mci(y[idx])


def main():
    y0 = int(sys.argv[1]) if len(sys.argv) > 1 else 2015
    df = build(y0)
    print(f"(114) (101)と(112)の食い違いを決着させる（{y0}年以降・{len(df):,}件）")
    print("★仮説: 矛盾ではなく (H1)10%では薄まる (H2)軸のλ補正top3が特徴に無かった\n")

    import lightgbm as lgb
    FEAT = [c for c in df.columns if c not in ("kind", "d")]
    FEAT_OLD = [c for c in FEAT if c not in ("axis_top3", "axis_E")]
    summary = []

    for kind in PARTS:
        g = df[df["kind"] == kind].reset_index(drop=True)
        if len(g) < 2000:
            continue
        need = -math.log(PAYBACK[kind])
        yrs_ = g["year"].to_numpy()
        y = g["d"].to_numpy()
        # 学習（新旧2つの特徴集合。**H2を分離するため**）
        preds = {}
        for tag, feats in (("新", FEAT), ("旧", FEAT_OLD)):
            pr = np.full(len(g), np.nan)
            for yy in sorted(set(yrs_.tolist())):
                tr, te = yrs_ < yy, yrs_ == yy
                if tr.sum() < 3000:
                    continue
                m = lgb.LGBMRegressor(n_estimators=300, num_leaves=31,
                                      learning_rate=0.05, verbose=-1)
                m.fit(g.loc[tr, feats], y[tr])
                pr[te] = m.predict(g.loc[te, feats])
            preds[tag] = pr
        ok = ~np.isnan(preds["新"])
        yy_ = y[ok]
        # ルール: axis_E は**小さいほど良い**ので符号を反転してスコアにする
        rule = -g.loc[ok, "axis_E"].to_numpy()
        pl = RNG.permutation(preds["新"][ok])

        print(f"── {kind}（必要量 {need:.4f}・{len(yy_):,}件・全体 {yy_.mean():+.4f}）")
        print(f"{'選ぶ深さ':>9}{'ルール(112)':>14}{'学習・新':>12}{'学習・旧':>12}"
              f"{'プラセボ':>11}{'完全オラクル':>13}")
        for fr in FRACS:
            r_m, r_lo, _ = topmean(yy_, rule, fr)
            n_m, n_lo, _ = topmean(yy_, preds["新"][ok], fr)
            o_m, _, _ = topmean(yy_, preds["旧"][ok], fr)
            p_m, _, _ = topmean(yy_, pl, fr)
            c_m, _, _ = topmean(yy_, yy_, fr)
            star = "★" if max(r_lo, n_lo) >= need else " "
            print(f"{fr:>8.0%}{r_m:>+14.4f}{n_m:>+12.4f}{o_m:>+12.4f}"
                  f"{p_m:>+11.4f}{c_m:>+13.4f} {star}")
            if fr == 0.02:
                r_hi = topmean(yy_, rule, fr)[2]
                summary.append((kind, need, r_m, r_lo, r_hi, p_m, c_m,
                                float(yy_.mean())))
        print()

    print("=" * 96)
    print("★深さ2%・ルール(112) の確定値（**主張はこの行だけ**）")
    print(f"{'券種':<8}{'全体':>10}{'2%裾':>10}{'99%CI':>20}{'プラセボ':>10}"
          f"{'実測−プラセボ':>14}{'必要量':>9}{'必要量の':>10}")
    for kind, need, m, lo, hi, pm, cm, all_m in summary:
        print(f"{kind:<8}{all_m:>+10.4f}{m:>+10.4f}"
              f"{'[' + format(lo, '+.4f') + ',' + format(hi, '+.4f') + ']':>20}"
              f"{pm:>+10.4f}{m - pm:>+14.4f}{need:>9.4f}{(m - pm) / need:>9.1%}")
    print("\n★読み方")
    print("  ・**ルール(112)** が深さ2%で全体より高いのに、**学習・旧**が低ければ H2（特徴の欠落）が原因。")
    print("  ・深さ10%で消えて2%で出るなら H1（薄まり）が原因。**(101)の結論は書き直す**。")
    print("  ・★が1つも付かなければ、**整合はしたが道は開いていない**。(112)の結論は変わらない。")
    print("  ・プラセボ列は同じ深さで選んだときの偶然の水準。効果量は「実測−プラセボ」で読むこと。")


if __name__ == "__main__":
    main()
