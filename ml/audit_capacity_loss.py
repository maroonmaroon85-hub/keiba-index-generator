"""(133) ★★(121)が開けたまま残した「(81)の順位主張」を、**ROIではなく1レース期待損失**で閉じる

★何が残っていたか
　(81)は1シードのROIで「**L5がL2より枠連 +2.78pt**・95%CI[+0.07,+5.51]★」と結論した。
　(121)はこれをDで測り直し、**確率の出どころとしてはL2が全w・0/7年で明確に上**と出したが、
　**「(81)を反証した」とは書けない**と明記して閉じた:
　> (121)が測ったのは **qの出どころとしての質**（対数スコア＝較正）。
　> (81)が主張したのは **固定の買い方（軸枠×紐枠2）のROI**＝**順位の質**。
　> 高容量モデルは順位が良くて較正が悪い、という両立はありうる。**別の量である**。
　→ ★**順位の質そのものは、いまも未検証のまま残っている**。

★★なぜ「ROIで測り直す」ではダメか（ここが本題）
　(80)が検出限界を出している: **枠連 軸枠×紐枠2 で 2.20pt**
　（1ptの差を検出するには 155,091R＝**約50年分**）。
　**(81)の +2.78pt・CI下端+0.07 はこの限界のすぐ上**＝もともと測れていない。
　**同じROIで測り直しても同じ限界に当たるだけ**で、何も決着しない。
★→ **(80)が指示した物差しに替える**: **1レース期待損失＝平均コスト − 平均払戻[円]**。
　 同じ比較が **2〜50倍精密**になる((80)④)。**これは(126)と同じ「物差しを疑う」形**。
　 ⚠**ROIは分母（賭け金）を割り算で捨てる**ので、**コストの違う買い方の比較には使えない**。
　 　軸枠×紐枠2 は**軸と紐が同じ枠に入ると1点に減る**ので、**モデルによってコストが変わる**。
　 　★**まさにROIが使えない形をしている**。(81)はここを踏んでいた。

★★事前登録（測る前に宣言する）
　1. 比較は **L2（現行）vs L5** の1組だけ。**梯子の他の段は見ない**（(81)の主張がこの2点だから）。
　2. **判定量は「1レース期待損失の差」**。同じレースで両モデルを走らせるので**対応のある差**で見る
　　 （分散が大きく下がる。★**入れ子ではなく同一レースの対**なので判定基準13に反しない）。
　3. **判定**: 差の99%CIが0を外れ、**L5の期待損失が小さい**なら「(81)の順位主張は生きている」。
　　 0をまたぐ／L5が悪いなら「**(81)の順位主張も否定された**」＝(121)と合わせて**完全に閉じる**。
　4. **ROIも併記する**（(81)との接続のため）。ただし**判定には使わない**。
　5. **コストそのものも報告する**（点数がモデルで変わるのが今回の肝なので）。
　6. **予想**: **差は0をまたぐ**と予想する。理由は(80)の検出限界そのもの——
　　 　+2.78ptが本物なら期待損失でも見えるはずだが、(121)で較正が大きく劣ることが分かっており、
　　 　**順位だけが良いという都合のよい両立**は考えにくい。
　　 ⚠**予想はあてにしない**。(117)(122)(123)(131)で外している。

⚠**実行時間**: L5は leaves255/2000本 で**1年あたり数分**かかる。年ごとにCSVへ保存するので
　途中で落ちても続きから走る（`data/cache/exp_caploss/`）。

実行: python3 ml/audit_capacity_loss.py [シード数(既定1)] [開始年(既定2019)]
"""
import math
import os
import sys
import warnings

import numpy as np
import pandas as pd
import lightgbm as lgb

warnings.filterwarnings("ignore")
sys.path.insert(0, "ml")
import features as F
from _cache import load_cached
from place_wide import PARAMS
from waku_umatan import load_wu, waku_of
from audit_capacity_wf import wakuren_cs

PAYOUT = "data/payout/a.csv"
CACHE = "data/cache/exp_caploss"
PAIR = [("L2", dict(PARAMS)),
        ("L5", dict(PARAMS, num_leaves=255, min_child_samples=10, n_estimators=2000))]


def zq(alpha):
    from statistics import NormalDist
    return NormalDist().inv_cdf(1 - alpha / 2)


def mci(x, alpha=0.01):
    x = np.asarray(x, float)
    if len(x) < 2:
        return float("nan"), float("nan"), float("nan")
    m = x.mean()
    se = x.std(ddof=1) / math.sqrt(len(x))
    z = zq(alpha)
    return m, m - z * se, m + z * se


def run_year(d, fx, y, year, yy, wu, n_seed):
    """1年分を走らせて (レースごとの払戻[円], コスト[円]) をモデル別に返す。

    ★ROIの比ではなく**円のまま**持つ。これをやらないと期待損失が出せない。
    """
    tr, te = year < yy, year == yy
    sub = d.loc[te, ["raceid", "umaban", "fieldsize", "odds"]].copy()
    for nm, par in PAIR:
        ps = [lgb.LGBMClassifier(random_state=s, **par)
              .fit(fx[tr], y[tr], categorical_feature=F.CAT_COLS)
              .predict_proba(fx[te])[:, 1] for s in range(n_seed)]
        sub[nm] = np.mean(ps, axis=0)
    rows = []
    for rid, g0 in sub.groupby("raceid", sort=False):
        w = wu.get(rid)
        if not w or not w["wakuren"] or len(g0) < 3:
            continue
        n = int(g0["fieldsize"].iloc[0])
        uma = g0["umaban"].astype(int).to_numpy()
        rec = {"raceid": rid, "year": yy}
        ok = True
        for nm, _ in PAIR:
            mo = uma[np.argsort(-g0[nm].to_numpy(float), kind="mergesort")]
            cs = wakuren_cs(mo, n)
            if not cs:
                ok = False
                break
            rec[f"pay_{nm}"] = float(sum(w["wakuren"].get(c, 0) for c in cs))
            rec[f"cost_{nm}"] = float(len(cs) * 100)     # ★点数がモデルで変わる
        if ok:
            rows.append(rec)
    return pd.DataFrame(rows)


def main():
    n_seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    y0 = int(sys.argv[2]) if len(sys.argv) > 2 else 2019
    os.makedirs(CACHE, exist_ok=True)

    d, fx = load_cached()
    y = (d["finish"] <= 3).astype(int).to_numpy()
    year = d["date"].dt.year.to_numpy()
    wu = load_wu(PAYOUT)
    years = [yy for yy in range(y0, int(year.max()) + 1) if (year == yy).sum() > 5000]
    print(f"(133) L2 vs L5 を1レース期待損失で比較（{years[0]}〜{years[-1]}・シード{n_seed}本）")
    print("※各年Yは「Yより前の全データ」で学習＝実運用の手続き\n", flush=True)

    parts = []
    for yy in years:
        fp = f"{CACHE}/{yy}.csv"
        if os.path.exists(fp):
            df = pd.read_csv(fp)
            print(f"  {yy} キャッシュから読込 ({len(df)}R)", flush=True)
        else:
            df = run_year(d, fx, y, year, yy, wu, n_seed)
            df.to_csv(fp, index=False)
            print(f"  {yy} 完了 ({len(df)}R)", flush=True)
        parts.append(df)

    a = pd.concat(parts, ignore_index=True)
    print(f"\n対象 {len(a):,}レース")

    print("\n── 参考: ROI と コスト（★判定には使わない） ──")
    print(f"{'モデル':<6}{'ROI':>9}{'平均コスト':>12}{'平均払戻':>12}")
    for nm, _ in PAIR:
        roi = a[f"pay_{nm}"].sum() / a[f"cost_{nm}"].sum() * 100
        print(f"{nm:<6}{roi:>8.2f}%{a[f'cost_{nm}'].mean():>11.1f}円"
              f"{a[f'pay_{nm}'].mean():>11.1f}円")
    dc = a["cost_L5"].mean() - a["cost_L2"].mean()
    print(f"　★コスト差 L5−L2 = {dc:+.2f}円/R"
          f"（{'点数が違うのでROIでは比較できない' if abs(dc) > 0.5 else '点数はほぼ同じ'}）")

    print("\n── ★判定: 1レース期待損失（コスト − 払戻・小さいほど良い） ──")
    loss = {}
    for nm, _ in PAIR:
        v = (a[f"cost_{nm}"] - a[f"pay_{nm}"]).to_numpy(float)
        loss[nm] = v
        m, lo, hi = mci(v)
        print(f"  {nm}  {m:>7.2f}円/R  99%CI[{lo:+.2f},{hi:+.2f}]")

    diff = loss["L5"] - loss["L2"]          # ★同一レースの対応のある差
    m, lo, hi = mci(diff)
    print(f"\n  ★対応のある差 L5 − L2 = {m:+.3f}円/R  99%CI[{lo:+.3f},{hi:+.3f}]")
    if lo > 0:
        v = "★**L5のほうが損失が大きい＝(81)の順位主張は否定された**"
    elif hi < 0:
        v = "★**L5のほうが損失が小さい＝(81)の順位主張は生きている**"
    else:
        v = "**0をまたぐ＝(81)の順位主張は支持されない**（差を検出できない）"
    print(f"  → {v}")

    print("\n── 年別（対応のある差 L5−L2・円/R） ──")
    pos = 0
    ys = sorted(a["year"].unique())
    for yy in ys:
        m2 = a["year"].to_numpy() == yy
        v, lo2, hi2 = mci(diff[m2])
        pos += v > 0
        print(f"  {yy}  n={int(m2.sum()):5d}  {v:+7.2f}円  99%CI[{lo2:+.2f},{hi2:+.2f}]")
    print(f"  → L5のほうが悪い年 {pos}/{len(ys)}")

    print("\n── ★(80)の検出限界との対応 ──")
    sd = diff.std(ddof=1)
    n_need = (zq(0.01) * sd / 1.0) ** 2
    print(f"  期待損失の差の標準偏差 {sd:.1f}円 → 1円/Rの差を99%で検出するのに約 {n_need:,.0f}R")
    print(f"  ※ROIだと枠連は1ptの検出に155,091R＝約50年（(80)）。"
          f"今回は {len(a):,}R で判定できている")


if __name__ == "__main__":
    main()
