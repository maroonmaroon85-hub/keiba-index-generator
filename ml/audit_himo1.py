"""D4: ★★「枠連は紐2ではなく紐1（1点）でよいのでは」を実運用の手続きで確かめる。

`audit_points.py` で2つ出た:
  ① モデルの買い目が1点に潰れたレース(1,884R)で、コストを揃えても人気順に **+9.06pt** 勝つ
  ② ★**全レースで『軸枠×紐枠1』の1点だけ買うと 85.24%**。現行(軸枠×紐枠2・平均194円)は84.86%。
     **対応ありで +0.38pt [−1.44,+2.21] ＝ ROIは誤差の内で同じなのに、コストは194円→100円**。

②は本プロジェクトが**(77)②/(78)②で枠連を本命に選んだのと同じ理屈**（ROIが同じならコストが
低い方が損失が小さい）に、そのまま当てはまる。1レースあたりの期待損失は
  現行 194×(1−0.8486)=29.4円  →  紐1 100×(1−0.8524)=14.8円  ＝ **ほぼ半分**。
しかも **枠連 軸枠×紐枠1 は (77)の8券種29通りに入っていない**（紐2・紐3・上位3頭BOXはある）。

だが `audit_points.py` の数字は**シード3本を平均した1本のモデル**から出ている。
★判定基準1（乱数だけで部分集合ROIは大きく動く）に対する対照が無い。そこで:

  ・**独立したシードごとに**ウォークフォワードを回し、**シード間の幅**を出す
  ・毎回 **人気順で同じ買い方をした対照**を並べる（作法3）
  ・**同じレースでの対応あり比較**で 紐1 − 紐2 を測る（作法5c）
  ・的中率も出す（作法7: この標本量でROI差は読めないことが多い）
  ・①の部分集合も**シードごとに**出して、1,884Rの主張が乱数で消えないか見る（作法1）

実行: python3 ml/audit_himo1.py [シード数(既定5)] [開始年(既定2016)]
"""
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

PAYOUT = "data/payout/a.csv"


def bets(order, n, k):
    """軸枠 × モデル上位k頭の枠（重複除去）。k=1 なら必ず1点。"""
    wa = waku_of(order[0], n)
    return sorted({tuple(sorted((wa, waku_of(h, n)))) for h in order[1:1 + k]})


def roi(cs, w):
    return sum(w.get(c, 0) for c in cs) / (len(cs) * 100.0)


def boot(x, rng, n=4000):
    b = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return np.percentile(b, 2.5) * 100, np.percentile(b, 97.5) * 100


def main():
    n_seed = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    y0 = int(sys.argv[2]) if len(sys.argv) > 2 else 2016
    d, fx = load_cached()
    y = (d["finish"] <= 3).astype(int).to_numpy()
    year = d["date"].dt.year.to_numpy()
    wu = load_wu(PAYOUT)
    years = [yy for yy in range(y0, int(year.max()) + 1) if (year == yy).sum() > 5000]
    print(f"ウォークフォワード（拡張窓＝実運用の手続き）{years[0]}〜{years[-1]}"
          f"・**シードを平均せず1本ずつ**{n_seed}本")

    # seed -> 各買い方の1レース回収倍率
    KEYS = ["紐1(1点)", "紐2(現行)", "紐3"]
    per = {s: {k: [] for k in KEYS} for s in range(n_seed)}
    pop = {k: [] for k in KEYS}
    onept = {s: [] for s in range(n_seed)}      # モデルが1点になったレースのマスク
    meta = []
    for yy in years:
        tr, te = year < yy, year == yy
        sub = d.loc[te, ["raceid", "umaban", "fieldsize", "odds"]].copy()
        for s in range(n_seed):
            sub[f"p{s}"] = (lgb.LGBMClassifier(random_state=100 + s, **PARAMS)
                            .fit(fx[tr], y[tr], categorical_feature=F.CAT_COLS)
                            .predict_proba(fx[te])[:, 1])
        for rid, g in sub.groupby("raceid", sort=False):
            w = wu.get(rid)
            if not (w and w["wakuren"] and len(g) >= 3):
                continue
            n = int(g["fieldsize"].iloc[0])
            uma = g["umaban"].astype(int).to_numpy()
            po = uma[np.argsort(np.log(g["odds"].to_numpy(float)), kind="mergesort")]
            for ki, k in enumerate((1, 2, 3)):
                pop[KEYS[ki]].append(roi(bets(po, n, k), w["wakuren"]))
            for s in range(n_seed):
                mo = uma[np.argsort(-g[f"p{s}"].to_numpy(float), kind="mergesort")]
                for ki, k in enumerate((1, 2, 3)):
                    per[s][KEYS[ki]].append(roi(bets(mo, n, k), w["wakuren"]))
                onept[s].append(len(bets(mo, n, 2)) == 1)
            meta.append(yy)
        print(f"  {yy} 完了")

    rng = np.random.default_rng(0)
    A = {s: {k: np.array(v) for k, v in per[s].items()} for s in range(n_seed)}
    P = {k: np.array(v) for k, v in pop.items()}
    O = {s: np.array(onept[s]) for s in range(n_seed)}
    nR = len(P["紐2(現行)"])
    print(f"\n{'='*104}\n★シードごとの結果（{nR:,}R・人気順は設定に依存しない固定値）")
    print(f"{'シード':<8}{'紐1(1点)':>12}{'紐2(現行)':>12}{'紐3':>10}"
          f"{'紐1−紐2':>11}{'紐1の対人気順':>15}{'紐2の対人気順':>15}{'1点率':>8}")
    for s in range(n_seed):
        a = A[s]
        print(f"{f'seed{s}':<8}{a['紐1(1点)'].mean()*100:>11.2f}%{a['紐2(現行)'].mean()*100:>11.2f}%"
              f"{a['紐3'].mean()*100:>9.2f}%{(a['紐1(1点)']-a['紐2(現行)']).mean()*100:>+10.2f}pt"
              f"{(a['紐1(1点)']-P['紐1(1点)']).mean()*100:>+14.2f}pt"
              f"{(a['紐2(現行)']-P['紐2(現行)']).mean()*100:>+14.2f}pt{O[s].mean()*100:>7.1f}%")
    print(f"{'人気順':<8}{P['紐1(1点)'].mean()*100:>11.2f}%{P['紐2(現行)'].mean()*100:>11.2f}%"
          f"{P['紐3'].mean()*100:>9.2f}%")
    for k in KEYS:
        v = np.array([A[s][k].mean() for s in range(n_seed)]) * 100
        print(f"  {k:<10} シード幅 {v.max()-v.min():.2f}pt（{v.min():.2f}〜{v.max():.2f}%）")

    print(f"\n★シード平均のモデルで判定（作法5cの対応あり比較・コストも併記）")
    avg = {k: np.mean([A[s][k] for s in range(n_seed)], axis=0) for k in KEYS}
    cost = {"紐1(1点)": 100.0, "紐2(現行)": None, "紐3": None}
    print(f"{'買い方':<14}{'ROI':>9}{'人気順':>9}{'対人気順':>11}{'95%CI':>18}"
          f"{'的中率':>9}{'平均コスト':>11}{'1R期待損失':>12}")
    ccs = {}
    for k, kk in zip(KEYS, (1, 2, 3)):
        c = np.array([len(bets(np.arange(1, 19), 18, kk)) * 100.0])  # 形式上の上限
        ccs[k] = c
    # 実測の平均コスト（人気順・モデルで同じ分布になるので紐2/紐3は実測から出す）
    print(f"  ※紐1は常に100円。紐2/紐3の平均コストは実測（重複除去で点数が減る）")
    for k in KEYS:
        m, q = avg[k], P[k]
        lo, hi = boot(m - q, rng)
        cst = 100.0 if k == "紐1(1点)" else None
        print(f"{k:<14}{m.mean()*100:>8.2f}%{q.mean()*100:>8.2f}%{(m-q).mean()*100:>+10.2f}pt"
              f"{f'[{lo:+.2f},{hi:+.2f}]':>18}{(m>0).mean()*100:>8.2f}%"
              f"{(f'{cst:.1f}円' if cst else '実測'):>10}"
              f"{(f'{cst*(1-m.mean()):.1f}円' if cst else '—'):>12}")
    lo, hi = boot(avg["紐1(1点)"] - avg["紐2(現行)"], rng)
    print(f"\n  ★対応あり: 紐1 − 紐2(現行) = {(avg['紐1(1点)']-avg['紐2(現行)']).mean()*100:+.2f}pt "
          f"[{lo:+.2f},{hi:+.2f}]")
    lo, hi = boot(avg["紐3"] - avg["紐2(現行)"], rng)
    print(f"  　対応あり: 紐3 − 紐2(現行) = {(avg['紐3']-avg['紐2(現行)']).mean()*100:+.2f}pt "
          f"[{lo:+.2f},{hi:+.2f}]")

    print(f"\n★①の部分集合（モデルの紐2が1点に潰れたレース）を**シードごとに**")
    print(f"{'シード':<8}{'R':>8}{'モデル(1点)':>13}{'人気順(1点で揃える)':>21}{'差':>10}{'95%CI':>18}")
    for s in range(n_seed):
        sel = O[s]
        m, q = A[s]["紐1(1点)"][sel], P["紐1(1点)"][sel]
        lo, hi = boot(m - q, rng)
        print(f"{f'seed{s}':<8}{int(sel.sum()):>8,}{m.mean()*100:>12.2f}%{q.mean()*100:>20.2f}%"
              f"{(m-q).mean()*100:>+9.2f}pt{f'[{lo:+.2f},{hi:+.2f}]':>18}")
    v = np.array([(A[s]['紐1(1点)'][O[s]] - P['紐1(1点)'][O[s]]).mean() for s in range(n_seed)]) * 100
    print(f"  シード幅 {v.max()-v.min():.2f}pt（{v.min():+.2f}〜{v.max():+.2f}pt）"
          f"  ★判定基準1: この幅が点推定と同程度なら主張は成立しない")

    print(f"\n★年別（紐1 vs 紐2・シード平均）")
    yr = np.array(meta)
    print(f"{'年':<7}{'R':>7}{'紐1':>9}{'紐2(現行)':>11}{'人気順(紐1)':>13}{'紐1の対人気順':>15}")
    for yy in sorted(set(yr)):
        s = yr == yy
        print(f"{yy:<7}{int(s.sum()):>7,}{avg['紐1(1点)'][s].mean()*100:>8.2f}%"
              f"{avg['紐2(現行)'][s].mean()*100:>10.2f}%{P['紐1(1点)'][s].mean()*100:>12.2f}%"
              f"{(avg['紐1(1点)'][s]-P['紐1(1点)'][s]).mean()*100:>+14.2f}pt")


if __name__ == "__main__":
    main()
