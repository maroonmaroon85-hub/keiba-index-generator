"""
三連単マルチの紐の本数を、対比較＋コストの観点で決め直す。

(31)では紐7が点推定の最大(71.7%)だったが、紐5-8は1pt以内に固まっている。
一方で**コストは倍以上違う**（紐5=60点6,000円 / 紐7=126点12,600円）。
ROIが統計的に同じなら**安い方が明確に有利**（同じ資金で賭けられる回数が増え、破産確率が下がる）。

そこで紐3-8を同一レース集合(9頭立て以上)で対比較し、
  ・紐7との差の95%CI（レース単位ブートストラップ・シード平均）
  ・1レースあたりの購入額
  ・的中率
を並べて、どこまで減らせるかを見る。

実行: python3 ml/himo_choice.py [シード数(既定5)]
"""
import sys

import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, "ml")
import features as F
from himo_sweep import POINTS, KEY, hits
from pocket_eval import load_payout_a

TN = "三連単マルチ"
HIMOS = [3, 4, 5, 6, 7, 8]
REF = 7
N_BOOT = 10000


def main():
    n_seed = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    d = F.to_model(F.load_files())
    f = F.build_features(d)
    keep = f["n_prior"] >= 1
    d, f = d[keep].reset_index(drop=True), f[keep].reset_index(drop=True)
    y = (d["finish"] == 1).astype(int).to_numpy()
    cut = d["date"].quantile(0.3)
    tr, te = (d["date"] < cut).to_numpy(), (d["date"] >= cut).to_numpy()
    fx, _ = F.encode_categoricals(f)
    pays = load_payout_a("data/payout/a.csv")
    meta = d.loc[te, ["raceid", "umaban"]].copy()
    print(f"train {tr.sum():,} / test {te.sum():,}  シード{n_seed}本  ブートストラップ{N_BOOT:,}回")

    acc = {n: [] for n in HIMOS}
    for seed in range(n_seed):
        m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=31, subsample=0.8,
                               colsample_bytree=0.8, min_child_samples=100, verbose=-1,
                               random_state=seed)
        m.fit(fx[tr], y[tr], categorical_feature=F.CAT_COLS)
        pr = meta.copy()
        pr["prob"] = m.predict_proba(fx[te])[:, 1]
        rows = {n: [] for n in HIMOS}
        for rid, g in pr.groupby("raceid", sort=False):
            p = pays.get(rid)
            if p is None or not p["sanrentan"] or len(g) < max(HIMOS) + 1:
                continue
            g = g.sort_values("prob", ascending=False, kind="mergesort")
            nums = g["umaban"].astype(int).tolist()
            for n in HIMOS:
                rows[n].append(hits(p[KEY[TN]], nums[0], set(nums[1:n + 1])))
        for n in HIMOS:
            acc[n].append(np.array(rows[n], float))
        print(f"  seed {seed} 完了")

    pay = {n: np.mean(acc[n], axis=0) for n in HIMOS}   # シード平均した払戻(円)
    R = len(pay[REF])
    rng = np.random.default_rng(0)
    idx = rng.integers(0, R, size=(N_BOOT, R))
    ref_roi = pay[REF] / (POINTS[TN](REF) * 100) * 100

    print(f"\n対象 {R:,}R（{max(HIMOS)+1}頭立て以上に統一）")
    print("=" * 88)
    print(f"{'紐':>3}{'点':>5}{'購入額':>9}{'的中率':>8}{'ROI':>8}"
          f"{'紐7との差':>10}{'95%CI':>18}  判定")
    for n in HIMOS:
        roi = pay[n] / (POINTS[TN](n) * 100) * 100
        hit = (pay[n] > 0).mean() * 100
        if n == REF:
            print(f"{n:>3}{POINTS[TN](n):>5}{POINTS[TN](n)*100:>8,}円{hit:>7.1f}%{roi.mean():>7.1f}%"
                  f"{'—':>10}{'（基準）':>18}")
            continue
        dif = roi - ref_roi
        b = dif[idx].mean(axis=1)
        lo, hi = np.percentile(b, 2.5), np.percentile(b, 97.5)
        v = "紐7と差なし" if lo <= 0 <= hi else ("紐7より良い" if lo > 0 else "紐7より悪い")
        print(f"{n:>3}{POINTS[TN](n):>5}{POINTS[TN](n)*100:>8,}円{hit:>7.1f}%{roi.mean():>7.1f}%"
              f"{dif.mean():>+9.2f}{f'[{lo:+.2f},{hi:+.2f}]':>18}  {v}")

    print("\n■ 1レースあたりの期待損失（購入額 × (100−ROI)%）")
    for n in HIMOS:
        cost = POINTS[TN](n) * 100
        roi = pay[n].mean() / cost * 100
        print(f"  紐{n}: {cost:>6,}円/R × (100−{roi:.1f})% = {cost * (100 - roi) / 100:>6,.0f}円の損/R")
    print("\n※ROIが100%未満なので『安ければ多く買える』は誤り（買うほど損が増える）。")
    print("  正しくは『同じレースを買うなら、ROIが同じで購入額が安い方が損失額が小さい』。")
    print("  紐5-8はROIが統計的に同じなので、**最も安い紐5が合理的な選択**。")


if __name__ == "__main__":
    main()
