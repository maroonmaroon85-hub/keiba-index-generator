"""
(40)で「点推定はプラスだがCIが0をまたぐ」となった2主張を煮詰める。

  A. 三連単マルチ×紐7 で **モデル順 > 人気順**（+5.09pt）
  B. 三連単マルチで **紐が多い方が良い**（紐7−紐4 = +5.15pt）

(40)の限界: レース数28,460が固定で、シードを増やしても誤差は縮まなかった。
そこで**推論の精度**を上げる方向で詰める。

  1. **ブートストラップ**: 配当の裾が極端に重い（中央値1万円 / 最大489万円）ので正規近似のCIは信用できない。
     レース単位の対ブートストラップで差のCIを出し直す。
  2. **分解**: ROI = 的中率 × 的中時の平均配当 / 購入額。**的中率は二項分布なので非常に精密に測れる**
     （28,460Rなら±0.5pt程度）。どちらの成分に効果があるのかを切り分ける。
  3. **配当に上限（winsorize）**: 大当たり1本に支配される問題を外し「普通のレースで効いているか」を見る。
     上限を変えて効果が消えるか残るかを追う。
  4. **シード平均**: レース標本誤差は全シード共通なので、シード方向は平均して潰す
     （モデル乱数の分だけノイズが減り、レース誤差はそのまま残る）。

実行: python3 ml/refine_two.py [シード数(既定5)]
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
N_BOOT = 10000


def boot_ci(diff, rng, n_boot=N_BOOT):
    """レース単位の対ブートストラップ。diff は1レースあたりの差(%)。"""
    n = len(diff)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = diff[idx].mean(axis=1)
    return np.percentile(means, 2.5), np.percentile(means, 97.5), float((means > 0).mean())


def report(name, diff, rng):
    lo, hi, p_pos = boot_ci(diff, rng)
    sig = "★差あり" if lo > 0 or hi < 0 else "0をまたぐ"
    print(f"  {name:<28}{diff.mean():>+7.2f}pt  95%CI[{lo:+6.2f},{hi:+6.2f}]  P(>0)={p_pos:.3f}  {sig}")


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
    meta = d.loc[te, ["raceid", "umaban", "odds"]].copy()
    print(f"train {tr.sum():,} / test {te.sum():,}  シード{n_seed}本  ブートストラップ{N_BOOT:,}回")

    market = {}
    for rid, g in meta.groupby("raceid", sort=False):
        g = g.sort_values("odds", ascending=True, kind="mergesort")
        market[rid] = g["umaban"].astype(int).tolist()

    # 生の払戻額（円）をレース×手法で貯める。シード方向は後で平均。
    keys = ["ml7", "ml4", "mk7"]
    acc = {k: [] for k in keys}
    rid_list = None
    for seed in range(n_seed):
        m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=31, subsample=0.8,
                               colsample_bytree=0.8, min_child_samples=100, verbose=-1,
                               random_state=seed)
        m.fit(fx[tr], y[tr], categorical_feature=F.CAT_COLS)
        pr = meta.copy()
        pr["prob"] = m.predict_proba(fx[te])[:, 1]
        rows, rids = {k: [] for k in keys}, []
        for rid, g in pr.groupby("raceid", sort=False):
            p = pays.get(rid)
            if p is None or not p["sanrentan"] or len(g) < 9:
                continue
            g = g.sort_values("prob", ascending=False, kind="mergesort")
            ml = g["umaban"].astype(int).tolist()
            mk = market[rid]
            rows["ml7"].append(hits(p[KEY[TN]], ml[0], set(ml[1:8])))
            rows["ml4"].append(hits(p[KEY[TN]], ml[0], set(ml[1:5])))
            rows["mk7"].append(hits(p[KEY[TN]], mk[0], set(mk[1:8])))
            rids.append(rid)
        for k in keys:
            acc[k].append(np.array(rows[k], float))
        rid_list = rids
        print(f"  seed {seed} 完了")

    R = len(rid_list)
    pay = {k: np.mean(acc[k], axis=0) for k in keys}       # シード平均した払戻(円)
    cost7, cost4 = POINTS[TN](7) * 100, POINTS[TN](4) * 100
    rng = np.random.default_rng(0)

    print(f"\n対象 {R:,}R   紐7=126点/12,600円  紐4=36点/3,600円")
    print("\n" + "=" * 78)
    print("【1】ブートストラップによる差のCI（シード平均・対比較）")
    print("=" * 78)
    a = pay["ml7"] / cost7 * 100
    b = pay["mk7"] / cost7 * 100
    c = pay["ml4"] / cost4 * 100
    print(f"  基準: モデル紐7 {a.mean():.1f}%  人気紐7 {b.mean():.1f}%  モデル紐4 {c.mean():.1f}%")
    report("A: モデル紐7 − 人気紐7", a - b, rng)
    report("B: モデル紐7 − モデル紐4", a - c, rng)

    print("\n" + "=" * 78)
    print("【2】分解: 的中率と的中時の平均配当（どちらに効果があるか）")
    print("=" * 78)
    print(f"{'手法':<14}{'的中率':>9}{'±':>7}{'的中時の平均配当':>16}{'中央値':>11}")
    for k, cost in [("ml7", cost7), ("mk7", cost7), ("ml4", cost4)]:
        h = (pay[k] > 0)
        se = 1.96 * np.sqrt(h.mean() * (1 - h.mean()) / R) * 100
        nz = pay[k][h]
        print(f"{k:<14}{h.mean()*100:>8.2f}%{se:>7.2f}{nz.mean():>15,.0f}円{np.median(nz):>10,.0f}円")
    h1, h2 = (pay["ml7"] > 0), (pay["mk7"] > 0)
    dh = (h1.astype(float) - h2.astype(float)) * 100
    report("的中率の差 モデル−人気(紐7)", dh, rng)

    print("\n" + "=" * 78)
    print("【3】配当に上限を設けた場合（大当たり依存を外す）")
    print("=" * 78)
    print(f"{'上限':<12}{'モデル紐7':>10}{'人気紐7':>10}{'差':>9}{'95%CI':>20}")
    for cap in [50_000, 100_000, 300_000, 1_000_000, 10_000_000]:
        aa = np.minimum(pay["ml7"], cap) / cost7 * 100
        bb = np.minimum(pay["mk7"], cap) / cost7 * 100
        lo, hi, _ = boot_ci(aa - bb, rng, 4000)
        mark = " ★" if lo > 0 else ""
        print(f"{cap:>10,}円{aa.mean():>9.1f}%{bb.mean():>9.1f}%{(aa-bb).mean():>+8.2f}"
              f"{f'[{lo:+.2f},{hi:+.2f}]':>20}{mark}")
    print(f"\n{'上限':<12}{'モデル紐7':>10}{'モデル紐4':>10}{'差':>9}{'95%CI':>20}")
    for cap in [50_000, 100_000, 300_000, 1_000_000, 10_000_000]:
        aa = np.minimum(pay["ml7"], cap) / cost7 * 100
        cc = np.minimum(pay["ml4"], cap) / cost4 * 100
        lo, hi, _ = boot_ci(aa - cc, rng, 4000)
        mark = " ★" if lo > 0 else ""
        print(f"{cap:>10,}円{aa.mean():>9.1f}%{cc.mean():>9.1f}%{(aa-cc).mean():>+8.2f}"
              f"{f'[{lo:+.2f},{hi:+.2f}]':>20}{mark}")


if __name__ == "__main__":
    main()
