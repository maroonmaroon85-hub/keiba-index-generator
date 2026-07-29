"""
ROIを直接の目的関数にしてハイパーパラメータを選べるか検証する。

(52)で「AUCで最適化するとモデルが市場の写しになりROIが下がる」と分かった。
ならばROIで選ぶべきだが、**ROIはAUDと違い測定誤差が極めて大きい**（本プロジェクトの主題）。
内側の検証データでROIを測っても、その値自体がノイズかもしれない。

そこで単に「ROI最良の設定」を選ぶのではなく、**内側検証ROIがテストROIを予測できているか**を確かめる:
  ・学習期間を さらに 内側学習 / 内側検証 に分割（テストには一切触らない）
  ・num_leaves × min_child_samples × 木の本数 のグリッドで、内側検証ROIとテストROIを両方出す
  ・両者の順位相関を見る
     相関が高い → ROIでチューニングできる。最良設定を採用してよい
     相関が低い → ROIでのチューニングは不可能。(52)の「現行が最良」は偶然だったことになる

木の本数は1回の学習から num_iteration を変えて予測することで、追加学習なしに複数点を評価する。

実行: python3 ml/tune_roi.py [シード数(既定3)]
"""
import itertools
import sys
import warnings

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
sys.path.insert(0, "ml")
import features as F
from place_wide import boot
from pocket_eval import load_payout_a

LEAVES = [15, 31, 63]
MIN_CHILD = [30, 100]
N_TREES = [100, 200, 400, 700, 1000]
BASE = dict(learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, verbose=-1)


def roi_of(pred, sub, pays):
    """三連複 軸1×紐3 のROI(%)。sub は raceid/umaban を持つ test 相当のフレーム。"""
    s = sub.copy()
    s["p"] = pred
    tot = cnt = 0
    for rid, g in s.groupby("raceid", sort=False):
        pp = pays.get(rid)
        if pp is None or not pp["sanrenpuku"] or len(g) < 4:
            continue
        t = g.sort_values("p", ascending=False, kind="mergesort")["umaban"].astype(int).tolist()
        cs = [tuple(sorted((t[0], a, b))) for a, b in itertools.combinations(t[1:4], 2)]
        tot += sum(pp["sanrenpuku"].get(c, 0) for c in cs)
        cnt += 1
    return (tot / (cnt * 300.0) * 100 if cnt else float("nan")), cnt


def main():
    n_seed = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    d = F.to_model(F.load_files())
    f = F.build_features(d)
    keep = (f["n_prior"] >= 1) & d["odds"].notna() & (d["odds"] > 0)
    d, f = d[keep].reset_index(drop=True), f[keep].reset_index(drop=True)
    y = (d["finish"] <= 3).astype(int).to_numpy()
    odds = d["odds"].to_numpy(float)
    inv = 1.0 / odds
    mkt = inv / pd.Series(inv).groupby(d["raceid"]).transform("sum").to_numpy()
    fx, _ = F.encode_categoricals(f)
    fx["log_odds"] = np.log(odds)
    fx["mkt_prob"] = mkt

    cut = d["date"].quantile(0.3)
    tr, te = (d["date"] < cut).to_numpy(), (d["date"] >= cut).to_numpy()
    inner = d.loc[tr, "date"].quantile(0.65)          # 学習期間の後半35%を内側検証に
    tr_in, tr_val = tr & (d["date"] < inner), tr & (d["date"] >= inner)
    pays = load_payout_a("data/payout/a.csv")
    sub_val = d.loc[tr_val, ["raceid", "umaban"]].copy()
    sub_te = d.loc[te, ["raceid", "umaban"]].copy()
    _, n_val_races = roi_of(np.zeros(tr_val.sum()), sub_val, pays)
    print(f"内側学習 {tr_in.sum():,} / 内側検証 {tr_val.sum():,}（{n_val_races:,}R） / テスト {te.sum():,}")
    print("※内側検証のレース数が少ないほどROIの測定はノイジーになる。そこが本検証の要点。\n")

    rows = []
    for leaves, mcs in itertools.product(LEAVES, MIN_CHILD):
        m_val = lgb.LGBMClassifier(n_estimators=max(N_TREES), num_leaves=leaves,
                                   min_child_samples=mcs, random_state=0, **BASE)
        m_val.fit(fx[tr_in], y[tr_in], categorical_feature=F.CAT_COLS)
        m_te = lgb.LGBMClassifier(n_estimators=max(N_TREES), num_leaves=leaves,
                                  min_child_samples=mcs, random_state=0, **BASE)
        m_te.fit(fx[tr], y[tr], categorical_feature=F.CAT_COLS)
        for nt in N_TREES:
            rv, _ = roi_of(m_val.predict_proba(fx[tr_val], num_iteration=nt)[:, 1], sub_val, pays)
            rt, _ = roi_of(m_te.predict_proba(fx[te], num_iteration=nt)[:, 1], sub_te, pays)
            rows.append({"leaves": leaves, "min_child": mcs, "trees": nt,
                         "val_roi": rv, "test_roi": rt})
            print(f"  leaves={leaves:>2} min_child={mcs:>3} 木={nt:>4}  "
                  f"内側検証ROI {rv:6.1f}%   テストROI {rt:6.1f}%")
    g = pd.DataFrame(rows)

    rho, p = spearmanr(g["val_roi"], g["test_roi"])
    print("\n" + "=" * 76)
    print(f"★内側検証ROI と テストROI の順位相関: ρ={rho:+.3f}  (p={p:.3f})")
    print("=" * 76)
    best_val = g.loc[g["val_roi"].idxmax()]
    best_te = g.loc[g["test_roi"].idxmax()]
    cur = g[(g["leaves"] == 31) & (g["min_child"] == 100) & (g["trees"] == 400)].iloc[0]
    print(f"  内側検証で最良の設定 : leaves={int(best_val.leaves)} min_child={int(best_val.min_child)} "
          f"木={int(best_val.trees)}  → **テストROI {best_val.test_roi:.1f}%**")
    print(f"  テストで最良（後知恵）: leaves={int(best_te.leaves)} min_child={int(best_te.min_child)} "
          f"木={int(best_te.trees)}  → テストROI {best_te.test_roi:.1f}%")
    print(f"  現行(31/100/400)                                      → テストROI {cur.test_roi:.1f}%")
    print(f"  テストROIの範囲: {g['test_roi'].min():.1f}〜{g['test_roi'].max():.1f}%")

    # 内側検証で選んだ設定を複数シードで確認
    lv, mc, nt = int(best_val.leaves), int(best_val.min_child), int(best_val.trees)
    print(f"\n■ 内側検証で選んだ設定({lv}/{mc}/{nt}) を{n_seed}シードで確認")
    vals = []
    for seed in range(n_seed):
        m = lgb.LGBMClassifier(n_estimators=nt, num_leaves=lv, min_child_samples=mc,
                               random_state=seed, **BASE)
        m.fit(fx[tr], y[tr], categorical_feature=F.CAT_COLS)
        r, _ = roi_of(m.predict_proba(fx[te])[:, 1], sub_te, pays)
        vals.append(r)
        print(f"   seed{seed}: {r:.1f}%")
    print(f"   平均 {np.mean(vals):.1f}%  シード幅 {np.ptp(vals):.1f}pt")


if __name__ == "__main__":
    main()
