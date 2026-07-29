"""
★まっさらから測り直す基準表。旧設定の定数を一切引き継がない。

引き継がないもの: gap閾値 / オッズ帯(2-5倍) / 馬場(ダート) / 紐4 / 券種の決め打ち / 好ポケット。
これらは全て (24)(25)(26) で崩れた土台の上で選ばれた定数で、(27)(28)(29)(30) で順に否定された。

測るもの:
  1. 券種(馬連/三連複/三連単マルチ) × 紐2-8 を **全レース**で
  2. **複数シード**で学習し直し、平均と幅を出す（(30)より: 部分集合でなくてもシードで数pt動く）
  3. 軸・紐の選び方を **モデル順 vs 人気順(単勝オッズ)** で比較
     ＝そもそもモデルは市場より良い並べ方をしているのか、という白紙なら当然の問い
  4. 各券種の控除率から決まる「無技能の上限」を併記し、そこからの距離で評価する

実行: python3 ml/baseline.py [シード数(既定3)]
"""
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, "ml")
import features as F
from himo_sweep import POINTS, KEY, hits
from pocket_eval import load_payout_a

# 券種 → JRA控除率。無技能でベタ買いしたときの期待回収率 = 100 - 控除率。
TAKEOUT = {"馬連 軸流し": 22.5, "三連複 軸流し": 25.0, "三連単マルチ": 27.5}
HIMOS = range(2, 9)


def eval_order(order_by_race, pays, tname, n):
    """order_by_race: {raceid: [馬番(優先順)]} → (回収率, R数, 的中率)"""
    pts = POINTS[tname](n) * 100
    tot = hit = cnt = 0
    for rid, nums in order_by_race.items():
        p = pays.get(rid)
        if p is None or not p["sanrentan"] or len(nums) < n + 1:
            continue
        pay = hits(p[KEY[tname]], nums[0], set(nums[1 : n + 1]))
        tot += pay
        hit += pay > 0
        cnt += 1
    return (tot / (cnt * pts) * 100 if cnt else float("nan"), cnt, hit / cnt * 100 if cnt else 0)


def main():
    n_seed = int(sys.argv[1]) if len(sys.argv) > 1 else 3
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
    print(f"train {tr.sum():,} / test {te.sum():,}  分割日 {cut.date()}  シード{n_seed}本")

    # --- 人気順（単勝オッズ昇順）。シードに依存しないので一度だけ ---
    mk = {}
    for rid, g in meta.groupby("raceid", sort=False):
        g = g.sort_values("odds", ascending=True, kind="mergesort")
        mk[rid] = g["umaban"].astype(int).tolist()

    res = defaultdict(list)  # (選び方, 券種, 紐) -> [ROI...]
    meta_n = {}
    for seed in range(n_seed):
        m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=31, subsample=0.8,
                               colsample_bytree=0.8, min_child_samples=100, verbose=-1,
                               random_state=seed)
        m.fit(fx[tr], y[tr], categorical_feature=F.CAT_COLS)
        pr = meta.copy()
        pr["prob"] = m.predict_proba(fx[te])[:, 1]
        ml = {}
        for rid, g in pr.groupby("raceid", sort=False):
            g = g.sort_values("prob", ascending=False, kind="mergesort")
            ml[rid] = g["umaban"].astype(int).tolist()
        for tname in TAKEOUT:
            for n in HIMOS:
                roi, cnt, hr = eval_order(ml, pays, tname, n)
                res[("モデル順", tname, n)].append(roi)
                meta_n[(tname, n)] = (cnt, hr)
        print(f"  seed {seed} 完了")

    for tname in TAKEOUT:
        for n in HIMOS:
            roi, cnt, hr = eval_order(mk, pays, tname, n)
            res[("人気順", tname, n)].append(roi)

    print("\n" + "=" * 78)
    print("★まっさら基準表: 全レース・絞り込み一切なし・軸=1位/紐=2位以降n頭")
    print("=" * 78)
    for tname in TAKEOUT:
        ceil = 100 - TAKEOUT[tname]
        print(f"\n■ {tname}   控除率{TAKEOUT[tname]}% → 無技能の上限 {ceil:.1f}%")
        print(f"{'紐':>3}{'点/R':>6}{'R':>7}{'的中率':>8}{'モデル順':>20}{'人気順':>9}{'差':>7}")
        for n in HIMOS:
            a = np.array(res[("モデル順", tname, n)], float)
            b = res[("人気順", tname, n)][0]
            cnt, hr = meta_n[(tname, n)]
            spread = f"{a.mean():.1f}%±{(a.max() - a.min()) / 2:.1f}" if len(a) > 1 else f"{a.mean():.1f}%"
            print(f"{n:>3}{POINTS[tname](n):>6}{cnt:>7}{hr:>7.1f}%{spread:>20}{b:>8.1f}%{a.mean() - b:>+7.1f}")
        best_n = max(HIMOS, key=lambda k: np.mean(res[("モデル順", tname, k)]))
        best = np.mean(res[("モデル順", tname, best_n)])
        print(f"  → 最良: 紐{best_n} {best:.1f}%   無技能の上限まで {best - ceil:+.1f}pt   "
              f"利益(100%)まで {best - 100:+.1f}pt")


if __name__ == "__main__":
    main()
