"""
ダート／芝／馬場状態に本当に差があるのかを、(31)のまっさら基準で決着させる。

経緯: ③で「ダート78.5%>芝75.4%」(旧設定・3pt差)→好ポケットに採用→(28)で正しく測ると+1pt→
(29)で道悪ダートが強く見える→(30)でシードノイズと判明し撤回。
だが**(31)のまっさら基準（複数シード・全レース・紐は各券種の最良）で馬場を測ってはいなかった**ため、ここで決着させる。

(30)の教訓に従い、**効果量をシードの幅と必ず並べて表示**する。幅より小さい差は主張しない。

実行: python3 ml/surface_check.py [シード数(既定3)]
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

# (31)の基準表で各券種の最良だった紐数
BEST = [("三連単マルチ", 7), ("馬連 軸流し", 6)]


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
    meta = d.loc[te, ["raceid", "umaban", "cond"]].copy()
    print(f"train {tr.sum():,} / test {te.sum():,}  シード{n_seed}本")

    res = defaultdict(list)
    counts = {}
    for seed in range(n_seed):
        m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=31, subsample=0.8,
                               colsample_bytree=0.8, min_child_samples=100, verbose=-1,
                               random_state=seed)
        m.fit(fx[tr], y[tr], categorical_feature=F.CAT_COLS)
        pr = meta.copy()
        pr["prob"] = m.predict_proba(fx[te])[:, 1]

        rows = []
        for rid, g in pr.groupby("raceid", sort=False):
            p = pays.get(rid)
            if p is None or not p["sanrentan"]:
                continue
            g = g.sort_values("prob", ascending=False, kind="mergesort")
            nums = g["umaban"].astype(int).tolist()
            r = {"surface": p["surface"], "cond": g["cond"].iloc[0], "nh": len(nums)}
            for tname, n in BEST:
                r[(tname, n)] = hits(p[KEY[tname]], nums[0], set(nums[1 : n + 1])) if len(nums) > n else np.nan
            rows.append(r)
        df = pd.DataFrame(rows)

        for tname, n in BEST:
            pts = POINTS[tname](n) * 100
            ok = df[df[(tname, n)].notna()]

            def roi(sub):
                return sub[(tname, n)].sum() / (len(sub) * pts) * 100 if len(sub) else float("nan")

            res[(tname, n, "全体")].append(roi(ok))
            counts[(tname, n, "全体")] = len(ok)
            for s in ["芝", "ダ"]:
                sub = ok[ok["surface"] == s]
                res[(tname, n, s)].append(roi(sub))
                counts[(tname, n, s)] = len(sub)
            for s in ["芝", "ダ"]:
                for c in ["良", "稍", "重", "不"]:
                    sub = ok[(ok["surface"] == s) & (ok["cond"] == c)]
                    if len(sub) < 200:
                        continue
                    res[(tname, n, s + c)].append(roi(sub))
                    counts[(tname, n, s + c)] = len(sub)
        print(f"  seed {seed} 完了")

    print("\n" + "=" * 72)
    print("★馬場の決着（全レース・(31)の最良紐・複数シード）")
    print("=" * 72)
    for tname, n in BEST:
        print(f"\n■ {tname}×紐{n}")
        print(f"{'条件':<8}{'R':>7}{'平均ROI':>10}{'シード幅':>10}{'全体との差':>12}  判定")
        base = np.mean(res[(tname, n, "全体")])
        base_w = np.ptp(res[(tname, n, "全体")])
        for k in [k for k in res if k[0] == tname and k[1] == n]:
            a = np.array(res[k], float)
            w = np.ptp(a)
            diff = a.mean() - base
            if k[2] == "全体":
                verdict = "—"
            else:
                verdict = "差はノイズ以下" if abs(diff) <= max(w, base_w) else "幅を超える差"
            print(f"{k[2]:<8}{counts[k]:>7}{a.mean():>9.1f}%{w:>9.1f}{diff:>+11.1f}pt  {verdict}")


if __name__ == "__main__":
    main()
