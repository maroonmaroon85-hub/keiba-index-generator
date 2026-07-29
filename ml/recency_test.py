"""
「後半で消える」のは市場効率か、それとも**モデルが古いだけ**かを切り分ける。

これまで(29)(32)(34)で繰り返し「前半は100%超・後半は基準並み」という形が出た（好ポケット・ダ不・組合せB）。
市場が効率化したと解釈してきたが、**学習の分割日は2017-02-05**であり、
  前半(2017-2021) = 学習直後   ／   後半(2022-2026) = 学習から5〜9年後
＝**モデルの陳腐化でも全く同じ形が出る**。この2つは今まで区別されていなかった。

切り分け方: **2021年末までで学習し直して2022-2026を予測する**（＝後半にとって"新鮮"なモデル）。
  ・優位が復活する → 原因は陳腐化。実運用では定期的に再学習すればよく、話が変わる。
  ・復活しない     → 原因は市場効率。組合せBは本当に死んでいる。

実行: python3 ml/recency_test.py [シード数(既定3)]
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

TNAME, NHIMO = "三連単マルチ", 7
FRESH_CUT = pd.Timestamp("2022-01-01")   # ここまでで学習し、以降を予測する


def conds(df):
    big = df["fieldsize"] >= 16
    keep_j = ~df["jockey_changed"].astype(bool)
    a = big & keep_j
    return {
        "全体": pd.Series(True, index=df.index),
        "A: 16頭〜&継続": a,
        "B: A&〜1200m": a & (df["distance"] <= 1200),
        "C: A&3勝-OP": a & (df["raceclass"] >= 4) & (df["raceclass"] <= 5),
    }


def main():
    n_seed = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    d = F.to_model(F.load_files())
    f = F.build_features(d)
    keep = f["n_prior"] >= 1
    d, f = d[keep].reset_index(drop=True), f[keep].reset_index(drop=True)
    d["prev_jockey"] = d.groupby("horse")["jockey"].shift(1)
    d["jockey_changed"] = (d["prev_jockey"].notna()) & (d["jockey"] != d["prev_jockey"])
    y = (d["finish"] == 1).astype(int).to_numpy()
    fx, _ = F.encode_categoricals(f)
    pays = load_payout_a("data/payout/a.csv")

    old_cut = d["date"].quantile(0.3)
    settings = {
        "旧モデル(2017-02まで学習)": (d["date"] < old_cut).to_numpy(),
        "新モデル(2021末まで学習)": (d["date"] < FRESH_CUT).to_numpy(),
    }
    te = (d["date"] >= FRESH_CUT).to_numpy()   # 評価は両方とも 2022-2026 に揃える
    print(f"評価期間を 2022-2026 に固定: {te.sum():,}頭")

    meta = d.loc[te, ["raceid", "umaban", "jockey_changed"]].copy()
    for c in ["fieldsize", "raceclass", "distance"]:
        meta[c] = f.loc[te, c].to_numpy()
    pts = POINTS[TNAME](NHIMO) * 100

    res = defaultdict(list)
    cnts = {}
    for label, tr in settings.items():
        print(f"\n{label}: train {tr.sum():,}頭")
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
                if p is None or not p["sanrentan"] or len(g) <= NHIMO:
                    continue
                g = g.sort_values("prob", ascending=False, kind="mergesort")
                nums = g["umaban"].astype(int).tolist()
                a = g.iloc[0]
                rows.append({"pay": hits(p[KEY[TNAME]], nums[0], set(nums[1:NHIMO + 1])),
                             "fieldsize": a["fieldsize"], "raceclass": a["raceclass"],
                             "distance": a["distance"], "jockey_changed": a["jockey_changed"]})
            df = pd.DataFrame(rows)
            for name, mask in conds(df).items():
                sub = df[mask.to_numpy()]
                cnts[name] = len(sub)
                res[(label, name)].append(sub["pay"].sum() / (len(sub) * pts) * 100 if len(sub) else np.nan)
            print(f"  seed {seed} 完了")

    print("\n" + "=" * 76)
    print(f"★陳腐化か市場効率か  評価期間=2022-2026に固定  {TNAME}×紐{NHIMO}")
    print("=" * 76)
    print(f"{'条件':<18}{'R':>7}{'旧モデル':>12}{'新モデル':>12}{'差':>9}")
    for name in ["全体", "A: 16頭〜&継続", "B: A&〜1200m", "C: A&3勝-OP"]:
        o = np.array(res[("旧モデル(2017-02まで学習)", name)], float)
        n = np.array(res[("新モデル(2021末まで学習)", name)], float)
        print(f"{name:<18}{cnts[name]:>7}{o.mean():>10.1f}%±{np.ptp(o) / 2:<4.1f}"
              f"{n.mean():>9.1f}%±{np.ptp(n) / 2:<4.1f}{n.mean() - o.mean():>+8.1f}")
    print("\n→ 新モデルで優位が復活すれば原因は陳腐化、しなければ市場効率。")


if __name__ == "__main__":
    main()
