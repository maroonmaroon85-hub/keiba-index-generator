"""
「部分集合で最も成績の良かったモデル(シード)を選んで使えばいいのでは？」を検証する。

(30)で、同じ仕様のモデルを学習し直すだけで ダ不(945R) の回収率が 61.7%〜110% で動くと分かった。
ここで自然に出る発想が「では一番良かったモデルを使えばいい」。これが成立する条件はただ一つ、
**前半で成績が良かったシードが、後半でも良いこと**。それを直接測る。

各シードについて ダ不 の回収率を 前半(2017-2021) と 後半(2022-2026) で別々に出し、
  ・前半と後半の相関
  ・前半で最良だったシードの、後半での成績
  ・全シードの後半平均（＝シードを選ばずに使った場合の期待値）
を比べる。相関が無ければ「良いモデルを選ぶ」は不可能＝過去の当たりを選んでいるだけ。

実行: python3 ml/seed_pick.py [シード数(既定8)] [部分集合(既定ダ不)]
"""
import sys

import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, "ml")
import features as F
from himo_sweep import POINTS, KEY, hits
from pocket_eval import load_payout_a

STRAT = ("三連単マルチ", 8)
SPLIT_YEAR = 2022


def main():
    n_seed = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    target_cs = sys.argv[2] if len(sys.argv) > 2 else "ダ不"

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
    tname, n = STRAT
    pts = POINTS[tname](n) * 100
    print(f"train {tr.sum():,} / test {te.sum():,}   部分集合={target_cs}   シード{n_seed}本")

    rec = []
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
            if p is None or not p["sanrentan"] or len(g) < n + 1:
                continue
            g = g.sort_values("prob", ascending=False)
            nums = g["umaban"].astype(int).tolist()
            rows.append({"year": p["date"].year, "cs": p["surface"] + g["cond"].iloc[0],
                         "pay": hits(p[KEY[tname]], nums[0], set(nums[1 : n + 1]))})
        df = pd.DataFrame(rows)
        sub = df[df["cs"] == target_cs]
        a = sub[sub["year"] < SPLIT_YEAR]
        b = sub[sub["year"] >= SPLIT_YEAR]
        ra = a["pay"].sum() / (len(a) * pts) * 100
        rb = b["pay"].sum() / (len(b) * pts) * 100
        rall = df["pay"].sum() / (len(df) * pts) * 100
        rec.append((seed, ra, rb, rall))
        print(f"  seed {seed}: 前半 {ra:6.1f}% ({len(a)}R)   後半 {rb:6.1f}% ({len(b)}R)   全体(全レース) {rall:.1f}%")

    r = pd.DataFrame(rec, columns=["seed", "first", "second", "all"])
    print(f"\n===== {target_cs} / {tname}×紐{n}  シード{n_seed}本 =====")
    print(f"前半: {r['first'].min():.1f}〜{r['first'].max():.1f}%  後半: {r['second'].min():.1f}〜{r['second'].max():.1f}%")
    corr = r["first"].corr(r["second"])
    best = r.loc[r["first"].idxmax()]
    print(f"\n前半と後半の相関: {corr:+.3f}")
    print(f"前半で最良のシード {int(best['seed'])}: 前半{best['first']:.1f}% → **後半{best['second']:.1f}%**")
    print(f"シードを選ばなかった場合の後半平均: {r['second'].mean():.1f}%")
    diff = best["second"] - r["second"].mean()
    print(f"→ 『良いモデルを選ぶ』ことで後半に得られた差: {diff:+.1f}pt")
    print(f"\n参考: 全レース(28,460R)での回収率は {r['all'].min():.1f}〜{r['all'].max():.1f}%（シード間で安定）")


if __name__ == "__main__":
    main()
