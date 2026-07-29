"""
(51)のやり直し。分割モデルを**それぞれ個別にチューニング**して公平に比較する。

(51)の欠陥（指摘を受けて修正）: 分割してデータが1/2〜1/4になったのに、
フルデータ向けのハイパーパラメータ(num_leaves=31, n_estimators=400, min_child_samples=100)を
全ての分割にそのまま流用していた。少ないデータでは過剰に複雑になるため、**分割側だけが不利**な検証だった。

やり直しの設計:
  ・学習期間の**後半20%を内側の検証データ**として切り出す（テストは一切触らない＝リークなし）
  ・部分モデルごとに num_leaves / min_child_samples をグリッド探索し、n_estimators は**早期終了**で決める
    （早期終了だけでもデータ量に応じた複雑さの調整になる。これが(51)に最も欠けていた）
  ・選んだ設定で学習期間全体を使って再学習し、テストで評価
  ・単一モデルにも同じ手続きを適用する（片側だけ最適化しない）

実行: python3 ml/split_models_tuned.py [シード数(既定3)]
"""
import itertools
import sys

import warnings

import numpy as np
import pandas as pd
import lightgbm as lgb

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*eval_set.*")
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "ml")
import features as F
from place_wide import boot
from pocket_eval import load_payout_a

GRID = [(15, 30), (15, 100), (31, 30), (31, 100), (63, 100)]   # (num_leaves, min_child_samples)
BASE = dict(learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, verbose=-1)


def tune(fx, y, m_tr, m_val, seed=0):
    """内側の検証データで (num_leaves, min_child_samples) と最適な木の本数を選ぶ。"""
    best = None
    for leaves, mcs in GRID:
        mdl = lgb.LGBMClassifier(n_estimators=1200, num_leaves=leaves,
                                 min_child_samples=mcs, random_state=seed, **BASE)
        mdl.fit(fx[m_tr], y[m_tr], eval_set=[(fx[m_val], y[m_val])], eval_metric="auc",
                categorical_feature=F.CAT_COLS,
                callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(0)])
        auc = mdl.best_score_["valid_0"]["auc"]
        if best is None or auc > best[0]:
            best = (auc, leaves, mcs, mdl.best_iteration_ or 400)
    return best


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
    # 学習期間の後半20%を内側の検証に回す（テストには触らない）
    inner = d.loc[tr, "date"].quantile(0.8)
    tr_in, tr_val = tr & (d["date"] < inner), tr & (d["date"] >= inner)
    print(f"train {tr.sum():,}（内訳: 学習{tr_in.sum():,} / 内側検証{tr_val.sum():,}） test {te.sum():,}")

    surface = np.where(d["surface"].to_numpy() == 1, "ダ", "芝")
    wet = np.where(d["cond"].isin(["稍", "重", "不"]), "道悪", "良")
    schemes = {
        "1.単一": np.array(["all"] * len(d)),
        "2.芝/ダ別": surface,
        "3.良/道悪別": wet,
    }

    pays = load_payout_a("data/payout/a.csv")
    sub = d.loc[te, ["raceid", "umaban"]].copy()
    sub["year"] = d.loc[te, "date"].dt.year.to_numpy()
    rng = np.random.default_rng(0)
    results = {}

    for name, grp in schemes.items():
        print(f"\n■ {name} のチューニング")
        params = {}
        for g in pd.unique(grp):
            if (tr & (grp == g)).sum() < 5000:
                continue
            auc, leaves, mcs, iters = tune(fx, y, tr_in & (grp == g), tr_val & (grp == g))
            params[g] = (leaves, mcs, iters)
            print(f"   {g:<6} n={ (tr&(grp==g)).sum():>7,}  → leaves={leaves} "
                  f"min_child={mcs} 木={iters}本  内側AUC {auc:.4f}")
        rois, aucs, per_seed = [], [], []
        for seed in range(n_seed):
            pred = np.full(te.sum(), np.nan)
            te_g = grp[te]
            for g, (leaves, mcs, iters) in params.items():
                m_tr = tr & (grp == g)
                m_te = te_g == g
                if m_te.sum() == 0:
                    continue
                mdl = lgb.LGBMClassifier(n_estimators=iters, num_leaves=leaves,
                                         min_child_samples=mcs, random_state=seed, **BASE)
                mdl.fit(fx[m_tr], y[m_tr], categorical_feature=F.CAT_COLS)
                pred[m_te] = mdl.predict_proba(fx.loc[te].loc[m_te])[:, 1]
            ok = ~np.isnan(pred)
            aucs.append(roc_auc_score(y[te][ok], pred[ok]))
            s2 = sub.copy()
            s2["p3"] = pred
            rows = []
            for rid, g2 in s2.groupby("raceid", sort=False):
                pp = pays.get(rid)
                if pp is None or not pp["sanrenpuku"] or len(g2) < 4 or g2["p3"].isna().any():
                    continue
                t = g2.sort_values("p3", ascending=False, kind="mergesort")["umaban"].astype(int).tolist()
                cs = [tuple(sorted((t[0], a, b))) for a, b in itertools.combinations(t[1:4], 2)]
                rows.append({"y": g2["year"].iloc[0],
                             "r": sum(pp["sanrenpuku"].get(c, 0) for c in cs) / 300.0})
            df = pd.DataFrame(rows)
            rois.append(df)
            per_seed.append(df["r"].mean() * 100)
        df = rois[0].copy()
        df["r"] = np.mean([x["r"].to_numpy(float) for x in rois], axis=0)
        results[name] = (np.mean(aucs), df, np.ptp(per_seed))

    print("\n" + "=" * 84)
    print("★個別チューニング後の比較（三連複 軸1×紐3）")
    print("=" * 84)
    print(f"{'分割の仕方':<16}{'AUC':>8}{'R':>8}{'的中率':>8}{'ROI':>8}{'95%CI':>15}{'シード幅':>9}{'年別':>14}")
    for name, (auc, df, sw) in results.items():
        x = df["r"].to_numpy(float)
        lo, hi = boot(x, rng, 2000)
        ys = df.groupby("y")["r"].mean() * 100
        print(f"{name:<16}{auc:>8.4f}{len(x):>8,}{(x>0).mean()*100:>7.1f}%{x.mean()*100:>7.1f}%"
              f"{f'[{lo:.1f},{hi:.1f}]':>15}{sw:>8.1f}{f'{ys.min():.0f}〜{ys.max():.0f}%':>14}")


if __name__ == "__main__":
    main()
