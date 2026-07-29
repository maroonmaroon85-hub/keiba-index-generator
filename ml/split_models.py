"""
「馬場ごとにモデルを分けるべきでは？ 全部同じで行こうとしていないか」を検証する。

現状は**単一モデル**で、cond(馬場状態)とsurface(芝/ダ)はカテゴリ特徴として入っている。
木は内部で分岐できるが、学習は**全データ共通の損失**を最小化するため、
少数派（道悪は約30%、不良は約4%）や、芝ダで効き方が変わる特徴を十分に拾えていない可能性がある。
芝とダートは実質別競技なので、そこも含めて分割学習を比較する。

比較する4通り（特徴・パラメータ・シードは全て共通。**分割の仕方だけ**を変える）:
  1. 単一        … 現行
  2. 芝/ダ 別     … surface で2分割
  3. 良/道悪 別   … cond が 良 か 稍重不 かで2分割
  4. 芝ダ×良道悪  … 4分割

評価は (48) の実用指標に合わせて **三連複 軸1×紐3 のROI**（top3目標モデル）と、
参考として test 全体の AUC。分割で標本が減る副作用もあるので、両方を見る必要がある。

実行: python3 ml/split_models.py [シード数(既定3)]
"""
import itertools
import sys

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "ml")
import features as F
from place_wide import boot, PARAMS
from pocket_eval import load_payout_a


def fit_predict(fx, y, tr, te, groups, seed):
    """groups（各行の分割ラベル）ごとに学習し、test の予測を1本にまとめて返す。
    groups が全て同じ値なら単一モデルと同じ。"""
    pred = np.full(te.sum(), np.nan)
    te_g = groups[te]
    for g in pd.unique(groups):
        m_tr = tr & (groups == g)
        m_te = te_g == g
        if m_tr.sum() < 5000 or m_te.sum() == 0:
            continue
        mdl = lgb.LGBMClassifier(random_state=seed, **PARAMS)
        mdl.fit(fx[m_tr], y[m_tr], categorical_feature=F.CAT_COLS)
        pred[m_te] = mdl.predict_proba(fx.loc[te].loc[m_te])[:, 1]
    return pred


def main():
    n_seed = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    d = F.to_model(F.load_files())
    f = F.build_features(d)
    keep = (f["n_prior"] >= 1) & d["odds"].notna() & (d["odds"] > 0)
    d, f = d[keep].reset_index(drop=True), f[keep].reset_index(drop=True)
    top3 = (d["finish"] <= 3).astype(int).to_numpy()
    odds = d["odds"].to_numpy(float)
    inv = 1.0 / odds
    mkt = inv / pd.Series(inv).groupby(d["raceid"]).transform("sum").to_numpy()
    fx, _ = F.encode_categoricals(f)
    fx["log_odds"] = np.log(odds)
    fx["mkt_prob"] = mkt
    cut = d["date"].quantile(0.3)
    tr, te = (d["date"] < cut).to_numpy(), (d["date"] >= cut).to_numpy()

    surface = d["surface"].astype(str).to_numpy() if "surface" in d else None
    if surface is None:
        # to_model に surface が無い場合は距離列の隣（馬場種別）を使う想定。無ければ cond のみで検証。
        surface = np.array(["?"] * len(d))
    wet = np.where(d["cond"].isin(["稍", "重", "不"]), "道悪", "良")
    schemes = {
        "1.単一": np.array(["all"] * len(d)),
        "2.芝/ダ別": surface,
        "3.良/道悪別": wet,
        "4.芝ダ×良道悪": np.char.add(surface.astype(str), wet.astype(str)),
    }
    for k, v in schemes.items():
        n = len(pd.unique(v))
        print(f"  {k}: {n}分割  " + " / ".join(f"{g}:{(tr & (v==g)).sum():,}"
                                              for g in pd.unique(v)[:6]))

    pays = load_payout_a("data/payout/a.csv")
    sub = d.loc[te, ["raceid", "umaban"]].copy()
    sub["odds"] = odds[te]
    sub["year"] = d.loc[te, "date"].dt.year.to_numpy()
    rng = np.random.default_rng(0)

    print(f"\ntrain {tr.sum():,} / test {te.sum():,}  シード{n_seed}本")
    print("=" * 88)
    print(f"{'分割の仕方':<16}{'AUC':>8}{'R':>8}{'的中率':>8}{'ROI':>8}{'95%CI':>15}"
          f"{'シード幅':>9}{'年別':>14}")
    for name, grp in schemes.items():
        rois, aucs, per_seed = [], [], []
        for seed in range(n_seed):
            p = fit_predict(fx, top3, tr, te, grp, seed)
            ok = ~np.isnan(p)
            aucs.append(roc_auc_score(top3[te][ok], p[ok]))
            s2 = sub.copy()
            s2["p3"] = p
            rows = []
            for rid, g in s2.groupby("raceid", sort=False):
                pp = pays.get(rid)
                if pp is None or not pp["sanrenpuku"] or len(g) < 4 or g["p3"].isna().any():
                    continue
                gm = g.sort_values("p3", ascending=False, kind="mergesort")
                t = gm["umaban"].astype(int).tolist()
                cs = [tuple(sorted((t[0], a, b))) for a, b in itertools.combinations(t[1:4], 2)]
                rows.append({"y": g["year"].iloc[0],
                             "r": sum(pp["sanrenpuku"].get(c, 0) for c in cs) / 300.0})
            df = pd.DataFrame(rows)
            rois.append(df)
            per_seed.append(df["r"].mean() * 100)
        df = rois[0].copy()
        df["r"] = np.mean([x["r"].to_numpy(float) for x in rois], axis=0)
        x = df["r"].to_numpy(float)
        lo, hi = boot(x, rng, 2000)
        ys = df.groupby("y")["r"].mean() * 100
        print(f"{name:<16}{np.mean(aucs):>8.4f}{len(x):>8,}{(x>0).mean()*100:>7.1f}%"
              f"{x.mean()*100:>7.1f}%{f'[{lo:.1f},{hi:.1f}]':>15}{np.ptp(per_seed):>8.1f}"
              f"{f'{ys.min():.0f}〜{ys.max():.0f}%':>14}")


if __name__ == "__main__":
    main()
