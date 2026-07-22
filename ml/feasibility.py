"""
Phase 5 実現可能性テスト: LightGBM がルールベース指数(アウトオブサンプル単勝EV回収率 約73%)を
超えられるか実測する。

方針:
- 特徴量は「発走前に見える情報」＝各馬の近走(prior runs)＋条件のみ。
  当該レースの着順(=ラベル)は使わない。単勝オッズも特徴に入れない（入れると市場追随になり
  edgeが出ないため。オッズはEV評価にだけ使う）。
- 日付で train(前半)/test(後半) を分割し、リークなしでアウトオブサンプル評価。
- 評価: 的中/複勝率、top-pick複勝率、単勝EV≥1戦略の回収率（レース内で確率を再正規化）。
"""
import glob
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

FILES = sorted(glob.glob("DS*.CSV"))

def load():
    frames = []
    for f in FILES:
        df = pd.read_csv(f, header=None, encoding="shift_jis", dtype=str, keep_default_na=False)
        frames.append(df)
    raw = pd.concat(frames, ignore_index=True)
    raw = raw[raw.shape[1] and raw[40].str.len() > 2]  # 有効行
    d = pd.DataFrame()
    d["date"] = pd.to_datetime("20" + raw[0].str.zfill(2) + raw[1].str.zfill(2) + raw[2].str.zfill(2), format="%Y%m%d", errors="coerce")
    d["course"] = raw[4].str.strip()
    d["surface"] = (raw[9].str.strip() == "ダ").astype(int)  # 1=ダ,0=芝
    d["distance"] = pd.to_numeric(raw[11], errors="coerce")
    d["cond"] = raw[12].str.strip()
    d["horse"] = raw[37].str.strip()  # 血統登録番号
    d["sex"] = raw[14].str.strip()
    d["age"] = pd.to_numeric(raw[15], errors="coerce")
    d["wtcarry"] = pd.to_numeric(raw[17], errors="coerce")
    d["fieldsize"] = pd.to_numeric(raw[18], errors="coerce")
    d["finish"] = pd.to_numeric(raw[20], errors="coerce")
    d["margin"] = pd.to_numeric(raw[23], errors="coerce")
    p = raw[[28, 29, 30, 31]].apply(pd.to_numeric, errors="coerce")
    p = p.where(p > 0)
    d["passavg"] = p.mean(axis=1)
    d["raceid"] = raw[40].str[:-2]
    d["sire"] = raw[43].str.strip()
    d["damsire"] = raw[45].str.strip()
    d["odds"] = pd.to_numeric(raw[48], errors="coerce")
    d = d.dropna(subset=["date", "distance", "finish", "fieldsize", "horse"])
    d = d.drop_duplicates(subset=["raceid", "horse"])
    d["finratio"] = (d["fieldsize"] - d["finish"] + 1) / d["fieldsize"]
    d["passratio"] = d["passavg"] / d["fieldsize"]
    return d.sort_values(["horse", "date"]).reset_index(drop=True)

def features(d):
    g = d.groupby("horse", sort=False)
    f = pd.DataFrame(index=d.index)
    # 近走(prior)＝shift(1..)。当該走は使わない。
    f["n_prior"] = g.cumcount()
    f["last1_fin"] = g["finratio"].shift(1)
    f["last2_fin"] = g["finratio"].shift(2)
    f["last3_fin"] = g["finratio"].shift(3)
    f["avg3_fin"] = g["finratio"].shift(1).rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
    f["best3_fin"] = g["finratio"].shift(1).rolling(3, min_periods=1).max().reset_index(level=0, drop=True)
    f["last_margin"] = g["margin"].shift(1)
    f["last_passratio"] = g["passratio"].shift(1)
    f["avg3_passratio"] = g["passratio"].shift(1).rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
    last_dist = g["distance"].shift(1)
    f["dist_change"] = d["distance"] - last_dist
    f["same_dist"] = (d["distance"] == last_dist).astype(float)
    last_surf = g["surface"].shift(1)
    f["surf_change"] = (d["surface"] != last_surf).astype(float)
    f["wt_change"] = d["wtcarry"] - g["wtcarry"].shift(1)
    days_since = (d["date"] - g["date"].shift(1)).dt.days
    f["weeks_since"] = days_since / 7.0
    f["weeks_before"] = (g["date"].shift(1) - g["date"].shift(2)).dt.days / 7.0
    # 当該レースの条件（発走前に既知）
    f["distance"] = d["distance"]
    f["surface"] = d["surface"]
    f["fieldsize"] = d["fieldsize"]
    f["wtcarry"] = d["wtcarry"]
    f["age"] = d["age"]
    for col in ["sex", "cond", "course", "sire", "damsire"]:
        f[col] = d[col].astype("category")
    return f

def main():
    d = load()
    f = features(d)
    keep = f["n_prior"] >= 1
    d, f = d[keep].reset_index(drop=True), f[keep].reset_index(drop=True)
    y = (d["finish"] == 1).astype(int)

    cut = d["date"].quantile(0.7)
    tr, te = d["date"] < cut, d["date"] >= cut
    cats = ["sex", "cond", "course", "sire", "damsire"]
    print(f"train {tr.sum()}頭 / test {te.sum()}頭  分割日 {cut.date()}  勝率(base) {y.mean():.3f}")

    model = lgb.LGBMClassifier(
        n_estimators=400, learning_rate=0.03, num_leaves=31,
        subsample=0.8, colsample_bytree=0.8, min_child_samples=100, verbose=-1,
    )
    model.fit(f[tr], y[tr], categorical_feature=cats)
    d = d.copy()
    d["pred"] = model.predict_proba(f)[:, 1]

    auc = roc_auc_score(y[te], d["pred"][te])
    # レース内で再正規化して確率に。
    dte = d[te].copy()
    dte["prob"] = dte["pred"] / dte.groupby("raceid")["pred"].transform("sum")

    # top-pick(モデル最上位)の複勝率
    top = dte.loc[dte.groupby("raceid")["prob"].idxmax()]
    toppick_place = (top["finish"] <= 3).mean() * 100
    toppick_win = (top["finish"] == 1).mean() * 100

    # 単勝EV戦略（contenderOnly: prob>=1/頭数）
    ev = dte[(dte["odds"] > 0) & (dte["prob"] >= 1 / dte["fieldsize"])].copy()
    ev["ev"] = ev["prob"] * ev["odds"]
    bet = ev[ev["ev"] >= 1.0]
    roi = (bet.loc[bet["finish"] == 1, "odds"].sum() * 100) / (len(bet) * 100) * 100 if len(bet) else 0

    print("\n===== ML アウトオブサンプル結果 =====")
    print(f"AUC(test)                 : {auc:.4f}   （0.5=でたらめ, 高いほど良い）")
    print(f"top-pick 複勝率           : {toppick_place:.1f}%   （ルールベース◎: 47%）")
    print(f"top-pick 勝率             : {toppick_win:.1f}%   （ルールベース◎: 19%）")
    print(f"単勝EV≥1 戦略 回収率      : {roi:.1f}%  ({len(bet)}点)   （ルールベースOOS: 約73%）")

    imp = pd.Series(model.feature_importances_, index=f.columns).sort_values(ascending=False)
    print("\n重要な特徴量 top10:")
    for k, v in imp.head(10).items():
        print(f"  {k:16s} {int(v)}")

if __name__ == "__main__":
    main()
