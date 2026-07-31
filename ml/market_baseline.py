"""本命推奨（枠連 軸枠×紐枠2）に「人気順で同じ買い方をする」対照実験を当てる。

**なぜ必要か**: HANDOFFは枠連84.5%を「控除率の天井77.5%を+7pt上回る」と書いてきたが、
天井77.5%は**pool平均**（＝掛け金を市場の投票比率どおりに散らした場合）の水準であって、
「人気馬に寄せた買い方」の基準ではない。人気に寄せるだけで人気バイアスのぶん天井を超えるので、
**モデルの正味の寄与を測るには同じ買い方を人気順で並べた対照が要る**。
(47)ワイド・(49)三連複には人気順の対照があったが、**本命の枠連には無かった**（(69)で追加）。

測るもの:
  1. 人気順ベースライン vs モデル順（対応のあるブートストラップ・年別・買い目一致率）
  2. ★判定基準1(シード)・3(設定) を**差**に当てる
     … (53)の通りハイパーパラメータは事前に選べないので、差が設定で動くなら差は確定できない

実行: python3 ml/market_baseline.py [シード数(既定3)]
"""
import itertools
import sys
import warnings

import numpy as np
import pandas as pd
import lightgbm as lgb

warnings.filterwarnings("ignore")
sys.path.insert(0, "ml")
import features as F
from place_wide import PARAMS, boot
from pocket_eval import load_payout_a
from waku_umatan import load_wu, waku_of

PAYOUT = "data/payout/a.csv"

# (53)のグリッドから代表4点。内側検証でもROIでも事前に選べないことが示されている。
CONFIGS = [
    ("現行 leaves31/mc100/400本", dict(PARAMS)),
    ("leaves15/mc30/400本 ((53)の内側検証が選ぶ)", dict(PARAMS, num_leaves=15, min_child_samples=30)),
    ("leaves63/mc30/1000本 ((53)の後知恵最良)",
     dict(PARAMS, num_leaves=63, min_child_samples=30, n_estimators=1000)),
    ("leaves15/mc100/100本 (単純)", dict(PARAMS, num_leaves=15, n_estimators=100)),
]


def wakuren_cs(nums, n):
    """軸枠×紐枠2 の買い目（同枠に入ると1点に縮む）。"""
    wa = waku_of(nums[0], n)
    return sorted({tuple(sorted((wa, waku_of(h, n)))) for h in nums[1:3]})


def load():
    d = F.to_model(F.load_files())
    f = F.build_features(d)
    keep = (f["n_prior"] >= 1) & d["odds"].notna() & (d["odds"] > 0)
    d, f = d[keep].reset_index(drop=True), f[keep].reset_index(drop=True)
    odds = d["odds"].to_numpy(float)
    inv = 1.0 / odds
    fx, _ = F.encode_categoricals(f)
    fx["log_odds"] = np.log(odds)
    fx["mkt_prob"] = inv / pd.Series(inv).groupby(d["raceid"]).transform("sum").to_numpy()
    return d, fx, odds


def main():
    n_seed = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    d, fx, odds = load()
    y = (d["finish"] <= 3).astype(int).to_numpy()
    cut = d["date"].quantile(0.3)
    tr, te = (d["date"] < cut).to_numpy(), (d["date"] >= cut).to_numpy()
    print(f"train {tr.sum():,} / test {te.sum():,}（分割 {cut.date()}）  シード{n_seed}本")

    wu, pa = load_wu(PAYOUT), load_payout_a(PAYOUT)
    sub = d.loc[te, ["raceid", "umaban", "fieldsize"]].copy()
    sub["odds"] = odds[te]
    sub["year"] = d.loc[te, "date"].dt.year.to_numpy()

    ms = [lgb.LGBMClassifier(random_state=s, **PARAMS).fit(fx[tr], y[tr], categorical_feature=F.CAT_COLS)
          for s in range(n_seed)]
    sub["p"] = np.mean([m.predict_proba(fx[te])[:, 1] for m in ms], axis=0)

    # ===== 1. 枠連 / 三連複BOX4 を モデル順 vs 人気順 で =====
    rw, rs = [], []
    for rid, g in sub.groupby("raceid", sort=False):
        w, s3 = wu.get(rid), pa.get(rid)
        n = int(g["fieldsize"].iloc[0])
        order = {"model": g.sort_values("p", ascending=False, kind="mergesort"),
                 "pop": g.sort_values("odds", ascending=True, kind="mergesort")}
        nums = {k: v["umaban"].astype(int).tolist() for k, v in order.items()}
        if w and w["wakuren"] and len(g) >= 3:
            row, buys = {"year": g["year"].iloc[0]}, {}
            for k, nm in nums.items():
                cs = wakuren_cs(nm, n)
                buys[k] = set(cs)
                row[f"{k}_pay"] = sum(w["wakuren"].get(c, 0) for c in cs)
                row[f"{k}_k"] = len(cs)
            row["same"] = buys["model"] == buys["pop"]
            rw.append(row)
        if s3 and s3["sanrenpuku"] and len(g) >= 9:
            row = {"year": g["year"].iloc[0]}
            for k, nm in nums.items():
                cs = [tuple(sorted(c)) for c in itertools.combinations(nm[:4], 3)]
                row[f"{k}_pay"] = sum(s3["sanrenpuku"].get(c, 0) for c in cs)
                row[f"{k}_k"] = 4
            rs.append(row)

    rng = np.random.default_rng(0)

    def compare(title, rows, ceiling):
        df = pd.DataFrame(rows)
        x = {k: (df[f"{k}_pay"] / (df[f"{k}_k"] * 100)).to_numpy(float) for k in ("model", "pop")}
        print(f"\n=== {title}  {len(df):,}R ===")
        print(f"  A. 控除率の天井(pool平均)     {ceiling:5.1f}%")
        print(f"  B. 人気順で同じ買い方         {x['pop'].mean()*100:5.1f}%"
              f"   ← 天井より{x['pop'].mean()*100-ceiling:+.1f}pt（人気バイアスぶん・モデル不要）")
        print(f"  C. モデル順                   {x['model'].mean()*100:5.1f}%"
              f"   ← Bより{(x['model']-x['pop']).mean()*100:+.2f}pt（モデルの正味の寄与）")
        diff = x["model"] - x["pop"]
        lo, hi = boot(diff, rng, 2000)
        print(f"  差の95%CI [{lo:+.2f}, {hi:+.2f}]pt")
        if "same" in df:
            print(f"  買い目が人気順と完全一致するレース: {df['same'].mean()*100:.1f}%")
        ys = pd.Series(diff * 100).groupby(df["year"].to_numpy()).mean()
        print("  年別の差: " + " ".join(f"{y_}:{v:+.1f}" for y_, v in ys.items()))
        print(f"  → 差がプラスの年 {int((ys>0).sum())}/{len(ys)}")

    compare("枠連 軸枠×紐枠2（本命推奨）", rw, 77.5)
    compare("三連複 BOX上位4（対抗）", rs, 75.0)

    # ===== 2. 差に ★判定基準1(シード)・3(設定) を当てる =====
    groups = [(rid, g) for rid, g in sub.groupby("raceid", sort=False)
              if len(g) >= 3 and wu.get(rid) and wu[rid]["wakuren"]]

    def wakuren_roi(frame, col, asc):
        pays, ks = [], []
        for rid, g in groups:
            gg = frame.loc[g.index].sort_values(col, ascending=asc, kind="mergesort")
            cs = wakuren_cs(gg["umaban"].astype(int).tolist(), int(g["fieldsize"].iloc[0]))
            pays.append(sum(wu[rid]["wakuren"].get(c, 0) for c in cs))
            ks.append(len(cs))
        return np.array(pays) / (np.array(ks) * 100.0)

    pop_roi = wakuren_roi(sub, "odds", True).mean() * 100
    print(f"\n=== ★判定基準3（設定の不確実性）を差に当てる — 枠連 {len(groups):,}R ===")
    print(f"人気順ベースライン（設定に依存しない）: {pop_roi:.2f}%")
    print(f"{'設定':<44}{'モデルROI':>11}{'差':>10}{'シード幅':>10}")
    for name, par in CONFIGS:
        per_seed = []
        for s in range(min(n_seed, 2)):
            m = lgb.LGBMClassifier(random_state=s, **par).fit(fx[tr], y[tr],
                                                              categorical_feature=F.CAT_COLS)
            fr = sub.copy()
            fr["q"] = m.predict_proba(fx[te])[:, 1]
            per_seed.append(wakuren_roi(fr, "q", False).mean() * 100)
        mu = float(np.mean(per_seed))
        print(f"{name:<44}{mu:>10.2f}%{mu-pop_roi:>+9.2f}pt{max(per_seed)-min(per_seed):>9.2f}pt")


if __name__ == "__main__":
    main()
