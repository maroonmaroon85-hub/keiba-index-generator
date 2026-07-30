"""
軸③: レース内の相対特徴を足す。

現在の特徴量は全て「その馬の絶対値」だが、目的は**レース内での順位付け**。
同じ「賞金1,500万円」でも新馬戦なら断然、G1なら最下位で、意味が正反対になる。
つまり**このレースの中で何番目か**という情報をモデルに与えていない。
（唯一の例外は(45)で足した `mkt_prob`＝オッズをレース内で正規化したもので、これは既に相対特徴。
　それが効いたことは、相対化に意味がある傍証でもある。）

そこで主要な数値特徴について**レース内パーセンタイル順位**を追加し、AUCとROIで比較する。
リークはない（すべて発走前に確定している値をレース内で並べ替えるだけ）。

比較:
  base … 現行（(45)のオッズ入り）
  rel  … base + レース内順位（下記 REL_COLS のパーセンタイル）

評価は AUC と、(55)(56)で最良だった 2つの買い方:
  枠連 軸枠×紐枠2 ／ 三連複 BOX上位4

実行: python3 ml/relative_features.py [シード数(既定3)]
"""
import itertools
import sys
import warnings

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
sys.path.insert(0, "ml")
import features as F
from place_wide import boot, PARAMS
from pocket_eval import load_payout_a
from waku_umatan import load_wu, waku_of

# レース内順位を足す列（近走の実力・仕上がりに関わる主要な数値）
REL_COLS = ["avg3_fin", "avg3_prize", "best3_fin", "best3_agari", "last_agari",
            "avg3_passratio", "wtcarry", "bodywt", "weeks_since", "fin_std", "raceclass"]


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

    fx_base, _ = F.encode_categoricals(f)
    fx_base["log_odds"] = np.log(odds)
    fx_base["mkt_prob"] = mkt

    fx_rel = fx_base.copy()
    rid = d["raceid"].to_numpy()
    added = []
    for c in REL_COLS:
        if c not in f.columns:
            continue
        fx_rel[f"rel_{c}"] = f[c].groupby(rid).rank(pct=True).to_numpy()
        added.append(c)
    print(f"レース内順位を追加した列（{len(added)}本）: {', '.join(added)}")

    cut = d["date"].quantile(0.3)
    tr, te = (d["date"] < cut).to_numpy(), (d["date"] >= cut).to_numpy()
    pa = load_payout_a("data/payout/a.csv")
    wu = load_wu("data/payout/a.csv")
    sub = d.loc[te, ["raceid", "umaban"]].copy()
    sub["year"] = d.loc[te, "date"].dt.year.to_numpy()
    sub["fieldsize"] = f.loc[te, "fieldsize"].to_numpy()
    rng = np.random.default_rng(0)
    print(f"train {tr.sum():,} / test {te.sum():,}  シード{n_seed}本\n")

    res = {}
    for tag, fxx in [("base", fx_base), ("rel", fx_rel)]:
        ps, aucs = [], []
        for s in range(n_seed):
            m = lgb.LGBMClassifier(random_state=s, **PARAMS)
            m.fit(fxx[tr], top3[tr], categorical_feature=F.CAT_COLS)
            p = m.predict_proba(fxx[te])[:, 1]
            ps.append(p)
            aucs.append(roc_auc_score(top3[te], p))
        res[tag] = (np.mean(ps, axis=0), float(np.mean(aucs)), np.ptp(aucs))
        print(f"  {tag}: AUC {np.mean(aucs):.4f}（シード幅{np.ptp(aucs):.4f}）")

    print(f"\n{'特徴':<8}{'買い方':<22}{'的中率':>8}{'的中時配当':>11}{'ROI':>8}{'95%CI':>14}{'年別':>13}")
    for tag in ["base", "rel"]:
        s2 = sub.copy()
        s2["p"] = res[tag][0]
        rows = {"枠連 軸枠×紐枠2": [], "三連複 BOX上位4": []}
        for r_id, g in s2.groupby("raceid", sort=False):
            gg = g.sort_values("p", ascending=False, kind="mergesort")
            nums = gg["umaban"].astype(int).tolist()
            w = wu.get(r_id)
            if w and w["wakuren"] and len(nums) >= 3:
                fs = int(gg["fieldsize"].iloc[0])
                wa = waku_of(nums[0], fs)
                cs = sorted({tuple(sorted((wa, waku_of(h, fs)))) for h in nums[1:3]})
                if cs:
                    rows["枠連 軸枠×紐枠2"].append(
                        {"y": g["year"].iloc[0], "pay": sum(w["wakuren"].get(c, 0) for c in cs), "k": len(cs)})
            p = pa.get(r_id)
            if p and p["sanrenpuku"] and len(nums) >= 9:
                cs = [tuple(sorted(c)) for c in itertools.combinations(nums[:4], 3)]
                rows["三連複 BOX上位4"].append(
                    {"y": g["year"].iloc[0], "pay": sum(p["sanrenpuku"].get(c, 0) for c in cs), "k": 4})
        for name, rr in rows.items():
            df = pd.DataFrame(rr)
            x = (df["pay"] / (df["k"] * 100)).to_numpy(float)
            lo, hi = boot(x, rng, 2000)
            nz = df.loc[df["pay"] > 0, "pay"]
            ys = pd.Series(x).groupby(df["y"].to_numpy()).mean() * 100
            print(f"{tag:<8}{name:<22}{(x>0).mean()*100:>7.2f}%{nz.mean():>10,.0f}円"
                  f"{x.mean()*100:>7.1f}%{f'[{lo:.0f},{hi:.0f}]':>14}"
                  f"{f'{ys.min():.0f}〜{ys.max():.0f}%':>13}")


if __name__ == "__main__":
    main()
