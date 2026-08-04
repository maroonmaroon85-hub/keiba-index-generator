"""A1の続き: 「候補から抜けた馬」を**理由別**に分ける。

audit_premise.py で、候補馬が実頭数より少ないレースほどROIが高いと分かった（三連複で顕著）。
だが抜ける理由は3つあり、正当性が全く違う:

  (a) 初出走・前走なし（n_prior=0）… **正当**。発走前に本当に過去走が無い馬で、
      実際の予想でも順位付けできない。除くのは戦略の一部。
  (b) アーカイブに存在しない        … **不当**。現実には過去走があるのに手元のデータに無いだけ。
      抜けるのは「その後どこにも出てこなかった馬」＝弱い馬に偏るので、
      これを候補から自動的に消すのは**発走後に知る情報を使っているのと同じ**。
  (c) 単勝オッズが無い              … ほぼ (b) と同時に起きる。

(b)がROIを押し上げているなら、報告値は水増しされている。
**(b)が1頭も無いレースだけ**（＝アーカイブが完全）で測り直すのが正しい基準線。

実行: python3 ml/audit_coverage2.py [シード数(既定3)]
"""
import itertools
import os
import sys
import warnings

import numpy as np
import pandas as pd
import lightgbm as lgb

warnings.filterwarnings("ignore")
sys.path.insert(0, "ml")
import features as F
from _cache import CACHE, load_cached
from place_wide import PARAMS
from pocket_eval import load_payout_a
from waku_umatan import load_wu, waku_of

PAYOUT = "data/payout/a.csv"
RAWC = f"{CACHE}/rawcount.pkl"


def race_counts():
    """レースごとに (アーカイブ在籍頭数, 実頭数) を返す。keep フィルタ前の数え上げ。"""
    if os.path.exists(RAWC):
        return pd.read_pickle(RAWC)
    d = F.to_model(F.load_files())
    f = F.build_features(d)
    g = pd.DataFrame({
        "raceid": d["raceid"], "fieldsize": d["fieldsize"],
        "n_prior": f["n_prior"].to_numpy(), "has_odds": d["odds"].notna() & (d["odds"] > 0)})
    out = g.groupby("raceid").agg(
        n_arch=("raceid", "size"), fieldsize=("fieldsize", "first"),
        n_debut=("n_prior", lambda s: int((s == 0).sum())),
        n_noodds=("has_odds", lambda s: int((~s).sum())))
    out["n_absent"] = out["fieldsize"] - out["n_arch"]      # (b) アーカイブに無い
    out.to_pickle(RAWC)
    return out


def wakuren_cs(nums, n):
    wa = waku_of(nums[0], n)
    return sorted({tuple(sorted((wa, waku_of(h, n)))) for h in nums[1:3]})


def boot(x, rng, n=2000):
    b = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return np.percentile(b, 2.5) * 100, np.percentile(b, 97.5) * 100


def main():
    n_seed = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    rc = race_counts()
    print(f"レース {len(rc):,}  アーカイブ欠損0のレース {float((rc['n_absent'] <= 0).mean())*100:.1f}%  "
          f"欠損頭数の平均 {rc['n_absent'].clip(lower=0).mean():.2f}頭 / 実頭数平均 {rc['fieldsize'].mean():.1f}")

    d, fx = load_cached()
    y = (d["finish"] <= 3).astype(int).to_numpy()
    cut = d["date"].quantile(0.3)
    tr, te = (d["date"] < cut).to_numpy(), (d["date"] >= cut).to_numpy()
    ms = [lgb.LGBMClassifier(random_state=s, **PARAMS).fit(fx[tr], y[tr], categorical_feature=F.CAT_COLS)
          for s in range(n_seed)]
    p = np.mean([m.predict_proba(fx[te])[:, 1] for m in ms], axis=0)

    sub = d.loc[te, ["raceid", "umaban", "fieldsize", "date", "odds"]].copy()
    sub["p"] = p
    wu, pa = load_wu(PAYOUT), load_payout_a(PAYOUT)

    rw, rs = [], []
    for rid, g in sub.groupby("raceid", sort=False):
        n = int(g["fieldsize"].iloc[0])
        r = rc.loc[rid] if rid in rc.index else None
        if r is None:
            continue
        base = {"absent": int(max(r["n_absent"], 0)), "debut": int(r["n_debut"]),
                "year": g["date"].iloc[0].year, "n": n}
        order = {"model": g.sort_values("p", ascending=False, kind="mergesort"),
                 "pop": g.sort_values("odds", ascending=True, kind="mergesort")}
        nums = {k: v["umaban"].astype(int).tolist() for k, v in order.items()}
        w, s3 = wu.get(rid), pa.get(rid)
        if w and w["wakuren"] and len(g) >= 3:
            row = dict(base)
            for k, nm in nums.items():
                cs = wakuren_cs(nm, n)
                row[f"{k}_x"] = sum(w["wakuren"].get(c, 0) for c in cs) / (len(cs) * 100)
            rw.append(row)
        if s3 and s3["sanrenpuku"] and len(g) >= 9:
            row = dict(base)
            for k, nm in nums.items():
                cs = [tuple(sorted(c)) for c in itertools.combinations(nm[:4], 3)]
                row[f"{k}_x"] = sum(s3["sanrenpuku"].get(c, 0) for c in cs) / 400
            rs.append(row)

    rng = np.random.default_rng(0)
    for title, rows in [("枠連 軸枠×紐枠2", rw), ("三連複 BOX上位4", rs)]:
        df = pd.DataFrame(rows)
        m, q = df["model_x"].to_numpy(), df["pop_x"].to_numpy()
        print(f"\n{'='*80}\n=== {title}  {len(df):,}R ===")
        print(f"{'区分':<34}{'R':>7}{'構成比':>7}{'モデル':>9}{'人気順':>9}{'差':>9}{'差の95%CI':>18}")

        def line(lab, sel):
            s = np.asarray(sel)
            if s.sum() < 100:
                return
            mm, qq = m[s], q[s]
            lo, hi = boot(mm - qq, rng)
            print(f"{lab:<34}{s.sum():>7,}{s.mean()*100:>6.1f}%{mm.mean()*100:>8.2f}%"
                  f"{qq.mean()*100:>8.2f}%{(mm-qq).mean()*100:>+8.2f}pt{f'[{lo:+.2f},{hi:+.2f}]':>18}")

        line("全体（現行の報告値）", np.ones(len(df), bool))
        line("★(b)アーカイブ欠損=0（正しい基準）", (df["absent"] == 0).to_numpy())
        line("　うち初出走も0（完全な出走馬）", ((df["absent"] == 0) & (df["debut"] == 0)).to_numpy())
        line("　うち初出走あり", ((df["absent"] == 0) & (df["debut"] > 0)).to_numpy())
        line("(b)アーカイブ欠損 1頭", (df["absent"] == 1).to_numpy())
        line("(b)アーカイブ欠損 2頭以上", (df["absent"] >= 2).to_numpy())
        print(f"  ※初出走の平均頭数 {df['debut'].mean():.2f} / アーカイブ欠損の平均 {df['absent'].mean():.2f}")


if __name__ == "__main__":
    main()
