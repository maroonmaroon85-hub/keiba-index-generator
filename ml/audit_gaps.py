"""
★全検証の見直しで見つかった「抜け」を実測する。

見つかった抜けは4つ:
  A. **馬番・枠番が特徴量に一度も入っていない**。42特徴のどこにも無い。
     内外の有利不利は競馬の基本で、しかも**本命の買い方が枠連**なのに枠を見ていない。
  B. **予測時に馬体重が欠損する**。DGに馬体重が無いので `bodywt`/`bodywt_change` はNaNになる。
     学習時には有る特徴なので train/serve 不一致だが、その影響を**測っていない**((64)③で「不利」と
     書いたが未測定）。
  C. **障害レースが混入している**（31,390行・1,479R・全体の2.1%）。`_classcode` は障害を認識せず、
     未勝利1/オープン5/G1 9 として平地と同じ扱いにしている。
  D. **学習量の違いが未検証**。検証は「前30%で学習」だが、`train_prod.py` は**全期間で学習**して保存する。
     (24)(52)(58)(63)で「モデルを良くするとROIが下がる」が5回起きているので、
     **学習量を増やすとROIが下がらない保証は無い**。

比較は同一のテスト集合・同一シードで行い、指標は AUC と 枠連 軸枠×紐枠2 / 三連複 BOX上位4 のROI。

実行: python3 ml/audit_gaps.py [シード数(既定3)]
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
from waku_umatan import load_wu, waku_of, wakuren_buy


def evaluate(p, sub, wu, pa, rng):
    """予測確率 → 枠連/三連複BOX4 の (ROI, 的中率)。"""
    s = sub.copy()
    s["p"] = p
    rows_w, rows_s = [], []
    for rid, g in s.groupby("raceid", sort=False):
        gg = g.sort_values("p", ascending=False, kind="mergesort")
        nums = gg["umaban"].astype(int).tolist()
        n = int(gg["fieldsize"].iloc[0])
        w = wu.get(rid)
        if w and w["wakuren"] and len(nums) >= 3:
            cs = wakuren_buy(nums, n, 2)
            rows_w.append(sum(w["wakuren"].get(c, 0) for c in cs) / (len(cs) * 100))
        q = pa.get(rid)
        if q and q["sanrenpuku"] and len(nums) >= 9:
            cs = [tuple(sorted(c)) for c in itertools.combinations(nums[:4], 3)]
            rows_s.append(sum(q["sanrenpuku"].get(c, 0) for c in cs) / 400)
    xw, xs = np.array(rows_w), np.array(rows_s)
    lo, hi = boot(xw, rng, 1500)
    return xw.mean() * 100, (xw > 0).mean() * 100, (lo, hi), xs.mean() * 100


def main():
    n_seed = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    d = F.to_model(F.load_files())
    f = F.build_features(d)
    keep = (f["n_prior"] >= 1) & d["odds"].notna() & (d["odds"] > 0)
    d, f = d[keep].reset_index(drop=True), f[keep].reset_index(drop=True)
    y = (d["finish"] <= 3).astype(int).to_numpy()
    odds = d["odds"].to_numpy(float)
    rid = d["raceid"].to_numpy()
    inv = 1.0 / odds
    fx0, _ = F.encode_categoricals(f)
    fx0["log_odds"] = np.log(odds)
    fx0["mkt_prob"] = inv / pd.Series(inv).groupby(rid).transform("sum").to_numpy()

    # A. 枠番・馬番
    n_arr = f["fieldsize"].to_numpy(int)
    uma = d["umaban"].to_numpy(int)
    fx_pos = fx0.copy()
    fx_pos["umaban"] = uma
    fx_pos["waku"] = [waku_of(u, n) for u, n in zip(uma, n_arr)]
    fx_pos["pos_ratio"] = uma / n_arr          # 内(0)〜外(1)。頭数で正規化した相対位置

    # B. 馬体重を落とす（予測時の欠損を再現）
    fx_nowt = fx0.copy()
    fx_nowt["bodywt"] = np.nan
    fx_nowt["bodywt_change"] = np.nan

    # C. 障害レースを除く（レース名で判定。_classcode は障害を認識していない）
    raw_names = F.load_files()[7].str.strip()
    # to_model と同じ絞り込みを再現して行を対応させる
    jr = F.load_files()
    jr = jr[jr[40].str.len() > 2]
    nm = pd.Series(jr[7].str.strip().to_numpy(), index=jr[40].str[:-2].to_numpy())
    is_sho = nm.groupby(level=0).first().str.contains("障", na=False)
    flat = ~pd.Series(rid).map(is_sho).fillna(False).to_numpy()

    cut = d["date"].quantile(0.3)
    tr, te = (d["date"] < cut).to_numpy(), (d["date"] >= cut).to_numpy()
    wu, pa = load_wu("data/payout/a.csv"), load_payout_a("data/payout/a.csv")
    sub = d.loc[te, ["raceid", "umaban"]].copy()
    sub["fieldsize"] = f.loc[te, "fieldsize"].to_numpy()
    rng = np.random.default_rng(0)
    print(f"train {tr.sum():,} / test {te.sum():,}  障害 {(~flat).sum():,}行\n")

    CONF = [
        ("現行（基準）", fx0, tr, te),
        ("A. +馬番/枠番/相対位置", fx_pos, tr, te),
        ("B. 馬体重なし(予測時再現)", fx_nowt, tr, te),
        ("C. 障害を除いて学習", fx0, tr & flat, te),
        ("D. 学習を全期間の8割に増量", fx0, (d["date"] < d["date"].quantile(0.8)).to_numpy(),
         (d["date"] >= d["date"].quantile(0.8)).to_numpy()),
    ]
    print(f"{'構成':<26}{'AUC':>8}{'枠連ROI':>9}{'的中率':>8}{'95%CI':>13}{'BOX4 ROI':>10}{'R数':>8}")
    for name, fxx, trm, tem in CONF:
        ps = []
        for s in range(n_seed):
            m = lgb.LGBMClassifier(random_state=s, **PARAMS)
            m.fit(fxx[trm], y[trm], categorical_feature=F.CAT_COLS)
            ps.append(m.predict_proba(fxx[tem])[:, 1])
        p = np.mean(ps, axis=0)
        auc = roc_auc_score(y[tem], p)
        sb = d.loc[tem, ["raceid", "umaban"]].copy()
        sb["fieldsize"] = f.loc[tem, "fieldsize"].to_numpy()
        roi, hit, (lo, hi), roi_s = evaluate(p, sb, wu, pa, rng)
        print(f"{name:<26}{auc:>8.4f}{roi:>8.1f}%{hit:>7.1f}%{f'[{lo:.0f},{hi:.0f}]':>13}"
              f"{roi_s:>9.1f}%{sb['raceid'].nunique():>8,}")
    print("\n※D はテスト期間が変わるので他と直接は比べられない（学習量を増やしても壊れないかの確認用）。")


if __name__ == "__main__":
    main()
