"""
(67)③で残った未開拓のうち、手元のデータだけで試せる**母（半兄弟の実績）**を検証する。

現状の血統特徴は**父(col43)と母父(col45)だけ**で、**母(col44)は一度も使っていない**。
ただし母をカテゴリにするのは筋が悪い: 21,781頭いて、行の50.4%は「200行未満の母」に乗る。
1頭の母の産駒は平均3.2頭しかいないので、カテゴリコード化すると**ほぼ馬個体ID＝過学習確定**。

そこで `jockey_form`/`trainer_form` と同じ**リーク防止の拡張平均**として渡す。
つまり「**この馬の半兄姉が、それまでに実際どれだけ走ったか**」という別種の情報にする。
血統表に現れない「その牝系が現役でどう機能しているか」を捉えられる可能性がある。

★自分自身の成績を必ず除く: 母の平均に自分の過去走が入ると `avg3_fin` 等の焼き直しになり、
　「母が効いた」のか「自分の近走が効いた」のか区別できなくなる。
　→ 母の累積から**自分の累積を引いて**半兄弟だけの平均を作る。

追加する特徴:
  dam_sib_form … 半兄姉の finratio の拡張平均（自分を除く）
  dam_sib_n    … その時点で半兄姉が走った本数（信頼度。少ないほど平均は当てにならない）

比較は同一テスト集合・同一シードで、AUC と 枠連 軸枠×紐枠2 / 三連複 BOX上位4 のROI。

実行: python3 ml/dam_test.py [シード数(既定3)]
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
from waku_umatan import load_wu, wakuren_buy


def sibling_form(d):
    """母ごとの finratio 拡張平均から**自分の分を差し引いた**「半兄姉だけの平均」と本数。

    日付順に (母の累積 − 自分の累積) / (母の頭数 − 自分の本数) を取る。
    どちらも「そのレースより前」だけを見るのでリークしない。
    """
    t = pd.DataFrame({"date": d["date"], "fr": d["finratio"],
                      "dam": d["dam"].astype(str), "horse": d["horse"].astype(str)})
    t = t.sort_values("date", kind="mergesort")
    gd, gh = t.groupby("dam")["fr"], t.groupby("horse")["fr"]
    dam_sum, dam_n = gd.cumsum() - t["fr"], gd.cumcount()
    own_sum, own_n = gh.cumsum() - t["fr"], gh.cumcount()
    sib_sum, sib_n = dam_sum - own_sum, dam_n - own_n
    form = np.where(sib_n > 0, sib_sum / sib_n.replace(0, np.nan), np.nan)
    return (pd.Series(form, index=t.index).reindex(d.index),
            sib_n.reindex(d.index).astype(float))


def evaluate(p, sub, wu, pa, rng):
    s = sub.copy()
    s["p"] = p
    rw, rs = [], []
    for rid, g in s.groupby("raceid", sort=False):
        gg = g.sort_values("p", ascending=False, kind="mergesort")
        nums = gg["umaban"].astype(int).tolist()
        n = int(gg["fieldsize"].iloc[0])
        w = wu.get(rid)
        if w and w["wakuren"] and len(nums) >= 3:
            cs = wakuren_buy(nums, n, 2)
            rw.append(sum(w["wakuren"].get(c, 0) for c in cs) / (len(cs) * 100))
        q = pa.get(rid)
        if q and q["sanrenpuku"] and len(nums) >= 9:
            cs = [tuple(sorted(c)) for c in itertools.combinations(nums[:4], 3)]
            rs.append(sum(q["sanrenpuku"].get(c, 0) for c in cs) / 400)
    xw, xs = np.array(rw), np.array(rs)
    lo, hi = boot(xw, rng, 1500)
    return xw.mean() * 100, (xw > 0).mean() * 100, (lo, hi), xs.mean() * 100


def main():
    n_seed = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    raw = F.load_files()
    d = F.to_model(raw)
    # to_model は母を持たないのでここで付ける（col44=母名）
    key = raw[40].str[:-2] + "|" + raw[37].str.strip()
    dam_map = pd.Series(raw[44].str.strip().to_numpy(), index=key.to_numpy())
    dam_map = dam_map[~dam_map.index.duplicated()]
    d["dam"] = (d["raceid"] + "|" + d["horse"]).map(dam_map).fillna("")

    sib_form, sib_n = sibling_form(d)
    f = F.build_features(d)
    keep = (f["n_prior"] >= 1) & d["odds"].notna() & (d["odds"] > 0)
    d, f = d[keep].reset_index(drop=True), f[keep].reset_index(drop=True)
    sib_form, sib_n = sib_form[keep].reset_index(drop=True), sib_n[keep].reset_index(drop=True)
    print(f"母が取れた行 {(d['dam'] != '').mean()*100:.1f}%  "
          f"半兄姉の実績がある行 {sib_form.notna().mean()*100:.1f}%  "
          f"半兄姉の本数の中央値 {sib_n[sib_n > 0].median():.0f}")

    y = (d["finish"] <= 3).astype(int).to_numpy()
    odds = d["odds"].to_numpy(float)
    rid = d["raceid"].to_numpy()
    inv = 1.0 / odds
    fx0, _ = F.encode_categoricals(f)
    fx0["log_odds"] = np.log(odds)
    fx0["mkt_prob"] = inv / pd.Series(inv).groupby(rid).transform("sum").to_numpy()
    fx_dam = fx0.copy()
    fx_dam["dam_sib_form"] = sib_form.to_numpy()
    fx_dam["dam_sib_n"] = sib_n.to_numpy()

    cut = d["date"].quantile(0.3)
    tr, te = (d["date"] < cut).to_numpy(), (d["date"] >= cut).to_numpy()
    wu, pa = load_wu("data/payout/a.csv"), load_payout_a("data/payout/a.csv")
    sub = d.loc[te, ["raceid", "umaban"]].copy()
    sub["fieldsize"] = f.loc[te, "fieldsize"].to_numpy()
    rng = np.random.default_rng(0)
    print(f"train {tr.sum():,} / test {te.sum():,}\n")

    print(f"{'構成':<22}{'AUC':>8}{'枠連ROI':>9}{'的中率':>8}{'95%CI':>13}{'BOX4 ROI':>10}")
    store = {}
    for name, fxx in [("現行（基準）", fx0), ("+ 母の半兄弟実績", fx_dam)]:
        ps, aucs = [], []
        for s in range(n_seed):
            m = lgb.LGBMClassifier(random_state=s, **PARAMS)
            m.fit(fxx[tr], y[tr], categorical_feature=F.CAT_COLS)
            p = m.predict_proba(fxx[te])[:, 1]
            ps.append(p)
            aucs.append(roc_auc_score(y[te], p))
        p = np.mean(ps, axis=0)
        store[name] = m
        roi, hit, (lo, hi), roi_s = evaluate(p, sub, wu, pa, rng)
        print(f"{name:<22}{np.mean(aucs):>8.4f}{roi:>8.1f}%{hit:>7.1f}%"
              f"{f'[{lo:.0f},{hi:.0f}]':>13}{roi_s:>9.1f}%")

    m = store["+ 母の半兄弟実績"]
    imp = pd.Series(m.feature_importances_, index=m.feature_name_)
    imp = imp / imp.sum() * 100
    r = imp.rank(ascending=False)
    for c in ["dam_sib_form", "dam_sib_n"]:
        print(f"  {c}: 重要度 {imp[c]:.2f}%（{int(r[c])}位/{len(imp)}）")


if __name__ == "__main__":
    main()
