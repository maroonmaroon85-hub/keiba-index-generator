"""
(39)で「未検証」として残した2つの主張を、対比較の正しい誤差で検定する。

対象:
  A. (31)「三連単マルチではモデル順が人気順に +4.9pt 勝つ」（馬連では逆に負ける）
  B. (27)「紐5-7 は 紐3-4 より 6-8pt 良い」

どちらも**同一レース上の比較**なので、条件間の比較((39)で誤った形)とは誤差の性質が違う。
正しくは**レースごとの差 d_i = 回収率A(i) − 回収率B(i)** を取り、その平均の標準誤差を使う。
同じレースの大穴配当は両方に（あるいは両方に等しく）効くため相殺され、
marginalなCI（(39)で使ったもの）よりずっと狭くなる。ここを混同すると
「差があるのに無いと言う」逆方向の誤りを犯すので、分けて検定する必要がある。

実行: python3 ml/paired_test.py [シード数(既定5)]
"""
import sys

import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, "ml")
import features as F
from himo_sweep import POINTS, KEY, hits
from pocket_eval import load_payout_a

TNAME_T, TNAME_U = "三連単マルチ", "馬連 軸流し"


def paired(name, a, b, extra=""):
    """a,b は同一レース上の1レースあたり回収率(1.0=元返し)。差の平均と95%CIを出す。"""
    d = (a - b) * 100
    n = len(d)
    se = d.std(ddof=1) / np.sqrt(n)
    lo, hi = d.mean() - 1.96 * se, d.mean() + 1.96 * se
    sig = "差あり" if (lo > 0 or hi < 0) else "差があるとは言えない"
    print(f"  {name:<34}{d.mean():>+7.2f}pt  95%CI [{lo:+6.2f},{hi:+6.2f}]  {sig}{extra}")
    return d.mean(), lo, hi


def main():
    n_seed = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    d = F.to_model(F.load_files())
    f = F.build_features(d)
    keep = f["n_prior"] >= 1
    d, f = d[keep].reset_index(drop=True), f[keep].reset_index(drop=True)
    y = (d["finish"] == 1).astype(int).to_numpy()
    cut = d["date"].quantile(0.3)
    tr, te = (d["date"] < cut).to_numpy(), (d["date"] >= cut).to_numpy()
    fx, _ = F.encode_categoricals(f)
    pays = load_payout_a("data/payout/a.csv")
    meta = d.loc[te, ["raceid", "umaban", "odds"]].copy()
    print(f"train {tr.sum():,} / test {te.sum():,}  シード{n_seed}本")

    # 人気順（単勝オッズ昇順）はシードに依存しない
    market = {}
    for rid, g in meta.groupby("raceid", sort=False):
        g = g.sort_values("odds", ascending=True, kind="mergesort")
        market[rid] = g["umaban"].astype(int).tolist()

    res_a, res_b = [], []
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
            if p is None or not p["sanrentan"] or len(g) < 9:
                continue
            g = g.sort_values("prob", ascending=False, kind="mergesort")
            ml = g["umaban"].astype(int).tolist()
            mk = market[rid]
            r = {}
            for n in [3, 4, 5, 6, 7, 8]:
                r[f"ml_t{n}"] = hits(p[KEY[TNAME_T]], ml[0], set(ml[1:n + 1])) / (POINTS[TNAME_T](n) * 100)
            r["mk_t7"] = hits(p[KEY[TNAME_T]], mk[0], set(mk[1:8])) / (POINTS[TNAME_T](7) * 100)
            r["ml_u6"] = hits(p[KEY[TNAME_U]], ml[0], set(ml[1:7])) / (POINTS[TNAME_U](6) * 100)
            r["mk_u6"] = hits(p[KEY[TNAME_U]], mk[0], set(mk[1:7])) / (POINTS[TNAME_U](6) * 100)
            rows.append(r)
        df = pd.DataFrame(rows)

        print(f"\n=== seed {seed}  ({len(df):,}R) ===")
        print("■ A: モデル順 vs 人気順（同一レース対比較）")
        a1 = paired("三連単マルチ×紐7  モデル−人気", df["ml_t7"].to_numpy(), df["mk_t7"].to_numpy())
        a2 = paired("馬連 軸流し×紐6   モデル−人気", df["ml_u6"].to_numpy(), df["mk_u6"].to_numpy())
        print("■ B: 紐の本数（同一レース対比較・三連単マルチ）")
        b1 = paired("紐7 − 紐4", df["ml_t7"].to_numpy(), df["ml_t4"].to_numpy())
        b2 = paired("紐7 − 紐3", df["ml_t7"].to_numpy(), df["ml_t3"].to_numpy())
        b3 = paired("紐5 − 紐4", df["ml_t5"].to_numpy(), df["ml_t4"].to_numpy())
        res_a.append((a1[0], a2[0]))
        res_b.append((b1[0], b2[0], b3[0]))

    ra, rb = np.array(res_a), np.array(res_b)
    print("\n" + "=" * 74)
    print("★シード間の安定性（各差分のシード平均と幅）")
    print("=" * 74)
    for i, lab in enumerate(["三連単 モデル−人気", "馬連 モデル−人気"]):
        print(f"  {lab:<24}平均{ra[:, i].mean():+6.2f}pt  幅{np.ptp(ra[:, i]):5.2f}pt")
    for i, lab in enumerate(["三連単 紐7−紐4", "三連単 紐7−紐3", "三連単 紐5−紐4"]):
        print(f"  {lab:<24}平均{rb[:, i].mean():+6.2f}pt  幅{np.ptp(rb[:, i]):5.2f}pt")


if __name__ == "__main__":
    main()
