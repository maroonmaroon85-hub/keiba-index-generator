"""
血統 × 馬場状態。「道悪が得意な血統は道悪で過小評価されている」を1つの検定にする。

素直に 父(1,225種) × 馬場(8区分) を総当たりすると数千セルになり、(38)のとおり
ランダムデータでも260%のセルが出るので何も言えない。そこで仮説を交互作用の検定に落とす:

  父ごとの「道悪適性」= (その父の産駒の 稍/重/不 での過去平均着順比) − (良での過去平均着順比)
  ※リーク防止: そのレースより前のデータだけで算出（日付順に累積し自分の結果を引く）

  検定する交互作用:
    [道悪適性・高 の馬を道悪で買ったROI − 道悪適性・低 の馬を道悪で買ったROI]
  − [道悪適性・高 の馬を良馬場で買ったROI − 道悪適性・低 の馬を良馬場で買ったROI]

  血統の道悪適性が市場に織り込まれていなければ、この差は正になるはず。
  適性そのものの効果（良馬場でも強い等）は引き算で消えるので、**道悪特有の上乗せだけ**が残る。

これなら検定は1回（＋父/母父の2通り）で、多重検定の問題が起きない。
併せて記述統計として 適性4分位 × 馬場 の表も出す（判断はあくまで上の交互作用で行う）。

実行: python3 ml/pedigree_going.py [シード数(既定5)]
"""
import sys

import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, "ml")
import features as F
from himo_sweep import POINTS, KEY, hits
from pocket_eval import load_payout_a

TN, NHIMO = "三連単マルチ", 5   # (42)より紐5を採用
MIN_N = 50                      # 適性を算出するのに必要な最低の過去出走数
N_BOOT = 10000


def wet_lift(d, keycol, min_n=MIN_N):
    """血統ごとの道悪適性を、そのレースより前のデータだけで算出する（リーク防止）。
    lift = 道悪での過去平均finratio − 良馬場での過去平均finratio。標本不足はNaN。"""
    t = pd.DataFrame({"k": d[keycol].astype(str), "fr": d["finratio"],
                      "wet": d["cond"].isin(["稍", "重", "不"]).astype(float),
                      "date": d["date"]}).sort_values("date", kind="mergesort")
    out = pd.Series(np.nan, index=t.index)
    # 道悪/良 それぞれの「自分より前」の累積平均
    res = {}
    for tag, w in [("wet", t["wet"]), ("dry", 1 - t["wet"])]:
        num = (t["fr"] * w).groupby(t["k"]).cumsum() - t["fr"] * w
        den = w.groupby(t["k"]).cumsum() - w
        res[tag] = (num / den.where(den >= min_n), den)
    lift = res["wet"][0] - res["dry"][0]
    lift[(res["wet"][1] < min_n) | (res["dry"][1] < min_n)] = np.nan
    out.loc[lift.index] = lift
    return out.reindex(d.index)


def boot_ci(x, rng, n_boot=N_BOOT):
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    m = x[idx].mean(axis=1)
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


def interaction(sub, pts, rng, label):
    """交互作用 (道悪での高−低) − (良での高−低) をブートストラップで検定する。
    適性そのものの効果（良馬場でも強い等）は引き算で消え、**道悪特有の上乗せだけ**が残る。"""
    arrs = []
    for wet in [True, False]:
        for hi in [True, False]:
            s = sub[(sub["wet"] == wet) & (sub["hi"] == hi)]
            arrs.append(s["pay"].to_numpy(float) / pts * 100)
    if min(len(a) for a in arrs) < 100:
        print(f"  {label:<14} 標本不足のため省略（最小群 {min(len(a) for a in arrs)}R）")
        return
    ms = []
    for a in arrs:
        idx = rng.integers(0, len(a), size=(N_BOOT, len(a)))
        ms.append(a[idx].mean(axis=1))
    b = (ms[0] - ms[1]) - (ms[2] - ms[3])
    inter = (arrs[0].mean() - arrs[1].mean()) - (arrs[2].mean() - arrs[3].mean())
    lo, hi_ = np.percentile(b, 2.5), np.percentile(b, 97.5)
    v = "★差あり" if (lo > 0 or hi_ < 0) else "0をまたぐ＝織り込み済み"
    print(f"  {label:<14}交互作用 {inter:>+7.2f}pt  95%CI[{lo:+7.2f},{hi_:+7.2f}]  "
          f"P(>0)={float((b > 0).mean()):.3f}  {v}")


def main():
    n_seed = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    d = F.to_model(F.load_files())
    f = F.build_features(d)
    keep = f["n_prior"] >= 1
    d, f = d[keep].reset_index(drop=True), f[keep].reset_index(drop=True)

    print("血統の道悪適性を算出中（リーク防止の累積平均）…")
    d["sire_lift"] = wet_lift(d, "sire")
    d["damsire_lift"] = wet_lift(d, "damsire")
    print(f"  父の適性が付いた行: {d['sire_lift'].notna().mean()*100:.1f}%  "
          f"母父: {d['damsire_lift'].notna().mean()*100:.1f}%")

    y = (d["finish"] == 1).astype(int).to_numpy()
    cut = d["date"].quantile(0.3)
    tr, te = (d["date"] < cut).to_numpy(), (d["date"] >= cut).to_numpy()
    fx, _ = F.encode_categoricals(f)
    pays = load_payout_a("data/payout/a.csv")
    meta = d.loc[te, ["raceid", "umaban", "cond", "sire_lift", "damsire_lift"]].copy()
    pts = POINTS[TN](NHIMO) * 100
    print(f"train {tr.sum():,} / test {te.sum():,}  シード{n_seed}本  戦略={TN}×紐{NHIMO}")

    acc = []
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
            if p is None or not p["sanrentan"] or len(g) <= NHIMO:
                continue
            g = g.sort_values("prob", ascending=False, kind="mergesort")
            nums = g["umaban"].astype(int).tolist()
            a = g.iloc[0]
            rows.append({"pay": hits(p[KEY[TN]], nums[0], set(nums[1:NHIMO + 1])),
                         "cond": a["cond"], "sire_lift": a["sire_lift"],
                         "damsire_lift": a["damsire_lift"], "surface": p["surface"],
                         "year": p["date"].year})
        acc.append(pd.DataFrame(rows))
        print(f"  seed {seed} 完了")

    base = acc[0].copy()
    base["pay"] = np.mean([x["pay"].to_numpy(float) for x in acc], axis=0)  # シード平均
    base["wet"] = base["cond"].isin(["稍", "重", "不"])
    rng = np.random.default_rng(0)

    # 馬場整備は年で変わる（排水改良・クッション砂等）ので、まず道悪の出現率の推移を確認する
    print("\n■ 年別の道悪(稍/重/不)比率と、その年の道悪レースのROI")
    print(f"{'年':>6}{'R':>7}{'道悪率':>8}{'道悪ROI':>9}{'良ROI':>9}")
    for yr, g in base.groupby("year"):
        w = g[g["wet"]]; dry = g[~g["wet"]]
        rw = w["pay"].mean() / pts * 100 if len(w) else float("nan")
        rd = dry["pay"].mean() / pts * 100 if len(dry) else float("nan")
        print(f"{yr:>6}{len(g):>7}{g['wet'].mean()*100:>7.1f}%{rw:>8.1f}%{rd:>8.1f}%")

    for col, label in [("sire_lift", "父"), ("damsire_lift", "母父")]:
        sub = base[base[col].notna()].copy()
        if len(sub) < 1000:
            print(f"\n{label}: 適性を算出できた行が少なすぎるため省略")
            continue
        med = sub[col].median()
        sub["hi"] = sub[col] > med
        print("\n" + "=" * 76)
        print(f"★{label}の道悪適性 × 馬場   （{TN}×紐{NHIMO} / {len(sub):,}R）")
        print("=" * 76)
        print(f"{'':<14}{'道悪適性 高':>14}{'道悪適性 低':>14}{'差':>10}")
        cells = {}
        for wet, wlab in [(True, "道悪(稍重不)"), (False, "良馬場")]:
            r = {}
            for hi, hlab in [(True, "hi"), (False, "lo")]:
                s = sub[(sub["wet"] == wet) & (sub["hi"] == hi)]
                r[hlab] = (s["pay"].to_numpy(float) / pts * 100, len(s))
            cells[wet] = r
            print(f"{wlab:<14}{r['hi'][0].mean():>11.1f}%({r['hi'][1]:>5}R)"
                  f"{r['lo'][0].mean():>10.1f}%({r['lo'][1]:>5}R)"
                  f"{r['hi'][0].mean()-r['lo'][0].mean():>+9.2f}pt")

        interaction(sub, pts, rng, "全期間")
        # 馬場整備は年で変わる（排水改良・クッション砂等）ため、期間を割って安定性を見る
        for lab, msk in [("前半 2017-21", sub["year"] < 2022), ("後半 2022-26", sub["year"] >= 2022)]:
            interaction(sub[msk], pts, rng, lab)


if __name__ == "__main__":
    main()
