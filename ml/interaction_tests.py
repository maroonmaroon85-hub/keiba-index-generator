"""
交互作用の検定 第2弾。(43)と同じ設計で、事前に宣言した3つの仮説を測る。

総当たりは(38)で否定済みなので、仮説ごとに**1つの対比**に落とし込む。事後に軸を追加しない。

事前宣言（2026-07-29）:
  A. 距離適性 × 延長/短縮
     父の長距離適性 = (産駒の1800m以上での過去平均finratio) − (1600m以下での過去平均)  ※リーク防止
     検定 = [距離延長時の 適性高−低] − [距離短縮時の 適性高−低]
     仮説: 距離が延びる時、その馬が持つかどうかを市場は判断しづらく、血統の適性が過小評価される
  B. コース適性 × 開催場
     父のその場適性 = (産駒のその開催場での過去平均finratio) − (産駒全体の過去平均)  ※父の地力を差し引く
     検定 = ROI(その場適性 高) − ROI(低)   ※既に父の地力で差分化済みなので2x2は不要
     仮説: 「この父はこの競馬場が得意」が市場に織り込まれていない
  C. 脚質 × 頭数
     脚質 = 軸馬の近3走の平均通過順位比(avg3_passratio)。小さい=先行、大きい=差し追込
     検定 = [少頭数での 先行−差し] − [多頭数での 先行−差し]
     仮説: 少頭数は隊列が短く前が止まりにくいので先行有利、多頭数は差しが届く

判定は(41)以降の基準どおり、レース単位ブートストラップ＋シード平均。
(43)で分かったとおり、この種の交互作用は効果+6pt程度に対しCIが±15ptになりがちなので、
**CIが0をまたいだら「確定できない」であって「無い」ではない**ことに注意して読む。

実行: python3 ml/interaction_tests.py [シード数(既定5)]
"""
import sys

import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, "ml")
import features as F
from himo_sweep import POINTS, KEY, hits
from pocket_eval import load_payout_a

TN, NHIMO = "三連単マルチ", 5
MIN_N = 50
N_BOOT = 10000


def prior_mean(d, keys, weight=None, min_n=MIN_N):
    """keys で決まるグループの finratio を、そのレースより前だけで平均する（自分は除く）。
    weight を渡すと weight=1 の行だけを対象にする。標本が min_n 未満は NaN。"""
    k = d[keys[0]].astype(str)
    for c in keys[1:]:
        k = k + "|" + d[c].astype(str)
    w = pd.Series(1.0, index=d.index) if weight is None else weight.astype(float)
    t = pd.DataFrame({"k": k, "fr": d["finratio"] * w, "w": w, "date": d["date"]})
    t = t.sort_values("date", kind="mergesort")
    num = t["fr"].groupby(t["k"]).cumsum() - t["fr"]
    den = t["w"].groupby(t["k"]).cumsum() - t["w"]
    out = (num / den.where(den >= min_n))
    return out.reindex(d.index)


def boot(arrs, rng):
    ms = []
    for a in arrs:
        idx = rng.integers(0, len(a), size=(N_BOOT, len(a)))
        ms.append(a[idx].mean(axis=1))
    return ms


def show_interaction(name, a_hi, a_lo, b_hi, b_lo, rng, la, lb):
    """[条件Aでの 高−低] − [条件Bでの 高−低] を検定。"""
    if min(len(x) for x in [a_hi, a_lo, b_hi, b_lo]) < 100:
        print(f"  {name}: 標本不足のため省略")
        return
    print(f"{'':<12}{'高':>16}{'低':>16}{'差':>10}")
    print(f"{la:<12}{a_hi.mean():>13.1f}%({len(a_hi):>5}R){a_lo.mean():>12.1f}%({len(a_lo):>5}R)"
          f"{a_hi.mean()-a_lo.mean():>+9.2f}pt")
    print(f"{lb:<12}{b_hi.mean():>13.1f}%({len(b_hi):>5}R){b_lo.mean():>12.1f}%({len(b_lo):>5}R)"
          f"{b_hi.mean()-b_lo.mean():>+9.2f}pt")
    ms = boot([a_hi, a_lo, b_hi, b_lo], rng)
    b = (ms[0] - ms[1]) - (ms[2] - ms[3])
    inter = (a_hi.mean() - a_lo.mean()) - (b_hi.mean() - b_lo.mean())
    lo, hi = np.percentile(b, 2.5), np.percentile(b, 97.5)
    v = "★差あり" if (lo > 0 or hi < 0) else "0をまたぐ＝確定できない"
    print(f"  交互作用 {inter:+.2f}pt  95%CI[{lo:+.2f},{hi:+.2f}]  P(>0)={float((b>0).mean()):.3f}  {v}")


def show_contrast(name, hi, lo, rng):
    if min(len(hi), len(lo)) < 100:
        print(f"  {name}: 標本不足のため省略")
        return
    ms = boot([hi, lo], rng)
    b = ms[0] - ms[1]
    dlo, dhi = np.percentile(b, 2.5), np.percentile(b, 97.5)
    v = "★差あり" if (dlo > 0 or dhi < 0) else "0をまたぐ＝確定できない"
    print(f"  適性高 {hi.mean():.1f}%({len(hi)}R)  適性低 {lo.mean():.1f}%({len(lo)}R)  "
          f"差 {hi.mean()-lo.mean():+.2f}pt  95%CI[{dlo:+.2f},{dhi:+.2f}]  {v}")


def main():
    n_seed = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    d = F.to_model(F.load_files())
    f = F.build_features(d)
    keep = f["n_prior"] >= 1
    d, f = d[keep].reset_index(drop=True), f[keep].reset_index(drop=True)

    print("血統の適性指標を算出中（リーク防止の累積平均）…")
    longm = (d["distance"] >= 1800).astype(float)
    shortm = (d["distance"] <= 1600).astype(float)
    d["sire_long"] = prior_mean(d, ["sire"], longm) - prior_mean(d, ["sire"], shortm)
    d["sire_course"] = prior_mean(d, ["sire", "course"]) - prior_mean(d, ["sire"])
    d["dist_change"] = f["dist_change"].to_numpy()
    d["pass3"] = f["avg3_passratio"].to_numpy()
    d["fieldsize"] = f["fieldsize"].to_numpy()
    print(f"  長距離適性 {d['sire_long'].notna().mean()*100:.1f}%  "
          f"コース適性 {d['sire_course'].notna().mean()*100:.1f}%  "
          f"脚質 {d['pass3'].notna().mean()*100:.1f}%")

    y = (d["finish"] == 1).astype(int).to_numpy()
    cut = d["date"].quantile(0.3)
    tr, te = (d["date"] < cut).to_numpy(), (d["date"] >= cut).to_numpy()
    fx, _ = F.encode_categoricals(f)
    pays = load_payout_a("data/payout/a.csv")
    cols = ["raceid", "umaban", "sire_long", "sire_course", "dist_change", "pass3", "fieldsize"]
    meta = d.loc[te, cols].copy()
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
            a = g.iloc[0].to_dict()
            a["pay"] = hits(p[KEY[TN]], nums[0], set(nums[1:NHIMO + 1]))
            rows.append(a)
        acc.append(pd.DataFrame(rows))
        print(f"  seed {seed} 完了")

    base = acc[0].copy()
    base["pay"] = np.mean([x["pay"].to_numpy(float) for x in acc], axis=0)
    base["roi"] = base["pay"] / pts * 100
    rng = np.random.default_rng(0)

    def arr(mask):
        return base.loc[mask, "roi"].to_numpy(float)

    print("\n" + "=" * 78)
    print("★A. 距離適性 × 距離の延長/短縮")
    print("=" * 78)
    s = base[base["sire_long"].notna() & base["dist_change"].notna()]
    med = s["sire_long"].median()
    hi_m = s["sire_long"] > med
    ext = s["dist_change"] > 0
    shr = s["dist_change"] < 0
    g = lambda m: s.loc[m, "roi"].to_numpy(float)
    show_interaction("A", g(ext & hi_m), g(ext & ~hi_m), g(shr & hi_m), g(shr & ~hi_m),
                     rng, "距離延長時", "距離短縮時")

    print("\n" + "=" * 78)
    print("★B. コース適性（父のその場適性・父の地力は差し引き済み）")
    print("=" * 78)
    s = base[base["sire_course"].notna()]
    med = s["sire_course"].median()
    show_contrast("B", s.loc[s["sire_course"] > med, "roi"].to_numpy(float),
                  s.loc[s["sire_course"] <= med, "roi"].to_numpy(float), rng)

    print("\n" + "=" * 78)
    print("★C. 脚質 × 頭数")
    print("=" * 78)
    s = base[base["pass3"].notna()]
    med = s["pass3"].median()
    front = s["pass3"] <= med          # 通過順位が前＝先行
    small = s["fieldsize"] <= 12
    big = s["fieldsize"] >= 16
    g = lambda m: s.loc[m, "roi"].to_numpy(float)
    show_interaction("C", g(small & front), g(small & ~front), g(big & front), g(big & ~front),
                     rng, "少頭数(〜12)", "多頭数(16〜)")


if __name__ == "__main__":
    main()
