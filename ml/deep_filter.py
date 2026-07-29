"""
「もっと絞れるか」に答える。絞れるが、**絞るほど検証が不可能になる**ことを同時に数字で出す。

(34)のC(A&3勝-OP)は141%に見えて、正体は228万円の払戻1本（全払戻の54%）だった。
＝絞り込みの評価には ROI だけでなく **①払戻の集中度 ②その標本で必要な検証レース数** が要る。
本スクリプトは条件ごとに次を出す:
  ・ROI（複数シードの平均と幅）と前後半
  ・最高配当1本が全払戻に占める割合 ／ その1本を除いたROI  ← 宝くじ検出
  ・1レースあたり収益率のSD → 95%CI半幅、および「±10ptの精度に必要なR数」と年数
    （年約 R/10年 で割って算出。ここが数十〜数百年になれば、その条件は原理的に検証不能）

事前宣言（2026-07-29・(33)(36)で最も安定だった条件のみ使用。事後に追加しない）:
  D: 16頭〜 & 継続騎乗 & 斤量56.5kg〜
  E: D & 〜1200m

実行: python3 ml/deep_filter.py [シード数(既定3)]
"""
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, "ml")
import features as F
from himo_sweep import POINTS, KEY, hits
from pocket_eval import load_payout_a

TNAME, NHIMO = "三連単マルチ", 7
SPLIT_YEAR = 2022
YEARS = 10  # OOSの年数（2017-2026）


def conds(df):
    big = df["fieldsize"] >= 16
    keep_j = ~df["jockey_changed"].astype(bool)
    heavy = df["wtcarry"] >= 56.5
    sprint = df["distance"] <= 1200
    a = big & keep_j
    dd = a & heavy
    return {
        "全体": pd.Series(True, index=df.index),
        "A: 16頭〜&継続": a,
        "B: A&〜1200m": a & sprint,
        "D: A&斤量56.5〜": dd,
        "E: D&〜1200m": dd & sprint,
    }


def main():
    n_seed = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    d = F.to_model(F.load_files())
    f = F.build_features(d)
    keep = f["n_prior"] >= 1
    d, f = d[keep].reset_index(drop=True), f[keep].reset_index(drop=True)
    d["prev_jockey"] = d.groupby("horse")["jockey"].shift(1)
    d["jockey_changed"] = (d["prev_jockey"].notna()) & (d["jockey"] != d["prev_jockey"])
    y = (d["finish"] == 1).astype(int).to_numpy()
    cut = d["date"].quantile(0.3)
    tr, te = (d["date"] < cut).to_numpy(), (d["date"] >= cut).to_numpy()
    fx, _ = F.encode_categoricals(f)
    pays = load_payout_a("data/payout/a.csv")
    meta = d.loc[te, ["raceid", "umaban", "jockey_changed"]].copy()
    for c in ["fieldsize", "distance", "wtcarry"]:
        meta[c] = f.loc[te, c].to_numpy()
    print(f"train {tr.sum():,} / test {te.sum():,}  シード{n_seed}本  戦略={TNAME}×紐{NHIMO}")

    pts = POINTS[TNAME](NHIMO) * 100
    acc = defaultdict(list)
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
            rows.append({"year": p["date"].year, "pay": hits(p[KEY[TNAME]], nums[0], set(nums[1:NHIMO + 1])),
                         "fieldsize": a["fieldsize"], "distance": a["distance"],
                         "wtcarry": a["wtcarry"], "jockey_changed": a["jockey_changed"]})
        df = pd.DataFrame(rows)
        for name, mask in conds(df).items():
            sub = df[mask.to_numpy()]
            if sub.empty:
                continue
            r = sub["pay"].to_numpy(float) / pts          # 1レースあたり回収率(1.0=元返し)
            top = np.sort(r)[::-1]
            acc[name].append({
                "R": len(sub), "roi": r.mean() * 100, "sd": r.std(ddof=1),
                "hits": int((r > 0).sum()),
                "top1": top[0] / r.sum() * 100 if r.sum() > 0 else 0,
                "ex1": (r.sum() - top[0]) / len(r) * 100,
                "r1": sub.loc[sub["year"] < SPLIT_YEAR, "pay"].mean() / pts * 100,
                "r2": sub.loc[sub["year"] >= SPLIT_YEAR, "pay"].mean() / pts * 100,
            })
        print(f"  seed {seed} 完了")

    print("\n" + "=" * 96)
    print(f"★深掘り絞り込み  {TNAME}×紐{NHIMO}  （宝くじ検出＋必要標本を併記）")
    print("=" * 96)
    hdr = (f"{'条件':<18}{'R':>6}{'ROI':>7}{'幅':>5}{'前半':>7}{'後半':>7}"
           f"{'的中':>5}{'最高1本':>7}{'除くと':>7}{'95%CI半幅':>10}{'±10ptに必要':>12}")
    print(hdr)
    for name in ["全体", "A: 16頭〜&継続", "B: A&〜1200m", "D: A&斤量56.5〜", "E: D&〜1200m"]:
        v = acc.get(name)
        if not v:
            continue
        roi = np.array([x["roi"] for x in v])
        sd = np.mean([x["sd"] for x in v])
        R = int(np.mean([x["R"] for x in v]))
        half = 1.96 * sd / np.sqrt(R) * 100
        need = int((1.96 * sd / 0.10) ** 2)
        yrs = need / (R / YEARS)
        print(f"{name:<18}{R:>6}{roi.mean():>6.1f}%{np.ptp(roi):>5.1f}"
              f"{np.mean([x['r1'] for x in v]):>6.1f}%{np.mean([x['r2'] for x in v]):>6.1f}%"
              f"{int(np.mean([x['hits'] for x in v])):>5}{np.mean([x['top1'] for x in v]):>6.0f}%"
              f"{np.mean([x['ex1'] for x in v]):>6.1f}%{half:>9.1f}pt{need:>7,}R({yrs:.0f}年)")
    print("\n『±10ptに必要』= 回収率を±10ptの精度で知るのに要するレース数と、その条件の年間出現数で割った年数。")
    print("ここが数十年を超える条件は、当たっているかを人生の時間内に確かめられない＝実質的に検証不能。")


if __name__ == "__main__":
    main()
