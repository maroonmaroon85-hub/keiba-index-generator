"""C2: ★(69)②「差は設定次第で0になる」を、訂正後の評価設計で測り直す。

(69)②は本プロジェクト最大の留保:
  「人気順に対する差は leaves/n_estimators を変えると **−0.04〜+5.72pt** に散らばる。
   (53)の通りどれが良いかは事前に選べないので、**枠連でモデルが市場を上回るとは確定できない**」
**だがこれは単一分割でしか測られていない**。(78)①で分かったとおり単一分割は
実運用より3.3倍少ない学習量のモデルを測っており、券種によって2pt以上ずれる。
**プロジェクト最大の留保が、穴のある評価設計の上に乗っている**ので、測り直す必要がある。

(69)②と同じ4設定を、実運用と同じウォークフォワード（各年をそれ以前の全データで学習）で測る。
特に見たいのは **「単純な設定(leaves15/100本)では差が0になる」が再現するか**。
再現すれば留保はそのまま。消えれば「モデルは人気順に勝つ」の確度が上がる。

実行: python3 ml/audit_config_wf.py [シード数(既定2)]
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
from _cache import load_cached
from place_wide import PARAMS
from pocket_eval import load_payout_a
from waku_umatan import load_wu, waku_of

PAYOUT = "data/payout/a.csv"

CONFIGS = [
    ("現行 leaves31/mc100/400本", dict(PARAMS)),
    ("leaves15/mc30/400本 ((53)内側検証)", dict(PARAMS, num_leaves=15, min_child_samples=30)),
    ("leaves63/mc30/1000本 ((53)後知恵最良)",
     dict(PARAMS, num_leaves=63, min_child_samples=30, n_estimators=1000)),
    ("leaves15/mc100/100本 (単純)", dict(PARAMS, num_leaves=15, n_estimators=100)),
]


def wakuren_cs(nums, n):
    wa = waku_of(nums[0], n)
    return sorted({tuple(sorted((wa, waku_of(h, n)))) for h in nums[1:3]})


def boot(x, rng, n=2000):
    b = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return np.percentile(b, 2.5) * 100, np.percentile(b, 97.5) * 100


def main():
    n_seed = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    d, fx = load_cached()
    y = (d["finish"] <= 3).astype(int).to_numpy()
    year = d["date"].dt.year.to_numpy()
    wu, pa = load_wu(PAYOUT), load_payout_a(PAYOUT)
    years = [yy for yy in range(2016, int(year.max()) + 1) if (year == yy).sum() > 5000]
    print(f"設定4種 × ウォークフォワード（{years[0]}〜{years[-1]}・シード{n_seed}本）")

    out = {name: {"w": [], "s": []} for name, _ in CONFIGS}
    popw, pops = [], []
    for yy in years:
        tr, te = year < yy, year == yy
        sub0 = d.loc[te, ["raceid", "umaban", "fieldsize", "odds"]].copy()
        for ci, (name, par) in enumerate(CONFIGS):
            ps = [lgb.LGBMClassifier(random_state=s, **par)
                  .fit(fx[tr], y[tr], categorical_feature=F.CAT_COLS)
                  .predict_proba(fx[te])[:, 1] for s in range(n_seed)]
            sub0[f"p{ci}"] = np.mean(ps, axis=0)
        for rid, g0 in sub0.groupby("raceid", sort=False):
            w, s3 = wu.get(rid), pa.get(rid)
            okw = w and w["wakuren"] and len(g0) >= 3
            oks = s3 and s3["sanrenpuku"] and len(g0) >= 9
            if not (okw or oks):
                continue
            n = int(g0["fieldsize"].iloc[0])
            uma = g0["umaban"].astype(int).to_numpy()
            po = uma[np.argsort(g0["odds"].to_numpy(float), kind="mergesort")]
            if okw:
                cs = wakuren_cs(po, n)
                popw.append(sum(w["wakuren"].get(c, 0) for c in cs) / (len(cs) * 100))
            if oks:
                cs = [tuple(sorted(c)) for c in itertools.combinations(po[:4], 3)]
                pops.append(sum(s3["sanrenpuku"].get(c, 0) for c in cs) / 400)
            for ci, (name, _) in enumerate(CONFIGS):
                mo = uma[np.argsort(-g0[f"p{ci}"].to_numpy(float), kind="mergesort")]
                if okw:
                    cs = wakuren_cs(mo, n)
                    out[name]["w"].append(sum(w["wakuren"].get(c, 0) for c in cs) / (len(cs) * 100))
                if oks:
                    cs = [tuple(sorted(c)) for c in itertools.combinations(mo[:4], 3)]
                    out[name]["s"].append(sum(s3["sanrenpuku"].get(c, 0) for c in cs) / 400)
        print(f"  {yy} 完了")

    rng = np.random.default_rng(0)
    pw, ps_ = np.array(popw), np.array(pops)
    for tag, key, pop in [("枠連 軸枠×紐枠2", "w", pw), ("三連複 BOX上位4", "s", ps_)]:
        print(f"\n{'='*94}\n=== {tag}  人気順 {pop.mean()*100:.2f}%（設定に依存しない固定値） ===")
        print(f"{'設定':<38}{'モデルROI':>11}{'対人気順':>11}{'95%CI':>18}")
        for name, _ in CONFIGS:
            m = np.array(out[name][key])
            lo, hi = boot(m - pop, rng)
            print(f"{name:<38}{m.mean()*100:>10.2f}%{(m-pop).mean()*100:>+10.2f}pt"
                  f"{f'[{lo:+.2f},{hi:+.2f}]':>18}")
        ds = [np.array(out[n][key]).mean() - pop.mean() for n, _ in CONFIGS]
        print(f"  ★設定による散らばり: {min(ds)*100:+.2f} 〜 {max(ds)*100:+.2f}pt"
              f"（幅 {(max(ds)-min(ds))*100:.2f}pt）")
        print(f"  ※(69)②の単一分割では 枠連 −0.04〜+5.72pt（幅5.76pt）だった")


if __name__ == "__main__":
    main()
