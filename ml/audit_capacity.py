"""C4: ★★「設定の±4ptはノイズではなく単調な構造ではないか」＋ プラセボとの接続。

C2(`audit_config_wf.py`)で、ウォークフォワードでも設定によって対人気順の差が
+0.08〜+5.26pt(枠連) / +0.81〜+7.74pt(三連複) に散らばることを確認した。
**だがその散らばりには向きがある**——4設定を「モデルの容量」順に並べると単調だった:

    leaves15/100本 81.52% < leaves15/400本 83.94% < leaves31/400本(現行) 84.71% < leaves63/1000本 86.70%

(53)は「ROIでは選べない／設定で±4pt動く」としてこれを**不確実性**として扱ってきたが、
単調なら**不確実性ではなく構造**であり、しかも本プロジェクトの中心的知見
「過学習しているモデルの方が儲かる」((63)③)と**同じ向き**。だとすれば現行設定は
まだ坂の途中で、**もっと容量を上げれば上がる**のかもしれない。

★同時に、C1(`audit_placebo.py`)で分かったことと繋げる必要がある:
  **人気順から適度に離れるだけで+1.4pt出る**（中身の無いノイズでも）。
  容量の大きいモデルほど正則化が弱く、**市場順から離れる**はずなので、
  この単調性は**プラセボ効果を辿っているだけ**の可能性がある。

そこで各設定について**同時に**測る:
  (a) ROI と 対人気順の差
  (b) 買い目が人気順と一致する率（＝市場順からの離れ方）
  (c) **同じ一致率になるプラセボ（人気順＋ノイズ）のROI**
  (d) ★ モデル − プラセボ ＝ 容量を上げて増えたぶんが技能なのか、ズレの効果なのか

判定:
  ・モデル−プラセボが容量とともに増える ⇒ 容量を上げると**技能が増える**。現行設定は坂の途中。
  ・モデル−プラセボが一定 ⇒ 容量の効果は**全部ズレの効果**。設定をいじる意味は無い。

実行: python3 ml/audit_capacity.py [シード数(既定2)] [開始年(既定2018)]
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
SIGMAS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0]
N_DRAW = 4

# 容量の梯子（単純 → 複雑）。現行は3段目。
LADDER = [
    ("1 leaves15 /  100本 (最単純)", dict(PARAMS, num_leaves=15, n_estimators=100)),
    ("2 leaves15 /  400本", dict(PARAMS, num_leaves=15, min_child_samples=30)),
    ("3 leaves31 /  400本 ★現行", dict(PARAMS)),
    ("4 leaves63 / 1000本", dict(PARAMS, num_leaves=63, min_child_samples=30, n_estimators=1000)),
    ("5 leaves127/ 2000本 (★未検証)",
     dict(PARAMS, num_leaves=127, min_child_samples=20, n_estimators=2000)),
]


def wakuren_cs(nums, n):
    wa = waku_of(nums[0], n)
    return sorted({tuple(sorted((wa, waku_of(h, n)))) for h in nums[1:3]})


def boot(x, rng, n=2000):
    b = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return np.percentile(b, 2.5) * 100, np.percentile(b, 97.5) * 100


def main():
    n_seed = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    y0 = int(sys.argv[2]) if len(sys.argv) > 2 else 2018
    d, fx = load_cached()
    y = (d["finish"] <= 3).astype(int).to_numpy()
    year = d["date"].dt.year.to_numpy()
    wu, pa = load_wu(PAYOUT), load_payout_a(PAYOUT)
    years = [yy for yy in range(y0, int(year.max()) + 1) if (year == yy).sum() > 5000]
    print(f"容量の梯子 × ウォークフォワード（{years[0]}〜{years[-1]}・シード{n_seed}本）")

    races = []          # プラセボ用にレース情報を貯める
    out = {nm: {"w": [], "s": []} for nm, _ in LADDER}
    agree = {nm: {"w": [], "s": []} for nm, _ in LADDER}
    popw, pops = [], []
    for yy in years:
        tr, te = year < yy, year == yy
        sub = d.loc[te, ["raceid", "umaban", "fieldsize", "odds"]].copy()
        for ci, (nm, par) in enumerate(LADDER):
            ps = [lgb.LGBMClassifier(random_state=s, **par)
                  .fit(fx[tr], y[tr], categorical_feature=F.CAT_COLS)
                  .predict_proba(fx[te])[:, 1] for s in range(n_seed)]
            sub[f"p{ci}"] = np.mean(ps, axis=0)
        for rid, g in sub.groupby("raceid", sort=False):
            w, s3 = wu.get(rid), pa.get(rid)
            okw = bool(w and w["wakuren"] and len(g) >= 3)
            oks = bool(s3 and s3["sanrenpuku"] and len(g) >= 9)
            if not (okw or oks):
                continue
            n = int(g["fieldsize"].iloc[0])
            uma = g["umaban"].astype(int).to_numpy()
            lo = np.log(g["odds"].to_numpy(float))
            po = uma[np.argsort(lo, kind="mergesort")]
            pwk = wakuren_cs(po, n) if okw else None
            ps3 = set(tuple(sorted(c)) for c in itertools.combinations(po[:4], 3)) if oks else None
            if okw:
                popw.append(sum(w["wakuren"].get(c, 0) for c in pwk) / (len(pwk) * 100))
            if oks:
                pops.append(sum(s3["sanrenpuku"].get(c, 0) for c in ps3) / 400)
            races.append({"n": n, "uma": uma, "lo": lo, "wk": w["wakuren"] if okw else None,
                          "s3": s3["sanrenpuku"] if oks else None, "pwk": pwk, "ps3": ps3})
            for ci, (nm, _) in enumerate(LADDER):
                mo = uma[np.argsort(-g[f"p{ci}"].to_numpy(float), kind="mergesort")]
                if okw:
                    cs = wakuren_cs(mo, n)
                    out[nm]["w"].append(sum(w["wakuren"].get(c, 0) for c in cs) / (len(cs) * 100))
                    agree[nm]["w"].append(cs == pwk)
                if oks:
                    cs = [tuple(sorted(c)) for c in itertools.combinations(mo[:4], 3)]
                    out[nm]["s"].append(sum(s3["sanrenpuku"].get(c, 0) for c in cs) / 400)
                    agree[nm]["s"].append(set(cs) == ps3)
        print(f"  {yy} 完了")

    # ---- プラセボ曲線（σ → 一致率, ROI）を一度だけ作る ----
    def placebo(sg, seed):
        g = np.random.default_rng(seed)
        rw, rs, aw, as_ = [], [], [], []
        for r in races:
            o = r["uma"][np.argsort(r["lo"] + g.normal(0, sg, len(r["uma"])), kind="mergesort")]
            if r["wk"] is not None:
                cs = wakuren_cs(o, r["n"])
                rw.append(sum(r["wk"].get(c, 0) for c in cs) / (len(cs) * 100))
                aw.append(cs == r["pwk"])
            if r["s3"] is not None:
                cs = [tuple(sorted(c)) for c in itertools.combinations(o[:4], 3)]
                rs.append(sum(r["s3"].get(c, 0) for c in cs) / 400)
                as_.append(set(cs) == r["ps3"])
        return np.mean(aw), np.mean(rw), np.mean(as_), np.mean(rs)

    print("\nプラセボ曲線を作成中…")
    curve = []
    for sg in SIGMAS:
        vals = [placebo(sg, 500 + k) for k in range(N_DRAW if sg > 0 else 1)]
        curve.append((sg,) + tuple(np.mean([v[i] for v in vals]) for i in range(4)))
    pw, ps_ = np.array(popw), np.array(pops)
    rng = np.random.default_rng(0)

    for tag, key, pop, ai, ri in [("枠連 軸枠×紐枠2", "w", pw, 1, 2),
                                  ("三連複 BOX上位4", "s", ps_, 3, 4)]:
        print(f"\n{'='*104}\n=== {tag}  人気順 {pop.mean()*100:.2f}% ===")
        print(f"{'設定（容量の順）':<30}{'ROI':>9}{'対人気順':>10}{'人気順との一致率':>17}"
              f"{'同じ一致率のプラセボ':>21}{'モデル−プラセボ':>17}{'95%CI':>17}")
        ax = np.array([c[ai] for c in curve])
        rx = np.array([c[ri] for c in curve])
        for nm, _ in LADDER:
            m = np.array(out[nm][key])
            ag = float(np.mean(agree[nm][key]))
            sg = float(np.interp(ag, ax[::-1], np.array(SIGMAS)[::-1]))
            pl = np.mean([placebo(sg, 900 + k)[ri - 1] for k in range(N_DRAW)])
            lo, hi = boot(m - pop, rng)
            print(f"{nm:<30}{m.mean()*100:>8.2f}%{(m-pop).mean()*100:>+9.2f}pt{ag*100:>16.1f}%"
                  f"{pl*100:>20.2f}%{(m.mean()-pl)*100:>+16.2f}pt{f'[{lo:+.2f},{hi:+.2f}]':>17}")
        print(f"  プラセボ曲線: " + " ".join(f"σ{c[0]:.2f}→{c[ai]*100:.0f}%/{c[ri]*100:.1f}%" for c in curve))


if __name__ == "__main__":
    main()
