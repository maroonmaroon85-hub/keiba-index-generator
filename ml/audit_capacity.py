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

実行: python3 ml/audit_capacity.py [シード数(既定2)] [開始年(既定2018)] [終了年] [固定窓の年数(既定3)]
⚠ 梯子の上段(leaves127/2000本)は**拡張窓だと1本1〜2分**かかり、11年×5設定×2シードで数時間になる。
　 既定を**固定窓3年**にしてあるのはそのため（(78)③で拡張窓と結果がほぼ同じと実測済み）。
　 拡張窓で回したいときは第4引数に 0 を渡す（時間はかかる）。
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
N_DRAW = 3
N_PL = 6          # 各設定のプラセボの引き直し回数

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
    y1 = int(sys.argv[3]) if len(sys.argv) > 3 else 9999
    # 第4引数に年数を渡すと固定窓（直近N年だけで学習）。梯子の上段(leaves127/2000本)は
    # 拡張窓だと1本1〜2分かかって現実的でないので、既定を固定窓3年にしてある。
    # ★(78)③で「固定窓(直近3年)と拡張窓(13年)で結果はほぼ同じ」（枠連84.38 vs 84.86 /
    #   三連複82.32 vs 81.82）と実測済みなので、容量の比較に使うぶんには等価。
    rolling = int(sys.argv[4]) if len(sys.argv) > 4 else 3
    d, fx = load_cached()
    y = (d["finish"] <= 3).astype(int).to_numpy()
    year = d["date"].dt.year.to_numpy()
    wu, pa = load_wu(PAYOUT), load_payout_a(PAYOUT)
    years = [yy for yy in range(y0, min(int(year.max()), y1) + 1) if (year == yy).sum() > 5000]
    print(f"容量の梯子 × ウォークフォワード（{years[0]}〜{years[-1]}・シード{n_seed}本"
          f"・{f'固定窓{rolling}年' if rolling else '拡張窓'}）")

    races = []          # プラセボ用にレース情報を貯める
    out = {nm: {"w": [], "s": []} for nm, _ in LADDER}
    agree = {nm: {"w": [], "s": []} for nm, _ in LADDER}
    popw, pops = [], []
    for yy in years:
        tr, te = year < yy, year == yy
        if rolling:
            tr = tr & (year >= yy - rolling)
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

    def placebo_rows(races, sg, seed, key):
        """1引きぶんの、レース単位の回収倍率の配列（key: w=枠連 / s=三連複）。"""
        g = np.random.default_rng(seed)
        row = []
        for r in races:
            if (key == "w" and r["wk"] is None) or (key == "s" and r["s3"] is None):
                continue
            o = r["uma"][np.argsort(r["lo"] + g.normal(0, sg, len(r["uma"])), kind="mergesort")]
            if key == "w":
                cs = wakuren_cs(o, r["n"])
                row.append(sum(r["wk"].get(c, 0) for c in cs) / (len(cs) * 100))
            else:
                cs = [tuple(sorted(c)) for c in itertools.combinations(o[:4], 3)]
                row.append(sum(r["s3"].get(c, 0) for c in cs) / 400)
        return np.array(row)

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
        print(f"{'設定（容量の順）':<30}{'ROI':>9}{'対人気順':>10}{'一致率':>9}{'σ':>7}"
              f"{'プラセボ':>10}{'±':>6}{'モデル−プラセボ':>17}{'その95%CI':>18}")
        ax = np.array([c[ai] for c in curve])
        for nm, _ in LADDER:
            m = np.array(out[nm][key])
            ag = float(np.mean(agree[nm][key]))
            sg = float(np.interp(ag, ax[::-1], np.array(SIGMAS)[::-1]))
            # プラセボは引くたびに数値が動くので、**レース単位の配列を N_PL 本引いて**
            # 平均（表の値）と引き間の散らばり（±列）とCIを1回の走査でまとめて出す。
            rows = np.array([placebo_rows(races, sg, 900 + k, key) for k in range(N_PL)])
            draws = rows.mean(axis=1)
            pl, plr = float(draws.mean()), rows.mean(axis=0)
            lo, hi = boot(m - plr, rng)
            print(f"{nm:<30}{m.mean()*100:>8.2f}%{(m-pop).mean()*100:>+9.2f}pt{ag*100:>8.1f}%"
                  f"{sg:>7.3f}{pl*100:>9.2f}%{draws.std()*100:>5.2f}"
                  f"{(m.mean()-pl)*100:>+16.2f}pt{f'[{lo:+.2f},{hi:+.2f}]':>18}")
        print(f"  プラセボ曲線: " + " ".join(f"σ{c[0]:.2f}→{c[ai]*100:.0f}%/{c[ri]*100:.1f}%" for c in curve))
        # ★対応ありで現行と直接比較（(77)②の作法）。同じレースで両方買った差。
        base = np.array(out["3 leaves31 /  400本 ★現行"][key])
        print(f"  ★現行との対応あり比較")
        for nm, _ in LADDER:
            if "現行" in nm:
                continue
            m = np.array(out[nm][key])
            lo, hi = boot(m - base, rng)
            v = "★現行より上" if lo > 0 else ("現行より下" if hi < 0 else "差は誤差の内")
            print(f"    {nm:<30}{(m-base).mean()*100:>+7.2f}pt  CI[{lo:+.2f},{hi:+.2f}]   {v}")


if __name__ == "__main__":
    main()
