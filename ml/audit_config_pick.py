"""
(80) ★(79)②「設定は事前に選べない」を、**選べる手続きがあるか**の検証に変える。

(53)以来ずっと「ハイパーパラメータでROIは散らばるが事前に選べない」を留保として置いてきた。
(79)②はそれを実運用の評価設計（ウォークフォワード）でも確認している:
    leaves15/mc100/100本 +0.08pt … leaves63/mc30/1000本 +5.26pt（幅5.18pt）
だが**「選べない」は主張であって、検証された事実ではない**。
「前年までの実績で選んで翌年に適用する」という**内側選択のウォークフォワード**は一度も試していない。
これは実運用でそのまま実行できる手続きなので、後知恵ではない。

やること: 各年Yについて、Yより前の全データで4設定を学習してYを予測し、**年ごとに記録**する。
そのうえで「Yを買うときにYより前の情報だけで設定を選ぶ」規則をシミュレートする:
  ・現行固定          … leaves31/mc100/400本 を使い続ける（今の運用）
  ・累積ROI最良       … Y未満の全年の累積ROIが最良の設定を選ぶ
  ・前年ROI最良       … 直前の1年のROIが最良の設定を選ぶ
  ・前年AUC最良       … 直前の1年のAUCが最良（★(52)より悪化するはず。反証用の対照）
  ・後知恵最良（参考） … 全期間で最良だった設定を最初から使う。**実行不可能**
  ・オラクル（参考）   … 各年ごとに最良を後から選ぶ。**実行不可能な上限**

★判定: 「累積ROI最良」または「前年ROI最良」が**現行固定を対応ありCIで上回る**なら、
　　　　(53)以来の留保は「選べる」に更新できる。上回らなければ留保はそのまま。

実行: python3 ml/audit_config_pick.py [シード数(既定2)]
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
    ("leaves15/mc30/400本", dict(PARAMS, num_leaves=15, min_child_samples=30)),
    ("leaves63/mc30/1000本", dict(PARAMS, num_leaves=63, min_child_samples=30,
                                 n_estimators=1000)),
    ("leaves15/mc100/100本", dict(PARAMS, num_leaves=15, n_estimators=100)),
]
CUR = 0          # 現行設定のインデックス


def wakuren_cs(nums, n):
    wa = waku_of(nums[0], n)
    return sorted({tuple(sorted((wa, waku_of(h, n)))) for h in nums[1:3]})


def boot(x, rng, n=2000):
    b = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return np.percentile(b, 2.5) * 100, np.percentile(b, 97.5) * 100


def auc(yy, s):
    o = np.argsort(s, kind="mergesort")
    r = np.empty(len(s), float)
    r[o] = np.arange(1, len(s) + 1)
    n1 = yy.sum()
    n0 = len(yy) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    return (r[yy == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def main():
    n_seed = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    d, fx = load_cached()
    y = (d["finish"] <= 3).astype(int).to_numpy()
    year = d["date"].dt.year.to_numpy()
    wu, pa = load_wu(PAYOUT), load_payout_a(PAYOUT)
    years = [yy for yy in range(2016, int(year.max()) + 1) if (year == yy).sum() > 5000]
    print(f"設定4種 × ウォークフォワード（{years[0]}〜{years[-1]}・シード{n_seed}本）")
    print("※各年Yは「Yより前の全データ」で学習したモデルで予測＝実運用の手続き\n")

    # roi[key][ci][year] = そのconfig・その年の レース単位ROI配列
    roi = {k: {ci: {} for ci in range(len(CONFIGS))} for k in ("w", "s")}
    pop = {k: {} for k in ("w", "s")}
    aucs = {ci: {} for ci in range(len(CONFIGS))}

    for yy in years:
        tr, te = year < yy, year == yy
        sub0 = d.loc[te, ["raceid", "umaban", "fieldsize", "odds"]].copy()
        for ci, (name, par) in enumerate(CONFIGS):
            ps = [lgb.LGBMClassifier(random_state=s, **par)
                  .fit(fx[tr], y[tr], categorical_feature=F.CAT_COLS)
                  .predict_proba(fx[te])[:, 1] for s in range(n_seed)]
            p = np.mean(ps, axis=0)
            sub0[f"p{ci}"] = p
            aucs[ci][yy] = auc(y[te], p)
        accw = {ci: [] for ci in range(len(CONFIGS))}
        accs = {ci: [] for ci in range(len(CONFIGS))}
        pw, ps_ = [], []
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
                pw.append(sum(w["wakuren"].get(c, 0) for c in cs) / (len(cs) * 100))
            if oks:
                cs = [tuple(sorted(c)) for c in itertools.combinations(po[:4], 3)]
                ps_.append(sum(s3["sanrenpuku"].get(c, 0) for c in cs) / 400)
            for ci in range(len(CONFIGS)):
                mo = uma[np.argsort(-g0[f"p{ci}"].to_numpy(float), kind="mergesort")]
                if okw:
                    cs = wakuren_cs(mo, n)
                    accw[ci].append(sum(w["wakuren"].get(c, 0) for c in cs) / (len(cs) * 100))
                if oks:
                    cs = [tuple(sorted(c)) for c in itertools.combinations(mo[:4], 3)]
                    accs[ci].append(sum(s3["sanrenpuku"].get(c, 0) for c in cs) / 400)
        pop["w"][yy], pop["s"][yy] = np.array(pw), np.array(ps_)
        for ci in range(len(CONFIGS)):
            roi["w"][ci][yy] = np.array(accw[ci])
            roi["s"][ci][yy] = np.array(accs[ci])
        print(f"  {yy} 完了（枠連{len(pw):,}R / 三連複{len(ps_):,}R）"
              + "  ROI " + " ".join(f"{np.mean(accw[c])*100:.1f}" for c in range(len(CONFIGS))))

    rng = np.random.default_rng(0)
    sel_years = years[1:]        # 1年ぶんの実績が無いと選べない

    def pick_cum(key, yy):
        """Y未満の全年の累積ROIが最良の設定。"""
        sc = [np.concatenate([roi[key][ci][p] for p in years if p < yy]).mean()
              for ci in range(len(CONFIGS))]
        return int(np.argmax(sc))

    def pick_prev(key, yy):
        prev = max(p for p in years if p < yy)
        return int(np.argmax([roi[key][ci][prev].mean() for ci in range(len(CONFIGS))]))

    def pick_auc(key, yy):
        prev = max(p for p in years if p < yy)
        return int(np.argmax([aucs[ci][prev] for ci in range(len(CONFIGS))]))

    for tag, key in [("枠連 軸枠×紐枠2", "w"), ("三連複 BOX上位4", "s")]:
        popv = np.concatenate([pop[key][yy] for yy in sel_years])
        print(f"\n{'='*104}")
        print(f"=== {tag}  評価期間 {sel_years[0]}〜{sel_years[-1]}"
              f"（{len(popv):,}R）  人気順 {popv.mean()*100:.2f}% ===")
        print(f"{'=' * 104}")

        rules = {}
        for nm, fn in [("累積ROI最良で選ぶ", pick_cum), ("前年ROI最良で選ぶ", pick_prev),
                       ("前年AUC最良で選ぶ", pick_auc)]:
            picks = {yy: fn(key, yy) for yy in sel_years}
            rules[nm] = (np.concatenate([roi[key][picks[yy]][yy] for yy in sel_years]), picks)
        for ci, (name, _) in enumerate(CONFIGS):
            rules[f"固定: {name}"] = (np.concatenate([roi[key][ci][yy] for yy in sel_years]),
                                     {yy: ci for yy in sel_years})
        orc = {yy: int(np.argmax([roi[key][ci][yy].mean() for ci in range(len(CONFIGS))]))
               for yy in sel_years}
        rules["オラクル(実行不可)"] = (np.concatenate([roi[key][orc[yy]][yy] for yy in sel_years]),
                                  orc)

        base = rules[f"固定: {CONFIGS[CUR][0]}"][0]
        print(f"{'規則':<30}{'ROI':>9}{'対人気順':>10}{'現行固定との差':>16}{'95%CI':>18}{'判定':>8}")
        order = ["累積ROI最良で選ぶ", "前年ROI最良で選ぶ", "前年AUC最良で選ぶ"] + \
                [f"固定: {n}" for n, _ in CONFIGS] + ["オラクル(実行不可)"]
        for nm in order:
            v, _ = rules[nm]
            dif = v - base
            if nm == f"固定: {CONFIGS[CUR][0]}":
                print(f"{nm:<30}{v.mean()*100:>8.2f}%{(v.mean()-popv.mean())*100:>+9.2f}pt"
                      f"{'—':>16}{'—':>18}{'基準':>8}")
                continue
            lo, hi = boot(dif, rng)
            mark = "★上" if lo > 0 else ("★下" if hi < 0 else "")
            print(f"{nm:<30}{v.mean()*100:>8.2f}%{(v.mean()-popv.mean())*100:>+9.2f}pt"
                  f"{dif.mean()*100:>+15.2f}pt{f'[{lo:+.2f},{hi:+.2f}]':>18}{mark:>8}")

        print("\n  年ごとに何が選ばれたか（数字はCONFIGSの番号。0=現行）")
        for nm in ["累積ROI最良で選ぶ", "前年ROI最良で選ぶ", "前年AUC最良で選ぶ",
                   "オラクル(実行不可)"]:
            _, picks = rules[nm]
            print(f"    {nm:<22}" + " ".join(f"{yy}:{picks[yy]}" for yy in sel_years))
        print("  各設定の年別ROI")
        for ci, (name, _) in enumerate(CONFIGS):
            print(f"    [{ci}] {name:<24}"
                  + " ".join(f"{roi[key][ci][yy].mean()*100:5.1f}" for yy in sel_years))

    print("\n★読み方: 「累積ROI最良」「前年ROI最良」が現行固定を**CIが0を外れて上回る**なら、"
          "(53)以来の「設定は事前に選べない」は『選べる』に更新できる。")
    print("　上回らなければ留保はそのまま＝**現行設定を使い続けるのが正しい**。")
    print("　オラクルとの差が大きいほど「良い設定は存在するが事前には掴めない」ことの裏付けになる。")


if __name__ == "__main__":
    main()
