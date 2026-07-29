"""
★事前指定した仮説を検定する（探索ではない）。

(32)で整理したとおり、これまでの絞り込みは全て「グリッドを舐めて良いセルを見つけた」もので、
「先に理由があって指定した」という条件を満たしていなかった。そこで**競馬の理屈として先に言える軸**を
5つ宣言してから測る。事後に軸を追加しないことが本スクリプトの価値。

事前宣言（2026-07-29）:
  1. 少頭数     … 頭数が少ないほど組合せが減り当てやすい／配当も安い、どちらが勝つか
  2. クラス     … 下級条件は力量差が読みにくい／上級は少数精鋭
  3. 距離帯     … 短距離は紛れが少ない／長距離は展開の影響が大きい
  4. 休み明け   … 前走からの間隔。仕上がり途上を市場が過大評価しやすい
  5. 乗り替わり … 騎手が替わった馬。市場は騎手を重視するので歪みが出やすい
計18区分 × 1戦略。18区分なら**偶然100%超が1個前後出るのが期待値**なので、その前提で読む。

判定は(30)(32)の基準に従う: 効果量をシード幅と並記し、前後半の両方で残るかを見る。

実行: python3 ml/hypothesis_test.py [シード数(既定3)]
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

TNAME, NHIMO = "三連単マルチ", 7  # (31)で最良だった形。事前に固定する。
SPLIT_YEAR = 2022
MIN_R = 300


def bins_of(df):
    """事前宣言した5軸 → {軸名: Series(区分ラベル)}。ここに後から軸を足さないこと。"""
    out = {}
    fs = df["fieldsize"]
    out["1.頭数"] = pd.cut(fs, [0, 9, 12, 15, 99], labels=["〜9頭", "10-12頭", "13-15頭", "16頭〜"])
    rc = df["raceclass"]
    out["2.クラス"] = pd.cut(rc, [-1, 0, 1, 3, 5, 99],
                           labels=["新馬", "未勝利", "1-2勝", "3勝-OP", "重賞"])
    ds = df["distance"]
    out["3.距離"] = pd.cut(ds, [0, 1200, 1600, 2000, 9999],
                         labels=["〜1200m", "1201-1600m", "1601-2000m", "2001m〜"])
    ws = df["weeks_since"]
    out["4.休み明け"] = pd.cut(ws, [-1, 3, 8, 16, 999],
                           labels=["〜3週", "4-8週", "9-16週", "17週〜"])
    out["5.乗り替わり"] = df["jockey_changed"].map({True: "乗り替わり", False: "継続"})
    return out


def main():
    n_seed = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    d = F.to_model(F.load_files())
    f = F.build_features(d)
    keep = f["n_prior"] >= 1
    d, f = d[keep].reset_index(drop=True), f[keep].reset_index(drop=True)

    # 乗り替わり: 同じ馬の前走の騎手と違うか（d は horse,date 順に並んでいる）
    d["prev_jockey"] = d.groupby("horse")["jockey"].shift(1)
    d["jockey_changed"] = (d["prev_jockey"].notna()) & (d["jockey"] != d["prev_jockey"])

    y = (d["finish"] == 1).astype(int).to_numpy()
    cut = d["date"].quantile(0.3)
    tr, te = (d["date"] < cut).to_numpy(), (d["date"] >= cut).to_numpy()
    fx, _ = F.encode_categoricals(f)
    pays = load_payout_a("data/payout/a.csv")

    meta = d.loc[te, ["raceid", "umaban", "jockey_changed"]].copy()
    for c in ["fieldsize", "raceclass", "distance", "weeks_since"]:
        meta[c] = f.loc[te, c].to_numpy() if c in f.columns else d.loc[te, c].to_numpy()
    print(f"train {tr.sum():,} / test {te.sum():,}  シード{n_seed}本  戦略={TNAME}×紐{NHIMO}")

    pts = POINTS[TNAME](NHIMO) * 100
    res = defaultdict(list)      # (軸,区分,期間) -> [ROI...]
    cnts = {}
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
            a = g.iloc[0]  # 軸馬の行（休み明け・乗り替わりは軸馬の属性）
            rows.append({"year": p["date"].year, "pay": hits(p[KEY[TNAME]], nums[0], set(nums[1:NHIMO + 1])),
                         "fieldsize": a["fieldsize"], "raceclass": a["raceclass"],
                         "distance": a["distance"], "weeks_since": a["weeks_since"],
                         "jockey_changed": a["jockey_changed"]})
        df = pd.DataFrame(rows)
        bb = bins_of(df)

        def roi(sub):
            return sub["pay"].sum() / (len(sub) * pts) * 100 if len(sub) else float("nan")

        res[("全体", "全体", "all")].append(roi(df))
        res[("全体", "全体", "1st")].append(roi(df[df["year"] < SPLIT_YEAR]))
        res[("全体", "全体", "2nd")].append(roi(df[df["year"] >= SPLIT_YEAR]))
        cnts[("全体", "全体")] = len(df)
        for axis, lab in bb.items():
            for v in [x for x in pd.unique(lab.dropna())]:
                sub = df[(lab == v).to_numpy()]
                if len(sub) < MIN_R:
                    continue
                res[(axis, v, "all")].append(roi(sub))
                res[(axis, v, "1st")].append(roi(sub[sub["year"] < SPLIT_YEAR]))
                res[(axis, v, "2nd")].append(roi(sub[sub["year"] >= SPLIT_YEAR]))
                cnts[(axis, v)] = len(sub)
        print(f"  seed {seed} 完了")

    base = np.mean(res[("全体", "全体", "all")])
    base_w = np.ptp(res[("全体", "全体", "all")])
    print("\n" + "=" * 84)
    print(f"★事前指定仮説の検定  戦略={TNAME}×紐{NHIMO}  全体={base:.1f}%（シード幅{base_w:.1f}）")
    print("=" * 84)
    print(f"{'軸':<12}{'区分':<12}{'R':>7}{'ROI':>8}{'幅':>6}{'前半':>8}{'後半':>8}   判定")
    axes = ["1.頭数", "2.クラス", "3.距離", "4.休み明け", "5.乗り替わり"]
    survivors = []
    for axis in axes:
        keys = [k for k in cnts if k[0] == axis]
        for k in sorted(keys, key=lambda x: -np.mean(res[(x[0], x[1], "all")])):
            a = np.array(res[(k[0], k[1], "all")], float)
            r1 = np.mean(res[(k[0], k[1], "1st")])
            r2 = np.mean(res[(k[0], k[1], "2nd")])
            w = np.ptp(a)
            diff = a.mean() - base
            if abs(diff) <= max(w, base_w):
                v = "ノイズ以下"
            elif (r1 > 100) and (r2 > 100):
                v = "★両期間で100%超"
                survivors.append((axis, k[1], a.mean(), r1, r2))
            elif a.mean() > 100:
                v = "全体100%超だが片期間のみ"
            else:
                v = "幅を超えるが100%未満"
            print(f"{axis:<12}{str(k[1]):<12}{cnts[k]:>7}{a.mean():>7.1f}%{w:>6.1f}{r1:>7.1f}%{r2:>7.1f}%   {v}")
        print()
    n_cells = len(cnts) - 1
    print(f"検定した区分数: {n_cells}（事前宣言どおり）")
    if survivors:
        print("★両期間とも100%超だった区分:")
        for s in survivors:
            print(f"   {s[0]} {s[1]}  全体{s[2]:.1f}%  前半{s[3]:.1f}%  後半{s[4]:.1f}%")
    else:
        print("→ 両期間とも100%超だった区分は **なし**。")


if __name__ == "__main__":
    main()
