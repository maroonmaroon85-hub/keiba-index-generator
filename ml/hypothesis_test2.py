"""
★事前指定した仮説の第2弾（(33)の続き。探索ではない）。

(33)で 頭数/クラス/距離/休み明け/乗り替わり の5軸18区分を検定した。その続きとして
**競馬の理屈として先に言える軸**を7つ宣言してから測る。事後に軸を追加しないこと。

事前宣言（2026-07-29・計29区分）:
  1. 枠順(3)      … 内枠は距離ロスが少ない／外枠は揉まれない。軸馬の馬番の相対位置で見る
  2. 性別(3)      … 牝馬は人気になりにくい／セン馬は特殊
  3. 年齢(3)      … 2-3歳は実績が薄く市場の値付けが粗い／古馬は安定
  4. 開催場(10)   … ローカルは頭数・レベルが違い、市場の厚みも違う
  5. 前走着順(4)  … 前走1着は人気を集めすぎる（過剰人気の典型）
  6. 斤量(3)      … ハンデ・別定で能力が均される
  7. 馬体重増減(3)… 大幅増減は仕上がりの不安。市場が反応しきれない可能性
29区分なら**偶然100%超が1-2個出るのが期待値**。その前提で読む。

判定は(30)(32)の基準どおり: シード幅を並記し、前後半の両方で100%超でなければ採用しない。

実行: python3 ml/hypothesis_test2.py [シード数(既定3)]
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
MIN_R = 300


def bins_of(df):
    """事前宣言した7軸。ここに後から足さないこと。"""
    out = {}
    pos = df["umaban"] / df["fieldsize"]
    out["1.枠順"] = pd.cut(pos, [0, 1 / 3, 2 / 3, 1.01], labels=["内", "中", "外"])
    out["2.性別"] = df["sex"]
    out["3.年齢"] = pd.cut(df["age"], [0, 3, 5, 99], labels=["2-3歳", "4-5歳", "6歳〜"])
    out["4.開催場"] = df["course"]
    out["5.前走着順"] = pd.cut(df["prev_finish"], [0, 1, 3, 5, 99],
                            labels=["前走1着", "前走2-3着", "前走4-5着", "前走6着〜"])
    out["6.斤量"] = pd.cut(df["wtcarry"], [0, 54, 56, 99], labels=["〜54kg", "54.5-56kg", "56.5kg〜"])
    out["7.馬体重増減"] = pd.cut(df["bodywt_change"], [-999, -8, 7, 999],
                            labels=["−8kg以下", "−7〜+7kg", "+8kg以上"])
    return out


def main():
    n_seed = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    d = F.to_model(F.load_files())
    f = F.build_features(d)
    keep = f["n_prior"] >= 1
    d, f = d[keep].reset_index(drop=True), f[keep].reset_index(drop=True)
    d["prev_finish"] = d.groupby("horse")["finish"].shift(1)

    y = (d["finish"] == 1).astype(int).to_numpy()
    cut = d["date"].quantile(0.3)
    tr, te = (d["date"] < cut).to_numpy(), (d["date"] >= cut).to_numpy()
    fx, _ = F.encode_categoricals(f)
    pays = load_payout_a("data/payout/a.csv")

    meta = d.loc[te, ["raceid", "umaban", "sex", "age", "course", "prev_finish"]].copy()
    for c in ["fieldsize", "wtcarry", "bodywt_change"]:
        meta[c] = f.loc[te, c].to_numpy()
    print(f"train {tr.sum():,} / test {te.sum():,}  シード{n_seed}本  戦略={TNAME}×紐{NHIMO}")

    pts = POINTS[TNAME](NHIMO) * 100
    res, cnts = defaultdict(list), {}
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
            a = g.iloc[0]   # 軸馬の属性で層別する
            rows.append({"year": p["date"].year, "pay": hits(p[KEY[TNAME]], nums[0], set(nums[1:NHIMO + 1])),
                         "umaban": a["umaban"], "fieldsize": a["fieldsize"], "sex": a["sex"],
                         "age": a["age"], "course": a["course"], "prev_finish": a["prev_finish"],
                         "wtcarry": a["wtcarry"], "bodywt_change": a["bodywt_change"]})
        df = pd.DataFrame(rows)

        def roi(sub):
            return sub["pay"].sum() / (len(sub) * pts) * 100 if len(sub) else float("nan")

        res[("全体", "全体", "all")].append(roi(df))
        cnts[("全体", "全体")] = len(df)
        for axis, lab in bins_of(df).items():
            for v in pd.unique(lab.dropna()):
                sub = df[(lab == v).to_numpy()]
                if len(sub) < MIN_R:
                    continue
                cnts[(axis, v)] = len(sub)
                res[(axis, v, "all")].append(roi(sub))
                res[(axis, v, "1st")].append(roi(sub[sub["year"] < SPLIT_YEAR]))
                res[(axis, v, "2nd")].append(roi(sub[sub["year"] >= SPLIT_YEAR]))
        print(f"  seed {seed} 完了")

    base = np.mean(res[("全体", "全体", "all")])
    base_w = np.ptp(res[("全体", "全体", "all")])
    print("\n" + "=" * 84)
    print(f"★事前指定仮説 第2弾  {TNAME}×紐{NHIMO}  全体={base:.1f}%（シード幅{base_w:.1f}）")
    print("=" * 84)
    print(f"{'軸':<12}{'区分':<12}{'R':>7}{'ROI':>8}{'幅':>6}{'前半':>8}{'後半':>8}   判定")
    survivors, n_cells = [], 0
    for axis in ["1.枠順", "2.性別", "3.年齢", "4.開催場", "5.前走着順", "6.斤量", "7.馬体重増減"]:
        keys = [k for k in cnts if k[0] == axis]
        for k in sorted(keys, key=lambda x: -np.mean(res[(x[0], x[1], "all")])):
            n_cells += 1
            a = np.array(res[(k[0], k[1], "all")], float)
            r1, r2 = np.mean(res[(k[0], k[1], "1st")]), np.mean(res[(k[0], k[1], "2nd")])
            w = np.ptp(a)
            diff = a.mean() - base
            if abs(diff) <= max(w, base_w):
                v = "ノイズ以下"
            elif r1 > 100 and r2 > 100:
                v = "★両期間100%超"
                survivors.append((axis, k[1], a.mean(), r1, r2, cnts[k]))
            elif a.mean() > 100:
                v = "全体100%超だが片期間のみ"
            else:
                v = "幅超・100%未満"
            print(f"{axis:<12}{str(k[1]):<12}{cnts[k]:>7}{a.mean():>7.1f}%{w:>6.1f}{r1:>7.1f}%{r2:>7.1f}%   {v}")
        print()
    print(f"検定した区分数: {n_cells}")
    if survivors:
        print("★両期間とも100%超:")
        for s in survivors:
            print(f"   {s[0]} {s[1]}  R={s[5]}  全体{s[2]:.1f}%  前半{s[3]:.1f}%  後半{s[4]:.1f}%")
    else:
        print("→ 両期間とも100%超だった区分は **なし**。")


if __name__ == "__main__":
    main()
