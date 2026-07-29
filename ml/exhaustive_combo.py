"""
★全組み合わせを総当たりし、その結果を**帰無分布と比較**する（多重検定の正しい扱い）。

「それぞれの組み合わせで全部検証できるか」への回答。総当たり自体は計算できるが、
単に最良セルを報告すれば必ず100%超が出るので意味がない。判定には
**「エッジが無かったら何個出るはずか」** との比較が要る。

方法:
  1. (33)(36)で宣言した12軸を使い、1軸/2軸/3軸の全組み合わせを列挙して各セルのROIを出す
  2. 同じ探索を、**払戻をレース間でシャッフルした偽データ**に対しても実行する
     （シャッフルすると条件と払戻の関係は完全に消えるので、出てくる100%超は純粋な偶然）
  3. 実データの「100%超セル数」が偽データの分布を上回るかを見る
     ・上回らない → 見つかった組み合わせは全て偶然の産物
     ・上回る     → そこで初めて何かがあると言える

計算はセルごとではなく groupby でまとめて行う（軸の組ごとに1回のgroupbyで全水準を得る）。

実行: python3 ml/exhaustive_combo.py [シャッフル回数(既定20)]
"""
import itertools
import sys

import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, "ml")
import features as F
from himo_sweep import POINTS, KEY, hits
from pocket_eval import load_payout_a

TNAME, NHIMO = "三連単マルチ", 7
SPLIT_YEAR = 2022
MIN_R = 300          # これ未満のセルは母集団が小さすぎるので対象外
MAX_AXES = 3         # 何軸まで重ねるか


def build_axes(df):
    """(33)(36)で事前宣言した12軸を、そのままカテゴリ列にする。"""
    a = pd.DataFrame(index=df.index)
    a["頭数"] = pd.cut(df["fieldsize"], [0, 9, 12, 15, 99], labels=["〜9", "10-12", "13-15", "16〜"])
    a["クラス"] = pd.cut(df["raceclass"], [-1, 0, 1, 3, 5, 99], labels=["新馬", "未勝利", "1-2勝", "3勝OP", "重賞"])
    a["距離"] = pd.cut(df["distance"], [0, 1200, 1600, 2000, 9999], labels=["〜1200", "〜1600", "〜2000", "2001〜"])
    a["休み明け"] = pd.cut(df["weeks_since"], [-1, 3, 8, 16, 999], labels=["〜3週", "4-8週", "9-16週", "17週〜"])
    a["乗替"] = np.where(df["jockey_changed"].astype(bool), "乗替", "継続")
    a["枠"] = pd.cut(df["umaban"] / df["fieldsize"], [0, 1 / 3, 2 / 3, 1.01], labels=["内", "中", "外"])
    a["性"] = df["sex"]
    a["齢"] = pd.cut(df["age"], [0, 3, 5, 99], labels=["2-3歳", "4-5歳", "6歳〜"])
    a["場"] = df["course"]
    a["前走"] = pd.cut(df["prev_finish"], [0, 1, 3, 5, 99], labels=["1着", "2-3着", "4-5着", "6着〜"])
    a["斤量"] = pd.cut(df["wtcarry"], [0, 54, 56, 99], labels=["〜54", "54.5-56", "56.5〜"])
    a["馬体重"] = pd.cut(df["bodywt_change"], [-999, -8, 7, 999], labels=["−8以下", "−7〜+7", "+8以上"])
    return a.astype(str)


def scan(axes, pay, second, pts):
    """全組み合わせを走査し (総セル数, 全体100%超, 両期間100%超, 最良セル) を返す。"""
    cols = list(axes.columns)
    n_cell = n_all = n_both = 0
    best = None
    for k in range(1, MAX_AXES + 1):
        for combo in itertools.combinations(cols, k):
            g = pd.DataFrame({"pay": pay, "second": second.astype(int), "pay2": pay * second})
            for c in combo:
                g[c] = axes[c].to_numpy()
            agg = g.groupby(list(combo), observed=True).agg(
                R=("pay", "size"), tot=("pay", "sum"), R2=("second", "sum"), tot2=("pay2", "sum"))
            agg = agg[agg["R"] >= MIN_R]
            if agg.empty:
                continue
            roi = agg["tot"] / (agg["R"] * pts) * 100
            r1 = (agg["tot"] - agg["tot2"]) / ((agg["R"] - agg["R2"]) * pts) * 100
            r2 = agg["tot2"] / (agg["R2"] * pts) * 100
            n_cell += len(agg)
            n_all += int((roi > 100).sum())
            n_both += int(((r1 > 100) & (r2 > 100)).sum())
            if len(roi) and (best is None or roi.max() > best[1]):
                i = roi.idxmax()
                best = (f"{combo}={i}", roi.max(), int(agg.loc[i, "R"]), r1.loc[i], r2.loc[i])
    return n_cell, n_all, n_both, best


def main():
    n_perm = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    d = F.to_model(F.load_files())
    f = F.build_features(d)
    keep = f["n_prior"] >= 1
    d, f = d[keep].reset_index(drop=True), f[keep].reset_index(drop=True)
    d["prev_jockey"] = d.groupby("horse")["jockey"].shift(1)
    d["jockey_changed"] = (d["prev_jockey"].notna()) & (d["jockey"] != d["prev_jockey"])
    d["prev_finish"] = d.groupby("horse")["finish"].shift(1)
    y = (d["finish"] == 1).astype(int).to_numpy()
    cut = d["date"].quantile(0.3)
    tr, te = (d["date"] < cut).to_numpy(), (d["date"] >= cut).to_numpy()
    fx, _ = F.encode_categoricals(f)
    pays = load_payout_a("data/payout/a.csv")

    meta = d.loc[te, ["raceid", "umaban", "sex", "age", "course", "prev_finish", "jockey_changed"]].copy()
    for c in ["fieldsize", "raceclass", "distance", "weeks_since", "wtcarry", "bodywt_change"]:
        meta[c] = f.loc[te, c].to_numpy()

    m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=31, subsample=0.8,
                           colsample_bytree=0.8, min_child_samples=100, verbose=-1, random_state=0)
    m.fit(fx[tr], y[tr], categorical_feature=F.CAT_COLS)
    meta["prob"] = m.predict_proba(fx[te])[:, 1]

    rows = []
    for rid, g in meta.groupby("raceid", sort=False):
        p = pays.get(rid)
        if p is None or not p["sanrentan"] or len(g) <= NHIMO:
            continue
        g = g.sort_values("prob", ascending=False, kind="mergesort")
        nums = g["umaban"].astype(int).tolist()
        a = g.iloc[0]
        r = a.to_dict()
        r["pay"] = hits(p[KEY[TNAME]], nums[0], set(nums[1:NHIMO + 1]))
        r["year"] = p["date"].year
        rows.append(r)
    df = pd.DataFrame(rows)
    axes = build_axes(df)
    pay = df["pay"].to_numpy(float)
    second = (df["year"] >= SPLIT_YEAR).to_numpy()
    pts = POINTS[TNAME](NHIMO) * 100
    print(f"対象 {len(df):,}R  軸{len(axes.columns)}本  最大{MAX_AXES}軸まで重ねる  最低R={MIN_R}")

    n_cell, n_all, n_both, best = scan(axes, pay, second, pts)
    print(f"\n■ 実データ")
    print(f"  総セル数 {n_cell:,}  /  全体100%超 {n_all}  /  **両期間100%超 {n_both}**")
    if best:
        print(f"  最良セル: {best[0]}  ROI {best[1]:.1f}%  R={best[2]}  前半{best[3]:.1f}% 後半{best[4]:.1f}%")

    print(f"\n■ 帰無分布（払戻をシャッフル×{n_perm}回。条件と払戻の関係が消えた偽データ）")
    rng = np.random.default_rng(0)
    na, nb, bs = [], [], []
    for i in range(n_perm):
        sh = rng.permutation(pay)
        _, a_, b_, be = scan(axes, sh, second, pts)
        na.append(a_)
        nb.append(b_)
        bs.append(be[1] if be else np.nan)
    print(f"  全体100%超:  平均{np.mean(na):.1f}  範囲{min(na)}〜{max(na)}   （実データ {n_all}）")
    print(f"  両期間100%超: 平均{np.mean(nb):.1f}  範囲{min(nb)}〜{max(nb)}   （実データ {n_both}）")
    print(f"  偽データの最良セルROI: 平均{np.nanmean(bs):.1f}%  最大{np.nanmax(bs):.1f}%   （実データ {best[1]:.1f}%）")
    p_all = (np.array(na) >= n_all).mean()
    p_both = (np.array(nb) >= n_both).mean()
    print(f"\n  → p値(全体100%超の個数): {p_all:.2f}   p値(両期間100%超の個数): {p_both:.2f}")
    print("  ※p値が小さくない＝実データの結果は偶然のシャッフルと区別がつかない。")


if __name__ == "__main__":
    main()
