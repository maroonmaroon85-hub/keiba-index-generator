"""
絞り込み軸の第2弾: 馬場状態(良/稍/重/不) と 血統(軸馬の父・母父) で絞れるかを検証する。

(28)でオッズ帯×馬場種別を引き直した続き。カテゴリ数が桁違いに多い（父1,225種・母父1,778種）ため、
**最良カテゴリを探して報告するのは無意味**（エッジ0でも100%超は必ず出る）。
そこで(28)と同じく期間を二分し、探索期間で選んだものを検証期間で一度だけ測る。

さらに血統では「探索期に100%を超えた父が、検証期でも100%を超え続ける割合」を出す。
エッジが無ければこの割合は偶然の水準（≒検証期に100%を超える父の全体割合）に一致するはずで、
そこを有意に上回るかどうかが唯一の判定基準になる。

使い方:
  python3 ml/filter_search2.py --min-races 150
"""
import argparse
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "ml")
import features as F
from himo_sweep import POINTS, KEY, hits
from pocket_eval import load_payout_a

STRATS = [("三連単マルチ", 8), ("馬連 軸流し", 4)]
CONDS = ["良", "稍", "重", "不"]


def roi_of(sub, tname, n):
    if len(sub) == 0:
        return float("nan"), 0
    ret = sub[(tname, n)].to_numpy(float).sum()
    return ret / (len(sub) * POINTS[tname](n) * 100) * 100, len(sub)


def report_axis(df, col, label, args, top=12):
    """カテゴリ列 col で層別し、探索期の上位を検証期と突き合わせる。"""
    tr, va = df[df["year"] < args.split_year], df[df["year"] >= args.split_year]
    for tname, n in STRATS:
        base_tr, _ = roi_of(tr, tname, n)
        base_va, _ = roi_of(va, tname, n)
        print(f"\n■ {label} × {tname}×紐{n}   絞り込みなし: 探索{base_tr:.1f}% / 検証{base_va:.1f}%")
        rows = []
        for v, g_tr in tr.groupby(col):
            if len(g_tr) < args.min_races:
                continue
            r_tr, n_tr = roi_of(g_tr, tname, n)
            r_va, n_va = roi_of(va[va[col] == v], tname, n)
            rows.append((v, r_tr, n_tr, r_va, n_va))
        if not rows:
            print("  （最低R数を満たすカテゴリなし）")
            continue
        rows.sort(key=lambda r: -r[1])
        print(f"  {'カテゴリ':<18}{'探索R':>7}{'探索ROI':>9}{'検証R':>7}{'検証ROI':>9}")
        for v, r_tr, n_tr, r_va, n_va in rows[:top]:
            print(f"  {str(v):<18}{n_tr:>7}{r_tr:>8.1f}%{n_va:>7}{r_va:>8.1f}%")
        if len(rows) > top:
            print(f"  （…全{len(rows)}カテゴリ中の上位{top}）")

        # 探索期100%超が検証期でも100%超を保つ割合 vs 偶然の水準
        hi = [r for r in rows if r[1] > 100]
        keep = [r for r in hi if r[3] > 100]
        all_va_hi = [r for r in rows if r[3] > 100]
        base_rate = len(all_va_hi) / len(rows) * 100
        line = f"  → 探索期100%超 {len(hi)}/{len(rows)}"
        if hi:
            line += f" → うち検証期も100%超 {len(keep)}/{len(hi)} ({len(keep)/len(hi)*100:.0f}%)"
        line += f"   ※偶然の水準={base_rate:.0f}%（検証期に100%超となるカテゴリの全体割合）"
        print(line)
        if hi:
            shrink = np.mean([r[1] - r[3] for r in hi])
            print(f"  → 探索期100%超カテゴリの平均縮小幅: {shrink:+.1f}pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default="out/ml_test_pred_early.csv")
    ap.add_argument("--payout", default="data/payout/a.csv")
    ap.add_argument("--split-year", type=int, default=2022)
    ap.add_argument("--min-races", type=int, default=150, help="探索期にこのR数未満のカテゴリは対象外")
    ap.add_argument("--min-horses", type=int, default=9)
    args = ap.parse_args()

    need = max(args.min_horses, max(n for _, n in STRATS) + 1)
    print("DS*.CSV を読み込み中（1分ほど）…")
    feat = F.to_model(F.load_files())[["raceid", "umaban", "sire", "damsire", "cond"]]
    feat["umaban"] = pd.to_numeric(feat["umaban"], errors="coerce")

    pays = load_payout_a(args.payout)
    d = pd.read_csv(args.pred, dtype={"raceid": str}).dropna(subset=["prob"])
    key = feat.set_index(["raceid", "umaban"])

    rows = []
    for rid, g in d.groupby("raceid", sort=False):
        p = pays.get(rid)
        if p is None or not p["sanrentan"] or len(g) < need:
            continue
        g = g.sort_values("prob", ascending=False)
        nums = g["umaban"].astype(int).tolist()
        axis = nums[0]
        try:
            info = key.loc[(rid, axis)]
        except KeyError:
            continue
        if isinstance(info, pd.DataFrame):
            info = info.iloc[0]
        row = {"year": p["date"].year, "surface": p["surface"],
               "cond": info["cond"], "sire": info["sire"], "damsire": info["damsire"]}
        for tname, n in STRATS:
            row[(tname, n)] = hits(p[KEY[tname]], axis, set(nums[1 : n + 1]))
        rows.append(row)

    df = pd.DataFrame(rows)
    tr = (df["year"] < args.split_year).sum()
    print(f"\n対象 {len(df):,}R（探索 {tr:,} / 検証 {len(df) - tr:,}）  最低R数={args.min_races}")

    print("\n" + "=" * 70 + "\n【軸1】馬場状態\n" + "=" * 70)
    report_axis(df, "cond", "馬場状態", args, top=8)

    print("\n" + "=" * 70 + "\n【軸2】馬場状態 × 馬場種別\n" + "=" * 70)
    df2 = df.copy()
    df2["cs"] = df2["surface"] + df2["cond"]
    report_axis(df2, "cs", "馬場種別×状態", args, top=10)

    print("\n" + "=" * 70 + "\n【軸3】軸馬の父\n" + "=" * 70)
    report_axis(df, "sire", "父", args, top=12)

    print("\n" + "=" * 70 + "\n【軸4】軸馬の母父\n" + "=" * 70)
    report_axis(df, "damsire", "母父", args, top=12)


if __name__ == "__main__":
    main()
