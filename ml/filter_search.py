"""
レース絞り込み条件（軸の単勝オッズ帯 × 馬場）を、旧設定を一切引き継がずゼロから探し直す。

背景: 現行の「軸2-5倍 & ダート」は ②③(2026-07-24) で **旧モデル × 3.8年** の層別探索から選んだ定数。
(21)の時点で「2.0倍の断崖は過学習の疑い」と自ら記録しており、(24)(25)(26)で土台が崩れた後も
この2条件だけは検証されずに残っていた。(27)で紐4が谷だったのと同じ疑いがかかるため引き直す。

**探索の罠への対策（これが本スクリプトの主眼）**:
グリッドを舐めて最良セルを報告するのは 141.9% を生んだ手順そのもの。そこで期間を二分し、
  探索期間(前半5年)で条件を選ぶ → 選んだ条件を検証期間(後半5年)で**一度だけ**測る
という順序を強制する。検証期間の数字は「選んだ後」に初めて見るため、探索バイアスが乗らない。
併せて、両期間で100%を超えたセル数を出し、偶然どれだけ出るかを可視化する。

使い方:
  python3 ml/filter_search.py --pred out/ml_test_pred_early.csv --payout data/payout/a.csv
"""
import argparse
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "ml")
from himo_sweep import POINTS, KEY, hits
from pocket_eval import load_payout_a

# (下限, 上限) 軸=モデル1位馬の単勝オッズ。上限は未満。
ODDS_BANDS = [(1.0, 1.5), (1.5, 2.0), (2.0, 3.0), (3.0, 5.0), (5.0, 7.0), (7.0, 10.0), (10.0, 1e9)]
SURFACES = ["芝", "ダ", "両方"]
# 検証する買い方（軸=モデル1位、紐=モデル2位以降n頭）
STRATS = [("三連単マルチ", 4), ("三連単マルチ", 8), ("馬連 軸流し", 4)]


def band_label(lo, hi):
    return f"{lo:g}倍〜" if hi > 1e8 else f"{lo:g}-{hi:g}倍"


def roi(sub, tname, n):
    """サブセットの回収率(%)とR数。購入額は 点数×100円×R。"""
    if sub.empty:
        return float("nan"), 0
    pts = POINTS[tname](n)
    ret = sub[(tname, n)].to_numpy(float).sum()
    return ret / (len(sub) * pts * 100) * 100, len(sub)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default="out/ml_test_pred_early.csv")
    ap.add_argument("--payout", default="data/payout/a.csv")
    ap.add_argument("--split-year", type=int, default=2022, help="この年以降を検証期間にする")
    ap.add_argument("--min-races", type=int, default=100, help="探索期間でこのR数未満のセルは候補にしない")
    ap.add_argument("--gap", type=float, default=0.0, help="gap下限（既定0＝gapで絞らない）")
    ap.add_argument("--min-horses", type=int, default=9)
    args = ap.parse_args()

    max_n = max(n for _, n in STRATS)
    need = max(args.min_horses, max_n + 1)

    pays = load_payout_a(args.payout)
    d = pd.read_csv(args.pred, dtype={"raceid": str}).dropna(subset=["prob"])
    print(f"配当A {len(pays):,}R / 予測 {d['raceid'].nunique():,}R  →  {need}頭立て以上に限定")

    rows = []
    for rid, g in d.groupby("raceid", sort=False):
        p = pays.get(rid)
        if p is None or not p["sanrentan"] or len(g) < need:
            continue
        g = g.sort_values("prob", ascending=False)
        probs = g["prob"].tolist()
        if probs[0] - probs[1] < args.gap:
            continue
        nums = g["umaban"].astype(int).tolist()
        axis = nums[0]
        row = {"year": p["date"].year, "surface": p["surface"], "odds": float(g["odds"].iloc[0]),
               "gap": probs[0] - probs[1]}
        for tname, n in STRATS:
            row[(tname, n)] = hits(p[KEY[tname]], axis, set(nums[1 : n + 1]))
        rows.append(row)

    df = pd.DataFrame(rows)
    tr = df[df["year"] < args.split_year]
    va = df[df["year"] >= args.split_year]
    print(f"\n探索期間 {df['year'].min()}-{args.split_year - 1}: {len(tr):,}R"
          f"   ／   検証期間 {args.split_year}-{df['year'].max()}: {len(va):,}R")
    if args.gap > 0:
        print(f"（gap≥{args.gap} で事前に絞った母集団）")

    for tname, n in STRATS:
        base_tr, _ = roi(tr, tname, n)
        base_va, _ = roi(va, tname, n)
        print(f"\n■ {tname}×紐{n}  絞り込みなし: 探索{base_tr:.1f}% / 検証{base_va:.1f}%")
        print(f"{'馬場':<5}{'オッズ帯':<10}{'探索R':>7}{'探索ROI':>9}{'検証R':>7}{'検証ROI':>9}")
        cells = []
        for surf in SURFACES:
            for lo, hi in ODDS_BANDS:
                def pick(x):
                    m = (x["odds"] >= lo) & (x["odds"] < hi)
                    if surf != "両方":
                        m &= x["surface"] == surf
                    return x[m]

                r_tr, n_tr = roi(pick(tr), tname, n)
                r_va, n_va = roi(pick(va), tname, n)
                cells.append((surf, (lo, hi), r_tr, n_tr, r_va, n_va))
                print(f"{surf:<5}{band_label(lo, hi):<10}{n_tr:>7}{r_tr:>8.1f}%{n_va:>7}{r_va:>8.1f}%")

        ok = [c for c in cells if c[3] >= args.min_races and not np.isnan(c[2])]
        if not ok:
            continue
        best = max(ok, key=lambda c: c[2])
        print(f"  → 探索期間の最良セル: {best[0]} {band_label(*best[1])} "
              f"探索{best[2]:.1f}%({best[3]}R)  ⇒ **検証{best[4]:.1f}%({best[5]}R)**")
        n100_tr = sum(1 for c in cells if c[2] > 100)
        n100_va = sum(1 for c in cells if c[4] > 100)
        both = sum(1 for c in cells if c[2] > 100 and c[4] > 100)
        print(f"  → 100%超のセル数: 探索{n100_tr}/{len(cells)}  検証{n100_va}/{len(cells)}  両方{both}")

    # 旧条件との突き合わせ（gapを含む固定条件を、探索/検証で別々に測る）
    print("\n\n===== 旧条件の分解（gapがどこで効いているかを見る） =====")
    conds = {
        "絞り込みなし": df["odds"] > 0,
        "旧: ダ & 2-5倍": (df["surface"] == "ダ") & (df["odds"] >= 2) & (df["odds"] < 5),
        "旧: ダ & 2-5倍 & gap≥0.2": (df["surface"] == "ダ") & (df["odds"] >= 2) & (df["odds"] < 5) & (df["gap"] >= 0.2),
        "ダートのみ": df["surface"] == "ダ",
        "2-5倍のみ": (df["odds"] >= 2) & (df["odds"] < 5),
        "10倍超を除外": df["odds"] < 10,
    }
    for tname, n in STRATS:
        print(f"\n■ {tname}×紐{n}")
        print(f"{'条件':<26}{'探索17-21':>14}{'検証22-26':>14}{'全体':>12}")
        for lab, m in conds.items():
            s = df[m]
            a = roi(s[s["year"] < args.split_year], tname, n)
            b = roi(s[s["year"] >= args.split_year], tname, n)
            c = roi(s, tname, n)
            print(f"{lab:<26}{a[0]:>7.1f}%({a[1]:>5}){b[0]:>7.1f}%({b[1]:>5}){c[0]:>7.1f}%")


if __name__ == "__main__":
    main()
