"""
紐（相手）の頭数を総当たりで振り、券種ごとの最適な形を探す。

背景: 現行の「紐4」は (18)(23) で決めたものだが、その根拠は
**旧モデル × 3.8年168R** ＝ (24)(25)(26) で不合格になった設定そのもの。
紐数だけが無傷である理由は無いので、(26)と同じ土台（新モデル × 10年OOS）で引き直す。

買い方（軸=モデル1位、紐=モデル2位以降のn頭）:
  三連単 一軸マルチ×紐n … 3n(n-1)点   軸が1〜3着のどれでもよく、相手2頭の順序も問わない
  三連複 軸1×紐n        … C(n,2)点
  馬連   軸1×紐n        … n点

高速化: 購入組を列挙して払戻を引くのではなく、**的中組が購入範囲に入るか**を判定する。
三連単マルチと三連複は「軸が含まれ、残り2頭がどちらも紐」で的中条件が同一（点数と配当だけが違う）。

使い方:
  python3 ml/himo_sweep.py --pred out/ml_test_pred_early.csv --payout data/payout/a.csv
  python3 ml/himo_sweep.py ... --no-pocket     # ポケット絞り込みを外す（(26)より高ROIの母集団）
"""
import argparse
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "ml")
from pocket_eval import GAP_TH, FAV_HI, FAV_LO, boot_ci, kelly_optimal, load_payout_a

# 券種 → 紐n頭のときの点数
POINTS = {
    "三連単マルチ": lambda n: 3 * n * (n - 1),
    "三連複 軸流し": lambda n: n * (n - 1) // 2,
    "馬連 軸流し": lambda n: n,
}
KEY = {"三連単マルチ": "sanrentan", "三連複 軸流し": "sanrenpuku", "馬連 軸流し": "umaren"}


def hits(wins, axis, partners):
    """的中組のうち購入範囲に入るものの払戻合計。partners は集合（同着があるので合計する）。"""
    total = 0
    for combo, pay in wins.items():
        s = set(combo)
        if axis not in s:
            continue
        if s - {axis} <= partners:  # 軸以外が全て紐に含まれる
            total += pay
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default="out/ml_test_pred_early.csv")
    ap.add_argument("--payout", default="data/payout/a.csv")
    ap.add_argument("--gap", type=float, default=GAP_TH)
    ap.add_argument("--fav-lo", type=float, default=FAV_LO)
    ap.add_argument("--fav-hi", type=float, default=FAV_HI)
    ap.add_argument("--no-pocket", action="store_true")
    ap.add_argument("--max-himo", type=int, default=8)
    ap.add_argument("--min-horses", type=int, default=8)
    args = ap.parse_args()

    ns = list(range(2, args.max_himo + 1))
    # 紐数で母集団が変わると比較にならないので、最大の紐数を賄える頭数のレースに固定する
    need = max(args.min_horses, args.max_himo + 1)

    pays = load_payout_a(args.payout)
    print(f"配当A: {len(pays):,}レース読込")
    d = pd.read_csv(args.pred, dtype={"raceid": str}).dropna(subset=["prob"])
    print(f"予測: {len(d):,}頭 / {d['raceid'].nunique():,}レース  （{need}頭立て以上に限定）")

    rows = []
    for rid, g in d.groupby("raceid", sort=False):
        p = pays.get(rid)
        if p is None or not p["sanrentan"]:
            continue
        if len(g) < need:
            continue
        g = g.sort_values("prob", ascending=False)
        probs = g["prob"].tolist()
        gap = probs[0] - probs[1]
        fav_odds = float(g["odds"].iloc[0])
        if not args.no_pocket:
            if gap < args.gap or not (args.fav_lo <= fav_odds <= args.fav_hi) or p["surface"] != "ダ":
                continue
        nums = g["umaban"].astype(int).tolist()
        axis = nums[0]
        row = {"raceid": rid, "year": p["date"].year}
        for n in ns:
            partners = set(nums[1 : n + 1])
            for tname in POINTS:
                row[(tname, n)] = hits(p[KEY[tname]], axis, partners)
        rows.append(row)

    if not rows:
        print("対象レース0件")
        return

    df = pd.DataFrame(rows)
    label = "全レース" if args.no_pocket else f"好ポケット(gap≥{args.gap} & 軸{args.fav_lo}-{args.fav_hi}倍 & ダート)"
    yr = df["year"]
    mid = int(np.median(yr))
    print(f"\n===== {label}: {len(df):,}レース  {yr.min()}-{yr.max()} =====")

    for tname in POINTS:
        print(f"\n■ {tname}")
        print(f"{'紐':>3}{'点/R':>6}{'的中率':>8}{'回収率':>9}{'95%CI':>16}{'>100%':>7}{'ケリーf':>8}{'成長':>7}"
              f"{'  前半/後半(' + str(mid) + '境)':>18}")
        for n in ns:
            pts = POINTS[tname](n)
            ret = df[(tname, n)].to_numpy(float)
            cost = np.full(len(ret), pts * 100.0)
            roi = ret.sum() / cost.sum() * 100
            hit = (ret > 0).mean() * 100
            lo, hi, pplus = boot_ci(cost, ret)
            f, growth = kelly_optimal(ret / cost - 1.0, pts)
            a, b = (yr < mid).to_numpy(), (yr >= mid).to_numpy()
            r1 = ret[a].sum() / cost[a].sum() * 100 if a.any() else float("nan")
            r2 = ret[b].sum() / cost[b].sum() * 100 if b.any() else float("nan")
            print(f"{n:>3}{pts:>6}{hit:>7.2f}%{roi:>8.1f}%{f'{lo:.0f}-{hi:.0f}%':>16}"
                  f"{pplus * 100:>6.0f}%{f * 100:>7.1f}%{growth * 100:>6.2f}{r1:>9.0f}%/{r2:.0f}%")


if __name__ == "__main__":
    main()
