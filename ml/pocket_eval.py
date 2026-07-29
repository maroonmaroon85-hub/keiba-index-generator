"""
好ポケット戦略の回収率を、ML予測CSV × 配当A（三連系を含む払戻）で実測する。

(24)の新モデル（騎手・厩舎あり）と (25)の期間拡大（9年OOS）を**同時に**かけた数字を出すための集計。
HANDOFF (25) の実施条件メモにある「新モデル × 9年797R の交差検証は未実施」を埋める。

ポケット定義（研究で最良だった条件・(25)と同一）:
  gap(モデル1位prob − 2位prob) ≥ 0.20  かつ  モデル1位馬の単勝オッズ 2.0〜5.0倍  かつ  ダート
買い方:
  三連単 一軸マルチ×紐4 (36点) / 三連複 軸1×紐4 (6点) / 馬連 軸1×紐4 (4点)
  軸=モデル1位、紐=モデル2〜5位（(23)で現行が最良と確認済み）

使い方:
  python3 ml/pocket_eval.py --pred out/ml_test_pred_early.csv --payout data/payout/a.csv
  python3 ml/pocket_eval.py --pred out/ml_test_pred.csv --payout data/payout/a.csv   # 従来168Rの再現
"""
import argparse, csv, io, itertools, json
from collections import defaultdict

import numpy as np
import pandas as pd

GAP_TH = 0.20
FAV_LO, FAV_HI = 2.0, 5.0

# 券種 → (推奨の点数, 着順どおりか, 表示名)
BETS = {
    "三連単マルチ×紐4": (36, True),
    "三連複 軸×紐4": (6, False),
    "馬連 軸×紐4": (4, False),
}


def _slots(r, start, count, width, nhorses, pay_idx):
    """(馬番×nhorses, 配当) のスロット列を読む。payout>0 のものだけ返す。check_result.py と同一ロジック。"""
    out = {}
    for s in range(count):
        b = start + s * width
        try:
            pay = int(r[b + pay_idx])
            horses = [int(r[b + i]) for i in range(nhorses)]
        except (ValueError, IndexError):
            continue
        if pay > 0 and all(h > 0 for h in horses):
            out[tuple(horses)] = pay
    return out


def load_payout_a(path):
    """配当A(224列) → {raceId: {"date","surface","umaren","sanrenpuku","sanrentan"}}。
    raceId=col14（成績col41の先頭8桁と一致）。馬場種別=col9 なので DS の再読込は不要。"""
    out = {}
    with open(path, "rb") as fh:
        txt = fh.read().decode("shift_jis", "replace")
    for r in csv.reader(io.StringIO(txt)):
        if len(r) < 224:
            continue
        rid = r[14].strip()
        if len(rid) != 8:
            continue
        try:
            date = pd.Timestamp(f"20{r[0].strip()}-{r[1].strip().zfill(2)}-{r[2].strip().zfill(2)}")
        except ValueError:
            continue
        out[rid] = {
            "date": date,
            "surface": r[9].strip(),
            "umaren": _slots(r, 115, 3, 4, 2, 2),
            "sanrenpuku": _slots(r, 179, 3, 5, 3, 3),
            "sanrentan": _slots(r, 194, 6, 5, 3, 3),
        }
    return out


def combos_for(axis, partners):
    """軸1頭＋紐4頭から各券種の購入組を作る。"""
    tan = [tuple(p) for a, b in itertools.permutations(partners, 2)
           for p in ([axis, a, b], [a, axis, b], [a, b, axis])]  # 一軸マルチ=軸が1〜3着のどれでも
    puku = [tuple(sorted((axis, a, b))) for a, b in itertools.combinations(partners, 2)]
    ren = [tuple(sorted((axis, a))) for a in partners]
    return {"三連単マルチ×紐4": tan, "三連複 軸×紐4": puku, "馬連 軸×紐4": ren}


def kelly_optimal(payoffs, points):
    """1レースあたりの純収益率サンプル（払戻/購入額 − 1）から、対数成長を最大化する f を数値探索。
    f=賭ける資金割合。全額購入時の純利益率が payoffs（-1 = 全没）。"""
    x = np.asarray(payoffs, dtype=float)
    best_f, best_g = 0.0, 0.0
    for f in np.arange(0.0, 0.2001, 0.0005):
        g = np.mean(np.log1p(f * x))
        if g > best_g:
            best_f, best_g = f, g
    return best_f, best_g


def boot_ci(cost, ret, n_boot=2000, seed=0):
    """レース単位のブートストラップで回収率の95%CIとプラス見込み確率を出す。"""
    rng = np.random.default_rng(seed)
    cost = np.asarray(cost, float)
    ret = np.asarray(ret, float)
    n = len(cost)
    idx = rng.integers(0, n, size=(n_boot, n))
    rois = ret[idx].sum(axis=1) / cost[idx].sum(axis=1) * 100
    return np.percentile(rois, 2.5), np.percentile(rois, 97.5), float((rois > 100).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default="out/ml_test_pred_early.csv")
    ap.add_argument("--payout", default="data/payout/a.csv")
    ap.add_argument("--gap", type=float, default=GAP_TH)
    ap.add_argument("--fav-lo", type=float, default=FAV_LO)
    ap.add_argument("--fav-hi", type=float, default=FAV_HI)
    ap.add_argument("--no-pocket", action="store_true", help="ポケット条件を外して全レースで集計")
    ap.add_argument("--min-horses", type=int, default=5, help="この頭数未満のレースは除外（TS側の既定は8）")
    ap.add_argument("--out", help="結果JSONの保存先")
    args = ap.parse_args()

    pays = load_payout_a(args.payout)
    print(f"配当A: {len(pays):,}レース読込")

    d = pd.read_csv(args.pred, dtype={"raceid": str})
    d = d.dropna(subset=["prob"])
    print(f"予測: {len(d):,}頭 / {d['raceid'].nunique():,}レース")

    # レースごとに モデル上位5頭・gap・軸オッズ を作る
    recs = defaultdict(lambda: {t: {"cost": 0.0, "ret": 0.0} for t in BETS})
    rows = []
    for rid, g in d.groupby("raceid", sort=False):
        p = pays.get(rid)
        if p is None or not p["sanrentan"]:
            continue
        g = g.sort_values("prob", ascending=False)
        if len(g) < max(5, args.min_horses):
            continue
        nums = g["umaban"].astype(int).tolist()
        probs = g["prob"].tolist()
        gap = probs[0] - probs[1]
        fav_odds = float(g["odds"].iloc[0])
        if not args.no_pocket:
            if gap < args.gap:
                continue
            if not (args.fav_lo <= fav_odds <= args.fav_hi):
                continue
            if p["surface"] != "ダ":
                continue
        axis, partners = nums[0], nums[1:5]
        cb = combos_for(axis, partners)
        row = {"raceid": rid, "year": p["date"].year, "gap": gap, "fav_odds": fav_odds}
        for tname, (npt, ordered) in BETS.items():
            key = {"三連単マルチ×紐4": "sanrentan", "三連複 軸×紐4": "sanrenpuku", "馬連 軸×紐4": "umaren"}[tname]
            wins = p[key]
            if not wins:
                row[tname] = None
                continue
            cost = len(cb[tname]) * 100
            ret = sum(wins.get(c, 0) for c in cb[tname])
            row[tname] = (cost, ret)
        rows.append(row)

    if not rows:
        print("対象レース0件"); return

    df = pd.DataFrame(rows)
    label = "全レース" if args.no_pocket else f"好ポケット(gap≥{args.gap} & 軸{args.fav_lo}-{args.fav_hi}倍 & ダート)"
    print(f"\n===== {label}: {len(df):,}レース  {df['year'].min()}-{df['year'].max()} =====")

    result = {"pred": args.pred, "races": len(df), "pocket": not args.no_pocket, "types": {}}
    print(f"{'券種':<22}{'R':>6}{'回収率':>9}{'95%CI':>18}{'>100%':>8}{'ケリーf':>9}{'成長':>8}")
    for tname in BETS:
        v = df[tname].dropna()
        if v.empty:
            continue
        cost = np.array([c for c, _ in v])
        ret = np.array([r for _, r in v])
        roi = ret.sum() / cost.sum() * 100
        lo, hi, pplus = boot_ci(cost, ret)
        f, growth = kelly_optimal(ret / cost - 1.0, len(cost))
        print(f"{tname:<22}{len(v):>6}{roi:>8.1f}%{f'{lo:.0f}-{hi:.0f}%':>18}{pplus*100:>7.0f}%{f*100:>8.1f}%{growth*100:>7.2f}")
        result["types"][tname] = {"races": int(len(v)), "roi": roi, "ci": [lo, hi],
                                  "p_profit": pplus, "kelly_f": f, "log_growth": growth}

    # 年別（本線=三連単）
    print("\n■ 年別 回収率")
    per_year = {}
    for tname in BETS:
        parts = []
        for y, gy in df.groupby("year"):
            v = gy[tname].dropna()
            if v.empty:
                continue
            c = sum(c for c, _ in v); r = sum(r for _, r in v)
            parts.append(f"{y}:{r/c*100:.0f}%({len(v)}R)")
            per_year.setdefault(tname, {})[int(y)] = r / c * 100
        print(f"  {tname:<22}" + " ".join(parts))
    result["per_year"] = per_year

    if args.out:
        json.dump(result, open(args.out, "w"), ensure_ascii=False, indent=1)
        print(f"\n→ {args.out}")


if __name__ == "__main__":
    main()
