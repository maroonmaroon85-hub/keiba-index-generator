"""(202) ★**直近1ヶ月の明細**（★記述のみ・判定しない）——「実際に買ったら何が起きたか」

⚠⚠**これは証拠ではない**（判定基準5）。**1ヶ月は数十レースしかない**。
　★**(200)で見た通り、三連単は上位1%のレースが払戻の81%を作る**——
　**1ヶ月では「大穴が1本入ったかどうか」しか分からない**。
★**それでも出す理由**: **利用者が「実際に買ったら何が起きたか」を見たいと言ったから**。
　★**数字の意味を誤らないよう、同じ表に「11年の実測ROI」を必ず並べる**。

■ ★買い方（**(197)〜(201)と同一**）
　**軸 = (推奨度≥0.15 かつ ズレ≥0.02) を満たす馬のうち ズレ最大**、**紐 = モデル上位順**。
　**複勝(軸1点) / 馬連(軸−紐1・1点) / 三連複(軸+紐2・1点) / 三連単(軸1着固定・2点)**。
■ ★**ウォークフォワードの予測を使う**（**その年を学習に含まない**）。

実行: python3 ml/audit_ana_recent.py
"""
import sys
from itertools import combinations

import numpy as np
import pandas as pd

sys.path.insert(0, "ml")
import features as F
from audit_crosspool import load_races, payoff
from audit_ana_odds import MIN_HORSES, roi_of
from audit_ana_marg import wf_predict
from audit_ana_board import NPLACE, load_fuku_boards, qpool
from train_prod import add_odds_features

FL = (0.15, 0.02)
ARMS = ["複勝", "馬連", "三連複", "三連単"]
NPT = {"複勝": 1, "馬連": 1, "三連複": 1, "三連単": 2}
LONG = {"複勝": 91.5, "馬連": 90.0, "三連複": 91.3, "三連単": 103.1}
DAYS = 30


def arm_tickets(kind, ax, h1, h2):
    if kind == "複勝":
        return [("複勝", [ax])]
    if kind == "馬連":
        return [("馬連", [ax, h1])]
    if kind == "三連複":
        return [("三連複", sorted([ax, h1, h2]))]
    return [("三連単", [ax, h1, h2]), ("三連単", [ax, h2, h1])]


def main():
    print("(202) ★**直近1ヶ月の明細**（★記述のみ・判定しない）")
    print("⚠⚠**これは証拠ではない**——**1ヶ月では「大穴が1本入ったか」しか分からない**\n")
    races = {r["rid"]: r for r in load_races()}
    boards = load_fuku_boards()
    d = F.to_model(F.load_files())
    f = F.build_features(d)
    keep = (f["n_prior"] >= 1) & d["odds"].notna() & (d["odds"] > 0)
    d, f = d[keep].reset_index(drop=True), f[keep].reset_index(drop=True)
    y = (d["finish"] <= 3).astype(int).to_numpy()
    fx, _ = F.encode_categoricals(f)
    fx = add_odds_features(fx, d["odds"].to_numpy(float), d["raceid"].to_numpy())
    pred = wf_predict(d, fx, y, 3)
    msk = ~np.isnan(pred)
    sub = d.loc[msk, ["raceid", "umaban", "odds", "date", "finish"]].copy()
    sub["p"] = pred[msk]
    last = sub["date"].max()
    lo = last - pd.Timedelta(days=DAYS)
    print(f"★データの最終日: **{last.date()}**　→ **{lo.date()} 〜 {last.date()} の{DAYS}日間**\n")
    recent = sub[sub["date"] > lo]

    rowsout, tot = [], {a: [0.0, 0.0, 0] for a in ARMS}
    for rid, g in recent.groupby("raceid"):
        r = races.get(str(rid))
        bd = boards.get(str(rid))
        if r is None or bd is None:
            continue
        nums = {u for u, _, _ in r["horses"]}
        if len(nums) < MIN_HORSES:
            continue
        gg = g[g["umaban"].astype(int).isin(nums)]
        ub = gg["umaban"].astype(int).to_numpy()
        if len(gg) < MIN_HORSES or not all(int(u) in bd for u in ub):
            continue
        od = gg["odds"].to_numpy(float)
        pv = gg["p"].to_numpy(float)
        if not np.isfinite(od).all() or (od <= 0).any() or pv.sum() <= 0:
            continue
        pn = pv / pv.sum() * NPLACE
        qp, _ = qpool([bd[int(u)] for u in ub], "harm")
        gap = pn - qp
        cand = np.where((pn >= FL[0]) & (gap >= FL[1]))[0]
        if not len(cand):
            continue
        ai = int(cand[int(np.argmax(gap[cand]))])
        axu = int(ub[ai])
        order_p = [int(u) for u in ub[np.argsort(-pv, kind="mergesort")]]
        himo = [u for u in order_p if u != axu]
        if len(himo) < 2:
            continue
        vals, okall = {}, True
        for a in ARMS:
            vs = [payoff(r, nm, sel) for nm, sel in arm_tickets(a, axu, himo[0], himo[1])]
            if any(v is None for v in vs):
                okall = False
                break
            vals[a] = sum(vs)
        if not okall:
            continue
        fin = {int(u): int(x) for u, x in zip(gg["umaban"].astype(int),
                                              gg["finish"].astype(int))}
        for a in ARMS:
            tot[a][0] += 100.0 * NPT[a]
            tot[a][1] += vals[a]
            tot[a][2] += 1
        rowsout.append({
            "date": gg["date"].iloc[0].date(), "rid": str(rid), "ax": axu,
            "axod": float(od[ai]), "axfin": fin.get(axu, 0),
            "h1": himo[0], "h1fin": fin.get(himo[0], 0),
            "h2": himo[1], "h2fin": fin.get(himo[1], 0),
            "gap": float(gap[ai]), "pn": float(pn[ai]),
            **{a: vals[a] for a in ARMS}})

    if not rowsout:
        print("⚠**対象レースが1つも無い**")
        return
    print(f"★対象 **{len(rowsout)}レース**（{DAYS}日間）\n")
    print(f"{'日付':<12}{'軸':>4}{'オッズ':>8}{'着':>4}{'推奨':>7}{'ズレ':>7}"
          f"{'紐1':>5}{'着':>4}{'紐2':>5}{'着':>4}"
          + "".join(f"{a:>9}" for a in ARMS))
    for x in rowsout:
        print(f"{str(x['date']):<12}{x['ax']:>4}{x['axod']:>7.1f}倍{x['axfin']:>4}"
              f"{x['pn']:>7.3f}{x['gap']:>7.3f}"
              f"{x['h1']:>5}{x['h1fin']:>4}{x['h2']:>5}{x['h2fin']:>4}"
              + "".join(f"{int(x[a]):>8,}円" if x[a] > 0 else f"{'−':>9}" for a in ARMS))

    print(f"\n{'='*100}")
    print(f"■ ★**{DAYS}日間の集計**　⚠**11年の実測ROIを必ず並べる**（**1ヶ月では決まらない**）")
    print(f"{'腕':<8}{'点数/R':>7}{'買った額':>11}{'払戻':>11}{'★収支':>11}"
          f"{'★ROI':>9}{'的中':>7}{'★11年のROI':>12}{'差':>9}")
    for a in ARMS:
        sp, pay, n = tot[a]
        hit = sum(1 for x in rowsout if x[a] > 0)
        roi = 100.0 * pay / sp if sp else 0.0
        print(f"{a:<8}{NPT[a]:>7}{sp:>10,.0f}円{pay:>10,.0f}円"
              f"{pay-sp:>+10,.0f}円{roi:>8.1f}%{hit:>4}/{n:<3}"
              f"{LONG[a]:>11.1f}%{roi-LONG[a]:>+8.1f}pt")
    print(f"\n⚠⚠**この{DAYS}日間の数字は、11年の実測ROIの推定にはならない**（判定基準5）。")
    print("★**(200)の通り、三連単は上位1%のレースが払戻の81%を作る**——"
          "**1ヶ月に大穴が入るかどうかは運**。")
    print("\n⚠**枠連の運用には触れない**。**設定変更は提案しない**。")


if __name__ == "__main__":
    sys.exit(main() or 0)
