"""(237) ★**買う2点の的中を全件出す** —— 前半・後半の日付の範囲と、払戻の一覧

★★**動機（2026-09-09・利用者の指定）**: ★**「後半っていつからいつまで？ 的中した払い戻し一覧を出して」**

■ ★**これは記述のみ**。**新しいマスを作らない。軸も買い目も基準も動かさない**。
　★**(234)で数えた的中を、日付と払戻つきで並べ直すだけ**。
■ ★**前半 = 2016〜2020 / 後半 = 2021〜**（`audit_ana_hole.SPLIT = 2021`）。
■ ★★★内部対照（**決定的**）——⚠★**訂正（実行後・結果を読む前）**
　⚠**初版は「的中数が (234) と一致」を対照にして落ちた**（**91/16・19/8 vs 89/16・19/7**）。
　★**(234)の的中数は1,383本（±20%の乱が引けたレース）で数えたもの**。
　★**この測定は乱を使わないので対象は全数1,398本**——**差の15レースに的中が4本ある**。
　⚠★★**これは判定基準37の8回目。同じ1,383/1,398の取り違えが3回連続**
　　（**(233)で注記を書き、(235)で踏み、ここでまた踏んだ**）。
　　→ ★**注記では効かないので、★対照の作り方を変える**:
　★★**① 対象レース数の合計が 1,398 であること**（**軸が立った全数・(234)(235)と共通**）
　★★**② 的中数が (234) の値を下回らないこと**（**乱の条件を外したので増えることはあっても減らない**）
　★★**③ 超過が15レース分（＝各券種15本）を超えないこと**
　⚠**3つ全部が立たなければ読まない**。

■ ★★★**結果（実測済み・2026-09-09）**
　★**内部対照3つとも立った**（対象1,398・的中が(234)以上・超過はP2本/X1本）。

■ ★**日付の範囲**
| | 期間 | レース | 開催日 |
|---|---|---|---|
| ★**前半** | ★**2016-01-05 〜 2020-12-26** | **1,071** | 443日 |
| ★**後半** | ★**2021-01-05 〜 2026-07-19** | ★**327** | 252日 |
　⚠**後半は5年半で327レース＝前半5年1,071レースの約1/3**（**(235)の該当減少がそのまま出ている**）。

■ ★★**P馬単M4点（主判定）の後半 — 16本・払戻合計 139,320円**
　★**中央 4,205円 / 平均 8,708円 / ★最大 71,990円 / 最小 1,670円**
　⚠★**(234)で「17,998円の1本」と書いたのは★1レースあたりのROI換算値**（**払戻÷点数×100**）。
　　★**実際の払戻は 71,990円**（**2025-01-06・軸16番48.9倍・買い目16-10**）。
　　⚠★**16本の払戻合計139,320円のうち、この1本が52%**。
　⚠**2023年は1本、2024年も1本**——**2年で2本しか当たっていない**。

■ ★★**X三連単A4点の後半 — 8本・払戻合計 263,730円**
　★**中央 30,690円 / 平均 32,966円 / 最大 80,940円（2024-02-24）/ 最小 8,500円**
　⚠**2023年と2025年は的中ゼロ**。★**5年半で8本＝8ヶ月に1回**。
　★**配当の分布は健全**（**最小8,500円**）——**P馬単M4点（最小1,670円・中央4,205円）より、
　　当たれば必ず大きい**。

■ ★**2点が同じ日に当たった日**: **2021-05-08 / 2021-08-14 / 2022-03-12 / 2022-12-04**
　★**同じ軸を共有しているので当然**だが、⚠**(226)Aの相関0.316が実物で見える**
　　——**当たる日は重なる**。

■ ⚠★★**この測定で私が踏んだ誤り（★判定基準37の8回目・3回連続）**
　★**初版の対照を「的中数が(234)と一致」にして落ちた**（**91/16・19/8 vs 89/16・19/7**）。
　★**(234)は1,383本（乱の条件つき）、この測定は全数1,398本**——**差の15レースに的中4本**。
　⚠**(233)で注記を書き、(235)で踏み、ここでまた踏んだ**。★**注記では効かなかった**。
　→ ★★**対照の作り方を変えた**: **「対象レース数を必ず1,398と突き合わせる」を
　　★対照の①に置いた**（**②的中数が(234)以上・③超過が15本以内**）。

実行: python3 ml/audit_ana_hits.py    自己テスト: python3 ml/audit_ana_hits.py --selftest
"""
import sys

import numpy as np

sys.path.insert(0, "ml")
import features as F
from audit_crosspool import load_races, payoff
from audit_ana_odds import MIN_HORSES
from audit_ana_board import NPLACE, load_fuku_boards, qpool
from audit_ana_band import PN_FLOOR
from audit_ana_fix import LFIX
from audit_ana_hole import GAP, SPLIT
from audit_ana_marg import wf_predict
from audit_ana_bet import BUY, tickets
from train_prod import add_odds_features

KNOWN = {"P馬単M4点": (89, 16), "X三連単A4点": (19, 7)}   # ★(234)＝乱の条件つき1,383本
KNOWN_R = 1398   # ★★軸が立った全数（(234)(235)と共通）。★新しいスクリプトは必ずここと突き合わせる
DIFF_MAX = 15    # ★乱が引けなかったレース数


def selftest():
    print(f"★前半 = 2016〜{SPLIT - 1} / 後半 = {SPLIT}〜")
    print(f"★買う2点: " + " / ".join(f"{lab}（紐{c[0]}・{c[1]}・{c[2]}点）" for lab, c in BUY))
    print(f"★★★内部対照（★訂正版・3つ全部）")
    print(f"　★① 対象レース数の合計 = **{KNOWN_R:,}**（軸が立った全数）")
    print(f"　★② 的中数が (234) 以上　★③ 超過が {DIFF_MAX} 本以内")
    for k, (a, b) in KNOWN.items():
        print(f"　　{k:<14} (234) 前半 {a:>3}本 / 後半 {b:>3}本"
              f"　⚠**これは1,383本（乱の条件つき）で数えた値**")
    print("⚠**記述のみ。マスも軸も買い目も基準も動かさない**")
    print("★自己テスト: 全部OK")
    return 0


def main():
    print("(237) ★**買う2点の的中を全件出す**\n")
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
    m = ~np.isnan(pred)
    sub = d.loc[m, ["raceid", "umaban", "odds", "date"]].copy()
    sub["p"] = pred[m]

    rows = {lab: [] for lab, _ in BUY}
    days = {"前半": [], "後半": []}
    nrace = {"前半": 0, "後半": 0}
    for rid, g in sub.groupby("raceid"):
        rid = str(rid)
        r, bd = races.get(rid), boards.get(rid)
        if r is None or bd is None:
            continue
        nums = {u for u, _, _ in r["horses"]}
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
        cand = np.where((pn >= PN_FLOOR) & (gap >= GAP) & (od >= LFIX))[0]
        if not len(cand):
            continue
        i = int(cand[int(np.argmax(pn[cand]))])
        ax = int(ub[i])
        op = [int(u) for u in ub[np.argsort(-pv, kind="mergesort")] if int(u) != ax]
        oq = [int(u) for u in ub[np.argsort(od, kind="mergesort")] if int(u) != ax]
        HM = {"P": op, "X": op[:2] + [u for u in oq if u not in op[:2]]}
        dt = gg["date"].iloc[0]
        half = "前半" if dt.year < SPLIT else "後半"
        ok = True
        buf = []
        for lab, c in BUY:
            tk = tickets(c[1], c[2], ax, HM[c[0]])
            if tk is None:
                ok = False
                break
            va = [payoff(r, k2, sel) for k2, sel in tk]
            if any(x is None for x in va):
                ok = False
                break
            buf.append((lab, sum(va), 100 * len(tk),
                        [sel for (k2, sel), v in zip(tk, va) if v]))
        if not ok:
            continue
        nrace[half] += 1
        days[half].append(dt)
        for lab, v, cost, wsel in buf:
            if v > 0:
                rows[lab].append({"d": dt, "rid": rid, "ax": ax, "od": float(od[i]),
                                  "gap": float(gap[i]), "v": v, "c": cost,
                                  "sel": wsel, "half": half})

    tot_r = nrace["前半"] + nrace["後半"]
    ok1 = tot_r == KNOWN_R
    print(f"★★★内部対照（★訂正版）")
    print(f"　★①対象レース数 **{tot_r:,}**（{KNOWN_R:,}であること）"
          f"　{'★立った' if ok1 else '⚠落ちた'}")
    ok23 = True
    for lab in rows:
        a = sum(1 for x in rows[lab] if x["half"] == "前半")
        b = sum(1 for x in rows[lab] if x["half"] == "後半")
        ka, kb = KNOWN[lab]
        g = (a >= ka and b >= kb and (a - ka) + (b - kb) <= DIFF_MAX)
        ok23 &= g
        print(f"　★②③{lab:<14} 前半 {a:>3}本 / 後半 {b:>3}本"
              f"　（(234)は {ka}/{kb}・乱の条件つき1,383本）"
              f"　超過 {(a - ka) + (b - kb)}本　{'★立った' if g else '⚠落ちた'}")
    okc = ok1 and ok23
    if not okc:
        print("⚠⚠**対照が落ちた。読まない**（判定基準32）。")
        return

    print(f"\n■ ★**日付の範囲**")
    for h in ("前半", "後半"):
        dd = sorted(days[h])
        print(f"　★**{h}**　{dd[0].date()} 〜 {dd[-1].date()}"
              f"　**{nrace[h]:,}レース**（{len(set(x.date() for x in dd)):,}開催日）")

    for lab in rows:
        print(f"\n{'='*104}")
        print(f"■ ★★**{lab} の的中 全{len(rows[lab])}件**")
        for h in ("前半", "後半"):
            rs = sorted([x for x in rows[lab] if x["half"] == h], key=lambda x: x["d"])
            if not rs:
                continue
            tot = sum(x["v"] for x in rs)
            print(f"\n　★**{h}（{len(rs)}本・払戻合計 {tot:,.0f}円）**")
            print(f"　{'日付':<12}{'レース':<11}{'軸':>4}{'単勝':>8}{'ズレ':>7}"
                  f"{'★払戻':>10}{'収支':>10}  当たった買い目")
            for x in rs:
                sel = " / ".join("-".join(str(z) for z in s) for s in x["sel"])
                print(f"　{str(x['d'].date()):<12}{x['rid']:<11}{x['ax']:>4}"
                      f"{x['od']:>7.1f}倍{x['gap']:>7.3f}{x['v']:>9,.0f}円"
                      f"{x['v']-x['c']:>+9,.0f}円  {sel}")
            v = np.array([x["v"] for x in rs], float)
            print(f"　　★中央 {np.median(v):,.0f}円 / 平均 {v.mean():,.0f}円 / "
                  f"最大 {v.max():,.0f}円 / 最小 {v.min():,.0f}円")
    print("\n⚠**枠連の運用には触れない**。**設定変更は提案しない**。")


if __name__ == "__main__":
    sys.exit(selftest() if "--selftest" in sys.argv else (main() or 0))
