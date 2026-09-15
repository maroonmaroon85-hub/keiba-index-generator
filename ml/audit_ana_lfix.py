"""(246sns) ★★★**単勝の下限を下げると何が起きるか** —— 利用者の問い「9倍にしたらどうなる？」

■ ⚠★★★**先に線を引く**
　★**`ANA_SNS_BRIEF.md` の禁止事項4は「★的中率を上げるために軸を人気馬に変えること」**。
　★**単勝の下限は、★◎が穴馬であることの根拠そのもの**。**ここを下げるのは禁止事項4に触れる**。
　★★**だからこれは「測るだけ」**。⚠**採用の提案ではない**。
　★**採るかどうかは利用者が決める**。**私は材料を出すところまで**。

■ ★★**測る条件（★現行のSNS条件から、単勝の下限だけを動かす）**
```
pn ≥ 0.15  かつ  ズレ ≥ 0.10  かつ  単勝 ≥ L　★障害戦は除外
L ∈ {10.0（現行）, 9.0, 8.0, 7.0, 5.0}
```
　⚠**他は一切動かさない**。★**L=10.0 は現行なので内部対照になる**。

■ ★★★★**ゲート2（判定基準42）—— ★何を見れば「◎が穴馬でなくなった」と言えるか**
　★★**核心は★的中率ではない。★◎の中央人気である**。
　★**下限を下げても◎の中央人気が6番のままなら、★入ってくるのは同じ性格の馬**
　　＝**「本数が増えただけ」**。
　⚠**中央人気が前に動くなら、★◎は人気側の馬になった**＝**禁止事項4に触れる**。
　⚠★**「的中率が上がる」は情報を持たない**——**短いオッズを入れるのだから★機械的に上がる**。
　★**読むのは ①中央人気 ②中央オッズ ③複勝ROI の3つ**。

■ ★★内部対照（⚠**落ちたら読まない**）
　★**L=10.0 が (243sns) の平地・ズレ0.10 と一致**:
　**該当 5,015本 / 複勝的中率 22.0% / 軸の中央オッズ 16.2倍**。

■ 予想（⚠**当てにしない**・★このセッションは6勝5敗）
　★**9.0倍では中央人気は6番のまま**と見る（**9倍の馬は6〜7番人気の圏内**）。
　★**該当は1〜2割増える**と見る。★**複凝的中率は +1pt 程度**。
　⚠**5.0倍まで下げると中央人気が4〜5番に動く**と見る＝**そこは明確に別物**。
　★**複勝ROIは下限を下げるほど下がる**と見る（**市場は人気馬ほど正確**）。
　⚠**もし9.0倍でROIが上がるなら、それは「10.0倍という線に根拠が無かった」ことを意味する**。

実行: python3 ml/audit_ana_lfix.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import features as F
from audit_crosspool import load_races, payoff
from audit_ana_odds import MIN_HORSES, roi_of
from audit_ana_marg import wf_predict
from audit_ana_board import NPLACE, load_fuku_boards, qpool
from audit_ana_band import PN_FLOOR
from train_prod import add_odds_features

GAP = 0.10
LS = [10.0, 9.0, 8.0, 7.0, 5.0]
KNOWN_N, KNOWN_HIT, KNOWN_OD = 5015, 22.0, 16.2


def is_jump(dist):
    try:
        v = float(dist)
    except (TypeError, ValueError):
        return False
    return v > 0 and (v % 100 != 0 or v > 3600)


def main():
    print("(246sns) ★★★**単勝の下限を下げると何が起きるか**（利用者の問い）")
    print("⚠★**これは測るだけ。★採用の提案ではない**"
          "——**単勝の下限は◎が穴馬であることの根拠そのもの**\n")

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
    sub = d.loc[m, ["raceid", "umaban", "odds", "date", "distance"]].copy()
    sub["p"] = pred[m]

    A = {L: {"od": [], "rk": [], "fu": [], "ta": []} for L in LS}
    yrs = set()
    for rid, g0 in sub.groupby("raceid"):
        rid = str(rid)
        r, bd = races.get(rid), boards.get(rid)
        if r is None or bd is None:
            continue
        if is_jump(g0["distance"].iloc[0]):
            continue                              # ★(243sns) 障害は除外
        nums = {u for u, _, _ in r["horses"]}
        if len(nums) < MIN_HORSES:
            continue
        gg = g0[g0["umaban"].astype(int).isin(nums)]
        ub = gg["umaban"].astype(int).to_numpy()
        if len(gg) < MIN_HORSES or not all(int(u) in bd for u in ub):
            continue
        od = gg["odds"].to_numpy(float)
        pv = gg["p"].to_numpy(float)
        if not np.isfinite(od).all() or (od <= 0).any() or pv.sum() <= 0:
            continue
        pn = pv / pv.sum() * NPLACE
        qp, _R = qpool([bd[int(u)] for u in ub], "harm")
        gap = pn - qp
        rank = np.argsort(np.argsort(od)) + 1
        yrs.add(int(gg["date"].iloc[0].year))
        for L in LS:
            c = np.where((pn >= PN_FLOOR) & (gap >= GAP) & (od >= L))[0]
            if not len(c):
                continue
            i = int(c[int(np.argmax(pn[c]))])
            ax = int(ub[i])
            v0 = payoff(r, "複勝", [ax])
            if v0 is None:
                continue
            vt = payoff(r, "単勝", [ax])
            a = A[L]
            a["od"].append(float(od[i])); a["rk"].append(int(rank[i]))
            a["fu"].append(float(v0)); a["ta"].append(float(vt) if vt is not None else 0.0)

    ny = max(len(yrs), 1)
    a10 = A[10.0]
    fu10 = np.array(a10["fu"])
    hr = 100.0 * np.mean(fu10 > 0)
    omed = float(np.median(a10["od"]))
    print("■ ★★**内部対照**（**L=10.0 が (243sns) の平地・ズレ0.10 と一致するか**）")
    ok = [("該当", len(fu10), KNOWN_N, 5, abs(len(fu10) - KNOWN_N) <= 5),
          ("複勝的中率(%)", hr, KNOWN_HIT, 0.5, abs(hr - KNOWN_HIT) <= 0.5),
          ("軸の中央オッズ", omed, KNOWN_OD, 0.5, abs(omed - KNOWN_OD) <= 0.5)]
    for nm, got, want, tol, good in ok:
        print(f"　{nm:<16}{got:>9.1f} vs {want:>8.1f}　{'★通った' if good else '⚠落ちた'}")
    if not all(g for *_, g in ok):
        print("\n⚠⚠**内部対照が落ちた。読まない**（判定基準32）。")
        return

    print(f"\n■ ★★★**単勝の下限ごと**（**11年 / {ny}年・ズレ0.10・障害除外**）")
    print(f"{'下限':<8}{'該当':>8}{'年あたり':>9}{'1日':>7}"
          f"{'★中央人気':>11}{'平均人気':>10}{'★中央オッズ':>12}"
          f"{'複勝的中':>10}{'★複勝ROI':>11}{'単勝的中':>9}")
    for L in LS:
        a = A[L]
        if not a["fu"]:
            continue
        fu, ta = np.array(a["fu"]), np.array(a["ta"])
        rk, odd = np.array(a["rk"]), np.array(a["od"])
        tag = "★現行" if L == 10.0 else ""
        print(f"{L:<8.1f}{len(fu):>8,}{len(fu)/ny:>9.0f}{len(fu)/ny/104:>7.2f}"
              f"{np.median(rk):>10.0f}番{rk.mean():>10.1f}{np.median(odd):>11.1f}倍"
              f"{100*np.mean(fu>0):>9.1f}%{roi_of(fu):>10.1f}%"
              f"{100*np.mean(ta>0):>8.1f}%  {tag}")

    print(f"\n■ ★★★★**ゲート2の答え —— ★◎はまだ穴馬か**")
    b = A[10.0]
    r0, o0 = float(np.median(b["rk"])), float(np.median(b["od"]))
    for L in LS:
        if L == 10.0 or not A[L]["fu"]:
            continue
        a = A[L]
        dr = float(np.median(a["rk"])) - r0
        do = float(np.median(a["od"])) - o0
        v = ("★**変わらない（本数が増えただけ）**" if abs(dr) < 0.5
             else "⚠**人気側へ動いた＝禁止事項4に触れる**")
        print(f"　L={L:<5.1f} 中央人気 {dr:+.1f}番 / 中央オッズ {do:+.1f}倍　→ {v}")
    print("\n⚠★★**「的中率が上がる」は情報を持たない**——**短いオッズを入れるのだから機械的に上がる**。")
    print("★**読むのは ①中央人気 ②中央オッズ ③複勝ROI の3つ**。")


if __name__ == "__main__":
    main()
