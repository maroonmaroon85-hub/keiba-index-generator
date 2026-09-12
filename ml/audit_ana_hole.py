"""(209) ★★★**穴側に寄せて、券種と点数を選ぶ** —— 前半で選び、後半で1回だけ

★★**動機（2026-09-07・利用者の指定）**:
　★**「同じだけ優位があるなら、高配当を取れる穴側に寄せたい。それがこのセッションの意図」**
　→ ★★**目的関数が変わった**。**「モデルの優位が在るか」ではなく「儲かるか」で選ぶ**。

⚠★**(208)の訂正を先に**: 私は「穴に寄せる操作には価値がない」と書いたが**言い過ぎ**だった。
| 下限L | 後半R | 後半の差 | 前半ROI | ★**後半ROI** |
|---|---|---|---|---|
| 1.0 | 1,324 | +17.1円 | 96.7% | 97.9% |
| 7.0（選ばれた） | 503 | +15.7円 | 100.5% | 97.0% |
| ★**10.0** | **318** | ★**+18.7円** | ★**101.9%** | ★★**102.9%** |
　★**L=10.0 は前半・後半とも100%超で、後半の対応差も全Lで最高**。
　⚠**主判定が落ちたのは前半が L=7.0 を選んだから**。**「価値がない」ではなく
　　「事前登録した選び方では拾えなかった」が正確**。⚠**post-hoc・後半318本**。
　→ ★**Lを10で止めたのが早すぎた。上限を見ていない**。

────────────────────────────────────────────────────────────
★★★ 事前登録（2026-09-07・**結果を見る前にコミットする**）
────────────────────────────────────────────────────────────

■ ★経路（判定基準25）: **q = MLモデル / q_pool = 複勝の板**。⚠**q側は弱い**。
■ ★軸: **pn≥0.15 かつ gap≥0.15 かつ ★単勝オッズ ≥ L の中で pn最大**。**候補が無ければ見送る**。
　**L ∈ {1, 5, 10, 15, 20, 30}**（★**(208)は10で止めた。その先を見る**）。
■ ★買い方: **紐 = モデル上位k頭**（**(198)で三連単はp降順でないと壊れると確認済み**）。
| 券種 | k | 点数 |
|---|---|---|
| **複勝** | — | 1 |
| **ワイド** | 1, 2 | k |
| **馬連** | 1, 2 | k |
| **三連複** | 2, 3 | C(k,2) |
| **三連単** | 2 | 2（軸1着固定） |
　**6 × 8 = 48マス**。

■ ★★★**選ぶ基準（★先に決める）**: **前半(2016-2020)の★ROI**。
　★**利用者の目的が「儲け」なので、優位の出どころは問わない**。
　⚠**危険**: **ROIで選ぶと(88)の帯の癖を拾う**——**モデル由来でない分も混じる**。
　★★**だから「同じオッズ帯の乱」を全マスで併記し、どれだけがモデル由来かを見えるようにする**。

■ ★★★★主判定（**2比較**）: **前半で選んだ1マスを、後半(2021-)で1回だけ試す**
　**① 本（モデル）② プラセボ（乱・10種平均・同じ手続き）**。
　★**判定: 後半ROIの99%CI下端が100%を超えるか**、**および 本の縮み vs プラセボの縮み**。
| ★**後半の下端が100%超** | ⚠**100%を跨ぐ／下回る** |
|---|---|
| **穴側に寄せる価値がある** | ⚠**48マスから選んだ下駄**。**現行のまま** |

■ ★★ゲート2（判定基準42）——**仮説が偽なら何を返すか**
　★**買い方に情報が無いなら、前半で選んだマスの後半ROIは、48マスの後半平均に一致する**。
　★**それも記述として出す**（**48マスの後半平均 vs 選ばれた1マスの後半**）。
　★**乱は同じオッズ帯から引くので、その帯のROIを返す**（⚠**払戻率ではない**・(197)で外した）。

■ ⚠ゲート1: (88)③④を再現（±3pt）／ ★ゲート板: 復元R 0.800±0.020 ／
　★陽性対照: 三連複BOX上位4 81.9%±1.5pt。
■ ★★★内部対照（**決定的・乱を使わない**）: **L=1・複勝 の全期間ROIが 95.9% ±0.2pt かつ 6,336R ±5**。

■ ★★★探索を守る（**主判定は2比較のみ**・Bonferroni α=0.01/2）
　★★**48マスの探索は前半で完結する**——**後半は1回しか見ない**。
　⚠**後半を見てからマスを変えない**。**変えたらこの設計は無意味になる**。
　⚠**前半・後半とも300本未満のマスは探索から除く**（判定基準5）。

■ ★記述（判定しない）
　**48マス全部の 買う率 / 1日の本数 / 軸の平均・中央オッズ / 平均人気 /
　　前半ROI / 後半ROI / 乱の前半・後半 / 的中率**。

■ 予想（⚠**当てにしない**・判定基準24。★**私は(198)(201)(203)で3回外した**）
　★**後半は100%を割ると見る**——**48マスから選ぶので下駄が大きい**（**(201)で25.8pt縮んだ**）。
　★**もし通れば、この線で初めて「勝てる」と言える**。⚠**通らなくても穴側の否定にはならない**
　　——**(208)のL=10のように「選び方が拾えなかっただけ」がありうる**。**そこは記述で見る**。

実行: python3 ml/audit_ana_hole.py    自己テスト: python3 ml/audit_ana_hole.py --selftest
"""
import math
import sys
from itertools import combinations
from zlib import crc32

import numpy as np

sys.path.insert(0, "ml")
import features as F
from audit_crosspool import LINE, load_races, payoff, zq
from audit_ana_odds import BANDS, MIN_HORSES, band_of, gate1, roi_of
from audit_ana_marg import WF_BOX4, WF_TOL, wf_predict
from audit_ana_board import BOARD_R, BOARD_TOL, NPLACE, load_fuku_boards, qpool
from audit_ana_band import PN_FLOOR
from audit_ana_ladder import FINE
from train_prod import add_odds_features

GAP = 0.15
LS = [1.0, 5.0, 10.0, 15.0, 20.0, 30.0]
BETS = [("複勝", 0), ("ワイド", 1), ("ワイド", 2), ("馬連", 1), ("馬連", 2),
        ("三連複", 2), ("三連複", 3), ("三連単", 2)]
SPLIT = 2021
NRAND = 10
SEED = 20260906
MINCELL = 300
KNOWN_ROI, KNOWN_N, ROI_TOL, N_TOL = 95.9, 6336, 0.2, 5
NCMP = 2
ALPHA = 0.01


def tickets(kind, k, ax, himo):
    """★買い目。無理なら None"""
    if kind == "複勝":
        return [("複勝", [ax])]
    hs = himo[:k]
    if len(hs) < k:
        return None
    if kind in ("ワイド", "馬連"):
        return [(kind, [ax, h]) for h in hs]
    if kind == "三連複":
        return [("三連複", sorted([ax, a, b])) for a, b in combinations(hs, 2)]
    if kind == "三連単":
        return [("三連単", [ax, hs[0], hs[1]]), ("三連単", [ax, hs[1], hs[0]])]
    return None


def selftest():
    ok = True
    z = zq(ALPHA / NCMP)
    print(f"★軸のオッズ下限 L: {LS}（★**(208)は10で止めた。その先を見る**）")
    print(f"★買い方 {len(BETS)}通り × L {len(LS)}通り = **{len(BETS)*len(LS)}マス**")
    print(f"{'券種':<8}{'k':>3}{'点数':>5}{'払戻率':>8}")
    for kind, k in BETS:
        t = tickets(kind, k, 1, [2, 3, 4])
        print(f"{kind:<8}{k:>3}{len(t) if t else 0:>5}{100*LINE[kind]:>7.1f}%")
        ok &= t is not None
    print(f"★★選ぶ基準は**前半のROI**（利用者の目的が「儲け」なので出どころは問わない）")
    print(f"　⚠**危険**: ROIで選ぶと(88)の帯の癖を拾う → ★**乱を全マスで併記する**")
    print(f"★主判定は {NCMP}比較のみ（本 / プラセボ）→ z = {z:.3f}"
          f"　★**48マスの探索は前半で完結する**")
    rng = np.random.default_rng(0)
    v = rng.choice([0.0, 1000.0], size=40_000, p=[0.9225, 0.0775])
    idx = rng.integers(0, len(v), size=(2000, len(v)))
    b = 100.0 * v[idx].mean(axis=1) / 100.0
    print(f"★ゲート2の検算: 払戻率77.5%の腕の「100%超の割合」= **{100*np.mean(b>100):.1f}%**"
          f"　{'★OK' if np.mean(b>100) < 0.01 else '⚠NG'}")
    ok &= np.mean(b > 100) < 0.01
    print(f"★★★内部対照（決定的・乱を使わない）: **L=1・複勝 が {KNOWN_ROI}% ±{ROI_TOL}pt "
          f"かつ {KNOWN_N:,}R ±{N_TOL}**")
    print("★自己テスト: " + ("全部OK" if ok else "⚠NG"))
    return 0 if ok else 1


def main():
    z = zq(ALPHA / NCMP)
    print("(209) ★★★**穴側に寄せて、券種と点数を選ぶ** —— 前半で選び、後半で1回だけ")
    print("★利用者の指定: **同じだけ優位があるなら、高配当を取れる穴側に寄せたい**\n")

    races = {r["rid"]: r for r in load_races()}
    rows, bad = gate1(list(races.values()))
    print("⚠**ゲート1**: (88)③④を別パーサで再現・許容±3pt")
    for nm, n, roi, known, dd, okg in rows:
        print(f"　{nm:<12}{roi:>7.1f}% vs {known:>5.1f}%　差 {dd:+.1f}pt"
              f"　{'★立った' if okg else '⚠落ちた'}")
    if bad:
        print("\n⚠⚠**ゲート1が落ちた。読まない**。")
        return

    print("\n★複勝の板(type=2)を読む…")
    boards = load_fuku_boards()
    print(f"　**{len(boards):,}レース分**")

    d = F.to_model(F.load_files())
    f = F.build_features(d)
    keep = (f["n_prior"] >= 1) & d["odds"].notna() & (d["odds"] > 0)
    d, f = d[keep].reset_index(drop=True), f[keep].reset_index(drop=True)
    y = (d["finish"] <= 3).astype(int).to_numpy()
    fx, _ = F.encode_categoricals(f)
    fx = add_odds_features(fx, d["odds"].to_numpy(float), d["raceid"].to_numpy())
    pred = wf_predict(d, fx, y, 3)
    msk = ~np.isnan(pred)
    sub = d.loc[msk, ["raceid", "umaban", "odds", "date"]].copy()
    sub["p"] = pred[msk]

    CELLS = [(L, kind, k) for L in LS for kind, k in BETS]
    K = {c: {"a": [], "r": [], "yr": [], "od": [], "rk": []} for c in CELLS}
    full = []
    Rs, box4, nall, ndays = [], [], 0, set()
    for rid, g in sub.groupby("raceid"):
        rid = str(rid)
        r = races.get(rid)
        bd = boards.get(rid)
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
        nall += 1
        ndays.add(gg["date"].iloc[0].date())
        pn = pv / pv.sum() * NPLACE
        qp, Rb = qpool([bd[int(u)] for u in ub], "harm")
        Rs.append(Rb)
        gap = pn - qp
        order_p = [int(u) for u in ub[np.argsort(-pv, kind="mergesort")]]
        bx = [payoff(r, "三連複", list(c)) for c in combinations(sorted(order_p[:4]), 3)]
        if not any(x is None for x in bx):
            box4.append(sum(bx) - 400.0)
        pos = {int(u): q for q, u in enumerate(ub)}
        rank = np.argsort(np.argsort(od)) + 1
        yr = int(gg["date"].iloc[0].year)
        base = (pn >= PN_FLOOR) & (gap >= GAP)
        c0 = np.where(base)[0]
        if len(c0):
            i0 = int(c0[int(np.argmax(pn[c0]))])
            v0 = payoff(r, "複勝", [int(ub[i0])])
            if v0 is not None:
                full.append(v0)
        for L in LS:
            cand = np.where(base & (od >= L))[0]
            if not len(cand):
                continue
            i = int(cand[int(np.argmax(pn[cand]))])
            ax = int(ub[i])
            himo = [u for u in order_p if u != ax]
            # ★乱: 軸と紐を、それぞれオッズ±20%以内の無作為な馬に置き換える
            draws = []
            okd = True
            for sd in range(NRAND):
                g2 = np.random.default_rng([SEED + sd, crc32(rid.encode())])
                out = []
                for u in [ax] + himo[:3]:
                    k0 = pos[u]
                    pl = [int(ub[q]) for q in range(len(ub))
                          if int(ub[q]) not in out
                          and od[k0] / FINE <= od[q] <= od[k0] * FINE]
                    if not pl:
                        okd = False
                        break
                    out.append(int(g2.choice(pl)))
                if not okd:
                    break
                draws.append(out)
            if not okd:
                continue
            for kind, k in BETS:
                tk = tickets(kind, k, ax, himo)
                if tk is None:
                    continue
                va = [payoff(r, nm2, sel) for nm2, sel in tk]
                if any(x is None for x in va):
                    continue
                cost = 100.0 * len(tk)
                acc, okr = [], True
                for t in draws:
                    tr = tickets(kind, k, t[0], t[1:])
                    if tr is None:
                        okr = False
                        break
                    vr = [payoff(r, nm2, sel) for nm2, sel in tr]
                    if any(x is None for x in vr):
                        okr = False
                        break
                    acc.append(sum(vr) / cost)
                if not okr:
                    continue
                c = K[(L, kind, k)]
                c["a"].append(sum(va) / cost)
                c["r"].append(float(np.mean(acc)))
                c["yr"].append(yr); c["od"].append(float(od[i])); c["rk"].append(int(rank[i]))
    print(f"\n★対象 **{nall:,}レース / {len(ndays):,}開催日**")

    med = float(np.median(Rs))
    okR = abs(med - BOARD_R) <= BOARD_TOL
    print(f"■ ★ゲート板: 復元R = **{med:.4f}** → **{'★立った' if okR else '⚠⚠落ちた'}**")
    box4 = np.asarray(box4, float)
    g0 = 100.0 * (box4.sum() + 400.0 * len(box4)) / (400.0 * len(box4))
    okb = abs(g0 - WF_BOX4) <= WF_TOL
    print(f"■ ⚠**陽性対照**: BOX上位4 **{g0:.1f}%** vs {WF_BOX4}%"
          f" → **{'★立った' if okb else '⚠⚠落ちた'}**")
    fv = np.asarray(full, float)
    okc = (abs(roi_of(fv) - KNOWN_ROI) <= ROI_TOL and abs(len(fv) - KNOWN_N) <= N_TOL)
    print(f"■ ★★★内部対照（決定的）: L=1・複勝 = **{roi_of(fv):.1f}%**（{len(fv):,}R） vs "
          f"{KNOWN_ROI}%（{KNOWN_N:,}R） → **{'★再現' if okc else '⚠⚠ズレた'}**")
    if not (okR and okb and okc):
        print("\n⚠⚠**ゲートが落ちた。読まない**（判定基準32）。")
        return

    tbl = []
    for c in CELLS:
        x = K[c]
        yr = np.asarray(x["yr"], int)
        if len(yr) == 0:
            continue
        h = yr < SPLIT
        if h.sum() < MINCELL or (~h).sum() < MINCELL:
            continue
        a = np.asarray(x["a"], float) * 100.0
        rv = np.asarray(x["r"], float) * 100.0
        od = np.asarray(x["od"], float)
        tbl.append({"c": c, "n1": int(h.sum()), "n2": int((~h).sum()),
                    "r1": a[h].mean(), "r2": a[~h].mean(),
                    "q1": rv[h].mean(), "q2": rv[~h].mean(),
                    "v2": a[~h], "w2": rv[~h],
                    "od": od.mean(), "mod": np.median(od), "rk": np.mean(x["rk"]),
                    "hit": 100.0 * np.mean(a > 0), "n": len(a),
                    "day": len(a) / len(ndays)})
    print(f"\n★探索できたマス **{len(tbl)}/{len(CELLS)}**")

    print(f"\n{'='*126}")
    print("■ ★★記述: **全マス**（**前半ROIの降順**・⚠**この表で選ばない。選ぶのは前半のROIだけ**）")
    print(f"{'L':>5}{'券種':<8}{'k':>3}{'点':>4}{'R数':>8}{'1日':>7}{'買う率':>8}"
          f"{'軸平均':>8}{'軸中央':>8}{'人気':>6}{'的中率':>8}"
          f"{'★前半ROI':>10}{'★後半ROI':>10}{'乱前半':>8}{'乱後半':>8}")
    for x in sorted(tbl, key=lambda t: -t["r1"]):
        L, kind, k = x["c"]
        npt = len(tickets(kind, k, 1, [2, 3, 4]))
        print(f"{L:>5.0f}{kind:<8}{k:>3}{npt:>4}{x['n']:>8,}{x['day']:>6.1f}本"
              f"{100*x['n']/nall:>7.1f}%{x['od']:>7.1f}倍{x['mod']:>7.1f}倍{x['rk']:>5.1f}番"
              f"{x['hit']:>7.1f}%{x['r1']:>9.1f}%{x['r2']:>9.1f}%"
              f"{x['q1']:>7.1f}%{x['q2']:>7.1f}%")

    print(f"\n{'='*126}")
    print("■ ★★★★主判定: **前半で選び、後半で1回だけ**")
    for who, k1, ky in (("本（モデル）", "r1", "v2"), ("プラセボ（乱）", "q1", "w2")):
        best = max(tbl, key=lambda t: t[k1])
        v = np.asarray(best[ky], float)
        mu = v.mean()
        se = v.std(ddof=1) / math.sqrt(len(v))
        lo, hi = mu - z * se, mu + z * se
        avg2 = float(np.mean([t["r2" if who.startswith("本") else "q2"] for t in tbl]))
        L, kind, k = best["c"]
        print(f"\n★**{who}**: 前半で選ばれたマス "
              f"**L={L:.0f} / {kind} / 紐{k}頭**（前半 {best[k1]:.1f}%・{best['n1']:,}R）")
        print(f"　★★**後半（1回だけ）**: **{mu:.1f}%**（{best['n2']:,}R）"
              f"　99%CI [{lo:.1f},{hi:.1f}]"
              f"　→ **{'★★100%超' if lo > 100 else '⚠通らない'}**")
        print(f"　★**縮み {best[k1]-mu:+.1f}pt** ／ **全マスの後半平均 {avg2:.1f}%**"
              f"　→ **選択の価値 {mu-avg2:+.1f}pt**")
        if who.startswith("本"):
            print(f"　★**同じマスの乱**: 前半 {best['q1']:.1f}% → 後半 {best['q2']:.1f}%"
                  f"　→ ★**モデル由来 {mu-best['q2']:+.1f}pt**")

    print("\n■ ★採用条件: **後半ROIの99%CI下端が100%超（主判定）**")
    print("⚠**後半を見てからマスを変えない**——**変えたらこの設計は無意味になる**。")
    print("\n⚠**枠連の運用には触れない**。**設定変更は提案しない**。")


if __name__ == "__main__":
    sys.exit(selftest() if "--selftest" in sys.argv else (main() or 0))
