"""(241sns) ★★★**ズレ下限0.10で4点セットを合わせたROI** —— 利用者の指定（2026-09-12）

★**利用者が(240sns)の梯子から G=0.10 を選んだ**（**1日4.63本・年の的中104.7本**）。
★**「roi出してみて」**。⚠**券種ごとのROIは(240sns)で測ってあるが、★合計は足し算では出ない**
　——**点数の重みが違い（100/100/200/400円）、★同じ軸を共有するので当たりが揃う**。

■ ★★これは何を足すのか（★マスは増えない）
　⚠**(240sns)で測った G=0.10 の4マスを、★1レース単位で合計するだけ**。
　★**新しい券種も新しい下限も足していない**。

■ ★測るもの
　1. ★**4点セット合計のROI**（**複勝100＋単勝100＋馬連200＋三連単400＝800円**）
　2. ★★**三連単を抜いた3点セット**（**400円**）← ⚠**(240sns)で三連単だけ採用保留にしたため**
　3. ★**乱との対応差の99%CI下端**／★**前半→後半の縮み**
　4. ★**4券種の相関**と、★**合計のse が「独立なら」からどれだけ離れるか**
　　（**(226)が G=0.15 で 1.19倍と出している。★同じ手続き**）

■ ★★内部対照（⚠**落ちたら読まない**）
　★**各券種のROIが(240sns)と一致すること**:
　**複勝 94.2% / 単勝 91.1% / 馬連2点 96.3% / 三連単A4点 161.3%**（**各 ±0.2pt・5,216本 ±5**）。
　★**該当レースは 5,297本**（**(240sns)①の G=0.10**）。

■ ★★★★ゲート2（判定基準42）—— **何を返せば「合計に意味が無い」か**
　★**4つが完全に独立なら、合計の se は √(Σ w²se²) に一致する**。
　★**完全に相関しているなら Σ w·se**。★**その2つを両端として出す**。
　⚠★★**合計ROIが100%を超えても、それが★三連単1本で作られているなら意味が無い**
　　——**(240sns)で三連単A4点G=0.10は「28マス中の1つ・当たり73本」として採用保留にした**。
　★**だから3点セット（三連単なし）を★必ず並べる**。**そちらが本体である**。

■ 予想（⚠**当てにしない**・★このセッションは5勝4敗）
　★**4点合計は100%を超える**と見る（**三連単の161.3%が400円分＝半分の重みを持つため**）。
　★★**3点セットは94〜96%**と見る（**複勝94.2・単勝91.1・馬連96.3の重み付き平均**）。
　⚠★**つまり「合計が100%超」は三連単だけで作られる**と予想する。**そうなら採用できない**。
　★**相関は(226)の0.259より高い**と見る（**同じ軸・同じ紐Pを4券種で共有しているため**）。


■ ★★★★**結果（実測済み・2026-09-12）**　★**内部対照6つとも通過**

| セット | 1レース | ROI | 的中(1点以上) | 対応差 | ★**99%CI下端** | ★**縮み** |
|---|---|---|---|---|---|---|
| **4点**（複勝＋単勝＋馬連2＋三連単A4） | 800円 | ★**127.9%** | 21.7% | +243.5円 | ★**+40.8** | **+3.5pt** |
| ★**3点**（三連単を抜く） | 400円 | **94.5%** | 21.7% | +34.3円 | ★**+0.8** | ★**+0.1pt** |

★★**差の +33.4pt は★全部三連単が作っている**（**事前登録した予想どおり**）。
⚠★**(240sns)で三連単A4点G=0.10は「28マス中の1つ・当たり73本」として採用保留**。
★★**ゲート2の答え: 合計が127.9%でも、★三連単だけで作られているので採用できない**。

★**相関と分散**（(226)と同じ手続き）:
　**相関の平均 0.360**（**(226)はG=0.15の別セットで0.259**）。
　**合計の se 15.50pt / 独立なら14.58 / 完全相関なら17.22 → ★独立比1.06倍**。
　★**分散はほとんど下がらない**——**4つとも同じ軸を共有しているため**。

★**3点セットの下端 +0.8 は0を超えている**が、⚠**ROIが94.5%＝★負ける買い目**。
　★**「乱より良い」と「儲かる」は別**（**(240sns)の裾外し比と同じ構図**）。

■ ★予想の答え合わせ（判定基準24・★このセッション7回目）
| 予想 | 実際 | |
|---|---|---|
| **4点合計は100%超** | ★**127.9%** | ★**当たり** |
| **3点は94〜96%** | ★**94.5%** | ★**当たり** |
| **「100%超」は三連単だけで作られる** | ★**+33.4pt が全部三連単** | ★**当たり** |
| **相関は(226)の0.259より高い** | ★**0.360** | ★**当たり** |

実行: python3 ml/audit_ana_g10.py
"""
import math
import os
import sys
from binascii import crc32

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import features as F
from audit_crosspool import load_races, payoff, zq
from audit_ana_odds import MIN_HORSES, roi_of
from audit_ana_marg import wf_predict
from audit_ana_board import NPLACE, load_fuku_boards, qpool
from audit_ana_band import PN_FLOOR
from audit_ana_hole import NRAND, SEED, SPLIT
from audit_ana_fix import LFIX
from audit_ana_ladder import FINE
from audit_ana_bet import tickets
from train_prod import add_odds_features

GAP = 0.10
SET = [("複勝", 0, 1), ("単勝", 0, 1), ("馬連", 2, 2), ("三連単A", 4, 4)]
KNOWN = {"複勝": 94.2, "単勝": 91.1, "馬連": 96.3, "三連単A": 161.3}
KNOWN_N, KNOWN_CAND, TOL, NTOL = 5216, 5297, 0.2, 5
ALPHA = 0.01


def main():
    z = zq(ALPHA)
    print(f"(241sns) ★★★**ズレ下限 {GAP} で4点セットを合わせたROI**（利用者の指定）")
    print("⚠**マスは増えない**——**(240sns)の4マスを1レース単位で合計するだけ**\n")

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

    rows, ncand = [], 0
    for rid, g0 in sub.groupby("raceid"):
        rid = str(rid)
        r, bd = races.get(rid), boards.get(rid)
        if r is None or bd is None:
            continue
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
        cand = np.where((pn >= PN_FLOOR) & (pn - qp >= GAP) & (od >= LFIX))[0]
        if not len(cand):
            continue
        i = int(cand[int(np.argmax(pn[cand]))])
        ax = int(ub[i])
        if payoff(r, "複勝", [ax]) is None:
            continue
        ncand += 1
        HP = [u for u in ub[np.argsort(-pv, kind="mergesort")] if int(u) != ax]
        HP = [int(u) for u in HP]
        pos = {int(u): q for q, u in enumerate(ub)}
        draws, okd = [], True
        for sd in range(NRAND):
            g2 = np.random.default_rng([SEED + sd, crc32(rid.encode())])
            out = []
            for u in [ax] + HP[:3]:
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
        vs, rs, ok = [], [], True
        for kind, k, _pts in SET:
            tk = tickets(kind, k, ax, HP)
            if tk is None:
                ok = False
                break
            va = [payoff(r, nm, sel) for nm, sel in tk]
            if any(x is None for x in va):
                ok = False
                break
            acc, okr = [], True
            for t in draws:
                tr = tickets(kind, k, t[0], t[1:])
                if tr is None:
                    okr = False
                    break
                vr = [payoff(r, nm, sel) for nm, sel in tr]
                if any(x is None for x in vr):
                    okr = False
                    break
                acc.append(sum(vr))
            if not okr:
                ok = False
                break
            vs.append(sum(va))
            rs.append(float(np.mean(acc)))
        if not ok:
            continue
        rows.append((int(gg["date"].iloc[0].year), vs, rs))

    if not rows:
        print("⚠**対象が0本**")
        return
    yr = np.array([x[0] for x in rows])
    V = np.array([x[1] for x in rows], float)     # 各券種の払戻[円]
    R = np.array([x[2] for x in rows], float)     # 乱の払戻[円]
    pts = np.array([p for _, _, p in SET], float)
    cost = pts * 100.0
    ny = len(set(yr.tolist()))

    print(f"★対象 **{ncand:,}レース**（**4券種すべて引けたのは {len(rows):,}本 / {ny}年**）")
    print("\n■ ★★**内部対照**（**(240sns)の G=0.10 と一致するか**）")
    okall = abs(ncand - KNOWN_CAND) <= NTOL and abs(len(rows) - KNOWN_N) <= NTOL
    print(f"　{'該当レース':<14}{ncand:>8,} vs {KNOWN_CAND:>8,}"
          f"　{'★通った' if abs(ncand-KNOWN_CAND)<=NTOL else '⚠落ちた'}")
    print(f"　{'4券種とも引けた':<14}{len(rows):>8,} vs {KNOWN_N:>8,}"
          f"　{'★通った' if abs(len(rows)-KNOWN_N)<=NTOL else '⚠落ちた'}")
    for j, (kind, _k, _p) in enumerate(SET):
        got = 100.0 * V[:, j].sum() / (cost[j] * len(rows))
        good = abs(got - KNOWN[kind]) <= TOL
        okall &= good
        print(f"　{kind:<14}{got:>7.1f}% vs {KNOWN[kind]:>7.1f}%"
              f"　{'★通った' if good else '⚠落ちた'}")
    if not okall:
        print("\n⚠⚠**内部対照が落ちた。読まない**（判定基準32）。")
        return

    def block(idx, lab):
        c = cost[idx].sum()
        v = V[:, idx].sum(axis=1)
        rr = R[:, idx].sum(axis=1)
        roi = 100.0 * v.sum() / (c * len(v))
        dd = v - rr
        lo = dd.mean() - z * dd.std(ddof=1) / math.sqrt(len(dd))
        r1 = 100.0 * v[yr < SPLIT].sum() / (c * max((yr < SPLIT).sum(), 1))
        r2 = 100.0 * v[yr >= SPLIT].sum() / (c * max((yr >= SPLIT).sum(), 1))
        print(f"\n■ ★★**{lab}**（**1レース {c:,.0f}円**）")
        print(f"　★**ROI {roi:.1f}%**　"
              f"的中（1点以上） {100*np.mean(v>0):.1f}%　"
              f"対応差 {dd.mean():+.1f}円　★**99%CI下端 {lo:+.1f}**")
        print(f"　前半 {r1:.1f}% → 後半 {r2:.1f}%　★**縮み {r2-r1:+.1f}pt**")
        return roi, lo

    roi4, lo4 = block(np.array([0, 1, 2, 3]), "4点セット（複勝＋単勝＋馬連2＋三連単A4）")
    roi3, lo3 = block(np.array([0, 1, 2]), "★3点セット（三連単を抜く）")

    print(f"\n■ ★★★**合計の分散は下がるのか**（**(226)と同じ手続き**）")
    ret = V / cost                                   # 券種ごとの倍率
    C = np.corrcoef(ret.T)
    print(f"{'':<10}" + "".join(f"{k:>10}" for k, _, _ in SET))
    for a, (k, _, _) in enumerate(SET):
        print(f"{k:<10}" + "".join(f"{C[a, b]:>10.3f}" for b in range(len(SET))))
    w = cost / cost.sum()
    se = np.array([ret[:, j].std(ddof=1) / math.sqrt(len(ret)) for j in range(len(SET))])
    tot = (V.sum(axis=1) / cost.sum())
    se_tot = tot.std(ddof=1) / math.sqrt(len(tot))
    indep = math.sqrt(float(((w * se) ** 2).sum()))
    full = float((w * se).sum())
    print(f"\n　★**相関の平均（対角を除く） {(C.sum()-len(SET))/(len(SET)**2-len(SET)):.3f}**")
    print(f"　★**合計の se {100*se_tot:.2f}pt**"
          f"　／ **独立なら {100*indep:.2f}pt**　／ **完全相関なら {100*full:.2f}pt**")
    print(f"　★**独立比 {se_tot/indep:.2f}倍**"
          f"（**(226)は G=0.15 の別セットで1.19倍**）")

    print(f"\n■ ★★★★**読み方**")
    print(f"　★**4点 {roi4:.1f}% / 3点 {roi3:.1f}%**"
          f"　→ ★**差の {roi4-roi3:+.1f}pt は三連単が作っている**")
    print("　⚠★**(240sns)で三連単A4点G=0.10は「28マス中の1つ・当たり73本」として"
          "★採用を保留にした**。")
    print("　★★**合計が100%を超えても、それが三連単だけで作られているなら採用できない**"
          "（**事前登録したゲート2**）。")


if __name__ == "__main__":
    main()
