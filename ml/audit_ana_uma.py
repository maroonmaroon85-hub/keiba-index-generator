"""(232) ★★★**馬連を3・4・5点まで伸ばす** —— 利用者の指定「馬連4-5点測って」（2026-09-09）

■ ★★★★**先に警告を書く（★これは探索を広げる測定である）**
　⚠**ブリーフ**: **「探索を広げるほど良い数字は出るが、良い数字が出るほど下駄である確率が上がる」**。
　★**(211)は 15→19→27→90→98→160→170→189 とマスを増やしてきた**。
　★**これは★9マス増やす**（**189 → 198**）。⚠**増やす理由は利用者の指定であって、
　　私が「ここに何かある」と踏んだからではない**。★**その区別を記録に残す**。

■ ★**なぜ馬連にこの穴が空いていたか**
　★**`audit_ana_hole.BETS`（(209)）が馬連を k=1,2 でしか刻んでいなかった**。
　⚠**根拠があって止めたのではなく、★(209)が「紐1〜2頭」の設計だったのを(211)が引き継いだだけ**
　　——**(212)の馬単・(224)の単勝と同じ「取りこぼし」の形**。
　★**`tickets()` は既に任意の点数の馬連を組める**ので、⚠**`audit_ana_bet.py` は1行も変更しない**。
　　★**既存189マスの記録を汚さないため、別スクリプトとして足す**。

■ ★**測るマス（★9マス。これ以上増やさない）**
　★**馬連 k ∈ {3, 4, 5} × 紐 ∈ {P, G, Q}**。**買い方は 軸-紐1 / 軸-紐2 / … の k点**。
　⚠**紐Xは足さない**（**(227)で券種を三連単A/M・馬単A/Mに絞ると決めた。★そこを動かさない**）。
　★**k=1,2 も同じスクリプトで測るが、★これは新マスではなく内部対照である**（下記）。

■ ★★★**内部対照（⚠これが落ちたら結果を読まない・判定基準32）**
　★**同じ経路で k=1,2 を測り、(211)の凍結値と一致すること**:
| マス | 凍結値 ROI | 本数 |
|---|---|---|
| **P馬連1点** | **102.1%** | **1,383** |
| **P馬連2点** | **108.3%** | **1,383** |
| **G馬連2点** | **140.0%** | **1,383** |
| **Q馬連2点** | **107.2%** | **1,383** |
　★**許容 ±0.2pt / ±5本**。★**加えて条件なしの全数が 1,398レース**であること。
　⚠★★**1,383と1,398は別の量である**（**1,383＝乱が引けたレースだけ / 1,398＝全数**）。
　　★**(231)で私はここを取り違えて「環境差だ」と誤診した。★同じ穴に二度落ちない**。

■ ★★★★**ゲート2（判定基準42）—— ★この測定は何を返せば「意味なし」か**
　⚠★★**「点数を増やすと的中率が上がる」は★情報を持たない**——**買い目を足すのだから
　　的中率は★機械的に単調増加する**。**偽でも0を返さないので、これで判断してはいけない**
　　（**(213)で「上位k本を抜くと必ずROIが下がる」と同じ形**）。
　★**情報を持つのは次の3つだけ**:
　　1. ★**必要年数**（**ROI・分散・本数を同時に見る唯一の量**）
　　2. ★**対応差の99%CI下端**（**乱と区別がつくか**）
　　3. ★**前半→後半の縮み**（**下駄が乗っていれば必ず縮む**）
　★★**「馬連を広げる意味が無い」なら、ROIは点数とともに払戻率(0.775)へ単調に寄り、
　　必要年数は単調に増え、下端は0を割ったままになる**。★**それがこの測定の偽の返り値**。

■ ★**ブリーフが要求する3つの物差しを全部当てる**
　1. ★**前半(〜2020) → 後半(2021〜) の縮み**
　2. ★**裾を外したROI÷払戻率**（**各的中を中央配当に置き換える。1.00超なら裾に頼っていない**）
　3. ⚠**朝9時の一致率は `audit_ana_morn.py` の担当なので、★ここでは測らない**
　　（**必要なら、選ばれたマスにだけ後から当てる**）

■ 予想（⚠**当てにしない**・判定基準24。★**このセッションの私は1勝1敗**）
　★**k=3,4,5 は全部 k=2 より悪くなる**と見る——**馬単Aの梯子が 2点136.5% → 3点116.3%
　　→ 4点106.0% → 5点103.6% と落ちた**のと同じ形になるはず。
　★**Pの馬連は k=2 の108.3%が頂点**と見る。★**Gは k=2 の140.0%から急落**すると見る
　　（**Gは平÷中2.30の裾依存なので、点数を増やすと裾が薄まる**）。
　⚠**もし k=4,5 で必要年数が k=2 より縮んだら、それは私の読み筋が間違っている**
　　——★**そのときは「紐を広げるのは高くつく」という(213)以来の結論を書き換える**。

実行: python3 ml/audit_ana_uma.py
"""
import math
import os
import sys
from binascii import crc32

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import features as F
from audit_crosspool import LINE, load_races, payoff, zq
from audit_ana_odds import MIN_HORSES, band_of, BANDS, roi_of
from audit_ana_marg import wf_predict
from audit_ana_board import NPLACE, load_fuku_boards, qpool
from audit_ana_band import PN_FLOOR
from audit_ana_hole import GAP, NRAND, SEED, SPLIT
from audit_ana_fix import LFIX
from audit_ana_ladder import FINE
from audit_ana_bet import need_years, tickets
from train_prod import add_odds_features

KS = [1, 2, 3, 4, 5]          # ★1,2 は内部対照 / 3,4,5 が新マス
NEWK = [3, 4, 5]
HIMO = ["P", "G", "Q"]
ALPHA = 0.01
RATE = LINE["馬連"]
KNOWN = {("P", 1): 102.1, ("P", 2): 108.3, ("G", 2): 140.0, ("Q", 2): 107.2}
KNOWN_N, ROI_TOL, N_TOL = 1383, 0.2, 5
TOTAL_N = 1398                # ★条件なしの全数（⚠1,383とは別の量）


def main():
    z = zq(ALPHA)
    print("(232) ★★★**馬連を3・4・5点まで伸ばす** —— 利用者の指定（2026-09-09）")
    print(f"⚠**これは探索を★{len(NEWK)*len(HIMO)}マス広げる測定である**"
          f"（**189 → {189 + len(NEWK)*len(HIMO)}**）。"
          "★**増やす理由は利用者の指定であって、私が踏んだからではない**\n")

    races = {r["rid"]: r for r in load_races()}
    print("★複勝の板(type=2)を読む…")
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

    cs = [(h, k) for h in HIMO for k in KS]
    K = {c: {"a": [], "r": [], "hit": [], "yr": []} for c in cs}
    nfix, yrs = 0, set()
    for rid, g in sub.groupby("raceid"):
        rid = str(rid)
        r, bd = races.get(rid), boards.get(rid)
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
        qp, _R = qpool([bd[int(u)] for u in ub], "harm")
        gap = pn - qp
        yrs.add(int(gg["date"].iloc[0].year))
        cand = np.where((pn >= PN_FLOOR) & (gap >= GAP) & (od >= LFIX))[0]
        if not len(cand):
            continue
        i = int(cand[int(np.argmax(pn[cand]))])
        ax = int(ub[i])
        if payoff(r, "複勝", [ax]) is None:
            continue
        nfix += 1
        order_p = [int(u) for u in ub[np.argsort(-pv, kind="mergesort")]]
        order_g = [int(u) for u in ub[np.argsort(-gap, kind="mergesort")]]
        order_q = [int(u) for u in ub[np.argsort(od, kind="mergesort")]]
        HM = {"P": [u for u in order_p if u != ax],
              "G": [u for u in order_g if u != ax],
              "Q": [u for u in order_q if u != ax]}
        pos = {int(u): q for q, u in enumerate(ub)}
        # ★乱: audit_ana_bet と★厳密に同じ引き方（同じ SEED・同じ順・同じ FINE）
        draws, okd = [], True
        for sd in range(NRAND):
            g2 = np.random.default_rng([SEED + sd, crc32(rid.encode())])
            out = []
            for u in [ax] + HM["P"][:3]:
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
            for u in HM["P"][3:5]:
                k0 = pos[u]
                pl = [int(ub[q]) for q in range(len(ub))
                      if int(ub[q]) not in out
                      and od[k0] / FINE <= od[q] <= od[k0] * FINE]
                if not pl:
                    break
                out.append(int(g2.choice(pl)))
            draws.append(out)
        if not okd:
            continue
        nmin = min(len(t) - 1 for t in draws)
        for h, k in cs:
            if k > nmin:
                continue
            tk = tickets("馬連", k, ax, HM[h])
            if tk is None:
                continue
            va = [payoff(r, nm, sel) for nm, sel in tk]
            if any(x is None for x in va):
                continue
            cost = 100.0 * len(tk)
            acc, okr = [], True
            for t in draws:
                tr = tickets("馬連", k, t[0], t[1:])
                if tr is None:
                    okr = False
                    break
                vr = [payoff(r, nm, sel) for nm, sel in tr]
                if any(x is None for x in vr):
                    okr = False
                    break
                acc.append(sum(vr) / cost)
            if not okr:
                continue
            c = K[(h, k)]
            c["a"].append(sum(va) / cost)
            c["r"].append(float(np.mean(acc)))
            c["hit"].append(sum(va))
            c["yr"].append(int(gg["date"].iloc[0].year))

    ny = max(len(yrs), 1)
    print(f"\n★対象 **{nfix:,}レース / {ny}年**（⚠**これは条件なしの全数**）")

    # ── ★★内部対照 ────────────────────────────────────────
    print("\n■ ★★**内部対照**（**(211)の凍結値と一致するか**）")
    okall = abs(nfix - TOTAL_N) <= N_TOL
    print(f"　{'全数(条件なし)':<16}{nfix:>8,} vs {TOTAL_N:>8,}"
          f"　{'★通った' if okall else '⚠落ちた'}"
          f"　⚠**1,383本と比べてはいけない（あちらは乱が引けたレースだけ）**")
    for (h, k), want in KNOWN.items():
        a = np.asarray(K[(h, k)]["a"], float) * 100.0
        got, n = roi_of(a), len(a)
        good = abs(got - want) <= ROI_TOL and abs(n - KNOWN_N) <= N_TOL
        okall &= good
        print(f"　{h}馬連{k}点{'':<9}{got:>7.1f}% / {n:>5,}本 vs "
              f"{want:>6.1f}% / {KNOWN_N:,}本　{'★通った' if good else '⚠落ちた'}")
    if not okall:
        print("\n⚠⚠**内部対照が落ちた。★結果を読まない**（判定基準32）。")
        return

    # ── ★全マス ────────────────────────────────────────────
    print(f"\n{'='*118}")
    print("■ ★★**馬連の梯子**（★**k=1,2 は既知の対照 / k=3,4,5 が新マス**）")
    print(f"{'紐':<3}{'点':>3}{'本数':>7}{'ROI':>8}{'優位比':>8}{'★対応差':>10}"
          f"{'99%CI下端':>11}{'的中率':>8}{'平均配当':>10}{'平÷中':>7}"
          f"{'★必要年数':>12}{'★縮み':>9}{'裾外し比':>9}")
    for h in HIMO:
        for k in KS:
            a = np.asarray(K[(h, k)]["a"], float) * 100.0
            rv = np.asarray(K[(h, k)]["r"], float) * 100.0
            if len(a) < 100:
                continue
            yr = np.asarray(K[(h, k)]["yr"], int)
            roi = roi_of(a)
            dd = a - rv
            mu = dd.mean()
            lo = mu - z * dd.std(ddof=1) / math.sqrt(len(dd))
            hv = np.array([x for x in K[(h, k)]["hit"] if x > 0], float)
            hm = hv.mean() if len(hv) else 0.0
            hmd = float(np.median(hv)) if len(hv) else 0.0
            tail = hm / hmd if hmd > 0 else float("nan")
            yy, _ = need_years(a, len(a) / ny, z)
            # ★前半→後半の縮み
            r1 = roi_of(a[yr < SPLIT]) if (yr < SPLIT).sum() else float("nan")
            r2 = roi_of(a[yr >= SPLIT]) if (yr >= SPLIT).sum() else float("nan")
            # ★裾を外したROI÷払戻率: 各的中を中央配当に置き換える
            rmed = 100.0 * (len(hv) * hmd) / (100.0 * len(a)) if len(a) else 0.0
            mark = "★" if k in NEWK else " "
            print(f"{mark}{h:<2}{k:>3}{len(a):>7,}{roi:>7.1f}%{roi/100.0/RATE:>8.3f}"
                  f"{mu:>+9.1f}円{lo:>+10.1f}{100*np.mean(a>0):>7.1f}%{hm:>9,.0f}円"
                  f"{tail:>7.2f}{('%.0f年' % yy) if yy else '到達せず':>12}"
                  f"{r2-r1:>+8.1f}pt{rmed/100.0/RATE:>9.3f}")
        print()
    print("★**縮み** = 後半ROI − 前半ROI（**下駄が乗っていれば必ず負に大きい**）")
    print("★**裾外し比** = 各的中を中央配当に置き換えたROI ÷ 払戻率"
          f"（**{RATE}**）。**1.00超なら裾に頼っていない**")
    print("\n⚠★★**的中率が点数とともに上がるのは機械的で、情報を持たない**"
          "（**事前登録のゲート2**）。★**読むのは必要年数・下端・縮みの3つだけ**。")


if __name__ == "__main__":
    main()
