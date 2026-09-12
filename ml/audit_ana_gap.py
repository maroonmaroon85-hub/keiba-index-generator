"""(240sns) ★★★★**ズレの下限を下げると何が増えて何が失われるか** —— 利用者の指定（2026-09-12）

■ ★★★**なぜこれを測るのか**
　★**利用者の指定**: **「レースはこっちの方が多く出ると思うんだよね」→ ★選択肢Bを採用**。
　⚠★**該当が枯れている**——**年360本(2016) → 37本(2026)＝★10分の1**。
　★**規則側(235)の分解**: **穴馬は減っていない**（**単勝≥10倍が 10.4→9.7頭・0.93倍**）。
　　⚠**減っているのは★ズレだけ**（**1レースあたり 0.441→0.144頭・0.33倍**）。
　→ ★**SNSとして成立する本数が出ない**。**そこでズレの下限だけを下げる**。

■ ⚠★★**先に、これが何を壊すかを書く**
| | ★**どうなるか** |
|---|---|
| **禁止事項4（軸を人気馬に変えない）** | ★**壊れない**——**単勝≥10.0倍の下限は残すので◎は穴馬のまま** |
| ★**判定基準25（別の母集団を持ち込まない）** | ⚠⚠**壊れる**——**ズレ0.15で測った11年の数字は、★別の下限には持ち込めない** |
| **`ANA_RULE.md` の主判定** | ★**触らない**——**あちらはズレ≥0.15のまま。★これはSNS側だけの線** |
| ⚠**実運用の◎とSNSの◎** | ⚠★**別の馬になりうる**。**投稿と購入が食い違う**（★下記の注意） |
　★★**だから測る**。**11年はあるので、下限ごとに測り直せば持ち込む必要が無い**。

■ ★測るマス（⚠**探索を広げる。★7水準 × 4券種 = 28マス**）
　★**ズレの下限 G ∈ {0.15, 0.12, 0.10, 0.08, 0.06, 0.04, 0.02}**
　⚠**他の条件は据え置き**: **pn ≥ 0.15 / 単勝 ≥ 10.0倍 / その中で pn最大の1頭**。
　★**券種はSNSの4点セット**: **複勝1 / 単勝1 / P馬連2 / P三連単A4**。
　★**G=0.15 は既知の線**＝★**内部対照**（**1,398本 / 複勝23.5% / 軸の中央15.8倍**）。

■ ★★★★**ゲート2（判定基準42）—— ★何を返せば「下げてはいけない」か**
　★★**ズレが優位の源なら、下限を下げるほど ROI と裾外し比は★単調に下がる**。
　★**逆に、ズレが効いていないなら、下げても水準は変わらず本数だけ増える**。
　⚠★**「本数が増える」は情報を持たない**——**条件を緩めるのだから★機械的に増える**。
　★**読むのは次の3つだけ**:
　　1. ★**ROI と裾外し比の落ち方**（**払戻率を割ったら、そこから先は買う意味が無い**）
　　2. ★**前半→後半の縮み**（**下げるほど下駄が乗りやすい**）
　　3. ★★**「年あたりの的中本数」**＝**本数 × 的中率**（**★SNSはここで決まる**）
　★★**折り返しがあるはず**——**本数は増えるが優位は減るので、★どこかで積が最大になる**。

■ ★★内部対照（⚠**落ちたら読まない**・判定基準32）
　★**G=0.15 が 1,398本 / 複勝の的中率 23.5% / 軸の中央オッズ 15.8倍**。
　⚠★**1,383本と比べない**——**あれは乱が引けたレースだけの条件つきの数**
　　（**このプロジェクトで★9回踏まれている取り違え**）。

■ 予想（⚠**当てにしない**・★このセッションは4勝2敗）
　★**ROIは単調に下がる**と見る。★**複勝は最後まで粘る**（**的中率が高く裾が軽いため**）。
　★**三連単A4点は早い段階で100%を割る**と見る。
　★★**年あたりの的中本数の折り返しは G=0.06〜0.10 のあたり**と見る。
　⚠**もし G=0.02 まで下げても複勝のROIが落ちないなら、それは
　　「ズレは複勝の優位の源ではなかった」ことを意味する**——★**そのときは線の理解を書き換える**。


■ ★★★★**結果（実測済み・2026-09-12）**　★**内部対照3つとも通過**（1,398本 / 23.5% / 15.8倍）

**★★①頻度 —— ★狙いどおり増える**
| ズレ下限 | 該当 | 年あたり | ★**1日** | 中央オッズ | 複勝的中率 | ★**年の的中** |
|---|---|---|---|---|---|---|
| **0.15**(現行) | 1,398 | 127 | **1.22** | 15.8倍 | **23.5%** | **29.8** |
| **0.12** | 3,142 | 286 | 2.75 | 16.1倍 | 22.2% | 63.3 |
| ★**0.10** | 5,297 | 482 | ★**4.63** | 16.4倍 | **21.7%** | ★**104.7** |
| **0.08** | 8,512 | 774 | 7.44 | 16.5倍 | 21.6% | 167.0 |
| 0.06 | 12,781 | 1,162 | 11.17 | 16.3倍 | 21.8% | 253.5 |
| 0.04 | 17,508 | 1,592 | 15.30 | 16.0倍 | 22.3% | 354.7 |
| 0.02 | 21,761 | 1,978 | 19.02 | 15.5倍 | 22.7% | 449.8 |

■ ★★★★**核心: ◎の「見た目」はほとんど変わらないが、ROIは即座に壊れる**
| | G=0.15 | G=0.10 | ★**変化** |
|---|---|---|---|
| **1日の本数** | 1.22 | ★**4.63** | ★**3.8倍** |
| ★**複勝的中率** | 23.5% | **21.7%** | ★**−1.8pt だけ** |
| ★**◎の中央オッズ** | 15.8倍 | 16.4倍 | ★**ほぼ同じ** |
| ★**年の的中本数** | 29.8 | ★**104.7** | ★**3.5倍** |
| ⚠⚠**複勝ROI** | ★**101.5%** | ⚠**94.2%** | ⚠⚠**−7.3pt・100%割れ** |
　★★★**「◎が来るか」という★見せ物としての性質は、下限を下げてもほとんど落ちない**。
　⚠⚠**壊れるのは★儲けの側だけ**。→ ★**SNS（ROIを主張しない）と購入（ROIが全て）で、
　　★最適な下限が★別々になる**。

**⚠②儲けの側は 0.15 以外ぜんぶ 100% を割る**
| 券種 | 0.15 | 0.12 | 0.10 | 0.08 | 0.06 | 0.04 | 0.02 |
|---|---|---|---|---|---|---|---|
| **複勝** | ★**101.5%** | 94.6% | 94.2% | 92.3% | 91.0% | 90.9% | 89.1% |
| **単勝** | ★**102.1%** | 93.9% | 91.1% | 93.2% | 90.7% | 89.5% | 89.0% |
| **馬連2** | ★**108.3%** | 100.0% | 96.3% | 93.7% | 96.5% | 95.3% | 92.2% |
| ★**三連単A4** | 124.6% | **142.9%** | ★★**161.3%** | 122.2% | 103.9% | 104.6% | ⚠90.2% |
　★**複勝の裾外し比は全下限で1.00超**（1.02〜1.08）＝**払戻率は超えている**。
　⚠**それでもROIが100%を割る**——**「控除率に勝つ」と「儲かる」は別**。

**⚠⚠★★③三連単A4点・G=0.10 が異常に良い（★だから疑う）**
　★**ROI 161.3% / 必要年数16年 / ★99%CI下端 +5.3 / 縮み +7.0pt / 裾外し比1.483**。
　★★**このプロジェクトで下端が0を超えたのは2例目**。
　⚠★**1例目は(216)の Q三連単M6点(+3.0)で、(217)が後半−3.4円に転じて★下駄と判明した**。
　⚠★**今回は28マス増やした中の1つ**——**ブリーフ:「良い数字が出るほど下駄である確率が上がる」**。
　★**違う点**: **縮みが +7.0pt（Q三連単M6点は −84.4pt）**＝**後半で壊れていない**。
　⚠**だが的中1.4% × 5,216本 ＝ 約73本の当たりしかない**。★**本数で判断できない**。
　★★**結論: ★採用しない**。**(217)と同じ検算（前半/後半・Bonferroni）を通すまでは材料に留める**。

■ ★予想の答え合わせ（判定基準24・★このセッション5回目）
| 予想 | 実際 | |
|---|---|---|
| **ROIは単調に下がる** | ★**複勝・単勝はほぼ単調** | ★**当たり** |
| **複勝が最後まで粘る** | ★**裾外し比が全下限で1.00超** | ★**当たり** |
| **三連単A4点は早く100%を割る** | ⚠**逆。0.10で161.3%に上がった** | ⚠**外れ** |
| **年の的中の折り返しは0.06〜0.10** | ⚠**折り返さなかった**（**的中率がほぼ落ちないため単調増加**） | ⚠**外れ** |
　★**外した2つは同じ理由**——**私は「緩めると的中率も落ちる」と思っていたが、
　　★複勝的中率は 23.5%→21.7%→22.7% でほぼ動かない**。⚠**ズレは的中率の源ではなかった**。

実行: python3 ml/audit_ana_gap.py
"""
import math
import os
import sys
from binascii import crc32

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import features as F
from audit_crosspool import LINE, load_races, payoff, zq
from audit_ana_odds import MIN_HORSES, roi_of
from audit_ana_marg import wf_predict
from audit_ana_board import NPLACE, load_fuku_boards, qpool
from audit_ana_band import PN_FLOOR
from audit_ana_hole import NRAND, SEED, SPLIT
from audit_ana_fix import LFIX
from audit_ana_ladder import FINE
from audit_ana_bet import need_years, tickets
from train_prod import add_odds_features

GAPS = [0.15, 0.12, 0.10, 0.08, 0.06, 0.04, 0.02]
BETS = [("複勝", 0), ("単勝", 0), ("馬連", 2), ("三連単A", 4)]
ALPHA = 0.01
KNOWN_N, KNOWN_HIT, KNOWN_OD = 1398, 23.5, 15.8


def main():
    z = zq(ALPHA)
    print("(240sns) ★★★★**ズレの下限を下げると何が増えて何が失われるか**（利用者の指定）")
    print(f"⚠**探索を {len(GAPS)*len(BETS)} マス広げる**"
          "　★**禁止事項4は壊さない（単勝≥10.0倍は据え置き＝◎は穴馬のまま）**\n")

    races = {r["rid"]: r for r in load_races()}
    boards = load_fuku_boards()
    print(f"★複勝の板 **{len(boards):,}レース分**")
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

    K = {(g, b): {"a": [], "r": [], "hit": [], "yr": []} for g in GAPS for b in BETS}
    NAX = {g: {"n": 0, "od": [], "hit": 0} for g in GAPS}
    yrs = set()
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
        gap = pn - qp
        yr = int(gg["date"].iloc[0].year)
        yrs.add(yr)
        order_p = [int(u) for u in ub[np.argsort(-pv, kind="mergesort")]]
        pos = {int(u): q for q, u in enumerate(ub)}
        drawn = None
        for G in GAPS:
            cand = np.where((pn >= PN_FLOOR) & (gap >= G) & (od >= LFIX))[0]
            if not len(cand):
                continue
            i = int(cand[int(np.argmax(pn[cand]))])
            ax = int(ub[i])
            v0 = payoff(r, "複勝", [ax])
            if v0 is None:
                continue
            NAX[G]["n"] += 1
            NAX[G]["od"].append(float(od[i]))
            NAX[G]["hit"] += 1 if v0 > 0 else 0
            HP = [u for u in order_p if u != ax]
            if drawn is None:          # ★乱は1レース1回だけ引く（★軸は下限で変わりうるので毎回作る）
                drawn = True
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
            for kind, k in BETS:
                tk = tickets(kind, k, ax, HP)
                if tk is None:
                    continue
                va = [payoff(r, nm, sel) for nm, sel in tk]
                if any(x is None for x in va):
                    continue
                cost = 100.0 * len(tk)
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
                    acc.append(sum(vr) / cost)
                if not okr:
                    continue
                c = K[(G, (kind, k))]
                c["a"].append(sum(va) / cost)
                c["r"].append(float(np.mean(acc)))
                c["hit"].append(sum(va))
                c["yr"].append(yr)

    ny = max(len(yrs), 1)
    # ── ★★内部対照 ──────────────────────────────────────
    n15 = NAX[0.15]
    hr = 100.0 * n15["hit"] / max(n15["n"], 1)
    omed = float(np.median(n15["od"])) if n15["od"] else 0.0
    print("\n■ ★★**内部対照**（**G=0.15 が既知の線と一致するか**）")
    ok = [("該当レース", n15["n"], KNOWN_N, 0, n15["n"] == KNOWN_N),
          ("複勝の的中率(%)", hr, KNOWN_HIT, 0.5, abs(hr - KNOWN_HIT) <= 0.5),
          ("軸の中央オッズ", omed, KNOWN_OD, 0.5, abs(omed - KNOWN_OD) <= 0.5)]
    for nm, got, want, tol, good in ok:
        print(f"　{nm:<16}{got:>10.1f} vs {want:>8.1f}　{'★通った' if good else '⚠落ちた'}")
    print("　⚠**1,383本と比べていない**（**あちらは乱が引けたレースだけ。★9回踏まれた取り違え**）")
    if not all(g for *_, g in ok):
        print("\n⚠⚠**内部対照が落ちた。読まない**（判定基準32）。")
        return

    # ── ★軸の頻度 ────────────────────────────────────────
    print(f"\n■ ★★★**①ズレの下限ごとの★頻度**（**11年 / {ny}年**）")
    print(f"{'ズレ下限':<10}{'該当':>8}{'年あたり':>10}{'1日':>8}"
          f"{'中央オッズ':>12}{'複勝的中率':>12}{'★年の的中':>11}")
    for G in GAPS:
        a = NAX[G]
        if not a["n"]:
            continue
        h2 = 100.0 * a["hit"] / a["n"]
        print(f"{G:<10.2f}{a['n']:>8,}{a['n']/ny:>10.1f}{a['n']/ny/104:>8.2f}"
              f"{np.median(a['od']):>11.1f}倍{h2:>11.1f}%{a['hit']/ny:>11.1f}")
    print("　★**1日 = 年あたり ÷ 104開催日**。★**年の的中 = 該当 × 的中率 ÷ 11年**")

    # ── ★券種ごと ────────────────────────────────────────
    for kind, k in BETS:
        rt = LINE["三連単" if kind.startswith("三連単") else kind]
        print(f"\n■ ★**{kind}{k if k else ''}点**（**払戻率 {rt}**）")
        print(f"{'ズレ下限':<10}{'本数':>8}{'ROI':>9}{'的中率':>9}{'平÷中':>8}"
              f"{'★裾外し比':>11}{'99%下端':>10}{'★縮み':>10}{'必要年数':>11}")
        for G in GAPS:
            c = K[(G, (kind, k))]
            a = np.asarray(c["a"], float) * 100.0
            if len(a) < 100:
                continue
            rv = np.asarray(c["r"], float) * 100.0
            yr = np.asarray(c["yr"], int)
            roi = roi_of(a)
            dd = a - rv
            lo = dd.mean() - z * dd.std(ddof=1) / math.sqrt(len(dd))
            hv = np.array([x for x in c["hit"] if x > 0], float)
            tail = hv.mean() / np.median(hv) if len(hv) else float("nan")
            ha = a[a > 0]
            rmed = (len(ha) * float(np.median(ha)) / len(a)) if len(ha) else 0.0
            yy, _ = need_years(a, len(a) / ny, z)
            r1 = roi_of(a[yr < SPLIT]) if (yr < SPLIT).sum() else float("nan")
            r2 = roi_of(a[yr >= SPLIT]) if (yr >= SPLIT).sum() else float("nan")
            mk = "★" if rmed / 100.0 / rt >= 1.0 else "⚠"
            print(f"{G:<10.2f}{len(a):>8,}{roi:>8.1f}%{100*np.mean(a>0):>8.1f}%"
                  f"{tail:>8.2f}{mk}{rmed/100.0/rt:>10.3f}{lo:>10.1f}"
                  f"{r2-r1:>+9.1f}pt{('%.0f年' % yy) if yy else '到達せず':>11}")
    print("\n★**裾外し比** = 各的中を中央配当に置き換えたROI ÷ 払戻率。**1.00超なら裾に頼っていない**")
    print("⚠★★**「本数が増える」は情報を持たない**（**条件を緩めるのだから機械的に増える**）。"
          "★**読むのは ROI・裾外し比・縮み・年の的中の4つ**。")


if __name__ == "__main__":
    main()
