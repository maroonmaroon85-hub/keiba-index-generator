"""(238) ★★★★**4券種の資金配分を最適化する** —— 前半で決めて、後半で1回だけ試す

★★**動機（2026-09-09・利用者の指定）**:
　★**「馬単・三連単・単勝・複勝の複合で最適な資金配分を研究して」**

■ ★**対象の4本**（**すべて同じ軸＝pn≥0.15 かつ ズレ≥0.15 かつ 単勝≥10.0倍 の中で pn最大**）
| 記号 | 買い目 | 全期間ROI | 的中率 |
|---|---|---|---|
| **T** | **単勝1点** | 102.1% | 6.7% |
| **F** | **複勝1点** | 101.5% | 23.4% |
| **U** | ★**P馬単M4点**（**紐＝モデル上位2頭・マルチ**） | 113.4% | 7.6% |
| **S** | ★**X三連単A4点**（**紐＝モデル上位2頭＋人気1頭・軸1着**） | 128.8% | 1.9% |

■ ⚠★★**これは新しい選択操作である**
　★**配分は連続の最適化なので、11年全部で最適化すれば必ず良く見える**。
　★★**だから (201) と同じ設計にする**——**前半(2016-2020)で配分を決め、後半(2021-2026)で1回だけ試す**。
　⚠**後半を見てから配分を変えない。変えたらこの測定は無意味になる**。

────────────────────────────────────────────────────────────
★★★ 事前登録（2026-09-09・**結果を見る前にコミットする**）
────────────────────────────────────────────────────────────

■ ★**配分の定義**: **各券種に「投資額の割合」w を与える**（**Σw = 1・w ≥ 0**）。
　★**1レースあたりの収益率 = Σ wᵢ × rᵢ**（**rᵢ = その券種の100円あたりの払戻**）。
　★**探索は0.05刻みの格子**（**Σw=1 を満たす組み合わせ全部**）。

■ ★★★★**主判定（★先に決める・これ1つ）**
　★★**前半で「必要年数」が最小になる配分を選び、後半でその配分の必要年数を見る**。
　　**必要年数 = (z·s/(ROI−100))² / 年あたり本数**（**z は 99%・両側**）。
　★**理由**: **(211)以来ずっと使っている基準**。**ROI・分散・本数を同時に最適化する唯一の量**。
　⚠**「ROI最大」では選ばない**——**それは必ず一番ROIの高い券種に全額になり、分散を無視する**。

■ ★★ゲート2（判定基準42）——**この測定は何を返せば「配分に意味がない」か**
　★**配分に意味がないなら、前半の最適配分の後半成績は、★等分配分（1/4ずつ）と変わらない**。
　★**等分配分を必ず併記する**。**さらに「後半で最適化した配分」も出し、前半最適との差＝縮みを見る**。
　★**単独4本の後半成績も併記**（**混ぜる価値があるのかを見る**）。

■ ★★★内部対照（**決定的**）: ★**対象レース数が 1,383**
　**かつ 4本の全期間ROIが T 102.1% / F 101.5% / U 113.4% / S 128.8%（±0.2pt）**。
　⚠**ずれたら読まない**。

■ ⚠★★★**対照を1度書き直した（判定基準37・9回目）——先に記録する**
　★**最初の事前登録では「1,398本 かつ T 101.8 / F 101.6 / U 113.4 / S 128.8」と書いた**。
　★**走らせたら U 112.9% / S 141.9% でずれ、対照が落ちた**（**結果は読んでいない・判定基準32**）。
　★★**原因: 2つの母集団の値を1つの対照に混ぜていた**（★**判定基準25**）。
　　**1,398 = 軸が立った全数**。**101.8 / 101.6 はこちらの値**。
　　**1,383 = ±20%の乱が「軸+紐3」で引けたレース**（`audit_ana_bet.py` の `if not okd: continue`）。
　　★**113.4 / 128.8 はこちらの値**。**同じ表の単勝・複勝は 102.1 / 101.5**（**1,383**）。
　★★**直し方: 母集団を1,383に揃える**（**乱の引けるレースだけに絞る**）。
　　★**そうすると4本とも `audit_ana_bet.py` の同じ表から取れる**（**485-486行・175-176行**）。
　⚠**乱そのものは配分の測定に使わない。レースの絞り込みにだけ使う**。

■ ⚠★**先に書いておく限界**
　★**必要年数の最小化は「決着の速さ」であって「儲け」ではない**。
　　**年間の期待利益も併記するが、主判定には使わない**。
　★**配分は連続なので、格子の刻み（0.05）を細かくすれば前半の値はいくらでも良くなる**。
　　⚠**だから前半の値は読まない。後半だけを読む**。

■ 予想（⚠**当てにしない**・★**私は11回外した**）
　★**前半の最適は複勝に大きく寄ると見る**（**分散が最小なので**）。
　⚠**そして後半で縮むと見る**——**(201)25.8pt・(209)97.4pt・(217)19〜86ptの前例**。

────────────────────────────────────────────────────────────
★★★★ 結果（2026-09-09・**事前登録を2本コミットしたあとに走らせた**）
────────────────────────────────────────────────────────────
★★★内部対照: **対象 1,383（1,383）／ T 102.1% / F 101.5% / U 113.4% / S 128.8%** → ★★**立った**
　★**前半 1,058本（5年・年212本） / 後半 325本（6年・年54本）**

| | T | F | U | S | ROI | se | 的中率 | ★必要年数 | 年間利益 |
|---|---|---|---|---|---|---|---|---|---|
| **前半の最適配分**（前半） | 0.00 | 0.00 | **0.85** | 0.15 | 116.8% | 16.28 | 8.6% | **31年** | +3,551円 |
| 等分（前半） | .25 | .25 | .25 | .25 | 110.2% | 13.63 | 23.8% | 59年 | +2,158円 |
| ★★**前半の最適配分（後半）** | 0.00 | 0.00 | **0.85** | 0.15 | 112.2% | 50.31 | 5.8% | ⚠**679年** | +660円 |
| ★**等分（後半）** | .25 | .25 | .25 | .25 | 115.6% | 27.47 | 21.8% | ★**123年** | +846円 |
| ⚠後半で最適化（下駄） | 0.45 | 0.00 | 0.00 | 0.55 | 128.3% | 38.30 | 7.1% | 73年 | +1,532円 |
| 単独 T 単勝1点（後半） | 1 | | | | 113.2% | 26.50 | 7.1% | 160年 | +717円 |
| 単独 F 複勝1点（後半） | | 1 | | | 101.4% | 12.02 | 21.8% | 2,749年 | +78円 |
| 単独 U P馬単M4点（後半） | | | 1 | | 107.2% | 57.35 | 4.9% | 2,547年 | +388円 |
| 単独 S X三連単A4点（後半） | | | | 1 | 140.6% | 57.21 | 2.2% | ★**79年** | +2,200円 |

■ ★★★★**主判定の返り値: ⚠「配分の最適化には意味がない」**
　★**前半で選んだ配分 [0, 0, 0.85, 0.15] を後半で試すと 679年**。
　★**等分（1/4ずつ）なら 123年**。★★**等分に5.5倍負けた**。
　★★**つまり「前半のデータで配分を決める」という操作は、何もしないより悪い**。

■ ★★下駄の大きさ（ゲート2）
　★**前半の最適は前半で31年。同じ配分が後半では679年**——★**22倍に伸びた**。
　★**後半で最適化し直せば73年**（**[0.45,0,0,0.55]**）。⚠**これは後半を見て選んだ値なので読めない**。
　★★**前半の最適(U0.85)と後半の最適(T0.45/S0.55)は重なる券種が一つもない**。
　　★**配分は年で入れ替わる＝安定した最適配分は存在しない**。

■ ⚠★**予想は外れた（★12回目）**
　★**「前半の最適は複勝に大きく寄る」と書いた。実際は複勝 0.00・馬単 0.85**。
　★**複勝は分散が最小だが ROI が 101.5% しかなく、必要年数の分子より分母が効かなかった**。

■ ⚠★★**先に書いた限界のとおり読むこと**
　★**後半は 325本・年54本しかない**。**se は 27〜57pt**。★**1標本のばらつき**（判定基準43）。
　★**「単独Sが後半79年で最良」は後半を見てから言っている＝選択操作**。★**主判定ではない**。
　★**必要年数は「決着の速さ」であって儲けではない**。**年間利益では等分846円 < 下駄1,532円**。

実行: python3 ml/audit_ana_alloc.py    自己テスト: python3 ml/audit_ana_alloc.py --selftest
"""
import math
import sys
from itertools import product

from zlib import crc32

import numpy as np

sys.path.insert(0, "ml")
import features as F
from audit_crosspool import load_races, payoff, zq
from audit_ana_odds import MIN_HORSES, roi_of
from audit_ana_board import NPLACE, load_fuku_boards, qpool
from audit_ana_band import PN_FLOOR
from audit_ana_fix import LFIX
from audit_ana_hole import GAP, NRAND, SEED, SPLIT
from audit_ana_ladder import FINE
from audit_ana_marg import wf_predict
from audit_ana_bet import tickets
from train_prod import add_odds_features

NAMES = ["T 単勝1点", "F 複勝1点", "U P馬単M4点", "S X三連単A4点"]
KNOWN_R, KNOWN_ROI, TOL = 1383, [102.1, 101.5, 113.4, 128.8], 0.2
STEP = 0.05
ALPHA = 0.01


def grid(step=STEP):
    """★Σw=1・w≥0 の格子（★0.05刻み）"""
    k = int(round(1.0 / step))
    out = []
    for a in range(k + 1):
        for b in range(k + 1 - a):
            for c in range(k + 1 - a - b):
                out.append(np.array([a, b, c, k - a - b - c], float) / k)
    return out


def need_years(v, per_year, z):
    roi = roi_of(v)
    if roi <= 100.0 or per_year <= 0:
        return None, roi
    return ((z * float(np.std(v, ddof=1)) / (roi - 100.0)) ** 2) / per_year, roi


def selftest():
    ok = True
    g = grid()
    print(f"★配分の格子（{STEP}刻み・Σw=1）: **{len(g):,}通り**")
    ok &= all(abs(w.sum() - 1.0) < 1e-9 and (w >= 0).all() for w in g)
    print(f"　Σw=1 かつ w≥0 が全部成立 {'★OK' if ok else '⚠NG'}")
    print(f"★★主判定: **前半で必要年数が最小の配分を選び、後半で1回だけ試す**")
    print(f"★★ゲート2: **等分配分(1/4ずつ)と、後半で最適化した配分を必ず併記**")
    print(f"★★★内部対照: **対象 {KNOWN_R:,}本** かつ **全期間ROI "
          + " / ".join(f"{n.split()[0]} {r}%" for n, r in zip(NAMES, KNOWN_ROI))
          + f"（±{TOL}pt）**")
    v = np.array([0.0] * 90 + [2000.0] * 10)   # ★ROI 200%（100超でないと必要年数が定義されない）
    ny, roi = need_years(v, 100.0, zq(ALPHA))
    print(f"★必要年数の検算: ROI {roi:.1f}% / 必要年数 {ny:,.1f}"
          f"　{'★OK' if abs(ny - ((zq(ALPHA)*v.std(ddof=1)/(roi-100))**2)/100) < 1e-9 else '⚠NG'}")
    print("⚠**後半を見てから配分を変えない**")
    print("★自己テスト: " + ("全部OK" if ok else "⚠NG"))
    return 0 if ok else 1


def main():
    z = zq(ALPHA)
    print("(238) ★★★★**4券種の資金配分を最適化する** —— 前半で決めて、後半で1回だけ\n")
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

    R, YR = [], []
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
        c = np.where((pn >= PN_FLOOR) & (gap >= GAP) & (od >= LFIX))[0]
        if not len(c):
            continue
        i = int(c[int(np.argmax(pn[c]))])
        ax = int(ub[i])
        # ★★★母集団を1,383に揃える: ±20%の乱が「軸+紐3」で引けるレースだけ
        #   （audit_ana_bet.py の `if not okd: continue` と厳密に同じ）
        op0 = [int(u) for u in ub[np.argsort(-pv, kind="mergesort")]]
        HMP = [u for u in op0 if u != ax]
        pos = {int(u): q for q, u in enumerate(ub)}
        okd = True
        for sd in range(NRAND):
            g2 = np.random.default_rng([SEED + sd, crc32(rid.encode())])
            out = []
            for u in [ax] + HMP[:3]:
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
        if not okd:
            continue
        op = HMP
        oq = [int(u) for u in ub[np.argsort(od, kind="mergesort")] if int(u) != ax]
        X = op[:2] + [u for u in oq if u not in op[:2]]
        vt, vf = payoff(r, "単勝", [ax]), payoff(r, "複勝", [ax])
        tu, ts = tickets("馬単M", 4, ax, op), tickets("三連単A", 4, ax, X)
        if vt is None or vf is None or tu is None or ts is None:
            continue
        pu = [payoff(r, k2, s2) for k2, s2 in tu]
        ps = [payoff(r, k2, s2) for k2, s2 in ts]
        if any(x is None for x in pu) or any(x is None for x in ps):
            continue
        R.append([vt, vf, sum(pu) / 4.0, sum(ps) / 4.0])   # ★100円あたりの払戻に揃える
        YR.append(int(gg["date"].iloc[0].year))
    A = np.array(R, float)
    YR = np.array(YR, int)
    rois = [roi_of(A[:, j]) for j in range(4)]
    okc = (len(A) == KNOWN_R
           and all(abs(a - b) <= TOL for a, b in zip(rois, KNOWN_ROI)))
    print(f"★★★内部対照: 対象 **{len(A):,}**（{KNOWN_R:,}）／ 全期間ROI "
          + " / ".join(f"{n.split()[0]} {r:.1f}%" for n, r in zip(NAMES, rois))
          + f"　{'★★立った' if okc else '⚠⚠落ちた'}")
    if not okc:
        print("⚠⚠**対照が落ちた。読まない**（判定基準32）。")
        return

    h1 = YR < SPLIT
    n1, n2 = int(h1.sum()), int((~h1).sum())
    per1, per2 = n1 / 5.0, n2 / 6.0          # ★前半5年・後半6年（2021〜2026）
    G = grid()

    def best(mask, per):
        bw, bn = None, None
        for w in G:
            v = A[mask] @ w
            ny, roi = need_years(v, per, z)
            if ny is not None and (bn is None or ny < bn):
                bw, bn = w, ny
        return bw, bn

    w1, ny1 = best(h1, per1)
    w2, ny2 = best(~h1, per2)
    eq = np.array([0.25] * 4)

    def rep(lab, w, mask, per):
        v = A[mask] @ w
        ny, roi = need_years(v, per, z)
        se = v.std(ddof=1) / math.sqrt(len(v))
        prof = per * (roi - 100.0) / 100.0 * 100.0
        print(f"{lab:<22}" + " ".join(f"{x:>5.2f}" for x in w)
              + f"{roi:>9.1f}%{se:>7.2f}{100*np.mean(v > 0):>7.1f}%"
              + f"{(f'{ny:,.0f}年' if ny else '到達せず'):>11}"
              + f"{prof:>+9,.0f}円")

    print(f"\n★前半 {n1:,}本（5年・年{per1:.0f}本） / 後半 {n2:,}本（6年・年{per2:.0f}本）")
    print(f"\n{'':<22}{'  T':>5}{'  F':>5}{'  U':>5}{'  S':>5}"
          f"{'ROI':>9}{'se':>7}{'的中率':>7}{'★必要年数':>11}{'年間利益':>9}")
    print("■ ★★前半（**ここは読まない。選ぶだけ**）")
    rep("★前半の最適配分", w1, h1, per1)
    rep("等分（1/4ずつ）", eq, h1, per1)
    print("\n■ ★★★★後半（**主判定はここ**）")
    rep("★★前半の最適配分", w1, ~h1, per2)
    rep("★等分（1/4ずつ）", eq, ~h1, per2)
    rep("⚠後半で最適化（下駄）", w2, ~h1, per2)
    print("\n■ ★単独（後半）")
    for j, nm in enumerate(NAMES):
        w = np.zeros(4); w[j] = 1.0
        rep(nm, w, ~h1, per2)

    v1 = A[~h1] @ w1
    ve = A[~h1] @ eq
    n1y, _ = need_years(v1, per2, z)
    ney, _ = need_years(ve, per2, z)
    print(f"\n■ ★★★**主判定**")
    print(f"　★**前半の最適配分 {np.round(w1, 2)} を後半で試すと "
          f"{(f'{n1y:,.0f}年' if n1y else '到達せず')}**")
    print(f"　★**等分配分なら {(f'{ney:,.0f}年' if ney else '到達せず')}**"
          f"　→ ★**{'配分に意味がある' if (n1y and ney and n1y < ney) else '⚠等分に負けた＝配分に意味がない'}**")
    print(f"　⚠**後半で最適化すれば {(f'{ny2:,.0f}年' if ny2 else '到達せず')}**"
          f"（**{np.round(w2, 2)}**）——★**これが下駄の大きさ**")
    print("\n⚠**枠連の運用には触れない**。**設定変更は提案しない**。")


if __name__ == "__main__":
    sys.exit(selftest() if "--selftest" in sys.argv else (main() or 0))
