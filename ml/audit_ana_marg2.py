"""(185) ★★**(182)を板ベースの穴でやり直す** — 同じ漏れを持っていたので整合性のために測る

★★**なぜやるか**——**(184)で「複勝を買うのに値段は単勝で測っていた」漏れが実在すると分かった**。
　⚠**(182)の「★穴」軸も同じ漏れを持っている**（**穴 = 5-20倍で `gap = share − mk(単勝)` 最大**）。
　★**(184)では、板に差し替えると 有意な帯が 1/9 → 6/9、最良の水準が 84.9% → 88.4%、
　　穴の26.1%が別の馬になった**。→ ★**(182)の穴軸の数字は、そのままでは信用できない**。
　⚠**(182)の「モ1」軸（モデル確率1位）は q_pool を使わないので影響を受けない**。**そこは再利用する**。

⚠**結論が変わる見込みは低い（先に書く）**——**(182)の限界ROIは 63.5〜89.9%**で、
　**100%まで最低でも10pt**。**(184)で板に替えても水準は +3.5pt しか上がらなかった**。
　★**それでも測るのは「同じ漏れを直さずに閉じた」という記録を残さないため**。

────────────────────────────────────────────────────────────
★★★ 事前登録（2026-09-06・**結果を見る前にコミットする**）
────────────────────────────────────────────────────────────

■ ★経路（判定基準25）
　　**q      = MLモデルの top3 確率 p**（Σ=3 に正規化）
　　★**q_pool = 複勝の板**（type=2・**価格は下限と上限の調和平均**・(184)で R=0.8009 を確認）
　⚠**q 側は依然としてML＝弱い**。**「閉じた」とは書けない**。

■ ★★**(182)から変えるのは1点だけ** —— **穴の選び方**
　| | (182) | ★**(185)** |
　|---|---|---|
　| 穴 | 5-20倍で `share − mk(単勝)` 最大 | ★**5-20倍で `p_norm − q_pool_fuku(板)` 最大** |
　| 紐の順序 | モデル確率の降順 | **同じ**（**q_pool を使わないので影響なし**） |
　| 券種・k・手続き | 4券種・k=1〜6・ウォークフォワード | **同じ** |
　⚠**帯を 5-20倍 のままにする**（**(182)と揃えるため**）。
　　★**(184)で最良だったのは12-20倍だが、そこに寄せると post-hoc になる**ので**変えない**。

■ ★★★主判定 = **限界ROI − 100%**（**穴軸のみ・22比較・Bonferroni α=0.01/22・z=3.582**）
　**k頭目で増える組**: 馬連1点 / 馬単1点 / 三連複 k−1点 / 三連単 2(k−1)点。
　★**ゲート2（判定基準42）**: **仮説が偽（k頭目が損益に中立）なら
　　「1点あたり払戻 − 100円」の期待値は厳密に0を返す**。**限界ROIは点数で割ってあるので
　　幅を変えても自動では動かない**。

■ ⚠ゲート1: (88)③④を別パーサで再現（±3pt）。
■ ★ゲート板: **復元R = 3/Σ(1/調和平均) の中央値が 0.800±0.020**（**(184)で0.8009**）。
■ ★陽性対照: **三連複BOX上位4が 81.9%±1.5pt**。
■ ★内部対照: ★**モ1軸の限界ROIが(182)の値を±2ptで再現**すること
　　（**モ1は q_pool を使わないので、同じ値が出なければ装置が違う**）。

■ ★記述（判定しない）
　1. ★**穴(板) と 穴(単勝) が同じ馬になる割合**（**(184)は73.9%だった**）。
　2. **(182)の穴(単勝)の限界ROIを併記**（**差し替えで何が動いたかを見るため**）。

■ ★採用条件
　1. **限界ROIの99%CI下端が100%を超える k が存在する**
　2. ★**券種をまたいで同じ向き**（**1マスだけ跳ねるのは(172)で崩れた形**）
　⚠**通らなければ「穴の価格を直しても、紐を1頭足す増分は常に損」**と書いて閉じる。

■ 予想
　★**通らないと見る**（**(182)で最良89.9%・(184)で板の上積みは+3.5pt**）。
　⚠**判定基準24: 類推は当たらない**。**どちらでも驚かない**。

実行: python3 ml/audit_ana_marg2.py    自己テスト: python3 ml/audit_ana_marg2.py --selftest
"""
import math
import sys
from itertools import combinations

import numpy as np

sys.path.insert(0, "ml")
import features as F
from audit_crosspool import load_races, payoff, zq
from audit_ana_odds import COST, MIN_HORSES, gate1, roi_of
from audit_ana_marg import (ANA_HI, ANA_LO, KMAX, SPECS, WF_BOX4, WF_TOL,
                            marginal_combos, wf_predict)
from audit_ana_board import (BOARD_R, BOARD_TOL, NPLACE, load_fuku_boards, qpool)
from train_prod import add_odds_features

NCMP = KMAX + KMAX + (KMAX - 1) + (KMAX - 1)      # ★穴軸のみ22比較
ALPHA = 0.01
SEED = 20260906
INNER_TOL = 2.0
# (182)の実測（内部対照＋記述）
M1_182 = {("馬連", 1): 82.3, ("馬連", 2): 82.2, ("馬連", 3): 81.1, ("馬連", 4): 75.0,
          ("馬連", 5): 81.8, ("馬連", 6): 81.9,
          ("馬単", 1): 78.0, ("馬単", 2): 79.9, ("馬単", 3): 77.0, ("馬単", 4): 73.4,
          ("馬単", 5): 76.9, ("馬単", 6): 75.9,
          ("三連複", 2): 81.9, ("三連複", 3): 81.4, ("三連複", 4): 78.7,
          ("三連複", 5): 77.5, ("三連複", 6): 73.0,
          ("三連単", 2): 72.4, ("三連単", 3): 75.8, ("三連単", 4): 70.7,
          ("三連単", 5): 73.8, ("三連単", 6): 72.2}
ANA_182 = {("馬連", 1): 82.9, ("馬連", 2): 89.9, ("馬連", 3): 81.9, ("馬連", 4): 79.8,
           ("馬連", 5): 77.1, ("馬連", 6): 64.5,
           ("馬単", 1): 78.5, ("馬単", 2): 89.2, ("馬単", 3): 81.0, ("馬単", 4): 80.3,
           ("馬単", 5): 83.8, ("馬単", 6): 63.5,
           ("三連複", 2): 87.3, ("三連複", 3): 81.0, ("三連複", 4): 80.1,
           ("三連複", 5): 78.0, ("三連複", 6): 75.0,
           ("三連単", 2): 80.8, ("三連単", 3): 77.0, ("三連単", 4): 83.2,
           ("三連単", 5): 81.9, ("三連単", 6): 68.7}


def selftest():
    ok = True
    assert len(M1_182) == NCMP and len(ANA_182) == NCMP
    print(f"★(182)の実測値の自己テスト: モ1 {len(M1_182)}マス / 穴 {len(ANA_182)}マス"
          f"（要 {NCMP}）　★OK")
    q, R = qpool([(2.0, 2.0)] * 10)
    assert abs(q.sum() - NPLACE) < 1e-9
    print(f"★板→含意の自己テスト: Σq={q.sum():.3f}　★OK")
    rng = np.random.default_rng(0)
    n = 200_000
    pay = np.where(rng.random(n) < 0.10, 1000.0, 0.0)
    m = float(np.mean([(rng.permutation(pay) - 100.0).mean() for _ in range(200)]))
    print(f"★ゲート2の自己テスト: 中立な増分200回の平均 {m:+.3f}円"
          f" → **仮説が偽なら0を返す**: {'★OK' if abs(m) < 2 else '⚠NG'}")
    ok &= abs(m) < 2
    print(f"★比較数 {NCMP}（★穴軸のみ）→ z = {zq(ALPHA/NCMP):.3f}")
    print(f"★穴の帯 = {ANA_LO}-{ANA_HI}倍（**(182)と揃える。12-20倍に寄せると post-hoc**）")
    print("★自己テスト: " + ("全部OK" if ok else "⚠NG"))
    return 0 if ok else 1


def main():
    z = zq(ALPHA / NCMP)
    print("(185) ★★**(182)を板ベースの穴でやり直す**")
    print("★変えるのは1点だけ: **穴 = 5-20倍で `p_norm − q_pool_fuku(板)` 最大**")
    print("　（(182)は `share − mk(単勝)` 最大だった）")
    print("⚠**モ1軸は q_pool を使わないので影響を受けない**＝**内部対照に使う**\n")

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
    m = ~np.isnan(pred)
    sub = d.loc[m, ["raceid", "umaban", "odds", "date"]].copy()
    sub["p"] = pred[m]

    acc = {(a, s[0], k): [] for a in ("モ1", "★穴(板)")
           for s in SPECS for k in range(s[2], s[3] + 1)}
    Rs, box4 = [], []
    same, tot, nrace = 0, 0, 0
    for rid, g in sub.groupby("raceid"):
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
        qp, Rb = qpool([bd[int(u)] for u in ub], "harm")
        Rs.append(Rb)
        gapf = pn - qp
        gapt = pv / pv.sum() - (1.0 / od) / (1.0 / od).sum()
        order = [int(u) for u in ub[np.argsort(-pv, kind="mergesort")]]
        bx = [payoff(r, "三連複", list(c)) for c in combinations(sorted(order[:4]), 3)]
        if not any(x is None for x in bx):
            box4.append(sum(bx) - 400.0)
        idx = np.where((od >= ANA_LO) & (od < ANA_HI))[0]
        axes = {"モ1": order[0]}
        if len(idx):
            ab = int(ub[int(idx[int(np.argmax(gapf[idx]))])])
            at = int(ub[int(idx[int(np.argmax(gapt[idx]))])])
            axes["★穴(板)"] = ab
            same += (ab == at); tot += 1
        used = False
        for a, axu in axes.items():
            himo = [u for u in order if u != axu]
            if len(himo) < KMAX:
                continue
            for kind, kk, lo, hi in SPECS:
                for k in range(lo, hi + 1):
                    cs = marginal_combos(kind, axu, himo, k)
                    vs = [payoff(r, kk, list(c)) for c in cs]
                    if any(v is None for v in vs):
                        continue
                    acc[(a, kind, k)].append(sum(vs) / len(cs))
                    used = True
        nrace += used
    print(f"\n★突き合わせ {nrace:,}レース")

    med = float(np.median(Rs))
    okR = abs(med - BOARD_R) <= BOARD_TOL
    print(f"■ ★ゲート板: 復元R の中央値 = **{med:.4f}**（要 {BOARD_R}±{BOARD_TOL}）"
          f" → **{'★立った' if okR else '⚠⚠落ちた'}**")
    box4 = np.asarray(box4, float)
    g0 = 100.0 * (box4.sum() + 400.0 * len(box4)) / (400.0 * len(box4))
    okb = abs(g0 - WF_BOX4) <= WF_TOL
    print(f"■ ⚠**陽性対照**: 三連複BOX上位4 **{g0:.1f}%** vs {WF_BOX4}%"
          f" → **{'★立った' if okb else '⚠⚠落ちた'}**")
    if not (okR and okb):
        print("\n⚠⚠**ゲートが落ちた。読まない**（判定基準32）。")
        return
    print(f"■ 記述: **穴(板) と 穴(単勝) が同じ馬になる割合 = {100*same/max(tot,1):.1f}%**"
          f"（**(184)は73.9%**）")

    print(f"\n{'='*100}")
    print("■ ★内部対照: **モ1軸が(182)を再現するか**（許容±{:.0f}pt）".format(INNER_TOL))
    ng = 0
    for kind, kk, lo, hi in SPECS:
        for k in range(lo, hi + 1):
            v = np.asarray(acc[("モ1", kind, k)], float)
            if len(v) < 300:
                continue
            roi = 100.0 * v.mean() / COST
            dd = roi - M1_182[(kind, k)]
            bad_ = abs(dd) > INNER_TOL
            ng += bad_
            if bad_:
                print(f"　⚠{kind} k={k}: {roi:.1f}% vs (182) {M1_182[(kind,k)]:.1f}%"
                      f"　差 {dd:+.1f}pt")
    print(f"　→ **{'⚠⚠' + str(ng) + 'マスがズレた（板のある部分集合なので注意して読む）' if ng else '★全22マスで再現'}**")

    print(f"\n{'='*100}")
    print(f"■ ★★★主判定: **k頭目の限界ROI**（★**穴(板)軸のみ**・"
          f"**{NCMP}比較・α={ALPHA}/{NCMP}・z={z:.3f}**）")
    print("　★ゲート2: **仮説が偽（k頭目が損益に中立）なら「1点あたり払戻−100円」の期待値は0**")
    print(f"\n{'券種':<8}{'k':>3}{'増分の点数':>10}{'標本R':>9}{'★穴(板)':>10}"
          f"{'(182)穴(単勝)':>14}{'差':>8}{'99%CI(Bonf)':>24}{'判定':>18}")
    hits = []
    for kind, kk, lo, hi in SPECS:
        for k in range(lo, hi + 1):
            v = np.asarray(acc[("★穴(板)", kind, k)], float)
            if len(v) < 300:
                continue
            mu = v.mean() - COST
            se = v.std(ddof=1) / math.sqrt(len(v))
            roi = 100.0 * v.mean() / COST
            sig = mu - z * se > 0
            if sig:
                hits.append((kind, k, roi))
            old = ANA_182[(kind, k)]
            npts = len(marginal_combos(kind, 99, list(range(1, KMAX + 1)), k))
            print(f"{kind:<8}{k:>3}{npts:>10}{len(v):>9,}{roi:>9.1f}%"
                  f"{old:>13.1f}%{roi-old:>+7.1f}"
                  f"{f'[{100+mu-z*se:.1f},{100+mu+z*se:.1f}]':>24}"
                  f"{'★★100%超' if sig else '⚠100%を超えない':>18}")

    print(f"\n■ ★採用条件")
    print(f"　1. 限界ROIの99%CI下端が100%超の k … **{len(hits)}マス**"
          f" → {'★満たす' if hits else '⚠満たさない'}")
    if not hits:
        print("\n★★★**結論: 穴の価格を板に直しても、紐を1頭足す増分は常に損**。")
        print("★**(182)の結論は保った**——**「組を買うこと自体が損」**。")
        print("⚠**経路は弱いので「閉じた」とは書けない**（判定基準25）。")
        return
    print("\n■ ★裾の検算（**通ったマス**・(77)）")
    for kind, k, roi in hits:
        v = np.asarray(acc[("★穴(板)", kind, k)], float)
        print(f"　{kind} k={k}: 限界ROI {roi:.1f}% / "
              f"**上位3本が全払戻の {100*np.sort(v)[-3:].sum()/max(v.sum(),1e-9):.1f}%**")
    print("\n⚠**枠連の運用には触れない**。**設定変更は提案しない**。")


if __name__ == "__main__":
    sys.exit(selftest() if "--selftest" in sys.argv else (main() or 0))
