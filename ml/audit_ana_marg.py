"""(182) ★★★**何頭流すべきか — 「k頭目の紐」の限界ROIを直接測る**

★★問い（ユーザー・2026-09-05/06）——
　「**穴馬軸で三連複で○頭流すとかも考えられる。馬単・馬連・三連単。そこらへんも考慮してある？**」
　「**一つずつ潰そう**」

★★**記録での答え（先に書く）**——**4券種とも(172)で測定済み**:
　**(172) `audit_elbow_kinds.py`**: **軸=差最大**で
　　**馬連 軸+紐1頭 / 馬単 軸を1着固定 / 三連複 軸+紐2頭 / 三連単 軸1着固定+紐2頭の順列**、
　　**紐の切り方3通り（固定3頭・固定4頭・可変）** → ❌**4券種とも主判定を通らず**。
　**(171)**: 三連複 軸1×紐4(6点)79.2% / 紐6(15点)77.7% → ❌**6比較とも有意差なし・損失5倍**。
　**(178)**: 穴×1番人気の2軸で 三連複1点97.2% → 3点62.0%（**選別後のほうが悪化が激しい**）。
⚠★**だが「紐の頭数そのもの」を1〜6頭でスイープしてはいない**（**(172)は3通りの切り方だけ**）。
　→ ★**本件でそこを潰す**。

★★★**決定的な形にする（これが本件の設計の核）**——
　⚠**「点数を増やすと1R損失(円)が増える」はROI<100%ならほぼ自動**（**(170)の形**・判定基準42）。
　★**非自明な問いはひとつだけ**: **「k頭目の紐を足したとき、その追加分だけのROIは100%を超えるか」**。
　→ ★★**超えるなら足すべき／超えないなら足すべきでない**。**これが「何頭流すか」の完全な答え**。
　　 **最適な幅は、限界ROIが100%を割る直前の k**。

────────────────────────────────────────────────────────────
★★★ 事前登録（2026-09-06・**結果を見る前にコミットする**）
────────────────────────────────────────────────────────────

■ ★経路（判定基準25）: 軸・紐の順序に MLモデルの p ／ 穴の帯は単勝オッズ。⚠**弱い経路**。

■ ★★手続き —— **ウォークフォワード**（判定基準6）
　**各年Yを Y年より前の全データで学習して予測**（シード3本）。
　⚠**(181)で単一分割は2〜3pt楽観的と判明した**（BOX4 84.5%→81.9%）。**もう単一分割では測らない**。
　★**予測は `data/cache/wf_pred.npz` にキャッシュする**（**次のスクリプトが再学習しなくて済む**）。

■ ★軸2通り（**これ以外に増やさない**）
　**モ1** … モデル確率1位（**現行**・対照）
　★**穴** … **5-20倍の帯で gap = share − mk が最大の馬**
　　⚠**post-hoc**: **(180)(181)で水準が最も高かった中位帯を見てから選んだ**。**登録に明記する**。

■ ★紐の順序 —— **モデル確率の降順**（軸を除く）。**k番目の紐 = k番目に確率が高い馬**。

■ ★★券種と「k頭目を足したときに増える組」（**限界の定義**）
　| 券種 | 買い目 | k頭目で増える組 | kの範囲 |
　|---|---|---|---|
　| **馬連** | 軸 × 紐 | **軸×紐k**（1点） | 1〜6 |
　| **馬単** | 軸→紐（1着固定） | **軸→紐k**（1点） | 1〜6 |
　| **三連複** | 軸+紐2頭 | **軸+紐k+紐j (j<k)**（k−1点） | 2〜6 |
　| **三連単** | 軸1着固定+紐2頭 | **軸→{紐k,紐j}の順列 (j<k)**（2(k−1)点） | 2〜6 |
　★**合計 2軸 × (6+6+5+5) = 44比較**。**Bonferroni α=0.01/44・z=3.891**。
　⚠**生のCIも併記するが、判定はBonferroniで行う**。

■ ★★★主判定 = **限界ROI − 100%**（**44比較**）
　★**統計量**: **その増分だけを買ったときの「1点あたり払戻 − 100円」の平均**。
　★★**ゲート2（判定基準42）**: **仮説が偽（k頭目を足すことが損益に対して中立）なら、
　　この統計量の期待値は厳密に 0 を返す**。**「点数が増えれば必ず動く量」ではない**——
　　**限界ROIは点数で割ってあるので、幅を変えても自動では動かない**。
　・**絞り込みをしない**（**その紐位置が存在する全レースを使う**）＝**(168)の形にならない**。

■ ⚠ゲート1（判定基準32）: (88)③④を別パーサで再現（±3pt）。
　★**陽性対照**: **ウォークフォワードの三連複BOX上位4が 81.9%±1.5pt**（**(181)の実測値**）。

■ ★採用条件（判定基準39/40/41）
　1. **限界ROIの99%CI(Bonf)下端が100%を超える k が存在する**
　2. ★**その k が券種をまたいで同じ向き**（**1マスだけ跳ねるのは(172)で崩れた形**）
　3. **裾の検算で符号が反転しない**（上位3本・前後半・年別）
　⚠**通らなければ「どの券種でも、紐を1頭足すことは常に損」＝「流す幅は最小が最適」**と書く。

■ ⚠**最良のマスを拾わない**（**(178)でプラセボが109.9%を出した**）。**主判定だけを読む**。

■ 予想（⚠**類推なので当てにしない**・判定基準24）
　★**(77)のBOX4→BOX5が −6.11pt [−8.64,−3.61]（唯一CIが0を跨がなかった比較）**なので、
　**限界ROIは k が増えるほど下がると見る**。⚠**だが「k=1の限界ROIが100%を超えるか」は別問題**で、
　**そこは(180)(181)の水準（最良で84.9%）から見て届かないと予想する**。**どちらでも驚かない**。

────────────────────────────────────────────────────────────
★★★ 実測（2026-09-06・29,485レース・ウォークフォワード）—— **44マス中0マス**
────────────────────────────────────────────────────────────

■ ゲート1: 4帯とも通過 ／ ★陽性対照: 三連複BOX上位4 = **(181)の81.9%を再現して立った**

■ ★★★**限界ROI（k頭目を足したときの追加分だけ・1点あたり）**
| 券種 | k=1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| **馬連**（軸=モ1） | 82.3 | 82.2 | 81.1 | 75.0 | 81.8 | 81.9 |
| **馬単**（軸=モ1） | 78.0 | 79.9 | 77.0 | 73.4 | 76.9 | 75.9 |
| **三連複**（軸=モ1） | — | 81.9 | 81.4 | 78.7 | 77.5 | 73.0 |
| **三連単**（軸=モ1） | — | 72.4 | 75.8 | 70.7 | 73.8 | 72.2 |
| **馬連**（軸=★穴） | 82.9 | 89.9 | 81.9 | 79.8 | 77.1 | 64.5 |
| **馬単**（軸=★穴） | 78.5 | 89.2 | 81.0 | 80.3 | 83.8 | 63.5 |
| **三連複**（軸=★穴） | — | 87.3 | 81.0 | 80.1 | 78.0 | 75.0 |
| **三連単**（軸=★穴） | — | 80.8 | 77.0 | 83.2 | 81.9 | 68.7 |
★**44比較すべてで99%CI(Bonf)の下端が100%に届かない**（**0マス**）。

■ ★★**k=1（いちばん最初の紐）ですら 78.0〜82.9%**
　→ ★★★**最適な幅は「1頭」ではない。「組を買わない」**。
　　 **(77)(171)(172)(178)の「点数を増やすほど悪化」の、限界での説明がこれ**——
　　 **増分が最初から負だから、足せば足すほど損が積み上がる**。

■ ★★★**限界ROIは、その券種の払戻率に貼り付いている**（**紐の位置にほぼ依らない**）
| 券種 | 払戻率 | 限界ROI平均(モ1) | 差 | 限界ROI平均(穴) | 差 |
|---|---|---|---|---|---|
| 馬連 | 77.5% | 80.7% | **+3.2pt** | 79.4% | +1.9pt |
| 馬単 | 75.0% | 76.9% | **+1.9pt** | 79.4% | +4.4pt |
| 三連複 | 75.0% | 78.5% | **+3.5pt** | 80.3% | +5.3pt |
| 三連単 | 72.5% | 73.0% | **+0.5pt** | 78.3% | +5.8pt |
★★**モデルの確率で並べた紐の順序には、払戻率+0.5〜+5.8pt しか情報が乗っていない**。
　**必要なのは +22.5〜+27.5pt**。→ ★**何頭流しても届かない**。
⚠**三連複モ1が 81.9→81.4→78.7→77.5→73.0% と緩やかに落ちる**のは
　**(77)のBOX4→BOX5 −6.11pt と整合**。**だが落ちる前から既に負**。

■ ★★結論
　★**「○頭流す」に伸びしろは無い**。**どの券種・どの軸・どの位置でも、紐の増分は最初から負**。
　⚠**経路は弱いので「閉じた」とは書けない**（判定基準25）。
　★**書けるのは「モデルの確率で並べた紐では、足すほど損になる」まで**。


実行: python3 ml/audit_ana_marg.py [シード数(既定3)]  自己テスト: python3 ml/audit_ana_marg.py --selftest
"""
import math
import os
import sys
from itertools import combinations, permutations

import numpy as np

sys.path.insert(0, "ml")
import features as F
from audit_crosspool import LINE, load_races, payoff, zq
from audit_ana_odds import COST, MIN_HORSES, gate1, roi_of
from train_prod import CAPACITY, add_odds_features, fit_seeds

FIRST_YEAR = 2016
CACHE = "data/cache/wf_pred.npz"
ANA_LO, ANA_HI = 5.0, 20.0        # ⚠post-hoc: (180)(181)で水準が最も高かった中位帯
KMAX = 6
ALPHA = 0.01
WF_BOX4, WF_TOL = 81.9, 1.5       # ★(181)の実測値
SPECS = [("馬連", "馬連", 1, KMAX), ("馬単", "馬単", 1, KMAX),
         ("三連複", "三連複", 2, KMAX), ("三連単", "三連単", 2, KMAX)]
AXES = ["モ1", "★穴"]
NCMP = 2 * (KMAX + KMAX + (KMAX - 1) + (KMAX - 1))


def marginal_combos(kind, ax, himo, k):
    """k頭目の紐を足したときに**増える**組。"""
    hk = himo[k - 1]
    if kind == "馬連":
        return [(ax, hk)]
    if kind == "馬単":
        return [(ax, hk)]
    if kind == "三連複":
        return [tuple(sorted((ax, hk, himo[j]))) for j in range(k - 1)]
    if kind == "三連単":
        out = []
        for j in range(k - 1):
            out.append((ax, hk, himo[j]))
            out.append((ax, himo[j], hk))
        return out
    raise ValueError(kind)


def selftest():
    ok = True
    ax, himo = 9, [1, 2, 3, 4, 5, 6]
    assert marginal_combos("馬連", ax, himo, 3) == [(9, 3)]
    assert marginal_combos("馬単", ax, himo, 3) == [(9, 3)]
    assert len(marginal_combos("三連複", ax, himo, 3)) == 2
    assert len(marginal_combos("三連単", ax, himo, 3)) == 4
    # ★増分の総和が、全体の点数に一致する（恒等式を数値で見せる・判定基準27）
    for kind, n_at in (("馬連", lambda k: k), ("馬単", lambda k: k),
                       ("三連複", lambda k: k * (k - 1) // 2),
                       ("三連単", lambda k: k * (k - 1))):
        for K in range(2, KMAX + 1):
            lo = 1 if kind in ("馬連", "馬単") else 2
            tot = sum(len(marginal_combos(kind, ax, himo, k)) for k in range(lo, K + 1))
            base = 0 if kind in ("馬連", "馬単") else 0
            assert tot + base == n_at(K), (kind, K, tot, n_at(K))
    print("★限界の定義の自己テスト: 増分の総和が全体の点数に一致（4券種・K=2..6）　★OK")
    # ★ゲート2: 中立なら「1点あたり払戻 − 100円」の平均は0
    rng = np.random.default_rng(0)
    n = 200_000
    pay = np.where(rng.random(n) < 0.10, 1000.0, 0.0)   # 期待払戻ちょうど100円
    ms = [float((rng.permutation(pay) - 100.0).mean()) for _ in range(200)]
    m = float(np.mean(ms))
    print(f"★ゲート2の自己テスト: 中立な増分200回の平均 {m:+.3f}円"
          f" → **仮説が偽なら0を返す**: {'★OK' if abs(m) < 2 else '⚠NG'}")
    ok &= abs(m) < 2
    print(f"★比較数 {NCMP} → z = {zq(ALPHA/NCMP):.3f}"
          f"　／ 陽性対照 三連複BOX4 = {WF_BOX4}±{WF_TOL}%（(181)の実測）")
    print(f"★穴の帯 = {ANA_LO}-{ANA_HI}倍　⚠**post-hoc**（(180)(181)を見てから選んだ）")
    print("★自己テスト: " + ("全部OK" if ok else "⚠NG"))
    return 0 if ok else 1


def wf_predict(d, fx, y, nseed):
    """ウォークフォワード予測。キャッシュがあれば使う。"""
    year = d["date"].dt.year.to_numpy()
    if os.path.exists(CACHE):
        z = np.load(CACHE, allow_pickle=True)
        if z["n"] == len(d) and int(z["nseed"]) == nseed:
            print(f"★予測キャッシュを使う: {CACHE}")
            return z["pred"]
    MODEL_DIR, PAR = CAPACITY["l2"]
    ys = [u for u in range(FIRST_YEAR, int(year.max()) + 1) if (year == u).sum() > 5000]
    pred = np.full(len(d), np.nan)
    for u in ys:
        tr, te = year < u, year == u
        ms = fit_seeds(fx[tr], y[tr], nseed, PAR)
        pred[te] = np.mean([m.predict_proba(fx[te])[:, 1] for m in ms], axis=0)
        print(f"　{u}: 学習 {tr.sum():,} → 予測 {te.sum():,}")
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez_compressed(CACHE, pred=pred, n=len(d), nseed=nseed)
    print(f"★予測をキャッシュに保存: {CACHE}")
    return pred


def main(nseed=3):
    z = zq(ALPHA / NCMP)
    print("(182) ★★★**何頭流すべきか — 「k頭目の紐」の限界ROIを直接測る**")
    print("★問い: **k頭目を足したとき、その追加分だけのROIは100%を超えるか**")
    print("　→ ★**超えるなら足すべき／超えないなら足すべきでない**。**これが「何頭流すか」の完全な答え**")
    print("★経路: 軸・紐の順序に MLモデルの p / 穴の帯は単勝オッズ。⚠**弱い経路**")
    print(f"⚠★**穴の帯 {ANA_LO}-{ANA_HI}倍 は post-hoc**（(180)(181)を見てから選んだ）\n")

    races = {r["rid"]: r for r in load_races()}
    rows, bad = gate1(list(races.values()))
    print("⚠**ゲート1**: (88)③④を別パーサで再現・許容±3pt")
    for nm, n, roi, known, dd, ok in rows:
        print(f"　{nm:<12}{roi:>7.1f}% vs {known:>5.1f}%　差 {dd:+.1f}pt"
              f"　{'★立った' if ok else '⚠落ちた'}")
    if bad:
        print("\n⚠⚠**ゲート1が落ちた。読まない**。")
        return

    d = F.to_model(F.load_files())
    f = F.build_features(d)
    keep = (f["n_prior"] >= 1) & d["odds"].notna() & (d["odds"] > 0)
    d, f = d[keep].reset_index(drop=True), f[keep].reset_index(drop=True)
    y = (d["finish"] <= 3).astype(int).to_numpy()
    fx, _ = F.encode_categoricals(f)
    fx = add_odds_features(fx, d["odds"].to_numpy(float), d["raceid"].to_numpy())
    print(f"\n★ウォークフォワード（シード{nseed}本・判定基準6）")
    pred = wf_predict(d, fx, y, nseed)
    m = ~np.isnan(pred)
    sub = d.loc[m, ["raceid", "umaban", "odds", "date"]].copy()
    sub["p"] = pred[m]

    acc = {(a, s[0], k): [] for a in AXES for s in SPECS
           for k in range(s[2], s[3] + 1)}
    box4, nrace = [], 0
    for rid, g in sub.groupby("raceid"):
        r = races.get(str(rid))
        if r is None:
            continue
        nums = {u for u, _, _ in r["horses"]}
        if len(nums) < MIN_HORSES:
            continue
        gg = g[g["umaban"].astype(int).isin(nums)]
        if len(gg) < MIN_HORSES:
            continue
        od = gg["odds"].to_numpy(float)
        pv = gg["p"].to_numpy(float)
        ub = gg["umaban"].astype(int).to_numpy()
        if not np.isfinite(od).all() or (od <= 0).any() or pv.sum() <= 0:
            continue
        gap = pv / pv.sum() - (1.0 / od) / (1.0 / od).sum()
        order = [int(u) for u in ub[np.argsort(-pv, kind="mergesort")]]
        bx = [payoff(r, "三連複", list(c)) for c in combinations(sorted(order[:4]), 3)]
        if not any(x is None for x in bx):
            box4.append(sum(bx) - 400.0)
        idx = np.where((od >= ANA_LO) & (od < ANA_HI))[0]
        axes = {"モ1": order[0]}
        if len(idx):
            axes["★穴"] = int(ub[int(idx[int(np.argmax(gap[idx]))])])
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
                    acc[(a, kind, k)].append(sum(vs) / len(cs))   # ★1点あたり払戻
                    used = True
        nrace += used
    print(f"\n★突き合わせ {nrace:,}レース")

    box4 = np.asarray(box4, float)
    g0 = 100.0 * (box4.sum() + 400.0 * len(box4)) / (400.0 * len(box4))
    okb = abs(g0 - WF_BOX4) <= WF_TOL
    print(f"⚠**陽性対照**: 三連複BOX上位4 **{g0:.1f}%** vs (181)の {WF_BOX4}%"
          f" → **{'★立った' if okb else '⚠⚠落ちた'}**")
    if not okb:
        print("\n⚠⚠**陽性対照が落ちた。読まない**（判定基準32）。")
        return

    print(f"\n{'='*104}")
    print(f"■ ★★★限界ROI（**k頭目を足したときの追加分だけ**・1点あたり）"
          f"　**{NCMP}比較・Bonferroni α={ALPHA}/{NCMP}・z={z:.3f}**")
    print("　★ゲート2: **仮説が偽（k頭目が損益に中立）なら「1点あたり払戻−100円」の期待値は0**")
    hits = []
    for a in AXES:
        print(f"\n── 軸 = {a} ──")
        print(f"{'券種':<8}{'k':>3}{'増分の点数':>10}{'標本R':>9}{'限界ROI':>10}"
              f"{'99%CI(Bonf)':>24}{'判定':>18}")
        for kind, kk, lo, hi in SPECS:
            for k in range(lo, hi + 1):
                v = np.asarray(acc[(a, kind, k)], float)
                if len(v) < 300:
                    continue
                mu = v.mean() - COST
                se = v.std(ddof=1) / math.sqrt(len(v))
                lo_, hi_ = mu - z * se, mu + z * se
                roi = 100.0 * v.mean() / COST
                sig = lo_ > 0
                if sig:
                    hits.append((a, kind, k, roi))
                npts = len(marginal_combos(kind, 99, list(range(1, KMAX + 1)), k))
                print(f"{kind:<8}{k:>3}{npts:>10}{len(v):>9,}{roi:>9.1f}%"
                      f"{f'[{100+lo_:.1f},{100+hi_:.1f}]':>24}"
                      f"{'★★100%超' if sig else '⚠100%を超えない':>18}")

    print(f"\n■ ★採用条件")
    print(f"　1. 限界ROIの99%CI下端が100%超の k … **{len(hits)}マス**"
          f" → {'★満たす' if hits else '⚠満たさない'}")
    if not hits:
        print("\n★★★**結論: どの券種・どの軸でも、紐を1頭足す増分は100%を超えない**。")
        print("★**＝「流す幅は最小が最適」**。**○頭流す形に伸びしろは無い**。")
        print("⚠**経路は弱いので「閉じた」とは書けない**（判定基準25）。")
        print("★**書けるのは「モデルの確率で並べた紐では、足すほど損になる」まで**。")
        return
    print(f"\n■ ★裾の検算（**通ったマス**・(77)）")
    for a, kind, k, roi in hits:
        v = np.asarray(acc[(a, kind, k)], float)
        print(f"　{a} / {kind} k={k}: 限界ROI {roi:.1f}% / "
              f"**上位3本が全払戻の {100*np.sort(v)[-3:].sum()/max(v.sum(),1e-9):.1f}%**")
    print("⚠**枠連の運用には触れない**。**設定変更は提案しない**。")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    n = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 3
    sys.exit(main(n) or 0)
