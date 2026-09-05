"""(177) ★★★**精緻化の応答で交絡を切り分ける** — 「オッズの幅→0」に外挿する

⚠⚠★★**先に書く: これは post-hoc の再定式化である**（**判定基準25/37**）。
　**(176)の記述で NBIN=20 の効果（第10−第1 = +13.1円）と NBIN=40 の表を見た後に書いた**。
　★**未見なのは NBIN=10 と NBIN=80 の2点**。**そこが falsifiable な部分**。
　⚠**「見た2点に合う話」はいくらでも書ける**（判定基準39/40）。**80分位で効果が縮めばこの案は落ちる**。

★★経緯（`ANA_TRACK.md`）——
　**(174)** 差で切った → 機構は在るが張れない（複勝87.6%）
　**(175)** 推奨度で切った → **曲線は5帯とも出たが平均オッズの ρ=+1.000 で全部説明が付いた**
　**(176)** オッズを40分位で中和 → ★**ゲート0が落ちた。だが落ちたのはゲートの定式化のほう**——
　　　　　**ρ は尺度を持たない量なので、どんなに細かくしても +1.000 のまま**＝**達成不能な条件**。
　→ ★**本スクリプトは ρ を捨て、「精緻化したとき効果が縮むか」を判定そのものにする**。

★★★**論理**——
　**もし効果が「帯の中に残ったオッズの傾き」の産物なら、統制を細かくして
　　残余オッズ幅 s を 0 に近づけると、効果 e も 0 に近づかなければならない**。
　→ ★**e を s に回帰して s=0 に外挿した切片が、効果のうち交絡で説明できない部分**。
　⚠**これは「恒等式」ではなく「線形の当てはめ」**（判定基準27: **恒等式は名乗るものではない**）。
　　★**だから外挿は線形性の仮定に乗っている**。**4点の当てはまり(残差)も必ず出す**。

────────────────────────────────────────────────────────────
★★★ 事前登録（2026-09-05・**結果を見る前にコミットする**）
────────────────────────────────────────────────────────────

■ ★経路（判定基準25）: q = MLモデルの top3 確率 p ／ 統制 = 単勝オッズ。⚠**弱い経路**。
　⚠**標本は(174)(175)(176)と同一**＝**独立な検証ではない**（判定基準35）。

■ 券種 —— ★**複勝のみ**（(174)の実測で単勝は検出限界に負ける・`ANA_TRACK.md` 4.）

■ 設計
　★**NBIN = 10 / 20 / 40 / 80** の4点。**各点で**:
　　・単勝オッズの分位で NBIN 個のビンに切る
　　・**ビンの中で** `ratio = share/mk` の十分位を作り、**ビンをまたいで集計**
　　・**s = (十分位の平均オッズ max/min) − 1**（★**残余オッズ幅**）
　　・**e = 第10十分位 − 第1十分位 の1点あたり損益差（円）**
　★**e を s に回帰**（**重みは 1/SE²**）→ ★**s=0 の切片 b0**。

■ ⚠ゲート0（訂正版）—— ★**大きさの条件だけ残す。ρ は使わない**
　**NBIN=40 と 80 で s < 0.05**（**残余オッズ幅5%未満**）であること。
　★**(176)で 40分位の s=0.045 は既に満たしている**。**80分位は未見**。
　⚠**ρ(平均オッズ) は印字するが判定に使わない**——**尺度を持たないので達成不能**((176)で判明)。

■ ⚠ゲート1（判定基準32）—— (174)(175)(176)と同じ（(88)③④を別パーサで再現・±3pt）

■ ★★ゲート2（判定基準42）——**仮説が偽なら何を返すか**
　★**主判定 = 切片 b0**。
　★**仮説が偽（同じオッズなら p は払戻について何も持たない）なら、
　　どの NBIN でも e の期待値は 0 になり、したがって b0 も 0 を返す**。
　★**帰無分布は「ビン内で複勝払戻をシャッフルして4点すべてを作り直し、同じ回帰をかける」**
　　（**NPERM=200**）——**回帰の手続きごと帰無に通す**ので、**当てはめの自由度も帰無に入る**。
　・**買う頭数は全十分位で同じ**＝(170)の形にならない／**絞り込みをしない**＝(168)の形にならない。

■ ★★採用条件（判定基準39/40/41）
　1. ★**b0 が帰無分布の99%点を超える**（**片側ではなく |b0|**）
　2. ★★**単調に縮んでいない**——**s が小さくなるほど e が小さくなる並びなら交絡**。
　　 **判定: e を s で並べたときの Spearman ρ が +0.8 以上なら、外挿が有意でも採用しない**
　3. ★**4点の当てはまりが破綻していない**（**残差の最大が |b0| の半分未満**）
　4. **裾の検算で符号が反転しない**（上位3本・前後半・年別）
　5. ★**ROI>100% でなければ「機構は在るが張れない」**（(174)(175)と同じ分岐）

■ ⚠既に測ってあること（判定基準25）
　★**(45)③「オッズ帯を揃えても全帯で市場を上回る」(+0.4〜+19.2pt)**。
　⚠**(46)がその帯を細かく割ると −12.9/+15.6/+58.4/−20.4 と暴れた＝区分の産物**。
　→ ★**本件の差分は「区分を細かくしたときの応答そのものを判定にした」こと**。
　　 **(46)は暴れを見て諦めたが、暴れは「最良セル」を見ていたから**。**本件は十分位の端の差を見る**。

■ 予想
　⚠**見た2点（20分位13.0pt / 40分位15.6pt）からは「縮まない」に見える**。
　★**だがそれが本当なら、b0 は 13〜16円あたりに出て、採用条件2も通るはず**。
　★**逆に、10分位で効果が大きく80分位で小さければ、それは交絡の署名**＝**案は落ちる**。
　⚠**どちらでも驚かない**。**2点で法則を書かない**（判定基準39）。

────────────────────────────────────────────────────────────
★★★ 実測（2026-09-05）—— **採用条件は3つとも通った。だが水準が足りない**
────────────────────────────────────────────────────────────

■ ゲート1: 4帯とも通過（88.1 / 82.4 / 54.4 / 38.2）
■ 標本: 26,583レース / 366,754頭（⚠**(174)(175)(176)と同一**）

■ ★4点（**未見だった NBIN=10 と 80 を含む**）
| NBIN | 残余オッズ幅 s | ρ(平均オッズ) | **第10−第1 e** | SE | 第1 ROI | 第10 ROI |
|---|---|---|---|---|---|---|
| 10 | 0.2147 | +1.000 | **+12.4円** | 2.03 | 68.1% | 80.5% |
| 20 | 0.0968 | +1.000 | **+13.1円** | 2.00 | 67.9% | 80.9% |
| **40** | **0.0454** | +1.000 | ★**+15.6円** | 1.97 | 65.9% | **81.5%** |
| **80** | ★**0.0203** | +1.000 | **+14.4円** | 1.98 | 66.9% | 81.3% |
■ ゲート0（訂正版・大きさのみ）: **s(40)=0.0454 / s(80)=0.0203 < 0.05** → ★**通った**

■ ★★主判定
　★**切片 b0 = +15.16円**（**s=0 への外挿**）／ **傾き b1 = −13.51**
　**帰無99%点 = 3.75円**（ビン内シャッフル200回・**回帰の手続きごと**） → ★**超えた**
| 採用条件 | 結果 |
|---|---|
| 1. b0 が帰無99%点を超える | ★**満たす**（15.16 vs 3.75） |
| 2. **s が小さいほど e も小さい並びでない** | ★**満たす**（**ρ=−0.800**＝**むしろ逆**） |
| 3. 4点の当てはまり | ★**満たす**（残差の最大 1.05円 < 7.58） |

■ ★★★**交絡は効果を作っていたのではなく、隠していた**（**傾き b1 が負**）
　★**残余オッズは第10十分位を高オッズ側へ +4.5% 押す**——
　　**`ratio = share/mk ∝ p·o` なので、同じ p なら高オッズの馬ほど比が大きくなる**から。
　　**(88)で高オッズほど悪い**ので、**この残差は e を下振れさせる**。
　→ ★**中和すると e は 12.4 → 15.2円へ増える**。**(175)とは符号が逆**——
　　 **(175)は「帯の中で share が高い＝低オッズ」で有利方向の交絡だった**。
　⚠★**同じ「オッズの残差」でも、変数の作り方で符号が入れ替わる**。**向きを予想で決めない**（判定基準34）。

■ ★★★得られた答え —— **「同じオッズの馬どうしを比べても、モデルは残余の情報を持つ」**
　★**交絡をゼロに外挿しても +15.2円/100円 の分離が残る**。**帰無99%点の4倍**。
　★**4点すべてで +12.4〜+15.6円**、**残差1.05円**＝**極めて安定**。
　→ ★**モデルは単勝オッズの読み直しではない**。**(63)(67)A(68)の「オッズが全公開情報を吸収」を
　　 精緻化する**——**吸収しきっていない部分が、複勝ROIで15pt分ある**。

■ ⚠⚠**それでも張れない（採用条件5）**
　★**第10十分位の ROI は 81.5%**。**複勝の払戻率 80.0% に対し +1.5pt だけ**。
　**100%に要るのは +20.0pt**。→ ★**「機構は在るが張れない」**。
| 検算 | 値 |
|---|---|
| 上位3本が全払戻に占める割合 | **1.1%**（★健全） |
| 前半 / 後半 | **81.7% / 81.3%**（★割れない） |
| ★**100%超の年** | ★**0/10** |
| モデル1位率 / 平均オッズ / 人気中央 | 9.3% / 69.4倍 / 7番 |

■ ★★**(174)と並べて読む（同一標本なので水準のCIは見比べない・判定基準35）**
| | 選び方 | 最良の水準 | 払戻率との差 |
|---|---|---|---|
| **(174)** | **オッズ帯を残したまま**「差」の上位10% | **複勝 87.6%**（3-6倍） | **+7.6pt** |
| **(177)** | ★**オッズを中和して**「比」の第10十分位 | **複勝 81.5%** | ★**+1.5pt** |
★★**中和すると水準が落ちる**。→ ★**(174)の +7.6pt のうち、大半は「どのオッズ帯を買うか」から来ていた**。
　**モデルの残余情報そのものは +1.5pt しかない**。
　⚠**これは(88)と完全に整合する**——**市場の誤りは最大でも+10ptで、その大半はオッズ帯に貼り付いている**。

■ ★★結論
　**「モデルが市場より高く評価している穴馬に、その馬自身の1頭の札を張る」は、
　　機構としては実在するが、水準は払戻率+1.5〜7.6pt で、必要な+20ptに届かない**。
　⚠**経路は弱い（単勝オッズ由来）ので「閉じた」とは書けない**（判定基準25）。
　★**書けるのは「モデルの確率では、1頭の札は100%に届かない」まで**。


実行: python3 ml/audit_ana_refine.py     自己テスト: python3 ml/audit_ana_refine.py --selftest
"""
import math
import sys

import numpy as np

sys.path.insert(0, "ml")
import features as F
from audit_crosspool import load_races, payoff, zq
from audit_ana_odds import COST, MIN_HORSES, gate1, roi_of
from audit_ana_reco import NDEC, SEED, dec_roi, spearman_vs_rank
from audit_ana_neut import bin_perm_index, within_bin_decile
from train_prod import CAPACITY, add_odds_features, fit_seeds

NBINS = [10, 20, 40, 80]
NPERM = 200
SMAX = 0.05                  # ゲート0: 残余オッズ幅（NBIN=40,80で要求）
SHRINK_RHO = 0.8             # 採用条件2: e を s で並べた ρ がこれ以上なら不採用


def wls_intercept(s, e, se):
    """e = b0 + b1·s を 1/SE² 重みで当てて (b0, b1, 残差) を返す。"""
    w = 1.0 / np.maximum(np.asarray(se, float) ** 2, 1e-12)
    X = np.column_stack([np.ones(len(s)), np.asarray(s, float)])
    W = np.diag(w)
    beta = np.linalg.solve(X.T @ W @ X, X.T @ W @ np.asarray(e, float))
    resid = np.asarray(e, float) - X @ beta
    return float(beta[0]), float(beta[1]), resid


def selftest():
    ok = True
    # 完全な交絡（e = 3·s、切片0）を復元できるか
    s = np.array([0.20, 0.10, 0.05, 0.025])
    b0, b1, r = wls_intercept(s, 3.0 * s, np.ones(4))
    assert abs(b0) < 1e-9 and abs(b1 - 3.0) < 1e-9
    # 縮まない場合（e一定）は切片がその値
    b0, b1, r = wls_intercept(s, np.full(4, 12.0), np.ones(4))
    assert abs(b0 - 12.0) < 1e-9 and abs(b1) < 1e-9
    print(f"★回帰の自己テスト: 交絡(e=3s)→切片 0.000 / 一定(e=12)→切片 12.000　★OK")
    # ★ゲート2: ビン内シャッフルなら4点とも0付近＝切片も0
    rng = np.random.default_rng(0)
    n = 60_000
    binid = rng.integers(0, 20, n)
    dec = within_bin_decile(rng.random(n), binid, 20)
    pay = rng.choice([0.0, 800.0], size=n, p=[0.875, 0.125])
    b0s = []
    for _ in range(100):
        sp = pay[bin_perm_index(binid, 20, rng)]
        es = [sp[dec == NDEC-1].mean() - sp[dec == 0].mean() for _ in range(4)]
        b0s.append(wls_intercept(s, es, np.ones(4))[0])
    m = float(np.mean(b0s))
    print(f"★ゲート2の自己テスト: シャッフル100回の切片の平均 {m:+.2f}円"
          f" → **仮説が偽なら0を返す**: {'★OK' if abs(m) < 15 else '⚠NG'}")
    ok &= abs(m) < 15
    print("★自己テスト: " + ("全部OK" if ok else "⚠NG"))
    return 0 if ok else 1


def main():
    print("(177) ★★★**精緻化の応答で交絡を切り分ける** — 「オッズの幅→0」に外挿する")
    print("⚠⚠**post-hoc の再定式化である**（(176)の20分位・40分位を見た後に書いた）。")
    print("　★**未見は NBIN=10 と 80**。**そこが falsifiable な部分**")
    print("★経路: q = MLモデルの top3 確率 / 統制 = 単勝オッズ。⚠**弱い経路**")
    print("⚠**標本は(174)(175)(176)と同一＝独立な検証ではない**（判定基準35）\n")

    races = {r["rid"]: r for r in load_races()}
    rows, bad = gate1(list(races.values()))
    print("⚠**ゲート1**: (88)③④を別パーサで再現・許容±3pt")
    for nm, n, roi, known, dd, ok in rows:
        print(f"　{nm:<12}{roi:>7.1f}% vs {known:>5.1f}%　差 {dd:+.1f}pt"
              f"　{'★立った' if ok else '⚠落ちた'}")
    if bad:
        print("\n⚠⚠**ゲート1が落ちた。読まない**。")
        return

    MODEL_DIR, PAR = CAPACITY["l2"]
    d = F.to_model(F.load_files())
    f = F.build_features(d)
    keep = (f["n_prior"] >= 1) & d["odds"].notna() & (d["odds"] > 0)
    d, f = d[keep].reset_index(drop=True), f[keep].reset_index(drop=True)
    y = (d["finish"] <= 3).astype(int).to_numpy()
    fx, _ = F.encode_categoricals(f)
    fx = add_odds_features(fx, d["odds"].to_numpy(float), d["raceid"].to_numpy())
    cut = d["date"].quantile(0.3)
    tr, te = (d["date"] < cut).to_numpy(), (d["date"] >= cut).to_numpy()
    print(f"\n学習 {tr.sum():,} / 検証 {te.sum():,}・分割 {cut.date()}（3シード平均）")
    ms = fit_seeds(fx[tr], y[tr], 3, PAR)
    p = np.mean([m.predict_proba(fx[te])[:, 1] for m in ms], axis=0)
    sub = d.loc[te, ["raceid", "umaban", "odds", "date"]].copy()
    sub["p"] = p

    keys = ("odds", "ratio", "date", "fuku", "top1", "pop")
    rec = {k: [] for k in keys}
    nrace = 0
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
        if not np.isfinite(od).all() or (od <= 0).any() or pv.sum() <= 0:
            continue
        share = pv / pv.sum()
        mk = (1.0 / od) / (1.0 / od).sum()
        fv, ok = [], True
        for u in gg["umaban"].astype(int):
            b = payoff(r, "複勝", [int(u)])
            if b is None:
                ok = False
                break
            fv.append(b)
        if not ok:
            continue
        nrace += 1
        t1 = np.zeros(len(pv), bool); t1[int(np.argmax(pv))] = True
        rec["odds"].append(od); rec["ratio"].append(share / np.maximum(mk, 1e-12))
        rec["date"].append(gg["date"].to_numpy()); rec["fuku"].append(np.asarray(fv, float))
        rec["top1"].append(t1); rec["pop"].append(np.argsort(np.argsort(od)) + 1)
    for k in keys:
        rec[k] = np.concatenate(rec[k])
    od, pay = rec["odds"], rec["fuku"]
    print(f"★突き合わせ {nrace:,}レース / {len(od):,}頭　⚠**(174)(175)(176)と同一標本**\n")

    # 各 NBIN で s と e
    setup, S, E, SE = {}, [], [], []
    print(f"{'NBIN':>6}{'残余オッズ幅 s':>16}{'ρ(平均オッズ)':>15}{'第10−第1 e':>13}"
          f"{'SE':>8}{'第1 ROI':>10}{'第10 ROI':>10}")
    for nb in NBINS:
        qs = np.quantile(od, np.linspace(0, 1, nb + 1)[1:-1])
        binid = np.searchsorted(qs, od, side="right")
        dec = within_bin_decile(rec["ratio"], binid, nb)
        mo = np.array([od[dec == i].mean() for i in range(NDEC)])
        s = float(mo.max() / mo.min() - 1.0)
        hi, lo = pay[dec == NDEC-1], pay[dec == 0]
        e = float(hi.mean() - lo.mean())
        se = math.sqrt(hi.var(ddof=1)/len(hi) + lo.var(ddof=1)/len(lo))
        setup[nb] = (binid, dec)
        S.append(s); E.append(e); SE.append(se)
        print(f"{nb:>6}{s:>15.4f}{spearman_vs_rank(mo):>+15.3f}{e:>+12.1f}円"
              f"{se:>8.2f}{roi_of(lo):>9.1f}%{roi_of(hi):>9.1f}%")

    # ゲート0（訂正版・大きさの条件だけ）
    s40, s80 = S[NBINS.index(40)], S[NBINS.index(80)]
    g0 = s40 < SMAX and s80 < SMAX
    print(f"\n⚠**ゲート0（訂正版・大きさのみ）**: NBIN=40 の s={s40:.4f} / "
          f"NBIN=80 の s={s80:.4f}（要 <{SMAX}）→ **{'★通った' if g0 else '⚠⚠落ちた'}**")
    print("　★**ρ(平均オッズ)は印字するが判定に使わない**——"
          "**尺度を持たないので達成不能**((176)で判明・判定基準37）")
    if not g0:
        print("\n⚠⚠**ゲート0が落ちた。以降を読まない**（事前登録どおり）。")
        return

    b0, b1, resid = wls_intercept(S, E, SE)
    # 帰無: ビン内シャッフルで4点を作り直し、同じ回帰をかける
    rng = np.random.default_rng(SEED)
    nb0 = np.zeros(NPERM)
    for t in range(NPERM):
        es = []
        for nb in NBINS:
            binid, dec = setup[nb]
            sp = pay[bin_perm_index(binid, nb, rng)]
            es.append(sp[dec == NDEC-1].mean() - sp[dec == 0].mean())
        nb0[t] = wls_intercept(S, es, SE)[0]
    thr = float(np.quantile(np.abs(nb0), 0.99))
    shrink = spearman_vs_rank(np.array(E)[np.argsort(S)])   # s の小さい順に e を並べる
    rmax = float(np.max(np.abs(resid)))

    print(f"\n■ ★★主判定（**帰無はビン内シャッフル{NPERM}回・回帰の手続きごと通す**）")
    print("　★ゲート2: **仮説が偽ならどの NBIN でも e の期待値は0＝切片も0を返す**")
    print(f"　**切片 b0 = {b0:+.2f}円**（s=0 への外挿）　傾き b1 = {b1:+.2f}")
    print(f"　**帰無の99%点 = {thr:.2f}円** → **{'★超えた' if abs(b0) > thr else '⚠超えない'}**")
    print(f"\n■ ★採用条件")
    c1 = abs(b0) > thr
    c2 = shrink < SHRINK_RHO
    c3 = rmax < abs(b0) / 2
    print(f"　1. b0 が帰無99%点を超える … {'★満たす' if c1 else '⚠満たさない'}")
    print(f"　2. ★**s が小さいほど e も小さい（＝縮む）並びでない** … "
          f"ρ={shrink:+.3f}（要 <{SHRINK_RHO}）→ {'★満たす' if c2 else '⚠満たさない'}")
    print(f"　3. 4点の当てはまり … 残差の最大 {rmax:.2f}円 < |b0|/2 = {abs(b0)/2:.2f}"
          f" → {'★満たす' if c3 else '⚠満たさない'}")
    if not (c1 and c2 and c3):
        print("\n★★**結論: 採用条件を満たさない＝陰性**。")
        if not c2:
            print("⚠★**s が小さくなるほど e も小さくなった＝交絡の署名**。"
                  "**(176)で見えた「縮まない」は2点のまぐれだった**（判定基準39）。")
        print("⚠**経路は弱いので「閉じた」とは書けない**（判定基準25）。")
        return

    binid, dec = setup[40]
    sel = pay[dec == NDEC-1]
    dt = rec["date"][dec == NDEC-1].astype("datetime64[D]").astype(int)
    yr = rec["date"][dec == NDEC-1].astype("datetime64[Y]").astype(int) + 1970
    ys = sorted(set(yr))
    over = sum(1 for u in ys if roi_of(sel[yr == u]) > 100.0)
    med = np.median(rec["date"].astype("datetime64[D]").astype(int))
    print(f"\n■ ★裾の検算（NBIN=40 の第10十分位・{len(sel):,}頭）")
    print(f"　ROI {roi_of(sel):.1f}% / **上位3本が全払戻の "
          f"{100*np.sort(sel)[-3:].sum()/max(sel.sum(),1e-9):.1f}%** / "
          f"前半 {roi_of(sel[dt<=med]):.1f}%・後半 {roi_of(sel[dt>med]):.1f}% / "
          f"**100%超の年 {over}/{len(ys)}**")
    print(f"　モデル1位率 {100*rec['top1'][dec==NDEC-1].mean():.1f}% / "
          f"平均オッズ {od[dec==NDEC-1].mean():.1f}倍 / "
          f"人気中央 {np.median(rec['pop'][dec==NDEC-1]):.0f}番")
    print("\n⚠**ROIが100%を超えなければ運用の候補にならない**（事前登録）。")
    print("⚠**枠連の運用には触れない**。**設定変更は提案しない**。")


if __name__ == "__main__":
    sys.exit(selftest() if "--selftest" in sys.argv else (main() or 0))
