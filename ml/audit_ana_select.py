"""(188) ★★★**穴自身の信号でレースを絞る** — 推奨度 / ズレの大きさ

★★**問い（ユーザー・2026-09-06）**——
　「**レース絞ろう。今のモデルで考えるなら穴馬の推奨度に応じて考える感じになると思うけど、
　　推奨度によって特徴ないの？**」

★★**これは本当に測っていない切り口**——
　**(183)はレース属性6次元（頭数・距離・芝ダ・クラス・荒れ度・(112)軸E）で
　「穴の優位にレース単位の構造があるか」を測って陰性だった**。
　⚠★**だが6次元に「穴自身の信号の強さ」は入っていなかった**。
　★**穴の信号は馬の量だが、穴を選んだ時点でレースの量になる**——**そこが空いている**。

⚠**近い先行と、何が違うか（判定基準25）**:
　**(174)** 帯内で gap 上位10%の馬を買う → **馬の選別**（**レースは全部使う**）
　**(184)** 帯内で gap 最大の馬を買う → **馬の選別**（**帯に馬が居る全レース**）
　★**(188)** ★**穴の信号が強いレースだけ買う** → ★**レースの選別**（**弱いレースは見送る**）
　→ ★**「何頭買うか」ではなく「何レース買うか」が変わる**。**これが差**。

────────────────────────────────────────────────────────────
★★★ 事前登録（2026-09-06・**結果を見る前にコミットする**）
────────────────────────────────────────────────────────────

■ ★経路（判定基準25）: **q = MLモデルの top3 確率 / q_pool = 複勝の板**。⚠**q 側は弱い**。
■ ★手続き: **ウォークフォワード**（`data/cache/wf_pred.npz` を再利用）。
■ ★穴の定義は(185)(187)と同一: **5-20倍の帯で `gap_f = p_norm − q_pool_fuku` が最大の馬**。
　⚠**帯が5-20倍なのは(182)で post-hoc に選んだもの**（**(180)(181)で水準が高かった中位帯**）。
■ ★買うのは**その穴の複勝1点**。**対照は同じ帯の無作為な1頭（乱）**——**(184)(186)と同じ形**。

■ ★★絞りの信号（**2つ。これ以外に増やさない**）
　★**S1 = 穴の gap_f**（**板に対するズレの大きさ**）
　★**S2 = 穴の推奨度 `p_norm`**（★**ユーザーが言った「推奨度」そのもの**）
　★**それぞれ十分位（10区分）に切る**。**上の分位ほど「信号が強いレース」**。

■ ★★★主判定（**4比較・Bonferroni α=0.01/4・z=2.807**）
　★**A**: **S1 の第10十分位の 穴の複勝ROI の 99%CI下端が 100% を超えるか**
　★**B**: **S2 の第10十分位で同じ**
　★**C**: **S1 の十分位に対する「穴 − 乱」の差の単調性 ρ**
　★**D**: **S2 で同じ**
　★**C/D の帰無分布**: **各ペアの差の符号を独立に ±1 で反転**（**完全な randomization test**・1,000回）。
　★**A/B の帰無**: **ゲート2どおり「板が正しければ払戻率80.0%を返す」**——
　　**100%を超えるかを問う**（**払戻率からの超過20.0ptが必要**）。

■ ★★ゲート2（判定基準42）——**仮説が偽なら何を返すか**
　★**C/D**: **仮説が偽（穴の信号がレースの良し悪しについて何も持たない）なら、
　　十分位は無作為分割と同じになり、ρ の期待値は 0 を返す**。
　★**A/B**: **仮説が偽（板が正しく値付けしている）なら、どの十分位でも ROI は 80.0% を返す**。
　・**買う頭数は全十分位で同じ**（**各1頭**）＝**(170)の形にならない**。
　・**絞り込みで穴側だけを落とさない**（**乱も同じレースで買う**）＝**(168)の形にならない**。

■ ★★★最重要の対照（**(175)で踏んだ形**）
　⚠**5-20倍の帯の中でも、信号の十分位が平均オッズと相関する可能性がある**。
　★**各十分位の平均オッズを必ず併記し、判定に入れる**——
　　**ROIの単調性が平均オッズの単調性で説明できるなら、それは(88)の再発見であって新しくない**。
　★**判定: |ρ(平均オッズ)| > 0.6 の信号は、その結果を「オッズで説明できる」と書く**。

■ ⚠ゲート1: (88)③④を再現（±3pt）／ ★ゲート板: 復元R 0.800±0.020 ／
　★陽性対照: 三連複BOX上位4 81.9%±1.5pt。

■ ★記述（判定しない・**ユーザーの「推奨度によって特徴ないの？」への答え**）
　**十分位ごとに**: **穴の複勝ROI / 乱のROI / 差 / 的中率 / 的中時の平均配当 /
　　平均オッズ / モデル1位率 / 人気中央 / 買うレース割合**。
　★**「信号が強い穴はどんな馬か」がこの表で見える**。

■ ★採用条件（判定基準39/40/41）
　1. **A または B が通る**（**第10十分位で100%超**）
　2. ★**平均オッズの単調性で説明されない**
　3. **裾の検算で符号が反転しない**（上位3本・前後半・年別）
　4. ★**S1 と S2 で同じ向き**（**片方だけなら信号の選び方の産物を疑う**）

■ 予想（⚠**類推なので当てにしない**・判定基準24）
　★**通らないと見る**——**(184)で最良の帯(12-20倍)でも88.4%**、**100%に11.6pt足りない**。
　**信号で絞って +11.6pt を作るには、十分位の上端で劇的に効く必要がある**。
　⚠**(183)で「レース属性6次元に構造なし」も出ている**。**どちらでも驚かない**。

────────────────────────────────────────────────────────────
★★★ 実測（2026-09-06・28,562レース・ウォークフォワード・板ベース）—— **4比較とも通らない**
────────────────────────────────────────────────────────────

■ ゲート1 4帯とも通過 ／ ゲート板 復元R **0.8009** ／ 陽性対照 BOX4 **81.9%** → 全部立った
■ **28,562レース**（板と穴の条件を満たす**全レースの96.9%**）

■ ★★★**ユーザーの「推奨度によって特徴ないの？」への答え — 特徴ははっきりある**
**S2（穴の推奨度）の十分位**
| 十分位 | 穴ROI | 乱ROI | 差 | ★**的中率** | ★**的中時配当** | 平均オッズ | モ1位率 | 人気中央 |
|---|---|---|---|---|---|---|---|---|
| **1**（低） | ★**92.5%** | 74.9% | +17.6円 | **21.1%** | ★**438円** | **16.6倍** | 0.0% | 6番 |
| 3 | 87.4% | 79.0% | +8.4円 | 25.7% | 341円 | 13.3倍 | 0.0% | 5番 |
| 5 | 86.8% | 75.4% | +11.5円 | 30.3% | 286円 | 10.7倍 | 0.4% | 5番 |
| 7 | 87.9% | 82.1% | +5.8円 | 35.7% | 246円 | 8.6倍 | 2.3% | 4番 |
| **10**（高） | ★**91.5%** | 81.6% | +10.0円 | ★**49.0%** | **187円** | **6.3倍** | 13.0% | 3番 |
★★**推奨度が高いほど的中率が2.3倍（21.1→49.0%）になり、配当が2.3分の1（438→187円）になる**。
★★★**完全に打ち消し合ってROIは 86.8〜92.5% でほぼ平坦**。**ρ(差) = −0.491 で単調ですらない**
　（**両端が高いU字**）。→ ★**「推奨度で絞る」は効かない。市場が推奨度をきれいに織り込んでいる**。

■ **S1（穴のgap(板)）の十分位** —— ★**単調に効くが、対照に落ちる**
| 十分位 | 穴ROI | 差 | ★**平均オッズ** | モ1位率 |
|---|---|---|---|---|
| 1 | 78.3% | +8.3円 | **11.5倍** | 0.0% |
| 5 | 88.9% | +16.5円 | 10.9倍 | 1.3% |
| 8 | 94.5% | +16.7円 | 10.2倍 | 3.9% |
| **10** | ★**97.5%** | +16.7円 | **9.8倍** | 13.2% |
★**gapの上位10%に絞ると 97.5%**＝**この線の全記録で最高**。⚠**しかし**:
| | |
|---|---|
| **99%CI** | **[89.1, 105.9]** → ★**100%を跨ぐ＝下端が届かない** |
| ★★**ρ(平均オッズ)** | ⚠⚠**−1.000**（**完全に単調**） |
| **ρ(差)** | **+0.733** vs **帰無99%点 +0.861** → ⚠**届かない** |
★★**平均オッズが 11.5 → 9.8倍 と完全に単調に下がる**＝
　★**「gapの上位＝その帯の低オッズ側」で、(88)の再発見と区別できない**。
　⚠**(175)で全く同じ形を踏んでいる**（**あのときも ρ(平均オッズ)=+1.000 だった**）。

■ ★★主判定: **4比較すべて通らない**
| 判定 | 信号 | 第10十分位ROI | 99%CI | ρ(差) | ρ(平均オッズ) | 結果 |
|---|---|---|---|---|---|---|
| A/C | ★S1 穴のgap(板) | **97.5%** | [89.1,105.9] | +0.733 | ⚠**−1.000** | ⚠通らない |
| B/D | ★S2 穴の推奨度 | 91.5% | [85.6,97.5] | −0.491 | ⚠**−1.000** | ⚠通らない |

■ ★★結論
　★**穴自身の信号でレースを絞っても届かない**。
　★**(183)の「レース属性6次元に構造なし」に、「穴自身の信号にも無い」を足して閉じる**。
　★★**副産物がこの測定の本体**——**推奨度は的中率と配当をきれいに交換するだけで、
　　ROIを動かさない**。**市場が推奨度を正しく値付けしている**ことの直接の実測。
　⚠**経路は弱いので「閉じた」とは書けない**（判定基準25）。


実行: python3 ml/audit_ana_select.py    自己テスト: python3 ml/audit_ana_select.py --selftest
"""
import math
import sys
from itertools import combinations

import numpy as np

sys.path.insert(0, "ml")
import features as F
from audit_crosspool import load_races, payoff, zq
from audit_ana_odds import COST, MIN_HORSES, gate1, roi_of
from audit_ana_marg import ANA_HI, ANA_LO, WF_BOX4, WF_TOL, wf_predict
from audit_ana_board import BOARD_R, BOARD_TOL, NPLACE, load_fuku_boards, qpool
from audit_ana_reco import NDEC, dec_roi, spearman_vs_rank
from train_prod import add_odds_features

SIGS = ["★S1 穴のgap(板)", "★S2 穴の推奨度"]
NPERM = 1000
NCMP = 4
ALPHA = 0.01
SEED = 20260906
FUKU_LINE = 80.0
ODDS_RHO = 0.6


def selftest():
    ok = True
    assert abs(spearman_vs_rank(np.arange(10.0)) - 1.0) < 1e-9
    print("★単調性の自己テスト: 昇順→+1.000　★OK")
    rng = np.random.default_rng(0)
    n = 150_000
    dec = rng.integers(0, NDEC, n)
    diff = rng.normal(0, 300, n)
    rs = []
    for _ in range(300):
        pm = diff * rng.choice([-1.0, 1.0], size=n)
        s = np.bincount(dec, weights=pm, minlength=NDEC)
        c = np.bincount(dec, minlength=NDEC)
        rs.append(spearman_vs_rank(s / np.maximum(c, 1)))
    m = float(np.mean(rs))
    print(f"★ゲート2(C/D)の自己テスト: 符号反転300回の ρ の平均 {m:+.3f}"
          f" → **仮説が偽なら0**: {'★OK' if abs(m) < 0.10 else '⚠NG'}")
    ok &= abs(m) < 0.10
    print(f"★ゲート2(A/B): **板が正しければどの十分位もROIは{FUKU_LINE}%を返す**（100%ではない）")
    print(f"★比較数 {NCMP} → z = {zq(ALPHA/NCMP):.3f}"
          f"　／ ★最重要の対照: |ρ(平均オッズ)| > {ODDS_RHO} なら「オッズで説明できる」")
    print(f"★穴の帯 = {ANA_LO}-{ANA_HI}倍（(182)(185)(187)と同一・⚠post-hoc）")
    print("★自己テスト: " + ("全部OK" if ok else "⚠NG"))
    return 0 if ok else 1


def main():
    z = zq(ALPHA / NCMP)
    print("(188) ★★★**穴自身の信号でレースを絞る** — 推奨度 / ズレの大きさ")
    print("★(183)はレース属性6次元で陰性だったが、**「穴自身の信号の強さ」は入っていなかった**")
    print("★**(174)(184)は馬の選別**（レースは全部使う）。★**本件はレースの選別**\n")

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

    rng = np.random.default_rng(SEED)
    rec = {k: [] for k in ("s1", "s2", "a", "r", "od", "t1", "pop", "yr", "dt")}
    Rs, box4, nall = [], [], 0
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
        nall += 1
        pn = pv / pv.sum() * NPLACE
        qp, Rb = qpool([bd[int(u)] for u in ub], "harm")
        Rs.append(Rb)
        gapf = pn - qp
        order = [int(u) for u in ub[np.argsort(-pv, kind="mergesort")]]
        bx = [payoff(r, "三連複", list(c)) for c in combinations(sorted(order[:4]), 3)]
        if not any(x is None for x in bx):
            box4.append(sum(bx) - 400.0)
        idx = np.where((od >= ANA_LO) & (od < ANA_HI))[0]
        if len(idx) < 2:
            continue
        a = int(idx[int(np.argmax(gapf[idx]))])
        b = int(rng.choice([int(k) for k in idx if k != a]))
        pa = payoff(r, "複勝", [int(ub[a])])
        pb = payoff(r, "複勝", [int(ub[b])])
        if pa is None or pb is None:
            continue
        rec["s1"].append(float(gapf[a])); rec["s2"].append(float(pn[a]))
        rec["a"].append(pa); rec["r"].append(pb); rec["od"].append(float(od[a]))
        rec["t1"].append(a == int(np.argmax(pv)))
        rec["pop"].append(int(np.argsort(np.argsort(od))[a] + 1))
        rec["yr"].append(int(gg["date"].iloc[0].year))
        rec["dt"].append(gg["date"].iloc[0])
    for k in rec:
        rec[k] = np.asarray(rec[k]) if k != "dt" else np.array(rec[k], dtype="datetime64[D]")
    N = len(rec["a"])
    print(f"\n★突き合わせ **{N:,}レース**（板と穴の条件を満たす全レース・**{100*N/max(nall,1):.1f}%**）")

    med = float(np.median(Rs))
    okR = abs(med - BOARD_R) <= BOARD_TOL
    print(f"■ ★ゲート板: 復元R = **{med:.4f}** → **{'★立った' if okR else '⚠⚠落ちた'}**")
    box4 = np.asarray(box4, float)
    g0 = 100.0 * (box4.sum() + 400.0 * len(box4)) / (400.0 * len(box4))
    okb = abs(g0 - WF_BOX4) <= WF_TOL
    print(f"■ ⚠**陽性対照**: BOX上位4 **{g0:.1f}%** vs {WF_BOX4}%"
          f" → **{'★立った' if okb else '⚠⚠落ちた'}**")
    if not (okR and okb):
        print("\n⚠⚠**ゲートが落ちた。読まない**（判定基準32）。")
        return

    A_, R_ = rec["a"].astype(float), rec["r"].astype(float)
    diff = A_ - R_
    res = {}
    for si, key in enumerate(("s1", "s2")):
        s = rec[key].astype(float)
        q = np.quantile(s, np.linspace(0, 1, NDEC + 1)[1:-1])
        dec = np.searchsorted(q, s, side="right")
        res[key] = dec
        print(f"\n{'='*104}")
        print(f"■ ★記述: **{SIGS[si]} の十分位**（★**「推奨度によって特徴があるか」の答え**）")
        print(f"{'十分位':<7}{'R数':>8}{'穴ROI':>9}{'乱ROI':>8}{'差':>9}"
              f"{'的中率':>9}{'的中時配当':>11}{'平均オッズ':>11}{'モ1位率':>9}{'人気中央':>9}")
        for i in range(NDEC):
            m = dec == i
            if m.sum() < 200:
                continue
            print(f"{i+1:<7}{m.sum():>8,}{roi_of(A_[m]):>8.1f}%{roi_of(R_[m]):>7.1f}%"
                  f"{diff[m].mean():>+8.1f}円{100*np.mean(A_[m] > 0):>8.1f}%"
                  f"{(A_[m][A_[m] > 0].mean() if (A_[m] > 0).any() else 0):>10,.0f}円"
                  f"{rec['od'][m].mean():>10.1f}倍{100*np.mean(rec['t1'][m]):>8.1f}%"
                  f"{np.median(rec['pop'][m]):>8.0f}番")

    print(f"\n{'='*104}")
    print(f"■ ★★★主判定（**{NCMP}比較・α={ALPHA}/{NCMP}・z={z:.3f}**）")
    print(f"　★ゲート2: **A/B は「板が正しければ{FUKU_LINE}%を返す」／"
          f"C/D は「信号が何も持たなければ ρ=0」**")
    print(f"\n{'判定':<6}{'信号':<18}{'第10十分位ROI':>16}{'99%CI':>22}"
          f"{'ρ(差)':>9}{'ρ(平均オッズ)':>14}{'結果':>16}")
    hits = []
    for si, key in enumerate(("s1", "s2")):
        dec = res[key]
        m = dec == NDEC - 1
        v = A_[m]
        mu, se = v.mean() - COST, v.std(ddof=1) / math.sqrt(len(v))
        lo_, hi_ = 100 + mu - z * se, 100 + mu + z * se
        okAB = lo_ > 100.0
        # C/D: 差の単調性
        cur = np.array([diff[dec == i].mean() if (dec == i).sum() >= 200 else 0.0
                        for i in range(NDEC)])
        rho = spearman_vs_rank(cur)
        nr = np.zeros(NPERM)
        for t in range(NPERM):
            pm = diff * rng.choice([-1.0, 1.0], size=len(diff))
            c2 = np.array([pm[dec == i].mean() if (dec == i).sum() >= 200 else 0.0
                           for i in range(NDEC)])
            nr[t] = spearman_vs_rank(c2)
        thr = float(np.quantile(np.abs(nr), 1 - ALPHA / NCMP))
        okCD = abs(rho) > thr
        mo = np.array([rec["od"][dec == i].mean() if (dec == i).sum() >= 200 else np.nan
                       for i in range(NDEC)])
        rho_o = spearman_vs_rank(np.nan_to_num(mo, nan=float(np.nanmean(mo))))
        expl = abs(rho_o) > ODDS_RHO
        if okAB:
            hits.append((key, "水準"))
        if okCD and not expl:
            hits.append((key, "単調性"))
        tag = ("⚠オッズで説明できる" if (okAB or okCD) and expl
               else "★★通った" if (okAB or okCD) else "⚠通らない")
        print(f"{'A/C' if si == 0 else 'B/D':<6}{SIGS[si]:<18}{roi_of(v):>15.1f}%"
              f"{f'[{lo_:.1f},{hi_:.1f}]':>22}{rho:>+9.3f}{rho_o:>+14.3f}{tag:>16}")
        print(f"{'':<6}{'  帰無99%点':<18}{'':>15} {'':>22}{thr:>+9.3f}"
              f"{'  要 |ρ|<' + str(ODDS_RHO):>14}")

    print(f"\n■ ★採用条件")
    print(f"　1. A または B（第10十分位で100%超）… {'★満たす' if any(h[1]=='水準' for h in hits) else '⚠満たさない'}")
    print(f"　2. オッズの単調性で説明されない … 上表の判定を参照")
    if not hits:
        print("\n★★★**結論: 穴自身の信号でレースを絞っても届かない**。")
        print("★**(183)の「レース属性6次元に構造なし」に、"
              "「穴自身の信号にも無い」を足して閉じる**。")
        print("⚠**経路は弱いので「閉じた」とは書けない**（判定基準25）。")
        return
    print("\n■ ★裾の検算（**通ったもの**・(77)）")
    for key, kind in hits:
        m = res[key] == NDEC - 1
        v = A_[m]
        yr, dt = rec["yr"][m], rec["dt"][m].astype(int)
        ys = sorted(set(yr))
        ov = sum(1 for u in ys if roi_of(v[yr == u]) > 100.0)
        print(f"　{key}/{kind}: ROI {roi_of(v):.1f}% / **上位3本が全払戻の "
              f"{100*np.sort(v)[-3:].sum()/max(v.sum(),1e-9):.1f}%** / "
              f"前半 {roi_of(v[dt<=np.median(dt)]):.1f}%・後半 {roi_of(v[dt>np.median(dt)]):.1f}% / "
              f"**100%超の年 {ov}/{len(ys)}**")
    print("\n⚠**枠連の運用には触れない**。**設定変更は提案しない**。")


if __name__ == "__main__":
    sys.exit(selftest() if "--selftest" in sys.argv else (main() or 0))
