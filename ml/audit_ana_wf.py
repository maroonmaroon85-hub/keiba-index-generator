"""(181) ★★★**(180)の曲線をウォークフォワードで測り直し、形を事前登録して検定する**

★★なぜこれをやるか（**2つの穴を同時に埋める**）——

**穴1: この線は一度もウォークフォワードを通していない**
　★**判定基準6**: **「新しい主張は `ml/audit_walkforward.py` で確認すること」**。
　**(78)で三連複BOX4が 84.5% → 82.2% と2.3pt動いた**——**券種によって2pt以上ずれる**。
　⚠**(174)〜(180)の7本はすべて「前30%学習・後70%検証」の単一分割**＝
　　**実運用より3.3倍少ない学習量のモデルで測っている**。**一度も確認していない**。

**穴2: (180)の「差が程度とともに増える」は post-hoc**
　★**9帯すべてで差が正（両側符号検定 p=0.0039）・しかも単調に近く増える**——
　　⚠**事前登録に符号検定も単調性も書いていない**。→ ★**本件で先に登録して検定する**。
　⚠★**「モデルは同じオッズでも残余情報を持つ」自体は(177)で証明済み**（切片+15.16円 vs 帰無3.75円）。
　　→ ★**そこを測り直すのではない**。**新しいのは「形」（程度との関係）だけ**。

⚠⚠★**先に限界を書く（判定基準35）**——**これは独立な標本ではない**。
　**2017年以降は(174)〜(180)と同じレース・同じ払戻を使う**。**新しいのは2016年だけ**。
　★**ウォークフォワードが与えるのは「実運用と同じ学習量・同じ手続き」であって、新しい標本ではない**。
　→ ★**「まぐれでない」ことの証拠にはならない**。**言えるのは「手続きを実運用に揃えても残るか」まで**。

────────────────────────────────────────────────────────────
★★★ 事前登録（2026-09-05・**結果を見る前にコミットする**）
────────────────────────────────────────────────────────────

■ ★経路（判定基準25）: 穴の選定に MLモデルの p ／ 程度の固定は単勝オッズ。⚠**弱い経路**。

■ ★★手続き（**実運用と同じ形**）
　**各年 Y（2016〜）について、Y年より前の全データで学習 → Y年を予測**（**expanding window**）。
　**シード3本の平均**（判定基準1）。**これが `train_prod.py` の手続きそのもの**。

■ ★設計は(180)と同一（**変えない**）
```
各レース × 各オッズ帯 について
  穴 = その帯にいる馬のうち gap = share − mk が最大の1頭
  乱 = その帯にいる別の1頭（無作為）
  両方の複勝を100円ずつ買い、対応差を取る
```
　**程度は9段**: 1.5-3 / 3-5 / 5-8 / 8-12 / 12-20 / 20-35 / 35-60 / 60-120 / 120倍〜

■ ★★★主判定（**2つ。α=0.01/2・z=2.807**）
　★**A 単調性**: **9帯の対応差(円)と帯の順序の Spearman ρ**。
　★**B 符号**: **9帯のうち差が正である帯の数**。
　★**帰無分布は「対応のある符号の入れ替え」**——**各ペア（同じレース・同じ帯の 穴 と 乱）の
　　差の符号を独立に ±1 で反転**（**完全な randomization test**）。**1,000回**。
　⚠**AとBは同じデータの別の側面なので独立ではない**。**両方通って初めて「形がある」と書く**。

■ ★★ゲート2（判定基準42）——**仮説が偽なら何を返すか**
　★**仮説が偽（gap が帯の中で何も持たない）なら、穴と乱は交換可能なので
　　各ペアの差の符号は対称になり、A の期待値は 0、B の期待値は 4.5（9帯の半分）を返す**。
　★**符号の入れ替えによる帰無分布が、まさにその状態**。
　・**買う頭数・コスト・券種は完全に同一**＝(170)の形にならない／**絞り込みをしない**＝(168)の形にならない。

■ ⚠ゲート1（判定基準32）: (88)③④を別パーサで再現（±3pt）。
　★**加えてウォークフォワードの陽性対照**: **三連複BOX上位4が (78)の 82.2%〜(77)の84.5% の範囲**に入ること。
　　（⚠**単一分割の84.5%ではなく、ウォークフォワードの値と比べる**。**(78)で2.3pt下がると分かっている**）

■ ★採用条件（判定基準39/40/41）
　1. **A と B の両方が帰無分布の99%点を超える**
　2. ★**単一分割((180))と同じ向き**（**手続きを変えても残るか**）
　3. **裾の検算で符号が反転しない**（上位3本・前後半・年別）
　4. ★**それでも水準(a)が払戻率80.0%を超える帯に差が無ければ「機構は在るが張れない」**
　　 ⚠**(180)では差が有意な帯の水準が60.1%だった**。**そこは変わらないと予想する**。

■ ★★何が分かっても運用は変わらない（**先に書く**）
　⚠**(180)で2本の曲線が逆向きだと分かっている**——**差が出る程度では水準が足りない**。
　→ ★**Aが通っても「穴ほどモデルの差が大きい」が確認されるだけで、張れる場所は増えない**。
　★**本件の価値は「形の確認」と「判定基準6の宿題を果たすこと」**。**運用の提案はしない**。

■ 予想（★**恒等式ではなく類推なので当てにしない**・判定基準24）
　⚠**(180)の単一分割では ρ が高そうに見える**が、**post-hocで見た形**。
　★**ウォークフォワードは学習量が増えるので、モデルが市場に近づいて差が縮む可能性がある**
　　（**(52)「AUCで最適化するとモデルは市場の写しになる」**の親戚）。**どちらでも驚かない**。

実行: python3 ml/audit_ana_wf.py [シード数(既定3)]   自己テスト: python3 ml/audit_ana_wf.py --selftest
"""
import math
import sys

import numpy as np

sys.path.insert(0, "ml")
import features as F
from audit_crosspool import load_races, payoff, zq
from audit_ana_odds import COST, MIN_HORSES, gate1, roi_of
from audit_ana_degree import DEG
from train_prod import CAPACITY, add_odds_features, fit_seeds

FIRST_YEAR = 2016
NPERM = 1000
NCMP = 2
ALPHA = 0.01
SEED = 20260905
FUKU_LINE = 80.0
WF_BOX4 = (82.2, 84.5)            # ★ウォークフォワードの陽性対照の範囲（(78)〜(77)）


def spearman_rank(v):
    """値 v と 1..k の順位相関。"""
    k = len(v)
    r = np.argsort(np.argsort(np.asarray(v, float))) + 1.0
    return float(np.corrcoef(np.arange(1.0, k + 1), r)[0, 1])


def stats_from(diffs):
    """帯ごとの差の配列リスト → (単調性ρ, 正の帯数, 帯ごとの平均)。"""
    mus = [float(np.mean(dd)) if len(dd) else 0.0 for dd in diffs]
    return spearman_rank(mus), int(sum(1 for m in mus if m > 0)), mus


def selftest():
    ok = True
    assert abs(spearman_rank([1, 2, 3, 4]) - 1.0) < 1e-9
    assert abs(spearman_rank([4, 3, 2, 1]) + 1.0) < 1e-9
    print("★単調性の自己テスト: 昇順→+1.000 / 降順→−1.000　★OK")
    # ★ゲート2: 符号を反転させると ρ は0のまわり、正の帯数は4.5のまわり
    rng = np.random.default_rng(0)
    diffs = [rng.normal(0, 300, 8000) for _ in range(len(DEG))]
    rs, bs = [], []
    for _ in range(300):
        pm = [dd * rng.choice([-1.0, 1.0], size=len(dd)) for dd in diffs]
        r, b, _ = stats_from(pm)
        rs.append(r); bs.append(b)
    mr, mb = float(np.mean(rs)), float(np.mean(bs))
    okg = abs(mr) < 0.10 and abs(mb - len(DEG) / 2) < 0.5
    print(f"★ゲート2の自己テスト: 符号反転300回 → ρ の平均 {mr:+.3f}（要 ≒0）/ "
          f"正の帯数の平均 {mb:.2f}（要 ≒{len(DEG)/2}）→ "
          f"**仮説が偽なら0と{len(DEG)/2}を返す**: {'★OK' if okg else '⚠NG'}")
    ok &= okg
    print(f"★比較数 {NCMP} → z = {zq(ALPHA/NCMP):.3f}　"
          f"／ ウォークフォワードの陽性対照 三連複BOX4 = {WF_BOX4[0]}〜{WF_BOX4[1]}%")
    print("★自己テスト: " + ("全部OK" if ok else "⚠NG"))
    return 0 if ok else 1


def main(nseed=3):
    print("(181) ★★★**(180)の曲線をウォークフォワードで測り直し、形を検定する**")
    print("★手続き: **各年Yについて Y年より前の全データで学習 → Y年を予測**（expanding window）")
    print(f"　**シード{nseed}本の平均**（判定基準1）。**train_prod.py の手続きそのもの**")
    print("⚠⚠**独立な標本ではない**——**2017年以降は(174)〜(180)と同じレース・同じ払戻**。")
    print("　★**新しいのは2016年だけ**。**言えるのは「手続きを実運用に揃えても残るか」まで**"
          "（判定基準35）\n")

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
    year = d["date"].dt.year.to_numpy()
    ys = [u for u in range(FIRST_YEAR, int(year.max()) + 1) if (year == u).sum() > 5000]
    print(f"\n評価年 {ys[0]}〜{ys[-1]}（{len(ys)}年）・全 {len(d):,}行")

    pred = np.full(len(d), np.nan)
    for u in ys:
        tr, te = year < u, year == u
        ms = fit_seeds(fx[tr], y[tr], nseed, PAR)
        pred[te] = np.mean([m.predict_proba(fx[te])[:, 1] for m in ms], axis=0)
        print(f"　{u}: 学習 {tr.sum():,} → 予測 {te.sum():,}")
    m = ~np.isnan(pred)
    sub = d.loc[m, ["raceid", "umaban", "odds", "date"]].copy()
    sub["p"] = pred[m]

    rng = np.random.default_rng(SEED)
    acc = {i: {"a": [], "r": [], "ao": [], "t1": [], "dt": []} for i in range(len(DEG))}
    box4, nrace = [], 0
    from itertools import combinations
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
        top1 = int(np.argmax(pv))
        mo = sorted(int(u) for u in ub[np.argsort(-pv, kind="mergesort")][:4])
        bx = [payoff(r, "三連複", list(c)) for c in combinations(mo, 3)]
        if not any(x is None for x in bx):
            box4.append(sum(bx) - 400.0)
        used = False
        for i, (nm, lo, hi) in enumerate(DEG):
            idx = np.where((od >= lo) & (od < hi))[0]
            if len(idx) < 2:
                continue
            a = int(idx[int(np.argmax(gap[idx]))])
            b = int(rng.choice([int(k) for k in idx if k != a]))
            pa = payoff(r, "複勝", [int(ub[a])])
            pb = payoff(r, "複勝", [int(ub[b])])
            if pa is None or pb is None:
                continue
            acc[i]["a"].append(pa); acc[i]["r"].append(pb)
            acc[i]["ao"].append(float(od[a])); acc[i]["t1"].append(a == top1)
            acc[i]["dt"].append(gg["date"].iloc[0])
            used = True
        nrace += used
    print(f"\n★突き合わせ {nrace:,}レース"
          f"　⚠**2017年以降は(174)〜(180)と同じレース**")

    box4 = np.asarray(box4, float)
    g0 = 100.0 * (box4.sum() + 400.0 * len(box4)) / (400.0 * len(box4))
    okb = WF_BOX4[0] - 1.0 <= g0 <= WF_BOX4[1] + 1.0
    print(f"⚠**陽性対照（ウォークフォワード）**: 三連複BOX上位4 **{g0:.1f}%** vs "
          f"想定 {WF_BOX4[0]}〜{WF_BOX4[1]}% → **{'★立った' if okb else '⚠⚠落ちた'}**")
    if not okb:
        print("\n⚠⚠**陽性対照が落ちた。読まない**（判定基準32）。")
        return

    print(f"\n{'='*100}")
    print("■ ★★曲線（**ウォークフォワード**・複勝1点）"
          "　⚠**単一分割の(180)と並べて読む**")
    print(f"{'穴の程度':<10}{'標本R':>9}{'穴の平均':>9}{'(a)水準':>10}{'乱の水準':>10}"
          f"{'★(b)差':>10}{'(180)の差':>11}{'モデル1位率':>12}")
    D180 = [2.7, 2.7, 4.4, 4.5, 6.0, 4.8, 5.1, 11.0, 17.9]
    diffs, lvls = [], []
    for i, (nm, lo, hi) in enumerate(DEG):
        A = np.asarray(acc[i]["a"], float)
        R = np.asarray(acc[i]["r"], float)
        diffs.append(A - R)
        lvls.append(roi_of(A) if len(A) else float("nan"))
        if len(A) < 300:
            print(f"{nm:<10}{len(A):>9,}　⚠標本不足")
            continue
        print(f"{nm:<10}{len(A):>9,}{np.mean(acc[i]['ao']):>8.1f}倍{roi_of(A):>9.1f}%"
              f"{roi_of(R):>9.1f}%{(A-R).mean():>+9.1f}円{D180[i]:>+10.1f}円"
              f"{100*np.mean(acc[i]['t1']):>11.1f}%")

    rho, npos, mus = stats_from(diffs)
    nr, nb = np.zeros(NPERM), np.zeros(NPERM)
    for t in range(NPERM):
        pm = [dd * rng.choice([-1.0, 1.0], size=len(dd)) if len(dd) else dd
              for dd in diffs]
        nr[t], nb[t], _ = stats_from(pm)
    tr_ = float(np.quantile(np.abs(nr), 0.99))
    tb_ = float(np.quantile(nb, 0.99))
    print(f"\n■ ★★★主判定（**帰無は対応のある符号反転{NPERM:,}回・完全な randomization test**）")
    print("　★ゲート2: **仮説が偽なら穴と乱は交換可能＝ρの期待値0・正の帯数の期待値4.5**")
    okA = abs(rho) > tr_
    okB = npos > tb_
    print(f"　**A 単調性 ρ = {rho:+.3f}**　帰無99%点 {tr_:+.3f}"
          f" → **{'★超えた' if okA else '⚠超えない'}**")
    print(f"　**B 正の帯数 = {npos}/{len(DEG)}**　帰無99%点 {tb_:.1f}"
          f" → **{'★超えた' if okB else '⚠超えない'}**")

    print(f"\n■ ★採用条件")
    print(f"　1. A と B の両方 … {'★満たす' if okA and okB else '⚠満たさない'}")
    same = (rho > 0) == (spearman_rank(D180) > 0)
    print(f"　2. 単一分割((180) ρ={spearman_rank(D180):+.3f})と同じ向き …"
          f" {'★満たす' if same else '⚠満たさない'}")
    over = [DEG[i][0] for i in range(len(DEG))
            if len(diffs[i]) >= 300 and mus[i] > 0 and lvls[i] > FUKU_LINE]
    print(f"　4. 差が正で水準が払戻率{FUKU_LINE}%超の帯 … **{len(over)}帯**"
          f"{'（' + ' / '.join(over) + '）' if over else ''}")

    if not (okA and okB):
        print("\n★★**結論: 主判定を通らない＝形は確認できない**。")
        print("★**(180)で見えた「差が程度とともに増える」は、実運用と同じ手続きでは残らない**。")
        print("⚠**経路は弱いので「閉じた」とは書けない**（判定基準25）。")
        return
    print(f"\n■ ★裾の検算（**差が最大の帯**・(77)）")
    i = int(np.argmax(mus))
    A = np.asarray(acc[i]["a"], float)
    dt = np.array(acc[i]["dt"], dtype="datetime64[D]").astype(int)
    yr = np.array(acc[i]["dt"], dtype="datetime64[Y]").astype(int) + 1970
    yy = sorted(set(yr))
    ov = sum(1 for u in yy if roi_of(A[yr == u]) > 100.0)
    print(f"　{DEG[i][0]}: ROI {roi_of(A):.1f}% / **上位3本が全払戻の "
          f"{100*np.sort(A)[-3:].sum()/max(A.sum(),1e-9):.1f}%** / "
          f"前半 {roi_of(A[dt<=np.median(dt)]):.1f}%・後半 {roi_of(A[dt>np.median(dt)]):.1f}% / "
          f"**100%超の年 {ov}/{len(yy)}**")
    print(f"\n★★**形は確認できた**。⚠**だが(180)と同じく、差が出る程度では水準が足りない**"
          if not over else "\n★★**形も水準も通った帯がある**")
    print("⚠**枠連の運用には触れない**。**設定変更は提案しない**。")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    n = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 3
    sys.exit(main(n) or 0)
