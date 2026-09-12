"""(192) ★★★**(191)で見えた2つを詰める** — (A) 96.3%は交絡か / (B) 紐にもズレの床

★★**動機（2026-09-06）**——**(191)で2つ見えた**:
　★**A: ズレ床を延長すると 93.0 → 96.3%**（**この線の複勝の最高値**）。
　　⚠**だが平均オッズが 8.2 → 5.9倍 と下がる**——**(189)の格子内では横ばいだった**。
　　→ ★★**(175)(188)で2回落ちた「(88)の再発見」が戻ってきている疑い**。**そこを決める**。
　★**B: 紐もズレ順にすると馬連が +6.3円（89.0 → 95.3%）**。**券種でまっぷたつに割れる**
　　（三連複 −20.0 / 三連単 −35.9円）。**機構は「馬連は2頭とも来ればいい」**。
　　⚠**Bonferroniを通っていない**（post-hoc）。→ ★**事前登録し直し、さらに紐にも床を置く**。

────────────────────────────────────────────────────────────
★★★ 事前登録（2026-09-06・**結果を見る前にコミットする**）
────────────────────────────────────────────────────────────

■ ★経路（判定基準25）: **q = MLモデル / q_pool = 複勝の板**（調和平均）。⚠**q 側は弱い**。
■ ★手続き: **ウォークフォワード**（`data/cache/wf_pred.npz` を再利用）。

■ ★★★家族A（**10比較**）: **96.3%は交絡か本物か**
　★**対照は「乱 = 穴と同じオッズ帯の無作為な1頭」**——**この線で一貫して使ってきた形**。
　★★**オッズが揃っているので、差が残れば交絡ではない**。
　**推奨度の床 {0.15, 0.20} × ズレの床 {0.08, 0.10, 0.12, 0.15, 0.20}**、**複勝1点**。
　★**主判定: 各マスの「穴 − 乱」の対応差（円）**。
　★★**読み方を先に書く**:
| 差が伸びる | 差が縮む |
|---|---|
| ★**交絡ではない**。**ズレ床は本当に効いている** | ⚠**(88)の再発見**。**オッズが下がっただけ** |
　⚠**(189)の格子内では差が +9.3〜+15.0円 だった**。**そこからの動きを見る**。

■ ★★★家族B（**12比較**）: **馬連 × 紐もズレ順 ＋ 紐にも床**
　★**(191)で見えた +6.3円 を事前登録し直す**（**post-hocだったので**）。
　★**さらに紐にもズレの床 C を置く**——**軸と同じ発想を紐にも当てる**。
　**絞り2つ**（(0.15,0.02) / (0.20,0.06)）× **紐の床 C ∈ {0.00, 0.02, 0.04}** = **6マス**。
　**券種は馬連のみ**（★**(191)で三連複 −20.0 / 三連単 −35.9円 と大暴落しており、
　　「2頭とも来ればいい」券種でしか成立しないと機構が言っている**）。
　**紐 = ズレ順（gap_f 降順）で、gap_f ≥ C を満たす馬から2頭**。
　　⚠**足りなければそのレースは買わない**（**これも実質レース選別になる**）。
　★**主判定は2つ**: **(1) ROI の99%CI下端が100%超か**、**(2) 現行(p降順・床なし)との対応差**。

■ ★★ゲート2（判定基準42）——**仮説が偽なら何を返すか**
　★**家族A**: **仮説が偽（穴と乱が同じ帯で交換可能）なら、対応差の期待値は厳密に0**。
　★**家族B(2)**: **仮説が偽（紐の選び方が損益について何も持たない）なら、対応差の期待値は0**。
　★**家族B(1)**: **仮説が偽（板が正しい）なら、ROIは馬連の払戻率77.5%を返す**（100%ではない）。
　・**点数もコストも同一**＝**(170)の形にならない**。

■ ⚠ゲート1: (88)③④を再現（±3pt）／ ★ゲート板: 復元R 0.800±0.020 ／
　★陽性対照: 三連複BOX上位4 81.9%±1.5pt。
■ ★★内部対照:
　**家族Aの(0.20,0.08)の穴ROIが(191)の93.0%を±1ptで再現**（⚠**乱の要求で標本が減るので±2ptに緩める**）。
　**家族Bの (0.20,0.06)・C=0.00 が(191)の95.3%を±1ptで再現**。

■ ★★★探索を守る（**合計22比較・Bonferroni α=0.01/22・z=3.582**）
　★**採用条件に「隣接と同じ向き」**——**孤立した1マスは採らない**（**(172)で崩れた形**）。
　⚠**(178)でプラセボが109.9%を出した**。**最良のマスを見出しにしない**（**(185)で訂正した誤り**）。

■ ★記述（判定しない）
　**各マスの 乱ROI / 的中率 / 平均オッズ / 買うレース割合 / 平均オッズの差(穴−乱)**。
　★**穴と乱の平均オッズが揃っていることを毎マスで確認する**（**対照が機能している証拠**）。

■ ★採用条件
　1. **家族Aで差が縮まない**（**ズレ床を上げても対応差が維持または増加**）
　2. **家族BでROIの99%CI下端が100%超、または対応差が有意に正**
　3. **隣接と同じ向き** / 4. **裾の検算で符号が反転しない**

■ 予想（⚠**類推なので当てにしない**・判定基準24）
　★**家族A: 差は縮むと見る**——**平均オッズが下がるほど穴も乱も「良い側」に寄るから**。
　　⚠**だが(189)の格子内では横ばいだったので、確信はない**。
　★**家族B: 通らないと見る**（**95.3%で4.7pt足りない**）。⚠**紐の床がどう効くかは予想を持たない**。

────────────────────────────────────────────────────────────
★★★★ 実測（2026-09-06）—— **96.3%は交絡ではなかった。だが水準はまだ通らない**
────────────────────────────────────────────────────────────

■ ゲート1 4帯とも通過 ／ ゲート板 復元R 0.8009 ／ 陽性対照 BOX4 81.9% → 全部立った
■ ★★内部対照: **家族A (0.20,0.08) 93.6% vs (191) 93.0%（+0.6pt）／
　家族B (0.20,0.06) C=0.00 95.6% vs (191) 95.3%（+0.3pt）** → **両方★再現**

■ ★★★★家族A: **差は縮むどころか伸びた＝交絡ではない**
| 推奨度≥0.20 | ズレ≥0.08 | 0.10 | 0.12 | ★**0.15** | 0.20 |
|---|---|---|---|---|---|
| **穴ROI** | 93.6% | 94.6% | 94.9% | ★**97.6%** | 95.1% |
| **乱ROI** | 80.4% | 77.0% | 78.6% | 79.3% | 90.4% |
| ★**差** | +13.2円 | +17.6円 | +16.4円 | ★**+18.2円** | +4.7円 |
| **穴の平均オッズ** | 10.7倍 | 10.7 | 10.7 | **10.5倍** | 9.6倍 |
| **乱の平均オッズ** | 12.6倍 | 12.8 | 13.1 | **12.9倍** | 11.0倍 |
★★**差は (189)の +9.3〜+15.0円 から +11.5〜+18.2円 に伸びた**。**8/10マスがBonferroniを通る**。
→ ★**「(88)の再発見」ではなかった**。⚠**事前登録の予想（「差は縮む」）は外れた**（判定基準24）。
★★**穴の平均オッズは 10.7 → 10.5倍 でほぼ横ばい**——
　⚠**(191)で見た 8.2 → 5.9倍 の低下は、乱が取れないレースを含む母集団の違い**であって、
　**対照付きの母集団では起きていない**。★**交絡の疑いは晴れた**。
⚠**ズレ床0.20の2マスだけ検出できない**（**967レースしかない**・判定基準5）。

■ ★★★家族B: **紐にも床を置くと100%を超える。だが通らない**
| 絞り | 紐の床 | R数 | **ズレ順ROI** | 99%CI | 現行 | 対応差 |
|---|---|---|---|---|---|---|
| (0.15,0.02) | 0.00 | 29,099 | 94.9% | [80.0,109.9] | 89.6% | +5.3円 |
| (0.15,0.02) | 0.02 | 24,842 | 98.7% | [81.5,115.9] | 89.7% | +9.0円 |
| **(0.15,0.02)** | **0.04** | 12,644 | ★**102.2%** | [80.2,124.1] | 89.7% | **+12.4円** |
| (0.20,0.06) | 0.00 | 25,758 | 95.6% | [80.7,110.5] | 89.2% | +6.4円 |
| (0.20,0.06) | 0.02 | 21,895 | 99.4% | [82.2,116.6] | 88.6% | +10.8円 |
| ★**(0.20,0.06)** | ★**0.04** | 11,827 | ★★**103.7%** | [82.0,125.4] | 88.2% | ★**+15.4円** |
★★**紐にも床を置くと、両方の絞りで単調に上がり100%を超える**
　（94.9 → 98.7 → **102.2%** ／ 95.6 → 99.4 → **103.7%**）。
⚠**しかし6マスとも通らない**——**99%CIが ±22pt** と広く、**対応差も [−6.8,+37.7] で0を跨ぐ**。
★**現行(p降順)は 88.2〜89.7% で床にほとんど反応しない**＝**動いているのはズレ順のほうだけ**。

■ ★★結論
　★★**(191)の96.3%は交絡ではなかった**——**オッズを揃えた対照で差が +18.2円 まで伸びた**。
　★**紐にも床を置くと 103.7% まで上がる**が、**CIが広くて通らない**。
　⚠**この線は「機構は確立、水準は測定限界」という状態に来ている**。
　⚠**経路は弱いので「閉じた」とは書けない**（判定基準25）。

■ ◇**post-hoc・未検定（拾わない）**: ★**「軸も紐もズレの床」で 102.2 / 103.7%**（2絞りとも）。
　★**単調（床 0.00 → 0.02 → 0.04 で 94.9→98.7→102.2 / 95.6→99.4→103.7）**。
　⚠**CIが±22ptで通らない**。**測るには標本か、分散の小さい判定量が要る**。


実行: python3 ml/audit_ana_ext2.py    自己テスト: python3 ml/audit_ana_ext2.py --selftest
"""
import math
import sys
from itertools import combinations

import numpy as np

sys.path.insert(0, "ml")
import features as F
from audit_crosspool import LINE, load_races, payoff, zq
from audit_ana_odds import BANDS, COST, MIN_HORSES, band_of, gate1, roi_of
from audit_ana_marg import WF_BOX4, WF_TOL, wf_predict
from audit_ana_board import BOARD_R, BOARD_TOL, NPLACE, load_fuku_boards, qpool
from audit_ana_ext import A_EXT, B_EXT
from train_prod import add_odds_features

FILT_B = [(0.15, 0.02), (0.20, 0.06)]
CS = [0.00, 0.02, 0.04]                 # ★紐の床
MINCELL = 300
NCMP = len(A_EXT) * len(B_EXT) + len(FILT_B) * len(CS) * 2      # 10 + 12
ALPHA = 0.01
SEED = 20260906
UMAREN_LINE = 100.0 * LINE["馬連"]


def selftest():
    ok = True
    print(f"★家族A {len(A_EXT)*len(B_EXT)}比較（穴 − 乱 の対応差・ズレ床 {B_EXT}）")
    print(f"★家族B {len(FILT_B)*len(CS)*2}比較（馬連×紐ズレ順・紐の床 {CS}・ROIと対応差）")
    print(f"★合計 {NCMP}比較 → z = {zq(ALPHA/NCMP):.3f}")
    rng = np.random.default_rng(0)
    n = 150_000
    pay = rng.choice([0.0, 250.0, 900.0], size=n, p=[0.72, 0.20, 0.08])
    m = float(np.mean([(pay[rng.permutation(n)] - pay[rng.permutation(n)]).mean()
                       for _ in range(200)]))
    print(f"★ゲート2（対応差）: 200回の平均 {m:+.3f}円 → **仮説が偽なら0**: "
          f"{'★OK' if abs(m) < 2 else '⚠NG'}")
    ok &= abs(m) < 2
    print(f"★ゲート2（家族B水準）: **板が正しければ馬連のROIは払戻率{UMAREN_LINE:.1f}%を返す**")
    print("★★家族Aの読み方: **差が伸びれば交絡ではない / 差が縮めば(88)の再発見**")
    print("★自己テスト: " + ("全部OK" if ok else "⚠NG"))
    return 0 if ok else 1


def main():
    z = zq(ALPHA / NCMP)
    print("(192) ★★★**(191)で見えた2つを詰める**")
    print("★A: **96.3%は交絡か本物か**（**オッズを揃えた乱との対応差で決める**）")
    print("★B: **馬連×紐もズレ順**を事前登録し直し、**紐にも床**を置く\n")

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
    A = {(a, b): {"a": [], "r": [], "oa": [], "orr": []} for a in A_EXT for b in B_EXT}
    Bk = {(fl, c, w): [] for fl in FILT_B for c in CS for w in ("ズレ順", "現行")}
    Bm = {(fl, c): {"yr": [], "dt": [], "od": []} for fl in FILT_B for c in CS}
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
        order_p = [int(u) for u in ub[np.argsort(-pv, kind="mergesort")]]
        order_g = list(np.argsort(-gapf, kind="mergesort"))
        bx = [payoff(r, "三連複", list(c)) for c in combinations(sorted(order_p[:4]), 3)]
        if not any(x is None for x in bx):
            box4.append(sum(bx) - 400.0)
        bi = np.array([band_of(float(o), BANDS) for o in od])
        yr = int(gg["date"].iloc[0].year)
        # ── 家族A: 穴 vs 乱（同じ帯） ──
        for a in A_EXT:
            for b in B_EXT:
                cand = np.where((pn >= a) & (gapf >= b))[0]
                if not len(cand):
                    continue
                ai = int(cand[int(np.argmax(gapf[cand]))])
                others = [int(k) for k in range(len(ub)) if bi[k] == bi[ai] and k != ai]
                if not others:
                    continue
                ri = int(rng.choice(others))
                pa = payoff(r, "複勝", [int(ub[ai])])
                pr_ = payoff(r, "複勝", [int(ub[ri])])
                if pa is None or pr_ is None:
                    continue
                c = A[(a, b)]
                c["a"].append(pa); c["r"].append(pr_)
                c["oa"].append(float(od[ai])); c["orr"].append(float(od[ri]))
        # ── 家族B: 馬連 × 紐ズレ順 × 紐の床 ──
        for fl in FILT_B:
            cand = np.where((pn >= fl[0]) & (gapf >= fl[1]))[0]
            if not len(cand):
                continue
            ai = int(cand[int(np.argmax(gapf[cand]))])
            axu = int(ub[ai])
            himo_p = [u for u in order_p if u != axu]
            for cth in CS:
                hg = [int(ub[k]) for k in order_g
                      if int(ub[k]) != axu and gapf[k] >= cth]
                if len(hg) < 2 or len(himo_p) < 2:
                    continue
                v1 = [payoff(r, "馬連", [axu, hg[0]]), payoff(r, "馬連", [axu, hg[1]])]
                v0 = [payoff(r, "馬連", [axu, himo_p[0]]),
                      payoff(r, "馬連", [axu, himo_p[1]])]
                if any(x is None for x in v1 + v0):
                    continue
                Bk[(fl, cth, "ズレ順")].append(sum(v1) / 2.0)
                Bk[(fl, cth, "現行")].append(sum(v0) / 2.0)
                Bm[(fl, cth)]["yr"].append(yr)
                Bm[(fl, cth)]["dt"].append(gg["date"].iloc[0])
                Bm[(fl, cth)]["od"].append(float(od[ai]))
    print(f"\n★対象 **{nall:,}レース**")

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

    v = np.asarray(A[(0.20, 0.08)]["a"], float)
    print(f"\n■ ★★内部対照")
    print(f"　家族A (0.20,0.08) 穴ROI: {roi_of(v):.1f}% vs (191) 93.0%"
          f"　差 {roi_of(v)-93.0:+.1f}pt　"
          f"{'★再現' if abs(roi_of(v)-93.0) <= 2.0 else '⚠ズレた'}"
          f"（⚠**乱の要求で標本が {len(v):,} に減っている**）")
    vb = np.asarray(Bk[((0.20, 0.06), 0.00, "ズレ順")], float)
    print(f"　家族B (0.20,0.06) C=0.00 ズレ順: {roi_of(vb):.1f}% vs (191) 95.3%"
          f"　差 {roi_of(vb)-95.3:+.1f}pt　"
          f"{'★再現' if abs(roi_of(vb)-95.3) <= 1.0 else '⚠ズレた'}")

    print(f"\n{'='*104}")
    print(f"■ ★★★家族A: **96.3%は交絡か本物か**（**穴 − 乱 の対応差**・複勝1点）")
    print(f"　★★読み方: **差が伸びれば交絡ではない / 差が縮めば(88)の再発見**")
    print(f"　⚠**(189)の格子内では差が +9.3〜+15.0円 だった**")
    print(f"\n{'推奨度≥':>8}{'ズレ≥':>7}{'R数':>9}{'穴ROI':>9}{'乱ROI':>8}"
          f"{'★差':>9}{'99%CI':>21}{'穴オッズ':>10}{'乱オッズ':>10}{'判定':>14}")
    keepA = []
    for a in A_EXT:
        for b in B_EXT:
            c = A[(a, b)]
            av = np.asarray(c["a"], float); rv = np.asarray(c["r"], float)
            if len(av) < MINCELL:
                print(f"{a:>8.2f}{b:>7.2f}{len(av):>9,}　⚠標本不足")
                continue
            dd = av - rv
            mu, se = dd.mean(), dd.std(ddof=1) / math.sqrt(len(dd))
            sig = abs(mu) > z * se
            if sig and mu > 0:
                keepA.append((a, b, mu))
            print(f"{a:>8.2f}{b:>7.2f}{len(av):>9,}{roi_of(av):>8.1f}%{roi_of(rv):>7.1f}%"
                  f"{mu:>+8.1f}円{f'[{mu-z*se:+.1f},{mu+z*se:+.1f}]':>21}"
                  f"{np.mean(c['oa']):>9.1f}倍{np.mean(c['orr']):>9.1f}倍"
                  f"{'★差がある' if sig else '⚠検出できない':>14}")

    print(f"\n{'='*104}")
    print(f"■ ★★★家族B: **馬連 × 紐もズレ順 × 紐の床**")
    print(f"　★ゲート2: **板が正しければROIは馬連の払戻率{UMAREN_LINE:.1f}%を返す**")
    print(f"\n{'絞り':<16}{'紐の床':>8}{'R数':>9}{'ズレ順ROI':>11}{'99%CI':>21}"
          f"{'現行ROI':>10}{'★対応差':>10}{'99%CI':>21}{'判定':>14}")
    passB = []
    for fl in FILT_B:
        for cth in CS:
            v1 = np.asarray(Bk[(fl, cth, "ズレ順")], float)
            v0 = np.asarray(Bk[(fl, cth, "現行")], float)
            if len(v1) < MINCELL:
                continue
            mu1, se1 = v1.mean() - COST, v1.std(ddof=1) / math.sqrt(len(v1))
            l1, h1 = 100 + mu1 - z * se1, 100 + mu1 + z * se1
            dd = v1 - v0
            mu2, se2 = dd.mean(), dd.std(ddof=1) / math.sqrt(len(dd))
            ok1, ok2 = l1 > 100.0, mu2 - z * se2 > 0
            if ok1 or ok2:
                passB.append((fl, cth, ok1, ok2, roi_of(v1)))
            tag = ("★★両方" if ok1 and ok2 else "★★水準" if ok1
                   else "★★差" if ok2 else "⚠通らない")
            print(f"{str(fl):<16}{cth:>8.2f}{len(v1):>9,}{roi_of(v1):>10.1f}%"
                  f"{f'[{l1:.1f},{h1:.1f}]':>21}{roi_of(v0):>9.1f}%{mu2:>+9.1f}円"
                  f"{f'[{mu2-z*se2:+.1f},{mu2+z*se2:+.1f}]':>21}{tag:>14}")

    print(f"\n■ ★採用条件")
    print(f"　1. 家族A: 差が有意なマス … **{len(keepA)}/10**")
    if keepA:
        mus = [m for _, _, m in keepA]
        print(f"　　★**差の範囲 {min(mus):+.1f} 〜 {max(mus):+.1f}円**"
              f"（(189)の格子内は +9.3〜+15.0円）")
    print(f"　2. 家族B: 通ったマス … **{len(passB)}/6**")
    if not passB:
        print("\n★★★**結論: 家族Bは通らない**——**馬連×紐ズレ順は、紐に床を置いても100%に届かない**。")
    if not keepA:
        print("★★**家族Aで差が消えた＝96.3%は(88)の再発見だった**。")
    else:
        print("★★**家族Aで差が残った＝ズレ床の効果は交絡ではない**"
              "（⚠**ただし水準は100%に届いていない**）。")
    print("⚠**経路は弱いので「閉じた」とは書けない**（判定基準25）。")
    if passB:
        print(f"\n■ ★裾の検算（**通ったマス**・(77)）")
        for fl, cth, ok1, ok2, roi in passB:
            v1 = np.asarray(Bk[(fl, cth, "ズレ順")], float)
            yr = np.asarray(Bm[(fl, cth)]["yr"], int)
            dt = np.array(Bm[(fl, cth)]["dt"], dtype="datetime64[D]").astype(int)
            ys = sorted(set(yr))
            ov = sum(1 for u in ys if roi_of(v1[yr == u]) > 100.0)
            print(f"　{fl} C={cth:.2f}: ROI {roi:.1f}% / "
                  f"**上位3本が全払戻の "
                  f"{100*np.sort(v1)[-3:].sum()/max(v1.sum(),1e-9):.1f}%** / "
                  f"前半 {roi_of(v1[dt<=np.median(dt)]):.1f}%・"
                  f"後半 {roi_of(v1[dt>np.median(dt)]):.1f}% / "
                  f"**100%超の年 {ov}/{len(ys)}**")
    print("\n⚠**枠連の運用には触れない**。**設定変更は提案しない**。")


if __name__ == "__main__":
    sys.exit(selftest() if "--selftest" in sys.argv else (main() or 0))
