"""(205) ★★★**混合** — 「小さいズレは残差モデル、大きいズレは現行」は成り立つか

★★**動機（2026-09-06・(204)の◇post-hoc）**:
　★**λ=1.0 は小さいズレで現行を上回った**（**+9.1 vs +5.7円（ズレ0.030）/ +10.3 vs +8.1（0.050）**）。
　★**現行は大きいズレで上**（**+16.5円（0.186）vs +10.9円（0.239）**）。
　→ ★★**役割分担が成り立つなら、混ぜれば両方取れるはず**。
　⚠**(100)(102)で「容量の違うモデルの混合」は採用実績がある**（+0.0025）。
　　★**「容量の違う相手がいて初めて目標の違いが活きる」**——**今回は「市場への寄り方が違う相手」**。

⚠⚠**これはpost-hoc由来の仮説**。★**同じデータで同じマスを測り直しても検証にならない**。
　★★**だから (201) の設計を使う**——**前半で重みを選び、後半で1回だけ試す**。
　**そうすれば「post-hocで見つけた」ことによる下駄が、後半には持ち込まれない**。

────────────────────────────────────────────────────────────
★★★ 事前登録（2026-09-06・**結果を見る前にコミットする**）
────────────────────────────────────────────────────────────

■ ★経路（判定基準25）: **q = 各腕のモデル / q_pool = 複勝の板**。⚠**q側は弱い**。
■ ★手続き: **(204)の学習キャッシュ（`data/cache/form_pred.npz`）をそのまま使う**
　＝**ウォークフォワード・3シード。★再学習しないので腕Aは厳密に再現するはず**。

■ ★★★**混合の作り方**
　**pn_mix = w × pn_A + (1−w) × pn_E1**（**どちらもレース内で合計3に正規化済みなので、
　　混ぜても合計3のまま**）。**w は「現行モデルの重み」**。
| 腕 | w |
|---|---|
| ★**A 現行** | **1.00**（★**内部対照**） |
| **混合 0.75** | 0.75 |
| **混合 0.50** | 0.50 |
| **混合 0.25** | 0.25 |
| **E λ=1.0** | **0.00** |

■ ★★★家族A（**5比較**）: **全期間・帯[0.15,∞) の「軸 − 乱」対応差**（**(204)と同じ物差し**）
　⚠**これは記述に近い**（**post-hocのデータを含む**）。★**判定はするが、主判定ではない**。

■ ★★★家族B（**2比較・★主判定**）: ★**前半(2016-2020)で腕を選び、後半(2021-)で1回だけ試す**
　★**選ぶ基準は前半の対応差のみ**。**後半は1回だけ見る**。**① 本 ② プラセボ（乱）**。
　★★**「混合が本物なら、前半で選ばれた腕は後半でも現行を上回るはず」**。
| ★**後半で現行を上回る** | ⚠**上回らない** |
|---|---|
| **役割分担は本物**。**混合を採る** | ⚠**post-hocの下駄だった**。**現行のまま** |
　★**この表を先に書いておく**（判定基準42）。

■ ★★家族C（**記述**）: ★**対応差 vs 軸の平均ズレ の曲線**（**各腕 × 5帯**）。
　★**混合の曲線が現行の曲線より上に来るか**を目で確認する。

■ ★★ゲート2（判定基準42）——**仮説が偽なら何を返すか**
　★**モデルに残余情報が無いなら、どの腕でも対応差は0**（**乱は同じオッズ帯**）。
　★**w=1.00 は腕Aそのもの**——**(204)の +16.5円 / 97.1% / 3,677R を厳密に返すはず**。

■ ⚠ゲート1: (88)③④を再現（±3pt）／ ★ゲート板: 復元R 0.800±0.020 ／
　★陽性対照: 腕Aで三連複BOX上位4 81.9%±1.5pt。
■ ★★★内部対照（**決定的**）: **w=1.00 が (204) の 複勝ROI 97.1% ±0.2pt かつ 3,677R ±5**。
　⚠**(204)で内部対照を3回外している**（**母集団の取り違え**）。★**今回は同じ条件・同じ
　　キャッシュなので、厳密一致を要求する**。

■ ★★★探索を守る（**7比較・Bonferroni α=0.01/7**）
　⚠**標本300レース未満は判定しない**（判定基準5）。
　★**採用条件に「wについて単調」**——**孤立した1点は採らない**。

■ ★採用条件
　1. ★**家族Bで、前半に選ばれた腕の後半対応差が、後半の腕Aの対応差を明確に上回る**
　2. **本の縮み幅がプラセボの縮み幅より小さい**
　3. **家族Aで w について単調**
　4. **家族Cで混合の曲線が現行の曲線より上**

■ 予想（⚠**当てにしない**・判定基準24。★**私は(198)(201)(203)で3回連続外した**）
　★**混合は現行をわずかに上回るが、有意にはならないと見る**——
　　**(204)で差が出たのは小さいズレの帯で、そこは対応差自体が小さいから**。
　⚠**(100)(102)の混合の効果も +0.0025 と小さかった**。
　★**「後半で現行を上回らない」なら、post-hocの下駄だったと結論する**。

────────────────────────────────────────────────────────────
★★★★ 実測（2026-09-06）—— **混合は採らない。post-hocの下駄だった**
────────────────────────────────────────────────────────────

■ ゲート1 ／ ゲート板 0.8009 ／ 陽性対照 BOX4 81.9% ／
　★★★内部対照（決定的）: **w=1.00・帯[0.15,∞) = 97.1%（3,677R）** → **(204)と厳密一致**

■ ★★★★家族B（**主判定**）: ★**前半で選ばれたのは「現行そのもの」**
| 腕 | 前半R | 後半R | ★前半の差 | ★**後半の差** | 前半ROI | ★**後半ROI** | 乱後半 |
|---|---|---|---|---|---|---|---|
| ★**w=1.00（現行）** | 2,353 | 1,324 | ★**+16.2円** | **+17.1円** | 96.7% | 97.9% | 80.8% |
| **w=0.75** | 2,336 | 1,697 | +13.9円 | ★**+20.9円** | 98.0% | ★**99.2%** | 78.4% |
| w=0.50 | 2,801 | 2,492 | +14.4円 | +17.2円 | 96.7% | 97.1% | 79.9% |
| w=0.25 | 3,492 | 3,486 | +12.3円 | +13.3円 | 92.9% | 93.5% | 80.2% |
| w=0.00（残差） | 4,120 | 4,428 | +10.9円 | +10.9円 | 91.5% | 90.8% | 79.9% |
★★**前半で最良だったのは w=1.00（現行）**＝**混合は前半の時点で現行に負けている**。
→ **後半の判定は「現行 vs 現行」で差 +0.0円**。⚠**採用条件1は満たされない**。
⚠**w=0.75 の後半（+20.9円・ROI 99.2%）は目を引くが、前半では現行に2.3円負けている**
　＝**再現性がない**。★**後半を見てから選び直したら設計が無意味になるので拾わない**。

■ ★★★家族A（**全期間・post-hocのデータを含む**）
| 腕 | R数 | 買う率 | 軸オッズ | 平均ズレ | 複勝ROI | ★**対応差** |
|---|---|---|---|---|---|---|
| w=1.00 | 3,677 | 12.5% | 10.4倍 | 0.186 | 97.1% | **+16.5円** |
| ★**w=0.75** | 4,033 | 13.7% | 7.3倍 | 0.190 | ★**98.5%** | ★**+16.9円** |
| w=0.50 | 5,293 | 18.0% | 5.7倍 | 0.202 | 96.9% | +15.7円 |
| w=0.25 | 6,978 | 23.7% | 5.1倍 | 0.218 | 93.2% | +12.8円 |
| w=0.00 | 8,548 | 29.0% | 5.0倍 | 0.239 | 91.1% | +10.9円 |
★**5腕とも有意**。⚠**山型（w=0.75が頂点）で単調ではない**＝**採用条件3も満たさない**。

■ ★★家族C: **(204)と同じ形が再現した**
| 軸の平均ズレ | w=1.00（現行） | w=0.75 | w=0.00（残差） |
|---|---|---|---|
| 0.030 | +5.7円 | ★**+9.3円** | ★**+9.1円** |
| 0.050 | +8.1円 | +9.9円 | ★**+10.3円** |
| 0.078 | +9.2円 | +9.0円 | +8.4円 |
| 0.122 | +13.0円 | +13.9円 | +13.4円 |
| ★**最大帯** | ★**+16.5円**(0.186) | **+16.9円**(0.190) | +10.9円(0.239) |
★**小さいズレでは残差側が上、大きいズレでは現行が上**＝**役割分担の形自体は再現した**。
⚠**しかし混ぜても、前半→後半で再現する利得にはならなかった**。

■ ★★★結論（**採用条件 0/4・主判定で不合格**）
　★★**(204)の◇post-hoc は post-hoc の下駄だった**。⚠**混合は採らない**。
　★**この線で「モデルの形を変える」道は、残差学習(204)・混合(205)の両方で閉じた**。
　⚠**測っていない形はまだある**（**別目的の学習・容量の違う相手との混合など**）。
　　★**だが(100)(102)で測った混合の効果は +0.0025 と小さく、期待値は低い**。

■ ◇**post-hoc・未検定（拾わない）**: **w=0.75 の全期間 複勝ROI 98.5%**（買う率13.7%）
　＝**この線の複勝の最高値**（**従来の最高は(192)の97.6% / (203)の97.3%**）。
　⚠**前半で現行に負けているので採らない**。★**再開条件: 別の年で前半検証を繰り返し、
　　w=0.75 が2期以上で現行を上回ったとき**。


実行: python3 ml/audit_ana_mix.py    自己テスト: python3 ml/audit_ana_mix.py --selftest
"""
import math
import os
import sys
from itertools import combinations
from zlib import crc32

import numpy as np
import pandas as pd

sys.path.insert(0, "ml")
import features as F
from audit_crosspool import load_races, payoff, zq
from audit_ana_odds import BANDS, MIN_HORSES, band_of, gate1, roi_of
from audit_ana_marg import WF_BOX4, WF_TOL
from audit_ana_board import BOARD_R, BOARD_TOL, NPLACE, load_fuku_boards, qpool
from audit_ana_band import GBANDS, PN_FLOOR
from audit_ana_form import CACHE, EPS, NRAND, SEED
from train_prod import add_odds_features

WS = [1.00, 0.75, 0.50, 0.25, 0.00]
ARMS = [f"w={w:.2f}" for w in WS]
SPLIT = 2021
MINCELL = 300
KNOWN_ROI, KNOWN_N, ROI_TOL, N_TOL = 97.1, 3677, 0.2, 5
NCMP = len(WS) + 2          # 5 + 2
ALPHA = 0.01
TOPB = GBANDS[-1]


def selftest():
    ok = True
    z = zq(ALPHA / NCMP)
    print(f"★腕 {len(ARMS)}本（w＝現行モデルの重み）: " + " / ".join(ARMS))
    print(f"　**w=1.00 は現行そのもの（★内部対照） / w=0.00 は残差モデル λ=1.0**")
    print(f"★家族A {len(WS)}比較（全期間・帯[0.15,∞)）＋ ★家族B 2比較"
          f"（**前半で選び後半で1回・主判定**）→ **{NCMP}比較**・z = {z:.3f}")
    # ★混合が合計3を保つこと
    a = np.array([1.2, 0.9, 0.6, 0.3])
    b = np.array([0.4, 1.1, 1.0, 0.5])
    for w in WS:
        m = w * a + (1 - w) * b
        ok &= abs(m.sum() - 3.0) < 1e-9
    print(f"★混合が合計3を保つ: {'★OK（全wで）' if ok else '⚠NG'}")
    rng = np.random.default_rng(0)
    n = 120_000
    pay = rng.choice([0.0, 250.0, 900.0], size=n, p=[0.62, 0.30, 0.08])
    m2 = float(np.mean([(pay[rng.permutation(n)] - pay[rng.permutation(n)]).mean()
                        for _ in range(200)]))
    print(f"★ゲート2（対応差）: **偽なら0** → {m2:+.3f}円　"
          f"{'★OK' if abs(m2) < 2 else '⚠NG'}")
    ok &= abs(m2) < 2
    print(f"★★★内部対照（決定的）: **w=1.00 が (204)の 複勝ROI {KNOWN_ROI}% ±{ROI_TOL}pt "
          f"かつ {KNOWN_N:,}R ±{N_TOL}**")
    print("　⚠**(204)で内部対照を3回外している（母集団の取り違え）**。"
          "★**今回は同じキャッシュ・同じ条件なので厳密一致を要求する**")
    print("★★主判定の読み方: **後半で現行を上回る→役割分担は本物 / "
          "上回らない→post-hocの下駄だった**")
    print("★自己テスト: " + ("全部OK" if ok else "⚠NG"))
    return 0 if ok else 1


def main():
    z = zq(ALPHA / NCMP)
    print("(205) ★★★**混合** — 「小さいズレは残差モデル、大きいズレは現行」は成り立つか")
    print("⚠**post-hoc由来なので、★前半で選び後半で1回だけ試す**\n")

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
    fx0, _ = F.encode_categoricals(f)
    odds = d["odds"].to_numpy(float)
    rid_arr = d["raceid"].to_numpy()
    inv = 1.0 / odds
    mkt = inv / pd.Series(inv).groupby(rid_arr).transform("sum").to_numpy()
    pmk = np.clip(NPLACE * mkt, EPS, 1 - EPS)
    init = np.log(pmk / (1 - pmk))

    if not os.path.exists(CACHE):
        print(f"⚠⚠**学習キャッシュ {CACHE} が無い。(204)を先に走らせること**")
        return
    z0 = np.load(CACHE, allow_pickle=True)
    if int(z0["n"]) != len(d):
        print("⚠⚠**キャッシュの行数が合わない。読まない**")
        return
    print(f"★学習キャッシュを使う: {CACHE}")
    pA, mE = z0["pA"], z0["mE"]
    pE = 1.0 / (1.0 + np.exp(-(init + 1.0 * mE)))
    msk = ~np.isnan(pA) & ~np.isnan(mE)
    sub = d.loc[msk, ["raceid", "umaban", "odds", "date"]].copy()
    sub["pA"] = pA[msk]
    sub["pE"] = pE[msk]
    print(f"★予測できた行 **{msk.sum():,}**")

    K = {(a, b): {"fa": [], "fr": [], "gap": [], "od": [], "yr": []}
         for a in ARMS for b in GBANDS}
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
        if not np.isfinite(od).all() or (od <= 0).any():
            continue
        va = gg["pA"].to_numpy(float); ve = gg["pE"].to_numpy(float)
        if va.sum() <= 0 or ve.sum() <= 0:
            continue
        nall += 1
        qp, Rb = qpool([bd[int(u)] for u in ub], "harm")
        Rs.append(Rb)
        pnA = va / va.sum() * NPLACE
        pnE = ve / ve.sum() * NPLACE
        bx = [payoff(r, "三連複", list(c))
              for c in combinations(sorted(int(u) for u in ub[np.argsort(-va)[:4]]), 3)]
        if not any(x is None for x in bx):
            box4.append(sum(bx) - 400.0)
        bi = np.array([band_of(float(o), BANDS) for o in od])
        yr = int(gg["date"].iloc[0].year)
        picks = {}
        for sd in range(NRAND):
            picks[sd] = np.random.default_rng([SEED + sd, crc32(str(rid).encode())])
        for w, a in zip(WS, ARMS):
            pn = w * pnA + (1 - w) * pnE
            gap = pn - qp
            for b in GBANDS:
                lo, hi = b
                cand = np.where((pn >= PN_FLOOR) & (gap >= lo) & (gap < hi))[0]
                if not len(cand):
                    continue
                ai = int(cand[int(np.argmax(pn[cand]))])
                fa = payoff(r, "複勝", [int(ub[ai])])
                if fa is None:
                    continue
                vr, okall = [], True
                for sd in range(NRAND):
                    g2 = np.random.default_rng([SEED + sd, crc32(str(rid).encode())])
                    pl = [int(ub[q]) for q in range(len(ub))
                          if bi[q] == bi[ai] and q != ai]
                    if not pl:
                        okall = False
                        break
                    v = payoff(r, "複勝", [int(g2.choice(pl))])
                    if v is None:
                        okall = False
                        break
                    vr.append(v)
                if not okall:
                    continue
                c = K[(a, b)]
                c["fa"].append(fa); c["fr"].append(float(np.mean(vr)))
                c["gap"].append(float(gap[ai])); c["od"].append(float(od[ai]))
                c["yr"].append(yr)
    print(f"★対象 **{nall:,}レース**")

    med = float(np.median(Rs))
    okR = abs(med - BOARD_R) <= BOARD_TOL
    print(f"■ ★ゲート板: 復元R = **{med:.4f}** → **{'★立った' if okR else '⚠⚠落ちた'}**")
    box4 = np.asarray(box4, float)
    g0 = 100.0 * (box4.sum() + 400.0 * len(box4)) / (400.0 * len(box4))
    okb = abs(g0 - WF_BOX4) <= WF_TOL
    print(f"■ ⚠**陽性対照（w=1.00）**: BOX上位4 **{g0:.1f}%** vs {WF_BOX4}%"
          f" → **{'★立った' if okb else '⚠⚠落ちた'}**")
    c1 = K[("w=1.00", TOPB)]
    av = np.asarray(c1["fa"], float)
    okc = abs(roi_of(av) - KNOWN_ROI) <= ROI_TOL and abs(len(av) - KNOWN_N) <= N_TOL
    print(f"■ ★★★内部対照（決定的）: w=1.00・帯[0.15,∞) → **{roi_of(av):.1f}%**"
          f"（{len(av):,}R） vs (204) {KNOWN_ROI}%（{KNOWN_N:,}R）"
          f" → **{'★再現' if okc else '⚠⚠ズレた'}**")
    if not (okR and okb and okc):
        print("\n⚠⚠**ゲートが落ちた。読まない**（判定基準32）。")
        return

    print(f"\n{'='*112}")
    print("■ ★★★家族A: **全期間・帯[0.15,∞)**（⚠**post-hocのデータを含む。主判定ではない**）")
    print(f"\n{'腕':<10}{'R数':>8}{'買う率':>8}{'軸オッズ':>9}{'平均ズレ':>9}"
          f"{'複勝ROI':>9}{'乱':>8}{'★対応差':>10}{'99%CI':>19}{'判定':>12}")
    for a in ARMS:
        c = K[(a, TOPB)]
        av = np.asarray(c["fa"], float); rv = np.asarray(c["fr"], float)
        if len(av) < MINCELL:
            print(f"{a:<10}{len(av):>8,}　⚠**標本不足**")
            continue
        dd = av - rv
        mu, se = dd.mean(), dd.std(ddof=1) / math.sqrt(len(dd))
        print(f"{a:<10}{len(av):>8,}{100*len(av)/nall:>7.1f}%"
              f"{np.mean(c['od']):>8.1f}倍{np.mean(c['gap']):>9.3f}"
              f"{roi_of(av):>8.1f}%{roi_of(rv):>7.1f}%{mu:>+9.1f}円"
              f"{f'[{mu-z*se:+.1f},{mu+z*se:+.1f}]':>19}"
              f"{('★★有意' if mu - z*se > 0 else '⚠通らない'):>12}")

    print(f"\n{'='*112}")
    print("■ ★★★★家族B（**主判定**）: ★**前半で腕を選び、後半で1回だけ試す**")
    tbl = []
    for a in ARMS:
        c = K[(a, TOPB)]
        yr = np.asarray(c["yr"], int)
        h = yr < SPLIT
        if h.sum() < MINCELL or (~h).sum() < MINCELL:
            continue
        av = np.asarray(c["fa"], float); rv = np.asarray(c["fr"], float)
        dd = av - rv
        tbl.append({"a": a, "n1": int(h.sum()), "n2": int((~h).sum()),
                    "d1": dd[h].mean(), "d2": dd[~h].mean(),
                    "r1": 100 * rv[h].mean() / 100.0, "v2": dd[~h],
                    "roi1": roi_of(av[h]), "roi2": roi_of(av[~h]),
                    "rr1": roi_of(rv[h]), "rr2": roi_of(rv[~h])})
    print(f"\n{'腕':<10}{'前半R':>8}{'後半R':>8}{'★前半の差':>11}{'★後半の差':>11}"
          f"{'縮み':>9}{'前半ROI':>9}{'後半ROI':>9}{'乱後半':>9}")
    for x in tbl:
        print(f"{x['a']:<10}{x['n1']:>8,}{x['n2']:>8,}{x['d1']:>+10.1f}円"
              f"{x['d2']:>+10.1f}円{x['d1']-x['d2']:>+8.1f}円"
              f"{x['roi1']:>8.1f}%{x['roi2']:>8.1f}%{x['rr2']:>8.1f}%")
    base = [x for x in tbl if x["a"] == "w=1.00"]
    b2 = base[0]["d2"] if base else float("nan")
    best = max(tbl, key=lambda x: x["d1"])
    v = np.asarray(best["v2"], float)
    mu, se = v.mean(), v.std(ddof=1) / math.sqrt(len(v))
    print(f"\n★**本（モデル）**: 前半で選ばれた腕 **{best['a']}**（前半 {best['d1']:+.1f}円）")
    print(f"　★★**後半（1回だけ）**: **{mu:+.1f}円**　99%CI [{mu-z*se:+.1f},{mu+z*se:+.1f}]")
    print(f"　★★**後半の現行(w=1.00) は {b2:+.1f}円** → **差 {mu-b2:+.1f}円**"
          f"　→ **{'★★現行を上回った' if mu > b2 else '⚠上回らなかった'}**")
    print(f"　★**縮み {best['d1']-mu:+.1f}円** ／ **全腕の後半平均 "
          f"{np.mean([x['d2'] for x in tbl]):+.1f}円**")

    print(f"\n{'='*112}")
    print("■ ★★家族C（**記述**）: **対応差 vs 軸の平均ズレ の曲線**")
    print(f"\n{'帯':<14}" + "".join(f"{a:>20}" for a in ARMS))
    for b in GBANDS:
        lo, hi = b
        nm2 = f"[{lo:.2f},{hi:.2f})" if hi < 9 else f"[{lo:.2f},∞)"
        cells = []
        for a in ARMS:
            c = K[(a, b)]
            av = np.asarray(c["fa"], float); rv = np.asarray(c["fr"], float)
            if len(av) < MINCELL:
                cells.append(f"{'—':>20}")
                continue
            cells.append(f"{np.mean(c['gap']):>6.3f}→{(av-rv).mean():>+6.1f}円({len(av):>5,})")
        print(f"{nm2:<14}" + "".join(cells))

    print("\n■ ★採用条件: **1.後半で現行を上回る（主判定） / 2.縮みがプラセボより小 / "
          "3.wについて単調 / 4.曲線が上**")
    print("⚠**後半を見てから w を変えない**——**変えたらこの設計は無意味になる**。")
    print("\n⚠**枠連の運用には触れない**。**設定変更は提案しない**。")


if __name__ == "__main__":
    sys.exit(selftest() if "--selftest" in sys.argv else (main() or 0))
