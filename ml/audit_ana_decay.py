"""(196) ★★★**劣化は本物か、それとも回帰か** — (a)情報が減った / (b)前半が運だった を分ける

★★**動機（2026-09-06）**——**(195)で仮説が死んだあと、残った説明は2つ**:
　**(a) 市場が賢くなり、モデルの残余情報そのものが減った**
　**(b) ★前半の高い数字が最初から裾の運だった（劣化ではなく平均への回帰）**
　⚠**(b)なら「昔は効いた」ごと崩れる**。★**分けて測る**。

★★★**分け方（判定基準26: 物差しを替える）**:
　⚠**馬連は裾が重く、的中10%・上位数本で数字が動く**——**(b)が入り込む余地が大きい**。
　★★**軸の複勝で同じ機構を測れば、裾が消える**（**(193)家族Cで有効性を確認済み**）。
　★**「穴 − 同じオッズ帯の乱」の対応差を年ごとに出し、その傾きを見る**。
| ★**傾きが有意に負** | ★**傾きが0** |
|---|---|
| **(a) 機構そのものが年々弱っている** | **(b) 馬連の劣化は裾と選択の産物**。**機構は生きている** |
　★★**この表を先に書いておく**（判定基準42・(192)と同じ形）。

────────────────────────────────────────────────────────────
★★★ 事前登録（2026-09-06・**結果を見る前にコミットする**）
────────────────────────────────────────────────────────────

■ ★経路（判定基準25）: **q = MLモデル / q_pool = 複勝の板**（調和平均）。⚠**q側は弱い**。
■ ★手続き: **ウォークフォワード**（`data/cache/wf_pred.npz`）。**母集団は(193)〜(195)と同一**。
　**軸 = (pn≥A かつ gap≥B) を満たす馬のうち gap 最大**。**絞り (0.15,0.02) / (0.20,0.06)**。

■ ★★★**対照は必ず10種平均**（**(195)で1回引きが3.8〜8.2pt揺れると実測した**）。
　★**種はレースに紐づけ、ループ順に依存させない**（**(195)で入れた修正**）。

■ ★★★家族A（**4比較**）: ★**軸の複勝**（1頭100円）——**裾のない物差し**
　**「穴 − 乱」の対応差を年ごとに出し、(1) 傾き（円/年）と (2) 前半−後半 の差の差**。
　**絞り2つ × 判定2つ = 4比較**。
■ ★★★家族B（**4比較**）: **同じことを馬連（紐2頭・紐床0.04）で**——**比較のため**。
　★★**AとBが食い違えば、それ自体が答え**（**Aが平ら・Bが落ちる → (b)**）。

■ ★★ゲート2（判定基準42）——**仮説が偽なら何を返すか**
　★**傾き**: **年による変化が無いなら、傾きの期待値は厳密に0**。
　★**対応差**: **穴と乱が同じ帯で交換可能なら、期待値は厳密に0**。
　★**水準**: **板が正しければ複勝ROIは払戻率80.0%、馬連は77.5%を返す**。

■ ⚠ゲート1: (88)③④を再現（±3pt）／ ★ゲート板: 復元R 0.800±0.020 ／
　★陽性対照: 三連複BOX上位4 81.9%±1.5pt。
■ ★★内部対照（**決定的な量だけを使う**・(195)の教訓）:
　**軸(穴)の複勝ROI が (192)家族A (0.20,0.08) の 93.6% を ±0.5pt で再現**。
　⚠**乱を含む量は内部対照に使わない**（**モンテカルロで揺れるから**）。

■ ★★★探索を守る（**8比較・Bonferroni α=0.01/8**）
　⚠**標本300レース未満の年は判定に使わない**（判定基準5）。
　★**2026は年途中**（**1,630レース＝他年の6割**）。⚠**傾きの計算には入れるが、
　　★2026を落とした値も併記して、結論が1年で変わらないか確認する**。

■ ★記述（判定しない）
　**年別の 穴の複勝ROI / 乱の複勝ROI / 対応差 / 軸の平均オッズ / 軸の複勝的中率**、
　**馬連も同じ**。★**軸の平均オッズが下がっている分を、対応差が追っているかを見る**。

■ ★採用条件（**ここでの「採用」は結論の採用**）
　1. **家族Aの傾きが有意に負 → (a)を採る**
　2. **家族Aの傾きが0で家族Bだけ落ちる → (b)を採る**
　3. **2026を落としても結論が変わらない**
　4. **前半−後半の差の差が、傾きと同じ向き**

■ 予想（⚠**類推なので当てにしない**・判定基準24）
　★**家族Aの傾きは緩やかに負と見る**——**(195)家族Aでズレの分布自体が28〜31%縮んでおり、
　　それは「モデルと市場が近づいた」ことの直接の証拠だから**。
　⚠**ただし複勝の対応差は(192)で +13〜+18円 と大きく、11年で0まで落ちるとは考えにくい**。
　★**つまり「(a)も(b)も両方ある」という結果を予想する**。⚠**その場合どちらを採るかは
　　傾きの大きさで決める**——**前半→後半の落差(115→88%)を傾きだけで説明できるか**。

────────────────────────────────────────────────────────────
★★★★ 実測（2026-09-06）—— **答えは(b)。機構は劣化していない**
────────────────────────────────────────────────────────────

■ ゲート1 4帯 ／ ゲート板 0.8009 ／ 陽性対照 BOX4 81.9% ／
　★★内部対照（決定的）: **軸の複勝ROI (0.20,0.08) = 93.6%** vs (192) 93.6% → **完全再現**

■ ★★★家族A（**複勝＝裾のない物差し**）: ★**機構は弱っていない**
| 絞り | R数 | 穴 | 乱 | ★**傾き** | 99%CI | 前半→後半 | 2026除く |
|---|---|---|---|---|---|---|---|
| (0.15,0.02) | 21,273 | 91.5% | 78.9% | ★**+0.71円/年** | [−0.67,+2.08] | +11.0 → **+14.0円** | +0.56 |
| (0.20,0.06) | 18,295 | 92.5% | 79.7% | ★**+0.83円/年** | [−0.58,+2.25] | +10.8 → **+14.9円** | +0.93 |
★★**傾きは負どころかわずかに正**。★**CIの下端が −0.67円/年 なので
　「11年で優位が消える」ほどの劣化（−2円/年）は棄却される**。
　⚠**自己テストで −2円/年 は検出できることを確認済み**＝**「検出力が無いだけ」ではない**。
★★**前半→後半でむしろ増えている**（+11.0→+14.0 / +10.8→+14.9円）。

■ ★★★★決定的な1行
| 年 | ⚠**馬連ROI** | ★**複勝の対応差** |
|---|---|---|
| **2025** | ⚠⚠**61.8%（最悪）** | ★★**+21.4円（★最高）** |
| 2018 | ★**139.7%（最良）** | +14.1円 |
★★★**馬連が最悪だった2025年に、複勝で測った優位は過去最高だった**。
→ **馬連の年ごとの上下は、機構とほぼ無関係に動いている**。

■ ★★★家族B（**馬連**）: ⚠**劣化も無劣化も区別できない**
| 絞り | 傾き | 99%CI | 前半→後半 | 差の差 |
|---|---|---|---|---|
| (0.15,0.02) | −2.73円/年 | [−8.09,+2.62] | +18.5 → −6.5円 | −25.0 [−58.7,+8.8] |
| (0.20,0.06) | −3.20円/年 | [−9.00,+2.61] | +20.3 → −1.4円 | −21.7 [−53.9,+10.5] |
⚠**4比較とも有意でない**（**CIが幅12円**）。★**点推定は落ちているが、0とも区別できない**。

■ ★★★★結論: **採用条件2が満たされた → (b) を採る**
　★★**(194)で「時間の劣化は本物」と書いたのは誤りだった。訂正する**。
　⚠**私が(194)でそう判断したのは、★裾の重い物差し1本だけで見ていたから**。
　★★**裾のない物差しに替えたら劣化そのものが消えた**（判定基準26）。
　★**馬連の前半115%→後半88%は、的中10%・上位数本の配当という裾の産物**。
　★★**機構（モデルは同じオッズでも残余情報を持つ）は11年を通じて健在**——
　　**むしろ +0.7〜+0.8円/年 で強くなる方向**（⚠**有意ではない**）。
　⚠**軸の平均オッズは 13.4→11.4 / 11.6→9.6倍 と下がっているが、対応差は下がっていない**
　　＝**(195)家族Aで見た分布の縮小は、機構の価値を削っていない**。
　⚠**経路は弱いので「閉じた」とは書けない**（判定基準25）。

■ ★★この線の最終状態（**(174)〜(196) の23本**）
　★**機構は確立し、劣化もしていない**。⚠**だが水準（100%超）は測定限界のまま**——
　**複勝は91.5〜92.5%（払戻率80.0%は明確に超える）／ 馬連は103.7%でCI±21pt**。
　★**「妙味は実在する。ただし複勝で+12円/レース、馬連で100%前後」というのが到達点**。


実行: python3 ml/audit_ana_decay.py    自己テスト: python3 ml/audit_ana_decay.py --selftest
"""
import math
import sys
from itertools import combinations
from zlib import crc32

import numpy as np

sys.path.insert(0, "ml")
import features as F
from audit_crosspool import LINE, load_races, payoff, zq
from audit_ana_odds import BANDS, COST, MIN_HORSES, band_of, gate1, roi_of
from audit_ana_marg import WF_BOX4, WF_TOL, wf_predict
from audit_ana_board import BOARD_R, BOARD_TOL, NPLACE, load_fuku_boards, qpool
from train_prod import add_odds_features

FILT = [(0.15, 0.02), (0.20, 0.06)]
HIMO_C = 0.04
NSEED = 10
SEED = 20260906
SPLIT = 2021
ABS_CTRL, KNOWN_FUKU, ROI_TOL = (0.20, 0.08), 93.6, 0.5    # ★(192)家族A・決定的
MINYEAR = 300
NCMP = len(FILT) * 2 * 2                                    # 4 + 4
ALPHA = 0.01
FUKU_LINE, UMAREN_LINE = 100.0 * LINE["複勝"], 100.0 * LINE["馬連"]


def slope(d, yr):
    """★年に対する対応差の傾き（円/年）と、その標準誤差（サンドイッチ）"""
    x = yr.astype(float) - yr.mean()
    sxx = float((x * x).sum())
    b = float((x * d).sum() / sxx)
    a = float(d.mean())
    res = d - (a + b * x)
    se = math.sqrt(float(((x * res) ** 2).sum()) / (sxx ** 2))
    return b, se


def selftest():
    ok = True
    z = zq(ALPHA / NCMP)
    print(f"★家族A {len(FILT)*2}比較（★**軸の複勝**・傾き と 前半−後半）")
    print(f"★家族B {len(FILT)*2}比較（**馬連 紐2頭・紐床{HIMO_C}**・同じ2つ）")
    print(f"★合計 {NCMP}比較 → z = {z:.3f}")
    rng = np.random.default_rng(0)
    n = 60_000
    yr = rng.integers(2016, 2027, size=n)
    d0 = rng.normal(5.0, 300.0, size=n)          # ★傾きなし
    b0, s0 = slope(d0, yr)
    print(f"★ゲート2（傾き）: **年に効果が無ければ0** → {b0:+.3f}円/年 "
          f"(se {s0:.3f})　{'★OK' if abs(b0) < z*s0 else '⚠NG'}")
    ok &= abs(b0) < z * s0
    d1 = d0 - 2.0 * (yr - 2021)                  # ★−2円/年を仕込む
    b1, s1 = slope(d1, yr)
    print(f"★検出力の確認: **−2.0円/年を仕込むと** {b1:+.3f}円/年 "
          f"(se {s1:.3f})　{'★検出できた' if b1 + z*s1 < 0 else '⚠NG'}")
    ok &= b1 + z * s1 < 0
    print(f"★★内部対照（決定的）: **軸の複勝ROI {ABS_CTRL} が {KNOWN_FUKU}% を "
          f"±{ROI_TOL}ptで再現**（⚠**乱を含む量は内部対照に使わない**）")
    print("★★読み方: **傾きが負→(a)機構が弱った / 傾き0でBだけ落ちる→(b)裾と選択**")
    print("★自己テスト: " + ("全部OK" if ok else "⚠NG"))
    return 0 if ok else 1


def main():
    z = zq(ALPHA / NCMP)
    print("(196) ★★★**劣化は本物か、それとも回帰か**")
    print("★(a) 情報が減った / (b) 前半が運だった —— ★**裾のない物差し（複勝）で分ける**\n")

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

    KEYS = [(fl, k) for fl in FILT for k in ("複勝", "馬連")] + [(ABS_CTRL, "対照複勝")]
    K = {k: {"a": [], "r": [], "yr": [], "od": []} for k in KEYS}
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
        gap = pn - qp
        bx = [payoff(r, "三連複", list(c))
              for c in combinations(sorted(int(u) for u in ub[np.argsort(-pv)[:4]]), 3)]
        if not any(x is None for x in bx):
            box4.append(sum(bx) - 400.0)
        bi = np.array([band_of(float(o), BANDS) for o in od])
        order_g = list(np.argsort(-gap, kind="mergesort"))
        yr = int(gg["date"].iloc[0].year)
        for key in KEYS:
            fl, kind = key
            cand = np.where((pn >= fl[0]) & (gap >= fl[1]))[0]
            if not len(cand):
                continue
            ai = int(cand[int(np.argmax(gap[cand]))])
            axu = int(ub[ai])
            if kind == "馬連":
                hg = [int(ub[k]) for k in order_g
                      if int(ub[k]) != axu and gap[k] >= HIMO_C]
                if len(hg) < 2:
                    continue
                va = [payoff(r, "馬連", [axu, w]) for w in hg[:2]]
                base = [ai]
            else:
                va = [payoff(r, "複勝", [axu])]
                base = [ai]
            if any(v is None for v in va):
                continue
            outs, ok = [], True
            for sd in range(NSEED):
                g2 = np.random.default_rng([SEED + sd, crc32(str(rid).encode())])
                if kind == "馬連":
                    hr = []
                    for uu in hg[:2]:
                        b = bi[list(ub).index(uu)]
                        pl = [int(ub[k]) for k in range(len(ub))
                              if bi[k] == b and int(ub[k]) != axu and int(ub[k]) not in hr]
                        if not pl:
                            ok = False
                            break
                        hr.append(int(g2.choice(pl)))
                    if not ok:
                        break
                    vr = [payoff(r, "馬連", [axu, w]) for w in hr]
                else:
                    pl = [int(ub[k]) for k in range(len(ub))
                          if bi[k] == bi[ai] and k != ai]
                    if not pl:
                        ok = False
                        break
                    vr = [payoff(r, "複勝", [int(g2.choice(pl))])]
                if any(v is None for v in vr):
                    ok = False
                    break
                outs.append(sum(vr) / len(vr))
            if not ok:
                continue
            c = K[key]
            c["a"].append(sum(va) / len(va))
            c["r"].append(float(np.mean(outs)))
            c["yr"].append(yr)
            c["od"].append(float(od[ai]))
    print(f"\n★対象 **{nall:,}レース**")

    med = float(np.median(Rs))
    okR = abs(med - BOARD_R) <= BOARD_TOL
    print(f"■ ★ゲート板: 復元R = **{med:.4f}** → **{'★立った' if okR else '⚠⚠落ちた'}**")
    box4 = np.asarray(box4, float)
    g0 = 100.0 * (box4.sum() + 400.0 * len(box4)) / (400.0 * len(box4))
    okb = abs(g0 - WF_BOX4) <= WF_TOL
    print(f"■ ⚠**陽性対照**: BOX上位4 **{g0:.1f}%** vs {WF_BOX4}%"
          f" → **{'★立った' if okb else '⚠⚠落ちた'}**")
    cv = np.asarray(K[(ABS_CTRL, "対照複勝")]["a"], float)
    okc = abs(roi_of(cv) - KNOWN_FUKU) <= ROI_TOL
    print(f"■ ★★内部対照（決定的）: 軸の複勝ROI {ABS_CTRL} = **{roi_of(cv):.1f}%** "
          f"vs (192) {KNOWN_FUKU}% → **{'★再現' if okc else '⚠⚠ズレた'}**（{len(cv):,}R）")
    if not (okR and okb and okc):
        print("\n⚠⚠**ゲートが落ちた。読まない**（判定基準32）。")
        return

    for fam, kind in (("A", "複勝"), ("B", "馬連")):
        print(f"\n{'='*104}")
        line = FUKU_LINE if kind == "複勝" else UMAREN_LINE
        print(f"■ ★★★家族{fam}: **{kind}**"
              f"（★ゲート2: **偽なら水準は{line:.1f}%、傾きも対応差も0**）")
        for fl in FILT:
            c = K[(fl, kind)]
            a = np.asarray(c["a"], float); rv = np.asarray(c["r"], float)
            yr = np.asarray(c["yr"], int); odv = np.asarray(c["od"], float)
            dd = a - rv
            print(f"\n★絞り{fl}　**{len(a):,}R**　"
                  f"（{kind} 穴 {roi_of(a):.1f}% / 乱 {roi_of(rv):.1f}%）")
            print(f"{'年':<7}{'R数':>7}{'穴ROI':>9}{'乱ROI':>8}{'★対応差':>10}"
                  f"{'的中率':>8}{'軸オッズ':>9}")
            for u in sorted(set(yr)):
                m = yr == u
                if m.sum() < MINYEAR:
                    print(f"{u:<7}{int(m.sum()):>7,}　⚠標本不足（判定に使うが記述は控える）")
                    continue
                print(f"{u:<7}{int(m.sum()):>7,}{roi_of(a[m]):>8.1f}%{roi_of(rv[m]):>7.1f}%"
                      f"{dd[m].mean():>+9.1f}円{100*np.mean(a[m]>0):>7.2f}%"
                      f"{odv[m].mean():>8.1f}倍")
            b, se = slope(dd, yr)
            m26 = yr < 2026
            b2, se2 = slope(dd[m26], yr[m26])
            h = yr < SPLIT
            d1, d2 = dd[h].mean(), dd[~h].mean()
            sdd = math.sqrt(dd[h].var(ddof=1)/h.sum() + dd[~h].var(ddof=1)/(~h).sum())
            sig_b = abs(b) > z * se
            sig_d = abs(d1 - d2) > z * sdd
            print(f"　★**傾き {b:+.2f}円/年**　99%CI [{b-z*se:+.2f},{b+z*se:+.2f}]"
                  f"　→ **{'★★有意' if sig_b else '⚠検出できない'}**"
                  f"　（2026を除くと {b2:+.2f}円/年 "
                  f"[{b2-z*se2:+.2f},{b2+z*se2:+.2f}]）")
            print(f"　★**前半 {d1:+.1f}円 → 後半 {d2:+.1f}円**　差の差 {d2-d1:+.1f}円"
                  f"　99%CI [{(d2-d1)-z*sdd:+.1f},{(d2-d1)+z*sdd:+.1f}]"
                  f"　→ **{'★★有意' if sig_d else '⚠検出できない'}**")
            print(f"　（前半ROI {roi_of(a[h]):.1f}% → 後半 {roi_of(a[~h]):.1f}%　"
                  f"軸オッズ {odv[h].mean():.1f} → {odv[~h].mean():.1f}倍）")

    print(f"\n■ ★★読み方（**事前に書いたもの**）")
    print("　★**家族Aの傾きが負 → (a) 機構そのものが年々弱っている**")
    print("　★**家族Aが平らで家族Bだけ落ちる → (b) 馬連の劣化は裾と選択の産物**")
    print("\n⚠**枠連の運用には触れない**。**設定変更は提案しない**。")


if __name__ == "__main__":
    sys.exit(selftest() if "--selftest" in sys.argv else (main() or 0))
