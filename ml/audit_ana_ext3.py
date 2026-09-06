"""(193) ★★★**103.7%を正面から測る** — 紐の床の延長 / 必要標本 / **物差しを替える**

★★**動機（2026-09-06）**——**(192)で ◇post-hoc として置いた3つを事前登録に上げる**:
　★**「軸も紐もズレの床」で馬連 102.2 / 103.7%**、**2絞りとも単調**（床 0.00→0.02→0.04）。
　⚠**だが99%CIが ±22pt** で通らない。**対応差も [−6.8,+37.7] で0を跨ぐ**。
　★**(192)で書いた通り「測るには標本か、分散の小さい判定量が要る」**。**その両方をやる**。
　⚠**post-hocのマスなので、事前登録し直さないと読めない**（**(185)で犯した誤りの再発防止**）。

★★★**判定基準26**——**「データが足りない」と書く前に、物差しを替えられないか考える**。
　**(190)で三連単103.1%に「925年必要」と出した**。★**今回は先に必要標本を計算する**——
　**925年なら追わない。10年なら追う。測る前にそれが分かる**。

────────────────────────────────────────────────────────────
★★★ 事前登録（2026-09-06・**結果を見る前にコミットする**）
────────────────────────────────────────────────────────────

■ ★経路（判定基準25）: **q = MLモデル / q_pool = 複勝の板**（調和平均）。⚠**q 側は弱い**。
■ ★手続き: **ウォークフォワード**（`data/cache/wf_pred.npz` を再利用）。
■ ★母集団: **(192)家族Bと同一**——**絞り (0.15,0.02) / (0.20,0.06)**、
　**軸 = 条件を満たす馬のうち gap_f 最大**、**紐 = gap_f 降順で gap_f ≥ C の上位2頭**。
　⚠**紐が2頭に満たないレースは買わない**（**これも実質レース選別**・(192)と同じ扱い）。

■ ★★★家族A（**8比較**）: **紐の床の延長** C ∈ {0.04, 0.06, 0.08, 0.10} × 絞り2つ
　★**(192)は C=0.04 で打ち切った**。**単調が続くのか、標本が尽きるのかを見る**。
　★**主判定は2つ**: **(1) 馬連ROIの99%CI下端が100%超か**、
　　**(2) 現行(p降順・床なし)紐との対応差**（**軸が同じなので払戻が相関し、分散が下がる**）。

■ ★★★家族C（**8比較**）: ★★**物差しを替える**——**紐そのものを複勝で測る**
　★**機構の主張は「ズレ順は p降順より良い紐を選ぶ」**。
　★**それなら紐2頭を複勝で買っても差が出るはず**——**馬連の裾が要らないぶん分散が小さい**。
　**同じ8マスで、紐2頭の複勝ROI（1頭100円）を ズレ順 vs 現行 で対応差**。
　⚠**これは馬連の水準を測る道具ではない**。**「紐の選び方に情報があるか」だけを測る**。
　★★**家族Aが通らず家族Cが通れば、「機構はあるが馬連の裾では測れない」と確定できる**。

■ ★★家族B（**判定しない・記述**）: **必要標本の計算**
　**各マスで「99%CI下端が100%を超えるのに要るレース数」= (z·s/(ROI−100))²·n**、
　**それを実測のレース/年で割って年数に直す**。★**(190)の925年と同じ手続き**。

■ ★★ゲート2（判定基準42）——**仮説が偽なら何を返すか**
　★**家族A(1)**: **板が正しければ馬連ROIは払戻率77.5%を返す**（100%ではない）。
　★**家族A(2)・C**: **紐の選び方が損益について何も持たないなら、対応差の期待値は厳密に0**。
　★**家族C の水準**: **偽なら複勝の払戻率80.0%を返す**。
　・**点数もコストも軸も同一**＝**(170)の形にならない**。

■ ⚠ゲート1: (88)③④を再現（±3pt）／ ★ゲート板: 復元R 0.800±0.020 ／
　★陽性対照: 三連複BOX上位4 81.9%±1.5pt。
■ ★★内部対照: **家族A (0.20,0.06)・C=0.04 が(192)の103.7%を±1ptで再現**、
　**(0.15,0.02)・C=0.04 が(192)の102.2%を±1ptで再現**。

■ ★★★探索を守る（**合計16比較・Bonferroni α=0.01/16**）
　★**採用条件に「隣接と同じ向き」**——**孤立した1マスは採らない**（**(172)で崩れた形**）。
　⚠**(178)でプラセボが109.9%を出した**。**最良のマスを見出しにしない**（**(185)で訂正した誤り**）。
　⚠**標本300レース未満のマスは判定しない**（判定基準5）。

■ ★記述（判定しない）
　**各マスの 的中率 / 平均オッズ / 買うレース割合 / 上位3本の払戻占有 / 前後半 / 100%超の年数**。

■ ★採用条件
　1. **家族Aで99%CI下端が100%超、または対応差が有意に正**
　2. **家族Cで対応差が有意に正**（**機構の独立確認**）
　3. **隣接と同じ向き（単調が続く）** / 4. **裾の検算で符号が反転しない**

■ 予想（⚠**類推なので当てにしない**・判定基準24）
　★**家族A: 通らないと見る**——**床を上げるほど標本が減り、CIは広がる方が速い**。
　　⚠**(192)で 29,099 → 12,644 と半減している**。**C=0.10 では判定不能になる可能性が高い**。
　★**家族C: 差は出ると見る**——**(184)(189)(192)で軸の複勝は一貫して差を出している**。
　　⚠**ただし紐は軸より弱いはずなので、差は小さいと見る**。
　★**家族B: 必要年数は(190)の925年よりずっと小さいと見る**（**馬連は的中率が高い**）。

実行: python3 ml/audit_ana_ext3.py    自己テスト: python3 ml/audit_ana_ext3.py --selftest
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
from train_prod import add_odds_features

FILT = [(0.15, 0.02), (0.20, 0.06)]
CS = [0.04, 0.06, 0.08, 0.10]           # ★紐の床（(192)は0.04で打ち切った）
KNOWN192 = {(0.15, 0.02): 102.2, (0.20, 0.06): 103.7}
MINCELL = 300
NCMP = len(FILT) * len(CS) * 2          # 家族A 8 + 家族C 8
ALPHA = 0.01
UMAREN_LINE = 100.0 * LINE["馬連"]
FUKU_LINE = 100.0 * LINE["複勝"]


def need_years(v, roi, races_per_year):
    """★99%CI下端が100%を超えるのに要る標本と年数（(190)と同じ手続き）"""
    n = len(v)
    s = float(np.std(v, ddof=1))
    z = zq(ALPHA / NCMP)
    if roi <= 100.0:
        return None, None
    need = (z * s / (roi - 100.0)) ** 2
    return need, need / max(races_per_year, 1e-9)


def selftest():
    ok = True
    z = zq(ALPHA / NCMP)
    print(f"★家族A {len(FILT)*len(CS)}比較（馬連ROI + 現行との対応差・紐の床 {CS}）")
    print(f"★家族C {len(FILT)*len(CS)}比較（★**物差しを替える**: 紐2頭の複勝）")
    print(f"★合計 {NCMP}比較 → z = {z:.3f}")
    rng = np.random.default_rng(0)
    n = 150_000
    pay = rng.choice([0.0, 250.0, 4000.0], size=n, p=[0.90, 0.07, 0.03])
    m = float(np.mean([(pay[rng.permutation(n)] - pay[rng.permutation(n)]).mean()
                       for _ in range(200)]))
    print(f"★ゲート2（対応差）: 200回の平均 {m:+.3f}円 → **仮説が偽なら0**: "
          f"{'★OK' if abs(m) < 3 else '⚠NG'}")
    ok &= abs(m) < 3
    print(f"★ゲート2（水準）: **偽なら馬連は{UMAREN_LINE:.1f}% / 複勝は{FUKU_LINE:.1f}%を返す**")
    # 必要標本の式の自己テスト: ROIが100+z*se ちょうどなら need == n
    v = rng.normal(100.0, 300.0, 50_000)
    se = v.std(ddof=1) / math.sqrt(len(v))
    need, _ = need_years(v, 100.0 + z * se, 1.0)
    print(f"★必要標本の式: 下端ちょうどのとき need={need:,.0f} vs n={len(v):,} → "
          f"{'★OK' if abs(need-len(v))/len(v) < 0.01 else '⚠NG'}")
    ok &= abs(need - len(v)) / len(v) < 0.01
    print("★★家族Cの読み方: **Aが落ちてCが通れば「機構はあるが馬連の裾では測れない」**")
    print("★自己テスト: " + ("全部OK" if ok else "⚠NG"))
    return 0 if ok else 1


def main():
    z = zq(ALPHA / NCMP)
    print("(193) ★★★**103.7%を正面から測る**")
    print("★A: **紐の床を 0.04 → 0.10 まで延長**（**単調が続くか、標本が尽きるか**）")
    print("★B: ★**必要標本を先に計算する**（判定基準26・**925年なら追わない**）")
    print("★C: ★★**物差しを替える**——**紐そのものを複勝で測る**\n")

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

    K = {(fl, c): {"um1": [], "um0": [], "fk1": [], "fk0": [],
                   "hit1": [], "hit0": [], "od": [], "yr": [], "dt": []}
         for fl in FILT for c in CS}
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
        yr = int(gg["date"].iloc[0].year)
        for fl in FILT:
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
                h1, h0 = hg[:2], himo_p[:2]
                v1 = [payoff(r, "馬連", [axu, u]) for u in h1]
                v0 = [payoff(r, "馬連", [axu, u]) for u in h0]
                f1 = [payoff(r, "複勝", [u]) for u in h1]
                f0 = [payoff(r, "複勝", [u]) for u in h0]
                if any(x is None for x in v1 + v0 + f1 + f0):
                    continue
                c = K[(fl, cth)]
                c["um1"].append(sum(v1) / 2.0); c["um0"].append(sum(v0) / 2.0)
                c["fk1"].append(sum(f1) / 2.0); c["fk0"].append(sum(f0) / 2.0)
                c["hit1"].append(1.0 if max(v1) > 0 else 0.0)
                c["hit0"].append(1.0 if max(v0) > 0 else 0.0)
                c["od"].append(float(od[ai])); c["yr"].append(yr)
                c["dt"].append(gg["date"].iloc[0])
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

    print(f"\n■ ★★内部対照（**(192)家族Bの C=0.04 を再現するか**）")
    for fl in FILT:
        v = np.asarray(K[(fl, 0.04)]["um1"], float)
        k = KNOWN192[fl]
        print(f"　{str(fl):<16} C=0.04: {roi_of(v):.1f}% vs (192) {k:.1f}%"
              f"　差 {roi_of(v)-k:+.1f}pt　"
              f"{'★再現' if abs(roi_of(v)-k) <= 1.0 else '⚠ズレた'}"
              f"（{len(v):,}R）")

    print(f"\n{'='*118}")
    print(f"■ ★★★家族A: **紐の床の延長**（馬連・ズレ順2点）")
    print(f"　★ゲート2: **板が正しければROIは馬連の払戻率{UMAREN_LINE:.1f}%を返す**")
    print(f"\n{'絞り':<16}{'紐の床':>8}{'R数':>9}{'買う率':>8}{'ズレ順ROI':>11}{'99%CI':>21}"
          f"{'的中率':>8}{'現行ROI':>9}{'★対応差':>10}{'99%CI':>21}{'判定':>13}")
    passA, statA = [], {}
    for fl in FILT:
        for cth in CS:
            c = K[(fl, cth)]
            v1 = np.asarray(c["um1"], float); v0 = np.asarray(c["um0"], float)
            if len(v1) < MINCELL:
                print(f"{str(fl):<16}{cth:>8.2f}{len(v1):>9,}　⚠**標本不足で判定しない**")
                continue
            mu1, se1 = v1.mean() - COST, v1.std(ddof=1) / math.sqrt(len(v1))
            l1, h1 = 100 + mu1 - z * se1, 100 + mu1 + z * se1
            dd = v1 - v0
            mu2, se2 = dd.mean(), dd.std(ddof=1) / math.sqrt(len(dd))
            ok1, ok2 = l1 > 100.0, mu2 - z * se2 > 0
            statA[(fl, cth)] = (roi_of(v1), l1, mu2, ok1, ok2)
            if ok1 or ok2:
                passA.append((fl, cth, ok1, ok2, roi_of(v1)))
            tag = ("★★両方" if ok1 and ok2 else "★★水準" if ok1
                   else "★★差" if ok2 else "⚠通らない")
            print(f"{str(fl):<16}{cth:>8.2f}{len(v1):>9,}{100*len(v1)/nall:>7.1f}%"
                  f"{roi_of(v1):>10.1f}%{f'[{l1:.1f},{h1:.1f}]':>21}"
                  f"{100*np.mean(c['hit1']):>7.2f}%{roi_of(v0):>8.1f}%{mu2:>+9.1f}円"
                  f"{f'[{mu2-z*se2:+.1f},{mu2+z*se2:+.1f}]':>21}{tag:>13}")

    print(f"\n{'='*118}")
    print(f"■ ★★家族B: **必要標本**（**判定しない・記述**）——**{ALPHA}/{NCMP} の両側で下端を100%超にするには**")
    span = None
    for fl in FILT:
        c = K[(fl, CS[0])]
        yrs = sorted(set(c["yr"]))
        span = len(yrs)
    print(f"{'絞り':<16}{'紐の床':>8}{'ROI':>8}{'R数':>9}{'レース/年':>10}"
          f"{'必要R数':>14}{'★必要年数':>13}")
    for fl in FILT:
        for cth in CS:
            c = K[(fl, cth)]
            v1 = np.asarray(c["um1"], float)
            if len(v1) < MINCELL:
                continue
            yrs = sorted(set(c["yr"]))
            rpy = len(v1) / max(len(yrs), 1)
            roi = roi_of(v1)
            need, ny = need_years(v1, roi, rpy)
            if need is None:
                print(f"{str(fl):<16}{cth:>8.2f}{roi:>7.1f}%{len(v1):>9,}{rpy:>10,.0f}"
                      f"{'—':>14}{'⚠**100%未満なので到達しない**':>13}")
            else:
                print(f"{str(fl):<16}{cth:>8.2f}{roi:>7.1f}%{len(v1):>9,}{rpy:>10,.0f}"
                      f"{need:>14,.0f}{ny:>12,.0f}年")

    print(f"\n{'='*118}")
    print(f"■ ★★★家族C: ★**物差しを替える**——**紐2頭を複勝で買う**（1頭100円）")
    print(f"　★ゲート2: **偽なら複勝の払戻率{FUKU_LINE:.1f}%を返す** / **対応差の期待値は0**")
    print(f"\n{'絞り':<16}{'紐の床':>8}{'R数':>9}{'ズレ順複勝':>11}{'現行複勝':>10}"
          f"{'★対応差':>10}{'99%CI':>21}{'判定':>13}")
    passC = []
    for fl in FILT:
        for cth in CS:
            c = K[(fl, cth)]
            f1 = np.asarray(c["fk1"], float); f0 = np.asarray(c["fk0"], float)
            if len(f1) < MINCELL:
                continue
            dd = f1 - f0
            mu, se = dd.mean(), dd.std(ddof=1) / math.sqrt(len(dd))
            ok = mu - z * se > 0
            if ok:
                passC.append((fl, cth, mu))
            print(f"{str(fl):<16}{cth:>8.2f}{len(f1):>9,}{roi_of(f1):>10.1f}%"
                  f"{roi_of(f0):>9.1f}%{mu:>+9.1f}円"
                  f"{f'[{mu-z*se:+.1f},{mu+z*se:+.1f}]':>21}"
                  f"{'★★差がある' if ok else '⚠検出できない':>13}")

    print(f"\n■ ★採用条件")
    print(f"　1. 家族A: 通ったマス … **{len(passA)}/{len(FILT)*len(CS)}**")
    print(f"　2. 家族C: 差が有意なマス … **{len(passC)}/{len(FILT)*len(CS)}**")
    if passA:
        print(f"\n■ ★裾の検算（**通ったマス**・(77)）")
        for fl, cth, ok1, ok2, roi in passA:
            c = K[(fl, cth)]
            v1 = np.asarray(c["um1"], float)
            yr = np.asarray(c["yr"], int)
            dt = np.array(c["dt"], dtype="datetime64[D]").astype(int)
            ys = sorted(set(yr))
            ov = sum(1 for u in ys if roi_of(v1[yr == u]) > 100.0)
            print(f"　{fl} C={cth:.2f}: ROI {roi:.1f}% / "
                  f"**上位3本が全払戻の "
                  f"{100*np.sort(v1)[-3:].sum()/max(v1.sum(),1e-9):.1f}%** / "
                  f"前半 {roi_of(v1[dt<=np.median(dt)]):.1f}%・"
                  f"後半 {roi_of(v1[dt>np.median(dt)]):.1f}% / "
                  f"**100%超の年 {ov}/{len(ys)}**")
    if not passA and passC:
        print("\n★★★**機構はあるが、馬連の裾では水準を測れない**"
              "——**家族Cが「紐の選び方に情報がある」ことを別の物差しで示した**。")
    if not passA and not passC:
        print("\n★★★**どちらも通らない**——**この線はここで測定限界**。")
    print("⚠**経路は弱いので「閉じた」とは書けない**（判定基準25）。")
    print("\n⚠**枠連の運用には触れない**。**設定変更は提案しない**。")


if __name__ == "__main__":
    sys.exit(selftest() if "--selftest" in sys.argv else (main() or 0))
