"""(189) ★★★★**推奨度とズレの絶対閾値を組み合わせてレースを選別する**

★★**問い（ユーザー・2026-09-06）**——
　「**市場とのズレが何%以上とか。あと、いくら市場とのズレがあっても来ない馬は来ないから、
　　ある程度閾値は決めてレース選別した方がいい。
　　シェア何%以上の穴馬が市場と何%ズレている時に穴馬として、みたいな**」

★★★**これは本当に測っていない**——
　**(188)は推奨度とズレを「別々に」十分位で見ただけ**。★**両方の絶対閾値を同時に課す形は未測定**。
　**(174)は帯内の gap 上位10%＝相対的な分位**で、**絶対閾値でも推奨度の床でもない**。
　**(184)の絶対閾値(ratio≥1.25)は ズレだけ**で、**推奨度の床が無かった**
　　（**その結果1レース2.34頭も引っかかり、閾値が選別になっていなかった**）。

★★**機構（ユーザーの指摘・実測の裏づけあり）**——
　**gap が大きいのは「市場がその馬をとても安く見ている」から**で、**p 自体が小さければ来ない**。
　★**(188)の実測**: **推奨度の第1十分位は的中率21.1%**（**第10十分位は49.0%**）。
　→ ★**推奨度に床を置くと、そういう馬が落ちる**。**(184)で閾値が選別にならなかった理由もこれ**。

★★**オッズ帯の制限(5-20倍)を外す**——**ご提案の定義は帯を必要としない**ので、
　**(182)以降ずっと引きずっていた post-hoc な選択を1つ減らせる**。

────────────────────────────────────────────────────────────
★★★ 事前登録（2026-09-06・**結果を見る前にコミットする**）
────────────────────────────────────────────────────────────

■ ★経路（判定基準25）: **q = MLモデルの top3 確率 p_norm（Σ=3に正規化）**／
　**q_pool = 複勝の板**（調和平均・(184)で復元R=0.8009）。⚠**q 側は弱い**。
■ ★手続き: **ウォークフォワード**（`data/cache/wf_pred.npz` を再利用）。

■ ★★★選別の規則（**ユーザーの提案そのまま**）
```
各レースについて:
  条件を満たす馬 = { p_norm ≥ A  かつ  gap_f = p_norm − q_pool_fuku ≥ B }
  1頭も居なければ → ★そのレースは見送る（＝レース選別）
  居れば 穴 = その中で gap_f が最大の1頭
  対照 乱 = 穴と同じオッズ帯（9段）の別の1頭
  それぞれの複勝を100円ずつ
```
　★**p_norm は「3着以内に来る確率」そのもの**（**Σ=3 に正規化してあるので解釈できる**）。
　★**gap_f は「その確率が板より何pt高いか」**。

■ ★★格子（**16マス。これ以外に増やさない**）
　**推奨度の床 A**: **0.15 / 0.20 / 0.25 / 0.30**（＝**3着以内率 15% / 20% / 25% / 30% 以上**）
　**ズレの床 B**: **0.02 / 0.04 / 0.06 / 0.08**（＝**板より 2 / 4 / 6 / 8pt 高い**）

■ ★★★主判定（**32比較 = 16マス × 2統計・Bonferroni α=0.01/32・z=3.891**）
　★**A統計（水準）**: **穴の複勝ROIの99%CI下端が 100% を超えるか**
　★**B統計（差）**: **「穴 − 乱」の対応差が 0 を超えるか**
　⚠**マスの標本が300未満なら「判定不能」**（**判定基準5: 絞ると測れなくなる**）。

■ ★★ゲート2（判定基準42）——**仮説が偽なら何を返すか**
　★**A統計**: **仮説が偽（板が正しく値付けしている）なら、どのマスでもROIは払戻率80.0%を返す**
　　（**100%ではない**）。→ **100%超を問うのは、払戻率からの超過20.0ptを問うこと**。
　★**B統計**: **仮説が偽（条件を満たす馬が同じ帯の他の馬と交換可能）なら、対応差の期待値は0**。
　・**買う頭数もコストも同一**（**各1頭**）＝**(170)の形にならない**。
　・**穴側だけを落とさない**（**乱も同じレースで買う**）＝**(168)の形にならない**。

■ ★★★探索を守る（**ここを外すと必ず偽陽性が出る**）
　⚠**(178)でプラセボ（無作為な穴）が 109.9% を出した**（397レース）。
　⚠**(172)は24マス並べて跳ねたマス(92.0%)を追い、上位3本除去で82.1%に崩れた**。
　⚠**判定基準4: 総当たり6,206セルでは偶然だけで34.4セルが通過する**。
　→ ★**採用条件に「隣接するマスと同じ向き」を入れる**——**孤立した1マスは採らない**。
　→ ★**全16マスは記述として出すが、そこから最良を選んで結論にしない**。

■ ★★★最重要の対照（**(175)(188)で2回踏んだ形**）
　⚠**閾値を上げると、選ばれる馬が低オッズ側に寄る可能性がある**。
　★**各マスの平均オッズを必ず併記**。**ROIの上昇が平均オッズの低下で説明できるなら
　　(88)の再発見であって新しくない**。
　★★**ただし今回は「推奨度の床」が入っているので、そこが効いて交絡が切れるかもしれない**——
　　**それ自体が見どころ**。

■ ⚠ゲート1: (88)③④を再現（±3pt）／ ★ゲート板: 復元R 0.800±0.020 ／
　★陽性対照: 三連複BOX上位4 81.9%±1.5pt。

■ ★記述（判定しない・**「組み合わせ」への答え**）
　1. **全16マス**: **買うレース割合 / 穴ROI / 乱ROI / 差 / 的中率 / 的中時配当 /
　　　平均オッズ / モデル1位率 / 人気中央**。
　2. ★**(112)の軸E との組み合わせ**——**標本が残るマスで、軸E の中央値で2分割**して並べる。
　　 ⚠**(178)で「裾2%は397レースで検出力ゼロ」と実測済み**なので、**中央値分割にする**。

■ ★採用条件（判定基準39/40/41）
　1. **A統計またはB統計を通るマスがある**
　2. ★**そのマスが隣接マスと同じ向き**（**孤立したマスは採らない**）
　3. ★**平均オッズの低下で説明されない**
　4. **裾の検算で符号が反転しない**（上位3本・前後半・年別）

■ 予想（⚠**類推なので当てにしない**・判定基準24）
　★**閾値を上げるほど買うレースが減り、CIが広がって判定不能になる**と見る（**判定基準5**）。
　★**(184)の最良88.4%・(188)の最良97.5%（CI下端89.1）から、100%超は難しい**と見る。
　⚠**だが「推奨度の床」は今回が初めてで、そこが効く可能性は残っている**。**どちらでも驚かない**。

実行: python3 ml/audit_ana_grid.py    自己テスト: python3 ml/audit_ana_grid.py --selftest
"""
import math
import sys
from itertools import combinations

import numpy as np

sys.path.insert(0, "ml")
import features as F
import soft_axis as SA
from audit_crosspool import load_races, payoff, zq
from audit_ana_odds import BANDS, COST, MIN_HORSES, band_of, gate1, roi_of
from audit_ana_marg import WF_BOX4, WF_TOL, wf_predict
from audit_ana_board import BOARD_R, BOARD_TOL, NPLACE, load_fuku_boards, qpool
from train_prod import add_odds_features

AS = [0.15, 0.20, 0.25, 0.30]      # ★推奨度の床（3着以内率）
BS = [0.02, 0.04, 0.06, 0.08]      # ★ズレの床（板より何pt高いか）
MINCELL = 300
NCMP = len(AS) * len(BS) * 2       # 32
ALPHA = 0.01
SEED = 20260906
FUKU_LINE = 80.0


def selftest():
    ok = True
    print(f"★格子: 推奨度の床 {AS} × ズレの床 {BS} = {len(AS)*len(BS)}マス")
    print(f"★比較数 {NCMP}（16マス × 2統計）→ z = {zq(ALPHA/NCMP):.3f}")
    q, R = qpool([(2.0, 2.0)] * 12)
    assert abs(q.sum() - NPLACE) < 1e-9
    print(f"★板→含意の自己テスト: Σq={q.sum():.3f}（＝3着ぶん）　★OK")
    # ★p_norm は「3着以内に来る確率」として解釈できる
    pv = np.array([0.5, 0.3, 0.2, 0.1, 0.1] * 3)
    pn = pv / pv.sum() * NPLACE
    assert abs(pn.sum() - NPLACE) < 1e-9 and (pn <= 1.0 + 1e-9).all()
    print(f"★正規化の自己テスト: Σp_norm={pn.sum():.3f} / 最大 {pn.max():.3f}（≤1）　★OK")
    # ★ゲート2B: 交換可能な2頭の対応差は0
    rng = np.random.default_rng(0)
    n = 150_000
    pay = rng.choice([0.0, 250.0, 900.0], size=n, p=[0.72, 0.20, 0.08])
    m = float(np.mean([(pay[rng.permutation(n)] - pay[rng.permutation(n)]).mean()
                       for _ in range(200)]))
    print(f"★ゲート2Bの自己テスト: 対応差200回の平均 {m:+.3f}円 → "
          f"**仮説が偽なら0**: {'★OK' if abs(m) < 2 else '⚠NG'}")
    ok &= abs(m) < 2
    print(f"★ゲート2A: **板が正しければどのマスもROIは{FUKU_LINE}%を返す**（100%ではない）")
    print(f"★セルの最小標本 {MINCELL}（未満は「判定不能」・判定基準5）")
    print("★自己テスト: " + ("全部OK" if ok else "⚠NG"))
    return 0 if ok else 1


def main():
    z = zq(ALPHA / NCMP)
    print("(189) ★★★★**推奨度とズレの絶対閾値を組み合わせてレースを選別する**")
    print("★規則: **p_norm ≥ A かつ gap_f ≥ B の馬が居るレースだけ買う**")
    print("　**居なければ見送る**＝★**これがレース選別**")
    print("★**オッズ帯の制限(5-20倍)は外した**（ご提案の定義は帯を必要としない）\n")

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
    K = {(i, j): {"a": [], "r": [], "od": [], "t1": [], "pop": [],
                  "yr": [], "dt": [], "e": []}
         for i in range(len(AS)) for j in range(len(BS))}
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
        top1 = int(np.argmax(pv))
        order = [int(u) for u in ub[np.argsort(-pv, kind="mergesort")]]
        bx = [payoff(r, "三連複", list(c)) for c in combinations(sorted(order[:4]), 3)]
        if not any(x is None for x in bx):
            box4.append(sum(bx) - 400.0)
        bi = np.array([band_of(float(o), BANDS) for o in od])
        _k, e_ax, _q = SA.axis_expect([float(o) for o in od])
        pops = np.argsort(np.argsort(od)) + 1
        yr = int(gg["date"].iloc[0].year)
        for i, A in enumerate(AS):
            for j, B in enumerate(BS):
                cand = np.where((pn >= A) & (gapf >= B))[0]
                if not len(cand):
                    continue                     # ★このレースは見送り
                a = int(cand[int(np.argmax(gapf[cand]))])
                others = [int(k2) for k2 in range(len(ub))
                          if bi[k2] == bi[a] and k2 != a]
                if not others:
                    continue
                b = int(rng.choice(others))
                pa = payoff(r, "複勝", [int(ub[a])])
                pb = payoff(r, "複勝", [int(ub[b])])
                if pa is None or pb is None:
                    continue
                c = K[(i, j)]
                c["a"].append(pa); c["r"].append(pb); c["od"].append(float(od[a]))
                c["t1"].append(a == top1); c["pop"].append(int(pops[a]))
                c["yr"].append(yr); c["dt"].append(gg["date"].iloc[0])
                c["e"].append(float(e_ax) if e_ax is not None else np.nan)
    print(f"\n★対象 **{nall:,}レース**（板があり8頭以上）")

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

    print(f"\n{'='*112}")
    print("■ ★★記述: **全16マス**"
          "　⚠★**ここから最良を選んで結論にしない**（(178)でプラセボが109.9%）")
    print(f"{'推奨度≥':>8}{'ズレ≥':>7}{'買うR':>9}{'割合':>8}{'穴ROI':>9}{'乱ROI':>8}"
          f"{'差':>9}{'的中率':>8}{'配当':>9}{'平均オッズ':>11}{'モ1位率':>9}{'人気中央':>9}")
    for i, A in enumerate(AS):
        for j, B in enumerate(BS):
            c = K[(i, j)]
            n = len(c["a"])
            if n < MINCELL:
                print(f"{A:>8.2f}{B:>7.2f}{n:>9,}　⚠標本不足（<{MINCELL}）")
                continue
            av = np.asarray(c["a"], float); rv = np.asarray(c["r"], float)
            print(f"{A:>8.2f}{B:>7.2f}{n:>9,}{100*n/nall:>7.1f}%{roi_of(av):>8.1f}%"
                  f"{roi_of(rv):>7.1f}%{(av-rv).mean():>+8.1f}円"
                  f"{100*np.mean(av>0):>7.1f}%"
                  f"{(av[av>0].mean() if (av>0).any() else 0):>8,.0f}円"
                  f"{np.mean(c['od']):>10.1f}倍{100*np.mean(c['t1']):>8.1f}%"
                  f"{np.median(c['pop']):>8.0f}番")

    print(f"\n{'='*112}")
    print(f"■ ★★★主判定（**{NCMP}比較・α={ALPHA}/{NCMP}・z={z:.3f}**）")
    print(f"　★ゲート2: **A統計は「板が正しければ{FUKU_LINE}%を返す」／"
          f"B統計は「交換可能なら差は0」**")
    print(f"\n{'推奨度≥':>8}{'ズレ≥':>7}{'A: 穴ROI':>11}{'99%CI':>22}"
          f"{'B: 差':>10}{'99%CI':>22}{'判定':>16}")
    passA, passB = set(), set()
    for i, A in enumerate(AS):
        for j, B in enumerate(BS):
            c = K[(i, j)]
            n = len(c["a"])
            if n < MINCELL:
                continue
            av = np.asarray(c["a"], float); rv = np.asarray(c["r"], float)
            mu, se = av.mean() - COST, av.std(ddof=1) / math.sqrt(n)
            lo1, hi1 = 100 + mu - z * se, 100 + mu + z * se
            dd = av - rv
            md, sd = dd.mean(), dd.std(ddof=1) / math.sqrt(n)
            lo2, hi2 = md - z * sd, md + z * sd
            oa, ob = lo1 > 100.0, lo2 > 0.0
            if oa:
                passA.add((i, j))
            if ob:
                passB.add((i, j))
            tag = ("★★A通過" if oa and not ob else "★★B通過" if ob and not oa
                   else "★★★両方" if oa and ob else "⚠通らない")
            print(f"{A:>8.2f}{B:>7.2f}{roi_of(av):>10.1f}%"
                  f"{f'[{lo1:.1f},{hi1:.1f}]':>22}{md:>+9.1f}円"
                  f"{f'[{lo2:+.1f},{hi2:+.1f}]':>22}{tag:>16}")

    def adjacent(s, cell):
        i, j = cell
        return any((i + di, j + dj) in s
                   for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)))

    print(f"\n■ ★採用条件")
    hits = [(c, "A") for c in passA if adjacent(passA, c)] + \
           [(c, "B") for c in passB if adjacent(passB, c)]
    print(f"　1. 通るマス … A={len(passA)} / B={len(passB)}")
    print(f"　2. ★**隣接マスと同じ向き**（孤立は採らない）… **{len(hits)}マス**"
          f" → {'★満たす' if hits else '⚠満たさない'}")
    if passA or passB:
        iso = [c for c in (passA | passB) if not adjacent(passA, c) and not adjacent(passB, c)]
        if iso:
            print(f"　⚠**孤立して通ったマス {len(iso)}個は採らない**"
                  f"（**(172)で崩れた形**）: "
                  + " / ".join(f"({AS[i]:.2f},{BS[j]:.2f})" for i, j in iso))

    print(f"\n■ 記述: ★**(112)の軸E との組み合わせ**（**軸Eの中央値で2分割**）")
    print("　⚠**(178)で「裾2%は397レースで検出力ゼロ」と実測済みなので中央値分割にした**")
    print(f"{'推奨度≥':>8}{'ズレ≥':>7}{'軸E 小さい側':>15}{'軸E 大きい側':>15}{'R数':>9}")
    for i, A in enumerate(AS):
        for j, B in enumerate(BS):
            c = K[(i, j)]
            if len(c["a"]) < 2 * MINCELL:
                continue
            e = np.asarray(c["e"], float)
            av = np.asarray(c["a"], float)
            ok2 = np.isfinite(e)
            if ok2.sum() < 2 * MINCELL:
                continue
            m = np.median(e[ok2])
            lo_ = av[ok2 & (e <= m)]; hi_ = av[ok2 & (e > m)]
            print(f"{A:>8.2f}{B:>7.2f}{roi_of(lo_):>14.1f}%{roi_of(hi_):>14.1f}%"
                  f"{len(av):>9,}")

    if not hits:
        print("\n★★★**結論: 推奨度とズレの絶対閾値を組み合わせても届かない**。")
        print("★**(183)(188)に続く7件目としてレース選別を閉じる**。")
        print("⚠**経路は弱いので「閉じた」とは書けない**（判定基準25）。")
        return
    print(f"\n■ ★裾の検算（**隣接条件も満たしたマス**・(77)）")
    for (i, j), kind in hits:
        c = K[(i, j)]
        av = np.asarray(c["a"], float)
        yr = np.asarray(c["yr"], int)
        dt = np.array(c["dt"], dtype="datetime64[D]").astype(int)
        ys = sorted(set(yr))
        ov = sum(1 for u in ys if roi_of(av[yr == u]) > 100.0)
        print(f"　({AS[i]:.2f},{BS[j]:.2f})/{kind}: ROI {roi_of(av):.1f}% / "
              f"**上位3本が全払戻の {100*np.sort(av)[-3:].sum()/max(av.sum(),1e-9):.1f}%** / "
              f"前半 {roi_of(av[dt<=np.median(dt)]):.1f}%・"
              f"後半 {roi_of(av[dt>np.median(dt)]):.1f}% / "
              f"**100%超の年 {ov}/{len(ys)}**")
    print("\n⚠**枠連の運用には触れない**。**設定変更は提案しない**。")


if __name__ == "__main__":
    sys.exit(selftest() if "--selftest" in sys.argv else (main() or 0))
