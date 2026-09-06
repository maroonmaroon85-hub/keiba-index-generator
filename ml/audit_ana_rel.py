"""(195) ★★★**劣化の原因を測る** — 絶対閾値 vs ★**相対閾値（その年までの分布の上位◯%）**

★★**動機（2026-09-06）**——**(194)追記で劣化が本物だと分かった**:
　⚠**ズレ順 前半115.0% → 後半88.0%**。**対照は横ばい（乱 81.8→83.4 / 現行 89.4→86.6）**
　＝**市場やデータ全体の変化ではなく、★この買い方に固有の劣化**。**優位性が +33.2 → +4.6円**。
　★**手がかり**: **同じルールなのに軸の平均オッズが 10.1 → 8.2倍、的中率が 9.44 → 12.25%**
　　＝**選ばれる馬が「穴」でなくなってきている**。

★★★**仮説（判定基準41: 分解して真の原因を探す）**:
　**ウォークフォワードは年々学習データが増え、モデルの精度が上がる
　　→ 市場との食い違い（ズレ）の分布そのものが縮む
　　→ 固定した絶対閾値（pn≥0.20, gap≥0.06, 紐床≥0.04）が拾う馬が年ごとに変わる
　　→ 後半は「ズレが大きい穴馬」ではなく「単に上位の人気馬」を拾っている**
★**もしそうなら、閾値を「その年までの分布の上位◯%」にすれば劣化は止まるはず**。

────────────────────────────────────────────────────────────
★★★ 事前登録（2026-09-06・**結果を見る前にコミットする**）
────────────────────────────────────────────────────────────

■ ★経路（判定基準25）: **q = MLモデル / q_pool = 複勝の板**（調和平均）。⚠**q側は弱い**。
■ ★手続き: **ウォークフォワード**（`data/cache/wf_pred.npz`）。**券種は馬連・紐2頭（2点）**。
■ ★★★**先読みを絶対に入れない**: **年Yの閾値は、Y未満の年だけをプールした分位から作る**。
　⚠**予測そのものもウォークフォワード**なので、**分位の推定にも未来は入らない**。
　⚠**2016は前年が無いので相対腕から外す**。★**比較は2017〜2026で両腕そろえる**。

■ ★★★家族A（**判定しない・記述**）: **仮説そのものを直接見る**
　**年ごとに、全馬の gap(=pn−qp) の 90/95/98分位、pn の 80/90分位、
　　絶対閾値(0.20,0.06)を満たす馬の割合、その馬の平均オッズ**。
　★★**分布が縮んでいれば仮説は生きる。横ばいなら仮説は死ぬ**（**先に書いておく**）。

■ ★★★家族B（**24比較**）: **相対閾値にすると劣化が止まるか**
　**軸: gap ≥ その年までの上位 {2, 5, 10}% ／ pn ≥ 上位 {10, 20}%**
　**紐の床: gap ≥ 上位 {10, 20}%** → **3×2×2 = 12マス**。
　★**主判定は2つ**: **(1) 全期間の「ズレ順 − 乱」対応差**、**(2) ★後半(2021〜)の同じ対応差**。
　★★**(2)が本命**——**劣化を直すのが目的だから**。

■ ★★ゲート2（判定基準42）——**仮説が偽なら何を返すか**
　★**乱は「軸と同じオッズ帯の無作為2頭」**＝**軸もコストも点数も同一**。
　★**紐の選び方が損益について何も持たないなら、対応差の期待値は厳密に0**。

■ ⚠ゲート1: (88)③④を再現（±3pt）／ ★ゲート板: 復元R 0.800±0.020 ／
　★陽性対照: 三連複BOX上位4 81.9%±1.5pt。
■ ★★内部対照: **絶対閾値(0.20,0.06)・床0.04 の腕を同じコードで走らせ、
　(194)の 前半+33.2円 / 後半+4.6円 を ±3円 で再現する**。★**これが立たなければ読まない**。

■ ★★★探索を守る（**24比較・Bonferroni α=0.01/24**）
　⚠**標本300レース未満のマスは判定しない**（判定基準5）。
　★**採用条件に「隣接と同じ向き」**——**孤立した1マスは採らない**。
　⚠**(194)で分かった通り紐の順序はほとんど効かない**。**ここで振るのは閾値だけ**。

■ ★採用条件
　1. **★後半の「ズレ順−乱」対応差が、絶対閾値版の +4.6円 を明確に上回る**
　2. **それが複数マスで起き、隣接と同じ向き**
　3. **前半も壊れていない**（**前半だけ良くして後半を捨てた形になっていない**）
　4. **家族Aで分布の縮小が実際に見えている**（**機構の裏づけ**）

■ 予想（⚠**類推なので当てにしない**・判定基準24）
　★**家族A: 分布は縮んでいると見る**（**軸オッズ 10.1→8.2倍 がそれを示唆している**）。
　⚠**家族B: 半分だけ直ると見る**——**分布の縮小が原因の一部でしかない可能性が高い**。
　　⚠**「市場が賢くなった」「モデルの残余情報が減った」なら相対化しても直らない**。
　★**もし家族Aで分布が縮んでいないなら、仮説は即死**。**その時は素直にそう書く**。

実行: python3 ml/audit_ana_rel.py    自己テスト: python3 ml/audit_ana_rel.py --selftest
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

QG = [98.0, 95.0, 90.0]        # ★軸のズレ: 上位2 / 5 / 10%
QP = [90.0, 80.0]              # ★軸の推奨度: 上位10 / 20%
QC = [90.0, 80.0]              # ★紐の床: ズレ上位10 / 20%
ABS_FILT, ABS_C = (0.20, 0.06), 0.04       # ★内部対照（(194)と同じ絶対閾値）
KNOWN_H1, KNOWN_H2, KTOL = 33.2, 4.6, 3.0
SPLIT = 2021                   # ★後半の定義（(194)と同じ）
MINCELL = 300
NCMP = len(QG) * len(QP) * len(QC) * 2     # 12マス × (全期間 / 後半)
ALPHA = 0.01
SEED = 20260906
UMAREN_LINE = 100.0 * LINE["馬連"]


def selftest():
    ok = True
    z = zq(ALPHA / NCMP)
    print(f"★家族B {len(QG)*len(QP)*len(QC)}マス"
          f"（軸ズレ上位{QG}% × 軸推奨度上位{QP}% × 紐床上位{QC}%）")
    print(f"★判定は各マス2つ（全期間 / ★後半{SPLIT}〜） → **{NCMP}比較**・z = {z:.3f}")
    rng = np.random.default_rng(0)
    n = 120_000
    pay = rng.choice([0.0, 300.0, 6000.0], size=n, p=[0.90, 0.07, 0.03])
    m = float(np.mean([(pay[rng.permutation(n)] - pay[rng.permutation(n)]).mean()
                       for _ in range(200)]))
    print(f"★ゲート2（対応差）: 200回の平均 {m:+.3f}円 → **仮説が偽なら0**: "
          f"{'★OK' if abs(m) < 4 else '⚠NG'}")
    ok &= abs(m) < 4
    # ★先読み防止の自己テスト: 年Yの閾値にYのデータが入っていないこと
    hist = {2016: np.array([1.0]), 2017: np.array([100.0])}
    thr = float(np.percentile(np.concatenate([hist[u] for u in hist if u < 2017]), 90.0))
    print(f"★先読み防止: 2017の閾値 = **{thr:.1f}**（2017の値100は入っていない）: "
          f"{'★OK' if thr < 50 else '⚠NG'}")
    ok &= thr < 50
    print(f"★★内部対照: **絶対閾値{ABS_FILT}・床{ABS_C} が (194)の "
          f"前半+{KNOWN_H1}円 / 後半+{KNOWN_H2}円 を ±{KTOL}円 で再現すること**")
    print("★★家族Aの読み方: **分布が縮んでいれば仮説は生きる。横ばいなら仮説は死ぬ**")
    print("★自己テスト: " + ("全部OK" if ok else "⚠NG"))
    return 0 if ok else 1


def pick2(ub, keyneg, pool):
    o = sorted(pool, key=lambda k: (keyneg[k], k))
    return [int(ub[k]) for k in o[:2]]


def main():
    z = zq(ALPHA / NCMP)
    print("(195) ★★★**劣化の原因を測る** — 絶対閾値 vs ★**相対閾値（上位◯%）**")
    print("★仮説: **モデルが賢くなり、ズレの分布そのものが縮んでいる**")
    print("★★先読み防止: **年Yの閾値はY未満の年だけから作る**\n")

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

    # ── 第1周: レースごとの素材を作る ──
    R = []
    box4 = []
    Rs = []
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
        bx = [payoff(r, "三連複", list(c))
              for c in combinations(sorted(int(u) for u in ub[np.argsort(-pv)[:4]]), 3)]
        if not any(x is None for x in bx):
            box4.append(sum(bx) - 400.0)
        R.append({"r": r, "yr": int(gg["date"].iloc[0].year), "ub": ub, "od": od,
                  "pn": pn, "gap": pn - qp,
                  "bi": np.array([band_of(float(o), BANDS) for o in od])})
    print(f"\n★対象 **{len(R):,}レース**")

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

    # ── 家族A: 年ごとの分布（記述） ──
    yrs = sorted({x["yr"] for x in R})
    print(f"\n{'='*112}")
    print("■ ★★★家族A（**記述**）: **ズレの分布は縮んでいるか**")
    print("　★★読み方: **縮んでいれば仮説は生きる。横ばいなら仮説は死ぬ**")
    print(f"\n{'年':<7}{'R数':>7}{'gap上位2%':>11}{'上位5%':>10}{'上位10%':>10}"
          f"{'pn上位10%':>11}{'上位20%':>10}{'★絶対閾値を満たす馬の割合':>26}{'その平均オッズ':>14}")
    GH, PH = {}, {}
    for u in yrs:
        gs = np.concatenate([x["gap"] for x in R if x["yr"] == u])
        ps = np.concatenate([x["pn"] for x in R if x["yr"] == u])
        os_ = np.concatenate([x["od"] for x in R if x["yr"] == u])
        GH[u], PH[u] = gs, ps
        hit = (ps >= ABS_FILT[0]) & (gs >= ABS_FILT[1])
        print(f"{u:<7}{sum(1 for x in R if x['yr']==u):>7,}"
              f"{np.percentile(gs,98):>11.4f}{np.percentile(gs,95):>10.4f}"
              f"{np.percentile(gs,90):>10.4f}{np.percentile(ps,90):>11.4f}"
              f"{np.percentile(ps,80):>10.4f}{100*hit.mean():>25.2f}%"
              f"{(os_[hit].mean() if hit.any() else float('nan')):>13.1f}倍")

    # ── 第2周: 賭ける ──
    rng = np.random.default_rng(SEED)
    CELLS = [(a, b, c) for a in QG for b in QP for c in QC]
    K = {k: {"z": [], "r": [], "yr": []} for k in CELLS}
    K["ABS"] = {"z": [], "r": [], "yr": []}
    for x in R:
        u = x["yr"]
        prior = [v for v in yrs if v < u]
        thr = None
        if prior:
            gpool = np.concatenate([GH[v] for v in prior])
            ppool = np.concatenate([PH[v] for v in prior])
            thr = ({q: float(np.percentile(gpool, q)) for q in set(QG) | set(QC)},
                   {q: float(np.percentile(ppool, q)) for q in QP})
        ub, od, pn, gap, bi = x["ub"], x["od"], x["pn"], x["gap"], x["bi"]
        order_g = list(np.argsort(-gap, kind="mergesort"))
        arms = []
        arms.append(("ABS", ABS_FILT[0], ABS_FILT[1], ABS_C))
        if thr is not None:
            for (a, b, c) in CELLS:
                arms.append(((a, b, c), thr[1][b], thr[0][a], thr[0][c]))
        for key, pth, gth, cth in arms:
            cand = np.where((pn >= pth) & (gap >= gth))[0]
            if not len(cand):
                continue
            ai = int(cand[int(np.argmax(gap[cand]))])
            axu = int(ub[ai])
            hg = [int(ub[k]) for k in order_g if int(ub[k]) != axu and gap[k] >= cth]
            if len(hg) < 2:
                continue
            hr, ok = [], True
            for uu in hg[:2]:
                b = bi[list(ub).index(uu)]
                pl = [int(ub[k]) for k in range(len(ub))
                      if bi[k] == b and int(ub[k]) != axu and int(ub[k]) not in hr]
                if not pl:
                    ok = False
                    break
                hr.append(int(rng.choice(pl)))
            if not ok:
                continue
            vz = [payoff(x["r"], "馬連", [axu, w]) for w in hg[:2]]
            vr = [payoff(x["r"], "馬連", [axu, w]) for w in hr]
            if any(v is None for v in vz + vr):
                continue
            K[key]["z"].append(sum(vz) / 2.0)
            K[key]["r"].append(sum(vr) / 2.0)
            K[key]["yr"].append(u)

    def halves(k):
        z_ = np.asarray(K[k]["z"], float)
        r_ = np.asarray(K[k]["r"], float)
        y_ = np.asarray(K[k]["yr"], int)
        h = y_ < SPLIT
        return z_, r_, y_, h

    z_, r_, y_, h = halves("ABS")
    d1, d2 = (z_[h] - r_[h]).mean(), (z_[~h] - r_[~h]).mean()
    ok1 = abs(d1 - KNOWN_H1) <= KTOL and abs(d2 - KNOWN_H2) <= KTOL
    print(f"\n■ ★★内部対照（**絶対閾値 {ABS_FILT}・床{ABS_C}**・{len(z_):,}R）")
    print(f"　前半 {d1:+.1f}円 vs (194) +{KNOWN_H1}円　／　後半 {d2:+.1f}円 vs +{KNOWN_H2}円"
          f"　→ **{'★再現' if ok1 else '⚠⚠ズレた'}**")
    print(f"　ROI: 全期間 {roi_of(z_):.1f}% / 前半 {roi_of(z_[h]):.1f}% / "
          f"後半 {roi_of(z_[~h]):.1f}%　（乱 {roi_of(r_):.1f}%）")
    if not ok1:
        print("\n⚠⚠**内部対照が立たない。読まない**（判定基準32）。")
        return

    print(f"\n{'='*112}")
    print(f"■ ★★★家族B: **相対閾値（その年までの分布の上位◯%）**")
    print(f"　★主判定は「ズレ順 − 乱」の対応差。**★本命は後半({SPLIT}〜)**"
          f"（**絶対閾値版は +{KNOWN_H2}円**）")
    print(f"\n{'軸ズレ':>8}{'軸推奨':>8}{'紐床':>7}{'R数':>8}{'ROI':>8}"
          f"{'前半ROI':>9}{'後半ROI':>9}{'★全差':>9}{'99%CI':>19}"
          f"{'★★後半差':>11}{'99%CI':>19}{'判定':>13}")
    passed = []
    for (a, b, c) in CELLS:
        z2, r2, y2, h2 = halves((a, b, c))
        if len(z2) < MINCELL or (~h2).sum() < MINCELL:
            print(f"{'上位'+str(100-a)+'%':>8}{'上位'+str(100-b)+'%':>8}"
                  f"{'上位'+str(100-c)+'%':>7}{len(z2):>8,}　⚠**標本不足**")
            continue
        dd = z2 - r2
        mu, se = dd.mean(), dd.std(ddof=1) / math.sqrt(len(dd))
        d2v = dd[~h2]
        mu2, se2 = d2v.mean(), d2v.std(ddof=1) / math.sqrt(len(d2v))
        s1, s2 = mu - z * se > 0, mu2 - z * se2 > 0
        better = mu2 > KNOWN_H2
        if s2 and better:
            passed.append(((a, b, c), mu2, roi_of(z2)))
        tag = ("★★★通った" if s2 and better else "★全期間のみ" if s1
               else "⚠通らない")
        print(f"{'上位'+str(int(100-a))+'%':>8}{'上位'+str(int(100-b))+'%':>8}"
              f"{'上位'+str(int(100-c))+'%':>7}{len(z2):>8,}{roi_of(z2):>7.1f}%"
              f"{roi_of(z2[h2]):>8.1f}%{roi_of(z2[~h2]):>8.1f}%{mu:>+8.1f}円"
              f"{f'[{mu-z*se:+.1f},{mu+z*se:+.1f}]':>19}{mu2:>+10.1f}円"
              f"{f'[{mu2-z*se2:+.1f},{mu2+z*se2:+.1f}]':>19}{tag:>13}")

    print(f"\n■ ★採用条件")
    print(f"　1. ★**後半の差が有意かつ絶対閾値版(+{KNOWN_H2}円)を上回る** … "
          f"**{len(passed)}/{len(CELLS)}**")
    if not passed:
        print("\n★★★**結論: 相対閾値にしても劣化は直らない**"
              "——**原因は「分布の縮小」ではない**。⚠**仮説は死んだ**。")
    else:
        print("\n★★**相対閾値で後半の優位が回復したマスがある**"
              "（⚠**水準ではなく対照との差であることに注意**）。")
    print("⚠**経路は弱いので「閉じた」とは書けない**（判定基準25）。")
    print("\n⚠**枠連の運用には触れない**。**設定変更は提案しない**。")


if __name__ == "__main__":
    sys.exit(selftest() if "--selftest" in sys.argv else (main() or 0))
