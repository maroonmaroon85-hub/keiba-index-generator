"""(200) ★★★**裾で100%を超えるのは「あり」か** — ★**年数と確率で出す**（判定ではなく意思決定の材料）

★★**動機（2026-09-06・利用者の指摘）**:
　★**「★95.1%で、裾で100%超えられるなら全然ありじゃない？」**
　★★**正しい指摘**——**ROIの算術平均が期待値そのもの**。**裾は偽物ではない**。
　⚠**私が(199)で「実力は95%、差はまるごと裾」と書いたのは「裾は無効」と読める**。**訂正する**。
　★**正しくは「期待値の点推定は103.5%。ただしその推定の確からしさが低い」**。

★★★**だが実務的な問いは別にある**——**確かめられないまま賭けたら何が起きるか**。
　⚠**判定基準4**: **6,206マスの総当たりで34.4マスが偶然100%超になる**。
　　★**この103.1%は(190)で42マス試した中の最大値**。**「運」の部分は配当の裾に乗る**。
　　★**(199)で中央値の比が1.000倍だったのは、そのことと整合する**。
　★★**そこで、判定ではなく「意思決定の材料」を出す**。

────────────────────────────────────────────────────────────
★★★ 事前登録（2026-09-06・**結果を見る前にコミットする**）
────────────────────────────────────────────────────────────

■ ★★**これは判定ではない**（**採用も棄却もしない**）。**すべて記述**。
　⚠**新しい主張を作らない**——**既に測った腕の、時間軸での挙動を出すだけ**。
　★**だから多重比較の補正は要らない**（**判定をしないから**）。**その代わり「有意」とも書かない**。

■ ★経路（判定基準25）: **母集団・買い方は(198)(199)と同一**（**軸 = gap最大、紐 = p降順の2頭**）。
　**絞りは (0.15,0.02)**（**三連単103.5%が出た側**）。

■ ★★★出すもの（**4つの腕を並べる**）
| 腕 | 買い方 | 1レースの費用 | (199)実測ROI |
|---|---|---|---|
| ★**複勝** | 軸1点 | 100円 | **91.5%**（堅い側） |
| **馬連** | 軸−紐1の1点 | 100円 | **90.3%** |
| **三連複** | 軸+紐2頭の1点 | 100円 | **91.3%** |
| ★**三連単** | 軸1着固定・紐入替の2点 | 200円 | ★**103.5%** |

■ ★★★①**時間軸のブートストラップ**（**レース単位で復元抽出**）
　**1年 = その腕が買うレース数**。**1 / 3 / 5 / 10 / 20年 を 10,000回ずつ試行**。
　★**出す値: プラスで終わっている確率 / 累積ROIの中央値 / 5%点 / 95%点 / 最悪の資金減少**。
　★★**「期待値が100%超か」ではなく「20年賭けてプラスでいる確率」を出す**。

■ ★★②**利益の集中度**: **上位1% / 5% / 10%の的中が、全払戻の何%を占めるか**。
　★**複勝と三連単を並べれば、「堅い」と「裾頼み」の違いが数字で出る**。

■ ★★③**破産の目安**: **各年の収支の分布**、**最悪の年**、**連続でマイナスの最長年数**。

■ ★★ゲート2（判定基準42）——**この測定は何を返せば「意味なし」か**
　★**ブートストラップは元データの平均に収束する**——**1年の中央値ROIが実測ROIから
　　大きくズレたら実装が壊れている**。★**それを毎腕で確認して表示する**。

■ ⚠ゲート1: (88)③④を再現（±3pt）／ ★ゲート板: 復元R 0.800±0.020 ／
　★陽性対照: 三連複BOX上位4 81.9%±1.5pt。
■ ★★内部対照（決定的）: **三連単 (0.15,0.02) のROIが 103.5% を ±0.5pt で再現**。

■ ★★★注意（**先に書いておく**）
　⚠**ブートストラップは「103.5%が真の期待値である」という前提の下での話**。
　★**その前提こそが不確かだというのが(199)の内容**。**だから「これで勝てる」とは読めない**。
　★★**読み方は「もし103.5%が本当なら、それでも20年でこうなる」**という条件付き。
　⚠**選択バイアス（42マスの最大値）はブートストラップでは直らない**——**別途明記する**。

実行: python3 ml/audit_ana_bank.py    自己テスト: python3 ml/audit_ana_bank.py --selftest
"""
import math
import sys
from itertools import combinations

import numpy as np

sys.path.insert(0, "ml")
import features as F
from audit_crosspool import load_races, payoff
from audit_ana_odds import MIN_HORSES, gate1, roi_of
from audit_ana_marg import WF_BOX4, WF_TOL, wf_predict
from audit_ana_board import BOARD_R, BOARD_TOL, NPLACE, load_fuku_boards, qpool
from train_prod import add_odds_features

FL = (0.15, 0.02)
ARMS = ["複勝", "馬連", "三連複", "三連単"]
NPT = {"複勝": 1, "馬連": 1, "三連複": 1, "三連単": 2}
YEARS = [1, 3, 5, 10, 20]
NSIM = 10000
NY = 11.0
KNOWN_TRI, ROI_TOL = 103.5, 0.5
SEED = 20260906


def arm_tickets(kind, ax, h1, h2):
    if kind == "複勝":
        return [("複勝", [ax])]
    if kind == "馬連":
        return [("馬連", [ax, h1])]
    if kind == "三連複":
        return [("三連複", sorted([ax, h1, h2]))]
    return [("三連単", [ax, h1, h2]), ("三連単", [ax, h2, h1])]


def selftest():
    ok = True
    print(f"★★**これは判定ではない**——**採用も棄却もしない。すべて記述**")
    print(f"★腕: {ARMS}（費用 {[NPT[a]*100 for a in ARMS]}円/R）")
    print(f"★年数 {YEARS}　試行 {NSIM:,}回")
    rng = np.random.default_rng(0)
    # ★ゲート2: ブートストラップは元の平均に収束する
    v = rng.choice([0.0, 5000.0], size=20000, p=[0.97, 0.03])
    idx = rng.integers(0, len(v), size=(3000, len(v)))
    m = v[idx].mean(axis=1)
    print(f"★ゲート2: **元の平均 {v.mean():.1f}円 vs ブートストラップ中央 "
          f"{np.median(m):.1f}円**　{'★OK' if abs(np.median(m)-v.mean()) < 20 else '⚠NG'}")
    ok &= abs(np.median(m) - v.mean()) < 20
    # ★集中度の自己テスト
    s = np.sort(v)[::-1]
    top1 = s[:len(s)//100].sum() / s.sum()
    print(f"★集中度の計算: 上位1%が全払戻の {100*top1:.1f}%"
          f"（**的中3%なので上位1%は的中の1/3 → 33%前後が正しい**）"
          f"　{'★OK' if 0.25 < top1 < 0.40 else '⚠NG'}")
    ok &= 0.25 < top1 < 0.40
    print("⚠**注意**: **ブートストラップは「実測ROIが真の期待値」という前提の下の話**。")
    print("　★**選択バイアス（42マスの最大値）はこれでは直らない**。**別途明記する**。")
    print("★自己テスト: " + ("全部OK" if ok else "⚠NG"))
    return 0 if ok else 1


def main():
    print("(200) ★★★**裾で100%を超えるのは「あり」か** — ★**年数と確率で出す**")
    print("★★**判定ではない。意思決定の材料**\n")

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

    K = {a: {"pay": [], "yr": []} for a in ARMS}
    Rs, box4 = [], []
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
        gap = pn - qp
        bx = [payoff(r, "三連複", list(c))
              for c in combinations(sorted(int(u) for u in ub[np.argsort(-pv)[:4]]), 3)]
        if not any(x is None for x in bx):
            box4.append(sum(bx) - 400.0)
        cand = np.where((pn >= FL[0]) & (gap >= FL[1]))[0]
        if not len(cand):
            continue
        ai = int(cand[int(np.argmax(gap[cand]))])
        axu = int(ub[ai])
        hp = [int(u) for u in ub[np.argsort(-pv, kind="mergesort")] if int(u) != axu]
        if len(hp) < 2:
            continue
        vals, okall = {}, True
        for a in ARMS:
            vs = [payoff(r, nm, sel) for nm, sel in arm_tickets(a, axu, hp[0], hp[1])]
            if any(v is None for v in vs):
                okall = False
                break
            vals[a] = sum(vs)
        if not okall:
            continue
        yr = int(gg["date"].iloc[0].year)
        for a in ARMS:
            K[a]["pay"].append(vals[a]); K[a]["yr"].append(yr)

    med = float(np.median(Rs))
    okR = abs(med - BOARD_R) <= BOARD_TOL
    print(f"\n■ ★ゲート板: 復元R = **{med:.4f}** → **{'★立った' if okR else '⚠⚠落ちた'}**")
    box4 = np.asarray(box4, float)
    g0 = 100.0 * (box4.sum() + 400.0 * len(box4)) / (400.0 * len(box4))
    okb = abs(g0 - WF_BOX4) <= WF_TOL
    print(f"■ ⚠**陽性対照**: BOX上位4 **{g0:.1f}%** vs {WF_BOX4}%"
          f" → **{'★立った' if okb else '⚠⚠落ちた'}**")
    tv = np.asarray(K["三連単"]["pay"], float)
    tr = 100.0 * tv.mean() / (100.0 * NPT["三連単"])
    okc = abs(tr - KNOWN_TRI) <= ROI_TOL
    print(f"■ ★★内部対照（決定的）: 三連単 ROI = **{tr:.1f}%** vs {KNOWN_TRI}% "
          f"→ **{'★再現' if okc else '⚠⚠ズレた'}**（{len(tv):,}R）")
    if not (okR and okb and okc):
        print("\n⚠⚠**ゲートが落ちた。読まない**（判定基準32）。")
        return

    rng = np.random.default_rng(SEED)
    print(f"\n{'='*112}")
    print("■ ★★★①**時間軸**（**レース単位の復元抽出・10,000回**）"
          "　★★**「もし実測ROIが真の期待値なら」という条件付き**")
    for a in ARMS:
        v = np.asarray(K[a]["pay"], float)
        cost = 100.0 * NPT[a]
        n = len(v)
        per_year = int(round(n / NY))
        roi = 100.0 * v.mean() / cost
        print(f"\n★**{a}**　{cost:.0f}円/R　**年 {per_year:,}レース**（投資 {cost*per_year/10000:.1f}万円/年）"
              f"　**実測ROI {roi:.1f}%**")
        print(f"{'年数':<6}{'総投資':>11}{'★プラスの確率':>15}{'累積ROI中央':>13}"
              f"{'5%点':>9}{'95%点':>9}{'★中央の収支':>14}{'★5%点の収支':>15}")
        for yy in YEARS:
            m = per_year * yy
            idx = rng.integers(0, n, size=(NSIM, m))
            tot = v[idx].sum(axis=1)
            spend = cost * m
            r_ = 100.0 * tot / spend
            print(f"{yy:<6}{spend/10000:>9.1f}万{100*np.mean(tot > spend):>14.1f}%"
                  f"{np.median(r_):>12.1f}%{np.percentile(r_,5):>8.1f}%"
                  f"{np.percentile(r_,95):>8.1f}%"
                  f"{(np.median(tot)-spend)/10000:>+12.1f}万"
                  f"{(np.percentile(tot,5)-spend)/10000:>+13.1f}万")
        # ★ゲート2: 1年の中央値が実測ROIから大きくズレていないか
        idx = rng.integers(0, n, size=(2000, per_year))
        r1 = 100.0 * v[idx].sum(axis=1) / (cost * per_year)
        print(f"　★ゲート2: **ブートストラップ1年の平均 {r1.mean():.1f}% vs 実測 {roi:.1f}%**"
              f"　{'★OK' if abs(r1.mean()-roi) < 2.0 else '⚠⚠実装が壊れている'}")

    print(f"\n{'='*112}")
    print("■ ★★②**利益の集中度**（**全払戻のうち、上位の的中が占める割合**）")
    print(f"{'腕':<8}{'的中率':>9}{'的中数':>9}{'★上位1%':>10}{'上位5%':>9}{'上位10%':>9}"
          f"{'★1回配当(平均)':>16}{'(中央値)':>12}{'平均/中央':>10}")
    for a in ARMS:
        v = np.asarray(K[a]["pay"], float)
        h = v[v > 0]
        s = np.sort(v)[::-1]
        tot = max(v.sum(), 1e-9)
        n1 = max(1, len(v) // 100)
        n5 = max(1, len(v) // 20)
        n10 = max(1, len(v) // 10)
        print(f"{a:<8}{100*np.mean(v>0):>8.2f}%{len(h):>9,}"
              f"{100*s[:n1].sum()/tot:>9.1f}%{100*s[:n5].sum()/tot:>8.1f}%"
              f"{100*s[:n10].sum()/tot:>8.1f}%{h.mean():>15,.0f}円"
              f"{np.median(h):>11,.0f}円{h.mean()/max(np.median(h),1e-9):>9.2f}倍")

    print(f"\n{'='*112}")
    print("■ ★★③**年ごとの収支**（**実測の11年**・1レースあたりの費用で正規化）")
    print(f"{'腕':<8}" + "".join(f"{u:>8}" for u in range(2016, 2027)) + f"{'★最悪の年':>11}{'連続マイナス':>12}")
    for a in ARMS:
        v = np.asarray(K[a]["pay"], float)
        yr = np.asarray(K[a]["yr"], int)
        cost = 100.0 * NPT[a]
        rs, run, best = [], 0, 0
        for u in range(2016, 2027):
            m = yr == u
            rr = 100.0 * v[m].mean() / cost if m.sum() else float("nan")
            rs.append(rr)
            if rr < 100.0:
                run += 1
                best = max(best, run)
            else:
                run = 0
        print(f"{a:<8}" + "".join(f"{x:>7.0f}%" for x in rs)
              + f"{min(rs):>10.0f}%{best:>11}年")

    print("\n■ ★★★読み方（**先に書いたもの**）")
    print("　⚠**ブートストラップは「実測ROIが真の期待値」という前提の下の話**")
    print("　★**その前提こそが不確かだというのが(199)の内容**")
    print("　⚠⚠**選択バイアス（(190)の42マスの最大値）はブートストラップでは直らない**")
    print("\n⚠**枠連の運用には触れない**。**設定変更は提案しない**。")


if __name__ == "__main__":
    sys.exit(selftest() if "--selftest" in sys.argv else (main() or 0))
