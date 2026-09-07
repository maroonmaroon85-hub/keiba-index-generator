"""(210) ★★★**主張を1つに固定する** — L=10・複勝。★検証ではなく「今後どう確かめるか」を決める

⚠⚠**これは検証ではない**。**(208)(209)で同じ場所が2回良い数字を出したが、
　母集団がほぼ同じなので独立な確認ではない**。★**同じデータで3回目を測っても意味がない**。
★★**やるのは3つ**:
　**1. 主張を1つに固定する**（**もう探索しない**）
　**2. その主張の全期間の姿を、水準・対応差・年別・裾で正しく出す**
　**3. ★今後どれだけ集めれば決着するかを計算し、監視の停止規則を書く**

■ ★★★**固定する主張**（**これ以降、探索しない**）
　**軸 = pn≥0.15 かつ gap≥0.15 かつ ★単勝オッズ≥10.0 の中で pn最大**、**複勝1点**。
　⚠**候補が無ければ見送る**。**1日約1.2本・年約126本**。

────────────────────────────────────────────────────────────
★★★ 事前登録（2026-09-07・**結果を見る前にコミットする**）
────────────────────────────────────────────────────────────

■ ★経路（判定基準25）: **q = MLモデル / q_pool = 複勝の板**。⚠**q側は弱い**。
■ ★★**判定はしない**（**新しい主張を作らない**）。**多重比較の補正も要らない**。
　**その代わり「有意」「採用」とは書かない**。★**出すのは姿と、必要な標本だけ**。

■ ★出すもの
　**1. 水準**: **ROI / 標準誤差 / 99%CI（正規・ブートストラップ10,000回）/ 100%超の割合**
　**2. 対照**: **同じオッズ帯の乱（10種平均）との対応差とそのCI**
　**3. 年別**: **各年のROIと本数 / 100%超の年数 / 前半・後半**
　**4. 裾**: **上位1%・5%が全払戻に占める割合 / 平均÷中央値**
　**5. ★必要標本**: **99%CI下端を100%超にするのに要る本数と年数**（**(193)と同じ手続き**）
　**6. ★これから集めるぶんだけで決着するか**（**年126本で何年かかるか**）

■ ★★ゲート2（判定基準42）——**この測定は何を返せば「意味なし」か**
　★**乱との対応差は、穴と乱が交換可能なら厳密に0**。
　★**必要標本の式は、下端がちょうど100%のとき現在の本数を返す**（**自己テストで確認**）。

■ ⚠ゲート1: (88)③④を再現（±3pt）／ ★ゲート板: 復元R 0.800±0.020 ／
　★陽性対照: 三連複BOX上位4 81.9%±1.5pt。
■ ★★★内部対照（**決定的・乱を使わない**）: **L=1・複勝 が 95.9% ±0.2pt かつ 6,336R ±5**。

■ ★★★**監視の停止規則（先に書く・別セッションの指摘に従う）**
　★**貯めるのは毎開催・読むのは年1回**。
　★★**読む前の主判定はこれ1つ**:
　　**「2026-09-07以降に集めたぶんだけで、複勝ROIの99%CI下端が100%を超えるか」**
　⚠**途中で読んで「今年は良かった」と言わない**（**判定基準43**）。
　⚠**規則は一切変えない**（**変えたら集めた標本は使えなくなる**）。
　★**必要年数がこの測定で出る**。**それを超えても届かなければ、この線は閉じる**。

実行: python3 ml/audit_ana_fix.py    自己テスト: python3 ml/audit_ana_fix.py --selftest
"""
import math
import sys
from itertools import combinations
from zlib import crc32

import numpy as np

sys.path.insert(0, "ml")
import features as F
from audit_crosspool import load_races, payoff, zq
from audit_ana_odds import BANDS, MIN_HORSES, band_of, gate1, roi_of
from audit_ana_marg import WF_BOX4, WF_TOL, wf_predict
from audit_ana_board import BOARD_R, BOARD_TOL, NPLACE, load_fuku_boards, qpool
from audit_ana_band import PN_FLOOR
from train_prod import add_odds_features

GAP, LFIX = 0.15, 10.0
NRAND = 10
NBOOT = 10000
SEED = 20260906
SPLIT = 2021
KNOWN_ROI, KNOWN_N, ROI_TOL, N_TOL = 95.9, 6336, 0.2, 5
ALPHA = 0.01


def need(v, roi, z):
    """★99%CI下端を100%超にするのに要る本数"""
    s = float(np.std(v, ddof=1))
    if roi <= 100.0:
        return None
    return (z * s / (roi - 100.0)) ** 2


def selftest():
    ok = True
    z = zq(ALPHA / 2)
    print("★★**固定する主張**: **軸 = pn≥0.15 かつ gap≥0.15 かつ "
          f"★単勝オッズ≥{LFIX:.0f}倍 の中で pn最大**、**複勝1点**")
    print("⚠⚠**これは検証ではない**——**同じデータで3回目を測っても意味がない**")
    print("★やるのは **主張の固定 / 姿を正しく出す / 必要標本と停止規則**")
    rng = np.random.default_rng(0)
    v = rng.normal(100.0, 300.0, 50_000)
    se = v.std(ddof=1) / math.sqrt(len(v))
    n = need(v, 100.0 + z * se, z)
    print(f"★必要標本の式: 下端ちょうどのとき need={n:,.0f} vs n={len(v):,}"
          f"　{'★OK' if abs(n-len(v))/len(v) < 0.01 else '⚠NG'}")
    ok &= abs(n - len(v)) / len(v) < 0.01
    n2 = 120_000
    pay = rng.choice([0.0, 250.0, 900.0], size=n2, p=[0.62, 0.30, 0.08])
    m = float(np.mean([(pay[rng.permutation(n2)] - pay[rng.permutation(n2)]).mean()
                       for _ in range(200)]))
    print(f"★ゲート2（対応差）: **偽なら0** → {m:+.3f}円　{'★OK' if abs(m) < 2 else '⚠NG'}")
    ok &= abs(m) < 2
    print(f"★★★内部対照（決定的）: **L=1・複勝 が {KNOWN_ROI}% ±{ROI_TOL}pt "
          f"かつ {KNOWN_N:,}R ±{N_TOL}**")
    print("★★停止規則: **貯めるのは毎開催・読むのは年1回**。"
          "**主判定は「2026-09-07以降のぶんだけで下端が100%超か」1つ**")
    print("★自己テスト: " + ("全部OK" if ok else "⚠NG"))
    return 0 if ok else 1


def main():
    z = zq(ALPHA / 2)
    print(f"(210) ★★★**主張を1つに固定する** — L={LFIX:.0f}・複勝")
    print("⚠⚠**検証ではない**。**固定・姿・必要標本・停止規則を出す**\n")

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

    A = {"a": [], "r": [], "yr": [], "od": [], "rk": []}
    full = []
    Rs, box4, nall, ndays = [], [], 0, set()
    for rid, g in sub.groupby("raceid"):
        rid = str(rid)
        r = races.get(rid)
        bd = boards.get(rid)
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
        ndays.add(gg["date"].iloc[0].date())
        pn = pv / pv.sum() * NPLACE
        qp, Rb = qpool([bd[int(u)] for u in ub], "harm")
        Rs.append(Rb)
        gap = pn - qp
        order_p = [int(u) for u in ub[np.argsort(-pv, kind="mergesort")]]
        bx = [payoff(r, "三連複", list(c)) for c in combinations(sorted(order_p[:4]), 3)]
        if not any(x is None for x in bx):
            box4.append(sum(bx) - 400.0)
        bi = np.array([band_of(float(o), BANDS) for o in od])
        rank = np.argsort(np.argsort(od)) + 1
        base = (pn >= PN_FLOOR) & (gap >= GAP)
        c0 = np.where(base)[0]
        if len(c0):
            i0 = int(c0[int(np.argmax(pn[c0]))])
            v0 = payoff(r, "複勝", [int(ub[i0])])
            if v0 is not None:
                full.append(v0)
        cand = np.where(base & (od >= LFIX))[0]
        if not len(cand):
            continue
        i = int(cand[int(np.argmax(pn[cand]))])
        va = payoff(r, "複勝", [int(ub[i])])
        if va is None:
            continue
        vr, okall = [], True
        for sd in range(NRAND):
            g2 = np.random.default_rng([SEED + sd, crc32(rid.encode())])
            pl = [int(ub[q]) for q in range(len(ub)) if bi[q] == bi[i] and q != i]
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
        A["a"].append(va); A["r"].append(float(np.mean(vr)))
        A["yr"].append(int(gg["date"].iloc[0].year))
        A["od"].append(float(od[i])); A["rk"].append(int(rank[i]))
    print(f"\n★対象 **{nall:,}レース / {len(ndays):,}開催日**")

    med = float(np.median(Rs))
    okR = abs(med - BOARD_R) <= BOARD_TOL
    print(f"■ ★ゲート板: 復元R = **{med:.4f}** → **{'★立った' if okR else '⚠⚠落ちた'}**")
    box4 = np.asarray(box4, float)
    g0 = 100.0 * (box4.sum() + 400.0 * len(box4)) / (400.0 * len(box4))
    okb = abs(g0 - WF_BOX4) <= WF_TOL
    print(f"■ ⚠**陽性対照**: BOX上位4 **{g0:.1f}%** vs {WF_BOX4}%"
          f" → **{'★立った' if okb else '⚠⚠落ちた'}**")
    fv = np.asarray(full, float)
    okc = (abs(roi_of(fv) - KNOWN_ROI) <= ROI_TOL and abs(len(fv) - KNOWN_N) <= N_TOL)
    print(f"■ ★★★内部対照（決定的）: L=1・複勝 = **{roi_of(fv):.1f}%**（{len(fv):,}R） vs "
          f"{KNOWN_ROI}%（{KNOWN_N:,}R） → **{'★再現' if okc else '⚠⚠ズレた'}**")
    if not (okR and okb and okc):
        print("\n⚠⚠**ゲートが落ちた。読まない**（判定基準32）。")
        return

    a = np.asarray(A["a"], float)
    rv = np.asarray(A["r"], float)
    yr = np.asarray(A["yr"], int)
    od = np.asarray(A["od"], float)
    n = len(a)
    roi = roi_of(a)
    se = a.std(ddof=1) / math.sqrt(n)
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, n, size=(NBOOT, n))
    b = a[idx].mean(axis=1)
    per_year = n / 11.0
    print(f"\n{'='*104}")
    print(f"■ ★★**1. 水準**（**{n:,}本 / 年約{per_year:.0f}本 / 1日{n/len(ndays):.1f}本**）")
    print(f"　★**ROI {roi:.1f}%**　標準誤差 {se:.2f}pt"
          f"　99%CI（正規）[{roi-z*se:.1f}, {roi+z*se:.1f}]")
    print(f"　★ブートストラップ99%区間 [{np.percentile(b,0.5):.1f}, "
          f"{np.percentile(b,99.5):.1f}]　★**100%超の割合 {100*np.mean(b>100):.1f}%**")
    print(f"　軸: 平均 {od.mean():.1f}倍 / 中央 {np.median(od):.1f}倍 / "
          f"平均 {np.mean(A['rk']):.1f}番人気 / 的中率 {100*np.mean(a>0):.1f}%")

    dd = a - rv
    mu, dse = dd.mean(), dd.std(ddof=1) / math.sqrt(n)
    print(f"\n■ ★★**2. 対照**（**同じオッズ帯の乱・10種平均**）")
    print(f"　乱のROI **{roi_of(rv):.1f}%**　★**対応差 {mu:+.1f}円**"
          f"　99%CI [{mu-z*dse:+.1f}, {mu+z*dse:+.1f}]"
          f"　→ **{'★0を含まない' if mu-z*dse > 0 else '⚠0を含む'}**")

    print(f"\n■ ★**3. 年別**")
    ys = sorted(set(yr))
    print(f"{'年':<7}" + "".join(f"{u:>8}" for u in ys) + f"{'★100%超':>10}")
    print(f"{'本数':<7}" + "".join(f"{int((yr==u).sum()):>8}" for u in ys))
    rs = [roi_of(a[yr == u]) for u in ys]
    print(f"{'ROI':<7}" + "".join(f"{x:>7.0f}%" for x in rs)
          + f"{sum(1 for x in rs if x > 100):>7}/{len(ys)}")
    h = yr < SPLIT
    print(f"　★前半(〜2020) {roi_of(a[h]):.1f}%（{int(h.sum())}本） / "
          f"後半(2021〜) {roi_of(a[~h]):.1f}%（{int((~h).sum())}本）")

    s = np.sort(a)[::-1]
    tot = max(a.sum(), 1e-9)
    hit = a[a > 0]
    print(f"\n■ ★**4. 裾**")
    print(f"　上位1% {100*s[:max(1,n//100)].sum()/tot:.1f}% / "
          f"上位5% {100*s[:max(1,n//20)].sum()/tot:.1f}% / "
          f"上位10% {100*s[:max(1,n//10)].sum()/tot:.1f}%")
    print(f"　1回配当: 平均 {hit.mean():,.0f}円 / 中央 {np.median(hit):,.0f}円"
          f"　★**平均÷中央 {hit.mean()/max(np.median(hit),1e-9):.2f}倍**")

    print(f"\n■ ★★★**5. 必要標本**（**99%CI下端を100%超にするには**）")
    nd = need(a, roi, z)
    if nd is None:
        print(f"　⚠**ROIが100%未満なので、標本を増やしても到達しない**")
    else:
        print(f"　★**必要本数 {nd:,.0f}本**（現在 {n:,}本 → ★**{nd/n:.1f}倍**）")
        print(f"　★**年{per_year:.0f}本なら {nd/per_year:,.0f}年**")
        rest = max(nd - n, 0)
        print(f"\n■ ★★★**6. これから集めるぶんだけで決着するか**")
        print(f"　⚠**過去11年は「探索に使った」ので、決着には使えない**"
              f"（**同じデータで3回目を測っても意味がない**）")
        print(f"　★**2026-09-07以降にゼロから集める場合: {nd:,.0f}本 ÷ 年{per_year:.0f}本 "
              f"= ★{nd/per_year:,.0f}年**")

    print(f"\n{'='*104}")
    print("■ ★★★**監視の停止規則（事前に書いたもの）**")
    print("　★**貯めるのは毎開催・読むのは年1回**")
    print("　★★**主判定はこれ1つ**: "
          "**「2026-09-07以降に集めたぶんだけで、複勝ROIの99%CI下端が100%を超えるか」**")
    print("　⚠**途中で読んで「今年は良かった」と言わない**（判定基準43）")
    print("　⚠**規則は一切変えない**（**変えたら集めた標本は使えなくなる**）")
    print("\n⚠**枠連の運用には触れない**。**設定変更は提案しない**。")


if __name__ == "__main__":
    sys.exit(selftest() if "--selftest" in sys.argv else (main() or 0))
