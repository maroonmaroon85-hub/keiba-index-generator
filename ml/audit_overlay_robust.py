"""(142) ★★★★(141)を叩く — **時間分割・層別・容量**（2026-08-13）

★★まず(141)の懸念を1つ**取り下げる**（自分の書いた注の訂正）
　(141)の検算③で「実測O/E 1.572 が自分のqの言い分 1.405 を上回る →
　**q_pool側の揺らぎ（薄い組の最終オッズが偶然高い）を選んでいる疑い**」と書いた。
　⚠**これは筋が悪い**。**パリミュチュエルの最終オッズは投票金額の正確な比であって、測定ノイズではない**。
　　**公衆がある組を過小に賭けたなら、その組の払戻は本当に高い**。**そこを買えば本当に儲かる**。
　　→ ★**「q_poolが揺らいでいる」は反証ではなく、むしろ利益の源そのもの**。
　★**残る本物のリスクは2つだけ**:
　　**(a) 時間** — 締切前に張るしかない。**T-10分の比が最終の比と同じか**は未検証。
　　**(b) 容量** — 枠連プールは薄い。**自分の賭けが最終オッズを動かす**。
　★**(a)は過去データでは原理的に検証できない**（netkeibaは過去レースについて確定オッズしか返さない）。
　　→ **これから毎週 T-10分 と確定を突き合わせて貯めるしかない**。**別タスク**。
　★**このスクリプトが answerable なのは「過去に実在したか」と「どこに偏っているか」**。

★★事前登録（測る前に宣言する。**閾値は(141)の 1.290 に固定。動かさない**）
　1. **★時間分割（これが本体）**: 2015-2020（前半）で見て 2021-2026（後半）でも立つか。
　　 ★**閾値1.290は理論値（1/払戻率）であって当てはめていない**ので過学習の余地は無い。
　　 　それでも**期間で崩れないこと**は別途要る（(94)続で最良帯が年代で崩れた前例がある）。
　　 **判定: 後半だけで ROI の99%CI下端 > 100%**。
　2. **層別（すべて発走前に分かる。後から増やさない）**
　　 L1 **頭数**（9-12 / 13-15 / 16-18）… 多いほど組が薄い
　　 L2 **枠連の板の組数**（少/多）… 同上
　　 L3 **競馬場**（10場）… 開催規模の代理
　　 L4 **選んだ組の q_pool 水準**（十分位）… (141)③のオッズ帯を細かくした版
　　 ★**判定は「どこかで消えるか」ではなく「全体に広く出ているか」**。
　　 　**1つの層に集中していたら、その層の事情（薄さ）を疑う**。
　3. **★容量の見積もり（(b)への最初の答え）**: 選んだ組の**実際の投票シェア**は板から出る
　　 （q_pool = その組に入った金額の割合）。**1点を X 円にしたときオッズが何%下がるか**を
　　 **プール総額の仮定を置かずに**出せる: 自分が S 円入れると、その組のシェアは
　　 `(m + S)/(M + S)` になる（m=その組の金額, M=総額）。`m/M = q_pool` なので
　　 **オッズの下落率は S/M だけで決まる**。**M（枠連プールの総額）は外部知識が要る**ので、
　　 **M = 500万円 / 1000万円 / 2000万円 の3通りで感度を出す**（★仮定であることを明示する）。
　4. **予想**: ★**当てにしてよい予想は持っていない**。
　　 ⚠**強いて言えば「後半で崩れる」ほうを警戒する**。(94)続の前例があるから。**これは類推**。

実行: python3 ml/audit_overlay_robust.py [開始年(既定2015)]
"""
import math
import sys

import numpy as np

sys.path.insert(0, "ml")
from audit_capacity_d import mkt_waku_dist
from audit_cond_split import load_boards
from audit_crosspool import PAYBACK, load_races, probs, zq
from audit_crosspool2 import realized
from audit_lbs import build_matrix, fit_lambda
from audit_waku_vs_umaren import load_type
from waku_umatan import waku_of

R = PAYBACK["枠連"]
TH = 1.0 / R                       # ★(141)の閾値に固定。動かさない
JYO = {1: "札幌", 2: "函館", 3: "福島", 4: "新潟", 5: "東京",
       6: "中山", 7: "中京", 8: "京都", 9: "阪神", 10: "小倉"}


def roi_ci(prof, cost_per_race, alpha=0.01):
    p = np.asarray(prof, float)
    if len(p) < 2:
        return float("nan"), float("nan"), float("nan")
    m = p.mean()
    se = p.std(ddof=1) / math.sqrt(len(p))
    z = zq(alpha)
    c = cost_per_race
    return 1 + m / c, 1 + (m - z * se) / c, 1 + (m + z * se) / c


def main():
    y0 = int(sys.argv[1]) if len(sys.argv) > 1 else 2015
    races = load_races()
    wb, ub = load_boards(), load_type(4, 4)
    if not wb or not ub:
        sys.exit("枠連(--type 3)と馬連(--type 4)の板が両方要る。")

    P, i1, i2, i3, yrs = build_matrix(races, y0)
    lam = {}
    for yy in sorted(set(yrs.tolist())):
        tr = yrs < yy
        if tr.sum() < 3000:
            lam[yy] = None
            continue
        ok3 = tr & (i3 >= 0)
        lam[yy] = (fit_lambda(P[tr], i1[tr], i2[tr]),
                   fit_lambda(P[ok3], i1[ok3], i2[ok3], stage3=True, ic=i3[ok3]))

    rows = []
    for r in races:
        yy = r["year"]
        if yy < y0 or not lam.get(yy) or not r["wakuren"]:
            continue
        W, U = wb.get(r["rid"]), ub.get(r["rid"])
        if not W or not U:
            continue
        rl = realized(r)
        if rl is None:
            continue
        a, b, _ = rl
        n = r["n"]
        nums = [u for u, _, _ in r["horses"]]
        if a not in nums or b not in nums:
            continue
        p = probs(r["horses"])
        l2, _ = lam[yy]
        md = mkt_waku_dist(r, p, l2)
        if not md:
            continue
        agg = {}
        for (x, y), o in U.items():
            if o <= 0:
                continue
            wx, wy = sorted((waku_of(x, n), waku_of(y, n)))
            agg[(wx, wy)] = agg.get((wx, wy), 0.0) + 1.0 / o
        keys = [k for k in sorted(md) if k in W and k in agg]
        if len(keys) < 3:
            continue
        key = tuple(sorted((waku_of(a, n), waku_of(b, n))))
        if key not in keys:
            continue
        inv = np.array([1.0 / W[k] for k in keys])
        qp = inv / inv.sum()
        qu = np.array([agg[k] for k in keys])
        qu /= qu.sum()
        sel = (qu / qp) >= TH
        if not sel.any():
            continue
        odds = np.array([W[k] for k in keys])
        win = np.array([k == key for k in keys])
        prof = float(odds[sel & win].sum() * 100.0) - 100.0 * sel.sum()
        try:
            jyo = int(str(r["rid"])[:2])
        except Exception:
            jyo = 0
        rows.append(dict(y=yy, n=n, nk=len(keys), jyo=jyo, npt=int(sel.sum()),
                         prof=prof, qsel=float(qp[sel].mean()),
                         hit=int((sel & win).sum())))

    ys = np.array([x["y"] for x in rows])
    pr = np.array([x["prof"] for x in rows])
    npt = np.array([x["npt"] for x in rows])
    nn = np.array([x["n"] for x in rows])
    nk = np.array([x["nk"] for x in rows])
    jj = np.array([x["jyo"] for x in rows])
    qs = np.array([x["qsel"] for x in rows])
    N = len(rows)
    tot = npt.sum()
    print(f"(142) (141)を叩く（閾値 {TH:.3f} 固定・{N:,}レース・延べ {tot:,}点）")
    print(f"　全体 ROI {1 + pr.sum()/(100*tot):.3f}  1R収支 {pr.mean():+.1f}円\n")

    def show(lab, m):
        if m.sum() < 100:
            print(f"  {lab:<22}{int(m.sum()):>7}本  標本不足")
            return
        c = 100.0 * npt[m].mean()
        roi, lo, hi = roi_ci(pr[m], c)
        mark = "★" if lo > 1.0 else ("★負" if hi < 1.0 else "")
        print(f"  {lab:<22}{int(m.sum()):>7,}本 {int(npt[m].sum()):>7,}点"
              f"  ROI {100*roi:>6.1f}%  99%CI [{100*lo:>5.1f},{100*hi:>6.1f}]  {mark}")

    print("■ ★時間分割（これが本体。閾値は理論値なので当てはめていないが、期間で崩れないか）")
    show("2015-2020（前半）", ys <= 2020)
    show("2021-2026（後半）", ys >= 2021)
    print("\n■ 年別")
    for u in sorted(set(ys.tolist())):
        show(str(u), ys == u)

    print("\n■ L1 頭数（多いほど1組あたりが薄い）")
    for lo_, hi_, lab in ((9, 12, "9-12頭"), (13, 15, "13-15頭"), (16, 18, "16-18頭")):
        show(lab, (nn >= lo_) & (nn <= hi_))
    print("\n■ L2 枠連の板の組数")
    med = int(np.median(nk))
    show(f"組数 ≤{med}", nk <= med)
    show(f"組数 >{med}", nk > med)
    print("\n■ L3 競馬場")
    for k in sorted(set(jj.tolist())):
        if k in JYO:
            show(JYO[k], jj == k)
    print("\n■ L4 選んだ組の q_pool 水準（十分位。小さいほど人気薄）")
    dec = np.clip((np.argsort(np.argsort(qs)) * 10) // N, 0, 9)
    for k in range(10):
        show(f"第{k+1}十分位", dec == k)

    print("\n" + "=" * 96)
    print("■ ★★容量の見積もり（(b)への最初の答え。★Mは仮定であることを明示する）")
    print("　 自分が S 円入れると、その組の最終オッズは **(1 − S/(M+S)) 倍** ではなく")
    print("　 **元の配当 × (M+S)/M × m/(m+S)** になる。**1点あたりの下落率は S/m が効く**。")
    print(f"　 選んだ組の平均シェア q_pool = {qs.mean():.4f}  ＝ プールの {100*qs.mean():.2f}%")
    print(f"{'プール総額M':>12}{'その組の金額m':>15}{'1点1万円':>12}{'1点5万円':>12}{'1点10万円':>12}")
    for M in (5e6, 1e7, 2e7):
        m_ = M * qs.mean()
        cells = "".join(f"{-100*(1-1/(1+S/m_)):>11.1f}%" for S in (1e4, 5e4, 1e5))
        print(f"{M/1e4:>10.0f}万{m_/1e4:>13.0f}万{cells}")
    print("　 ↑ **配当の下落率**。年2,859点なので1点1万円なら年2,859万円を投じることになる。")
    print("　 ⚠**Mの実測値は持っていない**。JRAの公式発表を1回見れば確定する（Mac作業）。")

    print("\n★読み方（事前登録のとおり）")
    print("  ・**後半だけでCI下端>100%** なら時間で崩れていない。")
    print("  ・層別は **1つの層に集中していたら疑う**。広く出ていれば構造。")
    print("  ・⚠**(a)時間の検証は過去データでは原理的にできない**。T-10分の収集が要る。")


if __name__ == "__main__":
    main()
