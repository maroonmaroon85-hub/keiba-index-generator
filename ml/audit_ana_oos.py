"""(201) ★★★**前半で決めて、後半で1回だけ試す** — ★選択バイアスを実際に測る

★★**動機（2026-09-06・利用者の指示）**:
　★**「超える想定である程度進めたい」**——**懸念は既に伝えた。前提を受け入れて進める**。
　★★**その前提で最も価値があるのは、「42マスの最大値」という選択バイアスを実測すること**。
　⚠**(200)で分かったこと**: **三連単103.1%は2019年の184%が作っており、
　　2019年を除くと94.3%**。**上位1%のレースが払戻の81%**。
　★**だが「選択バイアスで説明できるか」は、まだ一度も直接測っていない**。

★★★**設計（判定基準4への正面からの回答）**:
　**1. 前半（2016-2020）だけを見て、36通りの買い方から★最良の1つを選ぶ**。
　**2. その1つを固定して、後半（2021-2026）で★1回だけ試す**。
　★★**多重比較が発生しない**——**選ぶのは前半、試すのは後半、判定は1回**。
| ★**後半でも100%超** | ⚠**後半で100%未満** |
|---|---|
| **「たまたま選んだ」では説明できない**。**前提を支持する** | **選択バイアスで説明できる**。**前提は支持されない** |
　★**この表を先に書いておく**（判定基準42）。

★★★**さらに、プラセボで同じ手続きを踏む**（判定基準23・32）:
　★**乱（オッズを揃えた無作為な馬）でも、36通りから前半最良を選び、後半で試す**。
　★★**これが「選択だけで前半→後半がどれだけ縮むか」の物差しになる**。
　⚠**本とプラセボの縮み方が同じなら、本の縮みも選択の産物**。

────────────────────────────────────────────────────────────
★★★ 事前登録（2026-09-06・**結果を見る前にコミットする**）
────────────────────────────────────────────────────────────

■ ★経路（判定基準25）: **q = MLモデル / q_pool = 複勝の板**。⚠**q側は弱い**。
■ ★手続き: **ウォークフォワード**（`data/cache/wf_pred.npz`）。

■ ★★★探索空間（**36通り・前半だけで探す**）
　**推奨度の床 A ∈ {0.15, 0.20} × ズレの床 B ∈ {0.02, 0.04, 0.06}
　　× 紐の頭数 k ∈ {2, 3} × 券種 ∈ {三連単, 三連複, 馬連}**
　**三連単は軸1着固定・紐の順列**（k=2で2点、k=3で6点）。
　**三連複は軸+紐の組合せ**（k=2で1点、k=3で3点）。**馬連は軸−紐**（k=2で2点、k=3で3点）。
　★**選ぶ基準は前半のROIのみ**（**後半は一切見ない**）。

■ ★★★家族A（**2比較**）: **前半最良の構成を、後半で1回だけ試す**
　**① 本（モデル）／② プラセボ（乱・10種平均）**。
　★**主判定: 後半ROIの99%CI下端が100%を超えるか**。
　★**併せて「縮み幅（前半ROI − 後半ROI）」を本とプラセボで比べる**。

■ ★★家族B（**記述・判定しない**）: **36通り全部の前半ROIと後半ROI**。
　★**前半の順位と後半の順位の相関**（**順位が保たれるか**）。
　★**前半上位5つの後半平均**。⚠**これが「選択の縮み」の形を見せる**。

■ ★★ゲート2（判定基準42）——**仮説が偽なら何を返すか**
　★**買い方に情報が無いなら、前半で選んだ構成の後半ROIは、
　　36通りの後半ROIの平均に一致する**（**選択の価値がゼロ**）。
　★**それも記述として出す**（**36通りの後半平均 vs 選ばれた1つの後半**）。

■ ⚠ゲート1: (88)③④を再現（±3pt）／ ★ゲート板: 復元R 0.800±0.020 ／
　★陽性対照: 三連複BOX上位4 81.9%±1.5pt。
■ ★★内部対照（決定的）: **(0.15,0.02)・三連単・k=2 の全期間ROIが 103.1% を ±0.5pt で再現**。

■ ★★★探索を守る（**2比較・Bonferroni α=0.01/2**）
　★★**本体の探索は前半で完結しており、後半の判定は2回だけ**。
　⚠**後半を見てから構成を変えない**——**変えたらこの設計は無意味になる**。
　⚠**標本300レース未満の構成は探索から除く**（判定基準5）。

■ ★★★**事前登録の修正（2026-09-06・★家族Aの結果を一度も見る前に）**
　⚠**第1版は内部対照が落ちた**（**105.0% vs 103.1%**・**28,571R vs 29,437R**）。
　★**原因は母集団**——**(201)は「紐3頭」と「プラセボ4頭」が作れるレースだけを使うので
　　866レース減る**（判定基準25）。★**装置が壊れたのではなく、母集団が違うだけ**。
　⚠**私の内部対照の設計ミス**（**別の母集団の数字を許容±0.5ptで要求していた**）。
　★★**修正: 内部対照を「制限なしの母集団」で別に計算する**——
　　**(0.15,0.02)・三連単k=2 を、紐2頭さえあれば記録する腕を1本足し、
　　それが 103.1% を ±0.5pt で再現するかを見る**。★**装置の同一性を直接測る**。
　★**制限ありの母集団の値（105.0%）は記述として併記し、母集団差として明示する**。
　★**この修正はゲートを見て入れたもので、家族Aの判定は一度も見ていない**（判定基準38）。

■ ★採用条件
　1. **後半ROIの99%CI下端が100%超**
　2. **本の縮み幅が、プラセボの縮み幅より明確に小さい**
　3. **前半と後半で順位相関が正**（**買い方の良し悪しに再現性がある**）
　4. **選ばれた構成の後半が、36通りの後半平均を上回る**

■ 予想（⚠**類推なので当てにしない**・判定基準24）
　⚠**私は(198)で予想を2つとも外した**。**予想は弱い**。
　★**後半は100%を割ると見る**——**(200)で2019年（前半）が全部を作っていたから**。
　★**プラセボも大きく縮むと見る**（**36通りの最大値を選ぶので当然**）。
　★★**注目は「縮み幅の差」**——**本のほうが縮みが小さければ、選択だけではないと言える**。

実行: python3 ml/audit_ana_oos.py    自己テスト: python3 ml/audit_ana_oos.py --selftest
"""
import math
import sys
from itertools import combinations, permutations
from zlib import crc32

import numpy as np

sys.path.insert(0, "ml")
import features as F
from audit_crosspool import load_races, payoff, zq
from audit_ana_odds import MIN_HORSES, gate1, roi_of
from audit_ana_marg import WF_BOX4, WF_TOL, wf_predict
from audit_ana_board import BOARD_R, BOARD_TOL, NPLACE, load_fuku_boards, qpool
from audit_ana_ladder import FINE
from train_prod import add_odds_features

AS = [0.15, 0.20]
BS = [0.02, 0.04, 0.06]
KS = [2, 3]
KINDS = ["三連単", "三連複", "馬連"]
SPLIT = 2021
NSEED = 10
SEED = 20260906
MINCELL = 300
CTRL, KNOWN, ROI_TOL = (0.15, 0.02, 2, "三連単"), 103.1, 0.5
NCMP = 2
ALPHA = 0.01


def bet(kind, ax, himo, k):
    """★軸1頭・紐k頭からの買い目"""
    hs = himo[:k]
    if kind == "馬連":
        return [("馬連", [ax, h]) for h in hs]
    if kind == "三連複":
        return [("三連複", sorted([ax, a, b])) for a, b in combinations(hs, 2)]
    return [("三連単", [ax, a, b]) for a, b in permutations(hs, 2)]


def selftest():
    ok = True
    z = zq(ALPHA / NCMP)
    n = len(AS) * len(BS) * len(KS) * len(KINDS)
    print(f"★探索空間: 推奨度{AS} × ズレ{BS} × 紐{KS}頭 × {KINDS} = **{n}通り**")
    print(f"★★**前半({2016}-{SPLIT-1})だけで選び、後半({SPLIT}-2026)で1回だけ試す**")
    print(f"★判定は {NCMP}回だけ（本 / プラセボ）→ z = {z:.3f}")
    for kind in KINDS:
        for k in KS:
            t = bet(kind, 1, [2, 3, 4], k)
            print(f"　{kind} 紐{k}頭 → **{len(t)}点**　{[x[1] for x in t]}")
    exp = {("馬連", 2): 2, ("馬連", 3): 3, ("三連複", 2): 1, ("三連複", 3): 3,
           ("三連単", 2): 2, ("三連単", 3): 6}
    for (kind, k), v in exp.items():
        got = len(bet(kind, 1, [2, 3, 4], k))
        ok &= got == v
        if got != v:
            print(f"⚠**点数が違う**: {kind} k={k} → {got} (期待 {v})")
    print(f"★点数の自己テスト: {'★全部OK' if ok else '⚠NG'}")
    print("★★読み方: **後半でも100%超→選択では説明できない / "
          "100%未満→選択バイアスで説明できる**")
    print("★★プラセボにも同じ手続きを踏む——**縮み幅の差が本命**")
    print(f"★★内部対照（決定的）: **{CTRL} の全期間ROIが {KNOWN}% ±{ROI_TOL}pt**")
    print("★自己テスト: " + ("全部OK" if ok else "⚠NG"))
    return 0 if ok else 1


def main():
    z = zq(ALPHA / NCMP)
    print("(201) ★★★**前半で決めて、後半で1回だけ試す** — ★選択バイアスを実際に測る")
    print("★★利用者の指示「超える想定である程度進めたい」を受けての設計\n")

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

    CFG = [(a, b, k, kind) for a in AS for b in BS for k in KS for kind in KINDS]
    K = {c: {"a": [], "r": [], "yr": []} for c in CFG}
    full = []          # ★内部対照: 制限なしの母集団での (0.15,0.02) 三連単k=2
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
        pos = {int(u): k for k, u in enumerate(ub)}
        order_p = [int(u) for u in ub[np.argsort(-pv, kind="mergesort")]]
        yr = int(gg["date"].iloc[0].year)
        for a in AS:
            for b in BS:
                cand = np.where((pn >= a) & (gap >= b))[0]
                if not len(cand):
                    continue
                ai = int(cand[int(np.argmax(gap[cand]))])
                axu = int(ub[ai])
                himo = [u for u in order_p if u != axu]
                if (a, b) == CTRL[:2] and len(himo) >= 2:
                    vf = [payoff(r, nm, sel)
                          for nm, sel in bet("三連単", axu, himo, 2)]
                    if not any(v is None for v in vf):
                        full.append(sum(vf) / (100.0 * len(vf)))
                if len(himo) < max(KS):
                    continue
                # ★プラセボ: 軸も紐も同じオッズ(±20%)の無作為な馬に置き換える
                lst, okall = [], True
                for sd in range(NSEED):
                    g2 = np.random.default_rng([SEED + sd, crc32(str(rid).encode())])
                    out = []
                    for u in [axu] + himo[:max(KS)]:
                        k0 = pos[u]
                        pl = [int(ub[q]) for q in range(len(ub))
                              if int(ub[q]) not in out
                              and od[k0] / FINE <= od[q] <= od[k0] * FINE]
                        if not pl:
                            okall = False
                            break
                        out.append(int(g2.choice(pl)))
                    if not okall:
                        break
                    lst.append(out)
                if not okall:
                    continue
                for k in KS:
                    for kind in KINDS:
                        tk = bet(kind, axu, himo, k)
                        va = [payoff(r, nm, sel) for nm, sel in tk]
                        if any(v is None for v in va):
                            continue
                        acc, bad2 = [], False
                        for t in lst:
                            tr = bet(kind, t[0], t[1:], k)
                            vr = [payoff(r, nm, sel) for nm, sel in tr]
                            if any(v is None for v in vr):
                                bad2 = True
                                break
                            acc.append(sum(vr) / (100.0 * len(tr)))
                        if bad2:
                            continue
                        c = K[(a, b, k, kind)]
                        c["a"].append(sum(va) / (100.0 * len(tk)))
                        c["r"].append(float(np.mean(acc)))
                        c["yr"].append(yr)

    med = float(np.median(Rs))
    okR = abs(med - BOARD_R) <= BOARD_TOL
    print(f"\n■ ★ゲート板: 復元R = **{med:.4f}** → **{'★立った' if okR else '⚠⚠落ちた'}**")
    box4 = np.asarray(box4, float)
    g0 = 100.0 * (box4.sum() + 400.0 * len(box4)) / (400.0 * len(box4))
    okb = abs(g0 - WF_BOX4) <= WF_TOL
    print(f"■ ⚠**陽性対照**: BOX上位4 **{g0:.1f}%** vs {WF_BOX4}%"
          f" → **{'★立った' if okb else '⚠⚠落ちた'}**")
    fv = np.asarray(full, float)
    fr = 100.0 * fv.mean()
    okc = abs(fr - KNOWN) <= ROI_TOL
    print(f"■ ★★内部対照（決定的・**制限なしの母集団**）: {CTRL[:2]} 三連単k=2 "
          f"ROI = **{fr:.1f}%** vs {KNOWN}% → **{'★再現' if okc else '⚠⚠ズレた'}**"
          f"（{len(fv):,}R）")
    cv = np.asarray(K[CTRL]["a"], float)
    cr = 100.0 * cv.mean()
    print(f"　★**制限ありの母集団**（紐3頭とプラセボ4頭が作れるレースのみ）: "
          f"**{cr:.1f}%**（{len(cv):,}R）"
          f"　⚠**母集団差 {len(fv)-len(cv):,}R で {cr-fr:+.1f}pt**（判定基準25）")
    if not (okR and okb and okc):
        print("\n⚠⚠**ゲートが落ちた。読まない**（判定基準32）。")
        return

    # ── 前半で選ぶ ──
    tbl = []
    for c in CFG:
        v = np.asarray(K[c]["a"], float)
        rv = np.asarray(K[c]["r"], float)
        yr = np.asarray(K[c]["yr"], int)
        h = yr < SPLIT
        if h.sum() < MINCELL or (~h).sum() < MINCELL:
            continue
        tbl.append({"c": c, "n1": int(h.sum()), "n2": int((~h).sum()),
                    "a1": 100 * v[h].mean(), "a2": 100 * v[~h].mean(),
                    "r1": 100 * rv[h].mean(), "r2": 100 * rv[~h].mean(),
                    "v2": v[~h], "rv2": rv[~h]})
    print(f"\n★探索できた構成 **{len(tbl)}/{len(CFG)}**")

    for who, k1, k2, ky in (("本（モデル）", "a1", "a2", "v2"),
                            ("プラセボ（乱）", "r1", "r2", "rv2")):
        best = max(tbl, key=lambda x: x[k1])
        v = np.asarray(best[ky], float)
        mu = 100 * v.mean()
        se = 100 * v.std(ddof=1) / math.sqrt(len(v))
        lo, hi = mu - z * se, mu + z * se
        avg2 = float(np.mean([x[k2] for x in tbl]))
        print(f"\n{'='*104}")
        print(f"■ ★★★**{who}**")
        print(f"　★**前半で選ばれた構成**: **推奨度≥{best['c'][0]} / ズレ≥{best['c'][1]} / "
              f"紐{best['c'][2]}頭 / {best['c'][3]}**")
        print(f"　　前半 **{best[k1]:.1f}%**（{best['n1']:,}R）")
        print(f"　★★**後半（1回だけの判定）**: **{mu:.1f}%**（{best['n2']:,}R）"
              f"　99%CI [{lo:.1f},{hi:.1f}]"
              f"　→ **{'★★100%超' if lo > 100 else '⚠通らない'}**")
        print(f"　★**縮み幅（前半 − 後半）**: **{best[k1]-mu:+.1f}pt**")
        print(f"　★ゲート2: **36通りの後半平均 {avg2:.1f}%** vs 選ばれた1つ {mu:.1f}%"
              f"　→ **選択の価値 {mu-avg2:+.1f}pt**")

    print(f"\n{'='*104}")
    print("■ ★★家族B（**記述**）: **全構成の前半と後半**（**前半ROIの降順**）")
    print(f"{'推奨度':>7}{'ズレ':>6}{'紐':>4}{'券種':<8}{'点数':>5}{'前半R':>8}{'後半R':>8}"
          f"{'★前半ROI':>10}{'★後半ROI':>10}{'縮み':>9}{'乱前半':>9}{'乱後半':>9}")
    for x in sorted(tbl, key=lambda t: -t["a1"]):
        a, b, k, kind = x["c"]
        npt = len(bet(kind, 1, [2, 3, 4], k))
        print(f"{a:>7.2f}{b:>6.2f}{k:>4}{kind:<8}{npt:>5}{x['n1']:>8,}{x['n2']:>8,}"
              f"{x['a1']:>9.1f}%{x['a2']:>9.1f}%{x['a1']-x['a2']:>+8.1f}pt"
              f"{x['r1']:>8.1f}%{x['r2']:>8.1f}%")
    r1 = np.array([x["a1"] for x in tbl])
    r2 = np.array([x["a2"] for x in tbl])
    from scipy.stats import spearmanr
    rho = spearmanr(r1, r2).statistic
    top5 = sorted(tbl, key=lambda t: -t["a1"])[:5]
    print(f"\n　★**前半と後半の順位相関（スピアマン）: {rho:+.3f}**"
          f"　→ **{'★順位に再現性がある' if rho > 0.3 else '⚠順位は再現しない'}**")
    print(f"　★**前半上位5つの後半平均: {np.mean([x['a2'] for x in top5]):.1f}%**"
          f"（**全構成の後半平均 {r2.mean():.1f}%**）")

    print("\n■ ★★★**事前登録の修正（2026-09-06・★家族Aの結果を一度も見る前に）**
　⚠**第1版は内部対照が落ちた**（**105.0% vs 103.1%**・**28,571R vs 29,437R**）。
　★**原因は母集団**——**(201)は「紐3頭」と「プラセボ4頭」が作れるレースだけを使うので
　　866レース減る**（判定基準25）。★**装置が壊れたのではなく、母集団が違うだけ**。
　⚠**私の内部対照の設計ミス**（**別の母集団の数字を許容±0.5ptで要求していた**）。
　★★**修正: 内部対照を「制限なしの母集団」で別に計算する**——
　　**(0.15,0.02)・三連単k=2 を、紐2頭さえあれば記録する腕を1本足し、
　　それが 103.1% を ±0.5pt で再現するかを見る**。★**装置の同一性を直接測る**。
　★**制限ありの母集団の値（105.0%）は記述として併記し、母集団差として明示する**。
　★**この修正はゲートを見て入れたもので、家族Aの判定は一度も見ていない**（判定基準38）。

■ ★採用条件: **1.後半の下端が100%超 / 2.本の縮みがプラセボより小さい / "
          "3.順位相関が正 / 4.選ばれた後半が平均を上回る**")
    print("⚠**後半を見てから構成を変えない**——**変えたらこの設計は無意味になる**。")
    print("\n⚠**枠連の運用には触れない**。**設定変更は提案しない**。")


if __name__ == "__main__":
    sys.exit(selftest() if "--selftest" in sys.argv else (main() or 0))
