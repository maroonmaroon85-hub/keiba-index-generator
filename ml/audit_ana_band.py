"""(203) ★★★**ズレの「帯」で軸を選ぶ** — ★「大きいほど良い」ではないかもしれない

★★**動機（2026-09-06・利用者の指摘「0.02-0.04で合算するとどう？」）**:
　⚠**単純な合算は意味がない**——**≥0.04 は ≥0.02 の部分集合で、差は147レース(1.1%)しかない**。
　★**意味があるのは「帯で軸を選ぶ」ほう**——**「ズレ最大の馬」をやめて
　　「ズレが [下限,上限) に入る馬」を軸にする**。★**これは一度も測っていない**。

★★★**測る価値がある理由（2つの独立な測定が同じ方向を指している）**:
　★**(184)**: **「優位は極端な穴に集中」は誤りだった**——**実際は中間(5〜35倍)に優位があり、
　　120倍〜はむしろ悪い**（**単勝で値付けしていたことによる人工物だった**）。
　★**(201)**: **床を上げる（＝ズレの大きい方に寄せる）と後半が悪化**（95.8 → 92.0%）。
　→ ★★**「ズレが大きいほど良い」は成り立っていない**。**帯で切れば直接確かめられる**。

────────────────────────────────────────────────────────────
★★★ 事前登録（2026-09-06・**結果を見る前にコミットする**）
────────────────────────────────────────────────────────────

■ ★経路（判定基準25）: **q = MLモデル / q_pool = 複勝の板**（調和平均）。⚠**q側は弱い**。
■ ★手続き: **ウォークフォワード**（`data/cache/wf_pred.npz`）。

■ ★★★**軸の選び方を変える**（**ここが新しい**）
　**従来**: **pn ≥ 0.15 かつ gap ≥ 0.02 を満たす馬のうち ★gap最大**。
　★**今回**: **pn ≥ 0.15 かつ gap が帯 [lo,hi) に入る馬のうち ★pn最大**。
　　⚠**「gap最大」をやめる**——**gapは帯に入っていれば十分で、順位付けはモデルに任せる**。
　**帯: [0.02,0.04) / [0.04,0.06) / [0.06,0.10) / [0.10,0.15) / [0.15,∞)**
　⚠**帯はレース単位では排他ではない**（**1レースに複数の帯の馬がいる**）。
　　★**各帯の買うレース数を必ず併記する**（判定基準25）。
　**紐は従来どおりモデル上位順**（**(198)で三連単はp降順でないと壊れると確認済み**）。

■ ★★★家族A（**5比較**）: ★**どの帯に優位があるか**——**複勝で測る**（**裾が無く分散が小さい**）
　**各帯で「軸の複勝 − 同じオッズ帯の乱の複勝」の対応差**（**乱は10種平均**）。
　★★**(196)で確立した低分散の物差しをそのまま使う**。

■ ★★★家族B（**2比較**）: ★**前半で帯を選び、後半で1回だけ試す**（**(201)と同じ形**）
　**三連単・紐2頭**。**① 本 ② プラセボ（乱）**。
　★**選ぶ基準は前半のROIのみ**。**後半は1回だけ見る**。
　⚠**後半を見てから帯を変えない**。

■ ★★ゲート2（判定基準42）——**仮説が偽なら何を返すか**
　★**帯に情報が無いなら、家族Aの対応差は全帯で0**。
　★**買い方に情報が無いなら、前半で選んだ帯の後半ROIは全帯の後半平均に一致する**。
　⚠**「対照は払戻率を返す」とは書かない**（**(197)で外した**）。**対照は帯のROIを返す**。

■ ⚠ゲート1: (88)③④を再現（±3pt）／ ★ゲート板: 復元R 0.800±0.020 ／
　★陽性対照: 三連複BOX上位4 81.9%±1.5pt。
■ ★★内部対照（**決定的**）: **従来の軸（gap最大・床0.02）で三連単k=2 が 103.1% を ±0.5pt で再現**。
　★**軸の規則が変わるので、従来の腕を別に1本走らせて装置の同一性を測る**。

■ ★★★探索を守る（**7比較・Bonferroni α=0.01/7**）
　⚠**標本300レース未満の帯は判定しない**（判定基準5）。
　★**採用条件に「隣接と同じ向き」**——**孤立した1帯は採らない**。

■ ★記述（判定しない）
　**各帯の 買うレース割合 / 軸の平均オッズ / 軸の平均推奨度 / 軸の平均ズレ /
　　三連単の的中率とROI（前半・後半）/ 複勝ROI**。

■ ★採用条件
　1. **家族Aで、ある帯の対応差が有意に正で、隣接と同じ向き**
　2. ★**その帯が「最大のズレ」ではない**（**＝「大きいほど良い」の否定になる**）
　3. **家族Bで後半ROIが全帯平均を上回る**
　4. **本の縮み幅がプラセボの縮み幅より小さい**（**(201)で満たせなかった条件**）

■ 予想（⚠**類推なので当てにしない**・判定基準24）
　⚠**私は(198)で予想を2つとも外し、(201)でも「後半は100%を割る」以外は当てていない**。
　★**中間の帯（0.04〜0.10）が最良と見る**——**(184)(201)の方向から**。
　⚠**ただし最大の帯 [0.15,∞) は(192)で複勝97.6%を出しており、そちらが良い可能性もある**。
　★**家族Bは後半100%を割ると見る**（**(201)で95.8%が上限だったから**）。

実行: python3 ml/audit_ana_band.py    自己テスト: python3 ml/audit_ana_band.py --selftest
"""
import math
import sys
from itertools import combinations, permutations
from zlib import crc32

import numpy as np

sys.path.insert(0, "ml")
import features as F
from audit_crosspool import load_races, payoff, zq
from audit_ana_odds import BANDS, MIN_HORSES, band_of, gate1, roi_of
from audit_ana_marg import WF_BOX4, WF_TOL, wf_predict
from audit_ana_board import BOARD_R, BOARD_TOL, NPLACE, load_fuku_boards, qpool
from audit_ana_ladder import FINE
from train_prod import add_odds_features

PN_FLOOR = 0.15
GBANDS = [(0.02, 0.04), (0.04, 0.06), (0.06, 0.10), (0.10, 0.15), (0.15, 9.99)]
SPLIT = 2021
NSEED = 10
SEED = 20260906
MINCELL = 300
KNOWN, ROI_TOL = 103.1, 0.5
NCMP = len(GBANDS) + 2      # 5 + 2
ALPHA = 0.01


def tri2(ax, h1, h2):
    return [("三連単", [ax, h1, h2]), ("三連単", [ax, h2, h1])]


def selftest():
    ok = True
    z = zq(ALPHA / NCMP)
    print(f"★帯 {len(GBANDS)}本: " + " / ".join(
        f"[{lo:.2f},{hi:.2f})" if hi < 9 else f"[{lo:.2f},∞)" for lo, hi in GBANDS))
    print(f"★家族A {len(GBANDS)}比較（**複勝の対応差**）＋ 家族B 2比較"
          f"（**前半で選び後半で1回**）→ **{NCMP}比較**・z = {z:.3f}")
    print(f"★★軸の規則を変える: **従来=gap最大 → 今回=帯の中でpn最大**")
    # ★帯が重ならないこと
    for i in range(len(GBANDS) - 1):
        ok &= abs(GBANDS[i][1] - GBANDS[i + 1][0]) < 1e-9
    print(f"★帯の連続性: {'★OK（隙間も重なりも無い）' if ok else '⚠NG'}")
    rng = np.random.default_rng(0)
    n = 120_000
    pay = rng.choice([0.0, 250.0, 900.0], size=n, p=[0.62, 0.30, 0.08])
    m = float(np.mean([(pay[rng.permutation(n)] - pay[rng.permutation(n)]).mean()
                       for _ in range(200)]))
    print(f"★ゲート2（対応差）: **偽なら0** → {m:+.3f}円　"
          f"{'★OK' if abs(m) < 2 else '⚠NG'}")
    ok &= abs(m) < 2
    t = tri2(5, 3, 9)
    print(f"★券の構成: 三連単(軸5・紐3,9) → {[x[1] for x in t]}（**2点**）　"
          f"{'★OK' if len(t) == 2 else '⚠NG'}")
    ok &= len(t) == 2
    print(f"★★内部対照（決定的）: **従来の軸（gap最大・床0.02）で三連単k=2 が "
          f"{KNOWN}% ±{ROI_TOL}pt**")
    print("★★読み方: **中間の帯が最良→「大きいほど良い」の否定 / "
          "最大の帯が最良→従来の設計が正しい**")
    print("★自己テスト: " + ("全部OK" if ok else "⚠NG"))
    return 0 if ok else 1


def main():
    z = zq(ALPHA / NCMP)
    print("(203) ★★★**ズレの「帯」で軸を選ぶ**")
    print("★★軸の規則を変える: **従来=gap最大 → 今回=帯の中でpn最大**\n")

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

    K = {b: {"fa": [], "fr": [], "ta": [], "tr": [], "yr": [],
             "od": [], "pn": [], "gap": []} for b in GBANDS}
    full, Rs, box4, nall = [], [], [], 0
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
        pos = {int(u): k for k, u in enumerate(ub)}
        order_p = [int(u) for u in ub[np.argsort(-pv, kind="mergesort")]]
        yr = int(gg["date"].iloc[0].year)
        # ── ★内部対照: 従来の軸（gap最大・床0.02）──
        c0 = np.where((pn >= PN_FLOOR) & (gap >= 0.02))[0]
        if len(c0):
            a0 = int(c0[int(np.argmax(gap[c0]))])
            h0 = [u for u in order_p if u != int(ub[a0])]
            if len(h0) >= 2:
                vf = [payoff(r, nm, sel) for nm, sel in tri2(int(ub[a0]), h0[0], h0[1])]
                if not any(v is None for v in vf):
                    full.append(sum(vf) / 200.0)
        # ── ★今回: 帯の中で pn 最大 ──
        for b in GBANDS:
            lo, hi = b
            cand = np.where((pn >= PN_FLOOR) & (gap >= lo) & (gap < hi))[0]
            if not len(cand):
                continue
            ai = int(cand[int(np.argmax(pn[cand]))])
            axu = int(ub[ai])
            himo = [u for u in order_p if u != axu]
            if len(himo) < 2:
                continue
            fa = payoff(r, "複勝", [axu])
            ta = [payoff(r, nm, sel) for nm, sel in tri2(axu, himo[0], himo[1])]
            if fa is None or any(v is None for v in ta):
                continue
            # ★乱: 軸と紐を、それぞれ同じオッズ(±20%)の無作為な馬に置き換える
            fr_, tr_, okall = [], [], True
            for sd in range(NSEED):
                g2 = np.random.default_rng([SEED + sd, crc32(str(rid).encode())])
                out = []
                for u in [axu, himo[0], himo[1]]:
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
                # ★複勝の乱は「軸と同じオッズ帯の1頭」（(196)と同じ形）
                plf = [int(ub[q]) for q in range(len(ub)) if bi[q] == bi[ai] and q != ai]
                if not plf:
                    okall = False
                    break
                vf2 = payoff(r, "複勝", [int(g2.choice(plf))])
                vt2 = [payoff(r, nm, sel) for nm, sel in tri2(out[0], out[1], out[2])]
                if vf2 is None or any(v is None for v in vt2):
                    okall = False
                    break
                fr_.append(vf2)
                tr_.append(sum(vt2) / 200.0)
            if not okall:
                continue
            c = K[b]
            c["fa"].append(fa); c["fr"].append(float(np.mean(fr_)))
            c["ta"].append(sum(ta) / 200.0); c["tr"].append(float(np.mean(tr_)))
            c["yr"].append(yr); c["od"].append(float(od[ai]))
            c["pn"].append(float(pn[ai])); c["gap"].append(float(gap[ai]))
    print(f"\n★対象 **{nall:,}レース**")

    med = float(np.median(Rs))
    okR = abs(med - BOARD_R) <= BOARD_TOL
    print(f"■ ★ゲート板: 復元R = **{med:.4f}** → **{'★立った' if okR else '⚠⚠落ちた'}**")
    box4 = np.asarray(box4, float)
    g0 = 100.0 * (box4.sum() + 400.0 * len(box4)) / (400.0 * len(box4))
    okb = abs(g0 - WF_BOX4) <= WF_TOL
    print(f"■ ⚠**陽性対照**: BOX上位4 **{g0:.1f}%** vs {WF_BOX4}%"
          f" → **{'★立った' if okb else '⚠⚠落ちた'}**")
    fv = np.asarray(full, float)
    fr0 = 100.0 * fv.mean()
    okc = abs(fr0 - KNOWN) <= ROI_TOL
    print(f"■ ★★内部対照（決定的・**従来の軸**）: 三連単k=2 ROI = **{fr0:.1f}%** vs "
          f"{KNOWN}% → **{'★再現' if okc else '⚠⚠ズレた'}**（{len(fv):,}R）")
    if not (okR and okb and okc):
        print("\n⚠⚠**ゲートが落ちた。読まない**（判定基準32）。")
        return

    def nm(b):
        lo, hi = b
        return f"[{lo:.2f},{hi:.2f})" if hi < 9 else f"[{lo:.2f},∞)"

    print(f"\n{'='*116}")
    print("■ ★★★家族A: **どの帯に優位があるか**（**複勝・軸1点・乱との対応差**）")
    print(f"\n{'帯':<14}{'R数':>8}{'買う率':>8}{'軸オッズ':>10}{'軸推奨':>8}{'軸ズレ':>8}"
          f"{'★複勝ROI':>10}{'乱':>8}{'★対応差':>10}{'99%CI':>19}{'判定':>12}")
    keepA = []
    for b in GBANDS:
        c = K[b]
        a = np.asarray(c["fa"], float); rv = np.asarray(c["fr"], float)
        if len(a) < MINCELL:
            print(f"{nm(b):<14}{len(a):>8,}　⚠**標本不足**")
            continue
        dd = a - rv
        mu, se = dd.mean(), dd.std(ddof=1) / math.sqrt(len(dd))
        sig = mu - z * se > 0
        if sig:
            keepA.append((b, mu))
        print(f"{nm(b):<14}{len(a):>8,}{100*len(a)/nall:>7.1f}%"
              f"{np.mean(c['od']):>9.1f}倍{np.mean(c['pn']):>8.3f}{np.mean(c['gap']):>8.3f}"
              f"{roi_of(a):>9.1f}%{roi_of(rv):>7.1f}%{mu:>+9.1f}円"
              f"{f'[{mu-z*se:+.1f},{mu+z*se:+.1f}]':>19}"
              f"{('★★有意' if sig else '⚠通らない'):>12}")

    print(f"\n{'='*116}")
    print("■ ★★★家族B: ★**前半で帯を選び、後半で1回だけ試す**（三連単・紐2頭）")
    tbl = []
    for b in GBANDS:
        c = K[b]
        yr = np.asarray(c["yr"], int)
        h = yr < SPLIT
        if h.sum() < MINCELL or (~h).sum() < MINCELL:
            continue
        ta = np.asarray(c["ta"], float); tr = np.asarray(c["tr"], float)
        tbl.append({"b": b, "n1": int(h.sum()), "n2": int((~h).sum()),
                    "a1": 100 * ta[h].mean(), "a2": 100 * ta[~h].mean(),
                    "r1": 100 * tr[h].mean(), "r2": 100 * tr[~h].mean(),
                    "v2": ta[~h], "rv2": tr[~h],
                    "hit": 100 * np.mean(ta > 0)})
    print(f"\n{'帯':<14}{'前半R':>8}{'後半R':>8}{'★前半ROI':>10}{'★後半ROI':>10}"
          f"{'縮み':>9}{'的中率':>8}{'乱前半':>9}{'乱後半':>9}")
    for x in tbl:
        print(f"{nm(x['b']):<14}{x['n1']:>8,}{x['n2']:>8,}{x['a1']:>9.1f}%"
              f"{x['a2']:>9.1f}%{x['a1']-x['a2']:>+8.1f}pt{x['hit']:>7.2f}%"
              f"{x['r1']:>8.1f}%{x['r2']:>8.1f}%")
    for who, k1, k2, ky in (("本（モデル）", "a1", "a2", "v2"),
                            ("プラセボ（乱）", "r1", "r2", "rv2")):
        best = max(tbl, key=lambda x: x[k1])
        v = np.asarray(best[ky], float)
        mu = 100 * v.mean()
        se = 100 * v.std(ddof=1) / math.sqrt(len(v))
        avg2 = float(np.mean([x[k2] for x in tbl]))
        print(f"\n★**{who}**: 前半で選ばれた帯 **{nm(best['b'])}**（前半 {best[k1]:.1f}%）")
        print(f"　★★**後半（1回だけ）**: **{mu:.1f}%**　99%CI [{mu-z*se:.1f},{mu+z*se:.1f}]"
              f"　→ **{'★★100%超' if mu - z*se > 100 else '⚠通らない'}**")
        print(f"　★**縮み {best[k1]-mu:+.1f}pt** ／ **全帯の後半平均 {avg2:.1f}%**"
              f"　→ **選択の価値 {mu-avg2:+.1f}pt**")

    print(f"\n■ ★採用条件: **1.ある帯の対応差が有意で隣接と同じ向き … {len(keepA)}/{len(GBANDS)}"
          " / 2.それが最大のズレの帯でない / 3.後半が全帯平均超 / 4.本の縮みがプラセボより小**")
    if keepA:
        bb = max(keepA, key=lambda t: t[1])
        print(f"　★**対応差が最大の帯: {nm(bb[0])}（{bb[1]:+.1f}円）**"
              f"　→ **{'★★「大きいほど良い」の否定' if bb[0] != GBANDS[-1] else '⚠最大の帯が最良＝従来の設計が正しい'}**")
    print("⚠**後半を見てから帯を変えない**——**変えたらこの設計は無意味になる**。")
    print("\n⚠**枠連の運用には触れない**。**設定変更は提案しない**。")


if __name__ == "__main__":
    sys.exit(selftest() if "--selftest" in sys.argv else (main() or 0))
