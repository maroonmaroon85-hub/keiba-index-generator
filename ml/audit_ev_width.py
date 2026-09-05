"""(173) ★★★点数をレースごとに決める**唯一の理に適った規則**——「期待値が正の組を全部買う」

★★ユーザーの指摘（2026-09-05）
　「**固定3頭とかじゃなくてレースごとに変えるべき。レースごとに荒れる確率が異なるから。
　　固定は理に適ってない**」——**正しい**。**(171)(172)で私が測ったのは全部、恣意的な幅だった**
　（固定3/4/6/8/10頭、あるいは「前比0.70で切る」という**これも恣意的な閾値**）。

★★★**恣意的でない規則は1つだけある**: **`q_k / q_pool,k ≥ 1/払戻率` の組を全部買う**。
　**点数はレースごとに自動で決まる**（荒れるレースでは多く、断然のレースでは0点＝見送り）。
　**数字を1つも手で選んでいない**——**閾値は払戻率から出る**。
　★**これは(141)で枠連117.3%を出した規則そのもの**。

⚠★**(141)との違い＝ここが未測定の理由**:
　**(141)は q を「別のプールの板」から作った**（馬連の板→枠へ厳密集約）。**確定オッズのオラクル**。
　★**本件は q を「我々のMLモデル」から作る**——**発走前に手に入る**。**運用に落とせる形**。
　⚠**(150)は同じことを「単勝オッズ+λHarville」でやって陰性**だったが、
　　**MLモデルの確率は単勝由来の確率とは別物**（(112)の選択の情報などが入っている）。**未測定**。

★★★事前登録（**測る前に書いている**）
　**q の作り方**: **MLモデルの p を正規化 → λ補正Harville → 組の確率**（λは年ごとに学習側で当てはめ）。
　**q_pool**: **その券種の板**（枠連/馬連/三連複）。**手元にある**。
　**買い方**: **比が 1/払戻率 以上の組を全部100円**。**該当0なら見送り**（★**これが可変幅の本体**）。
　**券種3つ**: 枠連(0.775) / 馬連(0.775) / 三連複(0.750)。
　★**主判定**: **各券種のROIの99%CI下端 > 100%**（Bonferroni α=0.01/3）。
　　**加えて 年91%割れ ≤2 と 2021年以降も100%超**（(142)の教訓）。
　⚠**ゲート（判定基準32）**: **同じ枠組みで「q=馬連の板→枠」を測り、(141)の117.3%を±5ptで再現**。
　　**再現しなければ何も読まない**。
　★**併記する**: **1レースあたりの平均点数**（★**可変幅がどれくらい可変か**）と**買ったレースの割合**。
　⚠**裾依存の検算**（(77)の教訓）: **上位3本の配当が全払戻に占める割合**と**前半/後半の分割**。
　　★**これを最初から入れる**——**(172)で後から気づいて崩れたので**。

⚠**予想は持たない**。★**ただし逆風**: **(150)は単勝由来のqで陰性**、**(141)の弱い経路(枠連)は93.9%**。
　**MLの確率が単勝由来より良い保証はない**（(52): AUCを上げると市場の写しになる）。

★★★実測（2026-09-05・26,918レース・λ2=0.8560/λ3=0.7440）
　⚠**ゲート: 対照(馬連板→枠) 112.9% vs (141)の117.3%＝−4.4pt → 立った**
| | 買ったR | 買う率 | ★平均点数 | ROI | 99%CI(Bonf) | 上位3本 | 前半 | 後半 |
|---|---|---|---|---|---|---|---|---|
| ⚠**対照 馬連板→枠** | 13,633 | **50.6%** | ★**1.2** | **112.9%** | [98.2,127.6] | 2.8% | 117.6% | 108.3% |
| ★ML→枠連 | 25,641 | 95.3% | ★**12.3** | ❌**73.2%** | [69.1,77.2] | 0.9% | 73.8% | 72.4% |
| ★ML→馬連 | 26,911 | 100.0% | ★**47.7** | ❌**60.7%** | [56.1,65.3] | 1.5% | 62.6% | 58.5% |
| ★ML→三連複 | 26,911 | 100.0% | ★**192.9** | ❌**57.2%** | [51.0,63.5] | 3.5% | 60.5% | 53.3% |

★★★**数字より「点数の違い」が答え**——**同じ規則・同じ閾値なのに点数が100倍以上違う**。
　★**対照は「ほとんどのレースで+EVの組は無い」と言い、たまに1.2点だけ見つける**
　　＝**正しく較正された q の振る舞い**（板とほぼ一致し、ズレたところだけが意味を持つ）。
　⚠⚠**MLは「ほぼ全レースで十数〜200点が+EV」と言う**＝**エッジが大量にあるのではなく、
　　MLの確率が板から大きくズレている＝較正が粗い**。
　　**規則が+EVと判定した組が実際には57〜73%しか返さない**のがその証拠。

★★**結論: 可変幅の規則は正しい。だが我々のqではまだ使えない**。
　**規則が悪いのではなく、qの精度が足りない**。→ ★**改善の余地は「買い方」ではなく「qの較正」側**。
⚠**裾依存ではない**（上位3本 0.9〜3.5%）。**前半後半も一致**（73.8/72.4%）。**構造的な陰性**。
★**(52)と整合**——「AUCで最適化すると市場の写しになる」が、**組の確率に変換した瞬間に板から大きくズレる**。
　⚠**Harville変換が粗いのか、モデルの確率自体が粗いのかは、この測定では分けられない**。

⚠**λは単勝由来の確率で当てはめてMLの確率に適用している（近似）**。**厳密にやるならMLの確率でλを当てはめ直す**。

実行: python3 ml/audit_ev_width.py
"""
import math
import sys
from itertools import combinations

import numpy as np

sys.path.insert(0, "ml")
import features as F
from audit_cond_split import load_boards
from audit_crosspool import LINE, load_races, payoff, zq
from audit_crosspool2 import realized
from audit_lbs import build_matrix, fit_lambda
from audit_overlay_all import harville_pair, harville_trio, load_board
from train_prod import CAPACITY, add_odds_features, fit_seeds
from waku_umatan import waku_of

NCMP = 3
KNOWN_141, TOL = 117.3, 5.0


def summarize(name, rows, z, npts, nbuy, ntot):
    if not rows:
        print(f"{name:>20}   該当なし")
        return None
    pr = np.array([v - c for _, c, v in rows], float)
    cost = sum(c for _, c, _ in rows); ret = sum(v for _, _, v in rows)
    se = pr.std(ddof=1) / math.sqrt(len(pr)) * len(pr) / cost * 100.0
    roi = 100.0 * ret / cost
    ys = np.array([y for y, _, _ in rows])
    bad = 0
    for u in sorted(set(ys.tolist())):
        if (ys == u).sum() >= 30:
            cc = sum(c for (y, c, _) in rows if y == u)
            vv = sum(v for (y, _, v) in rows if y == u)
            bad += int(vv / max(cc, 1) < 0.91)
    c21 = sum(c for y, c, _ in rows if y >= 2021); v21 = sum(v for y, _, v in rows if y >= 2021)
    gains = np.array([v for _, _, v in rows]); s = np.sort(gains)[::-1]
    mid = np.median(ys)
    fh = ys <= mid
    r1 = 100 * gains[fh].sum() / max(sum(c for (y, c, _) in rows if y <= mid), 1)
    r2 = 100 * gains[~fh].sum() / max(sum(c for (y, c, _) in rows if y > mid), 1)
    mark = "★★買える" if roi - z * se > 100.0 else ""
    print(f"{name:>20}{len(pr):>8,}{100*nbuy/ntot:>7.1f}%{npts:>8.1f}{roi:>8.1f}%"
          f"{f'[{roi-z*se:.1f},{roi+z*se:.1f}]':>19}{bad:>7}{100*v21/max(c21,1):>8.1f}%"
          f"{100*s[:3].sum()/max(gains.sum(),1):>8.1f}%{r1:>8.1f}%{r2:>8.1f}% {mark}")
    return roi


def main():
    MODEL_DIR, PAR = CAPACITY["l2"]
    d = F.to_model(F.load_files()); f = F.build_features(d)
    keep = (f["n_prior"] >= 1) & d["odds"].notna() & (d["odds"] > 0)
    d, f = d[keep].reset_index(drop=True), f[keep].reset_index(drop=True)
    y = (d["finish"] <= 3).astype(int).to_numpy()
    fx, _ = F.encode_categoricals(f)
    fx = add_odds_features(fx, d["odds"].to_numpy(float), d["raceid"].to_numpy())
    cut = d["date"].quantile(0.3)
    tr, te = (d["date"] < cut).to_numpy(), (d["date"] >= cut).to_numpy()
    print("(173) ★点数をレースごとに決める＝「期待値が正の組を全部買う」")
    print(f"　学習 {tr.sum():,} / 検証 {te.sum():,}・分割 {cut.date()}")
    print("★q = **MLモデル → λ補正Harville**（発走前に手に入る）／q_pool = **その券種の板**")
    print("⚠**(141)はqを別プールの板から作った確定オッズのオラクル。本件は運用に落とせる形**\n")
    ms = fit_seeds(fx[tr], y[tr], 3, PAR)
    p_ml = np.mean([m.predict_proba(fx[te])[:, 1] for m in ms], axis=0)
    sub = d.loc[te, ["raceid", "umaban"]].copy(); sub["p"] = p_ml

    # λ を学習側だけで当てはめる（★事前に決めている・年ごとではなく1組に単純化）
    # ⚠**λ は単勝オッズ由来の確率で当てはめ、MLの確率に適用する**——
    # 　**既存コード(audit_lbs.build_matrix)がそう作られているため**。**近似であることを明記する**。
    race_list = load_races()
    cutyear = int(str(cut)[:4])
    P, i1, i2, i3, yrs = build_matrix(race_list, 2013)
    trm = yrs < cutyear
    ok3 = trm & (i3 >= 0)
    l2 = fit_lambda(P[trm], i1[trm], i2[trm])
    l3 = fit_lambda(P[ok3], i1[ok3], i2[ok3], stage3=True, ic=i3[ok3])
    print(f"　λ2={l2:.4f} / λ3={l3:.4f}（**{cutyear}年より前だけで当てはめ**・"
          f"⚠**単勝由来の確率で当てはめてMLの確率に適用＝近似**）\n")

    wb, ub, tb = load_boards(), load_board(4, 4), load_board(7, 6)
    races = {r["rid"]: r for r in race_list}
    KIND = [("枠連", 0.775), ("馬連", 0.775), ("三連複", 0.750)]
    rows = {k: [] for k, _ in KIND}
    rows["対照 馬連板→枠"] = []
    pts = {k: [] for k in list(rows)}
    ntot = 0

    for rid, g in sub.groupby("raceid"):
        r = races.get(str(rid))
        if r is None:
            continue
        rl = realized(r)
        if rl is None:
            continue
        a, b, c = rl
        n, nums0 = r["n"], [u for u, _, _ in r["horses"]]
        if a not in nums0 or b not in nums0 or c not in nums0:
            continue
        gg = g.sort_values("p", ascending=False, kind="mergesort")
        nums = [int(u) for u in gg["umaban"]]
        pv = gg["p"].to_numpy(float)
        if len(nums) < 6:
            continue
        ntot += 1
        pp = pv / pv.sum()
        yy = r["year"]
        pair = harville_pair(pp, l2)
        trio = harville_trio(pp, l2, l3)

        # 枠連: q = ペア確率を枠へ集約
        W = wb.get(str(rid))
        if W and r.get("wakuren"):
            ag = {}
            for (i, j), v in pair.items():
                k2 = tuple(sorted((waku_of(nums[i], n), waku_of(nums[j], n))))
                ag[k2] = ag.get(k2, 0.0) + v
            keys = [k for k in sorted(ag) if k in W]
            if len(keys) >= 3:
                q = np.array([ag[k] for k in keys]); q /= q.sum()
                odds = np.array([W[k] for k in keys])
                inv = 1.0 / odds; qp = inv / inv.sum()
                sel = (q / qp) >= 1.0 / 0.775
                pts["枠連"].append(int(sel.sum()))
                if sel.any():
                    real = tuple(sorted((waku_of(a, n), waku_of(b, n))))
                    v = payoff(r, "枠連(人気順)", [real[0], real[1]]) or 0.0
                    win = np.array([k == real for k in keys])
                    rows["枠連"].append((yy, 100.0 * int(sel.sum()),
                                        float(odds[sel & win].sum() * 100.0) if (sel & win).any() and v > 0 else 0.0))
                # 対照: q = 馬連の板→枠（(141)と同じ）
                U = ub.get(str(rid))
                if U:
                    ag2 = {}
                    for k, o in U.items():
                        if len(k) == 4 and k.isdigit() and o > 0:
                            x, yv = int(k[:2]), int(k[2:])
                            if x in nums and yv in nums and x != yv:
                                k2 = tuple(sorted((waku_of(x, n), waku_of(yv, n))))
                                ag2[k2] = ag2.get(k2, 0.0) + 1.0 / o
                    k2s = [k for k in sorted(ag2) if k in W]
                    if len(k2s) >= 3:
                        q2 = np.array([ag2[k] for k in k2s]); q2 /= q2.sum()
                        o2 = np.array([W[k] for k in k2s]); i2_ = 1.0 / o2; qp2 = i2_ / i2_.sum()
                        s2 = (q2 / qp2) >= 1.0 / 0.775
                        pts["対照 馬連板→枠"].append(int(s2.sum()))
                        if s2.any():
                            real = tuple(sorted((waku_of(a, n), waku_of(b, n))))
                            v = payoff(r, "枠連(人気順)", [real[0], real[1]]) or 0.0
                            w2 = np.array([k == real for k in k2s])
                            rows["対照 馬連板→枠"].append(
                                (yy, 100.0 * int(s2.sum()),
                                 float(o2[s2 & w2].sum() * 100.0) if (s2 & w2).any() and v > 0 else 0.0))
        # 馬連
        U = ub.get(str(rid))
        if U:
            bd = {tuple(sorted((int(k[:2]), int(k[2:])))): o for k, o in U.items()
                  if len(k) == 4 and k.isdigit() and o > 0}
            keys = [k for k in sorted({tuple(sorted((nums[i], nums[j]))): 1 for (i, j) in pair}) if k in bd]
            if len(keys) >= 5:
                pm = {tuple(sorted((nums[i], nums[j]))): v for (i, j), v in pair.items()}
                q = np.array([pm[k] for k in keys]); q /= q.sum()
                odds = np.array([bd[k] for k in keys]); inv = 1.0 / odds; qp = inv / inv.sum()
                sel = (q / qp) >= 1.0 / 0.775
                pts["馬連"].append(int(sel.sum()))
                if sel.any():
                    real = tuple(sorted((a, b)))
                    win = np.array([k == real for k in keys])
                    rows["馬連"].append((yy, 100.0 * int(sel.sum()),
                                        float(odds[sel & win].sum() * 100.0) if (sel & win).any() else 0.0))
        # 三連複
        T = tb.get(str(rid))
        if T:
            bd = {tuple(sorted((int(k[0:2]), int(k[2:4]), int(k[4:6])))): o
                  for k, o in T.items() if len(k) == 6 and k.isdigit() and o > 0}
            tm = {tuple(sorted((nums[i], nums[j], nums[k]))): v for (i, j, k), v in trio.items()}
            keys = [k for k in sorted(tm) if k in bd]
            if len(keys) >= 10:
                q = np.array([tm[k] for k in keys]); q /= q.sum()
                odds = np.array([bd[k] for k in keys]); inv = 1.0 / odds; qp = inv / inv.sum()
                sel = (q / qp) >= 1.0 / 0.750
                pts["三連複"].append(int(sel.sum()))
                if sel.any():
                    real = tuple(sorted((a, b, c)))
                    win = np.array([k == real for k in keys])
                    rows["三連複"].append((yy, 100.0 * int(sel.sum()),
                                         float(odds[sel & win].sum() * 100.0) if (sel & win).any() else 0.0))

    z = zq(0.01 / NCMP)
    print(f"★対象 {ntot:,}レース\n")
    print(f"{'':>20}{'買ったR':>8}{'買う率':>7}{'平均点数':>8}{'ROI':>8}"
          f"{'99%CI(Bonf)':>19}{'年割れ':>7}{'2021-':>8}{'上位3本':>8}{'前半':>8}{'後半':>8}")
    ctrl = summarize("⚠対照 馬連板→枠", rows["対照 馬連板→枠"], z,
                     np.mean(pts["対照 馬連板→枠"]) if pts["対照 馬連板→枠"] else 0,
                     len(rows["対照 馬連板→枠"]), ntot)
    for k, R in KIND:
        summarize(f"★ML→{k}", rows[k], z, np.mean(pts[k]) if pts[k] else 0, len(rows[k]), ntot)

    print("\n" + "=" * 110)
    if ctrl is not None:
        ok = abs(ctrl - KNOWN_141) <= TOL
        print(f"⚠ゲート: 対照 **{ctrl:.1f}%** vs (141)の {KNOWN_141}%　差 {ctrl-KNOWN_141:+.1f}pt"
              f" → **{'★立った' if ok else '⚠⚠立っていない'}**")
        if not ok:
            print("⚠⚠**立っていない。★ML→ の行を読まないこと**（判定基準32）。")
    print("★読み方（**事前登録のとおり**）")
    print("  ★**点数はレースごとに自動で決まる**（買う率<100%＝見送るレースがある）。**恣意的な数字は無い**。")
    print("  ⚠**上位3本の割合が高い／前半後半が割れる場合は、(77)と同じ壊れ方**。**最初から見る**。")
    print("  ★**ML→ が陽性なら、それは(141)と違って『発走前に手に入るq』**＝**運用に落とせる**。")


if __name__ == "__main__":
    main()
