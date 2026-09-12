"""(143) ★★★★「何分前なら成り立つか」— **手元の時系列オッズで答えられる範囲を出す**（2026-08-13・ユーザー依頼）

★ユーザーの問い「崩れた場合は何時間前で成り立つか確認して」
　(142)で(141)が時間分割に落ちたので、**締切前のどの時点なら比が成立するか**を知りたい。

⚠★★**まず答えられないことを書く（ここを曖昧にしない）**
　手元の時系列オッズ `data/odds_ts/`（396レース）は **TARGETの「時系列オッズ→単勝」＝単勝だけ**。
　**枠連の板も馬連の板も時系列では持っていない**。
　→ ★**(141)そのもの（馬連プール vs 枠連プール）の時間依存は、過去データでは出せない**。
　　 **これから毎週スナップショットを貯めるしかない**（→ `ml/nk_odds_snap.py`）。

★★答えられること（3つ。すべて331レース＝単勝時系列と両板が揃うもの）
　1. **票数の入り具合** — 発走X分前に**最終票数の何%**が入っているか。**市場がいつ固まるか**の実測。
　2. **単勝の含意確率がX分前と確定でどれだけ違うか** — qの側の陳腐化の大きさ。
　3. ★**(141)と同じ手続きを「X分前のqで選び、確定オッズで払い戻す」形で回す**。
　　 ⚠**qは λ補正Harville→枠**（単勝由来）。**(141)の馬連→枠ではない**。
　　 　(141)でλHarville版は93.9%だったので**ROIの水準は比較にならない**。
　　 　★**見るのは水準ではなく「確定で選んだ集合をX分前にどれだけ復元できるか（一致率）」**。
　　 　 **一致率は組が数千あるので331レースでも精度が出る**。**ROIは参考**。

★★事前登録
　1. 時点は **前日21時 / 当日9時 / 60分前 / 30分前 / 20分前 / 10分前 / 5分前 / 確定**。**後から増やさない**。
　2. **主判定は一致率**（確定で選ばれた組のうち、X分前にも選ばれる割合＝再現率／
　　 X分前に選んだ組のうち確定でも選ばれる割合＝適合率）。
　3. ⚠**331レースではROIのCIは±40pt級**になる。**ROIで判定しない**と先に決めておく。
　4. **予想**: ★**当てにしてよい予想は持っていない**。
　　 ⚠強いて言えば「**10分前ならほぼ一致**」と思うが**類推**（判定基準24）。

実行: python3 ml/audit_overlay_timing.py
"""
import math
import sys

import numpy as np

sys.path.insert(0, "ml")
from audit_capacity_d import mkt_waku_dist
from audit_cond_split import load_boards
from audit_crosspool import PAYBACK, load_races, zq
from audit_crosspool2 import realized
from audit_lbs import build_matrix, fit_lambda
from audit_waku_vs_umaren import load_type
from odds_ts import load_dir, odds_at
from waku_umatan import waku_of

R = PAYBACK["枠連"]
TH = 1.0 / R
WHENS = [("prev", 21, 0), ("day", 9, 0), ("before", 60), ("before", 30),
         ("before", 20), ("before", 10), ("before", 5), ("final",)]
LAB = {("prev", 21, 0): "前日21時", ("day", 9, 0): "当日9時", ("before", 60): "60分前",
       ("before", 30): "30分前", ("before", 20): "20分前", ("before", 10): "10分前",
       ("before", 5): "5分前", ("final",): "確定"}


def main():
    ts = load_dir("data/odds_ts")
    wb, ub = load_boards(), load_type(4, 4)
    races = {r["rid"]: r for r in load_races()}
    P, i1, i2, i3, yrs = build_matrix(list(races.values()), 2015)
    ok3 = i3 >= 0
    l2 = fit_lambda(P, i1, i2)
    print(f"(143) 何分前なら成り立つか（λ2={l2:.3f} 全期間当てはめ・参考値）")
    print("⚠**手元の時系列オッズは単勝だけ**。枠連・馬連の板の時系列は持っていない。")
    print("　 → **(141)そのものの時間依存は過去データでは出せない**。ここで測るのは")
    print("　 　 **qの側（単勝由来）の陳腐化**と**市場がいつ固まるか**\n")

    # ── 1. 票数の入り具合 ──
    print("■ 1. 単勝票数の入り具合（最終を100%として）")
    print(f"{'時点':>10}{'レース':>8}{'票数の割合(中央値)':>20}")
    import pandas as pd
    for w in WHENS:
        vs = []
        for rid, rec in ts.items():
            t = rec["times"]
            fin = rec["votes"][-1]
            if fin <= 0:
                continue
            if w[0] == "final":
                vs.append(1.0)
                continue
            if w[0] == "before":
                cut = rec["post"] - pd.Timedelta(minutes=w[1])
            else:
                day = rec["date"] - pd.Timedelta(days=1) if w[0] == "prev" else rec["date"]
                cut = day + pd.Timedelta(hours=w[1], minutes=w[2])
            idx = np.where(t <= cut)[0]
            if len(idx) == 0:
                continue
            vs.append(rec["votes"][idx[-1]] / fin)
        if vs:
            print(f"{LAB[w]:>10}{len(vs):>8}{np.median(vs):>19.1%}")

    # ── 2〜3. X分前のqで選び、確定オッズで払い戻す ──
    base = {}
    rows = {w: dict(hit=0, cost=0.0, ret=0.0, npt=0, prof=[]) for w in WHENS}
    inter = {w: [0, 0, 0] for w in WHENS}       # [共通, 確定で選んだ, X分前に選んだ]
    for rid, rec in ts.items():
        r = races.get(rid)
        W, U = wb.get(rid), ub.get(rid)
        if r is None or not W or not U or not r.get("wakuren"):
            continue
        rl = realized(r)
        if rl is None:
            continue
        a, b, _ = rl
        n = r["n"]
        nums = [u for u, _, _ in r["horses"]]
        if a not in nums or b not in nums or len(nums) != rec["odds"].shape[1]:
            continue
        key = tuple(sorted((waku_of(a, n), waku_of(b, n))))
        sels = {}
        for w in WHENS:
            o = odds_at(rec, w)
            if o is None or np.isnan(o).any() or (o <= 0).any():
                continue
            p = 1.0 / np.asarray(o, float)
            p = p / p.sum()
            hs = [(u, float(pi), 0) for u, pi in zip(nums, p)]
            md = mkt_waku_dist({"horses": hs, "n": n}, p, l2)
            if not md:
                continue
            keys = [k for k in sorted(md) if k in W]
            if key not in keys or len(keys) < 3:
                continue
            inv = np.array([1.0 / W[k] for k in keys])
            qp = inv / inv.sum()
            qq = np.array([md[k] for k in keys])
            qq /= qq.sum()
            sels[w] = ({keys[i] for i in range(len(keys)) if qq[i] / qp[i] >= TH}, keys)
        if ("final",) not in sels:
            continue
        fin_set = sels[("final",)][0]
        for w, (st, keys) in sels.items():
            inter[w][0] += len(st & fin_set)
            inter[w][1] += len(fin_set)
            inter[w][2] += len(st)
            if not st:
                continue
            c = 100.0 * len(st)
            v = 100.0 * W[key] if key in st else 0.0
            d = rows[w]
            d["cost"] += c
            d["ret"] += v
            d["npt"] += len(st)
            d["hit"] += int(key in st)
            d["prof"].append(v - c)

    print("\n■ 2-3. ★確定のqで選んだ集合を、X分前のqでどれだけ復元できるか")
    print(f"{'時点':>10}{'選んだ点数':>11}{'再現率':>9}{'適合率':>9}"
          f"{'買ったR':>9}{'的中':>6}{'ROI(参考)':>11}{'99%CI(参考)':>22}")
    for w in WHENS:
        c, f_, s = inter[w]
        d = rows[w]
        if d["cost"] <= 0:
            continue
        pr = np.array(d["prof"])
        roi = d["ret"] / d["cost"]
        mc = d["cost"] / len(pr)
        se = pr.std(ddof=1) / math.sqrt(len(pr)) if len(pr) > 1 else float("nan")
        lo, hi = 1 + (pr.mean() - zq(0.01) * se) / mc, 1 + (pr.mean() + zq(0.01) * se) / mc
        ci = "[" + format(100 * lo, ".0f") + "," + format(100 * hi, ".0f") + "]"
        print(f"{LAB[w]:>10}{d['npt']:>11,}{c/max(f_,1):>8.1%}{c/max(s,1):>9.1%}"
              f"{len(pr):>9,}{d['hit']:>6}{100*roi:>10.1f}%{ci:>22}")

    print("\n" + "=" * 96)
    print("★読み方（事前登録のとおり）")
    print("  ・**主判定は再現率／適合率**。ROIは331レースなのでCIが±40pt級＝**判定に使わない**。")
    print("  ・再現率が高い時点までは**qの側は陳腐化していない**。")
    print("  ⚠**q_pool（枠連の板）の側の陳腐化はここでは測れていない**。")
    print("    **枠連は薄いので単勝より遅く固まる可能性がある**。→ `ml/nk_odds_snap.py` で貯める。")


if __name__ == "__main__":
    main()
