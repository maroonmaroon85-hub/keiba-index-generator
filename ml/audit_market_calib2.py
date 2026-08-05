"""(88b) 市場の較正 第2パス — レース恒等式を壊した形で測り直す。

★第1パス(`audit_market_calib.py`)で判明した構造的な穴:
　市場含意 p をレース内で正規化すると **Σp = 1 = Σwin がレースごとに恒等式**になる。
　したがって「レース単位で定義された区分」(障害/頭数/重賞/芝ダ/距離帯/2歳戦)の
　q̄−p̄ は **必ず厳密に0**。18区分中12個は「較正が完璧」なのではなく **測っていない**。
　同じ理由でランダム疑似区分によるヌル較正(0/500)も無効だった（レース単位で選んだため）。

→ 本パスは恒等式が効かない2つの形で測る:
　【A】区分 × オッズ帯 のセル（レースの一部だけを取るので Σp≠Σwin になる）
　【B】人気順（1番人気/2番人気/…）— レース横断で比較できる馬単位の切り方

────────────────────────────────────────────────────────────────
★測る前に宣言する
────────────────────────────────────────────────────────────────
【A】セル = 区分12 × オッズ帯4 = 48セル。
  判定 (a) diff の CI が0を外れる（Bonferroni α=0.05/48）
       (b) ROI の CI 下端 > 100%（同）
       (c) 前後半で符号一致
       (d) ★maxT 並べ替え検定: レースへの区分ラベルを無作為に付け替えた
           帰無分布の max|z| と観測 max|z| を比べる（(38)の作法。FWER制御）
【B】人気順1〜8位 × 全体、および 1番人気 × 区分12。判定は同じ。
【C】★閾値ずらし: 第1パスで最も大きかった 1.0-1.5倍(+5.89pt / ROI 86.6%) を
     1.0-1.1 / 1.1-1.2 / 1.2-1.3 / 1.3-1.4 / 1.4-1.5 に割る。単一ビン突出なら偽陽性((86))。
【E】★ヌル較正のやり直し: **馬単位**のランダム疑似区分500個で誤検出率を測る。
────────────────────────────────────────────────────────────────

★控除率の基準線について（結論の読み方に必須）
　市場が完璧に較正されていれば、**どう切っても ROI は厳密に 1/S ≈ 79.5%** になる。
　つまり「80%」が較正完璧の線であり、100%が損益分岐。
　ROI が 80% を超えた分が **市場の誤りの大きさ（取り出せる分）**。
　埋めるべき溝は 20pt。**市場の誤りの最大振幅がそれに届くか**が本当の問い。

実行: python3 ml/audit_market_calib2.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "ml")
from audit_market_calib import load_table, prepare  # noqa: E402

NPERM = 1000
RNG = np.random.default_rng(20260805)

BANDS = [("1-3倍", 1.0, 3.0), ("3-10倍", 3.0, 10.0),
         ("10-30倍", 10.0, 30.0), ("30倍超", 30.0, 1e9)]


def race_seg_masks(d):
    """レース単位の区分（レースの全馬が同じ値を取るもの）。"""
    flat = d["jump"] == 0
    return {
        "新馬戦": flat & (d["n_prior"] == 0),
        "障害": d["jump"] == 1,
        "少頭数≤9": flat & (d["fieldsize"] <= 9),
        "多頭数≥16": flat & (d["fieldsize"] >= 16),
        "重賞": flat & (d["grade"] == 1),
        "芝": flat & (d["surface"] == 0),
        "ダート": flat & (d["surface"] == 1),
        "距離≤1400": flat & (d["distance"] <= 1400),
        "距離1401-1800": flat & (d["distance"] > 1400) & (d["distance"] <= 1800),
        "距離1801-2200": flat & (d["distance"] > 1800) & (d["distance"] <= 2200),
        "距離≥2201": flat & (d["distance"] > 2200),
        "2歳戦": flat & (d["age"] == 2),
    }


# ── レース単位クラスタの比推定量の分散（デルタ法・O(n)） ──
def cluster_ratio(num_r, den_r):
    """θ=Σnum/Σden のクラスタ頑健SE。num_r,den_r はレース単位の集計値。"""
    N, A = den_r.sum(), num_r.sum()
    if N <= 0:
        return np.nan, np.nan
    th = A / N
    var = ((num_r - th * den_r) ** 2).sum() / (N ** 2)
    return th, np.sqrt(max(var, 0.0))


def cell_stats(agg, cols):
    """agg: (nR,4) [n, Σwin, Σp, Σpay] → diff と ROI の点推定・SE。"""
    n_r, w_r, p_r, y_r = agg[:, 0], agg[:, 1], agg[:, 2], agg[:, 3]
    diff, se_d = cluster_ratio(w_r - p_r, n_r)
    roi, se_r = cluster_ratio(y_r, n_r)
    return diff, se_d, roi, se_r


def build_race_agg(d, mask, rcodes, nR):
    m = np.asarray(mask)
    out = np.zeros((nR, 4))
    idx = rcodes[m]
    out[:, 0] = np.bincount(idx, minlength=nR)
    out[:, 1] = np.bincount(idx, weights=d["win"].to_numpy()[m], minlength=nR)
    out[:, 2] = np.bincount(idx, weights=d["p"].to_numpy()[m], minlength=nR)
    out[:, 3] = np.bincount(idx, weights=d["pay"].to_numpy()[m], minlength=nR)
    return out


def z_from_ci(alpha):
    from math import sqrt
    # 正規近似の両側分位点
    from statistics import NormalDist
    return NormalDist().inv_cdf(1 - alpha / 2)


def main():
    d = prepare(load_table())
    rcodes, uniq = pd.factorize(d["raceid"], sort=False)
    nR = len(uniq)
    flat = (d["jump"] == 0).to_numpy()

    # 人気順（オッズの小さい順）
    d["pop"] = d.groupby("raceid")["odds"].rank(method="first").astype(int)

    segs = race_seg_masks(d)
    band_masks = {b: ((d["odds"] >= lo) & (d["odds"] < hi)).to_numpy()
                  for b, lo, hi in BANDS}

    NCELL = len(segs) * len(BANDS)
    alpha = 0.05 / NCELL
    zc = z_from_ci(alpha)
    med = d["date"].median()
    first = (d["date"] <= med).to_numpy()

    # 参照: オッズ帯全体の ROI / diff
    print(f"\n{'='*100}")
    print("【基準線】オッズ帯ごとの全体値（完璧に較正されていれば ROI は一律 1/S≒79.5%）")
    print("=" * 100)
    band_ref = {}
    for b, lo, hi in BANDS:
        agg = build_race_agg(d, band_masks[b] & flat, rcodes, nR)
        diff, sd, roi, sr = cell_stats(agg, None)
        band_ref[b] = (diff, roi)
        n = int(agg[:, 0].sum())
        print(f"  {b:<10} {n:>8,}頭  diff {100*diff:>+6.2f}pt  "
              f"ROI {100*roi:>6.2f}%  [{100*(roi-zc*sr):.1f},{100*(roi+zc*sr):.1f}]"
              f"  控除率線80%との差 {100*(roi-1/d['S'].mean()):>+5.2f}pt")

    print(f"\n{'='*100}")
    print(f"【A】区分 × オッズ帯 = {NCELL}セル  Bonferroni α=0.05/{NCELL}={alpha:.5f} (z={zc:.2f})")
    print("=" * 100)
    print(f"{'区分':<16}{'オッズ帯':<9}{'頭数':>8}{'diff':>9}{'(a)':>4}"
          f"{'ROI':>8}{'ROI下端':>9}{'(b)':>4}{'前半/後半diff':>17}{'(c)':>4}")
    print("-" * 100)
    cells, zobs = [], []
    aggs_by_seg = {}
    for sname, smask in segs.items():
        sm = smask.to_numpy()
        for b, lo, hi in BANDS:
            m = sm & band_masks[b]
            if m.sum() < 200:
                continue
            agg = build_race_agg(d, m, rcodes, nR)
            diff, sd, roi, sr = cell_stats(agg, None)
            a1 = build_race_agg(d, m & first, rcodes, nR)
            a2 = build_race_agg(d, m & ~first, rcodes, nR)
            d1 = cell_stats(a1, None)[0]
            d2 = cell_stats(a2, None)[0]
            lo_ci, hi_ci = diff - zc * sd, diff + zc * sd
            rlo = roi - zc * sr
            fa = "○" if (lo_ci > 0 or hi_ci < 0) else "×"
            fb = "○" if rlo > 1.0 else "×"
            fc = "○" if np.sign(d1) == np.sign(d2) else "×"
            cells.append((sname, b, fa, fb, fc, roi))
            zobs.append(abs((roi - band_ref[b][1]) / sr) if sr > 0 else 0.0)
            print(f"{sname:<16}{b:<9}{int(m.sum()):>8,}{100*diff:>+8.2f}pt{fa:>4}"
                  f"{100*roi:>7.1f}%{100*rlo:>8.1f}%{fb:>4}"
                  f"{100*d1:>+8.2f}/{100*d2:>+7.2f}{fc:>4}")
        aggs_by_seg[sname] = int(sm.sum())

    na = sum(c[2] == "○" for c in cells)
    nb = sum(c[3] == "○" for c in cells)
    allp = [c for c in cells if c[2] == "○" and c[3] == "○" and c[4] == "○"]
    print(f"\n  (a)通過 {na}/{len(cells)} / (b)通過 {nb}/{len(cells)} / "
          f"**(a)(b)(c)全通過 {len(allp)}件** {[(c[0],c[1]) for c in allp]}")
    print(f"  最良ROIセル: " + ", ".join(
        f"{c[0]}×{c[1]}={100*c[5]:.1f}%" for c in sorted(cells, key=lambda x: -x[5])[:3]))

    # ── (d) maxT 並べ替え検定 ──
    print(f"\n{'='*100}")
    print(f"(d) maxT 並べ替え検定 — レースへの区分ラベルを無作為化（{NPERM}回・FWER制御）")
    print("=" * 100)
    # 各バンドについて per-race の [n, pay] を用意
    band_agg = {}
    for b, lo, hi in BANDS:
        m = band_masks[b] & flat
        A = build_race_agg(d, m, rcodes, nR)
        band_agg[b] = A[:, [0, 3]]                      # n, pay
    # 区分のレース数（レース単位のラベルなので、レース集合の大きさ）
    seg_nrace = {}
    for sname, smask in segs.items():
        rr = np.zeros(nR, dtype=bool)
        rr[rcodes[smask.to_numpy()]] = True
        seg_nrace[sname] = int(rr.sum())

    null_max = np.zeros(NPERM)
    chunk = 100
    # セルごとの並べ替えROIを貯めて z を作る
    perm_roi = {(s, b): np.zeros(NPERM) for s in segs for b, _, _ in BANDS}
    for st in range(0, NPERM, chunk):
        k = min(chunk, NPERM - st)
        r = RNG.random((k, nR))
        order = np.argsort(r, axis=1)
        for sname in segs:
            ksz = seg_nrace[sname]
            if ksz < 5:
                continue
            Z = np.zeros((k, nR), dtype=np.float32)
            np.put_along_axis(Z, order[:, :ksz], 1.0, axis=1)
            for b, _, _ in BANDS:
                A = band_agg[b]
                num = Z @ A[:, 1].astype(np.float32)
                den = Z @ A[:, 0].astype(np.float32)
                perm_roi[(sname, b)][st:st + k] = np.where(den > 0, num / np.maximum(den, 1e-9), np.nan)
    # z 化してセル横断の max
    zmat = []
    for sname in segs:
        for b, _, _ in BANDS:
            v = perm_roi[(sname, b)]
            if np.all(v == 0) or np.nanstd(v) == 0:
                continue
            zmat.append((sname, b, np.nanmean(v), np.nanstd(v), v))
    Zn = np.vstack([np.abs((m[4] - m[2]) / m[3]) for m in zmat])
    null_max = np.nanmax(Zn, axis=0)
    # 観測 z（同じ帰無SDで基準化）
    obs = []
    for sname, b, mu, sdv, _ in zmat:
        m = segs[sname].to_numpy() & band_masks[b]
        if m.sum() < 200:
            continue
        agg = build_race_agg(d, m, rcodes, nR)
        _, _, roi, _ = cell_stats(agg, None)
        obs.append((abs((roi - mu) / sdv), sname, b, roi))
    obs.sort(reverse=True)
    obs_max = obs[0][0]
    pval = (np.sum(null_max >= obs_max) + 1) / (NPERM + 1)
    print(f"  観測 max|z| = {obs_max:.2f}（{obs[0][1]} × {obs[0][2]}, ROI {100*obs[0][3]:.1f}%）")
    print(f"  帰無 max|z| の 95%点 = {np.percentile(null_max,95):.2f} / 中央値 {np.median(null_max):.2f}")
    print(f"  ★p = {pval:.4f} → {'有意（どこかの区分は本当に違う）' if pval<0.05 else '有意でない。区分間のROI差は偶然の範囲'}")
    print("  上位5セル: " + ", ".join(f"{s}×{b}|z={z:.2f}|ROI{100*r:.1f}%" for z, s, b, r in obs[:5]))

    # ── 【B】人気順 ──
    print(f"\n{'='*100}")
    print("【B】人気順（馬単位・レース横断で比較可能な切り方）")
    print("=" * 100)
    npop = 8
    apop = 0.05 / npop
    zp = z_from_ci(apop)
    print(f"{'人気':<8}{'頭数':>9}{'実現q':>9}{'含意p':>9}{'diff':>9}{'CI':>20}{'ROI':>8}{'ROI CI':>18}")
    for k in range(1, npop + 1):
        m = flat & (d["pop"] == k).to_numpy()
        agg = build_race_agg(d, m, rcodes, nR)
        diff, sd, roi, sr = cell_stats(agg, None)
        print(f"{k}番人気{'':<3}{int(m.sum()):>9,}"
              f"{100*cluster_ratio(agg[:,1],agg[:,0])[0]:>8.2f}%"
              f"{100*cluster_ratio(agg[:,2],agg[:,0])[0]:>8.2f}%"
              f"{100*diff:>+8.2f}pt [{100*(diff-zp*sd):>+6.2f},{100*(diff+zp*sd):>+6.2f}]"
              f"{100*roi:>7.1f}% [{100*(roi-zp*sr):>5.1f},{100*(roi+zp*sr):>5.1f}]")

    print(f"\n  1番人気 × 区分（{len(segs)}セル・α=0.05/{len(segs)}）")
    a2 = 0.05 / len(segs)
    z2 = z_from_ci(a2)
    print(f"{'区分':<16}{'頭数':>9}{'実現q':>9}{'含意p':>9}{'diff':>9}{'ROI':>8}{'ROI下端':>9}")
    best = []
    for sname, smask in segs.items():
        m = smask.to_numpy() & (d["pop"] == 1).to_numpy()
        if m.sum() < 200:
            continue
        agg = build_race_agg(d, m, rcodes, nR)
        diff, sd, roi, sr = cell_stats(agg, None)
        best.append((roi - z2 * sr, sname, roi))
        print(f"{sname:<16}{int(m.sum()):>9,}"
              f"{100*cluster_ratio(agg[:,1],agg[:,0])[0]:>8.2f}%"
              f"{100*cluster_ratio(agg[:,2],agg[:,0])[0]:>8.2f}%"
              f"{100*diff:>+8.2f}pt{100*roi:>7.1f}%{100*(roi-z2*sr):>8.1f}%")
    best.sort(reverse=True)
    print(f"  → ROI下端が最も高い区分: {best[0][1]} 下端{100*best[0][0]:.1f}% "
          f"(100%超え: {'あり' if best[0][0]>1 else '**なし**'})")

    # ── 【C】閾値ずらし: 1.0-1.5倍を割る ──
    print(f"\n{'='*100}")
    print("【C】閾値ずらし — 第1パス最大の 1.0-1.5倍(+5.89pt/ROI86.6%) を細分")
    print("=" * 100)
    fine = [(1.0, 1.1), (1.1, 1.2), (1.2, 1.3), (1.3, 1.4), (1.4, 1.5),
            (1.5, 1.6), (1.6, 1.8), (1.8, 2.0), (2.0, 2.5)]
    a3 = 0.05 / len(fine)
    z3 = z_from_ci(a3)
    print(f"{'区間':<12}{'頭数':>8}{'実現q':>9}{'含意p':>9}{'diff':>9}{'ROI':>8}{'ROI CI':>20}")
    for lo, hi in fine:
        m = flat & ((d["odds"] >= lo) & (d["odds"] < hi)).to_numpy()
        if m.sum() < 50:
            continue
        agg = build_race_agg(d, m, rcodes, nR)
        diff, sd, roi, sr = cell_stats(agg, None)
        print(f"{lo:.1f}-{hi:.1f}{'':<5}{int(m.sum()):>8,}"
              f"{100*cluster_ratio(agg[:,1],agg[:,0])[0]:>8.2f}%"
              f"{100*cluster_ratio(agg[:,2],agg[:,0])[0]:>8.2f}%"
              f"{100*diff:>+8.2f}pt{100*roi:>7.1f}% "
              f"[{100*(roi-z3*sr):>5.1f},{100*(roi+z3*sr):>5.1f}]")

    # ── 【E】ヌル較正のやり直し（馬単位） ──
    print(f"\n{'='*100}")
    print("【E】ヌル較正のやり直し — **馬単位**のランダム疑似区分500個（第1パスの検査は無効だった）")
    print("=" * 100)
    trials, hits_a, hits_b = 500, 0, 0
    zc48 = z_from_ci(0.05 / NCELL)
    for t in range(trials):
        frac = [0.005, 0.02, 0.05, 0.15][t % 4]
        m = flat & (RNG.random(len(d)) < frac)
        agg = build_race_agg(d, m, rcodes, nR)
        diff, sd, roi, sr = cell_stats(agg, None)
        hits_a += (diff - zc48 * sd > 0) or (diff + zc48 * sd < 0)
        hits_b += (roi - zc48 * sr > 1.0)
    print(f"  {trials}個中 (a)誤検出 {hits_a}個 / (b)誤検出 {hits_b}個"
          f"（期待値 {trials*0.05/NCELL:.1f}個）")
    print(f"  → クラスタ頑健SEは{'妥当' if hits_a <= 3*trials*0.05/NCELL + 2 else '★甘い。判定は無効'}")


if __name__ == "__main__":
    main()
