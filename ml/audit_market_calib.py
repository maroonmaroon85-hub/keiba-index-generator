"""(88) 市場の較正が崩れる区分を探す — **モデルを一切使わない**。

HANDOFF「次にやること 1-a」。(87)で「モデルと市場の食い違い」は使えないと決着したので、
市場含意確率と実現頻度を直接比べる。モデルが無いので (30)シードノイズ / (53)設定の不確実性 /
(79)プラセボ が原理的に発生しない——この検証の唯一の利点。

────────────────────────────────────────────────────────────────
★測る前に宣言する（(86)の作法。結果を見てから基準を動かさないため）
────────────────────────────────────────────────────────────────
【対象】単勝。市場含意 p_i = (1/odds_i) / Σ_j(1/odds_j)（レース内正規化＝控除率を除去）。
        実現 q_i = 1{1着}。全馬に単勝オッズがあり勝ち馬がちょうど1頭のレースのみ。

【区分】18個を事前宣言（ユーザー指定）。障害以外は平地のみで計算する。
  1 新馬(過去走なし) / 2 障害 / 3 少頭数≤9 / 4 多頭数≥16 / 5 重賞(G1-G3) /
  6-9 オッズ帯 1-3 / 3-10 / 10-30 / 30倍超 /
  10 芝 / 11 ダ / 12-15 距離 ≤1400 / 1401-1800 / 1801-2200 / ≥2201 /
  16 2歳戦 / 17 初芝 / 18 初ダート

【判定】(a)〜(e) を全部通ったものだけ「発見」とする。
  (a) 較正ズレ: diff = q̄ − p̄ の**レース単位クラスタブートストラップCI**が0を外れる。
      多重性は区分数18でBonferroni補正（両側 α=0.05/18 → 99.72%CI）。
  (b) ★儲かるか（別問題）: 正規化は控除率を消しているので、diff>0 でも回収率が100%を
      超えるとは限らない。**必要なのは q̄/p̄ > S**（S=Σ1/odds≈1.25）。
      よって単勝ベタ買い回収率 ROI=mean(odds·win) のBonferroni補正CI下端>100% を別途課す。
      (a)だけ通って(b)が落ちるなら「市場は歪んでいるが取れない」＝入口ではない。
  (c) 期間安定: 前半/後半で diff の符号が一致する。
  (d) ★閾値ずらし((86)の教訓): 順序のある区分(オッズ帯・距離・頭数)は細かい区分に割って
      **連続的か**を見る。単一ビンだけ突出＝不合格。
  (e) ★ヌル較正: 同規模のランダム疑似区分500個に同じ手続きを当て、(a)の誤検出率が
      α近傍であることを確認する。手続き自体が甘ければ全部無効。

【方針】(a)は教科書的な favorite-longshot bias があるのでオッズ帯では通る見込み。
        本番は(b)。(b)が通る区分が1つも無ければ「1-aは否定側で決着」。
────────────────────────────────────────────────────────────────

実行: python3 ml/audit_market_calib.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "ml")
import features as F

SCRATCH = os.environ.get(
    "KEIBA_SCRATCH",
    "/tmp/claude-0/-home-user-keiba-index-generator/"
    "0a3a3d64-e6d0-486d-a283-b1489d9d7266/scratchpad/cache")
NBOOT = 2000
NSEG = 18                      # 事前宣言した区分数（Bonferroniの分母）
ALPHA = 0.05 / NSEG
RNG = np.random.default_rng(20260805)


# ───────────────────────── データ ─────────────────────────
def load_table():
    """モデル用キャッシュは n_prior>=1 で新馬を落とすので使わない。ここで作る。"""
    cache = f"{SCRATCH}/calib.pkl"
    if os.path.exists(cache):
        return pd.read_pickle(cache)
    rawpkl = f"{SCRATCH}/raw.pkl"
    raw = pd.read_pickle(rawpkl) if os.path.exists(rawpkl) else F.load_files()
    if not os.path.exists(rawpkl):
        os.makedirs(SCRATCH, exist_ok=True)
        raw.to_pickle(rawpkl)

    # レース単位の属性（クラス文字列）を id → 文字列で持っておく
    rid = raw[40].str[:-2]
    cls = pd.Series(raw[7].str.strip().values, index=rid.values)
    cls = cls[~cls.index.duplicated()]

    d = F.to_model(raw)
    d["clsname"] = d["raceid"].map(cls).fillna("")
    d["jump"] = d["clsname"].str.contains("障害").astype(int)
    d["grade"] = d["clsname"].str.contains("G1|G2|G3|Ｇ１|Ｇ２|Ｇ３", regex=True).astype(int)

    # 馬ごとの過去走数・芝ダ別の過去走数（リーク無し: 自分より前だけ）
    d = d.sort_values(["horse", "date"], kind="mergesort").reset_index(drop=True)
    g = d.groupby("horse", sort=False)
    d["n_prior"] = g.cumcount()
    d["n_prior_dirt"] = g["surface"].cumsum() - d["surface"]
    d["n_prior_turf"] = d["n_prior"] - d["n_prior_dirt"]
    d.to_pickle(cache)
    return d


def prepare(d):
    """正規化が正しく効くレースだけ残す。"""
    d = d[d["odds"].notna() & (d["odds"] > 0)].copy()
    n_all = d["raceid"].nunique()
    g = d.groupby("raceid")
    nh = g["horse"].transform("size")
    nwin = g["finish"].transform(lambda s: (s == 1).sum())
    ok = (nh == d["fieldsize"]) & (nwin == 1) & (d["fieldsize"] >= 4)
    d = d[ok].copy()
    inv = 1.0 / d["odds"]
    S = inv.groupby(d["raceid"]).transform("sum")
    d["S"] = S
    d["p"] = inv / S
    d["win"] = (d["finish"] == 1).astype(float)
    d["pay"] = d["odds"] * d["win"]
    print(f"レース {n_all:,} → 採用 {d['raceid'].nunique():,}"
          f"（全馬にオッズ・勝ち馬1頭）/ {len(d):,}頭 "
          f"{d['date'].min().date()}〜{d['date'].max().date()} / 平均S={d['S'].mean():.4f}")
    return d.reset_index(drop=True)


# ───────────────────────── 区分 ─────────────────────────
def segments(d):
    """事前宣言した18区分。障害以外は平地(jump==0)のみ。"""
    flat = d["jump"] == 0
    s = {}
    s["01 新馬(初出走)"] = flat & (d["n_prior"] == 0)
    s["02 障害"] = d["jump"] == 1
    s["03 少頭数 ≤9"] = flat & (d["fieldsize"] <= 9)
    s["04 多頭数 ≥16"] = flat & (d["fieldsize"] >= 16)
    s["05 重賞(G1-G3)"] = flat & (d["grade"] == 1)
    s["06 オッズ 1-3倍"] = flat & (d["odds"] < 3)
    s["07 オッズ 3-10倍"] = flat & (d["odds"] >= 3) & (d["odds"] < 10)
    s["08 オッズ 10-30倍"] = flat & (d["odds"] >= 10) & (d["odds"] < 30)
    s["09 オッズ 30倍超"] = flat & (d["odds"] >= 30)
    s["10 芝"] = flat & (d["surface"] == 0)
    s["11 ダート"] = flat & (d["surface"] == 1)
    s["12 距離 ≤1400"] = flat & (d["distance"] <= 1400)
    s["13 距離 1401-1800"] = flat & (d["distance"] > 1400) & (d["distance"] <= 1800)
    s["14 距離 1801-2200"] = flat & (d["distance"] > 1800) & (d["distance"] <= 2200)
    s["15 距離 ≥2201"] = flat & (d["distance"] > 2200)
    s["16 2歳戦"] = flat & (d["age"] == 2)
    s["17 初芝(経験あり)"] = flat & (d["surface"] == 0) & (d["n_prior"] >= 1) & (d["n_prior_turf"] == 0)
    s["18 初ダート(経験あり)"] = flat & (d["surface"] == 1) & (d["n_prior"] >= 1) & (d["n_prior_dirt"] == 0)
    return s


# ───────────────── クラスタブートストラップ ─────────────────
def race_index(d):
    codes, uniq = pd.factorize(d["raceid"], sort=False)
    return codes, len(uniq)


def per_race_stats(d, mask, rcodes, nR):
    """レース単位に集計（クラスタ＝レース）。列: n, Σwin, Σp, Σpay"""
    m = mask.to_numpy()
    out = np.zeros((nR, 4))
    idx = rcodes[m]
    out[:, 0] = np.bincount(idx, minlength=nR)
    out[:, 1] = np.bincount(idx, weights=d["win"].to_numpy()[m], minlength=nR)
    out[:, 2] = np.bincount(idx, weights=d["p"].to_numpy()[m], minlength=nR)
    out[:, 3] = np.bincount(idx, weights=d["pay"].to_numpy()[m], minlength=nR)
    return out


def boot_ci(M, alpha, nboot=NBOOT, chunk=200):
    """M: (nR,4). diff=(Σwin−Σp)/n と ROI=Σpay/n のブートストラップ分布。"""
    nR = M.shape[0]
    diffs, rois = [], []
    for st in range(0, nboot, chunk):
        k = min(chunk, nboot - st)
        cnt = RNG.multinomial(nR, np.full(nR, 1.0 / nR), size=k).astype(np.float64)
        agg = cnt @ M                                  # (k,4)
        n = np.maximum(agg[:, 0], 1e-9)
        diffs.append((agg[:, 1] - agg[:, 2]) / n)
        rois.append(agg[:, 3] / n)
    diffs, rois = np.concatenate(diffs), np.concatenate(rois)
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return (np.percentile(diffs, [lo, hi]), np.percentile(rois, [lo, hi]))


def point(d, mask):
    m = mask.to_numpy()
    n = m.sum()
    if n == 0:
        return dict(n=0)
    return dict(n=int(n), q=d["win"].to_numpy()[m].mean(), p=d["p"].to_numpy()[m].mean(),
                roi=d["pay"].to_numpy()[m].mean(), races=d["raceid"][m].nunique())


# ───────────────────────── 本体 ─────────────────────────
def main():
    d = prepare(load_table())
    rcodes, nR = race_index(d)
    segs = segments(d)

    # 期間の前後半（レース日で二分）
    med = d["date"].median()
    first, second = d["date"] <= med, d["date"] > med

    print(f"\n{'='*104}")
    print("(a)(b)(c) 事前宣言18区分 — 単勝の較正 / Bonferroni α=0.05/18=%.5f" % ALPHA)
    print("=" * 104)
    hdr = (f"{'区分':<20}{'頭数':>9}{'実現q':>8}{'含意p':>8}{'diff':>8}"
           f"{'diff の 99.72%CI':>22}{'ROI':>8}{'ROI の 99.72%CI':>20}  {'前半/後半 diff':>16}")
    print(hdr)
    print("-" * 104)
    results = {}
    for name, mask in segs.items():
        pt = point(d, mask)
        if pt["n"] < 100:
            print(f"{name:<20}{pt['n']:>9,}  （標本不足）")
            continue
        M = per_race_stats(d, mask, rcodes, nR)
        (dlo, dhi), (rlo, rhi) = boot_ci(M, ALPHA)
        p1, p2 = point(d, mask & first), point(d, mask & second)
        d1 = (p1["q"] - p1["p"]) if p1["n"] else float("nan")
        d2 = (p2["q"] - p2["p"]) if p2["n"] else float("nan")
        diff = pt["q"] - pt["p"]
        a = "○" if (dlo > 0 or dhi < 0) else "×"
        b = "○" if rlo > 1.0 else "×"
        c = "○" if (np.sign(d1) == np.sign(d2)) else "×"
        results[name] = dict(diff=diff, roi=pt["roi"], a=a, b=b, c=c,
                             ci=(dlo, dhi), rci=(rlo, rhi), n=pt["n"])
        print(f"{name:<20}{pt['n']:>9,}{100*pt['q']:>7.2f}%{100*pt['p']:>7.2f}%"
              f"{100*diff:>+7.2f}pt  [{100*dlo:>+6.2f},{100*dhi:>+6.2f}]{a}"
              f"{100*pt['roi']:>7.1f}%  [{100*rlo:>5.1f},{100*rhi:>5.1f}]{b}"
              f"   {100*d1:>+6.2f}/{100*d2:>+6.2f}{c}")

    print("\n★(a)較正ズレ有意 / (b)ROI下端>100% / (c)前後半で符号一致")
    passed = [k for k, v in results.items() if v["a"] == "○" and v["b"] == "○" and v["c"] == "○"]
    print(f"　(a)通過: {sum(v['a']=='○' for v in results.values())}件 / "
          f"(b)通過: {sum(v['b']=='○' for v in results.values())}件 / "
          f"**(a)(b)(c)全通過: {len(passed)}件** {passed}")

    # ───────── (d) 閾値ずらし ─────────
    print(f"\n{'='*104}")
    print("(d) 閾値ずらし — 順序区分を細かく割って連続性を見る（単一ビンだけ突出なら偽陽性）")
    print("=" * 104)
    flat = d["jump"] == 0
    ladders = {
        "オッズ": [("1.0-1.5", (1.0, 1.5)), ("1.5-2", (1.5, 2)), ("2-3", (2, 3)), ("3-5", (3, 5)),
                   ("5-7", (5, 7)), ("7-10", (7, 10)), ("10-15", (10, 15)), ("15-20", (15, 20)),
                   ("20-30", (20, 30)), ("30-50", (30, 50)), ("50-100", (50, 100)),
                   ("100-200", (100, 200)), ("200+", (200, 1e9))],
        "距離": [("≤1200", (0, 1200)), ("1201-1400", (1200, 1400)), ("1401-1600", (1400, 1600)),
                 ("1601-1800", (1600, 1800)), ("1801-2000", (1800, 2000)),
                 ("2001-2200", (2000, 2200)), ("2201-2400", (2200, 2400)), ("2401+", (2400, 1e9))],
    }
    for col, bins in ladders.items():
        key = "odds" if col == "オッズ" else "distance"
        print(f"\n[{col}]")
        print(f"{'区間':<12}{'頭数':>10}{'実現q':>9}{'含意p':>9}{'diff':>9}{'ROI':>9}")
        for label, (lo, hi) in bins:
            m = flat & (d[key] >= lo) & (d[key] < hi) if key == "odds" else \
                flat & (d[key] > lo) & (d[key] <= hi)
            pt = point(d, m)
            if pt["n"] < 100:
                continue
            print(f"{label:<12}{pt['n']:>10,}{100*pt['q']:>8.2f}%{100*pt['p']:>8.2f}%"
                  f"{100*(pt['q']-pt['p']):>+8.2f}pt{100*pt['roi']:>8.1f}%")
    print(f"\n[頭数]")
    print(f"{'頭数':<12}{'レース':>10}{'頭数計':>10}{'diff':>9}{'ROI':>9}")
    for fs in range(5, 19):
        m = flat & (d["fieldsize"] == fs)
        pt = point(d, m)
        if pt["n"] < 100:
            continue
        print(f"{fs:<12}{pt['races']:>10,}{pt['n']:>10,}"
              f"{100*(pt['q']-pt['p']):>+8.2f}pt{100*pt['roi']:>8.1f}%")

    # ───────── (e) ヌル較正 ─────────
    print(f"\n{'='*104}")
    print("(e) ヌル較正 — ランダム疑似区分500個で(a)の誤検出率を測る（手続きの妥当性検査）")
    print("=" * 104)
    sizes = [v["n"] for v in results.values()]
    n_all = len(d)
    hits = 0
    trials = 500
    for t in range(trials):
        frac = sizes[t % len(sizes)] / n_all
        # レース単位でランダムに選ぶ（区分は多くがレース属性なのでクラスタを揃える）
        pick_r = RNG.random(nR) < frac
        m = pd.Series(pick_r[rcodes], index=d.index)
        if m.sum() < 100:
            continue
        M = per_race_stats(d, m, rcodes, nR)
        (dlo, dhi), _ = boot_ci(M, ALPHA, nboot=400)
        hits += (dlo > 0 or dhi < 0)
    print(f"　ランダム疑似区分 {trials}個中 **{hits}個** が(a)を通過 "
          f"（期待値 {trials*ALPHA:.1f}個 = α×試行数）")
    print(f"　→ 手続きは{'妥当（誤検出率が名目通り）' if hits <= 3*trials*ALPHA + 2 else '★甘い。判定は無効'}")


if __name__ == "__main__":
    main()
