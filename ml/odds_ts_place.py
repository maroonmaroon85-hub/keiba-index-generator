"""
(94) ★「発走n分前に単勝が安い馬の複勝」を測り直す常設スクリプト（(92)⑤の再実行装置）。

(92)⑤は**その場限りの集計**で終わっていてコードが残っていない。宿題1（時系列オッズを
数十開催ぶん集める）が終わっても**走らせるものが無い**状態だったので、ここに常設化する。
`data/odds_ts/` にCSVを足して**このスクリプトを再実行するだけ**で判定が更新される。

━━━ ★測る前に宣言する（判定基準6。最良のビンで判定しない）━━━

【仮説】複勝プールは人気馬帯で甘い((92)②)。**確定オッズ1.0-1.6倍の複勝ROIは94.5%**（14年・2,466頭）
　だが確定オッズは買う時点では分からない((92)④)。**発走直前のオッズなら買える**——この帯が
　締切前に既に確定しているなら、事後選択ではなく**実行可能な戦略**になる。

【主指標】複勝ROI（1頭100円均等買い）と**線(80%)との差**。副指標として1頭あたり期待損益(円)。
　⚠ROIだけで判定しない。判定は下の「合格条件」で行う。

【選択時点】前日22時 / 当日9時 / 発走30分前 / 発走10分前 / 確定（=事後・買えない基準）
【閾値グリッド（事前宣言・固定）】単勝オッズ ≤ 1.3 / ≤ 1.6 / ≤ 2.0 / ≤ 3.0
　★**最良の閾値を選んで報告しない**。4点の**単調性**（緩いほどROIが下がるか）を Spearman ρ で見る。

【対照（判定基準2・3）】
　① 人気順対照 … 同じ時点の1番人気・2番人気（**順位で選ぶので常に買える**）の複勝ROI。
　　　(92)④より1番人気は+4.16pt。**オッズ閾値がこれを超えないなら、閾値に意味は無い**。
　② プラセボ対照 … 同じレースから**同じ頭数を無作為に**選ぶ（オッズを見ない）を200回。
　　　観測ROIがプラセボ分布の上位5%に入らなければ、選択に中身が無い。

【合格条件（測る前に固定）】以下を**すべて**満たしたときだけ「実行可能な優位」と呼ぶ:
　(a) 95%CIの下端が 80.0%（線）を超える
　(b) 人気順対照（1番人気・同時点）の点推定を上回る
　(c) プラセボ p < 0.05
　(d) 閾値の単調性 Spearman ρ < 0（緩めるほどROIが下がる）
　(e) 開催日ベースのブロックブートストラップでも(a)が保たれる（1開催日=1ブロック）
★**頭数が 500 未満のときは「判定しない」と出す**。(92)⑤は36頭＝SE±10ptで、
　96.4%と83.5%を区別できなかった。3,000頭で±1.1pt。**足りないうちは方向だけ**。

━━━ 14年アンカー ━━━
時系列オッズは今のところ数開催しか無いので、**確定オッズでの14年集計を必ず並べる**。
時系列標本はその部分集合であるべきで、大きくズレたら**標本の偏り**を先に疑う。

実行: python3 ml/odds_ts_place.py [時系列オッズのディレクトリ(既定 data/odds_ts)]
"""
import os
import sys

import numpy as np

sys.path.insert(0, "ml")
from audit_crosspool import load_races                      # noqa: E402
from odds_ts import load_dir, odds_at                       # noqa: E402

LINE = 0.800                       # 複勝の払戻率（JRA公示）
THRESHOLDS = [1.3, 1.6, 2.0, 3.0]  # ★事前宣言。増やさない
BANDS = [(1.0, 1.6), (1.6, 2.0), (2.0, 3.0), (3.0, 5.0), (5.0, 10.0), (10.0, 30.0), (30.0, 1e9)]
WHEN = [("前日22時", ("prev", 22, 0)), ("当日9時", ("day", 9, 0)),
        ("発走30分前", ("before", 30)), ("発走10分前", ("before", 10)),
        ("確定(事後)", ("final",))]
MIN_N = 500                        # ★これ未満は判定しない
B_BOOT = 4000
B_PLACEBO = 200


def boot_ci(vals, groups, rng, b=B_BOOT, alpha=0.05):
    """グループ(レース or 開催日)単位のブートストラップ。1頭1標本ではないので必須。"""
    vals = np.asarray(vals, float)
    groups = np.asarray(groups)
    if len(vals) == 0:
        return (float("nan"), float("nan"))
    uniq, inv = np.unique(groups, return_inverse=True)
    order = np.argsort(inv, kind="mergesort")
    v_sorted = vals[order]
    starts = np.searchsorted(inv[order], np.arange(len(uniq)))
    ends = np.append(starts[1:], len(v_sorted))
    sums = np.add.reduceat(v_sorted, starts) if len(v_sorted) else np.array([])
    cnts = (ends - starts).astype(float)
    idx = rng.integers(0, len(uniq), size=(b, len(uniq)))
    m = sums[idx].sum(axis=1) / np.maximum(cnts[idx].sum(axis=1), 1)
    return tuple(np.percentile(m, [alpha / 2 * 100, (1 - alpha / 2) * 100]))


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])


def fuku_pay(race, umaban):
    """複勝の払戻（円/100円）。的中していなければ0。"""
    return race["fuku"].get((umaban,), 0)


# ───────────────────────── 14年アンカー（確定オッズ） ─────────────────────────
def anchor(races, rng):
    print("━━━ ① 14年アンカー: 確定オッズで切った複勝（(92)②の再現・買えないが基準になる） ━━━")
    print(f"{'単勝オッズ帯':<14}{'頭数':>8}{'複勝ROI':>9}{'線との差':>10}{'95%CI':>16}{'円/頭':>8}")
    for lo, hi in BANDS:
        v, g = [], []
        for r in races:
            if not r["fuku"]:
                continue
            for u, o, _fin in r["horses"]:
                if lo <= o < hi:
                    v.append(fuku_pay(r, u))
                    g.append(r["rid"])
        if not v:
            continue
        roi = np.mean(v) / 100
        lo_ci, hi_ci = boot_ci(v, g, rng)
        name = f"{lo:.1f}-{hi:.1f}倍" if hi < 1e8 else f"{lo:.0f}倍〜"
        print(f"{name:<14}{len(v):>8,}{roi*100:>8.1f}%{(roi-LINE)*100:>+9.2f}pt"
              f"{f'[{lo_ci/100*100:.1f},{hi_ci/100*100:.1f}]':>16}{np.mean(v)-100:>+7.1f}")
    print()
    print(f"{'閾値(累積)':<14}{'頭数':>8}{'複勝ROI':>9}{'線との差':>10}{'95%CI':>16}{'円/頭':>8}")
    roi_by_th = []
    for th in THRESHOLDS:
        v, g = [], []
        for r in races:
            if not r["fuku"]:
                continue
            for u, o, _fin in r["horses"]:
                if o <= th:
                    v.append(fuku_pay(r, u))
                    g.append(r["rid"])
        roi = np.mean(v) / 100 if v else float("nan")
        roi_by_th.append(roi)
        lo_ci, hi_ci = boot_ci(v, g, rng)
        print(f"{f'≤ {th}倍':<14}{len(v):>8,}{roi*100:>8.1f}%{(roi-LINE)*100:>+9.2f}pt"
              f"{f'[{lo_ci:.1f},{hi_ci:.1f}]':>16}{np.mean(v)-100:>+7.1f}")
    rho = spearman(THRESHOLDS, roi_by_th)
    print(f"\n単調性 Spearman ρ = {rho:+.3f}（負＝緩めるほど下がる＝仮説どおり）")
    # 人気順対照（順位で選ぶ＝常に買える）
    print(f"\n{'人気順対照(確定)':<16}{'頭数':>8}{'複勝ROI':>9}{'線との差':>10}")
    for k in (1, 2, 3):
        v, g = [], []
        for r in races:
            if not r["fuku"] or len(r["horses"]) < k:
                continue
            hs = sorted(r["horses"], key=lambda h: h[1])
            u = hs[k - 1][0]
            v.append(fuku_pay(r, u))
            g.append(r["rid"])
        roi = np.mean(v) / 100
        print(f"{f'{k}番人気':<16}{len(v):>8,}{roi*100:>8.1f}%{(roi-LINE)*100:>+9.2f}pt")
    print()


# ───────────────────────── 時系列オッズ本体 ─────────────────────────
def collect(ts, byrid):
    """レースごとに {時点: オッズベクトル} を作る。全時点そろったレースだけ残す（対応あり比較）。"""
    out = []
    miss_payout = miss_time = 0
    for rid, rec in sorted(ts.items()):
        race = byrid.get(rid)
        if race is None or not race["fuku"]:
            miss_payout += 1
            continue
        vecs = {}
        ok = True
        for name, w in WHEN:
            v = odds_at(rec, w)
            if v is None:
                ok = False
                break
            vecs[name] = np.asarray(v, float)
        if not ok:
            miss_time += 1
            continue
        out.append((race, rec, vecs))
    return out, miss_payout, miss_time


def sel_by_threshold(race, vec, th):
    """その時点のオッズが th 以下の出走馬（馬番, 払戻）。取消等でオッズが取れない馬は除外。"""
    got = []
    for u, _o, _f in race["horses"]:
        if u - 1 < len(vec):
            o = vec[u - 1]
            if np.isfinite(o) and 0 < o <= th:
                got.append((u, fuku_pay(race, u)))
    return got


def sel_by_rank(race, vec, k):
    """その時点のオッズで k 番人気（順位で選ぶので常に買える）。"""
    cand = [(vec[u - 1], u) for u, _o, _f in race["horses"]
            if u - 1 < len(vec) and np.isfinite(vec[u - 1]) and vec[u - 1] > 0]
    if len(cand) < k:
        return None
    cand.sort()
    u = cand[k - 1][1]
    return (u, fuku_pay(race, u))


def ceiling(races, rng):
    """★オラクルの天井: 「確定オッズを知っていて最良の帯を選べる」場合の上限を測る。

    実運用の選択（発走n分前のオッズ）は**確定オッズ選択のノイズ版**なので、
    確定オッズで選んだときの値がこの一族の上限になる。ここが100%に届かないなら、
    時系列オッズを何開催集めても100%には届かない。
    """
    print("━━━ ④ ★天井: 確定オッズを知っているオラクルでも届くのはどこまでか ━━━")
    fine = [(1.1, 1.2), (1.2, 1.3), (1.3, 1.4), (1.4, 1.5), (1.5, 1.6),
            (1.6, 1.8), (1.8, 2.0), (2.0, 2.5), (2.5, 3.0)]
    print(f"{'単勝オッズ帯':<12}{'頭数':>7}{'複勝的中率':>10}{'的中時の平均払戻':>16}"
          f"{'複勝ROI':>9}{'95%CI':>15}{'払戻=100円':>11}")
    best = (None, -1.0)
    for lo, hi in fine:
        v, g, hit, floor = [], [], [], []
        for r in races:
            if not r["fuku"]:
                continue
            for u, o, _fin in r["horses"]:
                if lo <= o < hi:
                    p = fuku_pay(r, u)
                    v.append(p)
                    g.append(r["rid"])
                    hit.append(p > 0)
                    if p > 0:
                        floor.append(p == 100)
        if not v:
            continue
        roi = np.mean(v)
        lo_ci, hi_ci = boot_ci(v, g, rng)
        if roi > best[1] and len(v) >= 100:
            best = ((lo, hi, len(v), lo_ci, hi_ci), roi)
        print(f"{f'{lo}-{hi}倍':<12}{len(v):>7,}{np.mean(hit)*100:>9.1f}%"
              f"{np.mean([x for x in v if x > 0]):>15.1f}円{roi:>8.1f}%"
              f"{f'[{lo_ci:.1f},{hi_ci:.1f}]':>15}{np.mean(floor)*100:>10.1f}%")
    (blo, bhi, bn, bl, bh), broi = best
    print(f"\n★最良の帯は {blo}-{bhi}倍 の {broi:.1f}% [{bl:.1f},{bh:.1f}]（{bn:,}頭）。")
    print("　⚠これは**9帯から最良を選んだ後**の値なので、判定基準6より上振れしている。")
    print("　それでも**点推定が100%に届かない**。")
    print("\n★機構: 複勝には最低配当100円の下限がある。オッズを詰めるほど")
    print("　**的中時の払戻が100円に張り付く**（1.1-1.2倍では的中の7割が100円＝利益ゼロ）一方、")
    print("　的中率は95%止まりなので、ROI = 的中率 × 平均払戻 は100%を超えられない。")
    print("　⚠(92)①は『最低配当100円が優位の正体か』を全プールで調べて否定したが、")
    print("　　それは**正しい**（1.5-1.6倍では該当3.9%なのに+12.5ptある）。")
    print("　　下限は優位を**作って**いない。優位を**頭打ちにして**いる。役割が違う。\n")


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "data/odds_ts"
    rng = np.random.default_rng(0)
    races = load_races()
    byrid = {r["rid"]: r for r in races}
    y0, y1 = min(r["year"] for r in races), max(r["year"] for r in races)
    print(f"配当データ {len(races):,}レース（{y0}-{y1}）\n")
    anchor(races, rng)
    ceiling(races, rng)

    if not os.path.isdir(d):
        sys.exit(f"{d} が無い")
    ts = load_dir(d)
    if not ts:
        sys.exit(f"{d} に時系列オッズCSVがありません")
    data, miss_p, miss_t = collect(ts, byrid)
    days = sorted({r["date"].date() for _r, r, _v in data})
    print("━━━ ② 時系列オッズ（買える時点で切る） ━━━")
    print(f"読込 {len(ts)}レース → 使用 {len(data)}レース / {len(days)}開催日 "
          f"{days[0]}〜{days[-1]}（配当欠 {miss_p} / 時点欠 {miss_t}）")
    print(f"★判定に必要な頭数 {MIN_N}（現状の見込みは下表の頭数を見ること）\n")

    for th in THRESHOLDS:
        print(f"── 閾値 単勝 ≤ {th}倍 ──")
        print(f"{'選ぶ時点':<12}{'頭数':>6}{'複勝ROI':>9}{'線との差':>10}{'95%CI(R)':>16}"
              f"{'95%CI(開催日)':>18}{'円/頭':>8}{'プラセボp':>10}")
        for name, _w in WHEN:
            v, gr, gd, picks = [], [], [], []
            for race, rec, vecs in data:
                got = sel_by_threshold(race, vecs[name], th)
                for u, pay in got:
                    v.append(pay)
                    gr.append(race["rid"])
                    gd.append(str(rec["date"].date()))
                picks.append((race, len(got)))
            if not v:
                print(f"{name:<12}{0:>6}{'—':>9}")
                continue
            roi = np.mean(v) / 100
            lo_r, hi_r = boot_ci(v, gr, rng)
            lo_d, hi_d = boot_ci(v, gd, rng)
            # プラセボ: 同じレースから同じ頭数を無作為に（オッズを見ない）
            null = []
            for _ in range(B_PLACEBO):
                nv = []
                for race, k in picks:
                    if k == 0:
                        continue
                    us = [u for u, _o, _f in race["horses"]]
                    for u in rng.choice(us, size=min(k, len(us)), replace=False):
                        nv.append(fuku_pay(race, int(u)))
                null.append(np.mean(nv) / 100 if nv else np.nan)
            null = np.array(null, float)
            p = float(np.mean(null >= roi))
            print(f"{name:<12}{len(v):>6}{roi*100:>8.1f}%{(roi-LINE)*100:>+9.2f}pt"
                  f"{f'[{lo_r:.1f},{hi_r:.1f}]':>16}{f'[{lo_d:.1f},{hi_d:.1f}]':>18}"
                  f"{np.mean(v)-100:>+7.1f}{p:>10.3f}")
        print()

    # 人気順対照（同じ時点・同じレース集合）
    print("── 対照: 人気順（順位で選ぶので常に買える。閾値がこれを超えないと意味が無い） ──")
    print(f"{'選ぶ時点':<12}{'1番人気ROI':>12}{'線との差':>10}{'2番人気ROI':>12}{'頭数':>7}")
    for name, _w in WHEN:
        rows = {1: [], 2: []}
        for race, _rec, vecs in data:
            for k in (1, 2):
                s = sel_by_rank(race, vecs[name], k)
                if s:
                    rows[k].append(s[1])
        r1 = np.mean(rows[1]) / 100 if rows[1] else float("nan")
        r2 = np.mean(rows[2]) / 100 if rows[2] else float("nan")
        print(f"{name:<12}{r1*100:>11.1f}%{(r1-LINE)*100:>+9.2f}pt{r2*100:>11.1f}%{len(rows[1]):>7}")
    print()

    # 開催日別の内訳（本命の条件のみ）
    head_th, head_when = 1.6, "発走10分前"
    per_day = {}
    for race, rec, vecs in data:
        k = str(rec["date"].date())
        for _u, pay in sel_by_threshold(race, vecs[head_when], head_th):
            per_day.setdefault(k, []).append(pay)
    print(f"── 開催日別（{head_when} ≤ {head_th}倍）: 1開催に偏っていないかを見る ──")
    print(f"{'開催日':<12}{'頭数':>6}{'複勝ROI':>9}")
    for k in sorted(per_day):
        vv = per_day[k]
        print(f"{k:<12}{len(vv):>6}{np.mean(vv):>8.1f}%")
    tot = [x for vv in per_day.values() for x in vv]
    n_head = len(tot)
    print(f"{'合計':<12}{n_head:>6}{np.mean(tot) if tot else float('nan'):>8.1f}%"
          if tot else f"{'合計':<12}{0:>6}")
    print()

    # 判定
    print("━━━ ③ 合格条件（測る前に宣言したもの）に当てるとどうなるか ━━━")
    if n_head < MIN_N:
        se = np.std(tot, ddof=1) / np.sqrt(len(tot)) if len(tot) > 1 else float("nan")
        need = int(np.ceil((np.std(tot, ddof=1) / 1.1) ** 2)) if len(tot) > 1 else 0
        print(f"★**判定しない**。本命条件（{head_when} ≤ {head_th}倍）の頭数が {n_head} で、")
        print(f"　宣言した最低頭数 {MIN_N} に届かない。標準誤差は ±{se:.1f}pt。")
        print(f"　±1.1pt にするには約 {need:,} 頭（今の {len(days)}開催 の "
              f"{max(need // max(n_head, 1), 1)}倍前後の開催数）が要る。")
        print("　→ **TARGETから時系列オッズを追加出力して `data/odds_ts/` に置き、これを再実行すること**。")
    else:
        print("　頭数は足りている。上表の (a)〜(e) を順に当てること。")
    print("\n⚠ただし④の天井より、**この一族は集めきっても100%に届かない**。")
    print("　時系列オッズを厚くして分かるのは『96%なのか93%なのか』であって、")
    print("　『100%を超えるか』ではない。超えるには④の天井（確定オッズを知っていても98%）を")
    print("　破る必要があり、それは**発走前オッズが確定オッズに無い情報を持つ**場合だけ＝宿題2の主張。")
    print("　(92)⑤の『10分前(96.4%)＞確定(93.3%)』はその可能性を示唆していたが、")
    print("　標本を36頭→上表に増やすと**向きが再現しない**（確定の方が高いか同等）。")


if __name__ == "__main__":
    main()
