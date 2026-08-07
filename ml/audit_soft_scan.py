"""(110) 全組み合わせから「期待払戻が最小の組」を選ぶ（NEXT_SESSION 6-A-1）＋床の外側の帯（6-A-2）。

★★測る前の解析的整理（これで仕事の中身が変わる）
　期待払戻 E = 払戻率 / q なので、**E が最小 ⇔ q が最大**。
　λ補正Harville の q は、構成する各馬の p について**単調増加**である
　（g_pair_ordered = p_x·w_y/(W−w_x)、w=p^λ は p に単調。三連系も積の形で同じ）。
　→ **複勝・馬連・ワイド・馬単・三連複・三連単の argmax q は「人気上位N頭」そのもの**。
　　 つまり (105)(106) が測った「人気上位N頭に固定した組」は、**既に全組み合わせの最小E組だった**。
　　 NEXT_SESSION 6-A-1 が期待した「走査すればもっと甘い点がある」は、この6券種では**原理的に空**。
　→ 唯一の例外が **枠連**。枠は複数頭を合算するので、
　　 「上位2頭の枠」＝「最大確率の枠ペア」とは限らない。**走査に意味があるのは枠連だけ**。
　★そして枠連は (105)② の表に**唯一入っていない券種**（combo_q が枠連を返さないため）。
　　 運用の本命券種でありながら、期待払戻での層別が**一度も測られていない**。ここが本当の穴。

★★枠連は「床の外側」にある — だからこれは (106) の天井論の直球テストになる
　(106)② の結論は「複勝96.8%は最低配当100円の床が押し上げた値で、的中率92%が頭打ち」。
　枠連の最大 q はせいぜい 0.2〜0.45 なので E = 0.775/q は**常に 170円以上＝床の外**。
　→ 枠連で裾を詰めて甘さが伸びるなら「優位は床の産物ではない」。伸びないなら
　　 **「(105)の甘さは床が作っていた」**が確定し、単勝オッズだけの道は完全に閉じる。

★★事前登録（測る前に宣言。後から動かさない）
　【切り方】λ補正Harville の q（単勝オッズだけから発走前に計算できる）。λ はウォークフォワード
　　　　　　（各年を**それ以前の年だけ**で推定）。事後選択にならない。
　【本命】枠連。副次（記述のみ）: 複勝の期待払戻帯（6-A-2）。
　【判定】
　　J1 単調性: 枠連の6分位で Spearman ρ ≤ −0.6 なら「(105)の機構は枠連にも及ぶ」。
　　J2 到達点: 裾 20/10/5/2/1% のいずれかで **99%CI の下端が 100% を上回る**こと。
　　　　　　　 点推定では判定しない。多重性は 6分位+5裾=11セル → Bonferroni も併記。
　　J3 年代分割: 前半・後半とも線を上回ること。片方で崩れたら採らない。
　　J4 プラセボ: 払戻を(頭数,年)層内でシャッフルすると ρ と最下位層の差が消えること。
　　J5 交絡対照: 「1番人気の単勝オッズ」で層別しても同じ強さが出るなら、E である必要はない
　　　　　　　　（＝(105)の主張を弱める）。E の方が強いことを確認する。
　　J6 交絡対照2: 枠連の max-q 選択 vs 人気順上位2頭の枠。走査に意味があるかを直接見る。
　【予想（外れたらそう書く）】
　　・枠連も単調に出る（ρ ≤ −0.6）。最下位6分位で線+6〜12pt。
　　・**裾を詰めても 88〜93% で頭打ちし 100% は超えない**。床の外なので (106)② の押し上げが無い。
　　・複勝の 100〜130円帯は 91% 以下。96.8% を超えない（(105)の ρ=−1.000 から）。

実行: python3 ml/audit_soft_scan.py [開始年(既定2015)] [--verify]
　　　--verify を付けると「6券種の argmax q = 人気上位N頭」を300レースの全走査で実証する。
出力: data/cache/exp_scan110/{年}.csv に年ごと保存（落ちても続きから走る）。
"""
import itertools
import math
import os
import random
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, "ml")
from audit_crosspool import load_races, payoff, probs, zq
from audit_crosspool2 import realized
from audit_lbs import (fit_lambda, build_matrix, frame_members, g_pair_unordered,
                       g_tri_ordered, g_tri_unordered, g_pair_ordered, stage_w)
from waku_umatan import waku_of

CACHE = "data/cache/exp_scan110"
LINE_WAKU = 0.775
LINE_FUKU = 0.800
QUANT = 6                                    # ★事前登録: 6分位（(105)と同じ）
TAILS = [0.20, 0.10, 0.05, 0.02, 0.01]       # ★事前登録: (106)と同じ5段。増やさない
N_CELLS = QUANT + len(TAILS)                 # 多重性 11
# ★事前登録: 複勝の帯（円）。床(100円)の内側1本＋外側を細かく刻む
FUKU_BANDS = [(0, 90), (90, 100), (100, 110), (110, 120), (120, 130),
              (130, 150), (150, 200), (200, 1e9)]


def mci(x, alpha=0.01):
    x = np.asarray(x, float)
    if len(x) < 2:
        return float("nan"), float("nan"), float("nan")
    m = x.mean()
    se = x.std(ddof=1) / math.sqrt(len(x))
    z = zq(alpha)
    return m, m - z * se, m + z * se


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ra, rb = pd.Series(a).rank().to_numpy(), pd.Series(b).rank().to_numpy()
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


# ───────────────────────── 枠連: 全枠ペアの走査 ─────────────────────────
def waku_scan(r, p, l2):
    """全枠ペア（同枠＝ゾロ目を含む）の q を計算し、(最大qの枠ペア, q) を返す。"""
    w2 = stage_w(p, l2)
    W2 = w2.sum()
    wk = frame_members(r)
    frames = sorted(wk)
    best, bestq = None, -1.0
    for ia in range(len(frames)):
        for ib in range(ia, len(frames)):
            fa, fb = frames[ia], frames[ib]
            q = 0.0
            if fa == fb:
                mem = wk[fa]
                for x in range(len(mem)):
                    for y in range(x + 1, len(mem)):
                        q += g_pair_unordered(p, w2, W2, mem[x], mem[y])
            else:
                for x in wk[fa]:
                    for y in wk[fb]:
                        q += g_pair_unordered(p, w2, W2, x, y)
            if q > bestq:
                best, bestq = (fa, fb), q
    return best, bestq


# ───────────────────────── 行の生成（年ごとキャッシュ） ─────────────────────────
def build_rows(y0):
    os.makedirs(CACHE, exist_ok=True)
    races = load_races()
    years = sorted({r["year"] for r in races if r["year"] >= y0})
    todo = [y for y in years if not os.path.exists(f"{CACHE}/{y}.csv")]
    if todo:
        print(f"λ推定（ウォークフォワード）… 未計算年 {todo}", flush=True)
        P, i1, i2, i3, yrs = build_matrix(races, y0)
        lam = {}
        for yy in years:
            tr = yrs < yy
            if tr.sum() < 3000:
                lam[yy] = None
                continue
            ok3 = tr & (i3 >= 0)
            lam[yy] = (fit_lambda(P[tr], i1[tr], i2[tr]),
                       fit_lambda(P[ok3], i1[ok3], i2[ok3], stage3=True, ic=i3[ok3]))
        for yy in todo:
            if lam.get(yy) is None:
                pd.DataFrame().to_csv(f"{CACHE}/{yy}.csv", index=False)
                continue
            l2, l3 = lam[yy]
            rows = []
            for r in races:
                if r["year"] != yy or realized(r) is None:
                    continue
                hs = r["horses"]
                if len(hs) < 3:
                    continue
                p = probs(hs)
                order = np.argsort(-p)
                nums = [hs[k][0] for k in order]
                num2k = {num: k for k, (num, _, _) in enumerate(hs)}
                base = dict(rid=r["rid"], year=yy, n=r["n"], top_odds=hs[order[0]][1])

                # ── 枠連: 全走査（max-q） と 人気順上位2頭の枠（交絡対照 J6）
                if r["wakuren"]:
                    pair, q = waku_scan(r, p, l2)
                    v = payoff(r, "枠連(人気順)", pair)
                    if q > 0 and v is not None:
                        rows.append(dict(base, kind="枠連scan", q=q,
                                         exp_pay=LINE_WAKU / q * 100, pay=v))
                    pp = tuple(sorted((waku_of(nums[0], r["n"]), waku_of(nums[1], r["n"]))))
                    qp = waku_pair_q(r, p, l2, pp)
                    v2 = payoff(r, "枠連(人気順)", pp)
                    if qp > 0 and v2 is not None:
                        rows.append(dict(base, kind="枠連人気", q=qp,
                                         exp_pay=LINE_WAKU / qp * 100, pay=v2))

                # ── 複勝: 全馬走査（帯ごとに選べるよう全馬を残す）
                from audit_fuku_lbs import top3_probs
                t3 = top3_probs(p, 1.0, l2, l3)
                for k, (num, od, fin) in enumerate(hs):
                    q = float(t3[k])
                    if not (0 < q < 1):
                        continue
                    v = payoff(r, "複勝", (num,))
                    if v is None:
                        continue
                    rows.append(dict(base, kind="複勝", q=q, exp_pay=LINE_FUKU / q * 100,
                                     pay=v, rank=int(np.where(order == k)[0][0]) + 1))
            pd.DataFrame(rows).to_csv(f"{CACHE}/{yy}.csv", index=False)
            print(f"  {yy}: {len(rows):,}行 保存", flush=True)
    dfs = []
    for y in years:
        f = f"{CACHE}/{y}.csv"
        if os.path.exists(f) and os.path.getsize(f) > 10:
            dfs.append(pd.read_csv(f))
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def waku_pair_q(r, p, l2, pair):
    w2 = stage_w(p, l2)
    W2 = w2.sum()
    wk = frame_members(r)
    fa, fb = pair
    q = 0.0
    if fa == fb:
        mem = wk.get(fa, [])
        for x in range(len(mem)):
            for y in range(x + 1, len(mem)):
                q += g_pair_unordered(p, w2, W2, mem[x], mem[y])
    else:
        for x in wk.get(fa, []):
            for y in wk.get(fb, []):
                q += g_pair_unordered(p, w2, W2, x, y)
    return q


# ───────────────────────── 検証: argmax q = 人気上位N頭 か ─────────────────────────
def verify_argmax(y0, n_sample=300, seed=0):
    """★解析的主張の実証。6券種で全組み合わせを走査し、argmax q が人気上位N頭と一致するか。"""
    races = load_races()
    P, i1, i2, i3, yrs = build_matrix(races, y0)
    yy_all = sorted({r["year"] for r in races if r["year"] >= y0})
    lam = {}
    for yy in yy_all:
        tr = yrs < yy
        if tr.sum() < 3000:
            continue
        ok3 = tr & (i3 >= 0)
        lam[yy] = (fit_lambda(P[tr], i1[tr], i2[tr]),
                   fit_lambda(P[ok3], i1[ok3], i2[ok3], stage3=True, ic=i3[ok3]))
    pool = [r for r in races if r["year"] in lam and realized(r) is not None and len(r["horses"]) >= 6]
    random.Random(seed).shuffle(pool)
    pool = pool[:n_sample]
    hit = defaultdict(int)
    tot = defaultdict(int)
    for r in pool:
        hs = r["horses"]
        p = probs(hs)
        order = np.argsort(-p)
        nums = [hs[k][0] for k in order]
        num2k = {num: k for k, (num, _, _) in enumerate(hs)}
        l2, l3 = lam[r["year"]]
        w2 = stage_w(p, l2); W2 = w2.sum()
        w3 = stage_w(p, l3); W3 = w3.sum()
        idx = range(len(p))
        best = {}
        # 複勝
        from audit_fuku_lbs import top3_probs
        t3 = top3_probs(p, 1.0, l2, l3)
        best["複勝"] = (hs[int(np.argmax(t3))][0],)
        # 馬連 / ワイド / 馬単
        bp, bpv = None, -1.0
        bw, bwv = None, -1.0
        bo, bov = None, -1.0
        for x, y in itertools.combinations(idx, 2):
            v = g_pair_unordered(p, w2, W2, x, y)
            if v > bpv:
                bp, bpv = (x, y), v
            wq = v
            for a_, b_ in ((x, y), (y, x)):
                for z in idx:
                    if z in (x, y):
                        continue
                    wq += (g_tri_ordered(p, w2, W2, w3, W3, a_, z, b_)
                           + g_tri_ordered(p, w2, W2, w3, W3, z, a_, b_))
            if wq > bwv:
                bw, bwv = (x, y), wq
        for x, y in itertools.permutations(idx, 2):
            v = g_pair_ordered(p, w2, W2, x, y)
            if v > bov:
                bo, bov = (x, y), v
        best["馬連"] = tuple(sorted(hs[k][0] for k in bp))
        best["ワイド"] = tuple(sorted(hs[k][0] for k in bw))
        best["馬単"] = tuple(hs[k][0] for k in bo)
        # 三連複 / 三連単
        bt, btv = None, -1.0
        for c in itertools.combinations(idx, 3):
            v = g_tri_unordered(p, w2, W2, w3, W3, c)
            if v > btv:
                bt, btv = c, v
        bs, bsv = None, -1.0
        for c in itertools.permutations(idx, 3):
            v = g_tri_ordered(p, w2, W2, w3, W3, *c)
            if v > bsv:
                bs, bsv = c, v
        best["三連複"] = tuple(sorted(hs[k][0] for k in bt))
        best["三連単"] = tuple(hs[k][0] for k in bs)
        want = {"複勝": (nums[0],), "馬連": tuple(sorted(nums[:2])),
                "ワイド": tuple(sorted(nums[:2])), "馬単": (nums[0], nums[1]),
                "三連複": tuple(sorted(nums[:3])), "三連単": (nums[0], nums[1], nums[2])}
        for k in want:
            tot[k] += 1
            hit[k] += 1 if best[k] == want[k] else 0
    print("=" * 78)
    print(f"【検証0】argmax q（全組み合わせ走査）= 人気上位N頭 か（{len(pool)}レース全走査）")
    print("=" * 78)
    for k in ("複勝", "馬連", "ワイド", "馬単", "三連複", "三連単"):
        print(f"  {k:>6}: 一致 {hit[k]}/{tot[k]} = {100*hit[k]/max(tot[k],1):.1f}%")
    print("→ 一致率100%なら、(105)(106)の『人気上位N頭に固定』は**既に最小E組の選択だった**。\n")


# ───────────────────────── 集計 ─────────────────────────
def roi_row(s, line, alpha=0.01):
    v = s["pay"].to_numpy(float) / 100.0        # payoff は「100円あたりの払戻[円]」
    m, lo, hi = mci(v, alpha)
    return dict(n=len(v), roi=m, lo=lo, hi=hi, hit=100.0 * (v > 0).mean(),
                exp=s["exp_pay"].mean(), diff=m - line)


def report_waku(df, y0, n_years):
    for kind, title in (("枠連scan", "★本命【枠連・全走査(max-q)】"),
                        ("枠連人気", "交絡対照J6【枠連・人気順上位2頭の枠】")):
        g = df[df["kind"] == kind]
        if len(g) < 3000:
            continue
        line = LINE_WAKU
        print("=" * 112)
        print(f"{title} 線 {line*100:.1f}% / {len(g):,}R / {y0}年〜（{n_years}年）")
        print("=" * 112)
        ov = roi_row(g, line)
        print(f"  全体: ROI {100*ov['roi']:.1f}%  的中 {ov['hit']:.1f}%  "
              f"平均期待払戻 {ov['exp']:.0f}円  線との差 {100*ov['diff']:+.2f}pt")
        # 6分位
        g = g.copy()
        g["bin"] = pd.qcut(g["exp_pay"], QUANT, labels=False, duplicates="drop")
        print(f"\n  {'区分':>4}{'R数':>8}{'期待払戻':>10}{'的中率':>8}{'ROI':>8}"
              f"{'線との差':>10}{'99%CI':>22}")
        mids, diffs = [], []
        for b in sorted(g["bin"].dropna().unique()):
            s = g[g["bin"] == b]
            d = roi_row(s, line)
            mids.append(d["exp"]); diffs.append(d["diff"])
            print(f"  {int(b)+1:>4}{d['n']:>8,}{d['exp']:>10.0f}{d['hit']:>8.1f}"
                  f"{100*d['roi']:>8.1f}{100*d['diff']:>+10.2f}"
                  f"   [{100*d['lo']:>6.1f},{100*d['hi']:>6.1f}]")
        rho = spearman(mids, diffs)
        print(f"  → 単調性 Spearman ρ = {rho:+.3f}  "
              f"（J1: 本命は ρ≤−0.6 で通過）")
        # 裾
        half = g["year"].median()
        print(f"\n  {'裾':>6}{'R数':>8}{'年間R':>8}{'期待払戻':>10}{'的中率':>8}{'ROI':>8}"
              f"{'線との差':>10}{'99%CI':>22}{'前半':>8}{'後半':>8}")
        trho_x, trho_y = [], []
        for t in TAILS:
            thr = g["exp_pay"].quantile(t)
            s = g[g["exp_pay"] <= thr]
            d = roi_row(s, line)
            a = s[s["year"] <= half]["pay"].mean() / 100.0
            b = s[s["year"] > half]["pay"].mean() / 100.0
            trho_x.append(t); trho_y.append(d["diff"])
            print(f"  {100*t:>5.0f}%{d['n']:>8,}{d['n']/n_years:>8.0f}{d['exp']:>10.0f}"
                  f"{d['hit']:>8.1f}{100*d['roi']:>8.1f}{100*d['diff']:>+10.2f}"
                  f"   [{100*d['lo']:>6.1f},{100*d['hi']:>6.1f}]"
                  f"{100*a:>8.1f}{100*b:>8.1f}")
        bonf = 0.01 / N_CELLS
        print(f"  → J2判定（99%CI下端>100%）: "
              f"{'通過' if any(100*roi_row(g[g['exp_pay']<=g['exp_pay'].quantile(t)], line)['lo'] > 100 for t in TAILS) else '★不通過'}"
              f" / Bonferroni α={bonf:.4f} でも同様に判定する")
        print()


def report_confound(df, y0):
    """J5: 期待払戻でなく『1番人気の単勝オッズ』で層別しても同じ強さが出るか。"""
    g = df[df["kind"] == "枠連scan"].copy()
    if len(g) < 3000:
        return
    print("=" * 112)
    print("交絡対照J5【枠連】層別する指標を『1番人気の単勝オッズ』に替える（Eでなければ駄目か）")
    print("=" * 112)
    for col, name in (("exp_pay", "期待払戻E"), ("top_odds", "1番人気オッズ")):
        g["b"] = pd.qcut(g[col], QUANT, labels=False, duplicates="drop")
        mids, diffs = [], []
        for b in sorted(g["b"].dropna().unique()):
            s = g[g["b"] == b]
            d = roi_row(s, LINE_WAKU)
            mids.append(s[col].mean()); diffs.append(d["diff"])
        lo = g[g["b"] == 0]
        d0 = roi_row(lo, LINE_WAKU)
        print(f"  {name:>14}: ρ={spearman(mids, diffs):+.3f}  "
              f"最下位区分 ROI {100*d0['roi']:.1f}% 線との差 {100*d0['diff']:+.2f}pt "
              f"[{100*d0['lo']:.1f},{100*d0['hi']:.1f}]")
    print("  → E の方が強くなければ、『期待払戻だから』という説明は成立しない\n")


def report_placebo(df, kind, line, n_rep=20, seed=0, only_top=False, label=None):
    """J4: 払戻を(頭数,年)層内でシャッフル → 効果が消えることを確認。

    ★このプラセボの帰無は「Eと払戻の層内の対応だけを壊す」。頭数・年の**構成**は保つので、
    　プラセボが出す差＝**構成だけで説明できる分**。実測との差が『Eが効いている分』。
    """
    g0 = df[df["kind"] == kind].copy()
    if only_top:                      # 1レース1行にする（(106)と同じ「1番人気の複勝」）
        g0 = g0[g0["rank"] == 1]
    if len(g0) < 3000:
        return
    rng = np.random.default_rng(seed)
    print("=" * 112)
    print(f"プラセボJ4【{label or kind}】払戻を(頭数,年)層内でシャッフル×{n_rep}回 → 構成だけで出る分を測る")
    print("=" * 112)
    real, plac = {}, defaultdict(list)
    for t in TAILS:
        thr = g0["exp_pay"].quantile(t)
        real[t] = roi_row(g0[g0["exp_pay"] <= thr], line)["roi"]
    for _ in range(n_rep):
        g = g0.copy()
        g["pay"] = g.groupby(["n", "year"])["pay"].transform(
            lambda s: rng.permutation(s.to_numpy()))
        for t in TAILS:
            thr = g["exp_pay"].quantile(t)
            plac[t].append(roi_row(g[g["exp_pay"] <= thr], line)["roi"])
    print(f"  {'裾':>6}{'実測ROI':>10}{'プラセボROI':>13}{'実測−プラセボ':>14}")
    for t in TAILS:
        pv = np.array(plac[t])
        print(f"  {100*t:>5.0f}%{100*real[t]:>10.1f}{100*pv.mean():>13.1f}"
              f"{100*(real[t]-pv.mean()):>+14.2f}")
    print("  → プラセボが実測と同水準なら、その裾の甘さは『Eが効いている』ではなく"
          "**頭数・年の構成**で説明される\n")


def report_fuku_bands(df, n_years):
    """6-A-2: 床の外側の帯を細かく刻む。★全馬走査（帯に入る馬を各レース1頭選ぶ）。"""
    g = df[df["kind"] == "複勝"].copy()
    if len(g) < 3000:
        return
    print("=" * 112)
    print("副次【複勝・全馬走査】期待払戻の帯（★床100円の内側と外側を刻む / 6-A-2）")
    print("=" * 112)
    print(f"  {'帯(円)':>12}{'R数':>9}{'年間R':>8}{'選ぶ馬の人気':>13}{'期待払戻':>10}"
          f"{'的中率':>8}{'的中時払戻':>11}{'ROI':>8}{'線との差':>10}{'99%CI':>22}")
    mids, rois = [], []
    for lo_, hi_ in FUKU_BANDS:
        s = g[(g["exp_pay"] >= lo_) & (g["exp_pay"] < hi_)]
        if len(s) < 200:
            continue
        # 各レースから帯内で最小Eの馬を1頭だけ（レース間で独立にするため）
        s = s.sort_values("exp_pay").groupby("rid", as_index=False).first()
        d = roi_row(s, LINE_FUKU)
        hitpay = s[s["pay"] > 0]["pay"].mean() if (s["pay"] > 0).any() else 0.0
        mids.append(d["exp"]); rois.append(d["roi"])
        lbl = f"{lo_:.0f}-{hi_:.0f}" if hi_ < 1e8 else f"{lo_:.0f}+"
        print(f"  {lbl:>12}{d['n']:>9,}{d['n']/n_years:>8.0f}{s['rank'].mean():>13.2f}"
              f"{d['exp']:>10.0f}{d['hit']:>8.1f}{hitpay:>11.0f}{100*d['roi']:>8.1f}"
              f"{100*d['diff']:>+10.2f}   [{100*d['lo']:>6.1f},{100*d['hi']:>6.1f}]")
    print(f"  → 帯の単調性 ρ(期待払戻, ROI) = {spearman(mids, rois):+.3f}")
    print("  ★判定: どこかの帯で ROI が (106)の 96.8% を超え、かつ99%CI下端>100% なら到達。\n")


def main():
    y0 = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 2015
    if "--verify" in sys.argv:
        verify_argmax(y0)
        return
    df = build_rows(y0)
    if df.empty:
        print("行が無い")
        return
    n_years = df["year"].nunique()
    print(f"\n(110) 全組み合わせ走査と床の外側の帯（{y0}年以降・{n_years}年）")
    print("★切り方は単勝オッズだけから発走前に計算できる量。λはウォークフォワード。\n")
    report_waku(df, y0, n_years)
    report_confound(df, y0)
    report_placebo(df, "枠連scan", LINE_WAKU)
    report_fuku_bands(df, n_years)
    report_placebo(df, "複勝", LINE_FUKU, only_top=True, label="複勝・1番人気")


if __name__ == "__main__":
    main()
