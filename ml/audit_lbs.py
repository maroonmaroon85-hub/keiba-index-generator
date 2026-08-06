"""(96) ★Harville 変換そのものを疑う — 2着・3着の確率を **Lo–Bacon-Shone 型**に一般化して D を測り直す。

★なぜこれが要るのか（(89)の中心表が変わりうる）
　(88)〜(92)の D はすべて **q を Harville で作っている**。Harville は
　「1着が抜けた後、残りは元の勝率に比例して2着を争う」という**仮定**であって、実測ではない。
　競馬の実証では**この仮定は人気馬の2・3着確率を系統的に過大評価する**ことが知られている。
　もしそうなら、(89)で出た
　　　馬連 −0.0169 / 馬単 −0.0590 / 三連複 −0.0655 / 三連単 −0.0880
　という**負のD**は「市場が下手」ではなく**変換の誤差**かもしれない。
　★そして **枠連だけ +0.0153** だったのも、枠に集約すると誤差が相殺されるからかもしれない
　（(89)⑥がまさにこれを留保として残し、「枠連の発走前オッズが要る」としていた）。
　**λを推定すれば、オッズを追加で集めなくても分離できる**。ここが今回の眼目。

★一般化（λ=1 で Harville に一致する）
　　1着: P(i) = p_i                                  … 単勝プールそのものなので触らない
　　2着: P(j | 1着=i) = p_j^λ2 / Σ_{k≠i} p_k^λ2
　　3着: P(k | 1,2着=i,j) = p_k^λ3 / Σ_{l≠i,j} p_l^λ3
　λ<1 なら「人気馬は勝ちには強いが2・3着に残る力は相対的に弱い」＝Harvilleの過大評価を補正する向き。

★★事前登録（測る前に宣言）
　1. **予想する符号**: λ2, λ3 はともに **1未満**。これは文献の向きで、後付けではない。
　2. **推定はウォークフォワード**: 各年のλは**それ以前の年だけ**で最尤推定し、当年に適用する。
　   (78)より単一分割は使わない。**λを当年のデータで当てはめない**のが肝（それをやると必ず勝つ）。
　3. **判定**:
　　 a. λ が 1 から離れ、年をまたいで安定していること（年別のλを並べて目視＋範囲）
　　 b. **対応ありで D が上がること**（同じレース・同じ払戻で q だけ差し替える）。99%CIで判定。
　　 c. ★**枠連の優位が生き残るか**。これが本題。
　　　  ・生き残る → 枠連プールは本当に甘い。(89)⑥の留保が外れる
　　　  ・消える  → **(89)の「枠連だけ市場が市場自身に負けている」は変換の誤差だった**
　　　  　　　　　　→ 中心表を訂正する。宿題3（枠連オッズの収集）も不要になる
　4. **★上界は動かない**: λをどう直しても必要量（枠連0.2549）は変わらない。
　　 期待できるのは 0.01〜0.05 程度で、**儲かる話にはならない**。これは記述の精度の話。

実行: python3 ml/audit_lbs.py [開始年(既定2015)]
"""
import math
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, "ml")
from audit_crosspool import PAYBACK, load_races, payoff, probs, zq
from audit_crosspool2 import PARTS, PAYKEY, realized
from waku_umatan import waku_of

GRID = np.round(np.arange(0.30, 2.01, 0.02), 4)      # ★粗探索の範囲も事前に固定


# ───────────────────────── 一般化した変換（λ=1 で Harville） ─────────────────────────
def stage_w(p, lam):
    return p ** lam


def g_pair_ordered(p, w2, W2, x, y):
    d = W2 - w2[x]
    return p[x] * w2[y] / d if d > 1e-12 else 0.0


def g_pair_unordered(p, w2, W2, x, y):
    return g_pair_ordered(p, w2, W2, x, y) + g_pair_ordered(p, w2, W2, y, x)


def g_tri_ordered(p, w2, W2, w3, W3, x, y, z):
    d1, d2 = W2 - w2[x], W3 - w3[x] - w3[y]
    if d1 <= 1e-12 or d2 <= 1e-12:
        return 0.0
    return p[x] * (w2[y] / d1) * (w3[z] / d2)


def g_tri_unordered(p, w2, W2, w3, W3, idx):
    a, b, c = idx
    return sum(g_tri_ordered(p, w2, W2, w3, W3, x, y, z)
               for x, y, z in ((a, b, c), (a, c, b), (b, a, c),
                               (b, c, a), (c, a, b), (c, b, a)))


def frame_members(r):
    wk = defaultdict(list)
    for k, (num, _, _) in enumerate(r["horses"]):
        wk[waku_of(num, r["n"])].append(k)
    return wk


def q_of_lbs(kind, r, p, lam2, lam3, num2k, a, b, c):
    w2 = stage_w(p, lam2)
    W2 = w2.sum()
    if kind == "枠連":
        wa, wb = sorted((waku_of(a, r["n"]), waku_of(b, r["n"])))
        wk = frame_members(r)
        q = 0.0
        if wa == wb:
            mem = wk[wa]
            for x in range(len(mem)):
                for y in range(x + 1, len(mem)):
                    q += g_pair_unordered(p, w2, W2, mem[x], mem[y])
        else:
            for x in wk[wa]:
                for y in wk[wb]:
                    q += g_pair_unordered(p, w2, W2, x, y)
        return q, (wa, wb)
    if kind == "馬連":
        return g_pair_unordered(p, w2, W2, num2k[a], num2k[b]), tuple(sorted((a, b)))
    if kind == "馬単":
        return g_pair_ordered(p, w2, W2, num2k[a], num2k[b]), (a, b)
    if c is None:
        return 0.0, None
    w3 = stage_w(p, lam3)
    W3 = w3.sum()
    if kind == "三連複":
        return (g_tri_unordered(p, w2, W2, w3, W3,
                                (num2k[a], num2k[b], num2k[c])), tuple(sorted((a, b, c))))
    return g_tri_ordered(p, w2, W2, w3, W3, num2k[a], num2k[b], num2k[c]), (a, b, c)


# ───────────────────────── λ の最尤推定（ウォークフォワード） ─────────────────────────
def build_matrix(races, y0):
    """λ推定用に (レース × 最大頭数) の確率行列と、1〜3着の位置・年を作る。"""
    rows, i1, i2, i3, yrs = [], [], [], [], []
    mx = max(len(r["horses"]) for r in races)
    for r in races:
        if r["year"] < y0:
            continue
        rl = realized(r)
        if rl is None:
            continue
        a, b, c = rl
        hs = r["horses"]
        num2k = {num: k for k, (num, _, _) in enumerate(hs)}
        if a not in num2k or b not in num2k:
            continue
        p = probs(hs)
        v = np.zeros(mx)
        v[:len(p)] = p
        rows.append(v)
        i1.append(num2k[a])
        i2.append(num2k[b])
        i3.append(num2k[c] if (c is not None and c in num2k) else -1)
        yrs.append(r["year"])
    return (np.array(rows), np.array(i1), np.array(i2), np.array(i3), np.array(yrs))


def fit_lambda(P, ia, ib, stage3=False, ic=None):
    """条件付き尤度を最大化する λ を返す（粗探索→細探索）。

    2着: Σ [ λ log p_b − log Σ_{k≠a} p_k^λ ]
    3着: Σ [ λ log p_c − log Σ_{k≠a,b} p_k^λ ]
    """
    n = len(P)
    ar = np.arange(n)
    pa, pb = P[ar, ia], P[ar, ib]
    if stage3:
        pc = P[ar, ic]
        tgt = np.log(np.maximum(pc, 1e-12))
    else:
        tgt = np.log(np.maximum(pb, 1e-12))

    def ll(lam):
        W = (P ** lam).sum(axis=1)
        den = W - pa ** lam - (pb ** lam if stage3 else 0.0)
        ok = den > 1e-12
        return float((lam * tgt[ok] - np.log(den[ok])).sum())

    best = max(GRID, key=ll)
    fine = np.round(np.arange(best - 0.02, best + 0.021, 0.002), 4)
    return float(max(fine, key=ll))


def mci(x, alpha=0.01):
    x = np.asarray(x, float)
    m = x.mean()
    se = x.std(ddof=1) / math.sqrt(len(x))
    z = zq(alpha)
    return m, m - z * se, m + z * se


def main():
    y0 = int(sys.argv[1]) if len(sys.argv) > 1 else 2015
    races = load_races()
    P, i1, i2, i3, yrs = build_matrix(races, y0)
    print(f"(96) Harville を λ で一般化して D を測り直す（{y0}年以降・{len(P):,}レース）")
    print("★λ<1 なら『人気馬は勝ちには強いが2・3着に残る力は相対的に弱い』＝Harvilleの過大評価\n")

    years = sorted(set(yrs.tolist()))
    lam = {}
    print("=" * 88)
    print("【1】λ のウォークフォワード推定（各年は**それ以前の年だけ**で推定）")
    print("=" * 88)
    print(f"{'年':<8}{'学習R数':>10}{'λ2(2着)':>11}{'λ3(3着)':>11}")
    for yy in years:
        tr = yrs < yy
        if tr.sum() < 3000:
            lam[yy] = None
            print(f"{yy:<8}{tr.sum():>10,}   （学習データ不足のため除外）")
            continue
        ok3 = tr & (i3 >= 0)
        l2 = fit_lambda(P[tr], i1[tr], i2[tr])
        l3 = fit_lambda(P[ok3], i1[ok3], i2[ok3], stage3=True, ic=i3[ok3])
        lam[yy] = (l2, l3)
        print(f"{yy:<8}{tr.sum():>10,}{l2:>11.3f}{l3:>11.3f}")
    got = [v for v in lam.values() if v]
    if not got:
        sys.exit("λ を推定できる年がない")
    a2 = [v[0] for v in got]
    a3 = [v[1] for v in got]
    print(f"\n  λ2 の範囲 {min(a2):.3f}〜{max(a2):.3f}（幅 {max(a2)-min(a2):.3f}）")
    print(f"  λ3 の範囲 {min(a3):.3f}〜{max(a3):.3f}（幅 {max(a3)-min(a3):.3f}）")
    print("  ★1.000 が Harville。年をまたいで安定して1から離れていれば、変換に系統誤差がある。")

    # ───────── 全期間まとめての λ も出す（参考・当てはめなので判定には使わない） ─────────
    ok3 = i3 >= 0
    print(f"  参考（全期間当てはめ・**判定には使わない**）: "
          f"λ2={fit_lambda(P, i1, i2):.3f} / λ3={fit_lambda(P[ok3], i1[ok3], i2[ok3], True, i3[ok3]):.3f}")

    # ───────── D を測り直す ─────────
    print("\n" + "=" * 88)
    print("【2】同じレース・同じ払戻で **q だけ差し替えて** D を比べる（対応あり）")
    print("=" * 88)
    res = {k: [] for k in PARTS}
    for r in races:
        if r["year"] not in lam or lam[r["year"]] is None:
            continue
        rl = realized(r)
        if rl is None:
            continue
        a, b, c = rl
        hs = r["horses"]
        p = probs(hs)
        num2k = {num: k for k, (num, _, _) in enumerate(hs)}
        if a not in num2k or b not in num2k or (c is not None and c not in num2k):
            continue
        l2, l3 = lam[r["year"]]
        for kind, key in PARTS.items():
            if not r[key]:
                continue
            qh, combo = q_of_lbs(kind, r, p, 1.0, 1.0, num2k, a, b, c)   # Harville
            ql, _ = q_of_lbs(kind, r, p, l2, l3, num2k, a, b, c)         # λ補正
            if qh <= 0 or ql <= 0 or combo is None:
                continue
            v = payoff(r, PAYKEY[kind], combo)
            if not v or v <= 0:
                continue
            lp = math.log((v + 5) / 100.0) - math.log(PAYBACK[kind])
            res[kind].append((math.log(qh) + lp, math.log(ql) + lp))

    print(f"{'券種':<8}{'R数':>8}{'D(Harville)':>14}{'D(λ補正)':>12}{'改善':>10}"
          f"{'改善の99%CI':>22}{'必要量':>9}{'埋まる割合':>11}")
    out = {}
    for kind in PARTS:
        v = res[kind]
        if len(v) < 500:
            continue
        dh = np.array([x[0] for x in v])
        dl = np.array([x[1] for x in v])
        g, lo, hi = mci(dl - dh)
        need = -math.log(PAYBACK[kind])
        out[kind] = (dh.mean(), dl.mean(), g, lo, hi)
        print(f"{kind:<8}{len(v):>8,}{dh.mean():>+14.4f}{dl.mean():>+12.4f}{g:>+10.4f}"
              f"{f'[{lo:+.4f},{hi:+.4f}]':>22}{need:>9.4f}{max(dl.mean(),0)/need*100:>10.1f}%")

    print("\n" + "=" * 88)
    print("【3】★本題 — 枠連の優位は生き残るか")
    print("=" * 88)
    if "枠連" in out:
        wh, wl, *_ = out["枠連"]
        others = [(k, out[k][1]) for k in out if k != "枠連"]
        print(f"  枠連: Harville {wh:+.4f} → λ補正 {wl:+.4f}")
        for k, v in others:
            print(f"  {k:<6}: λ補正 {v:+.4f}   （枠連との差 {wl-v:+.4f}）")
        if wl > 0 and all(wl > v for _, v in others):
            print("  → ★枠連の優位は**生き残った**。(89)⑥の留保どおり枠連プールが甘い可能性が残る。")
        elif wl <= 0:
            print("  → ★**枠連の優位は消えた**。(89)の『枠連だけ市場が市場自身に負けている』は"
                  "**Harvilleの誤差だった**。中心表を訂正し、宿題3（枠連オッズ収集）は不要になる。")
        else:
            print("  → 枠連はもう最上位ではない。順位が入れ替わったので(89)⑤の解釈は書き直しが要る。")

    print("\n" + "=" * 88)
    print("★読み方")
    print("  ・改善のCIが0を含むなら、Harvilleの誤差は測れる大きさではない＝(89)の表はそのままでよい。")
    print("  ・改善が正でも、**必要量には遠いので運用は変わらない**（枠連0.2549）。記述の精度の話。")
    print("  ・λが年をまたいで動くなら、λ自体が推定ノイズ。範囲を見て判断すること。")


if __name__ == "__main__":
    main()
