"""(104) ★★(100)×(95) — 混合比 w を**レースごとに変える**。不一致が小さいレースほど濃く混ぜる。

★2つの結果を掛け合わせる
　・(100): 市場に L2 モデルを **14%** 混ぜると D が +0.0017 増える（プラセボ通過）
　・(95) : **シード間不一致 cv が小さいレースほどモデルはマシ**（除外50%で差が +0.0146 改善）
　→ **w を定数にする理由が無い**。不一致が小さいレースでは濃く、大きいレースでは薄く混ぜれば、
　　 同じモデルから**もっと多くの情報を引き出せる**はず。

★手続き
　1. cv（1位馬確率のシード間 変動係数）でレースを5区分。**区切りは前年までの分位で決める**
　　 （当年のデータで区切りを決めたら未来を見たことになる）。
　2. 各区分の w を**前年までのデータだけ**で最尤推定（1着の対数尤度）。
　3. 当年に適用して D を測る。比較は **定数w（(100)の形）** と **市場のみ**。

★★事前登録
　1. **予想**: w は cv に対して**単調減少**する（不一致が小さい区分ほど大きい）。
　　 (95)でその向きは実測済みなので、これは強い予想。
　2. **判定**: `D(条件付きw) − D(定数w)` が99%CIで正。**定数wとの比較が本命**
　　 （市場との比較では(100)の分が混ざるので判定にならない）。
　3. **★プラセボ**: cv を**レース間でシャッフル**する。区分の大きさも w の自由度も同じで、
　　 **cvとレースの対応だけ壊す**。ここで利得が出たら「区分を増やした」だけの効果。
　4. **⚠自由度が5倍になる**。ウォークフォワードでしか評価しないことと、
　　 プラセボで自由度の効果を測ることの2つで守る。
　5. **上界は動かない**見込み（必要量 枠連0.2549）。**運用は変わらない**。

実行: python3 ml/audit_mix_cond.py [開始年(既定2015)]
　　（data/cache/seedraw_5_2015/ が要る。無ければ audit_seed_disagree.py を先に）
"""
import math
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, "ml")
from audit_crosspool import PAYBACK, load_races, payoff, probs, zq
from audit_crosspool2 import PARTS, PAYKEY, realized
from audit_lbs import q_of_lbs
from audit_lbs_model import fit_exponent, race_probs

WGRID = np.round(np.arange(0.0, 0.601, 0.02), 4)
NBIN = 5
RNG = np.random.default_rng(20260806)


def load_seed(n_seed=5, y0=2015):
    dirp = f"data/cache/seedraw_{n_seed}_{y0}"
    if not os.path.isdir(dirp):
        sys.exit(f"{dirp} が無い。先に python3 ml/audit_seed_disagree.py {n_seed} {y0}")
    big = pd.concat([pd.read_csv(f"{dirp}/{f}") for f in sorted(os.listdir(dirp))],
                    ignore_index=True)
    cols = [f"p{s}" for s in range(n_seed)]
    out = {}
    for rid, um, *ps in zip(big["raceid"], big["umaban"].astype(int), *[big[c] for c in cols]):
        out.setdefault(rid, {})[int(um)] = np.array(ps, float)
    return out


def mix(pm, pM, w):
    lg = (1 - w) * np.log(np.maximum(pm, 1e-12)) + w * np.log(np.maximum(pM, 1e-12))
    lg -= lg.max()
    q = np.exp(lg)
    return q / q.sum()


def fit_w(rows):
    if len(rows) < 500:
        return None
    def ll(w):
        return sum(math.log(max(mix(pm, pM, w)[k], 1e-12)) for pm, pM, k in rows)
    return float(max(WGRID, key=ll))


def mci(x, alpha=0.01):
    x = np.asarray(x, float)
    m = x.mean()
    se = x.std(ddof=1) / math.sqrt(len(x))
    z = zq(alpha)
    return m, m - z * se, m + z * se


def build(races, smap, y0, shuffle_cv=False):
    rows = []
    for r in races:
        if r["year"] < y0:
            continue
        rl = realized(r)
        if rl is None:
            continue
        hs = r["horses"]
        mp = smap.get(r["rid"])
        if not mp:
            continue
        idx = [k for k, (num, _, _) in enumerate(hs) if num in mp]
        if len(idx) < 4:
            continue
        mat = np.array([mp[hs[k][0]] for k in idx], float)
        z = mat / mat.sum(axis=0, keepdims=True)
        top = int(np.bincount(z.argmax(axis=0)).argmax())
        v = z[top]
        cv = float(v.std(ddof=0) / v.mean()) if v.mean() > 0 else 0.0
        pm = race_probs(r, None)
        raw = np.full(len(hs), np.nan)
        raw[idx] = mat.mean(axis=1)
        miss = np.isnan(raw)
        s = raw[~miss].sum()
        if s <= 0:
            continue
        raw[~miss] = raw[~miss] / s * pm[~miss].sum()
        raw[miss] = pm[miss]
        pM = raw / raw.sum()
        num2k = {num: k for k, (num, _, _) in enumerate(hs)}
        if rl[0] not in num2k:
            continue
        rows.append({"r": r, "pm": pm, "pM": pM, "k": num2k[rl[0]],
                     "cv": cv, "year": r["year"]})
    if shuffle_cv:
        cvs = RNG.permutation([x["cv"] for x in rows])
        for x, c in zip(rows, cvs):
            x["cv"] = float(c)
    return rows


def measure(rows, wfun, mkt_par, label):
    """wfun(row, year) -> w。混合後に τ/λ を当てて D を集める。"""
    years = sorted({x["year"] for x in rows})
    mx = max(len(x["pm"]) for x in rows)
    P, i1, i2, i3, yrs, keep = [], [], [], [], [], []
    for x in rows:
        w = wfun(x)
        if w is None:
            continue
        r = x["r"]
        a, b, c = realized(r)
        num2k = {num: k for k, (num, _, _) in enumerate(r["horses"])}
        if b not in num2k:
            continue
        q = mix(x["pm"], x["pM"], w)
        v = np.zeros(mx)
        v[:len(q)] = q
        P.append(v)
        i1.append(num2k[a])
        i2.append(num2k[b])
        i3.append(num2k[c] if (c is not None and c in num2k) else -1)
        yrs.append(x["year"])
        keep.append((r, q))
    P, i1, i2, i3, yrs = (np.array(t) for t in (P, i1, i2, i3, yrs))
    par = {}
    for yy in sorted(set(yrs.tolist())):
        tr = yrs < yy
        if tr.sum() < 3000:
            par[yy] = None
            continue
        ok3 = tr & (i3 >= 0)
        par[yy] = (fit_exponent(P[tr], i1[tr]),
                   fit_exponent(P[tr], i2[tr], drop_idx=(i1[tr],)),
                   fit_exponent(P[ok3], i3[ok3], drop_idx=(i1[ok3], i2[ok3])))
    res = {k: [] for k in PARTS}
    for r, q in keep:
        yy = r["year"]
        if par.get(yy) is None or mkt_par.get(yy) is None:
            continue
        a, b, c = realized(r)
        num2k = {num: k for k, (num, _, _) in enumerate(r["horses"])}
        if a not in num2k or b not in num2k or (c is not None and c not in num2k):
            continue
        t, l2, l3 = par[yy]
        tk, l2k, l3k = mkt_par[yy]
        pm = probs(r["horses"])
        qx = q ** t / (q ** t).sum()
        pk = pm ** tk / (pm ** tk).sum()
        for kind, key in PARTS.items():
            if not r[key]:
                continue
            qa, combo = q_of_lbs(kind, r, qx, l2, l3, num2k, a, b, c)
            qb, _ = q_of_lbs(kind, r, pk, l2k, l3k, num2k, a, b, c)
            if qa <= 0 or qb <= 0 or combo is None:
                continue
            vv = payoff(r, PAYKEY[kind], combo)
            if not vv or vv <= 0:
                continue
            lp = math.log((vv + 5) / 100.0) - math.log(PAYBACK[kind])
            res[kind].append((r["rid"], math.log(qa) + lp, math.log(qb) + lp))
    return res


def main():
    y0 = int(sys.argv[1]) if len(sys.argv) > 1 else 2015
    print(f"(104) 混合比をレースごとに変える（{y0}年以降・5シード）")
    print("★判定の本命は『条件付きw − 定数w』。市場との比較は(100)の分が混ざるので判定にならない\n")
    smap = load_seed(5, y0)
    races = load_races()
    from audit_lbs_model import build as build_mat, fit_walk
    Pk, k1, k2, k3, kyr = build_mat(races, y0, None)
    mkt_par = fit_walk(Pk, k1, k2, k3, kyr)

    for tag, shuf in (("★本番", False), ("プラセボ（cvをレース間でシャッフル）", True)):
        rows = build(races, smap, y0, shuffle_cv=shuf)
        years = sorted({x["year"] for x in rows})
        print("\n" + "=" * 100)
        print(f"=== {tag}（{len(rows):,}レース） ===")
        print("=" * 100)

        # 区切りと w を「前年まで」で決める
        cuts, wmap, wconst = {}, {}, {}
        for yy in years:
            tr = [x for x in rows if x["year"] < yy]
            if len(tr) < 3000:
                continue
            qs = np.quantile([x["cv"] for x in tr], np.linspace(0, 1, NBIN + 1)[1:-1])
            cuts[yy] = qs
            wconst[yy] = fit_w([(x["pm"], x["pM"], x["k"]) for x in tr])
            for b in range(NBIN):
                sub = [x for x in tr if np.searchsorted(qs, x["cv"]) == b]
                wmap[(yy, b)] = fit_w([(x["pm"], x["pM"], x["k"]) for x in sub])
        yy_last = max(cuts)
        print(f"  {yy_last}年に使う w（cvの小さい区分から）: "
              + " ".join(f"{wmap.get((yy_last, b))}" for b in range(NBIN))
              + f"   定数w={wconst[yy_last]}")
        ws = [wmap.get((yy_last, b)) for b in range(NBIN)]
        if all(w is not None for w in ws):
            rho = pd.Series(ws).corr(pd.Series(range(NBIN)), method="spearman")
            print(f"  → 単調性 ρ={rho:+.3f}（負なら予想どおり『不一致が小さいほど濃く混ぜる』）")

        def w_cond(x):
            yy = x["year"]
            if yy not in cuts:
                return None
            return wmap.get((yy, int(np.searchsorted(cuts[yy], x["cv"]))))

        def w_const(x):
            return wconst.get(x["year"])

        rc = measure(rows, w_cond, mkt_par, "条件付き")
        rk = measure(rows, w_const, mkt_par, "定数")
        print(f"\n{'券種':<8}{'R数':>8}{'D(条件付き)':>13}{'D(定数w)':>12}{'D(市場)':>11}"
              f"{'条件付き−定数':>15}{'99%CI':>22}")
        for kind in PARTS:
            a = pd.DataFrame(rc[kind], columns=["rid", "dc", "dk"])
            b = pd.DataFrame(rk[kind], columns=["rid", "dw", "dk2"])
            if len(a) < 500 or len(b) < 500:
                continue
            m = a.merge(b, on="rid")                    # ★対応あり（同じレースで引く）
            g, lo, hi = mci(m["dc"] - m["dw"])
            print(f"{kind:<8}{len(m):>8,}{m['dc'].mean():>+13.4f}{m['dw'].mean():>+12.4f}"
                  f"{m['dk'].mean():>+11.4f}{g:>+15.4f}{f'[{lo:+.4f},{hi:+.4f}]':>22}")

    print("\n" + "=" * 100)
    print("★読み方")
    print("  ・本番で『条件付き−定数』が正・プラセボで0なら、**wをレースごとに変える価値がある**。")
    print("  ・プラセボでも出るなら、それは**自由度を5倍にしただけ**の効果。")
    print("  ・利得が出ても必要量には届かない見込み。**運用は変わらない**。")


if __name__ == "__main__":
    main()
