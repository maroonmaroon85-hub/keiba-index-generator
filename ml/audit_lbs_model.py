"""(97) ★★(90)の決着 — **モデルと市場に、それぞれ自分に最適な変換を当てて**Dを比べ直す。

★なぜ必要か
　(90)は「モデルは市場に 0.0846 負けている（枠連）」と結論したが、**両方とも Harville で測っていた**。
　(96)で Harville に系統誤差（λ2≈0.85 / λ3≈0.72）があると分かった以上、
　**モデル側の確率にも同じ誤差が乗っている**。しかも乗り方は同じとは限らない
　（モデルは3着以内目標で学習した特徴を使っているので、2・3着の段の性質が市場と違いうる）。
　→ **各side に最適な変換を当てて初めて公平な比較になる**。

★(96)からもう一段一般化する（τ＝1着の段）
　　1着: P(i) = p_i^τ / Σ p_k^τ        ← ★(96)では触っていない。τ<1 なら人気薄が過小評価されている
　　2着: P(j|i) = p_j^λ2 / Σ_{k≠i} p_k^λ2
　　3着: P(k|i,j) = p_k^λ3 / Σ_{l≠i,j} p_l^λ3
　τ は**単勝プールの人気馬バイアス（favourite-longshot bias）そのもの**。市場側で τ≠1 が出るなら、
　それは「単勝オッズは勝率としてズレている」という意味で、(89)の前提（q_pool = 正規化した1/オッズ）
　自体に効く。3つとも**各年それ以前の年だけで最尤推定**する。

★★事前登録（測る前に宣言）
　1. **予想する符号**: 市場側は τ<1（人気薄が本来より買われている＝FLB）。モデル側の τ は不明。
　2. **判定**: 補正後も `D(モデル) − D(市場)` が負なら、**(90)の結論は変換の産物ではない**＝確定。
　　 正に転じたら、**(90)は撤回**し「モデルは市場を超えていた」に書き換える。
　　 ★0.0846 は大きいので**転じるとは考えていない**。だが測らずに言い切ることはしない。
　3. **公平性**: モデル側だけ有利な自由度を与えない。**両方に同じ3パラメータを与える**。
　4. **上界は動かない**: どちらに転んでも必要量（枠連0.2549）には遠い。運用は変わらない。

実行: python3 ml/audit_lbs_model.py [シード数(既定3)] [開始年(既定2015)]
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
from audit_lbs import GRID, q_of_lbs

CACHE = "data/cache/winprob_{n}_{y}.csv"


# ───────────────────────── モデルの勝率（キャッシュ付き） ─────────────────────────
def model_win_probs(n_seed, y0):
    path = CACHE.format(n=n_seed, y=y0)
    if os.path.exists(path):
        d = pd.read_csv(path)
        print(f"  キャッシュを読んだ: {path}（{len(d):,}行）")
    else:
        import lightgbm as lgb
        import features as F
        from _cache import load_cached
        from place_wide import PARAMS
        dd, fx = load_cached()
        ywin = (dd["finish"] == 1).astype(int).to_numpy()
        year = dd["date"].dt.year.to_numpy()
        years = [yy for yy in range(y0, int(year.max()) + 1) if (year == yy).sum() > 5000]
        parts = []
        for yy in years:
            tr, te = year < yy, year == yy
            p = np.mean([lgb.LGBMClassifier(random_state=s, **PARAMS)
                         .fit(fx[tr], ywin[tr], categorical_feature=F.CAT_COLS)
                         .predict_proba(fx[te])[:, 1] for s in range(n_seed)], axis=0)
            sub = dd.loc[te, ["raceid", "umaban"]].copy()
            sub["p"] = p
            parts.append(sub)
            print(f"  {yy} 学習完了", flush=True)
        d = pd.concat(parts, ignore_index=True)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        d.to_csv(path, index=False)
        print(f"  キャッシュを書いた: {path}")
    out = {}
    for rid, um, p in zip(d["raceid"], d["umaban"].astype(int), d["p"]):
        out.setdefault(rid, {})[int(um)] = float(p)
    return out


# ───────────────────────── 3パラメータの最尤推定 ─────────────────────────
def fit_exponent(P, tgt_idx, drop_idx=()):
    """P (レース×頭数, 0埋め) から、条件付きロジットの指数を1次元最尤で求める。

    drop_idx に入れた列（既に着順の決まった馬）は分母から抜く。
    """
    n = len(P)
    ar = np.arange(n)
    tgt = np.log(np.maximum(P[ar, tgt_idx], 1e-12))
    drops = [P[ar, d] for d in drop_idx]

    def ll(e):
        W = (P ** e).sum(axis=1)
        den = W - sum(d ** e for d in drops) if drops else W
        ok = den > 1e-12
        return float((e * tgt[ok] - np.log(den[ok])).sum())

    best = max(GRID, key=ll)
    fine = np.round(np.arange(best - 0.02, best + 0.021, 0.002), 4)
    return float(max(fine, key=ll))


def build(races, y0, pmap):
    """λ推定用の行列を作る。pmap=None なら市場、そうでなければモデル（欠損は市場で埋める）。"""
    mx = max(len(r["horses"]) for r in races)
    rows, i1, i2, i3, yrs = [], [], [], [], []
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
        p = race_probs(r, pmap)
        if p is None:
            continue
        v = np.zeros(mx)
        v[:len(p)] = p
        rows.append(v)
        i1.append(num2k[a])
        i2.append(num2k[b])
        i3.append(num2k[c] if (c is not None and c in num2k) else -1)
        yrs.append(r["year"])
    return np.array(rows), np.array(i1), np.array(i2), np.array(i3), np.array(yrs)


def race_probs(r, pmap):
    """市場（pmap=None）またはモデルの確率ベクトル。(90)と同じ欠損の埋め方。"""
    hs = r["horses"]
    p_mkt = probs(hs)
    if pmap is None:
        return p_mkt
    mp = pmap.get(r["rid"])
    if not mp:
        return None
    raw = np.array([mp.get(num, np.nan) for num, _, _ in hs], float)
    miss = np.isnan(raw)
    if miss.all():
        return None
    s = raw[~miss].sum()
    if s <= 0:
        return None
    raw[~miss] = raw[~miss] / s * p_mkt[~miss].sum()
    raw[miss] = p_mkt[miss]
    return raw / raw.sum()


def fit_walk(P, i1, i2, i3, yrs):
    """{年: (τ, λ2, λ3)}。各年は**それ以前の年だけ**で推定。"""
    out = {}
    for yy in sorted(set(yrs.tolist())):
        tr = yrs < yy
        if tr.sum() < 3000:
            out[yy] = None
            continue
        ok3 = tr & (i3 >= 0)
        out[yy] = (fit_exponent(P[tr], i1[tr]),
                   fit_exponent(P[tr], i2[tr], drop_idx=(i1[tr],)),
                   fit_exponent(P[ok3], i3[ok3], drop_idx=(i1[ok3], i2[ok3])))
    return out


def mci(x, alpha=0.01):
    x = np.asarray(x, float)
    m = x.mean()
    se = x.std(ddof=1) / math.sqrt(len(x))
    z = zq(alpha)
    return m, m - z * se, m + z * se


def apply_tau(p, tau):
    w = p ** tau
    return w / w.sum()


def main():
    n_seed = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    y0 = int(sys.argv[2]) if len(sys.argv) > 2 else 2015
    print(f"(97) モデルと市場に**それぞれ最適な変換**を当ててDを比べ直す（{y0}年以降・シード{n_seed}本）")
    print("★判定: 補正後も差が負なら(90)は確定。正なら(90)を撤回する\n")
    pmap = model_win_probs(n_seed, y0)
    races = load_races()

    print("\n" + "=" * 92)
    print("【1】3パラメータのウォークフォワード推定（τ=1着 / λ2=2着 / λ3=3着）")
    print("=" * 92)
    par = {}
    for tag, pm in (("市場", None), ("モデル", pmap)):
        P, i1, i2, i3, yrs = build(races, y0, pm)
        par[tag] = fit_walk(P, i1, i2, i3, yrs)
        got = [v for v in par[tag].values() if v]
        arr = np.array(got)
        print(f"\n■ {tag}（{len(P):,}レース）")
        print(f"{'年':<8}{'τ(1着)':>10}{'λ2(2着)':>10}{'λ3(3着)':>10}")
        for yy, v in par[tag].items():
            if v:
                print(f"{yy:<8}{v[0]:>10.3f}{v[1]:>10.3f}{v[2]:>10.3f}")
        print(f"  範囲 τ {arr[:, 0].min():.3f}〜{arr[:, 0].max():.3f} / "
              f"λ2 {arr[:, 1].min():.3f}〜{arr[:, 1].max():.3f} / "
              f"λ3 {arr[:, 2].min():.3f}〜{arr[:, 2].max():.3f}")
    print("\n  ★τ<1 は『人気薄が本来より買われている（favourite-longshot bias）』の向き。")
    print("  ★市場のτが1から離れるなら、(89)の前提（q_pool=正規化した1/オッズ）自体に誤差がある。")

    # ───────── D を測る ─────────
    print("\n" + "=" * 92)
    print("【2】★同じレースで対応あり比較 — 各sideに自分の最適変換を当てた D")
    print("=" * 92)
    res = {k: [] for k in PARTS}
    for r in races:
        yy = r["year"]
        if yy < y0 or par["市場"].get(yy) is None or par["モデル"].get(yy) is None:
            continue
        rl = realized(r)
        if rl is None:
            continue
        a, b, c = rl
        hs = r["horses"]
        num2k = {num: k for k, (num, _, _) in enumerate(hs)}
        if a not in num2k or b not in num2k or (c is not None and c not in num2k):
            continue
        pk = race_probs(r, None)
        pm = race_probs(r, pmap)
        if pm is None:
            continue
        tk, l2k, l3k = par["市場"][yy]
        tm, l2m, l3m = par["モデル"][yy]
        pk_t, pm_t = apply_tau(pk, tk), apply_tau(pm, tm)
        for kind, key in PARTS.items():
            if not r[key]:
                continue
            qk, combo = q_of_lbs(kind, r, pk_t, l2k, l3k, num2k, a, b, c)
            qm, _ = q_of_lbs(kind, r, pm_t, l2m, l3m, num2k, a, b, c)
            qk0, _ = q_of_lbs(kind, r, pk, 1.0, 1.0, num2k, a, b, c)      # (90)の測り方
            qm0, _ = q_of_lbs(kind, r, pm, 1.0, 1.0, num2k, a, b, c)
            if min(qk, qm, qk0, qm0) <= 0 or combo is None:
                continue
            v = payoff(r, PAYKEY[kind], combo)
            if not v or v <= 0:
                continue
            lp = math.log((v + 5) / 100.0) - math.log(PAYBACK[kind])
            res[kind].append((math.log(qm) + lp, math.log(qk) + lp,
                              math.log(qm0) + lp, math.log(qk0) + lp))

    print(f"{'券種':<8}{'R数':>8}{'D(モデル)':>12}{'D(市場)':>11}{'差':>11}{'差の99%CI':>22}"
          f"{'(90)の差':>11}{'差の変化':>11}")
    flip = []
    for kind in PARTS:
        v = res[kind]
        if len(v) < 500:
            continue
        dm, dk, dm0, dk0 = (np.array([x[i] for x in v]) for i in range(4))
        g, lo, hi = mci(dm - dk)
        g0 = float((dm0 - dk0).mean())
        flip.append((kind, g, lo, hi, g0))
        print(f"{kind:<8}{len(v):>8,}{dm.mean():>+12.4f}{dk.mean():>+11.4f}{g:>+11.4f}"
              f"{f'[{lo:+.4f},{hi:+.4f}]':>22}{g0:>+11.4f}{g-g0:>+11.4f}")

    print("\n" + "=" * 92)
    print("【3】★判定")
    print("=" * 92)
    neg = [k for k, g, lo, hi, _ in flip if hi < 0]
    pos = [k for k, g, lo, hi, _ in flip if lo > 0]
    print(f"  差が負（モデルが市場に負け）と確定した券種: {len(neg)}/{len(flip)}  {neg}")
    print(f"  差が正（モデルが市場を超えた）と確定した券種: {len(pos)}/{len(flip)}  {pos}")
    if not pos:
        print("  → ★**(90)の結論は変換の産物ではなかった**。モデルは市場より情報量が少ない。")
        print("     機械学習を改善する方向は閉じたままで、以後も q に何を入れるかの勝負。")
    else:
        print("  → ★★**(90)を撤回する**。補正後はモデルが市場を超えている。中心表を書き直すこと。")
    print("  ⚠どちらに転んでも必要量（枠連0.2549）には遠い。**運用は変わらない**。")


if __name__ == "__main__":
    main()
