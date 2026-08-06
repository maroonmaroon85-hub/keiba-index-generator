"""(102) ★★(100)の続き — 専門家を増やす。対数線形プールを**多者混合**に広げる。

★(100)で分かったこと: 市場に L2-win モデルを 14% 混ぜると D が +0.0017 増える（プラセボ通過）。
　なら**別の専門家を足せばもっと増えるのでは**、が自然な次の問い。
　対数線形プールは専門家が何人でも同じ形で書ける:
　　　q ∝ Π_j p_j^{w_j}     （w_j ≥ 0、Σw_j = 1）

★専門家の候補（**互いに違う情報を持っていそうな順に並べた理由も書く**）
　1. **市場**（正規化した1/オッズ）… 土台。(97)よりこれ単体が最強
　2. **L2-win**  … (100)で使ったもの。1着を目標に学習
　3. **L2-top3** … **目標が違う**。3着以内を目標にすると「堅実に走る馬」を拾う。
　　　(48)で「全券種でtop3がwinを上回る」と実測されているので、winとは別の情報を持つはず
　4. **L5-win**  … **容量が違う**。(81)〜(83)で「AUCは下がるがROIは上がる」＝
　　　L2とは違う誤り方をしている。**違う誤り方をする予測器ほど混ぜる価値がある**
　★どれも単体では市場に負ける（(97)）。**それでも混ぜると増えるか**が問い。

★★事前登録
　1. **予想**: 3人とも重みが正になり、合計の利得は(100)の +0.0017 より大きくなる。
　　 ただし **L2-win と L2-top3 は相関が高い**ので、足し算にはならない（0.002〜0.004程度と予想）。
　2. **重みの推定**: 各年**それ以前の年だけ**で、1着の対数尤度を最大化する w を座標降下で求める。
　　 単体で劣る専門家に大きな重みが乗らないよう **w_j ≥ 0 と Σw_j = 1 を課す**。
　3. **★プラセボ**: 全モデルの確率をレース内でシャッフルした版で同じ手続きを踏む。
　　 (100)と同じく **w が市場に潰れて利得が消えること**を確認する。
　4. **★判定**: 利得が(100)を超え、CIが0を除外し、プラセボが通ること。
　　 ★それでも必要量（枠連0.2549）には届かない見込み。**運用は変わらない**。
　5. ⚠**専門家を増やすほど過学習しやすい**。重みは3つだけなので大きな自由度ではないが、
　　 **ウォークフォワードでしか評価しない**ことを守る。

実行: python3 ml/audit_mix_multi.py [開始年(既定2015)]
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
from audit_lbs_model import build, fit_exponent, fit_walk, race_probs

RNG = np.random.default_rng(20260806)
# (専門家名, 目標, 容量) — 学習して data/cache/ に置く
ALL_EXPERTS = [("L2-win", "win", "l2"), ("L2-top3", "top3", "l2"), ("L5-win", "win", "l5")]
# ★第2引数で専門家を絞れる。L5は学習が重く何度も落ちたので、まず2人で結果を出せるようにした
EXPERTS = ALL_EXPERTS


def expert_probs(name, target, cap, y0):
    """ウォークフォワード予測を年ごとに保存して返す。{raceid: {馬番: p}}"""
    dirp = f"data/cache/exp_{name}_{y0}"
    os.makedirs(dirp, exist_ok=True)
    import lightgbm as lgb
    import features as F
    from _cache import load_cached
    from place_wide import PARAMS
    par = PARAMS if cap == "l2" else dict(PARAMS, num_leaves=255,
                                          min_child_samples=10, n_estimators=2000)
    d, fx = load_cached()
    y = ((d["finish"] == 1) if target == "win" else (d["finish"] <= 3)).astype(int).to_numpy()
    year = d["date"].dt.year.to_numpy()
    years = [yy for yy in range(y0, int(year.max()) + 1) if (year == yy).sum() > 5000]
    for yy in years:
        f = f"{dirp}/{yy}.csv"
        if os.path.exists(f):
            continue
        tr, te = year < yy, year == yy
        p = np.mean([lgb.LGBMClassifier(random_state=s, **par)
                     .fit(fx[tr], y[tr], categorical_feature=F.CAT_COLS)
                     .predict_proba(fx[te])[:, 1] for s in range(3)], axis=0)
        sub = d.loc[te, ["raceid", "umaban"]].copy()
        sub["p"] = p
        sub.to_csv(f, index=False)
        print(f"    {name} {yy} 学習完了・保存", flush=True)
    del d, fx
    big = pd.concat([pd.read_csv(f"{dirp}/{yy}.csv") for yy in years], ignore_index=True)
    out = {}
    for rid, um, p in zip(big["raceid"], big["umaban"].astype(int), big["p"]):
        out.setdefault(rid, {})[int(um)] = float(p)
    return out


def pool(mat, w):
    """mat: (専門家 × 馬) の確率。対数線形プール。"""
    lg = (w[:, None] * np.log(np.maximum(mat, 1e-12))).sum(axis=0)
    lg -= lg.max()
    q = np.exp(lg)
    return q / q.sum()


def fit_weights(rows, n_exp, steps=60):
    """1着の対数尤度を最大化する w（w≥0・Σw=1）を座標降下で。市場を w[0] に置く。"""
    w = np.zeros(n_exp)
    w[0] = 1.0

    def ll(wv):
        return sum(math.log(max(pool(m, wv)[k], 1e-12)) for m, k in rows)

    best = ll(w)
    for _ in range(steps):
        improved = False
        for j in range(1, n_exp):
            for delta in (0.04, -0.04, 0.01, -0.01):
                cand = w.copy()
                cand[j] = max(0.0, cand[j] + delta)
                s = cand.sum()
                if s <= 0:
                    continue
                cand = cand / s
                v = ll(cand)
                if v > best + 1e-9:
                    w, best = cand, v
                    improved = True
        if not improved:
            break
    return w


def mci(x, alpha=0.01):
    x = np.asarray(x, float)
    m = x.mean()
    se = x.std(ddof=1) / math.sqrt(len(x))
    z = zq(alpha)
    return m, m - z * se, m + z * se


def main():
    global EXPERTS
    y0 = int(sys.argv[1]) if len(sys.argv) > 1 else 2015
    if len(sys.argv) > 2:
        want = set(sys.argv[2].split(","))
        EXPERTS = [e for e in ALL_EXPERTS if e[0] in want]
    print(f"(102) 多者混合（{y0}年以降）")
    print("★専門家: 市場 / " + " / ".join(n for n, _, _ in EXPERTS))
    maps = []
    for name, tgt, cap in EXPERTS:
        print(f"  {name} を用意")
        maps.append(expert_probs(name, tgt, cap, y0))
    races = load_races()

    Pk, k1, k2, k3, kyr = build(races, y0, None)
    mkt_par = fit_walk(Pk, k1, k2, k3, kyr)

    for tag, shuffle in (("★本番", False), ("プラセボ（レース内シャッフル）", True)):
        print("\n" + "=" * 100)
        print(f"=== {tag} ===")
        print("=" * 100)
        rows = []
        for r in races:
            if r["year"] < y0:
                continue
            rl = realized(r)
            if rl is None:
                continue
            pm = race_probs(r, None)
            ms = [race_probs(r, mp) for mp in maps]
            if any(x is None for x in ms):
                continue
            if shuffle:
                ms = [x[RNG.permutation(len(x))] for x in ms]
            num2k = {num: k for k, (num, _, _) in enumerate(r["horses"])}
            if rl[0] not in num2k:
                continue
            rows.append((np.vstack([pm] + ms), num2k[rl[0]], r["year"], r))

        years = sorted({x[2] for x in rows})
        ws = {}
        for yy in years:
            tr = [(m, k) for m, k, y, _ in rows if y < yy]
            ws[yy] = fit_weights(tr, len(EXPERTS) + 1) if len(tr) >= 3000 else None
        got = {k: v for k, v in ws.items() if v is not None}
        print("  重み（市場 / " + " / ".join(n for n, _, _ in EXPERTS) + "）")
        for yy, w in got.items():
            print(f"   {yy}: " + " ".join(f"{v:.3f}" for v in w))

        # 混合後に τ/λ を当てる
        mx = max(m.shape[1] for m, _, _, _ in rows)
        P, i1, i2, i3, yrs = [], [], [], [], []
        keep = []
        for m, k, yy, r in rows:
            if yy not in got:
                continue
            a, b, c = realized(r)
            num2k = {num: kk for kk, (num, _, _) in enumerate(r["horses"])}
            if b not in num2k:
                continue
            q = pool(m, got[yy])
            v = np.zeros(mx)
            v[:len(q)] = q
            P.append(v)
            i1.append(num2k[a])
            i2.append(num2k[b])
            i3.append(num2k[c] if (c is not None and c in num2k) else -1)
            yrs.append(yy)
            keep.append((r, q))
        P, i1, i2, i3, yrs = (np.array(x) for x in (P, i1, i2, i3, yrs))
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
            pm = race_probs(r, None)
            qx = q ** t / (q ** t).sum()
            pk = pm ** tk / (pm ** tk).sum()
            for kind, key in PARTS.items():
                if not r[key]:
                    continue
                qa, combo = q_of_lbs(kind, r, qx, l2, l3, num2k, a, b, c)
                qb, _ = q_of_lbs(kind, r, pk, l2k, l3k, num2k, a, b, c)
                if qa <= 0 or qb <= 0 or combo is None:
                    continue
                v = payoff(r, PAYKEY[kind], combo)
                if not v or v <= 0:
                    continue
                lp = math.log((v + 5) / 100.0) - math.log(PAYBACK[kind])
                res[kind].append((math.log(qa) + lp, math.log(qb) + lp))

        print(f"\n{'券種':<8}{'R数':>8}{'D(多者混合)':>13}{'D(市場)':>11}{'利得':>11}"
              f"{'利得の99%CI':>22}{'(100)の利得':>13}{'必要量':>9}")
        for kind in PARTS:
            v = res[kind]
            if len(v) < 500:
                continue
            dx = np.array([x[0] for x in v])
            dk = np.array([x[1] for x in v])
            g, lo, hi = mci(dx - dk)
            need = -math.log(PAYBACK[kind])
            ref = {"枠連": 0.0017, "馬連": 0.0024, "馬単": 0.0027,
                   "三連複": 0.0028, "三連単": 0.0034}.get(kind, float("nan"))
            print(f"{kind:<8}{len(v):>8,}{dx.mean():>+13.4f}{dk.mean():>+11.4f}{g:>+11.4f}"
                  f"{f'[{lo:+.4f},{hi:+.4f}]':>22}{ref:>+13.4f}{need:>9.4f}")

    print("\n" + "=" * 100)
    print("★読み方")
    print("  ・利得が(100)を超え、プラセボが通れば『専門家を増やすと情報は増える』が確定する。")
    print("  ・重みを見ること。L2-win と L2-top3 の重みが互いを食い合うなら、**同じ情報**を見ている。")
    print("  ・それでも必要量には届かない見込み。**運用は変わらない**。")


if __name__ == "__main__":
    main()
