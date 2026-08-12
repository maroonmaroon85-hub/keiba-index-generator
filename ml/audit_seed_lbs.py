"""(118) ★★(95)「シード間不一致による除外」を**λ補正した今のqで測り直す**（2026-08-11）

★なぜやり直すか
　(95)は D を測ってはいるが、市場側が **`probs(hs)`＝素のHarville**（D(市場)=+0.0151）。
　その後(96)でλ補正が入り、市場の枠連Dは **+0.0182** になっている。
　そして(116)で**補正の効き方は層によって違う**と実証された（ローカル/本場の差が3券種で消えた）。
　(95)は**レースを選別する研究**なので、まさにその影響を受ける型。

★(95)の核心は「差」だった
　条件3「上昇幅 **モデル +0.0102 vs 市場 +0.0019**」＝
　「当てやすいレースを選んだのではなく、モデルが効くレースを選んだ」という主張。
　**市場側が素のHarvilleのままなら、この差は道具の癖でも作れる**。そこを潰す。

★★事前登録（測る前に宣言）
　1. 除外率は **(95)と同じ 0/10/30/50%**。後から増やさない。
　2. 見るのは **市場D の上昇幅**（λ補正版）。(95)では +0.0019 で「ほぼ動かない」とされた。
　　 λ補正後にこれが大きくなるなら、(95)③の交絡対照は**通っていなかった**ことになる。
　3. **プラセボ**（無作為除外）を同じ本数で必ず並べる。
　4. **予想**: 市場Dの上昇幅は λ補正後も小さい（+0.005未満）まま。
　　 理由: `cv_top` はモデルの内部的なばらつきで、**市場の値付けとは独立**のはず。
　　 ただし「不一致が大きい＝荒れそうなレース」なら市場側にも効きうるので、断定しない。
　5. **判定の意味**: (95)の結論が保つなら「モデルが効くレースの選別」は生きている。
　　 崩れるなら、(104)（レースごとに混合比を変える＝利得ゼロ）と合わせて**この筋は閉じる**。

実行: python3 ml/audit_seed_lbs.py [開始年(既定2015)]
　　　（`data/cache/seedraw_5_2015/` のシード別確率を使う。無ければ(95)を先に回すこと）
"""
import glob
import math
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "ml")
from audit_crosspool import PAYBACK, load_races, payoff, probs, zq
from audit_crosspool2 import PAYKEY, realized
from audit_lbs import build_matrix, fit_lambda, q_of_lbs
from waku_umatan import waku_of

CUTS = [0.0, 0.10, 0.30, 0.50]          # ★(95)と同じ水準
CACHE = "data/cache/seedraw_5_2015"
RNG = np.random.default_rng(20260811)


def mci(x, alpha=0.01):
    x = np.asarray(x, float)
    if len(x) < 2:
        return float("nan"), float("nan"), float("nan")
    m = x.mean()
    se = x.std(ddof=1) / math.sqrt(len(x))
    return m, m - zq(alpha) * se, m + zq(alpha) * se


def load_seeds():
    """{raceid: {馬番: [シードごとのp]}}。"""
    out = {}
    for f in sorted(glob.glob(f"{CACHE}/*.csv")):
        df = pd.read_csv(f)
        cols = [c for c in df.columns if c.startswith("p") and c[1:].isdigit()]
        for t in df.itertuples(index=False):
            d = dict(zip(df.columns, t))
            out.setdefault(str(d["raceid"]), {})[int(d["umaban"])] = [
                float(d[c]) for c in cols]
    return out


def waku_q_model(r, pm):
    """モデル確率を枠に集約して、枠連の当たり枠組の確率を返す（(95)と同じ積ベース）。"""
    nums = [u for u, _, _ in r["horses"]]
    s = sum(pm[u] for u in nums)
    if s <= 0:
        return None
    bp = {}
    for u in nums:
        bp[waku_of(u, r["n"])] = bp.get(waku_of(u, r["n"]), 0.0) + pm[u] / s
    return bp


def main():
    y0 = int(sys.argv[1]) if len(sys.argv) > 1 else 2015
    seeds = load_seeds()
    if not seeds:
        sys.exit(f"{CACHE} が無い。先に `python3 ml/audit_seed_disagree.py 5 2015` を回すこと。")
    races = load_races()
    P, i1, i2, i3, yrs = build_matrix(races, y0)
    lam = {}
    for yy in sorted(set(yrs.tolist())):
        tr = yrs < yy
        if tr.sum() < 3000:
            lam[yy] = None
            continue
        ok3 = tr & (i3 >= 0)
        lam[yy] = (fit_lambda(P[tr], i1[tr], i2[tr]),
                   fit_lambda(P[ok3], i1[ok3], i2[ok3], stage3=True, ic=i3[ok3]))

    rows = []
    for r in races:
        yy = r["year"]
        if yy < y0 or not lam.get(yy) or not r["wakuren"]:
            continue
        sp = seeds.get(r["rid"])
        if not sp:
            continue
        rl = realized(r)
        if rl is None:
            continue
        a, b, c = rl
        hs = r["horses"]
        nums = [u for u, _, _ in hs]
        if any(u not in sp for u in nums):
            continue
        num2k = {u: k for k, (u, _, _) in enumerate(hs)}
        if a not in num2k or b not in num2k:
            continue
        # ★cv_top: 「最有力馬の確率」のシード間 変動係数（(95)Bと同じ）
        arr = np.array([sp[u] for u in nums], float)      # (馬, シード)
        top = arr[arr.mean(axis=1).argmax()]
        cv = float(top.std(ddof=0) / top.mean()) if top.mean() > 0 else 0.0

        p_mkt = probs(hs)
        l2, l3 = lam[yy]
        q_raw, combo = q_of_lbs("枠連", r, p_mkt, 1.0, 1.0, num2k, a, b, c)   # 素＝λ=1
        q_lbs, _ = q_of_lbs("枠連", r, p_mkt, l2, l3, num2k, a, b, c)
        if q_raw <= 0 or q_lbs <= 0 or combo is None:
            continue
        v = payoff(r, PAYKEY["枠連"], combo)
        if not v or v <= 0:
            continue
        bp = waku_q_model(r, {u: float(np.mean(sp[u])) for u in nums})
        if bp is None:
            continue
        wa, wb = sorted(combo)
        qm = bp.get(wa, 0.0) * bp.get(wb, 0.0) * (1.0 if wa == wb else 2.0)
        if qm <= 0:
            continue
        lg = math.log((v + 5) / 100.0) - math.log(PAYBACK["枠連"])
        rows.append(dict(year=yy, cv=cv,
                         d_raw=math.log(q_raw) + lg,     # (95)が使った素のHarville
                         d_lbs=math.log(q_lbs) + lg,     # ★λ補正した今のq
                         d_mdl=math.log(qm) + lg))       # モデル側

    df = pd.DataFrame(rows)
    print(f"(118) (95)をλ補正した今のqで測り直す（{y0}年以降・{len(df):,}レース）")
    print("★(95)の核心は「上昇幅 モデル+0.0102 vs 市場+0.0019」＝当てやすいレースを"
          "選んだのではない、という主張。市場側が素のHarvilleのままだった\n")

    cv = df["cv"].to_numpy()
    print(f"{'除外':>6}{'R数':>8}"
          f"{'D市場(素)':>12}{'D市場(λ)':>12}{'Dモデル':>11}"
          f"{'素の上昇':>10}{'λの上昇':>10}{'モデル上昇':>11}{'プラセボ(λ)':>12}")
    base = {}
    out_l = []
    for c in CUTS:
        th = np.quantile(cv, c) if c > 0 else -1.0
        m = cv > th
        n = int(m.sum())
        r0 = df.loc[m, "d_raw"].mean()
        r1 = df.loc[m, "d_lbs"].mean()
        rm = df.loc[m, "d_mdl"].mean()
        if c == 0.0:
            base = dict(raw=r0, lbs=r1, mdl=rm)
        pick = RNG.choice(len(df), size=n, replace=False)
        pl = df["d_lbs"].to_numpy()[pick].mean()
        out_l.append(r1 - base["lbs"])
        print(f"{c:>6.0%}{n:>8,}{r0:>+12.4f}{r1:>+12.4f}{rm:>+11.4f}"
              f"{r0-base['raw']:>+10.4f}{r1-base['lbs']:>+10.4f}"
              f"{rm-base['mdl']:>+11.4f}{pl-base['lbs']:>+12.4f}")

    lo, hi = mci(df.loc[cv > np.quantile(cv, 0.5), "d_lbs"])[1:]
    print(f"\n  50%除外時の λ補正D の99%CI: [{lo:+.4f}, {hi:+.4f}]  必要量 0.2549")
    print("\n★読み方（事前登録のとおり）")
    print("  ・**λの上昇**が(95)の +0.0019 と同程度なら、(95)③の交絡対照は本物。")
    print("  ・λの上昇が大きくなるなら、**素のHarvilleが市場側を不当に低く見せていた**だけ")
    print("    ＝『当てやすいレースを選んだのではない』という主張が崩れる。")
    print("  ・どちらでも必要量には遠い。**運用は変わらない**。")


if __name__ == "__main__":
    main()
