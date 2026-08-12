"""(117) ★★★運用の「枠連スコア下位20%を除外」を**Dで測り直す** — (62)はROI時代の結論（2026-08-11）

★なぜ最優先か
　`predict_nk.py` は毎週**この基準で実際にレースを外している**（`waku_score_p20`）。
　だが(62)は**ROIで決めた**: 「絞っても回収率は上がらない（下位20%を避けて84.5→85.6%が限界）」。
　(89)以降くり返し確認してきたとおり、**ROIでは1ptの差を検出できない**（枠連で50年かかる）。
　→ **運用に直結する基準だけが、いちばん古い道具のまま残っていた**。(111)→(112)と同じ構図。

★測るもの
　除外率 0/10/20/30/40% で **E[d|S]**（λ補正した市場Harville・(96)の最良形）。
　ケリーが上限なので、**E[d|S] が上がらなければ、その除外はどんな買い方をしても得をしない**。

★★事前登録（測る前に宣言）
　1. **閾値は 0/10/20/30/40% の5水準**。後から増やさない。
　2. **判定は単調性（Spearman ρ）**。最良のビンでは判定しない（(106)(111)(113)の教訓）。
　3. **プラセボ**: 同じ本数を無作為に除外して同じ手続き。効果量は「実測−プラセボ」。
　4. **閾値はウォークフォワード**（その年より前の年だけで決める）。運用と同じ手続き。
　5. **予想**: **ほぼ動かない**（+0.002未満）。理由は(114)——事前情報によるレース選択の正味は
　　 枠連で +0.0218＝必要量の8.5%が上限で、しかもそれは**軸のλ補正top3**という強い変数での話。
　　 枠連スコアはモデル由来なので、(102)で混合の重みが14%しか付かなかったことと整合する。
　6. **★判定の意味**: 上がらなければ「**除外は無意味だが有害でもない**」。
　　 下がれば「**除外をやめるべき**」＝運用が変わる。**変わる可能性がある実験はこれが久しぶり**。

実行: python3 ml/audit_excl_d.py [開始年(既定2015)]
"""
import glob
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "ml")
from audit_crosspool import PAYBACK, load_races, payoff, probs, zq
from audit_crosspool2 import PARTS, PAYKEY, realized
from audit_lbs import build_matrix, fit_lambda, q_of_lbs
from waku_umatan import bracket_probs, waku_score, wakuren_buy

CUTS = [0.0, 0.10, 0.20, 0.30, 0.40]     # ★先に宣言。20%が現行の運用
MODEL_CACHE = "data/cache/exp_L2-top3_2015"
RNG = np.random.default_rng(20260811)


def mci(x, alpha=0.01):
    x = np.asarray(x, float)
    if len(x) < 2:
        return float("nan"), float("nan"), float("nan")
    m = x.mean()
    se = x.std(ddof=1) / math.sqrt(len(x))
    z = zq(alpha)
    return m, m - z * se, m + z * se


def spearman(xs, ys):
    def rank(v):
        o = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for pos, i in enumerate(o):
            r[i] = pos
        return r
    a, b = rank(xs), rank(ys)
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    den = math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))
    return num / den if den else float("nan")


def load_model_p():
    """(97)が作ったウォークフォワードのモデル確率 → {raceid: {馬番: p}}。"""
    fs = sorted(glob.glob(f"{MODEL_CACHE}/*.csv"))
    if not fs:
        sys.exit(f"{MODEL_CACHE} が無い。先に (97)/(102) の実験を回してキャッシュを作ること。")
    out = {}
    for f in fs:
        for rid, u, p in pd.read_csv(f)[["raceid", "umaban", "p"]].itertuples(index=False):
            out.setdefault(str(rid), {})[int(u)] = float(p)
    return out


def main():
    y0 = int(sys.argv[1]) if len(sys.argv) > 1 else 2015
    mp = load_model_p()
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
        pm = mp.get(r["rid"])
        if not pm:
            continue
        rl = realized(r)
        if rl is None:
            continue
        a, b, c = rl
        hs = r["horses"]
        nums = [u for u, _, _ in hs]
        if any(u not in pm for u in nums):
            continue
        num2k = {u: k for k, (u, _, _) in enumerate(hs)}
        if a not in num2k or b not in num2k:
            continue
        order = sorted(nums, key=lambda u: -pm[u])          # モデル降順
        bp = bracket_probs(nums, [pm[u] for u in nums], r["n"])
        pairs = wakuren_buy(order, r["n"], 2)
        sc = waku_score(pairs, bp)
        p = probs(hs)
        l2, l3 = lam[yy]
        q, combo = q_of_lbs("枠連", r, p, l2, l3, num2k, a, b, c)
        if q <= 0 or combo is None:
            continue
        v = payoff(r, PAYKEY["枠連"], combo)
        if not v or v <= 0:
            continue
        d = math.log(q) + math.log((v + 5) / 100.0) - math.log(PAYBACK["枠連"])
        # 実際の買い目（軸枠×紐枠2）の払戻。参考値としてROIも出す
        ret = sum(payoff(r, PAYKEY["枠連"], list(pr)) or 0.0 for pr in pairs)
        rows.append(dict(year=yy, sc=sc, d=d, bet=len(pairs) * 100, ret=ret))

    df = pd.DataFrame(rows)
    print(f"(117) 「枠連スコア下位N%を除外」を D で測り直す（{y0}年以降・{len(df):,}レース）")
    print("★(62)はROIで決めた基準。運用が実際にこれでレースを外している\n")

    # ★閾値はウォークフォワード（その年より前の年だけで決める）＝運用と同じ手続き
    ys = sorted(df["year"].unique())
    keep = {c: np.zeros(len(df), bool) for c in CUTS}
    for c in CUTS:
        for yy in ys:
            te = (df["year"] == yy).to_numpy()
            tr = (df["year"] < yy).to_numpy()
            if c == 0.0:
                keep[c] |= te
                continue
            if tr.sum() < 2000:
                continue
            th = np.quantile(df.loc[tr, "sc"], c)
            keep[c] |= te & (df["sc"].to_numpy() > th)
    base = keep[0.0] & (df["year"] >= ys[1] if len(ys) > 1 else True)
    for c in CUTS:                       # 閾値を出せた年だけで揃える
        keep[c] &= (df["year"] > ys[0]).to_numpy()

    d = df["d"].to_numpy()
    bet = df["bet"].to_numpy(float)
    ret = df["ret"].to_numpy(float)
    need = -math.log(PAYBACK["枠連"])
    print(f"{'除外率':>7}{'R数':>8}{'E[d|S]':>10}{'99%CI下':>10}{'上':>9}"
          f"{'ROI':>8}{'プラセボE[d]':>13}{'実測−プラセボ':>14}")
    obs, pla = [], []
    for c in CUTS:
        m = keep[c]
        n = int(m.sum())
        dm, lo, hi = mci(d[m])
        roi = ret[m].sum() / bet[m].sum() * 100 if bet[m].sum() else float("nan")
        # ★プラセボは200回引いて平均する。1回引きだと標本誤差で勝手に単調に見える
        # 　（最初そうなっていた: プラセボρ=+0.900。無作為抽出は不偏なので本来は平坦）。
        idx = np.where(keep[0.0])[0]
        pm_ = float(np.mean([d[RNG.choice(idx, size=min(n, len(idx)), replace=False)].mean()
                             for _ in range(200)]))
        obs.append(dm)
        pla.append(pm_)
        print(f"{c:>7.0%}{n:>8,}{dm:>+10.4f}{lo:>+10.4f}{hi:>+9.4f}"
              f"{roi:>8.1f}{pm_:>+13.4f}{dm-pm_:>+14.4f}")
    rho = spearman(list(range(len(CUTS))), obs)
    rho_p = spearman(list(range(len(CUTS))), pla)
    print(f"\n  単調性 Spearman ρ = {rho:+.3f}（プラセボ {rho_p:+.3f}）  必要量 {need:.4f}")

    # ★★外した側と残した側を直接比べる（互いに素なので2標本検定ができる）。
    # 　入れ子の部分集合どうしのCIを見比べても有意性は判定できないため、これが本体。
    print(f"\n★外した側 vs 残した側（互いに素・こちらが判定の本体）")
    print(f"{'除外率':>7}{'外したR数':>10}{'外したD':>10}{'残したD':>10}"
          f"{'差':>10}{'99%CI':>22}{'判定':>7}")
    for c in CUTS[1:]:
        drop = keep[0.0] & ~keep[c] & (df["year"] > ys[0]).to_numpy()
        kp = keep[c]
        a, b = d[drop], d[kp]
        if len(a) < 100:
            continue
        diff = b.mean() - a.mean()
        se = math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
        z = zq(0.01)
        lo2, hi2 = diff - z * se, diff + z * se
        print(f"{c:>7.0%}{len(a):>10,}{a.mean():>+10.4f}{b.mean():>+10.4f}"
              f"{diff:>+10.4f}{f'[{lo2:+.4f},{hi2:+.4f}]':>22}"
              f"{'★' if lo2 > 0 else '':>7}")

    # ★年ごとに符号が揃うか（(112)と同じ確認）
    print(f"\n★年分割（除外20%・現行の運用）")
    pos = 0
    yl = [y for y in ys[1:]]
    for yy in yl:
        te = (df["year"] == yy).to_numpy()
        a, b = d[te & keep[0.0] & ~keep[0.20]], d[te & keep[0.20]]
        if len(a) < 30 or len(b) < 30:
            continue
        pos += (b.mean() - a.mean()) > 0
        print(f"   {yy}  外し{len(a):>4}本 {a.mean():+.4f} / 残し{len(b):>5}本 {b.mean():+.4f}"
              f"  差 {b.mean()-a.mean():+.4f}")
    print(f"   → {pos}/{len(yl)} 年で正")
    print("\n★読み方（事前登録のとおり）")
    print("  ・ρが0近傍 → **除外は無意味だが有害でもない**。現行のまま置いてよい。")
    print("  ・ρが負（除外するほどDが下がる） → **除外をやめるべき**。運用が変わる。")
    print("  ・ρが正でも、実測−プラセボが小さければ「人気の偏りを拾っただけ」。")
    print("  ・どの水準でも必要量 0.2549 には遠いはずで、**儲かるようにはならない**。")


if __name__ == "__main__":
    main()
