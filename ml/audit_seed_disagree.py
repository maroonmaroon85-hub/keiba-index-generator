"""(95) 宿題2 — **シード間の不一致を「除外」に使う**。判定は D（(89)の対数スコア差）で行う。

★なぜこれをやるのか
　(62)で「使えるのは選別ではなく**下位20%の除外**だけ」と分かっている。除外の基準は
　**枠連スコアだけ**しか試していない。(87)で**モデルの過信が高D領域に集中する**と分かったので、
　「**シード間で予測がばらつくレース＝モデルが自信を持てないレース**」を除外の基準にするのは筋が通る。
　(30)よりシードの乱数だけで部分集合ROIは61.7%↔116%動く＝**不一致は実在する量**。

★なぜROIでなくDで測るのか（(89)③・(53)(80)）
　ROIでの比較はこの標本量では測定限界に負ける（枠連で1ptの検出に50年）。Dは1レース1標本。
　しかも**ケリーが上限なので、Dが上がらないなら買い方をどう変えても無駄**。

★★事前登録（測る前に宣言する。(38)(46)より、後から基準を作ると必ず偽陽性が出る）

【指標】レース内でモデル確率を正規化してから、次の2つを作る。**主要指標はAで固定**。
　A `top1_flip` … 最頻の1位馬と一致しないシードの割合（0＝全シードが同じ馬を1位にした）
　　　**買うのは順位なので、順位の不安定さが本命**。
　B `cv_top` … 1位馬の確率のシード間 変動係数（std/mean）。**副次**。

【除外の水準】0%（除外なし）から 50% まで 10% 刻みの6水準。**不一致の大きい側から落とす**。
　★水準を後から増やさない。6水準で判定する。

【判定基準（4つ全部を満たしたときだけ「効いた」と言う）】
　1. **単調性**: 残った集合の `D(モデル) − D(市場)` が除外水準に対して単調に上がること
　   （6水準の Spearman ρ ≥ +0.7）。**最良の水準を選んで判定しない**((84)の教訓)。
　2. **プラセボ超え**: 除外30%の時点の差が、**同じ頭数を無作為に除外**した20回の分布の
　   99%上端を超えること。(79)①より「何かを除外したこと自体」で数字は動く。
　3. **市場と分離**: `D(市場)` の上昇幅より `D(モデル)` の上昇幅が大きいこと。
　   ★両方同じだけ上がるなら、それは「モデルが賢い」ではなく「**当てやすいレース**」を選んだだけ。
　4. **副次指標Bでも符号が同じ**こと（指標の作り方に依存しないこと）。

【★あらかじめ言っておく限界】
　(90)で `D(モデル) − D(市場)` は **−0.0846**（枠連）。**除外でこれが0を超えるとは考えにくい**。
　超えなければ「モデルは市場より下手だが、**この区分では下手さが小さい**」という記述にとどまり、
　**運用は変わらない**。それでも意味があるのは、(87)の「過信の集中」を独立に確認できるから。

★★第2版で足したもの（1回目の結果を見てから足した。**効果を殺す方向の検査なので後付けでも妥当**）
　1回目は指標Bで条件1〜3が通った（枠連: 差 −0.0838 → 除外50%で −0.0692）。だが
　**cv_top は「1位馬の確率が小さいほど大きくなる」＝モデルの自信の無さと同義になりうる**。
　もしそうなら、シードを5本引く必要すら無い（1本の予測から出る量で同じことができる）。
　→ **交絡対照を2つ追加**し、同じ除外率で比べる:
　　 C `p_top` … シード平均の1位馬確率（**シード1本でも作れる**）。これで同じ効果が出たら不一致は不要
　　 D `mkt_top` … 市場の1位馬含意確率（**モデル不要**）。これで出たら「堅いレース」を選んだだけ
　★判定を1つ足す: **条件5 — cv の効果が C・D の効果を同じ除外率で上回ること**。
　　下回るなら「シード間不一致」という道具は**不要**であり、宿題4は閉じる。
　なお指標Aは83.3%のレースが全シード一致（同点）で、パーセンタイル分割ができなかった。
　**実際に落とせるのは16.7%まで**なので、Aは自然な水準（flip>0 など）で出し直す。

実行: python3 ml/audit_seed_disagree.py [シード数(既定5)] [開始年(既定2015)]
"""
import math
import os
import sys
import warnings
from collections import Counter

import numpy as np
import pandas as pd
import lightgbm as lgb

warnings.filterwarnings("ignore")
sys.path.insert(0, "ml")
import features as F
from _cache import load_cached
from audit_crosspool import PAYBACK, load_races, payoff, probs, zq
from audit_crosspool2 import PARTS, PAYKEY, realized
from audit_crosspool3 import q_of
from place_wide import PARAMS

LEVELS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]      # ★事前登録。後から増やさない
PRIMARY = "枠連"                              # ★主要券種＝実運用の本命


def seed_probs(n_seed, y0):
    """ウォークフォワードで**シードごとの**勝率予測を作る。{raceid: {馬番: [p_seed...]}}

    ★production は seed 平均を使うので、q には平均を入れる。**シード別は不一致の測定にだけ使う**。
    """
    d, fx = load_cached()
    ywin = (d["finish"] == 1).astype(int).to_numpy()
    year = d["date"].dt.year.to_numpy()
    years = [yy for yy in range(y0, int(year.max()) + 1) if (year == yy).sum() > 5000]
    out = {}
    for yy in years:
        tr, te = year < yy, year == yy
        ps = np.stack([lgb.LGBMClassifier(random_state=s, **PARAMS)
                       .fit(fx[tr], ywin[tr], categorical_feature=F.CAT_COLS)
                       .predict_proba(fx[te])[:, 1] for s in range(n_seed)], axis=1)
        sub = d.loc[te, ["raceid", "umaban"]]
        for rid, um, row in zip(sub["raceid"], sub["umaban"].astype(int), ps):
            out.setdefault(rid, {})[int(um)] = row
        print(f"  {yy} 学習完了（{n_seed}シード）", flush=True)
    return out


def disagreement(mat):
    """シード別確率行列 (馬 × シード) から、事前登録した2指標を作る。

    レース内で**シードごとに正規化してから**比べる（絶対値の平行移動を落とす）。
    """
    z = mat / mat.sum(axis=0, keepdims=True)
    top = z.argmax(axis=0)                       # 各シードの1位馬
    mode = Counter(top.tolist()).most_common(1)[0][0]
    flip = float((top != mode).mean())            # A: 主要指標
    v = z[mode]
    cv = float(v.std(ddof=0) / v.mean()) if v.mean() > 0 else 0.0   # B: 副次
    return flip, cv


def collect(races, smap, y0):
    """(90)と同じ D を、レースごとの不一致つきで集める。"""
    rows = []
    for r in races:
        if r["year"] < y0:
            continue
        rl = realized(r)
        if rl is None:
            continue
        a, b, c = rl
        hs = r["horses"]
        p_mkt = probs(hs)
        mp = smap.get(r["rid"])
        if not mp:
            continue
        # ★過去走の無い馬はモデルが確率を出せない。(90)と同じく市場で埋めて再正規化する
        idx = [k for k, (num, _, _) in enumerate(hs) if num in mp]
        if len(idx) < 4:
            continue
        mat = np.array([mp[hs[k][0]] for k in idx], float)          # 馬 × シード
        flip, cv = disagreement(mat)
        raw = np.full(len(hs), np.nan)
        raw[idx] = mat.mean(axis=1)                                  # production は平均
        miss = np.isnan(raw)
        share = p_mkt[~miss].sum()
        s = raw[~miss].sum()
        if s <= 0:
            continue
        raw[~miss] = raw[~miss] / s * share
        raw[miss] = p_mkt[miss]
        p_use = raw / raw.sum()
        num2k = {num: k for k, (num, _, _) in enumerate(hs)}
        if a not in num2k or b not in num2k or (c is not None and c not in num2k):
            continue
        for kind, key in PARTS.items():
            if not r[key]:
                continue
            qm, combo = q_of(kind, r, p_use, num2k, a, b, c)
            qk, _ = q_of(kind, r, p_mkt, num2k, a, b, c)
            if qm <= 0 or qk <= 0 or combo is None:
                continue
            v = payoff(r, PAYKEY[kind], combo)
            if not v or v <= 0:
                continue
            lp = math.log((v + 5) / 100.0) - math.log(PAYBACK[kind])
            rows.append({"kind": kind, "rid": r["rid"], "year": r["year"],
                         "dm": math.log(qm) + lp, "dk": math.log(qk) + lp,
                         "flip": flip, "cv": cv, "n": r["n"],
                         # ★交絡対照。値が小さいほど「自信が無い」ので符号を反転して同じ向きに揃える
                         "p_top": -float(p_use.max()), "mkt_top": -float(p_mkt.max())})
    return pd.DataFrame(rows)


def mci(x, alpha=0.01):
    x = np.asarray(x, float)
    m = x.mean()
    se = x.std(ddof=1) / math.sqrt(len(x))
    z = zq(alpha)
    return m, m - z * se, m + z * se


def sweep(g, col, levels=None):
    """不一致の大きい側から落として、各水準の D を出す。

    ⚠**同点の扱い**: `flip` は83%が0で並ぶので分位では切れない（第2版で判明）。
    　分位が同点に当たった水準は**その水準を欠測にする**。無作為に同点を割るのは
    　「無作為除外」を混ぜることになり、プラセボと区別できなくなるため。
    """
    out = []
    for lv in (levels or LEVELS):
        if lv == 0:
            keep = g
        else:
            thr = g[col].quantile(1 - lv)
            keep = g[g[col] < thr]
            if len(keep) == 0 or len(keep) == len(g):
                out.append({"lv": lv, "n": np.nan, "actual": np.nan, "dm": np.nan,
                            "dk": np.nan, "diff": np.nan, "lo": np.nan, "hi": np.nan})
                continue
        dm, dk = keep["dm"].to_numpy(), keep["dk"].to_numpy()
        diff, lo, hi = mci(dm - dk)
        out.append({"lv": lv, "n": len(keep), "actual": 1 - len(keep) / len(g),
                    "dm": dm.mean(), "dk": dk.mean(), "diff": diff, "lo": lo, "hi": hi})
    return pd.DataFrame(out)


def sweep_value(g, col, thrs):
    """値そのもので切る（同点が多い指標A用）。thrs は「これ以上を落とす」境界。"""
    out = [{"thr": None, "n": len(g), "actual": 0.0,
            "dm": g["dm"].mean(), "dk": g["dk"].mean(),
            "diff": (g["dm"] - g["dk"]).mean()}]
    for t in thrs:
        keep = g[g[col] < t]
        if len(keep) < 500:
            continue
        dm, dk = keep["dm"].to_numpy(), keep["dk"].to_numpy()
        diff, lo, hi = mci(dm - dk)
        out.append({"thr": t, "n": len(keep), "actual": 1 - len(keep) / len(g),
                    "dm": dm.mean(), "dk": dk.mean(), "diff": diff})
    return pd.DataFrame(out)


def placebo(g, frac, n_rep=20, seed=0):
    """同じ頭数を無作為に除外したときの差の分布。(79)①より必ず並べる。"""
    rng = np.random.default_rng(seed)
    keepn = int(round(len(g) * (1 - frac)))
    d = (g["dm"] - g["dk"]).to_numpy()
    vals = [d[rng.choice(len(d), keepn, replace=False)].mean() for _ in range(n_rep)]
    return np.percentile(vals, 0.5), np.percentile(vals, 99.5)


def main():
    n_seed = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    y0 = int(sys.argv[2]) if len(sys.argv) > 2 else 2015
    print(f"(95) シード間不一致による除外（{y0}年以降・シード{n_seed}本・ウォークフォワード）")
    print("★事前登録した4条件＋交絡対照の条件5を全部満たしたときだけ『効いた』と言う\n")
    # ★学習に25分かかるので、集めた行を保存して再解析を軽くする
    cache = f"data/cache/seed_dis_{n_seed}_{y0}.csv"
    if os.path.exists(cache):
        df = pd.read_csv(cache)
        print(f"  キャッシュを読んだ: {cache}")
    else:
        smap = seed_probs(n_seed, y0)
        df = collect(load_races(), smap, y0)
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        df.to_csv(cache, index=False)
        print(f"  キャッシュを書いた: {cache}")
    print(f"\n対象 {len(df):,}件 / {df['rid'].nunique():,}レース")

    uni = df.drop_duplicates("rid")
    print(f"\n{'='*100}")
    print("【0】不一致そのものの分布（除外の道具になる量があるか）")
    print(f"{'='*100}")
    print(f"  A top1_flip: 平均 {uni['flip'].mean():.4f} / "
          f"全シード一致のレース {(uni['flip'] == 0).mean()*100:.1f}%")
    print(f"  B cv_top   : 平均 {uni['cv'].mean():.4f} / 中央値 {uni['cv'].median():.4f}")
    if (uni["flip"] == 0).mean() > 1 - max(LEVELS):
        print("  ⚠**Aは同点が多い**（全シード一致が多数）ので、"
              "上位x%を落とそうとしても実際の除外率が水準どおりにならない。実測値を下に出す。")

    # ───────── 指標A は同点だらけなので値で切る ─────────
    print(f"\n{'='*100}")
    print("★【A top1_flip（主要）】— 同点が多いので**値そのもの**で切る")
    print(f"{'='*100}")
    for kind in PARTS:
        g = df[df["kind"] == kind]
        if len(g) < 2000:
            continue
        s = sweep_value(g, "flip", [0.01, 0.25, 0.45])   # 「1本でも外れたら除外」「2本」「3本」
        base = s.iloc[0]["diff"]
        print(f"\n{'★' if kind == PRIMARY else ' '}{kind}")
        print(f"{'落とす条件':<22}{'実除外':>8}{'R数':>9}{'D(モデル)':>12}"
              f"{'D(市場)':>11}{'差':>11}{'0%からの変化':>14}")
        for _, r in s.iterrows():
            lab = "除外なし" if r["thr"] is None else f"flip≧{r['thr']:.2f} を除外"
            print(f"{lab:<22}{r['actual']*100:>7.1f}%{r['n']:>9,.0f}{r['dm']:>+12.4f}"
                  f"{r['dk']:>+11.4f}{r['diff']:>+11.4f}{r['diff']-base:>+14.4f}")

    for col, lab, tag in (("cv", "B cv_top（副次）", "★"),
                          ("p_top", "C p_top＝モデルの1位確率（交絡対照・シード1本で作れる）", "⚠"),
                          ("mkt_top", "D mkt_top＝市場の1位確率（交絡対照・モデル不要）", "⚠")):
        print(f"\n{'='*100}")
        print(f"{tag}【{lab}】で不安定な側を落とす")
        print(f"{'='*100}")
        for kind in PARTS:
            g = df[df["kind"] == kind]
            if len(g) < 2000:
                continue
            s = sweep(g, col)
            need = -math.log(PAYBACK[kind])
            star = "★" if kind == PRIMARY else " "
            print(f"\n{star}{kind}（必要量 {need:.4f}）")
            print(f"{'除外':>6}{'実除外':>8}{'R数':>9}{'D(モデル)':>12}{'D(市場)':>11}"
                  f"{'差':>11}{'差の99%CI':>22}{'0%からの変化':>14}")
            base = s.iloc[0]["diff"]
            for _, r in s.iterrows():
                print(f"{r['lv']*100:>5.0f}%{r['actual']*100:>7.1f}%{r['n']:>9,.0f}"
                      f"{r['dm']:>+12.4f}{r['dk']:>+11.4f}{r['diff']:>+11.4f}"
                      f"{f'[{r.lo:+.4f},{r.hi:+.4f}]':>22}{r['diff']-base:>+14.4f}")
            rho = pd.Series(s["diff"].to_numpy()).corr(pd.Series(LEVELS), method="spearman")
            dm_rise = s.iloc[3]["dm"] - s.iloc[0]["dm"]
            dk_rise = s.iloc[3]["dk"] - s.iloc[0]["dk"]
            plo, phi = placebo(g, s.iloc[3]["actual"])
            c1 = rho >= 0.7
            c2 = s.iloc[3]["diff"] > phi
            c3 = dm_rise > dk_rise
            print(f"   条件1 単調性 ρ={rho:+.3f} {'○' if c1 else '✕'}"
                  f" / 条件2 除外30%の差 {s.iloc[3]['diff']:+.4f} vs プラセボ99%上端 {phi:+.4f}"
                  f" {'○' if c2 else '✕'}")
            print(f"   条件3 上昇幅 モデル {dm_rise:+.4f} vs 市場 {dk_rise:+.4f} {'○' if c3 else '✕'}"
                  f" / 差が0を超えたか {'○' if s['diff'].max() > 0 else '✕'}")

    # ───────── ★条件5: 交絡対照との比較（同じ除外率で） ─────────
    print(f"\n{'='*100}")
    print("★★【条件5】交絡対照との比較 — シード間不一致は**そもそも要るのか**")
    print(f"{'='*100}")
    print("  同じ除外率で、B(cv) と C(p_top) と D(mkt_top) の『差の改善』を並べる。")
    print("  ★C が B と同等以上なら、**シードを5本引く意味は無い**（1本の予測から作れる量で足りる）。")
    print("  ★D が同等以上なら、**モデルすら要らない**（市場のオッズだけで同じことができる）。\n")
    print(f"{'券種':<8}{'除外率':>8}{'B cv(不一致)':>14}{'C p_top':>11}{'D mkt_top':>12}"
          f"{'B−C':>10}{'B−D':>10}{'判定':>18}")
    verdict = {}
    for kind in PARTS:
        g = df[df["kind"] == kind]
        if len(g) < 2000:
            continue
        sb, sc, sd = (sweep(g, c) for c in ("cv", "p_top", "mkt_top"))
        for i in (3, 5):                       # 30% と 50%
            b = sb.iloc[i]["diff"] - sb.iloc[0]["diff"]
            c = sc.iloc[i]["diff"] - sc.iloc[0]["diff"]
            d = sd.iloc[i]["diff"] - sd.iloc[0]["diff"]
            ok = b > c and b > d
            if i == 3:
                verdict[kind] = ok
            print(f"{kind:<8}{LEVELS[i]*100:>7.0f}%{b:>+14.4f}{c:>+11.4f}{d:>+12.4f}"
                  f"{b-c:>+10.4f}{b-d:>+10.4f}{'○ 不一致が上' if ok else '✕ 対照に負け':>18}")
    n_ok = sum(verdict.values())
    print(f"\n  → 除外30%で不一致が対照を上回った券種: **{n_ok}/{len(verdict)}**")
    if n_ok == 0:
        print("  ★**条件5は全滅**。シード間不一致という道具は不要。宿題4は閉じる。")
        print("  　（『モデルが自信を持てないレースを外す』という筋自体は生きているが、"
              "それは1位確率で足りる）")

    print(f"\n{'='*100}")
    print("★読み方")
    print("  ・条件1〜5が全部○のときだけ『不一致による除外は効く』と言える。1つでも✕なら効いていない。")
    print("  ・条件3が✕（市場のDも同じだけ上がる）なら、選んだのは"
          "**モデルが賢いレース**ではなく**当てやすいレース**。")
    print(f"  ・差が0を超えない限り、モデルは市場より下手なまま。(90)の枠連 −0.0846 が出発点。")
    print("  ・どの水準でも必要量には遠いはずで、**運用は変わらない**。")


if __name__ == "__main__":
    main()
