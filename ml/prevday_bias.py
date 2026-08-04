"""
(74) 「土曜の馬場傾向を日曜の予想に反映できるか」の検証。

⚠**(63)が強い反証**: モデルは `cond`（馬場状態）を一度も分岐に使っておらず（重要度0.00%・44位）、
全30,296レースの馬場を書き換えても買い目は**0.0%**しか変わらない。理由は**オッズが既に馬場を織り込んでいる**から。
同じ理屈が「前日の傾向」にも効くはずなので、ここでは**3段構え**で、安い順に潰す:

  段1. **傾向はそもそも翌日に持ち越すのか**
       前日の同場・同馬場種別の「内枠有利度」「先行有利度」が、**翌日の同じ指標を予測するか**。
       ここが0なら以降は無意味。
  段2. **持ち越すとして、それはオッズ／モデルに入っていないのか**
       残差（実際の3着以内 − モデル確率）を使う。モデルはオッズ入りなので、
       **残差に前日の傾向が効くなら、市場が織り込み残している**ということになる。
  段3. **買い目の後段フィルタとして効くのか**
       枠連の紐2頭を「モデル上位3頭から、前日の傾向に沿う枠を優先」に差し替えて、
       **的中率**で比べる（(66)④の通り、1券種の差はROIだと標本誤差で潰れる）。

前日＝同じ場で**8日以内の直前の開催日**（土日開催なら土曜、金土日なら前日）。
傾向は前日の**実際の結果**から作るので、当日の情報は一切使っていない。

実行: python3 ml/prevday_bias.py [シード数(既定3)]
"""
import sys
import warnings

import numpy as np
import pandas as pd
import lightgbm as lgb

warnings.filterwarnings("ignore")
sys.path.insert(0, "ml")
import features as F
from market_baseline import load
from place_wide import PARAMS, boot
from waku_umatan import load_wu, waku_of

PAYOUT = "data/payout/a.csv"
MIN_RACES = 5          # 前日にこの数以上のレースが無い場・馬場は傾向を作らない
INNER = 4              # 枠1..4を内、5..8を外とする
PACE = 0.4             # 通過順平均/頭数 がこれ以下なら「前で運ぶ馬」


def add_waku(d):
    n = d["fieldsize"].to_numpy(int)
    u = d["umaban"].to_numpy(int)
    d["waku"] = [waku_of(a, b) for a, b in zip(u, n)]
    return d


def day_bias(d):
    """{(場, 馬場, 日付): {"inner":内枠有利度, "pace":先行有利度, "n":レース数}}。

    有利度は finratio（＝(頭数−着順+1)/頭数、1に近いほど上位）の**群間差**。
    ＋なら内枠（前）が有利だった日。
    """
    out = {}
    g = d.groupby(["course", "surface", "date"], sort=False)
    for (c, s, dt), x in g:
        nr = x["raceid"].nunique()
        if nr < MIN_RACES:
            continue
        inn, out_ = x["waku"] <= INNER, x["waku"] > INNER
        fwd, bwd = x["passratio"] <= PACE, x["passratio"] > PACE
        if inn.sum() < 20 or out_.sum() < 20 or fwd.sum() < 20 or bwd.sum() < 20:
            continue
        out[(c, s, dt)] = {
            "inner": float(x.loc[inn, "finratio"].mean() - x.loc[out_, "finratio"].mean()),
            "pace": float(x.loc[fwd, "finratio"].mean() - x.loc[bwd, "finratio"].mean()),
            "n": nr}
    return out


def prev_day_map(d):
    """{(場, 日付): 直前の開催日}。同じ場で8日以内のものだけ（＝同じ開催の中）。"""
    mp = {}
    for c, x in d.groupby("course", sort=False):
        days = np.sort(x["date"].unique())
        for i in range(1, len(days)):
            if (days[i] - days[i - 1]) / np.timedelta64(1, "D") <= 8:
                mp[(c, pd.Timestamp(days[i]))] = pd.Timestamp(days[i - 1])
    return mp


def main():
    n_seed = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    d, fx, odds = load()
    d = add_waku(d)
    y = (d["finish"] <= 3).astype(int).to_numpy()
    cut = d["date"].quantile(0.3)
    tr, te = (d["date"] < cut).to_numpy(), (d["date"] >= cut).to_numpy()
    print(f"train {tr.sum():,} / test {te.sum():,}（分割 {cut.date()}）  シード{n_seed}本")

    bias = day_bias(d)
    prev = prev_day_map(d)
    print(f"傾向を作れた 場×馬場×日: {len(bias):,} / 直前の開催日が引ける 場×日: {len(prev):,}")

    # ===== 段1. 傾向は翌日に持ち越すか =====
    rows = []
    for (c, s, dt), b in bias.items():
        p = prev.get((c, dt))
        pb = bias.get((c, s, p)) if p is not None else None
        if pb:
            rows.append({"course": c, "surface": s, "date": dt,
                         "inner": b["inner"], "pace": b["pace"],
                         "p_inner": pb["inner"], "p_pace": pb["pace"]})
    t1 = pd.DataFrame(rows)
    print("\n" + "=" * 84)
    print(f"【段1】前日の傾向は翌日に持ち越すか  {len(t1):,} 組（場×馬場×日）")
    print("=" * 84)
    rng = np.random.default_rng(0)
    for k, name in [("inner", "内枠有利度"), ("pace", "先行有利度")]:
        a, b = t1[f"p_{k}"].to_numpy(float), t1[k].to_numpy(float)
        r = float(np.corrcoef(a, b)[0, 1])
        idx = rng.integers(0, len(a), size=(2000, len(a)))
        rs = np.array([np.corrcoef(a[i], b[i])[0, 1] for i in idx[:500]])
        print(f"  {name}: 前日→当日の相関 r={r:+.3f} "
              f"[{np.percentile(rs,2.5):+.3f},{np.percentile(rs,97.5):+.3f}]"
              f"   前日の標準偏差 {a.std():.4f} / 当日 {b.std():.4f}")
        hi = a > np.percentile(a, 75)
        lo = a < np.percentile(a, 25)
        print(f"    前日が{name}上位25%の日の当日値 {b[hi].mean():+.4f} / "
              f"下位25%の日 {b[lo].mean():+.4f}  差 {b[hi].mean()-b[lo].mean():+.4f}")

    # ===== モデルを学習（段2・段3用） =====
    ms = [lgb.LGBMClassifier(random_state=s, **PARAMS).fit(fx[tr], y[tr],
                                                           categorical_feature=F.CAT_COLS)
          for s in range(n_seed)]
    p_te = np.mean([m.predict_proba(fx[te])[:, 1] for m in ms], axis=0)
    print(f"  seed {n_seed}本の学習完了")

    sub = d.loc[te, ["raceid", "umaban", "fieldsize", "course", "surface", "date",
                     "waku"]].copy()
    sub["odds"] = odds[te]
    sub["p"] = p_te
    sub["top3"] = y[te]
    sub["style"] = fx.loc[te, "avg3_passratio"].to_numpy(float)   # 近3走の通過順（事前に見える）
    sub["pb_inner"] = [bias.get((c, s, prev.get((c, dt))), {}).get("inner", np.nan)
                       for c, s, dt in zip(sub["course"], sub["surface"], sub["date"])]
    sub["pb_pace"] = [bias.get((c, s, prev.get((c, dt))), {}).get("pace", np.nan)
                      for c, s, dt in zip(sub["course"], sub["surface"], sub["date"])]

    # ===== 段2. 傾向はモデル/オッズに入っていない情報か（残差で見る） =====
    print("\n" + "=" * 84)
    print("【段2】残差（実際の3着以内 − モデル確率）に前日の傾向が残っているか")
    print("=" * 84)
    s2 = sub[sub["pb_inner"].notna()].copy()
    s2["resid"] = s2["top3"] - s2["p"]
    print(f"対象 {len(s2):,}頭 / {s2['raceid'].nunique():,}レース"
          f"（全体の残差 {s2['resid'].mean()*100:+.3f}pt＝モデルの偏り）")
    print(f"{'仮説':<34}{'沿う馬':>12}{'逆らう馬':>12}{'差':>10}{'95%CI':>18}")
    for key, zfun, name in [
        ("pb_inner", lambda x: np.where(x["waku"] <= INNER, 1, -1), "前日内枠有利→内枠が来る"),
        ("pb_pace", lambda x: np.where(x["style"] <= PACE, 1, -1), "前日先行有利→先行馬が来る"),
    ]:
        z = zfun(s2) * np.sign(s2[key].to_numpy(float))
        a = s2.loc[z > 0, "resid"].to_numpy(float)
        b = s2.loc[z < 0, "resid"].to_numpy(float)
        if len(a) < 500 or len(b) < 500:
            print(f"{name:<34} 標本不足")
            continue
        la, ha = boot(a, rng, 1000)
        lb, hb = boot(b, rng, 1000)
        # 対応なしの差のCI
        ia = rng.integers(0, len(a), size=(1000, len(a)))
        ib = rng.integers(0, len(b), size=(1000, len(b)))
        dd = (a[ia].mean(axis=1) - b[ib].mean(axis=1)) * 100
        print(f"{name:<34}{a.mean()*100:>+11.3f}pt{b.mean()*100:>+11.3f}pt"
              f"{(a.mean()-b.mean())*100:>+9.3f}pt"
              f"{f'[{np.percentile(dd,2.5):+.3f},{np.percentile(dd,97.5):+.3f}]':>18}")
    print("  ※差が0を跨ぐなら「前日の傾向はオッズ＋モデルに既に入っている」＝(63)と同じ結論。")

    # ★プラセボ: 前日の傾向の符号を**日をまたいでシャッフル**して同じ統計を作る。
    #   「内枠有利の日」が多数派だと、沿う群が内枠馬ばかりになり、
    #   モデルが持つ枠の系統誤差がそのまま差として出てしまう。それを切り分ける。
    print("\n  プラセボ（前日の傾向を無関係な日のものと入れ替えて同じ計算）")
    day_key = s2["course"].astype(str) + "|" + s2["surface"].astype(str) + "|" \
        + s2["date"].astype(str)
    for key, zfun, name in [
        ("pb_inner", lambda x: np.where(x["waku"] <= INNER, 1, -1), "内枠"),
        ("pb_pace", lambda x: np.where(x["style"] <= PACE, 1, -1), "先行"),
    ]:
        sgn = np.sign(s2[key].to_numpy(float))
        z0 = zfun(s2)
        pos = float((sgn > 0).mean())
        days = pd.factorize(day_key)[0]
        vals = pd.Series(sgn).groupby(days).first().to_numpy()
        obs = None
        placebo = []
        for i in range(200):
            v = rng.permutation(vals)[days]
            z = z0 * v
            a = s2.loc[z > 0, "resid"].to_numpy(float)
            b = s2.loc[z < 0, "resid"].to_numpy(float)
            placebo.append((a.mean() - b.mean()) * 100)
        z = z0 * sgn
        obs = (s2.loc[z > 0, "resid"].mean() - s2.loc[z < 0, "resid"].mean()) * 100
        p = float((np.abs(placebo) >= abs(obs)).mean())
        print(f"    {name}: 観測 {obs:+.3f}pt / プラセボ {np.mean(placebo):+.3f}±"
              f"{np.std(placebo):.3f}pt（95%点 {np.percentile(np.abs(placebo),95):.3f}）"
              f" → p={p:.3f}   ※「有利」判定の日の割合 {pos*100:.1f}%")

    # ===== 段3. 枠連の紐選びを傾向で差し替える =====
    print("\n" + "=" * 84)
    print("【段3】枠連の紐2頭を「前日の傾向に沿う枠」優先に差し替える（評価は的中率）")
    print("=" * 84)
    wu = load_wu(PAYOUT)
    rows = []
    for rid, g in sub.groupby("raceid", sort=False):
        w = wu.get(rid)
        if not w or not w["wakuren"] or len(g) < 3:
            continue
        b = g["pb_inner"].iloc[0]
        if not np.isfinite(b) or b == 0:
            continue
        gg = g.sort_values("p", ascending=False, kind="mergesort")
        nums = gg["umaban"].astype(int).tolist()
        wk = dict(zip(gg["umaban"].astype(int), gg["waku"].astype(int)))
        n = int(g["fieldsize"].iloc[0])
        base = nums[1:3]
        cand = nums[1:4]
        # 前日が内枠有利(b>0)なら内枠から2頭、外枠有利なら外枠から2頭
        alt = sorted(cand, key=lambda u: (wk[u] if b > 0 else -wk[u]))[:2]
        r = {"year": g["date"].iloc[0].year}
        for tag, himo in [("base", base), ("bias", alt)]:
            cs = sorted({tuple(sorted((wk[nums[0]], wk[h]))) for h in himo})
            pay = sum(w["wakuren"].get(c, 0) for c in cs)
            r[f"{tag}_roi"] = pay / (len(cs) * 100.0)
            r[f"{tag}_hit"] = float(pay > 0)
        r["changed"] = set(base) != set(alt)
        rows.append(r)
    t3 = pd.DataFrame(rows)
    print(f"対象 {len(t3):,}R / 買い目が変わったのは {t3['changed'].mean()*100:.1f}%")
    for tag, name in [("base", "モデル順どおり（現行）"), ("bias", "前日の傾向で紐を差し替え")]:
        h = t3[f"{tag}_hit"].to_numpy(float)
        r_ = t3[f"{tag}_roi"].to_numpy(float)
        lo, hi = boot(r_, rng, 2000)
        print(f"  {name:<28} 的中率{h.mean()*100:>6.2f}%  ROI{r_.mean()*100:>6.1f}% "
              f"[{lo:.1f},{hi:.1f}]")
    dh = (t3["bias_hit"] - t3["base_hit"]).to_numpy(float)
    dr = (t3["bias_roi"] - t3["base_roi"]).to_numpy(float)
    lo, hi = boot(dh, rng, 2000)
    lo2, hi2 = boot(dr, rng, 2000)
    print(f"  差（対応あり）: 的中率 {dh.mean()*100:+.2f}pt [{lo:+.2f},{hi:+.2f}] / "
          f"ROI {dr.mean()*100:+.2f}pt [{lo2:+.2f},{hi2:+.2f}]")
    ch = t3[t3["changed"]]
    if len(ch) > 200:
        dh2 = (ch["bias_hit"] - ch["base_hit"]).to_numpy(float)
        lo3, hi3 = boot(dh2, rng, 2000)
        print(f"  買い目が変わったレースだけ（{len(ch):,}R）: 的中率 {dh2.mean()*100:+.2f}pt "
              f"[{lo3:+.2f},{hi3:+.2f}]")

    # ===== 段4. モデル順を壊さない形（拮抗しているときだけ入れ替える） =====
    # 段3が悪化したのは「モデルの順位を1つ落とすコスト」が段2の+0.64ptを上回るから。
    # ならば**モデルが甲乙つけがたいと言っているときだけ**入れ替えれば、そのコストは小さいはず。
    # 閾値は事前に決め打ちする（後から動かすと(38)の総当たりと同じことになる）。
    GAPS = (0.02, 0.05, 0.10)     # 紐候補のレース内シェアの差がこれ以下なら「拮抗」
    print("\n" + "=" * 84)
    print("【段4】拮抗しているときだけ入れ替える（モデル順を壊さない形）")
    print("=" * 84)
    rows = []
    for rid, g in sub.groupby("raceid", sort=False):
        w = wu.get(rid)
        if not w or not w["wakuren"] or len(g) < 4:
            continue
        b = g["pb_inner"].iloc[0]
        if not np.isfinite(b) or b == 0:
            continue
        gg = g.sort_values("p", ascending=False, kind="mergesort")
        nums = gg["umaban"].astype(int).tolist()
        wk = dict(zip(gg["umaban"].astype(int), gg["waku"].astype(int)))
        pv = gg["p"].to_numpy(float)
        share = pv / pv.sum()
        n = int(g["fieldsize"].iloc[0])
        base = nums[1:3]
        r = {"gap": float(share[2] - share[3])}       # 3番手と4番手の差＝入れ替え候補の拮抗度
        for tag, himo in [("base", base)] + [
                (f"g{int(gp*100)}", (sorted(nums[1:4], key=lambda u: (wk[u] if b > 0 else -wk[u]))[:2]
                                     if (share[2] - share[3]) <= gp else base))
                for gp in GAPS]:
            cs = sorted({tuple(sorted((wk[nums[0]], wk[h]))) for h in himo})
            pay = sum(w["wakuren"].get(c, 0) for c in cs)
            r[f"{tag}_roi"] = pay / (len(cs) * 100.0)
            r[f"{tag}_hit"] = float(pay > 0)
        rows.append(r)
    t4 = pd.DataFrame(rows)
    print(f"対象 {len(t4):,}R")
    print(f"{'規則':<28}{'発動率':>8}{'的中率':>9}{'的中率の差':>13}{'95%CI':>18}{'ROI':>8}")
    bh = t4["base_hit"].to_numpy(float)
    print(f"{'モデル順どおり（現行）':<28}{'—':>8}{bh.mean()*100:>8.2f}%{'—':>13}{'—':>18}"
          f"{t4['base_roi'].mean()*100:>7.1f}%")
    for gp in GAPS:
        tag = f"g{int(gp*100)}"
        d_ = (t4[f"{tag}_hit"] - t4["base_hit"]).to_numpy(float)
        lo, hi = boot(d_, rng, 2000)
        fire = (t4["gap"] <= gp).mean() * 100
        print(f"{f'シェア差{gp*100:.0f}pt以下なら入替':<28}{fire:>7.1f}%"
              f"{t4[f'{tag}_hit'].mean()*100:>8.2f}%{d_.mean()*100:>+12.2f}pt"
              f"{f'[{lo:+.2f},{hi:+.2f}]':>18}{t4[f'{tag}_roi'].mean()*100:>7.1f}%")
    print("  ※閾値は事前に3つだけ決め打ちしてある。ここを動かして良い値を探すと(38)の総当たりと同じ罠。")


if __name__ == "__main__":
    main()
