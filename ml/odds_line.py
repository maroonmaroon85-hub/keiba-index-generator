"""
(75-A) 「省くべきラインをオッズから引く」——買う前に分かる**市場含意確率**で買い目を切る。

**背景**: (73)で「小倉のROIは高配当依存では」を調べるのに *上位1%の払戻を落とす* という
順位での足切りを使ったが、これは筋が悪い:
  ・払戻はどの場でも右に長い裾を持つので、上位を切れば全場が同じくらい落ちるだけ
    （実測: 枠連で上位1%を落とすと10場すべて6.6〜9.5pt低下。差はσ1個分）
  ・的中率30%なので「上位5%除外」は的中の1/6を捨てる。何も読めない
  ・そもそも**当たった払戻は実際に受け取る金**なので、後から抜くとROIの過小評価になる

**正しい形**: 集計から抜くのではなく**買う前に弾く**。単勝オッズがあれば発走前に計算できる:
  1. 各馬の市場含意確率 = (1/単勝オッズ) をレース内で正規化（= `mkt_prob`）
  2. 枠ごとに合算 → 枠Aの確率 Pa、枠Bの確率 Pb
  3. その枠連の市場含意確率 = 2*Pa*Pb（ゾロ目は Pa^2）… `waku_score` と同じ積ベース
  4. **市場が見込む配当 ≈ (1−控除率) ÷ 確率**
→ 「この買い目は市場的に3,000円級」が発走前に分かる。

⚠**ユーザー方針（2026-08-03）: この線で買い目を弾くことはしない**。「モデルが推奨したなら買う」。
　したがって本スクリプトは**診断**として読むこと（(73)の小倉が高配当依存かの検証がもともとの動機）。
　「下位◯%を除外したときのROI」も参考値であって、運用に採るものではない。
　オッズの線を**モデルの作り方の側**に入れる検証は (75) `ml/model_line.py` で行う。

★(62)との違い: (62)は**モデルの**積ベーススコア下位20%除外（84.5→85.6%）。
　ここで測るのは**市場の**同じスコア。別物で、市場版は未測定だった。

判定は★判定基準どおり: 人気順で同じ線を引いた対照 / シード幅 / 十分位で単調性を見る。
(46)の通り**絞るほど検証が難しくなる**ので、線を1本引いて終わりにせず十分位の形で見ること。

実行: python3 ml/odds_line.py [シード数(既定3)]
"""
import itertools
import sys
import warnings

import numpy as np
import pandas as pd
import lightgbm as lgb

warnings.filterwarnings("ignore")
sys.path.insert(0, "ml")
import features as F
from market_baseline import load, wakuren_cs
from place_wide import PARAMS, boot
from pocket_eval import load_payout_a
from waku_umatan import load_wu, waku_of

PAYOUT = "data/payout/a.csv"
TAKEOUT = {"枠連": 0.225, "三連複": 0.25}


def implied(pairs, bp):
    """買い目の市場含意確率の合計。`waku_score` と同じ積ベース（ゾロ目は Pa^2）。"""
    s = 0.0
    for a, b in pairs:
        s += bp.get(a, 0.0) ** 2 if a == b else 2 * bp.get(a, 0.0) * bp.get(b, 0.0)
    return s


def build(sub, wu, pa):
    rows = []
    for rid, g in sub.groupby("raceid", sort=False):
        n = int(g["fieldsize"].iloc[0])
        w, s3 = wu.get(rid), pa.get(rid)
        gm = g.sort_values("p", ascending=False, kind="mergesort")
        gp = g.sort_values("odds", ascending=True, kind="mergesort")
        wk = {int(u): waku_of(int(u), n) for u in g["umaban"]}
        # 市場含意確率（発走前に分かる。1/オッズ をレース内で正規化して枠ごとに合算）
        inv = 1.0 / g["odds"].to_numpy(float)
        mp = inv / inv.sum()
        bp = {}
        for u, v in zip(g["umaban"].astype(int), mp):
            bp[wk[u]] = bp.get(wk[u], 0.0) + float(v)
        r = {"raceid": rid, "track": g["course"].iloc[0], "fieldsize": n,
             "year": g["year"].iloc[0]}
        if w and w["wakuren"] and len(g) >= 3:
            for k, gg in (("model", gm), ("pop", gp)):
                nums = gg["umaban"].astype(int).tolist()
                cs = wakuren_cs(nums, n)
                pay = sum(w["wakuren"].get(c, 0) for c in cs)
                r[f"waku_{k}"] = pay / (len(cs) * 100.0)
                r[f"waku_{k}_hit"] = float(pay > 0)
                r[f"waku_{k}_pay"] = float(pay)
                q = implied(cs, bp)
                r[f"waku_{k}_q"] = q                      # 買い目全体の市場含意的中率
                # 市場が見込む配当（100円あたり）。控除率ぶん割り引く
                r[f"waku_{k}_exp"] = (1 - TAKEOUT["枠連"]) / max(q, 1e-9) * 100
        if s3 and s3["sanrenpuku"] and len(g) >= 9:
            for k, gg in (("model", gm), ("pop", gp)):
                nums = gg["umaban"].astype(int).tolist()[:4]
                cs = [tuple(sorted(c)) for c in itertools.combinations(nums, 3)]
                pay = sum(s3["sanrenpuku"].get(c, 0) for c in cs)
                r[f"trio_{k}"] = pay / 400.0
                r[f"trio_{k}_hit"] = float(pay > 0)
                r[f"trio_{k}_pay"] = float(pay)
                mpd = dict(zip(g["umaban"].astype(int), mp))
                # 三連複BOX4の市場含意確率: 3頭の積の順列和（Harville近似）
                q = 0.0
                for c in cs:
                    for perm in itertools.permutations(c):
                        a, b, cc = (mpd[x] for x in perm)
                        d1 = 1 - a
                        d2 = 1 - a - b
                        if d1 > 1e-9 and d2 > 1e-9:
                            q += a * (b / d1) * (cc / d2)
                r[f"trio_{k}_q"] = q
                r[f"trio_{k}_exp"] = (1 - TAKEOUT["三連複"]) / max(q, 1e-9) * 100
        rows.append(r)
    return pd.DataFrame(rows)


def deciles(title, d, tag, rng):
    """市場含意確率の十分位ごとに ROI / 的中率 / 的中時配当 を出す。"""
    x = d[f"{tag}_model_q"].to_numpy(float)
    q = pd.qcut(x, 10, labels=False, duplicates="drop")
    print(f"\n=== {title}: 買い目の**市場含意的中率**の十分位（左ほど遠い買い目） ===")
    print(f"{'十分位':<8}{'R数':>8}{'市場含意的中率':>16}{'市場が見込む配当':>18}"
          f"{'モデルROI':>11}{'的中率':>8}{'的中時':>10}{'人気順ROI':>11}")
    for i in sorted(pd.unique(q[~pd.isna(q)])):
        m = q == i
        g = d[m]
        roi = g[f"{tag}_model"].to_numpy(float)
        hit = g[f"{tag}_model_hit"].to_numpy(float)
        pay = g[f"{tag}_model_pay"].to_numpy(float)
        print(f"{int(i)+1:<8}{len(g):>8,}{g[f'{tag}_model_q'].mean()*100:>15.1f}%"
              f"{g[f'{tag}_model_exp'].mean():>17,.0f}円{roi.mean()*100:>10.1f}%"
              f"{hit.mean()*100:>7.1f}%{(pay[pay>0].mean() if (pay>0).any() else 0):>9,.0f}円"
              f"{g[f'{tag}_pop'].mean()*100:>10.1f}%")

    print(f"\n  下位から切ったときのROI（★これが「省くべきライン」の答え）")
    print(f"{'切る割合':<12}{'残るR数':>10}{'モデルROI':>11}{'95%CI':>18}"
          f"{'人気順ROI':>11}{'差':>9}")
    base = d[f"{tag}_model"].to_numpy(float)
    lo0, hi0 = boot(base, rng, 2000)
    print(f"{'切らない':<12}{len(d):>10,}{base.mean()*100:>10.1f}%"
          f"{f'[{lo0:.1f},{hi0:.1f}]':>18}{d[f'{tag}_pop'].mean()*100:>10.1f}%"
          f"{(d[f'{tag}_model']-d[f'{tag}_pop']).mean()*100:>+8.2f}pt")
    for frac in (0.1, 0.2, 0.3, 0.5):
        th = np.quantile(x, frac)
        m = x >= th
        g = d[m]
        v = g[f"{tag}_model"].to_numpy(float)
        lo, hi = boot(v, rng, 2000)
        print(f"{f'下位{frac*100:.0f}%を除外':<12}{len(g):>10,}{v.mean()*100:>10.1f}%"
              f"{f'[{lo:.1f},{hi:.1f}]':>18}{g[f'{tag}_pop'].mean()*100:>10.1f}%"
              f"{(g[f'{tag}_model']-g[f'{tag}_pop']).mean()*100:>+8.2f}pt")


def by_track(d, tag, thresholds=(1000, 3000)):
    """場ごとに「総払戻のどれだけが“市場が高配当と見ていた買い目”から来ているか」。"""
    print(f"\n=== 場別: 払戻の出どころ（市場が見込む配当のどの帯から来たか） ===")
    print(f"{'場':<6}{'ROI':>8}{'市場含意的中率':>16}"
          + "".join(f"{f'{t:,}円超の寄与':>16}" for t in thresholds))
    exp = d[f"{tag}_model_exp"].to_numpy(float)
    pay = d[f"{tag}_model_pay"].to_numpy(float)
    roi = d[f"{tag}_model"].to_numpy(float)
    for t in d.groupby("track")[f"{tag}_model"].mean().sort_values(ascending=False).index:
        m = (d["track"] == t).to_numpy()
        tot = pay[m].sum()
        line = f"{t:<6}{roi[m].mean()*100:>7.1f}%{d.loc[m, f'{tag}_model_q'].mean()*100:>15.1f}%"
        for th in thresholds:
            sel = m & (exp > th)
            line += f"{pay[sel].sum()/max(tot,1)*100:>15.1f}%"
        print(line)
    print("  ※「◯円超の寄与」= 総払戻のうち、**買う前に市場が◯円超と値付けしていた**買い目から来た割合。")


def main():
    n_seed = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    d, fx, odds = load()
    y = (d["finish"] <= 3).astype(int).to_numpy()
    cut = d["date"].quantile(0.3)
    tr, te = (d["date"] < cut).to_numpy(), (d["date"] >= cut).to_numpy()
    print(f"train {tr.sum():,} / test {te.sum():,}（分割 {cut.date()}）  シード{n_seed}本")

    wu, pa = load_wu(PAYOUT), load_payout_a(PAYOUT)
    sub = d.loc[te, ["raceid", "umaban", "fieldsize", "course"]].copy()
    sub["odds"] = odds[te]
    sub["year"] = d.loc[te, "date"].dt.year.to_numpy()

    ps = []
    for s in range(n_seed):
        m = lgb.LGBMClassifier(random_state=s, **PARAMS).fit(fx[tr], y[tr],
                                                             categorical_feature=F.CAT_COLS)
        ps.append(m.predict_proba(fx[te])[:, 1])
        print(f"  seed {s} 完了")
    sub["p"] = np.mean(ps, axis=0)
    df = build(sub, wu, pa)

    rng = np.random.default_rng(0)
    w = df[df["waku_model"].notna()].copy()
    t3 = df[df["trio_model"].notna()].copy()
    deciles("枠連 軸枠×紐枠2", w, "waku", rng)
    by_track(w, "waku")
    deciles("三連複 BOX上位4", t3, "trio", rng)

    # ★判定基準1: 「下位20%除外」の効果がシードでどれだけ動くか
    print("\n★判定基準1: 市場含意確率 下位20%除外の効果をシード別に（枠連）")
    for s in range(n_seed):
        fr = sub.copy()
        fr["p"] = ps[s]
        g = build(fr, wu, pa)
        g = g[g["waku_model"].notna()]
        x = g["waku_model_q"].to_numpy(float)
        th = np.quantile(x, 0.2)
        a = g["waku_model"].mean() * 100
        b = g.loc[x >= th, "waku_model"].mean() * 100
        print(f"  seed {s}: 切らない {a:.2f}% → 下位20%除外 {b:.2f}%  効果 {b-a:+.2f}pt")


if __name__ == "__main__":
    main()
