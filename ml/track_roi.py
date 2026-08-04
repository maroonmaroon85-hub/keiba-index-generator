"""
(73) 競馬場ごとのROI。「場によってモデルの当たりやすさが違うのでは」という仮説の検証。

⚠**先に(39)と★判定基準を読むこと**。(33)(36)で同種の層別を47区分やって生存ゼロ、しかも
「阪神が良い」と一度判定したものが(39)で**誤差の当て方の誤り**と分かって撤回されている。
同じ轍を踏まないよう、ここでは判定に**3つ**当てる:

  1. ★判定基準2（標本誤差）… 場ごとにブートストラップCI
  2. ★判定基準1（シードノイズ）… 場ごとのROIをシード別に出して幅を見る
  3. ★判定基準4（帰無分布との比較）… **場のラベルを頭数層内でシャッフル**して、
     「場は無関係」という帰無仮説のもとで**10場のばらつき（最大−最小）がどこまで出るか**を作り、
     観測値をその分布と比べる。層別の判定でいちばん抜けやすいのがこれ（(38)）。

さらに**人気順で同じ買い方をした対照を必ず並べる**（(69)の教訓）。場ごとのROIの違いは
頭数構成の違い（＝配当水準の違い）でも出るので、**モデルの寄与を見るなら差(モデル−人気順)を見る**。

実行: python3 ml/track_roi.py [シード数(既定3)] [シャッフル回数(既定2000)]
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
from waku_umatan import load_wu

PAYOUT = "data/payout/a.csv"


def build(sub, wu, pa):
    """レース1行の表。券種ごとに モデル順/人気順 の回収倍率を持たせる。"""
    rows = []
    for rid, g in sub.groupby("raceid", sort=False):
        n = int(g["fieldsize"].iloc[0])
        nums = {"model": g.sort_values("p", ascending=False, kind="mergesort")
                          ["umaban"].astype(int).tolist(),
                "pop": g.sort_values("odds", ascending=True, kind="mergesort")
                        ["umaban"].astype(int).tolist()}
        w, s3 = wu.get(rid), pa.get(rid)
        r = {"raceid": rid, "track": g["course"].iloc[0], "fieldsize": n,
             "year": g["year"].iloc[0]}
        if w and w["wakuren"] and len(g) >= 3:
            for k, nm in nums.items():
                cs = wakuren_cs(nm, n)
                pay = sum(w["wakuren"].get(c, 0) for c in cs)
                r[f"waku_{k}"] = pay / (len(cs) * 100.0)
                r[f"waku_{k}_hit"] = float(pay > 0)
        if s3 and s3["sanrenpuku"] and len(g) >= 9:
            for k, nm in nums.items():
                cs = [tuple(sorted(c)) for c in itertools.combinations(nm[:4], 3)]
                pay = sum(s3["sanrenpuku"].get(c, 0) for c in cs)
                r[f"trio_{k}"] = pay / 400.0
                r[f"trio_{k}_hit"] = float(pay > 0)
        rows.append(r)
    return pd.DataFrame(rows)


def shuffle_null(df, col, rng, n_shuf, strat="fs_bin"):
    """場ラベルを**層内でシャッフル**したときの「10場のROIの最大−最小」の分布。

    層＝頭数ビン。場によって頭数構成が違い、それだけでもROIは動くので、
    層を跨いでシャッフルすると帰無分布が狭くなりすぎて偽陽性を招く。
    """
    v = df[col].to_numpy(float)
    codes, uniq = pd.factorize(df["track"])
    bins = df[strat].to_numpy()
    idx_by_bin = [np.flatnonzero(bins == b) for b in np.unique(bins)]
    ranges = np.empty(n_shuf)
    for i in range(n_shuf):
        perm = codes.copy()
        for ix in idx_by_bin:
            perm[ix] = rng.permutation(perm[ix])
        s = np.bincount(perm, weights=v, minlength=len(uniq))
        c = np.bincount(perm, minlength=len(uniq))
        means = s / np.maximum(c, 1)
        ranges[i] = means.max() - means.min()
    return ranges * 100


def section(title, df, mcol, pcol, rng, n_shuf, per_seed_tables):
    d = df[df[mcol].notna()].copy()
    d["fs_bin"] = pd.cut(d["fieldsize"], [0, 9, 12, 14, 16, 99], labels=False)
    print("\n" + "=" * 96)
    print(f"【{title}】 {len(d):,}R")
    print("=" * 96)
    print(f"{'場':<6}{'R数':>8}{'モデルROI':>11}{'95%CI':>18}{'人気順ROI':>11}{'差':>9}"
          f"{'的中率':>8}{'シード幅':>10}")
    order = d.groupby("track")[mcol].mean().sort_values(ascending=False).index
    for t in order:
        g = d[d["track"] == t]
        x, y = g[mcol].to_numpy(float), g[pcol].to_numpy(float)
        lo, hi = boot(x, rng, 2000)
        seeds = [tb[(tb["track"] == t) & tb[mcol].notna()][mcol].mean() * 100
                 for tb in per_seed_tables]
        print(f"{t:<6}{len(g):>8,}{x.mean()*100:>10.1f}%{f'[{lo:.1f},{hi:.1f}]':>18}"
              f"{y.mean()*100:>10.1f}%{(x-y).mean()*100:>+8.2f}pt"
              f"{g[mcol+'_hit'].mean()*100:>7.1f}%{max(seeds)-min(seeds):>9.2f}pt")
    tm = d.groupby("track")[mcol].mean() * 100
    td = d.groupby("track").apply(lambda g: (g[mcol] - g[pcol]).mean() * 100)
    print(f"{'全体':<6}{len(d):>8,}{d[mcol].mean()*100:>10.1f}%{'':>18}"
          f"{d[pcol].mean()*100:>10.1f}%{(d[mcol]-d[pcol]).mean()*100:>+8.2f}pt")

    # ★裾チェック: 平均ROIは払戻の裾が長いほど不安定になる。少数の高配当で持ち上がっているだけなら、
    #   「その場は良い」という主張は次の10年で再現しない。上位の払戻を落として何が残るかを見る。
    print(f"\n★裾チェック: 高配当を落としたときのROI（平均は少数の大当たりで持ち上がる）")
    print(f"{'場':<6}{'ROI':>8}{'上位1%除く':>13}{'上位5%除く':>13}"
          f"{'最大の1本の寄与':>17}{'的中時の倍率':>14}")
    for t in order:
        g = d[d["track"] == t]
        x = np.sort(g[mcol].to_numpy(float))[::-1]
        k1, k5 = max(1, int(len(x) * 0.01)), max(1, int(len(x) * 0.05))
        hit = x[x > 0]
        print(f"{t:<6}{x.mean()*100:>7.1f}%{x[k1:].mean()*100:>12.1f}%"
              f"{x[k5:].mean()*100:>12.1f}%{x[0]/x.sum()*100:>16.1f}%"
              f"{hit.mean():>13.2f}倍")

    # ★(33)(36)を(39)が撤回したときの教訓: 層別の「勝ち組」は期間を割ると入れ替わる。
    #   帰無分布を通っても、前半/後半で符号が揃わなければ運用の根拠にはならない。
    mid = int(np.median(d["year"]))
    print(f"\n★安定性: 検証期間を前半(〜{mid})/後半({mid+1}〜)に割った 差(モデル−人気順)")
    print(f"{'場':<6}{'前半R数':>9}{'前半の差':>11}{'後半R数':>9}{'後半の差':>11}"
          f"{'符号一致':>10}{'年別プラス':>12}")
    for t in order:
        g = d[d["track"] == t]
        a, b = g[g["year"] <= mid], g[g["year"] > mid]
        if len(a) < 100 or len(b) < 100:
            continue
        da = (a[mcol] - a[pcol]).mean() * 100
        db = (b[mcol] - b[pcol]).mean() * 100
        ys = g.groupby("year").apply(lambda x: (x[mcol] - x[pcol]).mean())
        print(f"{t:<6}{len(a):>9,}{da:>+10.2f}pt{len(b):>9,}{db:>+10.2f}pt"
              f"{('○' if da*db > 0 else '×'):>10}{f'{int((ys>0).sum())}/{len(ys)}':>12}")

    print(f"\n★判定基準4: 場ラベルを頭数層内で{n_shuf:,}回シャッフルした帰無分布との比較")
    for name, obs, col in [("モデルROI", tm.max() - tm.min(), mcol)]:
        null = shuffle_null(d, col, rng, n_shuf)
        p = float((null >= obs).mean())
        print(f"  {name}の最大−最小: 観測 {obs:.2f}pt / 偶然でも {np.percentile(null,50):.2f}pt "
              f"(中央値)・{np.percentile(null,95):.2f}pt (95%点) 出る → p={p:.3f}"
              + ("  ⇒ **偶然と区別できない**" if p >= 0.05 else "  ⇒ 偶然では説明しにくい"))
    d["_diff"] = d[mcol] - d[pcol]
    obs = td.max() - td.min()
    null = shuffle_null(d, "_diff", rng, n_shuf)
    p = float((null >= obs).mean())
    print(f"  差(モデル−人気順)の最大−最小: 観測 {obs:.2f}pt / 偶然でも "
          f"{np.percentile(null,50):.2f}pt・{np.percentile(null,95):.2f}pt → p={p:.3f}"
          + ("  ⇒ **偶然と区別できない**" if p >= 0.05 else "  ⇒ 偶然では説明しにくい"))


def main():
    n_seed = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    n_shuf = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
    d, fx, odds = load()
    y = (d["finish"] <= 3).astype(int).to_numpy()
    cut = d["date"].quantile(0.3)
    tr, te = (d["date"] < cut).to_numpy(), (d["date"] >= cut).to_numpy()
    print(f"train {tr.sum():,} / test {te.sum():,}（分割 {cut.date()}）"
          f"  シード{n_seed}本 / シャッフル{n_shuf:,}回")

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

    per_seed_tables = []
    for s in range(n_seed):
        fr = sub.copy()
        fr["p"] = ps[s]
        per_seed_tables.append(build(fr, wu, pa))
    sub["p"] = np.mean(ps, axis=0)
    df = build(sub, wu, pa)

    rng = np.random.default_rng(0)
    section("枠連 軸枠×紐枠2（本命推奨）", df, "waku_model", "waku_pop", rng, n_shuf,
            per_seed_tables)
    section("三連複 BOX上位4（対抗）", df, "trio_model", "trio_pop", rng, n_shuf,
            per_seed_tables)

    print("\n※(46)の通り、場で絞ると1区分あたり約2,900Rしか無く、"
          "**運用に使えるかを検証するには何十年もかかる**。ここは記述統計として読むこと。")


if __name__ == "__main__":
    main()
