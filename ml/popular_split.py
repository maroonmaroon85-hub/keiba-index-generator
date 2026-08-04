"""
(72) 「人気順と同じ買い目になったレース」と「違ったレース」に分けて、券種別に絶対値のROIを出す。

**なぜ必要か**: モデル順は人気順とほぼ同じ並びになる（軸＝1番人気が(49)で79.2%、8/01の実測で81.2%）。
だとすると**モデルの正味の寄与は「並びが違った分」に集中しているはず**で、そこを直接見たい。
(69)/`ml/market_baseline.py` は枠連で「一致46.7% / 不一致だけなら+5.37pt」まで出したが、
**差分しか無く、一致・不一致それぞれの絶対値**が無かった。ここを埋める。

出すもの（券種ごと・一致/不一致の2群）:
  ・レース数と構成比
  ・**モデル順ROIと人気順ROIの絶対値**（一致群では定義上同じ値になる＝検算になる）
  ・差と95%CI（対応のあるブートストラップ）
  ・**的中率と的中時払戻も分ける**（ROIだけだと標本誤差で潰れる。(66)④の教訓）

⚠読むときの注意:
  ・一致群は買い目が同一なので**差はきっかり0**。全体の差は「不一致率 × 不一致群の差」に分解される。
  ・群を2つに割ると各群は約半分。★判定基準2(標本誤差)と5(絞ると測れなくなる)に該当する。
    **「不一致のレースだけ買う」という運用はできない**（買う前に人気順との一致は分かるが、
    (46)の通り絞るほど検証に必要な年数が延び、ここでの差もCIで見ると確定しない）。
  ・★判定基準1(シードノイズ)として、不一致群の差をシード別にも出す。

実行: python3 ml/popular_split.py [シード数(既定3)]
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
from place_wide import PARAMS, boot, load_place_wide
from pocket_eval import load_payout_a
from waku_umatan import load_wu

PAYOUT = "data/payout/a.csv"


def race_rows(sub, wu, pa, pw):
    """レースごとに モデル順/人気順 の買い目を作り、券種別の払戻と一致フラグを返す。"""
    out = {"枠連": [], "三連複": [], "ワイド": []}
    for rid, g in sub.groupby("raceid", sort=False):
        n = int(g["fieldsize"].iloc[0])
        yr = g["year"].iloc[0]
        nums = {"model": g.sort_values("p", ascending=False, kind="mergesort")
                          ["umaban"].astype(int).tolist(),
                "pop": g.sort_values("odds", ascending=True, kind="mergesort")
                        ["umaban"].astype(int).tolist()}
        w, s3, wd = wu.get(rid), pa.get(rid), pw.get(rid)

        if w and w["wakuren"] and len(g) >= 3:
            row, buys = {"year": yr}, {}
            for k, nm in nums.items():
                cs = wakuren_cs(nm, n)
                buys[k] = set(cs)
                row[f"{k}_pay"] = sum(w["wakuren"].get(c, 0) for c in cs)
                row[f"{k}_k"] = len(cs)
            row["same"] = buys["model"] == buys["pop"]
            out["枠連"].append(row)

        if s3 and s3["sanrenpuku"] and len(g) >= 9:
            row, buys = {"year": yr}, {}
            for k, nm in nums.items():
                buys[k] = set(nm[:4])
                cs = [tuple(sorted(c)) for c in itertools.combinations(nm[:4], 3)]
                row[f"{k}_pay"] = sum(s3["sanrenpuku"].get(c, 0) for c in cs)
                row[f"{k}_k"] = 4
            row["same"] = buys["model"] == buys["pop"]      # BOXなので集合が同じなら同一
            out["三連複"].append(row)

        if wd and wd["wide"] and len(g) >= 4:
            row, buys = {"year": yr}, {}
            for k, nm in nums.items():
                buys[k] = set(nm[:3])
                cs = [tuple(sorted(c)) for c in itertools.combinations(nm[:3], 2)]
                row[f"{k}_pay"] = sum(wd["wide"].get(c, 0) for c in cs)
                row[f"{k}_k"] = 3
            row["same"] = buys["model"] == buys["pop"]
            out["ワイド"].append(row)
    return {k: pd.DataFrame(v) for k, v in out.items()}


def stats(df, who):
    """ROI・的中率・的中時払戻。ROIはレース単位の回収倍率の平均（点数で割ってある）。"""
    pay = df[f"{who}_pay"].to_numpy(float)
    k = df[f"{who}_k"].to_numpy(float)
    roi = pay / (k * 100.0)
    hit = pay > 0
    return roi, hit.mean() * 100, (pay[hit].mean() if hit.any() else 0.0)


def boot_indep(a, b, rng, n=2000, chunk=200):
    """**対応のない**2群の平均差のCI。同じレースを比べる `boot` とは別物なので混ぜないこと（(39)）。"""
    outs = []
    for start in range(0, n, chunk):
        k = min(chunk, n - start)
        ia = rng.integers(0, len(a), size=(k, len(a)))
        ib = rng.integers(0, len(b), size=(k, len(b)))
        outs.append(a[ia].mean(axis=1) - b[ib].mean(axis=1))
    x = np.concatenate(outs) * 100
    return np.percentile(x, 2.5), np.percentile(x, 97.5)


def report(title, df, ceiling, rng):
    print("\n" + "=" * 86)
    print(f"【{title}】 {len(df):,}R   控除率の天井(pool平均) {ceiling:.1f}%")
    print("=" * 86)
    print(f"{'群':<22}{'R数':>8}{'構成比':>8}{'モデルROI':>11}{'人気順ROI':>11}"
          f"{'差':>9}{'差の95%CI':>18}")
    buckets = [("全体", df),
               ("買い目が一致", df[df["same"]]),
               ("買い目が不一致", df[~df["same"]])]
    for name, g in buckets:
        if len(g) < 100:
            print(f"{name:<22}{len(g):>8,}  標本不足")
            continue
        rm, _, _ = stats(g, "model")
        rp, _, _ = stats(g, "pop")
        diff = rm - rp
        lo, hi = boot(diff, rng, 2000)
        print(f"{name:<22}{len(g):>8,}{len(g)/len(df)*100:>7.1f}%{rm.mean()*100:>10.1f}%"
              f"{rp.mean()*100:>10.1f}%{diff.mean()*100:>+8.2f}pt"
              f"{f'[{lo:+.2f},{hi:+.2f}]':>18}")

    print(f"\n{'群':<22}{'モデル的中率':>13}{'人気順的中率':>13}"
          f"{'モデル的中時':>14}{'人気順的中時':>14}")
    for name, g in buckets:
        if len(g) < 100:
            continue
        _, hm, am = stats(g, "model")
        _, hp, ap = stats(g, "pop")
        print(f"{name:<22}{hm:>12.2f}%{hp:>12.2f}%{am:>13,.0f}円{ap:>13,.0f}円")

    # 全体の差を「不一致率 × 不一致群の差」に分解して検算する
    mis = df[~df["same"]]
    if len(mis) >= 100:
        dm = (stats(mis, "model")[0] - stats(mis, "pop")[0]).mean() * 100
        share = len(mis) / len(df)
        all_d = (stats(df, "model")[0] - stats(df, "pop")[0]).mean() * 100
        print(f"\n  分解: 不一致率{share*100:.1f}% × 不一致群の差{dm:+.2f}pt = {share*dm:+.2f}pt"
              f"（全体の差 {all_d:+.2f}pt と一致すれば正しい）")
        ys = pd.Series((stats(mis, "model")[0] - stats(mis, "pop")[0]) * 100).groupby(
            mis["year"].to_numpy()).mean()
        print("  不一致群の年別の差: " + " ".join(f"{y}:{v:+.1f}" for y, v in ys.items())
              + f"  → プラスの年 {int((ys > 0).sum())}/{len(ys)}")

        # ★運用上の問い「不一致のレースだけ買えばいいのか」への答え。
        #   モデルの"寄与"が大きい群と、モデルで買ったときの"回収率"が高い群は別物。
        #   ここは同じレースを比べていないので**対応のない**CIを使う（(39)の取り違えを再発させない）。
        sm = stats(df[df["same"]], "model")[0]
        lo2, hi2 = boot_indep(stats(mis, "model")[0], sm, rng)
        dd = (stats(mis, "model")[0].mean() - sm.mean()) * 100
        print(f"  モデルで買ったときの絶対ROI: 不一致群{stats(mis,'model')[0].mean()*100:.1f}% "
              f"vs 一致群{sm.mean()*100:.1f}%  差{dd:+.2f}pt [{lo2:+.2f},{hi2:+.2f}]（対応なし）")


def main():
    n_seed = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    d, fx, odds = load()
    y = (d["finish"] <= 3).astype(int).to_numpy()
    cut = d["date"].quantile(0.3)
    tr, te = (d["date"] < cut).to_numpy(), (d["date"] >= cut).to_numpy()
    print(f"train {tr.sum():,} / test {te.sum():,}（分割 {cut.date()}）  シード{n_seed}本")

    wu, pa, pw = load_wu(PAYOUT), load_payout_a(PAYOUT), load_place_wide(PAYOUT)
    sub = d.loc[te, ["raceid", "umaban", "fieldsize"]].copy()
    sub["odds"] = odds[te]
    sub["year"] = d.loc[te, "date"].dt.year.to_numpy()

    per_seed_p = []
    for s in range(n_seed):
        m = lgb.LGBMClassifier(random_state=s, **PARAMS).fit(fx[tr], y[tr],
                                                             categorical_feature=F.CAT_COLS)
        per_seed_p.append(m.predict_proba(fx[te])[:, 1])
        print(f"  seed {s} 完了")

    rng = np.random.default_rng(0)
    sub["p"] = np.mean(per_seed_p, axis=0)
    tables = race_rows(sub, wu, pa, pw)
    report("枠連 軸枠×紐枠2（本命推奨）", tables["枠連"], 77.5, rng)
    report("三連複 BOX上位4（対抗）", tables["三連複"], 75.0, rng)
    report("ワイド BOX上位3", tables["ワイド"], 77.5, rng)

    # ★判定基準1（シードノイズ）: 不一致群の差はシードでどれだけ動くか
    print("\n" + "=" * 86)
    print("★判定基準1: 不一致群の差をシード別に（学習の乱数だけでどれだけ動くか）")
    print("=" * 86)
    print(f"{'seed':<8}{'枠連 不一致率':>15}{'枠連 差':>12}"
          f"{'三連複 不一致率':>17}{'三連複 差':>12}")
    spread = {"枠連": [], "三連複": []}
    # ★「不一致群の方が絶対ROIが高いのでは」への直接の答え。
    #   モデルで買ったときの 不一致群ROI − 一致群ROI が、学習の乱数だけでどれだけ動くか。
    absdiff = {"枠連": [], "三連複": []}
    seed_tables = []
    for s in range(n_seed):
        fr = sub.copy()
        fr["p"] = per_seed_p[s]
        t = race_rows(fr, wu, pa, pw)
        seed_tables.append(t)
        line = f"{s:<8}"
        for kind, w in (("枠連", 15), ("三連複", 17)):
            df = t[kind]
            mis = df[~df["same"]]
            dm = (stats(mis, "model")[0] - stats(mis, "pop")[0]).mean() * 100
            spread[kind].append(dm)
            absdiff[kind].append((stats(mis, "model")[0].mean()
                                  - stats(df[df["same"]], "model")[0].mean()) * 100)
            line += f"{(~df['same']).mean()*100:>{w-1}.1f}%{dm:>+11.2f}pt"
        print(line)
    for kind, v in spread.items():
        if len(v) > 1:
            print(f"  {kind}: シード幅 {max(v)-min(v):.2f}pt（平均 {np.mean(v):+.2f}pt）")

    print("\n★『不一致群の方が絶対ROIが高いのでは』の検証: 不一致群ROI − 一致群ROI をシード別に")
    for kind, v in absdiff.items():
        s_ = " / ".join(f"seed{i}:{x:+.2f}pt" for i, x in enumerate(v))
        note = ""
        if len(v) > 1:
            note = (f"  → シード幅 {max(v)-min(v):.2f}pt・平均 {np.mean(v):+.2f}pt"
                    + ("（**符号がシードで変わる＝乱数の産物**）"
                       if min(v) * max(v) < 0 else "（符号は安定）"))
        print(f"  {kind}: {s_}{note}")


if __name__ == "__main__":
    main()
