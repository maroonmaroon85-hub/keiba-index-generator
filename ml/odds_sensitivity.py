"""
前日オッズで運用した場合の劣化を感度分析で見積もる。

**これは測定ではなく感度分析**。理由: 手元のデータ(DS col49)は**確定オッズのみ**で、
前日オッズを持っていないため直接比較ができない。前日オッズを TARGET から出力できれば本当の検証になる。

代わりに実運用の構造を再現する:
  ・確定オッズに撹乱を加えて「前日オッズらしきもの」を作る
  ・**その撹乱オッズで買い目を決める**（実運用では前日オッズしか見えない）
  ・**払戻は本物の確定オッズ/配当で計算する**（実際に受け取る額は確定値で決まる）
撹乱は対数正規（odds' = odds × exp(σz)）。σ=0.2 なら概ね±20%程度の相対変動。
(21)の実例（前日6.2倍→確定5.0倍 ≒ −19%）はσ≒0.2に相当する。

モデルは確定オッズで学習したものを使う（手元にそれしか無い）。
これは実運用の train/serve 不一致そのものなので、測定対象として妥当。

評価: 枠連 軸枠×紐枠2 / 三連複 BOX上位4 / 単勝EV≥1.0、および買い目がどれだけ変わるか。

実行: python3 ml/odds_sensitivity.py [シード数(既定3)]
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
from place_wide import boot, PARAMS
from pocket_eval import load_payout_a
from waku_umatan import load_wu, waku_of

SIGMAS = [0.0, 0.05, 0.10, 0.20, 0.30, 0.50]


def main():
    n_seed = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    d = F.to_model(F.load_files())
    f = F.build_features(d)
    keep = (f["n_prior"] >= 1) & d["odds"].notna() & (d["odds"] > 0)
    d, f = d[keep].reset_index(drop=True), f[keep].reset_index(drop=True)
    win = (d["finish"] == 1).astype(int).to_numpy()
    top3 = (d["finish"] <= 3).astype(int).to_numpy()
    odds = d["odds"].to_numpy(float)
    rid_all = d["raceid"].to_numpy()

    def feats(o):
        inv = 1.0 / o
        mk = inv / pd.Series(inv).groupby(rid_all).transform("sum").to_numpy()
        fx, _ = F.encode_categoricals(f)
        fx["log_odds"] = np.log(o)
        fx["mkt_prob"] = mk
        return fx

    fx_true = feats(odds)
    cut = d["date"].quantile(0.3)
    tr, te = (d["date"] < cut).to_numpy(), (d["date"] >= cut).to_numpy()
    pa = load_payout_a("data/payout/a.csv")
    wu = load_wu("data/payout/a.csv")
    print(f"train {tr.sum():,} / test {te.sum():,}  シード{n_seed}本")
    print("※学習は確定オッズ。予測時のみ撹乱オッズを与える（実運用の train/serve 不一致を再現）\n")

    # 学習は確定オッズで1回だけ（実運用でも過去データは確定値しかない）
    models = {}
    for tag, y in [("win", win), ("top3", top3)]:
        ms = []
        for s in range(n_seed):
            m = lgb.LGBMClassifier(random_state=s, **PARAMS)
            m.fit(fx_true[tr], y[tr], categorical_feature=F.CAT_COLS)
            ms.append(m)
        models[tag] = ms
        print(f"  {tag} 学習完了")

    sub0 = d.loc[te, ["raceid", "umaban"]].copy()
    sub0["year"] = d.loc[te, "date"].dt.year.to_numpy()
    sub0["fieldsize"] = f.loc[te, "fieldsize"].to_numpy()
    o_true = odds[te]
    w_te = win[te]
    rng = np.random.default_rng(0)
    base_top = None

    print(f"\n{'σ':>6}{'枠連 軸枠×紐枠2':>18}{'三連複 BOX上位4':>18}{'単勝EV≥1.0':>20}{'軸が変わった率':>14}")
    for sg in SIGMAS:
        o_obs = o_true if sg == 0 else o_true * np.exp(rng.normal(0, sg, len(o_true)))
        # test 行だけ撹乱した特徴を作る
        inv = 1.0 / o_obs
        mk = inv / pd.Series(inv).groupby(sub0["raceid"].to_numpy()).transform("sum").to_numpy()
        fx_te = fx_true.loc[te].copy()
        fx_te["log_odds"] = np.log(o_obs)
        fx_te["mkt_prob"] = mk

        p3 = np.mean([m.predict_proba(fx_te)[:, 1] for m in models["top3"]], axis=0)
        pw = np.mean([m.predict_proba(fx_te)[:, 1] for m in models["win"]], axis=0)
        s2 = sub0.copy()
        s2["p3"] = p3
        ev = (pd.Series(pw) / pd.Series(pw).groupby(s2["raceid"].to_numpy()).transform("sum")).to_numpy() * o_obs

        rows_w, rows_s, tops = [], [], {}
        for r_id, g in s2.groupby("raceid", sort=False):
            gg = g.sort_values("p3", ascending=False, kind="mergesort")
            nums = gg["umaban"].astype(int).tolist()
            tops[r_id] = nums[0]
            w = wu.get(r_id)
            if w and w["wakuren"] and len(nums) >= 3:
                fs = int(gg["fieldsize"].iloc[0])
                wa = waku_of(nums[0], fs)
                cs = sorted({tuple(sorted((wa, waku_of(h, fs)))) for h in nums[1:3]})
                if cs:
                    rows_w.append({"y": g["year"].iloc[0],
                                   "pay": sum(w["wakuren"].get(c, 0) for c in cs), "k": len(cs)})
            p = pa.get(r_id)
            if p and p["sanrenpuku"] and len(nums) >= 9:
                cs = [tuple(sorted(c)) for c in itertools.combinations(nums[:4], 3)]
                rows_s.append({"y": g["year"].iloc[0],
                               "pay": sum(p["sanrenpuku"].get(c, 0) for c in cs), "k": 4})
        if base_top is None:
            base_top = tops
            chg = 0.0
        else:
            chg = np.mean([tops[k] != base_top[k] for k in tops]) * 100

        out = []
        for rr in [rows_w, rows_s]:
            df = pd.DataFrame(rr)
            x = (df["pay"] / (df["k"] * 100)).to_numpy(float)
            lo, hi = boot(x, rng, 1500)
            out.append(f"{x.mean()*100:5.1f}% [{lo:.0f},{hi:.0f}]")
        m = ev >= 1.0
        ev_roi = (o_true[m] * w_te[m]).sum() / m.sum() * 100   # 払戻は確定オッズ
        out.append(f"{ev_roi:5.1f}% (n={m.sum():,})")
        print(f"{sg:>6.2f}{out[0]:>18}{out[1]:>18}{out[2]:>20}{chg:>13.1f}%")

    print("\n※σ=0.2 が (21) の実例（前日6.2倍→確定5.0倍≒−19%）に相当する目安。")
    print("　本当の検証には TARGET から前日オッズを出力する必要がある。")


if __name__ == "__main__":
    main()
