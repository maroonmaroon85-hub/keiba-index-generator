"""B1/B2: 「定式化」そのものを変える2案（どちらも本プロジェクトで未実施）。

★事前宣言（(77)の3点セットを守る: (a)事前宣言 (b)人気順の対照 (c)対応ありでの直接比較）

**B1 レース内順位学習（LambdaRank）** — 一度も試されていない。
現行はすべて **binary classifier**（top3か否か）で、レース内の相対関係は
「同じレースの馬に同じ特徴を与える」ことでしか表現されていない（(57)のレース内相対特徴も効かなかった）。
だが買い方が要求するのは確率ではなく**レース内の順序**（軸＝1位、紐＝2〜4位）。
`LGBMRanker(objective="lambdarank", group=レース)` は**レース内の順序を直接最適化**する定式化で、
binaryとは損失が違う。(30)は「目的関数をEV回帰に」、(75)は「市場との差分」を試したが、
**レース内順位学習は候補に挙がったことがない**。

**B2 目標 top2** — オッズ入り構成では未実施。
(30)Aで win/top2/top3 を比べたが、あれは**オッズを特徴に入れる前**のモデル。
(45)で構成が変わったあと、top3以外は一度も測られていない。
本命の枠連は**1着と2着**で決まるので、top2 の方がラベルとして買い方に近い。

判定: 現行(A)を置き換えるには、**対応ありの差のCIが0を跨がずプラス**である必要がある。
      横断表で良く見えるだけでは(77)②の紐3と同じ罠にはまる。

実行: python3 ml/audit_model_alt.py [シード数(既定3)]
"""
import itertools
import sys
import warnings

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
sys.path.insert(0, "ml")
import features as F
from _cache import load_cached
from place_wide import PARAMS
from pocket_eval import load_payout_a
from waku_umatan import load_wu, waku_of

PAYOUT = "data/payout/a.csv"


def wakuren_cs(nums, n):
    wa = waku_of(nums[0], n)
    return sorted({tuple(sorted((wa, waku_of(h, n)))) for h in nums[1:3]})


def boot(x, rng, n=2000):
    b = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return np.percentile(b, 2.5) * 100, np.percentile(b, 97.5) * 100


def per_race(sub, wu, pa, col):
    """レースIDごとの回収額（1点=100円あたり）を返す。col で順位付ける。"""
    w_out, s_out = {}, {}
    for rid, g in sub.groupby("raceid", sort=False):
        n = int(g["fieldsize"].iloc[0])
        nums = g.sort_values(col, ascending=(col == "odds"), kind="mergesort")["umaban"].astype(int).tolist()
        w, s3 = wu.get(rid), pa.get(rid)
        if w and w["wakuren"] and len(g) >= 3:
            cs = wakuren_cs(nums, n)
            w_out[rid] = sum(w["wakuren"].get(c, 0) for c in cs) / (len(cs) * 100)
        if s3 and s3["sanrenpuku"] and len(g) >= 9:
            cs = [tuple(sorted(c)) for c in itertools.combinations(nums[:4], 3)]
            s_out[rid] = sum(s3["sanrenpuku"].get(c, 0) for c in cs) / 400
    return w_out, s_out


def main():
    n_seed = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    d, fx = load_cached()
    cut = d["date"].quantile(0.3)
    tr, te = (d["date"] < cut).to_numpy(), (d["date"] >= cut).to_numpy()
    fin = d["finish"].to_numpy()
    print(f"train {tr.sum():,} / test {te.sum():,}（分割 {cut.date()}）  シード{n_seed}本")

    # ranker はレースが連続していないと group を渡せないので、学習行を raceid 順に並べ替える
    tr_idx = np.flatnonzero(tr)
    ordr = tr_idx[np.argsort(d["raceid"].to_numpy()[tr_idx], kind="mergesort")]
    grp = pd.Series(d["raceid"].to_numpy()[ordr]).groupby(
        pd.Series(d["raceid"].to_numpy()[ordr]), sort=False).size().to_numpy()
    fx_r, fin_r = fx.iloc[ordr], fin[ordr]

    VARIANTS = {
        "A 現行 binary top3": ("bin", (fin <= 3).astype(int)),
        "B2 binary top2": ("bin", (fin <= 2).astype(int)),
        "   binary win（参考）": ("bin", (fin == 1).astype(int)),
        "B1 LambdaRank 段階ラベル": ("rank", np.clip(4 - fin_r, 0, 3).astype(int)),
        "B1 LambdaRank top3ラベル": ("rank", (fin_r <= 3).astype(int)),
    }

    wu, pa = load_wu(PAYOUT), load_payout_a(PAYOUT)
    sub = d.loc[te, ["raceid", "umaban", "fieldsize", "odds"]].copy()
    y_te3 = (fin[te] <= 3).astype(int)

    res, seedspread = {}, {}
    for label, (kind, target) in VARIANTS.items():
        preds = []
        for s in range(n_seed):
            if kind == "bin":
                m = lgb.LGBMClassifier(random_state=s, **PARAMS).fit(
                    fx[tr], target[tr], categorical_feature=F.CAT_COLS)
                preds.append(m.predict_proba(fx[te])[:, 1])
            else:
                m = lgb.LGBMRanker(random_state=s, objective="lambdarank", **PARAMS).fit(
                    fx_r, target, group=grp, categorical_feature=F.CAT_COLS)
                preds.append(m.predict(fx[te]))
        sub["p"] = np.mean(preds, axis=0)
        res[label] = per_race(sub, wu, pa, "p")
        # シード幅（★判定基準1）
        sp = []
        for pv in preds:
            sub["p1"] = pv
            w1, s1 = per_race(sub, wu, pa, "p1")
            sp.append((np.mean(list(w1.values())) * 100, np.mean(list(s1.values())) * 100))
        sp = np.array(sp)
        seedspread[label] = sp.max(axis=0) - sp.min(axis=0)
        auc = roc_auc_score(y_te3, np.mean(preds, axis=0))
        print(f"  学習完了 {label}  AUC(top3) {auc:.4f}")

    sub["p"] = sub["odds"]
    pop_w, pop_s = per_race(sub, wu, pa, "odds")
    rng = np.random.default_rng(0)

    for tag, ix in [("枠連 軸枠×紐枠2", 0), ("三連複 BOX上位4", 1)]:
        pop = [pop_w, pop_s][ix]
        print(f"\n{'='*92}\n=== {tag} ===")
        print(f"{'構成':<26}{'R':>7}{'ROI':>8}{'人気順':>8}{'差':>9}{'差の95%CI':>17}{'シード幅':>9}")
        for label in VARIANTS:
            dd = res[label][ix]
            ks = sorted(dd.keys() & pop.keys())
            m = np.array([dd[k] for k in ks])
            q = np.array([pop[k] for k in ks])
            lo, hi = boot(m - q, rng)
            print(f"{label:<26}{len(ks):>7,}{m.mean()*100:>7.2f}%{q.mean()*100:>7.2f}%"
                  f"{(m-q).mean()*100:>+8.2f}pt{f'[{lo:+.2f},{hi:+.2f}]':>17}"
                  f"{seedspread[label][ix]:>8.2f}pt")

        print(f"\n  ★対応ありで現行と直接比較（(77)②の作法。同じレースで両方買った差）")
        base = res["A 現行 binary top3"][ix]
        for label in VARIANTS:
            if label.startswith("A "):
                continue
            dd = res[label][ix]
            ks = sorted(dd.keys() & base.keys())
            diff = np.array([dd[k] - base[k] for k in ks])
            lo, hi = boot(diff, rng)
            verdict = "★現行より上" if lo > 0 else ("★現行より下" if hi < 0 else "変えない（差は誤差の内）")
            print(f"  {label:<26}{len(ks):>7,}R  差 {diff.mean()*100:>+6.2f}pt"
                  f"  CI[{lo:+.2f},{hi:+.2f}]   {verdict}")


if __name__ == "__main__":
    main()
