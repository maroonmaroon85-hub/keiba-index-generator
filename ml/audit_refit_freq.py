"""C3: ウォークフォワードの**再学習の頻度**が結果を変えるか。

(78)のウォークフォワードは**1年に1回**の再学習で測った。つまり12月のレースを
「前年12月までのデータで学習したモデル」＝最大12ヶ月古いモデルで予測している。
**実運用(`train_prod.py`)はデータが増えるたびに走らせる**ので、実際は数日〜数週の古さしかない。
＝**(78)の85.1%/82.2%は実運用よりやや悲観的な可能性がある**。

(35)は「新しいデータで学習し直しても回収率は改善しない」と結論しているが、
あれは旧構成(オッズなし・三連単)で、学習の打ち切り年を2つ比べただけだった。
ここでは**同じ構成のまま再学習の間隔だけを変えて**、(78)の数字が間隔に対して頑健かを見る。

判定: 間隔で数字が動かないなら(78)の値をそのまま運用の期待値としてよい。
      3ヶ月刻みの方が明確に良ければ、(78)は下振れなので上方修正する。

実行: python3 ml/audit_refit_freq.py [シード数(既定2)] [開始年(既定2020)]
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


def main():
    n_seed = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    y0 = int(sys.argv[2]) if len(sys.argv) > 2 else 2020
    d, fx = load_cached()
    y = (d["finish"] <= 3).astype(int).to_numpy()
    dt = d["date"].to_numpy()
    wu, pa = load_wu(PAYOUT), load_payout_a(PAYOUT)
    end = pd.Timestamp(d["date"].max())
    print(f"再学習の間隔の比較（評価 {y0}-01-01〜{end.date()}・シード{n_seed}本）")

    res = {}
    for months in (12, 3):
        bounds = pd.date_range(f"{y0}-01-01", end + pd.offsets.MonthBegin(1), freq=f"{months}MS")
        rows_w, rows_s, pw, ps = [], [], [], []
        for i in range(len(bounds) - 1):
            lo, hi = bounds[i], bounds[i + 1]
            tr = dt < np.datetime64(lo)
            te = (dt >= np.datetime64(lo)) & (dt < np.datetime64(hi))
            if te.sum() < 500:
                continue
            ps_ = [lgb.LGBMClassifier(random_state=s, **PARAMS)
                   .fit(fx[tr], y[tr], categorical_feature=F.CAT_COLS)
                   .predict_proba(fx[te])[:, 1] for s in range(n_seed)]
            sub = d.loc[te, ["raceid", "umaban", "fieldsize", "odds"]].copy()
            sub["p"] = np.mean(ps_, axis=0)
            for rid, g in sub.groupby("raceid", sort=False):
                w, s3 = wu.get(rid), pa.get(rid)
                n = int(g["fieldsize"].iloc[0])
                uma = g["umaban"].astype(int).to_numpy()
                mo = uma[np.argsort(-g["p"].to_numpy(float), kind="mergesort")]
                po = uma[np.argsort(g["odds"].to_numpy(float), kind="mergesort")]
                if w and w["wakuren"] and len(g) >= 3:
                    c1, c2 = wakuren_cs(mo, n), wakuren_cs(po, n)
                    rows_w.append(sum(w["wakuren"].get(c, 0) for c in c1) / (len(c1) * 100))
                    pw.append(sum(w["wakuren"].get(c, 0) for c in c2) / (len(c2) * 100))
                if s3 and s3["sanrenpuku"] and len(g) >= 9:
                    c1 = [tuple(sorted(c)) for c in itertools.combinations(mo[:4], 3)]
                    c2 = [tuple(sorted(c)) for c in itertools.combinations(po[:4], 3)]
                    rows_s.append(sum(s3["sanrenpuku"].get(c, 0) for c in c1) / 400)
                    ps.append(sum(s3["sanrenpuku"].get(c, 0) for c in c2) / 400)
        res[months] = tuple(np.array(v) for v in (rows_w, pw, rows_s, ps))
        print(f"  {months}ヶ月ごと再学習: 学習回数 {len(bounds)-1}回 / 枠連 {len(rows_w):,}R 完了")

    rng = np.random.default_rng(0)
    print(f"\n{'='*88}")
    print(f"{'再学習の間隔':<18}{'枠連ROI':>10}{'対人気順':>11}{'三連複ROI':>11}{'対人気順':>11}")
    for months in (12, 3):
        mw, qw, ms, qs = res[months]
        print(f"{f'{months}ヶ月ごと':<18}{mw.mean()*100:>9.2f}%{(mw-qw).mean()*100:>+10.2f}pt"
              f"{ms.mean()*100:>10.2f}%{(ms-qs).mean()*100:>+10.2f}pt")
    a, b = res[12], res[3]
    for tag, i in [("枠連 軸枠×紐枠2", 0), ("三連複 BOX上位4", 2)]:
        n = min(len(a[i]), len(b[i]))
        diff = b[i][:n] - a[i][:n]
        lo, hi = boot(diff, rng)
        print(f"  {tag}: 3ヶ月ごと − 12ヶ月ごと = {diff.mean()*100:+.2f}pt  CI[{lo:+.2f},{hi:+.2f}]"
              f"  {'→ 頻繁な再学習が有利' if lo > 0 else '→ 間隔は結果を変えない'}")


if __name__ == "__main__":
    main()
