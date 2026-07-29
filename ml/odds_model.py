"""
軸②: オッズを特徴量に入れたモデル。「市場が間違えている場所」を指せるか。

現行モデルは市場追随を避けるため単勝オッズを特徴から外している（features.py の設計）。
しかし勝ちたいなら必要なのは「勝つ馬」ではなく**「市場が間違えている馬」**であり、
それを狙うならモデルに市場の値を見せた上で、そこからのズレを学ばせるのが自然。

⑤で同じ趣旨の検定（モデル/市場≥閾値の組だけ買う）を行い「閾値を上げるほど回収率が下がる」
という結果が出ているが、**あれはオッズを見ていないモデルでの結果**。
オッズを見せた（＝市場を出発点にできる）モデルでどうなるかは未実施だった。

比較:
  モデルN（現行）  : オッズなし
  モデルO（新規）  : 現行 + log(単勝オッズ) + 市場の含意確率（1/オッズをレース内で正規化）

評価:
  1. AUC …… Oが大幅に上がるのは当然（オッズは最強の予測子）。ここは確認だけ。
  2. ★オーバーレイ検定 …… 期待値 EV = 予測確率 × オッズ が高い馬だけ単勝で買う。
     市場が間違っている場所を指せているなら、EVが高い層ほど回収率が上がるはず。
     ⑤では逆に下がった（＝モデルの誤りだった）。Oでどうなるかが本題。
  3. 連系 …… 三連単マルチ×紐5 の軸/紐をどちらのモデル順で選ぶか。
     Oは市場に近づくので、(41)で見た「モデルは高配当側に賭けている」優位が消えるはず＝その確認。

実行: python3 ml/odds_model.py [シード数(既定3)]
"""
import sys

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "ml")
import features as F
from himo_sweep import POINTS, KEY, hits
from pocket_eval import load_payout_a

TN, NHIMO = "三連単マルチ", 5
PARAMS = dict(n_estimators=400, learning_rate=0.03, num_leaves=31, subsample=0.8,
              colsample_bytree=0.8, min_child_samples=100, verbose=-1)


def deciles(name, score, odds, win, nb=10):
    q = pd.qcut(pd.Series(score).rank(method="first"), nb, labels=False).to_numpy()
    print(f"\n  ■ {name}")
    print(f"    {'十分位':>8}{'頭数':>9}{'平均オッズ':>10}{'的中率':>8}{'単勝回収率':>11}")
    for b in range(nb - 1, -1, -1):
        m = q == b
        roi = (odds[m] * win[m]).sum() / m.sum() * 100
        tag = "上位" if b == nb - 1 else ("下位" if b == 0 else "")
        print(f"    {tag:>6}{b+1:>2}{m.sum():>9,}{odds[m].mean():>10.1f}{win[m].mean()*100:>7.1f}%{roi:>10.1f}%")


def main():
    n_seed = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    d = F.to_model(F.load_files())
    f = F.build_features(d)
    keep = (f["n_prior"] >= 1) & d["odds"].notna() & (d["odds"] > 0)
    d, f = d[keep].reset_index(drop=True), f[keep].reset_index(drop=True)
    win = (d["finish"] == 1).astype(int).to_numpy()
    odds = d["odds"].to_numpy(float)

    # 市場の含意確率: 1/オッズをレース内で合計1に正規化（控除率を除いた市場の見立て）
    inv = 1.0 / odds
    mkt = inv / pd.Series(inv).groupby(d["raceid"]).transform("sum").to_numpy()

    fx_n, _ = F.encode_categoricals(f)
    fx_o = fx_n.copy()
    fx_o["log_odds"] = np.log(odds)
    fx_o["mkt_prob"] = mkt

    cut = d["date"].quantile(0.3)
    tr, te = (d["date"] < cut).to_numpy(), (d["date"] >= cut).to_numpy()
    print(f"train {tr.sum():,} / test {te.sum():,}  分割日 {cut.date()}  シード{n_seed}本")
    print(f"OOS 全馬ベタ買いの単勝回収率: {(odds[te]*win[te]).sum()/te.sum()*100:.1f}%")

    pays = load_payout_a("data/payout/a.csv")
    meta = d.loc[te, ["raceid", "umaban"]].copy()
    pts = POINTS[TN](NHIMO) * 100

    preds, exo = {}, {}
    for tag, fxx in [("N(オッズなし)", fx_n), ("O(オッズあり)", fx_o)]:
        ps, es = [], []
        for seed in range(n_seed):
            m = lgb.LGBMClassifier(random_state=seed, **PARAMS)
            m.fit(fxx[tr], win[tr], categorical_feature=F.CAT_COLS)
            p = m.predict_proba(fxx[te])[:, 1]
            ps.append(p)
            pr = meta.copy()
            pr["prob"] = p
            rows = []
            for rid, g in pr.groupby("raceid", sort=False):
                pp = pays.get(rid)
                if pp is None or not pp["sanrentan"] or len(g) <= NHIMO:
                    continue
                g = g.sort_values("prob", ascending=False, kind="mergesort")
                nums = g["umaban"].astype(int).tolist()
                rows.append(hits(pp[KEY[TN]], nums[0], set(nums[1:NHIMO + 1])))
            es.append(np.array(rows, float))
            print(f"  {tag} seed {seed} 完了")
        preds[tag] = np.mean(ps, axis=0)
        exo[tag] = np.mean(es, axis=0)

    o_te, w_te = odds[te], win[te]
    rid_te = d.loc[te, "raceid"].to_numpy()
    print("\n" + "=" * 78)
    print("【1】AUC（オッズを入れれば上がるのは当然。確認のみ）")
    print("=" * 78)
    for tag in preds:
        print(f"  {tag}: {roc_auc_score(w_te, preds[tag]):.4f}")
    print(f"  市場（1/オッズ）だけ: {roc_auc_score(w_te, mkt[te]):.4f}")

    print("\n" + "=" * 78)
    print("【2】★オーバーレイ検定: EV = 予測確率 × オッズ の高い順に単勝を買う")
    print("=" * 78)
    for tag in preds:
        pr = pd.Series(preds[tag])
        norm = (pr / pr.groupby(rid_te).transform("sum")).to_numpy()
        deciles(f"{tag}  EV=確率×オッズ", norm * o_te, o_te, w_te)

    print("\n  ■ EVがしきい値以上の馬だけ買った場合の単勝回収率")
    print(f"    {'閾値':>6}" + "".join(f"{t:>16}" for t in preds))
    for th in [1.0, 1.1, 1.2, 1.5, 2.0]:
        line = f"    {th:>6.1f}"
        for tag in preds:
            pr = pd.Series(preds[tag])
            norm = (pr / pr.groupby(rid_te).transform("sum")).to_numpy()
            m = norm * o_te >= th
            line += (f"{(o_te[m]*w_te[m]).sum()/m.sum()*100:>9.1f}%({m.sum():>4,})"
                     if m.sum() > 50 else f"{'—':>16}")
        print(line)

    print("\n" + "=" * 78)
    print(f"【3】連系: {TN}×紐{NHIMO} を各モデルの順位で買う")
    print("=" * 78)
    for tag in exo:
        v = exo[tag]
        print(f"  {tag}: {v.sum()/(len(v)*pts)*100:.1f}%  ({len(v):,}R)")


if __name__ == "__main__":
    main()
