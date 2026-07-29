"""
軸①: 目的関数を「的中精度」から「回収率」へ変える。

動機（(24)の発見）: 騎手・厩舎を足すと **AUC 0.764→0.776と精度は上がったのに回収率は141.9%→64.3%へ低下**した。
＝**予測精度と回収率は別物**。現行モデルは「1着になる確率」の精度を最大化しているだけで、
回収率を直接目指してはいない。ならば最初から回収率を目的関数にすべき、というのがこの実験。

やり方: 各馬について「単勝に1単位賭けたときの純収益」
    y = odds × 1{1着} − 1     （外れれば必ず −1、的中すればオッズ−1）
を**回帰**で直接予測する。y の期待値がそのまま期待値(EV)なので、
予測値が正の馬＝モデルが「賭ける価値がある」と判断した馬になる。

比較対象: 現行の分類器（1着確率）から作る EV = win_prob × odds − 1。
同じ土台（同じ特徴・同じ分割）で、どちらの並べ方が実際の回収率を高い順に並べられるかを見る。

評価: OOS を予測EVの十分位に分け、各バケットの**実際の単勝回収率**を出す。
エッジがあれば上位バケットが100%を超える。100%を超えなければ、目的関数を変えても勝てないと確定する。

実行: python3 ml/train_ev.py [分割位置(既定0.3)]
"""
import sys

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "ml")
import features as F

ODDS_CAP = 100.0  # 極端な大穴の的中1本が学習を支配するのを防ぐ（評価は生オッズで行う）


def buckets(name, score, odds, win, nb=10):
    """score の十分位ごとに実際の単勝回収率を出す。score が高い側が上位バケット。"""
    q = pd.qcut(score.rank(method="first"), nb, labels=False)
    print(f"\n■ {name} の十分位別 実測単勝回収率（OOS）")
    print(f"{'十分位':>6}{'頭数':>9}{'平均オッズ':>10}{'的中率':>8}{'回収率':>9}")
    for b in range(nb - 1, -1, -1):
        m = q == b
        roi = (odds[m] * win[m]).sum() / m.sum() * 100
        print(f"{'上位' if b == nb - 1 else ('下位' if b == 0 else ''):>4}{b + 1:>2}"
              f"{m.sum():>9,}{odds[m].mean():>10.1f}{win[m].mean() * 100:>7.1f}%{roi:>8.1f}%")


def main():
    q = float(sys.argv[1]) if len(sys.argv) > 1 else 0.3
    d = F.to_model(F.load_files())
    f = F.build_features(d)
    keep = (f["n_prior"] >= 1) & d["odds"].notna() & (d["odds"] > 0)
    d, f = d[keep].reset_index(drop=True), f[keep].reset_index(drop=True)

    win = (d["finish"] == 1).astype(int).to_numpy()
    odds = d["odds"].to_numpy(float)
    # 目的変数: 1単位賭けたときの純収益。外れ=-1、的中=オッズ-1。
    y_ev = np.where(win == 1, np.minimum(odds, ODDS_CAP), 0.0) - 1.0

    cut = d["date"].quantile(q)
    tr, te = (d["date"] < cut).to_numpy(), (d["date"] >= cut).to_numpy()
    fx, _ = F.encode_categoricals(f)
    print(f"分割 {q:.0%}: train {tr.sum():,} / test {te.sum():,}  分割日 {cut.date()}")
    print(f"OOS の全馬ベタ買い回収率: {(odds[te] * win[te]).sum() / te.sum() * 100:.1f}%（控除率どおりの基準線）")

    common = dict(n_estimators=400, learning_rate=0.03, num_leaves=31, subsample=0.8,
                  colsample_bytree=0.8, min_child_samples=100, verbose=-1)

    # === 軸①: 回収率(EV)を直接回帰 ===
    reg = lgb.LGBMRegressor(objective="huber", **common)  # 裾が重いのでHuber
    reg.fit(fx[tr], y_ev[tr], categorical_feature=F.CAT_COLS)
    ev_pred = reg.predict(fx)

    # === 比較: 現行の分類器（1着確率）→ EV = prob × odds − 1 ===
    clf = lgb.LGBMClassifier(**common)
    clf.fit(fx[tr], win[tr], categorical_feature=F.CAT_COLS)
    p = clf.predict_proba(fx)[:, 1]
    dd = pd.DataFrame({"raceid": d["raceid"], "p": p})
    prob = (dd["p"] / dd.groupby("raceid")["p"].transform("sum")).to_numpy()
    ev_clf = prob * odds - 1.0

    print(f"\n分類器のOOS AUC: {roc_auc_score(win[te], p[te]):.4f}"
          f"   EV回帰のOOS AUC(参考): {roc_auc_score(win[te], ev_pred[te]):.4f}")

    s = pd.Series(ev_pred[te])
    buckets("【軸①】EVを直接回帰したモデル", s, odds[te], win[te])
    buckets("【比較】現行分類器の win_prob × odds − 1", pd.Series(ev_clf[te]), odds[te], win[te])
    buckets("【参考】現行分類器の win_prob そのもの", pd.Series(prob[te]), odds[te], win[te])

    print("\n■ 予測EVが正（賭ける価値ありと判断）の馬だけを買った場合")
    for name, sc in [("軸①EV回帰", ev_pred[te]), ("比較 分類器EV", ev_clf[te])]:
        m = sc > 0
        if m.sum() == 0:
            print(f"  {name}: 該当なし")
            continue
        roi = (odds[te][m] * win[te][m]).sum() / m.sum() * 100
        print(f"  {name}: {m.sum():,}頭 ({m.mean() * 100:.1f}%)  的中率{win[te][m].mean() * 100:.1f}%  回収率 {roi:.1f}%")


if __name__ == "__main__":
    main()
