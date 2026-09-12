"""(174-枠) ★★穴馬側(236)の「モデルは市場を一度も超えていない」を**こちらで独立に測る**。

★**なぜ測るか**——**(236)は枠連側にとって最も重い主張**。渡された数字をそのまま採らない。
　⚠**ただし(141)の比の裾はモデルを使わない**（q=馬連の板→枠 / q_pool=枠連の板）ので、
　　★**この結果が(141)を否定することはない**。**効くのは運用（waku_score でレースと軸を選ぶ側）**。

★**事前登録（この2つだけ。後から増やさない）**
　①**主判定**: 検証期間の**年別 AUC(モデル) − AUC(市場)**。
　　★**仮説が偽（モデルに市場超過が無い）なら 0 かマイナスを返す**（判定基準42）。
　　⚠**「差が正の年が何年あるか」は数えるが、年ごとの有意性は主張しない**（判定基準39）。
　②**内部対照**: **検証行数**を最初に出す。**母集団の取り違えを防ぐ**（穴馬側は9回踏んだ）。

⚠**AUCは「順位付けの良さ」であって「賭けて勝てるか」ではない**。
　**市場を超えていなくても、控除率の低い券種・板の歪みを突く経路は別に在りうる**。
　★**逆に、超えていれば勝てるということでもない**（(88): 市場の誤りは最大+10pt、控除は20.5pt）。

実行: python3 ml/audit_model_vs_market.py
"""
import sys

import numpy as np

sys.path.insert(0, "ml")
import features as F
from train_prod import CAPACITY, add_odds_features, fit_seeds


def auc(y, s):
    """順位ベースのAUC（同値は平均順位）。scikit不要。"""
    o = np.argsort(s, kind="mergesort")
    r = np.empty(len(s), float)
    r[o] = np.arange(1, len(s) + 1)
    # 同値の平均順位
    su = np.sort(s)
    i = 0
    while i < len(su):
        j = i
        while j + 1 < len(su) and su[j + 1] == su[i]:
            j += 1
        if j > i:
            r[o[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def main():
    MODEL_DIR, PAR = CAPACITY["l2"]
    d = F.to_model(F.load_files())
    f = F.build_features(d)
    keep = (f["n_prior"] >= 1) & d["odds"].notna() & (d["odds"] > 0)
    d, f = d[keep].reset_index(drop=True), f[keep].reset_index(drop=True)
    y = (d["finish"] <= 3).astype(int).to_numpy()
    fx, _ = F.encode_categoricals(f)
    od = d["odds"].to_numpy(float)
    fx = add_odds_features(fx, od, d["raceid"].to_numpy())
    cut = d["date"].quantile(0.3)
    tr, te = (d["date"] < cut).to_numpy(), (d["date"] >= cut).to_numpy()

    print("(174-枠) ★穴馬側(236)「モデルは市場を一度も超えていない」の独立検算")
    print(f"■ ★②内部対照（母集団を最初に出す）")
    print(f"　全体 {len(d):,}行 / 学習 {int(tr.sum()):,}行（〜{cut.date()}） /"
          f" ★検証 {int(te.sum()):,}行（{cut.date()}〜{d['date'].max().date()}）")
    print(f"　検証レース {d.loc[te,'raceid'].nunique():,}\n")

    ms = fit_seeds(fx[tr], y[tr], 3, PAR)
    # ⚠★`predict()` は**0/1のラベル**を返す。**確率は `predict_proba(...)[:, 1]`**。
    # 　2026-09-09に間違えて AUC 0.6308 と出し、市場0.8081に「大敗」と誤読しかけた。
    # 　★他の監査（audit_gap_axis 等）は全部 predict_proba を使っている。**揃える**。
    p = np.mean([m.predict_proba(fx)[:, 1] for m in ms], axis=0)

    # 市場: レース内で 1/odds を正規化（穴馬側の qp と同じ定義）
    rid = d["raceid"].to_numpy()
    inv = 1.0 / od
    s = {}
    for r_, v in zip(rid, inv):
        s[r_] = s.get(r_, 0.0) + v
    mk = inv / np.array([s[r_] for r_ in rid])

    print("■ ★①主判定 — 年別 AUC（**検証期間のみ**）")
    print(f"{'年':>6}{'行':>9}{'AUC(モデル)':>13}{'AUC(市場)':>12}{'★差':>10}")
    yr = d["date"].dt.year.to_numpy()
    pos = neg = 0
    rows = []
    for Y in sorted(set(yr[te])):
        m_ = te & (yr == Y)
        if m_.sum() < 500:
            continue
        a1, a2 = auc(y[m_], p[m_]), auc(y[m_], mk[m_])
        rows.append((Y, int(m_.sum()), a1, a2, a1 - a2))
        pos += a1 > a2
        neg += a1 <= a2
        print(f"{Y:>6}{int(m_.sum()):>9,}{a1:>13.4f}{a2:>12.4f}{a1-a2:>+10.4f}")
    dif = np.array([r[4] for r in rows])
    print(f"\n　★差が正の年 {pos}/{len(rows)}　差の範囲 [{dif.min():+.4f}, {dif.max():+.4f}]"
          f"　中央値 {np.median(dif):+.4f}")
    a1, a2 = auc(y[te], p[te]), auc(y[te], mk[te])
    print(f"　★検証全体 AUC(モデル) {a1:.4f} / AUC(市場) {a2:.4f} / **差 {a1-a2:+.4f}**")

    print("\n" + "=" * 78)
    print("★読み方")
    print("  ⚠**AUCは順位付けの良さ。賭けて勝てるかではない**（控除率が入っていない）。")
    print("  ★**(141)の比の裾はモデルを使わない**ので、この結果はそこを否定しない。")
    print("  ★**効くのは運用側**——waku_score でレースと軸を選ぶ経路。")


if __name__ == "__main__":
    main()
