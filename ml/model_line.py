"""
(75) オッズの線を**モデルの作り方の側**に入れる。

**なぜ**: (73)(75-A)で「オッズから線を引いて買い目を弾く」話をしたが、
ユーザー方針は**「モデルが推奨したなら買う」＝後段フィルタは採らない**。
線を引くなら**モデルを作る段階**に入れるべき、という指摘を受けての検証。

HANDOFFの「次に試すべき軸」に、**未実施のまま残っていた項目**がこれ:
> 必要なのは「勝つ馬」ではなく **「市場が間違えている馬」**。
> 市場のオッズ由来確率と実結果のズレを予測対象にする形は未実施。

★既に潰してあるもの（重複して測らないこと）:
  ・オッズを特徴に入れる … (45)で実施済み＝**現行モデルがこれ**
  ・レース内相対特徴 … (57)で否定 ／ 馬番・枠番 … (67)Aで否定
  ・モデルの積スコアで買い目を絞る … (62)（84.5→85.6%が限界）
  ・目的関数をEV回帰に … (30)で改善せず

ここで測るのは4つ:
  A. 現行（top3分類・オッズ入り）… 基準
  B. **残差学習**: 目標 = 実際の3着以内 − **市場含意の3着以内確率**（Harville）。
     順位付け = 市場含意 + 予測残差（市場を土台に補正する）
  C. 残差学習・順位付け = **予測残差のみ**（純粋に「市場が間違えている順」）
  D. **学習データの足切り**: 単勝100倍超の馬を学習から外す（極端な人薄は結果がノイズ）
  E. **重み付け学習**: sample_weight = 市場含意確率（当たる可能性のある馬を重く）

★市場含意の3着以内確率は Harville で作る。単勝の含意確率(1/オッズを正規化)だけでは
　「1着になる確率」しか無く、3着以内目標の残差が作れないため。

判定: 枠連ROI / 三連複ROI / 的中率 / AUC を、**人気順の対照とシード幅つき**で並べる。
★(63)以下5例の「精度を上げるとROIが下がる」があるので、**AUCが上がってROIが下がったら採らない**。

実行: python3 ml/model_line.py [シード数(既定2)]
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


def harville_top3(p):
    """単勝の含意確率ベクトル → 各馬の「3着以内に入る」確率（Harville）。

    P(i∈top3) = P(1着) + P(誰かが1着で i が2着) + P(2頭が先着して i が3着)。
    レース単位で n×n 行列に落として計算する（n≤18なので十分速い）。
    """
    p = np.asarray(p, dtype=float)
    p = p / p.sum()
    n = len(p)
    if n <= 3:
        return np.ones(n)
    out = p.copy()
    d1 = 1.0 - p                                  # j が1着を取った後の残り
    d1 = np.where(d1 > 1e-9, d1, np.inf)
    M = p[None, :] / d1[:, None]                  # M[j,i] = p_i/(1-p_j)
    np.fill_diagonal(M, 0.0)
    out += (p[:, None] * M).sum(axis=0)           # i が2着
    d2 = 1.0 - p[:, None] - p[None, :]            # j,k が先着した後の残り
    W = p[:, None] * p[None, :] / d1[:, None]     # p_j * p_k/(1-p_j)
    np.fill_diagonal(W, 0.0)
    T = W / np.where(d2 > 1e-9, d2, np.inf)       # T[j,k]（i の分は後で掛ける）
    tot = T.sum()
    out += p * (tot - T.sum(axis=1) - T.sum(axis=0))
    return np.clip(out, 1e-6, 1 - 1e-6)


def market_top3(d, odds):
    """レースごとに Harville で市場含意の3着以内確率を作る。"""
    out = np.empty(len(d))
    inv = 1.0 / odds
    for _, idx in d.groupby("raceid", sort=False).indices.items():
        out[idx] = harville_top3(inv[idx])
    return out


def bets(sub, wu, pa, col):
    """`col` の降順で並べて 枠連/三連複BOX4 を買ったときの回収倍率。"""
    rw, rt = [], []
    for rid, g in sub.groupby("raceid", sort=False):
        n = int(g["fieldsize"].iloc[0])
        nums = g.sort_values(col, ascending=False, kind="mergesort")["umaban"].astype(int).tolist()
        w, s3 = wu.get(rid), pa.get(rid)
        if w and w["wakuren"] and len(g) >= 3:
            cs = wakuren_cs(nums, n)
            pay = sum(w["wakuren"].get(c, 0) for c in cs)
            rw.append(pay / (len(cs) * 100.0))
        if s3 and s3["sanrenpuku"] and len(g) >= 9:
            cs = [tuple(sorted(c)) for c in itertools.combinations(nums[:4], 3)]
            rt.append(sum(s3["sanrenpuku"].get(c, 0) for c in cs) / 400.0)
    return np.array(rw), np.array(rt)


def auc(y, s):
    o = np.argsort(s)
    r = np.empty(len(s))
    r[o] = np.arange(1, len(s) + 1)
    n1 = y.sum()
    n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def main():
    n_seed = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    d, fx, odds = load()
    y = (d["finish"] <= 3).astype(int).to_numpy()
    cut = d["date"].quantile(0.3)
    tr, te = (d["date"] < cut).to_numpy(), (d["date"] >= cut).to_numpy()
    print(f"train {tr.sum():,} / test {te.sum():,}（分割 {cut.date()}）  シード{n_seed}本")

    print("市場含意の3着以内確率（Harville）を作成中…")
    mt3 = market_top3(d, odds)
    print(f"  市場含意top3の平均 {mt3.mean():.4f} / 実際のtop3率 {y.mean():.4f}"
          f"（レース内合計は約3.0になるはず: {mt3.sum()/d['raceid'].nunique():.3f}）")
    print(f"  市場含意top3 単体のAUC(test) {auc(y[te], mt3[te]):.4f}  ← モデルはこれを超えないと意味がない")

    wu, pa = load_wu(PAYOUT), load_payout_a(PAYOUT)
    sub = d.loc[te, ["raceid", "umaban", "fieldsize"]].copy()
    sub["odds"] = odds[te]
    sub["mt3"] = mt3[te]

    # 対照: 人気順（=単勝オッズ昇順 = 市場含意の降順）
    sub["_pop"] = -sub["odds"]
    pw, pt = bets(sub, wu, pa, "_pop")
    print(f"\n対照 人気順: 枠連 {pw.mean()*100:.1f}%（{len(pw):,}R） / "
          f"三連複BOX4 {pt.mean()*100:.1f}%（{len(pt):,}R）")

    resid = y - mt3          # 残差学習の目標（市場がどちらに間違えているか）
    keep_d = odds <= 100     # D: 単勝100倍超を学習から外す
    print(f"D の足切りで学習から外れる行: {(~keep_d[tr]).sum():,} / {tr.sum():,}"
          f"（{(~keep_d[tr]).mean()*100:.1f}%）")

    rng = np.random.default_rng(0)
    print(f"\n{'構成':<40}{'AUC':>8}{'枠連ROI':>10}{'95%CI':>16}"
          f"{'三連複ROI':>11}{'的中率':>8}{'シード幅(枠連)':>16}")
    results = {}
    for tag, name in [("A", "A 現行（top3分類・オッズ入り）"),
                      ("B", "B 残差学習・順位=市場含意+予測残差"),
                      ("C", "C 残差学習・順位=予測残差のみ"),
                      ("D", "D 学習データ足切り（単勝100倍超を除外）"),
                      ("E", "E 重み付け学習（w=市場含意確率）")]:
        per_seed_w, per_seed_t, per_seed_auc = [], [], []
        for s in range(n_seed):
            if tag in ("A", "D", "E"):
                m = lgb.LGBMClassifier(random_state=s, **PARAMS)
                if tag == "A":
                    m.fit(fx[tr], y[tr], categorical_feature=F.CAT_COLS)
                elif tag == "D":
                    k = tr & keep_d
                    m.fit(fx[k], y[k], categorical_feature=F.CAT_COLS)
                else:
                    m.fit(fx[tr], y[tr], sample_weight=mt3[tr],
                          categorical_feature=F.CAT_COLS)
                score = m.predict_proba(fx[te])[:, 1]
            else:
                m = lgb.LGBMRegressor(random_state=s, **PARAMS)
                m.fit(fx[tr], resid[tr], categorical_feature=F.CAT_COLS)
                pred = m.predict(fx[te])
                score = (mt3[te] + pred) if tag == "B" else pred
            sub["_s"] = score
            bw, bt = bets(sub, wu, pa, "_s")
            per_seed_w.append(bw)
            per_seed_t.append(bt)
            per_seed_auc.append(auc(y[te], score))
        w_mean = np.mean([x.mean() for x in per_seed_w]) * 100
        t_mean = np.mean([x.mean() for x in per_seed_t]) * 100
        hit = np.mean([(x > 0).mean() for x in per_seed_w]) * 100
        spread = max(x.mean() for x in per_seed_w) - min(x.mean() for x in per_seed_w)
        lo, hi = boot(per_seed_w[0], rng, 2000)
        results[tag] = (w_mean, t_mean)
        print(f"{name:<40}{np.mean(per_seed_auc):>8.4f}{w_mean:>9.1f}%"
              f"{f'[{lo:.1f},{hi:.1f}]':>16}{t_mean:>10.1f}%{hit:>7.1f}%"
              f"{spread*100:>15.2f}pt")

    print("\n■ 読み方")
    print("  ・Aが基準。BCDEがAを**ROIで**上回り、かつ差がシード幅より大きいときだけ意味がある。")
    print("  ・(63)以下5例の通り、**AUCが上がってROIが下がる構成は採らない**。")
    print("  ・Cが極端に悪いなら「市場が間違えている順」だけでは買えない＝市場の土台が要る、ということ。")
    for tag in ("B", "C", "D", "E"):
        dw = results[tag][0] - results["A"][0]
        dt = results[tag][1] - results["A"][1]
        print(f"  {tag} − A: 枠連 {dw:+.2f}pt / 三連複 {dt:+.2f}pt")


if __name__ == "__main__":
    main()
