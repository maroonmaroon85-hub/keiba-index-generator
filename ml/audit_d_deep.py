"""
(87) 指数Dの上位区間を**記述的に**深掘りする。何を見ているのかを特定する。

(86)で「Dの上位区間だけ買っても100%を超えない」は決着した。ここで見るのは**別の問い**:
**Dは本当に「モデルと市場の乖離」を測っているのか**。

★疑い: Dの定義には**非対称**がある。
    D = 軸のシェア − 軸の市場含意確率
      ・軸のシェア        … モデルの**3着以内確率**をレース内で正規化したもの
      ・軸の市場含意確率  … **1÷単勝オッズ**を正規化したもの ＝ 実質**勝率**
    3着以内確率と勝率を引いている。構造的に、断然の1番人気は「勝率シェア>3着以内シェア」
    （勝ちが集中する）、人気薄は逆になる。
    → **Dは「乖離」ではなく単に「軸が人気薄か」を測っているだけかもしれない**。
      (86)で上位区間の軸の平均単勝が8.1倍だったのも、それで説明が付いてしまう。

そこで**揃えた版**と比べる:
    D2 = モデルの3着以内確率 − **市場含意の3着以内確率（Harville）**
D と D2 で曲線の形が変わるなら、(85)(86)が見ていたのは乖離ではなく**人気薄の軸**だった。

同時に見る3つ:
  ① 較正の比較 … 上位区間で軸の**実際の**勝率・複勝率が、モデル予測と市場含意のどちらに近いか
  ② 単勝と複勝の食い違い … (85)で単勝は D が ρ=+0.721 で通ったのに**複勝は ρ=−0.285 で逆向き**。
     「勝つが3着以内には入らない」馬を見つけているなら実在する構造
  ③ Dを駆動している特徴 … 上位区間の軸はどんな馬か（近走・休養明け・クラス・距離・頭数）

★これは**記述的な分析**。儲かる買い方を探すのではないので多重性の罠には入らないが、
　**ここで見つけた形を「だから儲かる」に繋げるときは(86)と同じ5条件を当てること**。

実行: python3 ml/audit_d_deep.py [シード数(既定3)] [開始年(既定2019)]
"""
import sys
import warnings

import numpy as np
import pandas as pd
import lightgbm as lgb

warnings.filterwarnings("ignore")
sys.path.insert(0, "ml")
import features as F
from _cache import load_cached
from model_line import harville_top3
from place_wide import PARAMS

PAR_L5 = dict(PARAMS, num_leaves=255, min_child_samples=10, n_estimators=2000)

# 軸の性質を見るのに使う特徴（build_features が作る列名）
TRAITS = [("n_prior", "出走経験数"), ("avg3_fin", "近3走の着順比"),
          ("last1_fin", "前走の着順比"), ("weeks_since", "前走からの週数"),
          ("class_change", "クラスの上下"), ("avg3_prize", "近3走の平均賞金"),
          ("last_passratio", "前走の通過順比"), ("best3_agari", "近3走の最速上がり"),
          ("bodywt", "馬体重"), ("age", "年齢"), ("wtcarry", "斤量"),
          ("distance", "距離"), ("fieldsize", "頭数"), ("raceclass", "クラス")]


def main():
    n_seed = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    y0 = int(sys.argv[2]) if len(sys.argv) > 2 else 2019
    d, fx = load_cached()
    y = (d["finish"] <= 3).astype(int).to_numpy()
    win = (d["finish"] == 1).astype(int).to_numpy()
    year = d["date"].dt.year.to_numpy()
    years = [yy for yy in range(y0, int(year.max()) + 1) if (year == yy).sum() > 5000]
    print(f"指数Dの深掘り（{years[0]}〜{years[-1]}・L5・シード{n_seed}本）\n")

    rows = []
    for yy in years:
        tr, te = year < yy, year == yy
        ps = [lgb.LGBMClassifier(random_state=s, **PAR_L5)
              .fit(fx[tr], y[tr], categorical_feature=F.CAT_COLS)
              .predict_proba(fx[te])[:, 1] for s in range(n_seed)]
        p = np.mean(ps, axis=0)
        sub = d.loc[te, ["raceid", "umaban", "fieldsize", "odds", "finish"]].copy()
        sub["p"] = p
        sub["win"] = win[te]
        sub["top3"] = y[te]
        # 市場含意の3着以内確率（Harville）。D2 のために揃えた量を作る
        mt3 = np.empty(len(sub))
        for _, ix in sub.groupby("raceid", sort=False).indices.items():
            mt3[ix] = harville_top3(1.0 / sub["odds"].to_numpy(float)[ix])
        sub["mkt_t3"] = mt3
        ftr = fx.loc[te].reset_index(drop=True)
        sub = sub.reset_index(drop=True)
        for rid, g in sub.groupby("raceid", sort=False):
            if len(g) < 4:
                continue
            i = g.index.to_numpy()
            pv = g["p"].to_numpy(float)
            od = g["odds"].to_numpy(float)
            inv = 1.0 / od
            mkw = inv / inv.sum()                    # 市場含意の勝率シェア
            share = pv / pv.sum()                    # モデルの3着以内シェア
            k = int(np.argmax(pv))                   # 軸＝モデル1位
            r = {"year": yy, "rid": rid,
                 "D": float(share[k] - mkw[k]),                       # (85)(86)で使った定義
                 "D2": float(pv[k] - g["mkt_t3"].to_numpy(float)[k]),  # ★揃えた版
                 "ax_odds": float(od[k]),
                 "ax_pop": int((od < od[k]).sum() + 1),                # 軸の人気順位
                 "ax_win": int(g["win"].to_numpy()[k]),
                 "ax_top3": int(g["top3"].to_numpy()[k]),
                 "p_top3": float(pv[k]), "mkt_top3": float(g["mkt_t3"].to_numpy(float)[k]),
                 "p_win_mkt": float(mkw[k])}
            for col, _ in TRAITS:
                if col in ftr:
                    r[col] = float(ftr[col].to_numpy()[i[k]])
            rows.append(r)
        print(f"  {yy} 完了", flush=True)

    df = pd.DataFrame(rows)
    print(f"\n対象 {len(df):,}レース\n")

    # ===== ① D と D2 は同じものを見ているか =====
    print("=" * 100)
    print("① D（3着以内シェア − 勝率シェア）と D2（3着以内確率どうし）の関係")
    print("=" * 100)
    print(f"  相関 r={df['D'].corr(df['D2']):+.3f} / 順位相関 ρ={df['D'].corr(df['D2'], method='spearman'):+.3f}")
    print(f"  D の平均 {df['D'].mean():+.4f} / D2 の平均 {df['D2'].mean():+.4f}")
    print(f"\n{'指数':<6}{'区間':<8}{'R数':>8}{'軸の平均単勝':>13}{'軸の平均人気':>13}"
          f"{'軸の実勝率':>11}{'軸の実複勝率':>13}{'モデル予測':>11}{'市場含意':>10}")
    for code in ("D", "D2"):
        q = pd.qcut(df[code], 10, labels=False, duplicates="drop")
        for b in (0, 4, 9):
            g = df[(q == b).to_numpy()]
            print(f"{code:<6}{'第'+str(b+1):<8}{len(g):>8,}{g['ax_odds'].mean():>12.1f}倍"
                  f"{g['ax_pop'].mean():>12.1f}番{g['ax_win'].mean()*100:>10.1f}%"
                  f"{g['ax_top3'].mean()*100:>12.1f}%{g['p_top3'].mean()*100:>10.1f}%"
                  f"{g['mkt_top3'].mean()*100:>9.1f}%")
    print("  ※Dの上位で軸のオッズが上がるだけなら、Dは『乖離』ではなく『人気薄の軸』を測っている。")
    print("  　D2で同じ形にならなければ、(85)(86)が見ていたのは乖離ではなかったことになる。")

    # ===== ② 較正: 上位区間で実測はモデルと市場のどちらに近いか =====
    print("\n" + "=" * 100)
    print("② 較正 — 軸の実際の3着以内率は、モデル予測と市場含意のどちらに近いか")
    print("=" * 100)
    for code in ("D", "D2"):
        q = pd.qcut(df[code], 10, labels=False, duplicates="drop")
        print(f"\n  【{code}】{'区間':<8}{'R数':>8}{'実測':>9}{'モデル':>9}{'市場':>9}"
              f"{'実測−モデル':>13}{'実測−市場':>12}{'どちらが近いか':>16}")
        for b in range(10):
            g = df[(q == b).to_numpy()]
            a, m, k = g["ax_top3"].mean(), g["p_top3"].mean(), g["mkt_top3"].mean()
            near = "モデル" if abs(a - m) < abs(a - k) else "市場"
            print(f"       {'第'+str(b+1):<8}{len(g):>8,}{a*100:>8.1f}%{m*100:>8.1f}%{k*100:>8.1f}%"
                  f"{(a-m)*100:>+12.1f}pt{(a-k)*100:>+11.1f}pt{near:>16}")
        gs = df.groupby(q).apply(lambda x: abs(x["ax_top3"].mean() - x["p_top3"].mean())
                                 < abs(x["ax_top3"].mean() - x["mkt_top3"].mean()))
        print(f"       → モデルの方が近い区間: {int(gs.sum())}/10")

    # ===== ③ 単勝と複勝の食い違い =====
    print("\n" + "=" * 100)
    print("③ 単勝と複勝の食い違い —『勝つが3着以内には入らない』馬を見つけているか")
    print("=" * 100)
    print(f"{'指数':<6}{'区間':<8}{'軸の実勝率':>11}{'市場含意勝率':>13}{'勝率の超過':>12}"
          f"{'軸の実複勝率':>13}{'市場含意複勝率':>15}{'複勝率の超過':>13}")
    for code in ("D", "D2"):
        q = pd.qcut(df[code], 10, labels=False, duplicates="drop")
        for b in (0, 4, 9):
            g = df[(q == b).to_numpy()]
            dw = g["ax_win"].mean() - g["p_win_mkt"].mean()
            dp = g["ax_top3"].mean() - g["mkt_top3"].mean()
            print(f"{code:<6}{'第'+str(b+1):<8}{g['ax_win'].mean()*100:>10.1f}%"
                  f"{g['p_win_mkt'].mean()*100:>12.1f}%{dw*100:>+11.2f}pt"
                  f"{g['ax_top3'].mean()*100:>12.1f}%{g['mkt_top3'].mean()*100:>14.1f}%"
                  f"{dp*100:>+12.2f}pt")
    print("  ※勝率の超過だけがプラスで複勝率の超過がマイナスなら、"
          "『勝つか飛ぶか』の馬を拾っていることになる（(85)の単勝○・複勝×と整合）。")

    # ===== ④ 上位区間の軸はどんな馬か =====
    print("\n" + "=" * 100)
    print("④ Dの上位10%の軸は、全体の軸と何が違うか（記述統計）")
    print("=" * 100)
    thr = df["D"].quantile(0.9)
    hi = df[df["D"] >= thr]
    print(f"{'特徴':<20}{'上位10%':>12}{'全体':>12}{'差':>12}{'標準化差':>11}")
    for col, name in TRAITS:
        if col not in df:
            continue
        a, b = hi[col].dropna(), df[col].dropna()
        if len(a) < 100 or b.std() == 0:
            continue
        z = (a.mean() - b.mean()) / b.std()
        mark = "  ★" if abs(z) > 0.3 else ""
        print(f"{name:<20}{a.mean():>12.3f}{b.mean():>12.3f}{a.mean()-b.mean():>+12.3f}"
              f"{z:>+11.2f}{mark}")
    print("  ※標準化差 |z|>0.3 に★。ここが『市場が見落としているもの』の候補になる。")
    print("  ⚠ただし記述統計。ここから買い方を作るなら(86)の5条件を当て直すこと。")


if __name__ == "__main__":
    main()
