"""
軸①: 複勝とワイドを測る。控除率が最も低い2券種で、(45)のオッズ入りモデルを使う。

これまで(31)以降は馬連/三連複/三連単しか測っていなかったが、配当A(a.csv)には
**複勝(col93)とワイド(col127)も入っている**（列レイアウトは src/backtest/payout-parser.ts の冒頭に記録）。
この2つは控除率が最も低く、天井が高い:
    複勝 20.0% → 天井80.0%   ／   ワイド 22.5% → 天井77.5%
さらにモデルの最大の強みは top-pick複勝率58-60%＝「3着以内に入る馬を当てる」能力で、複勝はそれを直接換金する券種。

使うモデルは(45)のオッズ入り（log(単勝オッズ)＋市場の含意確率を特徴に追加）。
複勝は事前オッズを持っていないため真のEVは計算できない。そこで
  ・単勝EV（=win確率×単勝オッズ）で選ぶ … (45)で機能が確認された選択規則をそのまま適用
  ・top3目標モデルの確率で選ぶ         … 複勝そのものを狙った選択
の2通りで測り、全馬ベタ買いおよびオッズ帯別と比較する（(45)③と同じ対照）。

ワイドは「上位3頭ボックス(3点)」で、モデル順と人気順を比較する。

実行: python3 ml/place_wide.py [シード数(既定3)]
"""
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, "ml")
import features as F
from pocket_eval import _slots, load_payout_a

PARAMS = dict(n_estimators=400, learning_rate=0.03, num_leaves=31, subsample=0.8,
              colsample_bytree=0.8, min_child_samples=100, verbose=-1)


def load_place_wide(path):
    """配当A から 複勝(93: (番,配当)×5) と ワイド(127: (番,番,配当,人気)×7) を読む。"""
    import csv, io
    out = {}
    with open(path, "rb") as fh:
        txt = fh.read().decode("shift_jis", "replace")
    for r in csv.reader(io.StringIO(txt)):
        if len(r) < 224:
            continue
        rid = r[14].strip()
        if len(rid) != 8:
            continue
        fuku = _slots(r, 93, 5, 2, 1, 1)     # {(馬番,): 配当}
        wide = _slots(r, 127, 7, 4, 2, 2)    # {(馬番,馬番): 配当}
        if fuku:
            out[rid] = {"fuku": {k[0]: v for k, v in fuku.items()}, "wide": wide}
    return out


def boot(x, rng, n=4000, chunk=200):
    """レース(点)単位のブートストラップ。標本が大きいと (n, len(x)) の一括生成で
    メモリが破綻するので、リサンプルを chunk 回ずつに分けて回す。"""
    means = []
    for start in range(0, n, chunk):
        k = min(chunk, n - start)
        i = rng.integers(0, len(x), size=(k, len(x)))
        means.append(x[i].mean(axis=1))
    b = np.concatenate(means) * 100
    return np.percentile(b, 2.5), np.percentile(b, 97.5)


def main():
    n_seed = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    d = F.to_model(F.load_files())
    f = F.build_features(d)
    keep = (f["n_prior"] >= 1) & d["odds"].notna() & (d["odds"] > 0)
    d, f = d[keep].reset_index(drop=True), f[keep].reset_index(drop=True)
    win = (d["finish"] == 1).astype(int).to_numpy()
    top3 = (d["finish"] <= 3).astype(int).to_numpy()
    odds = d["odds"].to_numpy(float)
    inv = 1.0 / odds
    mkt = inv / pd.Series(inv).groupby(d["raceid"]).transform("sum").to_numpy()

    fx, _ = F.encode_categoricals(f)
    fx["log_odds"] = np.log(odds)
    fx["mkt_prob"] = mkt
    cut = d["date"].quantile(0.3)
    tr, te = (d["date"] < cut).to_numpy(), (d["date"] >= cut).to_numpy()
    print(f"train {tr.sum():,} / test {te.sum():,}  シード{n_seed}本（(45)のオッズ入りモデル）")

    pw = load_place_wide("data/payout/a.csv")
    print(f"配当Aから複勝/ワイドを読込: {len(pw):,}レース")

    ps_win, ps_t3 = [], []
    for seed in range(n_seed):
        for tgt, store in [(win, ps_win), (top3, ps_t3)]:
            m = lgb.LGBMClassifier(random_state=seed, **PARAMS)
            m.fit(fx[tr], tgt[tr], categorical_feature=F.CAT_COLS)
            store.append(m.predict_proba(fx[te])[:, 1])
        print(f"  seed {seed} 完了")
    p_win = np.mean(ps_win, axis=0)
    p_t3 = np.mean(ps_t3, axis=0)

    sub = d.loc[te, ["raceid", "umaban"]].copy()
    sub["odds"] = odds[te]
    sub["top3"] = top3[te]
    sub["year"] = d.loc[te, "date"].dt.year.to_numpy()
    pw_ser = pd.Series(p_win)
    sub["ev"] = (pw_ser / pw_ser.groupby(sub["raceid"].to_numpy()).transform("sum")).to_numpy() * sub["odds"]
    sub["p3"] = p_t3

    # 複勝の払戻を各馬に付ける（無いレースは除外）
    fuku = []
    for rid, um in zip(sub["raceid"], sub["umaban"]):
        p = pw.get(rid)
        fuku.append(p["fuku"].get(int(um), 0) if p else np.nan)
    sub["fuku"] = fuku
    s = sub[sub["fuku"].notna()].copy()
    s["ret"] = s["fuku"] / 100.0          # 100円賭けたときの回収倍率
    rng = np.random.default_rng(0)
    print(f"\n複勝の対象: {len(s):,}頭 / {s['raceid'].nunique():,}レース")

    print("\n" + "=" * 74)
    print("【複勝】控除率20% → 無技能の天井 80.0%")
    print("=" * 74)
    print(f"{'選び方':<26}{'点数':>9}{'的中率':>8}{'ROI':>8}{'95%CI':>16}")

    def line(name, m):
        x = s.loc[m, "ret"].to_numpy(float)
        if len(x) < 200:
            print(f"{name:<26}{len(x):>9,}  標本不足")
            return
        lo, hi = boot(x, rng)
        print(f"{name:<26}{len(x):>9,}{(x>0).mean()*100:>7.1f}%{x.mean()*100:>7.1f}%"
              f"{f'[{lo:.1f},{hi:.1f}]':>16}")

    line("全馬ベタ買い", s["ret"].notna())
    for th in [1.0, 1.2, 1.5]:
        line(f"単勝EV≥{th} の馬", s["ev"] >= th)
    for q in [0.9, 0.95, 0.99]:
        line(f"top3確率 上位{(1-q)*100:.0f}%", s["p3"] >= s["p3"].quantile(q))
    line("各レースのtop3確率1位", s.groupby("raceid")["p3"].transform("max") == s["p3"])

    print("\n■ オッズ帯を揃えた対照（複勝・全部買う vs 単勝EV≥1.0）")
    print(f"{'オッズ帯':<12}{'全部':>10}{'EV≥1.0':>11}{'差':>8}{'点数':>8}")
    for lo_, hi_ in [(1, 3), (3, 8), (8, 20), (20, 50), (50, 10000)]:
        m = (s["odds"] >= lo_) & (s["odds"] < hi_)
        m2 = m & (s["ev"] >= 1.0)
        if m2.sum() < 200:
            continue
        a = s.loc[m, "ret"].mean() * 100
        b = s.loc[m2, "ret"].mean() * 100
        print(f"{f'{lo_}-{hi_}倍':<12}{a:>9.1f}%{b:>10.1f}%{b-a:>+8.1f}{m2.sum():>8,}")

    best = s[s["ev"] >= 1.0]
    print("\n  年別（単勝EV≥1.0 の複勝）: " +
          " ".join(f"{y}:{g['ret'].mean()*100:.0f}%" for y, g in best.groupby("year")))

    # ===== ワイド: 上位3頭ボックス(3点) =====
    print("\n" + "=" * 74)
    print("【ワイド】上位3頭ボックス(3点)  控除率22.5% → 天井77.5%")
    print("=" * 74)
    rows = []
    for rid, g in sub.groupby("raceid", sort=False):
        p = pw.get(rid)
        if p is None or not p["wide"] or len(g) < 4:
            continue
        ml = g.sort_values("ev", ascending=False)["umaban"].astype(int).tolist()[:3]
        mp = g.sort_values("p3", ascending=False)["umaban"].astype(int).tolist()[:3]
        mk = g.sort_values("odds", ascending=True)["umaban"].astype(int).tolist()[:3]
        r = {"year": g["year"].iloc[0]}
        for tag, top in [("EV順", ml), ("top3確率順", mp), ("人気順", mk)]:
            pay = 0
            for i in range(3):
                for j in range(i + 1, 3):
                    pay += p["wide"].get(tuple(sorted((top[i], top[j]))), 0)
            r[tag] = pay / 300.0
        rows.append(r)
    w = pd.DataFrame(rows)
    print(f"対象 {len(w):,}レース")
    print(f"{'選び方':<16}{'的中率':>9}{'ROI':>8}{'95%CI':>16}{'年別の範囲':>16}")
    for tag in ["EV順", "top3確率順", "人気順"]:
        x = w[tag].to_numpy(float)
        lo, hi = boot(x, rng)
        ys = w.groupby("year")[tag].mean() * 100
        print(f"{tag:<16}{(x>0).mean()*100:>8.1f}%{x.mean()*100:>7.1f}%"
              f"{f'[{lo:.1f},{hi:.1f}]':>16}{f'{ys.min():.0f}〜{ys.max():.0f}%':>16}")


if __name__ == "__main__":
    main()
