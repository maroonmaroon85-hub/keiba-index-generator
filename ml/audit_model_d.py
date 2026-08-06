"""
(90) ★現行MLモデルの D を測る — (89)の宿題1。**モデルが市場の上に何nat積めているか**を1つの数字で出す。

(89)は q に「単勝オッズのHarville」しか入れていない。ここでは **モデル由来の確率**で同じ D を測る。
　　D = E[log q] − E[log q_pool]、成長率 = log(払戻率) + D
★判定は一撃: **D ≥ |log(払戻率)| なら儲かる**。枠連なら D ≥ 0.2549 が必要。
　単勝オッズのHarvilleは +0.0145（必要量の5.7%）。モデルがこれを超えるか、下回るか。

★なぜROIでなくDで測るのか（(89)の要点）
　ROI比較は**この標本量では測定限界に負ける**（枠連で1ptを検出するのに50年分）。
　Dは**1レース1標本**なので桁違いに精密（枠連でCI幅0.008）。しかも
　**ケリーが上限なので、どんな買い方をしてもD以上には増えない**＝買い方の探索が不要になる。

★実装上の要点
 1. **Harvilleには勝率が要る**。現行モデルは3着以内目標なのでそのままでは入らない。
    → **同じ特徴・同じオッズ入りで「1着」を目標にしたモデル**を学習して使う（win-target）。
    参考として3着以内モデルの確率を正規化したもの（p3）も出すが、**これは勝率ではない**ので
    Harvilleの前提を満たさない。数字は載せるが解釈しないこと。
 2. **ウォークフォワード**（各年をそれ以前の全データで学習）＝実運用の手続き。(78)①より単一分割は使わない。
 3. **過去走の無い馬**（新馬など）はモデルが確率を出せない。その馬には**市場の含意勝率を入れて**
    レース内で再正規化する（モデルがある馬だけモデル、無い馬は市場）。カバー率を必ず出す。
 4. **同じレース集合で市場のHarvilleも計算し、対応ありで差を取る**。
    これが「モデルが市場の上に積んだ分」そのもの。
 5. 2015年以降に限定（(89)③: 2014年の控除率改定が t* の段差として実測で見えている）。

実行: python3 ml/audit_model_d.py [シード数(既定3)] [開始年(既定2015)]
"""
import math
import sys
import warnings

import numpy as np
import pandas as pd
import lightgbm as lgb

warnings.filterwarnings("ignore")
sys.path.insert(0, "ml")
import features as F
from _cache import load_cached
from audit_crosspool import PAYBACK, load_races, payoff, probs, zq
from audit_crosspool2 import PARTS, PAYKEY, realized
from audit_crosspool3 import q_of
from place_wide import PARAMS


def model_probs(n_seed, y0):
    """ウォークフォワードで各年の勝率予測を作る。{raceid: {馬番: p_win}}"""
    d, fx = load_cached()
    ywin = (d["finish"] == 1).astype(int).to_numpy()
    yt3 = (d["finish"] <= 3).astype(int).to_numpy()
    year = d["date"].dt.year.to_numpy()
    years = [yy for yy in range(y0, int(year.max()) + 1) if (year == yy).sum() > 5000]
    out_w, out_3 = {}, {}
    for yy in years:
        tr, te = year < yy, year == yy
        pw = np.mean([lgb.LGBMClassifier(random_state=s, **PARAMS)
                      .fit(fx[tr], ywin[tr], categorical_feature=F.CAT_COLS)
                      .predict_proba(fx[te])[:, 1] for s in range(n_seed)], axis=0)
        p3 = np.mean([lgb.LGBMClassifier(random_state=s, **PARAMS)
                      .fit(fx[tr], yt3[tr], categorical_feature=F.CAT_COLS)
                      .predict_proba(fx[te])[:, 1] for s in range(n_seed)], axis=0)
        sub = d.loc[te, ["raceid", "umaban"]].copy()
        for rid, um, a, b in zip(sub["raceid"], sub["umaban"].astype(int), pw, p3):
            out_w.setdefault(rid, {})[int(um)] = float(a)
            out_3.setdefault(rid, {})[int(um)] = float(b)
        print(f"  {yy} 学習完了", flush=True)
    return out_w, out_3


def collect(races, pmap, y0):
    """(89)と同じ形で D を集める。pmap が None なら市場のHarville（＝(89)の再現）。

    返り値: {券種: [(d_model, d_market, rid, year), ...]} と カバー率
    """
    out = {k: [] for k in PARTS}
    n_cov = n_tot = 0
    for r in races:
        if r["year"] < y0:
            continue
        rl = realized(r)
        if rl is None:
            continue
        a, b, c = rl
        hs = r["horses"]
        p_mkt = probs(hs)                       # 市場の含意勝率（正規化済み）
        if pmap is None:
            p_use = p_mkt
        else:
            mp = pmap.get(r["rid"])
            if not mp:
                continue
            # ★過去走の無い馬はモデルが確率を出せない。その馬だけ市場の値で埋めて再正規化する。
            raw = np.array([mp.get(num, np.nan) for num, _, _ in hs], float)
            miss = np.isnan(raw)
            n_cov += int((~miss).sum())
            n_tot += len(raw)
            if miss.all():
                continue
            # モデル側の合計を、モデルが担当する馬の市場シェアに合わせてから欠損を市場で埋める
            share_mkt = p_mkt[~miss].sum()
            s = raw[~miss].sum()
            if s <= 0:
                continue
            raw[~miss] = raw[~miss] / s * share_mkt
            raw[miss] = p_mkt[miss]
            p_use = raw / raw.sum()
        num2k = {num: k for k, (num, _, _) in enumerate(hs)}
        if a not in num2k or b not in num2k or (c is not None and c not in num2k):
            continue
        for kind, key in PARTS.items():
            if not r[key]:
                continue
            qm, combo = q_of(kind, r, p_use, num2k, a, b, c)
            qk, _ = q_of(kind, r, p_mkt, num2k, a, b, c)
            if qm <= 0 or qk <= 0 or combo is None:
                continue
            v = payoff(r, PAYKEY[kind], combo)
            if not v or v <= 0:
                continue
            lp = math.log((v + 5) / 100.0) - math.log(PAYBACK[kind])
            out[kind].append((math.log(qm) + lp, math.log(qk) + lp, r["rid"], r["year"]))
    cov = (n_cov / n_tot * 100) if n_tot else 100.0
    return out, cov


def mci(x, alpha=0.01):
    x = np.asarray(x, float)
    m = x.mean()
    se = x.std(ddof=1) / math.sqrt(len(x))
    z = zq(alpha)
    return m, m - z * se, m + z * se


def main():
    n_seed = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    y0 = int(sys.argv[2]) if len(sys.argv) > 2 else 2015
    print(f"現行モデルの D（{y0}年以降・シード{n_seed}本・ウォークフォワード）")
    print("★判定: D ≥ |log(払戻率)| なら儲かる。枠連なら 0.2549\n")
    pw, p3 = model_probs(n_seed, y0)
    races = load_races()

    for tag, pmap in (("勝率目標モデル（win）", pw),
                      ("3着以内モデルを正規化（★勝率ではない・参考）", p3)):
        res, cov = collect(races, pmap, y0)
        print(f"\n{'='*104}\n=== {tag}  カバー率 {cov:.1f}% ===\n{'='*104}")
        print(f"{'券種':<8}{'R数':>8}{'D(モデル)':>12}{'99%CI':>22}"
              f"{'D(市場)':>11}{'差':>11}{'差の99%CI':>22}{'必要量':>9}{'埋まる割合':>11}")
        for kind in PARTS:
            v = res[kind]
            if len(v) < 500:
                continue
            dm = np.array([x[0] for x in v])
            dk = np.array([x[1] for x in v])
            m, lo, hi = mci(dm)
            g, glo, ghi = mci(dm - dk)
            need = -math.log(PAYBACK[kind])
            print(f"{kind:<8}{len(v):>8,}{m:>+12.4f}{f'[{lo:+.4f},{hi:+.4f}]':>22}"
                  f"{dk.mean():>+11.4f}{g:>+11.4f}{f'[{glo:+.4f},{ghi:+.4f}]':>22}"
                  f"{need:>9.4f}{max(m,0)/need*100:>10.1f}%")
        print("  ※「差」＝モデル − 市場。**プラスならモデルが市場の上に情報を積めている**。")
        print("  ※「必要量」＝|log(払戻率)|。Dがこれを超えたときだけ儲かる。")

    print(f"\n{'='*104}")
    print("★読み方")
    print("  ・差がマイナスなら**モデルは市場より劣る**。(87)の『モデルのAUC 0.8067 < 市場 0.8101』と整合し、")
    print("    枠連84.5%は人気順82.8%の測定ノイズだった可能性が高くなる（(89)⑤の分解）。")
    print("  ・差がプラスなら**このプロジェクトで初めての『市場を超えた』直接証拠**。")
    print("    ただし D が必要量に届かない限り、どんな買い方をしても儲からない（ケリーが上限）。")
    print("  ・3着以内モデルの行は**Harvilleの前提（勝率）を満たさない**ので解釈しないこと。")


if __name__ == "__main__":
    main()
