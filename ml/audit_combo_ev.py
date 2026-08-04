"""B3: **組合せの「値段」を見て買う/買わないを決める** — 本プロジェクトで一度も測っていない軸。

★なぜこれが抜けているか
連系はこれまで**常に「モデル上位N点をベタ買い」**で、配当（＝その組合せの値段）を見ていない。
(62)は枠連スコア＝**確率**での絞りを試したが、確率が高い＝人気＝安い ので、
「安いものを確実に当てる」方向にしか動かない。**回収率は 確率×値段 で決まる**。
単勝では(45)が **EV＝確率×オッズ** で選んで機能した（EV≥1で83.3%・十分位が正しい順序）。
**その単勝で効いた選択規則を、連系では一度も適用していない**。

★なぜ今までできなかったか、どう解くか
連系は**当たった組合せの配当しか観測できない**（オッズ板が無い）。
そこで **市場の単勝オッズ → Harville → 組合せの市場確率 → 値段の推定** を作り、
**実際に観測できる的中時配当 46,917R で較正・検証**する。
値段が当てられるなら、買う前に全候補のEVが計算できる。

測るもの:
  ① 価格モデルの検算（推定した値段と実配当が合うか。合わなければ以降は無意味）
  ② ★EV十分位別の実測回収率 — (45)③の連系版。EVが高い組合せほど回収率が高いか
  ③ EVで買い目を選ぶ戦略 vs 現行（BOX上位4）の**対応ありの直接比較**
  ④ 市場の連系価格がHarvilleから系統的にずれているか（ずれ自体が独立した収益源になりうる）

実行: python3 ml/audit_combo_ev.py [シード数(既定3)]
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

PAYOUT = "data/payout/a.csv"
TOPK = 6          # 候補にする上位頭数（C(6,3)=20通り）
BUY = 4           # 1レースの購入点数（現行BOX4と同じコストに揃える）


def harville_top3(pw, combo):
    """win確率 pw（レース内合計1）のもとで、combo の3頭が1〜3着を占める確率（順不同）。"""
    tot = 0.0
    for a, b, c in itertools.permutations(combo):
        pa, pb, pc = pw[a], pw[b], pw[c]
        d1 = 1.0 - pa
        d2 = 1.0 - pa - pb
        if d1 <= 1e-12 or d2 <= 1e-12:
            continue
        tot += pa * (pb / d1) * (pc / d2)
    return tot


def boot(x, rng, n=2000):
    b = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return np.percentile(b, 2.5) * 100, np.percentile(b, 97.5) * 100


def main():
    n_seed = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    d, fx = load_cached()
    cut = d["date"].quantile(0.3)
    tr, te = (d["date"] < cut).to_numpy(), (d["date"] >= cut).to_numpy()
    fin = d["finish"].to_numpy()
    print(f"train {tr.sum():,} / test {te.sum():,}（分割 {cut.date()}）  シード{n_seed}本")

    # 順位付けは現行どおり top3 モデル。組合せ確率は win モデル（Harvilleはwin確率を要求する）
    pred = {}
    for tag, yv in [("top3", (fin <= 3).astype(int)), ("win", (fin == 1).astype(int))]:
        ps = [lgb.LGBMClassifier(random_state=s, **PARAMS)
              .fit(fx[tr], yv[tr], categorical_feature=F.CAT_COLS)
              .predict_proba(fx[te])[:, 1] for s in range(n_seed)]
        pred[tag] = np.mean(ps, axis=0)
        print(f"  学習完了 {tag}")

    sub = d.loc[te, ["raceid", "umaban", "fieldsize", "odds"]].copy()
    sub["p3"], sub["pw"] = pred["top3"], pred["win"]
    pa = load_payout_a(PAYOUT)

    rows = []          # 1行 = 1組合せ（候補20通りぶん）
    for rid, g in sub.groupby("raceid", sort=False):
        s3 = pa.get(rid)
        if not s3 or not s3["sanrenpuku"] or len(g) < 9:
            continue
        nums = g["umaban"].astype(int).to_numpy()
        pw = g["pw"].to_numpy(float)
        pw = pw / pw.sum()
        inv = 1.0 / g["odds"].to_numpy(float)
        mw = inv / inv.sum()
        pw_m = dict(zip(nums, pw))
        mw_m = dict(zip(nums, mw))
        top = g.sort_values("p3", ascending=False, kind="mergesort")["umaban"].astype(int).tolist()[:TOPK]
        cur = set(tuple(sorted(c)) for c in itertools.combinations(top[:4], 3))
        for combo in itertools.combinations(sorted(top), 3):
            pm = harville_top3(pw_m, combo)
            qm = harville_top3(mw_m, combo)
            rows.append({"raceid": rid, "combo": combo, "p_model": pm, "p_mkt": qm,
                         "pay": s3["sanrenpuku"].get(tuple(sorted(combo)), 0),
                         "cur": tuple(sorted(combo)) in cur})
    r = pd.DataFrame(rows)
    print(f"\n候補組合せ {len(r):,}件 / {r['raceid'].nunique():,}レース（上位{TOPK}頭のC({TOPK},3)）")

    # ===== ① 価格モデルの検算 =====
    hit = r[r["pay"] > 0].copy()
    hit["price"] = hit["pay"] / 100.0
    ok = hit["p_mkt"] > 1e-9
    hit = hit[ok]
    lx, ly = np.log(1.0 / hit["p_mkt"].to_numpy()), np.log(hit["price"].to_numpy())
    slope, intercept = np.polyfit(lx, ly, 1)
    corr = np.corrcoef(lx, ly)[0, 1]
    resid = ly - (slope * lx + intercept)
    print(f"\n[①価格モデルの検算] 的中組合せ {len(hit):,}件で log(実配当) ~ log(1/市場Harville確率)")
    print(f"  傾き {slope:.3f}（1.0なら価格は市場確率に正しく反比例）  相関 {corr:.3f}"
          f"  切片 exp={np.exp(intercept):.3f}（1−控除率=0.775 が理論値）")
    print(f"  残差の標準偏差 {resid.std():.3f}（対数）＝ 値段を実測の約 ×{np.exp(resid.std()):.2f}/÷ の幅でしか当てられない")
    r["price_hat"] = np.exp(intercept) / np.clip(r["p_mkt"].to_numpy(), 1e-9, None) ** slope
    r["ev"] = r["p_model"] * r["price_hat"]

    # ===== ④ 市場価格のHarvilleからのずれ =====
    print(f"\n[④市場の連系価格はHarvilleからずれているか] 的中組合せを市場確率の十分位で")
    hit2 = r[r["pay"] > 0].copy()
    hit2["dec"] = pd.qcut(hit2["p_mkt"], 10, labels=False, duplicates="drop")
    g4 = hit2.groupby("dec").apply(
        lambda x: pd.Series({"n": len(x), "実配当中央値": x["pay"].median(),
                             "理論配当中央値": (77.5 / x["p_mkt"]).median()}), include_groups=False)
    g4["実/理論"] = g4["実配当中央値"] / g4["理論配当中央値"]
    print(g4.round(2).to_string())

    # ===== ② EV十分位別の実測回収率（(45)③の連系版） =====
    print(f"\n[②EV十分位別の実測回収率] 全候補{len(r):,}件を推定EVで10等分")
    r["dec"] = pd.qcut(r["ev"], 10, labels=False, duplicates="drop")
    g2 = r.groupby("dec").apply(
        lambda x: pd.Series({"件数": len(x), "推定EV中央値": x["ev"].median(),
                             "的中率%": (x["pay"] > 0).mean() * 100,
                             "回収率%": x["pay"].mean()}), include_groups=False)
    print(g2.round(2).to_string())
    print("  ※(45)③の単勝では上位83.0%/下位55.5%と正しい順序が出た。ここで順序が出なければ"
          "「値段を見て選ぶ」は連系では機能しない。")

    # ===== ③ 戦略の対応ありの直接比較 =====
    print(f"\n[③戦略比較] 同じレースで買い比べる（1レース{BUY}点＝{BUY*100}円でコストを揃える）")
    strat = {}
    for rid, g in r.groupby("raceid", sort=False):
        cur = g[g["cur"]]
        by_ev = g.nlargest(BUY, "ev")
        by_p = g.nlargest(BUY, "p_model")
        strat[rid] = {"現行 BOX上位4": cur["pay"].sum() / (BUY * 100),
                      "EV上位4（値段を見る）": by_ev["pay"].sum() / (BUY * 100),
                      "モデル確率上位4（値段を見ない）": by_p["pay"].sum() / (BUY * 100)}
    sdf = pd.DataFrame(strat).T
    rng = np.random.default_rng(0)
    print(f"{'戦略':<32}{'R':>8}{'ROI':>9}{'的中率':>9}{'現行との差':>12}{'95%CI':>18}")
    base = sdf["現行 BOX上位4"].to_numpy()
    for c in sdf.columns:
        x = sdf[c].to_numpy()
        lo, hi = boot(x - base, rng)
        tail = "—" if c.startswith("現行") else f"{(x-base).mean()*100:>+10.2f}pt{f'[{lo:+.2f},{hi:+.2f}]':>18}"
        print(f"{c:<32}{len(x):>8,}{x.mean()*100:>8.2f}%{(x>0).mean()*100:>8.2f}%{tail}")

    print(f"\n[③b EVで「買うレースを選ぶ」] 現行BOX4のうち、レース内EV最大が閾値以上のレースだけ買う")
    evmax = r[r["cur"]].groupby("raceid")["ev"].max()
    cur = sdf["現行 BOX上位4"]
    j = pd.DataFrame({"x": cur, "evmax": evmax}).dropna()
    print(f"{'条件':<24}{'R':>8}{'構成比':>8}{'ROI':>9}")
    for q in [0.0, 0.2, 0.4, 0.6, 0.8]:
        th = j["evmax"].quantile(q)
        sel = j["evmax"] >= th
        print(f"{f'EV最大 上位{int((1-q)*100)}%':<24}{int(sel.sum()):>8,}{sel.mean()*100:>7.1f}%"
              f"{j.loc[sel,'x'].mean()*100:>8.2f}%")


if __name__ == "__main__":
    main()
