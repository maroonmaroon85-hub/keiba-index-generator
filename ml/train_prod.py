"""
実運用モデルの学習（predict.py が読む ml/model_prod/ を作る）。

`train.py` が作る `ml/model/` は「**オッズなし・win目標**」で、これは(45)以降の推奨構成とは**別物**。
実運用に使うのは検証で最良だった構成:
  ・目標  … **top3**(finish≤3)。枠連も三連複も上位3頭の並びで決まるため((48)(55))
  ・特徴  … build_features の全部 ＋ **log_odds** ＋ **mkt_prob**（1/oddsをレース内で正規化）((45))
  ・パラメータ … place_wide.PARAMS。AUC最適化はROIを下げるので**触らない**((52)(63))
  ・カテゴリ … 父/母父/騎手/調教師も**残す**。外すとAUCは上がるがROIは下がる((63))
  ・シード平均 … シード差だけでROIが数pt動く((30))ので**複数シードの平均**を保存する

出力: ml/model_prod/
  model_0.txt … model_{n-1}.txt  シードごとの学習済みモデル（予測は平均を使う）
  cat_maps.json / feature_cols.json
  meta.json  … 構成の記録＋**枠連スコアの下位20%閾値**（(62)の唯一使える絞り込み）

手順:
  1. (55)(62)と同じ日付分割（前30%で学習／後70%で検証）でOOSを測り、HANDOFFの数値
     （枠連 軸枠×紐枠2 ≒84.5% / 三連複 BOX上位4 ≒84.5%）が再現するか確認する
  2. 同じ検証期間で枠連スコア（積ベース）の20パーセンタイルを求め、除外閾値として保存
  3. **全期間で再学習**して保存（実運用では手元の全データを使う）

実行: python3 ml/train_prod.py [シード数(既定3)] [容量 l2|l5(既定 l2)]
  l5 は (81)(83) の高容量。**並行運用のために作るもので、既定は l2 のまま**。
  保存先が ml/model_prod / ml/model_prod_l5 に分かれるので共存できる。
"""
import itertools
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
sys.path.insert(0, "ml")
import features as F
from place_wide import PARAMS
from pocket_eval import load_payout_a
from waku_umatan import bracket_probs, load_wu, waku_score, wakuren_buy

MODEL_DIR = "ml/model_prod"
PAYOUT = "data/payout/a.csv"

# ★容量の選択肢。(81)(83)で「容量を上げるとAUCは下がるがROIは上がる」が5段の梯子で確認された。
#   ただし差は5シードで縮んでおりCIは0を跨ぐため、**既定は現行(L2)のまま**。
#   L5は並行運用（予測を両方記録して実運用で見比べる）用に作れるようにしてある。
#   ⚠L5は学習が10〜15倍重い（2分 → 20〜30分）。
CAPACITY = {
    "l2": ("ml/model_prod", dict(PARAMS)),
    "l5": ("ml/model_prod_l5", dict(PARAMS, num_leaves=255, min_child_samples=10,
                                    n_estimators=2000)),
}


def add_odds_features(fx, odds, raceid):
    """(45)のオッズ特徴。log_odds＝生オッズの対数、mkt_prob＝1/oddsをレース内で正規化した市場確率。"""
    inv = 1.0 / odds
    mkt = inv / pd.Series(inv).groupby(np.asarray(raceid)).transform("sum").to_numpy()
    fx = fx.copy()
    fx["log_odds"] = np.log(odds)
    fx["mkt_prob"] = mkt
    return fx


def fit_seeds(fx, y, n_seed, par=None):
    par = PARAMS if par is None else par
    return [lgb.LGBMClassifier(random_state=s, **par).fit(fx, y, categorical_feature=F.CAT_COLS)
            for s in range(n_seed)]


def main():
    n_seed = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    cap = (sys.argv[2] if len(sys.argv) > 2 else "l2").lower()
    if cap not in CAPACITY:
        sys.exit(f"容量は {'/'.join(CAPACITY)} のいずれか（既定 l2）")
    MODEL_DIR, PAR = CAPACITY[cap]
    if cap != "l2":
        print(f"★容量 {cap.upper()} で学習する（{PAR['num_leaves']}葉 × "
              f"{PAR['n_estimators']}本）。現行(L2)より10〜15倍重い")
    d = F.to_model(F.load_files())
    f = F.build_features(d)
    keep = (f["n_prior"] >= 1) & d["odds"].notna() & (d["odds"] > 0)
    d, f = d[keep].reset_index(drop=True), f[keep].reset_index(drop=True)
    y = (d["finish"] <= 3).astype(int).to_numpy()
    fx, cat_maps = F.encode_categoricals(f)
    fx = add_odds_features(fx, d["odds"].to_numpy(float), d["raceid"].to_numpy())
    cols = list(fx.columns)
    print(f"全体 {len(d):,}行 / {d['raceid'].nunique():,}レース  "
          f"{d['date'].min().date()}〜{d['date'].max().date()}  top3率 {y.mean():.3f}")

    # ===== 1. OOS検証（(55)(62)と同じ前30%学習・後70%検証） =====
    cut = d["date"].quantile(0.3)
    tr, te = (d["date"] < cut).to_numpy(), (d["date"] >= cut).to_numpy()
    print(f"\n[1/3] OOS検証: train {tr.sum():,} / test {te.sum():,}（分割日 {cut.date()}）")
    ms = fit_seeds(fx[tr], y[tr], n_seed, PAR)
    p = np.mean([m.predict_proba(fx[te])[:, 1] for m in ms], axis=0)
    print(f"  AUC {roc_auc_score(y[te], p):.4f}（シード{n_seed}本平均）")

    sub = d.loc[te, ["raceid", "umaban", "fieldsize"]].copy()
    sub["p"] = p
    wu, pa = load_wu(PAYOUT), load_payout_a(PAYOUT)
    rows_w, rows_s, scores = [], [], []
    for rid, g in sub.groupby("raceid", sort=False):
        gg = g.sort_values("p", ascending=False, kind="mergesort")
        nums = gg["umaban"].astype(int).tolist()
        n = int(gg["fieldsize"].iloc[0])
        w = wu.get(rid)
        if w and w["wakuren"] and len(nums) >= 3:
            cs = wakuren_buy(nums, n, 2)
            sc = waku_score(cs, bracket_probs(nums, gg["p"].to_numpy(), n))
            scores.append(sc)
            rows_w.append({"pay": sum(w["wakuren"].get(c, 0) for c in cs), "k": len(cs), "sc": sc})
        s = pa.get(rid)
        if s and s["sanrenpuku"] and len(nums) >= 9:
            cs = [tuple(sorted(c)) for c in itertools.combinations(nums[:4], 3)]
            rows_s.append({"pay": sum(s["sanrenpuku"].get(c, 0) for c in cs), "k": 4})

    oos = {}
    for name, rr, key in [("枠連 軸枠×紐枠2", rows_w, "wakuren2"), ("三連複 BOX上位4", rows_s, "sanrenpuku_box4")]:
        df = pd.DataFrame(rr)
        x = (df["pay"] / (df["k"] * 100)).to_numpy(float)
        oos[key] = {"races": int(len(x)), "roi": round(float(x.mean()) * 100, 2),
                    "hit": round(float((x > 0).mean()) * 100, 2)}
        print(f"  {name:<18}{len(x):>7,}R  的中{(x>0).mean()*100:5.2f}%  ROI {x.mean()*100:5.1f}%"
              f"  （HANDOFF記載 84.5%）")

    # ===== 2. 枠連スコアの下位20%閾値（(62): 絞りは「除外」にしか使えない） =====
    p20 = float(np.percentile(scores, 20))
    dfw = pd.DataFrame(rows_w)
    xw = (dfw["pay"] / (dfw["k"] * 100)).to_numpy(float)
    hi = dfw["sc"].to_numpy() >= p20
    print(f"\n[2/3] 枠連スコア下位20%閾値 = {p20:.4f}"
          f"  → 除外前 {xw.mean()*100:.1f}% / 除外後 {xw[hi].mean()*100:.1f}%"
          f"（下位20%だけ {xw[~hi].mean()*100:.1f}%）")
    oos["wakuren2_excl_bottom20"] = round(float(xw[hi].mean()) * 100, 2)

    # ===== 3. 全期間で再学習して保存 =====
    print(f"\n[3/3] 全期間 {len(d):,}行で再学習して保存（シード{n_seed}本）")
    ms_all = fit_seeds(fx, y, n_seed, PAR)
    os.makedirs(MODEL_DIR, exist_ok=True)
    for i, m in enumerate(ms_all):
        m.booster_.save_model(f"{MODEL_DIR}/model_{i}.txt")
    json.dump(cat_maps, open(f"{MODEL_DIR}/cat_maps.json", "w"), ensure_ascii=False)
    json.dump(cols, open(f"{MODEL_DIR}/feature_cols.json", "w"), ensure_ascii=False)
    meta = {
        "target": "top3 (finish<=3)",
        "odds_features": ["log_odds", "mkt_prob"],
        "params": PAR,
        "capacity": cap,
        "n_seed": n_seed,
        "models": [f"model_{i}.txt" for i in range(n_seed)],
        "rows": int(len(d)),
        "races": int(d["raceid"].nunique()),
        "date_from": str(d["date"].min().date()),
        "date_to": str(d["date"].max().date()),
        "waku_score_p20": p20,
        "oos": oos,
        "note": "枠連スコア閾値は前30%学習モデルのOOS分布から算出。保存モデルは全期間学習なので厳密には別分布。",
    }
    json.dump(meta, open(f"{MODEL_DIR}/meta.json", "w"), ensure_ascii=False, indent=1)
    print(f"保存: {MODEL_DIR}/ (model_0..{n_seed-1}.txt, cat_maps.json, feature_cols.json, meta.json)")


if __name__ == "__main__":
    main()
