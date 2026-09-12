"""(243) ★★★★**穴馬の前向き運用モデルを1回だけ作って固定する** —— `wf_predict` の2026年ぶんと同じ作り方

★★**なぜ要るか（2026-09-12）**
　★**当日の推奨を出すには、走る前にモデルの予測が要る**。
　⚠**`wf_predict` はキャッシュ済みの過去行にしか予測を持たない**（`data/cache/wf_pred.npz`）。
　→ ★**同じ作り方のモデルを1つ作って保存し、当日はそれで予測する**。

■ ★★**「同じ作り方」の中身**（★`ml/audit_ana_marg.py:193` の `wf_predict` の2026年ぶん）
| | ★**合わせるもの** |
|---|---|
| **学習対象** | ★**`year < 2026` の全行**（**2026年は1行も学習に使わない**） |
| **絞り込み** | ★**`n_prior >= 1` かつ `odds` があり `> 0`**（**wf_predict の呼び出し元と同じ**） |
| **容量** | ★**`CAPACITY["l2"]`**（`ml/model_prod` と同じパラメータ） |
| **シード** | ★**3本平均**（`fit_seeds(..., 3)`） |
| **オッズ特徴** | ★**`log_odds` と `mkt_prob`**（`add_odds_features`） |

■ ★**過去走の読み込み**: **ルート直下の `*.CSV` ＋ `data/nk/DSnk*.CSV`**
　★**DSnk は全部2026年なので、`year < 2026` の学習には1行も入らない**。
　★**入れる理由は「今日走る馬の直近の成績」を特徴量に反映させるため**（`reco_ana.py` と同じ）。

■ ⚠★★**これは凍結値（113.4% / 128.8%）を引き継がない**
　★**引き継げない理由は4つ**（`ANA_RULE.md` §6）:
　　**1. qp が朝9時の板／2. 単勝が朝9時／★3. このモデル（固定）と wf_predict は別物／
　　4. 過去走に繋がらない馬が落ちる**。
　★**3を最小にするための script がこれ**——**1を0にはできないが、作り方は揃えられる**。

■ ★**保存形式は `ml/train_prod.py` と同じ**（`predict.load_model` がそのまま読める）
　`model_0..2.txt` / `cat_maps.json` / `feature_cols.json` / `meta.json`

実行: python3 ml/train_ana_fwd.py        自己テスト: python3 ml/train_ana_fwd.py --selftest
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features as F
from train_prod import CAPACITY, add_odds_features, fit_seeds

OUT_DIR = "ml/model_ana_fwd"
CUT_YEAR = 2026          # ★この年より前だけで学習する（＝wf_predict の2026年ぶん）
NSEED = 3
CAP = "l2"


def load_panel():
    """★過去走のパネル: ルート直下の `*.CSV` ＋ `data/nk/DSnk*.CSV`（`reco_ana.py` と同じ）。"""
    ds = [p for p in sorted(glob.glob("data/nk/DSnk*.CSV")) if os.path.getsize(p) > 0]
    frames = [F.load_files()] + [
        pd.read_csv(p, header=None, encoding="shift_jis", encoding_errors="replace",
                    dtype=str, keep_default_na=False) for p in ds]
    return F.to_model(pd.concat(frames, ignore_index=True)), len(ds)


def selftest():
    ok = True
    print(f"★★作るもの: **{OUT_DIR}/**（`predict.load_model` が読める形）")
    print(f"★★学習対象: **year < {CUT_YEAR}**（★{CUT_YEAR}年は1行も使わない）"
          f"／容量 **{CAP.upper()}**／シード **{NSEED}本**")
    md, par = CAPACITY[CAP]
    print(f"　`CAPACITY['{CAP}']` = {md} / num_leaves={par.get('num_leaves')} "
          f"n_estimators={par.get('n_estimators')}")
    ok &= md == "ml/model_prod" and OUT_DIR != md
    print(f"　★**保存先が `ml/model_prod` と別**（枠連のモデルを上書きしない） "
          f"{'★OK' if ok else '⚠NG'}")
    ds = [p for p in sorted(glob.glob("data/nk/DSnk*.CSV")) if os.path.getsize(p) > 0]
    print(f"★過去走: ルート直下の *.CSV ＋ data/nk/DSnk*.CSV **{len(ds)}本**")
    print(f"⚠★**これは凍結値を引き継がない**（ANA_RULE.md §6 の4つの理由）")
    print("★自己テスト: " + ("全部OK" if ok else "⚠NG"))
    return 0 if ok else 1


def main():
    print("(243) ★★★★**穴馬の前向き運用モデルを固定する**\n")
    d, nds = load_panel()
    print(f"★過去走 {len(d):,}行（ルート *.CSV ＋ DSnk {nds}本）"
          f"　{d['date'].min().date()}〜{d['date'].max().date()}")
    f = F.build_features(d)
    keep = (f["n_prior"] >= 1) & d["odds"].notna() & (d["odds"] > 0)
    d, f = d[keep].reset_index(drop=True), f[keep].reset_index(drop=True)
    y = (d["finish"] <= 3).astype(int).to_numpy()
    fx, cat_maps = F.encode_categoricals(f)
    fx = add_odds_features(fx, d["odds"].to_numpy(float), d["raceid"].to_numpy())
    cols = list(fx.columns)

    yr = d["date"].dt.year.to_numpy()
    tr = yr < CUT_YEAR
    print(f"★★学習 **{int(tr.sum()):,}行**（year < {CUT_YEAR}）"
          f"／⚠**{CUT_YEAR}年の {int((~tr).sum()):,}行は使わない**"
          f"　top3率 {y[tr].mean():.3f}")
    if int(tr.sum()) < 100000:
        print("⚠⚠**学習行が少なすぎる。読み込みを疑うこと**")
        return 1

    _, par = CAPACITY[CAP]
    print(f"★学習中（容量 {CAP.upper()} × シード{NSEED}本）…")
    ms = fit_seeds(fx[tr], y[tr], NSEED, par)

    os.makedirs(OUT_DIR, exist_ok=True)
    for i, m in enumerate(ms):
        m.booster_.save_model(f"{OUT_DIR}/model_{i}.txt")
    json.dump(cat_maps, open(f"{OUT_DIR}/cat_maps.json", "w"), ensure_ascii=False)
    json.dump(cols, open(f"{OUT_DIR}/feature_cols.json", "w"), ensure_ascii=False)
    meta = {
        "purpose": "穴馬の前向き運用（当日の推奨）専用。wf_predict の2026年ぶんと同じ作り方",
        "target": "top3 (finish<=3)",
        "odds_features": ["log_odds", "mkt_prob"],
        "train_filter": f"year < {CUT_YEAR} かつ n_prior>=1 かつ odds>0",
        "capacity": CAP, "params": par, "n_seed": NSEED,
        "models": [f"model_{i}.txt" for i in range(NSEED)],
        "rows_train": int(tr.sum()),
        "date_from": str(d.loc[tr, "date"].min().date()),
        "date_to": str(d.loc[tr, "date"].max().date()),
        "panel_rows": int(len(d)),
        "note": ("⚠これは凍結値(113.4%/128.8%)を引き継がない。"
                 "wf_predict と同じ作り方だが同じ予測ではない（ANA_RULE.md §6）"),
    }
    json.dump(meta, open(f"{OUT_DIR}/meta.json", "w"), ensure_ascii=False, indent=1)
    print(f"★保存: **{OUT_DIR}/**（model_0..{NSEED-1}.txt / cat_maps.json / "
          f"feature_cols.json / meta.json）")
    print(f"　学習 {meta['date_from']}〜{meta['date_to']} / {meta['rows_train']:,}行")
    print("⚠**枠連の `ml/model_prod` には触れていない**")
    return 0


if __name__ == "__main__":
    sys.exit(selftest() if "--selftest" in sys.argv else main())
