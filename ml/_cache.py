"""検証用の共通キャッシュ。DS*.CSV の読込＋特徴量生成は毎回1分以上かかるので、
一度だけ作って parquet に落としておく。検証スクリプトはここから読む（結果は一切変えない）。

  from _cache import load_cached
  d, fx, meta = load_cached()

d  … to_model + build_features の keep 後の整形テーブル
fx … encode_categoricals + log_odds/mkt_prob 済みの特徴量（train_prod.py と同一）
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "ml")
import features as F

CACHE = os.environ.get("KEIBA_CACHE", "/tmp/claude-0/-home-user-keiba-index-generator/"
                       "feed58cc-f116-4f79-9af3-bbd71d0225d9/scratchpad/cache")


def build():
    os.makedirs(CACHE, exist_ok=True)
    d = F.to_model(F.load_files())
    f = F.build_features(d)
    keep = (f["n_prior"] >= 1) & d["odds"].notna() & (d["odds"] > 0)
    d, f = d[keep].reset_index(drop=True), f[keep].reset_index(drop=True)
    fx, cat_maps = F.encode_categoricals(f)
    odds = d["odds"].to_numpy(float)
    inv = 1.0 / odds
    fx["log_odds"] = np.log(odds)
    fx["mkt_prob"] = inv / pd.Series(inv).groupby(d["raceid"]).transform("sum").to_numpy()
    d.to_pickle(f"{CACHE}/d.pkl")
    fx.to_pickle(f"{CACHE}/fx.pkl")
    return d, fx


def load_cached():
    if not os.path.exists(f"{CACHE}/fx.pkl"):
        return build()
    return pd.read_pickle(f"{CACHE}/d.pkl"), pd.read_pickle(f"{CACHE}/fx.pkl")


if __name__ == "__main__":
    d, fx = build()
    print(f"cached: {len(d):,}行 / {d['raceid'].nunique():,}レース "
          f"{d['date'].min().date()}〜{d['date'].max().date()} / 特徴{fx.shape[1]}列")
