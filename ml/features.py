"""
共有の特徴量エンジニアリング（train.py と将来の predict.py で同一ロジックを使う）。
DS*.CSV（全馬成績フルセット+単オッズ, Shift_JIS, 52列）→ 特徴量DataFrame。
特徴は「発走前に見える情報」＝近走(prior runs)＋条件のみ。単勝オッズは特徴に入れない
（市場追随回避、EV評価にだけ使う）。
"""
import glob
import numpy as np
import pandas as pd

CAT_COLS = ["sex", "cond", "course", "sire", "damsire"]

def load_files(pattern="DS*.CSV"):
    frames = []
    for f in sorted(glob.glob(pattern)):
        frames.append(pd.read_csv(f, header=None, encoding="shift_jis", dtype=str, keep_default_na=False))
    return pd.concat(frames, ignore_index=True)

def to_model(raw):
    """生CSV → 1行1(馬,レース) の整形テーブル（重複除去・派生列つき）。"""
    raw = raw[raw[40].str.len() > 2]
    d = pd.DataFrame()
    d["date"] = pd.to_datetime("20" + raw[0].str.zfill(2) + raw[1].str.zfill(2) + raw[2].str.zfill(2), format="%Y%m%d", errors="coerce")
    d["course"] = raw[4].str.strip()
    d["surface"] = (raw[9].str.strip() == "ダ").astype(int)
    d["distance"] = pd.to_numeric(raw[11], errors="coerce")
    d["cond"] = raw[12].str.strip()
    d["horse"] = raw[37].str.strip()
    d["sex"] = raw[14].str.strip()
    d["age"] = pd.to_numeric(raw[15], errors="coerce")
    d["wtcarry"] = pd.to_numeric(raw[17], errors="coerce")
    d["fieldsize"] = pd.to_numeric(raw[18], errors="coerce")
    d["finish"] = pd.to_numeric(raw[20], errors="coerce")
    d["margin"] = pd.to_numeric(raw[23], errors="coerce")
    p = raw[[28, 29, 30, 31]].apply(pd.to_numeric, errors="coerce").where(lambda x: x > 0)
    d["passavg"] = p.mean(axis=1)
    d["agari"] = pd.to_numeric(raw[32], errors="coerce")
    d["prize"] = pd.to_numeric(raw[36], errors="coerce")
    d["raceid"] = raw[40].str[:-2]
    d["umaban"] = pd.to_numeric(raw[40].str[-2:], errors="coerce")
    d["sire"] = raw[43].str.strip()
    d["damsire"] = raw[45].str.strip()
    d["odds"] = pd.to_numeric(raw[48], errors="coerce")
    d = d.dropna(subset=["date", "distance", "finish", "fieldsize", "horse"])
    d = d.drop_duplicates(subset=["raceid", "horse"])
    d["finratio"] = (d["fieldsize"] - d["finish"] + 1) / d["fieldsize"]
    d["passratio"] = d["passavg"] / d["fieldsize"]
    return d.sort_values(["horse", "date"]).reset_index(drop=True)

def build_features(d):
    """整形テーブル → 特徴量(数値＋カテゴリは文字列のまま)。近走はshift(prior)のみ。"""
    g = d.groupby("horse", sort=False)
    f = pd.DataFrame(index=d.index)
    f["n_prior"] = g.cumcount()
    for k in (1, 2, 3):
        f[f"last{k}_fin"] = g["finratio"].shift(k)
    f["avg3_fin"] = g["finratio"].shift(1).rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
    f["best3_fin"] = g["finratio"].shift(1).rolling(3, min_periods=1).max().reset_index(level=0, drop=True)
    f["last_margin"] = g["margin"].shift(1)
    f["last_passratio"] = g["passratio"].shift(1)
    f["avg3_passratio"] = g["passratio"].shift(1).rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
    f["last_agari"] = g["agari"].shift(1)
    f["best3_agari"] = g["agari"].shift(1).rolling(3, min_periods=1).min().reset_index(level=0, drop=True)
    f["avg3_prize"] = g["prize"].shift(1).rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
    f["fin_std"] = g["finratio"].shift(1).rolling(3, min_periods=2).std().reset_index(level=0, drop=True)
    last_dist = g["distance"].shift(1)
    f["dist_change"] = d["distance"] - last_dist
    f["same_dist"] = (d["distance"] == last_dist).astype(float)
    f["surf_change"] = (d["surface"] != g["surface"].shift(1)).astype(float)
    f["wt_change"] = d["wtcarry"] - g["wtcarry"].shift(1)
    f["weeks_since"] = (d["date"] - g["date"].shift(1)).dt.days / 7.0
    f["weeks_before"] = (g["date"].shift(1) - g["date"].shift(2)).dt.days / 7.0
    f["month"] = d["date"].dt.month
    f["distance"] = d["distance"]
    f["surface"] = d["surface"]
    f["fieldsize"] = d["fieldsize"]
    f["wtcarry"] = d["wtcarry"]
    f["age"] = d["age"]
    for c in CAT_COLS:
        f[c] = d[c].astype(str)
    return f

def encode_categoricals(f, maps=None):
    """カテゴリ列を整数コード化。maps未指定なら学習(このfから作成)、指定なら適用（未知=-1）。"""
    f = f.copy()
    out_maps = {}
    for c in CAT_COLS:
        if maps is None:
            uniq = {v: i for i, v in enumerate(sorted(f[c].dropna().unique()))}
            out_maps[c] = uniq
        else:
            uniq = maps[c]
        f[c] = f[c].map(uniq).fillna(-1).astype(int)
    return (f, out_maps) if maps is None else f
