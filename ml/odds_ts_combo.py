"""TARGETの時系列オッズ（**枠連・馬連**）を読む。単勝用 `odds_ts.py` の組券版。

★これは (148) のための部品。**まず1ファイル見てから確定させる**方針なので、
　**列の意味を決め打ちせず、ヘッダ行から組キーを起こす**作りにしてある。

想定している形（単勝版 `odds_ts.py` と同じ枠組み・Shift_JIS・列名行あり）:
```
レースID | 区分 | 月日時分 | 頭数 | 票数 | <組1> | <組2> | … | <組K>
```
・レースID … 16桁 `YYYYMMDD` + 場(2) + 回(2) + 日(2) + R(2)
・区分     … 1=発売中 / 3=締切時点 / 4=確定
・月日時分 … `MMDDHHMM`
・**組の列名**は `1-2` `12` `1－2` などの表記ゆれがありうるので、
　**ヘッダから数字だけ取り出して2つに割る**（`1-2`→(1,2) / `0102`→(1,2)）。

★結合キーはファイル名の先頭2文字を除いた8桁（単勝版と同じ規則）。

使い方:
    python3 ml/odds_ts_combo.py data/odds_ts_waku      # ★まず中身を見る（列の解釈を表示）
"""
import csv
import glob
import io
import os
import re
import sys

import numpy as np
import pandas as pd

K_FINAL = "4"
K_CLOSE = "3"
FIXED = 5          # レースID・区分・月日時分・頭数・票数


def _key(cell):
    """ヘッダのセル → (a, b)。取れなければ None。"""
    ds = re.findall(r"\d+", str(cell))
    if not ds:
        return None
    s = "".join(ds)
    if len(ds) == 2:
        a, b = int(ds[0]), int(ds[1])
    elif len(s) == 4:
        a, b = int(s[:2]), int(s[2:])
    elif len(s) == 2:
        a, b = int(s[0]), int(s[1])
    else:
        return None
    return (a, b) if a <= b else (b, a)


def _dt(year, mmddhhmm):
    return pd.Timestamp(f"{year}-{mmddhhmm[:2]}-{mmddhhmm[2:4]} {mmddhhmm[4:6]}:{mmddhhmm[6:8]}")


def load_race(path):
    """1ファイル → {raceid, date, post, times, kubun, keys, odds(T×K)}。読めなければ None。"""
    txt = open(path, "rb").read().decode("shift_jis", "replace")
    rows = [r for r in csv.reader(io.StringIO(txt)) if r and r[0].strip()]
    if len(rows) < 2:
        return None
    head, body = rows[0], rows[1:]
    keys = [_key(c) for c in head[FIXED:]]
    if not any(k is not None for k in keys):
        return None
    rid16 = body[0][0].strip()
    year = rid16[:4]
    times, kubun, odds = [], [], []
    for r in body:
        if len(r) < FIXED + 1:
            continue
        times.append(_dt(year, r[2].strip()))
        kubun.append(r[1].strip())
        odds.append([pd.to_numeric(r[FIXED + i], errors="coerce")
                     if FIXED + i < len(r) else np.nan for i in range(len(keys))])
    if not times:
        return None
    kubun = np.array(kubun)
    times = pd.DatetimeIndex(times)
    close = times[kubun == K_CLOSE]
    return {"raceid": os.path.basename(path)[2:10], "rid16": rid16,
            "date": pd.Timestamp(f"{rid16[:4]}-{rid16[4:6]}-{rid16[6:8]}"),
            "post": close[-1] if len(close) else times[-1],
            "times": times, "kubun": kubun, "keys": keys,
            "odds": np.array(odds, dtype=float)}


def load_dir(d):
    out = {}
    for p in sorted(glob.glob(os.path.join(d, "*.CSV")) + glob.glob(os.path.join(d, "*.csv"))):
        rec = load_race(p)
        if rec:
            out[rec["raceid"]] = rec
    return out


def odds_at(rec, when):
    """`odds_ts.odds_at` と同じ規則。→ {組キー: オッズ} または None。"""
    if when[0] == "final":
        idx = np.where(rec["kubun"] == K_FINAL)[0]
        i = idx[-1] if len(idx) else -1
    else:
        if when[0] == "before":
            cut = rec["post"] - pd.Timedelta(minutes=when[1])
        else:
            day = rec["date"] - pd.Timedelta(days=1) if when[0] == "prev" else rec["date"]
            cut = day + pd.Timedelta(hours=when[1], minutes=when[2])
        idx = np.where(rec["times"] <= cut)[0]
        if len(idx) == 0:
            return None
        i = idx[-1]
    row = rec["odds"][i]
    return {k: float(v) for k, v in zip(rec["keys"], row)
            if k is not None and np.isfinite(v) and v > 0}


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "data/odds_ts_waku"
    fs = sorted(glob.glob(os.path.join(d, "*.CSV")) + glob.glob(os.path.join(d, "*.csv")))
    if not fs:
        sys.exit(f"{d} にCSVが無い。TARGETから出して置くこと。")
    print(f"{d}: {len(fs)} ファイル")
    txt = open(fs[0], "rb").read().decode("shift_jis", "replace")
    rows = [r for r in csv.reader(io.StringIO(txt)) if r and r[0].strip()]
    print(f"\n★1ファイル目 {os.path.basename(fs[0])} のヘッダ（先頭12列）:")
    print("  ", rows[0][:12])
    print(f"  列数 {len(rows[0])} / 行数 {len(rows)-1}")
    print(f"\n★組キーとして解釈できた列: ", end="")
    keys = [_key(c) for c in rows[0][FIXED:]]
    ok = [k for k in keys if k is not None]
    print(f"{len(ok)}/{len(keys)}   例: {ok[:8]}")
    rec = load_race(fs[0])
    if rec is None:
        sys.exit("⚠パースできなかった。ヘッダを見て `_key` を直すこと。")
    print(f"\n★時点数 {len(rec['times'])} / 発走(区分3) {rec['post']}")
    for w in (("prev", 21, 0), ("before", 30), ("before", 10), ("final",)):
        o = odds_at(rec, w)
        print(f"  {str(w):<20} 組数 {len(o) if o else 0}"
              + (f"  例 {list(o.items())[:3]}" if o else ""))


if __name__ == "__main__":
    main()
