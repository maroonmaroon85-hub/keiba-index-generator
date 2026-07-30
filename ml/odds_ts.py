"""
時系列オッズ(TARGET『時系列オッズ(フルCSV形式)→単勝』)の読み込み。

やり残しだった「前日オッズで運用した場合の劣化」を実測するための部品。
(59) `odds_sensitivity.py` は確定オッズにランダムノイズを乗せた**感度分析**でしかなく、
「直前に情報が集まって金が動く」という系統的な変化は再現できていなかった。これはその置き換え。

ファイル形式（1レース1ファイル・Shift_JIS・列名行あり）:
    レースID | 区分 | 月日時分 | 頭数 | 単勝票数 | 1番 | 2番 | … | N番
  ・レースID … 16桁 `YYYYMMDD` + 場(2) + 回(2) + 日(2) + R(2)
  ・区分     … 1=発売中 / 3=締切時点 / 4=確定
  ・月日時分 … `MMDDHHMM`（**前日から始まる**。実データは前日21:35〜当日の発走直後で147時点）
  ・各馬の列 … その時点の単勝オッズ（0/空 は出走取消等）

★結合キー: **ファイル名の先頭2文字"JT"を除いた8桁が、そのまま手元の `raceid`**。
  例 `JT01261201.CSV` → `01261201` = 場01(札幌) 年26 回1 日2 R01。
  日が10以上は A,B,… になる（`02261A01` 等）ので、ファイル名をそのまま使うのが安全。

実行（読み込み確認）: python3 ml/odds_ts.py [ディレクトリ(既定 data/odds_ts)]
"""
import csv
import glob
import io
import os
import sys

import numpy as np
import pandas as pd

DEFAULT_DIR = "data/odds_ts"
K_FINAL = "4"      # 区分4 = 確定
K_CLOSE = "3"      # 区分3 = 締切時点（発走時刻とみなす）


def _dt(year, mmddhhmm):
    """`MMDDHHMM` + 年 → Timestamp。年跨ぎ(12月→1月)は考慮不要な粒度なのでそのまま。"""
    return pd.Timestamp(f"{year}-{mmddhhmm[:2]}-{mmddhhmm[2:4]} {mmddhhmm[4:6]}:{mmddhhmm[6:8]}")


def load_race(path):
    """1ファイル → {raceid, date, post(発走), times, kubun, votes, odds(T×N), n}。"""
    txt = open(path, "rb").read().decode("shift_jis", "replace")
    rows = [r for r in csv.reader(io.StringIO(txt)) if r and r[0].strip()]
    if len(rows) < 2:
        return None
    body = rows[1:]
    rid16 = body[0][0].strip()
    year, date = rid16[:4], pd.Timestamp(f"{rid16[:4]}-{rid16[4:6]}-{rid16[6:8]}")
    n = int(body[0][3])
    times, kubun, votes, odds = [], [], [], []
    for r in body:
        times.append(_dt(year, r[2].strip()))
        kubun.append(r[1].strip())
        votes.append(int(r[4] or 0))
        odds.append([pd.to_numeric(r[5 + i], errors="coerce") if 5 + i < len(r) else np.nan
                     for i in range(n)])
    kubun = np.array(kubun)
    times = pd.DatetimeIndex(times)
    close = times[kubun == K_CLOSE]
    return {"raceid": os.path.basename(path)[2:10], "rid16": rid16, "date": date,
            "post": close[0] if len(close) else times[-1], "times": times, "kubun": kubun,
            "votes": np.array(votes), "odds": np.array(odds, dtype=float), "n": n}


def load_dir(d=DEFAULT_DIR):
    """ディレクトリ配下を全部読む → {raceid: レコード}。"""
    out = {}
    for p in sorted(glob.glob(os.path.join(d, "*.CSV")) + glob.glob(os.path.join(d, "*.csv"))):
        rec = load_race(p)
        if rec:
            out[rec["raceid"]] = rec
    return out


def odds_at(rec, when):
    """指定時点のオッズベクトル（長さ n）を返す。取れなければ None。

    when の指定:
      ("final",)            … 確定（区分4、無ければ最終行）
      ("prev", 21, 0)       … **前日**の21:00 時点で見えていた最後の値
      ("day", 9, 0)         … **当日**の09:00 時点で見えていた最後の値
      ("before", 10)        … 発走10分前の時点で見えていた最後の値
    いずれも「その時刻**以前**の最後のデータ」を取る（=実際に見られた値）。
    その時刻より前のデータが無ければ None＝「その時刻には買えなかった」。
    ★TARGETの`直後のデータ`のように後ろの値で埋めると、当日の情報が前日の列に混ざる。
    """
    kind = when[0]
    if kind == "final":
        idx = np.where(rec["kubun"] == K_FINAL)[0]
        return rec["odds"][idx[-1] if len(idx) else -1]
    if kind == "before":
        cut = rec["post"] - pd.Timedelta(minutes=when[1])
    else:
        day = rec["date"] - pd.Timedelta(days=1) if kind == "prev" else rec["date"]
        cut = day + pd.Timedelta(hours=when[1], minutes=when[2])
    ok = np.where(rec["times"] <= cut)[0]
    return rec["odds"][ok[-1]] if len(ok) else None


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DIR
    races = load_dir(d)
    if not races:
        sys.exit(f"{d} に時系列オッズCSVがありません")
    dates = sorted({r["date"] for r in races.values()})
    print(f"{len(races)}レース / {len(dates)}開催日 {dates[0].date()}〜{dates[-1].date()}")
    t = [len(r["times"]) for r in races.values()]
    starts = [r["times"][0] - r["date"] for r in races.values()]
    print(f"1レースあたり時点数 {min(t)}〜{max(t)}（中央値{int(np.median(t))}）")
    print(f"最初の時点は発走日の {min(starts)} 〜 {max(starts)}（負なら前日）\n")

    W = [("prev", 21, 0), ("day", 9, 0), ("before", 10), ("final",)]
    NAMES = ["前日21時", "当日9時", "発走10分前", "確定"]
    rows = []
    for rid, rec in races.items():
        vs = {nm: odds_at(rec, w) for nm, w in zip(NAMES, W)}
        fin = vs["確定"]
        for nm in NAMES[:-1]:
            v = vs[nm]
            if v is None or fin is None:
                rows.append({"t": nm, "ok": 0})
                continue
            m = np.isfinite(v) & np.isfinite(fin) & (v > 0) & (fin > 0)
            rows.append({"t": nm, "ok": 1, "n": int(m.sum()),
                         "fav_chg": int(np.argmin(np.where(m, v, np.inf))
                                        != np.argmin(np.where(m, fin, np.inf))),
                         "mad": float(np.median(np.abs(np.log(v[m] / fin[m])))),
                         "corr": float(np.corrcoef(np.log(v[m]), np.log(fin[m]))[0, 1])})
    df = pd.DataFrame(rows)
    print(f"{'時点':<12}{'取得できたR':>10}{'1番人気が確定と違う':>20}"
          f"{'オッズ乖離(中央値)':>20}{'log相関':>9}")
    for nm in NAMES[:-1]:
        s = df[df["t"] == nm]
        g = s[s["ok"] == 1]
        if g.empty:
            print(f"{nm:<12}{0:>10}")
            continue
        print(f"{nm:<12}{len(g):>7}/{len(s):<3}{g['fav_chg'].mean()*100:>18.1f}%"
              f"{g['mad'].mean()*100:>18.1f}%{g['corr'].mean():>9.3f}")
    print("\n※「1番人気が確定と違う」= その時点の最低オッズ馬と確定の最低オッズ馬が別の馬だったレースの割合。")
    print("　モデルの軸が入れ替わる率とは別物だが、前日オッズで買うことの影響の大きさの目安になる。")


if __name__ == "__main__":
    main()
