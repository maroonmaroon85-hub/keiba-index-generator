"""TARGETの「**指定**時系列オッズ(CSV形式)」（枠連・馬連）を読む。(148)のための部品。

★実データで形式を確定させた（2026-08-14）。**1ファイルに全レースが入る**（フル形式とは違う）:
```
レースID,区分,月日時分,頭数,<券種>票数,枠1-1,枠1-2,…,枠8-8      ← 枠連は36組
2026072501010101,1,07242058,8,0,0.0,0.0,…                      ← 前日20:58
2026072501010101,1,07250901,8,…                                 ← 当日09:01
2026072501010101,1,07250949,8,…                                 ← ★発走11分前
2026072501010101,4,07251005,8,…                                 ← 確定（区分4）
```
・レースID … 16桁 `YYYYMMDD` + 場(2) + 回(2) + 日(2) + R(2)
・**区分 1=発売中 / 4=確定**。実測では **1が3行・4が1行＝1レース4時点**。
・★**「最後の区分1」は発走の11分前でほぼ一定**（実測36レースで発走−11分）。
　→ **(148)の「10分前」はこのスナップを使う**。
・**行は時刻順に並んでいない**ので、**読んでから並べ替える**こと。
・**8頭以下は枠連が発売されないので全0**。それが正常（実測: 9頭以上30レースは全て値あり）。

★手元の8桁raceidへの変換: `場(2) + 年下2桁(2) + 回(1桁) + 日(1桁) + R(2)`
　例 `2026072501010101` → `01261101`（実測36レース中33本が既存データと一致）。

⚠★**Macには pandas も numpy も無い**。**このファイルは標準ライブラリだけで動く**
　（2026-08-13にここで一度落とした）。解析側（(148)）がクラウドで numpy に載せ替える。

使い方:
    python3 ml/odds_ts_combo.py data/odds_ts_waku/ts_waku.csv    # ★Macで実行可
"""
import csv
import glob
import io
import os
import re
import sys
from datetime import datetime

K_FINAL = "4"
K_SALE = "1"
FIXED = 5          # レースID・区分・月日時分・頭数・票数
LABELS = ("前日", "当日朝", "直前", "確定")


def _key(cell):
    """ヘッダのセル（`枠1-2` / `馬連1-2` など）→ (a, b)。取れなければ None。"""
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


def _dt(year, s):
    return datetime(int(year), int(s[:2]), int(s[2:4]), int(s[4:6]), int(s[6:8]))


def rid8_of(rid16):
    """16桁レースID → 手元の8桁raceid（場+年下2桁+回+日+R）。"""
    if len(rid16) != 16 or not rid16.isdigit():
        return None
    y, jyo, kai, hi, rr = rid16[2:4], rid16[8:10], rid16[10:12], rid16[12:14], rid16[14:16]
    return f"{jyo}{y}{int(kai)}{int(hi)}{rr}"


def load_file(path):
    """1ファイル → {rid8: {"n":頭数, "snaps":{ラベル: (時刻, {組:オッズ})}}}。

    ★ラベルは **前日 / 当日朝 / 直前 / 確定**。
    　「直前」＝**区分1の最後**（実測で発走−11分）。「前日」＝区分1の最初。
    　「当日朝」＝区分1の最後から1つ前。区分1が3行に満たなければ入らない。
    """
    txt = open(path, "rb").read().decode("shift_jis", "replace")
    rows = [r for r in csv.reader(io.StringIO(txt)) if r and r[0].strip()]
    if len(rows) < 2:
        return {}
    keys = [_key(c) for c in rows[0][FIXED:]]
    if not any(k is not None for k in keys):
        return {}
    per = {}
    for r in rows[1:]:
        if len(r) < FIXED + 1 or not r[0].strip().isdigit():
            continue
        rid8 = rid8_of(r[0].strip())
        if not rid8:
            continue
        vals = {}
        for i, k in enumerate(keys):
            if k is None or FIXED + i >= len(r):
                continue
            try:
                v = float(r[FIXED + i])
            except ValueError:
                continue
            if v > 0:
                vals[k] = v
        per.setdefault(rid8, {"n": int(r[3] or 0), "rows": []})["rows"].append(
            (r[1].strip(), _dt(r[0][:4], r[2].strip()), vals))
    out = {}
    for rid8, d in per.items():
        rs = sorted(d["rows"], key=lambda x: x[1])
        sale = [(t, v) for k, t, v in rs if k == K_SALE]
        fin = [(t, v) for k, t, v in rs if k == K_FINAL]
        snaps = {}
        if sale:
            snaps["前日"] = sale[0]
            snaps["直前"] = sale[-1]
            if len(sale) >= 3:
                snaps["当日朝"] = sale[-2]
        if fin:
            snaps["確定"] = fin[-1]
        if snaps:
            out[rid8] = {"n": d["n"], "snaps": snaps}
    return out


def load_dir(d):
    """ディレクトリでもファイルでも受ける。複数ファイルはマージする。"""
    if os.path.isfile(d):
        return load_file(d)
    out = {}
    for p in sorted(glob.glob(os.path.join(d, "*.CSV")) + glob.glob(os.path.join(d, "*.csv"))):
        out.update(load_file(p))
    return out


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "data/odds_ts_waku"
    rec = load_dir(d)
    if not rec:
        sys.exit(f"{d} が読めない。TARGETの「指定時系列オッズ(CSV形式)」で出したCSVを置くこと。")
    print(f"{d}: {len(rec):,} レース")
    ns = sorted(r["n"] for r in rec.values())
    print(f"　頭数の中央値 {ns[len(ns)//2]}　（枠連は9頭以上でしか発売されない）")
    for lab in LABELS:
        have = [r for r in rec.values() if lab in r["snaps"]]
        wv = [r for r in have if r["snaps"][lab][1]]
        print(f"　{lab:<4} {len(have):>4}レースに存在 / うちオッズあり {len(wv):>4}"
              + (f"　例 {list(list(wv[0]['snaps'][lab][1].items())[:3])}" if wv else ""))
    k = sorted(rec)[0]
    print(f"\n例 {k}（{rec[k]['n']}頭）:")
    for lab in LABELS:
        s = rec[k]["snaps"].get(lab)
        if s:
            print(f"  {lab:<4} {s[0]:%m-%d %H:%M}  組数 {len(s[1])}")


if __name__ == "__main__":
    main()
