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

★★「全オッズ」を1ファイルに出す場合（2026-08-14に対応）
　単勝・複勝・枠連・馬連・馬単・三連系が**同じファイル**に入る。
　⚠**`枠1-2` も `馬01-02` も組は (1,2)** なので、**券種で分けないと衝突する**。
　→ `_key()` が**列名から数字を除いた接頭辞**を券種として返し、`load_file(path, kind)` で選ぶ。
　⚠**「馬」を素朴に部分一致させると馬単にも当たる**ので `ALIAS` で優先順に完全一致させる。
　★`load_pool("枠")` / `load_pool("馬")` を呼べば、**置き場所も券種の並びも気にしなくてよい**
　　（`data/odds_ts_all/` `data/odds_ts_waku/` `data/odds_ts_umaren/` を全部見る）。
　★**列名は `枠1-2`/`馬01-02` 版と `枠連1-2`/`馬連01-02` 版の両方でテスト済み**（合成データ）。

使い方:
    python3 ml/odds_ts_combo.py data/odds_ts_waku/ts_waku.csv    # ★Macで実行可
    python3 ml/odds_ts_combo.py data/odds_ts_all 枠              # 全オッズ1ファイルのとき
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
# ★券種の別名（優先順）。実測の列名は `枠1-1` / `馬01-02` だが「全オッズ」1ファイルだと
#   `枠連〜` `馬連〜` の版もありうる。⚠**「馬」だけだと馬単にも当たる**ので順序が要る。
ALIAS = {"枠": ("枠連", "枠"), "馬": ("馬連", "馬")}


def _key(cell):
    """ヘッダのセル（`枠1-2` / `馬01-02` など）→ (種別, (a, b))。取れなければ None。

    ★★**種別＝数字を取り除いた接頭辞**（`枠` / `馬` など）。
    　⚠**「全オッズ」を1ファイルに出すと枠連と馬連が同じファイルに入る**。
    　**`枠1-2` も `馬01-02` も組は (1,2) なので、種別で分けないと衝突する**（2026-08-14）。
    """
    c = str(cell)
    ds = re.findall(r"\d+", c)
    if not ds:
        return None
    kind = re.sub(r"[\d\-‐－_\s]", "", c) or "?"
    s = "".join(ds)
    if len(ds) == 2:
        a, b = int(ds[0]), int(ds[1])
    elif len(s) == 4:
        a, b = int(s[:2]), int(s[2:])
    elif len(s) == 2:
        a, b = int(s[0]), int(s[1])
    else:
        return None
    return kind, ((a, b) if a <= b else (b, a))


def _dt(year, s):
    return datetime(int(year), int(s[:2]), int(s[2:4]), int(s[4:6]), int(s[6:8]))


def rid8_of(rid16):
    """16桁レースID → 手元の8桁raceid（場+年下2桁+回+日+R）。"""
    if len(rid16) != 16 or not rid16.isdigit():
        return None
    y, jyo, kai, hi, rr = rid16[2:4], rid16[8:10], rid16[10:12], rid16[12:14], rid16[14:16]
    return f"{jyo}{y}{int(kai)}{int(hi)}{rr}"


def kinds_of(path):
    """そのファイルに入っている券種（ヘッダの接頭辞）を返す。`{'枠': 36, '馬': 153}` など。"""
    txt = open(path, "rb").read().decode("shift_jis", "replace")
    rows = [r for r in csv.reader(io.StringIO(txt)) if r and r[0].strip()]
    if not rows:
        return {}
    out = {}
    for c in rows[0][FIXED:]:
        k = _key(c)
        if k:
            out[k[0]] = out.get(k[0], 0) + 1
    return out


def load_file(path, kind=None):
    """1ファイル → {rid8: {"n":頭数, "snaps":{ラベル: (時刻, {組:オッズ})}}}。

    ★ラベルは **前日 / 当日朝 / 直前 / 確定**。
    　「直前」＝**区分1の最後**（実測で発走−11分）。「前日」＝区分1の最初。
    　「当日朝」＝区分1の最後から1つ前。区分1が3行に満たなければ入らない。

    ★`kind` … 券種の接頭辞（`枠` / `馬`）。**部分一致で選ぶ**。
    　**None なら「1種類しか入っていない場合に限り」その1種類を使う**。
    　⚠**混在ファイルで None を渡すと組が衝突するので、そのときは例外にする**。
    """
    txt = open(path, "rb").read().decode("shift_jis", "replace")
    rows = [r for r in csv.reader(io.StringIO(txt)) if r and r[0].strip()]
    if len(rows) < 2:
        return {}
    keys = [_key(c) for c in rows[0][FIXED:]]
    have = sorted({k[0] for k in keys if k})
    if not have:
        return {}
    if kind is None:
        if len(have) > 1:
            raise ValueError(
                f"{path} には券種が複数入っている {have}。"
                f"　load_file(path, kind='枠') のように指定すること")
        want = set(have)
    else:
        # ★別名を**優先順に完全一致**で試す。⚠「馬」を素朴に部分一致させると
        #   **馬連と馬単の両方**に当たる（「全オッズ」を1ファイルに出すと実際に両方入る）。
        #   実測の列名は `枠1-1` / `馬01-02` だが、`枠連〜` `馬連〜` の版もありうるので両方見る。
        want = set()
        for c in ALIAS.get(kind, (kind,)):
            want = {h for h in have if h == c}
            if want:
                break
        if not want:
            for c in ALIAS.get(kind, (kind,)):
                w = {h for h in have if h.startswith(c)}
                if len(w) == 1:
                    want = w
                    break
                if len(w) > 1:
                    raise ValueError(
                        f"{path} で券種 '{c}' が {sorted(w)} の複数に当たる。"
                        f"　完全な列名の接頭辞で指定すること")
        if not want:
            return {}
    keys = [k if (k and k[0] in want) else None for k in keys]
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
                vals[k[1]] = v
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


POOL_DIRS = ("data/odds_ts_all", "data/odds_ts_waku", "data/odds_ts_umaren")


def load_pool(kind):
    """★券種を指定して、**置き場所を問わず**全部集める。解析側はこれだけ呼べばよい。

    `data/odds_ts_all/`（全オッズを1ファイルに出した場合）と
    `data/odds_ts_waku/` `data/odds_ts_umaren/`（券種ごとに出した場合）の**両方を見る**。
    ★**その券種の列を持たないファイルは黙って飛ばす**ので、**混在していても正しく動く**。
    """
    out = {}
    for d in POOL_DIRS:
        if os.path.isdir(d):
            out.update(load_dir(d, kind))
    return out


def paths_of(d):
    if os.path.isfile(d):
        return [d]
    return sorted(glob.glob(os.path.join(d, "*.CSV")) + glob.glob(os.path.join(d, "*.csv")))


def load_dir(d, kind=None):
    """ディレクトリでもファイルでも受ける。複数ファイルはマージする。

    ★`kind` を渡すと**その券種の列だけ**読む。**「全オッズ」1ファイルでも使える**。
    　★**券種を持たないファイルは黙って飛ばす**ので、
    　**枠連だけのファイルと全オッズのファイルが混ざっていても正しく動く**。
    """
    out = {}
    for p in paths_of(d):
        out.update(load_file(p, kind))
    return out


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "data/odds_ts_waku"
    kind = sys.argv[2] if len(sys.argv) > 2 else None
    ks = {}
    for p in paths_of(d):
        for k, c in kinds_of(p).items():
            ks[k] = max(ks.get(k, 0), c)
    if ks:
        print(f"★ファイルに入っている券種: " + " / ".join(f"{k}（{c}組）" for k, c in sorted(ks.items())))
        if len(ks) > 1 and kind is None:
            sys.exit("⚠券種が複数ある。`python3 ml/odds_ts_combo.py <パス> 枠` のように指定すること。")
    rec = load_dir(d, kind)
    if not rec:
        sys.exit(f"{d} が読めない。TARGETの「指定時系列オッズ(CSV形式)」で出したCSVを置くこと。")
    print(f"{d}{'（' + kind + '）' if kind else ''}: {len(rec):,} レース　"
          f"ファイル {len(paths_of(d))}本")
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
