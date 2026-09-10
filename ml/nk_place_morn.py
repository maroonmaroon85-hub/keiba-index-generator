"""(242) ★★★★**朝9時の複勝板を集める**（**Macで実行**。クラウドからは接続できない）

★★**なぜ要るか（2026-09-10 に判明）**
　★**穴馬の軸は `ズレ = pn − qp`。`qp` は複勝の板の[下限,上限]の調和平均からしか作れない**。
　⚠★**ところが朝9時の複勝板はどこにも無い**:
　　・`data/nk_odds/type2_*` は**確定板**（`nk_odds_verify.py` で的中組と実配当が8/8ズレ0円）
　　・`data/odds_ts` は**単勝だけ**（区分1/3/4 × 各馬1列）
　★★★**過去レースに netkeiba は確定板しか返さない＝朝9時の板は原理的に遡れない**。
　→ ★**これから走るレースを、朝9時に取るしかない。★取らなかった開催日は永久に失われる**。

■ ⚠★★**このスクリプトが守ること（★設計の中心）**
| | ★**規則** | ★**理由** |
|---|---|---|
| **1** | ★**枠連の運用に一切触らない** | `nk_fetch.py` も `nk_odds_bulk.py` も**読むだけ・書き換えない**。**別プロセスなので落ちても枠連は無事** |
| **2** | ★★**出力先を分ける**（`data/nk_odds_morn/`） | ⚠**`data/nk_odds/type2_*` は11年バックテストの母集団**。**朝の板を混ぜたら確定板の母集団が壊れる** |
| **3** | ★**自分の時計を正とする**（`fetched_at`） | ⚠**`official_datetime` は値の時刻を表さない**（09:57のスタンプで値が確定だった実測がある） |
| **4** | ★**連続失敗したら自分から止まる**（終了コード2） | **ブロックされているのに叩き続けない**（(70)⑤の方針） |
| **5** | ★**キャッシュを使わない・残さない** | ⚠`nk_fetch.get` はキャッシュ優先。**朝の板は時点が意味を持つ**ので鍵に時刻を入れる。★**そのぶん生JSONは読んだら消す**（放置すると `data/nk_cache` に毎開催36個ずつ溜まる） |

■ ★**取るもの**: **type=2（複勝）だけ**。1開催日36レース・1.5秒間隔で**約1分**。
　★**単勝は `nk_fetch.py entries` が既に取っている**ので、ここでは取らない（**二重に叩かない**）。

■ ⚠★**まだ確認できていないこと（★枠連側に確認中・2026-09-10）**
　★**未走のレースに type=2 を投げたとき、その時点の板が返るか**。
　　**返る想定だが検証していない**。→ ★**初回は必ず `--dry` で1レースだけ試すこと**。
　⚠**空が返るのは異常ではない**（`entries` も「発売前は空で返る」と書いてある）。

使い方（★Macで実行）:
    python3 ml/nk_place_morn.py 20260912 --dry     # ★まず1レースだけ試す（保存しない）
    python3 ml/nk_place_morn.py 20260912           # その日の全レースの複勝板を取る
    python3 ml/nk_place_morn.py 20260912 --status  # 何レース貯まっているか見るだけ（通信しない）

出力: **data/nk_odds_morn/place<YYYYMMDD>.jsonl**（1行1レース・追記）
　{"race_id","raceid","fetched_at","official_at","n","odds":{"01":[下限,上限],…}}
　★**同じ日に2回走らせたら2行入る**（**時点が違うので両方残すのが正しい**）。

自己テスト: python3 ml/nk_place_morn.py --selftest   （★通信しない）
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nk_fetch import CACHE, ODDS_API, get, race_ids_of_day
from nk_odds_combo import parse_combo
from nk_parse import nk_raceid

OUT = "data/nk_odds_morn"          # ★確定板の data/nk_odds とは必ず分ける
TYPE = 2                            # ★複勝のみ
MAX_FAIL = 5                        # ★連続でこれだけ失敗したら止まる


def out_path(ymd):
    return f"{OUT}/place{ymd}.jsonl"


def fetch_one(rid):
    """1レースの複勝板。→ (odds, official_at, drop) / 取れなければ (None, "", {})。

    ⚠**キャッシュを使わない**。`get` はキャッシュ優先なので、鍵に時刻を入れて必ず通信させる。
    """
    key = f"morn2_{rid}_{int(time.time())}.json"
    b = get(ODDS_API.format(t=TYPE, rid=rid), key,
            referer=f"https://race.netkeiba.com/race/shutuba.html?race_id={rid}")
    # ★鍵に時刻を入れるので、放っておくと data/nk_cache に毎回ゴミが増える。
    # 　★解析結果は自分の jsonl に持つので、生JSONは読んだら捨てる。
    try:
        os.remove(os.path.join(CACHE, key))
    except OSError:
        pass
    if not b:
        return None, "", {}
    od, at, drop = parse_combo(b.decode("utf-8", "replace"), TYPE)
    return od, at, drop


def status(ymd):
    p = out_path(ymd)
    if not os.path.exists(p):
        print(f"{p} はまだ無い")
        return
    rows = [json.loads(x) for x in open(p, encoding="utf-8") if x.strip()]
    ts = sorted({r["fetched_at"][:16] for r in rows})
    print(f"{p}: **{len(rows)}行 / {len({r['race_id'] for r in rows})}レース**")
    print(f"　収集した時点: {' / '.join(ts)}")


def selftest():
    ok = True
    print("★★★このスクリプトが守ること")
    print("　1. **枠連の運用に触らない**（nk_fetch も nk_odds_bulk も読むだけ）")
    print(f"　2. ★**出力先を分ける**: {OUT}/  ⚠**data/nk_odds とは別**")
    print("　3. ★**自分の時計 `fetched_at` を正とする**（official_datetime は値の時刻ではない）")
    print(f"　4. **連続{MAX_FAIL}回失敗したら止まる**（終了コード2）")
    print(f"　5. **キャッシュを使わない・残さない**（鍵に時刻を入れ、生JSONは読んだら {CACHE} から消す）")
    ok &= OUT != "data/nk_odds" and not OUT.rstrip("/").endswith("nk_odds")
    print(f"　→ 出力先が確定板と別 {'★OK' if ok else '⚠NG'}")

    # ★parse_combo が複勝を [下限,上限] で返すかの検算（★通信しない）
    fake = json.dumps({"data": {"official_datetime": "2026-09-12 09:00:00",
                                "odds": {"2": {"01": ["1.5", "2.1", "1"],
                                               "02": ["3.0", "4.4", "3"],
                                               "03": ["-3.0", "0", "0"]}}}})
    od, at, drop = parse_combo(fake, TYPE)
    o1 = od.get("01")
    good = (isinstance(o1, (list, tuple)) and abs(o1[0] - 1.5) < 1e-9
            and abs(o1[1] - 2.1) < 1e-9 and "03" not in od and len(drop) == 1)
    print(f"★複勝の板の検算: 01→{o1} / 02→{od.get('02')} / "
          f"03(負の番兵)→落とした {list(drop)}　{'★OK' if good else '⚠NG'}")
    ok &= good
    print(f"　official_datetime は読むが**使わない**: '{at}'")

    r = nk_raceid("202609040201")
    print(f"★raceid の検算: 202609040201 → {r}　{'★OK' if len(r) == 8 else '⚠NG'}")
    ok &= len(r) == 8
    print(f"★URL: type={TYPE} 固定（**単勝は entries が取るので叩かない**）")
    print("⚠★**未走レースで板が返るかは未検証**。★**初回は必ず `--dry` で1レースだけ試す**")
    print("⚠**枠連の運用には触れない**。**設定変更は提案しない**")
    print("★自己テスト: " + ("全部OK" if ok else "⚠NG"))
    return 0 if ok else 1


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        sys.exit("日付が要る: python3 ml/nk_place_morn.py 20260912 [--dry|--status]")
    ymd = args[0]
    if "--status" in sys.argv:
        status(ymd)
        return 0
    dry = "--dry" in sys.argv

    ids = race_ids_of_day(ymd)
    if not ids:
        print("⚠レース一覧が取れなかった（**当日でないか、まだ一覧が出ていない**）")
        return 1
    print(f"★{ymd}: **{len(ids)}レース**"
          + ("　⚠**--dry: 先頭1レースだけ・保存しない**" if dry else ""))
    if dry:
        ids = ids[:1]
    else:
        os.makedirs(OUT, exist_ok=True)

    fh = None if dry else open(out_path(ymd), "a", encoding="utf-8")
    nfail, nok, nempty = 0, 0, 0
    try:
        for rid in ids:
            od, at, drop = fetch_one(rid)
            now = time.strftime("%Y-%m-%d %H:%M:%S")
            if od is None:
                nfail += 1
                print(f"  {rid} ⚠取得失敗（連続{nfail}）")
                if nfail >= MAX_FAIL:
                    print(f"⚠⚠**連続{MAX_FAIL}回失敗した。ブロックの疑いで止める**")
                    return 2
                continue
            nfail = 0
            if not od:
                nempty += 1
                print(f"  {rid} ⚠**板が空**（発売前なら異常ではない）")
                continue
            nok += 1
            rec = {"race_id": rid, "raceid": nk_raceid(rid), "fetched_at": now,
                   "official_at": at, "n": len(od), "odds": {k: list(v) for k, v in od.items()}}
            if drop:
                rec["dropped"] = drop
            print(f"  {rid} {len(od)}頭 fetched_at={now}"
                  + (f"　official_at={at}" if at else "")
                  + (f"　⚠落とした {drop}" if drop else ""))
            if fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
    finally:
        if fh:
            fh.close()

    print(f"\n★**取れた {nok} / 空 {nempty} / 失敗 {nfail}**")
    if dry:
        print("⚠**--dry なので保存していない**。★**板が返っているなら --dry 無しで本番**")
        return 0
    print(f"★保存: **{out_path(ymd)}**")
    if nempty:
        print(f"⚠**{nempty}レースが空だった**。**朝9時前なら発売前の可能性**。"
              "★**時間を置いてもう一度走らせれば追記される**")
    print("⚠**枠連の運用には触れていない**")
    return 0


if __name__ == "__main__":
    sys.exit(selftest() if "--selftest" in sys.argv else main())
