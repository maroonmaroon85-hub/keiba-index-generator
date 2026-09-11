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
| **3** | ★**自分の時計を正とする**（`fetched_at`・★**オフセット付き**） | ⚠**`official_datetime` は値の時刻を表さない**（09:57のスタンプで値が確定だった実測がある）。★**「朝9時の板」の唯一の根拠になる値なので、タイムゾーンまで残す** |
| **4** | ★**連続失敗したら自分から止まる**（終了コード2） | **ブロックされているのに叩き続けない**（(70)⑤の方針）。⚠★**終了コード2で自動再開する形に包まないこと** |
| **4b** | ★**他の取得と同時に走らせない**（`nk_odds_bulk` / `nk_fetch` / `nk_odds_combo`） | ★**枠連側の回答【4】: 開催日の規則は時間帯ではなく同時実行の問題**。⚠**`entries` も netkeiba を叩くので同じ扱いにした**（回答の「entries の後に順番に流せば大丈夫」＝並走させない）。**起動時に `pgrep` で見る** |
| **4c** | ★**レース一覧が欠けていたら止まる** | ⚠**一覧はキャッシュされる。開催途中に取った欠けた一覧が残っていると欠けたまま集める**（**2026-08-09 に 13/36 で固まった前例**）。★**場ごとに12レース揃っているかを見る**。→ `--refresh` |
| **5** | ★**キャッシュを使わない・残さない** | ⚠`nk_fetch.get` はキャッシュ優先。**朝の板は時点が意味を持つ**ので鍵に時刻を入れる。★**そのぶん生JSONは読んだら消す**（放置すると `data/nk_cache` に毎開催36個ずつ溜まる） |

■ ★**取るもの**: **type=2（複勝）だけ**。1開催日36レース・1.5秒間隔で**約1分**。
　★**単勝は `nk_fetch.py entries` が既に取っている**ので、ここでは取らない（**二重に叩かない**）。

■ ★★**枠連側からの回答（2026-09-10）——★叩く前に読むこと**
| | ★**回答** |
|---|---|
| ★**未走で板が返るか** | ★**type=1 では返る**（**実測: 9/6 札幌6R が 08:04 2.3倍 → 08:45 2.5倍 → 15:02 2.5倍**）。⚠★**type=2 は双方とも未実測**。**同じエンドポイントで type 違いなので返る想定は妥当だが確証は無い** |
| ★**1.5秒間隔・UA** | ★**そのままでよい**（**42,000回を18時間流してブロックされたことは無い**）。⚠★**間隔は縮めないこと** |
| ★**Referer** | ★**`shutuba.html` でよい**（`entries` に合わせるのが正解） |
| ★★**開催日の規則** | ★**時間帯の規則ではない。★同時実行の問題**（`nk_odds_bulk` と衝突する）。→ ★**このスクリプトは起動時に `pgrep -f nk_odds_bulk` を見る**（下の `bulk_running`） |
| ★**キャッシュ** | ★**衝突しない**（`data/nk_cache` は .gitignore 済み・鍵に時刻が入るので `get` のキャッシュ判定に当たらない）。⚠★**消すときに `odds_*` の glob を使わないこと**——**`cmd_entries` が同じ接頭辞で書いている**。→ ★**このスクリプトは `morn2_` 接頭辞の★完全パスだけを消す** |
| ★**ブロックの兆候** | ⚠**本物のブロックの実例は無い**。**唯一の事故(2026-08-09)は全券種 HTTP 400 ＝仕様変更だった**。★**400 が出たら、まずブラウザの DevTools で実物のリクエストを見ること**（**症状から推測して外した前科がある**） |

■ ⚠★★**叩くときの禁止事項（★枠連側の実体験から）**
　1. ⚠★**終了コード2（連続失敗）で★自動再開しないこと**。
　　　**`until` ループで回して2でも5分後に叩き直し、「止まる」配慮を自分で無意味にした前科がある**。
　2. ⚠★**`--dry` は 1レース・1回だけ**。**2026-08-09 に「叩いて確かめる」で症状を悪化させた**
　　　（**切り分けのため連続で叩いた結果、成功していた type=7 まで含めて5券種すべて400になった**）。
　　　★**(70)⑤の負荷方針は「集めるとき」だけでなく★「調べるとき」にも適用する**。
　3. ⚠★**キャッシュのある道具で疎通確認をしないこと**（**`--old` がキャッシュを返して誤判定しかけた**）。
　　　★**このスクリプトはキャッシュを使わないので、疎通確認にも使える**。
　4. ⚠★**間隔（1.5秒）を縮めないこと**。

■ ⚠★**発売前は空で返るのが正常**（`ml/nk_parse.py` の `parse_odds_json` に明記）。
　★**「空＝異常」と判断しない**。⚠**複勝の発売開始時刻が単勝と同じかは双方とも未確認**。

使い方（★Macで実行）:
    python3 ml/nk_place_morn.py 20260912 --dry     # ★まず1レースだけ試す（保存しない）
    python3 ml/nk_place_morn.py 20260912           # その日の全レースの複勝板を取る
    python3 ml/nk_place_morn.py 20260912 --status  # 何レース貯まっているか見るだけ（通信しない）
    python3 ml/nk_place_morn.py 20260912 --refresh # ★レース一覧のキャッシュを捨てて取り直す
　★`nk_odds_bulk` が走っていると**起動時に止まる**（終了コード2）。⚠**押し切るなら `--force`（推奨しない）**

出力: **data/nk_odds_morn/place<YYYYMMDD>.jsonl**（1行1レース・追記）
　{"race_id","raceid","fetched_at","official_at","n","odds":{"01":[下限,上限],…}}
　★**同じ日に2回走らせたら2行入る**（**時点が違うので両方残すのが正しい**）。

自己テスト: python3 ml/nk_place_morn.py --selftest   （★通信しない）
"""
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nk_fetch import CACHE, ODDS_API, get, race_ids_of_day
from nk_odds_combo import parse_combo
from nk_parse import nk_raceid

OUT = "data/nk_odds_morn"          # ★確定板の data/nk_odds とは必ず分ける
TYPE = 2                            # ★複勝のみ
MAX_FAIL = 5                        # ★連続でこれだけ失敗したら止まる


def _ancestors():
    """★自分と自分の祖先のpid（★誤検知よけ）。

    ⚠**`pgrep -f` は「コマンドラインに文字列が含まれるプロセス」を拾うので、
    　★このスクリプトを起動したシェル自身が引っかかる**（実際に2026-09-10に踏んだ）。
    　→ ★**自分と祖先は必ず除く**。
    """
    out, pid = set(), os.getpid()
    for _ in range(12):                     # ★念のため上限を切る（循環よけ）
        out.add(pid)
        try:
            r = subprocess.run(["ps", "-o", "ppid=", "-p", str(pid)],
                               capture_output=True, text=True, timeout=10)
            pid = int(r.stdout.strip())
        except (OSError, ValueError, subprocess.SubprocessError):
            break
        if pid <= 1 or pid in out:
            break
    return out


# ★★同時に netkeiba を叩く可能性のあるもの（★枠連側の回答【4】: 規則は時間帯でなく同時実行）
#   ⚠**`nk_odds_bulk` だけでなく `nk_fetch`（entries/results）も叩く**。
#   　★枠連側の回答も「**entries の後に順番に流せば大丈夫**」＝**並走させない**という意味だった。
FETCHERS = ("nk_odds_bulk", "nk_fetch", "nk_odds_combo")


def bulk_running():
    """★他の取得プロセスが走っていないか。→ (走っている?, 説明)。

    ★`pgrep`/`ps` が使えない環境では None（**勝手に続けない**）。
    ⚠**自分と祖先は除く**。★**このスクリプト自身（`nk_place_morn`）を含む行も除く**。
    """
    try:
        r = subprocess.run(["pgrep", "-f", "|".join(FETCHERS)],
                           capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError) as e:
        return None, f"pgrep が使えない（{e}）"
    mine, hit = _ancestors(), []
    for x in r.stdout.split():
        try:
            pid = int(x)
        except ValueError:
            continue
        if pid in mine:
            continue                        # ★自分・自分を起動したシェル
        try:
            a = subprocess.run(["ps", "-o", "args=", "-p", str(pid)],
                               capture_output=True, text=True, timeout=10).stdout
        except (OSError, subprocess.SubprocessError):
            a = ""
        if "nk_place_morn" in a:
            continue                        # ★このスクリプトを起動した別の包み
        if not any(f in a for f in FETCHERS):
            continue                        # ★pgrep が拾ったが実体が違う
        hit.append(f"{pid}({next(f for f in FETCHERS if f in a)})")
    return (bool(hit), ("走っている: " + " ".join(hit)) if hit else "走っていない")


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
    rows, broken = [], 0
    for x in open(p, encoding="utf-8"):
        if not x.strip():
            continue
        try:
            rows.append(json.loads(x))      # ⚠**書き込み中に落ちると末尾が欠ける**
        except ValueError:
            broken += 1
    if broken:
        print(f"⚠**読めない行が {broken} 行ある**（**書き込み中に落ちた末尾の可能性**）")
    if not rows:
        print("⚠**読める行が無い**")
        return
    ts = sorted({r["fetched_at"][:16] for r in rows})
    print(f"{p}: **{len(rows)}行 / {len({r['race_id'] for r in rows})}レース**")
    print(f"　収集した時点: {' / '.join(ts)}")


def selftest():
    ok = True
    print("★★★このスクリプトが守ること")
    print("　1. **枠連の運用に触らない**（nk_fetch も nk_odds_bulk も読むだけ）")
    print(f"　2. ★**出力先を分ける**: {OUT}/  ⚠**data/nk_odds とは別**")
    print("　3. ★**自分の時計 `fetched_at` を正とする**（official_datetime は値の時刻ではない）")
    print(f"　4. **連続{MAX_FAIL}回失敗したら止まる**（終了コード2）"
          "　⚠★**自動再開する形に包まないこと**")
    b, why = bulk_running()
    print(f"　4b. ★**他の取得と同時に走らせない**（{' / '.join(FETCHERS)}／今: {why}）"
          + ("　⚠**確認できない環境**" if b is None else ""))
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
    # ★★★これは「朝9時の板」を作るための道具。★昼以降に走らせたら別物になる。
    #   ⚠**止めはしない**（発売前で空だったときの取り直しは正当）。★**必ず言う**。
    hh = int(time.strftime("%H"))
    if not 8 <= hh <= 11:
        print(f"⚠⚠**今 {time.strftime('%H:%M %Z')} です。★これは「朝9時の板」ではありません**"
              "　★**記録には残りますが、朝9時運用の標本としては使えません**")

    # ★★★叩く前に必ず: nk_odds_bulk と同時に走らせない（枠連側の回答【4】）
    busy, why = bulk_running()
    if busy:
        print(f"⚠⚠**他の取得プロセスが走っている（{why}）。同時実行はしない**"
              "　★**止まるのを待つか、`--force` で押し切る（★推奨しない）**")
        if "--force" not in sys.argv:
            return 2
    elif busy is None:
        print(f"⚠**同時実行の確認ができなかった（{why}）**"
              f"　★**手で `pgrep -f '{'|'.join(FETCHERS)}'` を見てから `--force` で実行すること**")
        if "--force" not in sys.argv:
            return 2
    else:
        print(f"★同時実行の確認（{' / '.join(FETCHERS)}）: {why}")

    ids = race_ids_of_day(ymd, refresh="--refresh" in sys.argv)
    if not ids:
        print("⚠レース一覧が取れなかった（**当日でないか、まだ一覧が出ていない**）")
        return 1
    # ★★一覧はキャッシュされる（`rlist_<日付>.html`）。⚠**開催途中に取った欠けた一覧が
    # 　残っていると、こちらも欠けたまま集めてしまう**（2026-08-09 に 13/36 で固まった前例）。
    # 　→ ★**場ごとに12レース揃っているかを見て、欠けていたら止める**。
    per = {}
    for i in ids:
        per[i[4:6]] = per.get(i[4:6], 0) + 1
    short = {k: v for k, v in per.items() if v < 12}
    print(f"　場ごとのレース数: " + " / ".join(f"{k}:{v}" for k, v in sorted(per.items())))
    if short:
        print(f"⚠⚠**12レースに満たない場がある {short}**"
              "　★**開催途中に取った一覧がキャッシュに残っている疑い**"
              "　→ ★**`--refresh` を付けて取り直すこと**")
        if "--refresh" not in sys.argv and "--force" not in sys.argv:
            return 2
    print(f"★{ymd}: **{len(ids)}レース**"
          + ("　⚠**--dry: 先頭1レースだけ・保存しない**" if dry else ""))
    if dry:
        ids = ids[:1]
    else:
        os.makedirs(OUT, exist_ok=True)

    fh = None if dry else open(out_path(ymd), "a", encoding="utf-8")
    # ⚠**`nfail` は「連続」失敗のカウンタ（MAX_FAIL 用）。★総数は別に持つ**
    #   （初版は最後に `nfail` を「失敗数」として出していたので、飛び飛びの失敗が 0 と出ていた）
    nfail, nok, nempty, nbad = 0, 0, 0, 0
    try:
        for rid in ids:
            od, at, drop = fetch_one(rid)
            now = time.strftime("%Y-%m-%d %H:%M:%S%z")   # ★実行機の地方時＋オフセット
            if od is None:
                nfail += 1
                nbad += 1
                print(f"  {rid} ⚠取得失敗（連続{nfail} / 通算{nbad}）")
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

    print(f"\n★**取れた {nok} / 空 {nempty} / 失敗 {nbad}**（対象 {len(ids)}）")
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
