"""(113) 過去レースのオッズ板を**一晩かけて遡って集める**。**Macで実行・放置して良い**。

★`nk_odds_combo.py --list` との違い（あちらは634レース16分用。全期間には作りが足りない）
　1. **再開できる**。取得済みを台帳(`done_type<N>.txt`)に持ち、2回目以降は**通信せず飛ばす**。
　　 途中で落ちても、Macが寝ても、同じコマンドをもう一度打てば続きから進む。
　2. **1レース1ファイルにしない**。42,181レース分のファイルを作るとgitが実用にならないので、
　　 **年ごとの `type<N>_<年>.jsonl.gz`** に追記する（1行1レース）。
　3. **生JSONをキャッシュに溜めない**。全期間だと数GBになる。ここでは解析後に捨てる。
　4. **★2%裾の634レースを最優先で先に取る**。途中で止まっても(113事前登録)の分析は成立する。
　5. **連続失敗したら自分から止まる**。ブロックされているのに叩き続けない（(70)⑤の方針）。

★所要時間の目安（1.5秒間隔）
　　2%裾のみ 634R → **約16分**
　　全期間 42,181R → **約18時間**（`--hours` で打ち切れる）

★Macで放置するときの注意
　・**スリープすると止まる**（前セッションで実測済み）。`caffeinate` で抑える。
　・ターミナルを閉じても続くように `nohup` を付け、出力はログに落とす。
　　　caffeinate -is nohup python3 ml/nk_odds_bulk.py > /tmp/odds.log 2>&1 &
　・様子を見る:  tail -f /tmp/odds.log     途中経過:  python3 ml/nk_odds_bulk.py --status

使い方:
    python3 ml/nk_odds_bulk.py                    # 全期間の三連複(type=7)。裾を先に取る
    python3 ml/nk_odds_bulk.py --type 3           # 枠連（9頭以上のレースだけ返る）
    python3 ml/nk_odds_bulk.py --from 2020        # 年で絞る
    python3 ml/nk_odds_bulk.py --hours 8          # 8時間で自分から止まる（続きは次回）
    python3 ml/nk_odds_bulk.py --list <file>      # race_id一覧だけを取る
    python3 ml/nk_odds_bulk.py --status           # 何件終わったか（通信しない）

出力:
    data/nk_odds/type<N>_<年>.jsonl.gz   1行 = {"race_id":…,"at":…,"odds":{…}}
    data/nk_odds/done_type<N>.txt        取得済み台帳（1行1race_id。空だった分は末尾に `-`）
"""
import gzip
import json
import os
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OUT = "data/nk_odds"
API = ("https://race.netkeiba.com/api/api_get_jra_odds.html"
       "?type={t}&locale=ja&race_id={rid}&action=init")
UA = "Mozilla/5.0 (compatible; personal-research/1.0)"
WAIT = 1.5
MAX_FAIL = 20        # ★これだけ連続で失敗したら止める（ブロックされている可能性が高い）


def fetch_raw(rid, t, retries=3):
    """1本取得。**キャッシュに残さない**（全期間だと数GBになるため）。"""
    url = API.format(t=t, rid=rid)
    req = urllib.request.Request(url, headers={
        "User-Agent": UA,
        "Referer": f"https://race.netkeiba.com/odds/index.html?race_id={rid}"})
    for i in range(retries):
        try:
            time.sleep(WAIT)
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.read().decode("utf-8", "replace")
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            if i == retries - 1:
                return f"__ERR__{e}"
            time.sleep(WAIT * (2 ** (i + 1)))    # 指数バックオフ
    return "__ERR__"


def load_done(t):
    """台帳 → 取得済みrace_idの集合。**通信を一切せずに再開点を決める**ための仕組み。"""
    path = f"{OUT}/done_type{t}.txt"
    if not os.path.exists(path):
        return set()
    return {ln.strip().rstrip("-").strip()
            for ln in open(path, encoding="utf-8") if ln.strip()}


def targets(y0, y1, list_path, tail_path):
    """取りに行くrace_id一覧。**2%裾を先頭に置く**（途中で止まっても分析が成立するように）。"""
    if list_path:
        return [x.strip() for x in open(list_path, encoding="utf-8") if x.strip()]
    from audit_crosspool import load_races
    from nk_odds_targets import to_nk_raceid
    ids = []
    for r in load_races():
        if not (y0 <= r["year"] <= y1):
            continue
        rid = to_nk_raceid(r["rid"])
        if rid:
            ids.append(rid)
    ids = sorted(set(ids))
    if tail_path and os.path.exists(tail_path):
        tail = [x.strip() for x in open(tail_path, encoding="utf-8") if x.strip()]
        head = [r for r in tail if r in set(ids)]
        rest = [r for r in ids if r not in set(head)]
        return head + rest
    return ids


def iter_records(t=7, years=None):
    """保存した板を1レースずつ読む（解析側から使う）。→ {"race_id","at","odds"}。

    ★壊れた末尾に強くしてある。放置収集中に落ちると最後の1件が切れることがあるが、
    　そこで全部読めなくなると困るので、**読めたところまでを返して警告する**。
    """
    import glob
    for path in sorted(glob.glob(f"{OUT}/type{t}_*.jsonl.gz")):
        y = int(os.path.basename(path).split("_")[1][:4])
        if years and y not in years:
            continue
        try:
            with gzip.open(path, "rt", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        yield json.loads(line)
        except (EOFError, OSError, ValueError) as e:
            print(f"  ※{path} は途中までしか読めなかった（{e}）", file=sys.stderr)


def main():
    a = sys.argv[1:]
    t = 7
    if "--type" in a:
        i = a.index("--type")
        t = int(a[i + 1])
        del a[i:i + 2]
    y0, y1 = 2000, 2100
    for flag, setter in (("--from", 0), ("--to", 1)):
        if flag in a:
            i = a.index(flag)
            v = int(a[i + 1])
            del a[i:i + 2]
            if setter == 0:
                y0 = v
            else:
                y1 = v
    hours = None
    if "--hours" in a:
        i = a.index("--hours")
        hours = float(a[i + 1])
        del a[i:i + 2]
    list_path = None
    if "--list" in a:
        i = a.index("--list")
        list_path = a[i + 1]
        del a[i:i + 2]

    done = load_done(t)
    if "--status" in a:
        print(f"type={t}  取得済み {len(done)} レース")
        for y in range(2013, 2027):
            p = f"{OUT}/type{t}_{y}.jsonl.gz"
            if os.path.exists(p):
                print(f"   {y}  {os.path.getsize(p)/1e6:>7.1f} MB  {p}")
        return

    from nk_odds_combo import parse_combo
    ids = targets(y0, y1, list_path, f"{OUT}/targets_2pct.txt")
    todo = [r for r in ids if r not in done]
    os.makedirs(OUT, exist_ok=True)

    print(f"type={t}  対象 {len(ids)} レース / 取得済み {len(done)} / これから {len(todo)}")
    print(f"見込み {len(todo)*WAIT/3600:.1f} 時間"
          + (f"（{hours}時間で打ち切る）" if hours else "") + "\n", flush=True)
    if not todo:
        print("すべて取得済み。")
        return

    ledger = open(f"{OUT}/done_type{t}.txt", "a", encoding="utf-8")
    files, t0, n_ok, n_empty, fails = {}, time.time(), 0, 0, 0
    try:
        for i, rid in enumerate(todo):
            if hours and time.time() - t0 > hours * 3600:
                print(f"\n{hours}時間たったので止める。**同じコマンドで続きから再開できる**。")
                break
            txt = fetch_raw(rid, t)
            if txt.startswith("__ERR__"):
                fails += 1
                print(f"  {rid}  取得失敗（連続{fails}）: {txt[7:][:80]}", flush=True)
                if fails >= MAX_FAIL:
                    print(f"\n★{MAX_FAIL}連続で失敗した。**ブロックされている可能性が高いので止める**。"
                          "\n　時間をおいて同じコマンドを打てば続きから再開する。")
                    break
                continue
            fails = 0
            odds, at, _ = parse_combo(txt, t)
            if not odds:
                # 空も台帳に残す（8頭立ての枠連など**そもそも存在しない**分を毎回叩き直さない）
                n_empty += 1
                ledger.write(f"{rid} -\n")
            else:
                # ★1件ごとに開いて閉じる。**放置中に落ちてもファイルが壊れない**ことを優先する
                # 　（gzipは複数メンバーの連結を読めるので、追記のたびに閉じてよい）。
                files[rid[:4]] = 1
                with gzip.open(f"{OUT}/type{t}_{rid[:4]}.jsonl.gz", "at",
                               encoding="utf-8") as fh:
                    fh.write(json.dumps(
                        {"race_id": rid, "at": at, "odds": odds},
                        ensure_ascii=False, separators=(",", ":")) + "\n")
                n_ok += 1
                ledger.write(f"{rid}\n")
            ledger.flush()
            if (i + 1) % 25 == 0 or i == 0:
                el = time.time() - t0
                eta = el / (i + 1) * (len(todo) - i - 1) / 3600
                print(f"  [{i+1}/{len(todo)}] {rid}  保存{n_ok} 空{n_empty}  "
                      f"経過{el/3600:.2f}h 残り{eta:.2f}h", flush=True)
    except KeyboardInterrupt:
        print("\n中断した。**同じコマンドで続きから再開できる**。")
    finally:
        ledger.close()

    print(f"\n保存 {n_ok} / 空 {n_empty} レース。台帳: {OUT}/done_type{t}.txt")
    print(f"続きをやるとき: python3 ml/nk_odds_bulk.py --type {t}"
          + (f" --from {y0}" if y0 > 2000 else ""))


if __name__ == "__main__":
    main()
