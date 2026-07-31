"""
netkeiba からの取得（**手元のMacで実行する**。クラウド環境からは接続できない）。

負荷をかけない設計にしてある:
  ・**1.5秒間隔**（`--wait` で変更可）＋失敗時は指数バックオフ
  ・**取得済みは再取得しない**（`data/nk_cache/` にHTMLをそのまま保存し、2度目以降はそれを読む）
  ・**必要なレースだけ**。過去13年の遡りは不要——学習データは既に手元にある(587,077行)ので、
    埋めるのは差分（週36レース程度）だけ。

使い方:
  # ① ある開催日の成績を取る → 学習データ(DS互換CSV)と払戻を作る
  python3 ml/nk_fetch.py results 20260801

  # ② 当日の出馬表＋オッズを取る → 予想用(DG相当)を作る
  python3 ml/nk_fetch.py entries 20260801

  # ③ 父・母父が空の馬を埋める（1頭1回だけ。キャッシュされる）
  python3 ml/nk_fetch.py pedigree

出力:
  data/nk_cache/      取得した生HTML/JSON（再取得を避けるため。gitignore推奨）
  data/nk/DSnk<日付>.CSV   DS互換52列（そのまま `*.CSV` として学習に使える）
  data/nk/pay<日付>.csv    払戻（券種・組・配当のtidy形式）
  data/nk/entries<日付>.json  出馬表＋オッズ（predict用）
"""
import json
import os
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nk_parse import parse_result, parse_shutuba, parse_odds_json, to_ds_rows, nk_raceid

UA = "Mozilla/5.0 (compatible; personal-research/1.0)"
CACHE = "data/nk_cache"
OUT = "data/nk"
WAIT = 1.5


def get(url, key, referer=None, wait=WAIT, retries=3):
    """キャッシュ優先で1本取得。取得済みなら通信しない。"""
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, key)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return open(path, "rb").read()
    req = urllib.request.Request(url, headers={"User-Agent": UA,
                                               **({"Referer": referer} if referer else {})})
    for i in range(retries):
        try:
            time.sleep(wait)
            with urllib.request.urlopen(req, timeout=30) as r:
                b = r.read()
            open(path, "wb").write(b)
            return b
        except (urllib.error.URLError, TimeoutError) as e:
            if i == retries - 1:
                print(f"  取得失敗 {url}: {e}")
                return b""
            time.sleep(wait * (2 ** (i + 1)))   # 指数バックオフ
    return b""


def race_ids_of_day(ymd):
    """その日のJRAのrace_id一覧。

    ★2つの落とし穴に対応している:
      ・`db.netkeiba.com/race/list/` は**結果データベース**なので、まだ走っていない日は空になる。
        当日の予想には使えないため、**先に `race_list_sub.html` を見る**（未来日でも返る・実確認済み）。
      ・一覧には**地方競馬が混ざる**（実データ: 7/26は78レース中42が地方）。
        場コードは JRA が 01〜10 なので、それ以外を落とす。落とさないと地方の行が学習データに入る。
    """
    import re
    b = get(f"https://race.netkeiba.com/top/race_list_sub.html?kaisai_date={ymd}",
            f"rlist_{ymd}.html")
    ids = set(re.findall(r"race_id=(\d{12})", b.decode("utf-8", "replace")))
    if not ids:   # 過去日でこちらが空なら結果DB側を見る
        b = get(f"https://db.netkeiba.com/race/list/{ymd}/", f"list_{ymd}.html")
        ids = set(re.findall(r"/race/(\d{12})", b.decode("euc_jp", "replace")))
    jra = sorted(i for i in ids if i.startswith(ymd[:4]) and i[4:6].isdigit()
                 and 1 <= int(i[4:6]) <= 10)
    if len(ids) > len(jra):
        print(f"  （地方競馬 {len(ids)-len(jra)}レースを除外）")
    return jra


def names_cache(update=None):
    p = os.path.join(CACHE, "names.json")
    d = json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {"jockey": {}, "trainer": {}}
    if update:
        for k in ("jockey", "trainer"):
            d[k].update(update.get(k, {}))
        os.makedirs(CACHE, exist_ok=True)
        json.dump(d, open(p, "w", encoding="utf-8"), ensure_ascii=False)
    return d


def cmd_results(ymd):
    import csv
    ids = race_ids_of_day(ymd)
    print(f"{ymd}: {len(ids)}レース")
    if not ids:
        print("  レース一覧が取れなかった。URLの形式が変わっている可能性がある。")
        return
    os.makedirs(OUT, exist_ok=True)
    rows, pays, nm = [], [], {"jockey": {}, "trainer": {}}
    for rid in ids:
        b = get(f"https://db.netkeiba.com/race/{rid}/", f"race_{rid}.html")
        if not b:
            continue
        race, horses, pay = parse_result(b, rid)
        if not horses:
            continue
        rows += to_ds_rows(race, horses)
        for h in horses:
            if h.get("jockey_id"):
                nm["jockey"][h["jockey_id"]] = h["jockey"]
            if h.get("trainer_id"):
                nm["trainer"][h["trainer_id"]] = h["trainer"]
        for kind, lst in pay.items():
            for combo, amt in lst:
                pays.append([race["raceid"], kind, combo, amt])
        print(f"  {rid} {race['place']}{int(rid[10:12])}R {race['name'][:12]} {len(horses)}頭")
    names_cache(nm)
    f1 = f"{OUT}/DSnk{ymd}.CSV"
    with open(f1, "w", encoding="shift_jis", errors="replace", newline="") as fh:
        csv.writer(fh).writerows(rows)
    f2 = f"{OUT}/pay{ymd}.csv"
    with open(f2, "w", encoding="utf-8", newline="") as fh:
        csv.writer(fh).writerows([["raceid", "kind", "combo", "payout"]] + pays)
    print(f"保存: {f1}（{len(rows)}行） / {f2}（{len(pays)}件）")
    print("※父・母父は空。`python3 ml/nk_fetch.py pedigree` で埋めること。")


def cmd_entries(ymd):
    ids = race_ids_of_day(ymd)
    if not ids:
        print("レース一覧が取れなかった")
        return
    os.makedirs(OUT, exist_ok=True)
    nm, out = names_cache(), []
    for rid in ids:
        b = get(f"https://race.netkeiba.com/race/shutuba.html?race_id={rid}",
                f"shutuba_{rid}.html")
        if not b:
            continue
        ttl, hs = parse_shutuba(b, nm)
        # オッズは毎回取り直す（キャッシュしない＝時点が変わるため）
        u = (f"https://race.netkeiba.com/api/api_get_jra_odds.html"
             f"?type=1&locale=ja&race_id={rid}&action=init")
        key = f"odds_{rid}_{int(time.time())}.json"
        ob = get(u, key, referer=f"https://race.netkeiba.com/race/shutuba.html?race_id={rid}")
        odds, at = parse_odds_json(ob.decode("utf-8", "replace")) if ob else ({}, "")
        for h in hs:
            h["odds"] = odds.get(int(h["umaban"]), None)
        out.append({"race_id": rid, "raceid": nk_raceid(rid), "title": ttl,
                    "odds_at": at, "horses": hs})
        print(f"  {rid} {ttl[:30]} {len(hs)}頭 オッズ{len(odds)}件 @{at}")
    f = f"{OUT}/entries{ymd}.json"
    json.dump(out, open(f, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"保存: {f}")


def cmd_pedigree():
    """DS互換CSVで父/母父が空の馬を、馬ページから埋める。1頭1回だけ。"""
    import csv
    import glob
    import re
    need = {}
    for p in sorted(glob.glob(f"{OUT}/DSnk*.CSV")):
        for r in csv.reader(open(p, encoding="shift_jis", errors="replace")):
            if len(r) > 45 and r[37] and not r[43]:
                need[r[37]] = True
    print(f"父/母父が未取得の馬: {len(need)}頭")
    ped = {}
    pp = os.path.join(CACHE, "pedigree.json")
    if os.path.exists(pp):
        ped = json.load(open(pp, encoding="utf-8"))
    todo = [h for h in need if h not in ped]
    print(f"うち今回取りに行くのは {len(todo)}頭（キャッシュ済み {len(need)-len(todo)}頭）")
    for i, hid in enumerate(todo, 1):
        b = get(f"https://db.netkeiba.com/horse/ped/{hid}/", f"ped_{hid}.html")
        if not b:
            continue
        s = b.decode("euc_jp", "replace")
        names = re.findall(r'/horse/(?:ped/)?\w+/"[^>]*>([^<]+)</a>', s)
        # 血統表は「父, 父父, …」の順に並ぶ。父=先頭、母父は母ブロックの先頭。
        ped[hid] = {"sire": names[0].strip() if names else "",
                    "damsire": names[15].strip() if len(names) > 15 else ""}
        if i % 20 == 0:
            json.dump(ped, open(pp, "w", encoding="utf-8"), ensure_ascii=False)
            print(f"  {i}/{len(todo)}")
    json.dump(ped, open(pp, "w", encoding="utf-8"), ensure_ascii=False)
    # CSVに書き戻す
    for p in sorted(glob.glob(f"{OUT}/DSnk*.CSV")):
        rows = list(csv.reader(open(p, encoding="shift_jis", errors="replace")))
        for r in rows:
            if len(r) > 45 and r[37] in ped:
                r[43], r[45] = ped[r[37]]["sire"], ped[r[37]]["damsire"]
        with open(p, "w", encoding="shift_jis", errors="replace", newline="") as fh:
            csv.writer(fh).writerows(rows)
        print(f"  更新: {p}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    cmd = sys.argv[1]
    if cmd == "results":
        cmd_results(sys.argv[2])
    elif cmd == "entries":
        cmd_entries(sys.argv[2])
    elif cmd == "pedigree":
        cmd_pedigree()
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
