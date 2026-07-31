"""
netkeiba の HTML/JSON → 手元の `DS*.CSV`（成績フルセット）と同じ52列レイアウトに変換する。

**なぜ52列に合わせるのか**: `ml/features.py` の `to_model()` が読む列番号は固定なので、
そこに合わせて吐けば **features / train_prod / predict は一切変更せずに動く**。
新しい経路のデータを既存の検証結果とそのまま比較できる（ここを崩すと過去の数字と接続できなくなる）。

取得元と取れるもの:
  ・成績ページ `db.netkeiba.com/race/<race_id>/`（EUC-JP）
      着順/枠/馬番/馬名/性齢/斤量/騎手/タイム/着差/通過/上り/単勝/人気/馬体重/調教師/賞金
      ＋レース条件（芝ダ・距離・馬場）＋払戻8券種 ＋ horse_id
  ・出馬表 `race.netkeiba.com/race/shutuba.html?race_id=<id>`（UTF-8）
      枠/馬番/馬名/性齢/斤量/騎手/厩舎/馬体重/horse_id
      ※**オッズはJSで後読みなので入っていない**（実データで確認: `---.-` が出走頭数ぶん並ぶだけ）
  ・オッズAPI `race.netkeiba.com/api/api_get_jra_odds.html?type=1&race_id=<id>&action=init`
      `data.odds["1"][馬番2桁] = [単勝オッズ, -, 人気]` ＋ `official_datetime`（取得時点）

★取れないもの: **父・母父**。馬ページ `/horse/<horse_id>/` が別途要る（1頭1回・キャッシュ可）。
　父/母父は空欄で出力し、`nk_pedigree.py`（未実装）で後から埋める設計にしてある。

★raceid の変換（ここを間違えると過去データと結合できない）:
  netkeiba: 年4 + 場2 + 回2 + 日2 + R2   例 202601010201
  手元DS  : 場2 + 年下2 + 回1 + 日1 + R2  例 01261201   ※日が10以上は A,B,C…
  さらに DS の col40 は「raceid8 + 馬番2」の10桁。

実行（サンプルで検算）: python3 ml/nk_parse.py
"""
import csv
import html as H
import json
import re
import sys

DS_COLS = 52
# 場コード → 場名。**成績ページのtitleに場所が入っていない**ので race_id から引く。
PLACES = {"01": "札幌", "02": "函館", "03": "福島", "04": "新潟", "05": "東京",
          "06": "中山", "07": "中京", "08": "京都", "09": "阪神", "10": "小倉"}


def _txt(x):
    return H.unescape(re.sub(r"<[^>]+>", "", x)).strip()


def _cells(tr):
    return [_txt(c) for c in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", tr, re.S)]


def _table(hs, cls):
    m = re.search(r'<table[^>]*class="[^"]*' + cls + r'[^"]*"[^>]*>(.*?)</table>', hs, re.S)
    return m.group(1) if m else ""


def nk_raceid(rid12):
    """netkeibaの12桁 → 手元DSの8桁。日が10以上は A,B,C…（実データの `02261A01` と同じ規則）。"""
    y, place, kai, day, r = rid12[:4], rid12[4:6], int(rid12[6:8]), int(rid12[8:10]), rid12[10:12]
    d = str(day) if day < 10 else chr(ord("A") + day - 10)
    return f"{place}{y[2:]}{kai}{d}{r}"


def parse_result(raw_bytes, rid12):
    """成績ページ → (レース情報, 馬のリスト, 払戻)。"""
    s = raw_bytes.decode("euc_jp", "replace")
    ttl = re.search(r"<title>(.*?)</title>", s, re.S)
    ttl = _txt(ttl.group(1)) if ttl else ""
    m = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", ttl)
    date = (m.group(1), m.group(2).zfill(2), m.group(3).zfill(2)) if m else ("", "", "")
    place = re.search(r"日\s*(\S+?)\d+R", ttl)
    # 条件行: 「ダ右1000m / 天候 : 晴 / ダート : 良 / 発走 : 10:00」
    cond = re.search(r"<(?:diary_snap_cut|p class=\"smalltxt\")>(.*?)</", s, re.S)
    dl = re.search(r"([芝ダ障])[^\d]*(\d{3,4})m", s)
    baba = re.search(r"(?:ダート|芝|障害)\s*:\s*(良|稍重|重|不良)", s)
    race = {
        "rid12": rid12, "raceid": nk_raceid(rid12), "y": date[0], "m": date[1], "d": date[2],
        "place": PLACES.get(rid12[4:6], ""),
        "name": ttl.split("｜")[0].split("|")[0].strip(),
        "surface": dl.group(1) if dl else "", "distance": dl.group(2) if dl else "",
        "cond": {"稍重": "稍", "不良": "不"}.get(baba.group(1), baba.group(1)) if baba else "",
    }
    horses, ids = [], re.findall(r"/horse/(\d+)", s)
    trs = re.findall(r"<tr[^>]*>(.*?)</tr>", _table(s, "race_table_01"), re.S)
    hdr = _cells(trs[0]) if trs else []
    ix = {k: hdr.index(k) for k in
          ["着順", "枠番", "馬番", "馬名", "性齢", "斤量", "騎手", "タイム", "着差", "通過",
           "上り", "単勝", "馬体重", "調教師"] if k in hdr}
    prize_i = next((i for i, h in enumerate(hdr) if h.startswith("賞金")), None)
    for tr in trs[1:]:
        c = _cells(tr)
        if len(c) < 10:
            continue
        hid = re.search(r"/horse/(\d+)", tr)
        jid = re.search(r"/jockey/[a-z/]*?(\d{5})/", tr)
        tid = re.search(r"/trainer/[a-z/]*?(\d{5})/", tr)
        g = lambda k: c[ix[k]] if k in ix and ix[k] < len(c) else ""
        sa = re.match(r"([牡牝セ])(\d+)", g("性齢"))
        bw = re.match(r"(\d+)", g("馬体重"))
        horses.append({
            "finish": g("着順"), "waku": g("枠番"), "umaban": g("馬番"), "name": g("馬名"),
            "sex": sa.group(1) if sa else "", "age": sa.group(2) if sa else "",
            "wtcarry": g("斤量"), "jockey": g("騎手"), "time": g("タイム"),
            "pass": g("通過"), "agari": g("上り"), "odds": g("単勝"),
            "bodywt": bw.group(1) if bw else "",
            "trainer": re.sub(r"^\[.\]\s*", "", g("調教師")).replace("\n", ""),
            "prize": c[prize_i].replace(",", "") if prize_i is not None and prize_i < len(c) else "",
            "horse_id": hid.group(1) if hid else "",
            "jockey_id": jid.group(1) if jid else "", "trainer_id": tid.group(1) if tid else "",
        })
    race["field"] = str(len(horses))
    pay = {}
    for tb in re.findall(r'<table[^>]*class="[^"]*pay_table_01[^"]*"[^>]*>(.*?)</table>', s, re.S):
        for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", tb, re.S):
            cs = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", tr, re.S)
            if len(cs) < 3:
                continue
            kind = _txt(cs[0])
            combos = [x.strip() for x in re.split(r"<br\s*/?>", cs[1]) if _txt(x)]
            pays = [x.strip() for x in re.split(r"<br\s*/?>", cs[2]) if _txt(x)]
            pay[kind] = [(_txt(a), int(_txt(b).replace(",", "") or 0))
                         for a, b in zip(combos, pays)]
    return race, horses, pay


def _sec(t):
    """`0:59.6` → 59.6 秒。空や異常値は None。"""
    m = re.match(r"(\d+):(\d+\.\d+)", t or "")
    return int(m.group(1)) * 60 + float(m.group(2)) if m else None


def margins_from_times(horses):
    """netkeibaの着差は**馬身**（`3.1/2`『クビ』）だが、DSのcol23は**秒差**で意味が違う。
    実データで確認した DS の定義は「1着=−(2着との差) / 2着以降=勝ち馬との差」。
    馬身からは復元できないので**タイムから作り直す**。
    """
    ts = [(_sec(h.get("time")), i) for i, h in enumerate(horses)]
    ok = sorted([(t, i) for t, i in ts if t is not None])
    out = [""] * len(horses)
    if len(ok) < 2:
        return out
    t1, t2 = ok[0][0], ok[1][0]
    for t, i in ts:
        if t is None:
            continue
        out[i] = f"{-(t2 - t1):.1f}" if t == t1 else f"{t - t1:.1f}"
    return out


def to_ds_rows(race, horses):
    """DS 52列レイアウトの行を作る（to_model が読む列だけ埋め、他は空）。"""
    out = []
    mg = margins_from_times(horses)
    for hi, h in enumerate(horses):
        r = [""] * DS_COLS
        r[0], r[1], r[2] = race["y"][2:], race["m"], race["d"]
        r[4] = race.get("place", "")
        r[7] = race["name"]
        r[9] = race["surface"]
        r[11] = race["distance"]
        r[12] = race["cond"]
        r[13] = h["name"]
        r[14], r[15] = h["sex"], h["age"]
        r[16] = h["jockey"]
        r[17] = h["wtcarry"]
        r[18] = race["field"]
        r[20] = h["finish"]
        r[23] = mg[hi]
        ps = [x for x in re.split(r"[-−]", h["pass"]) if x.strip().isdigit()]
        for i in range(4):
            r[28 + i] = ps[i] if i < len(ps) else "0"
        r[32] = h["agari"]
        r[33] = h["bodywt"]
        r[34] = h["trainer"]
        r[36] = h["prize"] or "0"   # DSは入着外を0で埋める。空にするとavg3_prizeがNaNになり不一致
        r[37] = h["horse_id"]          # 血統登録番号の代わり（netkeibaの一意な馬ID）
        r[40] = race["raceid"] + str(h["umaban"]).zfill(2)
        r[43], r[44], r[45] = "", "", ""   # 父/母/母父 は馬ページから後で埋める
        r[48] = h["odds"]
        out.append(r)
    return out


def parse_odds_json(txt):
    """オッズAPIのJSON → ({馬番int: 単勝オッズ}, 取得時点)。"""
    j = json.loads(txt)
    d = j.get("data", {})
    tan = d.get("odds", {}).get("1", {})
    return ({int(k): float(v[0]) for k, v in tan.items() if v and v[0] not in ("", "--")},
            d.get("official_datetime", ""))


def parse_shutuba(raw_bytes, names=None):
    """出馬表 → 馬のリスト（オッズは含まれない）。

    ★出馬表の騎手/調教師は**短縮表記**（「遠藤」「栗東荒川」）で、成績ページのフルネーム
    （「遠藤汰月」「小栗実」）と一致しない。そのままだと学習時のカテゴリと繋がらず未知(-1)になる。
    → **ID(5桁)で引く**。`names` に {"jockey": {id: 名}, "trainer": {id: 名}} を渡すと補完する
    （このマップは成績ページを読むたびに育つ）。
    """
    s = raw_bytes.decode("utf-8", "replace")
    ttl = _txt(re.search(r"<title>(.*?)</title>", s, re.S).group(1))
    tb = _table(s, "ShutubaTable")
    out = []
    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", tb, re.S):
        c = _cells(tr)
        hid = re.search(r"/horse/(\d+)", tr)
        if len(c) < 9 or not c[0].isdigit() or not hid:
            continue
        sa = re.match(r"([牡牝セ])(\d+)", c[4])
        jid = re.search(r"/jockey/[a-z/]*?(\d{5})/", tr)
        tid = re.search(r"/trainer/[a-z/]*?(\d{5})/", tr)
        jid, tid = (jid.group(1) if jid else ""), (tid.group(1) if tid else "")
        nm = names or {}
        out.append({"waku": c[0], "umaban": c[1], "name": c[3],
                    "sex": sa.group(1) if sa else "", "age": sa.group(2) if sa else "",
                    "wtcarry": c[5],
                    "jockey": nm.get("jockey", {}).get(jid) or re.sub(r"^[▲△☆◇★]", "", c[6]),
                    "trainer": nm.get("trainer", {}).get(tid) or c[7],
                    "jockey_id": jid, "trainer_id": tid, "horse_id": hid.group(1)})
    return ttl, out


def main():
    import glob
    p = "data/nk_sample/race.html"
    race, horses, pay = parse_result(open(p, "rb").read(), "202601010201")
    print(f"レース: {race['name']} {race['y']}/{race['m']}/{race['d']} "
          f"{race['surface']}{race['distance']} {race['cond']} {race['field']}頭")
    print(f"raceid変換: 202601010201 → {race['raceid']}")
    print(f"{'着':>3}{'枠':>3}{'番':>3} {'馬名':<12}{'性齢':>5}{'斤':>5}{'騎手':<8}"
          f"{'通過':>7}{'上り':>6}{'単勝':>7}{'体重':>6} horse_id")
    for h in horses[:4]:
        print(f"{h['finish']:>3}{h['waku']:>3}{h['umaban']:>3} {h['name']:<12}"
              f"{h['sex']+h['age']:>5}{h['wtcarry']:>5}{h['jockey']:<8}{h['pass']:>7}"
              f"{h['agari']:>6}{h['odds']:>7}{h['bodywt']:>6} {h['horse_id']}")
    print("\n払戻:", {k: v for k, v in pay.items()})

    # ★検算: 同じレースの時系列オッズ(確定)と単勝が一致するか
    f = f"data/odds_ts/JT{race['raceid']}.CSV"
    if glob.glob(f):
        rows = list(csv.reader(open(f, encoding="shift_jis", errors="replace")))
        fin = rows[-1]
        ts = {i + 1: float(v) for i, v in enumerate(fin[5:5 + int(race["field"])]) if v}
        nk = {int(h["umaban"]): float(h["odds"]) for h in horses if h["odds"]}
        same = sum(1 for k in nk if abs(nk[k] - ts.get(k, -1)) < 1e-9)
        print(f"\n★時系列オッズ(確定)との突合: {same}/{len(nk)}頭 一致"
              f"{'  ← パーサは正しい' if same == len(nk) else '  ⚠不一致あり'}")
        if same != len(nk):
            print("   nk:", nk, "\n   ts:", ts)
    else:
        print(f"\n（{f} が無いので突合はスキップ）")

    q = "data/nk_sample/shutuba.html"
    ttl, sh = parse_shutuba(open(q, "rb").read())
    print(f"\n出馬表: {ttl[:40]} … {len(sh)}頭")
    for h in sh[:3]:
        print("  ", h)


if __name__ == "__main__":
    main()
