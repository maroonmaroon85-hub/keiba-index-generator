"""レース1本だけ取り直して「甘い軸の三連複」を判定する。**標準ライブラリのみ・Macで動く**。

★何のためか
　(112運用②)で **「朝で候補、直前で確定」** が最良と実測した。
　だが `nk_fetch.py entries` は**その日の全レース**を取り直すので、直前確認には重い。
　これは**指定した1レースだけ**オッズを取り直して判定する。1リクエストで済む。

★使い方（3通り。どれでもよい）
```
python3 ml/nk_race.py 20260809 札幌 3      # 日付・場名・R
python3 ml/nk_race.py 20260809 3           # 日付・R（その日の全場の3Rを出す）
python3 ml/nk_race.py 20260809             # 日付だけ（その日の全レースを判定・オッズは全部取り直す）
python3 ml/nk_race.py --odds 1.4 4.5 7.2 … # ★ネット不要。手打ちしたオッズで判定だけする
```

★判定の中身は `ml/soft_axis.py` と同じ（オッズだけで決まる）
　軸＝1番人気 ／ 紐＝2・3番人気 → 三連複1点。
　買うのは「軸の複勝の期待払戻 E」が閾値以下のレースだけ（≤86円=裾2% / ≤90円=5% / ≤94円=10%）。

⚠**この買い目は「期待値がプラス」ではない**（(112)）。
　ROI 96.0% は2%裾でだけ跳ねており単調でなく、CIは[76.0,116.0]。Dで測ると情報量は足りていない。
　**承知の上で買うもの**。詳細は `HANDOFF.md` の(111)(112)。
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import soft_axis as SA


def judge(umabans, odds, label=""):
    """1レース分の判定を1行で出す。→ 買うなら True。"""
    r = SA.recommend(umabans, odds)
    if r is None:
        print(f"{label}  判定不可（頭数不足かオッズ欠損）")
        return False
    tag = f"軸{r['axis']}番({min(odds):.1f}倍) 複勝の期待払戻{r['e_axis']:.0f}円"
    if r["buy"]:
        print(f"★買う  {label}  三連複 {r['sanrenpuku']}（1点100円）  {tag}"
              f"  裾{int(r['tier']*100)}%")
        return True
    print(f"  見送り {label}  {tag}（閾値{SA.TIERS[-1][1]:.0f}円超）")
    return False


def main():
    a = sys.argv[1:]
    if not a:
        sys.exit(__doc__)

    # ── オッズ手打ちモード（ネット不要）──
    if a[0] == "--odds":
        odds = [float(x) for x in a[1:]]
        if len(odds) < 3:
            sys.exit("--odds のあとに単勝オッズを頭数ぶん並べる")
        judge(list(range(1, len(odds) + 1)), odds, "手入力")
        return

    from nk_fetch import CACHE, get, names_cache, race_ids_of_day
    from nk_parse import PLACES, parse_odds_json, parse_shutuba

    ymd = a[0]
    place = next((x for x in a[1:] if not x.isdigit()), None)
    rno = next((int(x) for x in a[1:] if x.isdigit()), None)

    ids = race_ids_of_day(ymd)
    if not ids:
        sys.exit("レース一覧が取れなかった（日付が違う／まだ公開されていない）")
    # 場名・R で絞る。★race_id は YYYY+場(2)+回(2)+日(2)+R(2)
    sel = [rid for rid in ids
           if (place is None or PLACES.get(rid[4:6], "") == place)
           and (rno is None or int(rid[10:12]) == rno)]
    if not sel:
        av = sorted({PLACES.get(r[4:6], "?") for r in ids})
        sys.exit(f"該当なし。その日の開催: {' '.join(av)}")

    nm = names_cache()
    n_buy = 0
    for rid in sel:
        b = get(f"https://race.netkeiba.com/race/shutuba.html?race_id={rid}",
                f"shutuba_{rid}.html")
        if not b:
            print(f"  {rid} 出馬表が取れない")
            continue
        _, hs, meta = parse_shutuba(b, nm)
        # ★オッズは毎回取り直す（キャッシュのキーに時刻を入れる）。これが直前確認の肝
        key = f"odds_{rid}_{int(time.time())}.json"
        ob = get(f"https://race.netkeiba.com/api/api_get_jra_odds.html"
                 f"?type=1&locale=ja&race_id={rid}&action=init", key,
                 referer=f"https://race.netkeiba.com/race/shutuba.html?race_id={rid}")
        odds, at = parse_odds_json(ob.decode("utf-8", "replace")) if ob else ({}, "")
        if not odds:
            try:
                os.remove(os.path.join(CACHE, key))
            except OSError:
                pass
            print(f"  {PLACES.get(rid[4:6],'')}{int(rid[10:12])}R "
                  f"単勝オッズがまだ出ていない（発売前）")
            continue
        um = [int(h["umaban"]) for h in hs if odds.get(int(h["umaban"]))]
        od = [odds[u] for u in um]
        lab = f"{PLACES.get(rid[4:6],'')}{int(rid[10:12])}R {meta['name'][:14]} @{at}"
        if judge(um, od, lab):
            n_buy += 1
    if len(sel) > 1:
        print(f"\n{len(sel)}レース中 {n_buy}レースが該当")
    print("⚠この買い目は期待値がプラスではない（(112)）。HANDOFF.md の(111)(112)を読むこと。")


if __name__ == "__main__":
    main()
