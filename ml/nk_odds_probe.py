"""netkeibaのオッズAPIが**単勝以外**も返すかを1レースだけ確かめる偵察スクリプト。

★なぜ要るか
　`nk_fetch.py` は `type=1`（単勝）しか叩いておらず、`nk_parse.parse_odds_json` も
　`data.odds["1"]` を決め打ちしている。**三連複や枠連が同じAPIで取れるなら**:
　　・(112)の買い目について「三連複のオッズが安いレースは避ける」を**検証できるようになる**
　　　（今は的中した組の配当しか手元に無いので、外れた組が何倍だったかが分からず検証不能）
　　・宿題3「枠連の発走前オッズを集める」も同時に片付く
　だが**レスポンスの形が分からない**ので、推測でパーサを書くと必ず外す。まず実物を見る。

★負荷への配慮（(70)⑤の方針を守る）
　叩くのは**1レース × 指定した券種だけ**。1.5秒間隔。**素のJSONをそのまま保存**するので、
　同じレースで何度も叩き直す必要が無い。

使い方（Macで実行）:
    python3 ml/nk_odds_probe.py 202607010511            # netkeibaのrace_id（12桁）
    python3 ml/nk_odds_probe.py 20260808 中京 5         # 日付・場・R でも指定できる
    python3 ml/nk_odds_probe.py 20260808 中京 5 6       # 券種を絞る（6=三連複）

netkeibaの type の割り当ては公表されていないので、**1〜9を順に叩いて中身を見る**。
保存先: data/nk_cache/probe_<race_id>_type<N>.json
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def summarize(txt, t):
    """レスポンスの構造だけを要約して出す（中身を全部出すと読めないので）。"""
    try:
        j = json.loads(txt)
    except (ValueError, TypeError):
        print(f"  type={t}  JSONとして読めない（先頭80字: {txt[:80]!r}）")
        return
    d = j.get("data")
    if not isinstance(d, dict):
        print(f"  type={t}  data が dict でない（{type(d).__name__}）＝この券種は返っていない")
        return
    od = d.get("odds")
    if not isinstance(od, dict):
        print(f"  type={t}  odds が dict でない＝発売前か未対応")
        return
    print(f"  type={t}  official_datetime={d.get('official_datetime','')}")
    for key, val in od.items():
        if not isinstance(val, dict) or not val:
            print(f"    odds[{key!r}] … 空 or dictでない")
            continue
        items = list(val.items())[:3]
        print(f"    odds[{key!r}] … {len(val)}件  例: " +
              " / ".join(f"{k!r}→{v!r}" for k, v in items))


def main():
    a = sys.argv[1:]
    if not a:
        sys.exit(__doc__)
    types = [int(x) for x in a if x.isdigit() and len(x) == 1]
    a = [x for x in a if not (x.isdigit() and len(x) == 1)]
    if not types:
        types = list(range(1, 10))

    from nk_fetch import CACHE, get, race_ids_of_day
    from nk_parse import PLACES

    if len(a) == 1 and len(a[0]) == 12:
        rid = a[0]
    else:
        ymd = a[0]
        place = next((x for x in a[1:] if not x.isdigit()), None)
        rno = next((int(x) for x in a[1:] if x.isdigit()), None)
        ids = race_ids_of_day(ymd)
        sel = [r for r in ids
               if (place is None or PLACES.get(r[4:6], "") == place)
               and (rno is None or int(r[10:12]) == rno)]
        if not sel:
            sys.exit("該当レースが無い")
        rid = sel[0]
    print(f"race_id={rid}　券種 {types} を順に叩く\n")

    for t in types:
        key = f"probe_{rid}_type{t}.json"
        b = get(f"https://race.netkeiba.com/api/api_get_jra_odds.html"
                f"?type={t}&locale=ja&race_id={rid}&action=init", key,
                referer=f"https://race.netkeiba.com/odds/index.html?race_id={rid}")
        if not b:
            print(f"  type={t}  取得できなかった")
            continue
        summarize(b.decode("utf-8", "replace"), t)
    print(f"\n素のJSONは {CACHE}/probe_{rid}_type*.json に保存した。")
    print("★この出力をそのまま貼ってもらえれば、パーサを書きます。")


if __name__ == "__main__":
    main()
