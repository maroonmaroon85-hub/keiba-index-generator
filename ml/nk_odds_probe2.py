"""(115) オッズAPIが400になった件 — **新しいパラメータの当て方を1回で決める**。**Macで実行**。

★経緯
　2026-08-09に `?type=1&locale=ja&race_id=…&action=init` が **HTTP 400** を返し始めた。
　サイト自体は正常にオッズを表示していたので、**締め出しではなく仕様変更**と分かった。
　ブラウザの実物（2026-08-11採取）:
```
?callback=jQuery11200973785211546272_1786417104845&pid=api_get_jra_odds
&input=UTF-8&output=jsonp&race_id=202607020608&type=1&action=init
&sort=odds&compress=1&_=1786417104846
```
　→ `locale=ja` が消え、`pid`/`input`/`output`/`callback` が増えている。**JSONP**になった。

★なぜ推測で直さないか
　`compress=1` が付いている。11頭で0.7 kBは小さく、**圧縮された別形式**で返っている疑いがある。
　その場合パーサを書き直すことになる。**1つずつ試すと無駄に叩く**ので、
　**候補4つを1回で当てて、生の先頭だけ見る**。(115)の教訓＝調べるときも負荷を考える。

★試す4つ（この順に意味がある）
　A. ブラウザと同一（JSONP・compress=1）      … 確実に通るはずの基準
　B. JSONPのまま compress=0                  … 従来のJSONが返るなら**パーサ据え置きで済む**
　C. output=json・callbackなし・compress=0    … JSONPを剥がす手間すら要らない形
　D. 旧URLに pid/input だけ足す               … 最小の修正で済むか

使い方（Macで実行）:
    python3 ml/nk_odds_probe2.py                      # 既定のレース・type=1
    python3 ml/nk_odds_probe2.py 202607020608 7       # レースと券種を指定
"""
import sys
import time
import urllib.error
import urllib.request

BASE = "https://race.netkeiba.com/api/api_get_jra_odds.html"
UA = "Mozilla/5.0 (compatible; personal-research/1.0)"
CB = "jQuery11200973785211546272_1786417104845"


def variants(rid, t):
    now = int(time.time() * 1000)
    common = f"pid=api_get_jra_odds&input=UTF-8&race_id={rid}&type={t}&action=init&sort=odds"
    return [
        ("A ブラウザと同一(JSONP/compress=1)",
         f"{BASE}?callback={CB}&{common}&output=jsonp&compress=1&_={now}"),
        ("B JSONPのまま compress=0",
         f"{BASE}?callback={CB}&{common}&output=jsonp&compress=0&_={now+1}"),
        ("C output=json・callbackなし",
         f"{BASE}?{common}&output=json&compress=0&_={now+2}"),
        ("D 旧URL＋pid/input だけ",
         f"{BASE}?pid=api_get_jra_odds&input=UTF-8&type={t}&race_id={rid}&action=init"),
    ]


def main():
    a = sys.argv[1:]
    rid = a[0] if a else "202607020608"
    t = int(a[1]) if len(a) > 1 else 1
    print(f"race_id={rid} type={t}　4通りを試す（1.5秒間隔）\n")
    ok = []
    for name, url in variants(rid, t):
        time.sleep(1.5)
        req = urllib.request.Request(url, headers={
            "User-Agent": UA,
            "Referer": f"https://race.netkeiba.com/odds/index.html?type=b1&race_id={rid}"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                b = r.read()
            s = b.decode("utf-8", "replace")
            print(f"★{name}  {len(b)}バイト  HTTP {r.status}")
            print(f"    先頭300字: {s[:300]}")
            ok.append(name)
        except urllib.error.HTTPError as e:
            print(f" {name}  HTTP {e.code} {e.reason}")
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            print(f" {name}  失敗: {e}")
        print()
    print(f"通ったのは: {ok or 'なし'}")
    print("★この出力をそのまま貼れば、通った形に合わせて nk_fetch.py と")
    print("　nk_odds_combo.py / nk_odds_bulk.py を直します。")


if __name__ == "__main__":
    main()
