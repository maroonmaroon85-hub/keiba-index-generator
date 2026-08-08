"""集めたオッズ板が**確定オッズか、それとも途中の板か**を検算する。**Macで実行**。

★なぜ要るか
　`nk_odds_combo.py --old 202507020501` は過去レースでもオッズを返したが、
　`official_datetime` が **09:57:20**（発走のはるか前）だった。
　　・**確定オッズ**なら → 過去11年を遡って集め、(112)の裾に「安い三連複は避ける」を当てられる。
　　・**朝の板**なら → 締切前に見える値としてはむしろ好都合だが、
　　　**「確定オッズで検証したつもりが朝の板だった」という取り違えは致命的**なので、必ず区別する。
　推測で決めない。**手元の払戻と突き合わせれば確定する**。

★検算の原理
　`data/nk/pay<日付>.csv` には netkeiba から取った**実際の配当**が入っている（100円あたり）。
　的中した組のオッズが板にあるので、**オッズ×100 が配当と一致するか**を見る。
　　一致する → その板は**確定オッズ**（発走時点まで更新されている）
　　一致しない → **途中の板**。ズレの大きさが「朝から発走までにどれだけ動くか」の実測にもなる。

　⚠️ 丸めがあるので厳密一致は求めない。netkeibaのオッズは小数1桁（大きい値は整数）、
　　配当は10円単位に切り捨て。**±1%以内なら一致とみなす**。

使い方（Macで実行）:
    python3 ml/nk_odds_verify.py 20260801          # その日の払戻CSVと突き合わせる
    python3 ml/nk_odds_verify.py 20260801 --n 30   # 見るレース数（既定8・1.5秒間隔なので）
    python3 ml/nk_odds_verify.py 20260801 --type 4 # 馬連で検算する
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PAY_KIND = {4: "馬連", 6: "馬単", 7: "三連複", 8: "三連単", 3: "枠連"}


def load_pay(ymd, kind):
    """pay<日付>.csv → {DS8桁raceid: (組文字列, 配当)}。"""
    path = f"data/nk/pay{ymd}.csv"
    if not os.path.exists(path):
        sys.exit(f"{path} が無い。先に `python3 ml/nk_fetch.py results {ymd}` を回して")
    out = {}
    for i, line in enumerate(open(path, encoding="utf-8")):
        if i == 0:
            continue
        c = line.rstrip("\n").split(",")
        if len(c) < 4 or c[1] != kind:
            continue
        try:
            out[c[0]] = (c[2], int(c[3]))
        except ValueError:
            continue
    return out


def combo_key(combo):
    """'3-5-7' → '030507'。板のキーは馬番2桁ずつの昇順連結。"""
    try:
        ns = sorted(int(x) for x in combo.replace("→", "-").split("-"))
    except ValueError:
        return ""
    return "".join(f"{n:02d}" for n in ns)


def main():
    a = sys.argv[1:]
    if not a:
        sys.exit(__doc__)
    t = 7
    if "--type" in a:
        i = a.index("--type")
        t = int(a[i + 1])
        del a[i:i + 2]
    n = 8
    if "--n" in a:
        i = a.index("--n")
        n = int(a[i + 1])
        del a[i:i + 2]
    ymd = a[0]

    from nk_fetch import race_ids_of_day
    from nk_parse import PLACES, nk_raceid
    from nk_odds_combo import fetch

    kind = PAY_KIND.get(t, "")
    pay = load_pay(ymd, kind)
    ids = [r for r in race_ids_of_day(ymd) if nk_raceid(r) in pay][:n]
    if not ids:
        sys.exit(f"{ymd} で {kind} の払戻と突き合わせられるレースが無い")

    print(f"{kind}（type={t}）の板と、手元の配当を突き合わせる … {len(ids)}レース\n")
    print(f"{'レース':<8}{'的中組':<12}{'板のオッズ':>10}{'×100':>9}"
          f"{'実配当':>9}{'ズレ':>9}  取得時点")
    hit = miss = 0
    for rid in ids:
        combo, yen = pay[nk_raceid(rid)]
        odds, at = fetch(rid, t)
        lab = f"{PLACES.get(rid[4:6],'')}{int(rid[10:12])}R"
        k = combo_key(combo)
        o = odds.get(k)
        if o is None:
            print(f"{lab:<8}{combo:<12}{'—':>10}{'':>9}{yen:>9}{'板に無い':>11}  {at}")
            continue
        d = o * 100 - yen
        ok = abs(d) <= max(10.0, yen * 0.01)
        hit += ok
        miss += not ok
        print(f"{lab:<8}{combo:<12}{o:>10.1f}{o*100:>9.0f}{yen:>9}"
              f"{d:>+9.0f}{'  ✓' if ok else '  ✗'}  {at}")

    print()
    if hit and not miss:
        print(f"★**確定オッズだった**（{hit}/{hit} 一致）")
        print("　→ 過去11年を遡って集められる。(112)の2%裾は約660レース＝1.5秒間隔で17分。")
    elif hit > miss:
        print(f"一致 {hit} / 不一致 {miss} … **おおむね確定オッズ**だが例外がある。上の✗を個別に見ること。")
    else:
        print(f"★**確定オッズではない**（一致 {hit} / 不一致 {miss}）。この板は発走前の途中経過。")
        print("　→ 「確定オッズでの検証」には使えない。ただし**締切前に見える値**なので、")
        print("　　運用側（買うか見送るかの判断）にはむしろこちらが正しい。用途を取り違えないこと。")


if __name__ == "__main__":
    main()
