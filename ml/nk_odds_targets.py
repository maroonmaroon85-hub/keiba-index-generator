"""(113) 遡って集めるべき**過去レースのrace_id一覧**を作る（cloud側で実行・通信しない）。

★なぜ要るか
　`nk_odds_combo.py` で過去のオッズ板を集められると分かったが、**全レースを集めるのは無駄**。
　(112)で意味があったのは **甘い軸の2%裾**（1番人気の複勝の期待払戻 E ≤ 86円）だけで、
　これは11年で約660レースしかない。**そこだけ集めれば1.5秒間隔で17分**で済む。
　全部（46,917R）だと**20時間**かかる。桁が違うので、先に対象を絞る。

★race_id の変換（手元8桁 → netkeiba12桁）
　手元: `place(2) + yy(2) + kai(1) + day(1文字) + r(2)`  例 `02131101`
　netkeiba: `yyyy + place(2) + kai(2) + day(2) + r(2)`   例 `201302110 1` → `202513110 1`
　日が10以上は手元側で A,B,C… になっている（`nk_parse.nk_raceid` の逆変換）。

★出力
　`data/nk_odds/targets_<裾>.txt` … 1行1つの12桁race_id（古い順）。
　`nk_odds_combo.py --list <file>` がこれを読んで順に集める。

使い方:
    python3 ml/nk_odds_targets.py            # 2%裾（既定・E≤86円）
    python3 ml/nk_odds_targets.py --tier 5   # 5%裾（E≤90円）
    python3 ml/nk_odds_targets.py --from 2015
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OUT = "data/nk_odds"


def to_nk_raceid(rid8):
    """手元8桁 → netkeiba12桁。`nk_parse.nk_raceid` の逆。無効なら ''。"""
    if len(rid8) != 8:
        return ""
    place, yy, kai, day, r = rid8[:2], rid8[2:4], rid8[4], rid8[5], rid8[6:8]
    if not (place.isdigit() and yy.isdigit() and kai.isdigit() and r.isdigit()):
        return ""
    if not 1 <= int(place) <= 10:      # JRA10場のみ（地方が混ざっていたら落とす）
        return ""
    if day.isdigit():
        d = int(day)
    elif "A" <= day.upper() <= "Z":
        d = ord(day.upper()) - ord("A") + 10
    else:
        return ""
    return f"20{yy}{place}{int(kai):02d}{d:02d}{r}"


def main():
    a = sys.argv[1:]
    tier = 0.02
    if "--tier" in a:
        i = a.index("--tier")
        tier = int(a[i + 1]) / 100.0
        del a[i:i + 2]
    y0 = 0
    if "--from" in a:
        i = a.index("--from")
        y0 = int(a[i + 1])
        del a[i:i + 2]

    import soft_axis as SA
    from audit_crosspool import load_races

    thr = next((th for tv, th, _ in SA.TIERS if abs(tv - tier) < 1e-9), None)
    if thr is None:
        sys.exit(f"裾 {tier:.0%} は soft_axis.TIERS に無い（{[t[0] for t in SA.TIERS]}）")

    races = load_races()
    sel, skipped, byyear = [], 0, {}
    for r in races:
        if r["year"] < y0:
            continue
        odds = [o for _, o, _ in r["horses"]]
        try:
            _, e, _ = SA.axis_expect(odds)
        except (ValueError, ZeroDivisionError):
            continue
        if e > thr:      # ★E は小さいほど軸が強い（=3着以内確率が高い）。裾は上限で切る
            continue
        rid12 = to_nk_raceid(r["rid"])
        if not rid12:
            skipped += 1
            continue
        sel.append((rid12, r["year"]))
        byyear[r["year"]] = byyear.get(r["year"], 0) + 1

    sel.sort()
    os.makedirs(OUT, exist_ok=True)
    path = f"{OUT}/targets_{int(tier*100)}pct.txt"
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("".join(f"{rid}\n" for rid, _ in sel))

    n_all = sum(1 for r in races if r["year"] >= y0)
    print(f"裾 {tier:.0%}（軸の複勝の期待払戻 E ≤ {thr:.0f}円＝軸の3着以内確率 {80/thr:.1%} 以上）")
    print(f"  対象 {len(sel)} / 全 {n_all} レース（{len(sel)/max(n_all,1):.2%}）"
          + (f"  ※race_id変換に失敗 {skipped}" if skipped else ""))
    print(f"  → {path}")
    print("\n年ごとの本数:")
    for y in sorted(byyear):
        print(f"   {y}  {byyear[y]:>4}")
    mins = len(sel) * 1.5 / 60
    print(f"\n収集にかかる時間: 約 {mins:.0f} 分（1.5秒間隔）")
    print(f"  Macで: python3 ml/nk_odds_combo.py --list {path}")


if __name__ == "__main__":
    main()
