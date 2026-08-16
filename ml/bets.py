"""★★実際に買った馬券だけを記録して採点する（2026-08-16新設）

★★なぜ要るか（ユーザー指摘 2026-08-16「全レースは買ってない」）
　⚠**それまでの「実運用記録」は `data/reco/*.json` の推奨レースを全部買った前提で集計していた**。
　**実際にはその一部しか買っていない**ので、**あの累計（54R・106.2%）は「実際の収支」ではない**。
　★**正しくは「推奨どおり全部買った場合の値」**。**混同すると自分の成績を誤認する**。
　→ ★**実際に買ったものだけを別ファイルに残す**。**これが唯一の「本当の記録」になる**。

★★台帳の形（`data/bets/bets.csv`・**追記していくだけ**・Excelでも開ける）
```
date,place,r,kind,combo,yen,src,note
20260816,札幌,11,枠連,4-8,100,pm,
```
・`date` … `YYYYMMDD`
・`place` … 場名（札幌/新潟/中京…）。`r` … レース番号
・`kind` … 券種（枠連/複勝/三連複/単勝/馬連/ワイド/馬単/三連単）
・`combo` … 買い目。`4-8` のようにハイフン区切り。**複勝・単勝は馬番1つ**
・`yen` … 賭け金（円）
・`src` … どの判定で買ったか。`am`(朝) / `pm`(取り直し) / `manual`(自分の判断) / `soft`(甘い軸)
　★**`manual` を必ず区別する**——**ルール外で買ったものを混ぜると(112)の標本が汚れる**
・`note` … 自由記入

★★使い方
```bash
# 1) 買ったものを足す（1本ずつ・Macで実行可。標準ライブラリだけ）
python3 ml/bets.py add 20260816 札幌 11 枠連 4-8 100 pm

# 2) 採点する（data/nk/pay*.csv と突き合わせる）
python3 ml/bets.py score              # 全期間
python3 ml/bets.py score 20260816     # その日だけ
```

⚠**採点は `data/nk/pay*.csv` があるレースだけ**。結果を取る前に score しても出ない。
⚠**同じ日・場・R・券種・買い目の重複は自動で弾く**（二重記録の防止）。

★★**推奨との違いを必ず出す**: `score` は
　**「実際に買った分」と「その日の推奨を全部買った場合」の両方**を並べる。
　→ **「買わなかったレースが当たっていたか」も分かる**（**選ばなかったことの評価**）。
　⚠**ただしそれを見て買い方を変えると後知恵**になる。**記録として見るだけ**。
"""
import csv
import glob
import os
import sys

PATH = "data/bets/bets.csv"
COLS = ["date", "place", "r", "kind", "combo", "yen", "src", "note"]
ORDERED = {"馬単", "三連単"}


def load_bets(path=PATH):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8", newline="") as fh:
        return [r for r in csv.DictReader(fh) if r.get("date")]


def save_bets(rows, path=PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in COLS})


def load_pays(pattern="data/nk/pay*.csv"):
    """{raceid: {券種: {組: 配当}}}。nk_score と同じ読み方。"""
    out = {}
    for p in sorted(glob.glob(pattern)):
        with open(p, encoding="utf-8", newline="") as fh:
            for r in csv.DictReader(fh):
                nums = [int(x) for x in r["combo"].split("-") if x.strip().isdigit()]
                if not nums:
                    continue
                key = tuple(nums) if r["kind"] in ORDERED else tuple(sorted(nums))
                out.setdefault(r["raceid"], {}).setdefault(r["kind"], {})[key] = int(r["payout"])
    return out


def race_index():
    """(date, place, r) → raceid。`data/nk/entries*.json` から作る。"""
    import json
    idx = {}
    for p in sorted(glob.glob("data/nk/entries*.json")):
        d = os.path.basename(p)[7:15]
        try:
            for e in json.load(open(p, encoding="utf-8")):
                idx[(d, e["place"], int(e["r"]))] = e["raceid"]
        except Exception:
            continue
    return idx


def cmd_add(args):
    if len(args) < 6:
        sys.exit("使い方: python3 ml/bets.py add <日付> <場> <R> <券種> <買い目> <金額> [src] [note]")
    date, place, r, kind, combo, yen = args[:6]
    src = args[6] if len(args) > 6 else "manual"
    note = args[7] if len(args) > 7 else ""
    combo = "-".join(str(int(x)) for x in combo.replace("−", "-").split("-") if x.strip())
    rows = load_bets()
    key = (date, place, str(int(r)), kind, combo)
    if any((x["date"], x["place"], x["r"], x["kind"], x["combo"]) == key for x in rows):
        print(f"⚠既に記録済み: {date} {place}{r}R {kind} {combo} → 追加しない")
        return
    rows.append(dict(date=date, place=place, r=str(int(r)), kind=kind,
                     combo=combo, yen=str(int(yen)), src=src, note=note))
    rows.sort(key=lambda x: (x["date"], x["place"], int(x["r"]), x["kind"], x["combo"]))
    save_bets(rows)
    print(f"★追加: {date} {place}{r}R {kind} {combo} {yen}円 ({src})　→ 台帳 {len(rows)}件")


def cmd_score(args):
    day = args[0] if args else None
    rows = [r for r in load_bets() if not day or r["date"] == day]
    if not rows:
        sys.exit(f"台帳に{'その日の' if day else ''}記録が無い（{PATH}）")
    pays, idx = load_pays(), race_index()
    print(f"★実際に買った分の採点（{PATH}{'・' + day if day else '・全期間'}）\n")
    per_kind, per_day, miss = {}, {}, 0
    print(f"{'日付':>9}{'場R':>9}{'券種':>7}{'買い目':>12}{'賭け':>7}{'払戻':>9}")
    for r in sorted(rows, key=lambda x: (x["date"], x["place"], int(x["r"]))):
        rid = idx.get((r["date"], r["place"], int(r["r"])))
        yen = int(r["yen"])
        p = None
        if rid and rid in pays:
            nums = [int(x) for x in r["combo"].split("-")]
            key = tuple(nums) if r["kind"] in ORDERED else tuple(sorted(nums))
            p = pays[rid].get(r["kind"], {}).get(key, 0)
        if p is None:
            miss += 1
            print(f"{r['date']:>9}{r['place'] + r['r'] + 'R':>9}{r['kind']:>7}"
                  f"{r['combo']:>12}{yen:>7}{'結果待ち':>9}")
            continue
        ret = p * yen // 100
        per_kind.setdefault(r["kind"], [0, 0, 0, 0])
        a = per_kind[r["kind"]]
        a[0] += yen; a[1] += ret; a[2] += 1; a[3] += int(ret > 0)
        b = per_day.setdefault(r["date"], [0, 0])
        b[0] += yen; b[1] += ret
        print(f"{r['date']:>9}{r['place'] + r['r'] + 'R':>9}{r['kind']:>7}"
              f"{r['combo']:>12}{yen:>7}{ret:>9,}{'  ★的中' if ret > 0 else ''}")

    print(f"\n{'券種':>8}{'点数':>7}{'的中':>6}{'購入':>10}{'払戻':>10}{'収支':>10}{'回収率':>9}")
    tc = tr = 0
    for k, (c, v, n, h) in sorted(per_kind.items()):
        tc += c; tr += v
        print(f"{k:>8}{n:>7}{h:>6}{c:>10,}{v:>10,}{v-c:>+10,}{100*v/max(c,1):>8.1f}%")
    print(f"{'合計':>8}{sum(a[2] for a in per_kind.values()):>7}"
          f"{sum(a[3] for a in per_kind.values()):>6}{tc:>10,}{tr:>10,}{tr-tc:>+10,}"
          f"{100*tr/max(tc,1):>8.1f}%")
    if miss:
        print(f"⚠結果待ち {miss}件（`python3 ml/nk_fetch.py results <日付>` を先に）")
    print("\n■ 日別")
    for d, (c, v) in sorted(per_day.items()):
        print(f"　{d}  購入{c:>7,}円 払戻{v:>8,}円 収支{v-c:>+8,}円 回収率{100*v/max(c,1):>7.1f}%")
    print("\n⚠**1日の数字に意味は無い**（枠連1点の的中率は約20%）。**長期の目安は85%**。")
    print("⚠**これは「実際に買ったもの」だけ**。`ml/nk_score.py` は**推奨を全部買った場合**を出す。")
    print("　**2つは別物**。混同しないこと（2026-08-16にこの区別を導入した）。")


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("add", "score"):
        sys.exit(__doc__.split("★★使い方")[1].split("⚠")[0].strip())
    (cmd_add if sys.argv[1] == "add" else cmd_score)(sys.argv[2:])


if __name__ == "__main__":
    main()
