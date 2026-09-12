"""★★★★開催日の夜の一括 —— 朝に凍結した買い目を4本まとめて答え合わせする

    python3 ml/raceday_night.py 20260913

■ ★入力は3つだけ
| | 中身 | 作るもの |
|---|---|---|
| `data/raceday/<日付>/tickets.json` | ★**朝に凍結した買い目** | `ml/raceday.py`（朝） |
| `data/nk/pay<日付>.csv` | **払戻**（券種・組・配当） | `ml/nk_fetch.py results`（★**Mac**） |
| `data/nk/DSnk<日付>.CSV` | **着順** | 同上 |

★★**朝のファイルだけを読む**。⚠**夜に買い目を作り直さない**——**作り直したら「結果を見た後の買い目」になる**。

■ ★出力
```
data/raceday/<日付>/result.json   ★その日の全明細
data/raceday/ledger.csv           ★通算（1行＝1レース1券種）★ここが唯一の台帳
data/reco/ana_forward.csv         ★穴馬の前向き標本（`ANA_RULE.md` §4-2 の形）
```

■ ★`ana_forward.csv` の列の意味（**`ANA_RULE.md` §4-2 に合わせてある**）
| 列 | 中身 |
|---|---|
| `<券種>_cost` | ★**その券種に投じた額**（P馬単M4点なら400） |
| `<券種>_ret` | ★**総払戻**（★当たらなければ0） |
| `<券種>_pay` | ★**100円あたりの払戻** = `_ret ÷ (_cost/100)`。★**ROI = _pay/100** |
★**`_pay` を持たせる理由**: ★**任意の配分 w の成績が `Σ wᵢ·payᵢ` で後から出せる**から
　（`ANA_RULE.md` §4-2「配分は今決める必要がない」）。⚠**判定に使う配分は §5-2 のまま**。

■ ⚠★★**先に書いておく限界（★数字を読む前に）**
　1. ★**`ANA_RULE.md` §5-2 の主判定は「読むのは年1回」**。⚠**開催日ごとにROIを読まないこと**（判定基準5・43）。
　2. ⚠**穴馬の「記録だけ4本」（P複勝1点・P単勝1点・G馬単M4点・Q三連単A4点）は
　　 　当日モード（`reco_ana_day.py`）が出していないので、ここでも記録できない**。
　　 ★**軸の単勝・複勝は記録する**ので P単勝1点・P複勝1点は復元できるが、
　　 　**G紐・Q紐は当日モードが計算していない**。★**足すかどうかは穴馬セッションの判断**。
　3. ⚠**②甘い軸の三連複は「朝の候補」で採点している**。★**直前に `nk_race.py` で
　　 　確定させた結果と食い違うことがある**。→ `--soft-actual 阪神2R,見送り` で上書きできる。

実行: python3 ml/raceday_night.py 20260913 [--soft-actual "阪神2R,見送り"] [--no-write]
"""
import argparse
import csv
import datetime
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

OUTDIR = "data/raceday"
LEDGER = "data/raceday/ledger.csv"
ANAFWD = "data/reco/ana_forward.csv"

ORDERED = {"馬単", "三連単"}          # ★着順どおりの券種（組を並べ替えない）
KIND_OF = {"馬単M": "馬単", "三連単A": "三連単"}


# ---------------------------------------------------------------- 読む
def load_pays(ymd):
    """→ {raceid: {券種: {組(tuple): 配当}}}。★組は券種ごとに正規化する。"""
    p = os.path.join(ROOT, f"data/nk/pay{ymd}.csv")
    if not os.path.exists(p):
        return None
    out = {}
    with io.open(p, encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            nums = [int(x) for x in r["combo"].split("-") if x.strip().isdigit()]
            if not nums:
                continue
            key = tuple(nums) if r["kind"] in ORDERED else tuple(sorted(nums))
            out.setdefault(r["raceid"], {}).setdefault(r["kind"], {})[key] = int(r["payout"])
    return out


def load_finish(ymd):
    """→ {(raceid, 馬番): 着順}。★DS互換CSVの col41=レースID+馬番 / col21=着順。"""
    p = os.path.join(ROOT, f"data/nk/DSnk{ymd}.CSV")
    if not os.path.exists(p):
        return {}
    out = {}
    with io.open(p, encoding="cp932", errors="replace", newline="") as fh:
        for row in csv.reader(fh):
            if len(row) < 41 or len(row[40]) < 10:
                continue
            try:
                out[(row[40][:8], int(row[40][8:]))] = int(row[20])
            except ValueError:
                continue
    return out


def pay_of(pays, rid, kind, combo):
    """→ (払戻, 判定できたか)。★配当表にそのレースが無ければ (0, False)＝未取得。"""
    t = (pays or {}).get(rid, {}).get(kind)
    if t is None:
        return 0, False
    key = tuple(combo) if kind in ORDERED else tuple(sorted(combo))
    return t.get(key, 0), True


# ---------------------------------------------------------------- 採点
def score_set(pays, rid, kind, combos):
    """1レース1券種。→ (cost, ret, 当たった組, 判定できたか)"""
    cost, ret, hit, known = 100 * len(combos), 0, [], False
    for c in combos:
        v, k = pay_of(pays, rid, kind, c)
        known |= k
        if v:
            ret += v
            hit.append(("-".join(str(x) for x in c), v))
    return cost, ret, hit, known


def row(date, label, rid, section, kind, cost, ret, hit, known, extra=""):
    return {"date": date, "raceid": rid, "label": label, "section": section, "kind": kind,
            "cost": cost, "ret": ret, "pay": round(ret / (cost / 100.0), 1) if cost else 0,
            "hit": "/".join(f"{c}:{v}" for c, v in hit), "known": int(known), "note": extra}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ymd")
    ap.add_argument("--soft-actual", default="",
                    help="甘い軸を直前判定で変えた場合。例 '阪神2R,見送り' / '阪神2R,2-5-7'（カンマ区切り・;で複数）")
    ap.add_argument("--no-write", action="store_true", help="台帳に書かず画面に出すだけ")
    a = ap.parse_args()
    ymd = a.ymd
    date = f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:8]}"
    tp = os.path.join(ROOT, OUTDIR, ymd, "tickets.json")
    if not os.path.exists(tp):
        sys.exit(f"⚠{OUTDIR}/{ymd}/tickets.json が無い。★朝に `python3 ml/raceday.py {ymd}` を走らせていない")
    T = json.load(open(tp, encoding="utf-8"))
    pays, fin = load_pays(ymd), load_finish(ymd)

    print("=" * 78)
    print(f"★★★★{date} の開催日一括（夜）")
    print("=" * 78)
    print(f"　朝の凍結 {T.get('generated_at','?')}　(git {T.get('git_head','')[:8]})")
    if pays is None:
        print(f"\n⚠★**data/nk/pay{ymd}.csv が無い**。★**Macで** `sh raceday_results.sh {ymd}` "
              "を先に走らせて push すること")
        return 1
    if not fin:
        print(f"　⚠**data/nk/DSnk{ymd}.CSV が無いので着順は出せない**（払戻だけで採点する）")

    override = {}
    for chunk in filter(None, a.soft_actual.split(";")):
        lab, _, v = chunk.partition(",")
        override[lab.strip()] = v.strip()

    rows, unknown = [], 0

    # ---- ① 本命 枠連
    print("\n■ ① 本命 枠連（軸枠×紐枠1）")
    for x in T["waku"]["wakuren"] if T.get("waku") else []:
        combos = [[int(n) for n in c.split("-")] for c in x["combos"]]
        c, r, h, k = score_set(pays, x["raceid"], "枠連", combos)
        unknown += (not k)
        rows.append(row(date, x["label"], x["raceid"], "本命", "枠連", c, r, h, k))
        print(f"　　{x['label']:>9}　{' / '.join(x['combos']):<12}"
              f"　{c:>5}円 → {r:>7,}円 {'★的中' if r else ''}{'' if k else '  ⚠配当未取得'}")

    # ---- ② 甘い軸の三連複
    print("\n■ ② 甘い軸の三連複　⚠**朝の候補で採点している**")
    for x in T["waku"]["soft_sanrenpuku"] if T.get("waku") else []:
        act = override.get(x["label"], x["combo"])
        if act in ("見送り", "見送", "skip", "-"):
            print(f"　　{x['label']:>9}　★**直前判定で見送り**（--soft-actual 指定）— 記録しない")
            continue
        note = "" if act == x["combo"] else f"直前に {x['combo']}→{act} へ変更"
        combos = [[int(n) for n in act.split("-")]]
        c, r, h, k = score_set(pays, x["raceid"], "三連複", combos)
        unknown += (not k)
        rows.append(row(date, x["label"], x["raceid"], "甘い軸", "三連複", c, r, h, k, note))
        print(f"　　{x['label']:>9}　{act:<12}　{c:>5}円 → {r:>7,}円 "
              f"{'★的中' if r else ''}{'' if k else '  ⚠配当未取得'}{'　' + note if note else ''}")

    # ---- ③ 穴馬（自分用）
    print("\n■ ③ 穴馬（自分用）　★8点800円/レース")
    ana_rows = []
    for rc in T.get("ana", {}).get("races", []):
        rid, ax = rc["raceid"], rc["axis"]
        f = fin.get((rid, ax))
        line = {"date": date, "raceid": rid, "label": rc["label"], "axis": ax,
                "axis_finish": f if f is not None else ""}
        print(f"　　{rc['label']}　軸 {ax}番"
              + (f"　→ ★**{f}着**" if f is not None else "　（着順不明）"))
        for t in rc["tickets"]:
            kind = KIND_OF.get(t["kind"], t["kind"])
            c, r, h, k = score_set(pays, rid, kind, t["combos"])
            unknown += (not k)
            rows.append(row(date, rc["label"], rid, "穴馬", t["label"], c, r, h, k))
            line[f"{t['label']}_cost"] = c
            line[f"{t['label']}_ret"] = r
            line[f"{t['label']}_pay"] = round(r / (c / 100.0), 1) if c else 0
            print(f"　　　　{t['label']:<12}{c:>5}円 → {r:>7,}円 "
                  f"{'★的中 ' + h[0][0] + ' ' + format(h[0][1], ',') + '円' if h else ''}"
                  f"{'' if k else '  ⚠配当未取得'}")
        # ★軸の単勝・複勝（**買っていないが記録する**＝`ANA_RULE.md` §4-2 の「記録だけ」のうち2本）
        for kind, col in (("単勝", "P単勝1点"), ("複勝", "P複勝1点")):
            v, k = pay_of(pays, rid, kind, [ax])
            line[f"{col}_cost"], line[f"{col}_ret"], line[f"{col}_pay"] = 100, v, float(v)
            rows.append(row(date, rc["label"], rid, "穴馬(記録だけ)", col, 100, v,
                            [(str(ax), v)] if v else [], k))
        print(f"　　　　（記録だけ）軸の単勝 {line['P単勝1点_ret']:,}円 / "
              f"複勝 {line['P複勝1点_ret']:,}円")
        ana_rows.append(line)
    if not T.get("ana", {}).get("races"):
        print("　　（朝の時点で該当0本）")

    # ---- ④ SNS（印）
    print("\n■ ④ SNS（印）　★◎の着順と単複だけ")
    for rc in T.get("sns", {}).get("races", []):
        ax = rc["axis"]
        f = fin.get((rc["raceid"], ax))
        tan, _ = pay_of(pays, rc["raceid"], "単勝", [ax])
        fuku, k = pay_of(pays, rc["raceid"], "複勝", [ax])
        unknown += (not k)
        rows.append(row(date, rc["label"], rc["raceid"], "SNS", "◎複勝", 100, fuku,
                        [(str(ax), fuku)] if fuku else [], k,
                        f"着順{f if f is not None else '?'} 単勝{tan}"))
        print(f"　　{rc['label']}　◎{ax}　"
              + (f"→ **{f}着**" if f is not None else "→ （着順不明）")
              + (f"　複勝{fuku:,}円★" if fuku else "　―")
              + (f"　単勝{tan:,}円★" if tan else ""))
    if not T.get("sns", {}).get("races"):
        print("　　（朝の時点で該当0本＝「見送り」を投稿した日）")

    # ---- まとめ
    buy = [r for r in rows if r["section"] in ("本命", "甘い軸", "穴馬")]
    c, r_ = sum(x["cost"] for x in buy), sum(x["ret"] for x in buy)
    print("\n" + "=" * 78)
    print(f"★**この日の自分の買い　{c:,}円 → {r_:,}円**（差引 {r_ - c:+,}円）")
    if unknown:
        print(f"⚠**{unknown}件が配当未取得**。★`python3 ml/nk_fetch.py results {ymd} --refresh` "
              "をMacで叩いてから、もう一度この採点を走らせること")
    print("⚠★**1日の数字には意味が無い**。★**穴馬の主判定は「読むのは年1回」**"
          "（`ANA_RULE.md` §5-2・判定基準43）")

    if a.no_write:
        print("\n（--no-write なので台帳に書いていない）")
        return 0

    os.makedirs(os.path.join(ROOT, OUTDIR, ymd), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "data/reco"), exist_ok=True)
    json.dump({"date": ymd, "scored_at": datetime.datetime.now().astimezone().isoformat(
        timespec="seconds"), "rows": rows, "ana": ana_rows},
        open(os.path.join(ROOT, OUTDIR, ymd, "result.json"), "w", encoding="utf-8"),
        ensure_ascii=False, indent=1)
    append(LEDGER, rows, ymd)
    if ana_rows:
        append(ANAFWD, ana_rows, ymd)
    print(f"\n★保存: {OUTDIR}/{ymd}/result.json ・ {LEDGER}"
          + (f" ・ {ANAFWD}" if ana_rows else ""))
    print(f"　　git add {OUTDIR} {ANAFWD} data/nk && git commit -m '{date} の答え合わせ'")
    return 0


def append(path, newrows, ymd):
    """★同じ日の行があれば**入れ替える**（走らせ直しても二重にならない）。"""
    p = os.path.join(ROOT, path)
    old, cols = [], []
    if os.path.exists(p):
        with io.open(p, encoding="utf-8", newline="") as fh:
            rd = csv.DictReader(fh)
            cols = list(rd.fieldnames or [])
            old = [r for r in rd if r.get("date", "").replace("-", "") != ymd]
    for r in newrows:
        for k in r:
            if k not in cols:
                cols.append(k)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with io.open(p, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in old + newrows:
            w.writerow({k: r.get(k, "") for k in cols})


if __name__ == "__main__":
    sys.exit(main())
