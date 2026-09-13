"""★★父・母父の空欄を、アーカイブの血統登録番号から埋める（★空欄だけ・既存値は触らない）

    python3 ml/nk_fill_ped.py --check     ★数えるだけ（書かない）
    python3 ml/nk_fill_ped.py             ★埋める

■ ★なぜ要るか
　★`ml/nk_fetch.py pedigree` は **netkeiba の10桁 horse_id でしか馬ページを引けない**。
　⚠**`ml/nk_link.py` で8桁の血統登録番号に名寄せされた行は、その経路から永久に外れる**。
　→ ★**そこを、手元のアーカイブ（ルートの `*.CSV`・血統登録番号がキー）から埋める**。

■ ⚠★**これは `nk_fetch.py pedigree` の代わりではない**
　★**10桁のまま（＝アーカイブに居ない新馬）は引けない**。**そちらは今までどおり `pedigree`**。
　★**実行順は変えない**: `results` → `pedigree` → `nk_link` →（必要なら）**この穴埋め**。

■ ★★守ること（**過去2回、血統を消す事故が起きているため**）
| | ★**規則** |
|---|---|
| **1** | ★**空欄にしか書かない**。**既存の値を上書きしない**（(7ee59e9)の不具合3と同じ方針） |
| **2** | ★**父が入っている行からしか読まない**（空を配らない） |
| **3** | ★**書く前に検算する**——**行数が変わらない／父44・母父46 以外の列が1文字も変わらない／
　　　　 　元が空だった欄しか変わらない**。**1つでも破れたら、そのファイルは書かない** |

実行: python3 ml/nk_fill_ped.py [--check] [対象glob(既定 data/nk/DSnk*.CSV)]
"""
import csv
import glob
import io
import os
import sys

SIRE, DAMSIRE, REG = 43, 45, 37        # ★0始まり（schema上は 父44 / 母父46 / 血統登録番号38）
ENC = "shift_jis"


def read_rows(path):
    raw = io.open(path, "rb").read().decode(ENC, errors="replace")
    return list(csv.reader(io.StringIO(raw, newline=""))), raw


def dump(rows):
    out = io.StringIO(newline="")
    csv.writer(out, lineterminator="\r\n").writerows(rows)
    return out.getvalue()


def archive_pedigree(pattern="*.CSV"):
    """→ {血統登録番号(8桁): (父, 母父)}。★父が入っている行からしか読まない。"""
    ped = {}
    for p in sorted(glob.glob(pattern)):
        try:
            for r in csv.reader(io.open(p, encoding=ENC, errors="replace", newline="")):
                if (len(r) > DAMSIRE and len(r[REG]) == 8 and r[REG].isdigit() and r[SIRE]):
                    ped.setdefault(r[REG], (r[SIRE], r[DAMSIRE]))
        except OSError:
            continue
    return ped


def verify(old, new):
    """★元と新で、父44・母父46 以外が1文字も変わっていないか。→ (OK, 変えた欄の数)"""
    if len(old) != len(new):
        return False, 0
    changed = 0
    for a, b in zip(old, new):
        if len(a) != len(b):
            return False, 0
        for i, (x, y) in enumerate(zip(a, b)):
            if x == y:
                continue
            if i not in (SIRE, DAMSIRE) or x != "":     # ★空欄以外を変えたら不合格
                return False, 0
            changed += 1
    return True, changed


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    check = "--check" in sys.argv
    pat = args[0] if args else "data/nk/DSnk*.CSV"
    files = sorted(glob.glob(pat))
    if not files:
        sys.exit(f"{pat} に対象がありません")

    print("アーカイブの血統表を作成中…")
    ped = archive_pedigree()
    print(f"　{len(ped):,}頭\n")

    tot_fill = tot_left = 0
    for p in files:
        rows, raw = read_rows(p)
        before = [list(r) for r in rows]
        fill = left = 0
        for r in rows:
            if len(r) <= DAMSIRE or r[SIRE]:
                continue                                  # ★既に入っている行は触らない
            v = ped.get(r[REG]) if len(r[REG]) == 8 and r[REG].isdigit() else None
            if v is None:
                left += 1
                continue
            r[SIRE], r[DAMSIRE] = v[0], v[1]
            fill += 1
        ok, nch = verify(before, rows)
        n = sum(1 for r in rows if len(r) > DAMSIRE)
        have = sum(1 for r in rows if len(r) > DAMSIRE and r[SIRE])
        if not ok:
            print(f"⚠{p}: **検算に失敗したので書かない**")
            continue
        mark = "" if fill else "　（変更なし）"
        print(f"{os.path.basename(p):<22} 埋めた {fill:>4}行 / 残り {left:>4}行"
              f" → 父あり {have}/{n} ({have/max(n,1):.0%}){mark}")
        tot_fill += fill
        tot_left += left
        if fill and not check:
            new = dump(rows)
            io.open(p, "wb").write(new.encode(ENC, errors="replace"))

    print(f"\n★合計 {tot_fill}行を埋めた / 残り {tot_left}行"
          "（★10桁のまま＝アーカイブに居ない馬。**`nk_fetch.py pedigree` の担当**）")
    if check:
        print("⚠**--check なので書いていない**")
    return 0


if __name__ == "__main__":
    sys.exit(main())
