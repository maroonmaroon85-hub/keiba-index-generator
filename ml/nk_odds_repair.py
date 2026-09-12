"""(113) 壊れた `type<N>_<年>.jsonl.gz` から**読めるメンバーだけ救出**して書き直す。

★なぜ要るか
　`nk_odds_bulk.py` は**1レースごとにgzipを開いて閉じて追記する**（放置中に落ちても
　ファイル全体が壊れないように、という設計）。gzipは複数メンバーの連結を読めるので
　これで正しいはずだった。**だが実際に途中のメンバーが壊れた**（2026-08-11・2025年分）:
　`zlib.error: Error -3 while decompressing data: invalid block type`。
　標準の `gzip.open` は**壊れたところで読むのをやめる**ので、その後ろの正常なメンバーが
　まるごと失われる。9.4MBのファイルから265件しか読めなかった。

★やること
　バイト列を先頭から見て、**gzipの magic (1f 8b 08) ごとにメンバーを切り出して個別に展開**する。
　壊れたメンバーは飛ばし、次の magic から再開する。
　⚠magicは圧縮データ中に偶然現れうるので、**展開できてJSONとして読めた行だけ**を採用する。

　救出後、**台帳から失われたrace_idを消す**ことで `nk_odds_bulk.py` が取り直せるようになる。

使い方:
    python3 ml/nk_odds_repair.py                 # 全ファイルを点検（書き換えない）
    python3 ml/nk_odds_repair.py --fix           # 壊れていたものを救出して書き直す
    python3 ml/nk_odds_repair.py --fix --ledger  # ★台帳からも失われた分を消す（取り直せる）
"""
import glob
import gzip
import json
import os
import sys
import zlib

OUT = "data/nk_odds"
MAGIC = b"\x1f\x8b\x08"


def salvage(path):
    """→ (救出した行のリスト, 壊れていたか)。読めるメンバーだけ集める。"""
    raw = open(path, "rb").read()
    lines, broken, i = [], False, raw.find(MAGIC)
    while i >= 0 and i < len(raw):
        d = zlib.decompressobj(16 + zlib.MAX_WBITS)
        try:
            out = d.decompress(raw[i:])
            rest = d.unused_data
        except zlib.error:
            broken = True
            j = raw.find(MAGIC, i + 3)      # 壊れたメンバーは捨てて次のmagicへ
            i = j
            continue
        got = 0
        for ln in out.decode("utf-8", "replace").splitlines():
            ln = ln.strip()
            if not ln:
                continue
            try:
                json.loads(ln)
            except ValueError:
                continue                    # 偶然のmagicから展開された屑
            lines.append(ln)
            got += 1
        if not got:
            broken = True
        i = (len(raw) - len(rest)) if rest else -1
        if rest and not rest.startswith(MAGIC):
            k = raw.find(MAGIC, i)
            broken = True
            i = k
    return lines, broken


def readable(path):
    """標準の読み方で何件読めるか（＝救出しなかった場合に失う量が分かる）。"""
    n = 0
    try:
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            for ln in fh:
                if ln.strip():
                    n += 1
    except (EOFError, OSError, ValueError, zlib.error):
        pass
    return n


def main():
    fix = "--fix" in sys.argv
    do_ledger = "--ledger" in sys.argv
    saved_ids = set()
    print("※「救出後」は**race_idで重複を除いた件数**。再開時に同じレースを2回書いた分が"
          "あるので、差がマイナスでも消失ではない。")
    print(f"{'ファイル':<34}{'素の読み':>9}{'救出後':>9}{'差':>8}")
    for path in sorted(glob.glob(f"{OUT}/type*_*.jsonl.gz")):
        n0 = readable(path)
        lines, broken = salvage(path)
        # 同じrace_idが複数回入っていることがある（再開時の重複）。最後のものを残す
        by_id = {}
        for ln in lines:
            try:
                by_id[json.loads(ln)["race_id"]] = ln
            except (ValueError, KeyError):
                continue
        n1 = len(by_id)
        mark = "  ★壊れていた" if broken or n1 > n0 else ""
        print(f"{os.path.basename(path):<34}{n0:>9,}{n1:>9,}{n1-n0:>+8,}{mark}")
        saved_ids |= set(by_id)
        if fix and (broken or n1 > n0):
            tmp = path + ".new"
            with gzip.open(tmp, "wt", encoding="utf-8") as fh:
                for rid in sorted(by_id):
                    fh.write(by_id[rid] + "\n")
            os.replace(tmp, path)
            print(f"    → 書き直した（{n1:,}件・1メンバーにまとめた）")

    if do_ledger:
        # ★台帳から「実際には残っていないrace_id」を消す → bulk が取り直せるようになる
        for t in (1, 3, 4, 6, 7, 8):
            led = f"{OUT}/done_type{t}.txt"
            if not os.path.exists(led) or not glob.glob(f"{OUT}/type{t}_*.jsonl.gz"):
                continue
            keep, lost = [], 0
            for ln in open(led, encoding="utf-8"):
                s = ln.strip()
                if not s:
                    continue
                rid = s.rstrip("-").strip()
                if s.endswith("-") or rid in saved_ids:
                    keep.append(s)          # 空だった分はそのまま（取り直す必要がない）
                else:
                    lost += 1
            if lost:
                with open(led, "w", encoding="utf-8") as fh:
                    fh.write("".join(x + "\n" for x in dict.fromkeys(keep)))
                print(f"\n★{led}: {lost:,}件を消した → もう一度 nk_odds_bulk.py を回すと取り直す")
            else:
                print(f"\n{led}: 消す必要のある行は無かった")

    if not fix:
        print("\n（点検のみ。書き直すなら --fix、台帳も直すなら --fix --ledger）")


if __name__ == "__main__":
    main()
