"""★★結果CSVの欠けている開催日を見つける（★「過去走なし」の正体を先に潰す）

    python3 ml/nk_gaps.py            ★欠けている日を出す（★穴があれば終了コード1）

■ ★なぜ要るか（**2026-09-19 中山3Rで判明**）
　★**`data/nk/DSnk<日付>.CSV` に穴があると、その日に走った馬は「過去走なし」になる**。
　⚠**モデルは特徴量を作れないので、その馬を表から落として計算する**。
　→ ★**9/19 中山3R は、落とした5番トルタパスカリーナ（単勝1.5倍）が勝った**。
　　★朝の出力は「※過去走なし2頭（来たら外れる）」と予告していたが、
　　⚠**それがデータの穴のせいだとは分からなかった**。★ここを見えるようにする。

■ ★見つけ方
　★**手元にある最初の日〜最後の日の間で、土日なのにファイルが無い日**を穴とする。
　⚠**期間の外（収集開始より前）は穴として数えない**。★**振替開催などの平日は見ない**
　　（⚠**取りこぼす可能性はあるが、偽の警告を出すよりまし**）。

■ ⚠★**埋めるのは Mac 側**（★クラウドからは netkeiba に届かない）
　★**順番を守ること**: `results` → `pedigree` → `nk_link`。
　⚠**`pedigree` を飛ばすと血統が消える**（**2026-09-12 に一度やらかしている**）。
"""
import datetime
import glob
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
# ★どこから叩いてもリポジトリの data/nk を見る（⚠カレントに依らせない）
NKDIR = os.path.join(ROOT, "data", "nk")


def covered(nkdir=NKDIR):
    """→ 手元にある結果CSVの日付（YYYYMMDD）の集合。"""
    out = set()
    for p in glob.glob(os.path.join(nkdir, "DSnk*.CSV")):
        m = re.search(r"(\d{8})", os.path.basename(p))
        if m:
            out.add(m.group(1))
    return out


def gaps(have):
    """→ [(YYYYMMDD, '土'/'日')]。★期間内で土日なのに無い日だけ。"""
    if len(have) < 2:
        return []
    lo, hi = min(have), max(have)
    d = datetime.date(int(lo[:4]), int(lo[4:6]), int(lo[6:]))
    end = datetime.date(int(hi[:4]), int(hi[4:6]), int(hi[6:]))
    out = []
    while d <= end:
        if d.weekday() in (5, 6):
            y = d.strftime("%Y%m%d")
            if y not in have:
                out.append((y, "土" if d.weekday() == 5 else "日"))
        d += datetime.timedelta(days=1)
    return out


def lines(nkdir=NKDIR):
    """→ (表示する行, 穴の日付リスト)。★raceday.py からも呼ぶ。"""
    have = covered(nkdir)
    g = gaps(have)
    if not have:
        return [f"⚠**{nkdir}/DSnk*.CSV が1つも無い**"], []
    if not g:
        return [f"　★結果CSVに穴なし（{min(have)}〜{max(have)} / {len(have)}日）"], []
    out = [f"⚠★★**結果CSVに穴が {len(g)}日ある**（{min(have)}〜{max(have)}）",
           "　　" + " ".join(f"{y}({w})" for y, w in g),
           "　⚠**この日に走った馬は「過去走なし」になり、モデルの表から落ちる**"
           "（★落ちた馬が勝つことがある・9/19 中山3R）",
           "　★**埋めるのは Mac 側**（⚠この順番で。`pedigree` を飛ばすと血統が消える）:"]
    out += [f"　　　python3 ml/nk_fetch.py results {y}" for y, _w in g]
    out += ["　　　python3 ml/nk_fetch.py pedigree",
            "　　　python3 ml/nk_link.py"]
    return out, [y for y, _w in g]


def main():
    txt, g = lines()
    print("\n".join(txt))
    return 1 if g else 0


if __name__ == "__main__":
    sys.exit(main())
