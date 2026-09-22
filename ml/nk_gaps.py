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

■ ★★**ルート直下の書庫も見る**（**SNS側の指摘・`ANA_SNS_GAP_REPLY.md` §4・2026-09-22**）
　⚠**`data/nk/` と★ルート直下の `*.CSV` は別物**。★**監査（＝凍結値を作った経路）が読むのは後者**
　（`audit_ana_*.py` の `F.load_files()` は**63箇所すべて引数なしで、既定は `"*.CSV"` ＝直下だけ**）。
　→ ★**`data/nk/` の穴は凍結値に届かない**。⚠**だが書庫の側にも穴は開きうる**。

　⚠★**「ファイルの有無」では見つからない**——★**行数で見る**。★**実測（2026-09-22）**:
| | |
|---|---|
| ★**書庫の土日** | **1,401日 / 中央値 542行** |
| ★**2026年の土日** | **01/04〜07/19 は1日も欠けていない**（**348〜1,521行**） |
| ⚠**2026-07-25 / 07-26** | ⚠**6行 / 2行**＝★**端数（前走ぶん）。書庫の終端** |
| ★**200行未満の土日** | ★**年末年始と中止のみ**（**2018-09-30・2019-10-12/13＝台風**）＝**正常** |

　★★**いちばん効く警告は「穴」ではなく「伸びた」のほう**——
　⚠**書庫が 2026-07-19 より先に伸びると、★監査の母集団が増えて凍結値が変わる**
　（★キャッシュ `data/cache/wf_pred.npz` は `len(d)` で判定しているので自動で無効化される）。
　★**「壊れた」ではなく「母集団が伸びた」**。★**その日が来たら測り直して凍結し直すこと**（★古い値も残す）。
"""
import csv
import datetime
import glob
import io
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
# ★どこから叩いてもリポジトリの data/nk を見る（⚠カレントに依らせない）
NKDIR = os.path.join(ROOT, "data", "nk")
# ★★ルート直下の書庫（＝監査の母集団）。⚠`data/nk/` とは別物
ARCHIVE = os.path.join(ROOT, "*.CSV")
ARCH_END = "20260719"     # ★実測した終端（2026-09-22）。★ここより伸びたら凍結値が変わる
ARCH_THIN = 200           # ★土日でこれ未満なら端数扱い（★中央値542行の約4割未満）


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


def archive_days(pattern=None):
    """→ {YYYYMMDD: 行数}。★ルート直下の書庫を日付ごとに数える。

    ⚠**前走ぶんの端数行が平日にも散る**ので、★**開催日かどうかは土日＋行数で見る**
    （★直下2,937日のうち609日が中央値の1割未満＝**端数の日を穴と呼ぶと使い物にならない**）。
    """
    cnt = {}
    for p in sorted(glob.glob(pattern or ARCHIVE)):
        try:
            for r in csv.reader(io.open(p, encoding="shift_jis",
                                        errors="replace", newline="")):
                if (len(r) > 3 and r[0].isdigit() and r[1].isdigit() and r[2].isdigit()):
                    k = f"20{r[0]}{r[1].zfill(2)}{r[2].zfill(2)}"
                    cnt[k] = cnt.get(k, 0) + 1
        except OSError:
            continue
    return cnt


def archive_lines(pattern=None):
    """→ 表示する行。★書庫が伸びたか／期間内に端数の土日が無いかを見る。"""
    cnt = archive_days(pattern)
    if not cnt:
        return ["⚠**ルート直下に *.CSV が無い**（★監査の母集団が読めない）"]
    full = sorted(d for d, n in cnt.items() if n >= ARCH_THIN)
    if not full:
        return ["⚠**書庫に開催日らしい日が無い**"]
    end = full[-1]
    out = [f"　★書庫（監査の母集団）{full[0]}〜{end} / 開催日 {len(full)}日"]
    if end > ARCH_END:
        out += [f"⚠★★**書庫が伸びている**（記録 {ARCH_END} → いま {end}）",
                "　⚠**監査の母集団が増えた＝★凍結値が変わる**"
                "（★キャッシュは len(d) 判定なので自動で無効化される）",
                "　★**壊れたのではない**。★**測り直して凍結し直すこと**（⚠古い値も消さずに残す）",
                f"　★直したら `ml/nk_gaps.py` の ARCH_END を {end} にすること"]
    # ★期間内の土日で端数のもの（⚠年末年始・中止も混じる）
    thin = []
    for d, n in sorted(cnt.items()):
        if not (full[0] <= d <= end) or n >= ARCH_THIN:
            continue
        if datetime.date(int(d[:4]), int(d[4:6]), int(d[6:])).weekday() in (5, 6):
            thin.append((d, n))
    if thin:
        out.append(f"　⚠期間内に端数の土日が {len(thin)}日"
                   f"（★年末年始・中止も混じる。★目で見ること）: "
                   + " ".join(f"{d}({n}行)" for d, n in thin[-6:]))
    return out


def main():
    txt, g = lines()
    print("\n".join(txt))
    print()
    print("\n".join(archive_lines()))
    return 1 if g else 0


if __name__ == "__main__":
    sys.exit(main())
