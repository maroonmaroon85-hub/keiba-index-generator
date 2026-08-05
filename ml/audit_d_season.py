"""
(91) (89)の D を **季節・開催規模・場**で割る — 「夏競馬は別物では」を測る。

★なぜ今やるのか: ③(2026-07-24)で季節はROIで層別済み（夏78.4% > 秋77.3% > 春77.2% > 冬74.9%）で
「小差」として片付いている。**だがあれは(89)以前の道具**。ROIで季節に割ると1区分年3,000R程度で、
(89)の指摘どおり**枠連で1ptを検出するのに50年**——「3.5pt差」は測定限界の中にあった。
**Dなら割っても持つ**（枠連全体でCI幅0.008。4分割しても0.016程度）。

★仮説には機構がある（事前宣言）
　夏競馬＝ローカル開催（小倉・新潟・札幌・函館）＝**プールが薄い**。そして:
　　・(76)で**市場AUCは開催規模でほぼ決まる**と実測（頭数との相関 r=0.956。東京0.8285 / 函館0.7844）
　　・(89)で**枠連プールだけが単勝プールに負けている**（D=+0.0145）
　　・**プールが薄いほど値付けが雑になるはず** → 夏のローカルでDが大きい
　つまり「夏が特別」ではなく「**薄いプールが特別**」で、夏はそれが集中する時期、という筋。
　(76)で小倉が理由不明のまま残ったのも、**小倉＝夏のローカル**なので同じ現象の別の見え方かもしれない。

★判定（測る前に宣言）
　1. 夏(6-8月)のDが他季節より大きい → 仮説どおり
　2. **場（開催規模）で説明が付く** → 季節ではなく**プールの厚さ**が本体
　3. どちらも出ない → 季節は無関係。③の結論のまま
　★**季節と場を同時に見る**のが肝。片方だけだと交絡する。
　★多重性: 4季節 + 10場 + 4頭数帯 = 18区分。α=0.05/18 で判定する。

実行: python3 ml/audit_d_season.py [開始年(既定2015)]
"""
import math
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, "ml")
from audit_crosspool import PAYBACK, load_races, payoff, probs, zq
from audit_crosspool2 import PARTS, PAYKEY, realized
from audit_crosspool3 import q_of

# raceid の先頭2桁 = 場コード（(70)の対応表と同じ）
PLACE = {"01": "札幌", "02": "函館", "03": "福島", "04": "新潟", "05": "東京",
         "06": "中山", "07": "中京", "08": "京都", "09": "阪神", "10": "小倉"}
LOCAL = {"札幌", "函館", "福島", "新潟", "小倉"}      # ローカル＝プールが薄い側
SEASON = {1: "冬", 2: "冬", 3: "春", 4: "春", 5: "春", 6: "夏", 7: "夏", 8: "夏",
          9: "秋", 10: "秋", 11: "秋", 12: "冬"}


def month_map(path="data/payout/a.csv"):
    """{raceid: 月}。`load_races` は月を持っていないが、配当Aの col1 が月。

    ★共有コードの `load_races` は並行セッションも使うので触らない。ここで独立に作る。
    """
    import csv
    import io
    with open(path, "rb") as fh:
        txt = fh.read().decode("shift_jis", "replace")
    out = {}
    for r in csv.reader(io.StringIO(txt)):
        if len(r) < 224:
            continue
        rid = r[14].strip()
        if len(rid) != 8:
            continue
        try:
            out[rid] = int(r[1])
        except ValueError:
            continue
    return out


def collect(races, y0, mmap):
    """(89)と同じ D を、区分の情報つきで集める。"""
    out = []
    for r in races:
        if r["year"] < y0:
            continue
        rl = realized(r)
        if rl is None:
            continue
        a, b, c = rl
        hs = r["horses"]
        p = probs(hs)
        num2k = {num: k for k, (num, _, _) in enumerate(hs)}
        if a not in num2k or b not in num2k or (c is not None and c not in num2k):
            continue
        rid = r["rid"]
        place = PLACE.get(rid[:2], "?")
        # raceid の先頭2桁が場コード。月は配当Aの col1 から引く
        mon = mmap.get(rid)
        for kind, key in PARTS.items():
            if not r[key]:
                continue
            q, combo = q_of(kind, r, p, num2k, a, b, c)
            if q <= 0 or combo is None:
                continue
            v = payoff(r, PAYKEY[kind], combo)
            if not v or v <= 0:
                continue
            d = math.log(q) + math.log((v + 5) / 100.0) - math.log(PAYBACK[kind])
            out.append({"kind": kind, "d": d, "year": r["year"], "n": r["n"],
                        "place": place, "local": place in LOCAL,
                        "season": SEASON.get(mon, "?") if mon else "?",
                        "month": mon or 0})
    return out


def mci(x, alpha):
    x = np.asarray(x, float)
    m = x.mean()
    se = x.std(ddof=1) / math.sqrt(len(x))
    z = zq(alpha)
    return m, m - z * se, m + z * se


def main():
    y0 = int(sys.argv[1]) if len(sys.argv) > 1 else 2015
    races = load_races()
    mmap = month_map()
    rows = collect(races, y0, mmap)
    print(f"月が引けたレース: {len(mmap):,}")
    import pandas as pd
    df = pd.DataFrame(rows)
    print(f"(89)の D を区分で割る（{y0}年以降・{len(df):,}件）")
    print("★判定は α=0.05/18（4季節+10場+4頭数帯）で行う\n")
    alpha = 0.05 / 18

    for kind in PARTS:
        dd = df[df["kind"] == kind]
        if len(dd) < 1000:
            continue
        m0, lo0, hi0 = mci(dd["d"], alpha)
        need = -math.log(PAYBACK[kind])
        print(f"{'='*104}")
        print(f"=== {kind}  {len(dd):,}件  全体 D={m0:+.4f} [{lo0:+.4f},{hi0:+.4f}]"
              f"  必要量 {need:.4f} ===")
        print(f"{'='*104}")
        for axis, label in (("season", "季節"), ("place", "場"), ("fsbin", "頭数帯"),
                            ("local", "ローカル開催か")):
            if axis == "fsbin":
                dd = dd.copy()
                dd["fsbin"] = pd.cut(dd["n"], [0, 11, 13, 15, 99],
                                     labels=["〜11頭", "12-13頭", "14-15頭", "16頭〜"])
            if axis == "season" and (dd["season"] == "?").all():
                continue
            print(f"\n■ {label}")
            print(f"{'区分':<12}{'件数':>9}{'D':>11}{'99.7%CI':>24}{'全体との差':>12}{'判定':>8}")
            for v, g in dd.groupby(axis, observed=True):
                if len(g) < 300:
                    continue
                m, lo, hi = mci(g["d"], alpha)
                mark = "★上" if lo > m0 else ("★下" if hi < m0 else "")
                print(f"{str(v):<12}{len(g):>9,}{m:>+11.4f}"
                      f"{f'[{lo:+.4f},{hi:+.4f}]':>24}{m-m0:>+12.4f}{mark:>8}")
        print()

    print("=" * 104)
    print("★読み方")
    print("  ・場と頭数帯でDが動き、季節では動かないなら → **季節ではなくプールの厚さ**が本体。")
    print("  ・季節だけが動くなら → 開催構成では説明できない季節固有の現象。")
    print("  ・どの区分でも必要量（枠連0.2549）には遠いはずで、**運用は変わらない**。")
    print("    ここで見ているのは『どこで市場が雑になるか』の記述であって、儲かる場所の探索ではない。")


if __name__ == "__main__":
    main()
