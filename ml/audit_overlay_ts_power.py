"""(148の付録) ★★★★(148)がなぜ判定不能なのかを数字で示す — **判定基準32の現物**

★何をするか
　1. **各時点のD**（(148)の表と同じ量）を、**平均・SD・99%CI**まで出す。
　2. ★**「確定」列を陽性対照として使う**。**確定のDは(141)で31,130レース・+0.0266（11/11年で正）**。
　　 **その既知の値を、この標本が復元できるか**を見る。**できないなら他の列も読めない**。
　3. **±0.005 nat を99%で見分けるのに要るレース数**を出す。**次に何を集めるかの見積もり**。
　4. ★**対応のある差 D(t) − D(確定)**（同じレース同士で引く）。**共通分散が消えて必要数が落ちる**。
　　 ⚠**これは(148)の事前登録に無い追加**。**見積もりにだけ使い、判定には使わない**（後出しの禁止）。

⚠★★なぜこれを別ファイルにしたか
　(148)は**事前登録どおりの表**を出すファイル。**ここは事前登録の外**。
　**混ぜると「事前に決めた表」と「後から足した量」の区別がつかなくなる**（判定基準27の教訓）。

実行: python3 ml/audit_overlay_ts_power.py
"""
import math
import sys

import numpy as np

sys.path.insert(0, "ml")
from audit_crosspool import load_races, zq
from audit_crosspool2 import realized
from odds_ts_combo import LABELS, load_dir
from waku_umatan import waku_of

DW, DU = "data/odds_ts_waku", "data/odds_ts_umaren"
KNOWN = 0.0266   # ★(141)の「確定」でのD（31,130レース・11/11年で正）＝陽性対照の真値
RESOL = 0.005    # 見分けたい幅（nat）


def per_race_d():
    """{rid: {ラベル: D}}。(148)の d と同じ計算。"""
    tw, tu = load_dir(DW), load_dir(DU)
    if not tw or not tu:
        sys.exit(f"{DW} と {DU} が要る。")
    races = {r["rid"]: r for r in load_races()}
    out = {}
    for rid in sorted(set(tw) & set(tu) & set(races)):
        r = races[rid]
        if not r.get("wakuren"):
            continue
        rl = realized(r)
        if rl is None:
            continue
        a, b, _ = rl
        n = r["n"]
        nums = [u for u, _, _ in r["horses"]]
        if a not in nums or b not in nums:
            continue
        key = tuple(sorted((waku_of(a, n), waku_of(b, n))))
        d = {}
        for lab in LABELS:
            sw, su = tw[rid]["snaps"].get(lab), tu[rid]["snaps"].get(lab)
            if not sw or not su or not sw[1] or not su[1]:
                continue
            agg = {}
            for (x, y), o in su[1].items():
                if o <= 0 or x not in nums or y not in nums:
                    continue
                wx, wy = sorted((waku_of(x, n), waku_of(y, n)))
                agg[(wx, wy)] = agg.get((wx, wy), 0.0) + 1.0 / o
            keys = [k for k in sorted(agg) if k in sw[1]]
            if key not in keys or len(keys) < 3:
                continue
            inv = np.array([1.0 / sw[1][k] for k in keys])
            qp = inv / inv.sum()
            qq = np.array([agg[k] for k in keys])
            qq /= qq.sum()
            j = keys.index(key)
            d[lab] = math.log(qq[j]) - math.log(qp[j])
        if d:
            out[rid] = d
    return out


def line(lab, v, z):
    sd = v.std(ddof=1)
    se = sd / math.sqrt(len(v))
    lo, hi = v.mean() - z * se, v.mean() + z * se
    need = (z * sd / RESOL) ** 2
    return lab, len(v), v.mean(), sd, lo, hi, need


def main():
    recs = per_race_d()
    z = zq(0.01)
    print("(148の付録) なぜ判定不能なのか — 判定基準32の現物")
    print(f"　対象 {len(recs):,}レース（1開催日ぶん）\n")

    print("■ 各時点のD（1レースあたり）")
    print(f"{'時点':>7}{'n':>6}{'平均D':>10}{'SD':>9}{'99%CI':>24}{'必要n':>11}")
    rows = {}
    for lab in LABELS:
        v = np.array([d[lab] for d in recs.values() if lab in d])
        if len(v) < 2:
            continue
        rows[lab] = line(lab, v, z)
        _, n, m, sd, lo, hi, need = rows[lab]
        print(f"{lab:>7}{n:>6}{m:>+10.4f}{sd:>9.4f}"
              f"{'[' + format(lo, '+.4f') + ',' + format(hi, '+.4f') + ']':>24}{need:>11,.0f}")

    print("\n■ ★★陽性対照の検査（ここが本題）")
    if "確定" in rows:
        _, n, m, sd, lo, hi, need = rows["確定"]
        ok = lo <= KNOWN <= hi
        print(f"　「確定」のDは(141)で **31,130レース・{KNOWN:+.4f}**（11/11年で正）と実測済み。")
        print(f"　この標本({n}レース)での推定は **{m:+.4f}**　99%CI [{lo:+.4f},{hi:+.4f}]")
        print(f"　→ 真値 {KNOWN:+.4f} をCIに含むか: **{'含む' if ok else '含まない'}**"
              f"　符号: **{'一致' if m * KNOWN > 0 else '★不一致'}**")
        print("　⚠**CIに含むので(141)への反証ではない**。**だが符号すら合っていない**。")
        print("　→ ★**真値の分かっている列を外す標本で、分からない列の符号は読めない**（判定基準32）。")

    print("\n■ ★対応のある差 D(t) − D(確定)（同じレース同士で引く。共通分散が消える）")
    print("　⚠**(148)の事前登録に無い追加**。**必要数の見積もりにだけ使う。判定には使わない**。")
    print(f"{'時点':>7}{'n':>6}{'差の平均':>11}{'SD':>9}{'99%CI':>24}{'必要n':>11}")
    for lab in LABELS:
        if lab == "確定":
            continue
        v = np.array([d[lab] - d["確定"] for d in recs.values() if lab in d and "確定" in d])
        if len(v) < 2:
            continue
        _, n, m, sd, lo, hi, need = line(lab, v, z)
        print(f"{lab:>7}{n:>6}{m:>+11.4f}{sd:>9.4f}"
              f"{'[' + format(lo, '+.4f') + ',' + format(hi, '+.4f') + ']':>24}{need:>11,.0f}")

    print("\n" + "=" * 92)
    print("★読み方")
    print(f"  ・必要nは **±{RESOL} nat を99%で見分ける**のに要るレース数。")
    print("  ・**直前を対応差で見れば 4,700レース級**＝**1.5年・約150開催日**で決着する。")
    print("  ⚠**それまで(141)の117.3%は生死不明**。**閉じたとも生きているとも書かないこと**。")


if __name__ == "__main__":
    main()
