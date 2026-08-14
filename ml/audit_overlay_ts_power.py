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
from odds_ts_combo import LABELS, load_pool
from waku_umatan import waku_of

KNOWN = 0.0266   # ★(141)の「確定」でのD（31,130レース・11/11年で正）＝陽性対照の真値
RESOL = 0.005    # 見分けたい幅（nat）
PER_DAY = 27     # ★1開催日あたり判定に使えたレース数（実測: 出力36R→27R）


def per_race_d():
    """{rid: {ラベル: D}}。(148)の d と同じ計算。"""
    tw, tu = load_pool("枠"), load_pool("馬")
    if not tw or not tu:
        sys.exit("枠連と馬連の時系列が両方要る（COLLECT_TS.md 参照）。")
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

    print("\n■ ★★必要レース数を「目的」から決め直す（⚠上の必要nの ±0.005 は恣意的だった）")
    sd_pair = np.array([d["直前"] - d["確定"] for d in recs.values()
                        if "直前" in d and "確定" in d]).std(ddof=1)
    sd_fin = np.array([d["確定"] for d in recs.values() if "確定" in d]).std(ddof=1)
    n0 = sum(1 for d in recs.values() if "確定" in d)
    rel = 1.0 / math.sqrt(2 * (n0 - 1))
    print(f"　⚠**SD自体が {n0}レース由来**。SDの相対誤差 ±{rel:.0%} → **必要nは±{2*rel:.0%}ぶれる**")
    print(f"{'目的':<44}{'見分ける幅':>11}{'必要R':>9}{'開催日':>8}{'開催日のぶれ':>16}")
    for name, sd, eff, one in (
            ("①陽性対照が立つ（確定単独のCIが0を外す）", sd_fin, KNOWN, True),
            ("②直前に確定と同じ優位が在るか／無いか", sd_pair, KNOWN, False),
            ("③直前の劣化が半分以下か", sd_pair, KNOWN / 2, False),
            (f"④±{RESOL} nat まで詰める", sd_pair, RESOL * 2, False)):
        n = (z * sd / eff) ** 2 if one else (2 * z * sd / eff) ** 2
        print(f"{name:<44}{eff:>11.4f}{n:>9,.0f}{n/PER_DAY:>8.0f}"
              f"{f'{n*0.7/PER_DAY:.0f}〜{n*1.3/PER_DAY:.0f}':>16}")
    print(f"　※1開催日 = **{PER_DAY}レース**（実測: 出力36Rのうち判定に使えたのが{PER_DAY}R）。")
    print("　⚠**歩留まりも27レース由来**。**実際にはもう少し良くなりうる**（raceid不一致3件は解消しうる）。")

    print("\n" + "=" * 92)
    print("★読み方")
    print("  ★**まず25開催日（約660R）集めれば ①②が立つ**。**そこで(141)の生死が分かる**。")
    print("  ・**①が立たなければ何も読まない**（判定基準32）。**①は最初に確認すること**。")
    print("  ⚠★★**途中で止めてよいのは①を見たときだけ**。**②を見て止めてはいけない**——")
    print("    **①は答えと無関係な既知量**なので、それを基準に止めても結論は歪まない。")
    print("    **②で止めると『良く見えたところで打ち切る』ことになり、任意停止のバイアスが入る**。")
    print("    → ★**5開催日ごとに①だけ見る。①が立った時点で②を1回だけ見て、そこで終わり**。")


if __name__ == "__main__":
    main()
