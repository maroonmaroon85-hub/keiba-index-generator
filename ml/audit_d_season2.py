"""(116) (91)の層別を**今のq**でやり直す — 「場・季節は決着済み」と言えるか（2026-08-11）

★なぜやり直すか（ユーザー指摘）
　(91)は `p = probs(hs)`＝**素の市場Harville**で測っていた。全体の枠連 D=+0.0145 がそれ。
　だが**その後Dは1.5倍近くになっている**:
　　素のHarville +0.0142 → **λ補正((96)) +0.0182** → モデル混合((100)(102)) +0.0207
　**層別だけ古い道具のまま**だった。「決着済み」と言い切るのは早い。

★何が変わりうるか
　(76)は「10場すべてでモデルAUC<市場AUC・場によらず一様」と実測しているが、
　**それはAUCであってDではない**。λ補正の効き方が場・季節で違えば、層別の絵は変わりうる。
　λ2/λ3はHarvilleの系統誤差の補正なので、**頭数や隊列の性質が違う場では効き方が違って当然**。
　→ (91)②「厚いプールほどDが大きい」が**λ補正の当て損ないの裏返し**だった可能性を潰す。

★★事前登録（測る前に宣言）
　1. **予想**: 結論は変わらない。季節は動かず、場・頭数帯は「厚いほど大きい」のまま。
　　 理由: λ補正は全体で+0.0040しか動かさない。(91)の場の差(−0.0037〜+0.0072)と同オーダーだが、
　　 　　　補正は**全レースに一様に効く形**（λは年ごとに1組しか推定しない）なので層は動きにくい。
　2. **★ただし1点だけ本気で疑う**: (91)②の「多頭数ほどDが大きい」（〜11頭 +0.0082 → 16頭〜 +0.0175）は
　　 **Harvilleの近似誤差が多頭数で相殺されているだけ**かもしれない——と(91)自身が書いている。
　　 λ補正はまさにその誤差を直す道具なので、**補正後に頭数の勾配が消えたら、(91)②は道具の癖だった**と確定する。
　　 消えなければ、**プールの厚さが本体**という記述が補強される。
　3. **判定**: α=0.05/18（4季節+10場+4頭数帯）。(91)と同じ基準を使う（後から緩めない）。
　4. **モデル混合は入れない**。混合の重みは年ごとに1組で、これも一様に効くため。
　　 ⚠入れないと決めた以上、**後から「混合なら違ったかも」と言い出さないこと**。

実行: python3 ml/audit_d_season2.py [開始年(既定2015)]
"""
import math
import sys

import numpy as np

sys.path.insert(0, "ml")
from audit_crosspool import PAYBACK, load_races, payoff, probs, zq
from audit_crosspool2 import PARTS, PAYKEY, realized
from audit_crosspool3 import q_of
from audit_d_season import LOCAL, PLACE, SEASON, month_map
from audit_lbs import build_matrix, fit_lambda, q_of_lbs


def mci(x, alpha):
    x = np.asarray(x, float)
    m = x.mean()
    se = x.std(ddof=1) / math.sqrt(len(x))
    z = zq(alpha)
    return m, m - z * se, m + z * se


def collect(races, y0, mmap, lam):
    """(91)と同じ形。**qだけ λ補正版に差し替える**（lam=None なら(91)と完全に同じ）。"""
    out = []
    for r in races:
        yy = r["year"]
        if yy < y0:
            continue
        if lam is not None and not lam.get(yy):
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
        mon = mmap.get(r["rid"])
        place = PLACE.get(r["rid"][:2], "?")
        for kind, key in PARTS.items():
            if not r[key]:
                continue
            if lam is None:
                q, combo = q_of(kind, r, p, num2k, a, b, c)
            else:
                l2, l3 = lam[yy]
                q, combo = q_of_lbs(kind, r, p, l2, l3, num2k, a, b, c)
            if q <= 0 or combo is None:
                continue
            v = payoff(r, PAYKEY[kind], combo)
            if not v or v <= 0:
                continue
            out.append({"kind": kind,
                        "d": math.log(q) + math.log((v + 5) / 100.0) - math.log(PAYBACK[kind]),
                        "year": yy, "n": r["n"], "place": place,
                        "local": place in LOCAL,
                        "season": SEASON.get(mon, "?") if mon else "?"})
    return out


def table(df, kind, alpha, pd):
    dd = df[df["kind"] == kind]
    if len(dd) < 1000:
        return None
    m0, lo0, hi0 = mci(dd["d"], alpha)
    dd = dd.copy()
    dd["fsbin"] = pd.cut(dd["n"], [0, 11, 13, 15, 99],
                         labels=["〜11頭", "12-13頭", "14-15頭", "16頭〜"])
    return dd, m0, lo0, hi0


def main():
    y0 = int(sys.argv[1]) if len(sys.argv) > 1 else 2015
    import pandas as pd
    races = load_races()
    mmap = month_map()

    # λ は(96)と同じくウォークフォワードで年ごとに推定する
    P, i1, i2, i3, yrs = build_matrix(races, y0)
    lam = {}
    for yy in sorted(set(yrs.tolist())):
        tr = yrs < yy
        if tr.sum() < 3000:
            lam[yy] = None
            continue
        ok3 = tr & (i3 >= 0)
        lam[yy] = (fit_lambda(P[tr], i1[tr], i2[tr]),
                   fit_lambda(P[ok3], i1[ok3], i2[ok3], stage3=True, ic=i3[ok3]))

    old = pd.DataFrame(collect(races, y0, mmap, None))
    new = pd.DataFrame(collect(races, y0, mmap, lam))
    alpha = 0.05 / 18
    print(f"(116) (91)の層別を**λ補正した今のq**でやり直す（{y0}年以降）")
    print(f"  (91)と同じ素のHarville: {len(old):,}件 / λ補正版: {len(new):,}件")
    print("  ★判定は(91)と同じ α=0.05/18。★は全体との差が有意な区分\n")

    for kind in PARTS:
        a = table(old, kind, alpha, pd)
        b = table(new, kind, alpha, pd)
        if a is None or b is None:
            continue
        (do, m0o, _, _), (dn, m0n, lo0n, hi0n) = a, b
        need = -math.log(PAYBACK[kind])
        print("=" * 100)
        print(f"=== {kind}  全体 D: 素{m0o:+.4f} → **λ補正 {m0n:+.4f}** "
              f"[{lo0n:+.4f},{hi0n:+.4f}]  必要量 {need:.4f} ===")
        for axis, label in (("season", "季節"), ("place", "場"),
                            ("fsbin", "頭数帯"), ("local", "ローカル開催か")):
            if axis == "season" and (dn["season"] == "?").all():
                continue
            print(f"\n■ {label}")
            print(f"{'区分':<12}{'件数':>9}{'素のD':>10}{'λ補正D':>10}"
                  f"{'全体との差':>12}{'99.7%CI':>24}{'判定':>7}")
            for v, g in dn.groupby(axis, observed=True):
                if len(g) < 300:
                    continue
                m, lo, hi = mci(g["d"], alpha)
                go = do[do[axis] == v] if axis != "fsbin" else do[
                    pd.cut(do["n"], [0, 11, 13, 15, 99],
                           labels=["〜11頭", "12-13頭", "14-15頭", "16頭〜"]) == v]
                mo = go["d"].mean() if len(go) else float("nan")
                mark = "★上" if lo > m0n else ("★下" if hi < m0n else "")
                print(f"{str(v):<12}{len(g):>9,}{mo:>+10.4f}{m:>+10.4f}"
                      f"{m-m0n:>+12.4f}{f'[{lo:+.4f},{hi:+.4f}]':>24}{mark:>7}")
        print()

    print("=" * 100)
    print("★読み方（事前登録2のとおり）")
    print("  ・**頭数帯の勾配が消えたら**、(91)②『厚いプールほどDが大きい』は")
    print("    **Harvilleの近似誤差が多頭数で相殺されていただけ**＝道具の癖だったと確定する。")
    print("  ・消えなければ、プールの厚さが本体という記述が補強される。")
    print("  ・季節が動かなければ(91)①のまま。**どちらにせよ必要量には遠く、運用は変わらない**。")


if __name__ == "__main__":
    main()
