"""(105) ★★「妙味の無い組は買われない」機構は他の券種にもあるか — (94)の横展開。

★根拠のある仮説（(94)③で機構まで特定できている）
　複勝プールは**短いオッズ帯だけが甘い**（確定1.0-1.6倍でROI 93〜95%＝線+13〜15pt）。
　(94)が特定した機構は「**複勝で1.2倍の馬を買っても110円しか返らないので買い手が集まらず、
　配当が高止まりする**」。★これは**知識ではなく動機の問題**なので、周知されても解消しない。

　→ **同じ動機は他の券種にもあるはず**。断然人気2頭のワイドは130円、馬連は150円。
　　 「当たっても増えない」組は買われにくい。**組の期待払戻が小さいほど甘くなる**という予想。
　★(77)は8券種29通りを横断したが、**モデルが選んだ買い目のROI**を比べていた。
　　ここで見るのは**市場の構造**であってモデルではない。**測っているものが違う**。

★★事前登録（測る前に宣言）
　1. **仮説**: 各券種で「**人気上位の組**」を買ったときのROIは、**組が短いほど線を上回る**。
　2. **判定は単調性**（(84)の教訓）。1番人気の単勝オッズ帯で切って、
　　 **ROIが単調に下がるか**を Spearman ρ で見る。最良の帯を選んで判定しない。
　3. **★事前に使える切り方も必ず出す**。確定オッズでの選択は事後（(94)④で一度やらかしている）。
　　 → 「1番人気の確定単勝オッズ」で切るのは**買う時点では分からない**。だが(92)⑤より
　　 　 **この帯は締切直前にはほぼ確定している**ので、方向を見る用途では使える。両方出す。
　4. **★線は券種ごとに違う**。単勝/複勝80% / 枠連・馬連・ワイド・馬単77.5% / 三連複75% / 三連単72.5%。
　　 **必ず各券種の線と比べる**。ROIの絶対値どうしを比べても意味が無い。
　5. **予想する順位**: 複勝 > ワイド > 馬連 ≒ 枠連 > 三連複。
　　 理由は**「当たっても増えない」度合いの順**。ワイドは複勝の次に的中しやすく配当が低い。
　6. ★**天井があるかどうかは券種で違う。ここは開いた問いとして測る**。
　　 (94)③が出した「複勝の天井98.2%」は**複勝に固有の議論**だった:
　　 　複勝には**最低配当100円**があるので、オッズを詰めるほど配当が100円に張り付き、
　　 　的中率は95%で頭打ち → `的中率 × 配当` が100%を超えられない。
　　 **ワイド・馬連にこの議論は移らない**。これらの配当は自然に100円より上にあり、床に当たらない。
　　 ＝**上振れの大きさに原理的な上限が無い**。断然人気2頭の馬連が110〜130円という領域は
　　 　「当たっても増えない」度合いが最も強く、複勝と同じ動機が最も効くはずの場所。
　　 ⚠**先に答えを決めない**。100%を超えるかどうかは測ってから言う。

★★機構をもっと直接突く（第2の層別）
　「1番人気の単勝オッズ」は proxy でしかない。機構が言っているのは
　**「その組の期待払戻が小さいほど買われない」**なので、**組の期待払戻そのもので層別する**。
　λ補正した Harville で組の確率 q を出せば、期待払戻 ≈ 払戻率/q が**発走前に計算できる**。
　→ **これが仮説の直球の検定**。単勝オッズ帯より鋭く出るはず。

実行: python3 ml/audit_soft_combo.py [開始年(既定2015)]
"""
import itertools
import math
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "ml")
from audit_crosspool import PAYBACK, load_races, payoff, probs, zq
from audit_crosspool2 import realized
from waku_umatan import waku_of

# (名前, 払戻キー, 点数の作り方)
MENU = [("複勝 1番人気", "複勝", 1), ("ワイド 上位2頭", "ワイド", 1),
        ("馬連 上位2頭", "馬連", 1), ("枠連 上位2頭の枠", "枠連(人気順)", 1),
        ("馬単 上位2頭", "馬単", 1), ("三連複 上位3頭", "三連複", 1),
        ("三連単 上位3頭", "三連単", 1)]
LINE = {"複勝": 0.800, "ワイド": 0.775, "馬連": 0.775, "枠連(人気順)": 0.775,
        "馬単": 0.775, "三連複": 0.750, "三連単": 0.725}


def mci(x, alpha=0.01):
    x = np.asarray(x, float)
    m = x.mean()
    se = x.std(ddof=1) / math.sqrt(len(x))
    z = zq(alpha)
    return m, m - z * se, m + z * se


def combo_q(kind, r, p, l2, l3, cb, num2k):
    """買い目の**発走前に計算できる**確率（λ補正Harville）。期待払戻の推定に使う。"""
    from audit_lbs import (g_pair_ordered, g_pair_unordered, g_tri_ordered,
                           g_tri_unordered, stage_w)
    w2 = stage_w(p, l2)
    W2 = w2.sum()
    if kind == "複勝":
        from audit_fuku_lbs import top3_probs
        return float(top3_probs(p, 1.0, l2, l3)[num2k[cb[0]]])
    if kind in ("馬連", "ワイド"):
        x, y = num2k[cb[0]], num2k[cb[1]]
        q2 = g_pair_unordered(p, w2, W2, x, y)
        if kind == "馬連":
            return q2
        # ワイドは「2頭とも3着以内」。3着までの枠に2頭が入る確率
        w3 = stage_w(p, l3)
        W3 = w3.sum()
        q = q2
        for a_, b_ in ((x, y), (y, x)):
            for z in range(len(p)):
                if z in (x, y):
                    continue
                q += (g_tri_ordered(p, w2, W2, w3, W3, a_, z, b_)
                      + g_tri_ordered(p, w2, W2, w3, W3, z, a_, b_))
        return q
    if kind == "馬単":
        return g_pair_ordered(p, w2, W2, num2k[cb[0]], num2k[cb[1]])
    w3 = stage_w(p, l3)
    W3 = w3.sum()
    if kind == "三連複":
        return g_tri_unordered(p, w2, W2, w3, W3,
                               (num2k[cb[0]], num2k[cb[1]], num2k[cb[2]]))
    if kind == "三連単":
        return g_tri_ordered(p, w2, W2, w3, W3,
                             num2k[cb[0]], num2k[cb[1]], num2k[cb[2]])
    return None                       # 枠連は枠の集約が要るのでここでは出さない


def main():
    y0 = int(sys.argv[1]) if len(sys.argv) > 1 else 2015
    from audit_lbs import build_matrix, fit_lambda
    all_races = load_races()
    P, i1, i2, i3, yrs = build_matrix(all_races, y0)
    lam = {}
    for yy in sorted(set(yrs.tolist())):
        tr = yrs < yy
        if tr.sum() < 3000:
            lam[yy] = None
            continue
        ok3 = tr & (i3 >= 0)
        lam[yy] = (fit_lambda(P[tr], i1[tr], i2[tr]),
                   fit_lambda(P[ok3], i1[ok3], i2[ok3], stage3=True, ic=i3[ok3]))
    rows = []
    for r in all_races:
        if r["year"] < y0:
            continue
        if realized(r) is None:
            continue
        hs = r["horses"]
        p = probs(hs)
        order = np.argsort(-p)
        nums = [hs[k][0] for k in order]
        if len(nums) < 3:
            continue
        n = r["n"]
        row = {"rid": r["rid"], "year": r["year"], "n": n,
               "o1": hs[order[0]][1], "o2": hs[order[1]][1], "o3": hs[order[2]][1]}
        combos = {
            "複勝": (nums[0],),
            "ワイド": tuple(sorted(nums[:2])),
            "馬連": tuple(sorted(nums[:2])),
            "枠連(人気順)": tuple(sorted((waku_of(nums[0], n), waku_of(nums[1], n)))),
            "馬単": (nums[0], nums[1]),
            "三連複": tuple(sorted(nums[:3])),
            "三連単": (nums[0], nums[1], nums[2]),
        }
        num2k = {num: k for k, (num, _, _) in enumerate(hs)}
        lm = lam.get(r["year"])
        for key, cb in combos.items():
            v = payoff(r, key, cb)
            row[key] = None if v is None else (v or 0.0)
            if lm and key != "枠連(人気順)":
                try:
                    q = combo_q(key, r, p, lm[0], lm[1], cb, num2k)
                except Exception:
                    q = None
                row[f"q_{key}"] = q if (q and 0 < q < 1) else None
        rows.append(row)
    df = pd.DataFrame(rows)
    print(f"(105)「妙味の無い組は買われない」機構の横展開（{y0}年以降・{len(df):,}レース）")
    print("★券種ごとに線が違う。必ず各券種の線と比べること\n")

    print("=" * 104)
    print("【1】全体 — 人気上位の組を機械的に買ったときのROI（1点・100円）")
    print("=" * 104)
    print(f"{'買い方':<20}{'R数':>9}{'的中率':>9}{'ROI':>9}{'99%CI':>22}{'線':>8}{'線との差':>11}")
    for lab, key, _ in MENU:
        g = df[df[key].notna()]
        if len(g) < 1000:
            continue
        x = g[key].to_numpy(float) - 100.0
        m, lo, hi = mci(x)
        line = LINE[key] * 100
        print(f"{lab:<20}{len(g):>9,}{(g[key] > 0).mean()*100:>8.1f}%"
              f"{(m+100):>8.1f}%{f'[{lo+100:.1f},{hi+100:.1f}]':>22}{line:>7.1f}%"
              f"{(m+100)-line:>+10.2f}pt")

    print("\n" + "=" * 104)
    print("【2】★1番人気の単勝オッズ帯で切る（(94)③と同じ切り方・仮説の本体）")
    print("=" * 104)
    bands = [0, 1.6, 2.0, 2.5, 3.0, 4.0, 6.0, 1e9]
    labs = ["〜1.6倍", "1.6-2倍", "2-2.5倍", "2.5-3倍", "3-4倍", "4-6倍", "6倍〜"]
    df["band"] = pd.cut(df["o1"], bands, labels=labs)
    for lab, key, _ in MENU:
        g0 = df[df[key].notna()]
        if len(g0) < 1000:
            continue
        line = LINE[key] * 100
        print(f"\n■ {lab}（線 {line:.1f}%）")
        print(f"{'1番人気のオッズ':<14}{'R数':>9}{'的中率':>9}{'ROI':>9}{'線との差':>11}{'99%CI':>22}")
        vals = []
        for b, g in g0.groupby("band", observed=True):
            if len(g) < 300:
                continue
            x = g[key].to_numpy(float) - 100.0
            m, lo, hi = mci(x)
            vals.append((labs.index(str(b)), m + 100))
            print(f"{str(b):<14}{len(g):>9,}{(g[key] > 0).mean()*100:>8.1f}%"
                  f"{(m+100):>8.1f}%{(m+100)-line:>+10.2f}pt"
                  f"{f'[{lo+100:.1f},{hi+100:.1f}]':>22}")
        if len(vals) >= 4:
            a = pd.DataFrame(vals, columns=["i", "roi"])
            print(f"   単調性 ρ={a['i'].corr(a['roi'], method='spearman'):+.3f}"
                  "（負なら仮説どおり『短いほど甘い』）")

    print("\n" + "=" * 104)
    print("【3】★券種の順位 — 最も短い帯（〜1.6倍）で線をどれだけ超えるか")
    print("=" * 104)
    print("  予想した順位: 複勝 > ワイド > 馬連 ≒ 枠連 > 三連複")
    print(f"{'買い方':<20}{'R数':>9}{'ROI':>9}{'線':>8}{'線との差':>11}{'99%CI':>22}")
    out = []
    for lab, key, _ in MENU:
        g = df[(df["band"] == "〜1.6倍") & df[key].notna()]
        if len(g) < 300:
            continue
        x = g[key].to_numpy(float) - 100.0
        m, lo, hi = mci(x)
        line = LINE[key] * 100
        out.append((lab, (m + 100) - line))
        print(f"{lab:<20}{len(g):>9,}{(m+100):>8.1f}%{line:>7.1f}%{(m+100)-line:>+10.2f}pt"
              f"{f'[{lo+100:.1f},{hi+100:.1f}]':>22}")
    print("  実際の順位: " + " > ".join(k for k, _ in sorted(out, key=lambda t: -t[1])))

    print("\n" + "=" * 104)
    print("【4】★★機構の直球検定 — 「その組の期待払戻」で層別する")
    print("=" * 104)
    print("  仮説が言っているのは『**その組の期待払戻が小さいほど買われない**』。")
    print("  λ補正Harvilleで組の確率qを出せば、期待払戻 ≈ 払戻率/q が**発走前に計算できる**。")
    print("  ★1番人気のオッズより直接的な検定。ここで単調に出れば機構は確定。\n")
    for lab, key, _ in MENU:
        col = f"q_{key}"
        if col not in df:
            continue
        g0 = df[df[key].notna() & df[col].notna()]
        if len(g0) < 2000:
            continue
        line = LINE[key] * 100
        g0 = g0.copy()
        g0["exp_pay"] = LINE[key] / g0[col] * 100          # 期待払戻[円]（発走前に分かる）
        g0["eb"] = pd.qcut(g0["exp_pay"], 6, labels=False, duplicates="drop")
        print(f"■ {lab}（線 {line:.1f}%）")
        print(f"{'期待払戻の区分':<16}{'R数':>9}{'平均期待払戻':>14}{'実際のROI':>11}"
              f"{'線との差':>11}{'99%CI':>22}")
        vals = []
        for b, g in g0.groupby("eb", observed=True):
            if len(g) < 300:
                continue
            x = g[key].to_numpy(float) - 100.0
            m, lo, hi = mci(x)
            vals.append((int(b), m + 100))
            print(f"{'第'+str(int(b)+1)+'区分':<16}{len(g):>9,}{g['exp_pay'].mean():>13.0f}円"
                  f"{(m+100):>10.1f}%{(m+100)-line:>+10.2f}pt"
                  f"{f'[{lo+100:.1f},{hi+100:.1f}]':>22}")
        if len(vals) >= 4:
            a = pd.DataFrame(vals, columns=["i", "roi"])
            rho = a["i"].corr(a["roi"], method="spearman")
            print(f"   単調性 ρ={rho:+.3f}（**負なら仮説どおり**: 期待払戻が小さい区分ほど甘い）\n")

    print("\n" + "=" * 104)
    print("★読み方")
    print("  ・単調性が複数の券種で出れば、(94)の機構は**複勝固有ではなく市場全体の性質**。")
    print("  ・★**天井の議論は複勝にしか当てはまらない**（最低配当100円に張り付くから）。")
    print("    ワイド・馬連は床に当たらないので、上振れの大きさに原理的な上限が無い。")
    print("    100%を超えるかどうかは**測ってから言う**。先に答えを決めない。")
    print("  ・★確定オッズでの層別なので事後選択が入る。(92)⑤より締切直前にはほぼ確定しているが、")
    print("    **『これで儲かる』と言うには時系列オッズでの裏取りが要る**（宿題1）。")
    print("  ・(94)③より複勝の天井は98.2%。**100%を超える見込みは無い**。")
    print("    ここで確定させたいのは『機構が実在するか』と『どの券種が一番マシか』。")


if __name__ == "__main__":
    main()
