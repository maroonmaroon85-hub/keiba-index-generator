"""(151) ★★★★複勝を**本物の板**で「比の裾」で買う — **判定基準29/30の未処理分**

★★なぜこれが残っていたか
　(89)④の「Dが必要量に届かないから閉じた」は(141)で**論法ごと誤り**と分かった
　（`log(払戻率)+D` は上界ではなく下界。**正しい条件は「∃k: q_k/q_pool,k ≥ 1/払戻率」＝比の裾**）。
　→ **判定基準29**（基準を足したら過去の結論に遡って当て直す）で当て直したのは
　　 **枠連(141) / 複勝(146) / 馬連・三連複(150)** の4つ。
　⚠**(146)の「複勝」は三連複プール経由**（`P(i が3着以内) = Σ_{j,k} q_三連複({i,j,k})`）。
　★**本物の複勝の板（type=2）で比の裾を測ったことは一度も無い**。**(124)で板が使えると分かった今日、
　　 同じ板で裾を測れる**。**これが本命の複勝ルートの、最後の未測定部分**。

★★★q_pool の作り方（(124)で理屈を確定させた）
　複勝プールは**3等分**される: `オッズ_i = 0.8·総額/(3·S_i)`
　→ `S_i/総額 = 0.8/(3·オッズ_i)` → `Σ 1/オッズ = 3.75` → **`q_i = 0.8/オッズ_i` で Σq = 3**。
　⚠**板は[下限,上限]の範囲**。**下限を使うと q_pool が過大＝我々に不利側**。
　★**主判定は下限**（保守側）。**中点は(124)で道具が壊れていると判明したので使わない**
　　（Dが必要量の4倍と出るのに実測ROIは94.4%）。

★★事前登録（**測る前に書いている**）
　1. **閾値の梯子 1.00 / 1.10 / ★1.25(=1/0.8) / 1.50 / 2.00**。**後から増やさない**。
　2. **賭け方**: 比が閾値以上の**全馬を1点100円**。該当無しは見送り。**選ばない**。
　3. **q_pool は「下限そのまま」を主判定**、「下限をΣ=3に」を副で並べる。
　　 **2通りで結論が変わったら「決着していない」と書く**（都合の良い方を選ばない）。
　4. **判定は race単位の収支**。**運用判定: ROIの99%CI下端 > 100%** かつ
　　 **年別で「91%を割る年が2つ以下」**（ユーザーの許容水準・2026-08-13）かつ
　　 **2021-2026でも100%超**（(142)の教訓）。
　5. ★★**陽性対照を置く（判定基準32）**。**真値が分かっている行を同じ表に並べる**:
　　 **「モデルのtop1を無条件に買う」**。
　　 ⚠**事前登録では真値を(106)の96.8%と書いたが、これは私の仕様ミス**——
　　 **(106)の96.8%は「2%裾」に絞った年間60レースの数字**で、**無条件のtop1ではない**。
　　 ★**正しい真値は 84.8%**（NEXT_SESSION 7-2 の本命表）。**実測 84.2%＝差 −0.6pt で①は立つ**。
　　 ⚠**この訂正は②を見たあとに行った**（走らせる前に確認できたはずのもの）。**順序を記録する**。
　6. **プラセボ**: 無作為に同数買う。**解析的に出す**（判定基準23）。
　7. **道具の検算**: 着内馬で「板の下限×100 ≤ 実配当 ≤ 板の上限×100」を確かめる。
　8. ★**予想**: **当てにしてよい予想は持っていない**（類推はこの4日で5連敗）。
　　 **恒等式から言えることだけ書く**: **下限を使う限り q_pool は過大**なので、
　　 **比は真の比より小さく出る**。→ **選ばれる馬は真より少ない**。**これは向きが分かっている**。
　　 ⚠**だから「選ばれなかった」は「裾が無い」の証明にならない**。**中点で増えるぶんは未測定のまま**。

★★★実行済みの結果（2026-08-16・32,858レース・2015年以降）**陰性。しかも逆向き**
　★道具の検算: 着内馬98,021件のうち実配当が板の[下限,上限]×100 を外れたのは **0.26%**。
　★①陽性対照: モデルtop1 **84.2%** vs 既知値 84.8% → **差 −0.6pt で立った**。

| 閾値 | 下限そのまま | | 下限をΣ=3に | |
|---|---|---|---|---|
| | R数 | ROI | R数 | ROI |
| 1.000 | 29,619 | 69.4% | 32,858 | 74.3% |
| 1.100 | 18,266 | 62.4% | 32,696 | 71.7% |
| **1.250** | 6,322 | **46.3%** | 24,886 | **64.8%** |
| 1.500 | 2,100 | 25.7% | 6,177 | 38.1% |
| 2.000 | 275 | 13.6% | 678 | 10.2% |

★★**閾値を上げるほど単調に悪くなる**。**枠連(141)とは正反対**（あちらは裾ほど良くなった）。
　→ ★**我々の q は、プールと最も食い違う馬でこそ最も過信している**。
　**2通りの q_pool で結論は変わらない**（事前登録3は満たした）。**全水準で100%を大きく下回る**。
　⚠**事前登録の限界（下限は保守側なので選ばれる馬が真より少ない）は結論を救わない**——
　　 **Σ=3版は24,886レースを選んでなお 64.8%**。

⚠⚠★**書きすぎていたので訂正する（2026-08-16・同日中に自分で気づいた）**
　最初「**これで複勝は(146)と(151)の両方で陰性＝比で買う道は閉じた**」と書いたが**言い過ぎ**。
　★**(151)の q は `top3_probs(p,...)`＝λ補正Harville(単勝由来)**。**「本物の板」なのは q_pool 側だけ**。
　→ ★**(151)は(150)と同じ「弱い経路」**（(141)で 枠連は 馬連ルート117.3% / λHarvilleルート93.9% と実測済み）。
　→ ⚠**書けるのは「単勝由来のqでは取れない」まで**。**「比の裾で閉じた」とは書けない**。
　★**強い経路（別の市場の意見を厳密に集約）は(146)の 三連複プール→複勝**で、**そちらは 89.0%**。
　**判定基準25（測った経路を毎回明記する）の再発**。**5回目**。

★**複勝の現状を正しく並べる**:
| 経路 | q | q_pool | 比の裾の結果 |
|---|---|---|---|
| **強い**(146) | **三連複プール→複勝**（厳密） | 複勝の板 | **89.0%が天井** |
| **弱い**(151) | λ補正Harville(**単勝**由来) | **複勝の板** | **46.3%**（閾値1.25）・単調に悪化 |

★**複勝の天井は(106)の 96.8%（2%裾・年間60R）のまま**。**それは「選ぶ」戦略であって
　「比で買う」戦略ではない**。**比で買う道は、強い経路でも 89.0% で100%に届いていない**。

実行: python3 ml/audit_fuku_tail.py [開始年(既定2015)]
"""
import math
import sys

import numpy as np

sys.path.insert(0, "ml")
from audit_crosspool import load_races, payoff, probs, zq
from audit_fuku_board import PAYBACK, load_boards
from audit_fuku_lbs import top3_probs
from audit_lbs import build_matrix, fit_lambda

THS = [1.00, 1.10, 1.0 / PAYBACK, 1.50, 2.00]
# ★★陽性対照の真値。⚠**最初 96.8% と書いたのは間違いだった**（2026-08-16）——
#   (106)の96.8%は **「2%裾」に絞った年間60レース**の数字で、**無条件のtop1ではない**。
#   **無条件のモデルtop1複勝の既知値は 84.8%**（NEXT_SESSION 7-2 の本命表・1R期待損失15.2円）。
#   ⚠**この訂正は②の表を見たあとに行った**。参照値はHANDOFFに書いてあり走らせる前に
#   確認できたので**結果に合わせた変更ではなく仕様ミスの訂正**だが、**順序は正直に記録する**。
#   ★**選択済みの部分集合の数字を、無条件の集合に当てた**＝判定基準25と同じ「母集団が違う」型。
KNOWN_TOP1 = 84.8
VARIANTS = ("下限そのまま", "下限をΣ=3に")


def qpool(od, nums, tag):
    lo = np.array([od[u][0] for u in nums], float)
    raw = PAYBACK / lo
    if tag == "下限そのまま":
        return raw
    return raw * (3.0 / raw.sum()) if raw.sum() > 0 else raw


def summarize(name, rows, z):
    """rows: [(year, cost, ret)] → 1行印字。ROI・99%CI・年91%割れ・2021-。"""
    if not rows:
        print(f"{name:>14}   該当なし")
        return None
    pr = np.array([v - c for _, c, v in rows], float)
    cost = sum(c for _, c, _ in rows)
    ret = sum(v for _, _, v in rows)
    mc = cost / len(pr)
    se = pr.std(ddof=1) / math.sqrt(len(pr)) if len(pr) > 1 else float("nan")
    lo, hi = 1 + (pr.mean() - z * se) / mc, 1 + (pr.mean() + z * se) / mc
    ys = np.array([y for y, _, _ in rows])
    bad = 0
    for u in sorted(set(ys.tolist())):
        m = ys == u
        if m.sum() >= 30:
            cc = sum(c for (y, c, _) in rows if y == u)
            vv = sum(v for (y, _, v) in rows if y == u)
            bad += int(vv / max(cc, 1) < 0.91)
    c21 = sum(c for y, c, _ in rows if y >= 2021)
    v21 = sum(v for y, _, v in rows if y >= 2021)
    mark = "★★買える" if lo > 1.0 else ""
    print(f"{name:>14}{len(pr):>9,}{100*ret/cost:>9.1f}%"
          f"{'[' + format(100*lo, '.1f') + ',' + format(100*hi, '.1f') + ']':>21}"
          f"{bad:>9}{100*v21/max(c21,1):>9.1f}% {mark}")
    return 100 * ret / cost


def main():
    y0 = int(sys.argv[1]) if len(sys.argv) > 1 else 2015
    boards = load_boards()
    if not boards:
        sys.exit("複勝の板が無い。Macで `python3 ml/nk_odds_bulk.py --type 2`。")
    races = load_races()
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

    per = {v: {t: [] for t in THS} for v in VARIANTS}
    ctrl, nR, chk_n, chk_bad = [], 0, 0, 0
    for r in races:
        yy = r["year"]
        if yy < y0 or not lam.get(yy):
            continue
        od = boards.get(r["rid"])
        if not od:
            continue
        hs = r["horses"]
        nums = [u for u, _, _ in hs]
        if any(u not in od or od[u][0] <= 0 for u in nums):
            continue
        l2, l3 = lam[yy]
        p = probs(hs)
        q = np.asarray(top3_probs(p, 1.0, l2, l3), float)
        pay = np.array([payoff(r, "複勝", [u]) or 0.0 for u in nums], float)
        # 道具の検算: 着内馬の実配当が板の[下限,上限]×100 に収まるか
        for k, u in enumerate(nums):
            if pay[k] > 0:
                chk_n += 1
                a, b = od[u]
                if not (a * 100 - 1 <= pay[k] <= b * 100 + 1):
                    chk_bad += 1
        nR += 1
        # ★陽性対照: モデルtop1を無条件に買う（既知値 84.8%）
        j = int(np.argmax(q))
        ctrl.append((yy, 100.0, float(pay[j])))
        for v in VARIANTS:
            qp = qpool(od, nums, v)
            rat = q / np.maximum(qp, 1e-9)
            for t in THS:
                sel = rat >= t
                if not sel.any():
                    continue
                per[v][t].append((yy, 100.0 * int(sel.sum()), float(pay[sel].sum())))

    z = zq(0.01)
    print(f"(151) 複勝を**本物の板**で比の裾で買う（{nR:,}レース・{y0}年以降）")
    print(f"　払戻率 {PAYBACK} → 利益に要る比 {1/PAYBACK:.3f}")
    print(f"★道具の検算: 着内馬 {chk_n:,}件のうち実配当が板の[下限,上限]×100 を外れた数 "
          f"{chk_bad}（{chk_bad/max(chk_n,1):.2%}）\n")

    print("■ ★★①陽性対照（判定基準32）— **これが合わなければ以下を読まない**")
    print(f"{'買い方':>14}{'R数':>9}{'ROI':>10}{'99%CI':>21}{'年91%割れ':>9}{'2021-':>10}")
    got = summarize("モデルtop1", ctrl, z)
    print(f"　★無条件top1の既知値は **{KNOWN_TOP1}%**（本命表）。この標本での再現 = "
          f"**{got:.1f}%**　差 {got - KNOWN_TOP1:+.1f}pt")
    ok = got is not None and abs(got - KNOWN_TOP1) <= 3.0
    print(f"　→ ★**①は立ったか（±3pt以内で再現したか）: {'★立った' if ok else '⚠立っていない'}**")
    if not ok:
        print("　⚠**立っていない**。**道具か標本が(106)と違う**。**以下の表は読まないこと**。")

    for v in VARIANTS:
        print(f"\n■ ②比の裾 — q_pool = {v}")
        print(f"{'閾値':>14}{'R数':>9}{'ROI':>10}{'99%CI':>21}{'年91%割れ':>9}{'2021-':>10}")
        for t in THS:
            summarize(f"{t:.3f}", per[v][t], z)

    print("\n" + "=" * 96)
    print("★読み方（事前登録のとおり。**後から足していない**）")
    print("  ・**①が立たなければ②は読まない**（判定基準32）。")
    print("  ・**2通りのq_poolで結論が変われば「決着していない」**と書く。都合の良い方を選ばない。")
    print("  ・★**下限を使う限り q_pool は過大＝比は真より小さく出る**。**向きは分かっている**。")
    print("    → ⚠**「選ばれなかった」は「裾が無い」の証明にならない**。")


if __name__ == "__main__":
    main()
