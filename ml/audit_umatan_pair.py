"""(159) ⚠**事前登録外・記述統計**。(158)の「Bは対照と一致するか」を**対応のある差**で読む。

★★なぜ要るか——**(158)のBと対照は独立ではない**。
　**同じレース・同じ枠連の払戻・同じ的中**を使い、**qの出どころだけ**が違う
　（B=馬単の板を枠へ厳密集約 / 対照=馬連の板を枠へ厳密集約）。
　→ **払戻のばらつきが丸ごと共通**。だから
　　⚠**BはROI 117.3%が「まぐれでない」ことの独立な証拠には一切ならない**。
　　★**Bが示せるのは「117.3%は馬連の板に固有の癖ではない」ことだけ**。
　→ **水準どうしのCIを見比べるのは誤り**。**対応のある差**で見る。

測るもの（**これ以上増やさない**）
　1. **レース単位の対応差** d = (Bの損益) − (対照の損益)、平均と99%CI。
　2. **買い目の重なり**（Jaccard）。**重なりが高いほど「別経路」は名ばかり**。
　3. 参考: 片方だけが買うレースの数と、その損益。

⚠**これは(158)の主判定を上書きしない**。(158)のBは事前登録の主判定に**未達**（下端99.8）。
⚠⚠**確定オッズのオラクルであることも変わらない**。**張れる時点は(148)待ち**。

実行: python3 ml/audit_umatan_pair.py [開始年(既定2015)]
"""
import math
import sys

import numpy as np

sys.path.insert(0, "ml")
from audit_cond_split import load_boards
from audit_crosspool import load_races, payoff, zq
from audit_overlay_all import load_board
from audit_crosspool2 import realized
from waku_umatan import waku_of

TH = 1.0 / 0.775          # 主閾値＝1/払戻率（(158)と同じ）
NCMP = 3                  # (158)と同じ Bonferroni


def main():
    y0 = int(sys.argv[1]) if len(sys.argv) > 1 else 2015
    tb, ub, wb = load_board(6, 4), load_board(4, 4), load_boards()
    races = load_races()
    print(f"(159) ⚠事前登録外・記述統計: Bと対照の**対応のある差**（{y0}年以降）")
    print(f"　q出どころ: B=馬単の板→枠（厳密） / 対照=馬連の板→枠（厳密）")
    print(f"　q_pool出どころ: **両方とも枠連の板**。**払戻も的中も同一**＝独立ではない\n")

    recs = []            # (year, pB, cB, pC, cC, nB, nC, nAnd, nOr)
    for r in races:
        yy = r["year"]
        if yy < y0 or not r.get("wakuren"):
            continue
        rl = realized(r)
        if rl is None:
            continue
        a, b, _ = rl
        n, nums = r["n"], [u for u, _, _ in r["horses"]]
        if a not in nums or b not in nums:
            continue
        T, U, W = tb.get(r["rid"]), ub.get(r["rid"]), wb.get(r["rid"])
        if not T or not U or not W:
            continue
        wkey = tuple(sorted((waku_of(a, n), waku_of(b, n))))
        if wkey not in W:
            continue
        v = payoff(r, "枠連(人気順)", [wkey[0], wkey[1]])
        if not v or v <= 0:
            continue

        def pairs(src):
            out = {}
            for k, o in src.items():
                if len(k) != 4 or not k.isdigit() or o <= 0:
                    continue
                i, j = int(k[:2]), int(k[2:])
                if i in nums and j in nums and i != j:
                    out[(i, j)] = out.get((i, j), 0.0) + 1.0 / o
            return out

        def sel_of(src):
            """枠へ厳密集約 → 比が閾値以上の枠組の集合を返す。"""
            ag = {}
            for (i, j), w in src.items():
                k2 = tuple(sorted((waku_of(i, n), waku_of(j, n))))
                ag[k2] = ag.get(k2, 0.0) + w
            keys = [k for k in sorted(ag) if k in W]
            if wkey not in keys or len(keys) < 3:
                return None
            qq = np.array([ag[k] for k in keys]); qq /= qq.sum()
            inv = np.array([1.0 / W[k] for k in keys]); qp = inv / inv.sum()
            rat = qq / qp
            return {k for k, x in zip(keys, rat) if x >= TH}

        sB, sC = sel_of(pairs(T)), sel_of(pairs(U))
        if sB is None or sC is None:
            continue
        cB, cC = 100.0 * len(sB), 100.0 * len(sC)
        pB = (v if wkey in sB else 0.0) - cB
        pC = (v if wkey in sC else 0.0) - cC
        recs.append((yy, pB, cB, pC, cC, len(sB), len(sC),
                     len(sB & sC), len(sB | sC)))

    if not recs:
        sys.exit("突き合わせできたレースが無い。")
    A = np.array([[x[1], x[2], x[3], x[4]] for x in recs], float)
    pB, cB, pC, cC = A[:, 0], A[:, 1], A[:, 2], A[:, 3]
    z = zq(0.01 / NCMP)
    N = len(recs)
    print(f"★突き合わせ {N:,}レース（両方が板を持ち、枠連の払戻がある）\n")

    print("■ 水準（参考・(158)と同じ数字）")
    for nm, p, c in (("B 馬単→枠", pB, cB), ("対照 馬連→枠", pC, cC)):
        buy = int((c > 0).sum())
        print(f"{nm:>14}  買ったR {buy:>7,}  ROI {100*(p.sum()+c.sum())/c.sum():>6.1f}%")

    d = pB - pC
    md, sd = d.mean(), d.std(ddof=1) / math.sqrt(N)
    mc = (cB.sum() + cC.sum()) / (2 * N)      # 1レースあたり平均投資（両者の平均）
    print("\n■ ★対応のある差 d = Bの損益 − 対照の損益（**同じ払戻・同じ的中**）")
    print(f"　1レースあたり {md:+.2f}円　99%CI(Bonf) [{md-z*sd:+.2f}, {md+z*sd:+.2f}]")
    print(f"　ROI換算の差 {100*md/mc:+.2f}pt"
          f"　99%CI [{100*(md-z*sd)/mc:+.2f}, {100*(md+z*sd)/mc:+.2f}]pt")
    print(f"　→ **{'差は検出できない＝同じものを見ている' if abs(md) < z*sd else '⚠差がある'}**")

    nand = np.array([x[7] for x in recs], float)
    nor = np.array([x[8] for x in recs], float)
    nB = np.array([x[5] for x in recs], float)
    nC = np.array([x[6] for x in recs], float)
    j = nand.sum() / max(nor.sum(), 1)
    print("\n■ ★買い目の重なり（Jaccard = |B∩対照| / |B∪対照|）")
    print(f"　{j:.3f}　（B {nB.sum():,.0f}点 / 対照 {nC.sum():,.0f}点 /"
          f" 共通 {nand.sum():,.0f}点）")
    print(f"　→ **{'⚠ほぼ同じ買い目。「別経路」は名ばかり' if j > 0.8 else 'ある程度は別の買い目'}**")

    onlyB = (nB > 0) & (nC == 0)
    onlyC = (nC > 0) & (nB == 0)
    print("\n■ 片方だけが買うレース")
    for nm, m, p, c in (("Bだけ", onlyB, pB, cB), ("対照だけ", onlyC, pC, cC)):
        if m.sum() == 0:
            print(f"{nm:>10}  なし")
            continue
        print(f"{nm:>10}  {int(m.sum()):>6,}R  ROI {100*(p[m].sum()+c[m].sum())/max(c[m].sum(),1):>6.1f}%")

    print("\n" + "=" * 88)
    print("★★読み方（**(158)の主判定は上書きしない**）")
    print("  ⚠**Bは「117.3%がまぐれでない」ことの独立な証拠にならない**——払戻が共通だから。")
    print("  ★**Bが言えるのは「馬連の板に固有の癖ではない」ことだけ**。それは言えた。")
    print("  ⚠⚠**確定オッズのオラクル**。**張れる時点で成立するかは(148)待ちのまま**。")


if __name__ == "__main__":
    main()
