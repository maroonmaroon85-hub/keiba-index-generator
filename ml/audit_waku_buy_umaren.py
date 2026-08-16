"""(153) ★★★★枠連の買い目を**馬連プール**で選ぶ — **(148)を待たずに試せる唯一の筋**

★★なぜこれが残っていたか（2026-08-16の監査で発見）
　(127)(135)は「**枠連の q は λ補正Harville(単勝)→枠 より 馬連プール→枠 のほうが良い**」
　（**D +0.0182 → +0.0266**）と実測し、「**発走前に同じAPIで取れるので運用可能**」と書いた。
　⚠**だが運用（`predict_nk.py` → `waku_umatan.bracket_probs`）は今もモデル確率を枠に集約している**。
　★**この改善は「オーバーレイ(141)の土台」としてだけ使われ、「買い目の選び方」としては
　　一度も試されていない**。

★★★ここが肝: **これは「順位で買う」戦略なので(148)を待たなくてよい**
　判定基準31: **順位で買う戦略は時点に頑健／比で買う戦略は時点に脆い**。
　・(141)のオーバーレイは **q/q_pool の比**で買う → **張れる時点で成立するか未確定((148))**。
　・★**本件は「q_馬連→枠 が最大の枠組を1点買う」＝純粋な順位**。
　　 → **(66)の「当日朝以降なら順位は劣化しない」がそのまま効く**。**今日から試せる**。

★★事前登録（**測る前に書いている**）
　1. **買い方は3つだけ**。**後から増やさない**:
　　 **A 現行**: 軸枠×紐枠1（**モデル**降順の1位馬の枠 × 2位馬の枠）**1点**
　　 **B 馬連プール**: `q_枠(a,b) = Σ_{i∈a,j∈b} 1/馬連オッズ(i,j)` が**最大の枠組1点**
　　 **C 枠連プール自身**: 枠連の板で**最も人気の枠組1点**（＝**市場そのもの。負けて当然の対照**）
　2. ★**判定は ROI ではなく「1レース期待損失（円）」**（HANDOFF冒頭3・(80)）。
　　 **ROI差は検出限界未満で測れない**。**AとBの差にCIを付ける（対応のある差）**。
　3. ★★**陽性対照（判定基準32）**: **Aの既知値は ROI 85.2% / 1R期待損失 14.8円**。
　　 → **±3pt / ±3円 で再現しなければ、BもCも読まない**。
　4. **年別**: **91%を割る年が2つ以下**（ユーザーの許容水準）。**2021-2026 も出す**（(142)）。
　5. **プラセボ**: **無作為な枠組1点**。**解析的に出す**（判定基準23）。
　6. **道具の検算**: 的中組で「枠連の板×100 と実配当」のずれ。**(127)は0.07%だった**。
　7. ★**予想**: **当てにしてよい予想は持っていない**（類推はこの4日で5連敗）。
　　 **恒等式から言えることだけ**: **Bは市場情報だけで作る**ので、**Cに近づく方向**。
　　 **Cは市場の最人気＝控除率ぶんきっちり負けるはず**。**BがAに勝つ保証はどこにも無い**。
　　 ⚠**(127)の「qが良い」は較正の話**。**「買い目が良い」は順位の話**。**別物**（判定基準31・(121)(133)）。

⚠★**この実験が陰性でも「馬連プールは使えない」とは書かない**。書けるのは
　**「1点買いの順位付けとしては現行に勝てない」**まで。**較正としての優位((127))は別に生きている**。

★★★実行済みの結果（2026-08-16・34,160レース・2015年以降）**差は無い**
　★道具の検算: 的中34,160件のうち板×100と実配当のずれ **0.07%**（(127)と一致）。
　★①陽性対照: A(現行)の再現 **83.3% / 16.7円** vs 既知値 85.2% / 14.8円 → **±3pt以内で立った**。

| 買い方 | ROI | 1R期待損失 | 2021- | Aとの差（対応のある差） |
|---|---|---|---|---|
| **A 現行(モデル)** | 83.3% | 16.7円 | 83.4% | — |
| **B 馬連プール** | 83.3% | 16.7円 | 84.1% | **+0.03円 99%CI[−2.46,+2.51]** ＝**差は無い** |
| C 枠連の最人気 | 80.6% | 19.4円 | 80.5% | +2.72円 [+0.21,+5.22] ＝**Aが有意に勝つ** |
| プラセボ(無作為) | 61.9% | 38.1円 | — | 解析的（判定基準23） |

★★**結論: 馬連プールで買い目を選んでも現行と全く同じ**。**運用は変えない**。
　⚠**だがこれは「情報が無い」ではない**——**AとBは30%のレースで違う枠組を選んでいる**
　（一致率69.9%・**事前登録に無い後から足した診断**）。**別の組を選んでなお同点**。

★★★**これが本件のいちばんの収穫**: **(127)は「馬連→枠 のほうが良い q」と実測した**
　（D +0.0182 → +0.0266）。**なのに「どの組を買うか」では1円も勝てない**。
　→ ★**較正の質と順位の質は本当に別物**（(121)(133)・判定基準31）。**3度目の確認**。
　→ ★**(127)の +0.0266 は「オーバーレイ((141)(148))の土台としてのみ価値がある」**と確定した。
　　 **順位で使う道はここで閉じた**。**(148)の結果が出るまで、この筋に上積みは無い**。

⚠**書けるのはここまで**: **「1点買いの順位付けとしては現行に勝てない」**。
　**較正としての優位((127))は別に生きている**（事前登録の限界どおり）。

実行: python3 ml/audit_waku_buy_umaren.py [開始年(既定2015)]
"""
import math
import sys

import numpy as np

sys.path.insert(0, "ml")
from audit_cond_split import load_boards
from audit_crosspool import PAYBACK, load_races, payoff, probs, zq
from audit_crosspool2 import realized
from audit_overlay_all import load_board
from waku_umatan import waku_of

KNOWN_ROI, KNOWN_YEN = 85.2, 14.8      # ★現行(A)の既知値。陽性対照の真値


def agg_umaren(ub, nums, n):
    """馬連の板 → 枠組の含意確率（正規化前）。**Harville不使用・厳密な集約**（(127)）。"""
    out = {}
    for k, o in ub.items():
        if len(k) != 4 or not k.isdigit() or o <= 0:
            continue
        x, y = int(k[:2]), int(k[2:])
        if x not in nums or y not in nums:
            continue
        a, b = sorted((waku_of(x, n), waku_of(y, n)))
        out[(a, b)] = out.get((a, b), 0.0) + 1.0 / o
    return out


def summarize(name, rows, z, base=None):
    if not rows:
        print(f"{name:>16}   該当なし")
        return None
    pr = np.array([v - 100.0 for _, v in rows], float)
    ys = np.array([y for y, _ in rows])
    roi = 100.0 * (1 + pr.mean() / 100.0)
    se = pr.std(ddof=1) / math.sqrt(len(pr))
    bad = sum(1 for u in sorted(set(ys.tolist()))
              if (ys == u).sum() >= 30 and (1 + pr[ys == u].mean() / 100.0) < 0.91)
    m21 = ys >= 2021
    r21 = 100.0 * (1 + pr[m21].mean() / 100.0) if m21.sum() else float("nan")
    ex = f"{-pr.mean():>9.1f}円"
    d = ""
    if base is not None:
        dd = pr - base
        sd = dd.std(ddof=1) / math.sqrt(len(dd))
        d = (f"  差 {-dd.mean():+7.2f}円 99%CI["
             f"{-(dd.mean()+z*sd):+.2f},{-(dd.mean()-z*sd):+.2f}]")
    print(f"{name:>16}{len(pr):>9,}{roi:>8.1f}%{ex}"
          f"{'[' + format(roi - 100*z*se/100, '.1f') + ',' + format(roi + 100*z*se/100, '.1f') + ']':>16}"
          f"{bad:>8}{r21:>8.1f}%{d}")
    return pr


def main():
    y0 = int(sys.argv[1]) if len(sys.argv) > 1 else 2015
    wb = load_boards()                    # 枠連の板 {rid: {(a,b): odds}}
    ub = load_board(4, 4)                 # 馬連の板 {rid: {"0102": odds}}
    if not wb or not ub:
        sys.exit("枠連(--type 3)と馬連(--type 4)の板が要る。")
    races = load_races()
    A, B, C, PL, SAME, chk_n, chk_bad = [], [], [], [], [], 0, 0
    for r in races:
        yy = r["year"]
        if yy < y0 or not r.get("wakuren"):
            continue
        rl = realized(r)
        if rl is None:
            continue
        a, b, _ = rl
        n = r["n"]
        hs = r["horses"]
        nums = [u for u, _, _ in hs]
        if a not in nums or b not in nums:
            continue
        W, U = wb.get(r["rid"]), ub.get(r["rid"])
        if not W or not U:
            continue
        key = tuple(sorted((waku_of(a, n), waku_of(b, n))))
        if key not in W:
            continue
        v = payoff(r, "枠連(人気順)", [key[0], key[1]])
        if not v or v <= 0:
            continue
        chk_n += 1
        if abs(W[key] * 100 - v) > max(10.0, v * 0.01):
            chk_bad += 1
        # A: 現行（モデル≒単勝オッズ降順の1位×2位の枠）
        p = probs(hs)
        order = np.argsort(-p)
        w1, w2 = waku_of(nums[order[0]], n), waku_of(nums[order[1]], n)
        ka = tuple(sorted((w1, w2)))
        # B: 馬連プール→枠 が最大の組
        ag = {k: x for k, x in agg_umaren(U, nums, n).items() if k in W}
        if not ag:
            continue
        kb = max(ag, key=lambda k: ag[k])
        # C: 枠連の板で最も人気（オッズ最小）の組
        kc = min(W, key=lambda k: W[k])
        SAME.append(int(ka == kb))
        A.append((yy, v if ka == key else 0.0))
        B.append((yy, v if kb == key else 0.0))
        C.append((yy, v if kc == key else 0.0))
        PL.append((yy, v / len(W)))       # プラセボ: 無作為1点（解析的）

    z = zq(0.01)
    print(f"(153) 枠連の買い目を馬連プールで選ぶ（{len(A):,}レース・{y0}年以降）")
    print(f"★道具の検算: 的中 {chk_n:,}件のうち板×100と実配当のずれ {chk_bad}"
          f"（{chk_bad/max(chk_n,1):.2%}）  ※(127)は0.07%\n")
    print(f"{'買い方':>16}{'R数':>9}{'ROI':>9}{'1R期待損失':>10}{'99%CI':>16}"
          f"{'年91%割れ':>8}{'2021-':>9}  差（Aとの対応のある差）")
    pa = summarize("A 現行(モデル)", A, z)
    print(f"　★陽性対照: 既知値 ROI {KNOWN_ROI}% / 1R期待損失 {KNOWN_YEN}円。"
          f"再現 {100*(1+pa.mean()/100):.1f}% / {-pa.mean():.1f}円")
    ok = abs(100 * (1 + pa.mean() / 100) - KNOWN_ROI) <= 3.0
    print(f"　→ ★**①は立ったか（±3pt）: {'★立った' if ok else '⚠立っていない'}**")
    if not ok:
        print("　⚠**立っていない。以下を読まないこと**（判定基準32）。\n")
    summarize("B 馬連プール", B, z, pa)
    summarize("C 枠連の最人気", C, z, pa)
    pl = np.array([v - 100.0 for _, v in PL], float)
    print(f"{'プラセボ(無作為)':>16}{len(pl):>9,}{100*(1+pl.mean()/100):>8.1f}%"
          f"{-pl.mean():>9.1f}円   ※解析的（判定基準23）")

    print(f"\n■ ⚠**事前登録に無い後から足した診断**（**判定には使わない**・判定基準27）")
    print(f"　**AとBが同じ枠組を選んだ割合: {np.mean(SAME):.1%}**"
          f"　→ **差が出ないのは当然か、それとも別の組を選んでなお同点か**を読むための数字。")

    print("\n" + "=" * 104)
    print("★読み方（事前登録のとおり。**後から足していない**）")
    print("  ・★**判定はROIではなく1レース期待損失（円）の差**。**CIが0をまたげば「差は無い」**。")
    print("  ・**BがAに有意に勝てば運用を替える候補**。**負けても「馬連プールは使えない」とは書かない**")
    print("    （書けるのは**『1点買いの順位付けとしては勝てない』**まで。較正の優位((127))は別）。")
    print("  ★**これは順位で買う戦略なので(148)を待たずに運用に載せられる**（判定基準31・(66)）。")


if __name__ == "__main__":
    main()
