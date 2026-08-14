"""(141) ★★★(89)④の上界は向きが逆だった — **正しい条件で測り直す**（2026-08-13・ユーザー依頼の棚卸しで発見）

★★★何が間違っていたか
　HANDOFF冒頭5.と(89)④はこう書いている:
　> **成長率 = log(払戻率) + D**。ケリーが上限なので**どんな買い方でも超えられない**。
　> → **買い方・絞り込み・資金配分の探索はもう不要**。
　⚠**これは「資金を全額 q に比例して張る」ときの成長率**であって、**現金を残す自由を捨てている**。
　　ケリー最適は**現金を持てる**ので、**必ず 0 以上**。つまり `log(払戻率)+D` は**下界**であって上界ではない。

　★**反例（数値で確認済み）**: 2組・払戻率0.775・q_pool=(0.90,0.10)・p=(0.85,0.15) のとき
　　`D=+0.0122`、`log(R)+D = −0.2427`（負）。**それでも組2だけに f=0.024 張れば成長率 +0.0019（正）**。
　　理由: **組2の p/q_pool = 1.50 が 1/払戻率 = 1.290 を超えている**から。

★★★正しい条件
```
　利益の出る賭けが存在する  ⇔  ∃組 k:  p_k / q_pool,k  >  1 / 払戻率
```
　**D（平均の対数比）ではなく、比の「裾」が問題**。**平均が小さくても裾が 1.29 を超えれば賭けは成立する**。
　★(136)は**この条件（O/E ≥ 1.290）を正しく使っていた**が、**14個の粗いラベルに集約して**測ったので、
　　**集約が裾を潰していた**（最大 1.044）。**組み合わせ単位では潰れない**。

★★なぜ今できるのか（(127)のおかげ）
　比 `q/q_pool` に意味があるのは、**q と q_pool が別の市場から来ているとき**だけ。
　(127)で **馬連プール→枠**（Harvilleもモデルも通さない）が手に入った。
　→ **q＝馬連プール由来 / q_pool＝枠連プール** の比は、**2つの独立な市場の食い違い**そのもの。
　★**モデルを1つも使わないので過学習の余地がゼロ**。λも要らない（馬連→枠は集約するだけ）。

★★事前登録（測る前に宣言する）
　1. **閾値の梯子 1.00 / 1.10 / **1.29（=1/払戻率）** / 1.50 / 2.00**。**後から水準を増やさない**。
　2. **賭け方**: 各レースで比が閾値以上の**全組を1点100円ずつ**買う（選ばない。**恣意を入れない**）。
　　 該当が無いレースは**見送り**（★これがケリーの「現金を持つ」に当たる）。
　3. **★判定は race単位の収支**。1レースの収支 = Σ_{当たった組} 板オッズ×100 − 100×点数。
　　 **レース間は独立**なのでそのまま平均とCIが取れる。**組の間は従属（高々1組しか当たらない）**ので
　　 組単位でCIを出してはいけない。
　4. **★運用判定: ROIの99%CI下端 > 100%** かつ **年8/11以上でプラス**。
　　 満たさなければ**この道も閉じる**（ただし**(89)④の論法ではなく、実測で閉じたことになる**）。
　5. **プラセボ**: レース内で比を**組にランダムに割り当て直して**同じ点数だけ買う。**200回平均**。
　　 ★判定基準28の**不偏な部分標本ではない**（選択が結果と独立になる）ので**構造上ゼロにはならず**、
　　 　**「同じ点数を無作為に買ったときのROI」＝控除率まわり**に落ちるはず。**そこと比べる**。
　6. **比較のため λHarville→枠 でも同じ梯子を引く**（単一市場由来なので**弱いはず**）。
　7. ⚠**正直に書く限界（2つとも重い）**
　　 (a) **使っているのは確定オッズ**。実際には**締切前に張る**しかなく、自分の賭けが
　　 　 オッズを動かす。(94)は「**確定オッズを知っているオラクルでも単勝98.2%**」と測っている。
　　 　 → **ここで出る数字は上振れ側のオラクル値**。**100%を超えなければ本当に閉じる**。
　　 (b) **枠連は9頭以上でしか発売されない**ので対象は限られる。
　8. **予想**: ★**当てにしてよい予想は持っていない**。
　　 ⚠(136)が粗いラベルで最大1.044だったことは**集約後の話なので組単位には及ばない**（判定基準25）。
　　 ⚠(110)は「全組み合わせ走査」をしたが**選択基準は max q（＝最も当たりやすい組）**であって
　　 　**比 q/q_pool ではない**。**別の量なので(110)はこの問いに答えていない**。

実行: python3 ml/audit_overlay.py [開始年(既定2015)]
"""
import math
import sys

import numpy as np

sys.path.insert(0, "ml")
from audit_capacity_d import mkt_waku_dist
from audit_cond_split import load_boards
from audit_crosspool import PAYBACK, load_races, probs, zq
from audit_crosspool2 import realized
from audit_lbs import build_matrix, fit_lambda
from audit_waku_vs_umaren import load_type
from waku_umatan import waku_of

R = PAYBACK["枠連"]
THS = [1.00, 1.10, 1.0 / R, 1.50, 2.00]      # ★1/払戻率 = 1.290 を含む。後から増やさない
NPLA = 200
RNG = np.random.default_rng(20260813)


def mci(x, alpha=0.01):
    x = np.asarray(x, float)
    if len(x) < 2:
        return float("nan"), float("nan"), float("nan")
    m = x.mean()
    se = x.std(ddof=1) / math.sqrt(len(x))
    return m, m - zq(alpha) * se, m + zq(alpha) * se


def main():
    y0 = int(sys.argv[1]) if len(sys.argv) > 1 else 2015
    races = load_races()
    wb, ub = load_boards(), load_type(4, 4)
    if not wb or not ub:
        sys.exit("枠連(--type 3)と馬連(--type 4)の板が両方要る。")

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

    recs = []
    for r in races:
        yy = r["year"]
        if yy < y0 or not lam.get(yy) or not r["wakuren"]:
            continue
        W, U = wb.get(r["rid"]), ub.get(r["rid"])
        if not W or not U:
            continue
        rl = realized(r)
        if rl is None:
            continue
        a, b, _ = rl
        n = r["n"]
        nums = [u for u, _, _ in r["horses"]]
        if a not in nums or b not in nums:
            continue
        p = probs(r["horses"])
        l2, _ = lam[yy]
        md = mkt_waku_dist(r, p, l2)
        if not md:
            continue
        agg = {}
        for (x, y), o in U.items():
            if o <= 0:
                continue
            wx, wy = sorted((waku_of(x, n), waku_of(y, n)))
            agg[(wx, wy)] = agg.get((wx, wy), 0.0) + 1.0 / o
        keys = [k for k in sorted(md) if k in W and k in agg]
        if len(keys) < 3:
            continue
        key = tuple(sorted((waku_of(a, n), waku_of(b, n))))
        if key not in keys:
            continue
        inv = np.array([1.0 / W[k] for k in keys])
        qp = inv / inv.sum()                                  # 枠連プール
        qu = np.array([agg[k] for k in keys]); qu /= qu.sum()  # 馬連→枠
        qh = np.array([md[k] for k in keys]); qh /= qh.sum()   # λHarville→枠
        odds = np.array([W[k] for k in keys])
        win = np.array([k == key for k in keys])
        recs.append((yy, qp, qu, qh, odds, win, float(inv.sum())))

    n = len(recs)
    nyr = len(set(x[0] for x in recs))
    print(f"(141) 比 q/q_pool の裾で買えるか（{y0}年以降・{n:,}レース・{nyr}年）")
    print(f"★(89)④の『log(払戻率)+D が上界』は**向きが逆**（下界）。正しい条件は")
    print(f"　 **∃組: q/q_pool ≥ 1/払戻率 = {1/R:.3f}**。**組単位で測る**（集約は裾を潰す）")
    print(f"⚠確定オッズを使っている＝**オラクル**。(94)は単勝で98.2%が天井と測っている\n")

    for tag, QI in (("★★馬連→枠（(127)。モデルもHarvilleも使わない）", 2),
                    ("λHarville→枠（単一市場由来。弱いはず）", 3)):
        print(f"■ {tag}")
        print(f"{'閾値':>7}{'買ったR':>9}{'延べ点数':>10}{'的中':>7}{'ROI':>9}"
              f"{'99%CI':>22}{'1R収支':>9}{'年+':>6}{'プラセボROI':>12}")
        diag = {}
        for t in THS:
            prof, cost, ret, nb, hit, yl = [], 0, 0, 0, 0, []
            rat_sel, qp_sel, win_sel, odd_sel, ovr_sel = [], [], [], [], []
            plr = np.zeros(NPLA)
            plc = np.zeros(NPLA)
            for (yy, qp, qu, qh, odds, win, ovr) in recs:
                q = qu if QI == 2 else qh
                sel = (q / qp) >= t
                if not sel.any():
                    continue
                rat_sel.extend((q / qp)[sel]); qp_sel.extend(qp[sel])
                win_sel.extend(win[sel]); odd_sel.extend(odds[sel])
                ovr_sel.extend([ovr] * int(sel.sum()))
                c = 100.0 * sel.sum()
                v = float((odds[sel & win]).sum() * 100.0)
                prof.append(v - c)
                yl.append(yy)
                cost += c
                ret += v
                nb += int(sel.sum())
                hit += int((sel & win).sum())
                for j in range(NPLA):
                    idx = RNG.permutation(len(qp))[: int(sel.sum())]
                    m = np.zeros(len(qp), bool)
                    m[idx] = True
                    plc[j] += c
                    plr[j] += float((odds[m & win]).sum() * 100.0)
            if not prof:
                print(f"{t:>7.3f}   該当レース無し")
                continue
            diag[t] = (np.array(rat_sel), np.array(qp_sel), np.array(win_sel),
                       np.array(odd_sel), np.array(ovr_sel))
            roi = ret / cost
            pr = np.array(prof)
            m, lo, hi = mci(pr)
            # ROIのCIはレース単位の収支から作る（組の間は従属なので組単位では作らない）
            mc = cost / len(pr)
            rlo, rhi = 1 + lo / mc, 1 + hi / mc
            ya = np.array(yl)
            pos = sum(1 for u in sorted(set(ya.tolist()))
                      if (ya == u).sum() >= 30 and pr[ya == u].mean() > 0)
            mark = "★★買える" if rlo > 1.0 else ""
            ci = "[" + format(100 * rlo, ".1f") + "," + format(100 * rhi, ".1f") + "]"
            print(f"{t:>7.3f}{len(pr):>9,}{nb:>10,}{hit:>7,}{100*roi:>8.1f}%{ci:>22}"
                  f"{m:>+9.1f}{pos:>4}/{len(set(ya.tolist()))}"
                  f"{100*np.mean(plr/np.maximum(plc,1)):>11.1f}% {mark}")
        if QI == 2:
            diag_main = diag
        print()

    # ── ★検算（事後の診断。判定を増やすものではない）──
    print("=" * 106)
    print("★★検算 — 馬連→枠・閾値1.290 の中身を開ける（判定を増やすものではない）")
    rat, qpv, wv, odv, ovv = diag_main[1.0 / R]
    print(f"  ① 較正: 選んだ組の **q/q_pool の平均 = {rat.mean():.3f}**"
          f"  vs  **実際の O/E = {wv.sum()/qpv.sum():.3f}**")
    print(f"     （一致していれば『馬連プールの言い分が当たっている』。"
          f"ずれていれば道具か偶然を疑う）")
    zb = zq(0.01 / len(THS))
    se = (117.3 - 103.9) / zq(0.01)
    print(f"  ② Bonferroni（{len(THS)}水準）: 99%CI [103.9,130.8] → "
          f"{100 - 0:.0f}%は使わず、α=0.01/{len(THS)} で "
          f"[{117.3 - zb*se:.1f},{117.3 + zb*se:.1f}]")
    print(f"  ③ 板オッズ帯ごとの内訳（配当の裾に依存していないか）")
    print(f"{'オッズ帯':>12}{'点数':>9}{'的中':>7}{'ROI':>9}{'寄与(円/R)':>12}")
    nR = 15614
    for lo_, hi_ in ((0, 10), (10, 30), (30, 100), (100, 1e9)):
        m = (odv >= lo_) & (odv < hi_)
        if m.sum() == 0:
            continue
        ret_ = (odv[m & (wv > 0)]).sum() * 100.0
        cost_ = 100.0 * m.sum()
        print(f"{lo_:>7}-{hi_ if hi_ < 1e9 else 999:<4}{m.sum():>9,}{int((m&(wv>0)).sum()):>7,}"
              f"{100*ret_/cost_:>8.1f}%{(ret_-cost_)/nR:>+12.1f}")
    print(f"  ④ 板の過剰率（Σ1/odds）: 選んだ組を含むレースの中央値 {np.median(ovv):.3f}"
          f"（1.29前後が正常。**極端なら取消などの異常を疑う**）")
    print()
    print("★読み方（事前登録のとおり）")
    print("  ・ROIの99%CI下端 > 100% かつ 年8/11以上 → ★**買える**。")
    print("  ・満たさなければ**この道も閉じる**。ただし今度は**(89)④の論法ではなく実測で閉じた**ことになる。")
    print("  ・プラセボ（同じ点数を無作為に買う）は**控除率まわり（77.5%前後）**に落ちるはず。")
    print("  ・⚠確定オッズのオラクルなので**上振れ側**。(94)は単勝で98.2%が天井と測っている。")


if __name__ == "__main__":
    main()
