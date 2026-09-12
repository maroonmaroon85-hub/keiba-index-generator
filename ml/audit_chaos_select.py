"""(134) ★★「荒れ度」そのものを選択変数として測る（ユーザー発案・2026-08-12）

★なぜ測るか（既に閉じている方向なのに）
　ユーザーの問い「**荒れる可能性が高いレースから期待値の高い馬を推奨するのはどうか**」。
　この提案は2層に分かれ、**後半は既に閉じている**:
　　・**「期待値の高い馬」＝ 我々のpが市場のqより高い馬 ＝ 残差そのもの**。
　　　(120) 指数Dで選ぶと**上位5%で −0.0172**（99%CIが負側に0を除外）＝**詰めるほど悪い**。
　　　(75) 残差だけの順位付けは **AUC 0.3605＝逆相関**（残差順＝人気薄順になるから）。
　　　(88)④ その人気薄は市場の値付けが最悪の帯（100-200倍 54.9% / 200倍超 37.1%）。
　★**しかし前半「荒れ度」そのものは、事前登録して測った記録が無い**。
　　(112)の軸E・(117)の枠連スコアと強く相関するはずだが、**独立には測っていない**。
　　→ **穴を残さないために、ここだけ測る**。

⚠**測る前に上界を書いておく（期待値を釣り上げないため）**
　(114)で**レース選択という手段の天井**が出ている: 完全オラクルは +0.4947 だが
　**事前情報で取れるのはその1/20＝正味 +0.0218（必要量の8.5%）**。
　★**荒れ度が仮に効いても、この天井は超えない**。**道が開く実験ではない**。
　**やる意味は「(112)と同じ信号か、別の信号か」を確定させること**（(119)と同じ問い）。

★★事前登録（測る前に宣言する）
　1. **変数は1本**: 市場確率のエントロピー `H = −Σ p log p`（**発走前に計算できる**）。
　　 頭数でHの上限が変わるので、**(頭数, 年)の層内で順位→パーセンタイル**に直す。
　　 **後から増やさない**（HHIや1番人気オッズに乗り換えない）。
　2. **測る対象は枠連のd**（実際に買っている券種）。λは**ウォークフォワード**。
　3. **判定は単調性**（十分位のSpearman ρ）。**最良のビンでは判定しない**。
　4. **プラセボ**: (頭数,年)層内でシャッフル・**30回平均**（判定基準13）。
　　 **効果量は「実測−プラセボ」**（判定基準2）。
　5. **裾は先に宣言**: 高い側（荒れる側）・低い側とも 2/5/10/25/50%。
　6. ★**(112)との関係を必ず見る**（(119)の教訓＝「別の変数に見えて同じ信号だった」）:
　　 **Hと軸のEの順位相関**を出し、**(112)の裾2%の中でHが追加で効くか**も見る。
　7. **★運用が変わる条件**: 裾2%で **d ≥ +0.0394 かつ 99%CI下端 > +0.0182 かつ単調**。
　　 **全部満たしたときだけ**検討。それ未満なら**記述にとどめ運用は変えない**。
　8. **予想**: ★**荒れる側（H大）が負**と予想する。ρ は負（0〜−0.8）。
　　 理由: (112)は「断然人気がいるレース」で効き、(113)(A)は「三連複が安い＝断然人気3頭」を
　　 　　　**残すほうが良い**と出た。**Hが大きい＝断然人気がいない**＝その逆側。
　　 　　　さらに我々のqは84.7%が市場由来なので、**市場が薄い所では我々も薄い**。
　　 ⚠**予想はあてにしない**（この日 (131)で外している）。

実行: python3 ml/audit_chaos_select.py [開始年(既定2015)] [プラセボ反復(既定30)]
"""
import math
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, "ml")
from audit_capacity_d import mkt_waku_dist
from audit_crosspool import PAYBACK, load_races, payoff, probs, zq
from audit_crosspool2 import PAYKEY, realized
from audit_lbs import build_matrix, fit_lambda
from waku_umatan import waku_of
import soft_axis as SA

RNG = np.random.default_rng(20260812)
NEED = -math.log(PAYBACK["枠連"])
TAILS = (0.02, 0.05, 0.10, 0.25, 0.50)
D_112_TAIL = 0.0394
D_ALL = 0.0182
E_112_THR = 86.0


def mci(x, alpha=0.01):
    x = np.asarray(x, float)
    if len(x) < 2:
        return float("nan"), float("nan"), float("nan")
    m = x.mean()
    se = x.std(ddof=1) / math.sqrt(len(x))
    z = zq(alpha)
    return m, m - z * se, m + z * se


def spearman(xs, ys):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for pos, i in enumerate(order):
            r[i] = pos
        return r
    a, b = rank(xs), rank(ys)
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    den = math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))
    return num / den if den else float("nan")


def entropy(p):
    """★荒れ度: 市場確率のエントロピー。大きいほど「本命不在＝荒れそう」。"""
    q = np.asarray(p, float)
    q = q[q > 0]
    return float(-(q * np.log(q)).sum())


def pctile_in_strata(v, strata):
    out = np.zeros(len(v))
    idx = defaultdict(list)
    for i, s in enumerate(strata):
        idx[s].append(i)
    for _s, ii in idx.items():
        ii = np.array(ii)
        out[ii] = np.argsort(np.argsort(v[ii])) / max(len(ii) - 1, 1)
    return out


def main():
    y0 = int(sys.argv[1]) if len(sys.argv) > 1 else 2015
    reps = int(sys.argv[2]) if len(sys.argv) > 2 else 30

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

    rows = []
    for r in races:
        yy = r["year"]
        if yy < y0 or not lam.get(yy) or not r["wakuren"]:
            continue
        rl = realized(r)
        if rl is None:
            continue
        a, b, _ = rl
        num2k = {u: k for k, (u, _, _) in enumerate(r["horses"])}
        if a not in num2k or b not in num2k:
            continue
        p = probs(r["horses"])
        l2, _l3 = lam[yy]
        md = mkt_waku_dist(r, p, l2)
        if not md:
            continue
        key = tuple(sorted((waku_of(a, r["n"]), waku_of(b, r["n"]))))
        if key not in md:
            continue
        v = payoff(r, PAYKEY["枠連"], key)
        if not v or v <= 0:
            continue
        d = math.log(md[key]) + math.log((v + 5) / 100.0) - math.log(PAYBACK["枠連"])
        odds = [o for _, o, _ in r["horses"]]
        _k, e_axis, _q = SA.axis_expect(odds)
        rows.append((d, entropy(p), r["n"], yy, e_axis if e_axis else float("nan")))

    arr = np.array(rows, float)
    d, H, nn, yy, ea = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 4]
    strata = list(zip(nn.astype(int).tolist(), yy.astype(int).tolist()))
    pct = pctile_in_strata(H, strata)

    m_all, lo_all, hi_all = mci(d)
    print(f"対象 {len(rows)} レース（{int(yy.min())}〜{int(yy.max())}）")
    print(f"★検算: 枠連の全体D = {m_all:+.4f} 99%CI[{lo_all:+.4f},{hi_all:+.4f}]  ※(96)は +0.0182")
    print(f"　必要量 {NEED:.4f} ／ (112)裾2%の現行最良 {D_112_TAIL:+.4f}")
    print(f"⚠上界: (114)でレース選択の天井は正味 +0.0218（必要量の8.5%）。**ここは超えない**")

    print("\n── ★事前登録6: (112)の軸Eとの順位相関（同じ信号か） ──")
    ok = ~np.isnan(ea)
    rho_e = spearman(H[ok].tolist(), ea[ok].tolist())
    print(f"  エントロピーH と 軸のE の Spearman ρ = {rho_e:+.3f}")
    print("  ※(119)では(112)と(117)が『同じ信号』と判明した。ここも疑ってかかる")

    print("\n── 十分位（★判定は単調性） ──")
    dec = np.minimum((pct * 10).astype(int), 9)
    xs, ys = [], []
    for g in range(10):
        m = dec == g
        if m.sum() < 20:
            continue
        v2, lo2, hi2 = mci(d[m])
        xs.append(g)
        ys.append(v2)
        print(f"  第{g+1:2d}十分位（{'堅い' if g < 5 else '荒れる'}側）"
              f" n={int(m.sum()):5d}  d={v2:+.4f}  99%CI[{lo2:+.4f},{hi2:+.4f}]")
    rho = spearman(xs, ys)
    print(f"  → Spearman ρ = {rho:+.3f}")

    print(f"\n── 裾（効果量は「実測−プラセボ」・{reps}回平均） ──")
    pl_hi = np.zeros(len(TAILS))
    pl_lo = np.zeros(len(TAILS))
    idx = defaultdict(list)
    for i, s in enumerate(strata):
        idx[s].append(i)
    for _ in range(reps):
        sh = np.zeros(len(pct))
        for _s, ii in idx.items():
            ii = np.array(ii)
            sh[ii] = pct[ii][RNG.permutation(len(ii))]
        for j, t in enumerate(TAILS):
            pl_hi[j] += d[sh >= 1 - t].mean()
            pl_lo[j] += d[sh <= t].mean()
    pl_hi /= reps
    pl_lo /= reps

    for lab, sel, pl in (("荒れる側（H大）", lambda t: pct >= 1 - t, pl_hi),
                         ("堅い側（H小）", lambda t: pct <= t, pl_lo)):
        print(f"  【{lab}】")
        print("   裾      n      実測d      99%CI            プラセボ    実測−プラセボ")
        for j, t in enumerate(TAILS):
            m = sel(t)
            v2, l2, h2 = mci(d[m])
            print(f"   {int(t*100):3d}%  {int(m.sum()):5d}  {v2:+.4f}  "
                  f"[{l2:+.4f},{h2:+.4f}]  {pl[j]:+.4f}   {v2-pl[j]:+.4f}")

    print("\n── ★事前登録6: (112)の裾2%の中でHが追加で効くか ──")
    in112 = ea <= E_112_THR
    a112, l112, h112 = mci(d[in112])
    print(f"  (112)の裾 n={int(in112.sum())}  d={a112:+.4f} 99%CI[{l112:+.4f},{h112:+.4f}]")
    sub, dd = pct[in112], d[in112]
    for t in (0.25, 0.50):
        mh, ml = sub >= 1 - t, sub < 1 - t
        if mh.sum() >= 20 and ml.sum() >= 20:
            ah, _, _ = mci(dd[mh])
            al, _, _ = mci(dd[ml])
            se = math.sqrt(dd[mh].std(ddof=1)**2 / mh.sum() + dd[ml].std(ddof=1)**2 / ml.sum())
            z = zq(0.01)
            print(f"   H上位{int(t*100)}% {ah:+.4f}(n={int(mh.sum())}) vs "
                  f"残り {al:+.4f}(n={int(ml.sum())})  差 {ah-al:+.4f} "
                  f"99%CI[{ah-al-z*se:+.4f},{ah-al+z*se:+.4f}]")

    print("\n── ★事前登録7: 運用が変わる条件に当てる ──")
    v2, l2, _ = mci(d[pct >= 0.98])
    print(f"  荒れる側 裾2%: d={v2:+.4f} 99%CI下端={l2:+.4f}"
          f"（条件は d≥{D_112_TAIL:+.4f} かつ 下端>{D_ALL:+.4f}）")
    print(f"  単調性 ρ={rho:+.3f}")
    ok2 = (v2 >= D_112_TAIL) and (l2 > D_ALL)
    print("  → " + ("★条件を満たす" if ok2 else
                    "**条件を満たさない。記述にとどめ、運用は変えない**"))


if __name__ == "__main__":
    main()
