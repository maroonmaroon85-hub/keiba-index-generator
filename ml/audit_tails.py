"""D2: **信頼区間の当て方そのもの**を疑う ＋ 券種の併用(E)を閉じる。

このプロジェクトは誤差の当て方を3回間違えている（(39)(46)(78)）。だが**直した後も、
CIは一貫して `boot()` ＝ パーセンタイル・ブートストラップ**のままで、
**その手法がこのデータで妥当か**は一度も確かめていない。(78)④が確かめたのは
「レースが独立か（開催日ブロック）」だけで、**分布の形**は見ていない。

★問題の所在: 連系の1レース回収額は**極端に右に歪んだ分布**（大半が0、たまに数十倍）。
  パーセンタイル・ブートストラップは**歪んだ分布の平均**に対して有限標本で
  被覆率が名目95%を下回ることが知られている。もしここで効いているなら、
  (79)①の技能 +1.93pt CI[+0.33,+3.48]（辛うじて0を外す）のような**境界の判定が反転しうる**。

そこで同じ差に3種類の区間を当てて比較する:
  ・percentile … 現行 `boot()`
  ・BCa       … 歪み(a)と偏り(z0)を補正した標準的な区間
  ・bootstrap-t … スチューデント化。歪んだ平均に対して最も被覆が良いとされる
3つが一致すれば現行の判定はそのまま信用してよい。ズレるなら、境界の主張は取り下げる。

併せて★判定基準7（ROI差は読めないことが多い）を定量化する:
  **この標本量で検出できる最小の差**（power 80%）を券種ごとに出し、
  「読めない」を感覚ではなく数字にする。

§6 は E（券種の併用）。全ての測定は「1レース1券種を等額」で、**同じレースで2券種を
同時に買う**形は測っていない。ROI<100%なので利益は出ないが、**併用が現行より悪い**ことを
明示的に閉じておく（併用ROIは各券種ROIのコスト加重平均になるはず、という事前宣言つき）。

前提: `python3 ml/audit_units.py` を先に走らせて /tmp/units_races.pkl を作っておくこと。
実行: python3 ml/audit_tails.py
"""
import pickle
import sys

import numpy as np

sys.path.insert(0, "ml")

SPECS = [("枠連 軸枠×紐枠2", "wk_m_pay", "wk_m_cost", "wk_p_pay", "wk_p_cost"),
         ("三連複 BOX上位4", "s3_m_pay", None, "s3_p_pay", None),
         ("複勝 top1", "fk_m_pay", None, "fk_p_pay", None)]
FIXED = {"s3_m_pay": 400.0, "s3_p_pay": 400.0, "fk_m_pay": 100.0, "fk_p_pay": 100.0}


def arrays(races, key, cost_key):
    sel = [r for r in races if key in r]
    pay = np.array([r[key] for r in sel], float)
    cost = (np.array([r[cost_key] for r in sel], float) if cost_key
            else np.full(len(sel), FIXED[key]))
    return sel, pay / cost


def boot_means(x, rng, n=4000):
    idx = rng.integers(0, len(x), size=(n, len(x)))
    return x[idx].mean(axis=1), x[idx].std(axis=1, ddof=1)


def intervals(x, rng, n=4000):
    """percentile / BCa / bootstrap-t の3種。x は1レース当たりの差の配列。"""
    th = x.mean()
    bm, bs = boot_means(x, rng, n)
    pct = (np.percentile(bm, 2.5), np.percentile(bm, 97.5))
    # --- BCa ---
    z0 = _ppf(max(min((bm < th).mean(), 1 - 1e-6), 1e-6))
    jk = (x.sum() - x) / (len(x) - 1)          # jackknife
    dv = jk.mean() - jk
    a = (dv ** 3).sum() / (6.0 * ((dv ** 2).sum() ** 1.5) + 1e-300)
    def adj(q):
        z = _ppf(q)
        return _cdf(z0 + (z0 + z) / max(1 - a * (z0 + z), 1e-6))
    bca = (np.percentile(bm, 100 * adj(0.025)), np.percentile(bm, 100 * adj(0.975)))
    # --- bootstrap-t ---
    se = x.std(ddof=1) / np.sqrt(len(x))
    t = (bm - th) / np.maximum(bs / np.sqrt(len(x)), 1e-300)
    bt = (th - np.percentile(t, 97.5) * se, th - np.percentile(t, 2.5) * se)
    return pct, bca, bt, se


def _cdf(z):
    import math
    return 0.5 * (1 + math.erf(z / np.sqrt(2)))


def _ppf(p):
    # Acklam の有理近似で十分（表示用）
    import math
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    dd = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
          3.754408661907416e+00]
    pl, ph = 0.02425, 1 - 0.02425
    if p < pl:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((dd[0]*q+dd[1])*q+dd[2])*q+dd[3])*q+1)
    if p > ph:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((dd[0]*q+dd[1])*q+dd[2])*q+dd[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def main():
    with open("/tmp/units_races.pkl", "rb") as fh:
        races = pickle.load(fh)
    rng = np.random.default_rng(0)
    print(f"読み込み: {len(races):,}R  {races[0]['year']}〜{races[-1]['year']}")

    print(f"\n{'='*104}\n§5 ★区間推定の当て方: percentile(現行) / BCa / bootstrap-t を突き合わせる")
    print(f"  {'券種':<16}{'R':>8}{'差':>9}{'歪度':>8}{'percentile(現行)':>21}"
          f"{'BCa':>21}{'bootstrap-t':>21}")
    for name, mk, mc, pk, pc in SPECS:
        _, m = arrays(races, mk, mc)
        _, p = arrays(races, pk, pc)
        x = m - p
        sk = ((x - x.mean()) ** 3).mean() / (x.std() ** 3 + 1e-300)
        pct, bca, bt, se = intervals(x, rng)
        f = lambda t: f"[{t[0]*100:+.2f},{t[1]*100:+.2f}]"
        print(f"  {name:<16}{len(x):>8,}{x.mean()*100:>+8.2f}pt{sk:>8.2f}"
              f"{f(pct):>21}{f(bca):>21}{f(bt):>21}")
    print("  ※3つが一致 ⇒ 現行の判定はそのまま信用してよい。ズレる ⇒ 境界の主張は取り下げる。")

    print(f"\n  この標本量で検出できる最小の差（両側5%・検出力80%）＝★判定基準7の数値化")
    print(f"  {'券種':<16}{'R':>8}{'1レースSD':>11}{'標準誤差':>10}{'検出限界':>10}"
          f"{'現行の差':>10}{'必要R数(差1pt)':>16}")
    for name, mk, mc, pk, pc in SPECS:
        _, m = arrays(races, mk, mc)
        _, p = arrays(races, pk, pc)
        x = m - p
        se = x.std(ddof=1) / np.sqrt(len(x))
        mde = 2.802 * se
        need = (2.802 * x.std(ddof=1) / 0.01) ** 2
        print(f"  {name:<16}{len(x):>8,}{x.std(ddof=1):>11.3f}{se*100:>9.3f}pt"
              f"{mde*100:>9.2f}pt{x.mean()*100:>+9.2f}pt{need:>16,.0f}")

    print(f"\n{'='*104}\n§6 E: 券種の併用（同じレースで2券種を同時に買う）")
    print("  事前宣言: 併用ROIは各券種ROIの**コスト加重平均**になり、単独の枠連を超えない")
    W = {r["rid"]: r for r in races if "wk_m_pay" in r}
    S = {r["rid"]: r for r in races if "s3_m_pay" in r}
    F_ = {r["rid"]: r for r in races if "fk_m_pay" in r}
    both = [rid for rid in W if rid in S]
    print(f"  両方が成立するレース {len(both):,}R")
    wp = np.array([W[r]["wk_m_pay"] for r in both]); wc = np.array([W[r]["wk_m_cost"] for r in both])
    sp = np.array([S[r]["s3_m_pay"] for r in both]); sc = np.full(len(both), 400.0)
    tri = [rid for rid in both if rid in F_]
    fp = np.array([F_[r]["fk_m_pay"] for r in tri]); fc = np.full(len(tri), 100.0)
    print(f"  {'買い方':<28}{'1レース平均コスト':>18}{'金額加重ROI':>14}{'レース等額ROI':>15}{'95%CI(等額)':>20}")
    combos = [("枠連のみ", [(wp, wc)]), ("三連複BOX4のみ", [(sp, sc)]),
              ("枠連 + 三連複BOX4", [(wp, wc), (sp, sc)])]
    for lab, parts in combos:
        P = sum(a for a, _ in parts); C = sum(b for _, b in parts)
        eq = P / C
        lo = np.percentile([eq[rng.integers(0, len(eq), len(eq))].mean() for _ in range(2000)], 2.5)
        hi = np.percentile([eq[rng.integers(0, len(eq), len(eq))].mean() for _ in range(2000)], 97.5)
        print(f"  {lab:<28}{C.mean():>17.1f}円{P.sum()/C.sum()*100:>13.2f}%"
              f"{eq.mean()*100:>14.2f}%{f'[{lo*100:.2f},{hi*100:.2f}]':>20}")
    if len(tri) > 1000:
        wp2 = np.array([W[r]["wk_m_pay"] for r in tri]); wc2 = np.array([W[r]["wk_m_cost"] for r in tri])
        sp2 = np.array([S[r]["s3_m_pay"] for r in tri])
        for lab, P, C in [("複勝top1のみ", fp, fc),
                          ("枠連 + 複勝top1", wp2 + fp, wc2 + fc),
                          ("3券種すべて", wp2 + sp2 + fp, wc2 + 400.0 + fc)]:
            eq = P / C
            lo = np.percentile([eq[rng.integers(0, len(eq), len(eq))].mean() for _ in range(2000)], 2.5)
            hi = np.percentile([eq[rng.integers(0, len(eq), len(eq))].mean() for _ in range(2000)], 97.5)
            print(f"  {lab:<28}{C.mean():>17.1f}円{P.sum()/C.sum()*100:>13.2f}%"
                  f"{eq.mean()*100:>14.2f}%{f'[{lo*100:.2f},{hi*100:.2f}]':>20}")
    print("  ※併用は「悪い方を混ぜる」操作なので、ROIは必ず単独の最良と最悪の間に入る。")
    print("    分散は下がるが、ROI<100%では分散を下げる価値は無い（期待損失が増えるだけ）。")

    # ---- 開催日単位で見る（レース間の相関） ----
    print(f"\n{'='*104}\n§7 開催日を単位にする（レース間の相関・運用の体感）")
    print(f"  {'券種':<16}{'開催日数':>9}{'日ROIの中央値':>15}{'プラスの日':>11}"
          f"{'日内相関ICC':>13}{'実効レース数':>13}")
    for name, mk, mc, pk, pc in SPECS:
        sel, m = arrays(races, mk, mc)
        days = np.array([r["day"] for r in sel])
        import collections
        g = collections.defaultdict(list)
        for dd, v in zip(days, m):
            g[dd].append(v)
        dm = np.array([np.mean(v) for v in g.values()])
        nk = np.array([len(v) for v in g.values()])
        # ICC（1元配置）: 群間分散 / 全分散
        gm = m.mean()
        ssb = float((nk * (dm - gm) ** 2).sum())
        ssw = float(sum(((np.array(v) - np.mean(v)) ** 2).sum() for v in g.values()))
        k = nk.mean()
        msb, msw = ssb / (len(nk) - 1), ssw / (len(m) - len(nk))
        icc = max((msb - msw) / (msb + (k - 1) * msw), 0.0)
        neff = len(m) / (1 + (k - 1) * icc)
        print(f"  {name:<16}{len(nk):>9,}{np.median(dm)*100:>14.1f}%"
              f"{(dm>1).mean()*100:>10.1f}%{icc:>13.4f}{neff:>13,.0f}")
    print("  ※ICCが0近傍なら、開催日でまとめてもCIは変わらない（(78)④のブロックBSと整合するはず）")


if __name__ == "__main__":
    main()
