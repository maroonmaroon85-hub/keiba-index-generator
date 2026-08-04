"""D3: §1で出た「点数が1点になるレースだけ +10.6pt」を潰す。

`audit_units.py` §1 の副産物として、**枠連の買い目が1点(100円)になったレース**(1,884R)で
モデルが人気順を **+10.64pt [+4.79,+16.61]** 上回るという大きな差が出た。
2点(200円)のレースでは +2.97pt しかない。しかもこの条件は**発走前に分かる**
（軸馬の枠と紐2頭の枠を見れば点数は確定する）ので、**もし本物なら運用に落とせる**。

★だが疑う理由がある。`audit_units.py` の予備実行を**中身の無い偽モデル
（人気順にノイズを足しただけ）**で回したとき、同じ欄が **+10.74pt** だった。
つまりこれは「モデルが強いレース」ではなく、**点数が1点に潰れる状況そのものが持つ
機械的な効果**の可能性が高い。(79)①のプラセボ効果と同じ形。

そこで(79)①と同じプラセボ対照を、**点数別の部分集合に対して**当てる:
  ・モデルの買い目が1点だったレースを取り出す
  ・**同じレース**で、人気順 と プラセボ（人気順＋ノイズ・モデルと同じ一致率）を測る
  ・モデル − プラセボ が残るかを見る

判定:
  ・モデル−プラセボが1点レースでも大きい ⇒ 本物。運用に落とせる可能性がある
  ・プラセボでも同じだけ出る             ⇒ 機械的効果。使えない

守った作法: 3(人気順の対照) 5(事前宣言＋対照＋対応あり) 6(プラセボ対照) 2(標本誤差)
　　　　　 ★判定基準5(絞ると測れなくなる)も併記する（1,884Rの検出限界を出す）

前提: `python3 ml/audit_units.py` で /tmp/units_races.pkl を作っておくこと（学習は不要）。
実行: python3 ml/audit_points.py
"""
import pickle
import sys

import numpy as np

sys.path.insert(0, "ml")
from waku_umatan import load_wu, waku_of

PAYOUT = "data/payout/a.csv"
SIGMAS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1.0]
N_DRAW = 12


def wakuren_cs(nums, n):
    wa = waku_of(nums[0], n)
    return sorted({tuple(sorted((wa, waku_of(h, n)))) for h in nums[1:3]})


def boot(x, rng, n=4000):
    b = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return np.percentile(b, 2.5) * 100, np.percentile(b, 97.5) * 100


def main():
    with open("/tmp/units_races.pkl", "rb") as fh:
        races = [r for r in pickle.load(fh) if "wk_m_pay" in r]
    wu = load_wu(PAYOUT)
    R = [r for r in races if wu.get(r["rid"]) and wu[r["rid"]]["wakuren"]]
    print(f"枠連が成立したレース {len(R):,}R（{R[0]['year']}〜{R[-1]['year']}）")

    mroi = np.array([r["wk_m_pay"] / r["wk_m_cost"] for r in R])
    proi = np.array([r["wk_p_pay"] / r["wk_p_cost"] for r in R])
    mpts = np.array([int(r["wk_m_cost"] // 100) for r in R])
    ppts = np.array([int(r["wk_p_cost"] // 100) for r in R])
    fs = np.array([r["n"] for r in R])
    agree_m = np.array([wakuren_cs(r["mo"], r["n"]) ==
                        wakuren_cs(r["uma"][np.argsort(r["lo"], kind="mergesort")], r["n"])
                        for r in R])

    def placebo_run(sg, seed):
        g = np.random.default_rng(seed)
        roi, pts, agr = [], [], []
        for r in R:
            o = r["uma"][np.argsort(r["lo"] + g.normal(0, sg, len(r["uma"])), kind="mergesort")]
            cs = wakuren_cs(o, r["n"])
            w = wu[r["rid"]]["wakuren"]
            roi.append(sum(w.get(c, 0) for c in cs) / (len(cs) * 100))
            pts.append(len(cs))
            agr.append(cs == wakuren_cs(r["uma"][np.argsort(r["lo"], kind="mergesort")], r["n"]))
        return np.array(roi), np.array(pts), float(np.mean(agr))

    # モデルと同じ「市場順からの離れ方」になる σ を補間で求める（(79)①と同じ作法）
    print("プラセボ曲線を作成中…")
    curve = [(sg, placebo_run(sg, 300)[2]) for sg in SIGMAS]
    xs = np.array([c[1] for c in curve])
    sg = float(np.interp(agree_m.mean(), xs[::-1], np.array(SIGMAS)[::-1]))
    print(f"  モデルの買い目一致率 {agree_m.mean()*100:.1f}% → プラセボ σ={sg:.3f}")
    draws = [placebo_run(sg, 400 + k) for k in range(N_DRAW)]
    lroi = np.mean([d[0] for d in draws], axis=0)
    lpts = np.array([d[1] for d in draws])

    rng = np.random.default_rng(0)
    print(f"\n{'='*104}")
    print("★点数別: モデル / 人気順 / プラセボ（同じレース・対応あり）")
    print(f"{'区分':<20}{'R':>8}{'頭数':>7}{'モデル':>9}{'人気順':>9}{'プラセボ':>10}"
          f"{'M−人気順':>11}{'M−プラセボ':>12}{'そのCI':>18}{'検出限界':>10}")
    for lab, sel in (("全体", mpts > 0), ("★1点(100円)", mpts == 1), ("2点(200円)", mpts == 2)):
        m, p, l = mroi[sel], proi[sel], lroi[sel]
        lo, hi = boot(m - l, rng)
        mde = 2.802 * (m - l).std(ddof=1) / np.sqrt(sel.sum()) * 100
        print(f"{lab:<20}{int(sel.sum()):>8,}{fs[sel].mean():>7.1f}{m.mean()*100:>8.2f}%"
              f"{p.mean()*100:>8.2f}%{l.mean()*100:>9.2f}%{(m-p).mean()*100:>+10.2f}pt"
              f"{(m-l).mean()*100:>+11.2f}pt{f'[{lo:+.2f},{hi:+.2f}]':>18}{mde:>9.2f}pt")

    print(f"\n★プラセボ自身の点数別ROI（モデルを使わない順序でも同じ形が出るか）")
    print(f"{'区分':<20}{'プラセボR(平均)':>16}{'プラセボROI':>13}{'人気順ROI(同じ集合)':>21}{'差':>10}")
    for lab, k in (("1点(100円)", 1), ("2点(200円)", 2)):
        vals, cnt, pv = [], [], []
        for (droi, dpts, _) in draws:
            s = dpts == k
            vals.append(droi[s].mean())
            cnt.append(s.sum())
            pv.append(proi[s].mean())
        print(f"{lab:<20}{np.mean(cnt):>16,.0f}{np.mean(vals)*100:>12.2f}%"
              f"{np.mean(pv)*100:>20.2f}%{(np.mean(vals)-np.mean(pv))*100:>+9.2f}pt")
    print("  ※プラセボでも1点レースで同じ差が出るなら、それは『点数が潰れる状況』の性質であって")
    print("    モデルの手柄ではない（(79)①の機械的効果と同じ）。")

    print(f"\n★人気順の点数で切った場合（モデル基準でなく市場基準で分ける）")
    print(f"{'区分':<20}{'R':>8}{'モデル':>9}{'人気順':>9}{'プラセボ':>10}{'M−プラセボ':>12}{'そのCI':>18}")
    for lab, sel in (("人気順が1点", ppts == 1), ("人気順が2点", ppts == 2)):
        m, p, l = mroi[sel], proi[sel], lroi[sel]
        lo, hi = boot(m - l, rng)
        print(f"{lab:<20}{int(sel.sum()):>8,}{m.mean()*100:>8.2f}%{p.mean()*100:>8.2f}%"
              f"{l.mean()*100:>9.2f}%{(m-l).mean()*100:>+11.2f}pt{f'[{lo:+.2f},{hi:+.2f}]':>18}")

    print(f"\n★的中率で見る（★判定基準7: ROI差より的中率の方が読める）")
    print(f"{'区分':<20}{'R':>8}{'モデル的中率':>13}{'人気順的中率':>13}{'差':>10}{'95%CI':>18}"
          f"{'モデル的中時配当':>17}{'人気順':>10}")
    for lab, sel in (("全体", mpts > 0), ("★1点(100円)", mpts == 1), ("2点(200円)", mpts == 2)):
        mh = (mroi[sel] > 0).astype(float)
        ph = (proi[sel] > 0).astype(float)
        lo, hi = boot(mh - ph, rng)
        mc = np.array([r["wk_m_cost"] for r in R])[sel]
        pc = np.array([r["wk_p_cost"] for r in R])[sel]
        mpay = mroi[sel] * mc
        ppay = proi[sel] * pc
        print(f"{lab:<20}{int(sel.sum()):>8,}{mh.mean()*100:>12.2f}%{ph.mean()*100:>12.2f}%"
              f"{(mh-ph).mean()*100:>+9.2f}pt{f'[{lo:+.2f},{hi:+.2f}]':>18}"
              f"{mpay[mh>0].mean():>16,.0f}円{ppay[ph>0].mean():>9,.0f}円")

    print(f"\n★1点レースの優位は期間を通して安定しているか（★判定基準5の確認）")
    yrs = np.array([r["year"] for r in R])
    print(f"{'期間':<14}{'R':>7}{'モデル':>9}{'人気順':>9}{'プラセボ':>10}"
          f"{'M−人気順':>11}{'M−プラセボ':>12}{'そのCI':>18}")
    half = (yrs.min() + yrs.max()) // 2
    for lab, sel in (("前半", (mpts == 1) & (yrs <= half)),
                     ("後半", (mpts == 1) & (yrs > half)),
                     ("直近5年", (mpts == 1) & (yrs >= yrs.max() - 4))):
        m, p, l = mroi[sel], proi[sel], lroi[sel]
        lo, hi = boot(m - l, rng)
        print(f"{lab:<14}{int(sel.sum()):>7,}{m.mean()*100:>8.2f}%{p.mean()*100:>8.2f}%"
              f"{l.mean()*100:>9.2f}%{(m-p).mean()*100:>+10.2f}pt{(m-l).mean()*100:>+11.2f}pt"
              f"{f'[{lo:+.2f},{hi:+.2f}]':>18}")
    ys = [(yy, (mpts == 1) & (yrs == yy)) for yy in sorted(set(yrs))]
    pos = sum(1 for _, s in ys if s.sum() > 30 and (mroi[s] - proi[s]).mean() > 0)
    tot = sum(1 for _, s in ys if s.sum() > 30)
    print(f"  人気順を上回った年: {pos}/{tot}   "
          f"年別の差: " + " ".join(f"{yy}:{(mroi[s]-proi[s]).mean()*100:+.0f}"
                                  for yy, s in ys if s.sum() > 30))

    # ------------------------------------------------------------------
    # ★コストの交絡を外す: 「1点だけ買う」買い方を**両方の順序に**適用して比べる。
    #   §1で見たとおり、モデルが1点のレースでは人気順は多くが2点＝**分母が違う**。
    #   点数を揃えないと「当てる力」と「安く買えた」が混ざる。
    #   ついでに ★枠連 軸枠×紐枠1 は (77)の29通りに入っていない未検証の買い方でもある。
    # ------------------------------------------------------------------
    def one_point(order, n, w):
        c = tuple(sorted((waku_of(order[0], n), waku_of(order[1], n))))
        return w.get(c, 0) / 100.0

    m1 = np.array([one_point(r["mo"], r["n"], wu[r["rid"]]["wakuren"]) for r in R])
    p1 = np.array([one_point(r["uma"][np.argsort(r["lo"], kind="mergesort")], r["n"],
                             wu[r["rid"]]["wakuren"]) for r in R])
    print(f"\n{'='*104}\n★点数を揃える: 両方とも『軸枠×紐枠1』の1点だけ買う（どちらも100円）")
    print("  ((77)の29通りに無い未検証の買い方でもある)")
    print(f"{'区分':<20}{'R':>8}{'モデル1点':>11}{'人気順1点':>11}{'差':>10}{'95%CI':>18}"
          f"{'（参考）現行2点まで':>20}")
    for lab, sel in (("全体", mpts > 0), ("★モデルが1点の集合", mpts == 1), ("モデルが2点の集合", mpts == 2)):
        lo, hi = boot(m1[sel] - p1[sel], rng)
        print(f"{lab:<20}{int(sel.sum()):>8,}{m1[sel].mean()*100:>10.2f}%{p1[sel].mean()*100:>10.2f}%"
              f"{(m1[sel]-p1[sel]).mean()*100:>+9.2f}pt{f'[{lo:+.2f},{hi:+.2f}]':>18}"
              f"{mroi[sel].mean()*100:>19.2f}%")
    lo, hi = boot(m1 - mroi, rng)
    print(f"  ★対応あり: 1点だけ買う − 現行(2点まで)  {(m1-mroi).mean()*100:+.2f}pt "
          f"[{lo:+.2f},{hi:+.2f}]  （全{len(R):,}R・コストは194円→100円）")

    print(f"\n★1点になる条件は頭数で決まるのか（枠割当の副作用かの確認）")
    print(f"{'頭数':<10}{'R':>8}{'モデル1点率':>12}{'人気順1点率':>12}{'プラセボ1点率':>14}")
    for lo_n, hi_n in ((9, 12), (13, 14), (15, 16), (17, 18)):
        s = (fs >= lo_n) & (fs <= hi_n)
        if s.sum() < 200:
            continue
        pr = np.mean([(dp[s] == 1).mean() for _, dp, _ in draws])
        print(f"{f'{lo_n}〜{hi_n}頭':<10}{int(s.sum()):>8,}{(mpts[s]==1).mean()*100:>11.1f}%"
              f"{(ppts[s]==1).mean()*100:>11.1f}%{pr*100:>13.1f}%")


if __name__ == "__main__":
    main()
