"""D1: **評価の単位**と**裾への依存**を疑う — これまで一度も当てていない3つの対照。

(78)は「学習量」の穴を塞ぎ、(79)は「順序のズレ」の穴を塞いだ。だが**回収率という量そのものの
定義**は(31)から一度も見直されていない。ここを3方向から疑う。

★① 評価の単位: **レース等額 vs 金額加重**
  現行の全スクリプトは `payout / (点数*100)` を**レース単位で平均**している＝「1レースあたり等額」。
  ところが**枠連 軸枠×紐枠2 は点数がレースごとに1点か2点で変わる**（軸と紐が同じ枠に入ると重複除去
  で1点になる。`waku_umatan.wakuren_buy` の docstring に明記済み）。
  つまり現行のROIは「100円しか賭けないレース」と「200円賭けるレース」を**同じ重み**で平均している。
  実際に財布から出る金で測る＝**金額加重 Σ払戻/Σ投資** とは別物になりうる。
  ★しかもモデル順と人気順で**点数の分布が違えば、対人気順の差そのものが歪む**。
  本命(枠連)だけの問題（三連複BOX4は常に400円・複勝top1は常に100円で影響を受けない）。
  → 事前宣言する仮説: **モデルは人気順より1点レースの比率が高く、レース等額ROIは金額加重ROIより高く出る**。

★② 裾への依存: **配当を上限で切っても優位は残るか**
  (79)①で「人気順から適度に離れるだけで+1.4pt出る。配当分布の右の裾が長いから」と分かった。
  だとすると**技能ぶん(+1.93pt)も裾から来ているだけ**かもしれない。(41)でも「優位は裾だけ」が出ている。
  払戻に上限C円を掛けて（＝万馬券的中を切り捨てて）モデル/人気順/プラセボを測り直す。
  優位がCに強く依存するなら、それは「たまに大穴を拾う」性質で、**運用で体感できる年数では実現しない**。

★③ 優位の縮小(B)の分解: **市場が賢くなったのか、配当の形が変わったのか**
  (78)③は「学習量が原因ではない」までしか言えていない。だが(79)①のプラセボ効果は
  **配当分布の形だけで決まる量**（中身の無いノイズ順のROI）。年ごとに
  「プラセボ効果」と「モデル−プラセボ(＝技能)」に分ければ、
  ・プラセボ効果が縮んでいる ⇒ 配当分布が均されただけ（市場の中身とは別）
  ・技能が縮んでいる         ⇒ モデルの相対優位が本当に失われている
  を分けられる。年別に両方出す。

守った作法: 1(複数シード) 2(標本誤差) 3(人気順の対照) 5(事前宣言＋対照＋対応あり) 6(プラセボ対照)
           7(ROI差は読めないことが多い→的中率と点数も併記) 8(精度指標は使わない)

実行: python3 ml/audit_units.py [シード数(既定3)] [開始年(既定2016)]
"""
import itertools
import pickle
import sys
import warnings

import numpy as np
import pandas as pd
import lightgbm as lgb

warnings.filterwarnings("ignore")
sys.path.insert(0, "ml")
import features as F
from _cache import load_cached
from place_wide import PARAMS, load_place_wide
from pocket_eval import load_payout_a
from waku_umatan import load_wu, waku_of

PAYOUT = "data/payout/a.csv"
SIGMAS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1.0]
N_DRAW = 6                       # プラセボの引き直し回数
CAPS = [None, 20000, 10000, 5000, 3000, 2000, 1000]   # 払戻の上限（円）


def wakuren_cs(nums, n):
    wa = waku_of(nums[0], n)
    return sorted({tuple(sorted((wa, waku_of(h, n)))) for h in nums[1:3]})


def boot(x, rng, n=2000):
    b = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return np.percentile(b, 2.5) * 100, np.percentile(b, 97.5) * 100


def boot_ratio(pay_a, cost_a, pay_b, cost_b, rng, n=2000):
    """金額加重ROIの差（比の推定量）のCI。レース単位でリサンプル。"""
    out = []
    for _ in range(n):
        i = rng.integers(0, len(pay_a), len(pay_a))
        out.append(pay_a[i].sum() / cost_a[i].sum() - pay_b[i].sum() / cost_b[i].sum())
    return np.percentile(out, 2.5) * 100, np.percentile(out, 97.5) * 100


def build_races(n_seed, y0):
    """ウォークフォワードで各レースの買い目と払戻（円）を作る。"""
    d, fx = load_cached()
    y = (d["finish"] <= 3).astype(int).to_numpy()
    year = d["date"].dt.year.to_numpy()
    wu, pa, pw = load_wu(PAYOUT), load_payout_a(PAYOUT), load_place_wide(PAYOUT)
    years = [yy for yy in range(y0, int(year.max()) + 1) if (year == yy).sum() > 5000]
    print(f"ウォークフォワード（各年をそれ以前の全データで学習・シード{n_seed}本）"
          f" 評価年 {years[0]}〜{years[-1]}")
    races = []
    for yy in years:
        tr, te = year < yy, year == yy
        p = np.mean([lgb.LGBMClassifier(random_state=s, **PARAMS)
                     .fit(fx[tr], y[tr], categorical_feature=F.CAT_COLS)
                     .predict_proba(fx[te])[:, 1] for s in range(n_seed)], axis=0)
        sub = d.loc[te, ["raceid", "umaban", "fieldsize", "odds", "date", "course"]].copy()
        sub["p"] = p
        for rid, g in sub.groupby("raceid", sort=False):
            n = int(g["fieldsize"].iloc[0])
            uma = g["umaban"].astype(int).to_numpy()
            lo = np.log(g["odds"].to_numpy(float))
            po = uma[np.argsort(lo, kind="mergesort")]
            mo = uma[np.argsort(-g["p"].to_numpy(float), kind="mergesort")]
            w, s3, fw = wu.get(rid), pa.get(rid), pw.get(rid)
            r = {"rid": rid, "year": yy, "day": str(g["date"].iloc[0])[:10], "n": n,
                 "track": g["course"].iloc[0], "uma": uma, "lo": lo, "mo": mo}
            if w and w["wakuren"] and len(g) >= 3:
                r["wk"] = w["wakuren"]
                for tag, o in (("m", mo), ("p", po)):
                    cs = wakuren_cs(o, n)
                    r[f"wk_{tag}_pay"] = float(sum(w["wakuren"].get(c, 0) for c in cs))
                    r[f"wk_{tag}_cost"] = float(len(cs) * 100)
            if s3 and s3["sanrenpuku"] and len(g) >= 9:
                r["s3"] = s3["sanrenpuku"]
                for tag, o in (("m", mo), ("p", po)):
                    cs = [tuple(sorted(c)) for c in itertools.combinations(o[:4], 3)]
                    r[f"s3_{tag}_pay"] = float(sum(s3["sanrenpuku"].get(c, 0) for c in cs))
            if fw and fw["fuku"]:
                r["fk"] = fw["fuku"]
                r["fk_m_pay"] = float(fw["fuku"].get(int(mo[0]), 0))
                r["fk_p_pay"] = float(fw["fuku"].get(int(po[0]), 0))
            races.append(r)
        print(f"  {yy} 完了 ({len(races):,}R)")
    return races


def placebo_pays(races, sg, seed, key):
    """人気順の log(オッズ) にノイズを足しただけの順序での払戻（円）。key: wk / s3 / fk"""
    g = np.random.default_rng(seed)
    pay, cost, idx = [], [], []
    for i, r in enumerate(races):
        if key not in r:
            continue
        o = r["uma"][np.argsort(r["lo"] + g.normal(0, sg, len(r["uma"])), kind="mergesort")]
        if key == "wk":
            cs = wakuren_cs(o, r["n"])
            pay.append(float(sum(r["wk"].get(c, 0) for c in cs)))
            cost.append(float(len(cs) * 100))
        elif key == "s3":
            cs = [tuple(sorted(c)) for c in itertools.combinations(o[:4], 3)]
            pay.append(float(sum(r["s3"].get(c, 0) for c in cs)))
            cost.append(400.0)
        else:
            pay.append(float(r["fk"].get(int(o[0]), 0)))
            cost.append(100.0)
        idx.append(i)
    return np.array(pay), np.array(cost), np.array(idx)


def agree_rate(races, sg, seed, key):
    """買い目が人気順と完全一致する率。σ→一致率 の曲線用。"""
    g = np.random.default_rng(seed)
    hit = []
    for r in races:
        if key not in r:
            continue
        o = r["uma"][np.argsort(r["lo"] + g.normal(0, sg, len(r["uma"])), kind="mergesort")]
        po = r["uma"][np.argsort(r["lo"], kind="mergesort")]
        if key == "wk":
            hit.append(wakuren_cs(o, r["n"]) == wakuren_cs(po, r["n"]))
        elif key == "s3":
            hit.append(set(o[:4]) == set(po[:4]))
        else:
            hit.append(o[0] == po[0])
    return float(np.mean(hit))


def main():
    n_seed = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    y0 = int(sys.argv[2]) if len(sys.argv) > 2 else 2016
    races = build_races(n_seed, y0)
    rng = np.random.default_rng(0)

    # ============================================================ §1 評価の単位
    W = [r for r in races if "wk" in r]
    pm, cm = np.array([r["wk_m_pay"] for r in W]), np.array([r["wk_m_cost"] for r in W])
    pp, cp = np.array([r["wk_p_pay"] for r in W]), np.array([r["wk_p_cost"] for r in W])
    print(f"\n{'='*100}\n§1 ★評価の単位: レース等額 vs 金額加重（枠連 軸枠×紐枠2・{len(W):,}R）")
    print("  事前宣言した仮説: モデルは人気順より1点レースが多く、レース等額ROIは金額加重より高く出る")
    print(f"\n  {'順序':<10}{'1点レース':>10}{'2点レース':>10}{'平均コスト':>11}"
          f"{'レース等額ROI':>15}{'金額加重ROI':>13}{'差(等額−加重)':>15}")
    for tag, p_, c_ in (("モデル", pm, cm), ("人気順", pp, cp)):
        eq = (p_ / c_).mean() * 100
        mw = p_.sum() / c_.sum() * 100
        print(f"  {tag:<10}{(c_==100).mean()*100:>9.1f}%{(c_==200).mean()*100:>9.1f}%"
              f"{c_.mean():>10.1f}円{eq:>14.2f}%{mw:>12.2f}%{eq-mw:>+14.2f}pt")
    d_eq = (pm / cm) - (pp / cp)
    lo1, hi1 = boot(d_eq, rng)
    lo2, hi2 = boot_ratio(pm, cm, pp, cp, rng)
    print(f"\n  {'対人気順の差':<22}{'点推定':>10}{'95%CI':>20}")
    print(f"  {'レース等額（現行の報告値）':<22}{d_eq.mean()*100:>+9.2f}pt{f'[{lo1:+.2f},{hi1:+.2f}]':>20}")
    print(f"  {'★金額加重（財布ベース）':<22}"
          f"{(pm.sum()/cm.sum()-pp.sum()/cp.sum())*100:>+9.2f}pt{f'[{lo2:+.2f},{hi2:+.2f}]':>20}")
    print(f"\n  点数別の内訳（モデル基準で分ける。同じレースで人気順も測る）")
    print(f"  {'モデルの点数':<14}{'R':>8}{'モデルROI':>11}{'人気順ROI':>11}{'差':>10}{'95%CI':>18}")
    for k in (100, 200):
        s = cm == k
        lo, hi = boot(d_eq[s], rng)
        print(f"  {f'{k//100}点({k}円)':<14}{int(s.sum()):>8,}{(pm[s]/cm[s]).mean()*100:>10.2f}%"
              f"{(pp[s]/cp[s]).mean()*100:>10.2f}%{d_eq[s].mean()*100:>+9.2f}pt"
              f"{f'[{lo:+.2f},{hi:+.2f}]':>18}")
    print("  ※三連複BOX4(常に400円)・複勝top1(常に100円)は点数が固定なのでこの問題を受けない")
    print(f"\n  場ごとに見る（(73)の小倉+14ptが単位の取り方で説明できるかの確認）")
    print(f"  {'場':<8}{'R':>8}{'モデル1点率':>12}{'人気順1点率':>12}"
          f"{'等額の差':>10}{'加重の差':>10}{'ズレ':>9}")
    tw = np.array([r["track"] for r in W])
    for t in sorted(set(tw)):
        s = tw == t
        if s.sum() < 300:
            continue
        e = ((pm[s] / cm[s]) - (pp[s] / cp[s])).mean() * 100
        g = (pm[s].sum() / cm[s].sum() - pp[s].sum() / cp[s].sum()) * 100
        print(f"  {t:<8}{int(s.sum()):>8,}{(cm[s]==100).mean()*100:>11.1f}%"
              f"{(cp[s]==100).mean()*100:>11.1f}%{e:>+9.2f}pt{g:>+9.2f}pt{e-g:>+8.2f}pt")

    # ============================================================ §2 裾への依存
    #    プラセボは「モデルと同じ一致率」に合わせた σ で引く（(79)①と同じ作法）
    print(f"\n{'='*100}\n§2 ★裾への依存: 払戻に上限を掛けても優位は残るか")
    specs = [("枠連 軸枠×紐枠2", "wk", "wk_m_pay", "wk_m_cost"),
             ("三連複 BOX上位4", "s3", "s3_m_pay", None),
             ("複勝 top1", "fk", "fk_m_pay", None)]
    store = {}
    for name, key, mp, mc in specs:
        R = [r for r in races if key in r]
        mpay = np.array([r[mp] for r in R])
        mcost = np.array([r[mc] for r in R]) if mc else np.full(len(R), 400.0 if key == "s3" else 100.0)
        ppay = np.array([r[mp.replace("_m_", "_p_")] for r in R])
        pcost = (np.array([r[mc.replace("_m_", "_p_")] for r in R]) if mc else mcost)
        # モデルの一致率（買い目が人気順と完全一致する率） → 同じ一致率になる σ を補間で求める
        agr = float(np.mean([_same_bets(r, key) for r in R]))
        curve = [(sg, np.mean([agree_rate(R, sg, 700 + k, key) for k in range(2 if sg else 1)]))
                 for sg in SIGMAS]
        xs = np.array([c[1] for c in curve])
        sg = float(np.interp(agr, xs[::-1], np.array(SIGMAS)[::-1]))
        draws = [placebo_pays(R, sg, 800 + k, key) for k in range(N_DRAW)]
        # ★プラセボは点数（コスト）も引きごとに変わるので、**引きごとにROIを作ってから平均**する。
        #   払戻とコストを別々に平均してから割ると比の平均にならず、枠連でズレる。
        def pl_roi(cap, draws=draws):     # 既定引数で束縛（遅延束縛だと最後の券種を参照してしまう）
            f = (lambda a: a if cap is None else np.minimum(a, cap))
            return np.mean([f(dp) / dc for dp, dc, _ in draws], axis=0)
        store[key] = (R, mpay, mcost, ppay, pcost, pl_roi, agr, sg)
        print(f"\n  --- {name}（{len(R):,}R・買い目一致率 {agr*100:.1f}% → プラセボ σ={sg:.3f}） ---")
        print(f"  {'払戻の上限':<12}{'モデル':>9}{'人気順':>9}{'プラセボ':>10}"
              f"{'M−人気順':>11}{'M−プラセボ':>12}{'そのCI':>18}{'残存率':>8}")
        base_skill = None
        for cap in CAPS:
            f = (lambda a: a if cap is None else np.minimum(a, cap))
            m_, p_, l_ = f(mpay) / mcost, f(ppay) / pcost, pl_roi(cap)
            sk = (m_ - l_)
            lo, hi = boot(sk, rng)
            if base_skill is None:
                base_skill = sk.mean()
            lab = "なし" if cap is None else f"{cap:,}円"
            print(f"  {lab:<12}{m_.mean()*100:>8.2f}%{p_.mean()*100:>8.2f}%{l_.mean()*100:>9.2f}%"
                  f"{(m_-p_).mean()*100:>+10.2f}pt{sk.mean()*100:>+11.2f}pt"
                  f"{f'[{lo:+.2f},{hi:+.2f}]':>18}"
                  f"{(f'{sk.mean()/base_skill*100:.0f}%' if abs(base_skill) > 0.005 else '—'):>7}")
        top = np.sort(mpay)[::-1]
        k1 = max(1, len(mpay) // 100)
        print(f"  上位1%のレース({k1:,}R)が総払戻に占める割合: "
              f"モデル {top[:k1].sum()/mpay.sum()*100:.1f}% / "
              f"人気順 {np.sort(ppay)[::-1][:k1].sum()/ppay.sum()*100:.1f}%")

    # ============================================================ §3 縮小の分解
    print(f"\n{'='*100}\n§3 ★近年の優位縮小を「配当の形」と「技能」に分ける")
    print("  プラセボ効果 = 中身の無いノイズ順が人気順を上回る量（配当分布の形だけで決まる）")
    print("  技能         = モデル − プラセボ")
    for name, key in (("枠連 軸枠×紐枠2", "wk"), ("三連複 BOX上位4", "s3")):
        R, mpay, mcost, ppay, pcost, pl_roi, agr, sg = store[key]
        yr = np.array([r["year"] for r in R])
        m_, p_, l_ = mpay / mcost, ppay / pcost, pl_roi(None)
        print(f"\n  --- {name} ---")
        print(f"  {'年':<7}{'R':>7}{'人気順':>9}{'プラセボ':>10}{'モデル':>9}"
              f"{'プラセボ効果':>13}{'技能(M−P)':>12}{'対人気順':>10}{'上位1%依存':>11}")
        for yy in sorted(set(yr)):
            s = yr == yy
            k1 = max(1, int(s.sum()) // 100)
            dep = np.sort(mpay[s])[::-1][:k1].sum() / max(mpay[s].sum(), 1)
            print(f"  {yy:<7}{int(s.sum()):>7,}{p_[s].mean()*100:>8.2f}%{l_[s].mean()*100:>9.2f}%"
                  f"{m_[s].mean()*100:>8.2f}%{(l_[s]-p_[s]).mean()*100:>+12.2f}pt"
                  f"{(m_[s]-l_[s]).mean()*100:>+11.2f}pt{(m_[s]-p_[s]).mean()*100:>+9.2f}pt"
                  f"{dep*100:>10.1f}%")
        print(f"  {'期間':<14}{'プラセボ効果':>14}{'その95%CI':>20}{'技能(M−P)':>12}{'その95%CI':>20}")
        ymax = int(yr.max())
        for lab, sel in (("全期間", yr >= 0), ("前半", yr < (y0 + ymax) // 2 + 1),
                         ("直近5年", yr >= ymax - 4)):
            a, b = boot((l_ - p_)[sel], rng), boot((m_ - l_)[sel], rng)
            print(f"  {lab:<14}{(l_-p_)[sel].mean()*100:>+13.2f}pt"
                  f"{f'[{a[0]:+.2f},{a[1]:+.2f}]':>20}{(m_-l_)[sel].mean()*100:>+11.2f}pt"
                  f"{f'[{b[0]:+.2f},{b[1]:+.2f}]':>20}")

    # ============================================================ §4 小倉の正体
    print(f"\n{'='*100}\n§4 ★小倉(+14pt)への新しい仮説: 「配当の裾が長い場ほどプラセボ効果が大きい」")
    print("  (76)③は『小倉は荒れるので組合せ券種の配当が跳ね、モデルがその一部を拾える場』を")
    print("  最も整合的な説明としたが、**測ってはいなかった**。(79)①のプラセボ効果はまさに")
    print("  『配当分布の裾の長さだけで決まる量』なので、場ごとに出せば直接の検定になる。")
    print("  事前宣言: 場ごとの プラセボ効果 と 対人気順の差 は正の相関を持ち、")
    print("            小倉の対人気順+14ptのかなりの部分はプラセボ効果で説明される。")
    for name, key in (("枠連 軸枠×紐枠2", "wk"), ("三連複 BOX上位4", "s3")):
        R, mpay, mcost, ppay, pcost, pl_roi, agr, sg = store[key]
        tr = np.array([r["track"] for r in R])
        m_, p_, l_ = mpay / mcost, ppay / pcost, pl_roi(None)
        rows = []
        for t in sorted(set(tr)):
            s = tr == t
            if s.sum() < 300:
                continue
            k1 = max(1, int(s.sum()) // 100)
            rows.append((t, int(s.sum()), p_[s].mean() * 100, (l_[s] - p_[s]).mean() * 100,
                         (m_[s] - l_[s]).mean() * 100, (m_[s] - p_[s]).mean() * 100,
                         np.sort(mpay[s])[::-1][:k1].sum() / max(mpay[s].sum(), 1) * 100))
        rows.sort(key=lambda x: -x[5])
        print(f"\n  --- {name} ---")
        print(f"  {'場':<8}{'R':>8}{'人気順ROI':>11}{'プラセボ効果':>13}{'技能(M−P)':>12}"
              f"{'対人気順':>10}{'上位1%依存':>11}")
        for t, nn, pr, pe, sk, tot, dep in rows:
            print(f"  {t:<8}{nn:>8,}{pr:>10.2f}%{pe:>+12.2f}pt{sk:>+11.2f}pt"
                  f"{tot:>+9.2f}pt{dep:>10.1f}%")
        a = np.array([[x[3], x[4], x[5]] for x in rows], float)
        print(f"  場をまたいだ相関: プラセボ効果 vs 対人気順 r={np.corrcoef(a[:,0],a[:,2])[0,1]:+.3f} / "
              f"プラセボ効果 vs 技能 r={np.corrcoef(a[:,0],a[:,1])[0,1]:+.3f}")

    with open("/tmp/units_races.pkl", "wb") as fh:
        pickle.dump([{k: v for k, v in r.items() if k not in ("wk", "s3", "fk")} for r in races], fh)
    print("\n（レース単位の結果を /tmp/units_races.pkl に保存）")


def _same_bets(r, key):
    po = r["uma"][np.argsort(r["lo"], kind="mergesort")]
    # モデル順は build_races で払戻だけ残したので、ここでは払戻とコストの一致で近似せず
    # 再計算できるよう uma / lo とモデル順の情報を持たせている
    mo = r["mo"]
    if key == "wk":
        return wakuren_cs(mo, r["n"]) == wakuren_cs(po, r["n"])
    if key == "s3":
        return set(mo[:4]) == set(po[:4])
    return mo[0] == po[0]


if __name__ == "__main__":
    main()
