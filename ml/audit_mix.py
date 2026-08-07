"""(100) ★★★モデルは市場の**上に**情報を足せるか — 対数線形プールで混合比を最尤推定する。

★これまでの検証の穴（ここが本題）
　(90)(97)は「**モデルか市場か**」を比べていた。どちらも単体で q を作って D を比べる形。
　だが情報理論で普通に問うべきなのは「**市場に、モデルを少し混ぜたら良くなるか**」。
　**単体で劣る予測器でも、限界的に情報を足せることはある**（相関が1でない限り一般に起きる）。
　　例: 市場が見落とす方向にモデルの誤差が乗っていなければ、少量の混合は必ず利く。
　★この検定を**一度もやっていない**。(90)(97)の「モデルは市場に負けている」は
　　**「混ぜても無駄」を意味しない**。ここを埋める。

★混合の形（対数線形プール＝logarithmic opinion pool）
　　q_i(w) ∝ p_market_i^(1-w) · p_model_i^w
　w=0 が市場そのもの、w=1 がモデルそのもの。**wを各年それ以前の年だけで最尤推定**する。
　対数線形にする理由: 対数スコアで測るので、**同じ土俵（指数族）で最適化される形**が自然。
　線形混合 (1-w)p_m + w·p_M も併記して、形に依存しないことを確かめる。

★★事前登録（測る前に宣言）
　1. **予想**: w* > 0 になる。単体で劣るからといって情報が無いわけではないから。
　　 ただし **w* は小さい**（0.1〜0.3程度）と予想する。(97)でモデルは市場に0.056負けている。
　2. **判定**: `D(混合) − D(市場)` が **99%CIで0を除外して正**なら、
　　 **このプロジェクトで初めて「モデルが市場の上に情報を積んだ」直接証拠**になる。
　3. **★プラセボ必須**: モデル確率を**レース内でシャッフル**した偽モデルで同じ手続きを踏む。
　　 分布は同じで情報だけ無い。ここで w*≈0 かつ利得≈0 にならなければ、**手続き自体が嘘をついている**。
　4. **★これでも儲かるとは限らない**: 必要量は枠連0.2549。(97)の市場D=+0.0182との差は0.2367。
　　 **利得がこれを埋めることはまず無い**。だが「モデルに価値があるか」という別の問いには答えが出る。
　5. **較正も同時に**: (97)より混合後の分布にも τ/λ 補正が要る。**混合の後に補正を当てる**。

実行: python3 ml/audit_mix.py [シード数(既定3)] [開始年(既定2015)]
　　（(97)が作った data/cache/winprob_*.csv があれば学習をやり直さない）
"""
import math
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, "ml")
from audit_crosspool import PAYBACK, load_races, payoff, probs, zq
from audit_crosspool2 import PARTS, PAYKEY, realized
from audit_lbs import GRID, q_of_lbs
from audit_lbs_model import fit_exponent, model_win_probs, race_probs

WGRID = np.round(np.arange(0.0, 1.001, 0.02), 4)      # ★事前に固定
RNG = np.random.default_rng(20260806)


def mix_log(pm, pM, w):
    """対数線形プール。0はlogで落ちるので下限を入れる。"""
    q = np.exp((1 - w) * np.log(np.maximum(pm, 1e-12)) + w * np.log(np.maximum(pM, 1e-12)))
    return q / q.sum()


def mix_lin(pm, pM, w):
    q = (1 - w) * pm + w * pM
    return q / q.sum()


def fit_w(rows, mixer):
    """1着の的中を最大化する w（各年それ以前の年だけを渡すこと）。"""
    def ll(w):
        return sum(math.log(max(mixer(pm, pM, w)[k], 1e-12)) for pm, pM, k in rows)
    return float(max(WGRID, key=ll))


def build_rows(races, pmap, y0, shuffle=False):
    """[(p_market, p_model, 勝ち馬のindex, 年, race)]"""
    out = []
    for r in races:
        if r["year"] < y0:
            continue
        rl = realized(r)
        if rl is None:
            continue
        a = rl[0]
        pm = race_probs(r, None)
        pM = race_probs(r, pmap)
        if pM is None:
            continue
        if shuffle:
            pM = pM[RNG.permutation(len(pM))]      # ★分布は同じ・情報だけ壊す
        num2k = {num: k for k, (num, _, _) in enumerate(r["horses"])}
        if a not in num2k:
            continue
        out.append((pm, pM, num2k[a], r["year"], r))
    return out


def mci(x, alpha=0.01):
    x = np.asarray(x, float)
    m = x.mean()
    se = x.std(ddof=1) / math.sqrt(len(x))
    z = zq(alpha)
    return m, m - z * se, m + z * se


def run(rows, mixer, label):
    """wをウォークフォワード推定 → 混合後にτ/λを当てて → D を測る。"""
    years = sorted({r[3] for r in rows})
    ws = {}
    for yy in years:
        tr = [(pm, pM, k) for pm, pM, k, y, _ in rows if y < yy]
        ws[yy] = fit_w(tr, mixer) if len(tr) >= 3000 else None
    got = {k: v for k, v in ws.items() if v is not None}
    print(f"\n■ {label}: w のウォークフォワード推定")
    print("   " + " ".join(f"{y}:{v:.2f}" for y, v in got.items()))

    # 混合後の分布に τ/λ を当てる（(97)と同じ手続き・各年それ以前の年だけ）
    mx = max(len(r[0]) for r in rows)
    P, i1, i2, i3, yrs = [], [], [], [], []
    for pm, pM, k, yy, r in rows:
        if yy not in got:
            continue
        rl = realized(r)
        a, b, c = rl
        num2k = {num: kk for kk, (num, _, _) in enumerate(r["horses"])}
        if b not in num2k:
            continue
        q = mixer(pm, pM, got[yy])
        v = np.zeros(mx)
        v[:len(q)] = q
        P.append(v)
        i1.append(num2k[a])
        i2.append(num2k[b])
        i3.append(num2k[c] if (c is not None and c in num2k) else -1)
        yrs.append(yy)
    P, i1, i2, i3, yrs = (np.array(x) for x in (P, i1, i2, i3, yrs))
    par = {}
    for yy in sorted(set(yrs.tolist())):
        tr = yrs < yy
        if tr.sum() < 3000:
            par[yy] = None
            continue
        ok3 = tr & (i3 >= 0)
        par[yy] = (fit_exponent(P[tr], i1[tr]),
                   fit_exponent(P[tr], i2[tr], drop_idx=(i1[tr],)),
                   fit_exponent(P[ok3], i3[ok3], drop_idx=(i1[ok3], i2[ok3])))
    return got, par


def measure(races, pmap, got, par, mixer, mkt_par):
    """混合 q と 市場 q の D を、同じレースで対応ありで集める。"""
    res = {k: [] for k in PARTS}
    for r in races:
        yy = r["year"]
        if yy not in got or par.get(yy) is None or mkt_par.get(yy) is None:
            continue
        rl = realized(r)
        if rl is None:
            continue
        a, b, c = rl
        num2k = {num: k for k, (num, _, _) in enumerate(r["horses"])}
        if a not in num2k or b not in num2k or (c is not None and c not in num2k):
            continue
        pm = race_probs(r, None)
        pM = race_probs(r, pmap)
        if pM is None:
            continue
        qmix = mixer(pm, pM, got[yy])
        t, l2, l3 = par[yy]
        tk, l2k, l3k = mkt_par[yy]
        qmix = qmix ** t / (qmix ** t).sum()
        pmk = pm ** tk / (pm ** tk).sum()
        for kind, key in PARTS.items():
            if not r[key]:
                continue
            qx, combo = q_of_lbs(kind, r, qmix, l2, l3, num2k, a, b, c)
            qk, _ = q_of_lbs(kind, r, pmk, l2k, l3k, num2k, a, b, c)
            if qx <= 0 or qk <= 0 or combo is None:
                continue
            v = payoff(r, PAYKEY[kind], combo)
            if not v or v <= 0:
                continue
            lp = math.log((v + 5) / 100.0) - math.log(PAYBACK[kind])
            res[kind].append((math.log(qx) + lp, math.log(qk) + lp))
    return res


def main():
    n_seed = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    y0 = int(sys.argv[2]) if len(sys.argv) > 2 else 2015
    print(f"(100) 市場にモデルを混ぜると情報は増えるか（{y0}年以降・シード{n_seed}本）")
    print("★判定: D(混合) − D(市場) が99%CIで正 → 初めて『モデルが市場の上に積んだ』直接証拠\n")
    pmap = model_win_probs(n_seed, y0)
    races = load_races()

    # 市場側の τ/λ（比較の基準・(97)と同じ）
    from audit_lbs_model import build, fit_walk
    Pk, k1, k2, k3, kyr = build(races, y0, None)
    mkt_par = fit_walk(Pk, k1, k2, k3, kyr)

    rows = build_rows(races, pmap, y0)
    rows_pl = build_rows(races, pmap, y0, shuffle=True)
    print(f"対象 {len(rows):,}レース")

    for tag, rr, pm_ in (("★本番", rows, pmap), ("プラセボ（レース内シャッフル）", rows_pl, pmap)):
        print("\n" + "=" * 96)
        print(f"=== {tag} ===")
        print("=" * 96)
        for mixer, mlab in ((mix_log, "対数線形プール"), (mix_lin, "線形混合")):
            got, par = run(rr, mixer, mlab)
            if tag.startswith("プラセボ"):
                # プラセボはシャッフル済みの確率で測る必要があるので pmap を差し替える
                pl = {}
                for pmv, pMv, k, yy, r in rr:
                    pl[r["rid"]] = {num: float(pMv[i])
                                    for i, (num, _, _) in enumerate(r["horses"])}
                res = measure(races, pl, got, par, mixer, mkt_par)
            else:
                res = measure(races, pm_, got, par, mixer, mkt_par)
            print(f"{'券種':<8}{'R数':>8}{'D(混合)':>11}{'D(市場)':>11}{'利得':>11}"
                  f"{'利得の99%CI':>22}{'必要量':>9}{'残り':>10}")
            for kind in PARTS:
                v = res[kind]
                if len(v) < 500:
                    continue
                dx = np.array([x[0] for x in v])
                dk = np.array([x[1] for x in v])
                g, lo, hi = mci(dx - dk)
                need = -math.log(PAYBACK[kind])
                print(f"{kind:<8}{len(v):>8,}{dx.mean():>+11.4f}{dk.mean():>+11.4f}{g:>+11.4f}"
                      f"{f'[{lo:+.4f},{hi:+.4f}]':>22}{need:>9.4f}{need-dx.mean():>10.4f}")

    print("\n" + "=" * 96)
    print("★読み方")
    print("  ・本番で利得が正・プラセボで0なら、**モデルは市場の上に情報を積んでいる**。")
    print("    (90)(97)の『モデルは市場に負ける』と矛盾しない。単体で劣ることと限界的に足せることは別。")
    print("  ・本番でもプラセボでも同じだけ出るなら、**手続きが利得を生んでいる**ので全部無効。")
    print("  ・利得が正でも『残り』の欄が0にならない限り儲からない（ケリーが上限）。")


if __name__ == "__main__":
    main()
