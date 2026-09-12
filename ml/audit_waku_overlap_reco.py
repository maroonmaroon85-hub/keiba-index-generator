"""(138) ★★(137)を**推奨レースだけ**でやり直す — ユーザー指摘（2026-08-13）

★ユーザーの指摘「推奨レースだけに絞って考えて」
　(137)は28,567レース**全部**で測った。だが**運用が実際に買うのは枠連スコア上位60%だけ**
　（(117)で下位40%を除外することにした）。**枠の重なりが効くとすれば、買う場所で効くはず**。

★★この指摘には根拠がある（思いつきではない）
　(117)は**推奨レースが実際に違う集団だと実測している**:
　　除外0% → D=+0.0184 ／ 除外40%（現行の推奨）→ **D=+0.0235**（★全水準・10/10年）。
　**推奨レースはDが約28%高い**。だから「全体で平坦でも推奨レースでは違う」は**ありうる**。

⚠★★ただし**先に検出力を書く**（判定基準5・(46)「絞ると測れなくなる」）
　(137)の層間差のCI幅は S1 で 0.023（28,567レース）。**標本が減るとCI幅は 1/√n で広がる**:
　　除外 0%（28,567R）→ ゾロ目層 1,683R → CI幅 ≈ 0.023
　　除外40%（推奨・約17,100R）→ ゾロ目層 約1,010R → CI幅 ≈ **0.030**
　　除外70%（上位30%・約8,600R）→ ゾロ目層 約510R → CI幅 ≈ **0.042**
　★**(137)の判定条件「層間差 +0.005超」は、推奨レースでは検出できない**。
　　**検出できるのは ±0.015（推奨）／±0.021（上位30%）より大きい効果だけ**。
　→ ★**だから「有意差なし」は「効果なし」を意味しない**。**この実験が答えられるのは
　　「**大きい効果があるか**」だけ**であり、それを**測る前に**書いておく。

★★事前登録（測る前に宣言する）
　1. **層は(137)と同じ3つ**（S1 軸と紐1が同枠 / S2 上位3頭が2枠以内 / S3 枠集中度）。
　　 ⚠★**実装で事前登録から1点ずれた（正直に記録する）**: S3を(137)の**十分位**ではなく
　　 　**上位半分 vs 下位半分**にした。**絞ると十分位1つが数百本になり検出力が無くなる**ため。
　　 　**結論は変わらない**——(137)が十分位で測って +0.0052・CIが0をまたいでいる。
　　 　★だが**宣言と実装がずれたこと自体は誤り**なので、数字より先にここに書く。
　　 **後から層を増やさない**。q は**(127)の新しい土台＝馬連プール→枠**。
　2. **除外率の梯子 0% / 20% / 40%（現行の推奨）/ 60% / 70%** で層間差がどう動くかを見る。
　　 ★**「絞るほど層間差が大きくなる」なら単調性が出るはず**。**単調性(ρ)を判定に使う**
　　 （判定基準1: 最良のビンで判定しない）。
　3. **★判定条件**: 推奨レース(40%除外)で**層間差の99%CI下端>0** かつ
　　 **梯子の単調性 ρ ≥ +0.8** かつ **年8/11以上**。**3つ揃わなければ運用は変えない**。
　4. **プラセボ**: 判定基準23のとおり**層の無作為割り当ては構造上0**なので**実装の検査**に格下げ。
　　 ★代わりに**「無作為に同数だけ間引いた集団」で同じ梯子を引く**——これは**0にならない**ので
　　 　**本物のプラセボになる**。**絞ること自体が層間差を生むか**を検査する。30回平均。
　5. **予想（判定基準24に従い、根拠の種類を明記）**
　　 ・**［類推・あてにしない］どの除外率でも平坦のまま**。(137)が全体で平坦だったから。
　　 ・~~★［恒等式から言えること］層間差の「加重平均」は保存される~~
　　 　 ⚠★★**これは誤りだった。恒等式ではない**（測ったあとに実測で気づいた）。
　　 　 **群の割合が層ごとに違う**（推奨のS2=1率21.1% / 除外側13.0%）ので、
　　 　 **平均の差は「差の加重平均」にならない**（シンプソンの形）。
　　 　 実測: 素朴な加重平均 −0.00012 に対し**全体の層間差は +0.00080 で一致しない**。
　　 　 → ★**「恒等式だから信用してよい」と書く前に、実際に閉じるか数値で確かめること**（判定基準27）。
　　 ⚠**結局この事前登録は「当てにしてよい根拠」を1つも持っていなかった**。
　　 　 それでも**両側（推奨/除外）を並べて出す**という手続き自体は残す価値がある。

実行: python3 ml/audit_waku_overlap_reco.py [開始年(既定2015)]
"""
import glob
import math
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "ml")
from audit_capacity_d import mkt_waku_dist
from audit_cond_split import load_boards
from audit_crosspool import load_races, probs, zq
from audit_crosspool2 import realized
from audit_lbs import build_matrix, fit_lambda
from audit_waku_vs_umaren import load_type
from waku_umatan import bracket_probs, waku_of, waku_score, wakuren_buy

MODEL_CACHE = "data/cache/exp_L2-top3_2015"
CUTS = [0.0, 0.20, 0.40, 0.60, 0.70]      # ★先に宣言。0.40が現行の推奨
NPLA = 30
RNG = np.random.default_rng(20260813)


def mci(x, alpha=0.01):
    x = np.asarray(x, float)
    if len(x) < 2:
        return float("nan"), float("nan"), float("nan")
    m = x.mean()
    se = x.std(ddof=1) / math.sqrt(len(x))
    return m, m - zq(alpha) * se, m + zq(alpha) * se


def two_sample(a, b, alpha=0.01):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan"), float("nan")
    d = a.mean() - b.mean()
    se = math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return d, d - zq(alpha) * se, d + zq(alpha) * se


def load_model_p():
    fs = sorted(glob.glob(f"{MODEL_CACHE}/*.csv"))
    if not fs:
        sys.exit(f"{MODEL_CACHE} が無い。")
    out = {}
    for f in fs:
        for rid, u, p in pd.read_csv(f)[["raceid", "umaban", "p"]].itertuples(index=False):
            out.setdefault(str(rid), {})[int(u)] = float(p)
    return out


def main():
    y0 = int(sys.argv[1]) if len(sys.argv) > 1 else 2015
    mp = load_model_p()
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

    rows = []
    for r in races:
        yy = r["year"]
        if yy < y0 or not lam.get(yy) or not r["wakuren"]:
            continue
        W, U = wb.get(r["rid"]), ub.get(r["rid"])
        mm = mp.get(str(r["rid"]))
        if not W or not U or not mm:
            continue
        rl = realized(r)
        if rl is None:
            continue
        a, b, _ = rl
        n = r["n"]
        nums = [u for u, _, _ in r["horses"]]
        if a not in nums or b not in nums or any(u not in mm for u in nums):
            continue
        pm = np.array([mm[u] for u in nums], float)
        if pm.sum() <= 0:
            continue
        order = sorted(nums, key=lambda u: -mm[u])
        key = tuple(sorted((waku_of(a, n), waku_of(b, n))))
        if key not in W:
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
        if key not in keys or len(keys) < 3:
            continue
        sW = sum(1.0 / W[k] for k in keys)
        sU = sum(agg[k] for k in keys)
        du = math.log(agg[key] / sU) - math.log((1.0 / W[key]) / sW)

        bp = bracket_probs(nums, pm, n)
        sc = waku_score(wakuren_buy(order, n, 2), bp)        # ★(117)と同じ作り方
        w_axis, w_h1 = waku_of(order[0], n), waku_of(order[1], n)
        rows.append((yy, sc, du,
                     int(w_axis == w_h1),
                     int(len({waku_of(h, n) for h in order[:3]}) <= 2),
                     sum(v * v for v in bp.values())))

    if not rows:
        sys.exit("突き合わせできたレースが無い")
    ys = np.array([x[0] for x in rows])
    SC = np.array([x[1] for x in rows])
    DU = np.array([x[2] for x in rows])
    S1 = np.array([x[3] for x in rows])
    S2 = np.array([x[4] for x in rows])
    HHI = np.array([x[5] for x in rows])
    n = len(rows)
    print(f"(138) (137)を**推奨レースだけ**でやり直す（{y0}年以降・{n:,}レース）")
    print("★推奨レース＝枠連スコア上位60%（(117)で下位40%を除外することにした）")
    print("⚠**先に書いた検出力**: 推奨で ±0.015、上位30%で ±0.021 より大きい効果しか見えない\n")

    LAB = {"S1": "軸と紐1が同枠（ゾロ目になる）", "S2": "上位3頭が2枠以内",
           "S3": "枠集中度Σp² 上位半分 vs 下位半分 ⚠事前登録は十分位だった（冒頭の注）"}

    def strat(tag, keep):
        if tag == "S1":
            return S1[keep] == 1, S1[keep] == 0
        if tag == "S2":
            return S2[keep] == 1, S2[keep] == 0
        h = HHI[keep]
        return h >= np.median(h), h < np.median(h)

    for tag in ("S1", "S2", "S3"):
        print(f"■ {tag} {LAB[tag]}")
        print(f"{'除外率':>8}{'R数':>9}{'該当':>8}{'D(該当)':>11}{'D(非該当)':>12}"
              f"{'★層間差':>10}{'99%CI':>22}{'年':>6}{'プラセボ':>10}")
        diffs = []
        for c in CUTS:
            th = np.quantile(SC, c) if c > 0 else -np.inf
            keep = SC >= th
            ga, gb = strat(tag, keep)
            dk, du_, dv = DU[keep], DU[keep][ga], DU[keep][gb]
            if len(du_) < 30 or len(dv) < 30:
                print(f"{c:>7.0%}{keep.sum():>9,}   標本不足")
                diffs.append(np.nan)
                continue
            d, lo, hi = two_sample(du_, dv)
            diffs.append(d)
            yk = ys[keep]
            yl = sorted(set(yk.tolist()))
            pos = sum(1 for yy in yl
                      if (yk == yy).sum() >= 100
                      and ga[yk == yy].sum() >= 5
                      and (dk[(yk == yy)][ga[yk == yy]].mean()
                           - dk[(yk == yy)][gb[yk == yy]].mean()) * (1 if d >= 0 else -1) > 0)
            # ★プラセボ: 同数を無作為に間引く（絞ること自体が層間差を生むか）
            pl = []
            for _ in range(NPLA):
                idx = RNG.choice(n, size=int(keep.sum()), replace=False)
                ga2, gb2 = strat(tag, idx)
                if ga2.sum() >= 30 and gb2.sum() >= 30:
                    pl.append(DU[idx][ga2].mean() - DU[idx][gb2].mean())
            mark = "★" if lo > 0 else ("★負" if hi < 0 else "")
            ci = "[" + format(lo, "+.4f") + "," + format(hi, "+.4f") + "]"
            print(f"{c:>7.0%}{keep.sum():>9,}{ga.sum():>8,}{du_.mean():>+11.4f}{dv.mean():>+12.4f}"
                  f"{d:>+10.4f}{ci:>22}{pos:>4}/{len(yl)}{np.mean(pl):>+10.4f} {mark}")
        ok = [(c, d) for c, d in zip(CUTS, diffs) if not np.isnan(d)]
        if len(ok) >= 3:
            rho = np.corrcoef([c for c, _ in ok], [d for _, d in ok])[0, 1]
            print(f"  → 除外率に対する単調性 ρ={rho:+.3f}"
                  f"（判定条件は ρ≥+0.8 かつ 推奨(40%)でCI下端>0 かつ 年8/11以上）\n")

    # ★恒等式の確認（事前登録5の2つ目）— 除外した側も必ず出す
    print("── 除外した側も並べる ──")
    print("  ⚠★事前登録5に『層間差の加重平均は保存される（恒等式）』と書いたが**誤り**。")
    print("  　 群の割合が層で違うので平均の差は差の加重平均にならない（シンプソンの形）。")
    print("  　 **恒等式ではないので検算には使えない**。並べる意味は『片側だけ都合よく")
    print("  　 大きくなっていないか目で見る』ことに留まる。")
    th = np.quantile(SC, 0.40)
    for name, m in (("推奨（上位60%）", SC >= th), ("除外した側（下位40%）", SC < th)):
        ga, gb = strat("S2", m)
        d, lo, hi = two_sample(DU[m][ga], DU[m][gb])
        print(f"  S2 {name:<18}{m.sum():>8,}本  層間差 {d:+.4f} [{lo:+.4f},{hi:+.4f}]")
    ga, gb = strat("S2", np.ones(n, bool))
    d, lo, hi = two_sample(DU[ga], DU[gb])
    nr = int((SC >= th).sum()); ne = n - nr
    dr_, de_ = None, None
    for name, m in (("r", SC >= th), ("e", SC < th)):
        ga2, gb2 = strat("S2", m)
        v = DU[m][ga2].mean() - DU[m][gb2].mean()
        if name == "r":
            dr_ = v
        else:
            de_ = v
    print(f"  S2 {'全体':<18}{n:>8,}本  層間差 {d:+.4f} [{lo:+.4f},{hi:+.4f}]")
    print(f"  ⚠素朴な加重平均は {(nr*dr_+ne*de_)/n:+.5f} で**全体と一致しない**"
          f"（恒等式ではない。上の注のとおり）")

    print("\n" + "=" * 104)
    print("★読み方（事前登録のとおり）")
    print("  ・判定は**3つ揃ったときだけ**（推奨でCI下端>0・単調性ρ≥+0.8・年8/11以上）。")
    print("  ・⚠**「有意差なし」は「効果なし」ではない**。推奨レースでは ±0.015 より小さい効果は")
    print("    **原理的に見えない**（測る前に書いた検出力）。**そこは(46)のとおり答えられない**。")
    print("  ・★プラセボ（同数を無作為に間引く）が0から離れたら、**絞ること自体の副作用**を疑う。")


if __name__ == "__main__":
    main()
