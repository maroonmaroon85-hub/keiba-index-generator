"""(164) ★★運用は毎レース**三連複BOX4を400円**買っている。**投資額の2/3**。**これは正当か**。

★★動機——**文書の結論と運用が食い違っている**。
　`ROI_MAP` 項目24: **「券種の選択は決着。λ補正後に正なのは枠連 +0.0182 のみ」**
　（馬連 −0.0135 / **三連複 −0.0327**）。**なのに運用は三連複を買い続けている**。
⚠**ただし「Dが負だから買うな」は判定基準30で撤回済みの論法**——**Dは下界であって上界ではない**。
　**だから D では決められない**。★**ROIそのもので決める**（ユーザーの基準）。

測るもの（**事前登録。これだけ**）
　**(155)とまったく同じ選択**（MLモデル降順＋`waku_score` 下位を除外）の下で:
　　**A 枠連 紐1**（現行）／**B 三連複 BOX4**（現行）／**A+B の合算**（＝いま実際に張っている形）
　★**判定はユーザーの基準に従う**: **「Bを外すと合算ROIが上がる」なら外す提案をする**。
　★**判定基準8に従い、ROIだけでなく「1レースあたり期待損失（円）」も出す**——
　　**ROIは投資額で割るので、額の違う2つを混ぜると読み間違える**。
　★**対応のある差**で見る（**同じレース・同じ選択**なので独立ではない。判定基準35）。

⚠**ゲート（判定基準32）**: **除外0%のBが既知の84.5%を±2.5ptで再現しなければ何も読まない**。
⚠**運用の変更は提案までにする**。**勝手に変えない**。

実行: python3 ml/audit_puku_drop.py
"""
import math
import sys
from itertools import combinations

import numpy as np

sys.path.insert(0, "ml")
import features as F
from audit_crosspool import load_races, payoff, zq
from audit_crosspool2 import realized
from train_prod import CAPACITY, add_odds_features, fit_seeds
from waku_umatan import bracket_probs, waku_of, waku_score, wakuren_buy

RATES = [0.0, 0.30, 0.40, 0.60]
KNOWN_PUKU, TOL = 84.5, 2.5          # 三連複BOX4の既知ROI（陽性対照）


def stats(prof, cost, z):
    """(ROI, 1レース期待損益, seのROI換算) を返す。"""
    n = len(prof)
    se = prof.std(ddof=1) / math.sqrt(n)
    roi = 100.0 * (prof.sum() + cost.sum()) / cost.sum()
    return roi, prof.mean(), 100.0 * se * n / cost.sum(), se


def main():
    MODEL_DIR, PAR = CAPACITY["l2"]
    d = F.to_model(F.load_files())
    f = F.build_features(d)
    keep = (f["n_prior"] >= 1) & d["odds"].notna() & (d["odds"] > 0)
    d, f = d[keep].reset_index(drop=True), f[keep].reset_index(drop=True)
    y = (d["finish"] <= 3).astype(int).to_numpy()
    fx, _ = F.encode_categoricals(f)
    fx = add_odds_features(fx, d["odds"].to_numpy(float), d["raceid"].to_numpy())
    cut = d["date"].quantile(0.3)
    tr, te = (d["date"] < cut).to_numpy(), (d["date"] >= cut).to_numpy()
    print("(164) ★三連複BOX4を買い続けるのは正当か（**ROIで決める。Dでは決めない**）")
    print(f"　学習 {tr.sum():,} / 検証 {te.sum():,}・分割 {cut.date()}")
    print("　★経路: 買い目は**MLモデル降順**。除外の閾値は**検証側の分位**（実効率＝表示率）")
    ms = fit_seeds(fx[tr], y[tr], 3, PAR)
    p_ml = np.mean([m.predict_proba(fx[te])[:, 1] for m in ms], axis=0)
    sub = d.loc[te, ["raceid", "umaban"]].copy()
    sub["p"] = p_ml

    races = {r["rid"]: r for r in load_races()}
    rows = []                       # (score, 枠連損益, 枠連コスト, 三複損益, 三複コスト)
    for rid, g in sub.groupby("raceid"):
        r = races.get(str(rid))
        if r is None or not r.get("wakuren"):
            continue
        rl = realized(r)
        if rl is None:
            continue
        a, b, c = rl
        n, nums = r["n"], [u for u, _, _ in r["horses"]]
        if a not in nums or b not in nums or c not in nums:
            continue
        gg = g.sort_values("p", ascending=False)
        order = [int(u) for u in gg["umaban"].tolist()]
        pv = gg["p"].to_numpy(float)
        if len(order) < 4:
            continue
        key = tuple(sorted((waku_of(a, n), waku_of(b, n))))
        vw = payoff(r, "枠連(人気順)", [key[0], key[1]])
        vp = payoff(r, "三連複", sorted([a, b, c]))
        if not vw or vw <= 0 or vp is None:
            continue
        q = pv / pv.sum()
        bp = bracket_probs(order, q, n)
        # ⚠除外の判定は**紐2で作った分位**に合わせる（判定基準34・運用と同じ）
        sc = float(waku_score(wakuren_buy(order, n, 2), bp))
        pairs = wakuren_buy(order, n, 1)                  # ★運用は紐1
        cw = 100.0 * len(pairs)
        gw = vw if key in pairs else 0.0
        box = list(combinations(sorted(order[:4]), 3))    # ★運用は上位4頭BOX
        cp = 100.0 * len(box)
        gp = vp if tuple(sorted((a, b, c))) in box else 0.0
        rows.append((sc, gw - cw, cw, gp - cp, cp))

    if not rows:
        sys.exit("突き合わせできたレースが無い")
    A = np.array([[x[1], x[2], x[3], x[4]] for x in rows], float)
    sc = np.array([x[0] for x in rows])
    z = zq(0.01)
    print(f"　突き合わせ {len(rows):,}レース\n")

    print(f"{'除外率':>7}{'買うR':>8}"
          f"{'A枠連ROI':>11}{'B三複ROI':>11}{'A+B合算ROI':>13}{'A単独ROI':>11}"
          f"{'合算→A の差':>14}{'99%CI(対応差)':>20}")
    for rt in RATES:
        thr = np.quantile(sc, rt) if rt > 0 else -np.inf
        m = sc >= thr
        pw, cw, pp, cp = A[m, 0], A[m, 1], A[m, 2], A[m, 3]
        roi_w = 100.0 * (pw.sum() + cw.sum()) / cw.sum()
        roi_p = 100.0 * (pp.sum() + cp.sum()) / cp.sum()
        roi_ab = 100.0 * (pw.sum() + cw.sum() + pp.sum() + cp.sum()) / (cw.sum() + cp.sum())
        # ★対応のある差: 1レースあたり損益で比べる（判定基準8・35）
        d_yen = pw - (pw + pp)          # ＝ −pp（三連複を外すと損益はこう変わる）
        md = d_yen.mean()
        sd = d_yen.std(ddof=1) / math.sqrt(m.sum())
        print(f"{100*rt:>6.0f}%{int(m.sum()):>8,}"
              f"{roi_w:>10.1f}%{roi_p:>10.1f}%{roi_ab:>12.1f}%{roi_w:>10.1f}%"
              f"{roi_w-roi_ab:>+13.1f}"
              f"{f'[{md-z*sd:+.1f},{md+z*sd:+.1f}]円/R':>20}")

    m0 = np.ones(len(rows), bool)
    roi_p0 = 100.0 * (A[m0, 2].sum() + A[m0, 3].sum()) / A[m0, 3].sum()
    ok = abs(roi_p0 - KNOWN_PUKU) <= TOL
    print(f"\n⚠ゲート（判定基準32）: 除外0%の三連複BOX4 **{roi_p0:.1f}%** vs 既知 {KNOWN_PUKU}%"
          f"　差 {roi_p0-KNOWN_PUKU:+.1f}pt → **{'★立った' if ok else '⚠⚠落ちた'}**")
    if not ok:
        print("⚠⚠**対照が落ちた。上の表を読まない**。")
        return

    print("\n■ ★1レースあたりの期待損益（判定基準8: **ROIでなく円で比べる**）")
    print(f"{'除外率':>7}{'A枠連':>12}{'B三連複':>12}{'A+B合算':>12}{'Bを外す効果':>14}")
    for rt in RATES:
        thr = np.quantile(sc, rt) if rt > 0 else -np.inf
        m = sc >= thr
        pw, pp = A[m, 0], A[m, 2]
        print(f"{100*rt:>6.0f}%{pw.mean():>+11.1f}円{pp.mean():>+11.1f}円"
              f"{(pw+pp).mean():>+11.1f}円{-pp.mean():>+13.1f}円")

    print("\n" + "=" * 96)
    print("★読み方（**事前登録のとおり**）")
    print("  ★**ユーザーの基準は「よりROIが高くなるなら修正」**。")
    print("  ⚠**ROIだけを見ると誤る**——**三連複は1レース400円、枠連は100円**。")
    print("     **合算ROIは投資額の重みで決まる**ので、**円の表も一緒に読む**。")
    print("  ⚠**運用の変更は提案まで。勝手に変えない**。")


if __name__ == "__main__":
    main()
