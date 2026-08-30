"""(170) ★(165)で向きだけ合っていた3つの切り口を、**対応のある差**で測り直す

★★動機——**(165)で3つとも同じ向きに出た**が、**水準の比較しかしていなかった**:
　**未勝利 83.8%（全体86.6%）／2歳戦 80.4%（3歳以上87.6%）／見えないシェア2%超 78.5%**。
　★**「2歳戦を外すと 86.6% → 87.6%（+1.0pt）」を、対応のある差で確かめる**のが本件。

⚠⚠**これは新しい証拠にならない（先に明記する）**——
　**(165)とまったく同じ25,719レース・同じ切り口**。**独立な確認ではない**。
　★**やる意味は「水準の比較を、正しい統計量（対応差・選択込み・円）に直す」ことだけ**。
　**ここで有意に出ても、それは(165)の post-hoc な切り出しが有意になっただけ**（判定基準25）。
　→ ★**採用の条件を厳しくする**: **有意でも、別の期間で再現するまで運用に入れない**。

事前登録（**この3アームだけ。後から増やさない**）
　★**A 2歳戦を外す**（本件）／**B 未勝利を外す**／**C 見えないシェア2%超を外す**
　★**主判定は「合算の1レース期待損益」**（枠連100円＋三連複400円＝運用そのもの）。
　　**外したレースは損益0**として全レースで平均する（**判定基準8・35**）。
　　**Bonferroni α=0.01/3**。
　⚠**ゲート（判定基準32）**: **現行が既知85.3%（枠連）を±2.5ptで再現しなければ読まない**。
　⚠**「レースを減らすと1レース平均の損失が減る」のは当たり前**——**賭けを減らしただけ**。
　　★**だから ROI も併記する**。**ROIが上がって初めて「良くなった」と言える**。

★★★実測（2026-08-30・ゲート: 現行の枠連 86.7% vs 既知85.3%＝+1.4pt → 立った）
**主判定（訂正版）: 外す側のROI vs 残す側のROI**
| アーム | | 外す側 | 残す側 | 差 | 99%CI(Bonf) | |
|---|---|---|---|---|---|---|
| A 2歳戦 | 枠連 | 80.5% | 87.8% | −7.3 | [−18.6,+4.0] | ⚠検出できない |
| | 合算 | 81.6% | 87.3% | −5.6 | [−15.7,+4.5] | ⚠検出できない |
| B 未勝利 | 枠連 | 83.8% | 89.2% | −5.4 | [−14.3,+3.4] | ⚠検出できない |
| | 合算 | 86.9% | 86.1% | ★**+0.8** | [−8.0,+9.6] | ⚠**符号が逆転** |
| C 見えない>2% | 枠連 | 79.4% | 87.5% | −8.2 | [−22.6,+6.2] | ⚠検出できない |
| | 合算 | 87.4% | 86.3% | ★**+1.1** | [−13.3,+15.5] | ⚠**符号が逆転** |
★**3つとも枠連では正しい向きだが、どれも有意にならない**。
★★**合算（＝実際に張っている形）では B と C は符号が逆転する**——
　**未勝利や「見えない馬が多い」レースは三連複のほうが良い**ため。
→ ★**「2歳戦を外す」の根拠は無い。運用は変えない**。

⚠⚠**第1版の設計ミスを記録として残す（同日中に訂正）**
　**主判定に「合算1R損益の差」を置いた**が、**これは外せば必ずプラスになる量**——
　**ROIが100%未満なので、どんな部分集合を捨てても1レース平均の損失は減る**。
　**実際 A +8.1円[+4.2,+11.9]・B +18.4円[+10.0,+26.9] と「有意」に出たが、何も意味していない**。
　⚠**事前登録に自分でその注意を書いておきながら、それを主判定に置いた**。→ **判定基準42**。

実行: python3 ml/audit_drop_slices.py
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

EXCL, KNOWN, TOL, NCMP = 0.40, 85.3, 2.5, 3
COST_P = 400.0


def main():
    MODEL_DIR, PAR = CAPACITY["l2"]
    d0 = F.to_model(F.load_files())
    f = F.build_features(d0)
    keep = (f["n_prior"] >= 1) & d0["odds"].notna() & (d0["odds"] > 0)
    allsum, dropsum = {}, {}
    for rid, o, k in zip(d0["raceid"].to_numpy(), d0["odds"].to_numpy(float),
                         keep.to_numpy()):
        if not np.isfinite(o) or o <= 0:
            continue
        allsum[rid] = allsum.get(rid, 0.0) + 1.0 / o
        if not k:
            dropsum[rid] = dropsum.get(rid, 0.0) + 1.0 / o
    d, f = d0[keep].reset_index(drop=True), f[keep].reset_index(drop=True)
    y = (d["finish"] <= 3).astype(int).to_numpy()
    fx, _ = F.encode_categoricals(f)
    fx = add_odds_features(fx, d["odds"].to_numpy(float), d["raceid"].to_numpy())
    cut = d["date"].quantile(0.3)
    tr, te = (d["date"] < cut).to_numpy(), (d["date"] >= cut).to_numpy()
    print("(170) ★(165)の3つの切り口を対応のある差で測り直す")
    print(f"　学習 {tr.sum():,} / 検証 {te.sum():,}・分割 {cut.date()}")
    print("⚠⚠**(165)と同じデータ・同じ切り口。独立な確認ではない**\n")
    ms = fit_seeds(fx[tr], y[tr], 3, PAR)
    p = np.mean([m.predict_proba(fx[te])[:, 1] for m in ms], axis=0)
    sub = d.loc[te, ["raceid", "umaban", "raceclass", "age"]].copy()
    sub["p"] = p

    races = {r["rid"]: r for r in load_races()}
    rows = []
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
        sc = float(waku_score(wakuren_buy(order, n, 2), bp))
        pairs = wakuren_buy(order, n, 1)
        cw = 100.0 * len(pairs)
        box = list(combinations(sorted(order[:4]), 3))
        tot = allsum.get(str(rid), 0.0)
        rows.append(dict(
            sc=sc, w=(vw if key in pairs else 0.0) - cw, cw=cw,
            pp=(vp if tuple(sorted((a, b, c))) in box else 0.0) - COST_P,
            two=int(gg["age"].min()) <= 2,
            mis=int(gg["raceclass"].iloc[0]) <= 1,
            blind=(dropsum.get(str(rid), 0.0) / tot) if tot > 0 else 0.0))

    N = len(rows)
    sc = np.array([r["sc"] for r in rows])
    sel = sc >= np.quantile(sc, EXCL)
    w = np.where(sel, [r["w"] for r in rows], 0.0)
    cw = np.where(sel, [r["cw"] for r in rows], 0.0)
    pp = np.where(sel, [r["pp"] for r in rows], 0.0)
    cp = np.where(sel, COST_P, 0.0)
    z = zq(0.01 / NCMP)
    roi0 = 100.0 * (w.sum() + cw.sum()) / cw.sum()
    ok = abs(roi0 - KNOWN) <= TOL
    print(f"★{N:,}レース／買う {int(sel.sum()):,}")
    print(f"⚠ゲート: 現行の枠連 **{roi0:.1f}%** vs 既知 {KNOWN}%　差 {roi0-KNOWN:+.1f}pt"
          f" → **{'★立った' if ok else '⚠⚠落ちた'}**")
    if not ok:
        print("⚠⚠落ちた。読まない。"); return
    print(f"　現行: 枠連ROI {roi0:.1f}% ／ 三連複ROI "
          f"{100*(pp.sum()+cp.sum())/cp.sum():.1f}% ／ 合算1R {(w+pp).mean():+.1f}円\n")

    ARMS = [("A 2歳戦を外す", np.array([r["two"] for r in rows], bool)),
            ("B 未勝利を外す", np.array([r["mis"] for r in rows], bool)),
            ("C 見えない>2%を外す", np.array([r["blind"] > 0.02 for r in rows]))]
    # ⚠⚠★**第1版はここを間違えた（同日中に訂正）**——
    #   **主判定に「合算1R損益の差」を置いたが、これは外せば必ずプラスになる量**。
    #   **ROIが100%未満なので、どんな部分集合を捨てても1レース平均の損失は減る**。
    #   ★**正しい問いは「外す側のROIが、残す側より本当に悪いか」**。
    #   　**外す側と残す側は互いに素なので、2標本比較として正しく検定できる**。
    print("■ ★★主判定（訂正版）: **外す側のROI vs 残す側のROI**（互いに素・2標本）")
    print(f"{'アーム':<20}{'外すR':>8}{'外す側ROI':>11}{'残す側ROI':>11}"
          f"{'差(pt)':>9}{'99%CI(Bonf)':>21}{'判定':>16}")
    for name, drop in ARMS:
        for lab, cost, prof in (("枠連", cw, w), ("合算", cw + cp, w + pp)):
            din, dout = sel & drop, sel & ~drop
            ci_, co_ = cost[din], cost[dout]
            pi_, po_ = prof[din], prof[dout]
            ri = 100.0 * (pi_.sum() + ci_.sum()) / ci_.sum()
            ro = 100.0 * (po_.sum() + co_.sum()) / co_.sum()
            sei = pi_[din[din]].std(ddof=1) if False else \
                prof[din].std(ddof=1) / math.sqrt(int(din.sum())) * int(din.sum()) / ci_.sum() * 100
            seo = prof[dout].std(ddof=1) / math.sqrt(int(dout.sum())) * int(dout.sum()) / co_.sum() * 100
            dd = ri - ro
            sd = math.hypot(sei, seo)
            v = "★差がある" if abs(dd) > z * sd else "⚠差は検出できない"
            print(f"{(name if lab=='枠連' else '')+' '+lab:<20}"
                  f"{int(din.sum()) if lab=='枠連' else 0:>8,}"
                  f"{ri:>10.1f}%{ro:>10.1f}%{dd:>+9.1f}"
                  f"{f'[{dd-z*sd:+.1f},{dd+z*sd:+.1f}]':>21}{v:>16}")

    print("\n■ （参考）合算1R損益の差 — ⚠**外せば必ずプラスになる量。判定に使わない**")
    print(f"{'アーム':<20}{'外すR':>8}{'枠連ROI':>10}{'三複ROI':>10}"
          f"{'合算1R差':>11}{'99%CI(Bonf)':>21}{'判定':>16}")
    for name, drop in ARMS:
        s2 = sel & ~drop
        w2 = np.where(s2, [r["w"] for r in rows], 0.0)
        cw2 = np.where(s2, [r["cw"] for r in rows], 0.0)
        p2 = np.where(s2, [r["pp"] for r in rows], 0.0)
        c2 = np.where(s2, COST_P, 0.0)
        dd = (w2 + p2) - (w + pp)
        md, sd = dd.mean(), dd.std(ddof=1) / math.sqrt(N)
        v = "★差がある" if abs(md) > z * sd else "⚠差は検出できない"
        print(f"{name:<20}{int((sel & drop).sum()):>8,}"
              f"{100*(w2.sum()+cw2.sum())/cw2.sum():>9.1f}%"
              f"{100*(p2.sum()+c2.sum())/c2.sum():>9.1f}%"
              f"{md:>+10.1f}円{f'[{md-z*sd:+.1f},{md+z*sd:+.1f}]':>21}{v:>16}")

    print("\n" + "=" * 96)
    print("★読み方（**事前登録のとおり**）")
    print("  ⚠**1レース平均の損失が減るのは当たり前**——**賭けを減らしただけ**。**ROIを見る**。")
    print("  ⚠⚠**(165)と同じデータの post-hoc な切り出し**。**有意でも運用に入れない**——")
    print("     ★**別の期間で再現するまで待つ**（判定基準25）。")


if __name__ == "__main__":
    main()
