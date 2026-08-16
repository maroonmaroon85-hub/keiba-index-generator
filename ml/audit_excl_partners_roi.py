"""(155) ★★★★運用の2つの設定を**ROIそのもの**で決める — 保留4・5への回答

★★なぜ要るか（ユーザー指示 2026-08-16「運用に手をつける場合はよりROIが高くなるなら修正して」）
　いま運用に載っている2つの設定は、**ROIで比べたことが一度も無い**:
　・**紐1 か 紐2 か** … (80)は**1レース期待損失**で決めた（14.8円 vs 約2倍）。
　　 ⚠**期待損失が半分なのは「賭け金が半分だから」でもある**。**ROIが良いとは限らない**。
　・**除外率** … (117)(139)は**D**で決めた。**ROIでは比べていない**。
　　 ⚠**しかも(154)で「40%の設定が実効約60%」と分かった**。**どちらが良いのかも未測定**。

★★★事前登録（**測る前に書いている**）
　1. **リークを避ける**: **(55)(62)(80)と同じ「前30%で学習・後70%で検証」**。**モデル順で買う**。
　2. **紐は 1 と 2 の2通りだけ**。**後から3は増やさない**。
　3. **除外率は 0/10/20/30/40/50/60/70%**。**後から増やさない**。
　　 ★**閾値は「検証側の waku_score の分位」で作る**ので、**表示した率がそのまま実効率**になる
　　 （＝**(154)で見つかったずれを排除した、理想の状態での比較**）。
　4. ★★**判定は3つ並べて見る**（**ROIだけで決めない**）:
　　 **① ROI**（ユーザーの基準）／**② 1R期待損失（円）**（(80)の基準）／
　　 **③ 判定100レースあたりの総収支（円）**（★**買う本数の違いを吸収する唯一の量**）
　　 ⚠**除外を増やせばROIは上がりやすい**（悪いレースを外すので）。**だが買う本数も減る**。
　　 **③が「実際に手元に残る金」**なので、**③を主に見る**。**先に決めておく**。
　5. ★★**陽性対照（判定基準32）**: **除外0%・紐1 の既知値は ROI 85.2% / 1R期待損失 14.8円**。
　　 **±2pt で再現しなければ以下を読まない**。
　6. **プラセボ**: 除外を**無作為**に同数行う。**「絞ったら動いた」の対照**（判定基準28）。
　7. ★**予想**: **当てにしてよい予想は持っていない**。
　　 **恒等式から言えることだけ**: **③は除外率とともに単調に0へ近づく**（買わなければ損もしない）。
　　 → ⚠**だから「③が良い＝除外を増やせ」と読んではいけない**。**買わないのが③では最強になる**。
　　 ★**正しい読み方: ①ROIが100%を超えないなら、③は必ず「買わない」が最良**。
　　 　**そのうえで「買うと決めた場合にどれが最良か」を①と②で見る**。**この順序を先に決めておく**。

実行: python3 ml/audit_excl_partners_roi.py
"""
import math
import sys

import numpy as np

sys.path.insert(0, "ml")
import features as F
from audit_crosspool import load_races, payoff, zq
from audit_crosspool2 import realized
from train_prod import CAPACITY, add_odds_features, fit_seeds
from waku_umatan import bracket_probs, waku_of, waku_score, wakuren_buy

RATES = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]
KNOWN_ROI, KNOWN_YEN, TOL = 85.2, 14.8, 2.0


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
    print(f"(155) 紐の数と除外率を**ROIそのもの**で決める"
          f"（学習 {tr.sum():,} / 検証 {te.sum():,}・分割 {cut.date()}）")
    print("★経路: 買い目は **MLモデル**の降順。**除外の閾値は検証側の分位**なので実効率＝表示率。")
    ms = fit_seeds(fx[tr], y[tr], 3, PAR)
    p_ml = np.mean([m.predict_proba(fx[te])[:, 1] for m in ms], axis=0)
    sub = d.loc[te, ["raceid", "umaban"]].copy()
    sub["p"] = p_ml

    races = {r["rid"]: r for r in load_races()}
    rows = []                      # (score, {紐: 払戻合計}, {紐: コスト})
    for rid, g in sub.groupby("raceid"):
        r = races.get(str(rid))
        if r is None or not r.get("wakuren"):
            continue
        rl = realized(r)
        if rl is None:
            continue
        a, b, _ = rl
        n, nums = r["n"], [u for u, _, _ in r["horses"]]
        if a not in nums or b not in nums:
            continue
        gg = g.sort_values("p", ascending=False)
        order = [int(u) for u in gg["umaban"].tolist()]
        pv = gg["p"].to_numpy(float)
        if len(order) < 3:
            continue
        key = tuple(sorted((waku_of(a, n), waku_of(b, n))))
        v = payoff(r, "枠連(人気順)", [key[0], key[1]])
        if not v or v <= 0:
            continue
        q = pv / pv.sum()
        bp = bracket_probs(order, q, n)
        sc = float(waku_score(wakuren_buy(order, n, 1), bp))
        got, cost = {}, {}
        for k in (1, 2):
            pairs = wakuren_buy(order, n, k)
            cost[k] = 100.0 * len(pairs)
            got[k] = 100.0 * v / 100.0 if key in pairs else 0.0
            got[k] = v if key in pairs else 0.0
        rows.append((sc, got, cost))

    if not rows:
        sys.exit("突き合わせできたレースが無い")
    sc = np.array([x[0] for x in rows])
    z = zq(0.01)
    rng = np.random.default_rng(0)
    print(f"　突き合わせ {len(rows):,}レース\n")
    print(f"{'紐':>3}{'除外率':>7}{'買うR':>8}{'ROI':>8}{'99%CI':>15}"
          f"{'②1R期待損失':>12}{'③判定100Rの収支':>16}{'プラセボROI':>11}")
    base = None
    for k in (1, 2):
        ret = np.array([x[1][k] for x in rows])
        cst = np.array([x[2][k] for x in rows])
        for p in RATES:
            th = np.quantile(sc, p) if p > 0 else -np.inf
            m = sc >= th
            if m.sum() < 30:
                continue
            pr = ret[m] - cst[m]
            roi = 100.0 * ret[m].sum() / cst[m].sum()
            se = pr.std(ddof=1) / math.sqrt(m.sum())
            mc = cst[m].mean()
            lo, hi = 100 * (1 + (pr.mean() - z * se) / mc), 100 * (1 + (pr.mean() + z * se) / mc)
            per100 = 100.0 * pr.sum() / len(rows)
            # プラセボ: 無作為に同数だけ残す（判定基準28）
            pl = []
            for _ in range(20):
                idx = rng.choice(len(rows), m.sum(), replace=False)
                pl.append(100.0 * ret[idx].sum() / cst[idx].sum())
            if k == 1 and p == 0.0:
                base = roi
            print(f"{k:>3}{int(p*100):>6}%{m.sum():>8,}{roi:>7.1f}%"
                  f"{'[' + format(lo, '.1f') + ',' + format(hi, '.1f') + ']':>15}"
                  f"{-pr.mean():>11.1f}円{per100:>15.0f}円{np.mean(pl):>10.1f}%")
        print()

    ok = base is not None and abs(base - KNOWN_ROI) <= TOL
    print(f"★陽性対照: 紐1・除外0% の既知値 ROI {KNOWN_ROI}% → 再現 {base:.1f}%"
          f"　差 {base-KNOWN_ROI:+.1f}pt　→ **{'★立った' if ok else '⚠立っていない'}**")
    if not ok:
        print("⚠**立っていない。上の表を読まないこと**（判定基準32）。")

    print("\n" + "=" * 92)
    print("★読み方（事前登録のとおり。**後から足していない**）")
    print("  ⚠**③は除外率とともに必ず0へ近づく**（買わなければ損もしない）。")
    print("    → ★**ROIが100%を超えないなら、③の最良は常に「買わない」**。**そう読む**。")
    print("  ★**「買うと決めた場合にどれが最良か」を①ROIと②期待損失で見る**。")
    print("  ・**プラセボ（無作為に同数残す）と差が無ければ、除外は何もしていない**（判定基準28）。")


if __name__ == "__main__":
    main()
