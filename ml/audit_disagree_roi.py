"""(156) ★★★「モデルと市場が食い違うレース」だけ買ったらROIはどうなるか（ユーザー発案 2026-08-16）

★★問いの出どころ
　2026-08-16の新潟6R・札幌11Rで、**モデルが市場2番人気を4番手に落とす**などの食い違いが出た。
　→ ユーザー「**市場人気とズレがあるレースを切り取ってROIを出すとどんなもん？**」

★★先に書いておく前例（判定基準29・15）
　**この筋は初めてではない**:
　・**(101)** レース選択の上界 → **裾は実在するが事前にはまったく予測できない**
　・**(112)** 甘い軸（複勝の期待払戻）で選ぶ → **本プロジェクト最大の利得**（ただし(140)で半減）
　・**(130)** プール間の不整合を選択変数に → **陰性**
　・**(134)** 「荒れ度」を選択変数に → **効くが向きが逆。しかも(112)と同じ信号だった**
　→ ⚠**今回も「(112)と同じ信号を別の名前で見ているだけ」の可能性が高い**。**それを必ず確かめる**。

★★★事前登録（**測る前に書いている**）
　1. **切り方は2つだけ**。**後から増やさない**:
　　 **A 軸馬の乖離**: `モデルのレース内シェア − 市場の含意確率`（軸＝モデル1位馬）の**十分位**
　　 **B 買い目の一致**: **モデルの買う枠組が市場順の枠組と同じか否か**（2区分）
　2. **買い方は現行の本命に固定**: **モデル1位の枠 × モデル2位の枠・1点100円**。**除外はしない**
　　 （除外を掛けると層が入れ子になって判定基準13を踏む。**選択変数の効果だけを見る**）。
　3. ★★**判定は「層間の差」の2標本検定**（**判定基準13後半**）。
　　 ⚠**層のCIが100%を跨ぐかどうかでは判定しない**（(112)が(140)で崩れた形）。
　　 **最上位十分位 vs 残り** の差にCIを付ける。**Bも同じ形**。
　4. **多重比較**: 十分位10 + 2区分 = **12**。**Bonferroni（α=0.01/12）**。
　5. ★**プラセボ（判定基準28）**: **同数を無作為に抜いた対照**を必ず並べる。
　　 **絞ると標本が減るだけでCIも点推定も動く**ので。
　6. ★★**(112)と同じ信号かを必ず確かめる**（判定基準29・(134)の教訓）:
　　 **軸馬の期待払戻 E（＝(112)の選択変数）と、Aの乖離の順位相関 ρ** を出す。
　　 **|ρ| が大きければ「新しい変数ではない」と書く**。
　7. ★★**陽性対照（判定基準32）**: **全体（層を切らない）のROIは 85.3%**（(153b)/(155)で実測）。
　　 **±2pt で再現しなければ何も読まない**。
　8. ⚠**リークを避ける**: **(55)(62)(80)と同じ「前30%で学習・後70%で検証」**。
　9. ★**予想**: **当てにしてよい予想は持っていない**（類推はこの4日で5連敗）。
　　 **恒等式から言えることだけ**: **乖離が大きいほど買い目は市場順から離れる**ので、
　　 **配当は高く、的中率は低くなる**。**ROIの向きは決まらない**。

⚠★**この実験が陽性でも、すぐ運用に入れない**。**(112)が(140)で半減した前例がある**。
　**最低でも「2021-2026でも同じ向きか」を見てから**（(142)の教訓）。

★★★実行済みの結果（2026-08-16・検証25,722レース）**陰性。しかも(112)の焼き直しだった**
　★①陽性対照: 全体ROI **85.3%** vs 既知 85.3% → **差 0.0pt でぴたりと立った**。

■B 買い目が市場順と一致するか（**ユーザーの問いへの直接の答え**）
| 区分 | R数 | ROI | 99%CI | 的中率 | プラセボ |
|---|---|---|---|---|---|
| 一致する | 16,785 | **85.3%** | [81.7,89.0] | **21.3%** | 85.0% |
| ★食い違う | 8,937 | **85.2%** | [78.1,92.3] | **12.1%** | 85.5% |

**差 −0.14円/R 99%CI[−10.50,+10.22] ＝ 完全に差なし**。
★★**的中率は 21.3%→12.1% とほぼ半減するのに、ROIは0.1ptしか違わない**
　＝**配当が正確にその分だけ高い＝市場がきっちり値段を付けている**。

■A 軸馬の乖離の十分位: **78.1%〜91.1% で単調でない**。**プラセボは全層84〜86%で平坦**。
　2標本検定（Bonferroni α=0.01/12）:
　　最上位十分位 vs 残り **+4.19円/R 99%CI[−14.88,+23.26]** ＝**0をまたぐ**
　　最下位十分位 vs 残り **−4.95円/R 99%CI[−16.66,+6.77]** ＝**0をまたぐ**

★★★**事前登録6が効いた**: **乖離と(112)の選択変数E の順位相関 ρ = +0.864**。
　→ ★**新しい変数ではない**。**(112)「甘い軸」と同じ信号を別の名前で見ているだけ**。
　**(134)「荒れ度」がまったく同じ結末を辿った**（→**判定基準17の二重計上に直接かかる**）。

⚠★**2021-2026の最上位十分位が91.1%（1,476R）と出ているが、読まないこと**。
　**部分集合の部分集合**で、しかも**表を見てから目についた数字**。
　**上の2標本検定が0をまたぐ以上、運の範囲**（判定基準13・(112)が(140)で崩れた形）。

★**結論: 「市場とズレたレースだけ買う」は陰性**。**(101)(130)(134)に続いて4回目の陰性**。
　★**レース選択の変数は(112)の1本のまま**。

実行: python3 ml/audit_disagree_roi.py
"""
import math
import sys

import numpy as np

sys.path.insert(0, "ml")
import features as F
from audit_crosspool import load_races, payoff, probs, zq
from audit_crosspool2 import realized
from audit_fuku_lbs import top3_probs
from audit_lbs import build_matrix, fit_lambda
from train_prod import CAPACITY, add_odds_features, fit_seeds
from waku_umatan import waku_of

KNOWN_ROI, TOL, NCMP = 85.3, 2.0, 12


def two_sample(a, b, z):
    """a（層）と b（残り）の平均差と99%CI。**入れ子ではない2標本**（判定基準13後半）。"""
    d = a.mean() - b.mean()
    se = math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return d, d - z * se, d + z * se


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
    print(f"(156) モデルと市場の食い違いでレースを切る"
          f"（学習 {tr.sum():,} / 検証 {te.sum():,}・分割 {cut.date()}）")
    print("★経路: 買い目は **MLモデル** 1位×2位の枠・1点。**除外は掛けない**（層の入れ子を避ける）")
    ms = fit_seeds(fx[tr], y[tr], 3, PAR)
    p_ml = np.mean([m.predict_proba(fx[te])[:, 1] for m in ms], axis=0)
    sub = d.loc[te, ["raceid", "umaban"]].copy()
    sub["p"] = p_ml

    races = {r["rid"]: r for r in load_races()}
    P, i1, i2, i3, yrs = build_matrix(list(races.values()), 2015)
    ok3 = i3 >= 0
    l2 = fit_lambda(P, i1, i2)
    l3 = fit_lambda(P[ok3], i1[ok3], i2[ok3], stage3=True, ic=i3[ok3])

    gap, prof, same, expay, yr = [], [], [], [], []
    for rid, g in sub.groupby("raceid"):
        r = races.get(str(rid))
        if r is None or not r.get("wakuren"):
            continue
        rl = realized(r)
        if rl is None:
            continue
        a, b, _ = rl
        n, hs = r["n"], r["horses"]
        nums = [u for u, _, _ in hs]
        if a not in nums or b not in nums:
            continue
        gg = g.sort_values("p", ascending=False)
        order = [int(u) for u in gg["umaban"].tolist()]
        pv = gg["p"].to_numpy(float)
        if len(order) < 2 or pv.sum() <= 0:
            continue
        key = tuple(sorted((waku_of(a, n), waku_of(b, n))))
        v = payoff(r, "枠連(人気順)", [key[0], key[1]])
        if not v or v <= 0:
            continue
        share = pv / pv.sum()
        pm = probs(hs)
        mkt = {u: float(x) for u, x in zip(nums, pm)}
        # A: 軸馬（モデル1位）の乖離
        gap.append(float(share[0]) - mkt.get(order[0], 0.0))
        k_ml = tuple(sorted((waku_of(order[0], n), waku_of(order[1], n))))
        o2 = np.argsort(-pm)
        k_mk = tuple(sorted((waku_of(nums[o2[0]], n), waku_of(nums[o2[1]], n))))
        same.append(int(k_ml == k_mk))
        prof.append((v if k_ml == key else 0.0) - 100.0)
        # (112)の選択変数 E（軸馬の複勝の期待払戻）— 同じ信号かの確認用（事前登録6）
        q3 = np.asarray(top3_probs(pm, 1.0, l2, l3), float)
        j = nums.index(order[0])
        expay.append(0.8 / max(q3[j], 1e-6) * 100.0)
        yr.append(r["year"])

    gap = np.array(gap); prof = np.array(prof); same = np.array(same)
    expay = np.array(expay); yr = np.array(yr)
    z_all, z_bon = zq(0.01), zq(0.01 / NCMP)
    roi_all = 100 * (1 + prof.mean() / 100)
    ok = abs(roi_all - KNOWN_ROI) <= TOL
    print(f"　突き合わせ {len(prof):,}レース\n")
    print(f"■ ★①陽性対照: 全体ROI **{roi_all:.1f}%** vs 既知 {KNOWN_ROI}%"
          f"　差 {roi_all-KNOWN_ROI:+.1f}pt → **{'★立った' if ok else '⚠立っていない'}**")
    if not ok:
        print("　⚠**以下を読まないこと**（判定基準32）。")

    print(f"\n■ ★A 軸馬の乖離（モデルのシェア − 市場の含意確率）の十分位")
    print(f"{'十分位':>7}{'乖離の範囲':>18}{'R数':>8}{'ROI':>8}{'99%CI':>16}"
          f"{'的中率':>8}{'プラセボ':>9}")
    qs = np.quantile(gap, np.linspace(0, 1, 11))
    rng = np.random.default_rng(0)
    dec = []
    for i in range(10):
        lo, hi = qs[i], qs[i + 1]
        m = (gap >= lo) & (gap <= hi if i == 9 else gap < hi)
        if m.sum() < 30:
            continue
        p_ = prof[m]
        roi = 100 * (1 + p_.mean() / 100)
        se = p_.std(ddof=1) / math.sqrt(m.sum())
        pl = np.mean([100 * (1 + prof[rng.choice(len(prof), m.sum(), False)].mean() / 100)
                      for _ in range(20)])
        dec.append((i + 1, m, roi))
        print(f"{i+1:>7}{f'[{lo:+.3f},{hi:+.3f}]':>18}{m.sum():>8,}{roi:>7.1f}%"
              f"{'[' + format(roi-z_all*se, '.1f') + ',' + format(roi+z_all*se, '.1f') + ']':>16}"
              f"{(p_ > 0).mean():>8.1%}{pl:>8.1f}%")

    print(f"\n■ ★★層間の差の2標本検定（判定基準13後半・Bonferroni α=0.01/{NCMP}）")
    for lab, m in (("最上位十分位 vs 残り", dec[-1][1] if dec else None),
                   ("最下位十分位 vs 残り", dec[0][1] if dec else None)):
        if m is None:
            continue
        dd, lo, hi = two_sample(prof[m], prof[~m], z_bon)
        mark = "★有意" if lo > 0 or hi < 0 else "0をまたぐ"
        print(f"　{lab:<22} 差 {dd:+7.2f}円/R  99%CI[{lo:+.2f},{hi:+.2f}]  {mark}")

    print(f"\n■ ★B 買い目が市場順と一致するか")
    print(f"{'区分':>16}{'R数':>8}{'ROI':>8}{'99%CI':>16}{'的中率':>8}{'プラセボ':>9}")
    for lab, m in (("一致する", same == 1), ("★食い違う", same == 0)):
        p_ = prof[m]
        roi = 100 * (1 + p_.mean() / 100)
        se = p_.std(ddof=1) / math.sqrt(m.sum())
        pl = np.mean([100 * (1 + prof[rng.choice(len(prof), m.sum(), False)].mean() / 100)
                      for _ in range(20)])
        print(f"{lab:>16}{m.sum():>8,}{roi:>7.1f}%"
              f"{'[' + format(roi-z_all*se, '.1f') + ',' + format(roi+z_all*se, '.1f') + ']':>16}"
              f"{(p_ > 0).mean():>8.1%}{pl:>8.1f}%")
    dd, lo, hi = two_sample(prof[same == 0], prof[same == 1], z_bon)
    print(f"　差（食い違う − 一致する） {dd:+7.2f}円/R  99%CI[{lo:+.2f},{hi:+.2f}]  "
          f"{'★有意' if lo > 0 or hi < 0 else '0をまたぐ'}")

    print(f"\n■ ★★事前登録6: これは(112)と同じ信号か")
    ra = np.argsort(np.argsort(gap)).astype(float)
    rb = np.argsort(np.argsort(expay)).astype(float)
    rho = float(np.corrcoef(ra, rb)[0, 1])
    print(f"　乖離 と (112)の選択変数E の順位相関 ρ = **{rho:+.3f}**")
    print(f"　→ ★**|ρ|が大きければ「新しい変数ではない」**（(134)がこれで(112)と同じと判明した）")

    print(f"\n■ 時間分割（(142)の教訓）")
    for lab, m in (("2021-2026 全体", yr >= 2021),):
        print(f"　{lab}: ROI {100*(1+prof[m].mean()/100):.1f}%（{m.sum():,}R）")
        if dec:
            mm = m & dec[-1][1]
            if mm.sum() >= 30:
                print(f"　　うち最上位十分位: ROI {100*(1+prof[mm].mean()/100):.1f}%（{mm.sum():,}R）")

    print("\n" + "=" * 96)
    print("★読み方（事前登録のとおり。**後から足していない**）")
    print("  ・★**判定は「層間の差」の2標本検定**。**層のCIが100%を跨ぐかでは判定しない**（判定基準13）。")
    print("  ・**プラセボと差が無ければ、切ったことは何もしていない**（判定基準28）。")
    print("  ・★**ρが大きければ(112)の焼き直し**。**新しい利得として積み上げてはいけない**（判定基準17）。")


if __name__ == "__main__":
    main()
