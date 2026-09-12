"""(165) ★★「未勝利戦はデータが少ないからROIが低いのでは」を測る（ユーザーの問い・2026-08-29）

★★問いの機構が鋭い——**「未勝利だから」ではなく「モデルが値を出せない馬が多いから」**。
　実例: 2026-08-29 札幌3R は**11頭中7頭が過去走なし**で、**市場シェアの15.0%にモデルが値を持たない**。
　→ ★**測るべき変数は「クラス」だけでなく「モデルが見えていない市場シェア」そのもの**。

⚠**先行結果は逆向き**——`audit_premise.py` は「**候補馬が実頭数より少ないレースほどROIが高い**」
　と報告している。`audit_coverage2.py` はその理由を (a)初出走＝正当 / (b)アーカイブ欠損＝不当 に分けた。
　★**だから「少ないほど悪い」という素朴な予想は、既存の観測と食い違う**。**予想を持たずに測る**。

測るもの（**事前登録。この3つだけ。後から増やさない**）
　★①**クラス別ROI**（新馬0 / 未勝利1 / 1勝2 / 2勝3 / 3勝4 / OP以上5+）
　★★②**「モデルが値を出せない馬に乗っている市場シェア」の十分位別ROI** ← **ユーザーの機構そのもの**
　　（`n_prior=0` などで候補から抜けた馬の単勝オッズ由来シェアの合計）
　★③**2歳戦かどうか**（クラスと交絡するので分けて出す）

判定（**先に書く**）
　⚠**ゲート（判定基準32）**: **除外0%の全体ROIが既知84.5%を±2.5ptで再現しなければ何も読まない**。
　★**主判定は②**。**十分位に単調性があり、上位と下位のROI差のCIが0を跨がなければ「効いている」**。
　　**Bonferroni α=0.01/3**。
　⚠**「効いている」と出ても、それだけでは運用に入れない**——**(112)の裾と同じ信号かを確かめるまでは**
　　**新しい選択変数とは呼ばない**（(119)(134)で2回、同じ信号の言い換えだった）。
　★**ユーザーの基準は「よりROIが高くなるなら修正」**。**判定基準8に従い円でも出す**。

⚠**確定オッズではなく検証時点のデータで測る**（(155)(164)と同じ土俵）。

★★実測（2026-08-29・ゲート: 除外0%の全体 85.3% vs 既知84.5%＝+0.8pt → 立った）
| 切り口 | ROI | 99%CI(Bonf) |
|---|---|---|
| ★未勝利 | **83.8%** | [77.7,90.0] |
| 1勝 | 89.7% / 2勝 88.7% / 3勝 98.2% / OP以上 86.3% | |
| （全体） | 86.6% | |
| ★2歳戦 | **80.4%** | [70.2,90.5] |
| 3歳以上 | 87.6% | [82.7,92.5] |
| 見えないシェア 2%超 | **78.5%** | [64.8,92.1] |
| 〃 0-2% | 87.5% | [82.8,92.2] |
★**未勝利 −2.8pt / 2歳戦 −6.2pt / 見えないシェア2%超 −9.0pt と3つとも同じ向き**
　（**大部分が同じレース群**）。⚠**だがCIはすべて重なり、どれ一つ確立していない**。
★**上位20%(見えない側) − 下位20%(見える側) の1R損益差 −1.1円・99%CI[−7.7,+5.5]＝検出できない**。

⚠⚠**②の設計は失敗した（記録として残す）**——**「見えないシェア」は大半のレースで0**なので
　**十分位が潰れ、D1〜D8が空になった**。**実質「0-2% vs 2%超」の2群比較にしかならず、
　事前登録した単調性の判定ができなかった**。
　→ ★**教訓: 十分位で切る前に、その変数の分布（ゼロ点の割合）を先に見ること**。
　　 **ゼロが過半を占める変数は十分位では切れない**。**閾値を分布から決めるか、二値にする**。

★**次の一手（未実施）**: **2歳戦を外すと 86.6% → 87.6%（+1.0pt・対象は14.6%減）**。
　⚠**これが本物かは「対応のある差」で測り直す必要がある**（判定基準35）。**まだやっていない**。
　★**ユーザーの基準（ROIが上がるなら修正）に照らすと、測る価値のある一手**。

実行: python3 ml/audit_class_roi.py
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

EXCL = 0.40                 # 運用の設定
KNOWN, TOL = 84.5, 2.5
NCMP = 3
CLASSES = [(0, 0, "新馬"), (1, 1, "★未勝利"), (2, 2, "1勝"), (3, 3, "2勝"),
           (4, 4, "3勝"), (5, 9, "OP以上")]


def block(name, prof, cost, z):
    n = len(prof)
    if n < 30 or cost.sum() <= 0:
        return None
    se = prof.std(ddof=1) / math.sqrt(n) * n / cost.sum() * 100.0
    roi = 100.0 * (prof.sum() + cost.sum()) / cost.sum()
    return roi, n, se, prof.mean()


def show(name, prof, cost, z):
    r = block(name, prof, cost, z)
    if r is None:
        print(f"{name:>14}   （30レース未満）")
        return
    roi, n, se, mean = r
    print(f"{name:>14}{n:>9,}{roi:>9.1f}%"
          f"{f'[{roi-z*se:.1f},{roi+z*se:.1f}]':>19}{mean:>+11.1f}円")


def main():
    MODEL_DIR, PAR = CAPACITY["l2"]
    d0 = F.to_model(F.load_files())
    f = F.build_features(d0)
    keep = (f["n_prior"] >= 1) & d0["odds"].notna() & (d0["odds"] > 0)
    # ★落ちた馬の市場シェアを測るため、**落とす前**のオッズをレース単位で保持する
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
    print("(165) ★「未勝利戦はROIが低いのでは」を測る（枠連・紐1・除外40%）")
    print(f"　学習 {tr.sum():,} / 検証 {te.sum():,}・分割 {cut.date()}")
    ms = fit_seeds(fx[tr], y[tr], 3, PAR)
    p_ml = np.mean([m.predict_proba(fx[te])[:, 1] for m in ms], axis=0)
    sub = d.loc[te, ["raceid", "umaban", "raceclass", "age"]].copy()
    sub["p"] = p_ml

    races = {r["rid"]: r for r in load_races()}
    rows = []          # (score, 損益, コスト, クラス, 見えないシェア, 2歳か)
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
        sc = float(waku_score(wakuren_buy(order, n, 2), bp))   # 判定基準34: 除外は紐2で
        pairs = wakuren_buy(order, n, 1)
        c = 100.0 * len(pairs)
        gain = v if key in pairs else 0.0
        tot = allsum.get(rid, 0.0)
        blind = (dropsum.get(rid, 0.0) / tot) if tot > 0 else 0.0
        rows.append((sc, gain - c, c, int(gg["raceclass"].iloc[0]),
                     blind, int(gg["age"].min()) <= 2))

    if not rows:
        sys.exit("突き合わせできたレースが無い")
    A = np.array([[x[1], x[2], x[4]] for x in rows], float)
    sc = np.array([x[0] for x in rows])
    cls = np.array([x[3] for x in rows])
    two = np.array([x[5] for x in rows], bool)
    z = zq(0.01 / NCMP)
    print(f"　突き合わせ {len(rows):,}レース\n")

    g0 = 100.0 * (A[:, 0].sum() + A[:, 1].sum()) / A[:, 1].sum()
    ok = abs(g0 - KNOWN) <= TOL
    print(f"⚠ゲート（判定基準32）: 除外0%の全体 **{g0:.1f}%** vs 既知 {KNOWN}%"
          f"　差 {g0-KNOWN:+.1f}pt → **{'★立った' if ok else '⚠⚠落ちた'}**")
    if not ok:
        print("⚠⚠**落ちた。以下を読まない**。")
        return

    thr = np.quantile(sc, EXCL)
    m0 = sc >= thr
    print(f"\n■ ★①クラス別（除外{EXCL:.0%}後・{int(m0.sum()):,}レース）")
    print(f"{'':>14}{'レース':>9}{'ROI':>10}{'99%CI(Bonf)':>19}{'1R期待損益':>11}")
    for lo, hi, nm in CLASSES:
        m = m0 & (cls >= lo) & (cls <= hi)
        show(nm, A[m, 0], A[m, 1], z)
    print(f"{'（全体）':>14}{int(m0.sum()):>9,}"
          f"{100*(A[m0,0].sum()+A[m0,1].sum())/A[m0,1].sum():>9.1f}%")

    print("\n■ ★★②モデルが値を出せない馬の市場シェア（十分位・除外後）← **本命**")
    bl = A[m0, 2]
    ed = np.quantile(bl, np.linspace(0, 1, 11))
    ed[0], ed[-1] = -1.0, 2.0
    print(f"{'':>14}{'レース':>9}{'ROI':>10}{'99%CI(Bonf)':>19}{'1R期待損益':>11}")
    idx = np.where(m0)[0]
    dec = []
    for i in range(10):
        mm = idx[(bl >= ed[i]) & (bl < ed[i + 1])]
        r = block("", A[mm, 0], A[mm, 1], z)
        dec.append(r[0] if r else float("nan"))
        show(f"D{i+1} {100*ed[i]:.0f}-{100*ed[i+1]:.0f}%", A[mm, 0], A[mm, 1], z)
    lo_m = idx[bl <= np.quantile(bl, 0.2)]
    hi_m = idx[bl >= np.quantile(bl, 0.8)]
    dprof = A[hi_m, 0].mean() - A[lo_m, 0].mean()
    sd = math.hypot(A[hi_m, 0].std(ddof=1) / math.sqrt(len(hi_m)),
                    A[lo_m, 0].std(ddof=1) / math.sqrt(len(lo_m)))
    print(f"\n　★上位20%(見えない側) − 下位20%(見える側) の1R損益差 **{dprof:+.1f}円**"
          f"　99%CI [{dprof-z*sd:+.1f},{dprof+z*sd:+.1f}]")
    print(f"　→ **{'⚠差がある' if abs(dprof) > z*sd else '★差は検出できない'}**")
    good = sum(1 for i in range(9)
               if np.isfinite(dec[i]) and np.isfinite(dec[i+1]) and dec[i+1] > dec[i])
    print(f"　単調に上がった回数 {good}/9")

    print("\n■ ★③2歳戦か（クラスと交絡するので分けて出す）")
    print(f"{'':>14}{'レース':>9}{'ROI':>10}{'99%CI(Bonf)':>19}{'1R期待損益':>11}")
    show("2歳戦", A[m0 & two, 0], A[m0 & two, 1], z)
    show("3歳以上", A[m0 & ~two, 0], A[m0 & ~two, 1], z)

    print("\n" + "=" * 88)
    print("★読み方（**事前登録のとおり**）")
    print("  ⚠**先行結果(`audit_premise.py`)は「候補が少ないレースほどROIが高い」＝逆向き**。")
    print("  ★**②で差が出ても、(112)の裾と同じ信号かを確かめるまで新しい選択変数と呼ばない**")
    print("     （(119)(134)で2回、同じ信号の言い換えだった）。")
    print("  ⚠**クラス別は多重比較が6本ある**。**1本だけ低くても、それだけでは外さない**。")


if __name__ == "__main__":
    main()
