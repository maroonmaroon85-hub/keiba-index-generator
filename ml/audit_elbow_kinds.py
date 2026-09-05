"""(172) ★★ユーザー案「シェアがガクッと落ちる位置まで紐にする」を**4券種**で測る

★★経緯（2026-09-05・ユーザー）
　1.「軸をモデルと市場の差が最大の馬にして三連複で広く流す」→ **(171)で陰性**。
　2.「4までに絞らなくていい」→ **紐の梯子を全頭まで伸ばした → 広げるほど単調に悪化**（63.5%まで）。
　3. ★★「**各レースのシェアの分布によって来る確率が異なる。ガクッと下がる位置まではある程度くる**」
　　 → **これは固定幅ではなくレースごとに切る位置を変える規則**。**探索で 同じ6点で +6.7pt** が出た。
　4. ★「**三連単・三連複・馬単・馬連でやった方がいい**」→ **本スクリプト**。

⚠⚠**探索で見えた +6.7pt は、まだ何の証拠でもない**（判定基準25/41）。
　**梯子を何本も引いて良いマスを見ている状態**。**ここで事前登録して測り直す**。

★★★事前登録（**測る前に書いている**）
　**買い方は「軸1頭流し」**（軸が3着以内に入らなければ必ず外れる形）:
　　**馬連** 軸+紐1頭 = n点／**馬単** 軸を1着固定 = n点／
　　**三連複** 軸+紐2頭 = C(n,2)点／**三連単** 軸1着固定+紐2頭の順列 = n(n-1)点
　**軸2通り**: 確率1位 ／ ★差最大（モデルのシェア − 市場シェア が最大）
　**紐の切り方3通り**: **固定3頭** ／ **固定4頭** ／ ★**ガクッと r=0.70**（前比が0.70を下回る位置で切る・3〜8頭）

　★**主判定は4つだけ**（券種ごと1つ）: **軸=差最大 で「ガクッと r=0.70」vs「固定4頭」の対応差（円）**。
　　**Bonferroni α=0.01/4**。**他の欄は記述**。
　⚠★**同時に平均点数を必ず出す**——**点数が違えば比較にならない**。
　　**(77): ROI<100%では点数を増やすのは常に悪手**。**点数差が10%を超えたら「比較不可」と書く**。
　⚠**ゲート（判定基準32）**: **三連複BOX4が84.1%を±2.5ptで再現しなければ何も読まない**。
　⚠**採用条件（判定基準39/40/41）**: **有意なだけでは足りない**。
　　**4券種のうち複数で同じ向きに出ること**。**1券種だけ跳ねるのは偶然でも起きる**。

★**予想は持たない**。⚠**ただし逆風は明示する**: **払戻率は 馬連77.5 / 馬単75.0 / 三連複75.0 / 三連単72.5%**。
　**三連単がいちばん不利**。**(77)で「広げるほど悪い」も分かっている**。

★★★実測（2026-09-05・26,417レース・ゲート: 三連複BOX4 が既知84.1%を再現 → 立った）
**主判定（軸=差最大・elbow r=0.70 − 固定4頭の対応差）: 4券種すべて通らない**
| 券種 | 点数(elbow/fix4) | 1R損益差 | 99%CI(Bonf) | |
|---|---|---|---|---|
| 馬連 | 3.2/3.0 | +7.5円 | [−9.6,+24.6] | ⚠検出できない |
| 馬単 | 3.2/3.0 | +1.7円 | [−23.2,+26.6] | ⚠検出できない |
| 三連複 | 5.1/3.0 | −42.5円 | [−88.4,+3.4] | ⚠点数差>10%で比較不可 |
| 三連単 | 10.2/6.0 | −128.8円 | [−279.7,+22.1] | ⚠点数差>10%で比較不可 |
★**探索で見えた「同じ6点で+6.7pt」は再現しなかった**。**判定基準25の典型例**。

⚠⚠**表の別の場所に大きいマスが出たので、(77)と同じ検算をした**:
　**三連単×★差最大×固定3頭 = 92.0%**（払戻率72.5%に対し**+19.5pt**・2.0点・−16.1円/R）。
　**軸を確率1位→差最大に変えると 74.4% → 92.0%（+17.6pt）**。**三連単だけ特異**。
★★**検算の結果、本物ではなかった**（26,417レース）:
| | |
|---|---|
| 的中 | **254本＝0.96%**（**104レースに1回**）／平均配当19,129円／軸の平均オッズ**18.6倍** |
| 裾の依存 | **上位3本で全払戻の10.7%**・上位10本で23.6%・最高配当264,560円 |
| 除外すると | **上位3本を除くと 92.0% → 82.1%（−9.9pt）**／最高配当1本を除くと87.0% |
| 年別 | **34.7% 〜 147.5%** で暴れる |
| 時間分割 | ★**前半(≤2021) 100.2% / 後半(>2021) 83.0%** |
| CI | 95%CI **[73.4, 110.6]**（幅37pt） |
→ ★**(77)の馬連BOX4（100.9%→前半143.9%/後半75.2%・最高配当1本が13.1%）とまったく同じ壊れ方**。
→ ⚠**(88)の再現でもある**——**軸の平均オッズ18.6倍＝穴側は分散を増やすだけ**。

★**一貫して出た2つ**: **点数を増やすほど悪化**（(77)）／**穴軸は分散を増やすだけ**（(88)）。

実行: python3 ml/audit_elbow_kinds.py
"""
import math
import sys
from itertools import combinations, permutations

import numpy as np

sys.path.insert(0, "ml")
import features as F
from audit_crosspool import load_races, payoff, zq
from audit_crosspool2 import realized
from train_prod import CAPACITY, add_odds_features, fit_seeds

COST = 100.0
KNOWN_BOX4, TOL, NCMP = 84.1, 2.5, 4
KINDS = [("馬連", 0.775), ("馬単", 0.750), ("三連複", 0.750), ("三連単", 0.725)]
CUTS = [("固定3頭", "fix3"), ("固定4頭", "fix4"), ("★ガクッと r=0.70", "elbow70")]
AXES = [("確率1位", "prob"), ("★差最大", "gap")]


def width(sh, mode):
    if mode == "fix3":
        return 3
    if mode == "fix4":
        return 4
    n = 1
    for i in range(1, len(sh)):
        if sh[i] < 0.70 * sh[i - 1]:
            break
        n += 1
    return max(3, min(8, n))


def combos_of(kind, ax, partners):
    if kind == "馬連":
        return [sorted((ax, p)) for p in partners]
    if kind == "馬単":
        return [[ax, p] for p in partners]
    if kind == "三連複":
        return [sorted((ax,) + pr) for pr in combinations(partners, 2)]
    return [[ax, a, b] for a, b in permutations(partners, 2)]


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
    print("(172) ★「シェアがガクッと落ちる位置まで」を4券種で測る（軸1頭流し）")
    print(f"　学習 {tr.sum():,} / 検証 {te.sum():,}・分割 {cut.date()}")
    print("⚠**探索で見えた +6.7pt はまだ何の証拠でもない**。**ここで事前登録して測り直す**\n")
    ms = fit_seeds(fx[tr], y[tr], 3, PAR)
    p = np.mean([m.predict_proba(fx[te])[:, 1] for m in ms], axis=0)
    sub = d.loc[te, ["raceid", "umaban", "odds"]].copy()
    sub["p"] = p
    races = {r["rid"]: r for r in load_races()}

    rows = []
    for rid, g in sub.groupby("raceid"):
        r = races.get(str(rid))
        if r is None:
            continue
        rl = realized(r)
        if rl is None:
            continue
        a, b, c = rl
        nums = [u for u, _, _ in r["horses"]]
        if a not in nums or b not in nums or c not in nums:
            continue
        gg = g.sort_values("p", ascending=False, kind="mergesort")
        order = np.array([int(u) for u in gg["umaban"]])
        pv = gg["p"].to_numpy(float)
        od = gg["odds"].to_numpy(float)
        if len(order) < 8 or not np.isfinite(od).all() or (od <= 0).any():
            continue
        pay = {}
        for kind, _ in KINDS:
            key = {"馬連": sorted((a, b)), "馬単": [a, b],
                   "三連複": sorted((a, b, c)), "三連単": [a, b, c]}[kind]
            v = payoff(r, kind, key)
            pay[kind] = (v if v else 0.0, tuple(key))
        share = pv / pv.sum()
        mk = (1.0 / od) / (1.0 / od).sum()
        rows.append(dict(order=order, share=share, gap=share - mk, pay=pay))

    N = len(rows)
    print(f"★突き合わせ {N:,}レース\n")
    z = zq(0.01 / NCMP)

    # ゲート: 三連複BOX4
    prof = np.zeros(N); cost = np.zeros(N)
    for i, rec in enumerate(rows):
        top4 = sorted(int(u) for u in rec["order"][:4])
        cb = [list(x) for x in combinations(top4, 3)]
        cost[i] = COST * len(cb)
        v, key = rec["pay"]["三連複"]
        prof[i] = (v if list(key) in cb else 0.0) - cost[i]
    g0 = 100.0 * (prof.sum() + cost.sum()) / cost.sum()
    ok = abs(g0 - KNOWN_BOX4) <= TOL
    print(f"⚠ゲート: 三連複BOX4 **{g0:.1f}%** vs 既知 {KNOWN_BOX4}%　差 {g0-KNOWN_BOX4:+.1f}pt"
          f" → **{'★立った' if ok else '⚠⚠落ちた'}**")
    if not ok:
        print("⚠⚠落ちた。読まない。"); return

    def run(kind, axmode, cutmode):
        pr = np.zeros(N); co = np.zeros(N); ns = np.zeros(N)
        for i, rec in enumerate(rows):
            o = rec["order"]
            ax = int(o[0]) if axmode == "prob" else int(o[int(np.argmax(rec["gap"]))])
            k = width(rec["share"], cutmode)
            pool = [int(u) for u in o[:k]]
            if ax not in pool:
                pool = [ax] + pool[:-1]
            partners = [u for u in pool if u != ax]
            cb = combos_of(kind, ax, partners)
            v, key = rec["pay"][kind]
            co[i] = COST * len(cb); ns[i] = len(cb)
            pr[i] = (v if list(key) in cb else 0.0) - co[i]
        return pr, co, ns

    store = {}
    print(f"\n{'券種':<7}{'払戻率':>7}{'軸':<10}{'切り方':<18}{'平均点数':>9}{'ROI':>9}{'1R損益':>11}")
    for kind, R in KINDS:
        for albl, axmode in AXES:
            for clbl, cutmode in CUTS:
                pr, co, ns = run(kind, axmode, cutmode)
                roi = 100.0 * (pr.sum() + co.sum()) / co.sum()
                store[(kind, axmode, cutmode)] = (pr, co, ns)
                print(f"{kind:<7}{100*R:>6.1f}%{albl:<10}{clbl:<18}{ns.mean():>9.1f}"
                      f"{roi:>8.1f}%{pr.mean():>+10.1f}円")
        print()

    print("■ ★★主判定: **軸=差最大** で「ガクッと r=0.70」−「固定4頭」の対応差")
    print(f"{'券種':<8}{'点数(elbow/fix4)':>18}{'1R損益差':>11}{'99%CI(Bonf)':>21}{'判定':>16}")
    for kind, R in KINDS:
        pe, ce, ne = store[(kind, "gap", "elbow70")]
        pf, cf, nf = store[(kind, "gap", "fix4")]
        dd = pe - pf
        md, sd = dd.mean(), dd.std(ddof=1) / math.sqrt(N)
        gapn = abs(ne.mean() - nf.mean()) / nf.mean()
        v = ("⚠点数差>10%で比較不可" if gapn > 0.10
             else ("★差がある" if abs(md) > z * sd else "⚠差は検出できない"))
        print(f"{kind:<8}{f'{ne.mean():.1f} / {nf.mean():.1f}':>18}{md:>+10.1f}円"
              f"{f'[{md-z*sd:+.1f},{md+z*sd:+.1f}]':>21}{v:>16}")

    print("\n" + "=" * 96)
    print("★読み方（**事前登録のとおり**）")
    print("  ⚠**点数が違えば比較にならない**。**点数差>10%は「比較不可」**。")
    print("  ★**採用には4券種のうち複数で同じ向きが要る**。**1券種だけ跳ねるのは偶然でも起きる**。")
    print("  ⚠**どれが良くても100%には届かない見込み**。**払戻率が72.5〜77.5%**。")


if __name__ == "__main__":
    main()
