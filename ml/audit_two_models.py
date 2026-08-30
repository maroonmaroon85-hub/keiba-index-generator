"""(169) ★★★ユーザー提案: **枠連用(top2)と三連複用(top3)の2つのモデルを持つ**

★★提案（2026-08-30・ユーザー）——**(168)で見つかった結合を、設計で解く**。
　**(168)は「1つのモデルをどちらの目標にするか」だった**ので、
　**どちらを選んでも片方が損をする**（top2にすると三連複が −5.2円/R 悪化）。
　★**2つ持てば結合が消える**: **枠連は top2、三連複は top3 のまま**。
　→ **枠連で +1.3円 得て、三連複は何も失わない**（(168)の実測値から）。**設計として正しい**。

★★★測る前に確かめること——**運用はモデルを3か所で使っており、目標の要求が同じではない**:
| 使い道 | 欲しい確率 |
|---|---|
| 枠連の軸・紐 | ★**上位2頭** |
| `waku_score` の除外判定 | ★**上位2頭**（式が `2·Pa·Pb`＝上位2頭の形） |
| **(112)の裾・甘い軸の E ≤ 86円** | ⚠**複勝の期待払戻なので top3** |
| 三連複BOX4 | **top3** |
→ ★**除外判定を top2 と top3 のどちらから作るかで結果が変わりうる**。**両方測る**。

★★事前登録（**この4アームだけ。後から増やさない**）
　**現行**: 買い目 top3 ／ 除外 top3 ／ 三連複 top3
　★**P1（提案）**: 買い目 **top2** ／ 除外 **top2** ／ 三連複 **top3**
　★**P2**: 買い目 **top2** ／ 除外 **top3** ／ 三連複 **top3**
　　（**(112)の選択は複勝の量なので、除外も top3 のままにする版**）
　**参考 単一top2**: すべて top2（＝(168)のB。**結合が残る版**）

判定
　⚠**ゲート（判定基準32）**: **現行が既知 85.3%（枠連）を ±2.5pt で再現しなければ読まない**。
　★**主判定は「合算の1レース期待損益」**（枠連100円＋三連複400円＝運用そのもの）。
　　**選択込みの対応差**（全レース・買わない日は損益0）。**判定基準8・35**。
　　**Bonferroni α=0.01/3**（現行に対する3比較）。
　★**枠連だけの数字も併記**（三連複を外す判断と結合しているため。→(164)）。
　⚠★**判定基準34**: **除外の閾値は各アームが使うモデルの分位で作り直す**。
　　**同じ絶対値を使い回さない**。**紐1に変えたときの事故を繰り返さない**。

⚠**予想は持たない**。**(168)の +1.3円 は 99%CI[−1.6,+3.2] で未確立のまま**。
　★**設計が正しいことと、効果が確立していることは別**。**両方を混ぜて報告しない**。

★★★実測（2026-08-30・ゲート: 現行の枠連 86.7% vs 既知85.3%＝+1.4pt → 立った）
| アーム | 枠連ROI | 三複ROI | 枠連1R | 三複1R | 合算1R |
|---|---|---|---|---|---|
| 現行 (3/3/3) | 86.7% | 86.4% | −8.0円 | −32.7円 | **−40.7円** |
| ★P1 (2/2/3) | **88.0%** | **85.2%** | −7.2円 | −35.5円 | −42.7円 |
| ★P2 (2/3/3) | 86.9% | 86.4% | −7.9円 | −32.7円 | **−40.5円** |
| 参考 単一top2 | 88.0% | 85.1% | −7.2円 | −35.9円 | −43.0円 |
主判定（選択込みの対応差）: **P1 −2.1円[−9.9,+5.7] / P2 +0.1円[−2.1,+2.3]** → **どれも未確立**。

★★★**予想外だったこと: 結合は「買い目の目標」ではなく「除外」に住んでいた**。
　**P1は三連複の目標を top3 に固定したのに 86.4%→85.2% に落ちた**——
　**`waku_score` を top2 モデルから作ると買うレースの集合が変わる**ため。
　→ ★**2つモデルを持っても、除外が共有されている限り結合は消えない**。

★★**分解できた**:
| | 枠連ROI | 寄与 |
|---|---|---|
| 現行（買い目top3・除外top3） | 86.7% | — |
| **P2**（買い目**top2**・除外top3） | 86.9% | **買い目の寄与 +0.2pt** |
| **P1**（買い目top2・除外**top2**） | 88.0% | ★**除外の寄与 +1.1pt** |
⚠⚠**(168)で見えた +1.3pt のうち +1.1pt は除外側から来ていた**。
　**「目標が賭けと合っていない」ことの効果は +0.2pt しかない**。
　★**私が『目標のずれ』と呼んでいたものの正体は、大部分が別のものだった**。

◇**次の設計（事前登録し直してから測る。今日は測らない）**:
　★**除外も券種ごとに分ける**——**枠連は top2 の `waku_score`、三連複は top3 の基準で別々に除外**。
　⚠**結果を見てからアームを足すのは最も危ない手順**（判定基準25）なので、**今日はやらない**。

実行: python3 ml/audit_two_models.py
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

EXCL = 0.40
KNOWN, TOL = 85.3, 2.5
NCMP = 3
COST_W, COST_P = 100.0, 400.0
# (名前, 買い目の目標, 除外の目標, 三連複の目標)
ARMS = [("現行 (3/3/3)", 3, 3, 3),
        ("★P1 (2/2/3)", 2, 2, 3),
        ("★P2 (2/3/3)", 2, 3, 3),
        ("参考 単一top2", 2, 2, 2)]


def main():
    MODEL_DIR, PAR = CAPACITY["l2"]
    d = F.to_model(F.load_files())
    f = F.build_features(d)
    keep = (f["n_prior"] >= 1) & d["odds"].notna() & (d["odds"] > 0)
    d, f = d[keep].reset_index(drop=True), f[keep].reset_index(drop=True)
    fin = d["finish"].to_numpy(float)
    fx, _ = F.encode_categoricals(f)
    fx = add_odds_features(fx, d["odds"].to_numpy(float), d["raceid"].to_numpy())
    cut = d["date"].quantile(0.3)
    tr, te = (d["date"] < cut).to_numpy(), (d["date"] >= cut).to_numpy()
    print("(169) ★2つのモデルを持つ（枠連=top2 / 三連複=top3）— ユーザー提案の検証")
    print(f"　学習 {tr.sum():,} / 検証 {te.sum():,}・分割 {cut.date()}")
    print("⚠**設計が正しいことと、効果が確立していることは別**。**混ぜて報告しない**\n")

    P = {}
    for k in (2, 3):
        y = (fin <= k).astype(int)
        print(f"── 目標 finish<={k}（陽性率 {y[tr].mean():.3f}）学習中…")
        ms = fit_seeds(fx[tr], y[tr], 3, PAR)
        P[k] = np.mean([m.predict_proba(fx[te])[:, 1] for m in ms], axis=0)

    races = {r["rid"]: r for r in load_races()}
    base = d.loc[te, ["raceid", "umaban"]].copy()
    # レース単位で、目標kごとの順序と確率をまとめる
    per = {}
    for k in (2, 3):
        sub = base.copy(); sub["p"] = P[k]
        per[k] = {str(rid): (list(map(int, g.sort_values("p", ascending=False)["umaban"])),
                             g.sort_values("p", ascending=False)["p"].to_numpy(float))
                  for rid, g in sub.groupby("raceid")}

    rows = []
    for rid in per[3]:
        r = races.get(rid)
        if r is None or not r.get("wakuren"):
            continue
        rl = realized(r)
        if rl is None:
            continue
        a, b, c = rl
        n, nums = r["n"], [u for u, _, _ in r["horses"]]
        if a not in nums or b not in nums or c not in nums:
            continue
        if len(per[3][rid][0]) < 4 or len(per[2][rid][0]) < 4:
            continue
        key = tuple(sorted((waku_of(a, n), waku_of(b, n))))
        vw = payoff(r, "枠連(人気順)", [key[0], key[1]])
        vp = payoff(r, "三連複", sorted([a, b, c]))
        if not vw or vw <= 0 or vp is None:
            continue
        rec = {"rid": rid}
        for k in (2, 3):
            order, pv = per[k][rid]
            q = pv / pv.sum()
            bp = bracket_probs(order, q, n)
            rec[f"sc{k}"] = float(waku_score(wakuren_buy(order, n, 2), bp))
            pairs = wakuren_buy(order, n, 1)
            rec[f"w{k}"] = (vw if key in pairs else 0.0) - COST_W * len(pairs)
            rec[f"cw{k}"] = COST_W * len(pairs)
            box = list(combinations(sorted(order[:4]), 3))
            rec[f"p{k}"] = (vp if tuple(sorted((a, b, c))) in box else 0.0) - COST_P
        rows.append(rec)

    N = len(rows)
    print(f"\n★突き合わせ {N:,}レース\n")
    z = zq(0.01 / NCMP)
    res = {}
    print(f"{'アーム':<16}{'買うR':>8}{'枠連ROI':>10}{'三複ROI':>10}"
          f"{'枠連 1R':>10}{'三複 1R':>10}{'合算 1R':>11}")
    for name, kb, ke, kp in ARMS:
        sc = np.array([r[f"sc{ke}"] for r in rows])
        sel = sc >= np.quantile(sc, EXCL)          # ⚠判定基準34: そのモデルの分位で
        w = np.where(sel, [r[f"w{kb}"] for r in rows], 0.0)
        cw = np.where(sel, [r[f"cw{kb}"] for r in rows], 0.0)
        pp = np.where(sel, [r[f"p{kp}"] for r in rows], 0.0)
        cp = np.where(sel, COST_P, 0.0)
        roiw = 100.0 * (w.sum() + cw.sum()) / cw.sum()
        roip = 100.0 * (pp.sum() + cp.sum()) / cp.sum()
        res[name] = dict(w=w, pp=pp, tot=w + pp, sel=sel)
        print(f"{name:<16}{int(sel.sum()):>8,}{roiw:>9.1f}%{roip:>9.1f}%"
              f"{w.mean():>+9.1f}円{pp.mean():>+9.1f}円{(w+pp).mean():>+10.1f}円")

    cur = res["現行 (3/3/3)"]
    scc = np.array([r["sc3"] for r in rows])
    selc = scc >= np.quantile(scc, EXCL)
    roi0 = 100.0 * (cur["w"].sum() + np.where(selc, [r["cw3"] for r in rows], 0.0).sum()) \
        / np.where(selc, [r["cw3"] for r in rows], 0.0).sum()
    ok = abs(roi0 - KNOWN) <= TOL
    print(f"\n⚠ゲート: 現行の枠連 **{roi0:.1f}%** vs 既知 {KNOWN}%　差 {roi0-KNOWN:+.1f}pt"
          f" → **{'★立った' if ok else '⚠⚠落ちた'}**")
    if not ok:
        print("⚠⚠**落ちた。以下を読まない**。"); return

    print("\n■ ★★主判定: 現行との対応差（**全レース・買わない日は損益0**）")
    print(f"{'比較':<20}{'合算の差':>11}{'99%CI(Bonf)':>21}{'枠連だけの差':>13}{'判定':>16}")
    for name, kb, ke, kp in ARMS[1:]:
        dd = res[name]["tot"] - cur["tot"]
        dw = res[name]["w"] - cur["w"]
        md, sd = dd.mean(), dd.std(ddof=1) / math.sqrt(N)
        v = "★差がある" if abs(md) > z * sd else "⚠差は検出できない"
        print(f"{name+' − 現行':<20}{md:>+10.1f}円"
              f"{f'[{md-z*sd:+.1f},{md+z*sd:+.1f}]':>21}{dw.mean():>+12.1f}円{v:>16}")

    print("\n" + "=" * 96)
    print("★読み方（**事前登録のとおり**）")
    print("  ★**提案の設計は正しい**——**結合が消え、三連複を犠牲にせずに枠連だけ動かせる**。")
    print("  ⚠**だが「差が確立した」かは別**。**合算の差のCIが0を跨いだら、運用は変えない**。")
    print("  ★**枠連だけの差も併記した**——**三連複を外す判断と結合しているため**（→(164)）。")
    print("  ⚠**採用するなら `train_prod.py` は2モデルを保存し、`predict_nk.py` は**")
    print("     **買い目=top2 / 除外=採用したほう / 三連複=top3 / (112)の裾と甘い軸=top3 に分ける**。")


if __name__ == "__main__":
    main()
