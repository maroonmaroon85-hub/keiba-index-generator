"""(168) ★★★学習目標が賭けと合っていない。**枠連は上位2頭で決まるのに top3 で学習している**

★★見つけ方（2026-08-30・ユーザー「モデルの考慮不足は無いか」）
　`train_prod.py:88` は **`y = (finish <= 3)`**。コメントは
　**「枠連も三連複も上位3頭の並びで決まるため((48)(55))」**。
　⚠**これは三連複には正しく、枠連には正しくない**——**枠連は1着と2着の枠だけで決まる。3着は無関係**。
　⚠**(48)が比べたのは win(1着) vs top3 で、top2 は一度も候補に入っていない**。
　⚠**二重にずれている**: `waku_score` は **`2*Pa*Pb` という上位2頭の式**を、
　　**top3で学習した確率**に当てている。

★★★事前登録（**測る前に書いている**）
　**3つの目標で学習して、現行の買い方で比べる**:
　　**A top3**（`finish<=3`・**現行**）／**B ★top2**（`finish<=2`）／**C win**（`finish==1`）
　★**比較は枠連 紐1・除外40%**（＝運用そのもの）。**三連複BOX4も参考に出す**
　　（**三連複は本当に top3 で決まるので、B が三連複で悪化するのが自然**。**そうなるかを見る**）。
　⚠★**判定基準34**: **除外の閾値は各アームの `waku_score` の分位で作り直す**。
　　**同じ絶対値を使い回すと、目標が変われば分布が動いて判定がずれる**。
　★**判定基準8**: **ROIだけでなく「1レースあたり期待損益（円）」も出す**。
　★**判定基準35**: **同じレースを使うので独立ではない。対応のある差で見る**。
　**Bonferroni α=0.01/2**（B vs A、C vs A の2比較）。

　⚠**ゲート（判定基準32）**: **A が既知 85.3% を ±2.5pt で再現しなければ何も読まない**。

★★**予想は持たない**。⚠**むしろ逆を警戒する**——**このプロジェクトには
　「モデルを『より正しく』すると賭けが悪くなる」が5例ある**（(24)(30)(52)(58)(63)）。
　**目標を賭けに合わせるのは「より正しく」の一種**なので、**6例目になる可能性が十分ある**。
　★**だから「合っているほうが良いはずだ」で運用を変えない**。**測ってから決める**。

実行: python3 ml/audit_target_top2.py
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
from itertools import combinations

EXCL = 0.40
KNOWN, TOL = 85.3, 2.5
NCMP = 2
ARMS = [("A top3（現行）", 3), ("★B top2", 2), ("C win", 1)]


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
    print("(168) ★学習目標を賭けに合わせると良くなるか（枠連は上位2頭で決まる）")
    print(f"　学習 {tr.sum():,} / 検証 {te.sum():,}・分割 {cut.date()}")
    print("⚠**「モデルを正しくすると賭けが悪くなる」が5例ある**。**予想は持たない**\n")

    races = {r["rid"]: r for r in load_races()}
    out = {}
    for name, k in ARMS:
        y = (fin <= k).astype(int)
        print(f"── {name}: 目標 finish<={k}（陽性率 {y[tr].mean():.3f}）学習中…")
        ms = fit_seeds(fx[tr], y[tr], 3, PAR)
        p = np.mean([m.predict_proba(fx[te])[:, 1] for m in ms], axis=0)
        sub = d.loc[te, ["raceid", "umaban"]].copy()
        sub["p"] = p
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
            # ⚠判定基準34: 閾値は**このアーム自身の分位**で作り直す
            sc = float(waku_score(wakuren_buy(order, n, 2), bp))
            pairs = wakuren_buy(order, n, 1)
            cw = 100.0 * len(pairs)
            box = list(combinations(sorted(order[:4]), 3))
            cp = 100.0 * len(box)
            rows.append((str(rid), sc,
                         (vw if key in pairs else 0.0) - cw, cw,
                         (vp if tuple(sorted((a, b, c))) in box else 0.0) - cp, cp))
        out[name] = {r[0]: r[1:] for r in rows}
        print(f"　　{len(rows):,}レース")

    # ★全アームで共通のレースだけを使う（対応のある比較・判定基準35）
    common = sorted(set.intersection(*[set(v) for v in out.values()]))
    print(f"\n★全アーム共通 {len(common):,}レース で対応ありの比較\n")
    z = zq(0.01 / NCMP)
    res = {}
    print(f"{'アーム':<16}{'買うR':>8}{'枠連ROI':>10}{'99%CI(Bonf)':>19}"
          f"{'1R損益':>10}{'  三複ROI':>10}")
    for name, _ in ARMS:
        m = out[name]
        sc = np.array([m[k][0] for k in common])
        thr = np.quantile(sc, EXCL)
        sel = sc >= thr
        pw = np.array([m[k][1] for k in common])[sel]
        cw = np.array([m[k][2] for k in common])[sel]
        pp = np.array([m[k][3] for k in common])[sel]
        cp = np.array([m[k][4] for k in common])[sel]
        roi = 100.0 * (pw.sum() + cw.sum()) / cw.sum()
        se = pw.std(ddof=1) / math.sqrt(len(pw)) * len(pw) / cw.sum() * 100.0
        res[name] = dict(sel=sel, pw=pw, cw=cw, roi=roi, se=se,
                         roip=100.0 * (pp.sum() + cp.sum()) / cp.sum())
        print(f"{name:<16}{int(sel.sum()):>8,}{roi:>9.1f}%"
              f"{f'[{roi-z*se:.1f},{roi+z*se:.1f}]':>19}{pw.mean():>+9.1f}円"
              f"{res[name]['roip']:>9.1f}%")

    base = res["A top3（現行）"]
    ok = abs(base["roi"] - KNOWN) <= TOL
    print(f"\n⚠ゲート: A **{base['roi']:.1f}%** vs 既知 {KNOWN}%"
          f"　差 {base['roi']-KNOWN:+.1f}pt → **{'★立った' if ok else '⚠⚠落ちた'}**")
    if not ok:
        print("⚠⚠**落ちた。以下を読まない**。")
        return

    print("\n■ ★対応のある差（**同じレース・全アームが買うレースだけ**・判定基準35）")
    print(f"{'比較':<24}{'1R損益差':>11}{'99%CI(Bonf)':>21}{'判定':>16}")
    for name, _ in ARMS[1:]:
        both = base["sel"] & res[name]["sel"]
        A = np.array([out["A top3（現行）"][k][1] for k in common])[both]
        B = np.array([out[name][k][1] for k in common])[both]
        dd = (B - A)
        md, sd = dd.mean(), dd.std(ddof=1) / math.sqrt(len(dd))
        v = "⚠差がある" if abs(md) > z * sd else "★差は検出できない"
        print(f"{name+' − A':<24}{md:>+10.1f}円"
              f"{f'[{md-z*sd:+.1f},{md+z*sd:+.1f}]':>21}{v:>16}"
              f"  （n={len(dd):,}）")

    print("\n" + "=" * 92)
    print("★読み方（**事前登録のとおり**）")
    print("  ⚠**「合っているほうが良いはずだ」で運用を変えない**。**円の差のCIが0を跨いだら変えない**。")
    print("  ★**三連複は本当に top3 で決まる**ので、**Bが三連複で悪化するのが自然**。")
    print("     **そうならなければ、目標の効き方そのものを疑う**。")
    print("  ⚠**このプロジェクトは「モデルを正しくすると賭けが悪くなる」を5回踏んでいる**。")


if __name__ == "__main__":
    main()
