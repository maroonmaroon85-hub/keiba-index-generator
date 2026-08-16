"""(153b) ★★★★(153)の訂正 — **Aは「モデル」ではなく「市場順」だった**

⚠★★**(153)の誤りを正直に書く（2026-08-16・同日中に自分で見つけた）**
　(153)は買い方Aを「**A 現行(モデル)**」と表示したが、中身は
　`probs(hs)` ＝ **単勝オッズ由来の市場確率**で並べた 1番人気の枠 × 2番人気の枠 だった。
　★**MLモデルは1行も使っていない**。**「現行の本命」ではない**。
　→ **(153)が比べていたのは「市場順 vs 馬連プール」で、どちらも市場情報だけ**。
　　 **同点になるのは驚きではない**。**本来の問い「モデル順 vs 馬連プール」は未回答のまま**。
　★**陽性対照が 83.3% vs 既知 85.2% で −1.9pt ずれたのがヒントだった**——
　　 **±3ptの許容に入ったので見逃した**。⚠**許容幅は「近い」を保証するが「同じ」は保証しない**。
　　 → **判定基準32に足す教訓: 対照がズレた向きにも意味がある**。

★★これが測るもの: **モデル順（＝いまの本命）vs 馬連プール順** を**同じレース**で。
　1. **A_ml 現行の本命**: **MLモデル**降順の1位馬の枠 × 2位馬の枠 **1点**
　2. **A_mkt 市場順**: 単勝オッズ降順（(153)のA。**モデルの上積みを見るための対照**）
　3. **B 馬連プール**: `q_枠(a,b)=Σ 1/馬連オッズ` が最大の枠組 **1点**
　4. **C 枠連の最人気** … 市場そのもの

⚠★**リークを避ける**: **(55)(62)(80)と同じ「前30%で学習・後70%で検証」**。
　**A_ml はこの検証側でしか評価しない**。**他の3つも同じレースに揃える**（対応のある比較）。

★★事前登録（**測る前に書いている**。(153)の骨格をそのまま使う）
　1. **判定は ROI ではなく「1レース期待損失（円）」**。**A_ml との対応のある差にCI**。
　2. **陽性対照（判定基準32）**: **A_ml の既知値は ROI 85.2% / 1R期待損失 14.8円**。
　　 ★**今度はモデルを使うので、±3ptではなく ±2pt で見る**（前回はここで見逃した）。
　3. **プラセボ**: 無作為な枠組1点。**解析的**（判定基準23）。
　4. ★**予想**: **(153)から言えるのは「B ≈ A_mkt」だけ**。
　　 **モデルが市場順に +1.9pt 乗るなら、B は A_ml に負けるはず**。
　　 ⚠**これは2つの標本をまたいだ連鎖推論なので、当てにしない**（判定基準24）。**測って決める**。

★★★実行済みの結果（2026-08-16・検証25,714レース・分割日2017-01-14）
　★①陽性対照: **A_ml 85.3% / 14.7円** vs 既知 85.2% / 14.8円 → **+0.1pt でぴたりと立った**。
　　→ ★**(153)のAが「モデルでない」ことの裏も取れた**（あちらは83.3%＝A_mktの82.5%側）。

| 買い方 | ROI | 1R期待損失 | A_ml との差（対応のある差） |
|---|---|---|---|
| ★**A_ml 現行の本命** | **85.3%** | **14.7円** | — |
| A_mkt 市場順 | 82.5% | 17.5円 | +2.78円 99%CI[**−0.54**,+6.11] ＝**0をまたぐ** |
| **B 馬連プール** | 82.6% | 17.4円 | +2.66円 99%CI[**−0.94**,+6.26] ＝**0をまたぐ** |
| C 枠連の最人気 | 79.5% | 20.5円 | +5.73円 99%CI[+2.14,+9.31] ＝★**有意** |
| プラセボ(無作為) | 62.5% | 37.5円 | 解析的（判定基準23） |

★★**結論: 運用は変えない**。**ただし「A_ml が B に勝つ」とは言えない**——**差は0をまたぐ**。
　正しい言い方は **「馬連プールに替える理由が無い（優位を示せない）」**。**「モデルが勝つ」ではない**。

⚠★★**ついでに分かった、居心地の悪い数字**: **A_ml vs A_mkt（＝モデルの上積み）も 99%CIが0をまたぐ**
　（+2.78円 [−0.54,+6.11]）。**25,714レースあってなお、枠連1点買いでは
　「モデルは単勝人気順に勝つ」を99%水準で示せない**。
　★**点推定は +2.8円/R で一貫して正**であり、**(45)以降の「モデルは市場を上回る」と矛盾はしない**
　（あちらは単勝EVや複勝など**別の測り方**で示されている）。
　→ ⚠**だが「枠連1点買いという買い方では、モデルの上積みは検出限界すれすれ」**とは書ける。
　　 **(80)の「買い方どうしの比較はROIではなく円で」がここでも効いている**。

★**Cだけが有意に負ける**＝**枠連プールの最人気を買うのが最も悪い**。
　**A_ml も A_mkt も B も、市場そのものよりは良い**。**3つの間に差が無い**という形。

実行: python3 ml/audit_waku_buy_ml.py
"""
import math
import sys

import numpy as np

sys.path.insert(0, "ml")
import features as F
from audit_cond_split import load_boards
from audit_crosspool import load_races, payoff, probs, zq
from audit_crosspool2 import realized
from audit_overlay_all import load_board
from audit_waku_buy_umaren import agg_umaren
from train_prod import CAPACITY, add_odds_features, fit_seeds
from waku_umatan import waku_of

KNOWN_ROI, KNOWN_YEN, TOL = 85.2, 14.8, 2.0


def summarize(name, rows, z, base=None):
    pr = np.array([v - 100.0 for v in rows], float)
    roi = 100.0 * (1 + pr.mean() / 100.0)
    se = pr.std(ddof=1) / math.sqrt(len(pr))
    d = ""
    if base is not None:
        dd = pr - base
        sd = dd.std(ddof=1) / math.sqrt(len(dd))
        d = (f"  差 {-dd.mean():+7.2f}円 99%CI["
             f"{-(dd.mean()+z*sd):+.2f},{-(dd.mean()-z*sd):+.2f}]")
    print(f"{name:>18}{len(pr):>9,}{roi:>8.1f}%{-pr.mean():>9.1f}円"
          f"{'[' + format(roi - z*se, '.1f') + ',' + format(roi + z*se, '.1f') + ']':>16}{d}")
    return pr


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
    print(f"(153b) モデル順 vs 馬連プール順（学習 {tr.sum():,} / 検証 {te.sum():,}・分割 {cut.date()}）")
    ms = fit_seeds(fx[tr], y[tr], 3, PAR)
    p_ml = np.mean([m.predict_proba(fx[te])[:, 1] for m in ms], axis=0)
    sub = d.loc[te, ["raceid", "umaban"]].copy()
    sub["p"] = p_ml
    ml_order = {}
    for rid, g in sub.groupby("raceid"):
        gg = g.sort_values("p", ascending=False)
        ml_order[str(rid)] = [int(u) for u in gg["umaban"].tolist()]

    wb, ub = load_boards(), load_board(4, 4)
    races = {r["rid"]: r for r in load_races()}
    Aml, Amk, B, C, PL = [], [], [], [], []
    for rid, order in ml_order.items():
        r = races.get(rid)
        if r is None or not r.get("wakuren"):
            continue
        rl = realized(r)
        if rl is None:
            continue
        a, b, _ = rl
        n, hs = r["n"], r["horses"]
        nums = [u for u, _, _ in hs]
        if a not in nums or b not in nums or len(order) < 2:
            continue
        W, U = wb.get(rid), ub.get(rid)
        if not W or not U:
            continue
        key = tuple(sorted((waku_of(a, n), waku_of(b, n))))
        if key not in W:
            continue
        v = payoff(r, "枠連(人気順)", [key[0], key[1]])
        if not v or v <= 0:
            continue
        k_ml = tuple(sorted((waku_of(order[0], n), waku_of(order[1], n))))
        pm = probs(hs)
        o2 = np.argsort(-pm)
        k_mk = tuple(sorted((waku_of(nums[o2[0]], n), waku_of(nums[o2[1]], n))))
        ag = {k: x for k, x in agg_umaren(U, nums, n).items() if k in W}
        if not ag:
            continue
        k_b = max(ag, key=lambda k: ag[k])
        k_c = min(W, key=lambda k: W[k])
        Aml.append(v if k_ml == key else 0.0)
        Amk.append(v if k_mk == key else 0.0)
        B.append(v if k_b == key else 0.0)
        C.append(v if k_c == key else 0.0)
        PL.append(v / len(W))

    z = zq(0.01)
    print(f"　突き合わせ {len(Aml):,}レース")
    print("★経路: A_ml = **MLモデル** / A_mkt = **単勝オッズ** / "
          "B = **馬連の板→枠（厳密）** / C = **枠連の板そのもの**\n")
    print(f"{'買い方':>18}{'R数':>9}{'ROI':>9}{'1R期待損失':>10}{'99%CI':>16}"
          f"  差（A_ml との対応のある差）")
    pa = summarize("★A_ml 現行の本命", Aml, z)
    got = 100 * (1 + pa.mean() / 100)
    ok = abs(got - KNOWN_ROI) <= TOL
    print(f"　★陽性対照: 既知 ROI {KNOWN_ROI}% / {KNOWN_YEN}円 → 再現 {got:.1f}% / {-pa.mean():.1f}円"
          f"　差 {got-KNOWN_ROI:+.1f}pt")
    print(f"　→ ★**①は立ったか（±{TOL}pt）: {'★立った' if ok else '⚠立っていない'}**")
    if not ok:
        print("　⚠**立っていない。以下を読まないこと**（判定基準32）。")
    summarize("A_mkt 市場順", Amk, z, pa)
    summarize("B 馬連プール", B, z, pa)
    summarize("C 枠連の最人気", C, z, pa)
    pl = np.array([v - 100.0 for v in PL], float)
    print(f"{'プラセボ(無作為)':>18}{len(pl):>9,}{100*(1+pl.mean()/100):>8.1f}%"
          f"{-pl.mean():>9.1f}円   ※解析的（判定基準23）")

    print("\n" + "=" * 100)
    print("★読み方（事前登録のとおり）")
    print("  ・**A_ml が B に有意に勝てば、いまの本命のままでよい**（＝モデルに価値がある）。")
    print("  ・**差が0をまたげば「どちらでもよい」**。**Bのほうが良ければ運用を替える候補**。")
    print("  ⚠**(153)は A を市場順で書いていた**。**この(153b)が正しい比較**。")


if __name__ == "__main__":
    main()
