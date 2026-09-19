"""(178-枠) ★(177)の内外バイアスは、**枠連のROI**を動かすか。**4腕を一度に測る**。

★**(177)で分かったこと**: **`logit(p)` を差し引いた後も内外バイアスが残る**
　（当日累積 +0.0163[+0.0093,+0.0233] / 前日(土→日) +0.0220[+0.0121,+0.0319]・年別10/10で正）。
　★**機構**: **44特徴量に枠番も馬番も無い**ので **`p` は馬番を知らない**。
　　**知っているのは `log_odds` 経由で市場が織り込んだ分だけ**＝**残ったのは市場の付け損ね**。
　⚠**前残りは死んだ**（CIがゼロを跨ぎ、符号6/10）。★**内外だけを持ち込む**。

⚠⚠★**それでも「儲かる」ではない**——**+0.016〜0.022 は log_odds 単独(−0.4423)の1/20**。
　★**このプロジェクトでは「モデルをより正しくすると賭けが悪くなる」が5回出ている**。
　**(168)で+1.3pt出たと思ったら(169)の分解で+1.1ptは除外由来、目標の効果は+0.2ptだった**。
　★**今回もその形になる可能性が高いと見ている**。⚠**予想はあてにしない**。

━━━ ★★事前登録（**結果を見る前に書く**。腕は4つ。後から増やさない）━━━

| | 何をするか | バイアス |
|---|---|---|
| **対照** | ★**現行**（除外40%・紐1） | — |
| **①A** | ★`align_draw` を**特徴量に足して再学習** | 当日累積 |
| **①B** | 同上 | 前日(土→日) |
| **②A** | ★**モデルはそのまま、推奨の上に絞りを重ねる** | 当日累積 |
| **②B** | 同上 | 前日(土→日) |

⚠★**AとBのどちらかを選んで走らせるのは後出し**（Bが+0.0220でAより大きいのを**もう知っている**）。
　→ ★**両方走らせ、Bonferroni α=0.01/4**。

★**主判定**: **1レースあたり損益の対応のある差**（腕 − 対照）。**99%CI(Bonf)**。
　⚠⚠**買わなかったレースも0円として全判定レースを母数に入れる**——
　　**(168)で「両方が買うレースだけ」で測り、測りたい部分を先に落とした**失敗を繰り返さない。
　★**仮説が偽なら0を返す**（判定基準42）。

★**②の絞り幅は判定基準46で縛る**: **候補 {10,20,30}%**。
　★**検証の前半で選び、後半で1回だけ試す**。**①も比較のため後半で評価する**。

★★**対照は2種類置く**:
　**(a) 現行**（腕との対応差の基準）
　**(b) プラセボ**: ★**同じ日の別の場のバイアス**を同じ形で当てる（乱数を使わない決定的な量）。
　　**①のプラセボも再学習する**（**特徴量を足したこと自体の効果**と切り分けるため）。

★**採用条件（3つ全部。有意なだけでは足りない・判定基準39/40）**:
　**① 対応差の99%CI(Bonf)がゼロを外し、★プラス側**
　**② 年別で符号が 8/10年以上そろう**
　**③ プラセボを上回る**

⚠**ゲート（判定基準32）**: **除外0%・紐1 の ROI が (155) の既知値 85.2% と ±2pt で一致**。
　**外れたら装置が違うので何も読まない**。
⚠**内部対照は最初に出す**: **突き合わせレース数**と**各腕でバイアスが引けたレース数**。

━━━ ★★結果（2026-09-16・**事前登録どおり・条件は変えていない**）━━━

■ ゲート: **除外0%・紐1 の ROI 85.3%**（(155)の既知値 85.2%）→ ★一致。読んだ。
■ 主判定（後半 12,251レース・対照は現行 除外40%・紐1 **ROI 84.7%**）

| 腕 | 買うR | ROI | 1Rあたり対応差 | 99%CI(Bonf) |
|---|---|---|---|---|
| ①A 特徴量・当日 | 7,351 | 85.9% | +0.69円 | [−1.10, +2.47] ⚠ゼロを跨ぐ |
| ①B 特徴量・前日 | 7,351 | 85.6% | +0.51円 | [−1.21, +2.23] ⚠ゼロを跨ぐ |
| ①A **プラセボ** | 7,351 | 85.7% | ⚠**+0.61円** | |
| ①B **プラセボ** | 7,351 | 86.0% | ⚠**+0.77円** | |
| ②A 絞り・当日 | 2,053 | 87.2% | +7.02円 | [+3.94, +10.10] |
| ②B 絞り・前日 | 1,019 | 90.6% | +8.38円 | [+4.98, +11.79] |
| ⚠②A **プラセボ** | 2,040 | **91.4%** | ⚠⚠**+7.73円** | |
| ⚠②B **プラセボ** | 1,151 | 89.6% | ⚠⚠**+8.18円** | |

★★★**4腕とも不採用。内外バイアスは枠連のROIを動かさない。運用は変えない**（判定基準40）。

⚠⚠★★**主判定が壊れていた（記録する）**——**②の +7〜8円は (170) と同じ人工物**。
　**ROIが100%未満なので、どんな部分集合を捨てても1レースあたり損益は必ず上がる**
　（**買わなければ0円、買えば平均 −15円**）。
　★**つまりこの統計量は「仮説が偽でも正を返す」**。**判定基準42を事前登録に書いておきながら外した**。
　★**(168)の「両方が買うレースだけ」は塞いだのに、(170)の「外せば必ずプラス」を塞いでいなかった**。
　★★**捕まえたのはプラセボ**——**②Aはプラセボ(+7.73)のほうが大きく、②Bは差+0.20円（CI幅±3.4）**。
　→ ★**対照を置く意味そのもの**。**主判定が壊れていても、プラセボが同じ壊れ方をするので比で救える**。

⚠★**もう1つ設計ミス**: **採用条件②を「8/10年以上」と書いたが、後半は2022〜2026の5年しかない**。
　**満たしようがない条件だった**（②A・②Bの「5/5」が⚠と出ているのはこのため）。
　★**結論は変わらない**（③で既に落ちている）が、**閾値は母集団を見てから決めること**。

★**①（特徴量に足す）は全部ゼロ跨ぎ、しかもプラセボと同じ大きさ**（+0.6〜0.8円）
　＝**何を足しても同じだけ動く**。**バイアスの効果ではない**。

★**(177)との関係**: **着順の予測では確かに残っていた**（+0.0163〜+0.0220・年別10/10）。
　⚠**だがROIには変わらなかった**。**log_odds 単独が −0.4423 なのに対し 1/20 の大きさ**で、
　**(88)の「市場の誤りは最大+10pt・控除は20.5pt」に対して半分にも届かない**。
　→ ★★**「モデルをより正しくすると賭けが良くなるとは限らない」の6例目**。**事前の見立てどおり**。

実行: python3 ml/audit_bias_roi.py
"""
import math
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "ml")
import features as F
from audit_crosspool import load_races, payoff, zq
from audit_crosspool2 import realized
from audit_track_bias import build_bias
from train_prod import CAPACITY, add_odds_features, fit_seeds
from waku_umatan import bracket_probs, waku_of, waku_score, wakuren_buy

KNOWN_ROI, TOL = 85.2, 2.0
EXCL = 0.40
CUTS = [0.10, 0.20, 0.30]
NCMP = 4
NYEAR_OK = 8


def sim(sub, races, scorecol="p"):
    """検証レースごとに (raceid, year, waku_score, 払戻, コスト, 軸のalign各種) を作る。"""
    out = []
    for rid, g in sub.groupby("raceid"):
        r = races.get(str(rid))
        if r is None or not r.get("wakuren"):
            continue
        rl = realized(r)
        if rl is None:
            continue
        a, b0, _ = rl
        n, nums = r["n"], [u for u, _, _ in r["horses"]]
        if a not in nums or b0 not in nums:
            continue
        gg = g.sort_values(scorecol, ascending=False)
        order = [int(u) for u in gg["umaban"].tolist()]
        if len(order) < 3:
            continue
        key = tuple(sorted((waku_of(a, n), waku_of(b0, n))))
        v = payoff(r, "枠連(人気順)", [key[0], key[1]])
        if not v or v <= 0:
            continue
        pv = gg[scorecol].to_numpy(float)
        q = pv / pv.sum()
        bp = bracket_probs(order, q, n)
        pairs = wakuren_buy(order, n, 1)
        sc = float(waku_score(pairs, bp))
        row = {"raceid": str(rid), "year": int(r["year"]), "sc": sc,
               "ret": (v if key in pairs else 0.0), "cost": 100.0 * len(pairs)}
        top = gg.iloc[0]
        for c in ("A_draw", "B_draw", "Aplc_draw", "Bplc_draw"):
            row[c] = float(top[c]) if c in gg.columns and pd.notna(top[c]) else np.nan
        out.append(row)
    return pd.DataFrame(out)


def profit(t, keepmask):
    """買わないレースも0円で母数に入れた、1レースあたり損益の配列。"""
    pr = np.zeros(len(t))
    m = keepmask.to_numpy() if hasattr(keepmask, "to_numpy") else keepmask
    pr[m] = (t["ret"].to_numpy()[m] - t["cost"].to_numpy()[m])
    return pr


def main():
    MODEL_DIR, PAR = CAPACITY["l2"]
    raw = F.load_files()
    raw = raw[raw[40].str.len() > 2]
    d = F.to_model(raw)
    f = F.build_features(d)
    keep = (f["n_prior"] >= 1) & d["odds"].notna() & (d["odds"] > 0)
    d, f = d[keep].reset_index(drop=True), f[keep].reset_index(drop=True)
    y = (d["finish"] <= 3).astype(int).to_numpy()
    b = build_bias(raw, d, f)

    fx, _ = F.encode_categoricals(f)
    fx = add_odds_features(fx, d["odds"].to_numpy(float), d["raceid"].to_numpy())
    cut = d["date"].quantile(0.3)
    tr, te = (d["date"] < cut).to_numpy(), (d["date"] >= cut).to_numpy()
    print(f"(178-枠) ★内外バイアスは枠連のROIを動かすか（4腕・Bonferroni α=0.01/{NCMP}）")
    print(f"■ 内部対照: 全体 {len(d):,}行 / 学習 {int(tr.sum()):,}（〜{cut.date()}） /"
          f" 検証 {int(te.sum()):,}")

    races = {r["rid"]: r for r in load_races()}
    models = {}
    for tag, col in (("base", None), ("①A", "A_draw"), ("①B", "B_draw"),
                     ("①A_plc", "Aplc_draw"), ("①B_plc", "Bplc_draw")):
        X = fx if col is None else fx.assign(**{"align": b[col].to_numpy()})
        ms = fit_seeds(X[tr], y[tr], 3, PAR)
        models[tag] = (X, np.mean([m.predict_proba(X[te])[:, 1] for m in ms], axis=0))
        print(f"　学習ずみ {tag}")

    base_sub = d.loc[te, ["raceid", "umaban"]].copy()
    for c in ("A_draw", "B_draw", "Aplc_draw", "Bplc_draw"):
        base_sub[c] = b[c].to_numpy()[te]
    tabs = {}
    for tag, (_, pv) in models.items():
        s = base_sub.copy(); s["p"] = pv
        tabs[tag] = sim(s, races)
        print(f"　突き合わせ {tag}: {len(tabs[tag]):,}レース")

    t0 = tabs["base"]
    # ⚠ゲート: 除外0%・紐1 の ROI
    roi0 = 100.0 * t0["ret"].sum() / t0["cost"].sum()
    gate = abs(roi0 - KNOWN_ROI) <= TOL
    print(f"\n■ ⚠ゲート: 除外0%・紐1 の ROI {roi0:.1f}%（(155)の既知値 {KNOWN_ROI}%）"
          f" → {'★一致。読む' if gate else '⚠外れた。読まない'}")
    if not gate:
        return

    # 検証を前半/後半に割る（②の絞り幅は前半で選び、全腕を後半で評価する）
    yrs = np.sort(t0["year"].unique())
    mid = yrs[len(yrs) // 2]
    print(f"　★検証の前半 {yrs[0]}〜{mid-1} で②の幅を選び、後半 {mid}〜{yrs[-1]} で1回だけ試す")

    def keep_of(t, rate=EXCL, extra=None, xrate=None):
        th = np.quantile(t["sc"], rate)
        k = t["sc"] >= th
        if extra is not None and xrate:
            v = t[extra]
            ok = v.notna() & k
            if ok.sum() > 0:
                th2 = np.quantile(v[ok], xrate)
                k = k & v.notna() & (v >= th2)
        return k

    # ★②の幅を前半で選ぶ
    pre = t0["year"] < mid
    chosen = {}
    for arm, col in (("②A", "A_draw"), ("②B", "B_draw"),
                     ("②A_plc", "Aplc_draw"), ("②B_plc", "Bplc_draw")):
        best, bx = -1e18, CUTS[0]
        for x in CUTS:
            tp = t0[pre].reset_index(drop=True)
            dd = profit(tp, keep_of(tp, EXCL, col, x)) - profit(tp, keep_of(tp, EXCL))
            if dd.mean() > best:
                best, bx = dd.mean(), x
        chosen[arm] = bx
        print(f"　②の幅 {arm}: 前半で選ばれたのは下位 {int(bx*100)}% を追加除外")

    z = zq(0.01 / NCMP)
    post = t0["year"] >= mid
    tb = t0[post].reset_index(drop=True)
    p_ctl = profit(tb, keep_of(tb, EXCL))
    roi_ctl = 100.0 * tb["ret"].to_numpy()[keep_of(tb, EXCL).to_numpy()].sum() \
        / max(tb["cost"].to_numpy()[keep_of(tb, EXCL).to_numpy()].sum(), 1)
    print(f"\n■ ★★主判定（後半 {int(post.sum()):,}レース・対照は現行 除外40%・紐1 ROI {roi_ctl:.1f}%）")
    print(f"{'腕':>10}{'買うR':>8}{'ROI':>8}{'1Rあたり対応差':>16}{'99%CI(Bonf)':>24}")

    res = {}
    for arm in ("①A", "①B", "①A_plc", "①B_plc", "②A", "②B", "②A_plc", "②B_plc"):
        if arm.startswith("①"):
            ta = tabs[arm].merge(t0[["raceid"]], on="raceid")
            ta = ta[ta["year"] >= mid].reset_index(drop=True)
            ka = keep_of(ta, EXCL)
            pa = profit(ta, ka)
            # 対応をとるため raceid で揃える
            mm = pd.DataFrame({"raceid": ta["raceid"], "pa": pa}).merge(
                pd.DataFrame({"raceid": tb["raceid"], "pc": p_ctl}), on="raceid")
            dd = mm["pa"].to_numpy() - mm["pc"].to_numpy()
            nb, rr = int(ka.sum()), 100.0 * ta["ret"].to_numpy()[ka.to_numpy()].sum() \
                / max(ta["cost"].to_numpy()[ka.to_numpy()].sum(), 1)
        else:
            col = {"②A": "A_draw", "②B": "B_draw",
                   "②A_plc": "Aplc_draw", "②B_plc": "Bplc_draw"}[arm]
            ka = keep_of(tb, EXCL, col, chosen[arm])
            dd = profit(tb, ka) - p_ctl
            nb, rr = int(ka.sum()), 100.0 * tb["ret"].to_numpy()[ka.to_numpy()].sum() \
                / max(tb["cost"].to_numpy()[ka.to_numpy()].sum(), 1)
        md, se = dd.mean(), dd.std(ddof=1) / math.sqrt(len(dd))
        res[arm] = (md, md - z * se, md + z * se)
        print(f"{arm:>10}{nb:>8,}{rr:>7.1f}%{md:>+16.2f}円"
              f"{f'[{md-z*se:+.2f}, {md+z*se:+.2f}]':>24}")

    print("\n■ 採用条件② 年別の符号（後半のみ）")
    ok2 = {}
    for arm in ("①A", "①B", "②A", "②B"):
        sg = []
        for Y in sorted(set(tb["year"])):
            if arm.startswith("①"):
                ta = tabs[arm].merge(t0[["raceid"]], on="raceid")
                ta = ta[ta["year"] == Y].reset_index(drop=True)
                tc = tb[tb["year"] == Y].reset_index(drop=True)
                if len(ta) < 300 or len(tc) < 300:
                    continue
                mm = pd.DataFrame({"raceid": ta["raceid"],
                                   "pa": profit(ta, keep_of(ta, EXCL))}).merge(
                     pd.DataFrame({"raceid": tc["raceid"],
                                   "pc": profit(tc, keep_of(tc, EXCL))}), on="raceid")
                sg.append((mm["pa"] - mm["pc"]).mean())
            else:
                col = {"②A": "A_draw", "②B": "B_draw"}[arm]
                tc = tb[tb["year"] == Y].reset_index(drop=True)
                if len(tc) < 300:
                    continue
                sg.append((profit(tc, keep_of(tc, EXCL, col, chosen[arm]))
                           - profit(tc, keep_of(tc, EXCL))).mean())
        n_ok = max(sum(1 for s in sg if s > 0), sum(1 for s in sg if s < 0))
        ok2[arm] = (n_ok, len(sg))
        print(f"　{arm}: そろった年 {n_ok}/{len(sg)}　（{' '.join(f'{s:+.1f}' for s in sg)}）")

    print("\n" + "=" * 92)
    print("★判定（3つ全部そろった腕だけが採用候補）")
    any_ok = False
    for arm, plc in (("①A", "①A_plc"), ("①B", "①B_plc"),
                     ("②A", "②A_plc"), ("②B", "②B_plc")):
        md, lo, hi = res[arm]
        c1 = lo > 0
        c2 = ok2[arm][0] >= NYEAR_OK
        c3 = md > res[plc][0]
        ok = c1 and c2 and c3
        any_ok |= ok
        print(f"　{arm}: ①CI>0 {'★' if c1 else '⚠'} / ②符号 {ok2[arm][0]}/{ok2[arm][1]}"
              f" {'★' if c2 else '⚠'} / ③プラセボ超え {'★' if c3 else '⚠'}"
              f"　→ **{'★採用候補' if ok else '⚠不採用'}**")
    if not any_ok:
        print("\n⚠★どの腕も3条件を満たさない＝**内外バイアスはROIを動かさない**と読む。")
        print("　★**この線はここで閉じる。運用は変えない**（判定基準40）。")
    else:
        print("\n★★採用候補が出た。⚠**それでも運用にすぐ入れない**——")
        print("　★**利用者に材料として渡す**。**運用ファイルは触らない**（役割分担）。")


if __name__ == "__main__":
    main()
