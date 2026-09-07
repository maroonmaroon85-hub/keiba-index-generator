"""(207) ★**穴馬の線の推奨を、指定日について出す** —— 先に買い目、あとで答え合わせ

■ ★規則（**(203)(206)で測ったもの**・すべて事前に固定）
　**pn（推奨度）= モデルの複勝確率をレース内で合計3に正規化**
　**qp（市場）= 複勝の板の[下限,上限]の★調和平均から、合計3に正規化**
　**ズレ gap = pn − qp**
| 推奨 | 軸の選び方 | 買い目 | ★実測ROI | ★99%CI |
|---|---|---|---|---|
| ★**A 複勝** | **pn≥0.15 かつ gap≥0.15 の中で ★pn最大** | **複勝1点** | **95.9%** | ⚠**[91.6,100.2]** |
| **B 三連単** | **pn≥0.15 かつ gap≥0.02 のうち ★gap最大** | **軸1着固定・紐(p降順)2頭の2点** | **103.1%** | ⚠**[79.4,126.7]** |
　★**参考として同じ軸の 馬連(紐1)・三連複 も出す**（**90.0% / 91.2%**——**マイナスと言える**）。

⚠**どちらも100%を挟む/下回る**。★**「勝てる」とは測れていない**。

■ ★★**過去走の読み込み**（**別セッションからの指摘・2026-09-06**）
　⚠**`F.load_files()` はルート直下の *.CSV しか読まない**ので、**7/26 までしか見えなかった**。
　★**`data/nk/DSnk*.CSV`（8/1〜9/6 の14ファイル）を足すと 9/6 まで見える**
　（**`predict_nk.py:128-129` と同じ読み方**）。⚠**0バイトのDSnkは飛ばす**（8/09で実際に踏んだ）。
■ ⚠**キャッシュを分ける**: **行数が変わると `wf_predict` のキャッシュが無効化され、
　`data/cache/wf_pred.npz` を★上書きしてしまう**。**それを使っている(174)〜(206)の
　内部対照が全部ズレるので、★このスクリプト専用のキャッシュに逃がす**。
■ ★**2026-08-09 の欠けは別セッションで解消済み**（13/36 → 36/36・`--refresh` を追加）。
　★**7/19〜9/6 の14開催日504レースすべてに複勝の板がある**。**欠けはゼロ**。
■ ⚠⚠**馬IDの表記が2通りある**（**2026-09-06 に実測**）
　★**ルートの *.CSV は全部8桁**（669,951行）。⚠**DSnk には10桁のIDが混じる**
　（**通常のファイルで約15%、`--refresh` で取り直した 8/9・9/6 は★100%が10桁**）。
　★**10桁は「20」＋8桁**なので、**先頭の "20" を落とせばルートと一致する**
　（**9/6 の489頭: そのままだと一致0頭 → ★"20"を落とすと431頭一致**。残りは新馬とみられる）。
　⚠**直さないと `n_prior=0` になり、その馬は丸ごと落ちる**（**9/6 は 489行 → 55行になっていた**）。
　★★**正規化したら必ず並べ直すこと**——**`to_model` の最後が `sort_values(["horse","date"])`
　　なので、後から名前を変えると並びが壊れたままになる**。**`build_features` の `cumcount()` は
　　★行順で数える**ので、**9/6の行が先頭に来て n_prior=0 になる**（**実際に踏んだ**）。
■ ★★**払戻の出どころが2つある**（**日付で使い分ける**）
　★**`data/payout/a.csv`**（`audit_crosspool.load_races`）——**2026-07-26 まで**。
　★**`data/nk/pay*.csv`**（`nk_score.load_pays`）——**8月以降はこちら**。
　⚠**推奨を出すだけなら払戻は要らない**ので、**`--check` のときだけ要求する**。

■ ★**複数日をまとめて見る**: `--from YYYY-MM-DD`（**日別の集計だけ出す**）
　⚠⚠**日別の数字を並べて「今週は良かった／悪かった」と読まないこと**——
　**それは判定基準43（1標本のばらつきを結論と読む）を毎回やることになる**。
　★**貯めるのは毎回・読むのは年1回・読む前に主判定を書く**（**別セッションからの指摘**）。
　★**2026-08-01以降は(206)の測定に入っていない**＝**測定後のデータ**。**7/19・7/26は入っている**。

実行:
　python3 ml/reco_ana.py [YYYY-MM-DD]        ★推奨だけ（結果を見ない）
　python3 ml/reco_ana.py [YYYY-MM-DD] --check  ★答え合わせ
　python3 ml/reco_ana.py --from 2026-07-19 --check   ★日別の集計
"""
import glob
import os
import sys
from itertools import combinations

import numpy as np
import pandas as pd

sys.path.insert(0, "ml")
import features as F
from audit_crosspool import load_races, payoff
from audit_ana_odds import MIN_HORSES
from nk_score import load_pays
import audit_ana_marg as _M
# ★★このスクリプトは DSnk を足して行数が変わるので、共有キャッシュを上書きしない
_M.CACHE = "data/cache/wf_pred_reco.npz"
from audit_ana_marg import wf_predict
from audit_ana_board import NPLACE, load_fuku_boards, qpool
from audit_ana_band import PN_FLOOR
from train_prod import add_odds_features

GAP_A, GAP_B = 0.15, 0.02
# ★★(215) (210)〜(214)で出した穴側の候補4つを、同じ画面に並べる
#   ★軸 = pn≥0.15 かつ gap≥0.15 かつ ★単勝オッズ≥10.0倍 の中で pn最大（**(210)で固定**）
#   ★紐 P = pn降順（人気馬・平均2.3番人気）／ G = gap降順（穴馬・平均6.6番人気）
from audit_ana_bet import tickets as bet_tickets
from audit_ana_fix import LFIX

CAND = [("C1 G馬単M 4点", "G", "馬単M", 4, 139.2, 39),
        ("C2 G馬単B 2点", "G", "馬単B", 2, 144.1, 48),
        ("C3 P馬単A 2点", "P", "馬単A", 2, 136.5, 55),
        ("C4 P三連単A 3点", "P", "三連単A", 3, 137.5, 68)]


def picks_of_race(gg, ub, od, pv, bd, order_p):
    """★1レース分の推奨。無ければ None"""
    pn = pv / pv.sum() * NPLACE
    qp, _ = qpool([bd[int(u)] for u in ub], "harm")
    gap = pn - qp
    A = B = None
    cA = np.where((pn >= PN_FLOOR) & (gap >= GAP_A))[0]
    if len(cA):
        i = int(cA[int(np.argmax(pn[cA]))])
        A = int(ub[i])
    cB = np.where((pn >= PN_FLOOR) & (gap >= GAP_B))[0]
    if len(cB):
        i = int(cB[int(np.argmax(gap[cB]))])
        ax = int(ub[i])
        himo = [u for u in order_p if u != ax][:2]
        if len(himo) == 2:
            B = (ax, himo)
    return A, B


def per_day(sub, days, races, pays, boards, pay_of, check):
    """★日別の集計だけを出す（明細は出さない）"""
    print(f"{'日付':<12}{'R数':>5}{'A本':>5}{'A的中':>6}{'A投資':>8}{'A払戻':>8}{'A ROI':>8}"
          f"{'B本':>5}{'B的中':>6}{'B投資':>9}{'B払戻':>9}{'B ROI':>8}{'測定':>8}")
    TOT = {"a": [0.0, 0.0, 0, 0], "b": [0.0, 0.0, 0, 0]}
    NEW = {"a": [0.0, 0.0, 0, 0], "b": [0.0, 0.0, 0, 0]}
    for day in days:
        tgt = sub[sub["date"] == day]
        ta = [0.0, 0.0, 0, 0]
        tb = [0.0, 0.0, 0, 0]
        nr = 0
        for rid, g in tgt.groupby("raceid"):
            rid = str(rid)
            bd = boards.get(rid)
            if bd is None:
                continue
            r = races.get(rid)
            gg = g[g["umaban"].astype(int).isin({u for u, _, _ in r["horses"]})] if r else g
            ub = gg["umaban"].astype(int).to_numpy()
            if len(gg) < MIN_HORSES or not all(int(u) in bd for u in ub):
                continue
            od = gg["odds"].to_numpy(float)
            pv = gg["p"].to_numpy(float)
            if not np.isfinite(od).all() or (od <= 0).any() or pv.sum() <= 0:
                continue
            if rid not in races and rid not in pays:
                continue
            nr += 1
            order_p = [int(u) for u in ub[np.argsort(-pv, kind="mergesort")]]
            A, B = picks_of_race(gg, ub, od, pv, bd, order_p)
            if A is not None:
                v = pay_of(rid, "複勝", [A]) or 0.0
                ta[0] += 100.0; ta[1] += v; ta[2] += 1; ta[3] += 1 if v > 0 else 0
            if B is not None:
                ax, h = B
                vs = [pay_of(rid, "三連単", [ax, h[0], h[1]]),
                      pay_of(rid, "三連単", [ax, h[1], h[0]])]
                v = sum(x for x in vs if x)
                tb[0] += 200.0; tb[1] += v; tb[2] += 1; tb[3] += 1 if v > 0 else 0
        ts = pd.Timestamp(day)
        new = ts >= pd.Timestamp("2026-08-01")
        for k, t in (("a", ta), ("b", tb)):
            for i in range(4):
                TOT[k][i] += t[i]
                if new:
                    NEW[k][i] += t[i]
        ra = 100.0 * ta[1] / ta[0] if ta[0] else float("nan")
        rb = 100.0 * tb[1] / tb[0] if tb[0] else float("nan")
        print(f"{str(ts.date()):<12}{nr:>5}{ta[2]:>5}{ta[3]:>6}{ta[0]:>7,.0f}円"
              f"{ta[1]:>7,.0f}円{ra:>7.1f}%{tb[2]:>5}{tb[3]:>6}{tb[0]:>8,.0f}円"
              f"{tb[1]:>8,.0f}円{rb:>7.1f}%"
              f"{('★測定後' if new else '測定内'):>8}")
    print()
    for nm2, T in (("★全期間", TOT), ("★★2026-08-01以降（測定後）", NEW)):
        ra = 100.0 * T["a"][1] / T["a"][0] if T["a"][0] else float("nan")
        rb = 100.0 * T["b"][1] / T["b"][0] if T["b"][0] else float("nan")
        print(f"{nm2}: ★複勝 {T['a'][2]}本 {T['a'][3]}的中 "
              f"{T['a'][0]:,.0f}円→{T['a'][1]:,.0f}円 **{ra:.1f}%**"
              f"　／　三連単 {T['b'][2]}本 {T['b'][3]}的中 "
              f"{T['b'][0]:,.0f}円→{T['b'][1]:,.0f}円 **{rb:.1f}%**")
    print(f"\n★11年の実測: **複勝 95.9% [91.6,100.2] / 三連単 103.1% [79.4,126.7]**")
    print(f"⚠⚠**この表で「良かった／悪かった」を読まないこと**"
          f"——**判定基準43を毎回やることになる**（別セッションからの指摘）。")
    print(f"★**貯めるのは毎回・読むのは年1回・読む前に主判定を書く**。")
    print("\n⚠**枠連の運用には触れない**。**設定変更は提案しない**。")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    check = "--check" in sys.argv
    print("(207) ★**穴馬の線の推奨**" + ("　★★【答え合わせ】" if check else "　★推奨のみ（結果は見ない）"))
    races = {r["rid"]: r for r in load_races()}
    pays = load_pays()
    ORD = {"馬単", "三連単"}

    def pay_of(rid, kind, sel):
        """★払戻。a.csv にあればそれ、無ければ pay*.csv。★どちらにも無ければ None"""
        if rid in races:
            return payoff(races[rid], kind, list(sel))
        m = pays.get(rid)
        if m is None:
            return None
        key = tuple(sel) if kind in ORD else tuple(sorted(sel))
        return float(m.get(kind, {}).get(key, 0.0))

    boards = load_fuku_boards()
    print(f"★払戻: a.csv **{len(races):,}レース** ＋ pay*.csv **{len(pays):,}レース**")
    ds = [p for p in sorted(glob.glob("data/nk/DSnk*.CSV")) if os.path.getsize(p) > 0]
    print(f"★過去走: ルート直下の *.CSV ＋ **data/nk/DSnk*.CSV {len(ds)}本**")
    frames = [F.load_files()] + [
        pd.read_csv(p, header=None, encoding="shift_jis", encoding_errors="replace",
                    dtype=str, keep_default_na=False) for p in ds]
    d = F.to_model(pd.concat(frames, ignore_index=True))
    # ★★馬IDの表記ゆれを直す（10桁「20」＋8桁 → 8桁）。直さないと過去走がリンクしない
    h0 = d["horse"].astype(str)
    d["horse"] = h0.map(lambda x: x[2:] if len(x) == 10 and x.startswith("20") else x)
    nfix = int((h0 != d["horse"]).sum())
    # ★★★並べ直しが必須。`to_model` の最後は sort_values(["horse","date"]) なので、
    # 　正規化を後から当てると「2021100478」と「21100478」が別グループとして並んだままになり、
    # 　`build_features` の `cumcount()` が行順で数えるため n_prior=0 になる（実際に踏んだ）。
    d = d.sort_values(["horse", "date"], kind="mergesort").reset_index(drop=True)
    print(f"★馬IDの表記を統一: **{nfix:,}行**（10桁「20」＋8桁 → 8桁）"
          f"　→ ★**(horse, date) で並べ直した**")
    n0 = len(d)
    d = d.drop_duplicates(subset=["raceid", "umaban"], keep="first").reset_index(drop=True)
    print(f"　**{n0:,}行 → 重複を落として {len(d):,}行**"
          f"　（最終日 **{d['date'].max().date()}**）")
    f = F.build_features(d)
    keep = (f["n_prior"] >= 1) & d["odds"].notna() & (d["odds"] > 0)
    d, f = d[keep].reset_index(drop=True), f[keep].reset_index(drop=True)
    y = (d["finish"] <= 3).astype(int).to_numpy()
    fx, _ = F.encode_categoricals(f)
    fx = add_odds_features(fx, d["odds"].to_numpy(float), d["raceid"].to_numpy())
    pred = wf_predict(d, fx, y, 3)
    msk = ~np.isnan(pred)
    cols = ["raceid", "umaban", "odds", "date", "name"]
    cols = [c for c in cols if c in d.columns]
    sub = d.loc[msk, cols + (["finish"] if check else [])].copy()
    sub["p"] = pred[msk]
    have = sorted({str(k) for k in boards})
    sub["rid"] = sub["raceid"].astype(str)
    sub = sub[sub["rid"].isin(have)]
    frm = None
    for i, a in enumerate(sys.argv):
        if a == "--from" and i + 1 < len(sys.argv):
            frm = pd.Timestamp(sys.argv[i + 1])
    if frm is not None:
        days = sorted(x for x in sub["date"].unique() if x >= frm.to_datetime64())
        print(f"★**{len(days)}開催日**を日別に集計する"
              f"（{pd.Timestamp(days[0]).date()}〜{pd.Timestamp(days[-1]).date()}）\n")
        return per_day(sub, days, races, pays, boards, pay_of, check)
    if args:
        day = pd.Timestamp(args[0])
    else:
        day = sub["date"].max()
        print(f"⚠**日付を指定しなかったので、板がある最後の日を使う**")
    print(f"★対象日: **{day.date()}**\n")
    tgt = sub[sub["date"] == day]
    if not len(tgt):
        print("⚠**その日のデータが無い**")
        return

    out = []
    for rid, g in tgt.groupby("raceid"):
        rid = str(rid)
        bd = boards.get(rid)
        if bd is None:
            continue
        r = races.get(rid)
        if r is not None:
            nums = {u for u, _, _ in r["horses"]}
            gg = g[g["umaban"].astype(int).isin(nums)]
        else:
            gg = g
        ub = gg["umaban"].astype(int).to_numpy()
        if len(gg) < MIN_HORSES or not all(int(u) in bd for u in ub):
            continue
        if check and rid not in races and rid not in pays:
            continue
        od = gg["odds"].to_numpy(float)
        pv = gg["p"].to_numpy(float)
        if not np.isfinite(od).all() or (od <= 0).any() or pv.sum() <= 0:
            continue
        pn = pv / pv.sum() * NPLACE
        qp, _ = qpool([bd[int(u)] for u in ub], "harm")
        gap = pn - qp
        order_p = [int(u) for u in ub[np.argsort(-pv, kind="mergesort")]]
        nm = {int(u): (str(s) if "name" in gg.columns else "")
              for u, s in zip(gg["umaban"].astype(int),
                              gg["name"] if "name" in gg.columns else ub)}
        fin = ({int(u): int(x) for u, x in zip(gg["umaban"].astype(int),
                                               gg["finish"].astype(int))}
               if check else {})
        order_g = [int(u) for u in ub[np.argsort(-gap, kind="mergesort")]]
        rec = {"rid": rid, "A": None, "B": None, "C": None}
        cC = np.where((pn >= PN_FLOOR) & (gap >= GAP_A) & (od >= LFIX))[0]
        if len(cC):
            i = int(cC[int(np.argmax(pn[cC]))])
            ax = int(ub[i])
            rec["C"] = {"u": ax, "od": float(od[i]), "pn": float(pn[i]),
                        "gap": float(gap[i]),
                        "P": [u for u in order_p if u != ax][:5],
                        "G": [u for u in order_g if u != ax][:5]}
        cA = np.where((pn >= PN_FLOOR) & (gap >= GAP_A))[0]
        if len(cA):
            i = int(cA[int(np.argmax(pn[cA]))])
            rec["A"] = {"u": int(ub[i]), "od": float(od[i]),
                        "pn": float(pn[i]), "gap": float(gap[i])}
        cB = np.where((pn >= PN_FLOOR) & (gap >= GAP_B))[0]
        if len(cB):
            i = int(cB[int(np.argmax(gap[cB]))])
            ax = int(ub[i])
            himo = [u for u in order_p if u != ax][:2]
            if len(himo) == 2:
                rec["B"] = {"u": ax, "od": float(od[i]), "pn": float(pn[i]),
                            "gap": float(gap[i]), "h": himo}
        rec["nm"], rec["fin"] = nm, fin
        out.append(rec)

    nA = sum(1 for x in out if x["A"])
    nB = sum(1 for x in out if x["B"])
    print(f"★この日のレース **{len(out)}本**"
          f"　→ ★**推奨A（複勝・ズレ≥{GAP_A}）{nA}本** / **推奨B（三連単・ズレ≥{GAP_B}）{nB}本**\n")

    print(f"■ ★★**推奨A: 複勝1点**（**ズレ≥{GAP_A}**・★実測95.9% [91.6,100.2]）")
    print(f"{'レース':<14}{'軸':>4}{'馬名':<16}{'オッズ':>8}{'推奨度':>8}{'ズレ':>8}"
          + ("{:>6}{:>10}".format("着", "払戻") if check else ""))
    tA = [0.0, 0.0, 0]
    for x in out:
        if not x["A"]:
            continue
        a = x["A"]
        line = (f"{x['rid']:<14}{a['u']:>4}{x['nm'].get(a['u'],''):<16}"
                f"{a['od']:>7.1f}倍{a['pn']:>8.3f}{a['gap']:>8.3f}")
        if check:
            v = pay_of(x["rid"], "複勝", [a["u"]])
            tA[0] += 100.0; tA[1] += (v or 0.0); tA[2] += 1
            line += f"{x['fin'].get(a['u'],0):>6}{(f'{int(v):,}円' if v else '−'):>10}"
        print(line)

    print(f"\n■ ★★**推奨B: 三連単2点**（**ズレ≥{GAP_B}・軸1着固定・紐はモデル上位2頭**"
          f"・★実測103.1% [79.4,126.7]）")
    print(f"{'レース':<14}{'軸':>4}{'馬名':<16}{'オッズ':>8}{'ズレ':>8}{'紐1':>5}{'紐2':>5}"
          + ("{:>6}{:>6}{:>6}{:>12}".format("軸着", "紐1", "紐2", "払戻") if check else ""))
    tB = [0.0, 0.0, 0]
    tU = [0.0, 0.0, 0]
    tS = [0.0, 0.0, 0]
    for x in out:
        if not x["B"]:
            continue
        b = x["B"]
        line = (f"{x['rid']:<14}{b['u']:>4}{x['nm'].get(b['u'],''):<16}"
                f"{b['od']:>7.1f}倍{b['gap']:>8.3f}{b['h'][0]:>5}{b['h'][1]:>5}")
        if check:
            vs = [pay_of(x["rid"], "三連単", [b["u"], b["h"][0], b["h"][1]]),
                  pay_of(x["rid"], "三連単", [b["u"], b["h"][1], b["h"][0]])]
            v = sum(v for v in vs if v)
            tB[0] += 200.0; tB[1] += v; tB[2] += 1
            vu = pay_of(x["rid"], "馬連", [b["u"], b["h"][0]]) or 0.0
            tU[0] += 100.0; tU[1] += vu; tU[2] += 1
            vs3 = pay_of(x["rid"], "三連複", sorted([b["u"], b["h"][0], b["h"][1]])) or 0.0
            tS[0] += 100.0; tS[1] += vs3; tS[2] += 1
            line += (f"{x['fin'].get(b['u'],0):>6}{x['fin'].get(b['h'][0],0):>6}"
                     f"{x['fin'].get(b['h'][1],0):>6}"
                     f"{(f'{int(v):,}円' if v else '−'):>12}")
        print(line)

    nC = sum(1 for x in out if x["C"])
    print(f"\n{'='*104}")
    print(f"■ ★★★★**推奨C: 穴側の候補4つ**（**(210)〜(214)**・"
          f"**軸=pn≥{PN_FLOOR} かつ ズレ≥{GAP_A} かつ ★単勝≥{LFIX}倍 の中で pn最大**）")
    print(f"★**この日の該当 {nC}本**　⚠**11年の数字は98マスから選んだもので、下端は全部0を割る**")
    tC = {c[0]: [0.0, 0.0, 0, 0] for c in CAND}
    for x in out:
        if not x["C"]:
            continue
        c = x["C"]
        print(f"\n　★**{x['rid']}**　軸 **{c['u']}番 {x['nm'].get(c['u'],'')}**"
              f"　**{c['od']:.1f}倍**　推奨度 {c['pn']:.3f}　ズレ {c['gap']:.3f}"
              + (f"　→ **{c and x['fin'].get(c['u'],0)}着**" if check else ""))
        print(f"　　紐P（人気側）{c['P'][:4]}　/　紐G（穴側）{c['G'][:4]}")
        for lab, hk, kind, npt, roi11, yr11 in CAND:
            tk = bet_tickets(kind, npt, c["u"], c[hk])
            if tk is None:
                print(f"　　{lab:<18} ⚠**紐が足りず組めない**")
                continue
            line = f"　　{lab:<18} " + " / ".join("-".join(str(z) for z in sel)
                                                 for _, sel in tk)
            if check:
                vs = [pay_of(x["rid"], k2, sel) for k2, sel in tk]
                if any(v is None for v in vs):
                    print(line + "　⚠**払戻が引けない**")
                    continue
                v = sum(vs)
                t = tC[lab]
                t[0] += 100.0 * len(tk); t[1] += v; t[2] += 1; t[3] += 1 if v else 0
                line += f"　→ **{(f'{int(v):,}円' if v else '−')}**"
            print(line)

    if check:
        print(f"\n{'='*104}")
        print(f"■ ★★★★**推奨Cの答え合わせ**")
        print(f"{'買い方':<20}{'点':>4}{'本数':>6}{'的中':>6}{'買った額':>11}{'払戻':>11}"
              f"{'★収支':>11}{'★ROI':>9}{'11年':>9}{'必要年数':>10}")
        for lab, hk, kind, npt, roi11, yr11 in CAND:
            t = tC[lab]
            if not t[2]:
                print(f"{lab:<20}{npt:>4}{'—':>6}")
                continue
            print(f"{lab:<20}{npt:>4}{t[2]:>6}{t[3]:>6}{t[0]:>10,.0f}円{t[1]:>10,.0f}円"
                  f"{t[1]-t[0]:>+10,.0f}円{100*t[1]/t[0]:>8.1f}%{roi11:>8.1f}%{yr11:>9}年")
        print(f"⚠⚠**この表で「良かった／悪かった」を読まないこと**——"
              f"**本数が少なすぎる**（**11年1,383本でも下端は0を割っている**）。")
        print(f"⚠★**これは前向きの検定ではない**——"
              f"**直近のレースは、98マスを選ぶのに使った11年の中に入っている**。")

    if check:
        print(f"\n{'='*96}")
        print(f"■ ★★**答え合わせ**")
        print(f"{'買い方':<20}{'点数':>5}{'本数':>6}{'買った額':>11}{'払戻':>11}"
              f"{'★収支':>11}{'★ROI':>9}{'11年の実測':>12}")
        for nm2, t, npt, lg in (("★推奨A 複勝", tA, 1, 95.9),
                                ("★推奨B 三連単", tB, 2, 103.1),
                                ("（参考）馬連 紐1", tU, 1, 90.0),
                                ("（参考）三連複", tS, 1, 91.2)):
            if not t[2]:
                continue
            roi = 100.0 * t[1] / t[0]
            print(f"{nm2:<20}{npt:>5}{t[2]:>6}{t[0]:>10,.0f}円{t[1]:>10,.0f}円"
                  f"{t[1]-t[0]:>+10,.0f}円{roi:>8.1f}%{lg:>11.1f}%")
        print(f"\n⚠⚠**1日の数字は11年の実測の推定にはならない**（判定基準5）。"
              f"**推奨Bは的中2.63%＝38レースに1回**。")
    print("\n⚠**枠連の運用には触れない**。**設定変更は提案しない**。")


if __name__ == "__main__":
    sys.exit(main() or 0)
