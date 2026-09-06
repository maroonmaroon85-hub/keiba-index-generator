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
■ ⚠**2026-08-09 は結果データ自体が欠けている**（36レース中13レース・別セッションの報告）。
　★**「その日は推奨が少なかった」と読まないこと**——**データの欠け**。

実行:
　python3 ml/reco_ana.py [YYYY-MM-DD]        ★推奨だけ（結果を見ない）
　python3 ml/reco_ana.py [YYYY-MM-DD] --check  ★答え合わせ
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
import audit_ana_marg as _M
# ★★このスクリプトは DSnk を足して行数が変わるので、共有キャッシュを上書きしない
_M.CACHE = "data/cache/wf_pred_reco.npz"
from audit_ana_marg import wf_predict
from audit_ana_board import NPLACE, load_fuku_boards, qpool
from audit_ana_band import PN_FLOOR
from train_prod import add_odds_features

GAP_A, GAP_B = 0.15, 0.02


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    check = "--check" in sys.argv
    print("(207) ★**穴馬の線の推奨**" + ("　★★【答え合わせ】" if check else "　★推奨のみ（結果は見ない）"))
    races = {r["rid"]: r for r in load_races()}
    boards = load_fuku_boards()
    ds = [p for p in sorted(glob.glob("data/nk/DSnk*.CSV")) if os.path.getsize(p) > 0]
    print(f"★過去走: ルート直下の *.CSV ＋ **data/nk/DSnk*.CSV {len(ds)}本**")
    frames = [F.load_files()] + [
        pd.read_csv(p, header=None, encoding="shift_jis", encoding_errors="replace",
                    dtype=str, keep_default_na=False) for p in ds]
    d = F.to_model(pd.concat(frames, ignore_index=True))
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
        r = races.get(str(rid))
        bd = boards.get(str(rid))
        if r is None or bd is None:
            continue
        nums = {u for u, _, _ in r["horses"]}
        if len(nums) < MIN_HORSES:
            continue
        gg = g[g["umaban"].astype(int).isin(nums)]
        ub = gg["umaban"].astype(int).to_numpy()
        if len(gg) < MIN_HORSES or not all(int(u) in bd for u in ub):
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
        rec = {"rid": str(rid), "A": None, "B": None}
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
            v = payoff(races[x["rid"]], "複勝", [a["u"]])
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
            r = races[x["rid"]]
            vs = [payoff(r, "三連単", [b["u"], b["h"][0], b["h"][1]]),
                  payoff(r, "三連単", [b["u"], b["h"][1], b["h"][0]])]
            v = sum(v for v in vs if v)
            tB[0] += 200.0; tB[1] += v; tB[2] += 1
            vu = payoff(r, "馬連", [b["u"], b["h"][0]]) or 0.0
            tU[0] += 100.0; tU[1] += vu; tU[2] += 1
            vs3 = payoff(r, "三連複", sorted([b["u"], b["h"][0], b["h"][1]])) or 0.0
            tS[0] += 100.0; tS[1] += vs3; tS[2] += 1
            line += (f"{x['fin'].get(b['u'],0):>6}{x['fin'].get(b['h'][0],0):>6}"
                     f"{x['fin'].get(b['h'][1],0):>6}"
                     f"{(f'{int(v):,}円' if v else '−'):>12}")
        print(line)

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
