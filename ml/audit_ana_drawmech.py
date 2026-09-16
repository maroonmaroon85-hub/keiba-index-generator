"""(246) ★★★**軸が外枠寄りなのは、どの条件のせいか** —— (245)④の機構を分解する

★★**動機（2026-09-16）**: ★**(245)で ④ 軸のalign平均 = +0.0561 [+0.0376,+0.0756] とゼロを外した**
　（**平均で頭数の5.6%ぶん外寄り**）。★**だが ③ corr(align, gap) はゼロを含んだ**。
　→ ★★**ズレ経由ではない。★では何が外枠を拾っているのか**。

■ ★**軸の3条件**: `pn ≥ 0.15` かつ `ズレ ≥ 0.15` かつ `単勝 ≥ 10.0倍` の中で **pn最大の1頭**
　★**見立て（⚠未測定）**: ★**「単勝10.0倍以上」が拾っている**
　　（★**外枠不利なら人気が落ちてオッズが上がる**）。★**これを確かめる**。

■ ⚠★★**これは「絞ればROIが動くか」ではない。★記述だけ。マスを増やさない**
　★**買い目も軸も一切動かさない**。★**ROIを一度も計算しない**。
　→ ⚠**(182)(185)(187)(190)の135マス・(188)の家族には入らない**。

────────────────────────────────────────────────────────────
★★★ 事前登録（2026-09-16・**結果を見る前にコミットする**）
────────────────────────────────────────────────────────────

■ ★**母集団**: **(236)(240)(241)(245)と同じ**。★**align = 馬番/頭数 − 0.5**（(245)と同じ）。

■ ★★**測るもの（すべて align の平均・レース内で中心化せず生のまま）**
| # | 対象 | ★**意味** |
|---|---|---|
| **⓪** | ★**全馬** | ★**基準線**（**定義上ほぼ0だが頭数の偶奇でわずかにずれる**） |
| ★**①** | ★**単勝 ≥ 10.0倍 の馬**（**他の条件は課さない**） | ★**オッズ条件だけで外枠に寄るか** |
| **②** | **pn ≥ 0.15 の馬**（同上） | **モデル条件だけで寄るか**（⚠**pnは馬番を持たないので≈0のはず**） |
| **③** | **ズレ ≥ 0.15 の馬**（同上） | **ズレ条件だけで寄るか**（⚠**(245)③がゼロなので≈0のはず**） |
| ★**④** | ★**3条件すべて（軸候補）** | ★**組み合わせで寄るか** |
| ★★**⑤** | ★★**軸（④の中で pn最大の1頭）** | ★**(245)④と一致するはず＝内部対照** |
　★**あわせて ★単勝オッズ帯ごとの align 平均**（**[1,3) [3,5) [5,10) [10,20) [20,50) [50,∞)**）
　　→ ★**「人気薄ほど外枠」が実在するかを直接見る**。

■ ★★★★**主判定（★先に決める・これ1つ）**
　★★**① 単勝≥10.0倍の馬の align 平均が、⑤ 軸の +0.0561 を説明できるか**。
| ★**返り値** | ★**読み** |
|---|---|
| ★**① が +0.0561 と同程度（99%CIが重なる）** | ★★**オッズ条件が外枠を拾っている**。**軸固有の性質ではない** |
| ⚠**① が明らかに小さい** | ⚠**オッズでは説明できない。★②③④か「pn最大」の選択が効いている** |

■ ★★ゲート2（判定基準42）——**何が返れば「機構が分からない」か**
　★**①②③が全部ゼロ近傍なら、★偏りは3条件の組み合わせか「pn最大」の選択から来る**
　　→ ⚠**そのときは「説明できなかった」と書く**。★**後から別の説明を探さない**。

■ ★★★内部対照（**決定的・最初に見る**）
　**1.** ★**年別レース数が (236)(240)(241)(245) と一致**
　**2.** ★★**⑤ 軸の align 平均が (245)④ と一致**: ★**+0.0561（±0.0005）・軸1,398本**
　⚠**どちらかが落ちたら読まない**（判定基準32）。

■ ⚠★**先に書いておく限界**
　★**align は馬番ベース**。⚠**枠番ではない**（**枠連側(177)も馬番ベース**）。
　★**頭数が少ないレースでは align の粒度が粗い**（**8頭なら0.125刻み**）。
　★**これは通年の平均**。⚠**日ごと・場ごとの馬場差は見ない**。
　⚠★**「外枠が不利か」は測っていない**——**測るのは「軸がどこから来ているか」だけ**。

■ 予想（⚠**当てにしない**・★私は13回外した）
　★**① がほぼ全部を説明すると見る**（**オッズ条件が犯人**）。
　★**②③はゼロ近傍と見る**（**pnは馬番を持たず、(245)③もゼロだったので**）。

実行: python3 ml/audit_ana_drawmech.py    自己テスト: python3 ml/audit_ana_drawmech.py --selftest
"""
import sys

import numpy as np

sys.path.insert(0, "ml")
import features as F
from audit_crosspool import load_races
from audit_ana_odds import MIN_HORSES
from audit_ana_board import NPLACE, load_fuku_boards, qpool
from audit_ana_band import PN_FLOOR
from audit_ana_hole import GAP
from audit_ana_fix import LFIX
from audit_ana_marg import wf_predict
from audit_ana_draw import align_of, boot_ci
from train_prod import add_odds_features

KNOWN_R = [2842, 2807, 2773, 2767, 2786, 2813, 2791, 2780, 2723, 2758, 1630]
KNOWN_AX, KNOWN_N, ATOL = 0.0561, 1398, 0.0005     # ★(245)④
BANDS = [1.0, 3.0, 5.0, 10.0, 20.0, 50.0, float("inf")]


def band_lab(i):
    hi = "∞" if BANDS[i + 1] == float("inf") else f"{BANDS[i+1]:.0f}"
    return f"[{BANDS[i]:.0f},{hi})"


def selftest():
    ok = True
    print("★★★★主判定: **① 単勝≥10.0倍の馬の align 平均が、⑤ 軸の +0.0561 を説明できるか**")
    print("　★同程度 → **オッズ条件が外枠を拾っている** ／ ⚠小さい → **オッズでは説明できない**")
    print("★★ゲート2: **①②③が全部ゼロ近傍なら「説明できなかった」と書く。後から別の説明を探さない**")
    print(f"★★★内部対照: **年別レース数** かつ **⑤ 軸の align = {KNOWN_AX}（±{ATOL}）・{KNOWN_N:,}本**")
    a = align_of([1, 2, 3, 4, 5, 6, 7, 8])
    print(f"★alignの検算（8頭）: {np.round(a, 3)}　★**(245)と同じ関数を import している**")
    ok &= a[0] < 0 < a[-1]
    print(f"★オッズ帯: " + " / ".join(band_lab(i) for i in range(len(BANDS) - 1)))
    ok &= len(BANDS) == 7 and BANDS[3] == LFIX
    print(f"　★**10.0倍が帯の境目＝軸の条件と一致** {'★OK' if BANDS[3] == LFIX else '⚠NG'}")
    lo, hi = boot_ci([0.05] * 100)
    print(f"★CIの検算: 全部0.05 → [{lo:.4f}, {hi:.4f}]　{'★OK' if abs(hi-lo) < 1e-9 else '⚠NG'}")
    ok &= abs(hi - lo) < 1e-9
    print("⚠★**これは「絞ればROIが動くか」ではない。★記述だけ。ROIを一度も計算しない**")
    print("⚠**買い目も軸も動かさない。マスを増やさない**")
    print("★自己テスト: " + ("全部OK" if ok else "⚠NG"))
    return 0 if ok else 1


def main():
    print("(246) ★★★**軸が外枠寄りなのは、どの条件のせいか**\n")
    races = {r["rid"]: r for r in load_races()}
    boards = load_fuku_boards()
    d = F.to_model(F.load_files())
    f = F.build_features(d)
    keep = (f["n_prior"] >= 1) & d["odds"].notna() & (d["odds"] > 0)
    d, f = d[keep].reset_index(drop=True), f[keep].reset_index(drop=True)
    y = (d["finish"] <= 3).astype(int).to_numpy()
    fx, _ = F.encode_categoricals(f)
    fx = add_odds_features(fx, d["odds"].to_numpy(float), d["raceid"].to_numpy())
    pred = wf_predict(d, fx, y, 3)
    m = ~np.isnan(pred)
    sub = d.loc[m, ["raceid", "umaban", "odds", "date"]].copy()
    sub["p"] = pred[m]

    K = ["all", "odds", "pn", "gap", "cand", "axis"]
    V = {k: [] for k in K}
    BV = [[] for _ in range(len(BANDS) - 1)]
    nR = {}
    for rid, g in sub.groupby("raceid"):
        rid = str(rid)
        r, bd = races.get(rid), boards.get(rid)
        if r is None or bd is None:
            continue
        nums = {u for u, _, _ in r["horses"]}
        gg = g[g["umaban"].astype(int).isin(nums)]
        ub = gg["umaban"].astype(int).to_numpy()
        if len(gg) < MIN_HORSES or not all(int(u) in bd for u in ub):
            continue
        od, pv = gg["odds"].to_numpy(float), gg["p"].to_numpy(float)
        if not np.isfinite(od).all() or (od <= 0).any() or pv.sum() <= 0:
            continue
        pn = pv / pv.sum() * NPLACE
        qp, _ = qpool([bd[int(u)] for u in ub], "harm")
        gp = pn - qp
        al = align_of(ub)
        yr = int(gg["date"].iloc[0].year)
        nR[yr] = nR.get(yr, 0) + 1
        V["all"] += list(al)
        for k, sel in (("odds", od >= LFIX), ("pn", pn >= PN_FLOOR), ("gap", gp >= GAP),
                       ("cand", (pn >= PN_FLOOR) & (gp >= GAP) & (od >= LFIX))):
            V[k] += list(al[sel])
        c = np.where((pn >= PN_FLOOR) & (gp >= GAP) & (od >= LFIX))[0]
        if len(c):
            V["axis"].append(float(al[int(c[int(np.argmax(pn[c]))])]))
        for j in range(len(BANDS) - 1):
            s2 = (od >= BANDS[j]) & (od < BANDS[j + 1])
            BV[j] += list(al[s2])

    ys = sorted(k for k in nR if k >= 2016)
    got = [nR[u] for u in ys]
    ok1 = got == KNOWN_R
    ax = float(np.mean(V["axis"]))
    ok2 = abs(ax - KNOWN_AX) <= ATOL and len(V["axis"]) == KNOWN_N
    print("★★★内部対照（**最初に見る**）")
    print(f"　1. 年別レース数 {'★一致' if ok1 else '⚠ずれた'}")
    print(f"　2. ⑤ 軸の align = {ax:+.4f}（{KNOWN_AX}）／ {len(V['axis']):,}本（{KNOWN_N:,}）"
          f"　{'★一致' if ok2 else '⚠ずれた'}")
    if not (ok1 and ok2):
        print("⚠⚠**対照が落ちた。読まない**（判定基準32）。")
        return

    print(f"\n{'':<30}{'頭数':>10}{'★align平均':>12}{'99%CI':>26}")
    lab = {"all": "⓪ 全馬（基準線）", "odds": "★① 単勝≥10.0倍だけ",
           "pn": "② pn≥0.15だけ", "gap": "③ ズレ≥0.15だけ",
           "cand": "★④ 3条件すべて（軸候補）", "axis": "★★⑤ 軸（pn最大）"}
    res = {}
    for k in K:
        v = V[k]
        lo, hi = boot_ci(v)
        res[k] = (float(np.mean(v)), lo, hi)
        z = "★ゼロを含む" if lo <= 0 <= hi else "⚠**ゼロを外す**"
        print(f"{lab[k]:<28}{len(v):>10,}{np.mean(v):>+12.4f}   [{lo:>+7.4f},{hi:>+7.4f}] {z}")

    print(f"\n★**単勝オッズ帯ごとの align 平均**（★「人気薄ほど外枠」が実在するか）")
    print(f"{'帯':<14}{'頭数':>10}{'★align平均':>12}{'99%CI':>26}")
    for j in range(len(BANDS) - 1):
        lo, hi = boot_ci(BV[j])
        z = "★ゼロを含む" if lo <= 0 <= hi else "⚠**ゼロを外す**"
        print(f"{band_lab(j):<14}{len(BV[j]):>10,}{np.mean(BV[j]):>+12.4f}"
              f"   [{lo:>+7.4f},{hi:>+7.4f}] {z}")

    o_m, o_lo, o_hi = res["odds"]
    a_m, a_lo, a_hi = res["axis"]
    overlap = not (o_hi < a_lo or a_hi < o_lo)
    print(f"\n■ ★★★★**主判定: ① が ⑤ を説明できるか**")
    print(f"　★**① 単勝≥10.0倍だけ = {o_m:+.4f} [{o_lo:+.4f}, {o_hi:+.4f}]**")
    print(f"　★**⑤ 軸　　　　　　 = {a_m:+.4f} [{a_lo:+.4f}, {a_hi:+.4f}]**")
    print(f"　→ ★**99%CIが{'重なる＝オッズ条件で説明できる' if overlap else '重ならない＝オッズでは説明できない'}**")
    zs = [k for k in ("odds", "pn", "gap") if res[k][1] <= 0 <= res[k][2]]
    if len(zs) == 3:
        print("　⚠⚠**①②③が全部ゼロを含む＝機構は説明できなかった**（ゲート2）。"
              "★**後から別の説明を探さない**")
    print(f"\n⚠**これは「絞ればROIが動くか」ではない。★ROIを一度も計算していない**")
    print("⚠**枠連の運用には触れない**。**設定変更は提案しない**。")


if __name__ == "__main__":
    sys.exit(selftest() if "--selftest" in sys.argv else (main() or 0))
