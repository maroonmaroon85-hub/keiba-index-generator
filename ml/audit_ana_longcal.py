"""(240) ★★★★**市場は「穴の帯」だけ上手くなったのか** —— 単勝10倍以上に絞って校正を年別に測る

★★**動機（2026-09-10・利用者の指定）**:
　★**(236)で測った AUC は全馬だった**。**市場の伸びは +0.0078・人気1番の的中率は ±0.0pt**。
　⚠**だが「穴の帯だけ市場が上手くなった」なら、全馬のAUCではほとんど動かない**。
　★**利用者の指定「10倍以上の帯に絞った市場の校正を測ろう」**。

■ ★★**(236)との違い（★これが測定の主旨）**
| | (236) | ★**この測定(240)** |
|---|---|---|
| **対象の馬** | **全馬** | ★**単勝オッズ ≥ 10.0倍 の馬だけ** |
| **測るもの** | ★**順位づけ（AUC）** | ★★**校正（予測確率 vs 実際の複勝率）＋帯の中のAUC** |
　★**順位づけと校正は別物**——**順位が正しくても確率の水準がずれていることはある**。
　★**穴の妙味は「市場が穴を過小/過大評価している」＝★校正のずれ**から来る。

────────────────────────────────────────────────────────────
★★★ 事前登録（2026-09-10・**結果を見る前にコミットする**）
────────────────────────────────────────────────────────────

■ ★**母集団**: **(236)と同じ**（**複勝の板があり、出走頭数が足りるレース**）。
　★**その中で 単勝オッズ ≥ 10.0倍 の馬だけを見る**。★**軸の定義は使わない**（**帯だけで切る**）。

■ ★★**測るもの（すべて年別・帯の中）**
| # | 量 | ★**「市場が穴で上手くなった」ならどう動くか** |
|---|---|---|
| **①** | **帯の頭数**（1レースあたり） | ★**内部対照**（(235)の①と一致するはず） |
| **②** | ★**実際の複勝率** | — |
| **③** | ★**市場の予測 qp の平均** | — |
| ★**④** | ★★**市場の校正ずれ ＝ ② − ③** | ★**ゼロに近づけば「市場が上手くなった」** |
| **⑤** | **モデルの予測 pn の平均** | — |
| ★**⑥** | ★**モデルの校正ずれ ＝ ② − ⑤** | — |
| ★**⑦** | ★**帯の中の AUC(qp)** | ★**上がれば「市場が穴の順位づけも上手くなった」** |
| **⑧** | **帯の中の AUC(pn)** | — |
| ⑨ | **帯の中央オッズ** | ⚠**帯の中身が動いていないかの確認** |

■ ★★★★**主判定（★先に決める・これ1つ）**
　★★**④（市場の校正ずれ）が2016→2026でゼロに近づいたか**。
　★**近づいた → 「市場が穴で上手くなった」を支持**。
　★**横ばい → 支持しない**。★**(236)の「市場はほぼ動いていない」が帯の中でも成り立つ**。

■ ★★ゲート2（判定基準42）——**この測定は何を返せば「市場は穴で上手くなっていない」か**
　★**④が年で横ばいの線（絶対値が縮まない）を返せば、支持しない**。
　★**⑦（帯の中のAUC(qp)）も併記する**——**校正と順位づけの両方で見る**。
　⚠**④と⑦が食い違ったら、どちらも結論にしない。「校正は動いたが順位は動かない」等と書く**。

■ ★★★内部対照（**決定的・これを最初に見る**）
　★**① 年別の対象レース数が (236)(235) と一致すること**:
　　**2842 / 2807 / 2773 / 2767 / 2786 / 2813 / 2791 / 2780 / 2723 / 2758 / 1630**
　★**かつ 帯の頭数が (235)の① と一致**（**2016: 10.377頭 / 2026: 9.696頭・±0.02**）。
　⚠**ずれたら読まない**（判定基準32）。★**(238)で9回目を踏んだので、ここを最初に置く**。

■ ⚠★**先に書いておく限界**
　★**qp の作り方（複勝板の[下限,上限]の調和平均・Σ=3に正規化）に校正ずれは依存する**。
　　⚠**中点を使えば違う値になる。この測定は「この qp の定義のもとでの校正」しか言わない**。
　★**帯の中身が年で動けば、平均で見た校正ずれも動く**→★**⑨で確認する**。
　★**2026は1,630レース＝途中まで**（**他年の6割**）。⚠**1標本のばらつき**（判定基準43）。
　⚠**これは探索ではない。マスも軸も買い目も動かさない。運用は変えない**。

■ 予想（⚠**当てにしない**・★**私は12回外した**）
　⚠★**④は横ばいだと見る**（**(236)で人気1番の的中率が±0.0ptだったのと整合するはず**）。
　★**つまり「市場が穴で上手くなった」は支持されないと見る**。

実行: python3 ml/audit_ana_longcal.py    自己テスト: python3 ml/audit_ana_longcal.py --selftest
"""
import sys

import numpy as np

sys.path.insert(0, "ml")
import features as F
from audit_crosspool import load_races
from audit_ana_odds import MIN_HORSES
from audit_ana_board import NPLACE, load_fuku_boards, qpool
from audit_ana_fix import LFIX
from audit_ana_marg import wf_predict
from audit_ana_swallow import auc
from train_prod import add_odds_features

KNOWN_R = [2842, 2807, 2773, 2767, 2786, 2813, 2791, 2780, 2723, 2758, 1630]
KNOWN_N = {2016: 10.377, 2026: 9.696}     # ★(235)の① 帯の頭数
NTOL = 0.02


def selftest():
    ok = True
    print("★★★★主判定: **市場の校正ずれ ④=実際−qp が 2016→2026 でゼロに近づいたか**")
    print("★★ゲート2: **④が横ばいなら「市場が穴で上手くなった」を支持しない**")
    print("　★**⑦ 帯の中のAUC(qp) も併記。食い違ったらどちらも結論にしない**")
    print(f"★★★内部対照（**最初に見る**）: 年別レース数 {KNOWN_R}")
    print(f"　★かつ 帯の頭数 2016={KNOWN_N[2016]} / 2026={KNOWN_N[2026]}（±{NTOL}）")
    y = np.array([0, 0, 1, 1])
    print(f"★AUCの検算: 完全分離 {auc([1, 2, 3, 4], y):.3f} / 逆 {auc([4, 3, 2, 1], y):.3f}"
          f" / 同点 {auc([1, 1, 1, 1], y):.3f}")
    ok &= abs(auc([1, 2, 3, 4], y) - 1.0) < 1e-9 and abs(auc([1, 1, 1, 1], y) - 0.5) < 1e-9
    # ★校正ずれの検算: 予測0.20・実際0.25 なら +0.05
    pr, ac = np.full(100, 0.20), np.array([1] * 25 + [0] * 75)
    b = float(ac.mean() - pr.mean())
    print(f"★校正ずれの検算: 予測0.20・実際0.25 → {b:+.3f}（+0.050）"
          f"　{'★OK' if abs(b - 0.05) < 1e-9 else '⚠NG'}")
    ok &= abs(b - 0.05) < 1e-9
    print(f"★帯の定義: **単勝オッズ ≥ {LFIX}倍**（★軸の定義は使わない）")
    print("⚠**探索ではない。マスも軸も買い目も動かさない。運用は変えない**")
    print("★自己テスト: " + ("全部OK" if ok else "⚠NG"))
    return 0 if ok else 1


def main():
    print("(240) ★★★★**市場は「穴の帯」だけ上手くなったのか**\n")
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
    sub = d.loc[m, ["raceid", "umaban", "odds", "date", "finish"]].copy()
    sub["p"] = pred[m]

    A = {}
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
        od = gg["odds"].to_numpy(float)
        pv = gg["p"].to_numpy(float)
        if not np.isfinite(od).all() or (od <= 0).any() or pv.sum() <= 0:
            continue
        pn = pv / pv.sum() * NPLACE
        qp, _ = qpool([bd[int(u)] for u in ub], "harm")
        hit = (gg["finish"].astype(int).to_numpy() <= 3).astype(int)
        yr = int(gg["date"].iloc[0].year)
        a = A.setdefault(yr, {"R": 0, "n": 0, "hs": [], "qs": [], "ps": [], "od": []})
        a["R"] += 1
        sel = od >= LFIX                      # ★帯: 単勝10倍以上
        a["n"] += int(sel.sum())
        a["hs"] += list(hit[sel]); a["qs"] += list(qp[sel])
        a["ps"] += list(pn[sel]); a["od"] += list(od[sel])

    ys = sorted(k for k in A if k >= 2016)
    got = [A[u]["R"] for u in ys]
    nn = {u: A[u]["n"] / A[u]["R"] for u in ys}
    ok1 = got == KNOWN_R
    ok2 = all(abs(nn[u] - v) <= NTOL for u, v in KNOWN_N.items() if u in nn)
    print("★★★内部対照（**最初に見る**）")
    print(f"　年別レース数 {got}")
    print(f"　既知　　　　 {KNOWN_R}　{'★一致' if ok1 else '⚠ずれた'}")
    print(f"　帯の頭数 2016={nn.get(2016, float('nan')):.3f}（{KNOWN_N[2016]}） / "
          f"2026={nn.get(2026, float('nan')):.3f}（{KNOWN_N[2026]}）"
          f"　{'★一致' if ok2 else '⚠ずれた'}")
    if not (ok1 and ok2):
        print("⚠⚠**対照が落ちた。読まない**（判定基準32）。")
        return

    def row(lab, fn, fmt="{:>8.3f}"):
        print(f"{lab:<26}" + "".join(fmt.format(fn(A[u])) for u in ys))

    print("\n" + " " * 26 + "".join(f"{u:>8}" for u in ys))
    row("★① 帯の頭数/レース", lambda a: a["n"] / a["R"])
    row("★② 実際の複勝率", lambda a: float(np.mean(a["hs"])))
    row("③ 市場の予測 qp", lambda a: float(np.mean(a["qs"])))
    row("★★④ 市場の校正ずれ ②−③", lambda a: float(np.mean(a["hs"]) - np.mean(a["qs"])),
        "{:>+8.3f}")
    row("⑤ モデルの予測 pn", lambda a: float(np.mean(a["ps"])))
    row("★⑥ モデルの校正ずれ ②−⑤", lambda a: float(np.mean(a["hs"]) - np.mean(a["ps"])),
        "{:>+8.3f}")
    row("★⑦ 帯の中の AUC(qp)", lambda a: auc(a["qs"], a["hs"]), "{:>8.4f}")
    row("⑧ 帯の中の AUC(pn)", lambda a: auc(a["ps"], a["hs"]), "{:>8.4f}")
    row("⑨ 帯の中央オッズ", lambda a: float(np.median(a["od"])), "{:>8.1f}")

    a0, a1 = A[ys[0]], A[ys[-1]]
    b0 = float(np.mean(a0["hs"]) - np.mean(a0["qs"]))
    b1 = float(np.mean(a1["hs"]) - np.mean(a1["qs"]))
    q0, q1 = auc(a0["qs"], a0["hs"]), auc(a1["qs"], a1["hs"])
    print(f"\n■ ★★★★**主判定: ④（市場の校正ずれ）はゼロに近づいたか**")
    print(f"　★**{ys[0]} {b0:+.3f} → {ys[-1]} {b1:+.3f}**"
          f"（**絶対値 {abs(b0):.3f} → {abs(b1):.3f}**）")
    near = abs(b1) < abs(b0)
    print(f"　→ ★**{'ゼロに近づいた' if near else '⚠近づいていない'}**")
    print(f"　★**⑦ 帯の中の AUC(qp): {q0:.4f} → {q1:.4f}（{q1-q0:+.4f}）**")
    print(f"\n⚠**qpの定義（複勝板の調和平均・Σ=3）に依存する**。"
          f"**2026は1,630レース＝途中まで**（判定基準43）。")
    print("⚠**枠連の運用には触れない**。**設定変更は提案しない**。")


if __name__ == "__main__":
    sys.exit(selftest() if "--selftest" in sys.argv else (main() or 0))
