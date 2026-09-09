"""(236) ★★★★**モデルは賢くなったのか、市場に飲み込まれたのか** —— 年別に精度と一致度を測る

★★**動機（2026-09-09・利用者の指摘）**:
　★**(235)で「ズレ≥0.15の馬が 0.441→0.144頭（0.33倍）」と出た**。**市場側は不変**。
　★★**だが「ズレが縮んだ」には2つの読みがある**:
| | ★**読み** | ★**含意** |
|---|---|---|
| **A** | ★**モデルが正確になった**——**市場は元から正しく、こちらが追いついた** | **自然な収束** |
| ⚠**B** | ⚠**モデルが市場を真似るようになっただけ**（**`log_odds` が特徴量なので、
学習が進むほどオッズに引っ張られる**） | ★★**避けるべき。汚染前に戻す必要がある** |
　⚠**(235)ではAとBを区別できない**——**どちらでも「ズレが縮む」は起きる**。

────────────────────────────────────────────────────────────
★★★ 事前登録（2026-09-09・**結果を見る前にコミットする**）
────────────────────────────────────────────────────────────

■ ★★**測るもの（すべて年別）**
| # | 量 | ★**AとBのどちらを支持するか** |
|---|---|---|
| **①** | ★**pn上位1頭の複勝的中率** | ★**上がれば A** |
| **②** | **人気1番の複勝的中率**（★**市場の基準線**） | **横ばいのはず**（(235)で市場は不変） |
| **③** | ★★**① − ②（市場超過）** | ★**維持/拡大なら A**／⚠**ゼロに向かうなら B** |
| **④** | ★★**pn上位1頭 == 人気1番 の割合**（**一致度**） | ⚠**上がれば B** |
| **⑤** | **pn上位3頭と人気上位3頭の重なり** | ⚠**上がれば B** |
| **⑥** | ★**AUC(pn) と AUC(qp)**（**複勝的中の予測として**） | ★**AUC(pn)が上がれば A**／⚠**AUC(qp)に収束すれば B** |

■ ★★ゲート2（判定基準42）——**この測定は何を返せば「区別がつかない」か**
　★**モデルが変わっていないなら、①〜⑥すべてが横ばいの線を返す**。
　　⚠**そのとき(235)のズレ縮小は別の原因**（**qp側の変化など**）を探すことになる。
　★**AとBは③④で分かれる**——**③が縮み④が上がればB、③が維持され④が横ばいならA**。
■ ★★★内部対照（**決定的**）: ★**年別の対象レース数が (235) と一致すること**
　**2842 / 2807 / 2773 / 2767 / 2786 / 2813 / 2791 / 2780 / 2723 / 2758 / 1630**
　⚠**ずれたら読まない**（★**(235)で1,383と1,398を取り違えた直後なので、ここは特に慎重に**）。

■ ⚠**この測定は探索ではない**。★**マスも軸の定義も買い目も動かさない**。
■ ⚠★**先に書いておく限界**
　★**AUCは「順位づけの良さ」で、校正の良さではない**。**pnが市場に寄っても順位が良ければAUCは上がる**。
　→ ★**だから④の一致度と併せて読む**。**AUCが上がり、かつ一致度も上がるなら、
　　「市場と同じ順位を、より正確に再現できるようになった」＝★Bに近い**。

■ 予想（⚠**当てにしない**・★**私は11回外した**）
　⚠★**Bだと見る**——**`log_odds` を特徴量に入れている以上、学習が進むほど市場に寄るのが自然**。
　★**もしAなら、③（市場超過）が維持されているはず**。

実行: python3 ml/audit_ana_swallow.py    自己テスト: python3 ml/audit_ana_swallow.py --selftest
"""
import sys

import numpy as np

sys.path.insert(0, "ml")
import features as F
from audit_crosspool import load_races
from audit_ana_odds import MIN_HORSES
from audit_ana_board import NPLACE, load_fuku_boards, qpool
from audit_ana_marg import wf_predict
from train_prod import add_odds_features

KNOWN_R = [2842, 2807, 2773, 2767, 2786, 2813, 2791, 2780, 2723, 2758, 1630]


def auc(score, label):
    """★複勝的中の予測としてのAUC（Mann-Whitney）。★外部ライブラリを使わない"""
    s, y = np.asarray(score, float), np.asarray(label, int)
    p, n = int(y.sum()), int((1 - y).sum())
    if p == 0 or n == 0:
        return float("nan")
    r = np.empty(len(s), float)
    o = np.argsort(s, kind="mergesort")
    sv = s[o]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        r[o[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return (r[y == 1].sum() - p * (p + 1) / 2.0) / (p * n)


def selftest():
    ok = True
    print("★★AとBを分けるのは ③（市場超過）と ④（一致度）")
    print("　★③が維持され④が横ばい → **A（モデルが正確になった）**")
    print("　⚠③が縮み④が上がる → **B（市場を真似ただけ）**")
    y = np.array([0, 0, 1, 1])
    print(f"★AUCの検算: 完全分離 {auc([1, 2, 3, 4], y):.3f}（1.000）"
          f" / 逆 {auc([4, 3, 2, 1], y):.3f}（0.000）"
          f" / 同点 {auc([1, 1, 1, 1], y):.3f}（0.500）")
    ok &= (abs(auc([1, 2, 3, 4], y) - 1.0) < 1e-9
           and abs(auc([4, 3, 2, 1], y)) < 1e-9
           and abs(auc([1, 1, 1, 1], y) - 0.5) < 1e-9)
    print(f"★★★内部対照: **年別の対象レース数が (235) と一致**")
    print("　" + " / ".join(str(x) for x in KNOWN_R) + f"　（合計 {sum(KNOWN_R):,}）")
    print("　⚠**(235)で1,383と1,398を取り違えた直後なので、ここは特に慎重に**")
    print("⚠**探索ではない。マスも軸の定義も買い目も動かさない**")
    print("★自己テスト: " + ("全部OK" if ok else "⚠NG"))
    return 0 if ok else 1


def main():
    print("(236) ★★★★**モデルは賢くなったのか、市場に飲み込まれたのか**\n")
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
        fin = gg["finish"].astype(int).to_numpy()
        hit = (fin <= 3).astype(int)
        pn = pv / pv.sum() * NPLACE
        qp, _ = qpool([bd[int(u)] for u in ub], "harm")
        yr = int(gg["date"].iloc[0].year)
        a = A.setdefault(yr, {"R": 0, "m1": 0, "q1": 0, "same": 0, "ov": 0.0,
                              "ps": [], "qs": [], "hs": []})
        a["R"] += 1
        im, iq = int(np.argmax(pn)), int(np.argmin(od))
        a["m1"] += hit[im]
        a["q1"] += hit[iq]
        a["same"] += 1 if im == iq else 0
        top_m = set(np.argsort(-pn)[:3])
        top_q = set(np.argsort(od)[:3])
        a["ov"] += len(top_m & top_q) / 3.0
        a["ps"] += list(pn); a["qs"] += list(qp); a["hs"] += list(hit)

    ys = sorted(k for k in A if k >= 2016)
    got = [A[u]["R"] for u in ys]
    okc = got == KNOWN_R
    print(f"★★★内部対照（年別の対象レース数 vs (235)）")
    print(f"　実測 {got}")
    print(f"　既知 {KNOWN_R}　{'★★一致' if okc else '⚠⚠ずれた'}")
    if not okc:
        print("⚠⚠**対照が落ちた。読まない**（判定基準32）。")
        return

    print(f"\n{'':<28}" + "".join(f"{u:>8}" for u in ys))
    m1 = [100.0 * A[u]["m1"] / A[u]["R"] for u in ys]
    q1 = [100.0 * A[u]["q1"] / A[u]["R"] for u in ys]
    print(f"{'★① pn上位1頭の複勝的中率':<20}" + "".join(f"{x:>7.1f}%" for x in m1))
    print(f"{'② 人気1番の複勝的中率':<21}" + "".join(f"{x:>7.1f}%" for x in q1))
    print(f"{'★★③ 市場超過（①−②）':<20}"
          + "".join(f"{a - b:>+7.1f}pt" for a, b in zip(m1, q1)))
    print(f"{'★★④ pn1位==人気1番':<21}"
          + "".join(f"{100.0 * A[u]['same'] / A[u]['R']:>7.1f}%" for u in ys))
    print(f"{'⑤ 上位3頭の重なり':<22}"
          + "".join(f"{100.0 * A[u]['ov'] / A[u]['R']:>7.1f}%" for u in ys))
    ap = [auc(A[u]["ps"], A[u]["hs"]) for u in ys]
    aq = [auc(A[u]["qs"], A[u]["hs"]) for u in ys]
    print(f"{'★⑥ AUC(pn)':<26}" + "".join(f"{x:>8.4f}" for x in ap))
    print(f"{'　 AUC(qp)＝市場':<23}" + "".join(f"{x:>8.4f}" for x in aq))
    print(f"{'★★ AUCの差（pn−qp）':<21}"
          + "".join(f"{a - b:>+8.4f}" for a, b in zip(ap, aq)))

    print(f"\n■ ★★★**2016 → 2026**")
    for lab, v in (("★① pn上位1頭の的中率", m1), ("② 人気1番の的中率", q1),
                   ("★★③ 市場超過", [a - b for a, b in zip(m1, q1)]),
                   ("★★④ 一致度", [100.0 * A[u]["same"] / A[u]["R"] for u in ys]),
                   ("⑤ 上位3頭の重なり", [100.0 * A[u]["ov"] / A[u]["R"] for u in ys]),
                   ("★⑥ AUC(pn)", ap), ("　 AUC(qp)", aq),
                   ("★★ AUCの差", [a - b for a, b in zip(ap, aq)])):
        print(f"{lab:<22}{v[0]:>9.4f} → {v[-1]:>8.4f}　★**{v[-1] - v[0]:+8.4f}**")
    print("\n⚠**枠連の運用には触れない**。**設定変更は提案しない**。")


if __name__ == "__main__":
    sys.exit(selftest() if "--selftest" in sys.argv else (main() or 0))
