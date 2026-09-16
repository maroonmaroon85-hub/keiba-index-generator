"""(248) ★★★★**トラックバイアスで絞るとROIは動くか** —— ★無作為プラセボで測る

★★**動機（2026-09-16・利用者の指定）**: ★**(245)(246)(247)で「バイアスは買い目のどこにも
　入っていない」と出た**。→ ★**では「新たに入れたら」動くのか**。

■ ⚠★★★**この測定には罠が2つある。★先に両方塞ぐ**
| | ⚠**罠** | ★**塞ぎ方** |
|---|---|---|
| **1** | ★★**(178)②型**——**「1レースあたり損益」を統計量にすると、母集団のROIが100%から離れている分だけ、部分集合を捨てる操作が★機械的に符号を持つ**（**枠連側が実際に踏んだ**） | ★★**統計量は ROI（100円あたり）**。★**かつ ★同じ本数を無作為に捨てるプラセボと比べる**（**絞らない全体とは比べない**） |
| **2** | ★★**(188)型**——**絞りがオッズと相関すると「(88)の再発見」に落ちる**（**ρ(平均オッズ)=−1.000**） | ★**絞った側の平均オッズを必ず出す**。⚠**大きく動いたら「オッズで説明できる」と書く** |

■ ★**バイアスの作り方（★そのレースより前の情報だけ・時刻の栓）**
　★**同じ日・同じ場で、★そのレースより前に走った既走レース**（★**3レース未満なら使わない**）。
　★`bias = mean(3着以内だった馬の超過align)`　（**超過 = align − 1/(2n)**・(247)と同じ）
　→ ★**正なら「その日その場は外が来ている」**。

■ ★**絞り方（★1つだけ。梯子を振らない）**
　★**軸の超過align × bias > 0**（★**軸がその日のバイアスの有利側にいるレースだけ買う**）。
　⚠**閾値を振らない**。★**符号だけ**。**マスを1つしか作らない**。

────────────────────────────────────────────────────────────
★★★ 事前登録（2026-09-16・**結果を見る前にコミットする**）
────────────────────────────────────────────────────────────

■ ★**対象**: ★**主判定の券種 `P馬単M4点` だけ**（⚠**他の券種は測らない＝マスを増やさない**）。
■ ★**母集団**: ★**軸が立った1,398本のうち、★bias が作れるレース**（**同日同場で既走3R以上**）。

■ ★★★★**主判定（★先に決める・これ1つ）**
　★★**絞ったROIが、★同じ本数を無作為に選んだときのROI分布の★99%点を超えるか**。
　★**無作為は2,000回**（seed固定）。★**絞りと同じ本数を、同じ母集団から復元抽出なしで選ぶ**。
| ★**返り値** | ★**読み** |
|---|---|
| ⚠**超える** | ★**バイアスで絞るとROIが動く**（★**ただしρ(平均オッズ)を見てから**） |
| ★**超えない** | ★★**動かない。★この線は閉じる** |

■ ★★ゲート2（判定基準42）——**何が返れば「絞りの効果ではない」か**
　★**絞った側の平均オッズが大きく動いたら、★(188)と同じく「オッズの再発見」**
　　→ ⚠**主判定が通っても採用しない**。★**「オッズで説明できる」と書く**。
　★**絞りの本数が母集団の半分前後になるはず**（**符号だけなので**）。⚠**極端に偏ったら定義を疑う**。

■ ★★★内部対照（**決定的・最初に見る**）
　**1.** ★**軸が立った全体が 1,398本**
　**2.** ★★**絞る前の `P馬単M4点` の全期間ROIが 112.9%（±0.2pt）**
　　★**これは (238) で実測済みの「1,398本での値」**（**1,383本の 113.4% ではない**）。

■ ⚠★★★★**対照を1度書き直した（判定基準37・★10回目・★同じ罠）**
　★**初版は「113.4%（±0.5pt）」と書いた**。→ ★**走らせたら 112.9% で落ちた**
　　（**差がちょうど0.5に乗り、浮動小数で `<= 0.5` を外した**）。★**結果は読んでいない**（判定基準32）。
　⚠★★**112.9% は (238) で私自身が測った値**——**(238)の初版も同じ取り違えで対照が落ち、
　　その訂正コミットに「1,398本で測れば U 112.9%」と★自分で書いている**。
　⚠★**それを知っていながら、また1,383本の値を対照に書いた**。
　★★**1,398（軸の全数） / 1,383（乱が引ける） の取り違えは、★これで10回目**。
　→ ★**対策は変えない**（**対象レース数を最初の内部対照にする**）が、★**それだけでは足りなかった**。
　　★**追加: ★凍結値を対照に書くときは、★その値を「どの母集団で測ったか」を必ず併記する**。
　⚠**どちらかが落ちたら読まない**（判定基準32）。

■ ⚠★**先に書いておく限界**
　★**bias は「同日同場の既走3R以上」でしか作れない**。→ ★**1R〜3Rが落ちる＝母集団が減る**。
　★**内外だけ**（**前残りは枠連側(177)で死亡・符号6/10**）。
　★**符号だけで絞る**（**強さを使わない**）。⚠**閾値を振れば必ず良いマスが出る**ので振らない。
　⚠**これは135マスで閉じた家族の★136マス目**（**(182)(185)(187)(190)(188)**）。
　　★**通っても「1マス通った」であって、★下駄が乗っている可能性は消えない**。
　⚠**枠連側(177)(178)の結論は持ち込んでいない**（判定基準25）。

■ 予想（⚠**当てにしない**・★私は14回外して15回目に当てた）
　★**超えないと見る**（**(188)(178)と同じ結末**）。

実行: python3 ml/audit_ana_biasroi.py    自己テスト: python3 ml/audit_ana_biasroi.py --selftest
"""
import sys

import numpy as np

sys.path.insert(0, "ml")
import features as F
from audit_crosspool import load_races, payoff
from audit_ana_odds import MIN_HORSES, roi_of
from audit_ana_board import NPLACE, load_fuku_boards, qpool
from audit_ana_band import PN_FLOOR
from audit_ana_hole import GAP
from audit_ana_fix import LFIX
from audit_ana_marg import wf_predict
from audit_ana_draw import align_of
from audit_ana_bet import tickets
from train_prod import add_odds_features

KNOWN_R, KNOWN_ROI, RTOL = 1398, 112.9, 0.2   # ★(238)で実測した「1,398本での値」
MINPRIOR, NPERM, SEED = 3, 2000, 20260916


def selftest():
    ok = True
    print("★★★★主判定: **絞ったROIが、同じ本数を無作為に選んだROI分布の99%点を超えるか**")
    print(f"　★無作為 {NPERM:,}回・seed固定。⚠**絞らない全体とは比べない**（(178)②型の罠）")
    print("★★ゲート2: **絞った側の平均オッズが大きく動いたら「オッズの再発見」**（(188)型）")
    print(f"★★★内部対照: **軸 {KNOWN_R:,}本** かつ **絞る前のP馬単M4点ROIが {KNOWN_ROI}%（±{RTOL}pt）**")
    print(f"　⚠★**{KNOWN_ROI}% は (238) 実測の「1,398本での値」**。"
          "★**1,383本の 113.4% ではない**（★判定基準37を10回踏んだ箇所）")
    print(f"★biasの作り方: **同日同場・そのレースより前・既走{MINPRIOR}R以上**"
          "／`bias = mean(3着以内の超過align)`")
    print("★絞り: ★**軸の超過align × bias > 0 の1つだけ**（⚠**閾値を振らない**）")
    # ★無作為プラセボの検算: 全部同じ値なら分布は一点
    g = np.random.default_rng(SEED)
    v = np.full(100, 1.5)
    m = [v[g.choice(100, 40, replace=False)].mean() for _ in range(50)]
    print(f"★プラセボの検算: 全部1.5 → 分布 [{min(m):.3f}, {max(m):.3f}]（一点のはず）"
          f"　{'★OK' if max(m)-min(m) < 1e-12 else '⚠NG'}")
    ok &= max(m) - min(m) < 1e-12
    # ★ROIの検算
    print(f"★ROIの検算: [0]*90+[2000]*10 → {roi_of(np.array([0.]*90+[2000.]*10)):.1f}%（200.0）")
    ok &= abs(roi_of(np.array([0.]*90+[2000.]*10)) - 200.0) < 1e-6
    print("⚠★**135マスで閉じた家族の136マス目。通っても下駄の可能性は消えない**")
    print("⚠**枠連側(177)(178)の結論は持ち込まない**（判定基準25）")
    print("★自己テスト: " + ("全部OK" if ok else "⚠NG"))
    return 0 if ok else 1


def main():
    print("(248) ★★★★**トラックバイアスで絞るとROIは動くか**\n")
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

    # ---- 1周目: 全レースの「3着以内の超過align」を作る（★biasの材料）
    ex3, axis = {}, {}
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
        n = len(ub)
        al = align_of(ub) - 1.0 / (2.0 * n)
        fin = {u: fi for u, _, fi in r["horses"]}
        top = [al[q] for q, u in enumerate(ub) if fin.get(int(u), 99) <= 3]
        if top:
            ex3[rid] = float(np.mean(top))
        pn = pv / pv.sum() * NPLACE
        qp, _ = qpool([bd[int(u)] for u in ub], "harm")
        gp = pn - qp
        c = np.where((pn >= PN_FLOOR) & (gp >= GAP) & (od >= LFIX))[0]
        if not len(c):
            continue
        i = int(c[int(np.argmax(pn[c]))])
        ax = int(ub[i])
        op = [int(u) for u in ub[np.argsort(-pv, kind="mergesort")] if int(u) != ax]
        axis[rid] = {"ax": ax, "op": op, "exa": float(al[i]),
                     "od": float(od[i]), "day": str(gg["date"].iloc[0].date()),
                     "r": r}

    # ---- 2周目: 同日同場・そのレースより前 の bias を作る
    #   ★raceidは 場(2)+年(2)+回(1)+日(1)+R(2)。同日同場 = 先頭6桁、レース番号 = 末尾2桁
    order = {}
    for rid in ex3:
        order.setdefault(rid[:6], []).append(rid)
    for k in order:
        order[k].sort(key=lambda x: int(x[6:8]))

    rows, dropped = [], 0
    for rid, a in sorted(axis.items()):
        sib = order.get(rid[:6], [])
        prev = [s for s in sib if int(s[6:8]) < int(rid[6:8])]
        if len(prev) < MINPRIOR:
            dropped += 1
            continue
        bias = float(np.mean([ex3[s] for s in prev]))
        t = tickets("馬単M", 4, a["ax"], a["op"])
        if t is None:
            continue
        pay = [payoff(a["r"], k2, s2) for k2, s2 in t]
        if any(x is None for x in pay):
            continue
        rows.append({"rid": rid, "v": sum(pay) / 4.0,   # ★100円あたりの払戻
                     "keep": a["exa"] * bias > 0, "od": a["od"], "bias": bias})

    A = np.array([x["v"] for x in rows], float)
    KP = np.array([x["keep"] for x in rows], bool)
    OD = np.array([x["od"] for x in rows], float)

    # ---- 内部対照
    allv = []
    for rid, a in sorted(axis.items()):
        t = tickets("馬単M", 4, a["ax"], a["op"])
        if t is None:
            continue
        pay = [payoff(a["r"], k2, s2) for k2, s2 in t]
        if any(x is None for x in pay):
            continue
        allv.append(sum(pay) / 4.0)
    allv = np.array(allv, float)
    roi_all = roi_of(allv)
    ok1 = len(axis) == KNOWN_R
    ok2 = abs(roi_all - KNOWN_ROI) <= RTOL
    print("★★★内部対照（**最初に見る**）")
    print(f"　1. 軸が立った全体 {len(axis):,}本（{KNOWN_R:,}）　{'★一致' if ok1 else '⚠ずれた'}")
    print(f"　2. 絞る前の P馬単M4点 ROI = {roi_all:.1f}%（{KNOWN_ROI}%・±{RTOL}pt）"
          f"　{'★一致' if ok2 else '⚠ずれた'}")
    if not (ok1 and ok2):
        print("⚠⚠**対照が落ちた。読まない**（判定基準32）。")
        return

    nk = int(KP.sum())
    print(f"\n★**bias が作れたレース {len(rows):,}本**"
          f"（⚠**既走{MINPRIOR}R未満で落ちた {dropped:,}本**）")
    print(f"★**絞って残る {nk:,}本**（{100*nk/max(len(rows),1):.1f}%）")
    if nk < 30 or nk > len(rows) - 30:
        print("⚠⚠**絞りが極端に偏った。★定義を疑う**（ゲート2）。読まない。")
        return

    roi_k = roi_of(A[KP])
    roi_d = roi_of(A[~KP])
    print(f"\n{'':<28}{'本数':>8}{'★ROI':>10}{'平均オッズ':>12}")
    print(f"{'絞る前（biasが作れた分）':<26}{len(A):>8,}{roi_of(A):>9.1f}%{OD.mean():>11.1f}倍")
    print(f"{'★★絞って残した':<25}{nk:>8,}{roi_k:>9.1f}%{OD[KP].mean():>11.1f}倍")
    print(f"{'　捨てた':<27}{len(A)-nk:>8,}{roi_d:>9.1f}%{OD[~KP].mean():>11.1f}倍")

    # ---- ★無作為プラセボ（同じ本数を無作為に選ぶ）
    g = np.random.default_rng(SEED)
    perm = np.array([roi_of(A[g.choice(len(A), nk, replace=False)]) for _ in range(NPERM)])
    p99 = float(np.percentile(perm, 99))
    pval = float(np.mean(perm >= roi_k))
    print(f"\n■ ★★★★**主判定: 無作為プラセボ {NPERM:,}回との比較**")
    print(f"　★**無作為に {nk:,}本選んだときのROI**: 中央 {np.median(perm):.1f}%"
          f" / 99%点 ★**{p99:.1f}%** / 範囲 [{perm.min():.1f}, {perm.max():.1f}]")
    print(f"　★★**絞ったROI = {roi_k:.1f}%**　→ ★**{'⚠99%点を超えた' if roi_k > p99 else '★超えなかった'}**"
          f"（**p = {pval:.3f}**）")

    dod = OD[KP].mean() - OD[~KP].mean()
    print(f"\n■ ★★ゲート2: **平均オッズ 残した {OD[KP].mean():.1f}倍 − 捨てた {OD[~KP].mean():.1f}倍"
          f" = {dod:+.1f}倍**")
    if abs(dod) > 2.0:
        print("　⚠⚠**オッズが大きく動いた＝(188)型「オッズの再発見」の疑い**。★**主判定が通っても採用しない**")
    else:
        print("　★**オッズはほとんど動いていない＝(188)型の交絡は小さい**")

    print(f"\n■ ★**結論の書き方**")
    if roi_k > p99 and abs(dod) <= 2.0:
        print("　⚠**無作為より上。★だが135マスで閉じた家族の136マス目＝下駄の可能性は消えない**")
    else:
        print("　★★**無作為と区別できない＝★バイアスで絞ってもROIは動かない。この線は閉じる**")
    print(f"\n⚠**内外だけ・符号だけ・閾値を振っていない**。⚠**枠連側の結論は持ち込んでいない**。")
    print("⚠**枠連の運用には触れない**。**設定変更は提案しない**。")


if __name__ == "__main__":
    sys.exit(selftest() if "--selftest" in sys.argv else (main() or 0))
