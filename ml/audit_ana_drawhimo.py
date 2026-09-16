"""(247) ★★★**紐に枠順の偏りはあるか** —— ★基準線をレース重みに直して、軸と紐を並べる

★★**動機（2026-09-16・利用者の問い「紐には関係ある？」）**
　★**(245)(246)で「軸に枠順の偏りは無い」と出た**。★**では紐はどうか**。

■ ⚠★★★**先に、(246)の基準線の取り方を直す（★私の2つ目の雑さ）**
　★**`align` の全馬平均は `1/(2n)` で常に正**（(246)で判明）。
　⚠**だが(246)の⓪は「全馬」＝★頭数重み**だった。
　★★**軸も紐も「1レースに1頭」なので、正しい基準線は★レース重み `E_races[1/(2n)]`**。
　　→ ★**小頭数のレースが等しく1票を持つので、レース重みの方が大きくなる**。
　⚠**(246)の結論（軸に偏りは無い）は、正しい基準線ならもっと強くなる方向**。★**それも確認する**。

■ ★**測る対象（★すべて1レースに1頭）**
| | 選び方 | ★**馬番を知っているか** |
|---|---|---|
| ★**軸** | `pn≥0.15 かつ ズレ≥0.15 かつ 単勝≥10.0倍` の中で **pn最大** | ⚠**pnは知らない／オッズ条件は帯で見て単調性なし** |
| ★**紐P1** | **pvの降順・軸を除く1位** | ★**pnは馬番を知らない** |
| ★**紐P2** | **同2位** | 同上 |
| ★**紐X3** | ★**人気順（単勝昇順）で、紐P上位2頭に使われていない先頭** | ⚠**オッズ順＝市場の順。★市場は馬番に値段を付けていない((245)②)** |
　★**比較相手は同じレースの `1/(2n)`**（★**レース重みの基準線**）。

────────────────────────────────────────────────────────────
★★★ 事前登録（2026-09-16・**結果を見る前にコミットする**）
────────────────────────────────────────────────────────────

■ ★**母集団**: **(245)(246)と同じ**。★**軸が立ったレースだけ**（**紐は軸があって初めて決まる**）。

■ ★★**測るもの**: ★**超過 = align − 1/(2n)**（**そのレースの基準線を引いた残り**）
　★**これならレースごとに中心がゼロになる**。→ ★**「ゼロを外すか」を素直に読める**。
　★**CIはレース単位のブートストラップ**（200回・seed固定・(245)(246)と同じ関数）。

■ ★★★★**主判定（★先に決める・これ1つ）**
　★★**紐P1・紐P2・紐X3 の超過が、それぞれ99%CIでゼロを含むか**。
| ★**返り値** | ★**読み** |
|---|---|
| ★**3つともゼロを含む** | ★★**紐に枠順の偏りは無い**。→ **トラックバイアスは買い目のどこにも入っていない** |
| ⚠**どれかがゼロを外す** | ⚠**その紐は枠順を拾っている**。★**どれが外したかを書く** |

■ ★★ゲート2（判定基準42）——**何が返れば「この測定は何も言っていない」か**
　★**軸の超過がゼロを外したら、(246)の結論と食い違う**
　　→ ⚠**そのときは基準線の取り方をもう一度疑う**（判定基準37）。★**紐を読まない**。

■ ★★★内部対照（**決定的・最初に見る**）
　**1.** ★**軸の本数が 1,398**（(245)(246)と一致）
　**2.** ★**軸の生の align 平均が +0.0561（±0.0005）**（(245)④・(246)⑤と一致）
　⚠**どちらかが落ちたら読まない**（判定基準32）。

■ ⚠★**先に書いておく限界**
　★**これは「絞ればROIが動くか」ではない**。★**ROIを一度も計算しない。マスを増やさない**。
　★**align は馬番ベース**（**枠番ではない**）。★**通年の平均**（**日ごと・場ごとの馬場差は見ない**）。
　⚠**「外枠が有利／不利か」は測っていない**。★**測るのは「買い目が枠順を拾っているか」だけ**。

■ 予想（⚠**当てにしない**・★私は14回外した）
　★**3つともゼロを含むと見る**（**pnは馬番を持たず、市場も馬番に値段を付けていないので**）。

────────────────────────────────────────────────────────────
★★★★ 結果（2026-09-16・**事前登録をコミットしたあとに走らせた**）
────────────────────────────────────────────────────────────
★★★**内部対照は2つとも立った**（軸1,398本・軸の生align +0.0561）。
★★**基準線を直した**: ★**レース重み `E_races[1/(2n)]` = +0.0365**
　（⚠**(246)の頭数重み ⓪ = +0.0451**・**平均頭数 14.2**）。

| | 本数 | 生align | ★**超過** | ★**99%CI（超過）** | |
|---|---|---|---|---|---|
| ★**軸（pn最大）** | 1,398 | +0.0561 | **+0.0196** | [−0.0004, +0.0375] | ★**含む** |
| ★★**紐P1（pv1位）** | 1,398 | +0.0519 | **+0.0154** | [−0.0017, +0.0372] | ★**含む** |
| ★★**紐P2（pv2位）** | 1,398 | +0.0473 | **+0.0108** | [−0.0052, +0.0309] | ★**含む** |
| ★★**紐X3（人気順）** | 1,398 | +0.0534 | **+0.0169** | [−0.0023, +0.0369] | ★**含む** |

■ ★★ゲート2: ★**軸の超過 +0.0196 [−0.0004,+0.0375] がゼロを含む＝(246)と整合**。
　→ ★**正しい基準線でも軸に偏りは無い**。★**紐を読んでよい**。

■ ★★★★**主判定の返り値: 紐に枠順の偏りは無い**
　★**P1・P2・X3 の3つとも超過がゼロを含む**。
　→ ★★**(245)③（ズレは馬番の関数を含まない）・(246)（軸に偏り無し）とあわせて、
　　　★トラックバイアスはこの買い目のどこにも入っていない**。

■ ★**予想は当たった（★15回目の予想・当たり）**
　★**「3つともゼロを含むと見る」→ そのとおり**（**pnは馬番を持たず、市場も馬番に値段を付けていない**）。

■ ⚠★**これが意味しないこと（★先に書いたとおり）**
　⚠**「外枠が有利／不利か」は測っていない**（★**枠連側(177)は「内外バイアスは実在する」と測っている**）。
　★**測ったのは「買い目が枠順を拾っているか」だけ**。→ ★**拾っていない＝利用も被害もしていない**。
　⚠**「バイアスで絞ればROIが動くか」は依然として未測定**。
　★★**使うなら「今は入っていないものを新たに入れる」話になる**
　　→ ⚠**(188)の ρ(平均オッズ)=−1.000 と (178)②の壊れた統計量、★2つの罠が待っている**。

実行: python3 ml/audit_ana_drawhimo.py    自己テスト: python3 ml/audit_ana_drawhimo.py --selftest
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

KNOWN_N, KNOWN_AX, ATOL = 1398, 0.0561, 0.0005      # ★(245)④ / (246)⑤


def base_of(n):
    """★そのレースの基準線 = align の race 内平均 = 1/(2n)"""
    return 1.0 / (2.0 * n)


def selftest():
    ok = True
    print("★★★★主判定: **紐P1・紐P2・紐X3 の超過が、それぞれ99%CIでゼロを含むか**")
    print("　★3つとも含む → **紐に枠順の偏りは無い** ／ ⚠外す → **その紐は枠順を拾っている**")
    print("★★ゲート2: **軸の超過がゼロを外したら(246)と食い違う → 基準線を疑い、紐を読まない**")
    print(f"★★★内部対照: **軸 {KNOWN_N:,}本** かつ **軸の生align = {KNOWN_AX}（±{ATOL}）**")
    for n in (8, 12, 18):
        a = align_of(list(range(1, n + 1)))
        print(f"★基準線の検算 {n:>2}頭: align平均 {a.mean():+.4f} / 1/(2n) {base_of(n):+.4f}"
              f"　{'★一致' if abs(a.mean() - base_of(n)) < 1e-12 else '⚠NG'}")
        ok &= abs(a.mean() - base_of(n)) < 1e-12
    print("★★**超過 = align − 1/(2n)** → ★**レースごとに中心がゼロ**")
    a = align_of(list(range(1, 13)))
    print(f"　12頭・全馬の超過の平均: {np.mean(a - base_of(12)):+.10f}（★ゼロのはず）")
    ok &= abs(np.mean(a - base_of(12))) < 1e-12
    lo, hi = boot_ci([0.01] * 100)
    print(f"★CIの検算: 全部0.01 → [{lo:.4f}, {hi:.4f}]　{'★OK' if abs(hi-lo) < 1e-9 else '⚠NG'}")
    ok &= abs(hi - lo) < 1e-9
    print("⚠★**ROIを一度も計算しない。買い目も軸も動かさない。マスを増やさない**")
    print("★自己テスト: " + ("全部OK" if ok else "⚠NG"))
    return 0 if ok else 1


def main():
    print("(247) ★★★**紐に枠順の偏りはあるか**\n")
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
    sub = d.loc[m, ["raceid", "umaban", "odds"]].copy()
    sub["p"] = pred[m]

    K = ["axis", "P1", "P2", "X3"]
    raw = {k: [] for k in K}          # ★生のalign
    exc = {k: [] for k in K}          # ★超過 = align − 1/(2n)
    bases, nn = [], []
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
        c = np.where((pn >= PN_FLOOR) & (gp >= GAP) & (od >= LFIX))[0]
        if not len(c):
            continue
        n = len(ub)
        al = align_of(ub)
        b = base_of(n)
        pos = {int(u): q for q, u in enumerate(ub)}
        i = int(c[int(np.argmax(pn[c]))])
        ax = int(ub[i])
        op = [int(u) for u in ub[np.argsort(-pv, kind="mergesort")] if int(u) != ax]
        oq = [int(u) for u in ub[np.argsort(od, kind="mergesort")] if int(u) != ax]
        X = op[:2] + [u for u in oq if u not in op[:2]]
        pick = {"axis": ax}
        if len(op) >= 1:
            pick["P1"] = op[0]
        if len(op) >= 2:
            pick["P2"] = op[1]
        if len(X) >= 3:
            pick["X3"] = X[2]
        for k, u in pick.items():
            raw[k].append(float(al[pos[u]]))
            exc[k].append(float(al[pos[u]] - b))
        bases.append(b); nn.append(n)

    ok1 = len(raw["axis"]) == KNOWN_N
    ok2 = abs(float(np.mean(raw["axis"])) - KNOWN_AX) <= ATOL
    print("★★★内部対照（**最初に見る**）")
    print(f"　1. 軸の本数 {len(raw['axis']):,}（{KNOWN_N:,}）　{'★一致' if ok1 else '⚠ずれた'}")
    print(f"　2. 軸の生align {np.mean(raw['axis']):+.4f}（{KNOWN_AX}）　{'★一致' if ok2 else '⚠ずれた'}")
    if not (ok1 and ok2):
        print("⚠⚠**対照が落ちた。読まない**（判定基準32）。")
        return

    print(f"\n★★**基準線を直す**: レース重み `E_races[1/(2n)]` = ★**{np.mean(bases):+.4f}**"
          f"（⚠**(246)の頭数重み ⓪ = +0.0451**・平均頭数 {np.mean(nn):.1f}）")

    lab = {"axis": "★軸（pn最大）", "P1": "★★紐P1（pv1位）",
           "P2": "★★紐P2（pv2位）", "X3": "★★紐X3（人気順）"}
    print(f"\n{'':<24}{'本数':>8}{'生align':>10}{'★超過':>10}{'99%CI（超過）':>26}")
    res = {}
    for k in K:
        lo, hi = boot_ci(exc[k])
        res[k] = (float(np.mean(exc[k])), lo, hi)
        z = "★ゼロを含む" if lo <= 0 <= hi else "⚠**ゼロを外す**"
        print(f"{lab[k]:<22}{len(exc[k]):>8,}{np.mean(raw[k]):>+10.4f}"
              f"{np.mean(exc[k]):>+10.4f}   [{lo:>+7.4f},{hi:>+7.4f}] {z}")

    a_m, a_lo, a_hi = res["axis"]
    print(f"\n■ ★★ゲート2: **軸の超過 = {a_m:+.4f} [{a_lo:+.4f}, {a_hi:+.4f}]**")
    if not (a_lo <= 0 <= a_hi):
        print("　⚠⚠**軸がゼロを外した＝(246)と食い違う。★基準線を疑い、紐を読まない**（判定基準37）")
        return
    print("　★**ゼロを含む＝(246)の結論と整合**（**正しい基準線でも軸に偏りは無い**）")

    print(f"\n■ ★★★★**主判定: 紐3つの超過**")
    zero = []
    for k in ("P1", "P2", "X3"):
        m2, lo, hi = res[k]
        inz = lo <= 0 <= hi
        zero.append(inz)
        print(f"　{lab[k]:<22} {m2:+.4f}  [{lo:+.4f}, {hi:+.4f}]  "
              f"{'★ゼロを含む' if inz else '⚠**ゼロを外す**'}")
    if all(zero):
        print("　→ ★★**3つともゼロを含む＝紐に枠順の偏りは無い**")
        print("　→ ★**(245)③(246)とあわせて、★トラックバイアスは買い目のどこにも入っていない**")
    else:
        out = [lab[k] for k, z in zip(("P1", "P2", "X3"), zero) if not z]
        print(f"　→ ⚠**ゼロを外した: {' / '.join(out)}**＝★**その紐は枠順を拾っている**")
    print(f"\n⚠**ROIを一度も計算していない**。⚠**「外枠が有利か」は測っていない**。")
    print("⚠**枠連の運用には触れない**。**設定変更は提案しない**。")


if __name__ == "__main__":
    sys.exit(selftest() if "--selftest" in sys.argv else (main() or 0))
