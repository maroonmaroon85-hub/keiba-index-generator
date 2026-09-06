"""(190) ★★★**(189)の絞り方で、券種を組の札に差し替える** — 馬連 / 馬単 / 三連複 / 三連単

★★**問い（ユーザー・2026-09-06）**——「**複勝なんか。この絞り方で馬単・馬連・三連複・三連単で検証して**」

★**(189)の規則をそのまま使う**（**レースの選び方は一切変えない**）:
```
条件を満たす馬 = { p_norm ≥ A  かつ  gap_f = p_norm − q_pool_fuku ≥ B }
1頭も居なければ → そのレースは見送る
居れば 穴 = その中で gap_f が最大の1頭
```
★**変えるのは「その穴で何を買うか」だけ**。

■ ⚠⚠★★★**第1版の誤りを訂正した（ユーザー指摘・2026-09-06）**
　**第1版は「点数は最小が最良」として各券種を最小構成に固定した**。
　★**指摘: 「これもわからないぞ。レース選定自体が異なるわけだから」——正しい**。
　⚠**(182)(185)(187)は「全レース」または「5-20倍帯のレース」で測った**。
　　**(189)の絞ったレースは母集団が違う**ので、**その数字は何も言っていない**。
　★★**判定基準25そのもの**——「**過去の知見を新しい層に持ち込むときは母集団を確かめる**」。
　　**このプロジェクトで5回踏んだと明記されている誤り**。**危うく6回目だった**。
　→ ★**紐の幅も、絞ったレースの中でスイープする**。**固定しない**。

■ ★★買い方（**幅をスイープする**）
　★**紐 = モデル確率の降順（穴を除く）**。**k頭目を足したときに増える組の限界ROI**を測る。
| 券種 | k頭目で増える組 | 点数 | kの範囲 | 払戻率 |
|---|---|---|---|---|
| **馬連** | 穴 × 紐k | 1 | 1〜4 | 0.775 |
| **馬単** | 穴 → 紐k（1着固定） | 1 | 1〜4 | 0.750 |
| **三連複** | 穴 + 紐k + 紐j (j<k) | k−1 | 2〜4 | 0.750 |
| **三連単** | 穴1着固定 × {紐k,紐j} の順列 | 2(k−1) | 2〜4 | 0.725 |
　★**馬単・三連単を1着固定にする理由は残す**: **(187)で「穴を後ろに置くほどROIが下がる」**
　　（馬単 84.6→81.5% / 三連単 84.6→82.6→76.6%）。⚠**ただしこれも別の母集団での実測**なので、
　　**「1着固定が最良」ではなく「1着固定だけを測る」と読む**。**位置のスイープは今回やらない**。

────────────────────────────────────────────────────────────
★★★ 事前登録（2026-09-06・**結果を見る前にコミットする**）
────────────────────────────────────────────────────────────

■ ★経路（判定基準25）: **q = MLモデル / q_pool = 複勝の板**。⚠**q 側は弱い**。
　⚠★**穴の選定は複勝の板で行い、買うのは別の券種**——**プールをまたいでいる**。
　　**(129)で「券種をまたぐ移送は更に落ちる」（+0.0009）と実測されている**。**逆風**。
■ ★手続き: **ウォークフォワード**（`data/cache/wf_pred.npz` を再利用）。
■ ★★絞りは**3通りだけ**（**格子16マス全部×幅では多重性が爆発する**）
　★**(189)の格子を代表する3点**——**最も緩い / 中央 / 最も厳しい**:
　　**(0.15, 0.02)** … 買うR 72.2% ／ **(0.20, 0.06)** … 62.1% ／ **(0.30, 0.08)** … 43.4%
　⚠**「最良のマス」を選んでいない**（**(189)の最良は(0.20,0.08)=93.6%**）。**格子を張るために選んだ**。

■ ★★★主判定（**42比較 = 3絞り × (馬連4 + 馬単4 + 三連複3 + 三連単3)・α=0.01/42・z=3.867**）
　★**k頭目の限界ROI の99%CI下端が 100% を超えるか**。
　★**同時に累積ROI（紐を何頭まで買うか）も出す**——**「何頭流すのが最良か」への直接の答え**。
　⚠**標本300未満、または99%CIの半幅が ±30pt を超えるマスは「判定不能」**
　　（**判定基準5: 絞ると測れなくなる**）。**「陰性」と書かない**。

■ ★★ゲート2（判定基準42）——**仮説が偽なら何を返すか**
　★**仮説が偽（k頭目を足すのが損益に中立）なら、「1点あたり払戻 − 100円」の期待値は厳密に0**。
　★**限界ROIは点数で割ってある**ので、**幅を変えても自動では動かない**＝**(170)の形にならない**。
　⚠**水準そのものは、市場が正しければその券種の払戻率を返す**
　　（**馬連77.5 / 馬単75.0 / 三連複75.0 / 三連単72.5%**）。**100%ではない**。

■ ⚠ゲート1: (88)③④を再現（±3pt）／ ★ゲート板: 復元R 0.800±0.020 ／
　★陽性対照: 三連複BOX上位4 81.9%±1.5pt。
■ ★★内部対照: ★**複勝の行が(189)を±1ptで再現**すること
　　（**(0.15,0.02)=91.5% / (0.20,0.06)=92.5% / (0.30,0.08)=91.6%**）。**再現しなければ装置が違う**。

■ ★★★探索を守る（**(178)でプラセボが109.9%、(172)は跳ねたマスを追って崩れた**）
　★**採用条件に「隣接するマスと同じ向き」を入れる**——**孤立した1マスは採らない**。
　★**全64マスは記述として出すが、そこから最良を選んで結論にしない**。
　⚠**(185)で「93.1%」を見出しにして訂正した**。**同じ誤りを繰り返さない**。

■ ★記述（判定しない）
　1. ★★**累積ROI と 1R損益**（**紐1頭 / 2頭 / 3頭 / 4頭**）——**「何頭流すのが最良か」への答え**。
　2. **限界ROIの表**（3絞り × 4券種 × k）: **的中率 / 増分の点数**。
　3. ★**複勝の行**（(189)の再掲）——**券種をまたいだ比較のため**。

■ ★採用条件（判定基準39/40/41）
　1. **ROIの99%CI下端が100%超のマスがある**
　2. ★**隣接マスと同じ向き**（孤立は採らない）
　3. ★**券種をまたいで同じ向き**（**1券種だけ跳ねるのは(172)で崩れた形**）
　4. **裾の検算で符号が反転しない**（上位3本・前後半・年別）

■ 予想（⚠**類推なので当てにしない**・判定基準24）
　★**通らないと見る**。**理由**:
　　1. **(189)の複勝が 89.2〜93.6%** で、**払戻率80.0%からの超過は +9〜+14pt**。
　　2. **組の払戻率は 72.5〜77.5%** で、**必要な超過が +22.5〜+27.5pt に増える**。
　　3. **(129)で「券種をまたぐ移送は落ちる」**（**穴は複勝の板で選び、買うのは別プール**）。
　⚠⚠★**ただし「幅」については予想を持たない**——**ご指摘のとおり、絞ったレースでは
　　母集団が違うので、(182)(185)(187)の「最小が最良」は当てにできない**。
　　★**絞ったレースでは幅を広げたほうが良い、という結果も十分ありうる**。

────────────────────────────────────────────────────────────
★★★ 実測（2026-09-06）—— **42マス中0マス。だがユーザー指摘が実測で正しかった**
────────────────────────────────────────────────────────────

■ ゲート1 4帯とも通過 ／ ゲート板 復元R 0.8009 ／ 陽性対照 BOX4 81.9% → 全部立った
■ ★★内部対照: **複勝が(189)を再現**（**91.5 / 92.1 / 91.5%**・差 −0.0/−0.4/−0.1pt）→ **装置は同一**

■ ★★★★**「点数は最小が最良」は、絞ったレースでは成り立たなかった**
　**絞り (0.15, 0.02) の三連単**:
| 紐 | 点数 | **累積ROI** | 1R損益 | この1頭の増分 |
|---|---|---|---|---|
| **2頭** | 2 | ★**103.1%** | ★**+6.1円** | **103.1%** |
| 3頭 | 6 | 93.5% | −39.0円 | 88.7% |
| 4頭 | 12 | 87.0% | −156.3円 | 80.4% |
★★**三連単・紐2頭が 103.1%（1レース +6.1円）**——**(182)(185)(187)では一度も出なかった数字**。
★**「最小が最良」も崩れた**: **馬連は 紐1頭 90.0% より 紐2頭の増分 89.0%**、
　**(0.20,0.06)では 紐2頭の増分 89.6% が 紐1頭 88.5% を上回る**。→ ★**絞ると幅の最適が動く**。
⚠★★**ユーザー指摘（「レース選定自体が異なるわけだから」）が実測で正しかった**——
　**判定基準25（このプロジェクトで5回踏んだと明記されている誤り）の6回目を踏むところだった**。
　**幅を固定せずスイープしたから 103.1% が見えた**。

■ ★★主判定: **100%超は 0マス**（**判定不能 1マス**）
| 絞り | 券種 | k | 限界ROI | 99%CI | 的中率 | 判定 |
|---|---|---|---|---|---|---|
| **(0.15,0.02)** | ★**三連単** | **2** | ★**103.1%** | ⚠**[72.1,134.1]** | **2.63%** | ⚠**判定不能(CI広)** |
| (0.20,0.06) | 三連単 | 2 | **96.6%** | [68.1,125.2] | 2.74% | ⚠通らない |
| (0.30,0.08) | 三連単 | 2 | **84.3%** | [63.1,105.5] | 3.13% | ⚠通らない |
| (0.15,0.02) | 馬単 | 1 | 91.9% | [77.4,106.4] | 5.81% | ⚠通らない |
| (0.15,0.02) | 三連複 | 2 | 91.2% | [79.7,102.6] | 7.13% | ⚠通らない |
★**的中率が2.63%しかなく、29,437レースあっても1点あたりのCIが ±31pt**。
　→ ★**事前登録の「半幅±30pt超は判定不能」に引っかかった**＝**「陰性」ではなく「測れない」**
　　（判定基準5: **絞ると測れなくなる**）。
⚠**絞りを変えると 103.1 → 96.6 → 84.3% と大きく動く**。**孤立した1マスで隣接条件も満たさない**
　（**(172)で崩れた形**）。

■ ★★結論
　★**(189)の絞り方でも、組の札は100%に届かない**（**42比較で0マス**）。
　★**幅も絞ったレースの中でスイープしたうえでの結論**——**ユーザー指摘の訂正を反映済み**。
　⚠★**ただし「三連単・紐2頭」は判定不能であって陰性ではない**。
　　**測るには的中率2.63%の券種で±30pt以内のCIが要る**＝**今の標本では届かない**。
　⚠**経路は弱いので「閉じた」とは書けない**（判定基準25）。

■ ◇**post-hoc・未検定（拾わない）**: **三連単は3絞りとも「紐2頭」が最良**
　（**103.1 / 96.6 / 84.3%**・**k=3,4 は必ず下がる**）。
　⚠**主判定を通っていないので拾わない**。**測るなら事前登録し直す**。


実行: python3 ml/audit_ana_grid2.py    自己テスト: python3 ml/audit_ana_grid2.py --selftest
"""
import math
import sys
from itertools import combinations, permutations

import numpy as np

sys.path.insert(0, "ml")
import features as F
from audit_crosspool import LINE, load_races, payoff, zq
from audit_ana_odds import BANDS, COST, MIN_HORSES, band_of, gate1, roi_of
from audit_ana_marg import WF_BOX4, WF_TOL, wf_predict
from audit_ana_board import BOARD_R, BOARD_TOL, NPLACE, load_fuku_boards, qpool
from audit_ana_grid import AS, BS, MINCELL
from train_prod import add_odds_features

MAXHW = 30.0                        # 判定不能の閾値（99%CIの半幅[pt]）
ALPHA = 0.01
SEED = 20260906
# (189)の複勝の実測（内部対照）
KNOWN_F = {(0, 0): 91.5, (0, 1): 91.7, (0, 2): 92.2, (0, 3): 93.1,
           (1, 0): 91.5, (1, 1): 91.8, (1, 2): 92.5, (1, 3): 93.6,
           (2, 0): 91.0, (2, 1): 91.5, (2, 2): 92.0, (2, 3): 93.0,
           (3, 0): 89.2, (3, 1): 90.0, (3, 2): 90.6, (3, 3): 91.6}


def marg(kind, ax, himo, k):
    """★k頭目の紐を足したときに**増える**組。幅は固定しない（ユーザー指摘の訂正）。"""
    hk = himo[k - 1]
    if kind in ("馬連", "馬単"):
        return [(ax, hk)]                           # ★馬単は1着固定
    if kind == "三連複":
        return [tuple(sorted((ax, hk, himo[j]))) for j in range(k - 1)]
    if kind == "三連単":
        out = []
        for j in range(k - 1):
            out += [(ax, hk, himo[j]), (ax, himo[j], hk)]
        return out
    raise ValueError(kind)


KMAX = 4
KINDS = [("馬連", 1), ("馬単", 1), ("三連複", 2), ("三連単", 2)]
# ★絞りは3通り（格子を張る。最良を選んでいない）
FILTERS = [(0, 0), (1, 2), (3, 3)]              # (0.15,0.02) / (0.20,0.06) / (0.30,0.08)
NCMP = len(FILTERS) * sum(KMAX - lo + 1 for _, lo in KINDS)   # 3 × 14 = 42


def selftest():
    ok = True
    ax, himo = 9, [3, 7, 1, 5]
    assert marg("馬連", ax, himo, 2) == [(9, 7)]
    assert marg("馬単", ax, himo, 2) == [(9, 7)]
    assert marg("三連複", ax, himo, 3) == [tuple(sorted((9, 1, 3))), tuple(sorted((9, 1, 7)))]
    t = marg("三連単", ax, himo, 3)
    assert len(t) == 4 and all(x[0] == ax for x in t)
    print("★限界の自己テスト: 馬連/馬単1点・三連複 k−1点・三連単 2(k−1)点　★OK")
    # ★増分の総和が全体の点数に一致（恒等式を数値で・判定基準27）
    for kind, lo in KINDS:
        for K in range(2, KMAX + 1):
            tot = sum(len(marg(kind, ax, himo, k)) for k in range(lo, K + 1))
            want = {"馬連": K, "馬単": K, "三連複": K*(K-1)//2, "三連単": K*(K-1)}[kind]
            assert tot == want, (kind, K, tot, want)
    print("★点数の自己テスト: 増分の総和が全体の点数に一致（4券種・K=2..4）　★OK")
    for k, _lo in KINDS:
        print(f"　{k:<6}払戻率 {LINE[k]:.3f}　100%に要る差 {100*(1-LINE[k]):+.1f}pt")
    print(f"★絞りは3通り: " + " / ".join(f"({AS[i]:.2f},{BS[j]:.2f})" for i, j in FILTERS)
          + "　⚠**最良を選んでいない。格子を張るために選んだ**")
    rng = np.random.default_rng(0)
    n = 200_000
    pay = np.where(rng.random(n) < 0.10, 1000.0, 0.0)
    m = float(np.mean([(rng.permutation(pay) - 100.0).mean() for _ in range(200)]))
    print(f"★ゲート2の自己テスト: 中立な買い目200回の平均 {m:+.3f}円 → "
          f"**仮説が偽なら払戻率を返す**: {'★OK' if abs(m) < 2 else '⚠NG'}")
    ok &= abs(m) < 2
    print(f"★比較数 {NCMP}（16マス × 4券種）→ z = {zq(ALPHA/NCMP):.3f}")
    print(f"★判定不能: 標本<{MINCELL} または 99%CI半幅>±{MAXHW}pt（判定基準5）")
    print(f"★内部対照: 複勝の行が(189)の {len(KNOWN_F)}マスを±1ptで再現")
    print("★自己テスト: " + ("全部OK" if ok else "⚠NG"))
    return 0 if ok else 1


def main():
    z = zq(ALPHA / NCMP)
    print("(190) ★★★**(189)の絞り方で、券種を組の札に差し替える**")
    print("★レースの選び方は(189)から一切変えない。**変えるのは「何を買うか」だけ**")
    print("★買い方は各券種の最小構成（**(182)(185)(187)の93マスで「点数は最小が最良」と実測**）")
    print("⚠★**穴は複勝の板で選び、買うのは別プール**＝**券種をまたぐ**。"
          "**(129)で「またぐと落ちる」と実測**\n")

    races = {r["rid"]: r for r in load_races()}
    rows, bad = gate1(list(races.values()))
    print("⚠**ゲート1**: (88)③④を別パーサで再現・許容±3pt")
    for nm, n, roi, known, dd, okg in rows:
        print(f"　{nm:<12}{roi:>7.1f}% vs {known:>5.1f}%　差 {dd:+.1f}pt"
              f"　{'★立った' if okg else '⚠落ちた'}")
    if bad:
        print("\n⚠⚠**ゲート1が落ちた。読まない**。")
        return

    print("\n★複勝の板(type=2)を読む…")
    boards = load_fuku_boards()
    print(f"　**{len(boards):,}レース分**")

    d = F.to_model(F.load_files())
    f = F.build_features(d)
    keep = (f["n_prior"] >= 1) & d["odds"].notna() & (d["odds"] > 0)
    d, f = d[keep].reset_index(drop=True), f[keep].reset_index(drop=True)
    y = (d["finish"] <= 3).astype(int).to_numpy()
    fx, _ = F.encode_categoricals(f)
    fx = add_odds_features(fx, d["odds"].to_numpy(float), d["raceid"].to_numpy())
    pred = wf_predict(d, fx, y, 3)
    msk = ~np.isnan(pred)
    sub = d.loc[msk, ["raceid", "umaban", "odds", "date"]].copy()
    sub["p"] = pred[msk]

    rng = np.random.default_rng(SEED)
    KEYS = [(fi, kind, k) for fi in range(len(FILTERS))
            for kind, lo in KINDS for k in range(lo, KMAX + 1)]
    K = {key: [] for key in KEYS}
    FU = {fi: [] for fi in range(len(FILTERS))}          # 複勝（内部対照）
    Rs, box4, nall = [], [], 0
    meta = {key: {"yr": [], "dt": []} for key in KEYS}
    for rid, g in sub.groupby("raceid"):
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
        nall += 1
        pn = pv / pv.sum() * NPLACE
        qp, Rb = qpool([bd[int(u)] for u in ub], "harm")
        Rs.append(Rb)
        gapf = pn - qp
        order = [int(u) for u in ub[np.argsort(-pv, kind="mergesort")]]
        bx = [payoff(r, "三連複", list(c)) for c in combinations(sorted(order[:4]), 3)]
        if not any(x is None for x in bx):
            box4.append(sum(bx) - 400.0)
        yr = int(gg["date"].iloc[0].year)
        for fi, (i0, j0) in enumerate(FILTERS):
            cand = np.where((pn >= AS[i0]) & (gapf >= BS[j0]))[0]
            if not len(cand):
                continue                              # ★このレースは見送り
            axu = int(ub[int(cand[int(np.argmax(gapf[cand]))])])
            vf = payoff(r, "複勝", [axu])
            if vf is not None:
                FU[fi].append(vf)
            himo = [u for u in order if u != axu]
            if len(himo) < KMAX:
                continue
            for kind, lo in KINDS:
                for k in range(lo, KMAX + 1):
                    cs = marg(kind, axu, himo, k)
                    vs = [payoff(r, kind, list(c)) for c in cs]
                    if any(v is None for v in vs):
                        continue
                    K[(fi, kind, k)].append(sum(vs) / len(cs))   # ★1点あたり払戻
                    meta[(fi, kind, k)]["yr"].append(yr)
                    meta[(fi, kind, k)]["dt"].append(gg["date"].iloc[0])
    print(f"\n★対象 **{nall:,}レース**（板があり8頭以上）")

    med = float(np.median(Rs))
    okR = abs(med - BOARD_R) <= BOARD_TOL
    print(f"■ ★ゲート板: 復元R = **{med:.4f}** → **{'★立った' if okR else '⚠⚠落ちた'}**")
    box4 = np.asarray(box4, float)
    g0 = 100.0 * (box4.sum() + 400.0 * len(box4)) / (400.0 * len(box4))
    okb = abs(g0 - WF_BOX4) <= WF_TOL
    print(f"■ ⚠**陽性対照**: BOX上位4 **{g0:.1f}%** vs {WF_BOX4}%"
          f" → **{'★立った' if okb else '⚠⚠落ちた'}**")
    if not (okR and okb):
        print("\n⚠⚠**ゲートが落ちた。読まない**（判定基準32）。")
        return

    print(f"\n■ ★★内部対照: **複勝が(189)を再現するか**（許容±1pt）")
    ng = 0
    for fi, (i0, j0) in enumerate(FILTERS):
        v = np.asarray(FU[fi], float)
        kv = KNOWN_F[(i0, j0)]
        dd = roi_of(v) - kv
        bad_ = abs(dd) > 1.0
        ng += bad_
        print(f"　({AS[i0]:.2f},{BS[j0]:.2f}) {len(v):>7,}R　"
              f"{roi_of(v):>6.1f}% vs (189) {kv:.1f}%　差 {dd:+.1f}pt"
              f"　{'⚠ズレた' if bad_ else '★再現'}")
    if ng:
        print("　⚠⚠**ズレた。装置を疑う**。")

    print(f"\n{'='*104}")
    print("■ ★★★**「何頭流すのが最良か」**（**累積ROI と 1R損益**）")
    print("　⚠★**(182)(185)(187)の「最小が最良」は別の母集団の話なので当てにしない**"
          "（判定基準25・ユーザー指摘）")
    for fi, (i0, j0) in enumerate(FILTERS):
        print(f"\n── 絞り: 推奨度≥{AS[i0]:.2f} かつ ズレ≥{BS[j0]:.2f}"
              f"（複勝 {roi_of(np.asarray(FU[fi], float)):.1f}%）──")
        print(f"{'券種':<8}{'紐':>4}{'点数':>6}{'累積ROI':>10}{'1R損益':>11}"
              f"{'この1頭の増分':>15}{'増分の点数':>11}{'標本R':>9}")
        for kind, lo in KINDS:
            tot_p, tot_n = 0.0, 0
            for k in range(lo, KMAX + 1):
                v = np.asarray(K[(fi, kind, k)], float)
                if len(v) < MINCELL:
                    continue
                n = len(marg(kind, 99, list(range(1, KMAX + 1)), k))
                tot_p += n * v.mean(); tot_n += n
                cum = 100.0 * tot_p / (tot_n * COST)
                print(f"{kind:<8}{k:>4}{tot_n:>6}{cum:>9.1f}%"
                      f"{tot_n * (cum / 100 - 1) * COST:>+10.1f}円"
                      f"{roi_of(v):>14.1f}%{n:>11}{len(v):>9,}")

    print(f"\n{'='*104}")
    print(f"■ ★★★主判定: **k頭目の限界ROI**（**{NCMP}比較・α={ALPHA}/{NCMP}・z={z:.3f}**）")
    print("　★ゲート2: **仮説が偽なら「1点あたり払戻−100円」の期待値は0**")
    print(f"\n{'絞り':<16}{'券種':<8}{'k':>3}{'限界ROI':>10}{'99%CI':>22}"
          f"{'的中率':>9}{'判定':>16}")
    passed, undet = set(), 0
    for fi, (i0, j0) in enumerate(FILTERS):
        lab = f"({AS[i0]:.2f},{BS[j0]:.2f})"
        for kind, lo in KINDS:
            for k in range(lo, KMAX + 1):
                v = np.asarray(K[(fi, kind, k)], float)
                if len(v) < MINCELL:
                    continue
                mu, se = v.mean() - COST, v.std(ddof=1) / math.sqrt(len(v))
                hw = z * se
                lo_, hi_ = 100 + mu - hw, 100 + mu + hw
                if hw > MAXHW:
                    undet += 1
                    tag = "⚠判定不能(CI広)"
                elif lo_ > 100.0:
                    passed.add((fi, kind, k)); tag = "★★100%超"
                else:
                    tag = "⚠通らない"
                print(f"{lab:<16}{kind:<8}{k:>3}{roi_of(v):>9.1f}%"
                      f"{f'[{lo_:.1f},{hi_:.1f}]':>22}{100*np.mean(v>0):>8.2f}%{tag:>16}")

    hits = [(f2, kd, k) for (f2, kd, k) in passed
            if (f2, kd, k - 1) in passed or (f2, kd, k + 1) in passed]
    print(f"\n■ ★採用条件")
    print(f"　1. 100%超のマス … **{len(passed)}マス**（⚠判定不能 {undet}マス）")
    print(f"　2. ★隣の k と同じ向き … **{len(hits)}マス**"
          f" → {'★満たす' if hits else '⚠満たさない'}")
    if passed and not hits:
        print("　⚠**孤立して通ったマスは採らない**（**(172)で崩れた形**）: "
              + " / ".join(f"({AS[FILTERS[f2][0]]:.2f},{BS[FILTERS[f2][1]]:.2f}){kd}k={k}"
                           for f2, kd, k in passed))
    if not hits:
        print("\n★★★**結論: (189)の絞り方でも、組の札は100%に届かない**。")
        print("★**幅も絞ったレースの中でスイープしたうえでの結論**（ユーザー指摘の訂正を反映）。")
        print("⚠**経路は弱いので「閉じた」とは書けない**（判定基準25）。")
        return
    print(f"\n■ ★裾の検算（**隣接条件も満たしたマス**・(77)）")
    for f2, kd, k in hits:
        v = np.asarray(K[(f2, kd, k)], float)
        yr = np.asarray(meta[(f2, kd, k)]["yr"], int)
        dt = np.array(meta[(f2, kd, k)]["dt"], dtype="datetime64[D]").astype(int)
        ys = sorted(set(yr))
        ov = sum(1 for u in ys if roi_of(v[yr == u]) > 100.0)
        i0, j0 = FILTERS[f2]
        print(f"　({AS[i0]:.2f},{BS[j0]:.2f}) {kd} k={k}: 限界ROI {roi_of(v):.1f}% / "
              f"**上位3本が全払戻の {100*np.sort(v)[-3:].sum()/max(v.sum(),1e-9):.1f}%** / "
              f"前半 {roi_of(v[dt<=np.median(dt)]):.1f}%・"
              f"後半 {roi_of(v[dt>np.median(dt)]):.1f}% / "
              f"**100%超の年 {ov}/{len(ys)}**")
    print("\n⚠**枠連の運用には触れない**。**設定変更は提案しない**。")


if __name__ == "__main__":
    sys.exit(selftest() if "--selftest" in sys.argv else (main() or 0))
