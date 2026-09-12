"""(186) ★★★**「候補頭数」の交絡を切り分ける** — 差を作っているのは「程度」か「何頭から選ぶか」か

★★**自己監査の漏れB（2026-09-06）**——
　★**穴 = その帯にいる m 頭のうち gap 最大の馬**。**ところが m は帯によって大きく違う**:
| 帯 | 買うレース割合（m≥2の割合・(180)実測） |
|---|---|
| 1.5-3倍 | **5.7%** |
| 12-20倍 | 52.1% |
| **120倍〜** | ★**65.3%** |
　★★**m が大きいほど「最大を選ぶ」という選別が強くなる**——
　　**m=2 の最大と m=5 の最大では、1頭あたりの情報量が同じでも差が違って出る**。
　→ ★★★**(180)の「差が程度とともに増える」は、程度の効果ではなく候補頭数の効果かもしれない**。
　⚠**(181)でその形が消えた（ρ +0.950→+0.517）のとも整合する**。

★**価格は複勝の板を使う**（**(184)(185)で「板が正しい」ことが3本の独立な確認で揃った**）。
⚠**水準の結論は動かない**（**最良88.4%／93.1%で100%に届かない**）。
　★**本件の目的は、(180)(181)の解釈を完成させること**。**運用の提案はしない**。

────────────────────────────────────────────────────────────
★★★ 事前登録（2026-09-06・**結果を見る前にコミットする**）
────────────────────────────────────────────────────────────

■ ★経路（判定基準25）: **q = MLモデルの top3 確率 / q_pool = 複勝の板**。⚠**q 側は弱い**。
■ ★手続き: **ウォークフォワード**（`data/cache/wf_pred.npz` を再利用）。

■ ★★設計 —— **(帯 × m) の二元表**
```
各レース × 各オッズ帯 について、その帯にいる頭数を m とする（m≥2 のときだけ使う）
  穴 = その帯で gap_f = p_norm − q_pool_fuku が最大の1頭
  乱 = その帯の別の1頭（無作為）
  差 = 穴の複勝払戻 − 乱の複勝払戻
これを **帯9段 × m（2 / 3 / 4 / 5頭以上）= 36セル** に分ける
```
　⚠**セルの標本が200未満なら使わない**（**ragged な表になる**）。

■ ★★★主判定（**2つ。α=0.01/2**）
　★**A（m の効果）**: **各帯の中で「差 vs m」の Spearman ρ を出し、帯をまたいで平均する**。
　★**B（帯の効果）**: **各 m の中で「差 vs 帯の順位」の Spearman ρ を出し、m をまたいで平均する**。
　★★**帰無分布**: **各ペア（同じレース・同じ帯の 穴 と 乱）の差の符号を独立に ±1 で反転**
　　（**完全な randomization test**・**1,000回**）。**maxT で2つをまとめて制御**。

■ ★★ゲート2（判定基準42）——**仮説が偽なら何を返すか**
　★**仮説が偽（gap_f が帯の中で何も持たない）なら、穴と乱は交換可能なので
　　どのセルの差も期待値0になり、A も B も 0 を返す**。
　★**符号反転の帰無分布が、まさにその状態**。
　・**買う頭数もコストも券種も完全に同一**＝**(170)の形にならない**。
　・**m≥2 のセルは穴側も乱側も同じだけ使う**＝**(168)の形にならない**。

■ ⚠ゲート1: (88)③④を別パーサで再現（±3pt）。
■ ★ゲート板: **復元R = 3/Σ(1/調和平均) の中央値が 0.800±0.020**（**(184)(185)で0.8009**）。
■ ★陽性対照: **三連複BOX上位4が 81.9%±1.5pt**。

■ ★★読み方（**先に書く**）
　| A（m） | B（帯） | 読み |
　|---|---|---|
　| ★**通る** | ⚠**通らない** | ★★**差を作っているのは「何頭から選ぶか」であって「程度」ではない**<br>→ **(180)の「差が程度とともに増える」は交絡だった** |
　| ⚠**通らない** | ★**通る** | **程度の効果が本物**。**(180)の読みが正しかった** |
　| ★**両方通る** | | **両方効いている**。**どちらが大きいかを ρ の大小で記述する**（**判定はしない**） |
　| ⚠**両方通らない** | | **どちらとも言えない**。**(181)で形が消えたことと整合** |

■ ★採用条件
　1. **A または B が maxT の99%点を超える**
　2. ★**通ったほうの向きが、隣接するセルで一貫している**（**孤立したセルは採らない**）
　⚠**どちらも通らなければ「(180)の形は、どちらの説明でも支えられない」と書いて閉じる**。

■ 予想（⚠**類推なので当てにしない**・判定基準24）
　★**A が通り B が通らない**と見る（**m と帯が強く相関しているから**）。
　⚠**だが(181)で形そのものが消えているので、両方通らない可能性も高い**。**どちらでも驚かない**。

────────────────────────────────────────────────────────────
★★★ 実測（2026-09-06・ウォークフォワード・板ベース）—— **両方通らない。だが非対称が読める**
────────────────────────────────────────────────────────────

■ ゲート1: 4帯とも通過 ／ ゲート板: 復元R **0.8009** ／ 陽性対照: BOX4 **81.9%** → 全部立った

■ ★記述: **帯ごとの候補頭数 m**（**交絡の本体**）
| 帯 | 平均 m | m≥2 の割合 |
|---|---|---|
| 1.5-3倍 | **0.67** | **5.6%** |
| 5-8倍 | 1.46 | 44.7% |
| 12-20倍 | 1.70 | 52.4% |
| **120倍〜** | ★**2.50** | ★**65.4%** |

■ ★★二元表: **(帯 × m) の差[円]**
| 帯 | m=2 | m=3 | m=4 | m≥5 |
|---|---|---|---|---|
| 1.5-3倍 | +1.8 | — | — | — |
| 3-5倍 | +5.1 | +4.3 | — | — |
| 5-8倍 | +7.7 | +7.9 | +12.8 | — |
| 8-12倍 | +9.1 | +12.7 | +18.7 | — |
| **12-20倍** | +10.9 | **+15.4** | ★**+24.0** | +21.3 |
| 20-35倍 | +11.7 | +12.6 | +18.1 | +8.9 |
| 35-60倍 | +12.9 | +9.8 | +1.2 | +35.1 |
| 60-120倍 | +2.5 | +10.5 | +0.6 | +43.3 |
| **120倍〜** | ⚠**−5.8** | −1.1 | +29.0 | +2.4 |

■ ★★★主判定（**帰無は対応のある符号反転1,000回・maxTで2つを制御**・**帰無99%点 0.617**）
| | 実測 | 判定 |
|---|---|---|
| **A（m の効果）** | **+0.571** | ⚠**超えない**（**惜しい**） |
| **B（帯の効果）** | ★**−0.036** | ⚠**超えない**（★**ほぼ厳密に0**） |
★**事前登録どおり「どちらの説明でも支えられない」として閉じる**。

■ ★★**それでも非対称は読む価値がある**
　★★**B = −0.036 は「m を固定すると、オッズ帯は差について何も持たない」という意味**——
　　**帯を横に見ても勾配がない**。
　★**A = +0.571 は「帯の中で m が増えると差が増える」傾向**（**12-20倍: 10.9→15.4→24.0**）。
　　⚠**事前登録の閾値0.617には届かない**。
　→ ★**「差が程度とともに増える」という(180)の読みは、m を固定すると消える**。
　　 ⚠**ただし「m が原因だ」と言い切るには A が足りない**。
　　 ★**(181)で形そのものが消えたことと合わせて、ここで閉じる**。

■ ⚠**120倍〜の行が荒れている**（**m=2で−5.8円 / m=4で+29.0円**）
　→ ★**(184)「超人気薄では複勝の市場が単勝と違うことを言う」と整合**。

■ ★★結論
　★**(180)(181)の解釈は完成した**——**「差が程度とともに増える」は支えられない**。
　⚠**水準の結論は動かない**（**最良 88.4%／93.1% で100%に届かない**）。


実行: python3 ml/audit_ana_mcount.py    自己テスト: python3 ml/audit_ana_mcount.py --selftest
"""
import math
import sys
from itertools import combinations

import numpy as np

sys.path.insert(0, "ml")
import features as F
from audit_crosspool import load_races, payoff, zq
from audit_ana_odds import COST, MIN_HORSES, gate1, roi_of
from audit_ana_degree import DEG
from audit_ana_marg import WF_BOX4, WF_TOL, wf_predict
from audit_ana_board import BOARD_R, BOARD_TOL, NPLACE, load_fuku_boards, qpool
from train_prod import add_odds_features

MBINS = [(2, "m=2"), (3, "m=3"), (4, "m=4"), (5, "m≥5")]
MINCELL = 200
NPERM = 1000
NCMP = 2
ALPHA = 0.01
SEED = 20260906


def mbin(m):
    return min(m, 5) - 2                      # 2→0 / 3→1 / 4→2 / 5以上→3


def rho_vs_rank(v):
    """値の並び v と 1..k の順位相関。k<3 なら 0 を返す。"""
    v = np.asarray(v, float)
    if len(v) < 3:
        return 0.0
    r = np.argsort(np.argsort(v)) + 1.0
    return float(np.corrcoef(np.arange(1.0, len(v) + 1), r)[0, 1])


def two_way(cells):
    """(帯, m) → 差の配列 の辞書 → (A: m の効果, B: 帯の効果)。"""
    mu = {}
    for k, v in cells.items():
        if len(v) >= MINCELL:
            mu[k] = float(np.mean(v))
    # A: 各帯の中で m 方向
    ra = []
    for bi in range(len(DEG)):
        seq = [mu[(bi, mi)] for mi in range(len(MBINS)) if (bi, mi) in mu]
        if len(seq) >= 3:
            ra.append(rho_vs_rank(seq))
    # B: 各 m の中で 帯 方向
    rb = []
    for mi in range(len(MBINS)):
        seq = [mu[(bi, mi)] for bi in range(len(DEG)) if (bi, mi) in mu]
        if len(seq) >= 3:
            rb.append(rho_vs_rank(seq))
    return (float(np.mean(ra)) if ra else 0.0,
            float(np.mean(rb)) if rb else 0.0, mu)


def selftest():
    ok = True
    assert mbin(2) == 0 and mbin(3) == 1 and mbin(4) == 2 and mbin(7) == 3
    print("★m の群分けの自己テスト: 2/3/4/5以上　★OK")
    assert abs(rho_vs_rank([1, 2, 3, 4]) - 1.0) < 1e-9
    assert abs(rho_vs_rank([4, 3, 2, 1]) + 1.0) < 1e-9
    print("★順位相関の自己テスト: 昇順→+1.000 / 降順→−1.000　★OK")
    # ★「m だけが効く」人工データ → A が高く B が0付近
    rng = np.random.default_rng(0)
    cells = {}
    for bi in range(len(DEG)):
        for mi in range(len(MBINS)):
            cells[(bi, mi)] = rng.normal(10.0 * mi, 50, 3000)
    a, b, _ = two_way(cells)
    print(f"★切り分けの自己テスト（m だけが効く人工データ）: A={a:+.3f}（要≒+1）/ "
          f"B={b:+.3f}（要≒0）→ {'★OK' if a > 0.9 and abs(b) < 0.4 else '⚠NG'}")
    ok &= a > 0.9 and abs(b) < 0.4
    # ★「帯だけが効く」→ 逆
    cells = {}
    for bi in range(len(DEG)):
        for mi in range(len(MBINS)):
            cells[(bi, mi)] = rng.normal(3.0 * bi, 50, 3000)
    a, b, _ = two_way(cells)
    print(f"★切り分けの自己テスト（帯だけが効く人工データ）: A={a:+.3f}（要≒0）/ "
          f"B={b:+.3f}（要≒+1）→ {'★OK' if abs(a) < 0.4 and b > 0.9 else '⚠NG'}")
    ok &= abs(a) < 0.4 and b > 0.9
    # ★ゲート2: 符号反転なら A も B も 0
    base = {(bi, mi): rng.normal(0, 300, 2000)
            for bi in range(len(DEG)) for mi in range(len(MBINS))}
    aa, bb = [], []
    for _ in range(200):
        pm = {k: v * rng.choice([-1.0, 1.0], size=len(v)) for k, v in base.items()}
        a, b, _ = two_way(pm)
        aa.append(a); bb.append(b)
    ma, mbv = float(np.mean(aa)), float(np.mean(bb))
    okg = abs(ma) < 0.15 and abs(mbv) < 0.15
    print(f"★ゲート2の自己テスト: 符号反転200回 → A の平均 {ma:+.3f} / B の平均 {mbv:+.3f}"
          f" → **仮説が偽なら両方0**: {'★OK' if okg else '⚠NG'}")
    ok &= okg
    print(f"★比較数 {NCMP} → z = {zq(ALPHA/NCMP):.3f}　／ セルの最小標本 {MINCELL}")
    print("★自己テスト: " + ("全部OK" if ok else "⚠NG"))
    return 0 if ok else 1


def main():
    print("(186) ★★★**「候補頭数」の交絡を切り分ける**")
    print("★問い: **差を作っているのは「程度（オッズ帯）」か「何頭から選ぶか（m）」か**")
    print("★価格は**複勝の板**（(184)(185)で3本の独立な確認が揃った）")
    print("⚠**水準の結論は動かない**。**目的は(180)(181)の解釈を完成させること**\n")

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
    cells = {(bi, mi): [] for bi in range(len(DEG)) for mi in range(len(MBINS))}
    mcount = {bi: [] for bi in range(len(DEG))}
    Rs, box4 = [], []
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
        pn = pv / pv.sum() * NPLACE
        qp, Rb = qpool([bd[int(u)] for u in ub], "harm")
        Rs.append(Rb)
        gapf = pn - qp
        order = [int(u) for u in ub[np.argsort(-pv, kind="mergesort")]]
        bx = [payoff(r, "三連複", list(c)) for c in combinations(sorted(order[:4]), 3)]
        if not any(x is None for x in bx):
            box4.append(sum(bx) - 400.0)
        for bi, (nm, lo, hi) in enumerate(DEG):
            idx = np.where((od >= lo) & (od < hi))[0]
            m = len(idx)
            mcount[bi].append(m)
            if m < 2:
                continue
            a = int(idx[int(np.argmax(gapf[idx]))])
            b = int(rng.choice([int(k) for k in idx if k != a]))
            pa = payoff(r, "複勝", [int(ub[a])])
            pb = payoff(r, "複勝", [int(ub[b])])
            if pa is None or pb is None:
                continue
            cells[(bi, mbin(m))].append(pa - pb)

    med = float(np.median(Rs))
    okR = abs(med - BOARD_R) <= BOARD_TOL
    print(f"\n■ ★ゲート板: 復元R の中央値 = **{med:.4f}** → **{'★立った' if okR else '⚠⚠落ちた'}**")
    box4 = np.asarray(box4, float)
    g0 = 100.0 * (box4.sum() + 400.0 * len(box4)) / (400.0 * len(box4))
    okb = abs(g0 - WF_BOX4) <= WF_TOL
    print(f"■ ⚠**陽性対照**: 三連複BOX上位4 **{g0:.1f}%** vs {WF_BOX4}%"
          f" → **{'★立った' if okb else '⚠⚠落ちた'}**")
    if not (okR and okb):
        print("\n⚠⚠**ゲートが落ちた。読まない**（判定基準32）。")
        return

    print(f"\n■ 記述: ★**帯ごとの候補頭数 m**（**これが交絡の本体**）")
    print(f"{'帯':<10}{'平均 m':>9}{'m≥2 の割合':>13}")
    for bi, (nm, lo, hi) in enumerate(DEG):
        v = np.asarray(mcount[bi], float)
        print(f"{nm:<10}{v.mean():>9.2f}{100*np.mean(v >= 2):>12.1f}%")

    print(f"\n{'='*104}")
    print("■ ★★二元表: **(帯 × m) の差[円]**（**カッコは標本R**・"
          f"**{MINCELL}未満は「—」**）")
    print(f"{'帯':<10}" + "".join(f"{lab:>17}" for _, lab in MBINS))
    for bi, (nm, lo, hi) in enumerate(DEG):
        line = ""
        for mi in range(len(MBINS)):
            v = cells[(bi, mi)]
            line += (f"{np.mean(v):>+9.1f}({len(v):>5,})" if len(v) >= MINCELL
                     else f"{'—':>17}")
        print(f"{nm:<10}{line}")

    A, B, mu = two_way(cells)
    na, nb = np.zeros(NPERM), np.zeros(NPERM)
    arrs = {k: np.asarray(v, float) for k, v in cells.items()}
    for t in range(NPERM):
        pm = {k: v * rng.choice([-1.0, 1.0], size=len(v)) if len(v) else v
              for k, v in arrs.items()}
        na[t], nb[t], _ = two_way(pm)
    thr = float(np.quantile(np.maximum(np.abs(na), np.abs(nb)), 0.99))
    print(f"\n■ ★★★主判定（**帰無は対応のある符号反転{NPERM:,}回・maxTで2つを制御**）")
    print("　★ゲート2: **仮説が偽なら穴と乱は交換可能＝A も B も 0**")
    print(f"\n　★**maxT の帰無99%点 = {thr:.3f}**")
    okA, okB = abs(A) > thr, abs(B) > thr
    print(f"　**A（m の効果・各帯の中で m 方向） = {A:+.3f}** → "
          f"**{'★超えた' if okA else '⚠超えない'}**")
    print(f"　**B（帯の効果・各 m の中で 帯 方向） = {B:+.3f}** → "
          f"**{'★超えた' if okB else '⚠超えない'}**")

    print(f"\n■ ★★読み方（事前登録の表）")
    if okA and not okB:
        print("　★★★**差を作っているのは「何頭から選ぶか」であって「程度」ではない**。")
        print("　→ ★**(180)の「差が程度とともに増える」は候補頭数の交絡だった**。")
        print("　　**(181)で形が消えたこととも整合する**。")
    elif okB and not okA:
        print("　★**程度の効果が本物**。**(180)の読みが正しかった**。")
    elif okA and okB:
        print(f"　★**両方効いている**（A={A:+.3f} / B={B:+.3f}）。"
              f"**{'m' if abs(A) > abs(B) else '帯'}のほうが大きい**（記述）。")
    else:
        print("　⚠**どちらも通らない**＝**(180)の形は、どちらの説明でも支えられない**。")
        print("　→ ★**(181)で形が消えたことと整合する。ここで閉じる**。")
    print("\n⚠**水準の結論は動かない**（**最良88.4%／93.1%で100%に届かない**）。")
    print("⚠**枠連の運用には触れない**。**設定変更は提案しない**。")


if __name__ == "__main__":
    sys.exit(selftest() if "--selftest" in sys.argv else (main() or 0))
