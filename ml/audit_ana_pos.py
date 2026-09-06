"""(187) ★★★**穴を「どこに置くか」で分けて測る** — 1着 / 2着 / 3着

★★**問い（ユーザー・2026-09-06）**——「**k=2 の馬単は2通り買ってるってこと？**」
　→ ★**違った。(182)(185)の馬単は「軸(穴)を1着固定」の1点だけ**（`[(ax, hk)]`）。
　→ ★★**そして「穴を1着以外に置く」は、この線で一度も測っていない**。

| 券種 | 測った形 | ★**測っていない形** |
|---|---|---|
| **馬単** | 穴 → 紐（**穴が1着**） | ★**紐 → 穴**（**穴が2着**） |
| **三連単** | 穴 → 紐 → 紐（**穴が1着**） | ★**紐 → 穴 → 紐**（2着）／**紐 → 紐 → 穴**（3着） |

★★**機構的に筋が通る**——**穴は「勝つ」より「絡む」ほうが起きやすい**。
　**1着固定はいちばん厳しい置き方**で、**そこだけを測っていた**。

⚠**近い先行はあるが同じではない**:
　**(54)** 三連単の固定位置の工夫 → ❌閉鎖。⚠**軸はモデル確率1位**。
　**(84)③** 三連単の固定位置 × 軸のシェア十分位 → **ρ=+0.600 で判定閾値0.6に届かず**。⚠**軸はモデル1位**。
　**(178)** 馬単の両方向 → ❌陰性。⚠**穴×1番人気の2軸で、方向を分けていない**。
→ ★**「穴を、1着/2着/3着 のどこに置くか」を分けて測ったことは無い**。

⚠⚠**先に逆風を書く**——
　★**(180)(183)(184)で「穴の複勝（＝3着以内に来るか）」は測ってあり、最良でも88.4%**。
　**三連単で穴を3着に置くのは、その情報を使う形の一つ**なので、**複勝を大きく超える理由は見当たらない**。
　⚠**払戻率は 馬単75.0% / 三連単72.5%** で、**複勝80.0%より5.0〜7.5pt不利なところから始まる**。
　★**それでも「測っていない」のは事実**なので潰す。

────────────────────────────────────────────────────────────
★★★ 事前登録（2026-09-06・**結果を見る前にコミットする**）
────────────────────────────────────────────────────────────

■ ★経路（判定基準25）: **q = MLモデルの top3 確率 / q_pool = 複勝の板**。⚠**q 側は弱い**。
■ ★手続き: **ウォークフォワード**（`data/cache/wf_pred.npz` を再利用）。
■ ★軸・紐は(185)と同一: **穴 = 5-20倍で `p_norm − q_pool_fuku` 最大** ／ **紐 = モデル確率の降順**。

■ ★★k頭目で増える組（**軸の位置ごと**）
| 券種 | 軸の位置 | k頭目で増える組 | 点数 |
|---|---|---|---|
| **馬単** | **1着** | (穴, 紐k) | 1 |
| **馬単** | ★**2着** | ★**(紐k, 穴)** | 1 |
| **三連単** | **1着** | (穴,紐k,紐j) / (穴,紐j,紐k) | 2(k−1) |
| **三連単** | ★**2着** | ★**(紐k,穴,紐j) / (紐j,穴,紐k)** | 2(k−1) |
| **三連単** | ★**3着** | ★**(紐k,紐j,穴) / (紐j,紐k,穴)** | 2(k−1) |
　★**どの位置でも点数が同じ**なので、**位置どうしを直接比べられる**（判定基準8の心配が無い）。

■ ★★★主判定 = **限界ROI − 100%**（**27比較**: 馬単 2位置×k=1〜6 ＋ 三連単 3位置×k=2〜6）
　**Bonferroni α=0.01/27**。
　★**ゲート2（判定基準42）**: **仮説が偽（k頭目をその位置に足すのが損益に中立）なら
　　「1点あたり払戻 − 100円」の期待値は厳密に0を返す**。**限界ROIは点数で割ってあるので
　　位置や幅を変えても自動では動かない**。

■ ⚠ゲート1: (88)③④を別パーサで再現（±3pt）。
■ ★ゲート板: **復元R = 3/Σ(1/調和平均) の中央値が 0.800±0.020**。
■ ★陽性対照: **三連複BOX上位4が 81.9%±1.5pt**。
■ ★★内部対照: ★**「1着」の限界ROIが(185)の値を±2ptで再現**すること
　　（**馬単 84.2/93.1/85.8/86.9/89.4/67.9%・三連単 89.9/91.0/86.2/83.6/72.0%**）。
　　**再現しなければ装置が違う**。

■ ★記述（判定しない）
　1. ★**位置ごとの的中率**——**「穴は勝つより絡むほうが起きやすい」が数字で見えるはず**。
　2. **位置ごとの平均配当**——**当たりやすい位置ほど配当が小さいはず（市場が織り込む）**。
　★**この2つが打ち消し合ってROIが動かないなら、それが答え**。

■ ★採用条件（判定基準39/40/41）
　1. **限界ROIの99%CI下端が100%を超える (位置, k) が存在する**
　2. ★**同じ位置で複数の k が同じ向き**（**1マスだけ跳ねるのは(172)で崩れた形**）
　3. ★**券種をまたいで同じ位置が良い**（**馬単の2着が良いなら三連単の2着も良いはず**）
　⚠**通らなければ「穴をどこに置いても、紐を1頭足す増分は常に損」と書いて閉じる**。

■ ⚠**最良のマスを拾わない**——**(178)でプラセボが109.9%を出した**。**主判定だけを読む**。

■ 予想（⚠**類推なので当てにしない**・判定基準24）
　★**的中率は 3着 > 2着 > 1着 になるが、配当がそれを打ち消してROIは横並びになる**と見る
　（**市場が織り込んでいるなら、そうなるのが自然**）。⚠**どちらでも驚かない**。

実行: python3 ml/audit_ana_pos.py    自己テスト: python3 ml/audit_ana_pos.py --selftest
"""
import math
import sys
from itertools import combinations

import numpy as np

sys.path.insert(0, "ml")
import features as F
from audit_crosspool import load_races, payoff, zq
from audit_ana_odds import COST, MIN_HORSES, gate1, roi_of
from audit_ana_marg import ANA_HI, ANA_LO, KMAX, WF_BOX4, WF_TOL, wf_predict
from audit_ana_board import BOARD_R, BOARD_TOL, NPLACE, load_fuku_boards, qpool
from train_prod import add_odds_features

# (券種, 軸の位置, kの下限)
SPECS = [("馬単", 1, 1), ("馬単", 2, 1),
         ("三連単", 1, 2), ("三連単", 2, 2), ("三連単", 3, 2)]
NCMP = 2 * KMAX + 3 * (KMAX - 1)          # 12 + 15 = 27
ALPHA = 0.01
SEED = 20260906
INNER_TOL = 2.0
# (185)の「1着」実測（内部対照）
KNOWN = {("馬単", 1, 1): 84.2, ("馬単", 1, 2): 93.1, ("馬単", 1, 3): 85.8,
         ("馬単", 1, 4): 86.9, ("馬単", 1, 5): 89.4, ("馬単", 1, 6): 67.9,
         ("三連単", 1, 2): 89.9, ("三連単", 1, 3): 91.0, ("三連単", 1, 4): 86.2,
         ("三連単", 1, 5): 83.6, ("三連単", 1, 6): 72.0}


def combos_at(kind, pos, ax, himo, k):
    """★軸を pos 着に置いたとき、k頭目の紐を足すと増える組。"""
    hk = himo[k - 1]
    if kind == "馬単":
        return [(ax, hk)] if pos == 1 else [(hk, ax)]
    out = []
    for j in range(k - 1):
        hj = himo[j]
        if pos == 1:
            out += [(ax, hk, hj), (ax, hj, hk)]
        elif pos == 2:
            out += [(hk, ax, hj), (hj, ax, hk)]
        else:
            out += [(hk, hj, ax), (hj, hk, ax)]
    return out


def selftest():
    ok = True
    ax, himo = 9, [3, 7, 1, 5, 2, 8]
    assert combos_at("馬単", 1, ax, himo, 2) == [(9, 7)]
    assert combos_at("馬単", 2, ax, himo, 2) == [(7, 9)]
    print("★馬単の自己テスト: 1着→(9,7) / ★2着→(7,9)　★OK")
    for pos in (1, 2, 3):
        c = combos_at("三連単", pos, ax, himo, 3)
        assert len(c) == 4
        # ★軸が指定の位置にちょうど入っている
        assert all(t[pos - 1] == ax for t in c), (pos, c)
        # ★軸以外は紐3と紐j（j<3）
        assert all(sorted(x for x in t if x != ax) == sorted([1, himo[j]])
                   for t, j in zip(c, (0, 0, 1, 1)))
    print("★三連単の自己テスト: 1着/2着/3着とも4点・軸が指定位置に入る　★OK")
    # ★どの位置でも点数が同じ（位置どうしを直接比べられる）
    for k in range(2, KMAX + 1):
        n = {p: len(combos_at("三連単", p, ax, himo, k)) for p in (1, 2, 3)}
        assert len(set(n.values())) == 1 == len({2 * (k - 1)} & set(n.values()))
    print("★点数の自己テスト: 三連単は 1着/2着/3着 とも 2(k−1)点で同じ　★OK")
    # ★ゲート2
    rng = np.random.default_rng(0)
    n = 200_000
    pay = np.where(rng.random(n) < 0.10, 1000.0, 0.0)
    m = float(np.mean([(rng.permutation(pay) - 100.0).mean() for _ in range(200)]))
    print(f"★ゲート2の自己テスト: 中立な増分200回の平均 {m:+.3f}円"
          f" → **仮説が偽なら0を返す**: {'★OK' if abs(m) < 2 else '⚠NG'}")
    ok &= abs(m) < 2
    print(f"★比較数 {NCMP} → z = {zq(ALPHA/NCMP):.3f}")
    print(f"★内部対照 {len(KNOWN)} マス（(185)の「1着」の実測）")
    print("★自己テスト: " + ("全部OK" if ok else "⚠NG"))
    return 0 if ok else 1


def main():
    z = zq(ALPHA / NCMP)
    print("(187) ★★★**穴を「どこに置くか」で分けて測る** — 1着 / 2着 / 3着")
    print("★(182)(185)の馬単・三連単は**軸(穴)を1着固定**の形しか測っていなかった")
    print("★穴は「勝つ」より「絡む」ほうが起きやすいので、1着固定はいちばん厳しい置き方\n")

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
    acc = {(kd, ps, k): [] for kd, ps, lo in SPECS for k in range(lo, KMAX + 1)}
    Rs, box4, nrace = [], [], 0
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
        idx = np.where((od >= ANA_LO) & (od < ANA_HI))[0]
        if not len(idx):
            continue
        ax = int(ub[int(idx[int(np.argmax(gapf[idx]))])])
        himo = [u for u in order if u != ax]
        if len(himo) < KMAX:
            continue
        used = False
        for kd, ps, lo in SPECS:
            for k in range(lo, KMAX + 1):
                cs = combos_at(kd, ps, ax, himo, k)
                vs = [payoff(r, kd, list(c)) for c in cs]
                if any(v is None for v in vs):
                    continue
                acc[(kd, ps, k)].append(sum(vs) / len(cs))
                used = True
        nrace += used
    print(f"\n★突き合わせ {nrace:,}レース")

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

    print(f"\n■ ★★内部対照: **「1着」が(185)を再現するか**（許容±{INNER_TOL:.0f}pt）")
    ng = 0
    for (kd, ps, k), known in KNOWN.items():
        v = np.asarray(acc[(kd, ps, k)], float)
        if len(v) < 300:
            continue
        roi = 100.0 * v.mean() / COST
        if abs(roi - known) > INNER_TOL:
            ng += 1
            print(f"　⚠{kd} k={k}: {roi:.1f}% vs (185) {known:.1f}%　差 {roi-known:+.1f}pt")
    print(f"　→ **{'⚠⚠' + str(ng) + 'マスがズレた' if ng else '★全' + str(len(KNOWN)) + 'マスで再現'}**")

    print(f"\n{'='*104}")
    print(f"■ ★★★主判定: **k頭目の限界ROI（軸の位置ごと）**"
          f"（**{NCMP}比較・α={ALPHA}/{NCMP}・z={z:.3f}**）")
    print("　★ゲート2: **仮説が偽なら「1点あたり払戻−100円」の期待値は0**")
    print("　★**どの位置も点数が同じ**なので、**位置どうしを直接比べられる**")
    hits = []
    for kd in ("馬単", "三連単"):
        ps_list = [p for k2, p, _ in SPECS if k2 == kd]
        lo = [l for k2, p, l in SPECS if k2 == kd][0]
        print(f"\n── {kd} ──")
        print(f"{'k':>3}{'点数':>6}" + "".join(f"{'軸' + str(p) + '着':>21}" for p in ps_list))
        for k in range(lo, KMAX + 1):
            line = ""
            for ps in ps_list:
                v = np.asarray(acc[(kd, ps, k)], float)
                if len(v) < 300:
                    line += f"{'—':>21}"
                    continue
                mu = v.mean() - COST
                se = v.std(ddof=1) / math.sqrt(len(v))
                roi = 100.0 * v.mean() / COST
                sig = mu - z * se > 0
                if sig:
                    hits.append((kd, ps, k, roi))
                mark = "★★" if sig else ""
                line += f"{roi:>8.1f}%[{100+mu-z*se:.0f},{100+mu+z*se:.0f}]{mark:>2}"
            npts = len(combos_at(kd, ps_list[0], 99, list(range(1, KMAX + 1)), k))
            print(f"{k:>3}{npts:>6}{line}")

    print(f"\n■ 記述: **位置ごとの的中率と平均配当**"
          f"（★**「穴は勝つより絡む」が見えるはず**）")
    print(f"{'券種':<8}{'軸の位置':>9}{'的中率':>10}{'的中時の平均配当':>18}{'限界ROI(平均)':>15}")
    for kd, ps, lo in SPECS:
        vs = np.concatenate([np.asarray(acc[(kd, ps, k)], float)
                             for k in range(lo, KMAX + 1)
                             if len(acc[(kd, ps, k)]) >= 300])
        hit = float(np.mean(vs > 0))
        avg = float(vs[vs > 0].mean()) if (vs > 0).any() else 0.0
        print(f"{kd:<8}{ps:>8}着{100*hit:>9.2f}%{avg:>16,.0f}円"
              f"{100*vs.mean()/COST:>14.1f}%")

    print(f"\n■ ★採用条件")
    print(f"　1. 限界ROIの99%CI下端が100%超 … **{len(hits)}マス**"
          f" → {'★満たす' if hits else '⚠満たさない'}")
    if not hits:
        print("\n★★★**結論: 穴をどこに置いても、紐を1頭足す増分は常に損**。")
        print("★**(182)(185)の「組を買うこと自体が損」は、置き位置を変えても保つ**。")
        print("⚠**経路は弱いので「閉じた」とは書けない**（判定基準25）。")
        return
    print("\n■ ★裾の検算（**通ったマス**・(77)）")
    for kd, ps, k, roi in hits:
        v = np.asarray(acc[(kd, ps, k)], float)
        print(f"　{kd} 軸{ps}着 k={k}: 限界ROI {roi:.1f}% / "
              f"**上位3本が全払戻の {100*np.sort(v)[-3:].sum()/max(v.sum(),1e-9):.1f}%**")
    print("\n⚠**枠連の運用には触れない**。**設定変更は提案しない**。")


if __name__ == "__main__":
    sys.exit(selftest() if "--selftest" in sys.argv else (main() or 0))
