"""(208) ★★★**軸にオッズの下限を置く** —— 元の問い（穴馬）に戻る

★★**動機（2026-09-07・利用者との対話から）**:
　★**複勝の軸（ズレ≥0.15）のオッズ分布を測ったら、★穴馬ではなかった**:
| | 本数 | 平均 | ★**中央値** | ★**5倍未満** | ★**平均人気** |
|---|---|---|---|---|---|
| ★**複勝の軸** | 6,337 | 7.1倍 | ★**3.5倍** | ★**61.0%** | ★**2.8番人気** |
| （参考）全出走馬 | 408,271 | 67.7倍 | 24.3倍 | 13.8% | — |
　⚠**平均7.1倍に騙されていた**——**中央値3.5倍で、少数の高オッズが平均を引き上げていた**。
　★★**そして人気順位で切ると、1番人気が41%を占め、そこが最も悪い**:
| 人気 | 割合 | ★複勝ROI |
|---|---|---|
| ⚠**1番人気** | ⚠**41.0%** | ⚠**92.6%** |
| 2-3番人気 | 30.2% | 96.9% |
| ★**4-6番人気** | 19.5% | ★**99.5%** |
| ★**7番人気以下** | 9.4% | ★**99.3%** |
　→ ★**穴側のほうが良い**。**1番人気を混ぜていることが水準を押し下げている可能性**。

⚠⚠**これは post-hoc**（**人気順位で切った結果を見てから言っている**・判定基準4）。
　★**だから前半で下限を選び、後半で1回だけ試す**（(201)で確立した形）。
　⚠**さらに正直に書いておく**: **私は全期間の post-hoc の向きを既に知っている**ので、
　　**後半がそれを裏づけても、★真の盲検よりは弱い証拠**にしかならない。**それでも最善**。

────────────────────────────────────────────────────────────
★★★ 事前登録（2026-09-07・**結果を見る前にコミットする**）
────────────────────────────────────────────────────────────

■ ★経路（判定基準25）: **q = MLモデル / q_pool = 複勝の板**。⚠**q側は弱い**。
■ ★母集団: **(203)(206)と同じ**（**軸 = pn≥0.15 かつ gap≥0.15 の中で pn最大**）。
　★**そこに「軸の単勝オッズ ≥ L」を足す**。**候補が無ければそのレースは見送る**。
　**L ∈ {1.0（＝現行・下限なし）, 3.0, 5.0, 7.0, 10.0}**。

■ ★★★**水準は (88) と混ざる**（**先に書いておく**）
　⚠**Lを上げると別のオッズ帯に移り、その帯の払戻率の癖（(88)）を拾う**。
　★★**だから主判定は「同じオッズ帯の乱との対応差」**（**乱は10種平均・種はレースに紐づける**）。
　★**水準（複勝ROI）は記述として併記するが、L間で直接比べない**。

■ ★★★家族A（**5比較**）: **各Lの「軸 − 乱」の対応差**（**全期間・複勝1点**）。
　⚠**post-hocのデータを含むので主判定ではない**。
■ ★★★★家族B（**2比較・★主判定**）: ★**前半(2016-2020)でLを選び、後半(2021-)で1回だけ試す**
　★**選ぶ基準は前半の対応差のみ**。**① 本 ② プラセボ（乱）**。
| ★**後半でL>1が現行(L=1)を上回る** | ⚠**上回らない** |
|---|---|
| **オッズ下限は効く**。**穴側に寄せる価値がある** | ⚠**post-hocの下駄だった**。**現行のまま** |

■ ★★ゲート2（判定基準42）——**仮説が偽なら何を返すか**
　★**穴と乱が同じ帯で交換可能なら、対応差の期待値は厳密に0**。
　★**L=1.0 は現行そのもの**——**(206)の 95.9% / 6,336R を厳密に返すはず**。

■ ⚠ゲート1: (88)③④を再現（±3pt）／ ★ゲート板: 復元R 0.800±0.020 ／
　★陽性対照: 三連複BOX上位4 81.9%±1.5pt。
■ ★★★内部対照（**決定的・乱を使わない**）: **L=1.0 の複勝ROI が 95.9% ±0.2pt かつ 6,336R ±5**。
　⚠**(195)(197)(201)(204)で内部対照を落とした原因は全部「揺れる量／違う母集団に厳しい許容」**。
　★**今回は乱を1粒も使わない量で照合する**。

■ ★★★探索を守る（**7比較・Bonferroni α=0.01/7**）
　⚠**標本300レース未満は判定しない**（判定基準5）。
　★**採用条件に「Lについて単調」**——**孤立した1点は採らない**。

■ ★記述（判定しない）
　**各Lの 買うレース数 / 買う率 / 軸の平均・中央オッズ / 平均人気 / 複勝ROI / 乱のROI /
　　的中率 / 1日あたりの本数**。

■ ★採用条件
　1. ★**家族Bで、前半に選ばれたLの後半対応差が、後半のL=1.0の対応差を明確に上回る**
　2. **本の縮み幅がプラセボの縮み幅より小さい**
　3. **家族AでLについて単調**
　4. **後半の水準（複勝ROI）がL=1.0より高い**

■ 予想（⚠**当てにしない**・判定基準24。★**私は(198)(201)(203)で3回外した**）
　★**対応差はLを上げると増えると見る**（**post-hocの向き**）。⚠**だが本数が減るので検出力が落ちる**。
　⚠**水準が上がっても、それは(88)の帯の効果かもしれない**——**対応差で見ないと分からない**。
　★**もし後半で差が出なければ、「穴に寄せても変わらない」と結論する**。

実行: python3 ml/audit_ana_floor.py    自己テスト: python3 ml/audit_ana_floor.py --selftest
"""
import math
import sys
from itertools import combinations
from zlib import crc32

import numpy as np

sys.path.insert(0, "ml")
import features as F
from audit_crosspool import load_races, payoff, zq
from audit_ana_odds import BANDS, MIN_HORSES, band_of, gate1, roi_of
from audit_ana_marg import WF_BOX4, WF_TOL, wf_predict
from audit_ana_board import BOARD_R, BOARD_TOL, NPLACE, load_fuku_boards, qpool
from audit_ana_band import PN_FLOOR
from train_prod import add_odds_features

GAP = 0.15
LS = [1.0, 3.0, 5.0, 7.0, 10.0]
SPLIT = 2021
NRAND = 10
SEED = 20260906
MINCELL = 300
KNOWN_ROI, KNOWN_N, ROI_TOL, N_TOL = 95.9, 6336, 0.2, 5
NCMP = len(LS) + 2          # 5 + 2
ALPHA = 0.01


def selftest():
    ok = True
    z = zq(ALPHA / NCMP)
    print(f"★軸のオッズ下限 L: {LS}（**1.0 は現行＝下限なし**）")
    print(f"★家族A {len(LS)}比較（全期間の対応差）＋ ★家族B 2比較"
          f"（**前半でLを選び後半で1回・主判定**）→ **{NCMP}比較**・z = {z:.3f}")
    print("★★水準は(88)と混ざるので、★主判定は「同じオッズ帯の乱との対応差」")
    rng = np.random.default_rng(0)
    n = 120_000
    pay = rng.choice([0.0, 250.0, 900.0], size=n, p=[0.62, 0.30, 0.08])
    m = float(np.mean([(pay[rng.permutation(n)] - pay[rng.permutation(n)]).mean()
                       for _ in range(200)]))
    print(f"★ゲート2（対応差）: **偽なら0** → {m:+.3f}円　{'★OK' if abs(m) < 2 else '⚠NG'}")
    ok &= abs(m) < 2
    print(f"★★★内部対照（決定的・乱を使わない）: **L=1.0 の複勝ROI が {KNOWN_ROI}% "
          f"±{ROI_TOL}pt かつ {KNOWN_N:,}R ±{N_TOL}**")
    print("★★主判定の読み方: **後半でL>1が現行を上回る→穴側に寄せる価値がある / "
          "上回らない→post-hocの下駄**")
    print("⚠**私は全期間のpost-hocの向きを既に知っている**"
          "——**後半が裏づけても真の盲検よりは弱い証拠**")
    print("★自己テスト: " + ("全部OK" if ok else "⚠NG"))
    return 0 if ok else 1


def main():
    z = zq(ALPHA / NCMP)
    print("(208) ★★★**軸にオッズの下限を置く** —— 元の問い（穴馬）に戻る")
    print("★複勝の軸は中央値3.5倍・平均2.8番人気・61%が5倍未満＝**穴馬ではなかった**\n")

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

    K = {L: {"a": [], "r": [], "od": [], "rk": [], "yr": []} for L in LS}
    full = []
    Rs, box4, nall, ndays = [], [], 0, set()
    for rid, g in sub.groupby("raceid"):
        rid = str(rid)
        r = races.get(rid)
        bd = boards.get(rid)
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
        ndays.add(gg["date"].iloc[0].date())
        pn = pv / pv.sum() * NPLACE
        qp, Rb = qpool([bd[int(u)] for u in ub], "harm")
        Rs.append(Rb)
        gap = pn - qp
        bx = [payoff(r, "三連複", list(c))
              for c in combinations(sorted(int(u) for u in ub[np.argsort(-pv)[:4]]), 3)]
        if not any(x is None for x in bx):
            box4.append(sum(bx) - 400.0)
        bi = np.array([band_of(float(o), BANDS) for o in od])
        rank = np.argsort(np.argsort(od)) + 1
        yr = int(gg["date"].iloc[0].year)
        base = (pn >= PN_FLOOR) & (gap >= GAP)
        # ★内部対照（乱を使わない・下限なし）
        c0 = np.where(base)[0]
        if len(c0):
            i0 = int(c0[int(np.argmax(pn[c0]))])
            v0 = payoff(r, "複勝", [int(ub[i0])])
            if v0 is not None:
                full.append(v0)
        for L in LS:
            cand = np.where(base & (od >= L))[0]
            if not len(cand):
                continue
            i = int(cand[int(np.argmax(pn[cand]))])
            va = payoff(r, "複勝", [int(ub[i])])
            if va is None:
                continue
            vr, okall = [], True
            for sd in range(NRAND):
                g2 = np.random.default_rng([SEED + sd, crc32(rid.encode())])
                pl = [int(ub[q]) for q in range(len(ub)) if bi[q] == bi[i] and q != i]
                if not pl:
                    okall = False
                    break
                v = payoff(r, "複勝", [int(g2.choice(pl))])
                if v is None:
                    okall = False
                    break
                vr.append(v)
            if not okall:
                continue
            c = K[L]
            c["a"].append(va); c["r"].append(float(np.mean(vr)))
            c["od"].append(float(od[i])); c["rk"].append(int(rank[i])); c["yr"].append(yr)
    print(f"\n★対象 **{nall:,}レース / {len(ndays):,}開催日**")

    med = float(np.median(Rs))
    okR = abs(med - BOARD_R) <= BOARD_TOL
    print(f"■ ★ゲート板: 復元R = **{med:.4f}** → **{'★立った' if okR else '⚠⚠落ちた'}**")
    box4 = np.asarray(box4, float)
    g0 = 100.0 * (box4.sum() + 400.0 * len(box4)) / (400.0 * len(box4))
    okb = abs(g0 - WF_BOX4) <= WF_TOL
    print(f"■ ⚠**陽性対照**: BOX上位4 **{g0:.1f}%** vs {WF_BOX4}%"
          f" → **{'★立った' if okb else '⚠⚠落ちた'}**")
    fv = np.asarray(full, float)
    okc = (abs(roi_of(fv) - KNOWN_ROI) <= ROI_TOL and abs(len(fv) - KNOWN_N) <= N_TOL)
    print(f"■ ★★★内部対照（決定的・乱を使わない）: L=1.0 複勝ROI = **{roi_of(fv):.1f}%**"
          f"（{len(fv):,}R） vs (206) {KNOWN_ROI}%（{KNOWN_N:,}R）"
          f" → **{'★再現' if okc else '⚠⚠ズレた'}**")
    if not (okR and okb and okc):
        print("\n⚠⚠**ゲートが落ちた。読まない**（判定基準32）。")
        return

    print(f"\n{'='*118}")
    print("■ ★★★家族A: **各Lの「軸 − 乱」の対応差**（**全期間**・⚠post-hocのデータを含む）")
    print(f"\n{'下限L':>7}{'R数':>8}{'買う率':>8}{'1日':>7}{'平均':>8}{'中央':>8}{'人気':>7}"
          f"{'複勝ROI':>9}{'乱':>8}{'的中率':>8}{'★対応差':>10}{'99%CI':>19}{'判定':>12}")
    for L in LS:
        c = K[L]
        a = np.asarray(c["a"], float); rv = np.asarray(c["r"], float)
        if len(a) < MINCELL:
            print(f"{L:>7.1f}{len(a):>8,}　⚠**標本不足**")
            continue
        dd = a - rv
        mu, se = dd.mean(), dd.std(ddof=1) / math.sqrt(len(dd))
        od = np.asarray(c["od"], float)
        print(f"{L:>7.1f}{len(a):>8,}{100*len(a)/nall:>7.1f}%{len(a)/len(ndays):>6.1f}本"
              f"{od.mean():>7.1f}倍{np.median(od):>7.1f}倍{np.mean(c['rk']):>6.1f}番"
              f"{roi_of(a):>8.1f}%{roi_of(rv):>7.1f}%{100*np.mean(a>0):>7.1f}%"
              f"{mu:>+9.1f}円{f'[{mu-z*se:+.1f},{mu+z*se:+.1f}]':>19}"
              f"{('★★有意' if mu - z*se > 0 else '⚠通らない'):>12}")

    print(f"\n{'='*118}")
    print("■ ★★★★家族B（**主判定**）: ★**前半でLを選び、後半で1回だけ試す**")
    tbl = []
    for L in LS:
        c = K[L]
        yr = np.asarray(c["yr"], int)
        h = yr < SPLIT
        if h.sum() < MINCELL or (~h).sum() < MINCELL:
            continue
        a = np.asarray(c["a"], float); rv = np.asarray(c["r"], float)
        dd = a - rv
        tbl.append({"L": L, "n1": int(h.sum()), "n2": int((~h).sum()),
                    "d1": dd[h].mean(), "d2": dd[~h].mean(), "v2": dd[~h],
                    "roi1": roi_of(a[h]), "roi2": roi_of(a[~h]),
                    "rr2": roi_of(rv[~h]),
                    "r1": (a[h] - rv[h]), "rv1": rv[h]})
    print(f"\n{'下限L':>7}{'前半R':>8}{'後半R':>8}{'★前半の差':>11}{'★後半の差':>11}"
          f"{'縮み':>9}{'前半ROI':>9}{'後半ROI':>9}{'乱後半':>9}")
    for x in tbl:
        print(f"{x['L']:>7.1f}{x['n1']:>8,}{x['n2']:>8,}{x['d1']:>+10.1f}円"
              f"{x['d2']:>+10.1f}円{x['d1']-x['d2']:>+8.1f}円"
              f"{x['roi1']:>8.1f}%{x['roi2']:>8.1f}%{x['rr2']:>8.1f}%")
    base = [x for x in tbl if x["L"] == 1.0]
    b2 = base[0]["d2"] if base else float("nan")
    br2 = base[0]["roi2"] if base else float("nan")
    best = max(tbl, key=lambda x: x["d1"])
    v = np.asarray(best["v2"], float)
    mu, se = v.mean(), v.std(ddof=1) / math.sqrt(len(v))
    print(f"\n★**本（モデル）**: 前半で選ばれた下限 **L={best['L']:.1f}**"
          f"（前半 {best['d1']:+.1f}円）")
    print(f"　★★**後半（1回だけ）**: **{mu:+.1f}円**　99%CI [{mu-z*se:+.1f},{mu+z*se:+.1f}]")
    print(f"　★★**後半のL=1.0 は {b2:+.1f}円** → **差 {mu-b2:+.1f}円**"
          f"　→ **{'★★現行を上回った' if mu > b2 else '⚠上回らなかった'}**")
    print(f"　★**後半の水準**: L={best['L']:.1f} で **{best['roi2']:.1f}%** "
          f"／ L=1.0 で {br2:.1f}%　→ **{'★上回った' if best['roi2'] > br2 else '⚠上回らない'}**")
    print(f"　★**縮み {best['d1']-mu:+.1f}円**")

    print("\n■ ★採用条件: **1.後半でL=1を上回る（主判定） / 2.縮みがプラセボより小 / "
          "3.Lについて単調 / 4.後半の水準もL=1より上**")
    print("⚠**私は全期間のpost-hocの向きを既に知っている**"
          "——**後半が裏づけても真の盲検よりは弱い証拠**（**先に書いた通り**）。")
    print("\n⚠**枠連の運用には触れない**。**設定変更は提案しない**。")


if __name__ == "__main__":
    sys.exit(selftest() if "--selftest" in sys.argv else (main() or 0))
