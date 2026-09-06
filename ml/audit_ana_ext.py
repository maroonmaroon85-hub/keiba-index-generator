"""(191) ★★★**未測定を2つ潰す** — (A) ズレ床の延長 / (B) 紐の順序

★★**動機（ユーザー・2026-09-06）**——
　「**買い方・レース選びで数値改善してるから、もっと別軸で数値良くなりそうじゃない？
　　試してないものは全部やってみよう**」
★**指摘のとおり水準は上がってきている**:
| | 最良の複勝ROI |
|---|---|
| (174) 単勝価格・帯内top10% | 87.6%（⚠単一分割で2〜3pt楽観） |
| (184) 板ベース・帯内最大 | **88.4%** |
| ★**(189) 推奨度＋ズレの絶対閾値** | ★**93.6%** |

────────────────────────────────────────────────────────────
■ ★★家族A: **ズレ床の延長**（**(189)は0.08で止めた**）
　★**(189)の実測は4行すべてで単調に上がり続けていた**:
　　**0.15の行 91.5 → 91.7 → 92.2 → 93.1%** ／ **0.20の行 91.5 → 91.8 → 92.5 → 93.6%**
　→ ★★**格子の端で切ったので、その先を見ていない**。**延長する**。
　⚠**標本は急速に減る**（**0.08で既に買うR 50%前後**）。**判定基準5に当たる見込み**。

■ ★★家族B: **紐の順序**（**この線で一度も変えていない**）
　⚠**軸と紐で選ぶ基準が違っていた**:
| | 選び方 |
|---|---|
| **軸（穴）** | ★**gap（板とのズレ）が最大** |
| **紐** | ★**モデル確率 p が高い順** |
　★**(172)は幅(CUTS)と軸(AXES)を振ったが、紐の順序は常にモデル確率の降順**だった。
　★★**特に「人気順」の対照が無い**——**判定基準7/(69)は「人気順で同じ買い方をした対照を必ず並べる」
　　と定めており、軸については毎回やっているが、紐については一度もやっていない**。

────────────────────────────────────────────────────────────
★★★ 事前登録（2026-09-06・**結果を見る前にコミットする**）
────────────────────────────────────────────────────────────

■ ★経路（判定基準25）: **q = MLモデル / q_pool = 複勝の板**（調和平均）。⚠**q 側は弱い**。
■ ★手続き: **ウォークフォワード**（`data/cache/wf_pred.npz` を再利用）。
■ ★レースの選び方は(189)と同一: **`p_norm ≥ A かつ gap_f ≥ B` の馬が居るレースだけ買う**。

■ ★★家族A（**10比較**）
　**推奨度の床 {0.15, 0.20} × ズレの床 {0.08, 0.10, 0.12, 0.15, 0.20}**。**券種は複勝1点のみ**
　（★**(174)の実測で単勝は検出限界に負ける**・**複勝が最も検出力がある**）。
　★**主判定: ROI の99%CI下端が 100% を超えるか**。
　⚠**標本300未満、または99%CI半幅が±20ptを超えるマスは「判定不能」**（判定基準5）。

■ ★★家族B（**24比較**）
　**絞りは2つ**（**(0.15,0.02) と (0.20,0.06)**・**(190)と揃える**）。
　**券種4つ**（馬連 / 馬単 / 三連複 / 三連単）。**幅は紐2頭に固定**
　（★**4券種すべてが定義でき、(190)で三連単の103.1%が出た幅**）。
　**紐の順序4通り**:
| | 紐の基準 | 位置づけ |
|---|---|---|
| **現行** | **モデル確率 p の降順** | **基準** |
| ★**人気** | ★**単勝オッズの昇順** | ★★**判定基準7/(69)が要求する対照** |
| ★**ズレ** | ★**gap_f の降順**（軸と同じ基準） | **一貫性の検定** |
| ★**板** | ★**板の複勝確率の降順** | **市場の複勝評価** |
　★**主判定: 各代替 − 現行 の対応差（円）**。**2絞り × 4券種 × 3代替 = 24比較**。
　★**同じレース・同じ点数・同じ券種**なので**対応比較**。

■ ★★ゲート2（判定基準42）——**仮説が偽なら何を返すか**
　★**家族A**: **仮説が偽（板が正しく値付けしている）なら、どのマスでもROIは払戻率80.0%を返す**。
　★**家族B**: **仮説が偽（紐の順序が損益について何も持たない）なら、対応差の期待値は厳密に0**。
　　**点数もコストも券種もレースも同一**＝**(170)の形にならない**。

■ ⚠ゲート1: (88)③④を再現（±3pt）／ ★ゲート板: 復元R 0.800±0.020 ／
　★陽性対照: 三連複BOX上位4 81.9%±1.5pt。
■ ★★内部対照: **家族Aの(0.15,0.08)と(0.20,0.08)が(189)を±1ptで再現**すること
　　（**93.1% / 93.6%**）。**家族Bの「現行」が(190)の紐2頭を±1ptで再現**すること。

■ ★★★探索を守る（**合計34比較・Bonferroni α=0.01/34・z=3.750**）
　⚠**(178)でプラセボが109.9%、(172)は跳ねたマスを追って崩れた**。
　★**採用条件に「隣接と同じ向き」を入れる**——**孤立した1マスは採らない**。
　★**全マスは記述として出すが、最良を選んで結論にしない**（**(185)で同じ誤りをして訂正した**）。

■ ★採用条件（判定基準39/40/41）
　1. **家族Aで100%超のマスがある、または 家族Bで有意な代替がある**
　2. ★**隣接（Aは隣のズレ床、Bは隣の券種）と同じ向き**
　3. **裾の検算で符号が反転しない**（上位3本・前後半・年別）
　4. ★**平均オッズの単調性で説明されない**（**(175)(188)で2回踏んだ形**）

■ 予想（⚠**類推なので当てにしない**・判定基準24）
　★**家族A**: **単調が続けば100%に届く可能性がある**が、**標本が減ってCIが広がるほうが速い**と見る。
　★**家族B**: **人気順の紐が現行と大差ない**と見る（**(66)「順位は時点に頑健」・(69)「対人気順+1〜4pt」**）。
　⚠**どちらでも驚かない**。

実行: python3 ml/audit_ana_ext.py    自己テスト: python3 ml/audit_ana_ext.py --selftest
"""
import math
import sys
from itertools import combinations, permutations

import numpy as np

sys.path.insert(0, "ml")
import features as F
from audit_crosspool import LINE, load_races, payoff, zq
from audit_ana_odds import COST, MIN_HORSES, gate1, roi_of
from audit_ana_marg import WF_BOX4, WF_TOL, wf_predict
from audit_ana_board import BOARD_R, BOARD_TOL, NPLACE, load_fuku_boards, qpool
from train_prod import add_odds_features

A_EXT = [0.15, 0.20]
B_EXT = [0.08, 0.10, 0.12, 0.15, 0.20]
FILT_B = [(0.15, 0.02), (0.20, 0.06)]
KINDS = ["馬連", "馬単", "三連複", "三連単"]
ORDERS = ["現行(p降順)", "★人気(オッズ昇順)", "★ズレ(gap降順)", "★板(複勝確率降順)"]
MINCELL = 300
MAXHW_A = 20.0
NCMP = len(A_EXT) * len(B_EXT) + len(FILT_B) * len(KINDS) * (len(ORDERS) - 1)   # 10 + 24
ALPHA = 0.01
SEED = 20260906
KNOWN_A = {(0.15, 0.08): 93.1, (0.20, 0.08): 93.6}
KNOWN_B = {((0.15, 0.02), "馬連"): 89.5, ((0.15, 0.02), "馬単"): 90.0,
           ((0.15, 0.02), "三連複"): 91.2, ((0.15, 0.02), "三連単"): 103.1,
           ((0.20, 0.06), "馬連"): 89.0, ((0.20, 0.06), "馬単"): 88.8,
           ((0.20, 0.06), "三連複"): 88.0, ((0.20, 0.06), "三連単"): 96.6}


def bet2(kind, ax, himo):
    """★紐2頭ぶんの買い目（(190)の「紐2頭」と同じ）。"""
    if kind == "馬連":
        return [(ax, himo[0]), (ax, himo[1])]
    if kind == "馬単":
        return [(ax, himo[0]), (ax, himo[1])]
    if kind == "三連複":
        return [tuple(sorted((ax, himo[0], himo[1])))]
    if kind == "三連単":
        return [(ax, a, b) for a, b in permutations(himo[:2], 2)]
    raise ValueError(kind)


def selftest():
    ok = True
    ax, h = 9, [3, 7, 1]
    assert bet2("馬連", ax, h) == [(9, 3), (9, 7)]
    assert len(bet2("三連複", ax, h)) == 1 and len(bet2("三連単", ax, h)) == 2
    print("★紐2頭の自己テスト: 馬連2点/馬単2点/三連複1点/三連単2点　★OK")
    print(f"　→ **(190)の「紐2頭」と同じ点数**（馬連2・馬単2・三連複1・三連単2）")
    rng = np.random.default_rng(0)
    n = 150_000
    pay = rng.choice([0.0, 250.0, 900.0], size=n, p=[0.72, 0.20, 0.08])
    m = float(np.mean([(pay[rng.permutation(n)] - pay[rng.permutation(n)]).mean()
                       for _ in range(200)]))
    print(f"★ゲート2B: 対応差200回の平均 {m:+.3f}円 → **仮説が偽なら0**: "
          f"{'★OK' if abs(m) < 2 else '⚠NG'}")
    ok &= abs(m) < 2
    print(f"★ゲート2A: **板が正しければROIは払戻率80.0%を返す**（100%ではない）")
    print(f"★家族A {len(A_EXT)*len(B_EXT)}比較（ズレ床 {B_EXT}）")
    print(f"★家族B {len(FILT_B)*len(KINDS)*(len(ORDERS)-1)}比較（紐の順序 {len(ORDERS)}通り）")
    print(f"★合計 {NCMP}比較 → z = {zq(ALPHA/NCMP):.3f}")
    print("★自己テスト: " + ("全部OK" if ok else "⚠NG"))
    return 0 if ok else 1


def main():
    z = zq(ALPHA / NCMP)
    print("(191) ★★★**未測定を2つ潰す** — (A) ズレ床の延長 / (B) 紐の順序")
    print("★A: **(189)は ズレ床 0.08 で止めたが、4行とも単調に上がり続けていた**")
    print("★B: **紐の順序はこの線で一度も変えていない**"
          "（**軸はgap最大、紐はモデル確率降順で基準が違う**）\n")

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
    A = {(a, b): {"v": [], "od": [], "yr": [], "dt": []} for a in A_EXT for b in B_EXT}
    B = {(fl, k, o): [] for fl in FILT_B for k in KINDS for o in ORDERS}
    Rs, box4, nall = [], [], 0
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
        order_p = [int(u) for u in ub[np.argsort(-pv, kind="mergesort")]]
        bx = [payoff(r, "三連複", list(c)) for c in combinations(sorted(order_p[:4]), 3)]
        if not any(x is None for x in bx):
            box4.append(sum(bx) - 400.0)
        yr = int(gg["date"].iloc[0].year)
        # ── 家族A ──
        for a in A_EXT:
            for b in B_EXT:
                cand = np.where((pn >= a) & (gapf >= b))[0]
                if not len(cand):
                    continue
                axi = int(cand[int(np.argmax(gapf[cand]))])
                v = payoff(r, "複勝", [int(ub[axi])])
                if v is None:
                    continue
                c = A[(a, b)]
                c["v"].append(v); c["od"].append(float(od[axi]))
                c["yr"].append(yr); c["dt"].append(gg["date"].iloc[0])
        # ── 家族B ──
        ords = {"現行(p降順)": order_p,
                "★人気(オッズ昇順)": [int(u) for u in ub[np.argsort(od, kind="mergesort")]],
                "★ズレ(gap降順)": [int(u) for u in ub[np.argsort(-gapf, kind="mergesort")]],
                "★板(複勝確率降順)": [int(u) for u in ub[np.argsort(-qp, kind="mergesort")]]}
        for fl in FILT_B:
            cand = np.where((pn >= fl[0]) & (gapf >= fl[1]))[0]
            if not len(cand):
                continue
            axu = int(ub[int(cand[int(np.argmax(gapf[cand]))])])
            for kind in KINDS:
                vals = {}
                okall = True
                for onm, oseq in ords.items():
                    himo = [u for u in oseq if u != axu]
                    if len(himo) < 2:
                        okall = False
                        break
                    cs = bet2(kind, axu, himo)
                    vs = [payoff(r, kind, list(c)) for c in cs]
                    if any(x is None for x in vs):
                        okall = False
                        break
                    vals[onm] = sum(vs) / len(cs)
                if not okall:
                    continue
                for onm, vv in vals.items():
                    B[(fl, kind, onm)].append(vv)
    print(f"\n★対象 **{nall:,}レース**")

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

    print(f"\n■ ★★内部対照")
    for (a, b), kv in KNOWN_A.items():
        v = np.asarray(A[(a, b)]["v"], float)
        dd = roi_of(v) - kv
        print(f"　家族A ({a:.2f},{b:.2f}): {roi_of(v):.1f}% vs (189) {kv:.1f}%"
              f"　差 {dd:+.1f}pt　{'★再現' if abs(dd) <= 1.0 else '⚠ズレた'}")
    for (fl, kind), kv in KNOWN_B.items():
        v = np.asarray(B[(fl, kind, "現行(p降順)")], float)
        if len(v) < MINCELL:
            continue
        dd = roi_of(v) - kv
        print(f"　家族B {fl} {kind}: {roi_of(v):.1f}% vs (190) {kv:.1f}%"
              f"　差 {dd:+.1f}pt　{'★再現' if abs(dd) <= 1.0 else '⚠ズレた'}")

    print(f"\n{'='*100}")
    print(f"■ ★★★家族A: **ズレ床の延長**（複勝1点・**{NCMP}比較・z={z:.3f}**）")
    print(f"　★ゲート2A: **板が正しければROIは払戻率80.0%を返す**")
    print(f"\n{'推奨度≥':>8}{'ズレ≥':>7}{'買うR':>9}{'割合':>8}{'ROI':>9}{'99%CI':>22}"
          f"{'的中率':>9}{'平均オッズ':>11}{'判定':>16}")
    passA = set()
    for a in A_EXT:
        for b in B_EXT:
            c = A[(a, b)]
            v = np.asarray(c["v"], float)
            if len(v) < MINCELL:
                print(f"{a:>8.2f}{b:>7.2f}{len(v):>9,}　⚠標本不足")
                continue
            mu, se = v.mean() - COST, v.std(ddof=1) / math.sqrt(len(v))
            hw = z * se
            lo_, hi_ = 100 + mu - hw, 100 + mu + hw
            if hw > MAXHW_A:
                tag = "⚠判定不能(CI広)"
            elif lo_ > 100.0:
                passA.add((a, b)); tag = "★★100%超"
            else:
                tag = "⚠通らない"
            print(f"{a:>8.2f}{b:>7.2f}{len(v):>9,}{100*len(v)/nall:>7.1f}%"
                  f"{roi_of(v):>8.1f}%{f'[{lo_:.1f},{hi_:.1f}]':>22}"
                  f"{100*np.mean(v>0):>8.1f}%{np.mean(c['od']):>10.1f}倍{tag:>16}")

    print(f"\n{'='*100}")
    print(f"■ ★★★家族B: **紐の順序**（紐2頭・**現行との対応差**）")
    print(f"　★ゲート2B: **仮説が偽（順序が損益について何も持たない）なら対応差の期待値は0**")
    passB = []
    for fl in FILT_B:
        print(f"\n── 絞り {fl} ──")
        print(f"{'券種':<8}" + "".join(f"{o:>22}" for o in ORDERS))
        for kind in KINDS:
            base = np.asarray(B[(fl, kind, "現行(p降順)")], float)
            if len(base) < MINCELL:
                continue
            line = f"{roi_of(base):>21.1f}%"
            for onm in ORDERS[1:]:
                v = np.asarray(B[(fl, kind, onm)], float)
                dd = v - base
                mu, se = dd.mean(), dd.std(ddof=1) / math.sqrt(len(dd))
                sig = abs(mu) > z * se
                if sig and mu > 0:
                    passB.append((fl, kind, onm, roi_of(v)))
                line += f"{roi_of(v):>13.1f}%({mu:>+5.1f}{'★' if sig else ' '})"
            print(f"{kind:<8}{line}")
        print(f"{'':<8}（カッコ内は**現行との対応差[円]**・★はBonferroniを通った）")

    print(f"\n■ ★採用条件")
    adjA = any((a, b) in passA and any((a, b2) in passA for b2 in B_EXT
                                       if abs(B_EXT.index(b2) - B_EXT.index(b)) == 1)
               for a, b in passA)
    print(f"　1. 家族Aで100%超 … **{len(passA)}マス**"
          f"（隣接あり: {'★' if adjA else '⚠無し'}）")
    print(f"　2. 家族Bで有意な代替 … **{len(passB)}マス**")
    if not passA and not passB:
        print("\n★★★**結論: ズレ床を延長しても、紐の順序を変えても届かない**。")
        print("⚠**経路は弱いので「閉じた」とは書けない**（判定基準25）。")
        return
    print(f"\n■ ★裾の検算（**通ったもの**・(77)）")
    for a, b in passA:
        c = A[(a, b)]
        v = np.asarray(c["v"], float)
        yr = np.asarray(c["yr"], int)
        dt = np.array(c["dt"], dtype="datetime64[D]").astype(int)
        ys = sorted(set(yr))
        ov = sum(1 for u in ys if roi_of(v[yr == u]) > 100.0)
        print(f"　A({a:.2f},{b:.2f}): ROI {roi_of(v):.1f}% / "
              f"**上位3本が全払戻の {100*np.sort(v)[-3:].sum()/max(v.sum(),1e-9):.1f}%** / "
              f"前半 {roi_of(v[dt<=np.median(dt)]):.1f}%・"
              f"後半 {roi_of(v[dt>np.median(dt)]):.1f}% / "
              f"**100%超の年 {ov}/{len(ys)}**")
    for fl, kind, onm, roi in passB:
        print(f"　B {fl} {kind} {onm}: ROI {roi:.1f}%")
    print("\n⚠**枠連の運用には触れない**。**設定変更は提案しない**。")


if __name__ == "__main__":
    sys.exit(selftest() if "--selftest" in sys.argv else (main() or 0))
