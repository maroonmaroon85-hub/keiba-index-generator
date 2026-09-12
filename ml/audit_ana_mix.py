"""(241) ★★★★**④の動きは「帯の中身が重くなったから」か** —— 構成を固定して測り直す

★★**動機（2026-09-10・(240)で自分が残した未測定）**:
　★**(240)で ④（市場の校正ずれ）が −0.006 → −0.008 と出た**（**主判定は「近づかない」**）。
　⚠**だが同時に ⑨ 帯の中央オッズが 46.7 → 50.6（+8%）で、帯の中身が重くなっていた**。
　★★**重い馬ほど過大評価が強いなら、★中身が動いただけで④は広がる**。★**これを切り分ける**。

■ ★★**やること: 構成を2016年に固定して④を測り直す**
　★**帯（単勝10倍以上）を★固定のオッズ小帯に割る**。**小帯ごとに④を出す**。
　★**各年の④を「2016年の小帯構成」で重みづけし直す**（★**構成調整後の④**）。
　→ ★**構成が原因なら、調整後の④は年で動かなくなる**。

────────────────────────────────────────────────────────────
★★★ 事前登録（2026-09-10・**結果を見る前にコミットする**）
────────────────────────────────────────────────────────────

■ ★**小帯の境目（★先に決める・以後変えない）**: **[10,20) / [20,40) / [40,80) / [80,∞)**
　⚠**この4分割は恣意的**。★**だが結果を見る前に決めており、以後変えない**。
　★**中央オッズが46.7〜50.6なので、境目40と80が中央をまたぐように置いた**。

■ ★★★★**主判定（★先に決める・これ1つ）**
　★★**2016年の構成で重みづけした④（調整後）が、2016→2026 でどう動くか**。
| ★**返り値** | ★**読み** |
|---|---|
| ★**調整後も −0.006→−0.008 のまま** | ★**構成のせいではない**。**(240)の④はそのまま読める** |
| ★**調整後は横ばい（差が0.001未満）になる** | ★★**(240)の④の動きは構成のせいだった** |

■ ★★ゲート2（判定基準42）——**この測定は何を返せば「構成は関係ない」か**
　★**構成が関係ないなら、★各年で「生の④」と「調整後の④」がほぼ一致する（差 < 0.001）**。
　★**両方を年別に併記する**。★**小帯ごとの④と構成比も全部出す**。

■ ★★★**ばらつきも同時に出す（★(240)で出していなかった）**
　★**④のse をレース単位のブートストラップで出す**（**同一レースの馬は独立でない**）。
　★**200回・seed固定**。→ ★**−0.006 と −0.008 の差が se の何倍かを見る**。
　⚠**(240)で「広がった」と書いたが、se を出していなかった**。★**ここで補う**。

■ ★★★内部対照（**決定的・最初に見る**）
　★**① 年別レース数が (236)(235)(240) と一致**:
　　**2842 / 2807 / 2773 / 2767 / 2786 / 2813 / 2791 / 2780 / 2723 / 2758 / 1630**
　★**② 帯の頭数 2016=10.377 / 2026=9.696（±0.02）**
　★**③ 生の④が (240) と一致: 2016 −0.006 / 2026 −0.008（±0.001）**
　⚠**3つとも立たなければ読まない**（判定基準32）。

■ ⚠★**先に書いておく限界**
　★**小帯の中でも構成は動きうる**（**[80,∞)の中で200倍が増える等**）。★**この測定は4分割の粒度でしか切れない**。
　★**qp の定義（複勝板の調和平均・Σ=3）依存**。**2026は1,630レース＝途中まで**。
　⚠**これは探索ではない。マスも軸も買い目も動かさない。運用は変えない**。

■ 予想（⚠**当てにしない**・★**私は12回外した**）
　★**調整後もほとんど変わらないと見る**（**構成のせいではない**）。
　★★**そしてそもそも④には trend が無く、−0.006 と −0.008 の差は se 1〜2本分だと見る**。

────────────────────────────────────────────────────────────
★★★★ 結果（2026-09-10・**事前登録をコミットしたあとに走らせた**）
────────────────────────────────────────────────────────────
★★★**内部対照は3つとも立った**（年別レース数・帯の頭数・生の④が(240)と一致）。
★**2016年の構成**: [10,20) 22.5% / [20,40) 22.8% / [40,80) 19.5% / [80,∞) 35.2%

| | 2016 | 2018 | 2020 | 2022 | 2024 | 2026 |
|---|---|---|---|---|---|---|
| 構成比 [10,20) | 22.5% | 22.2% | 24.2% | 22.2% | 21.7% | 22.4% |
| 構成比 [20,40) | 22.8% | 21.4% | 23.3% | 21.8% | 20.9% | 21.0% |
| 構成比 [40,80) | 19.5% | 19.2% | 20.1% | 20.0% | 19.3% | 18.6% |
| 構成比 **[80,∞)** | **35.2%** | 37.2% | 32.4% | 36.1% | 38.0% | **38.0%** |
| ★④ [10,20) | −.0027 | −.0169 | −.0092 | −.0013 | −.0045 | **+.0062** |
| ★④ [20,40) | −.0067 | −.0039 | +.0000 | −.0122 | −.0120 | −.0123 |
| ★④ [40,80) | −.0059 | −.0064 | −.0017 | −.0115 | −.0123 | −.0181 |
| ★④ **[80,∞)** | −.0073 | −.0083 | −.0071 | −.0078 | −.0088 | −.0100 |
| ★★**生の④** | **−.0058** | −.0089 | −.0049 | −.0080 | −.0092 | **−.0084** |
| ★★★**調整後の④** | **−.0058** | −.0089 | −.0049 | −.0080 | −.0093 | **−.0085** |
| **差（調整後−生）** | +.0000 | +.0000 | −.0000 | −.0000 | −.0000 | **−.0001** |
| ★**④のse** | .0013 | .0014 | .0014 | .0014 | .0015 | .0017 |

■ ★★★★**主判定の返り値: 構成のせいではない**
　★**生の④ −0.0058 → −0.0084（−0.0025）／ 調整後 −0.0058 → −0.0085（−0.0026）**
　★★**構成で説明できる分 = +0.0001（生の動きの −4%）＝★実質ゼロ**。
　★**11年すべてで「調整後 − 生」が ±0.0001 以内**（**ゲート2の「差<0.001なら構成は関係ない」を満たす**）。
　→ ★★**(240)の④は構成の影響を受けていない。そのまま読める**。

■ ★★★★**そして、そもそも④は動いていない（★(240)の書き方の訂正）**
　★**2016と2026の差 −0.0025 / 差のse 0.0022 = ★1.16倍** → ★**seの2倍未満**。
　⚠★★**(240)で「絶対値 0.006 → 0.008・むしろ広がった」と書いたのは言い過ぎだった**
　　（**seを出していなかった**）。★**正しくは「動いていない」**（判定基準43）。
　★**(240)の主判定「ゼロに近づいたか → 近づいていない」は変わらない**。
　　★**変わるのは「広がった」の部分だけ**。→ ★★**11年ずっと −0.005〜−0.009 の横ばい**。

■ ★★**小帯ごとに見ると（★これは事前登録に無い記述・結論にしない）**
| 小帯 | ★**11年の振れ** | ★**読み** |
|---|---|---|
| **[10,20)** | ⚠**−.0169 〜 +.0062（符号が変わる）** | ★**軽い穴は年でばらつく。2026は逆に過小評価** |
| **[20,40)** | −.0133 〜 +.0000 | — |
| **[40,80)** | −.0181 〜 −.0017 | — |
| ★**[80,∞)** | ★**−.0115 〜 −.0071（11年すべて負・幅が最も狭い）** | ★★**重い穴の過大評価だけが一貫している** |
　⚠**これは4つの小帯を後から見比べた記述**。★**主判定ではない**。

■ ⚠**先に書いた限界のとおり**
　★**小帯の中でも構成は動きうる**（**[80,∞)の中で200倍が増える等は切れていない**）。
　★**qp の定義依存。2026は1,630レース＝途中まで**。

■ ★**予想は2つとも当たった**（「調整後もほとんど変わらない」「④に trend は無く se 1〜2本分」）。
　⚠**外し回数は12のまま**。

実行: python3 ml/audit_ana_mix.py    自己テスト: python3 ml/audit_ana_mix.py --selftest
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
from train_prod import add_odds_features

KNOWN_R = [2842, 2807, 2773, 2767, 2786, 2813, 2791, 2780, 2723, 2758, 1630]
KNOWN_N = {2016: 10.377, 2026: 9.696}
KNOWN_B = {2016: -0.006, 2026: -0.008}      # ★(240)の生の④
NTOL, BTOL = 0.02, 0.001
EDGES = [10.0, 20.0, 40.0, 80.0, float("inf")]   # ★小帯（先に決めた・変えない）
NBOOT, SEED = 200, 20260910
REF = 2016                                        # ★構成を固定する基準年


def sub_of(o):
    """★そのオッズがどの小帯か"""
    for i in range(len(EDGES) - 1):
        if EDGES[i] <= o < EDGES[i + 1]:
            return i
    return -1


def boot_se(per_race, n):
    """★レース単位のブートストラップで④のse（★同一レースの馬は独立でない）"""
    if not per_race:
        return float("nan")
    s = np.array([x[0] for x in per_race], float)
    c = np.array([x[1] for x in per_race], float)
    g = np.random.default_rng(SEED)
    out = []
    for _ in range(n):
        i = g.integers(0, len(s), len(s))
        tc = c[i].sum()
        out.append(s[i].sum() / tc if tc > 0 else np.nan)
    return float(np.nanstd(out, ddof=1))


def selftest():
    ok = True
    print(f"★小帯（先に決めた）: " + " / ".join(
        f"[{EDGES[i]:.0f},{'∞' if EDGES[i+1] == float('inf') else f'{EDGES[i+1]:.0f}'})"
        for i in range(len(EDGES) - 1)))
    ok &= sub_of(10.0) == 0 and sub_of(19.9) == 0 and sub_of(20.0) == 1
    ok &= sub_of(79.9) == 2 and sub_of(80.0) == 3 and sub_of(9.9) == -1
    print(f"　★割り当ての検算: 10.0→{sub_of(10.0)} / 19.9→{sub_of(19.9)} / 20.0→{sub_of(20.0)}"
          f" / 79.9→{sub_of(79.9)} / 80.0→{sub_of(80.0)} / 9.9→{sub_of(9.9)}（帯の外）"
          f"　{'★OK' if ok else '⚠NG'}")
    print(f"★★★★主判定: **{REF}年の構成で重みづけした④が 2016→2026 でどう動くか**")
    print("　★調整後も動く → **構成のせいではない** ／ ★横ばいになる → **構成のせいだった**")
    print("★★ゲート2: **生の④と調整後の④を年別に併記。差<0.001なら構成は関係ない**")
    print(f"★★★内部対照（**最初に見る**）: 年別レース数・帯の頭数・"
          f"**生の④が(240)と一致（2016 {KNOWN_B[2016]} / 2026 {KNOWN_B[2026]}・±{BTOL}）**")
    # ★ブートストラップの検算: 1頭1レース・ずれ0.05固定なら se は 0 に近い
    pr = [(0.05, 1.0)] * 500
    se0 = boot_se(pr, 50)
    print(f"★seの検算: 全レース同じずれ0.05 → se {se0:.6f}（0に近いはず）"
          f"　{'★OK' if se0 < 1e-9 else '⚠NG'}")
    ok &= se0 < 1e-9
    print("⚠**探索ではない。マスも軸も買い目も動かさない。運用は変えない**")
    print("★自己テスト: " + ("全部OK" if ok else "⚠NG"))
    return 0 if ok else 1


def main():
    print("(241) ★★★★**④の動きは「帯の中身が重くなったから」か**\n")
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

    NS = len(EDGES) - 1
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
        qp, _ = qpool([bd[int(u)] for u in ub], "harm")
        hit = (gg["finish"].astype(int).to_numpy() <= 3).astype(int)
        yr = int(gg["date"].iloc[0].year)
        a = A.setdefault(yr, {"R": 0, "n": 0, "pr": [],
                              "sb": [[0.0, 0, []] for _ in range(NS)]})
        a["R"] += 1
        sel = np.where(od >= LFIX)[0]
        a["n"] += len(sel)
        dv = hit[sel] - qp[sel]
        a["pr"].append((float(dv.sum()), float(len(sel))))
        for k, i in enumerate(sel):
            b = sub_of(float(od[i]))
            if b < 0:
                continue
            a["sb"][b][0] += float(dv[k])
            a["sb"][b][1] += 1
            a["sb"][b][2].append((float(dv[k]), 1.0))

    ys = sorted(k for k in A if k >= 2016)
    got = [A[u]["R"] for u in ys]
    nn = {u: A[u]["n"] / A[u]["R"] for u in ys}
    raw = {u: sum(x[0] for x in A[u]["pr"]) / sum(x[1] for x in A[u]["pr"]) for u in ys}
    ok1 = got == KNOWN_R
    ok2 = all(abs(nn[u] - v) <= NTOL for u, v in KNOWN_N.items() if u in nn)
    ok3 = all(abs(raw[u] - v) <= BTOL for u, v in KNOWN_B.items() if u in raw)
    print("★★★内部対照（**最初に見る**）")
    print(f"　① 年別レース数 {'★一致' if ok1 else '⚠ずれた'}")
    print(f"　② 帯の頭数 2016={nn.get(2016, float('nan')):.3f} / "
          f"2026={nn.get(2026, float('nan')):.3f}　{'★一致' if ok2 else '⚠ずれた'}")
    print(f"　③ 生の④ 2016={raw.get(2016, float('nan')):+.4f}（{KNOWN_B[2016]}） / "
          f"2026={raw.get(2026, float('nan')):+.4f}（{KNOWN_B[2026]}）"
          f"　{'★一致' if ok3 else '⚠ずれた'}")
    if not (ok1 and ok2 and ok3):
        print("⚠⚠**対照が落ちた。読まない**（判定基準32）。")
        return

    lab = [f"[{EDGES[i]:.0f},{'∞' if EDGES[i+1] == float('inf') else f'{EDGES[i+1]:.0f}'})"
           for i in range(NS)]
    wref = np.array([A[REF]["sb"][b][1] for b in range(NS)], float)
    wref = wref / wref.sum()
    print(f"\n★**{REF}年の構成（これで固定する）**: "
          + " / ".join(f"{lab[b]} {100*wref[b]:.1f}%" for b in range(NS)))

    print("\n" + " " * 26 + "".join(f"{u:>8}" for u in ys))
    for b in range(NS):
        print(f"{'構成比 ' + lab[b]:<26}"
              + "".join(f"{100*A[u]['sb'][b][1]/max(A[u]['n'],1):>7.1f}%" for u in ys))
    for b in range(NS):
        print(f"{'★④ ' + lab[b]:<26}"
              + "".join(f"{(A[u]['sb'][b][0]/A[u]['sb'][b][1] if A[u]['sb'][b][1] else float('nan')):>+8.4f}"
                        for u in ys))
    print(f"{'★★生の④':<24}" + "".join(f"{raw[u]:>+8.4f}" for u in ys))
    adj = {}
    for u in ys:
        v = np.array([(A[u]["sb"][b][0] / A[u]["sb"][b][1]) if A[u]["sb"][b][1] else np.nan
                      for b in range(NS)], float)
        adj[u] = float(np.nansum(v * wref) / np.nansum(wref * ~np.isnan(v)))
    print(f"{'★★★調整後の④':<22}" + "".join(f"{adj[u]:>+8.4f}" for u in ys))
    print(f"{'　差（調整後−生）':<22}" + "".join(f"{adj[u]-raw[u]:>+8.4f}" for u in ys))
    se = {u: boot_se(A[u]["pr"], NBOOT) for u in ys}
    print(f"{'★④のse（レース単位）':<21}" + "".join(f"{se[u]:>8.4f}" for u in ys))

    y0, y1 = ys[0], ys[-1]
    dr, da = raw[y1] - raw[y0], adj[y1] - adj[y0]
    sd = float(np.hypot(se[y0], se[y1]))
    print(f"\n■ ★★★★**主判定: {REF}年の構成で固定した④の動き**")
    print(f"　★**生の④　　 {raw[y0]:+.4f} → {raw[y1]:+.4f}（{dr:+.4f}）**")
    print(f"　★**調整後の④ {adj[y0]:+.4f} → {adj[y1]:+.4f}（{da:+.4f}）**")
    print(f"　→ ★**構成で説明できる分 = {dr - da:+.4f}**"
          f"（**生の動きの {100*(dr-da)/dr if dr else float('nan'):.0f}%**）")
    print(f"\n■ ★★**そもそも動いているのか（se との比較）**")
    print(f"　★**2016と2026の差 {dr:+.4f} / 差のse {sd:.4f} = ★{abs(dr)/sd if sd else float('nan'):.2f}倍**")
    print(f"　→ ★**{'seの2倍未満＝ばらつきと区別がつかない' if abs(dr) < 2*sd else '⚠seの2倍以上'}**")
    print(f"\n⚠**小帯の中でも構成は動きうる（4分割の粒度でしか切れない）**。"
          f"**2026は1,630レース＝途中まで**。")
    print("⚠**枠連の運用には触れない**。**設定変更は提案しない**。")


if __name__ == "__main__":
    sys.exit(selftest() if "--selftest" in sys.argv else (main() or 0))
