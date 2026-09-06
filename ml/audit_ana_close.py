"""(183) ★★★**残り2枝を潰す** — ◇12-20倍のマス と 「穴が来るレースを探す」

★★「一つずつ潰そう」（ユーザー・2026-09-06）の最後の2つ。

**枝A: ◇12-20倍のマス**（**(181)で post-hoc に目を引いた**: 水準84.9%・差+9.2円）
　⚠**同じ標本で測り直しても新証拠にならない**（判定基準35）。→ ★**年別で割って決着させる**。
　★**(180)の有意帯（120倍〜）は「100%超の年 0/10」だった**。**12-20倍はまだ見ていない**。

**枝B: 「穴が来るレースを探す」**
　⚠**(130)プール間不整合 −0.0043 / (134)荒れ度 −0.0154 / (156)(157)4件とも陰性**＝**4回閉鎖**。
　⚠**さらに(178)で「(112)裾2%は397レースで検出力ゼロ」と実測で確定**。
　★★**それでも「新しい変数を探せばいいのでは」が残る**。→ ★**探すのではなく、
　　「そもそもレース単位の構造があるか」を測って潰す**。
　★**構造が無ければ、どんな選別規則も存在しえない**——**探索を1本ずつ否定するより強い**。

────────────────────────────────────────────────────────────
★★★ 事前登録（2026-09-06・**結果を見る前にコミットする**）
────────────────────────────────────────────────────────────

■ ★経路（判定基準25）: 穴の選定に MLモデルの p ／ 帯は単勝オッズ。⚠**弱い経路**。
■ ★手続き: **ウォークフォワード**（判定基準6・`data/cache/wf_pred.npz` を再利用）。
■ ★設計は(180)(181)と同一: **同じレース・同じオッズ帯の 穴 vs 乱・複勝1点**。

■ **枝A の判定**
　★**12-20倍の穴の複勝ROIを年別に出す**。
　★**判定条件（先に書く）**: **100%を超える年が半数（5/10以上）無ければ閉じる**。
　⚠**理由**: **運用に載せるなら「その年に張っていたら勝てた」が過半で要る**。
　　**(180)の有意帯は 0/10 だった**。**同じ形なら同じ扱いにする**（判定基準29）。
　★**あわせて 全9帯の「100%超の年」も出す**（**どの程度にも無いことを示すため**）。

■ ★★**枝B の判定 — 「穴の優位にレース単位の構造があるか」**
　★**統計量**: **各レースの「穴の複勝払戻 − 乱の複勝払戻」を、レース属性で群に分け、
　　群間の最大差（円）**を取る。**6次元・各2〜3群**:
　　1. **頭数**（≤12 / 13-15 / ≥16）
　　2. **距離**（≤1400 / 1401-1900 / ≥1901）
　　3. **芝ダ**（2群）
　　4. **クラス**（raceclass の下位/上位）
　　5. ★**市場エントロピー（荒れ度）**（三分位・**(134)の変数**）
　　6. ★**(112)の軸の複勝E**（三分位・**プロジェクト唯一の効く選別変数**）
　★★**帰無分布**: **レースの群ラベルを無作為に並べ替え**（**1,000回**）→ **maxT で6次元をまとめて制御**。
　　⚠**判定基準28の「不偏な部分標本のプラセボ」に当たる＝有効な形**（**構造上0が強制される形ではない**）。
　★★**ゲート2（判定基準42）**: **仮説が偽（穴の優位がレースによらず一定）なら、
　　群は無作為分割と同じになり、群間の最大差の期待値は帰無分布と一致する＝超過は0を返す**。
　★**判定**: **6次元のいずれも maxT の99%点を超えなければ、
　　「穴の優位にレース単位の構造は無い」＝どんな選別規則も作れない**と書いて閉じる。
　⚠**「無い」を証明するのではない**——**書けるのは「この6次元には無い」まで**（判定基準25）。
　　★**ただし6次元には(134)の荒れ度と(112)の軸Eという、既知で最有力の2本が入っている**。

■ ⚠ゲート1（判定基準32）: (88)③④を別パーサで再現（±3pt）。
　★**陽性対照**: **ウォークフォワードの三連複BOX上位4が 81.9%±1.5pt**（**(181)(182)の実測**）。

■ 予想
　**枝A: 0〜2/10年と見る**（**(180)の有意帯が0/10だったので**）。⚠**類推なので当てにしない**。
　**枝B: 6次元とも超えないと見る**（**(130)(134)(156)(157)の4件と整合するから**）。
　⚠**判定基準24: 類推は当たらない**。**どちらでも驚かない**。

実行: python3 ml/audit_ana_close.py    自己テスト: python3 ml/audit_ana_close.py --selftest
"""
import math
import sys

import numpy as np

sys.path.insert(0, "ml")
import features as F
from audit_crosspool import load_races, payoff, zq
from audit_ana_odds import COST, MIN_HORSES, gate1, roi_of
from audit_ana_degree import DEG
from audit_ana_marg import WF_BOX4, WF_TOL, wf_predict
import soft_axis as SA

TARGET = "12-20倍"
YEAR_RULE = 0.5                  # 枝A: 100%超の年が半数無ければ閉じる
NPERM = 1000
SEED = 20260906
DIMS = ["頭数", "距離", "芝ダ", "クラス", "★荒れ度(市場エントロピー)", "★(112)軸の複勝E"]


def group_of(dim, v):
    """レース属性 → 群の添字。"""
    if dim == "頭数":
        return 0 if v <= 12 else (1 if v <= 15 else 2)
    if dim == "距離":
        return 0 if v <= 1400 else (1 if v <= 1900 else 2)
    return int(v)


def maxgap(diff, grp, ng):
    """群ごとの平均の最大差[円]。"""
    s = np.bincount(grp, weights=diff, minlength=ng)
    n = np.bincount(grp, minlength=ng)
    ok = n >= 200
    if ok.sum() < 2:
        return 0.0
    m = s[ok] / n[ok]
    return float(m.max() - m.min())


def selftest():
    ok = True
    assert group_of("頭数", 10) == 0 and group_of("頭数", 14) == 1 and group_of("頭数", 16) == 2
    assert group_of("距離", 1200) == 0 and group_of("距離", 1800) == 1 and group_of("距離", 2400) == 2
    print("★群分けの自己テスト: 頭数3群・距離3群　★OK")
    rng = np.random.default_rng(0)
    n = 60_000
    diff = rng.normal(0, 300, n)
    grp = rng.integers(0, 3, n)
    obs = maxgap(diff, grp, 3)
    nulls = [maxgap(diff, rng.permutation(grp), 3) for _ in range(300)]
    q = float(np.quantile(nulls, 0.99))
    print(f"★ゲート2の自己テスト: 構造の無いデータで 実測の最大差 {obs:.2f}円 / "
          f"帰無99%点 {q:.2f}円 → **超えない**: {'★OK' if obs <= q else '⚠NG'}")
    ok &= obs <= q
    print(f"★枝Aの判定: 100%超の年が {YEAR_RULE:.0%} 未満なら閉じる")
    print(f"★枝Bの次元: {len(DIMS)}本（maxTでまとめて制御）")
    print("★自己テスト: " + ("全部OK" if ok else "⚠NG"))
    return 0 if ok else 1


def main():
    print("(183) ★★★**残り2枝を潰す** — ◇12-20倍のマス と 「穴が来るレースを探す」")
    print("★経路: 穴の選定に MLモデルの p / 帯は単勝オッズ。⚠**弱い経路**")
    print("★手続き: ウォークフォワード（`data/cache/wf_pred.npz` を再利用）\n")

    races = {r["rid"]: r for r in load_races()}
    rows, bad = gate1(list(races.values()))
    print("⚠**ゲート1**: (88)③④を別パーサで再現・許容±3pt")
    for nm, n, roi, known, dd, ok in rows:
        print(f"　{nm:<12}{roi:>7.1f}% vs {known:>5.1f}%　差 {dd:+.1f}pt"
              f"　{'★立った' if ok else '⚠落ちた'}")
    if bad:
        print("\n⚠⚠**ゲート1が落ちた。読まない**。")
        return

    d = F.to_model(F.load_files())
    f = F.build_features(d)
    keep = (f["n_prior"] >= 1) & d["odds"].notna() & (d["odds"] > 0)
    d, f = d[keep].reset_index(drop=True), f[keep].reset_index(drop=True)
    y = (d["finish"] <= 3).astype(int).to_numpy()
    fx, _ = F.encode_categoricals(f)
    from train_prod import add_odds_features
    fx = add_odds_features(fx, d["odds"].to_numpy(float), d["raceid"].to_numpy())
    pred = wf_predict(d, fx, y, 3)
    m = ~np.isnan(pred)
    cols = ["raceid", "umaban", "odds", "date", "fieldsize", "distance",
            "surface", "raceclass"]
    sub = d.loc[m, cols].copy()
    sub["p"] = pred[m]

    rng = np.random.default_rng(SEED)
    from itertools import combinations
    acc = {i: {"a": [], "yr": []} for i in range(len(DEG))}
    tgt = [i for i, (nm, _, _) in enumerate(DEG) if nm == TARGET][0]
    B = {"diff": [], "頭数": [], "距離": [], "芝ダ": [], "クラス": [],
         "ent": [], "eax": []}
    box4 = []
    for rid, g in sub.groupby("raceid"):
        r = races.get(str(rid))
        if r is None:
            continue
        nums = {u for u, _, _ in r["horses"]}
        if len(nums) < MIN_HORSES:
            continue
        gg = g[g["umaban"].astype(int).isin(nums)]
        if len(gg) < MIN_HORSES:
            continue
        od = gg["odds"].to_numpy(float)
        pv = gg["p"].to_numpy(float)
        ub = gg["umaban"].astype(int).to_numpy()
        if not np.isfinite(od).all() or (od <= 0).any() or pv.sum() <= 0:
            continue
        mk = (1.0 / od) / (1.0 / od).sum()
        gap = pv / pv.sum() - mk
        order = [int(u) for u in ub[np.argsort(-pv, kind="mergesort")]]
        bx = [payoff(r, "三連複", list(c)) for c in combinations(sorted(order[:4]), 3)]
        if not any(x is None for x in bx):
            box4.append(sum(bx) - 400.0)
        yr = int(gg["date"].iloc[0].year)
        for i, (nm, lo, hi) in enumerate(DEG):
            idx = np.where((od >= lo) & (od < hi))[0]
            if len(idx) < 2:
                continue
            a = int(idx[int(np.argmax(gap[idx]))])
            b = int(rng.choice([int(k) for k in idx if k != a]))
            pa = payoff(r, "複勝", [int(ub[a])])
            pb = payoff(r, "複勝", [int(ub[b])])
            if pa is None or pb is None:
                continue
            acc[i]["a"].append(pa); acc[i]["yr"].append(yr)
            if i == tgt:
                B["diff"].append(pa - pb)
                B["頭数"].append(group_of("頭数", int(gg["fieldsize"].iloc[0])))
                B["距離"].append(group_of("距離", int(gg["distance"].iloc[0])))
                B["芝ダ"].append(int(gg["surface"].iloc[0]))
                B["クラス"].append(1 if int(gg["raceclass"].iloc[0]) >= 3 else 0)
                ent = float(-(mk * np.log(np.maximum(mk, 1e-12))).sum())
                B["ent"].append(ent)
                _k, e_ax, _q = SA.axis_expect([float(o) for o in od])
                B["eax"].append(float(e_ax) if e_ax is not None else np.nan)

    box4 = np.asarray(box4, float)
    g0 = 100.0 * (box4.sum() + 400.0 * len(box4)) / (400.0 * len(box4))
    okb = abs(g0 - WF_BOX4) <= WF_TOL
    print(f"\n⚠**陽性対照**: 三連複BOX上位4 **{g0:.1f}%** vs (181)(182)の {WF_BOX4}%"
          f" → **{'★立った' if okb else '⚠⚠落ちた'}**")
    if not okb:
        print("\n⚠⚠**陽性対照が落ちた。読まない**（判定基準32）。")
        return

    # ── 枝A ──
    print(f"\n{'='*96}\n■ ★★枝A: ◇**{TARGET}のマス**を年別で決着させる")
    print(f"　★判定: **100%を超える年が半数（{YEAR_RULE:.0%}）無ければ閉じる**")
    A = np.asarray(acc[tgt]["a"], float)
    yr = np.asarray(acc[tgt]["yr"], int)
    yy = sorted(set(yr))
    print(f"\n{'年':<7}" + "".join(f"{u:>8}" for u in yy))
    print(f"{'ROI%':<7}" + "".join(f"{roi_of(A[yr==u]):>8.1f}" for u in yy))
    over = sum(1 for u in yy if roi_of(A[yr == u]) > 100.0)
    print(f"\n　**{TARGET} 全体 {roi_of(A):.1f}%**・**100%超の年 {over}/{len(yy)}**"
          f" → **{'⚠⚠残る' if over/len(yy) >= YEAR_RULE else '★閉じる'}**")
    print(f"\n　★参考: **全9帯の「100%超の年」**")
    print(f"{'帯':<10}{'ROI':>9}{'100%超の年':>12}")
    for i, (nm, lo, hi) in enumerate(DEG):
        v = np.asarray(acc[i]["a"], float)
        yv = np.asarray(acc[i]["yr"], int)
        if len(v) < 300:
            continue
        ys = sorted(set(yv))
        o = sum(1 for u in ys if roi_of(v[yv == u]) > 100.0)
        print(f"{nm:<10}{roi_of(v):>8.1f}%{f'{o}/{len(ys)}':>12}")

    # ── 枝B ──
    diff = np.asarray(B["diff"], float)
    ent = np.asarray(B["ent"], float)
    eax = np.asarray(B["eax"], float)
    ok = np.isfinite(eax)
    q3e = np.quantile(ent, [1/3, 2/3])
    q3a = np.quantile(eax[ok], [1/3, 2/3])
    grps = {"頭数": np.asarray(B["頭数"], int), "距離": np.asarray(B["距離"], int),
            "芝ダ": np.asarray(B["芝ダ"], int), "クラス": np.asarray(B["クラス"], int),
            "★荒れ度(市場エントロピー)": np.searchsorted(q3e, ent, side="right"),
            "★(112)軸の複勝E": np.searchsorted(q3a, np.where(ok, eax, q3a[0]), side="right")}
    print(f"\n{'='*96}\n■ ★★★枝B: **穴の優位に「レース単位の構造」があるか**"
          f"（{TARGET}・{len(diff):,}レース）")
    print("　★ゲート2: **仮説が偽（優位がレースによらず一定）なら、群は無作為分割と同じ**")
    print(f"　★**帰無はレースの群ラベルを無作為に並べ替え{NPERM:,}回・maxTで6次元をまとめて制御**")
    obs = {k: maxgap(diff, v, int(v.max()) + 1) for k, v in grps.items()}
    mx = np.zeros(NPERM)
    for t in range(NPERM):
        mx[t] = max(maxgap(diff, rng.permutation(v), int(v.max()) + 1)
                    for v in grps.values())
    thr = float(np.quantile(mx, 0.99))
    print(f"\n　★**maxT の帰無99%点 = {thr:.2f}円**")
    print(f"\n{'次元':<26}{'群数':>5}{'群間の最大差':>14}{'判定':>18}")
    npass = 0
    for k in DIMS:
        v = grps[k]
        ng = int(v.max()) + 1
        sig = obs[k] > thr
        npass += sig
        print(f"{k:<26}{ng:>5}{obs[k]:>13.2f}円"
              f"{'★★構造がある' if sig else '⚠構造は検出できない':>18}")
    print(f"\n■ ★判定")
    if npass == 0:
        print("　★★**6次元のいずれも maxT を超えない**＝")
        print("　★★★**穴の優位にレース単位の構造は無い＝どんな選別規則も作れない**。")
        print("　⚠**「無い」を証明したのではない**——**書けるのは「この6次元には無い」まで**。")
        print("　★**ただし6次元には(134)の荒れ度と(112)の軸Eという、既知で最有力の2本が入っている**。")
        print("　→ ★**(130)(134)(156)(157)の4件と整合する。5件目として閉じる**。")
    else:
        print(f"　⚠**{npass}次元が maxT を超えた**。**事前登録どおり、そこを次に測る**。")
    print("\n⚠**枠連の運用には触れない**。**設定変更は提案しない**。")


if __name__ == "__main__":
    sys.exit(selftest() if "--selftest" in sys.argv else (main() or 0))
