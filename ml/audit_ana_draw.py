"""(245) ★★★★**ズレは馬番の関数を含んでいるか** —— qpは馬番を織り込み、pnは知らない

★★**動機（2026-09-16・利用者の指定「トラックバイアスは穴馬選びに役に立たないか」）**
　★**枠連側(177)が「内外バイアスは logit(p) を差し引いても残る」と測った**
　（**当日累積 +0.0163 / プラセボ +0.0052 → 超過 +0.0111・年別符号10/10**）。
　★**こちらのモデルも馬番を持っていない**（**`ml/model_ana_fwd` の44特徴量に枠番・馬番なし・実値確認済**）。

■ ★★★**だが「絞りに使えるか」より先に問うべきことがある**
　★**軸は `ズレ = pn − qp`**。★**qp は複勝の板＝市場**で、★**市場は馬番を見ている**。
　→ ⚠★★**もし qp が馬番を織り込み、pn が織り込んでいないなら、
　　　★ズレの一部は「妙味」ではなく「モデルの欠落」である**。
　★**(236)で踏んだのと同じ構造**（**2016年の大きなズレはモデルが外していたから**）。
　⚠**これを確かめずに「バイアスで絞る」を測っても、何を絞っているのか分からない**。

■ ★★**この測定が使える理由（★交絡が無い）**
　★**馬番は抽選で決まる＝馬の実力とは独立**。
　→ ★**qp が馬番に依存するなら、それは★市場が枠順に付けた値段そのもの**。
　→ ★**pn は馬番を特徴量に持たないので、依存は ≈0 のはず**（★これが内部対照になる）。

────────────────────────────────────────────────────────────
★★★ 事前登録（2026-09-16・**結果を見る前にコミットする**）
────────────────────────────────────────────────────────────

■ ★**母集団**: **(236)(240)(241)と同じ**（**複勝の板があり出走頭数が足りるレース・全馬**）。

■ ★**測る量**: ★**align = 馬番/頭数 − 0.5**（**内=負 / 外=正**・枠連側(177)と同じ定義）
| # | 量 | ★**意味** |
|---|---|---|
| ★**①** | ★**corr(align, pn)** | ★**≈0 のはず**（**pnは馬番を持たない**）→ ★**内部対照** |
| ★**②** | ★**corr(align, qp)** | ★**市場が枠順に付けた値段** |
| ★★**③** | ★★**corr(align, gap)** | ★★**主判定。gap = pn − qp なので ③ ≈ ① − ②** |
| **④** | **軸に選ばれた馬の align の平均** | ★**軸が枠順に偏っているか**（**0なら偏っていない**） |
　★**相関はレース内で計算して全レースで平均**（**レースをまたぐ比較をしない**）。
　★**CIは★レース単位のブートストラップ**（**同一レースの馬は独立でない**・200回・seed固定）。

■ ★★★★**主判定（★先に決める・これ1つ）**
　★★**③ corr(align, gap) の99%CIがゼロを含むか**。
| ★**返り値** | ★**読み** |
|---|---|
| ★**ゼロを含む** | ★**ズレは馬番の関数を含んでいない**。→ **トラックバイアスは「ズレの正体」ではない** |
| ⚠**ゼロを外す** | ⚠**ズレの一部が馬番で説明できる**＝★**妙味ではなくモデルの欠落が混じっている** |

■ ★★ゲート2（判定基準42）——**何が返れば「この測定は何も言っていない」か**
　★**① corr(align, pn) がゼロを外したら、★pnが馬番を知らないという前提が壊れている**
　　→ ⚠**そのときは③を読まない**（**前提が崩れているので分解できない**）。
　★**②がゼロを含むなら、市場も馬番に値段を付けていない**＝**③もゼロのはず**。★**整合を確認する**。

■ ★★★内部対照（**決定的・最初に見る**）
　★**年別の対象レース数が (236)(240)(241) と一致**:
　　**2842 / 2807 / 2773 / 2767 / 2786 / 2813 / 2791 / 2780 / 2723 / 2758 / 1630**
　⚠**ずれたら読まない**（判定基準32）。

■ ⚠★**先に書いておく限界**
　★**これは「バイアスで絞ればROIが動くか」ではない**。★**ズレの正体を見るだけ**。
　⚠**絞りを測るなら、別に事前登録が要る**——★**統計量はROI（100円あたり）、
　　★**同じ本数を無作為に捨てるプラセボを必ず置く**（**枠連側(178)の②が壊れた理由**）。
　★**align は「内外」だけ。★前残り（ペース）は測らない**（**枠連側(177)で死亡・符号6/10**）。
　★**日ごとの馬場差は見ない**（**(177)のA/Bに相当するものは作らない**）。★**ここでは通年の平均だけ**。
　⚠**枠連側(177)(178)の★結論は持ち込まない**（**払戻率も買い目も違う**・判定基準25）。

■ 予想（⚠**当てにしない**・★私は12回外した）
　★**② はゼロを外し、③ もゼロを外すと見る**（**市場は枠順に値段を付けているはずなので**）。
　⚠**つまり「ズレの一部はモデルの欠落」に倒れると見る**。★**外れたら嬉しい**。

実行: python3 ml/audit_ana_draw.py    自己テスト: python3 ml/audit_ana_draw.py --selftest
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
from train_prod import add_odds_features

KNOWN_R = [2842, 2807, 2773, 2767, 2786, 2813, 2791, 2780, 2723, 2758, 1630]
NBOOT, SEED = 200, 20260916


def align_of(ub):
    """★align = 馬番/頭数 − 0.5（★内=負 / 外=正・枠連側(177)と同じ）"""
    return np.asarray(ub, float) / len(ub) - 0.5


def rcorr(a, b):
    """★1レース内の相関。★分散がゼロなら nan"""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def boot_ci(v, n=NBOOT, q=(0.5, 99.5)):
    """★レース単位のブートストラップで99%CI（★1レース1値なのでそのまま再標本）"""
    v = np.asarray([x for x in v if np.isfinite(x)], float)
    if len(v) < 10:
        return float("nan"), float("nan")
    g = np.random.default_rng(SEED)
    m = [float(np.mean(v[g.integers(0, len(v), len(v))])) for _ in range(n)]
    return float(np.percentile(m, q[0])), float(np.percentile(m, q[1]))


def selftest():
    ok = True
    print("★★★★主判定: **③ corr(align, gap) の99%CIがゼロを含むか**")
    print("　★含む → **ズレは馬番の関数を含んでいない** ／ ⚠外す → **モデルの欠落が混じっている**")
    print("★★ゲート2: **① corr(align, pn) がゼロを外したら前提が壊れている → ③を読まない**")
    print(f"★★★内部対照（最初に見る）: 年別レース数 {KNOWN_R}")
    a = align_of([1, 2, 3, 4, 5, 6, 7, 8])
    print(f"★alignの検算（8頭）: {np.round(a, 3)}")
    ok &= abs(a.mean() - 0.0625) < 1e-9 and a[0] < 0 < a[-1]
    print(f"　★**内が負・外が正** {'★OK' if a[0] < 0 < a[-1] else '⚠NG'}")
    print(f"★相関の検算: 完全一致 {rcorr([1,2,3],[1,2,3]):.3f} / 逆 {rcorr([1,2,3],[3,2,1]):.3f}"
          f" / 定数 {rcorr([1,2,3],[5,5,5])}")
    ok &= abs(rcorr([1,2,3],[1,2,3]) - 1.0) < 1e-9 and np.isnan(rcorr([1,2,3],[5,5,5]))
    lo, hi = boot_ci([0.10] * 100)
    print(f"★CIの検算: 全部0.10 → [{lo:.4f}, {hi:.4f}]（幅ゼロのはず）"
          f"　{'★OK' if abs(hi - lo) < 1e-9 else '⚠NG'}")
    ok &= abs(hi - lo) < 1e-9
    print("⚠**これは「絞ればROIが動くか」ではない。★ズレの正体を見るだけ**")
    print("⚠**枠連側(177)(178)の結論は持ち込まない**（判定基準25）")
    print("★自己テスト: " + ("全部OK" if ok else "⚠NG"))
    return 0 if ok else 1


def main():
    print("(245) ★★★★**ズレは馬番の関数を含んでいるか**\n")
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
        od, pv = gg["odds"].to_numpy(float), gg["p"].to_numpy(float)
        if not np.isfinite(od).all() or (od <= 0).any() or pv.sum() <= 0:
            continue
        pn = pv / pv.sum() * NPLACE
        qp, _ = qpool([bd[int(u)] for u in ub], "harm")
        gap = pn - qp
        al = align_of(ub)
        yr = int(gg["date"].iloc[0].year)
        a = A.setdefault(yr, {"R": 0, "cp": [], "cq": [], "cg": [], "ax": []})
        a["R"] += 1
        a["cp"].append(rcorr(al, pn))
        a["cq"].append(rcorr(al, qp))
        a["cg"].append(rcorr(al, gap))
        c = np.where((pn >= PN_FLOOR) & (gap >= GAP) & (od >= LFIX))[0]
        if len(c):
            i = int(c[int(np.argmax(pn[c]))])
            a["ax"].append(float(al[i]))

    ys = sorted(k for k in A if k >= 2016)
    got = [A[u]["R"] for u in ys]
    print("★★★内部対照（**最初に見る**）")
    print(f"　実測 {got}")
    print(f"　既知 {KNOWN_R}　{'★一致' if got == KNOWN_R else '⚠ずれた'}")
    if got != KNOWN_R:
        print("⚠⚠**対照が落ちた。読まない**（判定基準32）。")
        return

    def allv(k):
        return [x for u in ys for x in A[u][k]]

    print("\n" + " " * 28 + "".join(f"{u:>8}" for u in ys) + f"{'全期間':>10}")
    for k, lab in (("cp", "★① corr(align, pn)"), ("cq", "★② corr(align, qp)"),
                   ("cg", "★★③ corr(align, gap)")):
        row = "".join(f"{np.nanmean(A[u][k]):>+8.4f}" for u in ys)
        print(f"{lab:<26}{row}{np.nanmean(allv(k)):>+10.4f}")
    print(f"{'④ 軸のalign平均':<25}"
          + "".join(f"{(np.mean(A[u]['ax']) if A[u]['ax'] else float('nan')):>+8.4f}" for u in ys)
          + f"{np.mean(allv('ax')):>+10.4f}")
    print(f"{'　軸の本数':<27}" + "".join(f"{len(A[u]['ax']):>8}" for u in ys)
          + f"{len(allv('ax')):>10}")

    print(f"\n■ ★★★★**主判定: ③ corr(align, gap) の99%CI**")
    for k, lab in (("cp", "① pn（★内部対照・≈0のはず）"), ("cq", "② qp（市場）"),
                   ("cg", "★★③ gap（★主判定）"), ("ax", "④ 軸のalign平均")):
        v = allv(k)
        lo, hi = boot_ci(v)
        z = "★ゼロを含む" if lo <= 0 <= hi else "⚠**ゼロを外す**"
        print(f"　{lab:<26} {np.nanmean(v):>+8.4f}  99%CI [{lo:>+7.4f}, {hi:>+7.4f}]  {z}")

    cp, cq, cg = np.nanmean(allv("cp")), np.nanmean(allv("cq")), np.nanmean(allv("cg"))
    print(f"\n　★**整合の確認: ③ ≈ ① − ②** → {cg:+.4f} ≈ {cp:+.4f} − ({cq:+.4f}) = {cp-cq:+.4f}"
          f"　{'★合う' if abs(cg - (cp - cq)) < 0.01 else '⚠ずれる'}")
    lo, hi = boot_ci(allv("cp"))
    if not (lo <= 0 <= hi):
        print("⚠⚠**①がゼロを外した＝pnが馬番を知らないという前提が壊れている。★③を読まない**")
    print(f"\n⚠**これは「絞ればROIが動くか」ではない**。★**絞りを測るなら別に事前登録**"
          f"（**統計量はROI・同じ本数を無作為に捨てるプラセボ必須**）。")
    print("⚠**枠連側(177)(178)の結論は持ち込んでいない**（判定基準25）。")
    print("⚠**枠連の運用には触れない**。**設定変更は提案しない**。")


if __name__ == "__main__":
    sys.exit(selftest() if "--selftest" in sys.argv else (main() or 0))
