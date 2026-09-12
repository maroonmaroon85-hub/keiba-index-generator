"""★(193)追試: 「上位k本を抜く」に**基準線を与える**（★判定基準42の修正）

⚠**前回の誤り**: 「上位5本を抜くと89.0%」と書いたが、**この操作はどんな標本でも必ず下げる**。
　**5という数にも根拠がない**。★**仮説が偽でも下がる統計量＝何も測っていない**。

★★**修正**: **まったく同じ操作を3つの腕に当てて比べる**。
　① ズレ順の紐（主張している買い方） ② 現行 p降順の紐（対照・軸は同一）
　③ 乱 = 軸と同じオッズ帯の無作為な2頭（対照・払戻分布の基準線）
★**もし3腕とも同じだけ下がるなら、「裾の1本勝負」は馬連という券種の性質であって、
　この買い方の欠陥ではない**——**私の前回の論法は崩れる**。
★**さらに、抜く本数に依存しない道具としてブートストラップCIも出す**（**恣意的な打ち切り不要**）。

────────────────────────────────────────────────────────────
★★★★ 実測（2026-09-06）—— **前回の論法は崩れた。訂正する**
────────────────────────────────────────────────────────────

■ ★★『上位k本抜き』の下げ幅（**同じ操作を3腕に当てた**）
| 絞り | 床 | 腕 | ROI | 上位5本抜き | ★下げ幅 |
|---|---|---|---|---|---|
| (0.20,0.06) | 0.06 | ★ズレ順 | 106.8% | 89.0% | 17.8pt |
| | | **乱(同帯)** | 86.9% | 76.4% | ⚠**10.5pt** |
| (0.15,0.02) | 0.06 | ★ズレ順 | 104.8% | 87.1% | 17.7pt |
| | | **乱(同帯)** | ⚠**109.6%** | 90.2% | ⚠**19.4pt** |
| (0.20,0.06) | 0.04 | ★ズレ順 | 103.7% | 95.8% | 7.9pt |
| | | **乱(同帯)** | 82.5% | 77.5% | 5.0pt |
| (0.15,0.02) | 0.04 | ★ズレ順 | 102.2% | 94.0% | 8.1pt |
| | | **乱(同帯)** | 96.5% | 87.3% | 9.3pt |
★★**オッズを揃えた乱でも下げ幅は 5.0〜19.4pt**——**ズレ順と同程度かそれ以上**。
→ ★★★**「上位5本を抜くと落ちる＝1本勝負だから駄目」は誤り。取り下げる**。
　**これはそのオッズ帯の馬連という券種の性質であって、この買い方の欠陥ではない**。
⚠**現行(p降順)が 2.3〜6.2pt しか下がらないのは人気馬を買っていて配当が小さく頻繁だから**。
　★**性質の違う腕と比べて論じたのが誤りの中身**（判定基準37: 対照の作り方をまず疑う）。

■ ★★ブートストラップ（10,000回・**打ち切りを使わない道具**）
| 絞り | 床 | ★ROI 99%区間 | ★**100%超の割合** | **ズレ順−現行が0超** |
|---|---|---|---|---|
| (0.20,0.06) | **0.04** | [88.2, 120.1] | 71.6% | ★**99.4%** |
| (0.15,0.02) | **0.04** | [86.6, 118.6] | 62.1% | 97.9% |
| (0.20,0.06) | 0.06 | [80.8, 137.4] | ⚠**72.5%** | 94.4% |
| (0.15,0.02) | 0.06 | [78.8, 134.8] | 65.7% | 89.8% |
★**「106.8%が本当に100%超である確率」は 72.5%**。⚠**4回に1回以上は100%未満**。

■ ★★★決定的: **薄いマスではプラセボに負ける**
　★**(0.15,0.02)・床0.06 で 乱(同帯) が 109.6% > ズレ順 104.8%**。
　→ ★★**3,991Rのマスでは、無作為に紐を選んだほうが良い数字を出す**＝**区別がつかない**。
　⚠**(178)でプラセボが109.9%を出したのと同じ形**。
　★**一方、厚いマス（床0.04・12,000R）では乱が 82.5 / 96.5% でズレ順が明確に上**。

■ ★★結論（**前回から根拠が入れ替わった。結論は変わらない**）
　★**取り下げ**: 「上位5本抜きで89%＝裾の1本勝負」——**恣意的で、対照も同じだけ下がる**。
　★**新しい根拠**: ⚠**薄いマスは無作為対照(109.6%)に負ける** ／ ⚠**100%超の確率72.5%**
　　／ ⚠**厚いマスは前半117→後半90** ／ ⚠**有意化に342年**（(193)家族B）。
　★**厚いマス（床0.04）は対照に対して健全**（乱 82.5% / 差が0超 99.4%）。
　　⚠**だが時間で崩れており水準としては採れない**。
　★★**(193)家族C（紐を複勝で+6.4円・Bonferroni通過）は今回の訂正の影響を受けない**。

実行: python3 ml/audit_ana_tail.py
"""
import sys
import numpy as np
sys.path.insert(0, "ml")
import features as F
from audit_crosspool import load_races, payoff
from audit_ana_odds import BANDS, COST, MIN_HORSES, band_of, roi_of
from audit_ana_marg import wf_predict
from audit_ana_board import NPLACE, load_fuku_boards, qpool
from train_prod import add_odds_features

FILT = [(0.15, 0.02), (0.20, 0.06)]
CS = [0.04, 0.06]
KS = [1, 3, 5, 10]
SEED = 20260906
NBOOT = 10000

races = {r["rid"]: r for r in load_races()}
boards = load_fuku_boards()
d = F.to_model(F.load_files()); f = F.build_features(d)
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
K = {(fl, c): {"z": [], "g": [], "r": []} for fl in FILT for c in CS}
for rid, g in sub.groupby("raceid"):
    r = races.get(str(rid)); bd = boards.get(str(rid))
    if r is None or bd is None: continue
    nums = {u for u, _, _ in r["horses"]}
    if len(nums) < MIN_HORSES: continue
    gg = g[g["umaban"].astype(int).isin(nums)]
    ub = gg["umaban"].astype(int).to_numpy()
    if len(gg) < MIN_HORSES or not all(int(u) in bd for u in ub): continue
    od = gg["odds"].to_numpy(float); pv = gg["p"].to_numpy(float)
    if not np.isfinite(od).all() or (od <= 0).any() or pv.sum() <= 0: continue
    pn = pv / pv.sum() * NPLACE
    qp, _ = qpool([bd[int(u)] for u in ub], "harm")
    gapf = pn - qp
    order_p = [int(u) for u in ub[np.argsort(-pv, kind="mergesort")]]
    order_g = list(np.argsort(-gapf, kind="mergesort"))
    for fl in FILT:
        cand = np.where((pn >= fl[0]) & (gapf >= fl[1]))[0]
        if not len(cand): continue
        ai = int(cand[int(np.argmax(gapf[cand]))]); axu = int(ub[ai])
        himo_p = [u for u in order_p if u != axu]
        for cth in CS:
            hg = [int(ub[k]) for k in order_g if int(ub[k]) != axu and gapf[k] >= cth]
            if len(hg) < 2 or len(himo_p) < 2: continue
            # ③乱: ズレ順で選んだ紐と**同じオッズ帯**の無作為な2頭
            bi = np.array([band_of(float(o), BANDS) for o in od])
            pos = {int(u): k for k, u in enumerate(ub)}
            hr = []
            okr = True
            for u in hg[:2]:
                b = bi[pos[u]]
                pool = [int(ub[k]) for k in range(len(ub))
                        if bi[k] == b and int(ub[k]) != axu and int(ub[k]) not in hr]
                if not pool: okr = False; break
                hr.append(int(rng.choice(pool)))
            if not okr: continue
            vz = [payoff(r, "馬連", [axu, u]) for u in hg[:2]]
            vg = [payoff(r, "馬連", [axu, u]) for u in himo_p[:2]]
            vr = [payoff(r, "馬連", [axu, u]) for u in hr]
            if any(x is None for x in vz + vg + vr): continue
            c = K[(fl, cth)]
            c["z"].append(sum(vz) / 2.0); c["g"].append(sum(vg) / 2.0)
            c["r"].append(sum(vr) / 2.0)

def drops(v):
    sv = np.sort(v)
    return [roi_of(sv[:-k]) for k in KS]

print("\n★★★『上位k本を抜く』に基準線を与える（★判定基準42の修正）")
print("★同じ操作を3腕に当てる: **ズレ順 / 現行(p降順) / 乱(同じ帯の無作為2頭)**\n")
for fl in FILT:
    for cth in CS:
        c = K[(fl, cth)]
        vz = np.asarray(c["z"], float); vg = np.asarray(c["g"], float)
        vr = np.asarray(c["r"], float)
        if len(vz) < 300: continue
        print(f"■ 絞り{fl} 紐の床{cth:.2f}　**{len(vz):,}R**")
        print(f"{'腕':<14}{'ROI':>8}" + "".join(f"{'上位'+str(k)+'本抜き':>13}" for k in KS)
              + f"{'★下げ幅(5本)':>14}")
        for nm, v in (("★ズレ順", vz), ("現行(p降順)", vg), ("乱(同帯)", vr)):
            dz = drops(v)
            print(f"{nm:<14}{roi_of(v):>7.1f}%" + "".join(f"{x:>12.1f}%" for x in dz)
                  + f"{roi_of(v)-dz[2]:>12.1f}pt")
        # ★ブートストラップ（打ち切り不要の道具）
        n = len(vz)
        idx = rng.integers(0, n, size=(NBOOT, n))
        bz = 100.0 * vz[idx].mean(axis=1) / COST
        bg = 100.0 * vg[idx].mean(axis=1) / COST
        bd_ = bz - bg
        q = lambda a, p: float(np.percentile(a, p))
        print(f"　★ブートストラップ({NBOOT:,}回) ズレ順ROI 99%区間 "
              f"[{q(bz,0.5):.1f}, {q(bz,99.5):.1f}]　"
              f"**100%超の割合 {100*np.mean(bz>100):.1f}%**")
        print(f"　★ズレ順 − 現行 の差 99%区間 [{q(bd_,0.5):+.1f}, {q(bd_,99.5):+.1f}]pt　"
              f"**0超の割合 {100*np.mean(bd_>0):.1f}%**")
        print()
