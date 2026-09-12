"""★★対照(乱)の再現性を測る —— **種を変えると乱のROIはどれだけ動くか**（判定基準37）

⚠**(195)で内部対照が落ちた**: **主腕は完全再現(103.7/115.0/88.0%)なのに乱だけ 82.5 → 99.0%**。
★**乱の引き直しでこれだけ動くなら、(194)のプラセボの数字は全部あやしい**。**20種で測る**。
"""
import sys
import numpy as np
sys.path.insert(0, "ml")
import features as F
from audit_crosspool import load_races, payoff
from audit_ana_odds import BANDS, MIN_HORSES, band_of, roi_of
from audit_ana_marg import wf_predict
from audit_ana_board import NPLACE, load_fuku_boards, qpool
from train_prod import add_odds_features

CASES = [((0.20, 0.06), 0.04), ((0.20, 0.06), 0.06), ((0.15, 0.02), 0.06)]
NSEED = 20
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

rngs = [np.random.default_rng(1000 + s) for s in range(NSEED)]
K = {c: {"z": [], "yr": [], "r": [[] for _ in range(NSEED)]} for c in CASES}
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
    gap = pn - qp
    order_g = list(np.argsort(-gap, kind="mergesort"))
    bi = np.array([band_of(float(o), BANDS) for o in od])
    yr = int(gg["date"].iloc[0].year)
    for (fl, cth) in CASES:
        cand = np.where((pn >= fl[0]) & (gap >= fl[1]))[0]
        if not len(cand): continue
        ai = int(cand[int(np.argmax(gap[cand]))]); axu = int(ub[ai])
        hg = [int(ub[k]) for k in order_g if int(ub[k]) != axu and gap[k] >= cth]
        if len(hg) < 2: continue
        vz = [payoff(r, "馬連", [axu, w]) for w in hg[:2]]
        if any(v is None for v in vz): continue
        outs, ok = [], True
        for s in range(NSEED):
            hr = []
            for uu in hg[:2]:
                b = bi[list(ub).index(uu)]
                pl = [int(ub[k]) for k in range(len(ub))
                      if bi[k] == b and int(ub[k]) != axu and int(ub[k]) not in hr]
                if not pl: ok = False; break
                hr.append(int(rngs[s].choice(pl)))
            if not ok: break
            vr = [payoff(r, "馬連", [axu, w]) for w in hr]
            if any(v is None for v in vr): ok = False; break
            outs.append(sum(vr) / 2.0)
        if not ok: continue
        K[(fl, cth)]["z"].append(sum(vz) / 2.0)
        K[(fl, cth)]["yr"].append(yr)
        for s in range(NSEED):
            K[(fl, cth)]["r"][s].append(outs[s])

print(f"\n★★★対照(乱)の再現性 —— **{NSEED}種で引き直す**（判定基準37）\n")
for c in CASES:
    z = np.asarray(K[c]["z"], float); yr = np.asarray(K[c]["yr"], int)
    if len(z) < 300: continue
    rs = [np.asarray(v, float) for v in K[c]["r"]]
    rois = np.array([roi_of(v) for v in rs])
    d_all = np.array([(z - v).mean() for v in rs])
    h = yr < 2021
    d_h1 = np.array([(z[h] - v[h]).mean() for v in rs])
    d_h2 = np.array([(z[~h] - v[~h]).mean() for v in rs])
    print(f"■ 絞り{c[0]} 床{c[1]:.2f}　**{len(z):,}R**　ズレ順 {roi_of(z):.1f}%"
          f"（前半 {roi_of(z[h]):.1f}% / 後半 {roi_of(z[~h]):.1f}%）")
    print(f"　★**乱のROI**: 平均 {rois.mean():.1f}%　**幅 {rois.min():.1f} 〜 {rois.max():.1f}%**"
          f"　標準偏差 **{rois.std(ddof=1):.1f}pt**")
    print(f"　★**ズレ順−乱（全期間）**: 平均 {d_all.mean():+.1f}円"
          f"　幅 {d_all.min():+.1f} 〜 {d_all.max():+.1f}円　sd {d_all.std(ddof=1):.1f}円")
    print(f"　★**前半**: 平均 {d_h1.mean():+.1f}円　幅 {d_h1.min():+.1f} 〜 {d_h1.max():+.1f}円")
    print(f"　★**後半**: 平均 {d_h2.mean():+.1f}円　幅 {d_h2.min():+.1f} 〜 {d_h2.max():+.1f}円"
          f"　★**0を上回る種 {int((d_h2>0).sum())}/{NSEED}**")
    print(f"　★**乱が100%を超えた種 {int((rois>100).sum())}/{NSEED}**\n")
