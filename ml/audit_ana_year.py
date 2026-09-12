"""★(194)追記(記述のみ): **年別の内訳**と**時間の劣化が本物か**

★★**判定基準37**: **「後半90%」を読む前に、対照も同じように落ちているかを見る**。
　**対照も落ちているなら、市場かデータの変化であって、この買い方の劣化ではない**。
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
SEED = 20260906
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
K = {(fl, c): {"z": [], "g": [], "r": [], "yr": [], "ax": []} for fl in FILT for c in CS}
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
    bi = np.array([band_of(float(o), BANDS) for o in od])
    yr = int(gg["date"].iloc[0].year)
    for fl in FILT:
        cand = np.where((pn >= fl[0]) & (gapf >= fl[1]))[0]
        if not len(cand): continue
        ai = int(cand[int(np.argmax(gapf[cand]))]); axu = int(ub[ai])
        himo_p = [u for u in order_p if u != axu]
        for cth in CS:
            hg = [int(ub[k]) for k in order_g if int(ub[k]) != axu and gapf[k] >= cth]
            if len(hg) < 2 or len(himo_p) < 2: continue
            hr, ok = [], True
            for u in hg[:2]:
                b = bi[list(ub).index(u)]
                pl = [int(ub[k]) for k in range(len(ub))
                      if bi[k] == b and int(ub[k]) != axu and int(ub[k]) not in hr]
                if not pl: ok = False; break
                hr.append(int(rng.choice(pl)))
            if not ok: continue
            vz = [payoff(r, "馬連", [axu, u]) for u in hg[:2]]
            vg = [payoff(r, "馬連", [axu, u]) for u in himo_p[:2]]
            vr = [payoff(r, "馬連", [axu, u]) for u in hr]
            if any(x is None for x in vz + vg + vr): continue
            c = K[(fl, cth)]
            c["z"].append(sum(vz)/2.0); c["g"].append(sum(vg)/2.0); c["r"].append(sum(vr)/2.0)
            c["yr"].append(yr); c["ax"].append(float(od[ai]))

print("\n★★★年別の内訳（**記述のみ**）——★**対照も一緒に落ちているか**")
for fl in FILT:
    for cth in CS:
        c = K[(fl, cth)]
        z = np.asarray(c["z"], float); gv = np.asarray(c["g"], float)
        rv = np.asarray(c["r"], float); yr = np.asarray(c["yr"], int)
        ax = np.asarray(c["ax"], float)
        if len(z) < 300: continue
        ys = sorted(set(yr))
        print(f"\n■ 絞り{fl} 紐の床{cth:.2f}　**{len(z):,}R**　"
              f"（ズレ順 {roi_of(z):.1f}% / 現行 {roi_of(gv):.1f}% / 乱 {roi_of(rv):.1f}%）")
        print(f"{'年':<7}{'R数':>7}{'★ズレ順':>10}{'現行':>9}{'乱':>9}"
              f"{'★ズレ順−乱':>12}{'的中率':>8}{'軸オッズ':>9}")
        for u in ys:
            m = yr == u
            print(f"{u:<7}{int(m.sum()):>7,}{roi_of(z[m]):>9.1f}%{roi_of(gv[m]):>8.1f}%"
                  f"{roi_of(rv[m]):>8.1f}%{(z[m]-rv[m]).mean():>+11.1f}円"
                  f"{100*np.mean(z[m]>0):>7.2f}%{ax[m].mean():>8.1f}倍")
        # 前半後半
        h = yr <= 2020
        print(f"{'前半(〜20)':<7}{int(h.sum()):>7,}{roi_of(z[h]):>9.1f}%{roi_of(gv[h]):>8.1f}%"
              f"{roi_of(rv[h]):>8.1f}%{(z[h]-rv[h]).mean():>+11.1f}円"
              f"{100*np.mean(z[h]>0):>7.2f}%{ax[h].mean():>8.1f}倍")
        print(f"{'後半(21〜)':<7}{int((~h).sum()):>7,}{roi_of(z[~h]):>9.1f}%"
              f"{roi_of(gv[~h]):>8.1f}%{roi_of(rv[~h]):>8.1f}%"
              f"{(z[~h]-rv[~h]).mean():>+11.1f}円"
              f"{100*np.mean(z[~h]>0):>7.2f}%{ax[~h].mean():>8.1f}倍")
