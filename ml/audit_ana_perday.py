"""★1日あたりの対象レース数（記述）"""
import sys
import numpy as np
sys.path.insert(0, "ml")
import features as F
from audit_crosspool import load_races, payoff
from audit_ana_odds import MIN_HORSES, roi_of
from audit_ana_marg import wf_predict
from audit_ana_board import NPLACE, load_fuku_boards, qpool
from train_prod import add_odds_features

FILT = [(0.15, 0.02), (0.20, 0.06)]
CS = [0.00, 0.02, 0.04, 0.06]
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

K = {(fl, c): [] for fl in FILT for c in CS}
alldays, nall = set(), 0
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
    nall += 1
    dt = str(gg["date"].iloc[0].date())
    alldays.add(dt)
    pn = pv / pv.sum() * NPLACE
    qp, _ = qpool([bd[int(u)] for u in ub], "harm")
    gapf = pn - qp
    order_g = list(np.argsort(-gapf, kind="mergesort"))
    for fl in FILT:
        cand = np.where((pn >= fl[0]) & (gapf >= fl[1]))[0]
        if not len(cand): continue
        ai = int(cand[int(np.argmax(gapf[cand]))]); axu = int(ub[ai])
        for cth in CS:
            hg = [int(ub[k]) for k in order_g if int(ub[k]) != axu and gapf[k] >= cth]
            if len(hg) < 2: continue
            K[(fl, cth)].append(dt)

nd = len(alldays)
print(f"\n★データ全体: **{nall:,}レース / {nd:,}開催日**　"
      f"= **1日あたり {nall/nd:.1f}レース**（⚠**モデルが予測できたレースのみ**）")
print(f"　年あたり: **{nall/11.0:,.0f}レース / {nd/11.0:.0f}日**\n")
print(f"{'絞り':<16}{'紐の床':>7}{'買うR数':>9}{'買う日数':>9}{'★1日あたり':>12}"
      f"{'★1開催日の買う率':>16}{'年あたりR':>11}{'★年あたり投資(200円)':>20}")
for fl in FILT:
    for cth in CS:
        v = K[(fl, cth)]
        if not v: continue
        days = len(set(v))
        print(f"{str(fl):<16}{cth:>7.2f}{len(v):>9,}{days:>9,}{len(v)/days:>11.2f}R"
              f"{100*days/nd:>15.1f}%{len(v)/11.0:>10,.0f}{200*len(v)/11.0:>17,.0f}円")
