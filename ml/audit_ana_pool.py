"""(199) ★★★**配当の上乗せ1.20倍は本物か** — ★**算術平均をやめて対数で測る**

★★**動機（2026-09-06・(198)の続き）**:
　★**(198)で三連単だけ配当が1.20倍だった**（**馬連1.03 / 馬単1点1.06 / 三連複1.06倍**）。
　★**オッズを±20%以内で揃え直しても消えず、むしろ増えた**＝**対照の粗さではない**。
　★★**意味**: **三連単プールは、単勝オッズが示す以上にこの馬たちを割安に扱っている**
　　＝**プールをまたぐ歪み**。**枠連117.3%を生んだのと同じ系統（ROI_MAP IV「比の裾」）**。
　⚠**だが1.20倍は「的中774回の算術平均」の比**。**算術平均は裾に振り回される**。

★★★**物差しを替える（判定基準26）——対数で測る**:
　★**的中したときの配当の「対数の平均」＝幾何平均**を比べる。
　★★**プールが同じ値付けをしているなら、対数配当の差は厳密に0**（**ゲート2**）。
　★**歪みが本物なら分布ごと右にずれるので、対数でも差が出る**。
　★**算術平均より桁違いに安定**——**1本の大穴で動かない**。
| ★**対数の差が正で有意** | ★**対数の差が0** |
|---|---|
| **プールをまたぐ歪みは実在** | ⚠**1.20倍は算術平均の裾の産物** |
　★**この表を先に書いておく**（判定基準42）。

────────────────────────────────────────────────────────────
★★★ 事前登録（2026-09-06・**結果を見る前にコミットする**）
────────────────────────────────────────────────────────────

■ ★経路（判定基準25）: **q = MLモデル / q_pool = 複勝の板**。⚠**q側は弱い**。
■ ★母集団・買い方は(198)と同一（**軸 = gap最大、紐 = p降順の2頭、絞り2つ**）。
　**対照は 乱細（オッズ±20%以内・10種）**——**(198)で「粗さではない」と確認した側を使う**。

■ ★★★家族A（**8比較**）: **的中時の対数配当の差**（**本 − 乱細**）、**4券種 × 絞り2つ**。
　★**券種を並べるのは対照のため**——**馬連・馬単1点・三連複で差が出ず、
　　三連単だけ出るなら、「3頭・着順のプールだけが歪んでいる」と言える**。
　★**標準誤差はレース単位のブートストラップ（2,000回）**——**同じレースの10種は相関するから**。

■ ★★家族B（**記述・判定しない**）: **算術の配当比のブートストラップ99%区間**、
　**中央値の比**、**対数の差を倍率に直した値**。★**算術と幾何がどれだけ違うかを見る**。

■ ★★ゲート2（判定基準42）——**仮説が偽なら何を返すか**
　★**両プールが同じ値付けなら、対数配当の差の期待値は厳密に0**。
　⚠**「対照は払戻率を返す」とは書かない**（**(197)で外した**）。**対照は帯のROIを返す**。

■ ⚠ゲート1: (88)③④を再現（±3pt）／ ★ゲート板: 復元R 0.800±0.020 ／
　★陽性対照: 三連複BOX上位4 81.9%±1.5pt。
■ ★★内部対照（**決定的な量だけ**）: **三連単 (0.15,0.02) のROIが 103.1% を ±0.5pt で再現**。

■ ★★★探索を守る（**8比較・Bonferroni α=0.01/8**）
　⚠**的中が300回未満の腕は判定しない**（判定基準5）。

■ ★採用条件
　1. **三連単で対数配当の差が有意に正**
　2. **馬連・馬単1点・三連複では差が小さい**（**三連単だけという形になっている**）
　3. **2つの絞りで同じ向き**
　4. **算術と幾何で符号が一致**（**裾だけの現象になっていない**）

■ 予想（⚠**類推なので当てにしない**・判定基準24）
　⚠**私は(198)で予想を2つとも外した**（「着順の腕前は無い」「粒度の産物」）。**予想は弱い**。
　★**対数でも差は残ると見る**——**±20%の揃え直しで増えたのが決め手**。
　⚠**ただし倍率は1.20より小さくなると見る**（**算術平均は裾で膨らむから**）。
　★**もし対数で差が消えるなら、三連単の103.1%は完全に裾の産物と結論する**。

実行: python3 ml/audit_ana_pool.py    自己テスト: python3 ml/audit_ana_pool.py --selftest
"""
import math
import sys
from itertools import combinations
from zlib import crc32

import numpy as np

sys.path.insert(0, "ml")
import features as F
from audit_crosspool import LINE, load_races, payoff, zq
from audit_ana_odds import MIN_HORSES, gate1, roi_of
from audit_ana_marg import WF_BOX4, WF_TOL, wf_predict
from audit_ana_board import BOARD_R, BOARD_TOL, NPLACE, load_fuku_boards, qpool
from audit_ana_ladder import FILT, LADDER, NPT, FINE, NSEED, SEED, tickets
from train_prod import add_odds_features

CTRL_TRI, KNOWN_TRI, ROI_TOL = (0.15, 0.02), 103.1, 0.5
MINHIT = 300
NBOOT = 2000
NCMP = len(LADDER) * len(FILT)      # 8
ALPHA = 0.01


def boot_logdiff(ha, hr, rng, nboot=NBOOT):
    """★レース単位のブートストラップ（同じレースの10種は相関するので束ねる）
    ha/hr: レースごとの [配当>0 のリスト]（本 / 乱細）"""
    n = len(ha)
    obs = []
    for _ in range(nboot):
        idx = rng.integers(0, n, size=n)
        va = [v for i in idx for v in ha[i]]
        vr = [v for i in idx for v in hr[i]]
        if not va or not vr:
            continue
        obs.append(float(np.mean(np.log(va)) - np.mean(np.log(vr))))
    return np.asarray(obs, float)


def selftest():
    ok = True
    z = zq(ALPHA / NCMP)
    print(f"★家族A {NCMP}比較（**的中時の対数配当の差**・{LADDER} × 絞り2つ）→ z = {z:.3f}")
    rng = np.random.default_rng(0)
    # ★ゲート2: 同じ分布から引けば対数の差は0
    ha = [list(rng.lognormal(8.0, 1.2, size=rng.integers(0, 3))) for _ in range(4000)]
    hr = [list(rng.lognormal(8.0, 1.2, size=rng.integers(0, 3))) for _ in range(4000)]
    b = boot_logdiff(ha, hr, np.random.default_rng(1), 400)
    lo, hi = np.percentile(b, [0.5, 99.5])
    print(f"★ゲート2（対数の差）: **同じ分布なら0** → 中央 {np.median(b):+.4f}"
          f"　99%区間 [{lo:+.4f},{hi:+.4f}]　{'★OK' if lo < 0 < hi else '⚠NG'}")
    ok &= lo < 0 < hi
    # ★検出力: 1.20倍(=log 0.182)を仕込めば検出できるか
    hr2 = [[v / 1.20 for v in x] for x in hr]
    b2 = boot_logdiff(ha, hr2, np.random.default_rng(2), 400)
    lo2 = np.percentile(b2, 0.5)
    print(f"★検出力: **1.20倍を仕込むと** 中央 {np.median(b2):+.4f}"
          f"（log1.20={math.log(1.2):.4f}）99%下端 {lo2:+.4f}"
          f"　{'★検出できた' if lo2 > 0 else '⚠NG'}")
    ok &= lo2 > 0
    print("★★読み方: **対数で差が残る→プールをまたぐ歪みは実在 / "
          "消える→103.1%は完全に裾の産物**")
    print(f"★★内部対照（決定的）: **三連単 {CTRL_TRI} ROI = {KNOWN_TRI}% ±{ROI_TOL}pt**")
    print("★自己テスト: " + ("全部OK" if ok else "⚠NG"))
    return 0 if ok else 1


def main():
    print("(199) ★★★**配当の上乗せ1.20倍は本物か** — ★**算術平均をやめて対数で測る**")
    print("★★プールが同じ値付けなら対数の差は厳密に0。**歪みが本物なら分布ごと右にずれる**\n")

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

    K = {(fl, k): {"a": [], "r": [], "pa": [], "pr": [], "cost": 100.0 * NPT[k]}
         for fl in FILT for k in LADDER}
    Rs, box4 = [], []
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
        gap = pn - qp
        bx = [payoff(r, "三連複", list(c))
              for c in combinations(sorted(int(u) for u in ub[np.argsort(-pv)[:4]]), 3)]
        if not any(x is None for x in bx):
            box4.append(sum(bx) - 400.0)
        pos = {int(u): k for k, u in enumerate(ub)}
        for fl in FILT:
            cand = np.where((pn >= fl[0]) & (gap >= fl[1]))[0]
            if not len(cand):
                continue
            ai = int(cand[int(np.argmax(gap[cand]))])
            axu = int(ub[ai])
            hp = [int(u) for u in ub[np.argsort(-pv, kind="mergesort")] if int(u) != axu]
            if len(hp) < 2:
                continue
            trio = [axu, hp[0], hp[1]]
            lst, okall = [], True
            for sd in range(NSEED):
                g2 = np.random.default_rng([SEED + sd, crc32(str(rid).encode())])
                out = []
                for u in trio:
                    k0 = pos[u]
                    pl = [int(ub[k]) for k in range(len(ub))
                          if int(ub[k]) not in out
                          and od[k0] / FINE <= od[k] <= od[k0] * FINE]
                    if not pl:
                        okall = False
                        break
                    out.append(int(g2.choice(pl)))
                if not okall:
                    break
                lst.append(out)
            if not okall:
                continue
            vals, okall = {}, True
            for kind in LADDER:
                va = [payoff(r, nm, sel) for nm, sel in tickets(kind, *trio)]
                if any(v is None for v in va):
                    okall = False
                    break
                acc = []
                for t in lst:
                    vs = [payoff(r, nm, sel) for nm, sel in tickets(kind, *t)]
                    if any(v is None for v in vs):
                        okall = False
                        break
                    acc.append(sum(vs))
                if not okall:
                    break
                vals[kind] = (sum(va), acc)
            if not okall:
                continue
            for kind in LADDER:
                pa, accs = vals[kind]
                c = K[(fl, kind)]
                c["a"].append([pa] if pa > 0 else [])
                c["r"].append([v for v in accs if v > 0])
                c["pa"].append(pa)
                c["pr"].append(float(np.mean(accs)))

    med = float(np.median(Rs))
    okR = abs(med - BOARD_R) <= BOARD_TOL
    print(f"\n■ ★ゲート板: 復元R = **{med:.4f}** → **{'★立った' if okR else '⚠⚠落ちた'}**")
    box4 = np.asarray(box4, float)
    g0 = 100.0 * (box4.sum() + 400.0 * len(box4)) / (400.0 * len(box4))
    okb = abs(g0 - WF_BOX4) <= WF_TOL
    print(f"■ ⚠**陽性対照**: BOX上位4 **{g0:.1f}%** vs {WF_BOX4}%"
          f" → **{'★立った' if okb else '⚠⚠落ちた'}**")
    tv = np.asarray(K[(CTRL_TRI, "三連単")]["pa"], float)
    tr = 100.0 * tv.mean() / (100.0 * NPT["三連単"])
    okc = abs(tr - KNOWN_TRI) <= ROI_TOL
    print(f"■ ★★内部対照（決定的）: 三連単 {CTRL_TRI} ROI = **{tr:.1f}%** vs "
          f"{KNOWN_TRI}% → **{'★再現' if okc else '⚠⚠ズレた'}**（{len(tv):,}R）")
    if not (okR and okb and okc):
        print("\n⚠⚠**ゲートが落ちた。読まない**（判定基準32）。")
        return

    z = zq(ALPHA / NCMP)
    rng = np.random.default_rng(SEED)
    print(f"\n{'='*116}")
    print("■ ★★★家族A: **的中時の対数配当の差**（**本 − 乱細**・レース単位ブートストラップ）")
    print("　★★読み方: **差が残る→プールをまたぐ歪みは実在 / 消える→103.1%は裾の産物**")
    for fl in FILT:
        print(f"\n★絞り{fl}")
        print(f"{'券種':<9}{'的中(本)':>9}{'的中(乱細)':>11}{'★幾何平均(本)':>15}"
              f"{'(乱細)':>11}{'★★対数の差':>12}{'99%区間':>21}{'→倍率':>9}{'判定':>12}")
        for kind in LADDER:
            c = K[(fl, kind)]
            ha, hr = c["a"], c["r"]
            na = sum(len(x) for x in ha)
            nr = sum(len(x) for x in hr)
            if na < MINHIT or nr < MINHIT:
                print(f"{kind:<9}{na:>9,}{nr:>11,}　⚠**的中不足で判定しない**")
                continue
            va = np.concatenate([np.asarray(x, float) for x in ha if x])
            vr = np.concatenate([np.asarray(x, float) for x in hr if x])
            ga, gr = math.exp(np.log(va).mean()), math.exp(np.log(vr).mean())
            b = boot_logdiff(ha, hr, rng)
            lo, hi = np.percentile(b, [0.5, 99.5])
            mu = float(np.median(b))
            sig = lo > 0
            print(f"{kind:<9}{na:>9,}{nr:>11,}{ga:>14,.0f}円{gr:>10,.0f}円"
                  f"{mu:>+11.4f}{f'[{lo:+.4f},{hi:+.4f}]':>21}{math.exp(mu):>8.3f}倍"
                  f"{('★★有意' if sig else '⚠通らない'):>12}")

    print(f"\n{'='*116}")
    print("■ ★★家族B（**記述**）: **算術と幾何はどれだけ違うか**")
    for fl in FILT:
        print(f"\n★絞り{fl}")
        print(f"{'券種':<9}{'★算術の配当比':>15}{'99%区間':>23}{'幾何の比':>11}"
              f"{'中央値の比':>12}{'ROI(本)':>10}{'ROI(乱細)':>11}")
        for kind in LADDER:
            c = K[(fl, kind)]
            ha, hr = c["a"], c["r"]
            na = sum(len(x) for x in ha)
            if na < MINHIT:
                continue
            va = np.concatenate([np.asarray(x, float) for x in ha if x])
            vr = np.concatenate([np.asarray(x, float) for x in hr if x])
            n = len(ha)
            bs = []
            for _ in range(NBOOT):
                idx = rng.integers(0, n, size=n)
                aa = [v for i in idx for v in ha[i]]
                rr = [v for i in idx for v in hr[i]]
                if aa and rr:
                    bs.append(np.mean(aa) / np.mean(rr))
            bs = np.asarray(bs, float)
            lo, hi = np.percentile(bs, [0.5, 99.5])
            ga, gr = math.exp(np.log(va).mean()), math.exp(np.log(vr).mean())
            pa = np.asarray(c["pa"], float); pr = np.asarray(c["pr"], float)
            print(f"{kind:<9}{va.mean()/vr.mean():>14.3f}倍"
                  f"{f'[{lo:.3f},{hi:.3f}]':>23}{ga/gr:>10.3f}倍"
                  f"{np.median(va)/np.median(vr):>11.3f}倍"
                  f"{100*pa.mean()/c['cost']:>9.1f}%{100*pr.mean()/c['cost']:>10.1f}%")

    print("\n■ ★採用条件: **1.三連単で対数の差が有意 / 2.他券種では小さい / "
          "3.2絞りで同じ向き / 4.算術と幾何で符号一致**")
    print("⚠**経路は弱いので「閉じた」とは書けない**（判定基準25）。")
    print("\n⚠**枠連の運用には触れない**。**設定変更は提案しない**。")


if __name__ == "__main__":
    sys.exit(selftest() if "--selftest" in sys.argv else (main() or 0))
