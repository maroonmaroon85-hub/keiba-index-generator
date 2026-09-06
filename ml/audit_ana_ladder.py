"""(198) ★★★**増幅は「頭数」から来るのか「着順」から来るのか** ＋ **1.18倍は対照の粗さか**

★★**動機（2026-09-06・(197)の続き）**——**(197)で2つ未解決が残った**:
　★**未解決1: 馬単を2点（両方の順序）で買ったので、★着順の腕前を測れていない**。
　　**的中率が馬連と完全に同一（11.377%）になり、順序の情報が消えていた**（設計の限界）。
　★**未解決2: 三連単だけ配当の上乗せが1.18倍**（**馬連は1.03 / 1.00倍**）。
　　⚠**103.1%の大半がここから来ている**。**オッズ帯の粒度が粗いせいかもしれない**。

★★★**梯子を完成させる**——**(197)は2段しかなかった**:
| 段 | 券種 | 頭数 | 着順 | 点数 | 払戻率 | 100%に要る比 |
|---|---|---|---|---|---|---|
| 1 | **馬連** | 2 | **不問** | 1 | 77.5% | 1.290倍 |
| 2 | ★**馬単1点（軸1着固定）** | 2 | ★**要** | 1 | 75.0% | 1.333倍 |
| 3 | ★**三連複** | ★**3** | **不問** | 1 | 75.0% | 1.333倍 |
| 4 | **三連単** | 3 | 要 | 2 | 72.5% | 1.379倍 |
★★**2段目と3段目を足すと、「頭数」と「着順」を分離できる**:
| ★三連複の比 ≈ 三連単の比 | ★三連複 > 三連単 |
|---|---|
| **増幅は頭数から来る。着順は無関係** | **着順で失っている**（**モデルは3着以内しか学習していない**） |
| ★馬単1点の比 ≈ 馬連の比 | ★馬単1点 > 馬連 |
| **着順の腕前は無い** | **着順の腕前がある** |
★**この表を先に書いておく**（判定基準42）。

────────────────────────────────────────────────────────────
★★★ 事前登録（2026-09-06・**結果を見る前にコミットする**）
────────────────────────────────────────────────────────────

■ ★経路（判定基準25）: **q = MLモデル / q_pool = 複勝の板**（調和平均）。⚠**q側は弱い**。
■ ★手続き: **ウォークフォワード**。**母集団は(197)と同一**（**軸 = gap最大、紐 = p降順の2頭**）。
　**絞り (0.15,0.02) / (0.20,0.06)**。

■ ★★★家族A（**8比較**）: **梯子4段 × 絞り2つ**、**的中率の差（本 − 乱帯）**。
　**乱帯 = (197)と同じ「同じオッズ帯の無作為な馬」**（10種平均・種はレースに紐づけ）。

■ ★★★家族B（**2比較**）: ★**対照を細かくする**——**1.18倍は粒度のせいか**
　**乱細 = 「オッズが ±20% 以内の馬」から無作為**（**帯ではなく実オッズで揃える**）。
　**三連単 × 絞り2つ**で、★**配当比が1.00に近づくか**を見る。
| ★配当比が1.0に近づく | ★1.18倍のまま |
|---|---|
| **上乗せは対照の粗さだった**。**103.1%は帯の効果** | **上乗せは実在**。⚠**ただし裾なので測れない** |
　⚠**乱細は該当馬がいないレースを落とすので母集団が変わる**（判定基準25）。
　★**落ちたレース数を必ず報告し、乱帯の値も同じ母集団で再計算して並べる**。

■ ★★ゲート2（判定基準42）——**仮説が偽なら何を返すか**
　★**モデルが馬を選べていないなら、的中率の差の期待値は厳密に0**。
　⚠**(197)で「対照は払戻率を返す」と書いたのは誤りだった**——**対照は同じオッズ帯から引くので
　　(88)の帯の効果を受け継ぐ**。★**今回は「対照は帯のROIを返す」とだけ書く**（判定基準37・5回目を避ける）。

■ ⚠ゲート1: (88)③④を再現（±3pt）／ ★ゲート板: 復元R 0.800±0.020 ／
　★陽性対照: 三連複BOX上位4 81.9%±1.5pt。
■ ★★内部対照（**決定的な量だけ**）:
　**三連単 (0.15,0.02) のROIが (190)(197)の 103.1% を ±0.5pt で再現**。

■ ★★★探索を守る（**10比較・Bonferroni α=0.01/10**）
　⚠**標本300レース未満は判定しない**。★**梯子が単調でなければ、そこに理由を探す**。

■ ★採用条件
　1. **家族Aで4段すべて有意**（**機構が全券種にある**）
　2. ★**三連複の比が三連単の比より明確に大きい → 「着順で失う」を採る**
　3. ★**馬単1点の比が馬連の比より明確に大きい → 「着順の腕前がある」を採る**
　4. **家族Bで配当比が1.0に近づく → 1.18倍は粒度の産物と結論する**

■ 予想（⚠**類推なので当てにしない**・判定基準24）
　★**着順の腕前は無いと見る**——**モデルは「3着以内か」の二値でしか学習していない**
　　（**(187)で後ろの着順ほど悪かったのも同じ話**）。**馬単1点の比 ≈ 馬連の比 を予想**。
　★**三連複の比は三連単の比と同程度かやや大きいと見る**。
　★**家族B: 配当比は1.0に近づくと見る**（**帯が粗いのが原因という読み**）。
　⚠**ただし(197)の予想（「対照は払戻率を返す」）は外したばかりなので、予想は当てにしない**。

実行: python3 ml/audit_ana_ladder.py    自己テスト: python3 ml/audit_ana_ladder.py --selftest
"""
import math
import sys
from itertools import combinations
from zlib import crc32

import numpy as np

sys.path.insert(0, "ml")
import features as F
from audit_crosspool import LINE, load_races, payoff, zq
from audit_ana_odds import BANDS, MIN_HORSES, band_of, gate1, roi_of
from audit_ana_marg import WF_BOX4, WF_TOL, wf_predict
from audit_ana_board import BOARD_R, BOARD_TOL, NPLACE, load_fuku_boards, qpool
from train_prod import add_odds_features

FILT = [(0.15, 0.02), (0.20, 0.06)]
LADDER = ["馬連", "馬単1点", "三連複", "三連単"]
NPT = {"馬連": 1, "馬単1点": 1, "三連複": 1, "三連単": 2}
LN = {"馬連": LINE["馬連"], "馬単1点": LINE["馬単"],
      "三連複": LINE["三連複"], "三連単": LINE["三連単"]}
FINE = 1.20            # ★乱細: オッズが ±20% 以内
NSEED = 10
SEED = 20260906
CTRL_TRI, KNOWN_TRI, ROI_TOL = (0.15, 0.02), 103.1, 0.5
MINCELL = 300
NCMP = len(LADDER) * len(FILT) + len(FILT)     # 8 + 2
ALPHA = 0.01


def tickets(kind, ax, h1, h2):
    if kind == "馬連":
        return [("馬連", [ax, h1])]
    if kind == "馬単1点":
        return [("馬単", [ax, h1])]
    if kind == "三連複":
        return [("三連複", sorted([ax, h1, h2]))]
    return [("三連単", [ax, h1, h2]), ("三連単", [ax, h2, h1])]


def selftest():
    ok = True
    z = zq(ALPHA / NCMP)
    print(f"★家族A {len(LADDER)*len(FILT)}比較（梯子4段 × 絞り2つ）"
          f" ＋ 家族B {len(FILT)}比較（★対照を細かく） → **{NCMP}比較**・z = {z:.3f}")
    print(f"{'段':<4}{'券種':<10}{'頭数':>5}{'着順':>6}{'点数':>5}{'払戻率':>8}{'★100%に要る比':>15}")
    meta = [("1", "馬連", 2, "不問"), ("2", "馬単1点", 2, "★要"),
            ("3", "三連複", 3, "不問"), ("4", "三連単", 3, "要")]
    for no, k, nh, ord_ in meta:
        print(f"{no:<4}{k:<10}{nh:>5}{ord_:>6}{NPT[k]:>5}{100*LN[k]:>7.1f}%"
              f"{1/LN[k]:>14.3f}倍")
    print("★★分離の論理: **三連複 ≈ 三連単 → 増幅は頭数から / "
          "三連複 > 三連単 → 着順で失う**")
    print("★★　　　　　 **馬単1点 ≈ 馬連 → 着順の腕前なし / "
          "馬単1点 > 馬連 → 腕前あり**")
    rng = np.random.default_rng(0)
    a = rng.random(200_000) < 0.0263
    b = rng.random(200_000) < 0.0263
    se = math.sqrt(2 * 0.0263 * 0.9737 / 200_000)
    print(f"★ゲート2（的中率の差）: **偽なら0** → {100*(a.mean()-b.mean()):+.4f}pp"
          f"　{'★OK' if abs(a.mean()-b.mean()) < 3*se else '⚠NG'}")
    # ★券の構成の自己テスト
    t = tickets("三連複", 5, 3, 9)
    print(f"★券の構成: 三連複(軸5・紐3,9) → {t}　"
          f"{'★OK' if t == [('三連複',[3,5,9])] else '⚠NG'}")
    ok &= t == [("三連複", [3, 5, 9])]
    t2 = tickets("馬単1点", 5, 3, 9)
    print(f"★券の構成: 馬単1点(軸5・紐3) → {t2}（**軸1着固定**）　"
          f"{'★OK' if t2 == [('馬単',[5,3])] else '⚠NG'}")
    ok &= t2 == [("馬単", [5, 3])]
    print(f"★★内部対照（決定的）: **三連単 {CTRL_TRI} ROI = {KNOWN_TRI}% ±{ROI_TOL}pt**")
    print("★自己テスト: " + ("全部OK" if ok else "⚠NG"))
    return 0 if ok else 1


def main():
    z = zq(ALPHA / NCMP)
    print("(198) ★★★**増幅は「頭数」から来るのか「着順」から来るのか**")
    print("★梯子を4段に: **馬連(2頭・不問) → 馬単1点(2頭・要) → 三連複(3頭・不問) → 三連単(3頭・要)**")
    print("★★あわせて: **1.18倍の配当上乗せは、対照の粗さのせいか**\n")

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

    ARMS = ("本", "乱帯", "乱細")
    K = {(fl, k): {a: {"pay": [], "hit": [], "fine": []} for a in ARMS}
         for fl in FILT for k in LADDER}
    nfine = {fl: [0, 0] for fl in FILT}
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
        gap = pn - qp
        bx = [payoff(r, "三連複", list(c))
              for c in combinations(sorted(int(u) for u in ub[np.argsort(-pv)[:4]]), 3)]
        if not any(x is None for x in bx):
            box4.append(sum(bx) - 400.0)
        bi = np.array([band_of(float(o), BANDS) for o in od])
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
            nfine[fl][0] += 1

            def draw(rngen, mode):
                out = []
                for u in trio:
                    k0 = pos[u]
                    if mode == "乱帯":
                        pl = [int(ub[k]) for k in range(len(ub))
                              if bi[k] == bi[k0] and int(ub[k]) not in out]
                    else:
                        pl = [int(ub[k]) for k in range(len(ub))
                              if int(ub[k]) not in out
                              and od[k0] / FINE <= od[k] <= od[k0] * FINE]
                    if not pl:
                        return None
                    out.append(int(rngen.choice(pl)))
                return out

            sets = {"本": [trio]}
            okall = True
            for mode in ("乱帯", "乱細"):
                lst = []
                for sd in range(NSEED):
                    g2 = np.random.default_rng([SEED + sd, crc32(str(rid).encode())])
                    t = draw(g2, mode)
                    if t is None:
                        okall = False
                        break
                    lst.append(t)
                if not okall:
                    break
                sets[mode] = lst
            fine_ok = okall
            if fine_ok:
                nfine[fl][1] += 1
            else:
                # ★乱細が作れない → 乱細だけ欠測にして、本と乱帯は残す
                sets.pop("乱細", None)
                lst = []
                bad2 = False
                for sd in range(NSEED):
                    g2 = np.random.default_rng([SEED + sd, crc32(str(rid).encode())])
                    t = draw(g2, "乱帯")
                    if t is None:
                        bad2 = True
                        break
                    lst.append(t)
                if bad2:
                    continue
                sets["乱帯"] = lst
            vals, okall = {}, True
            for kind in LADDER:
                for which, lst in sets.items():
                    acc, hh = [], []
                    for t in lst:
                        vs = [payoff(r, nm, sel) for nm, sel in tickets(kind, *t)]
                        if any(v is None for v in vs):
                            okall = False
                            break
                        acc.append(sum(vs))
                        hh.append(1.0 if max(vs) > 0 else 0.0)
                    if not okall:
                        break
                    vals[(kind, which)] = (float(np.mean(acc)), float(np.mean(hh)))
                if not okall:
                    break
            if not okall:
                continue
            for kind in LADDER:
                for which in sets:
                    pay, hit = vals[(kind, which)]
                    c = K[(fl, kind)][which]
                    c["pay"].append(pay); c["hit"].append(hit)
                    c["fine"].append(1.0 if fine_ok else 0.0)
    print(f"\n★対象 **{nall:,}レース**")

    med = float(np.median(Rs))
    okR = abs(med - BOARD_R) <= BOARD_TOL
    print(f"■ ★ゲート板: 復元R = **{med:.4f}** → **{'★立った' if okR else '⚠⚠落ちた'}**")
    box4 = np.asarray(box4, float)
    g0 = 100.0 * (box4.sum() + 400.0 * len(box4)) / (400.0 * len(box4))
    okb = abs(g0 - WF_BOX4) <= WF_TOL
    print(f"■ ⚠**陽性対照**: BOX上位4 **{g0:.1f}%** vs {WF_BOX4}%"
          f" → **{'★立った' if okb else '⚠⚠落ちた'}**")
    tv = np.asarray(K[(CTRL_TRI, "三連単")]["本"]["pay"], float)
    tr = 100.0 * tv.mean() / (100.0 * NPT["三連単"])
    okc = abs(tr - KNOWN_TRI) <= ROI_TOL
    print(f"■ ★★内部対照（決定的）: 三連単 {CTRL_TRI} ROI = **{tr:.1f}%** vs "
          f"{KNOWN_TRI}% → **{'★再現' if okc else '⚠⚠ズレた'}**（{len(tv):,}R）")
    if not (okR and okb and okc):
        print("\n⚠⚠**ゲートが落ちた。読まない**（判定基準32）。")
        return

    print(f"\n{'='*116}")
    print("■ ★★★家族A: **梯子4段の的中率**（**本 − 乱帯**）")
    print("　★★分離: **三連複 ≈ 三連単 → 増幅は頭数から / 三連複 > 三連単 → 着順で失う**")
    print("　★★　　　 **馬単1点 ≈ 馬連 → 着順の腕前なし / 馬単1点 > 馬連 → 腕前あり**")
    for fl in FILT:
        print(f"\n★絞り{fl}")
        print(f"{'段':<4}{'券種':<9}{'頭数':>5}{'着順':>6}{'R数':>8}{'★本の的中率':>13}"
              f"{'乱帯':>9}{'★差':>10}{'99%CI':>21}{'★★比':>9}{'必要':>7}{'ROI':>8}{'判定':>12}")
        meta = {"馬連": (2, "不問"), "馬単1点": (2, "★要"),
                "三連複": (3, "不問"), "三連単": (3, "要")}
        for i, kind in enumerate(LADDER, 1):
            c = K[(fl, kind)]
            a = np.asarray(c["本"]["hit"], float)
            rr = np.asarray(c["乱帯"]["hit"], float)
            pa = np.asarray(c["本"]["pay"], float)
            if len(a) < MINCELL:
                continue
            dd = a - rr
            mu = dd.mean(); se = dd.std(ddof=1) / math.sqrt(len(dd))
            ratio = a.mean() / max(rr.mean(), 1e-12)
            nh, ordr = meta[kind]
            print(f"{i:<4}{kind:<9}{nh:>5}{ordr:>6}{len(a):>8,}{100*a.mean():>12.3f}%"
                  f"{100*rr.mean():>8.3f}%{100*mu:>+9.3f}pp"
                  f"{f'[{100*(mu-z*se):+.3f},{100*(mu+z*se):+.3f}]':>21}"
                  f"{ratio:>8.3f}倍{1/LN[kind]:>6.2f}倍"
                  f"{100*pa.mean()/(100*NPT[kind]):>7.1f}%"
                  f"{('★★有意' if mu - z*se > 0 else '⚠通らない'):>12}")

    print(f"\n{'='*116}")
    print("■ ★★★家族B: ★**対照を細かくすると配当の上乗せは消えるか**"
          f"（**乱細 = オッズ±{100*(FINE-1):.0f}%以内**）")
    print("　★★読み方: **配当比が1.0に近づく → 上乗せは対照の粗さだった**")
    for fl in FILT:
        tot, fin = nfine[fl]
        print(f"\n★絞り{fl}　**乱細が作れたレース {fin:,}/{tot:,}（{100*fin/max(tot,1):.1f}%）**"
              f"　⚠**母集団が変わる**（判定基準25）")
        print(f"{'券種':<9}{'腕':<7}{'R数':>8}{'的中率':>10}{'★1回配当':>12}"
              f"{'★配当比':>10}{'ROI':>8}")
        for kind in LADDER:
            c = K[(fl, kind)]
            fm = np.asarray(c["本"]["fine"], float) > 0
            a = np.asarray(c["本"]["pay"], float)[fm]
            ah = np.asarray(c["本"]["hit"], float)[fm]
            if len(a) < MINCELL:
                continue
            pa = a.sum() / max(ah.sum(), 1e-9)
            row = []
            for which in ("乱帯", "乱細"):
                cc = K[(fl, kind)][which]
                fm2 = np.asarray(cc["fine"], float) > 0
                v = np.asarray(cc["pay"], float)[fm2]
                h = np.asarray(cc["hit"], float)[fm2]
                if not len(v):
                    row.append(None)
                    continue
                row.append((v, h, v.sum() / max(h.sum(), 1e-9)))
            cost = 100.0 * NPT[kind]
            print(f"{kind:<9}{'本':<7}{len(a):>8,}{100*ah.mean():>9.3f}%"
                  f"{pa:>11,.0f}円{'—':>10}{100*a.mean()/cost:>7.1f}%")
            for nm, rw in zip(("乱帯", "乱細"), row):
                if rw is None:
                    continue
                v, h, pr = rw
                print(f"{'':<9}{nm:<7}{len(v):>8,}{100*h.mean():>9.3f}%"
                      f"{pr:>11,.0f}円{pa/max(pr,1e-9):>9.2f}倍"
                      f"{100*v.mean()/cost:>7.1f}%")

    print("\n■ ★採用条件: **1.4段とも有意 / 2.三連複>三連単なら着順で失う / "
          "3.馬単1点>馬連なら着順の腕前あり / 4.配当比が1.0に近づけば粒度の産物**")
    print("⚠**経路は弱いので「閉じた」とは書けない**（判定基準25）。")
    print("\n⚠**枠連の運用には触れない**。**設定変更は提案しない**。")


if __name__ == "__main__":
    sys.exit(selftest() if "--selftest" in sys.argv else (main() or 0))
