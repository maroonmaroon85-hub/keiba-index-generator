"""(179) ★★★**穴の定義を(174)の形に取り直して2軸をやり直す**

★★動機——**(178)が陰性だったが、穴の定義が悪かった疑いが残った**:
| 複勝1点・全レース（(178)実測） | ROI |
|---|---|
| **モデル1位** | ★**85.1%** |
| 1番人気 | 83.9% |
| ★**穴＝「比 `share/mk` が最大の馬」** | ⚠**73.3%** |
| 乱＝オッズ帯を揃えた無作為 | 63.4% |
★**穴 − 乱 = +9.9pt** で**情報はある**（(177)の +15.2円 と同じ向き）。
⚠★**だが「比が最大」はモデル1位に11.8pt負ける**——**(171)で軸の平均オッズが18.6倍まで
　穴側に寄ったのと同じ**＝★**(75)(120)「残差順は人気薄順になる」にそのまま落ちている**。
→ ★★**(174)で実際に効いた定義は「レース内で最大」ではなく「オッズ帯の中で上位」だった**:
　**3-6倍 87.6% / 6-15倍 85.8%**（複勝・上位10%）。**帯を跨いで最大値を取ると穴に落ちる**。

────────────────────────────────────────────────────────────
★★★ 事前登録（2026-09-05・**結果を見る前にコミットする**）
────────────────────────────────────────────────────────────

■ ★経路（判定基準25）: 穴の選定に MLモデルの p ／ 相方・統制は単勝オッズ。⚠**弱い経路**。
　⚠**標本は(174)〜(178)と同一**＝**独立な検証ではない**（判定基準35）。

■ ★★穴の定義（**3通り。これ以外に増やさない**）
　**帯内パーセンタイル** … 各馬の `gap = share − mk` を、**その馬のオッズ帯の中で**順位付けし
　　　　　　　　　　　　　**0〜1に正規化**した値（**(174)の切り方の連続版**）。
　| 定義 | 中身 | 位置づけ |
　|---|---|---|
　| ★**穴A** | **全帯**の中で、帯内パーセンタイルが最大の馬 | **post-hocなし** |
　| ★**穴B** | ★**3-15倍に限定**して同じもの | ⚠**post-hoc**（**(174)の最良帯から取った**・後述） |
　| **穴C** | **比 `share/mk` が最大の馬** | ★**(178)と同一＝内部対照**（**73.3%が再現すべき**） |
　⚠⚠**穴Bは post-hoc である**——**(174)の 3-6倍(87.6%)・6-15倍(85.8%) を見てから帯を選んだ**。
　　★**だから穴Aを主に置き、穴Bは「同じ向きが出るか」の確認として読む**。**穴Bだけ跳ねたら採らない**。

■ ★★ゲート3（**新設・これが通らなければ2軸を読まない**）
　★**穴A・穴Bの「複勝1点」ROIが、複勝の払戻率 80.0% 以上であること**。
　★**理由**: **定義を取り直す目的は「73.3%を直すこと」**。**払戻率にすら届かないなら直っていない**。
　　→ **その場合は2軸を読まず、「穴の定義では直らない」と書いて閉じる**。
　★**内部対照**: **穴Cの複勝が(178)の 73.3% を±2ptで再現すること**（**装置が同じであることの確認**）。

■ ★★買い方（**主判定は点数1の3つだけ**。**(178)で3点が大きく悪いと実測済み**）
　| # | 券種 | 買い目 | 点数 | 払戻率 |
　|---|---|---|---|---|
　| 1 | **馬連** | 穴 × 1番人気 | **1** | 0.775 |
　| 2 | **ワイド** | 穴 × 1番人気 | **1** | 0.775 |
　| 3 | **三連複** | 穴 + 1番人気 + モデル最上位1頭 | **1** | 0.750 |
　★**記述**: 馬単2点・三連複3点・複勝1点・モ2・人2 も出すが**判定しない**。

■ ⚠**レース選別のアームは外す**（**(178)の実測に基づく**）
　★**(112)裾2%は397レース(1.5%)でCIが ±115〜±471円＝検出力ゼロ**と実測で確定した。
　→ ★**全レースのみ**。⚠**これは「選別が効かない」ではなく「この標本では測れない」**（判定基準5）。

■ ★★★主判定（**6比較・Bonferroni α=0.01/6・z=2.936**）
　★**「穴×人」−「乱×人」の対応差（円）**を、**2つの穴定義（A・B） × 3つの買い方**で。
　★**乱** … **その穴と同じオッズ帯**からの無作為な1頭（**穴A用・穴B用で別々に引く**）。

■ ★★ゲート2（判定基準42）——**仮説が偽なら何を返すか**
　★**仮説が偽（穴の選定が何も持たない）なら、穴と乱は同じ帯からの交換可能な2頭になり、
　　対応差の期待値は 0 を返す**。**点数もコストも券種も完全に同一**。
　・**(170)の「頭数が減れば必ず良くなる量」ではない**／**(168)の「両方が買うレースだけ」もしない**。

■ ⚠ゲート1（判定基準32）: (88)③④の再現 ／ ★陽性対照: 三連複BOX上位4が84.5%を±2.5ptで再現

■ ★採用条件（判定基準39/40/41）
　1. **主判定がBonferroniを通る**
　2. ★**穴Aと穴Bで同じ向き**（**穴Bだけならpost-hocの産物として採らない**）
　3. ★**券種をまたいで同じ向き**（**1マスだけ跳ねるのは(172)で崩れた形**）
　4. ★**人気順対照（人2）にも同符号で勝つ**（**(69)**）
　5. **裾の検算で符号が反転しない**（上位3本・前後半・年別）
　6. ★**ROI>100% でなければ「機構は在るが張れない」**

■ ★★「最良のマス」の扱い（**(178)でプラセボが109.9%を出した**）
　⚠**(178)の実測**: **中身が無作為な軸が、馬単2点で109.9%・馬連1点で99.6%**（397レース）。
　→ ★★**全マスのROIは記述として出すが、そこから最良を選んで結論にしない**。**主判定だけを読む**。

■ 予想
　**持たない**。⚠**逆風**: **馬連・ワイドは払戻率77.5%で100%に要るのは+22.5pt**、
　**三連複は75.0%で+25.0pt**。★**(174)の最良は +7.6pt**。**定義を直しても、この差は残る**。

実行: python3 ml/audit_ana_pair2.py    自己テスト: python3 ml/audit_ana_pair2.py --selftest
"""
import math
import sys
from itertools import combinations

import numpy as np

sys.path.insert(0, "ml")
import features as F
from audit_crosspool import LINE, load_races, payoff, zq
from audit_ana_odds import BANDS, COST, MIN_HORSES, band_of, gate1, roi_of
from audit_ana_pair import _puku1, _puku3, _umaren, _umatan, _wide
from train_prod import CAPACITY, add_odds_features, fit_seeds

NCMP = 6
ALPHA = 0.01
SEED = 20260905
KNOWN_BOX4, TOL_BOX4 = 84.5, 2.5
KNOWN_C, TOL_C = 73.3, 2.0        # 内部対照: (178)の穴C（比最大）の複勝ROI
GATE3_LINE = 80.0                 # ゲート3: 複勝の払戻率
BANDS_B = (1, 2)                  # ★穴B: 3-6倍 と 6-15倍（⚠post-hoc）

SHAPES = [("馬連 軸2頭", "馬連", 1, _umaren),
          ("ワイド 軸2頭", "ワイド", 1, _wide),
          ("三連複 軸2頭+モ1", "三連複", 1, _puku1)]
DESC = [("馬単 軸2頭 両方向", "馬単", 2, _umatan),
        ("三連複 軸2頭+モ3", "三連複", 3, _puku3)]
ANAS = ["★穴A", "★穴B", "穴C"]


def band_pct(gap, bidx, nband):
    """★オッズ帯の中での gap のパーセンタイル(0〜1)。(174)の切り方の連続版。"""
    out = np.zeros(len(gap))
    for b in range(nband):
        m = bidx == b
        if not m.any():
            continue
        r = np.argsort(np.argsort(gap[m]))
        out[m] = r / max(len(r) - 1, 1)
    return out


def selftest():
    ok = True
    g = np.array([1.0, 2.0, 3.0, 10.0, 20.0, 30.0])
    b = np.array([0, 0, 0, 1, 1, 1])
    p = band_pct(g, b, 2)
    assert np.allclose(p, [0.0, 0.5, 1.0, 0.0, 0.5, 1.0])
    print("★帯内パーセンタイルの自己テスト: 帯ごとに0〜1へ正規化　★OK")
    # ★帯をまたぐと「比が最大」は穴に寄るが、帯内パーセンタイルは寄らない
    #   （帯1の最大値20.0は帯0の最大3.0より大きいが、パーセンタイルはどちらも1.0）
    assert p[2] == p[5] == 1.0
    print("　→ **帯をまたいで最大値を取る定義とは別物になる**ことを確認")
    rng = np.random.default_rng(0)
    n = 200_000
    pay = rng.choice([0.0, 1500.0], size=n, p=[0.94, 0.06])
    ds = [float((pay[rng.permutation(n)] - pay[rng.permutation(n)]).mean())
          for _ in range(200)]
    m = float(np.mean(ds))
    print(f"★ゲート2の自己テスト: 交換可能な2群の対応差200回の平均 {m:+.3f}円"
          f" → **仮説が偽なら0を返す**: {'★OK' if abs(m) < 3 else '⚠NG'}")
    ok &= abs(m) < 3
    print(f"★比較数 {NCMP} → z = {zq(ALPHA/NCMP):.3f}")
    print(f"★ゲート3: 穴A・穴Bの複勝ROI ≥ {GATE3_LINE}%（複勝の払戻率）")
    print(f"★内部対照: 穴Cの複勝ROIが (178)の {KNOWN_C}% を ±{TOL_C}pt で再現")
    print(f"★穴Bの帯: " + " / ".join(BANDS[i][0] for i in BANDS_B) + "（⚠post-hoc）")
    print("★自己テスト: " + ("全部OK" if ok else "⚠NG"))
    return 0 if ok else 1


def main():
    z = zq(ALPHA / NCMP)
    print("(179) ★★★**穴の定義を(174)の形に取り直して2軸をやり直す**")
    print("★経路: 穴の選定に MLモデルの p / 相方・統制は単勝オッズ。⚠**弱い経路**")
    print("⚠**標本は(174)〜(178)と同一＝独立な検証ではない**（判定基準35）")
    print("⚠★**穴Bは post-hoc**（(174)の最良帯 3-6/6-15倍 を見てから選んだ）。"
          "**穴Aを主に読む**\n")

    races = {r["rid"]: r for r in load_races()}
    rows, bad = gate1(list(races.values()))
    print("⚠**ゲート1**: (88)③④を別パーサで再現・許容±3pt")
    for nm, n, roi, known, dd, ok in rows:
        print(f"　{nm:<12}{roi:>7.1f}% vs {known:>5.1f}%　差 {dd:+.1f}pt"
              f"　{'★立った' if ok else '⚠落ちた'}")
    if bad:
        print("\n⚠⚠**ゲート1が落ちた。読まない**。")
        return

    MODEL_DIR, PAR = CAPACITY["l2"]
    d = F.to_model(F.load_files())
    f = F.build_features(d)
    keep = (f["n_prior"] >= 1) & d["odds"].notna() & (d["odds"] > 0)
    d, f = d[keep].reset_index(drop=True), f[keep].reset_index(drop=True)
    y = (d["finish"] <= 3).astype(int).to_numpy()
    fx, _ = F.encode_categoricals(f)
    fx = add_odds_features(fx, d["odds"].to_numpy(float), d["raceid"].to_numpy())
    cut = d["date"].quantile(0.3)
    tr, te = (d["date"] < cut).to_numpy(), (d["date"] >= cut).to_numpy()
    print(f"\n学習 {tr.sum():,} / 検証 {te.sum():,}・分割 {cut.date()}（3シード平均）")
    ms = fit_seeds(fx[tr], y[tr], 3, PAR)
    p = np.mean([m.predict_proba(fx[te])[:, 1] for m in ms], axis=0)
    sub = d.loc[te, ["raceid", "umaban", "odds", "date"]].copy()
    sub["p"] = p

    # ── 第1パス: レースごとの share/mk/gap/ratio を作り、帯内パーセンタイルを全体で計算 ──
    per, allgap, allband = {}, [], []
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
        share = pv / pv.sum()
        mk = (1.0 / od) / (1.0 / od).sum()
        gap = share - mk
        bi = np.array([band_of(float(o), BANDS) for o in od])
        per[str(rid)] = {"ub": ub, "od": od, "pv": pv, "gap": gap,
                         "ratio": share / np.maximum(mk, 1e-12), "bi": bi,
                         "date": gg["date"].iloc[0], "n0": len(allgap)}
        allgap.append(gap); allband.append(bi)
    allgap = np.concatenate(allgap); allband = np.concatenate(allband)
    pct_all = band_pct(allgap, allband, len(BANDS))
    print(f"★第1パス {len(per):,}レース / {len(allgap):,}頭　⚠**(174)〜(178)と同一標本**")

    # ── 第2パス: 穴A/B/C を決め、買い目の払戻を集める ──
    rng = np.random.default_rng(SEED)
    recs, box4 = [], []
    for rid, v in per.items():
        r = races[rid]
        ub, od, pv, bi = v["ub"], v["od"], v["pv"], v["bi"]
        pct = pct_all[v["n0"]:v["n0"] + len(ub)]
        mo = [int(u) for u in ub[np.argsort(-pv, kind="mergesort")]]
        pop1 = int(ub[int(np.argmin(od))])
        pop2 = int(ub[int(np.argsort(od)[1])])
        mB = np.isin(bi, BANDS_B)
        if not mB.any():
            continue
        anas = {"★穴A": int(ub[int(np.argmax(pct))]),
                "★穴B": int(ub[np.where(mB)[0][int(np.argmax(pct[mB]))]]),
                "穴C": int(ub[int(np.argmax(v["ratio"]))])}
        if any(a == pop1 for a in anas.values()):
            continue
        rans = {}
        okr = True
        for k, a in anas.items():
            b = bi[list(ub).index(a)]
            cand = [int(u) for u, bb in zip(ub, bi) if bb == b and int(u) not in (a, pop1)]
            if not cand:
                okr = False
                break
            rans[k] = int(rng.choice(cand))
        if not okr:
            continue
        top4 = sorted(mo[:4])
        bx = [payoff(r, "三連複", list(c)) for c in combinations(top4, 3)]
        if any(x is None for x in bx):
            continue
        rec = {"date": v["date"], "fuku": {}, "pay": {},
               "axod": {k: float(od[list(ub).index(a)]) for k, a in anas.items()}}
        # 軸セット: 穴X×人 と 乱X×人（Xごと）＋ モ2・人2
        sets = {}
        for k in ANAS:
            sets[k] = (anas[k], pop1)
            sets["乱" + k[-1]] = (rans[k], pop1)
        sets["モ2"] = (mo[0], mo[1]); sets["人2"] = (pop1, pop2)
        bad_rec = False
        for who, (a, _b) in sets.items():
            vv = payoff(r, "複勝", [a])
            if vv is None:
                bad_rec = True
                break
            rec["fuku"][who] = vv
        if bad_rec:
            continue
        rec["fuku"]["モ1"] = payoff(r, "複勝", [mo[0]])
        if rec["fuku"]["モ1"] is None:
            continue
        for snm, kind, pts, fn in SHAPES + DESC:
            for who, (a, bb) in sets.items():
                cs = fn(a, bb, mo)
                if len(cs) != pts:
                    bad_rec = True
                    break
                tot = 0.0
                for c in cs:
                    vv = payoff(r, kind, list(c))
                    if vv is None:
                        bad_rec = True
                        break
                    tot += vv
                if bad_rec:
                    break
                rec["pay"][(snm, who)] = tot - COST * pts
            if bad_rec:
                break
        if bad_rec:
            continue
        box4.append(sum(bx) - 400.0)
        recs.append(rec)

    N = len(recs)
    box4 = np.asarray(box4, float)
    print(f"★突き合わせ {N:,}レース")
    g0 = 100.0 * (box4.sum() + 400.0 * N) / (400.0 * N)
    okb = abs(g0 - KNOWN_BOX4) <= TOL_BOX4
    print(f"⚠**陽性対照**: 三連複BOX上位4 **{g0:.1f}%** vs 既知 {KNOWN_BOX4}%"
          f" → **{'★立った' if okb else '⚠⚠落ちた'}**")
    if not okb:
        print("\n⚠⚠**陽性対照が落ちた。読まない**（判定基準32）。")
        return

    fk = {w: np.array([r["fuku"][w] for r in recs]) for w in recs[0]["fuku"]}
    print(f"\n■ ★★ゲート3: **穴の定義は直ったか**（複勝1点・{N:,}レース）")
    print(f"{'軸':<10}{'複勝ROI':>10}{'平均オッズ':>12}{'判定':>26}")
    for k in ANAS:
        ao = np.mean([r["axod"][k] for r in recs])
        if k == "穴C":
            ok = abs(roi_of(fk[k]) - KNOWN_C) <= TOL_C
            v = f"内部対照 vs (178){KNOWN_C}% → {'★再現' if ok else '⚠⚠ズレた'}"
        else:
            ok = roi_of(fk[k]) >= GATE3_LINE
            v = f"要 ≥{GATE3_LINE}% → {'★通った' if ok else '⚠⚠落ちた'}"
        print(f"{k:<10}{roi_of(fk[k]):>9.1f}%{ao:>11.1f}倍{v:>26}")
    for k in ("モ1", "人2", "乱A", "乱B", "乱C"):
        print(f"{'  '+k:<10}{roi_of(fk[k]):>9.1f}%")
    okC = abs(roi_of(fk["穴C"]) - KNOWN_C) <= TOL_C
    okAB = roi_of(fk["★穴A"]) >= GATE3_LINE and roi_of(fk["★穴B"]) >= GATE3_LINE
    if not okC:
        print("\n⚠⚠**内部対照が落ちた（(178)の穴Cを再現しない）。装置を疑う。読まない**。")
        return
    if not okAB:
        print(f"\n★★**ゲート3が落ちた: 穴の定義を取り直しても複勝の払戻率{GATE3_LINE}%に届かない**。")
        print("→ ★**2軸は読まない**（事前登録どおり）。")
        print("★**書けるのは「穴の定義を変えても直らない」まで**。")
        print("⚠**経路は弱いので「閉じた」とは書けない**（判定基準25）。")
        return

    print(f"\n{'='*100}")
    print("■ 記述: **買い方 × 軸セット**（ROI%）"
          "　⚠★**ここから最良のマスを選んで結論にしない**（(178)でプラセボが109.9%）")
    store = {}
    cols = ["★穴A", "乱A", "★穴B", "乱B", "穴C", "乱C", "モ2", "人2"]
    print(f"{'買い方':<22}{'点数':>5}" + "".join(f"{c:>10}" for c in cols))
    for snm, kind, pts, _fn in SHAPES + DESC:
        line = ""
        for who in cols:
            pr = np.array([r["pay"][(snm, who)] for r in recs])
            store[(snm, who)] = pr
            line += f"{100.0*(pr.mean()+COST*pts)/(COST*pts):>9.1f}%"
        print(f"{snm:<22}{pts:>5}{line}")

    print(f"\n■ ★★★主判定: **「穴×人」−「乱×人」の対応差**"
          f"（**{NCMP}比較・Bonferroni α={ALPHA}/{NCMP}・z={z:.3f}**）")
    print("　★ゲート2: **仮説が偽なら穴と乱は同じ帯で交換可能＝対応差の期待値は0**")
    print(f"\n{'穴の定義':<10}{'買い方':<22}{'対応差':>10}{'99%CI(Bonf)':>22}{'判定':>16}")
    hit = []
    for k in ("★穴A", "★穴B"):
        for snm, kind, pts, _fn in SHAPES:
            dd = store[(snm, k)] - store[(snm, "乱" + k[-1])]
            mu, se = dd.mean(), dd.std(ddof=1) / math.sqrt(N)
            sig = abs(mu) > z * se
            if sig and mu > 0:
                hit.append((k, snm, pts))
            print(f"{k:<10}{snm:<22}{mu:>+9.1f}円"
                  f"{f'[{mu-z*se:+.1f},{mu+z*se:+.1f}]':>22}"
                  f"{'★差がある' if sig else '⚠検出できない':>16}")

    print(f"\n■ 記述: **人気順対照（人2）との差**（採用条件4）")
    for k in ("★穴A", "★穴B"):
        for snm, kind, pts, _fn in SHAPES:
            print(f"{k:<10}{snm:<22}"
                  f"{(store[(snm,k)]-store[(snm,'人2')]).mean():>+13.1f}円")

    if not hit:
        print("\n★★**結論: 主判定を通るマスが無い＝陰性**。")
        print("★**穴の定義を(174)の形に直しても、2軸は無作為な穴に勝てない**。")
        print("⚠**経路は弱いので「閉じた」とは書けない**（判定基準25）。")
        return
    dates = np.array([r["date"] for r in recs], dtype="datetime64[D]")
    years = dates.astype("datetime64[Y]").astype(int) + 1970
    med = np.median(dates.astype(int))
    print(f"\n■ ★裾の検算（**主判定を通った {len(hit)} マス**）")
    for k, snm, pts in hit:
        pr = store[(snm, k)]
        c = COST * pts
        gain = pr + c
        ys = sorted(set(years))
        over = sum(1 for u in ys if 100.0*(pr[years == u].mean()+c)/c > 100.0)
        dt = dates.astype(int)
        print(f"　{k} / {snm}: ROI {100.0*(pr.mean()+c)/c:.1f}% / "
              f"**上位3本が全払戻の "
              f"{100*np.sort(gain)[-3:].sum()/max(gain.sum(),1e-9):.1f}%** / "
              f"前半 {100.0*(pr[dt<=med].mean()+c)/c:.1f}%・"
              f"後半 {100.0*(pr[dt>med].mean()+c)/c:.1f}% / "
              f"**100%超の年 {over}/{len(ys)}**")
    print("\n⚠**ROIが100%を超えなければ運用の候補にならない**（事前登録）。")
    print("⚠**枠連の運用には触れない**。**設定変更は提案しない**。")


if __name__ == "__main__":
    sys.exit(selftest() if "--selftest" in sys.argv else (main() or 0))
