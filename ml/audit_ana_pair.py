"""(178) ★★★**「穴馬 × 1番人気」の2軸** — 買い方そのものを、探索を守りながら測る

★★問い（ユーザー・2026-09-05）——
　「**BOXではなくて穴馬と1番人気の2軸とかの可能性もある。買い目はBOXだけじゃない**」
　「**買い方自体も研究してほしい。最適な買い方を見つけるところも考慮して**」

★★★**先に2つの訂正を書く（私の引用が甘かった）**——
　1. ⚠**「点数を増やすほど悪化」の根拠3件は、どれも「全レース」だった**。
　　 **(77) `menu_wide.py` / (171) `audit_gap_axis.py` / (172) `audit_elbow_kinds.py`
　　 のいずれにも `waku_score`・除外・裾の選別は入っていない**（grepで0件）。
　　 → ★**「選んだレースでも点数を増やすと悪化するか」は測っていない**。**本件で並べて検算する**。
　　 ⚠**さらに切り分け**: **「点数↑で1R期待損失(円)↑」はROI<100%ならほぼ自動**（(170)の形）。
　　 **「点数↑でROI自体が下がる」(84.1→79.2→77.7%)のほうが経験的な主張**で、**そちらが未検証**。
　2. ⚠**(77)の「三連複 軸2頭×紐3」の軸2頭は `t[0], t[1]`＝モデル上位2頭**であって、
　　 **「穴馬 + 1番人気」ではない**（`menu_wide.py:72`）。→ ★**ご指摘の形は本当に未測定**。

★★**なぜ2軸に見込みがあるか（機構・ただし類推なので当てにしない・判定基準24）**
　**(171)は差最大の馬を「軸1つ」にして広く流した**ので、**点数が6〜15点に膨らみ
　(77)の逆風を全部かぶった**（軸1×紐6は15点・1R −334円）。
　★**2軸なら点数は紐の数そのもの**（**馬連・ワイドなら1点**）＝**(171)より点数が少ない**。
　★**穴馬の弱点は「来る確率が低い」こと**で、**相方を1番人気に固定すれば外れる要因が1つ減る**。
　⚠**ただし類推である**。**判定基準24: 当てにしてよいのは恒等式＋実測の上界だけ**。

────────────────────────────────────────────────────────────
★★★ 事前登録（2026-09-05・**結果を見る前にコミットする**）
────────────────────────────────────────────────────────────

■ ★経路（判定基準25）: 穴の選定に MLモデルの p を使う ／ 相方・統制は単勝オッズ。
　⚠**弱い経路**。**陰性でも「閉じた」とは書けない**。
　⚠**標本は(174)〜(177)と同一**＝**独立な検証ではない**（判定基準35）。

■ ★★軸の定義（**4通り。これ以外に増やさない**）
　**穴** … そのレースで `ratio = share/mk` が最大の馬（**(171)の「差最大」の比版**・1頭）
　**人** … 1番人気（単勝オッズ最小）
　| 軸セット | 中身 | 役割 |
　|---|---|---|
　| ★**穴×人** | 穴 + 1番人気 | ★**本命の仮説** |
　| ★**乱×人** | ★**穴と同じオッズ帯からランダムに1頭** + 1番人気 | ★★**主判定の対照** |
　| モ2 | モデル1位 + モデル2位 | **(77)の形**（記述） |
　| 人2 | 1番人気 + 2番人気 | **人気順対照**（記述・(69)で必須） |
　★★**「乱」がプラセボの本体**（判定基準7）——**オッズ帯を揃えた無作為な穴**なので、
　　**「穴と人気で買う」という形そのものの効果を打ち消し、モデルの穴の選定だけを残す**。
　　⚠**(79)①「中身の無い順序でも+1.4pt出る」への対処**。**人気順との差だけでは技能を示せない**。

■ ★★買い方（**5通り。★点数の少ない側だけ**。**BOXも広い流しも入れない**）
　| # | 券種 | 買い目 | 点数 | 払戻率 | 100%に要る差 |
　|---|---|---|---|---|---|
　| 1 | **馬連** | 軸2頭 | **1** | 0.775 | +22.5pt |
　| 2 | **ワイド** | 軸2頭 | **1** | 0.775 | +22.5pt |
　| 3 | **馬単** | 軸2頭の両方向 | **2** | 0.750 | +25.0pt |
　| 4 | **三連複** | 軸2頭 + モデル最上位1頭 | **1** | 0.750 | +25.0pt |
　| 5 | **三連複** | 軸2頭 + モデル上位3頭 | **3** | 0.750 | +25.0pt |
　★**橋渡し**: **複勝1点（穴 / 乱 / モデル1位 / 1番人気）**も出す＝**(174)〜(177)と繋がる**。
　⚠**(171)が測った「軸1×紐4/紐6」は入れない**——**既に陰性で、点数が多い側だから**。

■ ★★レース選別（**2水準。★ここがご指摘への回答**）
　**全レース** ／ ★**(112)裾2%**（`soft_axis.axis_expect` の軸の複勝E ≤ 86円・**モデル不使用**）
　→ ★**同じ表の中で「点数を増やすと悪化するか」を選別あり/なしで検算できる**。

■ ★★★主判定（**10比較・Bonferroni α=0.01/10・z=3.291**）
　★**「穴×人」−「乱×人」の対応差（円）**を、**5つの買い方 × 2つの選別**で。
　⚠**モ2・人2との比較は記述**（**増やすと補正が重くなるだけで、問いに答えない**）。

■ ★★ゲート2（判定基準42）——**仮説が偽なら何を返すか**
　★**主判定は「穴×人」と「乱×人」の対応差（同じレース・同じ点数・同じ券種）**。
　★**仮説が偽（モデルの穴の選定が何も持たない）なら、穴と乱は同じオッズ帯からの
　　交換可能な2頭になり、対応差の期待値は 0 を返す**。
　・**点数もコストも両者で完全に同一**＝**(170)の「頭数が減れば必ず良くなる量」ではない**。
　・**両方が買うレースだけに絞らない**（**両者とも全対象レースで買う**）＝**(168)の形にならない**。

■ ⚠ゲート1（判定基準32）—— **(174)〜(177)と同じ**（(88)③④を別パーサで再現・±3pt）
　★**加えて陽性対照**: **三連複BOX上位4（全レース）が既知84.5%を±2.5ptで再現**すること。
　　（**(171)は84.1%で立った**。**買い方の装置が動いていることの確認**）

■ ★採用条件（判定基準39/40/41）
　1. **主判定がBonferroniを通る**
　2. ★**人2（人気順対照）にも同符号で勝つ**（**(69)で本命に抜けが見つかった形**）
　3. ★**全レースと裾2%の両方で同じ向き**（**片方だけなら選別との交互作用を疑う**）
　4. ★**券種をまたいで同じ向き**（**1マスだけ跳ねるのは(172)で崩れた形**）
　5. **裾の検算で符号が反転しない**（**上位3本・前後半・年別**）
　6. ★**ROI>100% でなければ「機構は在るが張れない」**（(174)〜(177)と同じ分岐）

■ ★★「最適な買い方を探す」ことの扱い（**ここを守らないと必ず偽陽性が出る**）
　⚠**(172)は24マス並べて跳ねたマス(92.0%)を追い、上位3本を除くと82.1%・前後半で17pt割れた**。
　⚠**判定基準4: 総当たり6,206セルでは偶然だけで34.4セルが「両期間100%超」を通過する**。
　→ ★★**本件は「最良のマスを探す」ことを主判定にしない**。**主判定は対照との対応差だけ**。
　　 ★**全マスのROIは記述として出すが、そこから「最良」を選んで結論にしない**。
　　 ★**「最適な買い方」に答えられるのは、採用条件を全部通ったマスがあった場合だけ**。
　⚠**判定基準8**: **点数の違う買い方どうしは1R期待損失(円)で並べる**。**ROIで並べない**。

■ 予想
　**持たない**。⚠**逆風は明示する**——**三連複・馬単の払戻率は75.0%で、100%に要るのは+25.0pt**。
　**(174)の最良は+7.6pt、(177)の中和後は+1.5pt**。★**2軸で穴の当たりやすさが上がっても、
　　払戻率が5pt低いところから始まる**。

実行: python3 ml/audit_ana_pair.py     自己テスト: python3 ml/audit_ana_pair.py --selftest
"""
import math
import sys
from itertools import combinations, permutations

import numpy as np

sys.path.insert(0, "ml")
import features as F
import soft_axis as SA
from audit_crosspool import LINE, load_races, payoff, zq
from audit_ana_odds import BANDS, COST, MIN_HORSES, band_of, gate1, roi_of
from train_prod import CAPACITY, add_odds_features, fit_seeds

E_112 = 86.0                     # (112) 軸の複勝E の閾値[円]（裾2%）
NCMP = 10                        # 5つの買い方 × 2つの選別
ALPHA = 0.01
SEED = 20260905
KNOWN_BOX4, TOL_BOX4 = 84.5, 2.5

# 買い方: (名前, 券種, 点数, 買い目を作る関数(ax_a, ax_b, model_order))
def _umaren(a, b, mo):   return [(a, b)]
def _wide(a, b, mo):     return [(a, b)]
def _umatan(a, b, mo):   return [(a, b), (b, a)]
def _puku1(a, b, mo):
    h = [u for u in mo if u not in (a, b)][:1]
    return [tuple(sorted((a, b, h[0])))] if h else []
def _puku3(a, b, mo):
    hs = [u for u in mo if u not in (a, b)][:3]
    return [tuple(sorted((a, b, h))) for h in hs]

SHAPES = [("馬連 軸2頭", "馬連", 1, _umaren),
          ("ワイド 軸2頭", "ワイド", 1, _wide),
          ("馬単 軸2頭 両方向", "馬単", 2, _umatan),
          ("三連複 軸2頭+モ1", "三連複", 1, _puku1),
          ("三連複 軸2頭+モ3", "三連複", 3, _puku3)]
AXSETS = ["★穴×人", "乱×人", "モ2", "人2"]
SELS = [("全レース", None), ("★(112)裾2%", E_112)]


def selftest():
    ok = True
    mo = [5, 3, 9, 1, 7]
    assert _umaren(5, 1, mo) == [(5, 1)]
    assert _umatan(5, 1, mo) == [(5, 1), (1, 5)]
    assert _puku1(5, 1, mo) == [(3, 5, 1)] or _puku1(5, 1, mo) == [tuple(sorted((5, 1, 3)))]
    assert _puku3(5, 1, mo) == [tuple(sorted((5, 1, h))) for h in (3, 9, 7)]
    assert len(_puku3(5, 1, mo)) == 3
    print("★買い目生成の自己テスト: 馬連1点 / 馬単2点 / 三連複1点・3点　★OK")
    # ★ゲート2: 穴と乱が交換可能なら対応差は0
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
    for nm, kind, pts, _fn in SHAPES:
        print(f"　{nm:<22}{kind:<6}{pts}点　払戻率 {LINE[kind]:.3f}"
              f"　100%に要る差 {100*(1-LINE[kind]):+.1f}pt")
    print("★自己テスト: " + ("全部OK" if ok else "⚠NG"))
    return 0 if ok else 1


def main():
    z = zq(ALPHA / NCMP)
    print("(178) ★★★**「穴馬 × 1番人気」の2軸** — 買い方そのものを、探索を守りながら測る")
    print("★経路: 穴の選定に MLモデルの p / 相方・統制は単勝オッズ。⚠**弱い経路**")
    print("⚠**標本は(174)〜(177)と同一＝独立な検証ではない**（判定基準35）")
    print("⚠★**主判定は「最良のマス探し」ではなく、オッズ帯を揃えた無作為な穴との対応差**\n")

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

    rng = np.random.default_rng(SEED)
    recs = []
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
        share = pv / pv.sum()
        mk = (1.0 / od) / (1.0 / od).sum()
        ratio = share / np.maximum(mk, 1e-12)
        ana = int(ub[int(np.argmax(ratio))])
        pop1 = int(ub[int(np.argmin(od))])
        pop2 = int(ub[int(np.argsort(od)[1])])
        mo = [int(u) for u in ub[np.argsort(-pv, kind="mergesort")]]
        if ana == pop1:
            continue                       # 穴＝1番人気なら2軸にならない
        bi = band_of(float(od[list(ub).index(ana)]), BANDS)
        cand = [int(u) for u, o in zip(ub, od)
                if band_of(float(o), BANDS) == bi and int(u) not in (ana, pop1)]
        if not cand:
            continue
        ran = int(rng.choice(cand))        # ★オッズ帯を揃えた無作為な穴
        _k, e_axis, _q = SA.axis_expect([float(o) for o in od])
        if e_axis is None:
            continue
        top4 = sorted(mo[:4])
        b = [payoff(r, "三連複", list(c)) for c in combinations(top4, 3)]
        if any(v is None for v in b):
            continue
        box4.append(sum(b) - 400.0)
        rec = {"e": float(e_axis), "date": gg["date"].iloc[0], "mo": mo,
               "ax": {"★穴×人": (ana, pop1), "乱×人": (ran, pop1),
                      "モ2": (mo[0], mo[1]), "人2": (pop1, pop2)},
               "fuku": {}, "pay": {}}
        bad_rec = False
        for who, u in (("★穴×人", ana), ("乱×人", ran), ("モ2", mo[0]), ("人2", pop1)):
            v = payoff(r, "複勝", [u])
            if v is None:
                bad_rec = True
                break
            rec["fuku"][who] = v
        if bad_rec:
            continue
        for snm, kind, pts, fn in SHAPES:
            for who in AXSETS:
                a, bb = rec["ax"][who]
                combos = fn(a, bb, mo)
                if len(combos) != pts:
                    bad_rec = True
                    break
                tot = 0.0
                for c in combos:
                    v = payoff(r, kind, list(c))
                    if v is None:
                        bad_rec = True
                        break
                    tot += v
                if bad_rec:
                    break
                rec["pay"][(snm, who)] = tot - COST * pts
            if bad_rec:
                break
        if bad_rec:
            continue
        recs.append(rec)

    N = len(recs)
    box4 = np.asarray(box4[:N], float)
    print(f"★突き合わせ {N:,}レース　⚠**(174)〜(177)と同一標本**")
    g0 = 100.0 * (box4.sum() + 400.0 * len(box4)) / (400.0 * len(box4))
    okb = abs(g0 - KNOWN_BOX4) <= TOL_BOX4
    print(f"⚠**陽性対照**: 三連複BOX上位4（全レース）**{g0:.1f}%** vs 既知 {KNOWN_BOX4}%"
          f"　差 {g0-KNOWN_BOX4:+.1f}pt → **{'★立った' if okb else '⚠⚠落ちた'}**")
    if not okb:
        print("\n⚠⚠**陽性対照が落ちた。買い方の装置を信用できない。読まない**（判定基準32）。")
        return

    E = np.array([r["e"] for r in recs])
    dates = np.array([r["date"] for r in recs], dtype="datetime64[D]")
    years = dates.astype("datetime64[Y]").astype(int) + 1970
    med = np.median(dates.astype(int))

    print(f"\n{'='*104}")
    print("■ 記述: **買い方 × 軸セット × レース選別**（ROI% ／ 下段は1R損益[円]）")
    print("　⚠★**ここから「最良のマス」を選んで結論にしない**（判定基準4・(172)）")
    store = {}
    for sname, thr in SELS:
        m = np.ones(N, bool) if thr is None else (E <= thr)
        print(f"\n── {sname}（{m.sum():,}レース・{100*m.mean():.1f}%）──")
        print(f"{'買い方':<22}{'点数':>5}" + "".join(f"{w:>13}" for w in AXSETS))
        for snm, kind, pts, _fn in SHAPES:
            line1, line2 = "", ""
            for who in AXSETS:
                pr = np.array([r["pay"][(snm, who)] for r in recs])[m]
                c = COST * pts
                store[(sname, snm, who)] = pr
                line1 += f"{100.0*(pr.mean()+c)/c:>12.1f}%"
                line2 += f"{pr.mean():>+12.1f}円"
            print(f"{snm:<22}{pts:>5}{line1}")
            print(f"{'':<27}{line2}")
        fk = {w: np.array([r["fuku"][w] for r in recs])[m] for w in AXSETS}
        print(f"{'[橋渡し] 複勝1点':<22}{1:>5}"
              + "".join(f"{roi_of(fk[w]):>12.1f}%" for w in AXSETS))
        for w in AXSETS:
            store[(sname, "複勝1点", w)] = fk[w] - COST

    print(f"\n{'='*104}")
    print(f"■ ★★★主判定: **「★穴×人」−「乱×人」の対応差**"
          f"（**{NCMP}比較・Bonferroni α={ALPHA}/{NCMP}・z={z:.3f}**）")
    print("　★ゲート2: **仮説が偽なら穴と乱は交換可能＝対応差の期待値は0**")
    print(f"\n{'選別':<12}{'買い方':<22}{'対応差':>10}{'99%CI(Bonf)':>22}{'判定':>16}")
    hit = []
    for sname, thr in SELS:
        for snm, kind, pts, _fn in SHAPES:
            dd = store[(sname, snm, "★穴×人")] - store[(sname, snm, "乱×人")]
            mu, se = dd.mean(), dd.std(ddof=1) / math.sqrt(len(dd))
            sig = abs(mu) > z * se
            if sig and mu > 0:
                hit.append((sname, snm))
            print(f"{sname:<12}{snm:<22}{mu:>+9.1f}円"
                  f"{f'[{mu-z*se:+.1f},{mu+z*se:+.1f}]':>22}"
                  f"{'★差がある' if sig else '⚠検出できない':>16}")

    print(f"\n■ 記述: **人気順対照（人2）との差**（**採用条件2**・有意性は見ない）")
    print(f"{'選別':<12}{'買い方':<22}{'穴×人 − 人2':>14}")
    for sname, thr in SELS:
        for snm, kind, pts, _fn in SHAPES:
            dd = (store[(sname, snm, "★穴×人")] - store[(sname, snm, "人2")]).mean()
            print(f"{sname:<12}{snm:<22}{dd:>+13.1f}円")

    print(f"\n■ ★★★ご指摘への回答: **「点数を増やすと悪化」は選別後でも成り立つか**")
    print("　⚠**(77)(171)(172)はどれも全レースだった**（grepで確認）。**ここで並べる**")
    print(f"{'選別':<12}{'買い方(穴×人)':<22}{'点数':>5}{'ROI':>9}{'1R損益':>11}")
    for sname, thr in SELS:
        for snm, kind, pts, _fn in SHAPES:
            if kind != "三連複":
                continue
            pr = store[(sname, snm, "★穴×人")]
            c = COST * pts
            print(f"{sname:<12}{snm:<22}{pts:>5}{100.0*(pr.mean()+c)/c:>8.1f}%"
                  f"{pr.mean():>+10.1f}円")
    print("　★**同じ軸・同じ券種で点数だけ 1→3 に増やした比較**。"
          "**ROIが下がるか／損失が増えるかを選別あり/なしで見る**")

    if not hit:
        print(f"\n★★**結論: 主判定を通るマスが無い＝陰性**。")
        print("⚠**経路は弱いので「閉じた」とは書けない**（判定基準25）。")
        print("★**書けるのは「モデルの穴の選定は、オッズ帯を揃えた無作為な穴に勝てない」まで**。")
    else:
        print(f"\n■ ★裾の検算（**主判定を通った {len(hit)} マス**・(77)）")
        for sname, snm in hit:
            pr = store[(sname, snm, "★穴×人")]
            m = np.ones(N, bool) if SELS[[s[0] for s in SELS].index(sname)][1] is None \
                else (E <= E_112)
            pts = [s[2] for s in SHAPES if s[0] == snm]
            c = COST * (pts[0] if pts else 1)
            gain = pr + c
            dt, yr = dates[m].astype(int), years[m]
            ys = sorted(set(yr))
            over = sum(1 for u in ys
                       if 100.0*(pr[yr == u].mean()+c)/c > 100.0)
            print(f"　{sname} / {snm}: ROI {100.0*(pr.mean()+c)/c:.1f}% / "
                  f"**上位3本が全払戻の "
                  f"{100*np.sort(gain)[-3:].sum()/max(gain.sum(),1e-9):.1f}%** / "
                  f"前半 {100.0*(pr[dt<=med].mean()+c)/c:.1f}%・"
                  f"後半 {100.0*(pr[dt>med].mean()+c)/c:.1f}% / "
                  f"**100%超の年 {over}/{len(ys)}**")
        print("\n⚠**ROIが100%を超えなければ運用の候補にならない**（事前登録）。")
    print("⚠**枠連の運用には触れない**。**設定変更は提案しない**。")


if __name__ == "__main__":
    sys.exit(selftest() if "--selftest" in sys.argv else (main() or 0))
