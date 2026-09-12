"""(194) ★★★**紐の選び方を正面から測る** ＋ **「100%超の確率」にプラセボの基準線を与える**

★★**動機（2026-09-06・ユーザーの指摘2つ）**:
　★①**「相手（紐）も同じ基準で選ぶ、ここが雑だと思う」**——**その通り**。
　　**軸と紐は役割が違う**（**軸は「来ること」、紐は「軸と一緒に来ること」**）。
　　⚠**同じ物差し（ズレ）を使う理由はどこにもない**。**(193)まではズレ順 vs p降順の2択しか見ていない**。
　★②**「106.8%が100%超である確率72.5%は許せる」**——★**決めるのは利用者**。
　　⚠**だが判定基準23**: **その72.5%が「本物の腕」でしか出ない数字なのかを確かめていない**。
　　★★**プラセボ（乱）に同じブートストラップを当てて、同じ72%が出るなら、その数字は無意味**。

────────────────────────────────────────────────────────────
★★★ 事前登録（2026-09-06・**結果を見る前にコミットする**）
────────────────────────────────────────────────────────────

■ ★経路（判定基準25）: **q = MLモデル / q_pool = 複勝の板**（調和平均）。⚠**q側は弱い**。
■ ★手続き: **ウォークフォワード**（`data/cache/wf_pred.npz`）。**母集団は(193)と同一**。
　**軸 = (推奨度≥A かつ ズレ≥B) を満たす馬のうち gap 最大**、**馬連 紐2頭（2点）**。
　**絞り (0.15,0.02) / (0.20,0.06)** × **紐の床 C ∈ {0.04, 0.06}**。

■ ★★★家族A（**24比較**）: **紐の選び方6通り**（**すべて「gap≥C を満たす馬の中から」2頭**）
| # | 腕 | 発想 |
|---|---|---|
| 0 | **現行(p降順)** | **基準線**（対応差の相手） |
| 1 | **ズレ(gap降順)** | **(191)〜(193)で使ってきた腕** |
| 2 | ★**床はズレ・順序はモデル(p降順)** | ★**利用者の問い「モデル順じゃなくていいの？」** |
| 3 | ★**比(pn/qp 降順)** | ★**判定基準30（差ではなく比の裾）** |
| 4 | ★**板(複勝確率 qp 降順)** | **市場が最も来ると見ている馬** |
| 5 | ★**人気(オッズ昇順)** | **最も素朴な相手** |
| P | ★★**乱(同じオッズ帯の無作為2頭)** | ★★**プラセボ**（**判定しない・基準線**） |
　★**主判定: 各腕の「現行との対応差（円）」**。**軸もコストも点数も同一**。
　★**6腕のうち現行を除く5腕 × 2絞り × 2床 = 20比較**＋**乱との対応差 4比較 = 24**。

■ ★★★家族B（**判定しない・記述**）: ★**「100%超の確率」にプラセボの基準線**
　**全腕（乱を含む）でブートストラップ10,000回**、**ROIが100%を超える割合を出す**。
　★★**読み方を先に書く**:
| 乱の割合が低い（≲20%） | 乱の割合も高い（≳50%） |
|---|---|
| ★**72.5%には意味がある**。**利用者の判断は成立する** | ⚠⚠**72.5%は偶然でも出る数字**。**判断材料にならない** |
　★★**これが②への答えになる**。**私が決めるのではなく、プラセボに決めさせる**。

■ ★★ゲート2（判定基準42）——**仮説が偽なら何を返すか**
　★**家族A**: **紐の選び方が損益について何も持たないなら、対応差の期待値は厳密に0**。
　★**水準**: **板が正しければ馬連ROIは払戻率77.5%を返す**（100%ではない）。

■ ⚠ゲート1: (88)③④を再現（±3pt）／ ★ゲート板: 復元R 0.800±0.020 ／
　★陽性対照: 三連複BOX上位4 81.9%±1.5pt。
■ ★★内部対照: **腕1（ズレ順）が(193)の 102.2 / 104.8 / 103.7 / 106.8% を±1ptで再現**。

■ ★★★探索を守る（**24比較・Bonferroni α=0.01/24**）
　⚠**標本300レース未満のマスは判定しない**（判定基準5）。
　⚠**(193)の追試で、床0.06の薄いマスは乱が109.6%を出した**。**そこは特に慎重に読む**。
　★**採用条件に「隣接と同じ向き」**——**孤立した1マスは採らない**。

■ ★採用条件
　1. **ある腕の対応差が、2絞り×2床のうち複数マスでBonferroniを通る**
　2. **その腕が乱（プラセボ）に対しても正の差を持つ**
　3. **家族Bで乱の「100%超の割合」が低い**（**でなければ水準の話はできない**）
　4. **裾の検算で符号が反転しない**

■ 予想（⚠**類推なので当てにしない**・判定基準24）
　★**腕2（床はズレ・順序はモデル）が最良と見る**——**床がオッズ妙味を担保し、
　　順序が「来る確率」を担保する**＝**役割分担が噛み合う**。⚠**だが(191)ではp降順が最下位だった**。
　★**家族B: 乱の割合はかなり高いと見る**（**(193)追試で乱が109.6%を出した**）。
　　→ ⚠**その場合「72.5%」は判断材料にならない、と書くことになる**。

────────────────────────────────────────────────────────────
★★★★ 実測（2026-09-06）—— **利用者の指摘2つとも当たっていた。私の説明が2箇所不正確だった**
────────────────────────────────────────────────────────────

■ ゲート1 4帯とも通過 ／ ゲート板 復元R 0.8009 ／ 陽性対照 BOX4 81.9% → 全部立った
■ ★★内部対照: **腕1（ズレ順）が(193)の 102.2 / 104.8 / 103.7 / 106.8% を差0.0ptで再現**

■ ★★★★家族A: **紐の順序はほとんど効かない。効いていたのは「床」だった**
**厚いマス (0.20,0.06)・床0.04・11,827R**（基準線 現行 88.2%）
| 紐の選び方 | ROI | 的中率 | 現行との差 | 乱との差 |
|---|---|---|---|---|
| ★ズレ(gap降順) | **103.7%** | 10.62% | +15.4円 | +21.2円 |
| ★**比(pn/qp降順)** | **103.8%** | 9.66% | +15.6円 | +21.4円 |
| 板(複勝確率降順) | 101.0% | 11.11% | +12.8円 | +18.5円 |
| ★**床ズレ・順序モデル** | 100.8% | 11.11% | +12.5円 | +18.3円 |
| 人気(オッズ昇順) | 100.5% | 11.14% | +12.3円 | +18.0円 |
| ⚠現行(**床なし**・p降順) | ⚠**88.2%** | — | — | — |
| ★★乱(同帯) | 82.5% | 10.20% | −5.7円 | — |
★★★**どの順序でも 100.5〜103.8%**。**差はほとんどない**。
→ ★★**効いていたのは「順序」ではなく「★床（gap≥C を満たす馬の中から選ぶ）」**。
⚠⚠**(191)〜(193)で「ズレ順が効く」と書いたのは、床の効果と混ざっていた**——
　**現行だけ床をかけていなかったので、順序の差ではなく「床の有無」を測っていた**。
　★**利用者の指摘「相手も同じ基準で選ぶ、ここが雑」は正しかった**（**予想と逆の意味で**）。
★**「モデル順じゃなくていいの？」の答え: モデル順でも100.8%出る**。
　★**的中率は 11.11% と高く配当が小さい / ズレ順は 10.62% で配当が大きい**＝**同じ場所に別の道で着く**。
⚠**24比較すべて通らない**（CIが ±22〜40pt）。

■ ★★★★家族B: ★**プラセボの基準線——「72.5%」には意味があった**
| マス | ★ズレ順の「100%超の確率」 | ⚠**乱(プラセボ)の同じ確率** | 乱のROI |
|---|---|---|---|
| (0.20,0.06)・床0.04 | **71.6%** | ★**0.0%** | 82.5% |
| ★**(0.20,0.06)・床0.06** | ★**72.5%** | ★**5.6%** | 86.9% |
| (0.15,0.02)・床0.04 | 62.1% | 29.0% | 96.5% |
| (0.15,0.02)・床0.06 | 65.7% | ⚠⚠**78.1%** | ⚠**109.6%** |
　⚠**現行(p降順)は 0.0 / 2.5 / 0.0 / 5.2%** ＝**基準線として機能している**。
★★★**106.8%のマスでは、プラセボは 5.6% しか出さない**。**72.5%はそこから明確に離れている**。
　→ ★★**利用者の「72.5%は許せる」という判断は成立する**。
⚠⚠**私が前回「プラセボが109.6%を出した」と言ったのは★別のマス（絞りが違う方）**。
　★**106.8%のマス自体はプラセボに +19.9円 勝っている**。⚠**私の説明が不正確だった**。
　★**プラセボが暴れるのは (0.15,0.02)・床0.06 の1マスだけ**（4マス中1マス）。

■ ★★結論（**この測定で2つ訂正が入った**）
　★**訂正1: 「ズレ順が効く」ではない。「床が効く」**——**順序はどれでもほぼ同じ**。
　★**訂正2: 「106.8%はプラセボと区別がつかない」は誤り**。**そのマスのプラセボは5.6%**。
　⚠**残る事実**: **24比較0通過** ／ **100%超の年 5〜6/11** ／ **厚いマスは前半117→後半90**。
　★★**「統計的に通らない」と「儲からない」は別**。⚠**私が言えるのは前者だけ**。
　★**規模感**: (0.20,0.06)・床0.04 は **年約1,075レース＝1レース200円で年21.5万円 → 期待+8,000円**。

■ ★次に測るべきこと（**今回の訂正から自然に出る**）
　★★**「床」を単独で測る**——**順序を固定して床だけを振る**。**それが本体だと分かったから**。
　⚠**(193)家族Aは床を振ったが、順序はズレ順に固定していた**。**逆をやっていない**。


実行: python3 ml/audit_ana_himo.py    自己テスト: python3 ml/audit_ana_himo.py --selftest
"""
import math
import sys
from itertools import combinations

import numpy as np

sys.path.insert(0, "ml")
import features as F
from audit_crosspool import LINE, load_races, payoff, zq
from audit_ana_odds import BANDS, COST, MIN_HORSES, band_of, gate1, roi_of
from audit_ana_marg import WF_BOX4, WF_TOL, wf_predict
from audit_ana_board import BOARD_R, BOARD_TOL, NPLACE, load_fuku_boards, qpool
from train_prod import add_odds_features

FILT = [(0.15, 0.02), (0.20, 0.06)]
CS = [0.04, 0.06]
ARMS = ["現行(p降順)", "★ズレ(gap降順)", "★床ズレ・順序モデル", "★比(pn/qp降順)",
        "★板(複勝確率降順)", "★人気(オッズ昇順)", "★★乱(同帯)"]
BASE, PLACEBO = ARMS[0], ARMS[-1]
KNOWN193 = {((0.15, 0.02), 0.04): 102.2, ((0.15, 0.02), 0.06): 104.8,
            ((0.20, 0.06), 0.04): 103.7, ((0.20, 0.06), 0.06): 106.8}
MINCELL = 300
NCMP = len(FILT) * len(CS) * ((len(ARMS) - 2) + 1)      # 20 + 4
ALPHA = 0.01
SEED = 20260906
NBOOT = 10000
UMAREN_LINE = 100.0 * LINE["馬連"]


def selftest():
    ok = True
    z = zq(ALPHA / NCMP)
    print(f"★家族A {NCMP}比較（**紐の選び方 {len(ARMS)-1}腕 + 乱**・現行との対応差）")
    for i, a in enumerate(ARMS):
        print(f"　{i}. {a}")
    print(f"★合計 {NCMP}比較 → z = {z:.3f}")
    rng = np.random.default_rng(0)
    n = 120_000
    pay = rng.choice([0.0, 300.0, 6000.0], size=n, p=[0.90, 0.07, 0.03])
    m = float(np.mean([(pay[rng.permutation(n)] - pay[rng.permutation(n)]).mean()
                       for _ in range(200)]))
    print(f"★ゲート2（対応差）: 200回の平均 {m:+.3f}円 → **仮説が偽なら0**: "
          f"{'★OK' if abs(m) < 4 else '⚠NG'}")
    ok &= abs(m) < 4
    print(f"★ゲート2（水準）: **板が正しければ馬連ROIは{UMAREN_LINE:.1f}%を返す**")
    # ★ブートストラップの自己テスト: 払戻率どおりの腕なら「100%超の割合」はほぼ0
    v = rng.choice([0.0, 1000.0], size=40_000, p=[0.9225, 0.0775])
    idx = rng.integers(0, len(v), size=(2000, len(v)))
    b = 100.0 * v[idx].mean(axis=1) / COST
    print(f"★ブートストラップの検算: 払戻率77.5%の腕の「100%超の割合」= "
          f"**{100*np.mean(b>100):.1f}%** → {'★OK' if np.mean(b>100) < 0.01 else '⚠NG'}")
    ok &= np.mean(b > 100) < 0.01
    print("★★家族Bの読み方: **乱の割合も高ければ「72.5%」は判断材料にならない**")
    print("★自己テスト: " + ("全部OK" if ok else "⚠NG"))
    return 0 if ok else 1


def main():
    z = zq(ALPHA / NCMP)
    print("(194) ★★★**紐の選び方を正面から測る**")
    print("★①**「相手も同じ基準で選ぶ、ここが雑」**——**6腕を並べる**")
    print("★②**「100%超の確率72.5%は許せる」**——★**プラセボに同じ計算を当てて決める**\n")

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

    rng = np.random.default_rng(SEED)
    K = {(fl, c): {a: [] for a in ARMS} for fl in FILT for c in CS}
    for k in K:
        K[k]["yr"], K[k]["dt"] = [], []
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
        gapf = pn - qp
        ratio = pn / np.maximum(qp, 1e-9)
        bx = [payoff(r, "三連複", list(c))
              for c in combinations(sorted(int(u) for u in ub[np.argsort(-pv)[:4]]), 3)]
        if not any(x is None for x in bx):
            box4.append(sum(bx) - 400.0)
        bi = np.array([band_of(float(o), BANDS) for o in od])
        yr = int(gg["date"].iloc[0].year)
        # ★腕ごとの「並べ替えのキー」（★すべて gap≥C の中から2頭を取る）
        keys = {"現行(p降順)": -pv, "★ズレ(gap降順)": -gapf, "★床ズレ・順序モデル": -pv,
                "★比(pn/qp降順)": -ratio, "★板(複勝確率降順)": -qp, "★人気(オッズ昇順)": od}
        for fl in FILT:
            cand = np.where((pn >= fl[0]) & (gapf >= fl[1]))[0]
            if not len(cand):
                continue
            ai = int(cand[int(np.argmax(gapf[cand]))])
            axu = int(ub[ai])
            for cth in CS:
                # ★★「現行(p降順)」だけは床をかけない＝(193)と同じ基準線
                pool_c = [k for k in range(len(ub)) if int(ub[k]) != axu and gapf[k] >= cth]
                pool_all = [k for k in range(len(ub)) if int(ub[k]) != axu]
                if len(pool_c) < 2 or len(pool_all) < 2:
                    continue
                picks, okall = {}, True
                for a in ARMS[:-1]:
                    pool = pool_all if a == BASE else pool_c
                    ordk = sorted(pool, key=lambda k: (keys[a][k], k))
                    picks[a] = [int(ub[k]) for k in ordk[:2]]
                # ★プラセボ: ズレ順が選んだ2頭と**同じオッズ帯**の無作為2頭
                hr = []
                for u in picks["★ズレ(gap降順)"]:
                    b = bi[list(ub).index(u)]
                    pl = [int(ub[k]) for k in range(len(ub))
                          if bi[k] == b and int(ub[k]) != axu and int(ub[k]) not in hr]
                    if not pl:
                        okall = False
                        break
                    hr.append(int(rng.choice(pl)))
                if not okall:
                    continue
                picks[PLACEBO] = hr
                vals = {}
                for a in ARMS:
                    vv = [payoff(r, "馬連", [axu, u]) for u in picks[a]]
                    if any(x is None for x in vv):
                        okall = False
                        break
                    vals[a] = sum(vv) / 2.0
                if not okall:
                    continue
                c = K[(fl, cth)]
                for a in ARMS:
                    c[a].append(vals[a])
                c["yr"].append(yr); c["dt"].append(gg["date"].iloc[0])
    print(f"\n★対象 **{nall:,}レース**")

    med = float(np.median(Rs))
    okR = abs(med - BOARD_R) <= BOARD_TOL
    print(f"■ ★ゲート板: 復元R = **{med:.4f}** → **{'★立った' if okR else '⚠⚠落ちた'}**")
    box4 = np.asarray(box4, float)
    g0 = 100.0 * (box4.sum() + 400.0 * len(box4)) / (400.0 * len(box4))
    okb = abs(g0 - WF_BOX4) <= WF_TOL
    print(f"■ ⚠**陽性対照**: BOX上位4 **{g0:.1f}%** vs {WF_BOX4}%"
          f" → **{'★立った' if okb else '⚠⚠落ちた'}**")
    if not (okR and okb):
        print("\n⚠⚠**ゲートが落ちた。読まない**（判定基準32）。")
        return

    print(f"\n■ ★★内部対照（**腕1（ズレ順）が(193)を再現するか**）")
    for fl in FILT:
        for cth in CS:
            v = np.asarray(K[(fl, cth)]["★ズレ(gap降順)"], float)
            k = KNOWN193[(fl, cth)]
            print(f"　{str(fl):<16} C={cth:.2f}: {roi_of(v):>6.1f}% vs (193) {k:.1f}%"
                  f"　差 {roi_of(v)-k:+.1f}pt　"
                  f"{'★再現' if abs(roi_of(v)-k) <= 1.0 else '⚠ズレた'}（{len(v):,}R）")

    print(f"\n{'='*116}")
    print(f"■ ★★★家族A: **紐の選び方6腕**（**現行との対応差**・馬連2点・軸は同一）")
    print(f"　★ゲート2: **紐の選び方が何も持たないなら対応差は0** / **板が正しければ{UMAREN_LINE:.1f}%**")
    best = {}
    for fl in FILT:
        for cth in CS:
            c = K[(fl, cth)]
            v0 = np.asarray(c[BASE], float)
            if len(v0) < MINCELL:
                continue
            print(f"\n★絞り{fl}　紐の床{cth:.2f}　**{len(v0):,}R**"
                  f"（**基準線 現行 {roi_of(v0):.1f}%**）")
            print(f"{'腕':<24}{'ROI':>8}{'的中率':>8}{'★現行との差':>12}{'99%CI':>21}"
                  f"{'★乱との差':>12}{'判定':>13}")
            vp = np.asarray(c[PLACEBO], float)
            for a in ARMS[1:]:
                v = np.asarray(c[a], float)
                dd = v - v0
                mu, se = dd.mean(), dd.std(ddof=1) / math.sqrt(len(dd))
                sig = mu - z * se > 0
                dp = v - vp
                mp = dp.mean()
                sp = dp.std(ddof=1) / math.sqrt(len(dp))
                sigp = mp - z * sp > 0
                tag = ("★★両方" if sig and sigp else "★現行のみ" if sig
                       else "★乱のみ" if sigp else "⚠通らない")
                if a == PLACEBO:
                    tag = "（基準線）"
                best.setdefault((fl, cth), []).append((a, roi_of(v), mu, sig))
                print(f"{a:<24}{roi_of(v):>7.1f}%{100*np.mean(v>0):>7.2f}%{mu:>+11.1f}円"
                      f"{f'[{mu-z*se:+.1f},{mu+z*se:+.1f}]':>21}"
                      f"{('—' if a==PLACEBO else f'{mp:+.1f}円'):>12}{tag:>13}")

    print(f"\n{'='*116}")
    print(f"■ ★★★家族B（**記述**）: ★**「100%超の確率」にプラセボの基準線を与える**")
    print(f"　★★読み方: **乱も高ければ、その確率は偶然でも出る＝判断材料にならない**")
    for fl in FILT:
        for cth in CS:
            c = K[(fl, cth)]
            n = len(c[BASE])
            if n < MINCELL:
                continue
            idx = rng.integers(0, n, size=(NBOOT, n))
            print(f"\n★絞り{fl}　紐の床{cth:.2f}　**{n:,}R**")
            print(f"{'腕':<24}{'ROI':>8}{'ブートストラップ99%区間':>26}{'★100%超の割合':>16}")
            for a in ARMS:
                v = np.asarray(c[a], float)
                b = 100.0 * v[idx].mean(axis=1) / COST
                print(f"{a:<24}{roi_of(v):>7.1f}%"
                      f"{f'[{np.percentile(b,0.5):.1f}, {np.percentile(b,99.5):.1f}]':>26}"
                      f"{100*np.mean(b>100):>15.1f}%")

    print("\n⚠**枠連の運用には触れない**。**設定変更は提案しない**。")


if __name__ == "__main__":
    sys.exit(selftest() if "--selftest" in sys.argv else (main() or 0))
