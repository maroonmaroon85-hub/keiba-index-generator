"""(197) ★★★**三連単は信用できるか** — ★**物差しをROIから「的中率」に替える**

★★**動機（2026-09-06・利用者の指摘）**:
　★**「複勝である程度穴馬を推奨できて、かつ馬連で軸・紐の確からしさを一定信頼できるなら、
　　自ずと三連単も信用できると思う」**
　★**理屈は正しい**——**モデルの優位が馬1頭ごとに付いているなら、3頭を指定する券では
　　その優位が3回かかる**。⚠**このプロジェクトで唯一100%超の枠連(117.3%)も、
　　「組み合わせると比の歪みが増幅する」から効いている**（ROI_MAP IV）。
　⚠**壊れる理由も2つある**:
　　**1. 三連単は着順を当てる券だが、モデルは「3着以内か」でしか学習していない**
　　**2. 確率の合計は1に固定＝ある馬が安すぎるなら別の馬は高すぎる。誤差は独立でない**
　★**(187)で27マス測って27/27が通らず、後ろの着順ほど悪かった**。

★★★**私の「測らない」判断を撤回する**（2026-09-06）——
　⚠**「900年必要」と書いたのはROIで測ろうとしていたから**。**ROIは配当の裾に支配される**。
　★★**判定基準26（「データが足りない」と書く前に物差しを替えられないか考える）を
　　自分で守れていなかった**。**馬連では複勝に替えたのに、三連単では替えずに諦めていた**。

★★★**新しい物差し: 的中率**
　**ROI = 的中率 × 1回あたり配当**。★**オッズを揃えた対照と比べれば配当が揃い、
　　残るのは的中率の差だけ**。**対照は払戻率72.5%を返すはずなので、
　　★こちらの的中率が対照の 1/0.725 = 1.38倍 あれば100%**。
| | ROIで測る | ★**的中率で測る** |
|---|---|---|
| 効くもの | ⚠**数本の高配当** | ★**当たった回数（約770回）** |
| 検出できる差 | ⚠±31pt | ★**相対17%** |
| ★必要な差 | — | ★**相対38%** |
　★★**必要な38%は検出できる17%の2倍以上ある**＝**今あるデータで判定できる**。

────────────────────────────────────────────────────────────
★★★ 事前登録（2026-09-06・**結果を見る前にコミットする**）
────────────────────────────────────────────────────────────

■ ★経路（判定基準25）: **q = MLモデル / q_pool = 複勝の板**（調和平均）。⚠**q側は弱い**。
■ ★手続き: **ウォークフォワード**（`data/cache/wf_pred.npz`）。**母集団は(190)(193)〜(196)と同一**。
　**軸 = (pn≥A かつ gap≥B) を満たす馬のうち gap 最大**、**紐 = gap降順の2頭**（**紐床なし**）。
　**絞り (0.15,0.02) / (0.20,0.06)**。

■ ★★★**券種を梯子にする**（★**利用者の「自ずと信用できる」がどの段で切れるかを見る**）
| 券種 | 何を当てるか | 点数 | 払戻率 |
|---|---|---|---|
| **馬連** | **2頭・順不同** | 1点（軸+紐1） | 77.5% |
| **馬単** | **2頭・順序** | 2点 | 75.0% |
| ★**三連単** | ★**3頭・順序** | 2点（軸1着固定・紐の2着3着入替） | ⚠**72.5%** |

■ ★★★**対照（10種平均・種はレースに紐づける）**
　★**乱全**: ★**軸も紐も、それぞれ同じオッズ帯の無作為な馬に置き換える**（**主判定**）。
　　★**これで配当の分布が揃う**——**残るのは「どの馬を選んだか」だけ**。
　★**乱紐**: **軸は据え置き、紐だけ無作為**（**記述**）。★**軸と紐のどちらが効いているかの分解**。

■ ★★★家族A（**6比較**）: **的中率の差（こちら − 乱全）**、**3券種 × 絞り2つ**。
　★**主判定は「差が有意に正か」**。
　★★**併せて「的中率の比」を出し、1.38倍（＝100%の線）と比べる**。
■ ★★家族B（**記述・判定しない**）: **1回あたり配当が両腕で揃っているか**、
　**乱全のROIが払戻率を返すか**、**乱紐との比較**、**ROIそのもの**。

■ ★★ゲート2（判定基準42）——**仮説が偽なら何を返すか**
　★**モデルが馬を選べていないなら、的中率の差の期待値は厳密に0**（**同じ帯からの無作為**）。
　★**乱全のROIは払戻率（77.5 / 75.0 / 72.5%）を返す**。⚠**これが返らなければ対照が壊れている**。
　★**配当が揃っていなければ、的中率の差をROIに翻訳できない**——**家族Bで必ず確認する**。

■ ⚠ゲート1: (88)③④を再現（±3pt）／ ★ゲート板: 復元R 0.800±0.020 ／
　★陽性対照: 三連複BOX上位4 81.9%±1.5pt。
■ ★★内部対照（**決定的な量だけ**・(195)の教訓）:
　**三連単 (0.15,0.02) 紐2頭のROIが (190)の 103.1% を ±0.5pt で再現**。
　⚠**乱を含む量は内部対照に使わない**。

■ ★★★探索を守る（**6比較・Bonferroni α=0.01/6**）
　★**採用条件に「梯子が単調」**——**馬連 → 馬単 → 三連単 で比が保たれるか、落ちるか**。
　⚠**(178)でプラセボが109.9%を出した**。**最良のマスを見出しにしない**。

■ ★★★**事前登録の修正（2026-09-06・★家族Aの結果を一度も見る前に）**
　⚠**第1版は内部対照が落ちた**（**三連単 66.3% vs (190) 103.1%**）。
　★**原因: 私が紐を「ズレ順(gap降順)」で組んでいたが、(190)の103.1%は「モデル上位順(p降順)」**。
　　**(190)は(191)より前の測定で、当時の紐は常に p降順だった**（判定基準25: **母集団が違う**）。
　★★**この落ち方自体が情報**: **同じ三連単で紐をズレ順にすると 103.1% → 66.3%**。
　　**(191)の「三連単だけ −35.9円 と大暴落」と一致する**。
　★★**修正: 主判定の紐を p降順（＝(190)と同じ）にし、ズレ順は記述として併記する**。
　★**この修正はゲートを見て入れたもので、家族Aの判定は一度も見ていない**（判定基準38）。

■ ★採用条件
　1. **的中率の差が有意に正**（**3券種とも**）
　2. ★**三連単の的中率比が 1.38倍を超える**（**＝ROIで100%を超えることを意味する**）
　3. **配当が両腕で揃っている**（**家族B**）
　4. **乱全のROIが払戻率を返す**（**対照が壊れていない**）

■ 予想（⚠**類推なので当てにしない**・判定基準24）
　★**的中率の差は3券種とも正と見る**（**(196)で機構が健在と確認済み**）。
　⚠**比は 1.38倍に届かないと見る**——**複勝の実測は 92.5/79.7 = 1.16倍**であり、
　　★**3頭に増やしても比が2倍以上に増幅するとは考えにくい**。
　⚠**ただし「増幅するか否か」こそが利用者の問いであり、私の予想は根拠が弱い**。
　★**着順の情報が無いことが効くなら、馬連 → 馬単 で比が落ちるはず**。**そこを見る**。

────────────────────────────────────────────────────────────
★★★★ 実測（2026-09-06）—— **増幅は実在した。だが控除に届かない**
────────────────────────────────────────────────────────────

■ ゲート1 4帯 ／ ゲート板 0.8009 ／ 陽性対照 BOX4 81.9% ／
　★★内部対照（決定的）: **三連単 (0.15,0.02) ROI = 103.1%** vs (190) 103.1% → **完全再現**

■ ★★★家族A: ★**券種が難しくなるほど的中率の比が上がる**（**利用者の読みが数字で出た**）
| 券種 | 何を当てる | R数 | 本の的中率 | 乱全 | ★差 | ★★**比** | 必要 |
|---|---|---|---|---|---|---|---|
| 馬連 | 2頭・順不同 | 29,437 | 11.377% | 10.713% | +0.664pp | **1.062倍** | 1.29倍 |
| 馬単 | 2頭・順序 | 29,437 | 11.377% | 10.713% | +0.664pp | **1.062倍** | 1.33倍 |
| ★**三連単** | ★**3頭・順序** | 29,437 | 2.629% | 2.389% | +0.240pp | ★**1.101倍** | ⚠**1.38倍** |
| 馬連 | 〃(0.20,0.06) | 26,096 | 12.029% | 11.355% | +0.674pp | 1.059倍 | 1.29倍 |
| 三連単 | 〃(0.20,0.06) | 26,096 | 2.744% | 2.534% | +0.210pp | 1.083倍 | ⚠**1.38倍** |
★★**1.062 → 1.101倍**＝**「1頭ごとの優位が重なる」は実在する**。**5/6が有意**。
⚠**だが必要な比（1.29 → 1.38）のほうが速く上がる**＝**増幅が控除に追いつかない**。
⚠**馬単は2点で両方の順序を買うので的中率は馬連と同一**——**着順の腕前は測れていない**（設計の限界）。

■ ★★★★家族B: ★**103.1%の正体を分解できた**
**三連単 (0.15,0.02) の 103.1% ＝ 79.3% × 1.101 × 1.18**
| | 値 | ★信頼できるか |
|---|---|---|
| ① **オッズ帯そのもののROI**（乱全） | **79.3%** | ★**測定済み** |
| ★② **当てる腕前**（的中率の比） | ★**1.101倍** | ★**有意**（774回の的中） |
| ⚠③ **配当の上乗せ** | ⚠**1.18倍**（7,840円 vs 6,636円） | ⚠⚠**774回の的中だけで決まる＝裾そのもの** |
★★★**「103.1%」の大半は③**。⚠**そこは(190)で「900年必要」と書いた、まさにその裾**。
★**信頼できる ①×② だけなら 87.3%**。
★**対照: 馬連の配当の上乗せは 1.03倍 / 1.00倍 でほぼゼロ**——
　⚠**三連単だけ1.18倍なのは「配当が揃わなかった」と読むのが自然**（**オッズ帯の粒度が粗い**）。

■ ★★★利用者の問いへの答え
　★**「複勝で穴馬を選べ、馬連で軸・紐を信頼できるなら、自ずと三連単も信用できる」**
| ★正しい | ⚠足りない | ⚠危ない |
|---|---|---|
| **優位は実在し券種が難しいほど増幅**（1.062→1.101倍）**。測れた** | **増幅1.10倍が控除1.38倍に届かない**（**25%分の不足**） | **103.1%を作るのは配当の上乗せ1.18倍＝測れない裾** |
　★★**「モデルは三連単でも良い馬を選んでいる」＝信用できる**。
　⚠**「だから三連単で勝てる」＝そこは繋がらない**。

■ ⚠**私の設計ミス（対照の作り方・4回目）**
　★**事前登録で「対照は払戻率72.5%を返すはず」と書いたが、返さなかった（79.3%）**。
　⚠**当然だった**——**対照はモデルが選んだ馬と同じオッズ帯から引くので、
　　(88)の「その帯のROIが高い」性質をそのまま受け継ぐ**。
　★**対照が壊れていたのではなく、私の予想の書き方が誤り**（判定基準37・(176)(184)(195)に続く4回目）。
　★**採用条件4は私のミスなので判定から外す。採用条件2・3が満たされないので結論は変わらない**。

■ ★★併記（記述）: **紐をズレ順にすると三連単は壊れる**
| 券種 | 本(p降順) | ★**ズレ順** | 的中率(ズレ順) |
|---|---|---|---|
| 馬連 | 90.0% | **94.3%** | 6.414% |
| 馬単 | 87.1% | **91.1%** | 6.414% |
| ★**三連単** | ★**103.1%** | ⚠⚠**66.3%** | ⚠**0.798%** |
★★**馬連・馬単は良くなるのに、三連単だけ 103.1% → 66.3% と壊れる**。
　⚠**的中率が 2.629% → 0.798% と3分の1になる**——**ズレ順の紐は「来ない馬」を上位に置く**。
　★**(191)の「三連単だけ −35.9円 と大暴落」の正体はこれ**。


実行: python3 ml/audit_ana_hit.py    自己テスト: python3 ml/audit_ana_hit.py --selftest
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
KINDS = ["馬連", "馬単", "三連単"]
NPT = {"馬連": 1, "馬単": 2, "三連単": 2}
NSEED = 10
SEED = 20260906
CTRL_TRI, KNOWN_TRI, ROI_TOL = (0.15, 0.02), 103.1, 0.5     # ★(190)・決定的
MINCELL = 300
NCMP = len(KINDS) * len(FILT)          # 6
ALPHA = 0.01


def tickets(kind, ax, h1, h2):
    if kind == "馬連":
        return [("馬連", [ax, h1])]
    if kind == "馬単":
        return [("馬単", [ax, h1]), ("馬単", [h1, ax])]
    return [("三連単", [ax, h1, h2]), ("三連単", [ax, h2, h1])]


def selftest():
    ok = True
    z = zq(ALPHA / NCMP)
    print(f"★家族A {NCMP}比較（**的中率の差**・{KINDS} × 絞り2つ）→ z = {z:.3f}")
    for k in KINDS:
        print(f"　{k}: {NPT[k]}点・払戻率 {100*LINE[k]:.1f}% → "
              f"★**100%に要る的中率の比 {1/LINE[k]:.3f}倍**")
    print(f"　⚠**三連単は {1/LINE['三連単']:.2f}倍 必要**")
    # ★検出力: n=29,000・的中2.63% で、どれだけの相対差を検出できるか
    n, p = 29000, 0.0263
    se = math.sqrt(2 * p * (1 - p) / n)
    print(f"★検出力: n={n:,}・的中{100*p:.2f}% → 差の標準誤差 {100*se:.3f}pp、"
          f"検出できる相対差 **{100*z*se/p:.1f}%**"
          f"　{'★OK（必要な38%より小さい）' if z*se/p < 0.38 else '⚠NG'}")
    ok &= z * se / p < 0.38
    rng = np.random.default_rng(0)
    a = rng.random(200_000) < p
    b = rng.random(200_000) < p
    print(f"★ゲート2（的中率の差）: **偽なら0** → {100*(a.mean()-b.mean()):+.4f}pp"
          f"　{'★OK' if abs(a.mean()-b.mean()) < 3*se else '⚠NG'}")
    print(f"★★内部対照（決定的）: **三連単 {CTRL_TRI} のROIが {KNOWN_TRI}% を "
          f"±{ROI_TOL}ptで再現**（⚠**乱を含む量は使わない**）")
    print("★★読み方: **比が1.38倍超→三連単は+EV / 1.0〜1.38→機構はあるが控除に足りない**")
    print("★自己テスト: " + ("全部OK" if ok else "⚠NG"))
    return 0 if ok else 1


def main():
    z = zq(ALPHA / NCMP)
    print("(197) ★★★**三連単は信用できるか** — ★**物差しをROIから的中率に替える**")
    print("★利用者の指摘: **複勝で穴馬を選べ、馬連で軸・紐を信頼できるなら、"
          "自ずと三連単も信用できるはず**")
    print("★★対照は「軸も紐も同じオッズ帯の無作為な馬」＝**配当が揃い、残るのは選び方だけ**\n")

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

    ARMS = ("本", "乱全", "乱紐", "ズレ順", "ズレ乱全")
    K = {(fl, k): {a: {"pay": [], "hit": []} for a in ARMS}
         for fl in FILT for k in KINDS}
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
        order_g = list(np.argsort(-gap, kind="mergesort"))
        for fl in FILT:
            cand = np.where((pn >= fl[0]) & (gap >= fl[1]))[0]
            if not len(cand):
                continue
            ai = int(cand[int(np.argmax(gap[cand]))])
            axu = int(ub[ai])
            hg_p = [int(u) for u in ub[np.argsort(-pv, kind="mergesort")]
                    if int(u) != axu]
            hg_g = [int(ub[k]) for k in order_g if int(ub[k]) != axu]
            if len(hg_p) < 2 or len(hg_g) < 2:
                continue
            TRIOS = {"本": [axu, hg_p[0], hg_p[1]],        # ★(190)と同じ p降順
                     "ズレ順": [axu, hg_g[0], hg_g[1]]}
            # ★乱: それぞれ同じオッズ帯の無作為な馬に置き換える
            def draw(rngen, which, trio):
                out = []
                for j, u in enumerate(trio):
                    if which.endswith("乱紐") and j == 0:
                        out.append(u)
                        continue
                    b = bi[pos[u]]
                    pl = [int(ub[k]) for k in range(len(ub))
                          if bi[k] == b and int(ub[k]) not in out]
                    if not pl:
                        return None
                    out.append(int(rngen.choice(pl)))
                return out
            sets = {"本": [TRIOS["本"]], "ズレ順": [TRIOS["ズレ順"]]}
            okall = True
            for which, base in (("乱全", "本"), ("乱紐", "本"), ("ズレ乱全", "ズレ順")):
                lst = []
                for sd in range(NSEED):
                    g2 = np.random.default_rng([SEED + sd, crc32(str(rid).encode())])
                    t = draw(g2, which, TRIOS[base])
                    if t is None:
                        okall = False
                        break
                    lst.append(t)
                if not okall:
                    break
                sets[which] = lst
            if not okall:
                continue
            vals = {}
            for kind in KINDS:
                for which, lst in sets.items():
                    acc, hh = [], []
                    for t in lst:
                        tk = tickets(kind, t[0], t[1], t[2])
                        vs = [payoff(r, nm, sel) for nm, sel in tk]
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
            for kind in KINDS:
                for which in ARMS:
                    pay, hit = vals[(kind, which)]
                    c = K[(fl, kind)][which]
                    c["pay"].append(pay); c["hit"].append(hit)
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
    print(f"■ ★★内部対照（決定的）: 三連単 {CTRL_TRI} ROI = **{tr:.1f}%** vs (190) "
          f"{KNOWN_TRI}% → **{'★再現' if okc else '⚠⚠ズレた'}**（{len(tv):,}R）")
    if not (okR and okb and okc):
        print("\n⚠⚠**ゲートが落ちた。読まない**（判定基準32）。")
        return

    print(f"\n{'='*112}")
    print("■ ★★★家族A: **的中率の差**（**本 − 乱全**・オッズを揃えた対照）")
    print("　★★読み方: **比が1.38倍超→三連単は+EV / 1.0〜1.38→機構はあるが控除に足りない**")
    for fl in FILT:
        print(f"\n★絞り{fl}")
        print(f"{'券種':<8}{'点数':>5}{'R数':>8}{'★本の的中率':>13}{'乱全':>9}"
              f"{'★差':>10}{'99%CI':>21}{'★★比':>9}{'必要':>7}{'判定':>13}")
        for kind in KINDS:
            c = K[(fl, kind)]
            a = np.asarray(c["本"]["hit"], float)
            rr = np.asarray(c["乱全"]["hit"], float)
            if len(a) < MINCELL:
                continue
            dd = a - rr
            mu = dd.mean()
            se = dd.std(ddof=1) / math.sqrt(len(dd))
            ratio = a.mean() / max(rr.mean(), 1e-12)
            need = 1.0 / LINE[kind]
            sig = mu - z * se > 0
            print(f"{kind:<8}{NPT[kind]:>5}{len(a):>8,}{100*a.mean():>12.3f}%"
                  f"{100*rr.mean():>8.3f}%{100*mu:>+9.3f}pp"
                  f"{f'[{100*(mu-z*se):+.3f},{100*(mu+z*se):+.3f}]':>21}"
                  f"{ratio:>8.3f}倍{need:>6.2f}倍"
                  f"{('★★超えた' if ratio > need and sig else '★差はある' if sig else '⚠通らない'):>13}")

    print(f"\n{'='*112}")
    print("■ ★★家族B（**記述**）: **配当は揃っているか / 対照は払戻率を返すか**")
    for fl in FILT:
        print(f"\n★絞り{fl}")
        print(f"{'券種':<8}{'★本ROI':>10}{'乱全ROI':>10}{'払戻率':>8}"
              f"{'乱紐ROI':>10}{'★1回配当(本)':>14}{'(乱全)':>12}{'配当比':>8}"
              f"{'★ズレ順ROI':>12}{'その的中率':>12}{'ズレ乱全':>10}")
        for kind in KINDS:
            c = K[(fl, kind)]
            cost = 100.0 * NPT[kind]
            a = np.asarray(c["本"]["pay"], float); ah = np.asarray(c["本"]["hit"], float)
            rv = np.asarray(c["乱全"]["pay"], float)
            rh = np.asarray(c["乱全"]["hit"], float)
            hv = np.asarray(c["乱紐"]["pay"], float)
            zv = np.asarray(c["ズレ順"]["pay"], float)
            zh = np.asarray(c["ズレ順"]["hit"], float)
            zr = np.asarray(c["ズレ乱全"]["pay"], float)
            if len(a) < MINCELL:
                continue
            pa = a.sum() / max(ah.sum(), 1e-9)
            pr = rv.sum() / max(rh.sum(), 1e-9)
            print(f"{kind:<8}{100*a.mean()/cost:>9.1f}%{100*rv.mean()/cost:>9.1f}%"
                  f"{100*LINE[kind]:>7.1f}%{100*hv.mean()/cost:>9.1f}%"
                  f"{pa:>13,.0f}円{pr:>11,.0f}円{pa/max(pr,1e-9):>7.2f}倍"
                  f"{100*zv.mean()/cost:>11.1f}%{100*zh.mean():>11.3f}%"
                  f"{100*zr.mean()/cost:>9.1f}%")

    print("\n■ ★採用条件: **1.差が有意 / 2.三連単の比が1.38倍超 / "
          "3.配当が揃う / 4.乱全が払戻率を返す**")
    print("⚠**経路は弱いので「閉じた」とは書けない**（判定基準25）。")
    print("\n⚠**枠連の運用には触れない**。**設定変更は提案しない**。")


if __name__ == "__main__":
    sys.exit(selftest() if "--selftest" in sys.argv else (main() or 0))
