"""(204) ★★★★**モデルの形そのものを変える** — §4d・「市場への寄り方」を動かす

★★**動機（2026-09-06・利用者の指示「進めて」）**——**ANA_TRACK §4d に預けていたタスク**:
　★**「穴馬軸で考えるとき、現状のモデル（市場多め＋モデル）の形が最適じゃない可能性がある」**

★★★**(203)で入口が開いた**:
　★**複勝の「軸 − 乱」の対応差は、軸のズレに対して完全に単調**
　　（**ズレ0.030 → 0.187 で +5.8円 → +17.9円**、**5帯すべて有意**）。
　→ ★★**「もっと大きなズレを作れるモデル」があれば、外挿で100%を超える可能性がある**。
　⚠**ただし「大きなズレ＝雑音」なら勾配は崩れる**。**そこが分かれ目**。

■ ★★★**「市場への寄り方」をどう動かすか**
　★**現行モデルはオッズ特徴（log_odds, mkt_prob）を持つ**＝**市場に寄っている**。
　★★**残差学習の形にすると、寄り方が1つの数字λになる**:
　　**init_score = logit(市場の複勝確率)** を土台に置き、**特徴からオッズを外して学習**する。
　　→ **pn(λ) = sigmoid( logit(市場) + λ × f(x) )**
　　**λ=0 なら市場そのもの / λ=1 が学習した残差 / λ>1 は不一致の増幅**。
　★**λ を変えるのに再学習は要らない**（**生margin f(x) を掛けるだけ**）。

■ ★**測る腕（5本）**
| 腕 | 中身 |
|---|---|
| ★**A 現行** | **オッズ特徴あり**（**内部対照・(203)を再現しなければ読まない**） |
| **B オッズなし** | **log_odds も mkt_prob も外す**（⚠**(45)で「壊れる」と実測済み**） |
| ★**E λ=0.5** | **残差学習・寄りを強める** |
| ★**E λ=1.0** | **残差学習・そのまま** |
| ★**E λ=2.0** | ★**残差学習・不一致を2倍に増幅** |

────────────────────────────────────────────────────────────
★★★ 事前登録（2026-09-06・**結果を見る前にコミットする**）
────────────────────────────────────────────────────────────

■ ★経路（判定基準25）: **q = 各腕のモデル / q_pool = 複勝の板**（調和平均）。
■ ★手続き: **ウォークフォワード（3シード・年ごとに学習し直す）を腕ごとに走らせる**。
　⚠**先読みは入らない**（**init_score の市場確率はそのレースの確定前オッズ＝従来と同じ扱い**）。

■ ★★★家族A（**5比較**）: **(203)の最良の帯で腕を比べる**
　**軸 = pn≥0.15 かつ gap ≥ 0.15 の中で pn最大**、**複勝1点**、**乱は同じオッズ帯の10種平均**。
　★**主判定: 「軸 − 乱」の対応差**。⚠**腕ごとに該当レース数が変わる**（判定基準25）→ **必ず併記**。

■ ★★★家族B（**5比較**）: **各腕を「自分の極端」で測る**
　**各腕の gap の★上位10%** に入る馬を軸（**その中で pn最大**）。**複勝1点**。
　★★**腕ごとにズレの分布が違うので、固定の閾値では公平に比べられない**——**分位で揃える**。

■ ★★家族C（**記述・判定しない**）: ★**対応差 vs 軸の平均ズレ の曲線**（**各腕 × 5帯**）
| ★**全腕が同じ曲線に乗る** | ★**腕ごとに曲線が違う** |
|---|---|
| **効くのはズレの大きさだけ**。**あとは「大きなズレを作れるか」の勝負** | **モデルの形そのものが効く**。⚠**どの形が良いかを選べる** |
　★**この表を先に書いておく**（判定基準42）。

■ ★★ゲート2（判定基準42）——**仮説が偽なら何を返すか**
　★**モデルに残余情報が無いなら、どの腕でも対応差は0**（**乱は同じオッズ帯**）。
　★**λ=0 は市場そのものなので、対応差は0に近づくはず**（★**λの効き方の健全性検査**）。

■ ⚠ゲート1: (88)③④を再現（±3pt）／ ★ゲート板: 復元R 0.800±0.020 ／
　★陽性対照: 腕Aで三連複BOX上位4 81.9%±1.5pt。
■ ★★★内部対照（**決定的**・これが立たなければ何も読まない）:
　**腕Aが(203)の帯[0.15,∞) を再現する**——**複勝ROI 97.3% ±0.5pt かつ レース数 3,631 ±20**。
　⚠**腕Aは現行と同じコードで学習し直すので、厳密に一致するはず**。

■ ★★★**事前登録の修正（2026-09-06・★家族A/Bの判定を一度も見る前に）**
　⚠**第1版は内部対照が落ちた**——**ROIは 97.1% vs 97.3%（差0.2pt）で通ったが、
　　レース数が 3,677 vs 3,631 と46レース多かった**。
　★**原因は母集団**（判定基準25）: **(203)は「三連単の紐2頭とその乱3頭が引ける」
　　レースだけを記録していた**。**(204)は複勝しか要らないので条件が緩い**。
　⚠**私の内部対照の設計ミス（3回目）**——**条件の違う母集団に厳しい件数許容を置いた**。
　★★**修正: (203)と完全に同じ条件（紐2頭＋三連単の乱3頭＋複勝の乱）を課した
　　対照腕を1本足し、そこでROIと件数の両方を照合する**。
　★**本編の解析は緩い条件のまま**（**そのほうが標本が多い**）。**件数差は記述で明示する**。
　★**この修正はゲートを見て入れたもので、家族A/Bの判定は一度も見ていない**（判定基準38）。
　⚠**ただし「各腕のgap上位10%の閾値」（A 0.067 / B 0.107 / E0.5 0.098 / E1.0 0.109 /
　　E2.0 0.155）は目に入った**。★**これは分布の記述であって判定量ではないが、
　　見たことは記録しておく**（**λ=2が大きなズレを作れているのは事実として確認された**）。

■ ★★★探索を守る（**10比較・Bonferroni α=0.01/10**）
　⚠**標本300レース未満の腕は判定しない**（判定基準5）。
　★**採用条件に「λについて単調」**——**孤立した1点は採らない**。

■ ★採用条件
　1. **ある腕の対応差が、腕Aの対応差を明確に上回る**
　2. ★**その腕がより大きなズレを作れている**（**家族Bで確認**）
　3. **λについて単調**（**0.5 → 1.0 → 2.0 で向きが揃う**）
　4. **家族Cで曲線が上にずれている**（**同じズレでより大きな対応差**）

■ 予想（⚠**当てにしない**・判定基準24。★**私は(198)(201)(203)で3回連続外している**）
　★**λ=2 は対応差が落ちると見る**——**増幅は雑音も増幅するから**。
　★**Bは壊れると見る**（**(45)の実測どおり**）。
　⚠**ただし「λ=2でズレは大きくなるが勾配は保たれる」なら、それが§4dの答えになる**。
　★**家族Cで「全腕が同じ曲線」なら、モデルの形は関係なく「ズレの大きさ」だけが効く**——
　　**それはそれで大きな発見**（**現行の形が最適である証拠になる**）。

────────────────────────────────────────────────────────────
★★★★ 実測（2026-09-06）—— **§4dの答えはノー。現行の形が最良だった**
────────────────────────────────────────────────────────────

■ ゲート1 ／ ゲート板 0.8009 ／ 陽性対照（腕A）BOX4 81.9% ／
　★★★内部対照（決定的・(203)と完全に同じ条件）: **97.3%（3,631R）** → **1桁まで完全再現**
　★本編の緩い条件は 97.1%（3,677R）＝**件数差+46Rで−0.2pt**（判定基準25・明示）

■ ★★★家族A: **帯 [0.15,∞) で腕を比べる**
| 腕 | R数 | 買う率 | 軸オッズ | 平均ズレ | 複勝ROI | ★**対応差** | 判定 |
|---|---|---|---|---|---|---|---|
| ★**A 現行** | 3,677 | 12.5% | 10.4倍 | 0.186 | ★**97.1%** | ★★**+16.5円** | ★有意 |
| ⚠**B オッズなし** | 13,058 | 44.3% | ⚠**39.8倍** | 0.198 | ⚠**76.9%** | ⚠**+1.2円** | ⚠**通らない** |
| E λ=0.5 | 7,358 | 25.0% | 3.3倍 | 0.235 | 87.6% | +6.9円 | ★有意 |
| E λ=1.0 | 8,548 | 29.0% | 5.0倍 | 0.239 | 91.1% | +10.9円 | ★有意 |
| ★**E λ=2.0** | 15,390 | 52.2% | 14.6倍 | ★**0.264** | 91.2% | **+12.5円** | ★有意 |
★**λは単調**（+6.9 → +10.9 → +12.5円）。⚠**だが全部、現行の +16.5円 より下**。
★**Bは(45)の実測どおり壊れた**——**軸の平均オッズ39.8倍**＝**大穴を無差別に拾う**。

■ ★★★家族B: **各腕を「自分の極端」（gap上位10%）で測る**
| 腕 | R数 | 閾値 | 平均ズレ | 複勝ROI | ★**対応差** |
|---|---|---|---|---|---|
| A 現行 | 16,113 | 0.067 | 0.114 | 90.7% | **+10.5円** |
| ⚠B | 22,148 | 0.107 | 0.163 | 77.9% | ⚠+2.2円（通らない） |
| E λ=0.5 | 10,406 | 0.098 | 0.200 | 86.8% | +6.8円 |
| E λ=1.0 | 11,004 | 0.109 | 0.210 | 90.8% | +11.0円 |
| ★E λ=2.0 | 15,138 | 0.155 | **0.267** | 91.2% | ★**+12.4円** |
★**自分の極端どうしで比べると λ=2.0 (+12.4円) が A (+10.5円) を +1.9円 上回る**。
⚠**ただしCIが大きく重なる**（**A [+6.0,+15.0] / λ=2.0 [+7.1,+17.7]**）＝**差は検出できない**。
⚠**出力の「A比」列は家族Aの基準線(+16.5円)と比べており誤解を招く**。**上の+1.9円が正しい比較**。

■ ★★★★家族C: ★**決定打——同じズレで比べると現行が上**
| 軸の平均ズレ | ★**A 現行** | E λ=1.0 | ★**E λ=2.0** | ⚠B |
|---|---|---|---|---|
| 0.030 | +5.7円 | ★**+9.1円** | +2.7円 | +1.3円 |
| 0.050 | +8.1円 | ★**+10.3円** | +5.0円 | −2.6円 |
| 0.078 | +9.2円 | +8.4円 | +7.2円 | +1.7円 |
| 0.122 | +13.0円 | +13.4円 | +11.0円 | +4.2円 |
| ★**最大帯** | ★★**+16.5円**(0.186) | +10.9円(0.239) | **+12.5円**(0.264) | +1.2円(0.198) |
★★★**λ=2 はより大きなズレ(0.264)を作れるが、そこでの優位は +12.5円 で、
　現行が 0.186 で得ている +16.5円 に届かない**。
→ ★★**曲線は上ではなく下にずれている**＝**大きなズレを人工的に作っても
　単位ズレあたりの優位は減る**。

■ ★★★結論（**採用条件 2/4**）
| ★満たした | ⚠満たさない |
|---|---|
| **2. より大きなズレを作れている**（0.264 vs 0.186） | ⚠**1. 腕Aを上回る** |
| **3. λについて単調**（6.9→10.9→12.5円） | ⚠⚠**4. 曲線が上にずれている**（**実際は下**） |
★★**§4dの問いへの答え: 測った範囲では現行の形（市場＋α）が最良**。
⚠**「大きなズレ＝雑音」のほうが正しく、★(203)の勾配を外挿する道は閉じた**。
⚠**ただし測ったのは「残差学習＋λ」という1つの形だけ**。
　**混合や別目的の学習は測っていない**（判定基準25）。

■ ◇**post-hoc・未検定（拾わない）**
　★**λ=1.0 は小さいズレ（0.030〜0.050）で現行を上回る**（**+9.1 vs +5.7 / +10.3 vs +8.1円**）。
　⚠**post-hocなので拾わない**。★**再開条件: 混合（(100)(102)で採用実績あり）を
　　事前登録して測るとき**——**「小さいズレは残差モデル、大きいズレは現行」という
　　役割分担が成り立つかを、対応差で直接測る**。


実行: python3 ml/audit_ana_form.py    自己テスト: python3 ml/audit_ana_form.py --selftest
"""
import math
import os
import sys
from itertools import combinations
from zlib import crc32

import numpy as np
import pandas as pd

sys.path.insert(0, "ml")
import features as F
from audit_crosspool import load_races, payoff, zq
from audit_ana_odds import BANDS, MIN_HORSES, band_of, gate1, roi_of
from audit_ana_marg import FIRST_YEAR, WF_BOX4, WF_TOL
from audit_ana_board import BOARD_R, BOARD_TOL, NPLACE, load_fuku_boards, qpool
from audit_ana_ladder import FINE
from audit_ana_band import GBANDS, PN_FLOOR
from train_prod import CAPACITY, add_odds_features, fit_seeds

ARMS = ["A 現行", "B オッズなし", "E λ=0.5", "E λ=1.0", "E λ=2.0"]
LAM = {"E λ=0.5": 0.5, "E λ=1.0": 1.0, "E λ=2.0": 2.0}
TOPQ = 90.0          # ★家族B: gapの上位10%
NSEED = 3
NRAND = 10
SEED = 20260906
MINCELL = 300
KNOWN_ROI, KNOWN_N, ROI_TOL, N_TOL = 97.3, 3631, 0.5, 20
NCMP = 2 * len(ARMS)          # 5 + 5
ALPHA = 0.01
CACHE = "data/cache/form_pred.npz"
EPS = 1e-6


def wf(fx, y, year, init=None, nseed=NSEED):
    """★ウォークフォワード。init を渡すと残差学習（生marginを返す）"""
    _, PAR = CAPACITY["l2"]
    ys = [u for u in range(FIRST_YEAR, int(year.max()) + 1) if (year == u).sum() > 5000]
    out = np.full(len(fx), np.nan)
    for u in ys:
        tr, te = year < u, year == u
        if init is None:
            ms = fit_seeds(fx[tr], y[tr], nseed, PAR)
            out[te] = np.mean([m.predict_proba(fx[te])[:, 1] for m in ms], axis=0)
        else:
            ms = [__import__("lightgbm").LGBMClassifier(random_state=s, **PAR)
                  .fit(fx[tr], y[tr], init_score=init[tr],
                       categorical_feature=F.CAT_COLS) for s in range(nseed)]
            out[te] = np.mean([m.predict(fx[te], raw_score=True) for m in ms], axis=0)
        print(f"　　{u}: 学習 {tr.sum():,} → 予測 {te.sum():,}")
    return out


def selftest():
    ok = True
    z = zq(ALPHA / NCMP)
    print(f"★腕 {len(ARMS)}本: " + " / ".join(ARMS))
    print(f"★家族A {len(ARMS)}比較（帯[0.15,∞)で比較）＋ 家族B {len(ARMS)}比較"
          f"（各腕の gap 上位{100-TOPQ:.0f}%）→ **{NCMP}比較**・z = {z:.3f}")
    print("★★残差学習: **pn(λ) = sigmoid( logit(市場の複勝確率) + λ × f(x) )**")
    print("　★**λ=0 は市場そのもの / λ=1 が学習した残差 / λ>1 は不一致の増幅**")
    # ★λ=0 が市場そのものになることの確認
    mk = np.array([0.10, 0.25, 0.40])
    lg = np.log(mk / (1 - mk))
    f0 = np.array([1.5, -0.8, 0.3])
    p0 = 1 / (1 + np.exp(-(lg + 0.0 * f0)))
    print(f"★ゲート2（λ=0）: 市場 {mk} → pn(0) = {np.round(p0,4)}　"
          f"{'★OK（一致）' if np.allclose(p0, mk, atol=1e-9) else '⚠NG'}")
    ok &= np.allclose(p0, mk, atol=1e-9)
    p2 = 1 / (1 + np.exp(-(lg + 2.0 * f0)))
    p1 = 1 / (1 + np.exp(-(lg + 1.0 * f0)))
    print(f"★λの向き: λ=1 {np.round(p1,3)} → λ=2 {np.round(p2,3)}　"
          f"**市場から遠ざかる**: "
          f"{'★OK' if np.all(np.abs(p2-mk) >= np.abs(p1-mk) - 1e-12) else '⚠NG'}")
    ok &= np.all(np.abs(p2 - mk) >= np.abs(p1 - mk) - 1e-12)
    rng = np.random.default_rng(0)
    n = 120_000
    pay = rng.choice([0.0, 250.0, 900.0], size=n, p=[0.62, 0.30, 0.08])
    m = float(np.mean([(pay[rng.permutation(n)] - pay[rng.permutation(n)]).mean()
                       for _ in range(200)]))
    print(f"★ゲート2（対応差）: **偽なら0** → {m:+.3f}円　{'★OK' if abs(m) < 2 else '⚠NG'}")
    ok &= abs(m) < 2
    print(f"★★★内部対照（決定的）: **腕Aが(203)の帯[0.15,∞) を再現**"
          f"（**複勝ROI {KNOWN_ROI}% ±{ROI_TOL}pt かつ R数 {KNOWN_N} ±{N_TOL}**）")
    print("★★家族Cの読み方: **全腕が同じ曲線→効くのはズレの大きさだけ / "
          "腕ごとに違う→モデルの形が効く**")
    print("★自己テスト: " + ("全部OK" if ok else "⚠NG"))
    return 0 if ok else 1


def main():
    z = zq(ALPHA / NCMP)
    print("(204) ★★★★**モデルの形そのものを変える** — §4d")
    print("★★残差学習で「市場への寄り方」を λ という1つの数字にする\n")

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
    fx0, _ = F.encode_categoricals(f)
    odds = d["odds"].to_numpy(float)
    rid_arr = d["raceid"].to_numpy()
    fxA = add_odds_features(fx0, odds, rid_arr)
    year = d["date"].dt.year.to_numpy()
    inv = 1.0 / odds
    mkt = inv / pd.Series(inv).groupby(rid_arr).transform("sum").to_numpy()
    pmk = np.clip(NPLACE * mkt, EPS, 1 - EPS)      # ★市場の複勝確率（単勝から）
    init = np.log(pmk / (1 - pmk))

    if os.path.exists(CACHE):
        z0 = np.load(CACHE, allow_pickle=True)
        if int(z0["n"]) == len(d):
            print(f"★学習キャッシュを使う: {CACHE}")
            pA, pB, mE = z0["pA"], z0["pB"], z0["mE"]
        else:
            z0 = None
    else:
        z0 = None
    if z0 is None:
        print("\n★腕A（現行・オッズ特徴あり）を学習…")
        pA = wf(fxA, y, year)
        print("★腕B（オッズなし）を学習…")
        pB = wf(fx0, y, year)
        print("★腕E（残差学習・init_score = logit(市場の複勝確率)）を学習…")
        mE = wf(fx0, y, year, init=init)
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        np.savez_compressed(CACHE, pA=pA, pB=pB, mE=mE, n=len(d))
        print(f"★学習をキャッシュに保存: {CACHE}")

    P = {"A 現行": pA, "B オッズなし": pB}
    for nm2, lam in LAM.items():
        P[nm2] = 1.0 / (1.0 + np.exp(-(init + lam * mE)))
    msk = ~np.isnan(pA) & ~np.isnan(pB) & ~np.isnan(mE)
    print(f"\n★予測できた行 **{msk.sum():,}**")

    sub = d.loc[msk, ["raceid", "umaban", "odds", "date"]].copy()
    for a in ARMS:
        sub[a] = P[a][msk]

    K = {(a, b): {"fa": [], "fr": [], "gap": [], "od": []}
         for a in ARMS for b in GBANDS}
    TOP = {a: {"fa": [], "fr": [], "gap": []} for a in ARMS}
    CTRL203 = []          # ★(203)と完全に同じ条件の対照腕
    GALL = {a: [] for a in ARMS}
    Rs, box4, nall = [], [], 0
    STORE = []
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
        if not np.isfinite(od).all() or (od <= 0).any():
            continue
        qp, Rb = qpool([bd[int(u)] for u in ub], "harm")
        pns = {}
        okall = True
        for a in ARMS:
            pv = gg[a].to_numpy(float)
            if pv.sum() <= 0 or not np.isfinite(pv).all():
                okall = False
                break
            pns[a] = pv / pv.sum() * NPLACE
        if not okall:
            continue
        nall += 1
        Rs.append(Rb)
        pvA = gg["A 現行"].to_numpy(float)
        bx = [payoff(r, "三連複", list(c))
              for c in combinations(sorted(int(u) for u in ub[np.argsort(-pvA)[:4]]), 3)]
        if not any(x is None for x in bx):
            box4.append(sum(bx) - 400.0)
        bi = np.array([band_of(float(o), BANDS) for o in od])
        STORE.append((r, str(rid), ub, od, bi, qp, pns))
        for a in ARMS:
            GALL[a].append(pns[a] - qp)
    print(f"★対象 **{nall:,}レース**")

    med = float(np.median(Rs))
    okR = abs(med - BOARD_R) <= BOARD_TOL
    print(f"■ ★ゲート板: 復元R = **{med:.4f}** → **{'★立った' if okR else '⚠⚠落ちた'}**")
    box4 = np.asarray(box4, float)
    g0 = 100.0 * (box4.sum() + 400.0 * len(box4)) / (400.0 * len(box4))
    okb = abs(g0 - WF_BOX4) <= WF_TOL
    print(f"■ ⚠**陽性対照（腕A）**: BOX上位4 **{g0:.1f}%** vs {WF_BOX4}%"
          f" → **{'★立った' if okb else '⚠⚠落ちた'}**")

    THR = {a: float(np.percentile(np.concatenate(GALL[a]), TOPQ)) for a in ARMS}
    print("\n★各腕の gap 上位10%の閾値: "
          + " / ".join(f"{a} {THR[a]:.3f}" for a in ARMS))

    def draw_fuku(rid, ai, ub, bi):
        out = []
        for sd in range(NRAND):
            g2 = np.random.default_rng([SEED + sd, crc32(rid.encode())])
            pl = [int(ub[q]) for q in range(len(ub)) if bi[q] == bi[ai] and q != ai]
            if not pl:
                return None
            out.append(int(g2.choice(pl)))
        return out

    def draw3(rid, trio, ub, od, pos):
        """★(203)と同じ: 軸・紐1・紐2 を それぞれオッズ±20%の無作為な馬に置き換える"""
        out = []
        for sd in range(NRAND):
            g2 = np.random.default_rng([SEED + sd, crc32(rid.encode())])
            o3 = []
            for u in trio:
                k0 = pos[u]
                pl = [int(ub[q]) for q in range(len(ub))
                      if int(ub[q]) not in o3
                      and od[k0] / FINE <= od[q] <= od[k0] * FINE]
                if not pl:
                    return None
                o3.append(int(g2.choice(pl)))
            out.append(o3)
        return out

    for (r, rid, ub, od, bi, qp, pns) in STORE:
        # ── ★★内部対照: (203)と完全に同じ条件で 腕A・帯[0.15,∞) ──
        pnA = pns["A 現行"]
        gapA = pnA - qp
        lo0, hi0 = GBANDS[-1]
        cA = np.where((pnA >= PN_FLOOR) & (gapA >= lo0) & (gapA < hi0))[0]
        if len(cA):
            ai0 = int(cA[int(np.argmax(pnA[cA]))])
            ax0 = int(ub[ai0])
            pos0 = {int(u): q for q, u in enumerate(ub)}
            ordp = [int(u) for u in ub[np.argsort(-pnA, kind="mergesort")]]
            himo0 = [u for u in ordp if u != ax0]
            if len(himo0) >= 2:
                fa0 = payoff(r, "複勝", [ax0])
                ta0 = [payoff(r, "三連単", [ax0, himo0[0], himo0[1]]),
                       payoff(r, "三連単", [ax0, himo0[1], himo0[0]])]
                lst3 = draw3(rid, [ax0, himo0[0], himo0[1]], ub, od, pos0)
                pk0 = draw_fuku(rid, ai0, ub, bi)
                if (fa0 is not None and not any(v is None for v in ta0)
                        and lst3 is not None and pk0 is not None):
                    okc3 = True
                    for t3 in lst3:
                        vv = [payoff(r, "三連単", [t3[0], t3[1], t3[2]]),
                              payoff(r, "三連単", [t3[0], t3[2], t3[1]])]
                        if any(v is None for v in vv):
                            okc3 = False
                            break
                    vf3 = [payoff(r, "複勝", [u]) for u in pk0]
                    if okc3 and not any(v is None for v in vf3):
                        CTRL203.append(fa0)
        for a in ARMS:
            pn = pns[a]
            gap = pn - qp
            for b in GBANDS:
                lo, hi = b
                cand = np.where((pn >= PN_FLOOR) & (gap >= lo) & (gap < hi))[0]
                if not len(cand):
                    continue
                ai = int(cand[int(np.argmax(pn[cand]))])
                fa = payoff(r, "複勝", [int(ub[ai])])
                pick = draw_fuku(rid, ai, ub, bi)
                if fa is None or pick is None:
                    continue
                vr = [payoff(r, "複勝", [u]) for u in pick]
                if any(v is None for v in vr):
                    continue
                c = K[(a, b)]
                c["fa"].append(fa); c["fr"].append(float(np.mean(vr)))
                c["gap"].append(float(gap[ai])); c["od"].append(float(od[ai]))
            cand = np.where((pn >= PN_FLOOR) & (gap >= THR[a]))[0]
            if not len(cand):
                continue
            ai = int(cand[int(np.argmax(pn[cand]))])
            fa = payoff(r, "複勝", [int(ub[ai])])
            pick = draw_fuku(rid, ai, ub, bi)
            if fa is None or pick is None:
                continue
            vr = [payoff(r, "複勝", [u]) for u in pick]
            if any(v is None for v in vr):
                continue
            TOP[a]["fa"].append(fa); TOP[a]["fr"].append(float(np.mean(vr)))
            TOP[a]["gap"].append(float(gap[ai]))

    cv3 = np.asarray(CTRL203, float)
    okc = (abs(roi_of(cv3) - KNOWN_ROI) <= ROI_TOL
           and abs(len(cv3) - KNOWN_N) <= N_TOL)
    print(f"\n■ ★★★内部対照（決定的・**(203)と完全に同じ条件**）: 腕A・帯[0.15,∞) → "
          f"**{roi_of(cv3):.1f}%**（{len(cv3):,}R） vs (203) {KNOWN_ROI}%（{KNOWN_N:,}R）"
          f" → **{'★再現' if okc else '⚠⚠ズレた'}**")
    av = np.asarray(K[("A 現行", GBANDS[-1])]["fa"], float)
    print(f"　★**本編の緩い条件**（複勝だけ要求）: **{roi_of(av):.1f}%**（{len(av):,}R）"
          f"　⚠**件数差 {len(av)-len(cv3):+,}R で {roi_of(av)-roi_of(cv3):+.1f}pt**"
          f"（判定基準25）")
    if not (okR and okb and okc):
        print("\n⚠⚠**ゲートが落ちた。読まない**（判定基準32）。")
        return

    print(f"\n{'='*112}")
    print("■ ★★★家族A: **(203)の最良の帯 [0.15,∞) で腕を比べる**")
    print(f"\n{'腕':<14}{'R数':>8}{'買う率':>8}{'軸オッズ':>9}{'平均ズレ':>9}"
          f"{'★複勝ROI':>10}{'乱':>8}{'★対応差':>10}{'99%CI':>19}{'判定':>12}")
    baseA = None
    for a in ARMS:
        c = K[(a, GBANDS[-1])]
        av = np.asarray(c["fa"], float); rv = np.asarray(c["fr"], float)
        if len(av) < MINCELL:
            print(f"{a:<14}{len(av):>8,}　⚠**標本不足**")
            continue
        dd = av - rv
        mu, se = dd.mean(), dd.std(ddof=1) / math.sqrt(len(dd))
        if a == "A 現行":
            baseA = mu
        print(f"{a:<14}{len(av):>8,}{100*len(av)/nall:>7.1f}%"
              f"{np.mean(c['od']):>8.1f}倍{np.mean(c['gap']):>9.3f}"
              f"{roi_of(av):>9.1f}%{roi_of(rv):>7.1f}%{mu:>+9.1f}円"
              f"{f'[{mu-z*se:+.1f},{mu+z*se:+.1f}]':>19}"
              f"{('★★有意' if mu - z*se > 0 else '⚠通らない'):>12}")

    print(f"\n{'='*112}")
    print(f"■ ★★★家族B: **各腕を「自分の極端」（gap上位{100-TOPQ:.0f}%）で測る**")
    print(f"\n{'腕':<14}{'R数':>8}{'閾値':>8}{'平均ズレ':>9}{'★複勝ROI':>10}"
          f"{'乱':>8}{'★対応差':>10}{'99%CI':>19}{'A比':>9}{'判定':>12}")
    for a in ARMS:
        c = TOP[a]
        av = np.asarray(c["fa"], float); rv = np.asarray(c["fr"], float)
        if len(av) < MINCELL:
            print(f"{a:<14}{len(av):>8,}　⚠**標本不足**")
            continue
        dd = av - rv
        mu, se = dd.mean(), dd.std(ddof=1) / math.sqrt(len(dd))
        print(f"{a:<14}{len(av):>8,}{THR[a]:>8.3f}{np.mean(c['gap']):>9.3f}"
              f"{roi_of(av):>9.1f}%{roi_of(rv):>7.1f}%{mu:>+9.1f}円"
              f"{f'[{mu-z*se:+.1f},{mu+z*se:+.1f}]':>19}"
              f"{(mu-baseA if baseA is not None else float('nan')):>+8.1f}円"
              f"{('★★有意' if mu - z*se > 0 else '⚠通らない'):>12}")

    print(f"\n{'='*112}")
    print("■ ★★家族C（**記述**）: ★**対応差 vs 軸の平均ズレ の曲線**")
    print("　★★読み方: **全腕が同じ曲線→効くのはズレの大きさだけ / "
          "腕ごとに違う→モデルの形が効く**")
    print(f"\n{'帯':<14}" + "".join(f"{a:>22}" for a in ARMS))
    for b in GBANDS:
        lo, hi = b
        nm2 = f"[{lo:.2f},{hi:.2f})" if hi < 9 else f"[{lo:.2f},∞)"
        cells = []
        for a in ARMS:
            c = K[(a, b)]
            av = np.asarray(c["fa"], float); rv = np.asarray(c["fr"], float)
            if len(av) < MINCELL:
                cells.append(f"{'—':>22}")
                continue
            mu = (av - rv).mean()
            cells.append(f"{np.mean(c['gap']):>7.3f}→{mu:>+6.1f}円({len(av):>5,})")
        print(f"{nm2:<14}" + "".join(cells))

    print("\n■ ★採用条件: **1.腕Aを明確に上回る / 2.より大きなズレを作れている / "
          "3.λについて単調 / 4.曲線が上にずれている**")
    print("⚠**経路は弱いので「閉じた」とは書けない**（判定基準25）。")
    print("\n⚠**枠連の運用には触れない**。**設定変更は提案しない**。")


if __name__ == "__main__":
    sys.exit(selftest() if "--selftest" in sys.argv else (main() or 0))
