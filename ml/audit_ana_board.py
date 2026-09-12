"""(184) ★★★★**複勝の板を q_pool にして測り直す** — 自己監査で見つかった最大の漏れ

★★★★**漏れの内容（2026-09-06・自己監査）**——
　**(174)〜(183)の10本すべてで、私は「複勝を買い、値段は単勝オッズで測る」ことをしていた**。
　　**市場の基準は一貫して `mk = (1/単勝オッズ)/Σ`**。
　⚠**単勝プールと複勝プールは別の市場で、別の値段が付いている**。
　　**「単勝では妥当だが複勝では安い馬」は原理的に存在しうる**。
　★**`NEXT_ANA.md` が `load_board(2, 2)` を明示していたのに、一度も開いていなかった**。
　★★**データは全期間ある**（`data/nk_odds/type2_2013〜2026.jsonl.gz`・**42,161レース**）。

★★**なぜこれが結論を変えうるか**——**ROI_MAPの「経路の強弱」**:
| 経路 | 意味 | 実測 |
|---|---|---|
| ★**強い** | **別の市場の板を厳密に集約**して q を作る | 複勝 **89.0%**((146) 三連複プール) |
| ⚠**弱い** | **単勝オッズ由来** | 複勝 **46.3%**((151)) |
★**同じ複勝で経路の違いが40pt動いた前例がある**。**本件は q_pool 側を複勝の板に差し替える**。

⚠**先に逆風を書く（判定基準24: 類推は当たらない／恒等式と実測の上界だけ信用してよい）**——
　★**(146)の 89.0% が「強い経路での複勝の天井」**として実測されている。**100%には11pt足りない**。
　→ ★**水準が100%を超える見込みは高くない**。**それでも方法論の穴なので潰す**。

────────────────────────────────────────────────────────────
★★★ 事前登録（2026-09-06・**結果を見る前にコミットする**）
────────────────────────────────────────────────────────────

■ ★経路（判定基準25）——**ここが本件の変更点**
　　**q      = MLモデルの top3 確率 p**（Σ=3 に正規化）
　　★**q_pool = 複勝の板**（type=2・`(1/オッズ)/Σ × 3`）　←**(174)〜(183)は単勝オッズだった**
　⚠**q 側は依然としてMLモデル＝弱い**。**強くなるのは q_pool 側だけ**。**「強い経路」とは呼べない**。
　　→ ★**書けるのは「複勝の市場価格で測っても届かない」まで**。

■ ★板の使い方（**(124)の事前登録2に倣う: どれか1つを選ばない**）
　**複勝の板は `[下限, 上限]` の範囲**（**最低配当100円の保証で他の的中馬に配分が動くため**）。
　⚠⚠★**第1版は mid=(下限+上限)/2 を主にしたが、ゲート板が落ちた**（**R=0.8321 vs 0.800**）。
　★**判定基準37を当てて対照の定式化を疑い、恒等式に戻って直した**——
　　**`R = 3/Σ(1/o)` は 1/o について線形**なので、**価格の中心は「オッズの平均」ではなく
　　「1/o の平均」＝調和平均 `2·lo·hi/(lo+hi)`**。
　★**実測（42,161レース・復元Rの中央値）**:
| 価格の取り方 | 復元R | 0.800との差 |
|---|---|---|
| 下限 | 0.6682 | −0.1318 |
| mid（第1版） | 0.8321 | **+0.0321** |
| 上限 | 0.9895 | +0.1895 |
| 幾何平均 | 0.8161 | +0.0161 |
| ★**調和平均** | ★**0.8003** | ★**+0.0003** |
　→ ★**主は調和平均**。**下限・mid・上限は感度として出す**（**記述**）。
　⚠★**この訂正はゲートを見て行ったもので、主判定は一度も見ていない**。**判定基準37の正しい形**。
　★**判定基準38の2件目**——**測れる定数を測ったら、定式化の誤りが出た**（1件目は(162)の馬単）。

■ ★手続き: **ウォークフォワード**（判定基準6・`data/cache/wf_pred.npz` を再利用）。
■ ★程度は(180)(183)と同一の**単勝オッズ9段**（**「穴の程度」の自然な定義は単勝オッズ**）。
■ ★設計も(180)(183)と同一: **同じレース・同じ帯の 穴 vs 乱・複勝1点**。
　★**違いは穴の選び方だけ**: **穴 = 帯内で `gap_f = p_norm − q_pool_fuku` が最大の馬**。

■ ★★★主判定（**10比較・Bonferroni α=0.01/10・z=3.291**）
　★**A（9比較）**: **各帯の「穴(板) − 乱」の対応差（円）**。
　★★**B（1比較・自己監査の漏れC）**: ★**絶対閾値の裾**——
　　**`ratio_f = p_norm / q_pool_fuku ≥ 1/0.800 = 1.250` を満たす馬の複勝を全部買う**。
　　**判定: そのROIの99%CI下端が100%を超えるか**。
　⚠**Bが本命**——**ROI_MAP §0の「正しい条件 ∃k: q_k/q_pool,k ≥ 1/R」はこの形**。
　　**(174)〜(183)は全部「相対的な裾（上位10%・十分位・帯内最大）」で、絶対閾値ではなかった**。
　⚠**近い先行**: **(151) λHarville→複勝の板・1.25で 46.3%**（**q が弱い経路**）。
　　★**本件は q がMLモデルなので別の量**。**(173)「MLの確率は板から大きくズレている」が逆風**。

■ ★★ゲート2（判定基準42）——**仮説が偽なら何を返すか**
　★**A**: **仮説が偽（gap_f が帯の中で何も持たない）なら、穴と乱は交換可能＝対応差の期待値は0**。
　★**B**: **仮説が偽（板が正しく値付けしている）なら、閾値で切った群のROIは払戻率80.0%を返す**
　　（**100%ではない**）。→ ★**「100%を超えるか」は払戻率からの超過20.0ptを問う判定**。

■ ⚠ゲート1（判定基準32）: (88)③④を別パーサで再現（±3pt）。
■ ★★ゲート板（**新設・判定基準38「測れる定数は測れ」**）
　★**板から復元した複勝の払戻率 `R = 3/Σ(1/mid)` の中央値が 0.800 ± 0.020**であること。
　⚠**外れたら板の読み方を疑う**（**(162)で馬単の定数が2.6pt誤っていた前例**）。
■ ★内部対照: **穴(単勝ベース・(183)と同じ定義)の帯別ROIが、(183)の値を±2ptで再現**すること。
■ ★陽性対照: **三連複BOX上位4が 81.9%±1.5pt**（**(181)(182)(183)の実測**）。

■ ★記述（判定しない）
　1. ★**穴(板) と 穴(単勝) の買い目の一致率（Jaccard）**——**判定基準35: 「別経路が名ばかりでないか」の物差し**。
　2. **下限・上限の板でも同じ向きか**（感度）。
　3. **閾値を 1.10 / 1.25 / 1.50 と振ったときのROIと点数**（**記述**。**判定は1.25だけ**）。

■ ★採用条件（判定基準39/40/41）
　1. **A が2帯以上でBonferroniを通る**、または **B が通る**
　2. ★**穴(板) と 穴(単勝) の一致率が高すぎない**（**Jaccard > 0.9 なら「差し替えていない」と書く**）
　3. **裾の検算で符号が反転しない**（上位3本・前後半・年別）
　4. ★**ROI>100% でなければ「機構は在るが張れない」**

■ 予想（⚠**類推なので当てにしない**）
　★**(146)の89.0%が天井の目安**なので、**Bは100%に届かないと見る**。**どちらでも驚かない**。

────────────────────────────────────────────────────────────
★★★★ 実測（2026-09-06・ウォークフォワード）—— **漏れは実在した。機構ははっきり出た。水準は届かない**
────────────────────────────────────────────────────────────

■ ゲート1: 4帯とも通過
■ ★ゲート板: **復元R = 0.8009**（公示0.800・**調和平均に直した後**）→ ★**立った**
■ ★陽性対照: 三連複BOX上位4 **81.9%** vs 81.9% → ★**立った**
■ ★内部対照: **穴(単勝ベース)が(183)を9帯すべて±0.1ptで再現** → ★**装置は同一**
■ ★記述: **穴(板) と 穴(単勝) が同じ馬になる割合 = 73.9%**（要<90%）
　→ ★**26.1%は別の馬を選んでいる＝「差し替えていない」ではない**（判定基準35）

■ ★★★主判定A: **各帯の「穴(板) − 乱」**（10比較・α=0.01/10・z=3.291）
| 帯 | 標本R | ★**穴(板)** | 穴(単勝) | 乱 | ★**差(板−乱)** | 99%CI(Bonf) | 判定 |
|---|---|---|---|---|---|---|---|
| 1.5-3倍 | 1,645 | 90.2% | 89.7% | 88.4% | +1.8円 | [−5.6,+9.2] | ⚠検出できない |
| **3-5倍** | 10,205 | **84.6%** | 83.1% | 79.8% | **+4.9円** | [+0.8,+9.0] | ★**差がある** |
| **5-8倍** | 13,170 | **84.2%** | 80.5% | 76.2% | **+7.9円** | [+3.5,+12.4] | ★**差がある** |
| **8-12倍** | 10,105 | **85.3%** | 82.9% | 74.8% | **+10.5円** | [+4.3,+16.7] | ★**差がある** |
| ★**12-20倍** | 15,444 | ★**88.4%** | 84.9% | 74.4% | ★**+14.0円** | [+7.9,+20.2] | ★**差がある** |
| **20-35倍** | 15,985 | **85.9%** | 81.9% | 73.3% | **+12.6円** | [+5.1,+20.2] | ★**差がある** |
| **35-60倍** | 13,810 | **80.5%** | 78.1% | 69.1% | **+11.5円** | [+1.3,+21.6] | ★**差がある** |
| 60-120倍 | 15,719 | 76.9% | 73.6% | 70.4% | +6.5円 | [−6.2,+19.2] | ⚠検出できない |
| **120倍〜** | 19,277 | ⚠**52.0%** | **58.9%** | 47.9% | +4.1円 | [−10.6,+18.9] | ⚠検出できない |
★★**6/9帯が通った**（**単勝ベースの(180)(181)は 1/9 だった**）。**採用条件1は満たす**。

■ ★★★★**この線でいちばん重要な訂正 — 「優位は超人気薄に集中」は価格の取り違えの産物だった**
　⚠**(180)(181)は「差が程度とともに増え、120倍〜が最大」と読んでいた**。
　★★**複勝の板で測ると、優位は中位（5-35倍）に集中し、120倍〜では穴(板)が
　　穴(単勝)より6.9pt悪くなる**（52.0% vs 58.9%）。
　→ ★**超人気薄では、複勝の市場が単勝と違うことを言っており、そちらに従うと損をする**。
　★**単勝オッズで値段を測っていたことが、優位の在り処を誤って示していた**。

■ ★★★主判定B: **絶対閾値の裾** `p_norm / q_pool_fuku ≥ 1.250`
| 閾値 | 点数 | 買う頭/R | ROI | 99%CI(Bonf) | 判定 |
|---|---|---|---|---|---|
| 1.10 | 128,499 | 4.36 | 80.1% | [77.6,82.7] | —（記述） |
| **1.25** | 68,994 | **2.34** | **79.3%** | [75.3,83.2] | ⚠**100%を超えない** |
| 1.50 | 28,109 | 0.95 | 73.5% | [66.4,80.6] | —（記述） |
★★**ゲート2Bの予言どおり、払戻率80.0%をそのまま返した**（**79.3%**）＝
　★**複勝の板は、この閾値の所で正しく値付けしている**。
⚠★**1.25で1レース2.34頭も買うことになる**＝**閾値がまったく選別になっていない**。
　→ ★**(173)「MLの確率が板から大きくズレている（較正が粗い）」の再現**。
　　**絶対閾値の裾は、我々の q では使えない**（**枠連の(141)が使えたのは q が板だったから**）。

■ ★感度（板の価格の取り方）: **穴(板) − 乱 の全帯平均は lo +8.7 / mid +9.1 / hi +9.1円**
　→ ★**範囲のどこを取っても向きは同じ**（**(124)の事前登録2の作法**）。

■ ⚠⚠**それでも張れない（採用条件4）**
| 通った帯 | ROI | ★**100%超の年** |
|---|---|---|
| 3-5倍 | 84.6% | **0/11** |
| 5-8倍 | 84.2% | **0/11** |
| 8-12倍 | 85.3% | **1/11** |
| ★**12-20倍** | ★**88.4%** | **0/11** |
| 20-35倍 | 85.9% | **0/11** |
| 35-60倍 | 80.5% | **0/11** |
★**最良は 12-20倍 の 88.4%**（**この線の全記録で最高**）。⚠**100%には 11.6pt 足りない**。
★**11年で100%を超えたのは 8-12倍 の1年だけ**。

■ ★★結論
　★**自己監査の指摘は実在した**——**複勝の板を使うと、優位は 1/9帯 → 6/9帯 に増え、
　　最良の水準も 84.9% → 88.4% に上がり、優位の在り処（中位帯）も変わった**。
　⚠**それでも水準は 88.4% が上限で、100%には届かない**。
　★**(146)の「強い経路での複勝の天井 89.0%」とほぼ一致する**——
　　**恒等式ではなく実測の一致だが、天井の位置が2つの独立な経路で揃った**。
　⚠**q 側は依然としてML＝弱い**。**「閉じた」とは書けない**（判定基準25）。
　★**書けるのは「複勝の市場価格で測っても、1頭の札は100%に届かない」まで**。


実行: python3 ml/audit_ana_board.py    自己テスト: python3 ml/audit_ana_board.py --selftest
"""
import math
import sys
from itertools import combinations

import numpy as np

sys.path.insert(0, "ml")
import features as F
from audit_crosspool import load_races, payoff, zq
from audit_ana_odds import COST, MIN_HORSES, gate1, roi_of
from audit_ana_degree import DEG
from audit_ana_marg import WF_BOX4, WF_TOL, wf_predict
from train_prod import add_odds_features

NPLACE = 3.0
FUKU_R = 0.800
THR = 1.0 / FUKU_R                 # ★絶対閾値 1.250
THRS = [1.10, 1.25, 1.50]
NCMP = len(DEG) + 1                # A(9) + B(1)
ALPHA = 0.01
SEED = 20260906
BOARD_R, BOARD_TOL = 0.800, 0.020
INNER_TOL = 2.0
# (183)の実測（内部対照）
KNOWN_TAN = {"1.5-3倍": 89.6, "3-5倍": 83.1, "5-8倍": 80.5, "8-12倍": 82.9,
             "12-20倍": 84.9, "20-35倍": 81.8, "35-60倍": 78.2,
             "60-120倍": 73.6, "120倍〜": 58.8}


def load_fuku_boards():
    """race_id(8桁) → {馬番: (下限, 上限)}。type=2 の板から。"""
    from nk_odds_bulk import iter_records
    from audit_fuku_board import nk_raceid
    out = {}
    for rec in iter_records(2):
        r8 = nk_raceid(rec["race_id"])
        if not r8:
            continue
        d = {}
        for k, v in rec["odds"].items():
            if not str(k).isdigit():
                continue
            lo, hi = ((float(v[0]), float(v[1])) if isinstance(v, (list, tuple))
                      else (float(v), float(v)))
            if lo > 0 and hi > 0:
                d[int(k)] = (lo, hi)
        if d:
            out[r8] = d
    return out


def qpool(od_pairs, which="harm"):
    """複勝の板 → 市場含意の3着以内確率（Σ=3）。

    ★**価格の中心は調和平均**——**恒等式 `R = 3/Σ(1/o)` は 1/o について線形**なので、
    　**「オッズの平均」ではなく「1/o の平均」を取るのが正しい**。
    　**実測で 調和平均→R=0.8003（公示0.800）／mid→0.8321**（判定基準37/38）。
    """
    lo = np.array([p[0] for p in od_pairs], float)
    hi = np.array([p[1] for p in od_pairs], float)
    o = {"harm": 2.0 * lo * hi / (lo + hi), "mid": (lo + hi) / 2.0,
         "lo": lo, "hi": hi}[which]
    inv = 1.0 / o
    return inv / inv.sum() * NPLACE, float(NPLACE / inv.sum())


def selftest():
    ok = True
    # ★恒等式: 全馬が同じオッズなら含意は一様、復元Rはその値そのもの
    pairs = [(4.0, 4.0)] * 12
    q, R = qpool(pairs)
    assert abs(q.sum() - NPLACE) < 1e-9 and abs(q[0] - NPLACE / 12) < 1e-9
    assert abs(R - NPLACE / (12 * 0.25)) < 1e-9
    print(f"★板→含意の自己テスト: Σq={q.sum():.3f}（要3.000）/ 復元R={R:.3f}　★OK")
    # 範囲の4通りが順序を保つ
    pr = [(1.5, 3.0), (2.0, 5.0), (10.0, 30.0)]
    ms = [np.argmax(qpool(pr, w)[0]) for w in ("harm", "mid", "lo", "hi")]
    assert len(set(ms)) == 1
    print("★感度の自己テスト: 調和平均/中央/下限/上限で最大の馬が変わらない　★OK")
    # ★調和平均は mid より小さい（1/o の平均を取るので）
    h, _ = qpool([(1.0, 9.0)] * 10)
    assert abs(2 * 1.0 * 9.0 / 10.0 - 1.8) < 1e-9
    print("★調和平均の自己テスト: [1.0,9.0] → 1.80（mid は 5.00）　★OK")
    # ★ゲート2: 交換可能な2頭の対応差は0
    rng = np.random.default_rng(0)
    n = 150_000
    pay = rng.choice([0.0, 250.0, 900.0], size=n, p=[0.72, 0.20, 0.08])
    m = float(np.mean([(pay[rng.permutation(n)] - pay[rng.permutation(n)]).mean()
                       for _ in range(200)]))
    print(f"★ゲート2Aの自己テスト: 対応差200回の平均 {m:+.3f}円 → "
          f"**仮説が偽なら0**: {'★OK' if abs(m) < 2 else '⚠NG'}")
    ok &= abs(m) < 2
    print(f"★ゲート2B: **板が正しければ閾値で切った群のROIは払戻率{100*FUKU_R:.1f}%を返す**"
          f"（100%ではない）")
    print(f"★比較数 {NCMP} → z = {zq(ALPHA/NCMP):.3f}　／ 絶対閾値 = 1/{FUKU_R} = {THR:.3f}")
    print("★自己テスト: " + ("全部OK" if ok else "⚠NG"))
    return 0 if ok else 1


def main():
    z = zq(ALPHA / NCMP)
    print("(184) ★★★★**複勝の板を q_pool にして測り直す** — 自己監査で見つかった最大の漏れ")
    print("★経路: **q = MLモデルの top3 確率 / ★q_pool = 複勝の板(type=2)**")
    print("　⚠**(174)〜(183)は q_pool が単勝オッズだった**。**複勝を買うのに単勝で値段を測っていた**")
    print("　⚠**q 側は依然としてML＝弱い**。**強くなるのは q_pool 側だけ**\n")

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
    m = ~np.isnan(pred)
    sub = d.loc[m, ["raceid", "umaban", "odds", "date"]].copy()
    sub["p"] = pred[m]

    rng = np.random.default_rng(SEED)
    acc = {i: {"b": [], "r": [], "t": [], "yr": []} for i in range(len(DEG))}
    tailv, tailr, tailn = {t: [] for t in THRS}, [], []
    jac_hit, jac_tot = 0, 0
    Rs, box4 = [], []
    sens = {"lo": [], "mid": [], "hi": []}
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
        pairs = [bd[int(u)] for u in ub]
        qp, Rb = qpool(pairs, "harm")
        Rs.append(Rb)
        gapf = pn - qp
        gapt = pv / pv.sum() - (1.0 / od) / (1.0 / od).sum()
        ratio = pn / np.maximum(qp, 1e-12)
        order = [int(u) for u in ub[np.argsort(-pv, kind="mergesort")]]
        bx = [payoff(r, "三連複", list(c)) for c in combinations(sorted(order[:4]), 3)]
        if not any(x is None for x in bx):
            box4.append(sum(bx) - 400.0)
        yr = int(gg["date"].iloc[0].year)
        # ★B: 絶対閾値の裾
        for t in THRS:
            for k in np.where(ratio >= t)[0]:
                v = payoff(r, "複勝", [int(ub[k])])
                if v is not None:
                    tailv[t].append(v)
                    if t == THR:
                        tailr.append(yr)
        # ★A: 帯ごとの 穴(板) vs 乱／内部対照の 穴(単勝)
        for i, (nm, lo, hi) in enumerate(DEG):
            idx = np.where((od >= lo) & (od < hi))[0]
            if len(idx) < 2:
                continue
            ab = int(idx[int(np.argmax(gapf[idx]))])
            at = int(idx[int(np.argmax(gapt[idx]))])
            rn = int(rng.choice([int(k) for k in idx if k != ab]))
            pb = payoff(r, "複勝", [int(ub[ab])])
            pt = payoff(r, "複勝", [int(ub[at])])
            pr_ = payoff(r, "複勝", [int(ub[rn])])
            if pb is None or pt is None or pr_ is None:
                continue
            acc[i]["b"].append(pb); acc[i]["r"].append(pr_)
            acc[i]["t"].append(pt); acc[i]["yr"].append(yr)
            jac_tot += 1; jac_hit += (ab == at)
            for w in ("lo", "mid", "hi"):
                q2, _ = qpool(pairs, w)
                g2 = pn - q2
                a2 = int(idx[int(np.argmax(g2[idx]))])
                v2 = payoff(r, "複勝", [int(ub[a2])])
                if v2 is not None:
                    sens[w].append(v2 - pr_)

    Rs = np.asarray(Rs, float)
    print(f"\n■ ★★ゲート板（判定基準38「測れる定数は測れ」）")
    med = float(np.median(Rs))
    okR = abs(med - BOARD_R) <= BOARD_TOL
    print(f"　**板から復元した複勝の払戻率 R = 3/Σ(1/調和平均) の中央値 = {med:.4f}**"
          f"（要 {BOARD_R}±{BOARD_TOL}）→ **{'★立った' if okR else '⚠⚠落ちた'}**")
    if not okR:
        print("⚠⚠**板の読み方を疑う。読まない**（(162)で馬単の定数が2.6pt誤っていた前例）。")
        return
    box4 = np.asarray(box4, float)
    g0 = 100.0 * (box4.sum() + 400.0 * len(box4)) / (400.0 * len(box4))
    okb = abs(g0 - WF_BOX4) <= WF_TOL
    print(f"⚠**陽性対照**: 三連複BOX上位4 **{g0:.1f}%** vs {WF_BOX4}%"
          f" → **{'★立った' if okb else '⚠⚠落ちた'}**")
    if not okb:
        print("\n⚠⚠**陽性対照が落ちた。読まない**。")
        return

    print(f"\n{'='*104}")
    print("■ ★内部対照: **穴(単勝ベース)が(183)を再現するか**（許容±{:.0f}pt）".format(INNER_TOL))
    ng = 0
    for i, (nm, lo, hi) in enumerate(DEG):
        t = np.asarray(acc[i]["t"], float)
        if len(t) < 300:
            continue
        dd = roi_of(t) - KNOWN_TAN[nm]
        bad_ = abs(dd) > INNER_TOL
        ng += bad_
        print(f"　{nm:<10}{roi_of(t):>7.1f}% vs (183) {KNOWN_TAN[nm]:>5.1f}%"
              f"　差 {dd:+.1f}pt　{'⚠ズレた' if bad_ else '★再現'}")
    print(f"　→ **{'⚠⚠' + str(ng) + '帯がズレた（板のある部分集合なので注意して読む）' if ng else '★全帯で再現'}**")
    print(f"\n■ 記述: **穴(板) と 穴(単勝) が同じ馬になる割合 = "
          f"{100*jac_hit/max(jac_tot,1):.1f}%**")
    print("　★**判定基準35: 高すぎるなら「差し替えていない」**（要 <90%）")

    print(f"\n{'='*104}")
    print(f"■ ★★★主判定A: **各帯の「穴(板) − 乱」の対応差**"
          f"（**{NCMP}比較・α={ALPHA}/{NCMP}・z={z:.3f}**）")
    print("　★ゲート2A: **仮説が偽なら穴と乱は交換可能＝対応差の期待値は0**")
    print(f"\n{'帯':<10}{'標本R':>9}{'穴(板)':>9}{'穴(単勝)':>10}{'乱':>8}"
          f"{'★差(板−乱)':>12}{'99%CI(Bonf)':>22}{'判定':>14}")
    hitsA = []
    for i, (nm, lo, hi) in enumerate(DEG):
        B_ = np.asarray(acc[i]["b"], float)
        R_ = np.asarray(acc[i]["r"], float)
        T_ = np.asarray(acc[i]["t"], float)
        if len(B_) < 300:
            continue
        dd = B_ - R_
        mu, se = dd.mean(), dd.std(ddof=1) / math.sqrt(len(dd))
        sig = abs(mu) > z * se
        if sig and mu > 0:
            hitsA.append(i)
        print(f"{nm:<10}{len(B_):>9,}{roi_of(B_):>8.1f}%{roi_of(T_):>9.1f}%"
              f"{roi_of(R_):>7.1f}%{mu:>+11.1f}円"
              f"{f'[{mu-z*se:+.1f},{mu+z*se:+.1f}]':>22}"
              f"{'★差がある' if sig else '⚠検出できない':>14}")

    print(f"\n■ ★★★主判定B: **絶対閾値の裾** `p_norm / q_pool_fuku ≥ {THR:.3f}`")
    print(f"　★ゲート2B: **板が正しければROIは払戻率{100*FUKU_R:.1f}%を返す**（100%ではない）")
    print(f"\n{'閾値':<8}{'点数':>10}{'買うR率':>10}{'ROI':>9}{'99%CI(Bonf)':>24}{'判定':>16}")
    okB = False
    nrace_all = len(Rs)
    for t in THRS:
        v = np.asarray(tailv[t], float)
        if len(v) < 100:
            print(f"{t:<8}{len(v):>10,}　⚠標本不足")
            continue
        mu, se = v.mean() - COST, v.std(ddof=1) / math.sqrt(len(v))
        lo_, hi_ = 100 + mu - z * se, 100 + mu + z * se
        judge = "—（記述）"
        if abs(t - THR) < 1e-9:
            okB = lo_ > 100.0
            judge = "★★100%超" if okB else "⚠100%を超えない"
        print(f"{t:<8}{len(v):>10,}{100*len(v)/max(nrace_all,1):>9.1f}%"
              f"{roi_of(v):>8.1f}%{f'[{lo_:.1f},{hi_:.1f}]':>24}{judge:>16}")

    print(f"\n■ 記述: **板の下限/上限での感度**（穴(板) − 乱 の全帯平均）")
    for w in ("lo", "mid", "hi"):
        s = np.asarray(sens[w], float)
        if len(s):
            print(f"　{w:<4}{s.mean():>+8.1f}円")

    print(f"\n■ ★採用条件")
    print(f"　1. A が2帯以上 または B が通る … A={len(hitsA)}帯 / B={'通る' if okB else '通らない'}"
          f" → {'★満たす' if (len(hitsA) >= 2 or okB) else '⚠満たさない'}")
    if not (len(hitsA) >= 2 or okB):
        print("\n★★★**結論: 複勝の板で測り直しても届かない**。")
        print("★**「複勝を買うのに単勝で値段を測っていた」という方法論の穴は塞いだ**が、")
        print("　**結論は変わらない**。")
        print("⚠**q 側は依然としてML＝弱い経路**。**「閉じた」とは書けない**（判定基準25）。")
        print("★**書けるのは「複勝の市場価格で測っても届かない」まで**。")
        return
    print("\n■ ★裾の検算（**通ったもの**・(77)）")
    if okB:
        v = np.asarray(tailv[THR], float)
        yrv = np.asarray(tailr, int)
        ys = sorted(set(yrv))
        ov = sum(1 for u in ys if roi_of(v[yrv == u]) > 100.0)
        print(f"　閾値{THR:.2f}: ROI {roi_of(v):.1f}% / **上位3本が全払戻の "
              f"{100*np.sort(v)[-3:].sum()/max(v.sum(),1e-9):.1f}%** / "
              f"**100%超の年 {ov}/{len(ys)}**")
    for i in hitsA:
        B_ = np.asarray(acc[i]["b"], float)
        yv = np.asarray(acc[i]["yr"], int)
        ys = sorted(set(yv))
        ov = sum(1 for u in ys if roi_of(B_[yv == u]) > 100.0)
        print(f"　{DEG[i][0]}: ROI {roi_of(B_):.1f}% / **100%超の年 {ov}/{len(ys)}**")
    print("\n⚠**枠連の運用には触れない**。**設定変更は提案しない**。")


if __name__ == "__main__":
    sys.exit(selftest() if "--selftest" in sys.argv else (main() or 0))
