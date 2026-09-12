"""(175) ★★**推奨度そのもの**（モデルのレース内シェア）で1頭の札を切る

★★問い（ユーザー・2026-09-05）——**「推奨の方からやろう」**。
　**(174)は「ズレ（差）」で切った**。**今度は「その馬の推奨度」そのもので切る**。
　⚠**「比 `share/mk` で測り直す」はユーザー判断で保留**（**(175)が陰性ならその時にやる**）。
　★**保留の記録は `ANA_TRACK.md` 3.**。**保留は陰性ではない**。

────────────────────────────────────────────────────────────
★★★ 事前登録（2026-09-05・**結果を見る前にコミットする**）
────────────────────────────────────────────────────────────

■ ★経路の明示（判定基準25・表の前に必ず印字）
　　推奨度 share = p/Σp（**MLモデルのレース内シェア**）
　　統制  オッズ帯（**単勝オッズ**）
　⚠**モデルは単勝オッズを特徴量に持つ**。**弱い経路**。
　→ ★**陰性でも「閉じた」とは書けない**。書けるのは
　　 「**モデルのシェアでは、1頭の札は100%に届かない**」まで。

■ 標本 —— ⚠**(174)と同一**（26,583R / 366,754頭・8頭以上・後70%検証・3シード）
　★★**独立な検証ではない**（**同じレース・同じ払戻**）。**判定基準35**。
　→ ⚠**(174)と水準のCIを見比べてはいけない**。**別々の問いとして読む**。

■ 券種 —— ★**複勝のみ**
　★**(174)の実測で単勝の検出限界は±11〜38pt**、**(88)の市場の誤りの最大は+10pt**。
　→ ★★**単勝は原理的に測れないと確定した**。**検出力を無駄にしない**（`ANA_TRACK.md` 4.）。

■ 切り方
　★**主軸 = 推奨度 `share`**。**各オッズ帯の中で十分位（10区間）**。
　**オッズ帯5段**: 〜3 / 3-6 / 6-15 / 15-50 / 50倍〜
　⚠★**オッズ帯での統制は必須**——**外すと(75)(120)の「残差順＝人気薄順」に戻る**。

■ ★★主判定 —— **「最良のマス」ではなく曲線の単調性**
　★**帯ごとに、share の十分位に対する複勝ROIの Spearman ρ**。**5帯＝5本の曲線**。
　★**帰無分布**: **帯の中で複勝払戻をシャッフル**（1,000回）。
　★**maxT（5本の max|ρ| の分布）で FWER 制御**（**(85)(86)と同じ作法**）。
　⚠**「10区間のどれかが高い」は偶然でも起きるが、「10区間が順に並ぶ」は起きにくい**——
　　**(84)(85)がこの判定量を選んだ理由**。**(86)は「最上位区間だけ高い」を追って消えた**。

■ ★★ゲート2（判定基準42）——**仮説が偽なら何を返すか**
　★**主判定は ρ（十分位の順序と複勝ROIの順位相関）**。
　★**仮説が偽（推奨度が帯内で払戻について何も持たない）なら、share と払戻は帯内で独立になり、
　　ρ の期待値は 0 を返す**。**シャッフル帰無分布がまさにその状態を作る**。
　・**買う頭数は全十分位で同じ**（帯の10%ずつ）＝**(170)の「頭数が減れば必ず良くなる量」ではない**。
　・**絞り込みをしない**（帯の全馬がどれかの十分位に入る）＝**(168)の形にならない**。

■ ⚠ゲート1（判定基準32）—— **(174)と同じ**。落ちたら何も読まない。
　**モデル不使用・全レース全馬の単勝ベタ買いROIで(88)③④を再現**（許容±3pt）:
　**1.3-1.4倍 88.9% / 10-30倍 82.3% / 100-200倍 54.9% / 200倍超 37.1%**
　（**(174)の実測は 88.1 / 82.4 / 54.4 / 38.2 で4帯とも通っている**）

■ ★★★最重要の対照 —— **平均オッズの単調性**
　⚠**帯には幅があるので、帯の中でも share が高い馬は低オッズ側に寄る**。
　★**(88)は帯の中でも低オッズ側ほど良い**ので、**ROIの単調性は「オッズの単調性」でも作れる**。
　→ ★★**各十分位の平均オッズを必ず併記し、次を判定に入れる**:
　　 **ROIの単調性が平均オッズの単調性で説明できるなら、それは(88)の再発見であって新しくない**。
　　 ⚠**これを入れないと、(174)の50倍〜(81.6倍 vs 210.6倍)と同じ読めないマスを作る**。

■ ⚠既に一度出ている（**陽性対照でもあり、危険でもある**）
　★**(85)で「複勝 top1 × 指数A（軸のシェア）」が ρ=+0.685 > 帰無95%点 0.600** を超えている。
　→ ★**同じ向きが出たら、それは(85)の再現であって新発見ではない**。
　⚠**(85)は「軸」「レース選別」、本件は「全馬」「馬の選別」**なので**同一ではない**が、
　　**近い**。**「新しい」と書く前に(85)との違いを1行書くこと**（判定基準25）。

■ ★採用条件（判定基準39/40/41）——**有意なだけでは足りない**
　1. **5帯のうち3帯以上**で ρ が **maxT の帰無分布95%点**を超える
　2. ★**平均オッズの単調性で説明されない**（上の対照）
　3. **裾の検算で符号が反転しない**（上位3本・前後半・年別）
　4. ★**ROI>100% でなければ「機構は在るが張れない」と書く**（**(174)と同じ分岐**）

■ 記述（**判定しない**）
　・★**帯内での `share` / `gap = share−mk` / `ratio = share/mk` の相互相関**
　　→ ★**「比でやり直す」（保留中）の判断材料**。**|ρ|>0.9 なら2軸は成立しない**
　・各十分位の**平均オッズ・モデル1位率・人気中央値・複勝100円率**

■ 予想
　**持たない**。★ただし**(120)から言えること**は書いておく——
　**(120)は「モデル−市場」型が逆効果だと示したが、「モデル」型((117)枠連スコア)は効いていた**。
　**推奨度は「モデル」型**なので、**(120)の逆効果の対象ではない**。

────────────────────────────────────────────────────────────
★★★ 実測（2026-09-05・26,583レース / 366,754頭・⚠(174)と同一標本）
────────────────────────────────────────────────────────────

■ ゲート1: 4帯とも通過（88.1 / 82.4 / 54.4 / 38.2 ＝ (174)と同値）

■ 帯 × 推奨度の十分位（**複勝ROI%**）
| 帯 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | ρ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 〜3倍 | 81.8 | 83.7 | 83.0 | 84.8 | 86.7 | 84.9 | 86.3 | 89.3 | 90.1 | **95.3** | +0.952 |
| 3-6倍 | 75.3 | 77.6 | 80.8 | 78.2 | 83.1 | 82.9 | 84.2 | 86.7 | 84.3 | **87.7** | +0.964 |
| 6-15倍 | 73.9 | 76.3 | 75.2 | 78.8 | 76.4 | 81.4 | 82.7 | 80.8 | 81.9 | **84.6** | +0.915 |
| 15-50倍 | 73.2 | 74.3 | 73.1 | 78.1 | 78.6 | 77.8 | 78.6 | 80.6 | **84.3** | 80.6 | +0.891 |
| 50倍〜 | **34.9** | 34.9 | 41.8 | 61.1 | 62.1 | 68.7 | 67.7 | 72.3 | 74.1 | **80.8** | +0.976 |

■ ★★主判定: **5帯すべてが maxT を超えた**（**帰無95%点 +0.794**・シャッフル1,000回）
　⚠**単独の帰無95%点は +0.624〜+0.661**＝**(84)が使った閾値0.6は、単独でも甘い**。
　　★**自己テストでも「帰無のもとで |ρ|>0.6 が10.5%出る」ことを確認した**。**maxTが要る**。

■ ★★★**それを対照が全部殺した** —— **平均オッズの単調性 ρ = +1.000（5帯すべて）**
| 帯 | 十分位1→10 の平均オッズ | 読み |
|---|---|---|
| 〜3倍 | 2.6 → **1.5倍** | ⚠**(88)の再発見と区別できない** |
| 3-6倍 | 5.1 → **3.6倍** | ⚠同上 |
| 6-15倍 | 12.1 → **7.6倍** | ⚠同上 |
| 15-50倍 | 39.1 → **19.9倍** | ⚠同上 |
| 50倍〜 | **363.5 → 69.2倍** | ⚠同上（★**帯が広すぎる**） |
★★**採用条件2は 0/5 帯**。→ ★**陰性**。
　**帯の中でも「推奨度が高い馬」＝「その帯の低オッズ側の馬」**なので、
　**十分位曲線は favorite-longshot bias をなぞっているだけ**。**新しい入口ではない**。
⚠★**曲線が5帯とも綺麗に出たのに全部説明が付いてしまう**——**対照を先に登録していなければ
　「5帯すべてで単調・maxT通過」と書いていた**。**判定基準32/37の効き目がそのまま出た例**。

■ ★最良マスも既知だった
　**〜3倍 第10十分位 = 95.3%**（**平均1.5倍・モデル1位率99.2%**）＝**ほぼ「複勝 top1 の断然人気」**。
　★**(26)(106)(110)で「最低配当100円の床」の天井は複勝96.8%と測ってある**。**95.3%はその内側**。
　→ **新しい発見ではなく、既知の床の再確認**。
　⚠**100%超の年は5マスとも 0/10**。**裾も健全**（上位3本が全払戻の0.2〜1.7%・前後半の割れ無し）。

■ ★★★副産物（**次の枝への青信号**）: **帯内での変数どうしの相関**
| 帯 | share−gap | ★**share−ratio** | gap−ratio |
|---|---|---|---|
| 〜3倍 | −0.248 | ★**+0.169** | +0.895 |
| 3-6倍 | +0.392 | ★**+0.539** | +0.979 |
| 6-15倍 | +0.640 | ★**+0.551** | +0.983 |
| 15-50倍 | +0.803 | ★**+0.563** | +0.925 |
| 50倍〜 | +0.893 | ★**+0.516** | +0.810 |
★★**`ratio = share/mk` は `share` と最大でも +0.563**＝**事前登録の「|ρ|>0.9なら成立しない」を
　大きく下回る**。→ ★**保留していた「比で測り直す」は、推奨度の言い換えではない**。**別の量**。
　**(119)(134)(156)(157)で4回起きた「新変数が(112)の言い換えだった」形には、今回は入らない**。
⚠**ただし gap と ratio は +0.810〜+0.983 でほぼ同じ**——**(174)の差と比は近い**。
　→ ★**「比でやり直す」は、(174)の焼き直しになる危険がある**。**設計で分ける必要がある**。

■ ★★次に測るなら（**設計の核心・ここが今回いちばんの学び**）
　⚠**オッズ帯5段では粗すぎる**。**帯の中でも平均オッズが 2.6→1.5 / 363→69倍 と動く**。
　→ ★★**統制は「十分位のあいだで平均オッズが平坦になる粒度」まで細かくすること**。
　　 **具体案: 単勝オッズの細かいビン（20〜40区分）の中で切り、ビンをまたいで集計する**。
　　 ★**平均オッズの ρ を判定に入れたまま、それが平坦になって初めて中身を読む**。
　⚠**この統制を入れずに測ると、何を変数にしても ρ≒+0.9 が出る**——**今回それを実演した**。


実行: python3 ml/audit_ana_reco.py        自己テスト: python3 ml/audit_ana_reco.py --selftest
"""
import math
import sys

import numpy as np

sys.path.insert(0, "ml")
import features as F
from audit_crosspool import load_races, payoff, zq
from audit_ana_odds import BANDS, COST, MIN_HORSES, band_of, gate1, roi_of
from train_prod import CAPACITY, add_odds_features, fit_seeds

NDEC = 10
NPERM = 1000
SEED = 20260905


def spearman_vs_rank(v):
    """十分位の順序(1..k) と値 v の順位相関。kが小さいので素朴に。"""
    k = len(v)
    r = np.argsort(np.argsort(v)) + 1.0
    x = np.arange(1.0, k + 1)
    return float(np.corrcoef(x, r)[0, 1])


def dec_roi(pay, dec, k=NDEC):
    """十分位ごとのROI[%]。"""
    s = np.bincount(dec, weights=pay, minlength=k)
    n = np.bincount(dec, minlength=k)
    return 100.0 * s / np.maximum(n * COST, 1e-9)


def selftest():
    ok = True
    # spearman: 完全単調なら +1 / 逆なら −1
    assert abs(spearman_vs_rank(np.arange(10.0)) - 1.0) < 1e-9
    assert abs(spearman_vs_rank(-np.arange(10.0)) + 1.0) < 1e-9
    # dec_roi: 100円買いで払戻が全部120円なら 120%
    d = np.repeat(np.arange(NDEC), 5)
    assert abs(dec_roi(np.full(len(d), 120.0), d).mean() - 120.0) < 1e-9
    # ★ゲート2の性質: 払戻を十分位内でシャッフルすると ρ は0のまわりに散る
    rng = np.random.default_rng(0)
    n = 200_000
    dec = rng.integers(0, NDEC, n)
    pay = rng.choice([0.0, 800.0], size=n, p=[0.875, 0.125])
    rs = [spearman_vs_rank(dec_roi(rng.permutation(pay), dec)) for _ in range(200)]
    m = float(np.mean(rs))
    hit = float(np.mean(np.abs(rs) > 0.6))
    print(f"★ゲート2の自己テスト: シャッフル200回の ρ の平均 {m:+.3f}"
          f"（|ρ|>0.6 の割合 {100*hit:.1f}%）→ **仮説が偽なら0を返す**: "
          f"{'★OK' if abs(m) < 0.10 else '⚠NG'}")
    ok &= abs(m) < 0.10
    print("★自己テスト: " + ("全部OK" if ok else "⚠NG"))
    return 0 if ok else 1


def banner():
    print("★経路（判定基準25）:")
    print("　　推奨度 share = p/Σp（MLモデルのレース内シェア）")
    print("　　統制  オッズ帯（単勝オッズ）")
    print("　⚠**モデルは単勝オッズを特徴量に持つ＝弱い経路**。**陰性でも「閉じた」とは書けない**")
    print("　⚠**標本は(174)と同一＝独立な検証ではない**（判定基準35）。"
          "**水準のCIを(174)と見比べない**\n")


def main():
    print("(175) ★★**推奨度そのもの**（モデルのレース内シェア）で1頭の札を切る")
    banner()
    races = {r["rid"]: r for r in load_races()}
    print(f"配当A {len(races):,}レース")

    rows, bad = gate1(list(races.values()))
    print(f"\n⚠**ゲート1（判定基準32）**: (88)③④を別パーサで再現・許容±3pt")
    print(f"{'帯':<12}{'頭数':>10}{'実測ROI':>10}{'(88)':>9}{'差':>9}{'判定':>8}")
    for nm, n, roi, known, dd, ok in rows:
        print(f"{nm:<12}{n:>10,}{roi:>9.1f}%{known:>8.1f}%{dd:>+8.1f}pt"
              f"{'★立った' if ok else '⚠落ちた':>8}")
    if bad:
        print("\n⚠⚠**ゲート1が落ちた。以降を読まない**（事前登録どおり）。")
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

    keys = ("odds", "share", "gap", "ratio", "date", "fuku", "top1", "pop")
    rec = {k: [] for k in keys}
    nrace = 0
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
        if not np.isfinite(od).all() or (od <= 0).any() or pv.sum() <= 0:
            continue
        share = pv / pv.sum()
        mk = (1.0 / od) / (1.0 / od).sum()
        fv, ok = [], True
        for u in gg["umaban"].astype(int):
            b = payoff(r, "複勝", [int(u)])
            if b is None:
                ok = False
                break
            fv.append(b)
        if not ok:
            continue
        nrace += 1
        t1 = np.zeros(len(pv), bool)
        t1[int(np.argmax(pv))] = True
        rec["odds"].append(od); rec["share"].append(share)
        rec["gap"].append(share - mk); rec["ratio"].append(share / np.maximum(mk, 1e-12))
        rec["date"].append(gg["date"].to_numpy()); rec["fuku"].append(np.asarray(fv, float))
        rec["top1"].append(t1); rec["pop"].append(np.argsort(np.argsort(od)) + 1)
    for k in keys:
        rec[k] = np.concatenate(rec[k])
    N = len(rec["odds"])
    print(f"★突き合わせ {nrace:,}レース / {N:,}頭　⚠**(174)と同一標本**\n")

    bidx = np.array([band_of(o, BANDS) for o in rec["odds"]])
    years = rec["date"].astype("datetime64[Y]").astype(int) + 1970
    med = np.median(rec["date"].astype("datetime64[D]").astype(int))
    rng = np.random.default_rng(SEED)

    # ── 記述: 帯内の共線性（★「比でやり直す」の判断材料） ──
    print("■ 記述: ★**帯の中での share / gap / ratio の相互相関**"
          "（**|ρ|>0.9 なら2軸は成立しない**）")
    print(f"{'帯':<9}{'頭数':>9}{'share-gap':>12}{'share-ratio':>13}{'gap-ratio':>12}")
    for bi, (bn, lo, hi) in enumerate(BANDS):
        m = bidx == bi
        if m.sum() < 100:
            continue
        s, g_, r_ = rec["share"][m], rec["gap"][m], rec["ratio"][m]
        def rho(a, b):
            return float(np.corrcoef(np.argsort(np.argsort(a)),
                                     np.argsort(np.argsort(b)))[0, 1])
        print(f"{bn:<9}{m.sum():>9,}{rho(s,g_):>+12.3f}{rho(s,r_):>+13.3f}"
              f"{rho(g_,r_):>+12.3f}")

    # ── 主判定 ──
    obs, curves, decs = {}, {}, {}
    for bi, (bn, lo, hi) in enumerate(BANDS):
        m = bidx == bi
        if m.sum() < 1000:
            continue
        s = rec["share"][m]
        q = np.quantile(s, np.linspace(0, 1, NDEC + 1)[1:-1])
        dec = np.searchsorted(q, s, side="right")
        pay = rec["fuku"][m]
        cur = dec_roi(pay, dec)
        obs[bn] = spearman_vs_rank(cur)
        curves[bn] = cur
        decs[bn] = (m, dec, pay)

    print(f"\n■ 記述: **帯 × 推奨度の十分位**（複勝ROI%・下段は平均オッズ）")
    print(f"{'帯':<9}" + "".join(f"{i+1:>7}" for i in range(NDEC)) + f"{'ρ':>8}")
    for bn in obs:
        m, dec, pay = decs[bn]
        print(f"{bn:<9}" + "".join(f"{v:>7.1f}" for v in curves[bn])
              + f"{obs[bn]:>+8.3f}")
        od = rec["odds"][m]
        mo = [od[dec == i].mean() for i in range(NDEC)]
        t1 = [100 * rec["top1"][m][dec == i].mean() for i in range(NDEC)]
        print(f"{'  平均オッズ':<9}" + "".join(f"{v:>7.1f}" for v in mo))
        print(f"{'  モデル1位率':<9}" + "".join(f"{v:>7.1f}" for v in t1))

    # ── maxT 帰無分布 ──
    print(f"\n■ ★★主判定: **十分位曲線の単調性 ρ**（**帯内で複勝払戻をシャッフル"
          f"{NPERM:,}回・maxTでFWER制御**）")
    print("　★ゲート2: **仮説が偽なら share と払戻は帯内で独立＝ρ の期待値は0**")
    names = list(obs)
    maxnull = np.zeros(NPERM)
    pernull = {bn: np.zeros(NPERM) for bn in names}
    for t in range(NPERM):
        mx = 0.0
        for bn in names:
            m, dec, pay = decs[bn]
            r = spearman_vs_rank(dec_roi(rng.permutation(pay), dec))
            pernull[bn][t] = r
            mx = max(mx, abs(r))
        maxnull[t] = mx
    thr = float(np.quantile(maxnull, 0.95))
    print(f"\n　★**maxT の帰無95%点 = {thr:+.3f}**（これを超えた曲線だけ読む）")
    print(f"\n{'帯':<9}{'ρ(実測)':>10}{'帰無95%(単独)':>15}{'maxT95%':>10}{'判定':>16}")
    npass = 0
    for bn in names:
        s95 = float(np.quantile(np.abs(pernull[bn]), 0.95))
        ok = abs(obs[bn]) > thr
        npass += ok and obs[bn] > 0
        print(f"{bn:<9}{obs[bn]:>+10.3f}{s95:>+15.3f}{thr:>+10.3f}"
              f"{'★超えた' if ok else '⚠超えない':>16}")

    # ── ★最重要の対照: 平均オッズの単調性 ──
    print(f"\n■ ★★★最重要の対照: **ROIの単調性は「平均オッズの単調性」で説明できないか**")
    print(f"{'帯':<9}{'ρ(ROI)':>10}{'ρ(平均オッズ)':>15}{'読み':>34}")
    nclean = 0
    for bn in names:
        m, dec, pay = decs[bn]
        od = rec["odds"][m]
        mo = np.array([od[dec == i].mean() for i in range(NDEC)])
        ro = -spearman_vs_rank(mo)     # 低オッズ側ほど良い＝(88)。符号を揃える
        clean = abs(obs[bn]) > thr and obs[bn] > 0 and ro <= 0.6
        nclean += clean
        v = ("⚠**(88)の再発見と区別できない**" if abs(obs[bn]) > thr and ro > 0.6
             else "★オッズでは説明できない" if abs(obs[bn]) > thr
             else "—")
        print(f"{bn:<9}{obs[bn]:>+10.3f}{ro:>+15.3f}{v:>34}")

    print(f"\n■ ★採用条件（判定基準39/40/41）")
    print(f"　1. 5帯のうち3帯以上が maxT を正に超える … **{npass}帯** →"
          f" {'★満たす' if npass >= 3 else '⚠満たさない'}")
    print(f"　2. ★**平均オッズの単調性で説明されない** … **{nclean}帯** →"
          f" {'★満たす' if nclean >= 3 else '⚠満たさない'}")
    if npass < 3 or nclean < 3:
        why = ("採用条件1を満たさない" if npass < 3
               else "★**採用条件2を満たさない＝ROIの単調性が平均オッズの単調性と区別できない**")
        print(f"\n★★**結論: {why}＝陰性**。")
        if nclean < 3 <= npass:
            print("⚠★**曲線そのものは出た（5帯とも maxT を超えた）が、"
                  "帯の中でも share が高い馬は低オッズ側に寄っているだけ**——")
            print("　**(88)の favorite-longshot bias の再発見であって、新しい入口ではない**。")
            print("★**次に測るなら、オッズの統制を「平均オッズが十分位で平坦になる粒度」まで"
                  "細かくすること**（`ANA_TRACK.md` 4.）。")
        print("⚠**経路は弱いので「閉じた」とは書けない**（判定基準25）。")
        print("　書けるのは「**モデルのシェアでは、1頭の札は100%に届かない**」まで。")
        print("★**保留していた「比 share/mk で測り直す」を再開する条件が満たされた**"
              "（`ANA_TRACK.md` 3.）。")
        return

    print(f"\n■ ★裾の検算（(77)・maxTを超えた帯の第10十分位）")
    for bn in names:
        if not (abs(obs[bn]) > thr and obs[bn] > 0):
            continue
        m, dec, pay = decs[bn]
        sel = pay[dec == NDEC - 1]
        top3 = 100.0 * np.sort(sel)[-3:].sum() / max(sel.sum(), 1e-9)
        dt = rec["date"][m][dec == NDEC - 1].astype("datetime64[D]").astype(int)
        yr = years[m][dec == NDEC - 1]
        ys = sorted(set(yr))
        over = sum(1 for u in ys if roi_of(sel[yr == u]) > 100.0)
        print(f"　{bn} 第10十分位: ROI {roi_of(sel):.1f}% / "
              f"**上位3本が全払戻の {top3:.1f}%** / "
              f"前半 {roi_of(sel[dt<=med]):.1f}%・後半 {roi_of(sel[dt>med]):.1f}% / "
              f"**100%超の年 {over}/{len(ys)}**")
    print("\n⚠**ROIが100%を超えなければ運用の候補にならない**（事前登録）。")
    print("⚠**枠連の運用には触れない**。**設定変更は提案しない**。")


if __name__ == "__main__":
    sys.exit(selftest() if "--selftest" in sys.argv else (main() or 0))
