"""(176) ★★★**オッズを細かく中和したとき、モデルは残余の情報を持つか**

★★経緯（`ANA_TRACK.md` 1.）——
　**(174)** 差 `share−mk` で切った → **機構は在るが張れない**（最良 複勝87.6%・+7.6pt）
　**(175)** 推奨度 `share` で切った → ★**曲線は5帯とも maxT を超えたが、
　　　　　「平均オッズの単調性」が5帯とも ρ=+1.000 で全部説明が付いた**＝**(88)の再発見**
　→ ★**(175)が出した設計の核心**: **オッズ帯5段は粗すぎる**（帯内で 2.6→1.5倍 / 363→69倍）。
　　 **統制は「十分位のあいだで平均オッズが平坦になる粒度」まで細かくすること**。

★★★**そして「比で測り直す」枝は、そのままでは測るに値しない**（**算数で先に分かる**）:
```
ratio = share/mk = (p/Σp)·o·Σ(1/o) ≒ p·o × (Σ(1/o)/Σp)   ← Σ(1/o)≒1.25・Σp≒3 でほぼ定数
      ∝ ★単勝EV (p×o)
```
　→ ★**「比」は単勝EVの単調変換**。**(45)がEV十分位を測っている**（上位83.0%/下位55.5%・
　　 EV≥1.0で83.3%）。**そのまま測れば6回目**（(84)(86)(120)(174)(175)に続く）。
★★★**さらに決定的**: **オッズを細かく統制すると `mk` が固定されるので、
　`share−mk`・`share/mk`・`share` の3つは同じ順位に潰れる**。
　→ ★**「差か比か推奨度か」という問いは、オッズを中和した瞬間に消える**。
　→ ★★**残る本当の問いは1つ**: **「同じオッズの馬どうしを比べたとき、モデルは残余の情報を持つか」**。
　　 **本スクリプトはそれを測る**。**3つの枝を同時に片づける**。

────────────────────────────────────────────────────────────
★★★ 事前登録（2026-09-05・**結果を見る前にコミットする**）
────────────────────────────────────────────────────────────

■ ★経路の明示（判定基準25）
　　p（MLモデルのtop3確率）／ 統制 = **単勝オッズ**
　⚠**モデルは単勝オッズを特徴量に持つ＝弱い経路**。**陰性でも「閉じた」とは書けない**。
　⚠**標本は(174)(175)と同一**（26,583R / 366,754頭）＝**独立な検証ではない**（判定基準35）。
　　→ **水準のCIを(174)(175)と見比べない**。

■ 券種 —— ★**複勝のみ**（**(174)の実測で単勝の検出限界は±11〜38pt**・`ANA_TRACK.md` 4.）

■ ★★統制（**ここが本体**）
　★**単勝オッズの分位で細かいビンに切る。主は NBIN=40**（**記述として20も出す**）。
　★**十分位は「ビンの中で」作り、ビンをまたいで集計する**。
　→ **十分位のあいだで平均オッズが平坦になるので、(175)を殺した交絡が構造的に消える**。

■ ★★ゲート0（**新設・これが通らなければ何も読まない**）
　★**中和の検算**: **十分位の平均オッズが平坦であること**。
　　**max/min < 1.05 かつ |ρ(平均オッズ)| < 0.6**。
　⚠**(175)は max/min が 1.7〜5.3・ρ=+1.000 だった**。**そこが直っていなければ同じ穴に落ちる**。
　→ ★**(175)の失敗を、次の測定の「入口の条件」に変換した**（判定基準29）。

■ ⚠ゲート1（判定基準32）—— **(174)(175)と同じ**（(88)③④を別パーサで再現・許容±3pt）

■ ★★ゲート2（判定基準42）——**仮説が偽なら何を返すか**
　★**主判定A = ビン内十分位に対する複勝ROIの Spearman ρ**。
　★**主判定B = 第10十分位 − 第1十分位 の1点あたり損益差（円）**。
　★**仮説が偽（同じオッズならモデルの p は払戻について何も持たない）なら、
　　p と払戻はビン内で独立になり、A の期待値は 0、B の期待値も 0 を返す**。
　★**帰無分布はビン内で複勝払戻をシャッフルして作る**——**まさにその状態**。
　・**買う頭数は全十分位で同じ**＝**(170)の形にならない**。
　・**絞り込みをしない**（全馬がどれかの十分位に入る）＝**(168)の形にならない**。
　・**A と B の2つを見るので Bonferroni α=0.01/2**。

■ ★記述（判定しない）
　1. ★**ビンの中で `share` / `gap` / `ratio` の十分位割り当てがどれだけ一致するか**
　　 → ★**「3つは同じ順位に潰れる」という上の主張の実測**。**一致率が高ければ枝は3本とも閉じる**
　2. ★**`ratio` と **単勝EV `p·o`** の順位相関** → **「比＝(45)のEV」の実測**
　3. 各十分位の 平均オッズ・モデル1位率・人気中央値・複勝100円率・ROI

■ ★採用条件（判定基準39/40/41）
　1. **A が帰無95%点を超える** かつ **B の99%CI(Bonf)が0を除外**
　2. ★**ゲート0が通っている**（平坦でない統制で出た単調性は読まない）
　3. **裾の検算で符号が反転しない**（上位3本・前後半・年別）
　4. ★**NBIN=20 でも同じ向き**（**ビン数の恣意性に頑健か**。記述だが向きが逆なら不採用）
　5. ★**ROI>100% でなければ「機構は在るが張れない」と書く**（(174)(175)と同じ分岐）

■ ⚠既に測ってあること（**新しいと書く前に差分を1行書く**・判定基準25）
　★**(45)③が「オッズ帯を揃えても全帯で市場を上回る」を測っている**
　　（1-2倍 +0.4 / 3-5 +1.2 / 5-8 +4.1 / 8-15 +3.4 / 15-30 +1.3 / **30-80 +13.4 / 80超 +19.2**）。
　⚠**だが(46)がその帯を細かく割ると −12.9 / +15.6 / +58.4 / −20.4 と暴れた**＝**区分の産物**。
　→ ★**本件の差分は3つ**: **(a) 統制を40分位まで細かくして平坦性を検算する**、
　　 **(b) 券種を複勝にする（唯一検出力がある）**、**(c) 判定を「最良の帯」ではなく単調性にする**。
　　 ⚠**(45)(46)は単勝・粗い帯・最良セルで見ていた**。**同じものではないが、近い**。

■ 予想
　**持たない**。★ただし**(45)③から言えること**は書いておく——
　**上乗せが在るとすれば人気薄側に集中する**（30倍超で+13〜19pt）。
　⚠**そして(46)がその安定性を否定している**。**どちらが出ても驚かない**。

実行: python3 ml/audit_ana_neut.py        自己テスト: python3 ml/audit_ana_neut.py --selftest
"""
import math
import sys

import numpy as np

sys.path.insert(0, "ml")
import features as F
from audit_crosspool import load_races, payoff, zq
from audit_ana_odds import COST, MIN_HORSES, gate1, roi_of
from audit_ana_reco import NDEC, NPERM, SEED, dec_roi, spearman_vs_rank
from train_prod import CAPACITY, add_odds_features, fit_seeds

NBIN_MAIN = 40
NBIN_ALT = 20
NCMP = 2                      # 主判定 A(ρ) と B(円)
ALPHA = 0.01
FLAT_RATIO = 1.05             # ゲート0: 十分位の平均オッズ max/min
FLAT_RHO = 0.6                # ゲート0: |ρ(平均オッズ)|


def within_bin_decile(x, binid, nbin):
    """ビンの中で x の十分位(0..NDEC-1)を作る。"""
    dec = np.empty(len(x), np.int64)
    for b in range(nbin):
        m = binid == b
        if not m.any():
            continue
        v = x[m]
        q = np.quantile(v, np.linspace(0, 1, NDEC + 1)[1:-1])
        dec[m] = np.searchsorted(q, v, side="right")
    return dec


def bin_perm_index(binid, nbin, rng):
    """ビン内だけを並べ替える添字を返す。"""
    key = rng.random(len(binid))
    src = np.lexsort((key, binid))
    dst = np.lexsort((np.arange(len(binid)), binid))
    out = np.empty(len(binid), np.int64)
    out[dst] = src
    return out


def selftest():
    ok = True
    rng = np.random.default_rng(0)
    n = 50_000
    binid = rng.integers(0, 10, n)
    x = rng.random(n)
    dec = within_bin_decile(x, binid, 10)
    # 各ビンで十分位がほぼ均等
    cnt = np.bincount(dec, minlength=NDEC)
    assert cnt.max() / cnt.min() < 1.05
    # ビン内並べ替えはビンを保つ
    idx = bin_perm_index(binid, 10, rng)
    assert (binid[idx] == binid).all()
    # ★ゲート2: ビン内シャッフルなら ρ も 円の差も 0 を返す
    pay = rng.choice([0.0, 800.0], size=n, p=[0.875, 0.125])
    rs, ds = [], []
    for _ in range(200):
        sp = pay[bin_perm_index(binid, 10, rng)]
        cur = dec_roi(sp, dec)
        rs.append(spearman_vs_rank(cur))
        ds.append(sp[dec == NDEC - 1].mean() - sp[dec == 0].mean())
    mr, md = float(np.mean(rs)), float(np.mean(ds))
    print(f"★ゲート2の自己テスト: ビン内シャッフル200回 → ρ の平均 {mr:+.3f} / "
          f"円の差の平均 {md:+.2f}円 → **仮説が偽なら0を返す**: "
          f"{'★OK' if abs(mr) < 0.10 and abs(md) < 15 else '⚠NG'}")
    ok &= abs(mr) < 0.10 and abs(md) < 15
    print(f"★比較数 {NCMP} → z = {zq(ALPHA/NCMP):.3f}")
    print("★自己テスト: " + ("全部OK" if ok else "⚠NG"))
    return 0 if ok else 1


def main():
    z = zq(ALPHA / NCMP)
    print("(176) ★★★**オッズを細かく中和したとき、モデルは残余の情報を持つか**")
    print("★経路（判定基準25）: p = MLモデルのtop3確率 / 統制 = 単勝オッズ")
    print("　⚠**モデルは単勝オッズを特徴量に持つ＝弱い経路**。**陰性でも「閉じた」とは書けない**")
    print("　⚠**標本は(174)(175)と同一＝独立な検証ではない**（判定基準35）\n")

    races = {r["rid"]: r for r in load_races()}
    rows, bad = gate1(list(races.values()))
    print(f"⚠**ゲート1（判定基準32）**: (88)③④を別パーサで再現・許容±3pt")
    for nm, n, roi, known, dd, ok in rows:
        print(f"　{nm:<12}{roi:>7.1f}% vs {known:>5.1f}%　差 {dd:+.1f}pt"
              f"　{'★立った' if ok else '⚠落ちた'}")
    if bad:
        print("\n⚠⚠**ゲート1が落ちた。以降を読まない**。")
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

    keys = ("odds", "share", "gap", "ratio", "ev", "date", "fuku", "top1", "pop")
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
        t1 = np.zeros(len(pv), bool); t1[int(np.argmax(pv))] = True
        rec["odds"].append(od); rec["share"].append(share)
        rec["gap"].append(share - mk)
        rec["ratio"].append(share / np.maximum(mk, 1e-12))
        rec["ev"].append(pv * od)
        rec["date"].append(gg["date"].to_numpy())
        rec["fuku"].append(np.asarray(fv, float))
        rec["top1"].append(t1); rec["pop"].append(np.argsort(np.argsort(od)) + 1)
    for k in keys:
        rec[k] = np.concatenate(rec[k])
    N = len(rec["odds"])
    print(f"★突き合わせ {nrace:,}レース / {N:,}頭　⚠**(174)(175)と同一標本**")

    od, pay = rec["odds"], rec["fuku"]
    years = rec["date"].astype("datetime64[Y]").astype(int) + 1970
    medd = np.median(rec["date"].astype("datetime64[D]").astype(int))
    rng = np.random.default_rng(SEED)

    def run(nbin, label, judge):
        qs = np.quantile(od, np.linspace(0, 1, nbin + 1)[1:-1])
        binid = np.searchsorted(qs, od, side="right")
        dec = within_bin_decile(rec["ratio"], binid, nbin)
        cur = dec_roi(pay, dec)
        mo = np.array([od[dec == i].mean() for i in range(NDEC)])
        flat = mo.max() / mo.min()
        rho_o = spearman_vs_rank(mo)
        rho = spearman_vs_rank(cur)
        print(f"\n{'='*96}\n■ {label}（ビン {nbin} 分位）")
        print(f"{'十分位':<9}" + "".join(f"{i+1:>7}" for i in range(NDEC)))
        print(f"{'  複勝ROI%':<9}" + "".join(f"{v:>7.1f}" for v in cur))
        print(f"{'  平均オッズ':<9}" + "".join(f"{v:>7.1f}" for v in mo))
        print(f"{'  モデル1位率':<9}"
              + "".join(f"{100*rec['top1'][dec==i].mean():>7.1f}" for i in range(NDEC)))
        print(f"{'  人気中央':<9}"
              + "".join(f"{np.median(rec['pop'][dec==i]):>7.0f}" for i in range(NDEC)))
        print(f"\n　⚠**ゲート0（中和の検算）**: 平均オッズ max/min = **{flat:.3f}**"
              f"（要 <{FLAT_RATIO}）・ρ(平均オッズ) = **{rho_o:+.3f}**（要 |ρ|<{FLAT_RHO}）"
              f" → **{'★通った' if flat < FLAT_RATIO and abs(rho_o) < FLAT_RHO else '⚠⚠落ちた'}**")
        print(f"　（⚠**(175)は max/min 1.7〜5.3・ρ=+1.000 だった**）")
        if not judge:
            print(f"　記述: ρ(ROI) = {rho:+.3f} / 第10−第1 = "
                  f"{pay[dec==NDEC-1].mean()-pay[dec==0].mean():+.1f}円")
            return None
        if not (flat < FLAT_RATIO and abs(rho_o) < FLAT_RHO):
            print("\n⚠⚠**ゲート0が落ちた。以降を読まない**（事前登録どおり）。")
            return "gate0"
        hi, lo = pay[dec == NDEC - 1], pay[dec == 0]
        dd = hi.mean() - lo.mean()
        se = math.sqrt(hi.var(ddof=1)/len(hi) + lo.var(ddof=1)/len(lo))
        hw = z * se
        nullr = np.zeros(NPERM)
        nulld = np.zeros(NPERM)
        for t in range(NPERM):
            sp = pay[bin_perm_index(binid, nbin, rng)]
            nullr[t] = spearman_vs_rank(dec_roi(sp, dec))
            nulld[t] = sp[dec == NDEC-1].mean() - sp[dec == 0].mean()
        thr = float(np.quantile(np.abs(nullr), 0.95))
        okA, okB = abs(rho) > thr, abs(dd) > hw
        print(f"\n■ ★★主判定（**α={ALPHA}/{NCMP}・z={z:.3f}**・帰無はビン内シャッフル{NPERM:,}回）")
        print(f"　★ゲート2: **仮説が偽なら p と払戻はビン内で独立＝A も B も 0 を返す**")
        print(f"　**A 単調性 ρ = {rho:+.3f}**　帰無95%点 {thr:+.3f}"
              f" → **{'★超えた' if okA else '⚠超えない'}**")
        print(f"　**B 第10−第1 = {dd:+.1f}円**　99%CI(Bonf) [{dd-hw:+.1f},{dd+hw:+.1f}]"
              f"　帰無の95%点 {float(np.quantile(np.abs(nulld),0.95)):+.1f}円"
              f" → **{'★0を除外' if okB else '⚠検出できない'}**")
        print(f"\n■ ★採用条件")
        print(f"　1. A かつ B → {'★満たす' if okA and okB else '⚠満たさない'}")
        print(f"　2. ゲート0 → ★満たす")
        if not (okA and okB):
            return "neg"
        sel = pay[dec == NDEC - 1]
        dt = rec["date"][dec == NDEC-1].astype("datetime64[D]").astype(int)
        yr = years[dec == NDEC-1]
        ys = sorted(set(yr))
        over = sum(1 for u in ys if roi_of(sel[yr == u]) > 100.0)
        top3 = 100.0 * np.sort(sel)[-3:].sum() / max(sel.sum(), 1e-9)
        print(f"\n■ ★裾の検算（第10十分位・{len(sel):,}頭）")
        print(f"　ROI {roi_of(sel):.1f}% / **上位3本が全払戻の {top3:.1f}%** / "
              f"前半 {roi_of(sel[dt<=medd]):.1f}%・後半 {roi_of(sel[dt>medd]):.1f}% / "
              f"**100%超の年 {over}/{len(ys)}**")
        return "pos"

    # 記述1: 3つの変数の十分位割り当ての一致
    qs = np.quantile(od, np.linspace(0, 1, NBIN_MAIN + 1)[1:-1])
    binid = np.searchsorted(qs, od, side="right")
    ds_ = within_bin_decile(rec["share"], binid, NBIN_MAIN)
    dg_ = within_bin_decile(rec["gap"], binid, NBIN_MAIN)
    dr_ = within_bin_decile(rec["ratio"], binid, NBIN_MAIN)
    print(f"\n■ 記述1: ★**ビン{NBIN_MAIN}分位の中で「差・比・推奨度」の十分位はどれだけ一致するか**")
    print(f"　share と ratio が同じ十分位: **{100*np.mean(ds_==dr_):.1f}%**"
          f"　（|差|≤1 なら {100*np.mean(np.abs(ds_-dr_)<=1):.1f}%）")
    print(f"　gap   と ratio が同じ十分位: **{100*np.mean(dg_==dr_):.1f}%**"
          f"　（|差|≤1 なら {100*np.mean(np.abs(dg_-dr_)<=1):.1f}%）")
    print(f"　share と gap   が同じ十分位: **{100*np.mean(ds_==dg_):.1f}%**")
    print("　→ ★**一致が高いなら「差か比か推奨度か」という問いはオッズ中和で消える**")
    rr = float(np.corrcoef(np.argsort(np.argsort(rec["ratio"])),
                           np.argsort(np.argsort(rec["ev"])))[0, 1])
    print(f"\n■ 記述2: ★**ratio と 単勝EV(p·o) の順位相関 = {rr:+.4f}**"
          f"　→ ★**比＝(45)のEVかどうかの実測**")

    run(NBIN_ALT, f"記述: 感度（ビン{NBIN_ALT}分位）", judge=False)
    res = run(NBIN_MAIN, f"★★主判定（ビン{NBIN_MAIN}分位）", judge=True)
    if res == "neg":
        print("\n★★**結論: 主判定を通らない＝陰性**。")
        print("⚠**経路は弱いので「閉じた」とは書けない**（判定基準25）。")
    elif res == "pos":
        print("\n⚠**ROIが100%を超えなければ運用の候補にならない**（事前登録）。")
    print("⚠**枠連の運用には触れない**。**設定変更は提案しない**。")


if __name__ == "__main__":
    sys.exit(selftest() if "--selftest" in sys.argv else (main() or 0))
