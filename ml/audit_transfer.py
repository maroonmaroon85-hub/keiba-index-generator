"""(166) ★★「中央に一度も出ていない3歳以上の馬」＝地方/海外からの移籍馬は、ROIを下げているか

★★ユーザーの問い（2026-08-29）:「地方馬がわからないのか。中央に参加する地方馬のデータも集めようか」
　**集める前に、集めて得られる利得の上限を測る**。**これが陰性なら収集は要らない**。

★診断（当日の実例・50頭の空欄の内訳）
　**2歳 38頭（76%）= ほぼ本当の初出走。集めても埋まらない**
　★**3歳以上 12頭（24%）= 中央に13年半で一度も出ていない＝地方/海外の可能性が高い**
　　（例: 中京4R マイコンプリート **単勝1.7倍の1番人気**にモデルが値を持てていない）

★★★事前登録（**測る前に書いている**）

⚠**いちばん危ない交絡を先に潰す**: **アーカイブは2013-01-05始まり**なので、
　**2013〜2016年のレースでは「それ以前に中央で走っていた馬」が偽の移籍馬に見える**。
　→ ★**馬齢から初出走年を逆算し、`レース年 −(馬齢−2) < 2014` の馬は判定から外す**。
　　 **この guard が無いと、古い年ほど移籍馬が多い、という完全な偽信号が出る**。

　★①**移籍候補の量**: 何レースに何頭出るか。**その市場シェアの分布**。
　★★②**シェア区分別ROI**（枠連・紐1・除外40%）。**主判定はここ**。
　　 ⚠**区分は事前に固定する**: **0% / 0超〜5% / 5〜15% / 15%超**。
　　 　**(165)で十分位が潰れた**（変数の過半がゼロ）**教訓を反映**——**分布を見てから切らない**。
　★③**市場は移籍馬を正しく価格付けしているか**: **移籍候補の実際の複勝圏内率 vs 市場含意**。
　　 ★**ここが要**——**市場が正しいなら、我々の欠損は「順位付けを誤る」形でしか効かない**。
　　 **市場も間違えているなら、そこは逆に取れる場所になりうる**。

判定
　⚠**ゲート（判定基準32）**: **除外0%の全体が84.5%を±2.5ptで再現しなければ何も読まない**。
　★**主判定**: **②でシェア15%超の区分のROIが、0%の区分より有意に低い**（Bonferroni α=0.01/3）。
　★**利得の上限**: **移籍候補を含むレースを全部落としたときのROI改善**。
　　 ⚠**これは「上限」であって達成可能な値ではない**——**データを集めても、
　　 　当てられるようになるだけで、落とすほどの改善は出ない**。**上限が小さければ収集は不要**。

⚠**予想は持たない**。**(165)では向きは合っていたが何も確立しなかった**。

★★★実測（2026-08-29・ゲート: 除外0%の全体 85.3% vs 既知84.5%＝+0.8pt → 立った）
　guard で 10,095頭を除外（アーカイブ開始前デビューと推定）。突き合わせ 25,719レース。
| | 結果 |
|---|---|
| ①量 | 移籍候補が1頭以上いるレース **17.8%**／シェア中央値0.0%・90%点2.0%・99%点21.4% |
| ②ROI | 0%:87.4% / 0-5%:85.3% / 5-15%:71.9% / **15%超:90.8%**（単調でない） |
| 　主判定 | 15%超 − 候補なし の1R損益差 **+3.4円・99%CI[−37.7,+44.6]** → **差は検出できない** |
| ★★③ | **実際の複勝圏内率 9.2% vs 市場含意 9.2%・差+0.1pt**（7,944頭） |
| 利得の上限 | 候補を含むレースを丸ごと落として **86.6% → 87.4%**（17.8%のレースを失う） |

★★★**結論: 地方馬データの収集は不要**。**理由は「効果が小さい」ではなく
　「市場が既に正しく価格付けしていて、取るものが無い」**（③）。
　**我々の欠損は「レース内の順位付けを誤る」形でしか効かず、②でそれが検出できない**。
　⚠**(89)以降の「特徴量を足してもオッズが吸収する」の5回目と同じ形**——
　　**今回は被覆の問題という別カテゴリだったのに、同じ結果になった**。

⚠**③の弱点を明記する**: **「市場含意」は単勝シェア×3の素朴な近似**で、**人気薄側で偏る**。
　**9.2%と9.2%の一致は近似の偏りを含む**。**厳密にやるならλ補正Harville**（プロジェクトにある）。
　★**ただし②が独立に陰性で上限も+0.8ptなので、③を精密化しても結論は動かない**。

◇**残った観察（お金にはならないが事実）**: **2026-08-29 中京4Rは単勝1.7倍の1番人気に
　モデルが値を持てていない**。**気持ち悪いが、そこにお金は無い**。

実行: python3 ml/audit_transfer.py
"""
import math
import sys

import numpy as np

sys.path.insert(0, "ml")
import features as F
from audit_crosspool import load_races, payoff, zq
from audit_crosspool2 import realized
from train_prod import CAPACITY, add_odds_features, fit_seeds
from waku_umatan import bracket_probs, waku_of, waku_score, wakuren_buy

EXCL = 0.40
KNOWN, TOL = 84.5, 2.5
NCMP = 3
ARCH_Y0 = 2013                      # アーカイブ開始年
GUARD = ARCH_Y0 + 1                 # 初出走年がこれ未満の馬は判定から外す
BANDS = [(0.0, 1e-9, "0%（移籍候補なし）"), (1e-9, 0.05, "0超〜5%"),
         (0.05, 0.15, "5〜15%"), (0.15, 9.9, "★15%超")]


def show(name, prof, cost, z, extra=""):
    n = len(prof)
    if n < 30 or cost.sum() <= 0:
        print(f"{name:>18}   （30レース未満）")
        return None
    se = prof.std(ddof=1) / math.sqrt(n) * n / cost.sum() * 100.0
    roi = 100.0 * (prof.sum() + cost.sum()) / cost.sum()
    print(f"{name:>18}{n:>9,}{roi:>9.1f}%"
          f"{f'[{roi-z*se:.1f},{roi+z*se:.1f}]':>19}{prof.mean():>+11.1f}円 {extra}")
    return roi, prof.mean(), prof.std(ddof=1) / math.sqrt(n)


def main():
    MODEL_DIR, PAR = CAPACITY["l2"]
    d0 = F.to_model(F.load_files())
    f = F.build_features(d0)
    keep = (f["n_prior"] >= 1) & d0["odds"].notna() & (d0["odds"] > 0)

    # ★レース単位で「移籍候補（3歳以上・アーカイブ初出現）」の市場シェアを作る
    yr = d0["date"].dt.year.to_numpy()
    age = d0["age"].to_numpy(float)
    debut = yr - (age - 2)                       # 初出走年の推定
    cand = (~keep.to_numpy()) & (age >= 3) & (debut >= GUARD)
    guard_out = (~keep.to_numpy()) & (age >= 3) & (debut < GUARD)
    allsum, candsum, candn = {}, {}, {}
    hit_n = hit_top3 = 0
    hit_mkt = 0.0
    fin = d0["finish"].to_numpy(float)
    for rid, o, c, ff in zip(d0["raceid"].to_numpy(), d0["odds"].to_numpy(float),
                             cand, fin):
        if not np.isfinite(o) or o <= 0:
            continue
        allsum[rid] = allsum.get(rid, 0.0) + 1.0 / o
        if c:
            candsum[rid] = candsum.get(rid, 0.0) + 1.0 / o
            candn[rid] = candn.get(rid, 0) + 1

    d, f = d0[keep].reset_index(drop=True), f[keep].reset_index(drop=True)
    y = (d["finish"] <= 3).astype(int).to_numpy()
    fx, _ = F.encode_categoricals(f)
    fx = add_odds_features(fx, d["odds"].to_numpy(float), d["raceid"].to_numpy())
    cut = d["date"].quantile(0.3)
    tr, te = (d["date"] < cut).to_numpy(), (d["date"] >= cut).to_numpy()
    print("(166) ★移籍候補（中央初出現・3歳以上）はROIを下げているか")
    print(f"　学習 {tr.sum():,} / 検証 {te.sum():,}・分割 {cut.date()}")
    print(f"　⚠guard: 初出走年 < {GUARD} と推定される馬は**アーカイブ開始前デビュー**として除外"
          f"（{int(guard_out.sum()):,}頭）")
    ms = fit_seeds(fx[tr], y[tr], 3, PAR)
    p_ml = np.mean([m.predict_proba(fx[te])[:, 1] for m in ms], axis=0)
    sub = d.loc[te, ["raceid", "umaban"]].copy()
    sub["p"] = p_ml

    # ③ 移籍候補そのものの成績（検証期間のレースだけ）
    te_rids = set(sub["raceid"].astype(str))
    for rid, o, c, ff in zip(d0["raceid"].astype(str).to_numpy(),
                             d0["odds"].to_numpy(float), cand, fin):
        if not c or str(rid) not in te_rids or not np.isfinite(o) or o <= 0:
            continue
        tot = allsum.get(rid, 0.0)
        if tot <= 0:
            continue
        hit_n += 1
        hit_top3 += int(np.isfinite(ff) and ff <= 3)
        hit_mkt += (1.0 / o) / tot * 3.0        # 市場含意の複勝圏内率（近似: シェア×3）

    races = {r["rid"]: r for r in load_races()}
    rows = []
    for rid, g in sub.groupby("raceid"):
        r = races.get(str(rid))
        if r is None or not r.get("wakuren"):
            continue
        rl = realized(r)
        if rl is None:
            continue
        a, b, _ = rl
        n, nums = r["n"], [u for u, _, _ in r["horses"]]
        if a not in nums or b not in nums:
            continue
        gg = g.sort_values("p", ascending=False)
        order = [int(u) for u in gg["umaban"].tolist()]
        pv = gg["p"].to_numpy(float)
        if len(order) < 3:
            continue
        key = tuple(sorted((waku_of(a, n), waku_of(b, n))))
        v = payoff(r, "枠連(人気順)", [key[0], key[1]])
        if not v or v <= 0:
            continue
        q = pv / pv.sum()
        bp = bracket_probs(order, q, n)
        sc = float(waku_score(wakuren_buy(order, n, 2), bp))
        pairs = wakuren_buy(order, n, 1)
        c = 100.0 * len(pairs)
        tot = allsum.get(rid, 0.0)
        share = (candsum.get(rid, 0.0) / tot) if tot > 0 else 0.0
        rows.append((sc, (v if key in pairs else 0.0) - c, c, share,
                     candn.get(rid, 0)))

    if not rows:
        sys.exit("突き合わせできたレースが無い")
    A = np.array([[x[1], x[2], x[3], x[4]] for x in rows], float)
    sc = np.array([x[0] for x in rows])
    z = zq(0.01 / NCMP)
    print(f"　突き合わせ {len(rows):,}レース\n")

    g0 = 100.0 * (A[:, 0].sum() + A[:, 1].sum()) / A[:, 1].sum()
    ok = abs(g0 - KNOWN) <= TOL
    print(f"⚠ゲート: 除外0%の全体 **{g0:.1f}%** vs 既知 {KNOWN}%　差 {g0-KNOWN:+.1f}pt"
          f" → **{'★立った' if ok else '⚠⚠落ちた'}**")
    if not ok:
        print("⚠⚠**落ちた。以下を読まない**。")
        return

    thr = np.quantile(sc, EXCL)
    m0 = sc >= thr
    sh, cn = A[:, 2], A[:, 3]
    print(f"\n■ ★①移籍候補の量（除外{EXCL:.0%}後・{int(m0.sum()):,}レース）")
    print(f"　1頭以上いるレース **{100*(cn[m0]>0).mean():.1f}%**"
          f"／2頭以上 {100*(cn[m0]>1).mean():.1f}%")
    print(f"　市場シェアの分位: 中央値 {100*np.median(sh[m0]):.1f}%"
          f"／90%点 {100*np.quantile(sh[m0],.9):.1f}%"
          f"／99%点 {100*np.quantile(sh[m0],.99):.1f}%")

    print("\n■ ★★②シェア区分別ROI（**区分は事前に固定・分布を見てから切っていない**）")
    print(f"{'':>18}{'レース':>9}{'ROI':>10}{'99%CI(Bonf)':>19}{'1R期待損益':>11}")
    res = {}
    for lo, hi, nm in BANDS:
        m = m0 & (sh >= lo) & (sh < hi)
        res[nm] = show(nm, A[m, 0], A[m, 1], z)

    base = res.get("0%（移籍候補なし）")
    top = res.get("★15%超")
    if base and top:
        dd = top[1] - base[1]
        sd = math.hypot(base[2], top[2])
        print(f"\n　★主判定: 15%超 − 移籍候補なし の1R損益差 **{dd:+.1f}円**"
              f"　99%CI [{dd-z*sd:+.1f},{dd+z*sd:+.1f}]")
        print(f"　→ **{'⚠差がある' if abs(dd) > z*sd else '★差は検出できない'}**")

    mm = m0 & (cn == 0)
    r_drop = 100.0 * (A[mm, 0].sum() + A[mm, 1].sum()) / A[mm, 1].sum()
    print(f"\n　★**利得の上限**: 移籍候補を含むレースを全部落とすと"
          f" **{100*(A[m0,0].sum()+A[m0,1].sum())/A[m0,1].sum():.1f}% → {r_drop:.1f}%**"
          f"（{100*(1-mm.sum()/m0.sum()):.1f}%のレースを失う）")
    print("　⚠**これは上限であって達成可能な値ではない**——**集めても当てられるようになるだけ**。")

    print("\n■ ★③市場は移籍候補を正しく価格付けしているか（検証期間・{:,}頭）".format(hit_n))
    if hit_n:
        print(f"　実際の複勝圏内率 **{100*hit_top3/hit_n:.1f}%**"
              f"　vs 市場含意 **{100*hit_mkt/hit_n:.1f}%**"
              f"　差 **{100*(hit_top3-hit_mkt)/hit_n:+.1f}pt**")
        print("　★**市場が正しければ、我々の欠損は「順位付けを誤る」形でしか効かない**。")
        print("　⚠**市場も外していれば、そこは逆に取れる場所になりうる**（別の検定が要る）。")

    print("\n" + "=" * 92)
    print("★読み方（**事前登録のとおり**）")
    print("  ★**上限が小さければ、地方馬データの収集は要らない**。**そこが今日の判断材料**。")
    print("  ⚠**②で差が出ても、(112)の裾と同じ信号かを確かめるまで新しい選択変数と呼ばない**。")


if __name__ == "__main__":
    main()
