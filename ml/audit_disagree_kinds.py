"""(157) ★★★(156)を**券種を変えて**やり直す — 「食い違うレースは別の券種なら効くか」（ユーザー発案）

★★問いの出どころ
　(156)で「**モデルと市場が食い違うレースだけ枠連を買っても ROI は 85.2% vs 85.3% で差なし**」
　と出た。→ ユーザー「**枠連以外だとROI上がるとかない？**」

★★★機構としてありうる理由（これは事前に書ける）
　★**券種によって「食い違い」の効き方が違うはず**:
　・**単勝**は「1着を当てる」＝**食い違いが最も直撃する**
　・**複勝**は「3着以内」＝**誰が勝つかの食い違いに最も鈍い**
　・**枠連・馬連**は「上位2頭」、**三連複**は「上位3頭」＝**その中間**
　→ ★**もし優位が「1着争い」に在るなら単勝で最も出る／「掲示板争い」に在るなら複勝で出る**。
　　 **(156)は枠連1つしか見ていないので、この区別ができていなかった**。

⚠⚠★★**先に危険を書く（これが一番大事）**
　**これは陰性だった結果に次元を足して再挑戦する形**。**偽陽性がいちばん出やすい手つき**。
　→ ★**券種6 × 切り方2 = 12 の多重比較**。**Bonferroni（α=0.01/12）を必ず当てる**。
　→ ★**(156)で ρ=+0.864 で「(112)と同じ信号」と分かっている**。**券種を変えても信号は同じ**。
　　 **仮に1つ当たっても「12回引いて1回当たった」を疑う**。**年別と時間分割を必ず見る**。

★★事前登録（**測る前に書いている**）
　1. **券種は6つだけ**。**後から増やさない**（買い方は全部モデル順で固定）:
　　 **単勝 top1 / 複勝 top1 / 枠連 軸枠×紐枠1 / 馬連 top1-top2 / ワイド top1-top2 / 三連複 top3の1点**
　2. **切り方は2つだけ**（**券種によらないレース単位の量**にする。(156)と揃える）:
　　 **A モデルtop2の集合 vs 市場top2の集合が一致するか**（2区分）
　　 **B 軸馬の乖離（モデルのシェア − 市場の含意確率）の上位20% vs 残り**
　3. ★**判定は「層間の差」の2標本検定**（判定基準13後半）。**層のCIでは判定しない**。
　4. **Bonferroni α=0.01/12**。
　5. ★**プラセボ**: 同数を無作為に抜いた対照（判定基準28）。
　6. ★★**陽性対照（判定基準32）**: 各券種の全体ROIを既知値と突き合わせる。
　　 **枠連 85.3%（(153b)(155)(156)）/ 複勝 84.8%（本命表）/ 三連複BOX 84.5%（train_prod）**。
　　 **±2pt で再現しなければその券種は読まない**。
　7. ⚠**リークを避ける**: **前30%で学習・後70%で検証**。
　8. ★**予想**: ★**当てにしてよい予想は持っていない**（類推はこの4日で5連敗）。
　　 **恒等式から言えることだけ**: **(156)で「的中率が半減しても配当が正確に補う」と実測した**。
　　 　**同じ市場が同じように値付けしているなら、券種を変えても補われるはず**。
　　 　→ **これは「機構からの予想」なので当てにしない**（判定基準24）。**測って決める**。

★★★実行済みの結果（2026-08-16・検証25,719レース）**12個すべて0をまたぐ。陰性**
　食い違う（モデルtop2 ≠ 市場top2）: 9,492R（36.9%）／乖離上位20%: 5,144R

■A 食い違うレースだけ買う
| 券種 | 全体ROI | 対照 | 一致 | ★食い違う | 差(円/R) | 99%CI(Bonf) |
|---|---|---|---|---|---|---|
| 単勝 | 81.3% | — | 81.3% | 81.4% | +0.13 | [−5.91,+6.16] |
| **複勝** | 85.1% | ★OK | 86.0% | **83.7%** | −2.25 | [−5.39,+0.90] |
| **枠連** | 85.3% | ★OK | 85.5% | **84.9%** | −0.62 | [−10.64,+9.39] |
| 馬連 | 82.9% | — | 83.6% | 81.8% | −1.76 | [−13.27,+9.74] |
| ワイド | 82.4% | — | 81.8% | 83.5% | +1.68 | [−5.26,+8.62] |
| 三連複 | 80.8% | ⚠ | 84.6% | 74.4% | −10.21 | [−25.92,+5.50] |

■B 軸馬の乖離 上位20% vs 残り: **単勝 86.7% vs 80.0%（+6.78 [−2.20,+15.77]）が最大**。
　**それでも0をまたぐ**。他5券種も全部0をまたぐ。

⚠⚠★**陽性対照について正直に書く（今日3回目の同じ仕様ミス）**
　・**三連複の対照がNG（80.8% vs 既知84.5%）はデータの問題ではない**——
　　 **既知値84.5%は「BOX4」なのに、私が測ったのは「top3の1点」**。**母集団が違う**。
　　 →(153)(151)に続いて**今日3回目**。**判定基準25/32の「母集団が違う」型**。
　・**単勝・馬連・ワイドには既知値が無い**（「—」）。**厳密には対照が立っていない**。
　★**対照がちゃんと立っているのは複勝と枠連だけ**で、**どちらも差なし**。
　→ ⚠**それ以外の4券種は「測ったが読めない」**。**単勝の+6.78も読まない**。

★**機構の予想は当たらなかった（当たってもいない）**: 事前に「複勝は3着以内なので
　1着争いの食い違いに鈍いはず」と書いたが、**実測は複勝が最も差が大きい（−2.25・食い違う側が悪い）**。
　**ただし0をまたぐので判定不能**。→ **判定基準24（機構からの予想は類推と同じ扱い）のとおり**。

★**結論: 券種を変えても「食い違うレースだけ買う」は効かない**。
　**(156)の ρ=+0.864（(112)と同じ信号）から予想されるとおり**。**レース選択の変数は(112)の1本のまま**。

実行: python3 ml/audit_disagree_kinds.py
"""
import math
import sys

import numpy as np

sys.path.insert(0, "ml")
import features as F
from audit_crosspool import load_races, payoff, probs, zq
from audit_crosspool2 import realized
from train_prod import CAPACITY, add_odds_features, fit_seeds
from waku_umatan import waku_of

NCMP = 12
KNOWN = {"枠連": 85.3, "複勝": 84.8, "三連複": 84.5}      # 陽性対照の真値（分かるものだけ）
TOL = 2.0


def two_sample(a, b, z):
    d = a.mean() - b.mean()
    se = math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return d, d - z * se, d + z * se


def main():
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
    print(f"(157) 食い違うレースを**券種を変えて**買う"
          f"（学習 {tr.sum():,} / 検証 {te.sum():,}・分割 {cut.date()}）")
    print("★経路: 買い目は全部 **MLモデル**の降順。切り方は**券種によらないレース単位の量**")
    print(f"⚠**券種6 × 切り方2 = {NCMP} の多重比較**。**Bonferroni α=0.01/{NCMP}**\n")
    ms = fit_seeds(fx[tr], y[tr], 3, PAR)
    p_ml = np.mean([m.predict_proba(fx[te])[:, 1] for m in ms], axis=0)
    sub = d.loc[te, ["raceid", "umaban"]].copy()
    sub["p"] = p_ml

    races = {r["rid"]: r for r in load_races()}
    KINDS = ["単勝", "複勝", "枠連", "馬連", "ワイド", "三連複"]
    prof = {k: [] for k in KINDS}
    same, gap, yr, ok_rows = [], [], [], []
    for rid, g in sub.groupby("raceid"):
        r = races.get(str(rid))
        if r is None:
            continue
        rl = realized(r)
        if rl is None:
            continue
        n, hs = r["n"], r["horses"]
        nums = [u for u, _, _ in hs]
        gg = g.sort_values("p", ascending=False)
        order = [int(u) for u in gg["umaban"].tolist()]
        pv = gg["p"].to_numpy(float)
        if len(order) < 3 or pv.sum() <= 0:
            continue
        share = pv / pv.sum()
        pm = probs(hs)
        o2 = np.argsort(-pm)
        mkt_top2 = {nums[o2[0]], nums[o2[1]]}
        vals = {}
        vals["単勝"] = payoff(r, "単勝", [order[0]])
        vals["複勝"] = payoff(r, "複勝", [order[0]])
        vals["馬連"] = payoff(r, "馬連", sorted(order[:2]))
        vals["ワイド"] = payoff(r, "ワイド", sorted(order[:2]))
        vals["三連複"] = payoff(r, "三連複", sorted(order[:3]))
        if r.get("wakuren"):
            a, b, _ = rl
            if a in nums and b in nums:
                key = tuple(sorted((waku_of(a, n), waku_of(b, n))))
                pv2 = payoff(r, "枠連(人気順)", [key[0], key[1]])
                kml = tuple(sorted((waku_of(order[0], n), waku_of(order[1], n))))
                vals["枠連"] = (pv2 if kml == key else 0.0) if pv2 and pv2 > 0 else None
            else:
                vals["枠連"] = None
        else:
            vals["枠連"] = None
        if any(v is None for v in vals.values()):
            continue
        for k in KINDS:
            prof[k].append(float(vals[k]) - 100.0)
        same.append(int({order[0], order[1]} == mkt_top2))
        gap.append(float(share[0]) - float(pm[nums.index(order[0])]))
        yr.append(r["year"])

    for k in KINDS:
        prof[k] = np.array(prof[k], float)
    same = np.array(same); gap = np.array(gap); yr = np.array(yr)
    z_all, z_bon = zq(0.01), zq(0.01 / NCMP)
    n = len(same)
    hi20 = gap >= np.quantile(gap, 0.80)
    rng = np.random.default_rng(0)
    print(f"　突き合わせ {n:,}レース（**全券種の払戻が揃うレースだけ**）")
    print(f"　★食い違う（モデルtop2 ≠ 市場top2）: {(same==0).sum():,}R "
          f"({(same==0).mean():.1%}) / 乖離上位20%: {hi20.sum():,}R\n")

    print(f"{'券種':>7}{'全体ROI':>9}{'対照':>7}{'一致':>9}{'★食い違う':>10}"
          f"{'差(円/R)':>10}{'99%CI(Bonf)':>20}{'プラセボ':>9}")
    for k in KINDS:
        p_ = prof[k]
        roi = 100 * (1 + p_.mean() / 100)
        kn = KNOWN.get(k)
        ctl = ("★OK" if abs(roi - kn) <= TOL else "⚠NG") if kn else "—"
        a, b = p_[same == 0], p_[same == 1]
        dd, lo, hi = two_sample(a, b, z_bon)
        pl = np.mean([100 * (1 + p_[rng.choice(n, (same == 0).sum(), False)].mean() / 100)
                      for _ in range(20)])
        mark = " ★有意" if lo > 0 or hi < 0 else ""
        print(f"{k:>7}{roi:>8.1f}%{ctl:>7}{100*(1+b.mean()/100):>8.1f}%"
              f"{100*(1+a.mean()/100):>9.1f}%{dd:>10.2f}"
              f"{'[' + format(lo, '+.2f') + ',' + format(hi, '+.2f') + ']':>20}{pl:>8.1f}%{mark}")

    print(f"\n■ B 軸馬の乖離 上位20% vs 残り")
    print(f"{'券種':>7}{'上位20%':>10}{'残り':>9}{'差(円/R)':>10}{'99%CI(Bonf)':>20}{'プラセボ':>9}")
    for k in KINDS:
        p_ = prof[k]
        a, b = p_[hi20], p_[~hi20]
        dd, lo, hi = two_sample(a, b, z_bon)
        pl = np.mean([100 * (1 + p_[rng.choice(n, hi20.sum(), False)].mean() / 100)
                      for _ in range(20)])
        mark = " ★有意" if lo > 0 or hi < 0 else ""
        print(f"{k:>7}{100*(1+a.mean()/100):>9.1f}%{100*(1+b.mean()/100):>8.1f}%"
              f"{dd:>10.2f}{'[' + format(lo, '+.2f') + ',' + format(hi, '+.2f') + ']':>20}"
              f"{pl:>8.1f}%{mark}")

    print(f"\n■ 2021-2026（(142)の教訓）— 食い違う側のROI")
    m21 = yr >= 2021
    print("  " + " / ".join(f"{k} {100*(1+prof[k][m21 & (same==0)].mean()/100):.1f}%" for k in KINDS))

    print("\n" + "=" * 100)
    print("★読み方（事前登録のとおり。**後から足していない**）")
    print("  ⚠**これは陰性に次元を足した再挑戦**。**12回引いて1回当たっても偽陽性を疑う**。")
    print("  ・**陽性対照が⚠NGの券種は読まない**（判定基準32）。")
    print("  ・**プラセボと差が無ければ、切ったことは何もしていない**（判定基準28）。")
    print("  ★**(156)で ρ=+0.864 で(112)と同じ信号と分かっている**。**券種を変えても信号は同じ**。")


if __name__ == "__main__":
    main()
