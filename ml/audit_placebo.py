"""C1: ★プラセボ対照 — **モデルの優位は「技能」か「市場順からズレたこと自体の効果」か**。

★これが未検証なのは本プロジェクト最大の穴かもしれない。
(69)〜(78)を通じて主張はずっと「モデル順は人気順より+3pt高い」だが、
**モデル順は人気順を"ある量だけ"崩した順序**でもある（買い目の46.7%は人気順と完全一致・(69)①）。
(72)はさらに「寄与は100%不一致側から来る／的中率は人気順より低く、的中時配当が高い」と測っている。
＝**「市場順から離れると、当たりにくくなる代わりに配当が上がる」**という形。

だとすると次の可能性が排除できていない:
  **人気順をランダムに揺らしただけでも同じ+3ptが出るのではないか**
  （配当分布は右に長い裾を持つので、ズレ方によっては期待回収額が動きうる）。
もしそうならモデルは**馬を選べていない**ことになり、(69)(72)(77)(78)の「優位」は全部
「市場順からの逸脱量」の関数でしかない。

**対照の作り方**（(74)のプラセボ対照と同じ発想を、本命の主張そのものに当てる）:
  1. 人気順（単勝オッズ昇順）の log(オッズ) に N(0, σ) のノイズを足して並べ替える
  2. σ を掃引し、**買い目が人気順と一致する率がモデルと同じになる σ** を見つける
     ＝「モデルと同じだけ市場順から離れた、中身が空の順序」
  3. その σ でのROIを、モデルのROIと比べる

判定:
  ・プラセボが人気順と同じ ⇒ **ズレただけでは何も起きない。モデルの+3ptは技能**
  ・プラセボもモデルと同じだけ上がる ⇒ **モデルは馬を選べていない。主張は全面的に見直し**

実行: python3 ml/audit_placebo.py [シード数(既定3)]
"""
import itertools
import sys
import warnings

import numpy as np
import pandas as pd
import lightgbm as lgb

warnings.filterwarnings("ignore")
sys.path.insert(0, "ml")
import features as F
from _cache import load_cached
from place_wide import PARAMS
from pocket_eval import load_payout_a
from waku_umatan import load_wu, waku_of

PAYOUT = "data/payout/a.csv"
SIGMAS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0]
N_DRAW = 5          # 各σにつきノイズを引き直す回数


def wakuren_cs(nums, n):
    wa = waku_of(nums[0], n)
    return sorted({tuple(sorted((wa, waku_of(h, n)))) for h in nums[1:3]})


def boot(x, rng, n=2000):
    b = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return np.percentile(b, 2.5) * 100, np.percentile(b, 97.5) * 100


def main():
    n_seed = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    d, fx = load_cached()
    y = (d["finish"] <= 3).astype(int).to_numpy()
    cut = d["date"].quantile(0.3)
    tr, te = (d["date"] < cut).to_numpy(), (d["date"] >= cut).to_numpy()
    print(f"train {tr.sum():,} / test {te.sum():,}（分割 {cut.date()}）  シード{n_seed}本")
    ms = [lgb.LGBMClassifier(random_state=s, **PARAMS).fit(fx[tr], y[tr], categorical_feature=F.CAT_COLS)
          for s in range(n_seed)]
    p = np.mean([m.predict_proba(fx[te])[:, 1] for m in ms], axis=0)

    sub = d.loc[te, ["raceid", "umaban", "fieldsize", "odds"]].copy()
    sub["p"] = p
    wu, pa = load_wu(PAYOUT), load_payout_a(PAYOUT)

    # レースごとに必要なものを一度だけ取り出す（σ掃引で何度も回すため）
    races = []
    for rid, g in sub.groupby("raceid", sort=False):
        w, s3 = wu.get(rid), pa.get(rid)
        if not ((w and w["wakuren"] and len(g) >= 3) or (s3 and s3["sanrenpuku"] and len(g) >= 9)):
            continue
        races.append({
            "n": int(g["fieldsize"].iloc[0]),
            "uma": g["umaban"].astype(int).to_numpy(),
            "logodds": np.log(g["odds"].to_numpy(float)),
            "p": g["p"].to_numpy(float),
            "wk": w["wakuren"] if (w and w["wakuren"] and len(g) >= 3) else None,
            "s3": s3["sanrenpuku"] if (s3 and s3["sanrenpuku"] and len(g) >= 9) else None,
        })
    print(f"評価対象 {len(races):,}レース")

    def run(order_key):
        """order_key(race) -> 並べ替え後の馬番配列 を受け取り、券種ごとの結果を返す。"""
        w_x, s_x, w_same, s_same = [], [], [], []
        for r in races:
            nums = order_key(r)
            if r["wk"] is not None:
                cs = wakuren_cs(nums, r["n"])
                w_x.append(sum(r["wk"].get(c, 0) for c in cs) / (len(cs) * 100))
                w_same.append(cs == r["pop_wk"])
            if r["s3"] is not None:
                cs = [tuple(sorted(c)) for c in itertools.combinations(nums[:4], 3)]
                s_x.append(sum(r["s3"].get(c, 0) for c in cs) / 400)
                s_same.append(set(cs) == r["pop_s3"])
        return (np.array(w_x), np.array(s_x), np.mean(w_same), np.mean(s_same))

    # 人気順（基準）
    for r in races:
        o = r["uma"][np.argsort(r["logodds"], kind="mergesort")]
        r["pop_order"] = o
        r["pop_wk"] = wakuren_cs(o, r["n"]) if r["wk"] is not None else None
        r["pop_s3"] = set(tuple(sorted(c)) for c in itertools.combinations(o[:4], 3)) if r["s3"] is not None else None

    pw, ps, _, _ = run(lambda r: r["pop_order"])
    mw, msr, mw_same, ms_same = run(lambda r: r["uma"][np.argsort(-r["p"], kind="mergesort")])
    rng = np.random.default_rng(0)

    print(f"\n{'='*96}")
    print(f"人気順        枠連 {pw.mean()*100:.2f}%   三連複BOX4 {ps.mean()*100:.2f}%")
    print(f"モデル順      枠連 {mw.mean()*100:.2f}% ({(mw-pw).mean()*100:+.2f}pt)   "
          f"三連複BOX4 {msr.mean()*100:.2f}% ({(msr-ps).mean()*100:+.2f}pt)")
    print(f"★モデルの買い目が人気順と完全一致する率:  枠連 {mw_same*100:.1f}%   三連複 {ms_same*100:.1f}%")

    print(f"\n★プラセボ: 人気順の log(オッズ) に N(0,σ) を足して並べ替えただけの順序（中身は空）")
    print(f"{'σ':<8}{'一致率(枠連)':>14}{'枠連ROI':>11}{'対人気順':>11}"
          f"{'一致率(三連複)':>16}{'三連複ROI':>11}{'対人気順':>11}")
    curve = []
    for sg in SIGMAS:
        ws, ss, wsame, ssame = [], [], [], []
        for k in range(N_DRAW if sg > 0 else 1):
            g = np.random.default_rng(1000 + k)
            a, b, c, e = run(lambda r: r["uma"][np.argsort(
                r["logodds"] + g.normal(0, sg, len(r["uma"])), kind="mergesort")])
            ws.append(a.mean()); ss.append(b.mean()); wsame.append(c); ssame.append(e)
        curve.append((sg, np.mean(wsame), np.mean(ws), np.mean(ssame), np.mean(ss)))
        print(f"{sg:<8.2f}{np.mean(wsame)*100:>13.1f}%{np.mean(ws)*100:>10.2f}%"
              f"{(np.mean(ws)-pw.mean())*100:>+10.2f}pt"
              f"{np.mean(ssame)*100:>15.1f}%{np.mean(ss)*100:>10.2f}%"
              f"{(np.mean(ss)-ps.mean())*100:>+10.2f}pt")

    # モデルと一致率が同じになる σ を線形補間で求め、そこを直接比較する
    print(f"\n★モデルと『市場順からの離れ方』を揃えた比較")
    for tag, same_target, model_roi, pop_roi, ix in [
            ("枠連 軸枠×紐枠2", mw_same, mw, pw, 1), ("三連複 BOX上位4", ms_same, msr, ps, 3)]:
        xs = np.array([c[ix] for c in curve])
        sig = np.array([c[0] for c in curve])
        j = int(np.argmin(np.abs(xs - same_target)))
        sg = float(np.interp(same_target, xs[::-1], sig[::-1]))
        gg = np.random.default_rng(77)
        acc = []
        for k in range(N_DRAW):
            g2 = np.random.default_rng(2000 + k)
            a, b, c, e = run(lambda r: r["uma"][np.argsort(
                r["logodds"] + g2.normal(0, sg, len(r["uma"])), kind="mergesort")])
            acc.append(a if ix == 1 else b)
        pl = np.mean(acc, axis=0)
        lo, hi = boot(model_roi - pl, gg)
        print(f"  {tag}: モデルの一致率 {same_target*100:.1f}% に合う σ={sg:.3f}"
              f"（最も近い掃引点 σ={sig[j]:.2f}・一致率 {xs[j]*100:.1f}%）")
        print(f"    人気順 {pop_roi.mean()*100:6.2f}%  /  プラセボ {pl.mean()*100:6.2f}%"
              f"  /  モデル {model_roi.mean()*100:6.2f}%")
        print(f"    ★モデル − プラセボ = {(model_roi-pl).mean()*100:+.2f}pt  CI[{lo:+.2f},{hi:+.2f}]"
              f"   {'→ 技能あり' if lo > 0 else '→ ★ズレただけで説明できる（技能とは言えない）'}")


if __name__ == "__main__":
    main()
