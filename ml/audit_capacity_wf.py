"""
(81) ★容量の梯子をウォークフォワードで伸ばす — 「設定を選ぶ問題」ではなく「容量の方向」なのかを見る。

(80)で4設定を実運用の手続き（各年をそれ以前の全データで学習）で測ったところ、
**容量順にきれいに単調**だった:
    leaves15/100本 81.48% → leaves15/400本 84.02% → 現行 leaves31/400本 84.85%
    → leaves63/mc30/1000本 86.54%
(80)は「累積ROIで選ぶ規則」を検証したが、**候補セットに(53)の後知恵最良が入っている**ので
フェアな検証にならない。単調なら、これは選択の問題ではなく**方向**の話になる。

そこで**梯子を上下に伸ばす**。★判定は単純:
  ・上に伸ばしても単調が続く → 「容量を上げるほど回収率が上がる」が実装できる方向になる。
    HANDOFFの「現行構成（leaves31/400本）がROIでは最良」は**訂正が必要**。
  ・どこかで頭打ち/反転する → 山の頂点が最適容量。その位置は事前に選べないので(53)の留保のまま。

★これは HANDOFF の「★5回繰り返された現象（モデルを"より正しく"するとROIが悪くなる）」の
　裏返しにあたる。過学習させるほど市場から離れ、市場から離れること自体に(79)①の効果がある。
　**だから単調が続いても不思議ではない**が、(79)①よりズレは山型（離れすぎると壊れる）なので
　どこかで反転するはず。**その位置を測るのがこのスクリプトの目的**。

⚠容量を上げると学習時間が急に伸びる。leaves255/2000本は1年あたり数分かかる。

実行: python3 ml/audit_capacity_wf.py [シード数(既定1)] [開始年(既定2019)] [段をカンマ区切りで絞る]
  例: python3 ml/audit_capacity_wf.py 3 2019 L2,L5   … 現行とL5だけ3シードで確かめ直す
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

# ★容量の梯子。葉数×本数を「モデルが表現できる複雑さ」の代理として並べる。
#   (80)で測った4点のうち両端と現行を残し、**上に3段**足した。
LADDER = [
    ("L1  leaves15/mc100/100本", dict(PARAMS, num_leaves=15, n_estimators=100)),
    ("L2  leaves31/mc100/400本（現行）", dict(PARAMS)),
    ("L3  leaves63/mc30/1000本", dict(PARAMS, num_leaves=63, min_child_samples=30,
                                     n_estimators=1000)),
    ("L4  leaves127/mc20/1500本", dict(PARAMS, num_leaves=127, min_child_samples=20,
                                      n_estimators=1500)),
    ("L5  leaves255/mc10/2000本", dict(PARAMS, num_leaves=255, min_child_samples=10,
                                      n_estimators=2000)),
    ("L6  leaves511/mc5/2500本", dict(PARAMS, num_leaves=511, min_child_samples=5,
                                     n_estimators=2500)),
]


def wakuren_cs(nums, n):
    wa = waku_of(nums[0], n)
    return sorted({tuple(sorted((wa, waku_of(h, n)))) for h in nums[1:3]})


def boot(x, rng, n=2000):
    b = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return np.percentile(b, 2.5) * 100, np.percentile(b, 97.5) * 100


def auc(yy, s):
    o = np.argsort(s, kind="mergesort")
    r = np.empty(len(s), float)
    r[o] = np.arange(1, len(s) + 1)
    n1, n0 = yy.sum(), len(yy) - yy.sum()
    return (r[yy == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) if n1 and n0 else float("nan")


def main():
    n_seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    y0 = int(sys.argv[2]) if len(sys.argv) > 2 else 2019
    only = set((sys.argv[3] if len(sys.argv) > 3 else "").split(",")) - {""}
    ladder = [(nm, par) for nm, par in LADDER if not only or nm.split()[0] in only]
    d, fx = load_cached()
    y = (d["finish"] <= 3).astype(int).to_numpy()
    year = d["date"].dt.year.to_numpy()
    wu, pa = load_wu(PAYOUT), load_payout_a(PAYOUT)
    years = [yy for yy in range(y0, int(year.max()) + 1) if (year == yy).sum() > 5000]
    print(f"容量の梯子 × ウォークフォワード（{years[0]}〜{years[-1]}・シード{n_seed}本）")
    print("※各年Yは「Yより前の全データ」で学習＝実運用の手続き。"
          "評価年を絞ってあるのは上位の設定が重いため\n")

    acc = {nm: {"w": [], "s": []} for nm, _ in LADDER}
    aucs = {nm: [] for nm, _ in LADDER}
    popw, pops = [], []
    for yy in years:
        tr, te = year < yy, year == yy
        sub0 = d.loc[te, ["raceid", "umaban", "fieldsize", "odds"]].copy()
        for nm, par in ladder:
            ps = [lgb.LGBMClassifier(random_state=s, **par)
                  .fit(fx[tr], y[tr], categorical_feature=F.CAT_COLS)
                  .predict_proba(fx[te])[:, 1] for s in range(n_seed)]
            sub0[nm] = np.mean(ps, axis=0)
            aucs[nm].append(auc(y[te], sub0[nm].to_numpy()))
        for rid, g0 in sub0.groupby("raceid", sort=False):
            w, s3 = wu.get(rid), pa.get(rid)
            okw = w and w["wakuren"] and len(g0) >= 3
            oks = s3 and s3["sanrenpuku"] and len(g0) >= 9
            if not (okw or oks):
                continue
            n = int(g0["fieldsize"].iloc[0])
            uma = g0["umaban"].astype(int).to_numpy()
            po = uma[np.argsort(g0["odds"].to_numpy(float), kind="mergesort")]
            if okw:
                cs = wakuren_cs(po, n)
                popw.append(sum(w["wakuren"].get(c, 0) for c in cs) / (len(cs) * 100))
            if oks:
                cs = [tuple(sorted(c)) for c in itertools.combinations(po[:4], 3)]
                pops.append(sum(s3["sanrenpuku"].get(c, 0) for c in cs) / 400)
            for nm, _ in ladder:
                mo = uma[np.argsort(-g0[nm].to_numpy(float), kind="mergesort")]
                if okw:
                    cs = wakuren_cs(mo, n)
                    acc[nm]["w"].append(sum(w["wakuren"].get(c, 0) for c in cs) / (len(cs) * 100))
                if oks:
                    cs = [tuple(sorted(c)) for c in itertools.combinations(mo[:4], 3)]
                    acc[nm]["s"].append(sum(s3["sanrenpuku"].get(c, 0) for c in cs) / 400)
        print(f"  {yy} 完了  この年までの累積 枠連ROI " + " ".join(
            f"{np.mean(acc[nm]['w'])*100:.1f}" for nm, _ in LADDER), flush=True)

    rng = np.random.default_rng(0)
    pw, ps_ = np.array(popw), np.array(pops)
    for tag, key, pop in [("枠連 軸枠×紐枠2", "w", pw), ("三連複 BOX上位4", "s", ps_)]:
        cur = next(nm for nm, _ in ladder if nm.startswith("L2"))
        base = np.array(acc[cur][key])
        print(f"\n{'='*100}\n=== {tag}  {len(pop):,}R  人気順 {pop.mean()*100:.2f}% ===")
        print(f"{'設定（容量の小さい順）':<32}{'AUC':>8}{'ROI':>9}{'対人気順':>10}"
              f"{'現行との差':>12}{'95%CI':>18}")
        for nm, _ in ladder:
            m = np.array(acc[nm][key])
            dif = m - base
            lo, hi = boot(dif, rng) if nm != cur else (0.0, 0.0)
            cell = "—" if nm == cur else f"[{lo:+.2f},{hi:+.2f}]"
            mark = "" if nm == cur else ("  ★上" if lo > 0 else ("  ★下" if hi < 0 else ""))
            print(f"{nm:<32}{np.mean(aucs[nm]):>8.4f}{m.mean()*100:>8.2f}%"
                  f"{(m.mean()-pop.mean())*100:>+9.2f}pt{dif.mean()*100:>+11.2f}pt{cell:>18}{mark}")
        rois = [np.array(acc[nm][key]).mean() for nm, _ in LADDER]
        top = int(np.argmax(rois))
        mono = all(rois[i] <= rois[i + 1] for i in range(len(rois) - 1))
        print(f"  ★最良は {ladder[top][0]}（{rois[top]*100:.2f}%）／"
              + ("**単調増加が続いている**" if mono else f"**{ladder[top][0]} で頭打ち＝山型**"))
        print("  ※AUCも併記した。AUCが下がりながらROIが上がるなら"
              "『精度を上げるとROIが下がる』の裏返しが起きている。")

    print("\n★読み方: 単調が続くなら『容量を上げるほど回収率が上がる』は実装できる方向で、"
          "HANDOFFの「現行構成がROIでは最良」は訂正が要る。")
    print("　頭打ち/反転するなら山の頂点が最適容量だが、その位置は事前に選べないので(53)の留保のまま。")


if __name__ == "__main__":
    main()
