"""A3: ウォークフォワード評価 — **実運用と同じ形で測り直す**。

現行の数字（枠連84.5% / 三連複BOX4 84.5%）は**すべて「前30%学習・後70%検証」の単一分割**から出ている。
だがこの分割は実運用と2点で違う:

  1. **学習量が違う**  … 2013-2016の17.6万行だけで学習したモデルで、2017-2026を予測している。
     実運用の `train_prod.py` は**全期間58.7万行**で学習する。3.3倍の学習量のモデルは測っていない。
  2. **経過年数が違う** … 2026年の予測を「9年前のモデル」で出している。実運用は毎回学習し直す。
     (35)で「陳腐化ではない」と切り分けたが、あれは旧構成(オッズなし・三連単)での話。

ウォークフォワード（expanding window）: 各年Yについて **Y年より前の全データで学習 → Y年を予測**。
これが実運用の手続きそのもの。ここで出る数字が、運用で期待してよい正味の値。

同時に見るもの:
  ・**近年でも優位が残っているか**（市場効率化で消えていれば運用の前提が崩れる）
  ・単一分割の84.5%が、学習量を実運用と揃えても再現するか

実行: python3 ml/audit_walkforward.py [シード数(既定3)]
"""
import itertools
import sys
import warnings

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
sys.path.insert(0, "ml")
import features as F
from _cache import load_cached
from place_wide import PARAMS, load_place_wide
from pocket_eval import load_payout_a
from waku_umatan import load_wu, waku_of

PAYOUT = "data/payout/a.csv"
FIRST_TEST_YEAR = 2016
ROLLING = 0


def wakuren_cs(nums, n):
    wa = waku_of(nums[0], n)
    return sorted({tuple(sorted((wa, waku_of(h, n)))) for h in nums[1:3]})


def boot(x, rng, n=2000):
    b = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return np.percentile(b, 2.5) * 100, np.percentile(b, 97.5) * 100


def evaluate(sub, wu, pa, pw):
    """1年ぶんの予測から、券種ごとの (モデル, 人気順) の1レース当たり回収額を作る。"""
    out = {k: [] for k in ("wakuren", "sanrenpuku", "fukusho")}
    for rid, g in sub.groupby("raceid", sort=False):
        n = int(g["fieldsize"].iloc[0])
        order = {"model": g.sort_values("p", ascending=False, kind="mergesort"),
                 "pop": g.sort_values("odds", ascending=True, kind="mergesort")}
        nums = {k: v["umaban"].astype(int).tolist() for k, v in order.items()}
        w, s3, fw = wu.get(rid), pa.get(rid), pw.get(rid)
        if w and w["wakuren"] and len(g) >= 3:
            r = {"raceid": rid}
            for k, nm in nums.items():
                cs = wakuren_cs(nm, n)
                r[k] = sum(w["wakuren"].get(c, 0) for c in cs) / (len(cs) * 100)
            out["wakuren"].append(r)
        if s3 and s3["sanrenpuku"] and len(g) >= 9:
            r = {"raceid": rid}
            for k, nm in nums.items():
                cs = [tuple(sorted(c)) for c in itertools.combinations(nm[:4], 3)]
                r[k] = sum(s3["sanrenpuku"].get(c, 0) for c in cs) / 400
            out["sanrenpuku"].append(r)
        if fw and fw["fuku"]:
            r = {"raceid": rid}
            r.update({k: fw["fuku"].get(nm[0], 0) / 100 for k, nm in nums.items()})
            out["fukusho"].append(r)
    return out


def main():
    n_seed = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    global ROLLING
    # 第2引数に年数を渡すと「直近N年だけで学習」する固定窓になる。
    # ★これは「近年の優位縮小」の原因の切り分けに使う:
    #   ・市場の効率化が原因なら → 固定窓でも同じように縮む
    #   ・学習量が増えてモデルが"正しく"なったのが原因なら((63)の5回繰り返された現象) → 固定窓では縮まない
    ROLLING = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    if ROLLING:
        print(f"★固定窓モード: 各年の学習は直近{ROLLING}年だけ")
    d, fx = load_cached()
    y = (d["finish"] <= 3).astype(int).to_numpy()
    year = d["date"].dt.year.to_numpy()
    wu, pa = load_wu(PAYOUT), load_payout_a(PAYOUT)
    pw = load_place_wide(PAYOUT)
    years = [yy for yy in range(FIRST_TEST_YEAR, int(year.max()) + 1) if (year == yy).sum() > 5000]
    print(f"ウォークフォワード: 各年について「その年より前の全データ」で学習（シード{n_seed}本）")
    print(f"評価年 {years[0]}〜{years[-1]}  全 {len(d):,}行\n")

    acc = {k: [] for k in ("wakuren", "sanrenpuku", "fukusho")}
    print(f"{'年':<6}{'学習行数':>10}{'AUC':>8}{'枠連 M/人':>18}{'三連複BOX4 M/人':>20}{'複勝top1 M/人':>18}")
    for yy in years:
        tr, te = year < yy, year == yy
        if ROLLING:
            tr = tr & (year >= yy - ROLLING)
        ms = [lgb.LGBMClassifier(random_state=s, **PARAMS).fit(fx[tr], y[tr], categorical_feature=F.CAT_COLS)
              for s in range(n_seed)]
        p = np.mean([m.predict_proba(fx[te])[:, 1] for m in ms], axis=0)
        auc = roc_auc_score(y[te], p)
        sub = d.loc[te, ["raceid", "umaban", "fieldsize", "odds"]].copy()
        sub["p"] = p
        res = evaluate(sub, wu, pa, pw)
        cells = []
        for k in ("wakuren", "sanrenpuku", "fukusho"):
            df = pd.DataFrame(res[k])
            df["year"] = yy
            acc[k].append(df)
            cells.append(f"{df['model'].mean()*100:5.1f}/{df['pop'].mean()*100:5.1f}"
                         f"({(df['model']-df['pop']).mean()*100:+5.2f})")
        print(f"{yy:<6}{int(tr.sum()):>10,}{auc:>8.4f}{cells[0]:>18}{cells[1]:>20}{cells[2]:>18}")

    rng = np.random.default_rng(0)
    print(f"\n{'='*84}\n★ウォークフォワード合計（実運用と同じ手続き）")
    print(f"{'券種':<16}{'R':>8}{'モデル':>9}{'人気順':>9}{'差':>9}{'差の95%CI':>18}"
          f"{'プラスの年':>10}{'単一分割の報告値':>16}")
    ref = {"wakuren": "84.5%", "sanrenpuku": "84.5%", "fukusho": "85.1%"}
    name = {"wakuren": "枠連 軸枠×紐枠2", "sanrenpuku": "三連複 BOX上位4", "fukusho": "複勝 top1"}
    for k in ("wakuren", "sanrenpuku", "fukusho"):
        df = pd.concat(acc[k], ignore_index=True)
        m, q = df["model"].to_numpy(), df["pop"].to_numpy()
        lo, hi = boot(m - q, rng)
        ys = pd.Series((m - q) * 100).groupby(df["year"].to_numpy()).mean()
        print(f"{name[k]:<16}{len(df):>8,}{m.mean()*100:>8.2f}%{q.mean()*100:>8.2f}%"
              f"{(m-q).mean()*100:>+8.2f}pt{f'[{lo:+.2f},{hi:+.2f}]':>18}"
              f"{f'{int((ys>0).sum())}/{len(ys)}':>10}{ref[k]:>16}")

    print(f"\n評価期間を単一分割と揃える（2017年以降だけ。単一分割の検証期間は2017-01-14〜）")
    print(f"{'券種':<16}{'R':>8}{'モデル':>9}{'人気順':>9}{'差':>9}{'差の95%CI':>18}{'単一分割':>10}")
    for k in ("wakuren", "sanrenpuku", "fukusho"):
        df = pd.concat(acc[k], ignore_index=True)
        df = df[df["year"] >= 2017]
        m, q = df["model"].to_numpy(), df["pop"].to_numpy()
        lo, hi = boot(m - q, rng)
        print(f"{name[k]:<16}{len(df):>8,}{m.mean()*100:>8.2f}%{q.mean()*100:>8.2f}%"
              f"{(m-q).mean()*100:>+8.2f}pt{f'[{lo:+.2f},{hi:+.2f}]':>18}{ref[k]:>10}")

    print(f"\n直近5年だけ（市場効率化で優位が消えていないかの確認）")
    print(f"{'券種':<16}{'R':>8}{'モデル':>9}{'人気順':>9}{'差':>9}{'差の95%CI':>18}")
    for k in ("wakuren", "sanrenpuku", "fukusho"):
        df = pd.concat(acc[k], ignore_index=True)
        df = df[df["year"] >= years[-1] - 4]
        m, q = df["model"].to_numpy(), df["pop"].to_numpy()
        lo, hi = boot(m - q, rng)
        print(f"{name[k]:<16}{len(df):>8,}{m.mean()*100:>8.2f}%{q.mean()*100:>8.2f}%"
              f"{(m-q).mean()*100:>+8.2f}pt{f'[{lo:+.2f},{hi:+.2f}]':>18}")

    # ===== 本命の選択を対応ありで直接比べる（(77)②の作法） =====
    print(f"\n★本命の選択: 枠連 vs 三連複BOX4 を**同じレースで**直接比較（(69)③の判断の再検討）")
    W = pd.concat(acc["wakuren"], ignore_index=True).set_index("raceid")
    S = pd.concat(acc["sanrenpuku"], ignore_index=True).set_index("raceid")
    j = W.join(S, how="inner", lsuffix="_w", rsuffix="_s")
    print(f"{'期間':<12}{'R':>8}{'枠連ROI':>10}{'三連複ROI':>11}{'ROI差':>10}{'95%CI':>18}"
          f"{'枠連の対人気順':>14}{'三連複の対人気順':>16}")
    for lab, lo_y in [("全期間", 0), ("2017年以降", 2017), ("直近5年", years[-1] - 4)]:
        x = j[j["year_w"] >= lo_y]
        dw = (x["model_w"] - x["pop_w"]).mean() * 100
        ds = (x["model_s"] - x["pop_s"]).mean() * 100
        diff = (x["model_w"] - x["model_s"]).to_numpy()
        lo, hi = boot(diff, rng)
        print(f"{lab:<12}{len(x):>8,}{x['model_w'].mean()*100:>9.2f}%{x['model_s'].mean()*100:>10.2f}%"
              f"{diff.mean()*100:>+9.2f}pt{f'[{lo:+.2f},{hi:+.2f}]':>18}{dw:>+13.2f}pt{ds:>+15.2f}pt")
    print("  ※枠連は1レース平均194円・三連複BOX4は400円。ROIが同じなら点数が少ない方が損失は小さい((77)②)")

    import pickle
    out = {k: pd.concat(v, ignore_index=True) for k, v in acc.items()}
    with open(f"/tmp/wf_acc{ROLLING}.pkl", "wb") as fh:
        pickle.dump(out, fh)
    print("\n（レース単位の結果を /tmp/wf_acc.pkl に保存）")


if __name__ == "__main__":
    main()
