"""前提の検算2本（既存の数字が正しいかを問う。新しい買い方は探さない）。

**A1 カバー率**: 評価は「手元のアーカイブに居る馬」だけを候補にして順位付けしている。
(69)⑤で出走馬カバー率97.6%・完全に揃うレース81.7%と測ったが、
**揃っていないレースでもROIを計上している**。欠けている馬は「その後どこにも出てこなかった馬」＝
弱い馬に偏るはずで、候補から自動的に消えることで**ROIが実際より良く出ている**可能性がある。
→ カバー率でROIを分解し、完全に揃うレースだけでも同じ数字が出るかを見る。

**A2 日クラスタ**: 既存の boot() は全てレース単位のi.i.d.リサンプル。
同一開催日のレースは馬場・天候・プール規模を共有するので独立ではない。
独立と仮定するとCIは狭く出る。**開催日ブロック**でリサンプルし直して比べる。

実行: python3 ml/audit_premise.py [シード数(既定3)]
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


def wakuren_cs(nums, n):
    wa = waku_of(nums[0], n)
    return sorted({tuple(sorted((wa, waku_of(h, n)))) for h in nums[1:3]})


def boot_iid(x, rng, n=2000):
    b = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return np.percentile(b, 2.5) * 100, np.percentile(b, 97.5) * 100


def boot_cluster(x, groups, rng, n=2000):
    """開催日ブロックブートストラップ。日を復元抽出し、その日のレースを丸ごと採る。"""
    codes, uniq = pd.factorize(groups)
    idx = [np.flatnonzero(codes == i) for i in range(len(uniq))]
    sizes = np.array([len(i) for i in idx])
    out = []
    for _ in range(n):
        pick = rng.integers(0, len(idx), len(idx))
        tot = np.concatenate([idx[p] for p in pick])
        out.append(x[tot].mean())
    return np.percentile(out, 2.5) * 100, np.percentile(out, 97.5) * 100, sizes.mean()


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

    sub = d.loc[te, ["raceid", "umaban", "fieldsize", "date", "odds"]].copy()
    sub["p"] = p
    wu, pa = load_wu(PAYOUT), load_payout_a(PAYOUT)

    rw, rs = [], []
    for rid, g in sub.groupby("raceid", sort=False):
        n = int(g["fieldsize"].iloc[0])
        cov = len(g) / n if n else 0.0
        base = {"date": g["date"].iloc[0], "year": g["date"].iloc[0].year,
                "n": n, "ncand": len(g), "cov": cov}
        order = {"model": g.sort_values("p", ascending=False, kind="mergesort"),
                 "pop": g.sort_values("odds", ascending=True, kind="mergesort")}
        nums = {k: v["umaban"].astype(int).tolist() for k, v in order.items()}
        w, s3 = wu.get(rid), pa.get(rid)
        if w and w["wakuren"] and len(g) >= 3:
            row = dict(base)
            for k, nm in nums.items():
                cs = wakuren_cs(nm, n)
                row[f"{k}_x"] = sum(w["wakuren"].get(c, 0) for c in cs) / (len(cs) * 100)
            rw.append(row)
        if s3 and s3["sanrenpuku"] and len(g) >= 9:
            row = dict(base)
            for k, nm in nums.items():
                cs = [tuple(sorted(c)) for c in itertools.combinations(nm[:4], 3)]
                row[f"{k}_x"] = sum(s3["sanrenpuku"].get(c, 0) for c in cs) / 400
            rs.append(row)

    rng = np.random.default_rng(0)
    for title, rows in [("枠連 軸枠×紐枠2", rw), ("三連複 BOX上位4", rs)]:
        df = pd.DataFrame(rows)
        m, q = df["model_x"].to_numpy(), df["pop_x"].to_numpy()
        print(f"\n{'='*78}\n=== {title}  {len(df):,}R ===")
        print(f"  全体: モデル {m.mean()*100:.2f}%  人気順 {q.mean()*100:.2f}%  "
              f"差 {(m-q).mean()*100:+.2f}pt")

        # ---------- A1 カバー率 ----------
        print(f"\n  [A1] アーカイブのカバー率別（候補馬数 / 実頭数）")
        print(f"  {'カバー率':<14}{'R':>7}{'構成比':>7}{'平均頭数':>8}{'モデルROI':>10}"
              f"{'人気順ROI':>10}{'差':>8}")
        bins = [(1.0, 1.01, "= 1.00 (完全)"), (0.95, 1.0, "0.95-1.00"),
                (0.90, 0.95, "0.90-0.95"), (0.0, 0.90, "< 0.90")]
        for lo, hi, lab in bins:
            sel = (df["cov"] >= lo) & (df["cov"] < hi)
            if sel.sum() < 50:
                continue
            mm, qq = m[sel.to_numpy()], q[sel.to_numpy()]
            print(f"  {lab:<14}{sel.sum():>7,}{sel.mean()*100:>6.1f}%"
                  f"{df.loc[sel,'n'].mean():>8.1f}{mm.mean()*100:>9.2f}%{qq.mean()*100:>9.2f}%"
                  f"{(mm-qq).mean()*100:>+7.2f}pt")
        full = (df["cov"] >= 1.0).to_numpy()
        lo1, hi1 = boot_iid(m[full] - q[full], rng)
        print(f"  → 完全に揃うレースだけ: モデル {m[full].mean()*100:.2f}% / "
              f"人気順 {q[full].mean()*100:.2f}% / 差 {(m[full]-q[full]).mean()*100:+.2f}pt "
              f"CI[{lo1:+.2f},{hi1:+.2f}]")
        print(f"  カバー率の年別平均: " + "  ".join(
            f"{yy}:{vv:.3f}" for yy, vv in df.groupby("year")["cov"].mean().items()))

        # ---------- A2 日クラスタ ----------
        print(f"\n  [A2] ブートストラップの単位を変える（差 {(m-q).mean()*100:+.2f}pt のCI）")
        li, hi_ = boot_iid(m - q, rng)
        lc, hc, avg = boot_cluster(m - q, df["date"].to_numpy(), rng)
        print(f"  レース単位 i.i.d.（既存）  CI[{li:+.2f}, {hi_:+.2f}]  幅 {hi_-li:.2f}pt")
        print(f"  開催日ブロック（{len(df['date'].unique()):,}日・平均{avg:.1f}R/日）"
              f"  CI[{lc:+.2f}, {hc:+.2f}]  幅 {hc-lc:.2f}pt"
              f"   → 拡大率 {(hc-lc)/(hi_-li):.2f}倍")
        lvi = boot_iid(m, rng)
        lvc = boot_cluster(m, df["date"].to_numpy(), rng)
        print(f"  （ROI水準そのもの: i.i.d. CI[{lvi[0]:.2f},{lvi[1]:.2f}] / "
              f"日ブロック CI[{lvc[0]:.2f},{lvc[1]:.2f}]）")


if __name__ == "__main__":
    main()
