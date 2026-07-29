"""
★自己監査: (33)(36)の判定に使った誤差が間違っていたので測り直す。

見つかった問題:
  ① 3シードの「幅」はばらつきを大幅に過小評価する。
     正規分布から3個引いた範囲の期待値は約1.7σだが、95%区間は3.9σ分ある＝実際の不確実性の半分以下。
  ② ★より深刻: シード幅は「モデルの乱数によるブレ」しか測っていない。
     **レース数が少ないことによるブレ**（標本誤差）を完全に無視していた。
     (37)で測ったとおり1,986Rなら±44.6ptもあるのに、(33)(36)では±5pt程度のシード幅と比べて
     「幅を超える差」と判定していた＝**判定の大半が無意味**だった可能性が高い。

本スクリプトは各条件について**標本誤差を含む正しい95%CI**を出し、
全体(基準)がそのCIに入るかどうかで判定し直す。入るなら「差があるとは言えない」。
併せて、シード幅と標本誤差の大きさを並べて、どちらが支配的かを示す。

実行: python3 ml/audit_errorbars.py [シード数(既定5)]
"""
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, "ml")
import features as F
from himo_sweep import POINTS, KEY, hits
from pocket_eval import load_payout_a

TNAME, NHIMO = "三連単マルチ", 7


def build_conds(df):
    """(33)(36)で検定した条件を再現する。"""
    c = {}
    fs, rc, ds = df["fieldsize"], df["raceclass"], df["distance"]
    c["頭数 16〜"] = fs >= 16
    c["頭数 13-15"] = (fs >= 13) & (fs <= 15)
    c["頭数 10-12"] = (fs >= 10) & (fs <= 12)
    c["頭数 〜9"] = fs <= 9
    c["クラス 3勝OP"] = (rc >= 4) & (rc <= 5)
    c["クラス 未勝利"] = rc == 1
    c["クラス 1-2勝"] = (rc >= 2) & (rc <= 3)
    c["クラス 重賞"] = rc >= 6
    c["距離 〜1200"] = ds <= 1200
    c["距離 1201-1600"] = (ds > 1200) & (ds <= 1600)
    c["距離 1601-2000"] = (ds > 1600) & (ds <= 2000)
    c["距離 2001〜"] = ds > 2000
    ws = df["weeks_since"]
    c["休み明け 〜3週"] = ws <= 3
    c["休み明け 17週〜"] = ws > 16
    c["乗替 継続"] = ~df["jockey_changed"].astype(bool)
    c["乗替 乗り替わり"] = df["jockey_changed"].astype(bool)
    pos = df["umaban"] / fs
    c["枠 内"] = pos <= 1 / 3
    c["枠 外"] = pos > 2 / 3
    c["性 牡"] = df["sex"] == "牡"
    c["性 牝"] = df["sex"] == "牝"
    c["齢 2-3歳"] = df["age"] <= 3
    c["齢 6歳〜"] = df["age"] >= 6
    for course in ["阪神", "中京", "中山", "東京", "京都", "新潟", "福島", "小倉", "函館", "札幌"]:
        c[f"場 {course}"] = df["course"] == course
    pf = df["prev_finish"]
    c["前走 1着"] = pf == 1
    c["前走 6着〜"] = pf >= 6
    wc = df["wtcarry"]
    c["斤量 56.5〜"] = wc >= 56.5
    c["斤量 〜54"] = wc <= 54
    bw = df["bodywt_change"]
    c["馬体重 −8以下"] = bw <= -8
    c["馬体重 +8以上"] = bw >= 8
    return c


def main():
    n_seed = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    d = F.to_model(F.load_files())
    f = F.build_features(d)
    keep = f["n_prior"] >= 1
    d, f = d[keep].reset_index(drop=True), f[keep].reset_index(drop=True)
    d["prev_jockey"] = d.groupby("horse")["jockey"].shift(1)
    d["jockey_changed"] = (d["prev_jockey"].notna()) & (d["jockey"] != d["prev_jockey"])
    d["prev_finish"] = d.groupby("horse")["finish"].shift(1)
    y = (d["finish"] == 1).astype(int).to_numpy()
    cut = d["date"].quantile(0.3)
    tr, te = (d["date"] < cut).to_numpy(), (d["date"] >= cut).to_numpy()
    fx, _ = F.encode_categoricals(f)
    pays = load_payout_a("data/payout/a.csv")
    meta = d.loc[te, ["raceid", "umaban", "sex", "age", "course", "prev_finish", "jockey_changed"]].copy()
    for c in ["fieldsize", "raceclass", "distance", "weeks_since", "wtcarry", "bodywt_change"]:
        meta[c] = f.loc[te, c].to_numpy()
    pts = POINTS[TNAME](NHIMO) * 100
    print(f"シード{n_seed}本  戦略={TNAME}×紐{NHIMO}")

    per_seed = []
    for seed in range(n_seed):
        m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=31, subsample=0.8,
                               colsample_bytree=0.8, min_child_samples=100, verbose=-1,
                               random_state=seed)
        m.fit(fx[tr], y[tr], categorical_feature=F.CAT_COLS)
        pr = meta.copy()
        pr["prob"] = m.predict_proba(fx[te])[:, 1]
        rows = []
        for rid, g in pr.groupby("raceid", sort=False):
            p = pays.get(rid)
            if p is None or not p["sanrentan"] or len(g) <= NHIMO:
                continue
            g = g.sort_values("prob", ascending=False, kind="mergesort")
            nums = g["umaban"].astype(int).tolist()
            r = g.iloc[0].to_dict()
            r["pay"] = hits(p[KEY[TNAME]], nums[0], set(nums[1:NHIMO + 1]))
            rows.append(r)
        per_seed.append(pd.DataFrame(rows))
        print(f"  seed {seed} 完了")

    base_by_seed = [df["pay"].mean() / pts * 100 for df in per_seed]
    base = np.mean(base_by_seed)
    print(f"\n全体(基準) = {base:.1f}%   シード{n_seed}本の幅 {np.ptp(base_by_seed):.1f}pt")

    stats = defaultdict(list)
    for df in per_seed:
        for name, mask in build_conds(df).items():
            r = df.loc[mask.to_numpy(), "pay"].to_numpy(float) / pts
            if len(r) < 300:
                continue
            stats[name].append((len(r), r.mean() * 100, r.std(ddof=1) * 100))

    print("\n" + "=" * 92)
    print("★正しい誤差での判定（標本誤差を含む95%CI に 全体が入るか）")
    print("=" * 92)
    print(f"{'条件':<16}{'R':>6}{'ROI':>7}{'シード幅':>8}{'標本誤差±':>10}{'95%CI':>18}  判定")
    n_sig = n_tot = 0
    rows_out = []
    for name, v in stats.items():
        R = int(np.mean([x[0] for x in v]))
        roi = np.mean([x[1] for x in v])
        seedw = np.ptp([x[1] for x in v])
        sd = np.mean([x[2] for x in v])
        half = 1.96 * sd / np.sqrt(R)
        lo, hi = roi - half, roi + half
        sig = not (lo <= base <= hi)
        n_tot += 1
        n_sig += sig
        rows_out.append((name, R, roi, seedw, half, lo, hi, sig))
    for name, R, roi, seedw, half, lo, hi, sig in sorted(rows_out, key=lambda x: -x[2]):
        print(f"{name:<16}{R:>6}{roi:>6.1f}%{seedw:>7.1f}{half:>9.1f}"
              f"{f'{lo:.0f}〜{hi:.0f}%':>18}  {'差あり' if sig else '差があるとは言えない'}")
    print(f"\n→ {n_tot}条件中、全体と有意に差があるのは **{n_sig}条件** のみ。")
    ratio = np.mean([x[4] / max(x[3], 1e-9) for x in rows_out])
    print(f"→ 標本誤差はシード幅の平均 {ratio:.1f}倍。**(33)(36)はシード幅だけで判定していたので誤り**。")


if __name__ == "__main__":
    main()
