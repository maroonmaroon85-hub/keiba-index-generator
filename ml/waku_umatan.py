"""
測定漏れの券種を埋める: 枠連(col103) と 馬単(col155)。

(47)で複勝・ワイドの測定漏れを埋めたら最良の券種が見つかったので、同じ抜けを潰す。
配当A には 枠連 と 馬単 も入っている（列レイアウトは src/backtest/payout-parser.ts 冒頭）:
    枠連 103: (枠,枠,配当,人気) × 3スロット      控除率22.5% → 天井77.5%
    馬単 155: (馬番,馬番,配当,人気) × 6スロット   控除率25.0% → 天井75.0%  ※着順どおり

枠連は**馬番ではなく枠番**の組なので、頭数から枠番を導出する必要がある。
JRAの割り当て規則:
    頭数 n ≤ 8  … 枠 = 馬番
    n ≥ 9      … q, r = divmod(n, 8) として、枠1..(8-r) に q頭、枠(8-r+1)..8 に q+1頭を
                 馬番の小さい順に詰める（例 12頭: 枠1-4が1頭ずつ、枠5-8が2頭ずつ）
この導出が正しいか、**実際の1着馬/2着馬の枠が枠連の的中キーと一致するか**で検算する。

実行: python3 ml/waku_umatan.py [シード数(既定3)]
"""
import csv
import io
import itertools
import sys
import warnings

import numpy as np
import pandas as pd
import lightgbm as lgb

warnings.filterwarnings("ignore")
sys.path.insert(0, "ml")
import features as F
from place_wide import boot, PARAMS
from pocket_eval import _slots


def waku_of(umaban, n):
    """馬番と頭数から枠番を返す。"""
    if n <= 8:
        return int(umaban)
    q, r = divmod(n, 8)
    edge = 8 - r          # 枠1..edge が q頭、枠edge+1..8 が q+1頭
    small = edge * q      # 小さい枠が占める馬番の数
    if umaban <= small:
        return (int(umaban) - 1) // q + 1
    return edge + (int(umaban) - small - 1) // (q + 1) + 1


def load_wu(path):
    """配当A から 枠連 と 馬単 を読む。"""
    out = {}
    with open(path, "rb") as fh:
        txt = fh.read().decode("shift_jis", "replace")
    for r in csv.reader(io.StringIO(txt)):
        if len(r) < 224:
            continue
        rid = r[14].strip()
        if len(rid) != 8:
            continue
        wakuren = _slots(r, 103, 3, 4, 2, 2)
        umatan = _slots(r, 155, 6, 4, 2, 2)
        if wakuren or umatan:
            out[rid] = {"wakuren": wakuren, "umatan": umatan}
    return out


def main():
    n_seed = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    d = F.to_model(F.load_files())
    f = F.build_features(d)
    keep = (f["n_prior"] >= 1) & d["odds"].notna() & (d["odds"] > 0)
    d, f = d[keep].reset_index(drop=True), f[keep].reset_index(drop=True)
    d["fieldsize"] = f["fieldsize"].to_numpy()
    wu = load_wu("data/payout/a.csv")
    print(f"配当Aから枠連/馬単を読込: {len(wu):,}レース")

    # ===== 枠番の導出を実データで検算 =====
    ok = ng = skip = 0
    for rid, g in d.groupby("raceid", sort=False):
        p = wu.get(rid)
        if p is None or not p["wakuren"]:
            continue
        fin = g.sort_values("finish")
        if len(fin) < 2 or fin["finish"].iloc[1] != 2:
            skip += 1
            continue
        n = int(fin["fieldsize"].iloc[0])
        pair = tuple(sorted((waku_of(fin["umaban"].iloc[0], n), waku_of(fin["umaban"].iloc[1], n))))
        if pair in p["wakuren"]:
            ok += 1
        else:
            ng += 1
    tot = ok + ng
    print(f"枠番導出の検算: 一致 {ok:,} / 不一致 {ng:,}（{ok/tot*100:.1f}%）  ※除外や同着で判定不能 {skip:,}")
    if tot and ok / tot < 0.95:
        print("⚠ 一致率が低い。枠番の導出規則が誤っている可能性が高いので枠連の結果は信用しない。")

    # ===== モデル =====
    win = (d["finish"] == 1).astype(int).to_numpy()
    top3 = (d["finish"] <= 3).astype(int).to_numpy()
    odds = d["odds"].to_numpy(float)
    inv = 1.0 / odds
    mkt = inv / pd.Series(inv).groupby(d["raceid"]).transform("sum").to_numpy()
    fx, _ = F.encode_categoricals(f)
    fx["log_odds"] = np.log(odds)
    fx["mkt_prob"] = mkt
    cut = d["date"].quantile(0.3)
    tr, te = (d["date"] < cut).to_numpy(), (d["date"] >= cut).to_numpy()
    pr = {}
    for tag, y in [("win", win), ("top3", top3)]:
        ps = []
        for s in range(n_seed):
            m = lgb.LGBMClassifier(random_state=s, **PARAMS)
            m.fit(fx[tr], y[tr], categorical_feature=F.CAT_COLS)
            ps.append(m.predict_proba(fx[te])[:, 1])
        pr[tag] = np.mean(ps, axis=0)
        print(f"  {tag} 学習完了")

    sub = d.loc[te, ["raceid", "umaban", "fieldsize"]].copy()
    sub["year"] = d.loc[te, "date"].dt.year.to_numpy()
    for k, v in pr.items():
        sub[k] = v
    rng = np.random.default_rng(0)

    def report(name, rows, pts):
        df = pd.DataFrame(rows)
        x = (df["pay"] / pts).to_numpy(float)
        lo, hi = boot(x, rng, 2000)
        nz = df.loc[df["pay"] > 0, "pay"]
        ys = (df["pay"] / pts).groupby(df["y"]).mean() * 100
        print(f"{name:<24}{len(x):>7,}{pts:>7,}円{(x>0).mean()*100:>7.2f}%"
              f"{nz.mean():>10,.0f}円{x.mean()*100:>7.1f}%{f'[{lo:.0f},{hi:.0f}]':>14}"
              f"{f'{ys.min():.0f}〜{ys.max():.0f}%':>13}")

    print(f"\n{'買い方':<24}{'R':>7}{'コスト':>8}{'的中率':>8}{'的中時配当':>11}{'ROI':>8}{'95%CI':>14}{'年別':>13}")
    for tag in ["win", "top3"]:
        print(f"— 順位付け: {tag} —")
        for n in [2, 3, 4]:
            kinds = {
                f"馬単 1着固定×紐{n}": ("umatan", lambda a, h: [(a, x) for x in h], n),
                f"馬単 2着固定×紐{n}": ("umatan", lambda a, h: [(x, a) for x in h], n),
                f"馬単 マルチ×紐{n}": ("umatan", lambda a, h: [(a, x) for x in h] + [(x, a) for x in h], 2 * n),
                f"枠連 軸枠×紐枠{n}": ("wakuren", None, None),
            }
            for name, (key, fn, npt) in kinds.items():
                rows = []
                for rid, g in sub.groupby("raceid", sort=False):
                    p = wu.get(rid)
                    if p is None or not p[key] or len(g) < n + 1:
                        continue
                    gg = g.sort_values(tag, ascending=False, kind="mergesort")
                    nums = gg["umaban"].astype(int).tolist()
                    if key == "umatan":
                        cs = fn(nums[0], nums[1:n + 1])
                        k = npt
                    else:
                        fs = int(gg["fieldsize"].iloc[0])
                        wa = waku_of(nums[0], fs)
                        cs = sorted({tuple(sorted((wa, waku_of(h, fs)))) for h in nums[1:n + 1]})
                        k = len(cs)
                    if k == 0:
                        continue
                    rows.append({"y": g["year"].iloc[0],
                                 "pay": sum(p[key].get(c, 0) for c in cs), "k": k})
                if len(rows) < 500:
                    continue
                df = pd.DataFrame(rows)
                # 枠連は点数がレースごとに変わるので平均コストで割る
                if key == "wakuren":
                    x = (df["pay"] / (df["k"] * 100)).to_numpy(float)
                    lo, hi = boot(x, rng, 2000)
                    nz = df.loc[df["pay"] > 0, "pay"]
                    ys = pd.Series(x).groupby(df["y"].to_numpy()).mean() * 100
                    print(f"{name:<24}{len(x):>7,}{df['k'].mean()*100:>6,.0f}円"
                          f"{(x>0).mean()*100:>7.2f}%{nz.mean():>10,.0f}円{x.mean()*100:>7.1f}%"
                          f"{f'[{lo:.0f},{hi:.0f}]':>14}{f'{ys.min():.0f}〜{ys.max():.0f}%':>13}")
                else:
                    report(name, rows, npt * 100)
        print()


if __name__ == "__main__":
    main()
