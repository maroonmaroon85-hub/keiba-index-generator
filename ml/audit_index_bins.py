"""
(84) 指数（モデルの推奨の強さ）で区間を切り、区間ごとのROIを出す。

⚠**先に(62)と(46)を読むこと**。(62)で枠連スコア（積ベース）による絞り込みを測っており、
**上位に絞っても回収率は上がらない／使えるのは下位20%の除外だけ**という結論が出ている。
(46)は「絞ると測れなくなる」（全体なら2年で検証できるものが、絞ると199年かかる）。

**それでも測り直す理由**: (62)は全てL2（現行容量）での話。(81)(83)で容量を上げると
**モデルが市場からさらに離れる**ことが分かったので、**指数の分布そのものが変わる**。
L5で同じ絞り込みが効くかは未測定。

★罠を避ける設計:
  ・指数は**5つ事前に宣言**し、結果を見てから足さない
  ・閾値を探すのではなく**十分位の曲線を全部出す**（最良の区間だけ報告しない）
  ・各区間に**人気順で同じ買い方をした対照**を並べる（(69)）
  ・L2とL5の両方で出し、**同じ形になるか**を見る（片方だけなら容量固有の現象）

★もう1つ、事前に宣言した仮説を検定する:
  **「圧倒的1番人気を3着固定で買う必要はない」**（＝固定位置は軸の強さに応じて変えるべき）。
  軸が断然なら1着に来る確率が高く、「ちょうど3着」に来る確率はむしろ下がるので、
  **断然の区間ほど1着固定が良く、非断然の区間ほど3着固定が良い**はず。
  (54)は固定位置を**全レース一律**で比べており、この交互作用は測っていない。
  → 予測: 軸のシェアの十分位で切ると、1着固定と3着固定のROIが**どこかで交差する**。
     交差しなければ「固定位置は軸の強さと無関係」＝(54)の一律の結論のままでよい。

指数（すべて発走前に計算できる量）:
  A 軸のシェア            … レース内で正規化したモデル確率の最大値。軸がどれだけ断然か
  B 軸と2位のシェア差      … 1位と2位が離れているか
  C 枠連スコア（積ベース）   … (62)と同じ指標。2*Pa*Pb（ゾロ目は Pa^2）
  D 軸のシェア − 市場含意   … ★本命。**モデルが市場とどれだけ違うと言っているか**
  E 買い目の市場含意確率    … 期待配当の代理。低いほど「市場的に遠い」買い目

★Dが本命の理由: (72)で「モデルの寄与は人気順と買い目が違うレースに集中する」と分かっているが、
　あれは**二値（一致/不一致）**でしか見ていない。連続量で測れば区間差が出るはず。

★券種を広げるときの設計（(85)で実施）:
  29通り×5指数×10区間＝**1,450セル**は(38)の罠に直撃する（6,206セルでは偶然34.4セルが通過）。
  そこで **(a)券種ごとに代表1つ＝8通りに絞る (b)判定を「最良の区間」から「曲線の単調性(ρ)」に変える
  (c)指数ラベルを層内でシャッフルした帰無分布とρを比べる** の3点で守る。
  「10区間のどれかが高い」は偶然でも起きるが、「10区間が指数の順に並ぶ」は偶然では起きにくい。

実行: python3 ml/audit_index_bins.py [シード数(既定1)] [開始年(既定2019)]
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

CONFIGS = [("L2 現行", dict(PARAMS)),
           ("L5", dict(PARAMS, num_leaves=255, min_child_samples=10, n_estimators=2000))]

IDX = [("A", "軸のシェア"), ("B", "軸と2位の差"), ("C", "枠連スコア"),
       ("D", "軸のシェア−市場含意"), ("E", "買い目の市場含意")]


def wakuren_cs(nums, n):
    wa = waku_of(nums[0], n)
    return sorted({tuple(sorted((wa, waku_of(h, n)))) for h in nums[1:3]})


def bracket_prob(nums, share, n):
    bp = {}
    for u, s in zip(nums, share):
        w = waku_of(u, n)
        bp[w] = bp.get(w, 0.0) + float(s)
    return bp


def waku_score(pairs, bp):
    t = 0.0
    for a, b in pairs:
        t += bp.get(a, 0.0) ** 2 if a == b else 2 * bp.get(a, 0.0) * bp.get(b, 0.0)
    return t


def main():
    n_seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    y0 = int(sys.argv[2]) if len(sys.argv) > 2 else 2019
    d, fx = load_cached()
    y = (d["finish"] <= 3).astype(int).to_numpy()
    year = d["date"].dt.year.to_numpy()
    wu, pa = load_wu(PAYOUT), load_payout_a(PAYOUT)
    years = [yy for yy in range(y0, int(year.max()) + 1) if (year == yy).sum() > 5000]
    print(f"指数の区間別ROI（{years[0]}〜{years[-1]}・シード{n_seed}本・L2/L5）\n")

    rows = []
    for yy in years:
        tr, te = year < yy, year == yy
        sub = d.loc[te, ["raceid", "umaban", "fieldsize", "odds"]].copy()
        for ci, (_, par) in enumerate(CONFIGS):
            ps = [lgb.LGBMClassifier(random_state=s, **par)
                  .fit(fx[tr], y[tr], categorical_feature=F.CAT_COLS)
                  .predict_proba(fx[te])[:, 1] for s in range(n_seed)]
            sub[f"c{ci}"] = np.mean(ps, axis=0)
        for rid, g in sub.groupby("raceid", sort=False):
            w, s3 = wu.get(rid), pa.get(rid)
            okw = w and w["wakuren"] and len(g) >= 3
            oks = s3 and s3["sanrenpuku"] and len(g) >= 9
            if not (okw or oks):
                continue
            n = int(g["fieldsize"].iloc[0])
            uma = g["umaban"].astype(int).to_numpy()
            od = g["odds"].to_numpy(float)
            inv = 1.0 / od
            mkt = inv / inv.sum()                       # 市場含意確率（レース内で正規化）
            po = uma[np.argsort(od, kind="mergesort")]
            r = {"year": yy}
            if okw:
                cs = wakuren_cs(po, n)
                r["pop_w"] = sum(w["wakuren"].get(c, 0) for c in cs) / (len(cs) * 100)
            if oks:
                cs = [tuple(sorted(c)) for c in itertools.combinations(po[:4], 3)]
                r["pop_s"] = sum(s3["sanrenpuku"].get(c, 0) for c in cs) / 400
            okt = s3 and s3["sanrentan"] and len(g) >= 6
            for ci in range(len(CONFIGS)):
                p = g[f"c{ci}"].to_numpy(float)
                o = np.argsort(-p, kind="mergesort")
                nums, share = uma[o], p[o] / p.sum()
                mk = mkt[o]
                pairs = wakuren_cs(nums, n)
                bp = bracket_prob(nums, share, n)
                bpm = bracket_prob(nums, mk, n)         # 市場側の枠確率
                r[f"A{ci}"] = float(share[0])
                r[f"B{ci}"] = float(share[0] - share[1])
                r[f"C{ci}"] = float(waku_score(pairs, bp))
                r[f"D{ci}"] = float(share[0] - mk[0])
                r[f"E{ci}"] = float(waku_score(pairs, bpm))   # 買い目の市場含意確率
                if okw:
                    r[f"w{ci}"] = sum(w["wakuren"].get(c, 0) for c in pairs) / (len(pairs) * 100)
                if oks:
                    cs = [tuple(sorted(c)) for c in itertools.combinations(nums[:4], 3)]
                    r[f"s{ci}"] = sum(s3["sanrenpuku"].get(c, 0) for c in cs) / 400
                if okt:
                    # 三連単の固定位置3種（紐3・各6点）。軸の強さとの交互作用を見るため
                    ax, himo = nums[0], nums[1:4]
                    for pos, mk_ in ((1, lambda a, b: (ax, a, b)), (2, lambda a, b: (a, ax, b)),
                                     (3, lambda a, b: (a, b, ax))):
                        combos = [mk_(a, b) for a, b in itertools.permutations(himo, 2)]
                        r[f"t{pos}_{ci}"] = sum(s3["sanrentan"].get(tuple(c), 0)
                                                for c in combos) / 600
            rows.append(r)
        print(f"  {yy} 完了", flush=True)

    df = pd.DataFrame(rows)
    for key, tag, popcol in [("w", "枠連 軸枠×紐枠2", "pop_w"), ("s", "三連複 BOX上位4", "pop_s")]:
        dd = df[df[f"{key}0"].notna()].copy()
        print(f"\n{'='*112}\n=== {tag}  {len(dd):,}R  "
              f"人気順 {dd[popcol].mean()*100:.2f}% ===")
        for code, name in IDX:
            print(f"\n■ 指数 {code}: {name}")
            print(f"{'区間':<8}{'R数':>8}{'指数の範囲':>22}"
                  f"{'L2 ROI':>10}{'L5 ROI':>10}{'人気順':>9}{'L5−人気順':>12}{'L5的中率':>10}")
            for ci, lab in ((0, "L2"), (1, "L5")):
                pass
            # 十分位は**L5の指数**で切る（運用時に使うのはL5の指数のため）
            q = pd.qcut(dd[f"{code}1"], 10, labels=False, duplicates="drop")
            for b in sorted(pd.Series(q).dropna().unique()):
                m = (q == b).to_numpy()
                g = dd[m]
                lo_, hi_ = g[f"{code}1"].min(), g[f"{code}1"].max()
                print(f"{'第'+str(int(b)+1)+'区間':<8}{len(g):>8,}"
                      f"{f'{lo_:.4f}〜{hi_:.4f}':>22}"
                      f"{g[f'{key}0'].mean()*100:>9.1f}%{g[f'{key}1'].mean()*100:>9.1f}%"
                      f"{g[popcol].mean()*100:>8.1f}%"
                      f"{(g[f'{key}1'].mean()-g[popcol].mean())*100:>+11.2f}pt"
                      f"{(g[f'{key}1']>0).mean()*100:>9.1f}%")
            # 単調性の検算: 区間番号とROIの順位相関
            gs = dd.groupby(q)[f"{key}1"].mean()
            rho = float(pd.Series(gs.index, dtype=float).corr(pd.Series(gs.values), method="spearman"))
            print(f"  → 区間とL5 ROIの順位相関 ρ={rho:+.3f}"
                  + ("（単調な傾向あり）" if abs(rho) > 0.6 else "（傾向なし）"))

    # ===== 事前宣言した仮説: 固定位置は軸の強さに応じて変えるべきか =====
    dt = df[df["t1_1"].notna()].copy()
    print(f"\n{'='*112}")
    print(f"=== ★仮説「圧倒的1番人気を3着固定で買う必要はない」 — 三連単の固定位置 × 軸の強さ ===")
    print(f"    {len(dt):,}R・L5・紐3（各6点600円）。(54)は固定位置を全レース一律でしか比べていない")
    print(f"{'='*112}")
    q = pd.qcut(dt["A1"], 10, labels=False, duplicates="drop")
    print(f"{'区間':<8}{'R数':>8}{'軸のシェア':>16}{'1着固定':>10}{'2着固定':>10}{'3着固定':>10}"
          f"{'最良':>8}{'1着−3着':>11}")
    cross = []
    for b in sorted(pd.Series(q).dropna().unique()):
        g = dt[(q == b).to_numpy()]
        v = [g[f"t{p}_1"].mean() * 100 for p in (1, 2, 3)]
        best = ["1着", "2着", "3着"][int(np.argmax(v))]
        cross.append((int(b) + 1, v[0] - v[2]))
        print(f"{'第'+str(int(b)+1)+'区間':<8}{len(g):>8,}"
              f"{f'{g[chr(65)+chr(49)].min():.3f}〜{g[chr(65)+chr(49)].max():.3f}':>16}"
              f"{v[0]:>9.1f}%{v[1]:>9.1f}%{v[2]:>9.1f}%{best:>8}{v[0]-v[2]:>+10.2f}pt")
    xs = np.array([c[0] for c in cross], float)
    ys = np.array([c[1] for c in cross], float)
    rho = float(pd.Series(xs).corr(pd.Series(ys), method="spearman"))
    print(f"  → 区間と「1着固定−3着固定」の順位相関 ρ={rho:+.3f}")
    if rho > 0.6:
        print("     ⇒ **予測どおり**: 断然の区間ほど1着固定が相対的に良い。固定位置を軸の強さで変える根拠になる")
    elif rho < -0.6:
        print("     ⇒ **予測と逆向き**: 断然の区間ほど3着固定が良い。(41)『優位は市場の期待と逆側』と整合する")
    else:
        print("     ⇒ **傾向なし**: 固定位置は軸の強さと無関係。(54)の一律の結論のままでよい")
    print("  ※三連単は的中率4〜9%と低く1シードでは特に振れる。ここは方向を見るだけに留めること。")

    print("\n★読み方: 区間ごとのROIに**単調な傾向が無ければ、その指数で絞る意味は無い**。")
    print("　傾向があっても(46)より『絞ると測れなくなる』ので、運用に載せる前に")
    print("　★判定基準1(シード)・2(標本誤差)・5(絞ると測れない)を当てること。")
    print("　(62)ではL2で『上位に絞っても回収率は上がらない／下位20%除外だけが使える』が結論だった。")


if __name__ == "__main__":
    main()
