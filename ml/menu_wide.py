"""
(77) 買い方を**広く**並べて比較する。候補に挙がっていない買い方を落とさないための横断表。

**なぜ必要か**: 検証が「枠連 軸枠×紐枠2」と「三連複BOX上位4」に寄りすぎていた。
既存の `bet_menu.py` は券種5つ・11通りで、**枠連・馬単・三連単の固定位置・BOX系が入っておらず、
人気順の対照もシード幅も無い**。ここでは8券種29通りを同じ土俵に載せる。

⚠**(38)の罠に注意**: 買い方を増やせば「良く見えるもの」は必ず出る。
　総当たり6,206セルでは偶然だけで34.4セルが「両期間100%超」を通過した。
　だからここでは:
　　・**買い方は事前に宣言してこのリストに固定する**（結果を見てから足さない）
　　・全通りに**人気順で同じ買い方をした対照**を並べる（(69)の教訓）
　　・**多重性を数字で出す**（29通り×5% = 偶然でも約1.5通りが「有意」を通る）
　　・上位のものは**前半/後半とシード幅**も見る

実行: python3 ml/menu_wide.py [シード数(既定3)]
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
from market_baseline import load
from place_wide import PARAMS, boot, load_place_wide
from pocket_eval import load_payout_a
from waku_umatan import load_wu, waku_of

PAYOUT = "data/payout/a.csv"

# ★事前宣言。(券種, 表示名, 買い目を作る関数, 最低頭数)
#   買い目は「モデル順に並んだ馬番リスト nums」と「頭数 n」から作る。
MENU = [
    ("tansho", "単勝 top1", lambda t, n: [(t[0],)], 4),
    ("fuku", "複勝 top1", lambda t, n: [(t[0],)], 4),
    ("fuku", "複勝 top2", lambda t, n: [(h,) for h in t[:2]], 4),
    ("fuku", "複勝 top3", lambda t, n: [(h,) for h in t[:3]], 4),
    ("wide", "ワイド 軸1×紐2", lambda t, n: [tuple(sorted((t[0], h))) for h in t[1:3]], 4),
    ("wide", "ワイド 軸1×紐3", lambda t, n: [tuple(sorted((t[0], h))) for h in t[1:4]], 5),
    ("wide", "ワイド BOX3", lambda t, n: [tuple(sorted(c)) for c in itertools.combinations(t[:3], 2)], 4),
    ("wide", "ワイド BOX4", lambda t, n: [tuple(sorted(c)) for c in itertools.combinations(t[:4], 2)], 5),
    ("umaren", "馬連 軸1×紐2", lambda t, n: [tuple(sorted((t[0], h))) for h in t[1:3]], 4),
    ("umaren", "馬連 軸1×紐3", lambda t, n: [tuple(sorted((t[0], h))) for h in t[1:4]], 5),
    ("umaren", "馬連 軸1×紐4", lambda t, n: [tuple(sorted((t[0], h))) for h in t[1:5]], 6),
    ("umaren", "馬連 BOX3", lambda t, n: [tuple(sorted(c)) for c in itertools.combinations(t[:3], 2)], 4),
    ("umaren", "馬連 BOX4", lambda t, n: [tuple(sorted(c)) for c in itertools.combinations(t[:4], 2)], 5),
    ("wakuren", "枠連 軸枠×紐枠2（現行）",
     lambda t, n: sorted({tuple(sorted((waku_of(t[0], n), waku_of(h, n)))) for h in t[1:3]}), 9),
    ("wakuren", "枠連 軸枠×紐枠3",
     lambda t, n: sorted({tuple(sorted((waku_of(t[0], n), waku_of(h, n)))) for h in t[1:4]}), 9),
    ("wakuren", "枠連 上位3頭の枠BOX",
     lambda t, n: sorted({tuple(sorted(c)) for c in
                          itertools.combinations({waku_of(h, n) for h in t[:3]}, 2)}) or
                  [(waku_of(t[0], n), waku_of(t[0], n))], 9),
    ("umatan", "馬単 1着固定×紐2", lambda t, n: [(t[0], h) for h in t[1:3]], 4),
    ("umatan", "馬単 1着固定×紐3", lambda t, n: [(t[0], h) for h in t[1:4]], 5),
    ("umatan", "馬単 マルチ×紐2",
     lambda t, n: [p for h in t[1:3] for p in ((t[0], h), (h, t[0]))], 4),
    ("sanrenpuku", "三連複 軸1×紐3",
     lambda t, n: [tuple(sorted((t[0], a, b))) for a, b in itertools.combinations(t[1:4], 2)], 6),
    ("sanrenpuku", "三連複 軸1×紐4",
     lambda t, n: [tuple(sorted((t[0], a, b))) for a, b in itertools.combinations(t[1:5], 2)], 7),
    ("sanrenpuku", "三連複 BOX上位4（現行）",
     lambda t, n: [tuple(sorted(c)) for c in itertools.combinations(t[:4], 3)], 9),
    ("sanrenpuku", "三連複 BOX上位5",
     lambda t, n: [tuple(sorted(c)) for c in itertools.combinations(t[:5], 3)], 9),
    ("sanrenpuku", "三連複 軸2頭×紐3",
     lambda t, n: [tuple(sorted((t[0], t[1], h))) for h in t[2:5]], 7),
    ("sanrentan", "三連単 1着固定×紐3",
     lambda t, n: [(t[0], a, b) for a, b in itertools.permutations(t[1:4], 2)], 6),
    ("sanrentan", "三連単 2着固定×紐3",
     lambda t, n: [(a, t[0], b) for a, b in itertools.permutations(t[1:4], 2)], 6),
    ("sanrentan", "三連単 3着固定×紐3",
     lambda t, n: [(a, b, t[0]) for a, b in itertools.permutations(t[1:4], 2)], 6),
    ("sanrentan", "三連単 マルチ×紐3",
     lambda t, n: [p for a, b in itertools.permutations(t[1:4], 2)
                   for p in ((t[0], a, b), (a, t[0], b), (a, b, t[0]))], 6),
    ("sanrentan", "三連単 2着固定×紐4",
     lambda t, n: [(a, t[0], b) for a, b in itertools.permutations(t[1:5], 2)], 7),
]


def load_all(path):
    """全券種の払戻を1つの辞書にまとめる。{raceid: {券種: {組: 配当}}}"""
    wu, pa, pw = load_wu(path), load_payout_a(path), load_place_wide(path)
    out = {}
    for rid in set(wu) | set(pa) | set(pw):
        e = {}
        if rid in wu:
            e["wakuren"], e["umatan"] = wu[rid]["wakuren"], wu[rid]["umatan"]
        if rid in pa:
            e["umaren"] = pa[rid]["umaren"]
            e["sanrenpuku"], e["sanrentan"] = pa[rid]["sanrenpuku"], pa[rid]["sanrentan"]
        if rid in pw:
            e["fuku"] = {(k,): v for k, v in pw[rid]["fuku"].items()}
            e["wide"] = pw[rid]["wide"]
        out[rid] = e
    return out


def main():
    n_seed = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    d, fx, odds = load()
    y = (d["finish"] <= 3).astype(int).to_numpy()
    cut = d["date"].quantile(0.3)
    tr, te = (d["date"] < cut).to_numpy(), (d["date"] >= cut).to_numpy()
    print(f"train {tr.sum():,} / test {te.sum():,}（分割 {cut.date()}）  シード{n_seed}本"
          f" / 買い方{len(MENU)}通り（事前宣言）")

    pays = load_all(PAYOUT)
    sub = d.loc[te, ["raceid", "umaban", "fieldsize", "finish"]].copy()
    sub["odds"] = odds[te]
    sub["year"] = d.loc[te, "date"].dt.year.to_numpy()

    ps = []
    for s in range(n_seed):
        m = lgb.LGBMClassifier(random_state=s, **PARAMS).fit(fx[tr], y[tr],
                                                             categorical_feature=F.CAT_COLS)
        ps.append(m.predict_proba(fx[te])[:, 1])
        print(f"  seed {s} 完了")

    def run(pvec):
        """1つの予測ベクトルで全メニューを評価。{名前: DataFrame(model/pop の回収倍率)}"""
        sub2 = sub.copy()
        sub2["p"] = pvec
        acc = {name: [] for _, name, _, _ in MENU}
        for rid, g in sub2.groupby("raceid", sort=False):
            pay = pays.get(rid)
            if not pay:
                continue
            n = int(g["fieldsize"].iloc[0])
            # 単勝の払戻はオッズから作る（配当Aに単勝が無いため）
            wnr = g[g["finish"] == 1]
            tansho = {(int(wnr["umaban"].iloc[0]),): float(wnr["odds"].iloc[0]) * 100} \
                if len(wnr) == 1 else {}
            order = {"model": g.sort_values("p", ascending=False, kind="mergesort"),
                     "pop": g.sort_values("odds", ascending=True, kind="mergesort")}
            tops = {k: v["umaban"].astype(int).tolist() for k, v in order.items()}
            for kind, name, fn, minf in MENU:
                table = tansho if kind == "tansho" else pay.get(kind)
                if not table or n < minf or len(g) < minf:
                    continue
                row = {"rid": rid, "year": g["year"].iloc[0]}   # ★対応ありの比較で結合キーになる
                for k, t in tops.items():
                    cs = fn(t, n)
                    row[k] = sum(table.get(c, 0) for c in cs) / (len(cs) * 100.0)
                    row[f"{k}_hit"] = float(any(c in table for c in cs))
                    row[f"{k}_pts"] = len(cs)
                acc[name].append(row)
        return {k: pd.DataFrame(v) for k, v in acc.items()}

    main_tab = run(np.mean(ps, axis=0))
    seed_tabs = [run(p) for p in ps] if n_seed > 1 else []

    rng = np.random.default_rng(0)
    print("\n" + "=" * 118)
    print("買い方の横断比較（モデル順 / 人気順で同じ買い方）")
    print("=" * 118)
    print(f"{'買い方':<26}{'R数':>8}{'点数':>6}{'的中率':>8}{'的中時':>10}"
          f"{'モデルROI':>11}{'95%CI':>16}{'人気順':>8}{'差':>9}{'差のCI':>17}{'シード幅':>9}")
    res = []
    for kind, name, _, _ in MENU:
        df = main_tab[name]
        if len(df) < 500:
            print(f"{name:<26}{len(df):>8,}  標本不足")
            continue
        x, z = df["model"].to_numpy(float), df["pop"].to_numpy(float)
        diff = x - z
        lo, hi = boot(x, rng, 1000)
        dlo, dhi = boot(diff, rng, 1000)
        hit = df["model_hit"].to_numpy(float)
        payhit = (x[hit > 0] * df["model_pts"].to_numpy(float)[hit > 0] * 100).mean() \
            if hit.sum() else 0.0
        sd = [t[name]["model"].mean() * 100 for t in seed_tabs if len(t[name]) >= 500]
        spread = (max(sd) - min(sd)) if len(sd) > 1 else float("nan")
        ys = pd.Series(diff * 100).groupby(df["year"].to_numpy()).mean()
        res.append({"name": name, "kind": kind, "n": len(df), "roi": x.mean() * 100,
                    "pop": z.mean() * 100, "diff": diff.mean() * 100, "dlo": dlo, "dhi": dhi,
                    "spread": spread, "yr_pos": int((ys > 0).sum()), "yr": len(ys),
                    "hit": hit.mean() * 100})
        print(f"{name:<26}{len(df):>8,}{df['model_pts'].mean():>6.1f}{hit.mean()*100:>7.1f}%"
              f"{payhit:>9,.0f}円{x.mean()*100:>10.1f}%{f'[{lo:.1f},{hi:.1f}]':>16}"
              f"{z.mean()*100:>7.1f}%{diff.mean()*100:>+8.2f}pt"
              f"{f'[{dlo:+.1f},{dhi:+.1f}]':>17}{spread:>8.2f}pt")

    r = pd.DataFrame(res)
    print("\n■ ROI上位5（買い方を増やすほど「良く見えるもの」は必ず出る。下の多重性の注記を必ず読むこと）")
    for _, x in r.sort_values("roi", ascending=False).head(5).iterrows():
        print(f"  {x['name']:<26} ROI{x['roi']:>6.1f}%  的中{x['hit']:>5.1f}%  "
              f"人気順比{x['diff']:+.2f}pt [{x['dlo']:+.1f},{x['dhi']:+.1f}]  "
              f"シード幅{x['spread']:.2f}pt  年別+{x['yr_pos']}/{x['yr']}")
    print("\n■ 「人気順に勝つ」がCIで確定した買い方（差のCIが0を跨がない）")
    win = r[r["dlo"] > 0]
    for _, x in win.sort_values("diff", ascending=False).iterrows():
        print(f"  {x['name']:<26} 差{x['diff']:+.2f}pt [{x['dlo']:+.1f},{x['dhi']:+.1f}]  "
              f"ROI{x['roi']:.1f}%  シード幅{x['spread']:.2f}pt")
    print(f"\n★多重性: {len(r)}通り評価した。有意水準5%なら**偶然だけで約{len(r)*0.05:.1f}通り**が"
          f"「差のCIが0を跨がない」を通る。今回通ったのは {len(win)}通り。")
    print("　→ 通った数がこの期待値と同程度なら**何も見つかっていない**のと同じ。"
          "はっきり多いときだけ、上位を(38)の層内シャッフルで再検証すること。")
    print("★どれもROIは100%未満（＝控除率を埋めない）。"
          "「相対的にマシな買い方」の順位づけであって、勝てる買い方の探索ではない。")

    # ===== 現行の買い方 vs 代替を**対応あり**で直接比較 =====
    # 上の表は各買い方を独立に測っただけなので、「AよりBが良い」は言えない。
    # 同じレースで両方買った差を取る（対応あり）と、レース間のばらつきが消えて検出力が上がる。
    PAIRS = [("枠連 軸枠×紐枠2（現行）", "枠連 軸枠×紐枠3"),
             ("枠連 軸枠×紐枠2（現行）", "枠連 上位3頭の枠BOX"),
             ("三連複 BOX上位4（現行）", "三連複 軸1×紐3"),
             ("三連複 BOX上位4（現行）", "三連複 BOX上位5"),
             ("三連複 BOX上位4（現行）", "三連複 軸2頭×紐3"),
             ("枠連 軸枠×紐枠2（現行）", "複勝 top1"),
             ("枠連 軸枠×紐枠2（現行）", "ワイド BOX4")]
    print("\n" + "=" * 118)
    print("現行 vs 代替（**対応あり**＝同じレースで両方買った差。上の表は独立測定なので比較に使えない）")
    print("=" * 118)
    print(f"{'比較':<48}{'R数':>8}{'現行ROI':>10}{'代替ROI':>10}{'差':>10}{'95%CI':>17}"
          f"{'的中率の差':>12}{'シード幅の差':>13}")
    for a, b in PAIRS:
        da, db = main_tab[a], main_tab[b]
        if len(da) < 500 or len(db) < 500:
            continue
        # 同じレース集合に揃える（対象頭数が違うと母集団がずれる）
        j = da.merge(db, on="rid", suffixes=("_a", "_b"))
        if len(j) < 500:
            continue
        x, z = j["model_a"].to_numpy(float), j["model_b"].to_numpy(float)
        diff = z - x
        lo, hi = boot(diff, rng, 2000)
        dh = (j["model_hit_b"] - j["model_hit_a"]).mean() * 100
        sa = [t[a]["model"].mean() * 100 for t in seed_tabs if len(t[a]) >= 500]
        sb = [t[b]["model"].mean() * 100 for t in seed_tabs if len(t[b]) >= 500]
        sp = ((max(sb) - min(sb)) - (max(sa) - min(sa))) if len(sa) > 1 and len(sb) > 1 else float("nan")
        mark = "" if lo <= 0 <= hi else ("  ★代替が上" if diff.mean() > 0 else "  ★現行が上")
        print(f"{a[:16]+' → '+b[:24]:<48}{len(j):>8,}{x.mean()*100:>9.1f}%{z.mean()*100:>9.1f}%"
              f"{diff.mean()*100:>+9.2f}pt{f'[{lo:+.2f},{hi:+.2f}]':>17}"
              f"{dh:>+11.2f}pt{sp:>+12.2f}pt{mark}")
    print("  ※差のCIが0を跨ぐなら**現行を変える理由は無い**。的中率とシード幅の差は参考情報"
          "（ROIが同じなら安定して当たる方が良い、という判断材料）。")


if __name__ == "__main__":
    main()
