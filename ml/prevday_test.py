"""
★前日オッズで運用した場合の劣化を**実測**する（(59)の感度分析の置き換え）。

(59) `odds_sensitivity.py` は確定オッズに対数正規ノイズを乗せた**感度分析**で、
「直前に情報が集まって金が動く」という系統的な変化を再現できていなかった。
`data/odds_ts/`（TARGETの時系列オッズ）が入ったので、本物のオッズで測る。

測り方（実運用そのままの手順を再現する）:
  ・モデルは**確定オッズで学習**（過去データには確定値しか無いので実運用でもこうなる）
  ・予測時には**その時刻に実際に見えていたオッズ**を与えて買い目を決める
  ・精算は**実際の払戻**（配当A）で行う
これは train/serve 不一致そのもので、実運用で起きることと同じ。

出す指標は2つ:
  ① 買い目の一致率 … 確定オッズで決めた買い目と何%一致するか。1レース1判定なので**標本が少なくても効く**
  ② ROI … 参考値。標本が数百Rでは誤差±10pt級になるので**これで結論を出さない**（(46)の教訓）

実行: python3 ml/prevday_test.py [シード数(既定3)]
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
from odds_ts import load_dir, odds_at
from place_wide import boot, PARAMS
from pocket_eval import load_payout_a
from waku_umatan import load_wu, waku_of, wakuren_buy

# 測る時点。名前 → odds_at の指定
WHEN = [("確定（基準）", ("final",)),
        ("発走10分前", ("before", 10)),
        ("発走30分前", ("before", 30)),
        ("当日9時", ("day", 9, 0)),
        ("前日22時", ("prev", 22, 0))]


def buys(nums, n):
    """モデル順(nums)から買い目を作る。predict.py と同じ定義。"""
    return {"軸": nums[0],
            "枠連": tuple(wakuren_buy(nums, n, 2)),
            "三連複BOX4": tuple(sorted(tuple(sorted(c)) for c in itertools.combinations(nums[:4], 3)))}


def main():
    n_seed = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    ts = load_dir()
    print(f"時系列オッズ {len(ts)}レース "
          f"{min(r['date'] for r in ts.values()).date()}〜{max(r['date'] for r in ts.values()).date()}")

    d = F.to_model(F.load_files())
    f = F.build_features(d)
    keep = (f["n_prior"] >= 1) & d["odds"].notna() & (d["odds"] > 0)
    d, f = d[keep].reset_index(drop=True), f[keep].reset_index(drop=True)
    top3 = (d["finish"] <= 3).astype(int).to_numpy()
    odds = d["odds"].to_numpy(float)
    rid = d["raceid"].to_numpy()
    inv = 1.0 / odds
    fx, _ = F.encode_categoricals(f)
    fx["log_odds"] = np.log(odds)
    fx["mkt_prob"] = inv / pd.Series(inv).groupby(rid).transform("sum").to_numpy()

    # 学習は(55)(62)と同じ前30%。評価対象(2026-06以降)は当然 test 側に入る
    cut = d["date"].quantile(0.3)
    tr = (d["date"] < cut).to_numpy()
    tgt = np.isin(rid, list(ts.keys()))
    print(f"学習 {tr.sum():,}行 / 評価対象 {tgt.sum():,}行（{pd.Series(rid[tgt]).nunique()}レース）")
    if tgt.sum() == 0:
        sys.exit("評価対象が0。時系列オッズのレースが成績データに存在しない（成績データが古い可能性）")

    models = [lgb.LGBMClassifier(random_state=s, **PARAMS).fit(fx[tr], top3[tr],
                                                               categorical_feature=F.CAT_COLS)
              for s in range(n_seed)]
    print(f"学習完了（シード{n_seed}本）\n")

    wu, pa = load_wu("data/payout/a.csv"), load_payout_a("data/payout/a.csv")
    sub = d.loc[tgt, ["raceid", "umaban", "fieldsize"]].copy()
    fx_t = fx.loc[tgt].reset_index(drop=True)
    sub = sub.reset_index(drop=True)

    # 各時点のオッズベクトルを作る（レース×馬番 → その時刻のオッズ）
    res, skipped = {}, 0
    for name, w in WHEN:
        o = np.full(len(sub), np.nan)
        for r_id, g in sub.groupby("raceid", sort=False):
            rec = ts.get(r_id)
            v = odds_at(rec, w) if rec else None
            if v is None:
                continue
            for i, u in zip(g.index, g["umaban"].astype(int)):
                if 1 <= u <= len(v):
                    o[i] = v[u - 1]
        ok = np.isfinite(o) & (o > 0)
        fxx = fx_t.copy()
        fxx.loc[~ok, :] = fxx.loc[~ok, :]          # 欠損行はそのまま（下でレースごと除外）
        oo = np.where(ok, o, np.nan)
        s = pd.Series(1.0 / oo)
        fxx["log_odds"] = np.log(oo)
        fxx["mkt_prob"] = (s / s.groupby(sub["raceid"].to_numpy()).transform("sum")).to_numpy()
        p = np.mean([m.predict_proba(fxx)[:, 1] for m in models], axis=0)
        res[name] = (p, ok)

    # レースごとに買い目を作る
    base_name = WHEN[0][0]
    out = {name: {"match": {"軸": [], "枠連": [], "三連複BOX4": []}, "w": [], "s": [],
                  "ov": [], "wh": [], "sh": []} for name, _ in WHEN}
    for r_id, g in sub.groupby("raceid", sort=False):
        n = int(g["fieldsize"].iloc[0])
        if n < 9:
            continue
        idx = g.index.to_numpy()
        if not all(res[nm][1][idx].all() for nm, _ in WHEN):
            skipped += 1
            continue
        w_pay, s_pay = wu.get(r_id), pa.get(r_id)
        base = None
        for name, _ in WHEN:
            order = idx[np.argsort(-res[name][0][idx], kind="mergesort")]
            nums = g.loc[order, "umaban"].astype(int).tolist()
            b = buys(nums, n)
            if name == base_name:
                base = b
            for k in b:
                out[name]["match"][k].append(int(b[k] == base[k]))
            # 枠連は完全一致だけだと厳しすぎる（1点でも重なれば的中の芽は残る）ので重なり率も見る
            bs, cs0 = set(b["枠連"]), set(base["枠連"])
            out[name]["ov"].append(len(bs & cs0) / max(len(cs0), 1))
            if w_pay and w_pay["wakuren"]:
                cs = list(b["枠連"])
                pay = sum(w_pay["wakuren"].get(c, 0) for c in cs)
                out[name]["w"].append(pay / (len(cs) * 100))
                out[name]["wh"].append(int(pay > 0))
            if s_pay and s_pay["sanrenpuku"]:
                cs = list(b["三連複BOX4"])
                pay = sum(s_pay["sanrenpuku"].get(c, 0) for c in cs)
                out[name]["s"].append(pay / 400)
                out[name]["sh"].append(int(pay > 0))

    nR = len(out[base_name]["match"]["軸"])
    print(f"評価 {nR}レース（9頭以上・全時点でオッズが揃ったもの。除外 {skipped}）\n")
    rng = np.random.default_rng(0)
    print("【買い目がどれだけ変わるか】確定オッズで決めた買い目との比較")
    print(f"{'時点':<12}{'軸が同じ':>9}{'枠連が同じ':>11}{'枠連の重なり':>13}{'BOX4が同じ':>11}")
    for name, _ in WHEN:
        m = {k: np.mean(v) * 100 for k, v in out[name]["match"].items()}
        print(f"{name:<12}{m['軸']:>8.1f}%{m['枠連']:>10.1f}%{np.mean(out[name]['ov'])*100:>12.1f}%"
              f"{m['三連複BOX4']:>10.1f}%")

    se = lambda x: np.sqrt(np.mean(x) * (1 - np.mean(x)) / len(x)) * 100
    print(f"\n【当たったか／回収できたか】{nR}レース")
    print(f"{'時点':<12}{'枠連 的中率':>13}{'差(基準比)':>12}{'枠連ROI':>10}{'95%CI':>13}"
          f"{'BOX4 的中率':>13}{'BOX4 ROI':>10}")
    base_wh = np.array(out[base_name]["wh"])
    base_sh = np.array(out[base_name]["sh"])
    for name, _ in WHEN:
        o = out[name]
        xw, xs = np.array(o["w"]), np.array(o["s"])
        wh, sh = np.array(o["wh"]), np.array(o["sh"])
        lo, hi = boot(xw, rng, 2000) if len(xw) else (0, 0)
        # 同じレースを比べているので**対応のある差**の標準誤差を使う（独立標本の式より小さくなる）
        dwh = (wh - base_wh)
        sed = np.std(dwh, ddof=1) / np.sqrt(len(dwh)) * 100
        dtxt = "—" if name == base_name else f"{dwh.mean()*100:+.1f}±{sed:.1f}pt"
        print(f"{name:<12}{wh.mean()*100:>11.1f}%{dtxt:>13}{xw.mean()*100:>9.1f}%"
              f"{f'[{lo:.0f},{hi:.0f}]':>13}{sh.mean()*100:>12.1f}%{xs.mean()*100:>9.1f}%")
    print(f"\n※的中率は1レース1判定なので{nR}Rでも誤差±{se(base_wh):.1f}pt程度。ROIは配当のばらつきが大きく")
    print("　この標本数では判定に使えない（(46)の教訓）。**的中率の差で読むこと**。")


if __name__ == "__main__":
    main()
