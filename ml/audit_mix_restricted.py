"""(107) ★★特徴を**わざと制限した**専門家を混ぜる — (102)で分かった原理を素直に押す。

★(102)で分かった原理
　・**目標を変えるだけでは効かない**（L2-top3 単独の重み 0.009）
　・**容量を変えると効く**（L5を足すと利得 +0.0019 → +0.0024、しかも top3 の重みが 0.047 に復活）
　→ **違う誤り方をする予測器ほど混合の相手として価値がある**。
　→ ならば **わざと情報を絞ったモデル**が最も違う誤り方をするはず。これが本命。

★★用意する専門家（**なぜ違う誤り方をするか**を先に書く）
　1. `noodds` … **オッズを落とす**（log_odds, mkt_prob を除く）
　　　現行モデルは(45)より**オッズに強く依存**している。落とすと市場と相関しない誤りをする。
　　　★(24)〜(26)で「オッズ無しモデルは賭ける価値なし」と結論しているが、
　　　　**あれは単体で使う話**。混合の相手としての価値は**一度も測っていない**。
　2. `formonly` … **近走成績だけ**（n_prior〜fin_std ＋ 距離・頭数）。血統・騎手・厩舎・オッズを落とす
　3. `idonly` … **識別子だけ**（sire/damsire/jockey/trainer/course/sex ＋ オッズ）。
　　　近走を見ずに「誰が乗るか・どこの厩舎か」だけで決める＝**記憶に頼る**モデル
　★どれも単体では現行より弱いはず。**それでも混ぜると増えるか**が問い。

★★事前登録（測る前に宣言）
　1. **予想**: `noodds` が最も効く（市場と最も相関しない）。3人足して利得は
　　 **+0.0024 → 0.003〜0.006** と予想する。
　2. **重みは各年それ以前の年だけ**で推定（座標降下・w≥0・Σw=1）。
　3. **★プラセボ**: 全専門家をレース内シャッフル。重みが市場に潰れ利得が消えることを確認。
　4. **判定**: 利得が(102)の +0.0024 を超え、CIが0を除外し、プラセボが通ること。
　5. **シードは1本**。★(102)③で **Dはシード数に依存しない**と実測したので1本で足りる。
　　 （判定基準1「1シードで報告しない」は**部分集合ROIの話**で、全レース集計のDには当てはまらない）
　6. ⚠**専門家が6人になると自由度が増える**。ウォークフォワードとプラセボの2枚で守る。
　　 **利得が増えてもプラセボが通らなければ全部無効**。

実行: python3 ml/audit_mix_restricted.py [開始年(既定2015)]
"""
import sys

sys.path.insert(0, "ml")
import audit_mix_multi as M

# 近走成績のかたまり（build_features が作る順に並べてある）
FORM = ["n_prior", "last1_fin", "last2_fin", "last3_fin", "avg3_fin", "best3_fin",
        "last_margin", "last_passratio", "avg3_passratio", "last_agari", "best3_agari",
        "avg3_prize", "fin_std", "dist_change", "same_dist", "surf_change", "wt_change",
        "weeks_since", "weeks_before", "bodywt", "bodywt_change"]
IDS = ["sire", "damsire", "jockey", "trainer", "course", "sex", "cond"]
ODDS = ["log_odds", "mkt_prob"]
RACE = ["distance", "surface", "fieldsize", "wtcarry", "age", "raceclass", "month", "season"]

# 名前 → 使う列（None は全列）。★落とす理由は docstring に書いた
RESTRICT = {
    "noodds": ("drop", ODDS),
    "formonly": ("keep", FORM + ["distance", "fieldsize"]),
    "idonly": ("keep", IDS + ODDS + ["distance", "fieldsize"]),
}


def patched_expert_probs(name, target, cap, y0, n_seed=3):
    """(102)の学習関数を、列を絞れるように差し替えたもの。"""
    import os
    import numpy as np
    import pandas as pd
    import lightgbm as lgb
    import features as F
    from _cache import load_cached
    from place_wide import PARAMS

    dirp = f"data/cache/exp_{name}_{y0}"
    os.makedirs(dirp, exist_ok=True)
    par = PARAMS if cap == "l2" else dict(PARAMS, num_leaves=255,
                                          min_child_samples=10, n_estimators=2000)
    d, fx = load_cached()
    if name in RESTRICT:
        how, cols = RESTRICT[name]
        use = [c for c in fx.columns if c not in cols] if how == "drop" \
            else [c for c in fx.columns if c in cols]
        fx = fx[use]
    cats = [c for c in F.CAT_COLS if c in fx.columns]
    y = ((d["finish"] == 1) if target == "win" else (d["finish"] <= 3)).astype(int).to_numpy()
    year = d["date"].dt.year.to_numpy()
    years = [yy for yy in range(y0, int(year.max()) + 1) if (year == yy).sum() > 5000]
    for yy in years:
        f = f"{dirp}/{yy}.csv"
        if os.path.exists(f):
            continue
        tr, te = year < yy, year == yy
        p = np.mean([lgb.LGBMClassifier(random_state=s, **par)
                     .fit(fx[tr], y[tr], categorical_feature=cats)
                     .predict_proba(fx[te])[:, 1] for s in range(n_seed)], axis=0)
        sub = d.loc[te, ["raceid", "umaban"]].copy()
        sub["p"] = p
        sub.to_csv(f, index=False)
        print(f"    {name} {yy} 学習完了・保存（{len(fx.columns)}列）", flush=True)
    del d, fx
    big = pd.concat([pd.read_csv(f"{dirp}/{yy}.csv") for yy in years], ignore_index=True)
    out = {}
    for rid, um, p in zip(big["raceid"], big["umaban"].astype(int), big["p"]):
        out.setdefault(rid, {})[int(um)] = float(p)
    return out


# ★(102)の main をそのまま使い、専門家の定義と学習関数だけ差し替える
M.expert_probs = patched_expert_probs
M.ALL_EXPERTS = [
    ("L2-win", "win", "l2", 3),
    ("L2-top3", "top3", "l2", 3),
    ("L5-win", "win", "l5", 3),
    ("noodds", "win", "l2", 1),      # ★オッズを落とす（本命）
    ("formonly", "win", "l2", 1),    # 近走成績だけ
    ("idonly", "win", "l2", 1),      # 識別子＋オッズだけ
]
M.EXPERTS = M.ALL_EXPERTS

if __name__ == "__main__":
    print(__doc__.split("実行:")[0])
    M.main()
