"""(179-枠) ★「過去走なし」で黙って落とすのをやめると、枠連のROIは動くか。

★**発端**: 運用ハブの `RACEDAY_GAP_HANDOFF.md`（2026-09-19）——
　**中山3Rで単勝1.5倍・1番人気が両モデルの表に1行も載らないまま1着**になった。
　**12頭立てなのに表は10頭**。⚠**原因はその馬の唯一の前走(7/25)の結果CSVが無かったこと**。
　★**7/25は埋まった**が、**「過去走が繋がらない馬を落とす」という扱いそのものは残っている**。

★**現行の扱い**: `keep = (f["n_prior"] >= 1) & d["odds"].notna() & (d["odds"] > 0)`
　⚠**学習でも予測でも落としている**。
★**だが44特徴量のうち `log_odds` / `mkt_prob` / `sex` / `age` / `course` / `distance` /
　`sire` / `damsire` / `jockey` / `trainer` などは新馬でも作れる**。
　**LightGBMはNaNを扱える**ので、★**過去走系だけNaNにして残すことは技術的に可能**。

★**この筋は (176)(177)(178) の3連敗とは別物**——**新しい信号を足すのではなく、
　すでに手元にある行を捨てるのをやめる**話。**判定基準40の「3つ目を探すな」には当たらない**と読む。

⚠⚠★**事前登録を書いている時点で、落ちた馬が勝つかどうかの数字は一切見ていない**。
　**先に見ると(172)の「跳ねたマスを追う」になる**（ハブも判定基準42で先に釘を刺してきた）。

━━━ ★★事前登録（**結果を見る前に書く**。腕は2つ。後から増やさない）━━━

| | 学習 | 予測 |
|---|---|---|
| **対照（現行）** | n_prior≥1 のみ | n_prior≥1 のみ |
| ★**腕①（予測だけ緩める）** | n_prior≥1 のみ（★**現行のモデルそのまま**） | ★**n_prior≥0**（過去走系はNaN） |
| ★**腕②（学習も緩める）** | ★**n_prior≥0** | ★**n_prior≥0** |

★**レース集合は3腕で同じ**——**対照が判定できるレース**（**n_prior≥1 の馬が3頭以上**）
　**かつ枠連の払戻があるもの**に限る。★**対応のある比較にするため**。
⚠**全馬が新馬のレース（現行では「予想不可」）は主判定に入れない**。**記述として別に出す**。
　★**丸ごと増えるレースは「落とす扱いの是非」より大きな変更**なので、**問いを混ぜない**。

★★**(170)の罠を構造で潰す**——**対照も腕も除外40%で、★買うレース数が同じ**。
　**「外せば必ずプラス」が起きない**。**違うのは「どのレース／どの枠組を選ぶか」だけ**。
　★**だから対応差は仮説が偽なら0を返す**（判定基準42）。

★**主判定**: **1レースあたり損益の対応のある差**（腕 − 対照）。**99%CI・Bonferroni α=0.01/2**。
　⚠**買わなかったレースも0円として全判定レースを母数に入れる**（(168)の失敗を繰り返さない）。

★★**プラセボ（腕①のみ・判定基準43に従い10種の平均）**:
　★**新馬を足す代わりに、同じ数だけ「すでに点のある馬」を無作為に落とす**。
　**表の頭数が同じだけ動く**ので、★**「表が変わったこと自体の効果」と切り分けられる**。
　⚠**腕②は再学習が要るのでプラセボを置けない**。**その限界を明記して読む**。

★**採用条件（3つ全部）**:
　**① 対応差の99%CI(Bonf)がゼロを外し、★プラス側**
　**② 年別で符号がそろった年が ★ceil(0.8×年数) 以上**
　　（⚠**(178)で「8/10」と書いたのに後半が5年しかなく満たしようがなかった**。**母集団から決める**）
　**③ 腕①はプラセボ(10種平均)を上回る**

⚠**ゲート（判定基準32）**: **対照の 除外0%・紐1 の ROI が (155) の既知値 85.2% と ±2pt**。
　⚠**腕②は再学習するのでこのゲートは当てない**（**装置が変わるのは織り込み済み**）。
⚠**内部対照は最初に出す**: **レース数**と、★**実際に馬が増えたレースの割合**。
　★**増えたレースが少なければ、どんな結果でも効果は小さい**。**先に見ておく**。

★**予想**: ⚠**当てにしないが書く**——**落ちる馬の大半は新馬で、市場は既に値段を付けている**
　（`log_odds` は腕でも使える）。★**ROIは動かないほうに賭ける**。
　⚠**9/19の1件は逸話であって根拠ではない**（判定基準39）。

実行: python3 ml/audit_keep_newcomers.py
"""
import math
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "ml")
import features as F
from audit_crosspool import load_races, payoff, zq
from audit_crosspool2 import realized
from train_prod import CAPACITY, add_odds_features, fit_seeds
from waku_umatan import bracket_probs, waku_of, waku_score, wakuren_buy

KNOWN_ROI, TOL = 85.2, 2.0
EXCL, NCMP, NSEED = 0.40, 2, 10
MIN_SCORED = 3


def sim(sub, races):
    out = []
    for rid, g in sub.groupby("raceid"):
        r = races.get(str(rid))
        if r is None or not r.get("wakuren"):
            continue
        rl = realized(r)
        if rl is None:
            continue
        a, b0, _ = rl
        n, nums = r["n"], [u for u, _, _ in r["horses"]]
        if a not in nums or b0 not in nums:
            continue
        gg = g.sort_values("p", ascending=False)
        order = [int(u) for u in gg["umaban"].tolist()]
        if len(order) < MIN_SCORED:
            continue
        key = tuple(sorted((waku_of(a, n), waku_of(b0, n))))
        v = payoff(r, "枠連(人気順)", [key[0], key[1]])
        if not v or v <= 0:
            continue
        pv = gg["p"].to_numpy(float)
        q = pv / pv.sum()
        bp = bracket_probs(order, q, n)
        pairs = wakuren_buy(order, n, 1)
        out.append({"raceid": str(rid), "year": int(r["year"]), "ntab": len(order),
                    "sc": float(waku_score(pairs, bp)),
                    "ret": (v if key in pairs else 0.0), "cost": 100.0 * len(pairs)})
    return pd.DataFrame(out)


def prof(t, k):
    p = np.zeros(len(t))
    m = k.to_numpy() if hasattr(k, "to_numpy") else k
    p[m] = t["ret"].to_numpy()[m] - t["cost"].to_numpy()[m]
    return p


def keepmask(t, rate=EXCL):
    return t["sc"] >= np.quantile(t["sc"], rate)


def paired(ta, tb):
    """raceid で揃えた 1Rあたり損益の対応差（腕 − 対照）。"""
    A = pd.DataFrame({"raceid": ta["raceid"], "a": prof(ta, keepmask(ta))})
    B = pd.DataFrame({"raceid": tb["raceid"], "b": prof(tb, keepmask(tb))})
    m = A.merge(B, on="raceid")
    return (m["a"] - m["b"]).to_numpy(), m["raceid"].to_numpy()


def main():
    MODEL_DIR, PAR = CAPACITY["l2"]
    d0 = F.to_model(F.load_files())
    f0 = F.build_features(d0)
    oddsok = d0["odds"].notna() & (d0["odds"] > 0)
    strict = (f0["n_prior"] >= 1) & oddsok
    loose = oddsok
    y0 = (d0["finish"] <= 3).astype(int).to_numpy()
    fxa, _ = F.encode_categoricals(f0)
    fxa = add_odds_features(fxa, d0["odds"].to_numpy(float), d0["raceid"].to_numpy())
    cut = d0["date"].quantile(0.3)
    early = (d0["date"] < cut).to_numpy()

    print("(179-枠) ★「過去走なし」で落とすのをやめると枠連のROIは動くか")
    print("■ 内部対照（母集団を最初に出す）")
    print(f"　全体 {len(d0):,}行 / n_prior≥1 {int(strict.sum()):,} / オッズありの全部 {int(loose.sum()):,}")
    print(f"　★落としている行 {int((loose & ~strict).sum()):,}"
          f"（{100*(loose & ~strict).sum()/max(loose.sum(),1):.1f}%）")

    races = {r["rid"]: r for r in load_races()}
    S, L = strict.to_numpy(), loose.to_numpy()
    ms_s = fit_seeds(fxa[S & early], y0[S & early], 3, PAR)
    ms_l = fit_seeds(fxa[L & early], y0[L & early], 3, PAR)
    print("　学習ずみ 対照(n_prior≥1) / 腕②(n_prior≥0)")

    def table(mask, models):
        te = mask & ~early
        p = np.mean([m.predict_proba(fxa[te])[:, 1] for m in models], axis=0)
        s = d0.loc[te, ["raceid", "umaban"]].copy()
        s["p"] = p
        return sim(s, races)

    t_ctl = table(S, ms_s)
    t_a1 = table(L, ms_s)       # ★腕①: 現行モデルのまま、予測だけ緩める
    t_a2 = table(L, ms_l)       # ★腕②: 学習も緩める
    # ★レース集合を対照に揃える（全馬新馬のレースは主判定に入れない）
    ok = set(t_ctl["raceid"])
    t_a1 = t_a1[t_a1["raceid"].isin(ok)].reset_index(drop=True)
    t_a2 = t_a2[t_a2["raceid"].isin(ok)].reset_index(drop=True)
    print(f"　★突き合わせ {len(t_ctl):,}レース（対照が判定できたもの）")
    mg = t_ctl[["raceid", "ntab"]].merge(t_a1[["raceid", "ntab"]], on="raceid",
                                         suffixes=("_c", "_a"))
    grew = (mg["ntab_a"] > mg["ntab_c"])
    print(f"　★★馬が増えたレース {int(grew.sum()):,} / {len(mg):,} = {100*grew.mean():.1f}%"
          f"（増えた頭数 中央 {int((mg['ntab_a']-mg['ntab_c'])[grew].median() or 0)}）")

    roi0 = 100.0 * t_ctl["ret"].sum() / t_ctl["cost"].sum()
    gate = abs(roi0 - KNOWN_ROI) <= TOL
    print(f"\n■ ⚠ゲート: 対照の 除外0%・紐1 ROI {roi0:.1f}%（既知 {KNOWN_ROI}%）"
          f" → {'★一致。読む' if gate else '⚠外れた。読まない'}")
    if not gate:
        return

    kc = keepmask(t_ctl)
    roic = 100.0 * t_ctl["ret"].to_numpy()[kc.to_numpy()].sum() \
        / t_ctl["cost"].to_numpy()[kc.to_numpy()].sum()
    z = zq(0.01 / NCMP)
    print(f"\n■ ★★主判定（除外40%・紐1／対照 ROI {roic:.1f}%・"
          f"★買うレース数は3腕とも同じ {int(kc.sum()):,}）")
    print(f"{'腕':>8}{'ROI':>8}{'1Rあたり対応差':>16}{'99%CI(Bonf)':>24}")
    res = {}
    for nm, t in (("腕①", t_a1), ("腕②", t_a2)):
        dd, _ = paired(t, t_ctl)
        k = keepmask(t)
        rr = 100.0 * t["ret"].to_numpy()[k.to_numpy()].sum() / t["cost"].to_numpy()[k.to_numpy()].sum()
        md, se = dd.mean(), dd.std(ddof=1) / math.sqrt(len(dd))
        res[nm] = (md, md - z * se, md + z * se)
        print(f"{nm:>8}{rr:>7.1f}%{md:>+16.2f}円{f'[{md-z*se:+.2f}, {md+z*se:+.2f}]':>24}")

    # ★プラセボ（腕①のみ・10種）: 新馬を足す代わりに、同数の「点のある馬」を無作為に落とす
    print(f"\n■ ★プラセボ（腕①のみ・{NSEED}種の平均）— 同じ数だけ無作為に落とす")
    add = (mg["ntab_a"] - mg["ntab_c"]).to_numpy()
    need = dict(zip(mg["raceid"].to_numpy(), add))
    plc = []
    for sd in range(NSEED):
        rng = np.random.default_rng(sd)
        sub = d0.loc[S & ~early, ["raceid", "umaban"]].copy()
        p = np.mean([m.predict_proba(fxa[S & ~early])[:, 1] for m in ms_s], axis=0)
        sub["p"] = p
        drop = []
        for rid, g in sub.groupby("raceid"):
            k = int(need.get(str(rid), 0))
            if k > 0 and len(g) - k >= MIN_SCORED:
                drop += list(rng.choice(g.index.to_numpy(), size=k, replace=False))
        t_p = sim(sub.drop(index=drop), races)
        t_p = t_p[t_p["raceid"].isin(ok)].reset_index(drop=True)
        dd, _ = paired(t_p, t_ctl)
        plc.append(dd.mean())
    pm = float(np.mean(plc))
    print(f"　プラセボ平均 {pm:+.2f}円　種ごとの幅 [{min(plc):+.2f}, {max(plc):+.2f}]"
          f"　標準偏差 {np.std(plc):.2f}")

    print("\n■ 採用条件② 年別の符号")
    ok2 = {}
    for nm, t in (("腕①", t_a1), ("腕②", t_a2)):
        sg = []
        for Y in sorted(set(t_ctl["year"])):
            ta = t[t["year"] == Y].reset_index(drop=True)
            tc = t_ctl[t_ctl["year"] == Y].reset_index(drop=True)
            if len(ta) < 300 or len(tc) < 300:
                continue
            dd, _ = paired(ta, tc)
            sg.append(dd.mean())
        need_y = math.ceil(0.8 * len(sg))
        n_ok = max(sum(1 for s in sg if s > 0), sum(1 for s in sg if s < 0))
        ok2[nm] = (n_ok, len(sg), need_y)
        print(f"　{nm}: そろった年 {n_ok}/{len(sg)}（必要 {need_y}）"
              f"　（{' '.join(f'{s:+.1f}' for s in sg)}）")

    print("\n" + "=" * 88)
    any_ok = False
    for nm in ("腕①", "腕②"):
        md, lo, hi = res[nm]
        c1 = lo > 0
        c2 = ok2[nm][0] >= ok2[nm][2]
        c3 = (md > pm) if nm == "腕①" else None
        okk = c1 and c2 and (c3 is not False)
        any_ok |= okk
        s3 = "—（プラセボ無し）" if c3 is None else ("★" if c3 else "⚠")
        print(f"　{nm}: ①CI>0 {'★' if c1 else '⚠'} / ②符号 {ok2[nm][0]}/{ok2[nm][1]}"
              f" {'★' if c2 else '⚠'} / ③プラセボ超え {s3}"
              f"　→ **{'★採用候補' if okk else '⚠不採用'}**")
    if not any_ok:
        print("\n⚠★どちらも3条件を満たさない＝**落とす扱いを変えてもROIは動かない**と読む。")
        print("　★**この線はここで閉じる。運用は変えない**（判定基準40）。")
    else:
        print("\n★★採用候補が出た。⚠**運用にすぐ入れない**。★材料として利用者とハブに渡す。")
        print("　⚠**腕②はプラセボを置けていない**（再学習が要るため）。**その限界つきで読む**。")


if __name__ == "__main__":
    main()
