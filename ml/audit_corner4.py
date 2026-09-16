"""(176-枠) ★4コーナーの形は、モデルが既に拾っているか。**安いゲート1本だけ**。

★**発端**: SNS設計セッションからの共有（2026-09-16）——
　**`features.to_model` が4コーナーの通過順を平均1つに潰している**（`features.py:51-52`）。
　**モデルに届くのは `last_passratio` / `avg3_passratio`（= passavg/fieldsize）だけ**で、
　★**4角単独の位置は44特徴量のどこにも入っていない**（確認済み）。

★**ここまでに分かっていること（事前測定・このスクリプトの外）**
　・`corr(4角, 平均) = 0.9598`。**ばらつきの約31%は平均で説明できない**。
　・**前走の4角残差 と 今走3着以内 の偏相関**（前走の平均位置と log単勝 を差し引いた後）
　　**= −0.0943  99%CI [−0.0965, −0.0921]  n=1,369,961** ＝ **ゼロを外す**。
　⚠**符号は発端の仮説と逆だった**——**残差がマイナス＝4角までに押し上げた馬**のほうが次走で走る。
　　★**発端（穴馬側）では、これは機構の否定になる**（「市場が見落とす穴馬」の話が成り立たない）。
　　⚠★**だがこのセッションには関係しない**——**枠連側は穴馬を探していない**。
　　**軸とレースを選ぶだけ**なので、★**符号がどちら向きでも、独立な情報が在るなら使える**。
　　→ ★**判定は「符号の向き」ではなく「モデルが既に拾っているか」だけ**。
　⚠**対照が弱い**: 差し引いたのは2つだけで、**モデルは44特徴量を持つ**。

━━━ ★★事前登録（**結果を見る前に書く**。このゲート1本だけ。後から増やさない）━━━

★**問い**: **モデルの予測 `p` を差し引いても、4角残差に独立な情報が残るか**。
　★**残らなければ、絞りを作る前にここで閉じる**。**絞りは作らない**。

★**主判定**: 検証期間で **偏相関 corr(prev_res, top3 | logit p)**。99%CI。
　★**仮説が偽（モデルが既に拾っている）なら 0 を返す**（判定基準42）。
　　**`p` は log_odds も mkt_prob も特徴量として含む**ので、**市場の分も同時に差し引かれる**。

★★**対照（判定基準45: 乱を使わない決定的な量に置く）**:
　★**`prev_ra`（前走の平均位置）＝ `last_passratio` として★確実にモデルに入っている変数**で
　　**同じ偏相関を計算する**。→ ★**これが「完全に吸収された変数の偏相関」の物差し**になる。
　⚠**恣意的な閾値を置かない**。**対照と比べる**。

★**採用条件（判定基準39/40。有意なだけでは足りない）**:
　**① `prev_res` の偏相関の絶対値が、対照 `prev_ra` のそれを明確に上回る**
　**② 年別で符号が 8/10年以上そろう**
　★**両方を満たさなければ「モデルが既に拾っている」と読み、この線を閉じる**。
　⚠**n が137万あるので、有意性はいくらでも出る**。**大きさと安定性で判定する**。

⚠**ゲート（判定基準32）**: **検証期間のモデルAUCが (174-枠) の 0.8027 と ±0.01 で一致すること**。
　**外れたら装置が違うので、主判定を読まない**。

⚠**内部対照は最初に出す**: **検証行数**（母集団の取り違えを防ぐ・穴馬側は9回踏んだ）。

★**参考列（2026-09-16 追加・★採用条件には使わない）**: **偶数年で学習し、奇数年で検証**。
　★**利点**: **両側が11年に均等に散る**ので、**時代の変化が効果の推定に乗らない**。
　⚠**主判定にできない理由**: **未来を見る**（2018年で学習して2017年を予測する）。
　　★**このゲートでは特に厄介**——**`p` が未来を見て強くなると差し引かれる量が過大になり、
　　「モデルが既に拾っている」側＝**線を閉じる方向に間違える**。
　　★**判定基準46（前半で選び後半で1回だけ試す）も時間順が前提**。
　→ ★**時間順を主判定に置き、偶奇は「時代の変化か本物か」の切り分けとして並べるだけ**。

実行: python3 ml/audit_corner4.py
"""
import math
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "ml")
import features as F
from train_prod import CAPACITY, add_odds_features, fit_seeds

AUC_REF = 0.8027        # (174-枠) の検証全体AUC
NYEAR_OK = 8            # 採用条件②


def auc(y, s):
    o = np.argsort(s, kind="mergesort")
    r = np.empty(len(s), float)
    r[o] = np.arange(1, len(s) + 1)
    n1 = int(y.sum()); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0) if n1 and n0 else float("nan")


def pcorr(x, y, Z):
    """Z を差し引いた後の x と y の偏相関と99%CI（Fisher変換）。"""
    def res(v):
        b = np.linalg.lstsq(Z, v, rcond=None)[0]
        return v - Z @ b
    rx, ry = res(x), res(y)
    r = float(np.corrcoef(rx, ry)[0, 1])
    n = len(x); se = 1.0 / math.sqrt(n - Z.shape[1] - 1)
    z = 0.5 * math.log((1 + r) / (1 - r))
    return r, math.tanh(z - 2.576 * se), math.tanh(z + 2.576 * se)


def main():
    MODEL_DIR, PAR = CAPACITY["l2"]
    raw = F.load_files()
    raw = raw[raw[40].str.len() > 2]
    # ★4角の生値はここでしか取れない（to_model が平均に潰すため）
    # ⚠★`to_model` は dropna と drop_duplicates で行を落とす（1,499,882 → 669,951）。
    # 　**位置で対応づけると壊れる**（2026-09-16に実際に踏んだ）。★**col40（raceid+馬番）で突き合わせる**。
    p4 = raw[[28, 29, 30, 31]].apply(pd.to_numeric, errors="coerce").where(lambda x: x > 0)
    key_raw = raw[40].str.strip()
    c4_raw = p4[31]
    d = F.to_model(raw)
    f = F.build_features(d)
    keep = (f["n_prior"] >= 1) & d["odds"].notna() & (d["odds"] > 0)

    # ★4角比と「平均で説明できない成分」を作る（to_model と同じ行の並びで）
    fs = d["fieldsize"].to_numpy(float)
    # ★col40 で引き当てる。重複キーは to_model と同じく最初の1行を採る
    c4map = pd.Series(c4_raw.to_numpy(), index=key_raw.to_numpy())
    c4map = c4map[~c4map.index.duplicated(keep="first")]
    key_d = d["raceid"].astype(str) + d["umaban"].astype(int).astype(str).str.zfill(2)
    r4 = c4map.reindex(key_d.to_numpy()).to_numpy(float) / fs
    ra = d["passavg"].to_numpy(float) / fs
    print(f"　4角が引けた行 {int(np.isfinite(r4).sum()):,} / {len(d):,}")
    ok = np.isfinite(r4) & np.isfinite(ra)
    X = np.c_[np.ones(ok.sum()), ra[ok]]
    b = np.linalg.lstsq(X, r4[ok], rcond=None)[0]
    res = np.full(len(d), np.nan)
    res[ok] = r4[ok] - X @ b

    aux = pd.DataFrame({"horse": d["horse"].to_numpy(), "date": d["date"].to_numpy(),
                        "res": res, "ra": ra})
    aux = aux.sort_values(["horse", "date"])
    g = aux.groupby("horse")
    aux["prev_res"] = g["res"].shift(1)
    aux["prev_ra"] = g["ra"].shift(1)
    aux = aux.sort_index()

    d, f = d[keep].reset_index(drop=True), f[keep].reset_index(drop=True)
    aux = aux[keep.to_numpy()].reset_index(drop=True)
    y = (d["finish"] <= 3).astype(int).to_numpy()
    fx, _ = F.encode_categoricals(f)
    fx = add_odds_features(fx, d["odds"].to_numpy(float), d["raceid"].to_numpy())
    cut = d["date"].quantile(0.3)
    tr, te = (d["date"] < cut).to_numpy(), (d["date"] >= cut).to_numpy()

    print("(176-枠) ★4コーナーの形は、モデルが既に拾っているか（ゲート1本）")
    print(f"■ 内部対照（母集団を最初に出す）")
    print(f"　全体 {len(d):,}行 / 学習 {int(tr.sum()):,}（〜{cut.date()}） /"
          f" ★検証 {int(te.sum()):,}（{cut.date()}〜{d['date'].max().date()}）")

    ms = fit_seeds(fx[tr], y[tr], 3, PAR)
    p = np.mean([m.predict_proba(fx)[:, 1] for m in ms], axis=0)

    a = auc(y[te], p[te])
    ok_gate = abs(a - AUC_REF) <= 0.01
    print(f"\n■ ⚠ゲート: 検証AUC {a:.4f}（(174-枠)は {AUC_REF:.4f}）"
          f" → {'★一致。主判定を読む' if ok_gate else '⚠外れた。主判定を読まない'}")
    if not ok_gate:
        return

    m = te & aux["prev_res"].notna().to_numpy() & aux["prev_ra"].notna().to_numpy()
    pp = np.clip(p[m], 1e-6, 1 - 1e-6)
    Z = np.c_[np.ones(int(m.sum())), np.log(pp / (1 - pp))]     # logit(p) だけを差し引く
    yy = y[m].astype(float)
    print(f"\n■ ★★主判定（検証 {int(m.sum()):,}行・logit(p) を差し引いた後の偏相関）")
    print(f"{'':>22}{'偏相関':>10}{'99%CI':>24}")
    rr = {}
    for nm, v in (("★本 prev_res（4角残差）", aux["prev_res"].to_numpy()[m]),
                  ("対照 prev_ra（モデル内）", aux["prev_ra"].to_numpy()[m])):
        r, lo, hi = pcorr(v, yy, Z)
        rr[nm[:2]] = abs(r)
        print(f"{nm:>22}{r:>+10.4f}{f'[{lo:+.4f}, {hi:+.4f}]':>24}")
    cond1 = rr["★本"] > rr["対照"]
    print(f"　→ ①本が対照を上回るか: **{'★はい' if cond1 else '⚠いいえ'}**"
          f"（{rr['★本']:.4f} vs {rr['対照']:.4f}）")

    print(f"\n■ 採用条件② 年別の符号")
    yr = d["date"].dt.year.to_numpy()
    sgn = []
    for Y in sorted(set(yr[m])):
        mm = m & (yr == Y)
        if mm.sum() < 5000:
            continue
        ppy = np.clip(p[mm], 1e-6, 1 - 1e-6)
        Zy = np.c_[np.ones(int(mm.sum())), np.log(ppy / (1 - ppy))]
        r, _, _ = pcorr(aux["prev_res"].to_numpy()[mm], y[mm].astype(float), Zy)
        sgn.append(r)
        print(f"　{Y}  {r:+.4f}")
    same = max(sum(1 for s in sgn if s > 0), sum(1 for s in sgn if s < 0))
    cond2 = same >= NYEAR_OK
    print(f"　→ ②符号がそろった年 {same}/{len(sgn)} → **{'★満たす' if cond2 else '⚠満たさない'}**")

    # ■ 参考: 偶数年で学習・奇数年で検証（★採用条件には使わない）
    print("\n■ 参考（★採用条件には使わない）— 偶数年で学習・奇数年で検証")
    print("　⚠**未来を見るので主判定にはできない**。時代の変化と本物を切り分けるためだけ。")
    ev = (yr % 2 == 0)
    ms2 = fit_seeds(fx[ev], y[ev], 3, PAR)
    p2 = np.mean([mm.predict_proba(fx)[:, 1] for mm in ms2], axis=0)
    m2 = (~ev) & aux["prev_res"].notna().to_numpy() & aux["prev_ra"].notna().to_numpy()
    a2 = auc(y[m2], p2[m2])
    pp2 = np.clip(p2[m2], 1e-6, 1 - 1e-6)
    Z2 = np.c_[np.ones(int(m2.sum())), np.log(pp2 / (1 - pp2))]
    print(f"　奇数年 {int(m2.sum()):,}行 / AUC {a2:.4f}")
    for nm, v in (("★本 prev_res", aux["prev_res"].to_numpy()[m2]),
                  ("対照 prev_ra", aux["prev_ra"].to_numpy()[m2])):
        r, lo, hi = pcorr(v, y[m2].astype(float), Z2)
        print(f"{nm:>22}{r:>+10.4f}{f'[{lo:+.4f}, {hi:+.4f}]':>24}")

    print("\n" + "=" * 78)
    if cond1 and cond2:
        print("★★①②とも満たした＝**モデルが拾いきれていない情報が残っている**。")
        print("　⚠**それでも「儲かる」ではない**。絞りを作るなら**別に事前登録する**。")
    else:
        print("⚠★①②のどちらかを満たさない＝**モデルが既に拾っている**と読む。")
        print("　★**この線はここで閉じる。絞りは作らない**（判定基準40）。")


if __name__ == "__main__":
    main()
