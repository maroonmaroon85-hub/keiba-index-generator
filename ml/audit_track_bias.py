"""(177-枠) ★トラックバイアスは、市場が織り込んだ後に何か残すか。**ゲート1本だけ**。

★**なぜ測るか**——**44特徴量に枠番も馬番も1つも入っていない**（確認済み）。
　`course` はあっても**「その日その場が内有利か前残りか」は無い**。
　★**(176)の4コーナーと違い、モデルに無いことが確実**。

⚠⚠★**致命的な罠（先に書く）**——**バイアスは「その日の結果」から測るが、
　**賭ける時点ではその日の結果を知らない**。**11年分で「内が有利だった日は内が来た」と
　示すのは何も言っていない**（(176)で私が踏んだ「同じレースの4角位置と同じレースの着順」と同型）。
★**使える形は1つだけ**: **その日・その場で、そのレースより前に終わったレースだけ**から
　推定し、**次のレースに当てる**。**1Rは使えない。前半は推定が粗い**。
　（**穴馬側(219)の「全レースに同じ時計を当てると朝のレースはレース後を読む」と同じ構造**）

⚠★**事前の見立ては厳しい側**——**市場も同じものを見ている**。
　**当日の1R〜5Rを見て「今日は内が伸びる」と判断し、それが後半のオッズに乗る**。
　★**(88): 市場の誤りは最大でも +10pt、埋めるべき控除率は 20.5pt**。
　→ ★**`p` を差し引いた後に残るのは「市場が見落とした分」だけ**。**それが欲しいもの**。

━━━ ★★事前登録（**結果を見る前に書く**。このゲート1本。後から増やさない）━━━

★**バイアスの作り方**（**当日・同じ場・同じ芝ダで、そのレースより前のレースだけ**）:
　各既走レースについて **上位3頭の「馬番/頭数」の平均 − 0.5** ＝ そのレースの内外の偏り。
　同じく **上位3頭の「4角位置/頭数」の平均 − 0.5** ＝ 前残り/差しの偏り。
　★**それを既走レースで平均**して `bias_draw` / `bias_pace`。
　⚠**既走が3レース未満のレースは使わない**（1レースの雑音でバイアスを名乗らない）。

★★**2つの作り方を並べる（2026-09-16・利用者の案で追加。★走らせる前に足した）**:
　**A 当日累積**: その日・その場で**そのレースより前**の既走レースだけ。
　　⚠**既走3R未満は使えない**ので**実質後半のレースだけ**になる。
　**B ★前日（土→日）**: ★**同じ場の1日前の開催日の全レース**から作る。
　　★**1Rから使える**（発走前に完全に確定している）。★**時刻の栓が要らない**。
　　⚠**弱点は「馬場は整備される」こと**——**日曜朝に均すので土曜の傾向がどれだけ残るかは未知**。
　　★**日付差がちょうど1日のときだけ使う**（週をまたぐ持ち越しは混ぜない）。

★**当てる量（A・Bそれぞれで2つ。計4つ）**:
　`align_draw = (自分の馬番/頭数 − 0.5) × bias_draw`   ★**馬番は発走前に分かる**
　`align_pace = (last_passratio − 0.5) × bias_pace`    ★**前走の位置を今日の脚質の代理にする**
　**符号が正＝その日の偏りに沿っている**。

★**主判定**: 検証期間で **偏相関 corr(align, top3 | logit p)**。**99%CI・Bonferroni α=0.01/2**。
　★**仮説が偽なら 0 を返す**（判定基準42）。**`p` は log_odds / mkt_prob を含む**ので
　**市場が織り込んだ分は同時に落ちる**。

★★**対照（判定基準45: 乱を使わない決定的な量）**: ★**同じ日の「別の場」のバイアス**を当てる。
　**同じ日・同じ時期なのに、この馬場の状態とは因果が無い**。★**プラセボ**。
　⚠**乱数を使わないので種のぶれが無い**（判定基準43を回避できる形）。

★**採用条件（判定基準39/40。有意なだけでは足りない）**:
　**① 本の偏相関の絶対値が、プラセボを明確に上回る**（4つのどれかで）
　**② 年別で符号が 8/10年以上そろう**
　**③ Bonferroni後の99%CIがゼロを外す**
　★**3つ揃わなければ「市場が織り込み済み」と読み、この線を閉じる**。

⚠**ゲート（判定基準32）**: **検証AUCが (174-枠) の 0.8027 と ±0.01 で一致**。外れたら読まない。
⚠**内部対照は最初に出す**: **検証行数と、バイアスが引けた行数**。

━━━ ★★結果（2026-09-16・**事前登録どおり・条件は変えていない**）━━━

■ ゲート: **検証AUC 0.8027 ＝ (174-枠) と完全一致**。装置は同じ。
■ ★★主判定（logit(p) を差し引いた後の偏相関・Bonferroni α=0.01/4）

| | n | 偏相関 | 99%CI(Bonf) |
|---|---|---|---|
| ★**本 A 当日累積 内外** | 187,230 | ★**+0.0163** | ★**[+0.0093, +0.0233]** |
| 　プラセボ A 内外 | 187,230 | +0.0052 | [−0.0018, +0.0122] ⚠ゼロを跨ぐ |
| 本 A 当日累積 前残り | 187,230 | +0.0027 | [−0.0042, +0.0097] ⚠ゼロを跨ぐ |
| 　プラセボ A 前残り | 187,230 | +0.0043 | [−0.0027, +0.0113] |
| ★**本 B 前日(土→日) 内外** | 93,115 | ★**+0.0220** | ★**[+0.0121, +0.0319]** |
| ⚠**プラセボ B 内外** | 93,115 | ⚠**+0.0134** | ⚠**[+0.0035, +0.0233]（ゼロを外す）** |
| 本 B 前日 前残り | 93,115 | −0.0010 | [−0.0109, +0.0089] ⚠ゼロを跨ぐ |
| 　プラセボ B 前残り | 93,115 | −0.0013 | [−0.0112, +0.0086] |

■ 年別の符号: ★**A_draw 10/10 ／ B_draw 10/10**（**全年で正**）。**A_pace 6/10 / B_pace 6/10**。

★★★**①②③すべて満たした**。⚠**ただし内外だけ。前残りは死んでいる**。

★**読み方**
　1. ★**「内外」は市場が織り込んだ後にも残る**。**10/10年で符号が正**＝**偶然ではない**。
　2. ⚠**「前残り」は残らない**。**プラセボに負け、CIもゼロを跨ぎ、符号も6/10**。★**閉じる**。
　3. ⚠★**Bのプラセボがゼロを外した**（+0.0134）——**別の場の前日バイアスでも当たる**。
　　 ★**時期に共通の成分がある**（季節・天候・馬場管理）。**それを差し引いた超過は
　　 　A が +0.0111、B が +0.0086 で、★当日累積のほうが「その馬場に固有」**。
　4. ⚠⚠**大きさは小さい**。**log_odds 単独が top3 と −0.4423** なのに対し **+0.016〜0.022**
　　 ＝**市場の1/20程度**。★**「3着以内に入りやすい」であって「儲かる」ではない**。
　5. ★**なぜ残りうるか（機構）**: **44特徴量に枠番も馬番も無い**ので、
　　 **`p` は馬番を一切知らない**。**知っているのは `log_odds` 経由で市場が織り込んだ分だけ**。
　　 → ★**残ったのは「市場が馬番に付け損ねた値段」**。

⚠**ここから先は別の事前登録が要る**——**買い方（絞り/特徴量追加）を作るのはこの測定の外**。
　⚠**特徴量に足すと `model_prod` が変わる＝運用が変わる**。**(141)の117.3%は板経路なので動かない**。

実行: python3 ml/audit_track_bias.py
"""
import math
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "ml")
import features as F
from train_prod import CAPACITY, add_odds_features, fit_seeds

AUC_REF = 0.8027
MIN_PRIOR_RACES = 3       # 既走がこれ未満のレースは使わない
NCMP = 4                  # A/B × draw/pace
NYEAR_OK = 8


def auc(y, s):
    o = np.argsort(s, kind="mergesort")
    r = np.empty(len(s), float); r[o] = np.arange(1, len(s) + 1)
    n1 = int(y.sum()); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0) if n1 and n0 else float("nan")


def pcorr(x, y, Z, ncmp=1):
    def res(v):
        b = np.linalg.lstsq(Z, v, rcond=None)[0]
        return v - Z @ b
    rx, ry = res(x), res(y)
    r = float(np.corrcoef(rx, ry)[0, 1])
    n = len(x); se = 1.0 / math.sqrt(n - Z.shape[1] - 1)
    from audit_crosspool import zq
    z = 0.5 * math.log((1 + r) / (1 - r)); zc = zq(0.01 / ncmp)
    return r, math.tanh(z - zc * se), math.tanh(z + zc * se)


def build_bias(raw, d, f):
    """★トラックバイアスの当てる量を作る。**定義はここ1か所**（(178)もこれを読む）。

    返す DataFrame の列: A_draw/A_pace/B_draw/B_pace と、それぞれのプラセボ。
    ★**A=当日累積（そのレースより前だけ）／B=前日（土→日・日付差1日）**。
    ⚠**プラセボは「同じ日の別の場のバイアス」**（乱数を使わない決定的な対照）。
    """
    # ★4角は to_model が平均に潰すので生から取る。★col40で突き合わせる（(176)で位置合わせを踏んだ）
    c4_raw = pd.to_numeric(raw[31], errors="coerce").where(lambda x: x > 0)
    c4map = pd.Series(c4_raw.to_numpy(), index=raw[40].str.strip().to_numpy())
    c4map = c4map[~c4map.index.duplicated(keep="first")]
    y = (d["finish"] <= 3).astype(int).to_numpy()

    key = d["raceid"].astype(str) + d["umaban"].astype(int).astype(str).str.zfill(2)
    fs = d["fieldsize"].to_numpy(float)
    b = pd.DataFrame({
        "raceid": d["raceid"].to_numpy(), "date": d["date"].to_numpy(),
        "course": d["course"].to_numpy(), "surface": d["surface"].to_numpy(),
        "R": d["raceid"].astype(str).str[6:8].astype(int).to_numpy(),
        "rd": d["umaban"].to_numpy(float) / fs,
        "rp": c4map.reindex(key.to_numpy()).to_numpy(float) / fs,
        "top3": y.astype(float),
        "lastp": f["last_passratio"].to_numpy(float)})

    # ★レースごとの偏り（上位3頭の平均 − 0.5）
    t = b[b["top3"] > 0]
    per = t.groupby("raceid").agg(rd=("rd", "mean"), rp=("rp", "mean")).reset_index()
    per[["rd", "rp"]] -= 0.5
    meta = b.groupby("raceid").agg(date=("date", "first"), course=("course", "first"),
                                   surface=("surface", "first"), R=("R", "first")).reset_index()
    per = per.merge(meta, on="raceid").sort_values(["course", "surface", "date", "R"])

    # ★そのレースより前の既走レースだけで平均（expanding→shift）
    g = per.groupby(["course", "surface", "date"])
    for c in ("rd", "rp"):
        per[f"bias_{c}"] = g[c].transform(lambda s: s.expanding().mean().shift(1))
    per["nprior"] = g.cumcount()
    per0 = per.copy()                      # ★Bはその日の全レースを使うので絞る前を残す
    per = per[per["nprior"] >= MIN_PRIOR_RACES]

    # ★プラセボ: 同じ日の「別の場」のバイアス（同じ芝ダ・その日の全レース平均）
    day = per.groupby(["date", "surface", "course"]).agg(
        pd_=("rd", "mean"), pp_=("rp", "mean")).reset_index()
    tot = day.groupby(["date", "surface"]).agg(s_d=("pd_", "sum"), s_p=("pp_", "sum"),
                                               n=("course", "count")).reset_index()
    day = day.merge(tot, on=["date", "surface"])
    day = day[day["n"] >= 2]
    day["plc_d"] = (day["s_d"] - day["pd_"]) / (day["n"] - 1)   # 自分以外の場の平均
    day["plc_p"] = (day["s_p"] - day["pp_"]) / (day["n"] - 1)
    per = per.merge(day[["date", "surface", "course", "plc_d", "plc_p"]],
                    on=["date", "surface", "course"], how="left")

    # ★B: 前日（土→日）のバイアス。★同じ場・同じ芝ダで日付差がちょうど1日のときだけ
    dayfull = per0.groupby(["course", "surface", "date"]).agg(
        pv_d=("rd", "mean"), pv_p=("rp", "mean")).reset_index()
    dayfull = dayfull.sort_values(["course", "surface", "date"])
    gg = dayfull.groupby(["course", "surface"])
    dayfull["prev_date"] = gg["date"].shift(1)
    dayfull["prev_d"] = gg["pv_d"].shift(1)
    dayfull["prev_p"] = gg["pv_p"].shift(1)
    gap = (dayfull["date"] - dayfull["prev_date"]).dt.days
    dayfull.loc[gap != 1, ["prev_d", "prev_p"]] = np.nan
    # ★Bのプラセボ: 同じ「前日」の別の場のバイアス
    tt = dayfull.groupby(["date", "surface"]).agg(
        s_d=("prev_d", "sum"), s_p=("prev_p", "sum"), n=("prev_d", "count")).reset_index()
    dayfull = dayfull.merge(tt, on=["date", "surface"])
    ok2 = dayfull["n"] >= 2
    dayfull["prevplc_d"] = np.where(ok2, (dayfull["s_d"] - dayfull["prev_d"].fillna(0))
                                    / (dayfull["n"] - 1), np.nan)
    dayfull["prevplc_p"] = np.where(ok2, (dayfull["s_p"] - dayfull["prev_p"].fillna(0))
                                    / (dayfull["n"] - 1), np.nan)
    per = per.merge(dayfull[["course", "surface", "date",
                             "prev_d", "prev_p", "prevplc_d", "prevplc_p"]],
                    on=["course", "surface", "date"], how="left")

    b = b.merge(per[["raceid", "bias_rd", "bias_rp", "plc_d", "plc_p",
                     "prev_d", "prev_p", "prevplc_d", "prevplc_p"]], on="raceid", how="left")
    b["A_draw"] = (b["rd"] - 0.5) * b["bias_rd"]
    b["A_pace"] = (b["lastp"] - 0.5) * b["bias_rp"]
    b["Aplc_draw"] = (b["rd"] - 0.5) * b["plc_d"]
    b["Aplc_pace"] = (b["lastp"] - 0.5) * b["plc_p"]
    b["B_draw"] = (b["rd"] - 0.5) * b["prev_d"]
    b["B_pace"] = (b["lastp"] - 0.5) * b["prev_p"]
    b["Bplc_draw"] = (b["rd"] - 0.5) * b["prevplc_d"]
    b["Bplc_pace"] = (b["lastp"] - 0.5) * b["prevplc_p"]
    return b


def main():
    MODEL_DIR, PAR = CAPACITY["l2"]
    raw = F.load_files()
    raw = raw[raw[40].str.len() > 2]
    d = F.to_model(raw)
    f = F.build_features(d)
    keep = (f["n_prior"] >= 1) & d["odds"].notna() & (d["odds"] > 0)
    d, f = d[keep].reset_index(drop=True), f[keep].reset_index(drop=True)
    y = (d["finish"] <= 3).astype(int).to_numpy()
    b = build_bias(raw, d, f)

    fx, _ = F.encode_categoricals(f)
    fx = add_odds_features(fx, d["odds"].to_numpy(float), d["raceid"].to_numpy())
    cut = d["date"].quantile(0.3)
    tr, te = (d["date"] < cut).to_numpy(), (d["date"] >= cut).to_numpy()

    print("(177-枠) ★トラックバイアスは市場が織り込んだ後に何か残すか（ゲート1本）")
    print("■ 内部対照（母集団を最初に出す）")
    print(f"　全体 {len(d):,}行 / 学習 {int(tr.sum()):,}（〜{cut.date()}） / 検証 {int(te.sum()):,}")
    okA = b[["A_draw", "A_pace", "Aplc_draw", "Aplc_pace"]].notna().all(axis=1).to_numpy()
    okB = b[["B_draw", "B_pace", "Bplc_draw", "Bplc_pace"]].notna().all(axis=1).to_numpy()
    print(f"　★A 当日累積が引けた行 {int(okA.sum()):,}（既走{MIN_PRIOR_RACES}R以上＋同日に別の場）")
    print(f"　★B 前日(土→日)が引けた行 {int(okB.sum()):,}（日付差1日＋前日に別の場）")

    ms = fit_seeds(fx[tr], y[tr], 3, PAR)
    p = np.mean([m.predict_proba(fx)[:, 1] for m in ms], axis=0)
    a = auc(y[te], p[te])
    gate = abs(a - AUC_REF) <= 0.01
    print(f"\n■ ⚠ゲート: 検証AUC {a:.4f}（(174-枠)は {AUC_REF:.4f}）"
          f" → {'★一致。主判定を読む' if gate else '⚠外れた。読まない'}")
    if not gate:
        return

    print(f"\n■ ★★主判定（logit(p) を差し引いた後の偏相関・Bonferroni α=0.01/{NCMP}）")
    print(f"{'':>30}{'n':>10}{'偏相関':>10}{'99%CI(Bonf)':>26}")
    vals = {}
    for tag, okm, pairs in (
            ("A 当日累積", okA, (("A_draw", "Aplc_draw", "内外"), ("A_pace", "Aplc_pace", "前残り"))),
            ("B 前日(土→日)", okB, (("B_draw", "Bplc_draw", "内外"), ("B_pace", "Bplc_pace", "前残り")))):
        m = te & okm
        pp = np.clip(p[m], 1e-6, 1 - 1e-6)
        Z = np.c_[np.ones(int(m.sum())), np.log(pp / (1 - pp))]
        yy = y[m].astype(float)
        for main, plc, lab in pairs:
            for nm, col in ((f"★本 {tag} {lab}", main), (f"　プラセボ {tag} {lab}", plc)):
                r, lo, hi = pcorr(b[col].to_numpy()[m], yy, Z, NCMP)
                vals[col] = (abs(r), lo, hi)
                print(f"{nm:>30}{int(m.sum()):>10,}{r:>+10.4f}{f'[{lo:+.4f}, {hi:+.4f}]':>26}")
    MAINS = ("A_draw", "A_pace", "B_draw", "B_pace")
    PLCS = ("Aplc_draw", "Aplc_pace", "Bplc_draw", "Bplc_pace")
    c1 = any(vals[a][0] > vals[c][0] for a, c in zip(MAINS, PLCS))
    c3 = any(vals[k][1] * vals[k][2] > 0 for k in MAINS)
    print(f"　→ ①本がプラセボを上回るか: **{'★はい' if c1 else '⚠いいえ'}**")
    print(f"　→ ③CIがゼロを外すか: **{'★はい' if c3 else '⚠いいえ'}**")

    print("\n■ 採用条件② 年別の符号")
    yr = d["date"].dt.year.to_numpy()
    cnt = {}
    for col, okm in (("A_draw", okA), ("A_pace", okA), ("B_draw", okB), ("B_pace", okB)):
        sg = []
        for Y in sorted(set(yr[te & okm])):
            mm = te & okm & (yr == Y)
            if mm.sum() < 3000:
                continue
            ppy = np.clip(p[mm], 1e-6, 1 - 1e-6)
            Zy = np.c_[np.ones(int(mm.sum())), np.log(ppy / (1 - ppy))]
            r, _, _ = pcorr(b[col].to_numpy()[mm], y[mm].astype(float), Zy)
            sg.append(r)
        cnt[col] = (max(sum(1 for s in sg if s > 0), sum(1 for s in sg if s < 0)), len(sg))
        print(f"　{col}: そろった年 {cnt[col][0]}/{cnt[col][1]}"
              f"　（{' '.join(f'{s:+.3f}' for s in sg)}）")
    c2 = any(v[0] >= NYEAR_OK for v in cnt.values())
    print(f"　→ ②{'★満たす' if c2 else '⚠満たさない'}")

    print("\n" + "=" * 84)
    if c1 and c2 and c3:
        print("★★①②③とも満たした＝**市場が織り込んだ後にも何か残っている**。")
        print("　⚠**それでも「儲かる」ではない**。買い方を作るなら**別に事前登録する**。")
    else:
        print("⚠★①②③のどれかを満たさない＝**市場が織り込み済み**と読む。")
        print("　★**この線はここで閉じる**（判定基準40）。")


if __name__ == "__main__":
    main()
