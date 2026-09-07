"""(219) ★★★★**朝のオッズだと、同じ馬が選ばれるのか** —— 物差しを「一致率」に替える

★★**動機（2026-09-07・利用者の指定「朝時点のオッズだとどう？」）**:
　★**(218)までの169マスは、全部★確定オッズで測った**（**CSVの単勝＝確定と実測確認済み・
　　勝ち馬3,000件で ±0.05倍以内が100%・平均絶対差0.000倍・相関1.0000**）。
　★**この軸はオッズに三重に依存する**: **①軸の条件（単勝≥10.0倍）②紐Q（人気順）
　　③モデルの特徴量 log_odds**。→ ⚠**朝のオッズだと別の馬が選ばれうる**。

■ ⚠★★**線引き（守る）**
　★**data/odds_ts は別セッションが(148)で集めたもの。★読むだけ**。
　　**書かない・集め方に触れない・枠連の運用にも触れない・設定変更の提案もしない**。

■ ★★★**なぜ物差しを替えるか（判定基準26）**
　⚠**複勝の板は1レース1枚（締切前後）しか無い**ので、★**朝版のROIは作れない**。
　⚠**そのうえ対象は396レース**で、**この軸は1日0.6本＝該当は数本**しか出ない。
　　→ ★**ROIでは何も測れない**（判定基準5）。
　★★**だから「朝と確定で、選ばれる馬がどれだけ入れ替わるか」を一致率で測る**。
　　★**396レース全部が使えて、分散が小さい**。

────────────────────────────────────────────────────────────
★★★ 事前登録（2026-09-07・**結果を見る前にコミットする**）
────────────────────────────────────────────────────────────

■ ★対象: **data/odds_ts の396レース**（**2026-06-20〜07-26 の11開催日・各36R**）。
■ ★**朝の定義: その開催日の 09:00 直前の最後のスナップショット**（**第1レースの前**）。
　★**参考に 前日21:00 と 締切直前（最後のスナップショット）も出す**。
■ ★**確定の定義: ルートCSVの単勝オッズ**（**＝確定と実測確認済み**）。

■ ★★★**測るもの（すべて一致率・ROIではない）**
| # | 量 | ★**これが答える問い** |
|---|---|---|
| **1** | **単勝オッズの朝↔確定の相関・平均絶対変化率** | ★**そもそもどれだけ動くのか** |
| **2** | ★**単勝10.0倍の線をまたぐ馬の割合** | ★**軸の候補集合が変わるか** |
| **3** | ★**人気1〜2番（紐Qの中身）の一致率** | ★**紐Qが変わるか** |
| **4** | ★**「10倍以上」の集合の一致率（Jaccard）** | ★**軸の候補が入れ替わる度合い** |

■ ★★ゲート2（判定基準42）——**この測定は何を返せば「朝でも同じ」か**
　★**オッズが一切動かないなら、相関1.000・またぎ0%・一致率100%を返す**。
　★**逆に、完全に無関係なら相関0・一致率は偶然の水準**。**その偶然の水準も併記する**。
■ ★★内部対照（**決定的**）: **締切直前のスナップショット vs 確定オッズ**は
　★**ほぼ一致するはず**（**相関>0.99・またぎ<2%**）。⚠**ここが合わなければ、
　　時系列とルートCSVの突き合わせが壊れている**。★**合わなければ読まない**。

■ ⚠**この測定で答えられないこと（先に書く）**
　★**朝版のROI・必要年数は出せない**（**板が1枚・該当が数本**）。
　★**モデルの予測値（log_odds経由）の変化も直接は測らない**——
　　**①②③のうち①②だけを見る**。⚠**③は同じ向きに動くはずだが、確認していない**。

■ 予想（⚠**当てにしない**・★**私は7回外した**）
　★**人気1〜2番の一致率は9割超と見る**（**上位人気は朝からほぼ決まっている**）。
　⚠**10倍の線をまたぐ馬は1〜2割あると見る**——**10倍前後は最も票が動く帯**。
　★**もしまたぎが多ければ、★朝イチ運用は「別の測定」になる**。

■ ★★★★**結果（実測済み・2026-09-07）—— ★朝9時では別の馬が選ばれる**
　★**内部対照が立った**: **締切直前 vs 確定 = 相関 0.9979・またぎ 0.9%**
　→ ★**時系列とルートCSVの突き合わせは正しい**。

| 時点 | R数 | 対数相関 | ★**平均変化** | ★**10倍またぎ** | 10倍集合の一致 | ★**人気1番一致** | 人気1-2番一致 |
|---|---|---|---|---|---|---|---|
| **前日 21:00** | 360 | 0.7885 | ⚠**65.1%** | ⚠**16.9%** | 77.8% | ⚠**56.4%** | 64.3% |
| ★**朝 09:00** | 360 | 0.9447 | ⚠**39.3%** | ⚠**8.4%** | 87.9% | ⚠**73.3%** | 80.3% |
| **11:00** | 360 | 0.9460 | 30.7% | 6.3% | 90.8% | 77.8% | 84.6% |
| **13:00** | 360 | 0.9674 | 19.1% | 4.4% | 93.5% | 85.8% | 89.2% |
| **15:00** | 360 | 0.9888 | **9.0%** | **1.8%** | 97.4% | **95.6%** | 95.3% |
| ★**締切直前**（対照） | 360 | ★**0.9979** | **4.0%** | **0.9%** | 98.7% | **98.3%** | 97.8% |

■ ⚠★**私の予想は8回目の外れ。★外し方が示唆的だった**
| 予想 | 実際 |
|---|---|
| **人気1〜2番の一致率は9割超** | ⚠**80.3%**（**人気1番だけなら 73.3%**） |
| **10倍またぎは1〜2割** | ★**8.4%**（**予想より少ない**） |
　★★**心配していた「10倍の線」より、人気順の方がはるかに不安定だった**。
　　**朝9時に人気1番だった馬が確定でも1番人気なのは4回に3回**。

■ ★★★**意味すること**
| 要素 | 朝9時の影響 |
|---|---|
| ★**軸の条件（10倍以上）** | **8.4%の馬が入れ替わる**（候補集合の一致87.9%）。★**比較的頑健** |
| ⚠★**紐Q（人気順）** | ★**人気1番が26.7%入れ替わる** → **Q三連単A/G馬単Mの買い目が大きく変わる** |
| ⚠**モデルの log_odds** | **オッズが平均39.3%動く**ので**予測値も動く**（★**未測定**） |
　★★**Q三連単A4点は、朝イチだと「別の買い目」になる**。
　　⚠**11年の141.6%・40年という数字は朝版には当てはまらない**。
　★**時刻ごとの劣化の勾配が出た**: **13時で人気1番一致85.8%、15時で95.6%**。
　　★**締切に近いほど測定と同じ条件になる**——**この線は1日0.6本なので、
　　★候補レースの締切前だけ確認する運用なら成立する**。

■ ⚠**修正した誤り2件（★どちらも結果を読む前）**
　1. ★**「前日21:00」を当日21時と解釈していた**——**レースは16時に終わるので
　　　締切直前と同じ行を拾い、★数字が完全一致して発覚**。→ **前日＝負の時間に直した**。
　2. **自前パーサをやめ、既存の `ml/odds_ts.py` の `load_dir` を使った**
　　　（**raceid＝ファイル名の3〜10文字目でルートCSVとそのまま突き合う**）。

■ ★★★★**(220) 追記（2026-09-07・★利用者の決定「朝九時かな」）**
　★**利用者は買う時点を「朝9時」に決めた**。★**記録する。以後この時点を前提にする**。
　⚠★**結果として、169マスの数字は朝9時版には当てはまらない**
　　（**139.2%・25年・141.6%・40年——全部、確定オッズでの値**）。★**朝版の必要年数は未知**。

■ ★★★**それでも測れること: 紐によって朝オッズの影響が違うはず**
| 紐 | 何で決まるか | ★**朝オッズの影響** |
|---|---|---|
| **Q**（人気順） | ★**単勝オッズそのもの** | ⚠**直撃**（(219)で人気1番が26.7%入れ替わる） |
| **P**（モデル順） | **モデルの予測値 pv** | **`log_odds` 経由で間接的** |
| ★**G**（ズレ順） | **pn − qp（モデル−複勝板）** | ★**単勝オッズを直接は使わない** |
　★★**仮説: G馬単M4点は朝9時でもあまり変わらない**。**Q三連単A4点は大きく変わる**。

■ ★★**(220) の測り方（★事前登録・実測前）**
　★**対象**: **odds_ts の396レースのうち、確定版で軸が立つレース**。
　★**朝9時の単勝オッズで、①軸の条件（≥10倍）②紐Qの並び を作り直す**。
　★**pn と qp は確定版のまま固定する**——⚠**朝の複勝板が無いため**。
　★★**測るのは一致率**: **軸が同じレースの割合 / 紐P・G・Qの上位2頭の一致率 /
　　買い目そのもの（G馬単M4点・Q三連単A4点）の一致率**。
　■ ★★ゲート2（判定基準42）: ★**朝オッズが効かないなら、全部100%を返す**。
　■ ⚠★**この測定の限界（先に書く）**
　　★**`log_odds` 経由でモデルの予測値が動く分は含まれない**——**pnを固定しているため**。
　　→ ★**ここで出る一致率は「上限」である**。**実際はこれより低い**。
　　⚠**だから「朝9時でも大丈夫」の証明には使えない。「朝9時だとここまでは崩れる」の下限**。

■ ★★★★**(220) の結果（実測済み・2026-09-07）—— ★軸は変わらない。★消えるだけ**
| | 結果 |
|---|---|
| **確定版で軸が立つ** | **9本** |
| ⚠★**朝9時でも軸が立つ** | ⚠**6本（66.7%）**——★**3本は朝9時だと候補が消える** |
| ★★**そのうち軸が同一** | ★**6本すべて（100.0%）** |
　★**消えた3本は、確定では10倍以上だった馬が朝9時には10倍未満だった**
　　＝★**取りこぼしであって誤爆ではない**。

　★★**仮説どおり、紐で差が出た**:
| 紐 | 上位2頭の一致率 | 買い目そのものの一致率 |
|---|---|---|
| ★**P**（モデル順） | ★**100.0%** | — |
| ★**G**（ズレ順） | ★**100.0%** | ★**G馬単M4点 = 100.0%** |
| ⚠**Q**（人気順） | ⚠**75.0%** | ⚠**Q三連単A4点 = 50.0%** |
　★**理由は事前に立てた仮説どおり**——**Gは pn−qp（モデル−複勝板）で決まり単勝オッズを
　　直接使わない。Qは単勝オッズそのもの**。

■ ⚠★**この数字が弱い3つの理由（★先に書いたとおり）**
　1. ★**pnを固定している**——**朝オッズだと log_odds が動き pn も動く**。**100%は上限**。
　2. ⚠**6本しかない**。**100.0%といっても「6回とも一致した」だけ**。
　3. ★**朝9時版のROIは依然として未知**。**一致率が高くても、取りこぼした3本の分だけ本数が減る**。

■ ★**朝9時を選んだ場合に測定が示す形**（⚠**推奨ではない**）
| | ★**朝9時での安定性** |
|---|---|
| ★**G馬単M4点**（紐=ズレ順） | ★**高い**（買い目一致100%・6/6） |
| ⚠**Q三連単A4点**（紐=人気順） | ⚠**低い**（買い目一致50%） |
| **P複勝1点**（紐なし） | ★**軸だけなので高いはず**（軸一致100%） |
| **共通** | ⚠**買える本数が約2/3に減る**（**軸が立つのが 6/9**） |

実行: python3 ml/audit_ana_morn.py    自己テスト: python3 ml/audit_ana_morn.py --selftest
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "ml")
import features as F
from odds_ts import load_dir

LFIX = 10.0
GAP_F = 0.15
# ★時点（レース当日の基準時刻からの時間。★負なら前日）
#   ⚠**訂正(2026-09-07)**: **初版は EVE=21 を「当日21時」と解釈していた**。
#   　**レースは16時に終わるので、それは締切直前と同じ行を拾っていた**（**数字が完全一致して発覚**）。
POINTS = [("前日 21:00", -3), ("★★朝 09:00", 9), ("11:00", 11),
          ("13:00", 13), ("15:00", 15)]
MORN = 9
CORR_MIN, CROSS_MAX = 0.99, 2.0


def snap_at(rec, hour):
    """★その時点までで最後のスナップショット（区分1）の行番号。無ければ None。

    ★**hour は当日0時からの時間。負なら前日**（**−3 は前日21:00**）。
    ⚠**前日夜の行は時刻文字列だけでは朝より大きく見える**ので、★**日付ごと比較する**。
    """
    lim = pd.Timestamp(rec["date"].date()) + pd.Timedelta(hours=hour)
    ok = np.where((rec["times"] <= lim) & (rec["kubun"] == "1"))[0]
    return int(ok[-1]) if len(ok) else None


def last_win(rec):
    """★最後の単勝スナップショット（＝締切直前）の行番号"""
    ok = np.where(rec["kubun"] == "1")[0]
    return int(ok[-1]) if len(ok) else None


def jacc(a, b):
    u = a | b
    return 1.0 if not u else len(a & b) / len(u)


def selftest():
    ok = True
    rec = {"date": pd.Timestamp("2026-07-26"),
           "times": pd.DatetimeIndex(["2026-07-25 21:35", "2026-07-26 09:00",
                                      "2026-07-26 14:00"]),
           "kubun": np.array(["1", "1", "1"])}
    print(f"★朝の定義: **レース当日 {MORN}時 までの最後のスナップショット**")
    print(f"　⚠**前日夜の行は時刻の文字列だけでは朝より大きく見える**"
          f"　→ ★**日付ごと比較する**")
    for h, want, lab in ((MORN, 1, "09時"), (8, 0, "08時（★前日夜を拾う）"),
                         (-2, 0, "★前日22時（=−2時間）"),
                         (-3, None, "★前日21時（21:35はまだ無い）"), (15, 2, "15時")):
        got = snap_at(rec, h)
        w = f"（{rec['times'][got]}）" if got is not None else "（該当なし）"
        print(f"　{lab:<24} → 行 {got}{w}"
              f"　{'★OK' if got == want else '⚠NG'}")
        ok &= got == want
    rec0 = dict(rec, date=pd.Timestamp("2026-07-25"))
    print(f"　★前日の朝なら無い → {snap_at(rec0, 5)}"
          f"　{'★OK' if snap_at(rec0, 5) is None else '⚠NG'}")
    ok &= snap_at(rec0, 5) is None
    print(f"　★締切直前は最後の行 → {last_win(rec)}"
          f"　{'★OK' if last_win(rec) == 2 else '⚠NG'}")
    ok &= last_win(rec) == 2
    print(f"★Jaccardの検算: 同じ集合 {jacc({1,2},{1,2}):.2f} / "
          f"半分 {jacc({1,2},{2,3}):.2f} / 空 {jacc(set(),set()):.2f}")
    ok &= jacc({1, 2}, {1, 2}) == 1.0 and abs(jacc({1, 2}, {2, 3}) - 1/3) < 1e-9
    print(f"★★ゲート2: **オッズが動かなければ 相関1.000・またぎ0%・一致率100%**")
    print(f"★★内部対照: **締切直前 vs 確定 が 相関>{CORR_MIN} かつ またぎ<{CROSS_MAX}%**"
          f"　⚠**合わなければ読まない**")
    print("★自己テスト: " + ("全部OK" if ok else "⚠NG"))
    return 0 if ok else 1


def axis_check(ts):
    """★(220) 朝9時の単勝オッズで、★軸と紐がどれだけ変わるか（pn・qpは確定版で固定）"""
    import audit_ana_bet as B
    from audit_ana_board import NPLACE, load_fuku_boards, qpool
    from audit_ana_band import PN_FLOOR
    from audit_ana_fix import LFIX as L
    from audit_ana_marg import wf_predict
    from audit_crosspool import load_races
    from train_prod import add_odds_features

    print(f"\n{'='*104}")
    print("■ ★★★★**(220) 朝9時だと、★軸馬は変わるのか**"
          "（**pnとqpは確定版で固定＝この一致率は上限**）")
    races = {r["rid"]: r for r in load_races()}
    boards = load_fuku_boards()
    d = F.to_model(F.load_files())
    f = F.build_features(d)
    keep = (f["n_prior"] >= 1) & d["odds"].notna() & (d["odds"] > 0)
    d, f = d[keep].reset_index(drop=True), f[keep].reset_index(drop=True)
    y = (d["finish"] <= 3).astype(int).to_numpy()
    fx, _ = F.encode_categoricals(f)
    fx = add_odds_features(fx, d["odds"].to_numpy(float), d["raceid"].to_numpy())
    pred = wf_predict(d, fx, y, 3)
    m = ~np.isnan(pred)
    sub = d.loc[m, ["raceid", "umaban", "odds"]].copy()
    sub["p"] = pred[m]

    nax = [0, 0, 0]          # 確定で軸あり / 朝も軸あり / 軸が同一
    hit = {h: [] for h in ("P", "G", "Q")}
    bet = {"G馬単M4点": [], "Q三連単A4点": []}
    for rid, g in sub.groupby("raceid"):
        rid = str(rid)
        rec, bd = ts.get(rid), boards.get(rid)
        if rec is None or bd is None or rid not in races:
            continue
        i9 = snap_at(rec, MORN)
        if i9 is None:
            continue
        ub = g["umaban"].astype(int).to_numpy()
        if not all(int(u) in bd for u in ub):
            continue
        od = g["odds"].to_numpy(float)
        pv = g["p"].to_numpy(float)
        row = rec["odds"][i9]
        om = np.array([row[u - 1] if u - 1 < len(row) else np.nan for u in ub], float)
        if pv.sum() <= 0 or not np.isfinite(om).all() or (om <= 0).any():
            continue
        pn = pv / pv.sum() * NPLACE
        qp, _ = qpool([bd[int(u)] for u in ub], "harm")
        gap = pn - qp
        base = (pn >= PN_FLOOR) & (gap >= GAP_F)
        cf = np.where(base & (od >= L))[0]
        if not len(cf):
            continue
        nax[0] += 1
        cm = np.where(base & (om >= L))[0]
        if not len(cm):
            continue
        nax[1] += 1
        af = int(ub[int(cf[int(np.argmax(pn[cf]))])])
        am = int(ub[int(cm[int(np.argmax(pn[cm]))])])
        same = af == am
        nax[2] += same
        ordf = {"P": np.argsort(-pv), "G": np.argsort(-gap), "Q": np.argsort(od)}
        ordm = {"P": np.argsort(-pv), "G": np.argsort(-gap), "Q": np.argsort(om)}
        HF, HM2 = {}, {}
        for h in ("P", "G", "Q"):
            HF[h] = [int(ub[q]) for q in ordf[h] if int(ub[q]) != af]
            HM2[h] = [int(ub[q]) for q in ordm[h] if int(ub[q]) != am]
            hit[h].append(len(set(HF[h][:2]) & set(HM2[h][:2])) / 2.0)
        for lab, h, kind, k in (("G馬単M4点", "G", "馬単M", 4),
                                ("Q三連単A4点", "Q", "三連単A", 4)):
            tf = B.tickets(kind, k, af, HF[h])
            tm = B.tickets(kind, k, am, HM2[h])
            if tf is None or tm is None:
                continue
            sf = {tuple(x[1]) for x in tf}
            sm = {tuple(x[1]) for x in tm}
            bet[lab].append(len(sf & sm) / len(sf | sm))

    if not nax[0]:
        print("　⚠**対象レースが0本**")
        return
    print(f"　★**確定版で軸が立つ {nax[0]}本**"
          f"　→ **朝9時でも軸が立つ {nax[1]}本（{100*nax[1]/nax[0]:.1f}%）**")
    print(f"　★★**そのうち軸が同一のレース: {nax[2]}本"
          f"（{100*nax[2]/max(nax[1],1):.1f}%）**")
    print(f"\n{'紐':<4}{'★上位2頭の一致率':>18}")
    for h in ("P", "G", "Q"):
        print(f"{h:<4}{100*np.mean(hit[h]):>17.1f}%")
    print(f"\n{'買い目':<16}{'本数':>6}{'★買い目そのものの一致率':>24}")
    for lab in bet:
        if bet[lab]:
            print(f"{lab:<16}{len(bet[lab]):>6}{100*np.mean(bet[lab]):>23.1f}%")
    print(f"　⚠★**pnを固定しているので、これは上限**。**実際はこれより低い**。")


def main():
    print("(219) ★★★★**朝のオッズだと、同じ馬が選ばれるのか**")
    print("⚠**data/odds_ts は別セッションが(148)で集めたもの。★読むだけ**\n")
    ts = load_dir()
    print(f"★時系列オッズ: **{len(ts):,}レース**"
          f"（★**ml/odds_ts.py の既存ローダーを使う**）")

    d = F.to_model(F.load_files())
    fin = {}
    for rid, g in d.groupby("raceid"):
        o = {int(u): float(x) for u, x in zip(g["umaban"], g["odds"]) if np.isfinite(x) and x > 0}
        if o:
            fin[str(rid)] = o

    def cmp_snap(hour, label, last=False):
        pa, pb, cross, jac, top1, top2, n = [], [], [], [], [], [], 0
        for rid, rec in ts.items():
            b = fin.get(rid)                    # ★raceid はそのまま突き合う
            if b is None:
                continue
            i = last_win(rec) if last else snap_at(rec, hour)
            if i is None:
                continue
            row = rec["odds"][i]
            a = {u + 1: float(row[u]) for u in range(rec["n"])
                 if np.isfinite(row[u]) and row[u] > 0}
            sh = [u for u in a if u in b]
            if len(sh) < 5:
                continue
            n += 1
            va = np.array([a[u] for u in sh])
            vb = np.array([b[u] for u in sh])
            pa += list(va); pb += list(vb)
            cross.append(np.mean((va >= LFIX) != (vb >= LFIX)))
            jac.append(jacc({u for u in sh if a[u] >= LFIX},
                            {u for u in sh if b[u] >= LFIX}))
            oa = sorted(sh, key=lambda u: a[u])
            ob = sorted(sh, key=lambda u: b[u])
            top1.append(oa[0] == ob[0])
            top2.append(len(set(oa[:2]) & set(ob[:2])) / 2.0)
        if not n:
            print(f"{label:<16} ⚠**突き合わせ0件**")
            return None
        pa, pb = np.array(pa), np.array(pb)
        r = float(np.corrcoef(np.log(pa), np.log(pb))[0, 1])
        chg = float(np.mean(np.abs(pa - pb) / pb))
        print(f"{label:<16}{n:>6}R{r:>9.4f}{100*chg:>10.1f}%"
              f"{100*np.mean(cross):>10.1f}%{100*np.mean(jac):>10.1f}%"
              f"{100*np.mean(top1):>9.1f}%{100*np.mean(top2):>9.1f}%")
        return dict(n=n, r=r, chg=chg, cross=float(np.mean(cross)),
                    jac=float(np.mean(jac)), t1=float(np.mean(top1)),
                    t2=float(np.mean(top2)))

    print(f"\n{'時点':<16}{'R数':>7}{'★対数相関':>9}{'平均変化':>10}"
          f"{'★10倍またぎ':>10}{'★10倍一致':>10}{'人気1一致':>9}{'人気1-2':>9}")
    last = cmp_snap(None, "★締切直前(対照)", last=True)
    res = {lab: cmp_snap(h, lab) for lab, h in POINTS}
    morn = res.get("★★朝 09:00")

    if last is None:
        print("\n⚠⚠**突き合わせができない。読まない**。")
        return
    okc = last["r"] > CORR_MIN and 100 * last["cross"] < CROSS_MAX
    print(f"\n★★内部対照（締切直前 vs 確定）: 相関 {last['r']:.4f}（>{CORR_MIN}）・"
          f"またぎ {100*last['cross']:.1f}%（<{CROSS_MAX}%）　"
          f"{'★★立った' if okc else '⚠⚠落ちた'}")
    if not okc:
        print("⚠⚠**対照が落ちた。読まない**（判定基準32）。")
        return

    if morn:
        print(f"\n■ ★★★**答え（朝09:00 vs 確定）**")
        print(f"　★**単勝オッズは平均 {100*morn['chg']:.1f}% 動く**（対数相関 {morn['r']:.4f}）")
        print(f"　★**10.0倍の線をまたぐ馬は {100*morn['cross']:.1f}%**"
              f"　→ ★**軸の候補集合の一致は {100*morn['jac']:.1f}%**")
        print(f"　★**人気1番の一致 {100*morn['t1']:.1f}% / 人気1-2番の一致 {100*morn['t2']:.1f}%**"
              f"　→ ★**紐Qがどれだけ変わるか**")

    axis_check(ts)

    print(f"\n■ ⚠**この測定で答えられないこと**")
    print("　★**朝版のROI・必要年数は出せない**（**複勝の板が1レース1枚・該当が数本**）。")
    print("　★**モデルの予測値（log_odds経由）の変化は直接は測っていない**。")
    print("\n⚠**枠連の運用には触れない**。**(148)の集め方にも触れない**。**設定変更は提案しない**。")


if __name__ == "__main__":
    sys.exit(selftest() if "--selftest" in sys.argv else (main() or 0))
