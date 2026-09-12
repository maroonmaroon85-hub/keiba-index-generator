"""(243sns) ★★★**障害戦は平地と同じ線に乗るのか** —— 利用者の指定（2026-09-12）

★**きっかけ**: **2026-09-12 のSNS用の印で、★阪神1R（障2970m）が◎に選ばれた**。
　⚠★**私も規則側も、障害戦を一度も区別していなかった**。

■ ⚠★★**先に、データの状態を書く（★ここが問題の本体）**
　★**生データのトラック種別は `芝` と `ダ` の2値だけ**（**749,951 / 749,931行**）。
　⚠⚠**`障` というカテゴリが存在しない**。**障害戦は芝かダに混ぜて記録されている**。
　★**`reco_ana_day.py` の変換も `surf = 1 if surface=="ダ" else 0`** なので、
　　⚠**障害は「芝」として扱われる**。→ ★**モデルは障害を平地として学習・予測している**。

■ ★★**障害の判定（★距離で見分ける・事前に決める）**
```
障害 = （距離が100の倍数でない） または （距離 > 3600m）
```
　★**根拠**: **JRAの平地は100mの倍数で最長3,600m**（**ステイヤーズS**）。
　★**障害は 2,970 / 3,110 / 3,170 / 3,250 / 3,390 / 3,930 / 4,250m など★半端な距離**。
　⚠★**限界を書く**: **障害にも 3,000/3,100/3,200/3,300m の切りの良い距離がある**ので、
　　★**この判定は障害を★取りこぼす方向に外れる**（**混入は平地側に残る**）。
　★**つまり「障害を除いた側」はまだ少し汚れている**。⚠**それでも向きは分かる**。

■ ★測るもの（⚠**マスを増やさない・(210)の軸と(240sns)の下限を使うだけ**）
　★**ズレ下限 0.15（規則側）と 0.10（SNS側）の両方で**、
　★**該当を「平地」と「障害」に割り、次を並べる**:
　　**該当数 / 全レースに占める割合 / 複勝的中率 / 複勝ROI / 単勝的中率 / 軸の中央オッズ**

■ ★★★ゲート2（判定基準42）—— **何を返せば「区別しなくてよい」か**
　★**障害が平地と同じ線に乗るなら、★複勝的中率とROIが平地と一致する**。
　⚠**大きく外れるなら、★障害は別の母集団**＝**混ぜて測っていた11年の数字が少し汚れている**。
　★**どちらに転んでも、★障害の本数が少なければ全体への影響は小さい**——**そこも出す**。

■ ★★内部対照（⚠**落ちたら読まない**）
　★**平地＋障害＝全体**が、**G=0.15 で 1,398本 / G=0.10 で 5,297本**（**(240sns)①**）。

■ 予想（⚠**当てにしない**・★このセッションは5勝4敗）
　★**障害での該当率は平地より★高い**と見る（**頭数が少なく、モデルと市場がずれやすい**）。
　★**複勝的中率は平地より★低い**と見る（**落馬・競走中止がある**）。
　⚠**本数は全体の5%未満**と見るので、**11年の数字への影響は小さい**と見る。
　★**もし障害の方が成績が良ければ、それは「平地として学習したモデルが障害で効いている」
　　という妙な話**になる。★**そのときは除外ではなく、理由を調べる側に回る**。


■ ★★★★**結果（実測済み・2026-09-12）**　★**内部対照2つとも通過**（1,398 / 5,297）
　★**対象レース: 平地 28,569 / ★障害 901（3.1%）**

| ズレ下限 | | 該当 | ★**該当率** | ★**複勝的中** | 複勝ROI | 単勝的中 | 中央オッズ |
|---|---|---|---|---|---|---|---|
| **0.15** | 平地 | 1,321 | **4.6%** | ★**24.0%** | 101.8% | 6.8% | 15.7倍 |
| **0.15** | ★**障害** | 77 | ⚠**8.5%** | ⚠**14.3%** | 98.3% | 3.9% | 20.6倍 |
| **0.10** | 平地 | 5,015 | **17.6%** | **22.0%** | 93.9% | 5.6% | 16.2倍 |
| **0.10** | ★**障害** | 282 | ⚠**31.3%** | ⚠**17.4%** | 99.0% | 4.3% | 19.9倍 |

■ ★★★**読み取れること（★3つ）**
　★**① 障害では軸が★約2倍の頻度で立つ**（**4.6→8.5% / 17.6→31.3%**）。
　★**② なのに複勝は★来ない**（**24.0→14.3pt＝−9.7pt / 22.0→17.4%**）。
　★★**①と②は同じことの表と裏**——**モデルは障害を「芝」として学習している**ので、
　　★**障害では系統的に外す → ズレが大きく見える → 軸が立つ → だが当たらない**。
　　⚠★**つまり障害のズレは「妙味」ではなく★モデルの誤差である**。
　★**③ 全体のROIへの影響は小さい**（**±0.2〜0.3pt**）——**該当の5.3〜5.5%しかないため**。
　⚠**ROIの符号は下限で逆転する**（**0.15で−3.5pt / 0.10で+5.1pt**）。
　　★**77本と282本なので、★ROIの差は読まない**（判定基準5）。★**読むのは①②**。

■ ★★★**結論: ★障害は除外する（★理由はROIではない）**
| ★**除外する理由** | |
|---|---|
| **1** | ★★**モデルが障害を平地として扱っている**＝**構造的な誤りで、測定以前の問題** |
| **2** | ★**複勝的中率が明確に低い**（**−9.7pt / −4.6pt**）＝**SNSの見せ物として弱い** |
| **3** | ★**失うものが小さい**（**該当の5%・ROIへの影響±0.3pt**） |
　⚠★**ROIを理由にしない**——**本数が少なすぎて符号が安定しない**。

■ ★予想の答え合わせ（判定基準24・★このセッション6回目）
| 予想 | 実際 | |
|---|---|---|
| **障害の該当率は平地より高い** | ★**約2倍**（4.6→8.5% / 17.6→31.3%） | ★**当たり** |
| **複勝的中率は平地より低い** | ★**−9.7pt / −4.6pt** | ★**当たり** |
| **本数は全体の5%未満** | ⚠**レースでは3.1%だが★該当では5.3〜5.5%** | ⚠**惜しい外れ** |
　★**外した理由が①そのもの**——**障害は該当率が2倍なので、レース比より★該当比で膨らむ**。

実行: python3 ml/audit_ana_jump.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import features as F
from audit_crosspool import load_races, payoff
from audit_ana_odds import MIN_HORSES, roi_of
from audit_ana_marg import wf_predict
from audit_ana_board import NPLACE, load_fuku_boards, qpool
from audit_ana_band import PN_FLOOR
from audit_ana_fix import LFIX
from train_prod import add_odds_features

GAPS = [0.15, 0.10]
KNOWN = {0.15: 1398, 0.10: 5297}


def is_jump(dist):
    """★障害の判定（★事前に決めた・距離だけで見る）"""
    return (float(dist) % 100 != 0) or (float(dist) > 3600)


def main():
    print("(243sns) ★★★**障害戦は平地と同じ線に乗るのか**（利用者の指定）")
    print("⚠**生データに『障』が無い**——**障害は芝/ダに混ぜて記録され、"
          "★モデルは平地として学習している**\n")

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
    sub = d.loc[m, ["raceid", "umaban", "odds", "distance"]].copy()
    sub["p"] = pred[m]

    ACC = {g: {0: {"n": 0, "fu": [], "ta": [], "od": []},
               1: {"n": 0, "fu": [], "ta": [], "od": []}} for g in GAPS}
    NR = {0: 0, 1: 0}
    for rid, g0 in sub.groupby("raceid"):
        rid = str(rid)
        r, bd = races.get(rid), boards.get(rid)
        if r is None or bd is None:
            continue
        nums = {u for u, _, _ in r["horses"]}
        if len(nums) < MIN_HORSES:
            continue
        gg = g0[g0["umaban"].astype(int).isin(nums)]
        ub = gg["umaban"].astype(int).to_numpy()
        if len(gg) < MIN_HORSES or not all(int(u) in bd for u in ub):
            continue
        od = gg["odds"].to_numpy(float)
        pv = gg["p"].to_numpy(float)
        if not np.isfinite(od).all() or (od <= 0).any() or pv.sum() <= 0:
            continue
        J = 1 if is_jump(gg["distance"].iloc[0]) else 0
        NR[J] += 1
        pn = pv / pv.sum() * NPLACE
        qp, _R = qpool([bd[int(u)] for u in ub], "harm")
        gap = pn - qp
        for G in GAPS:
            c = np.where((pn >= PN_FLOOR) & (gap >= G) & (od >= LFIX))[0]
            if not len(c):
                continue
            i = int(c[int(np.argmax(pn[c]))])
            ax = int(ub[i])
            v0 = payoff(r, "複勝", [ax])
            if v0 is None:
                continue
            vt = payoff(r, "単勝", [ax])
            a = ACC[G][J]
            a["n"] += 1
            a["fu"].append(float(v0))
            a["ta"].append(float(vt) if vt is not None else 0.0)
            a["od"].append(float(od[i]))

    print(f"■ ★**対象レース**　平地 **{NR[0]:,}** / ★**障害 {NR[1]:,}**"
          f"（**{100*NR[1]/max(NR[0]+NR[1],1):.1f}%**）")

    print("\n■ ★★**内部対照**（**平地＋障害＝(240sns)①の該当数**）")
    okall = True
    for G in GAPS:
        tot = ACC[G][0]["n"] + ACC[G][1]["n"]
        good = abs(tot - KNOWN[G]) <= 5
        okall &= good
        print(f"　ズレ≥{G}　{tot:>6,} vs {KNOWN[G]:>6,}　{'★通った' if good else '⚠落ちた'}")
    if not okall:
        print("\n⚠⚠**内部対照が落ちた。読まない**（判定基準32）。")
        return

    for G in GAPS:
        print(f"\n■ ★★★**ズレ下限 {G}**")
        print(f"{'':<8}{'該当':>8}{'レース':>9}{'★該当率':>10}"
              f"{'複勝的中':>10}{'複勝ROI':>10}{'単勝的中':>10}{'中央オッズ':>12}")
        for J, lab in ((0, "平地"), (1, "★障害")):
            a = ACC[G][J]
            if not a["n"]:
                print(f"{lab:<8}{'0':>8}{NR[J]:>9,}{'—':>10}")
                continue
            fu = np.array(a["fu"]); ta = np.array(a["ta"])
            print(f"{lab:<8}{a['n']:>8,}{NR[J]:>9,}{100*a['n']/max(NR[J],1):>9.1f}%"
                  f"{100*np.mean(fu>0):>9.1f}%{roi_of(fu):>9.1f}%"
                  f"{100*np.mean(ta>0):>9.1f}%{np.median(a['od']):>11.1f}倍")
        a0, a1 = ACC[G][0], ACC[G][1]
        if a1["n"]:
            f0 = np.array(a0["fu"]); f1 = np.array(a1["fu"])
            dh = 100*np.mean(f1 > 0) - 100*np.mean(f0 > 0)
            dr = roi_of(f1) - roi_of(f0)
            sh = 100.0 * a1["n"] / max(a0["n"] + a1["n"], 1)
            print(f"　★**障害は該当の {sh:.1f}%**"
                  f"　★**複勝的中の差 {dh:+.1f}pt**　★**ROIの差 {dr:+.1f}pt**")
            # ★全体への影響: 障害を抜いたら全体のROIはどう動くか
            allf = np.concatenate([f0, f1])
            print(f"　★**障害を抜くと 複勝ROI {roi_of(allf):.1f}% → {roi_of(f0):.1f}%"
                  f"（{roi_of(f0)-roi_of(allf):+.1f}pt）**")

    print("\n⚠★**判定の限界**: **障害にも切りの良い距離がある**ので、"
          "★**この判定は障害を取りこぼす方向に外れる**（**平地側に混入が残る**）。")


if __name__ == "__main__":
    main()
