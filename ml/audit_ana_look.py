"""(237) ★★★**複勝が当たったとき、どれくらい「かっこつく」のか** —— 配当と人気の分布を出す

★★**動機（2026-09-09・利用者「複勝当たれば多少かっこはつくと思うんだよね」）**
　★**同意する。そして私は前に複勝を過小評価した**——**「平均配当435円で見せ場が無い」と書いたが、
　　それは `ANA_RULE.md` の★「高配当」という目的で判定した言葉**。**SNSの基準は別**。
　★**100円→435円で、来るのは6.6番人気・15.9倍の馬**。**穴が的中している絵ではある**。

■ ⚠★★★**だが、ここに私が今日2回踏んだのと同じ罠がある**
　★**21.2倍・6.6番人気は「軸ぜんぶ」の値**。⚠**「当たったときの軸」の値ではない**。
　★**人気薄ほど来ないので、的中は★短いオッズ側に寄っているはず**。
　★★**寄っているなら「かっこつく」度合いは見かけより低い**——
　　**「平均21.2倍の馬が23.4%で来る」と書いたら、それは★別の量を混ぜたことになる**（**判定基準37**）。
　⚠**私は今日 1,383と1,398 を取り違え、対照に0本の日を混ぜた。★3回目をやらない**。

■ ★★これは記述であって検定ではない
　★**マスを1つも増やさない**。**(210)で凍結した複勝1点の★中身を分解するだけ**。

■ ★★★測るもの
　1. ★**複勝配当の分布**（**中央 / 四分位 / 500円以上・1,000円以上の割合 / 最大**）
　2. ★★**的中したときの軸の単勝オッズ・人気の分布** ← ★**罠の本体**
　3. ★**それを「軸ぜんぶ」の分布と並べる**（**どれだけ短い側に寄ったか**）
　4. ★**オッズ帯ごとの複勝的中率**（**10-15倍 / 15-20倍 / 20-30倍 / 30倍以上**）
　5. **参考: 単勝（軸が1着）の同じ分解**——★**「一番かっこつく」のはこれ**

■ ★★ゲート2（判定基準42）—— **何を返せば「かっこつかない」か**
　★**的中が短いオッズ側に寄っていなければ、的中時の中央オッズは全体の15.9倍と一致する**。
　⚠**大きく下回れば「当たるのは人気側の穴だけ」**＝★**穴として見せられる回数は23.4%より少ない**。
　★**逆に一致すれば、23.4%はそのまま「穴が来た」と言ってよい**。

■ ★内部対照（⚠落ちたら読まない）
　★**全数1,398本 / 複勝の的中率23.5% / 軸の中央オッズ15.8倍**（**(231)と同じ**）。

■ 予想（⚠**当てにしない**・★このセッションは3勝1敗）
　★**的中時の中央オッズは 13〜15倍**と見る（**全体15.8倍より少し短い**）。
　★**大きくは寄らない**と見る——**pnで選んでいるので、モデルは既にオッズと別の情報を見ている**。
　⚠**もし10倍近くまで寄っていたら、それは「軸の下限10.0倍にへばりついた馬しか来ない」ことを意味し、
　　★SNSの見せ方を考え直す必要がある**。


■ ★★★★**結果（実測済み・2026-09-09）**　★**内部対照3つとも通過**（1,398本 / 23.5% / 15.8倍）

**★①複勝配当の分布**（**的中 328本**）
| 中央 | 25% | 75% | 最大 |
|---|---|---|---|
| ★**370円** | 310円 | 482円 | **2,130円** |
| ★**500円以上**は的中の**24.1%** | ⚠**全体では5.7%＝約18本に1本** | | |
| **1,000円以上**は的中の**2.1%** | ⚠**全体では0.5%＝約200本に1本** | | |
　★**典型的な的中は370円**。⚠**「万馬券」的な絵にはならない**。

**★★②ゲート2の答え —— ⚠寄っていた（★だが人気はほとんど動かない）**
| | 中央オッズ | 平均オッズ | 中央人気 | 平均人気 |
|---|---|---|---|---|
| **軸ぜんぶ** | 15.8倍 | **21.2倍** | **6番** | 6.6番 |
| ★**複勝が当たった軸** | ★**13.8倍** | ★**16.9倍** | ★**6番** | 6.1番 |
| 　うち1着 | 13.2倍 | 15.3倍 | 5番 | 5.8番 |
| ⚠**外れた軸** | 16.6倍 | 22.6倍 | 7番 | 6.8番 |
　★**中央オッズは −2.0倍**（**事前登録した閾値1.5倍を超えた＝寄っている**）。
　★★**だが★中央人気は6番のまま動かない**——**平均が 21.2→16.9倍 に動いたのは
　　★30倍以上がほとんど来ないから**で、**「6番人気の馬が来た」は嘘にならない**。
　⚠★★**書いてよい**: **「6番人気の馬が23.5%で複勝に来る」**。
　⚠★★**書いてはいけない**: **「平均21.2倍の馬が23.4%で来る」**（**的中の平均は16.9倍**）。

**★★★③オッズ帯ごと —— ★的中率と配当が真っ向から交換される**
| 帯 | 本数 | ★複勝的中率 | 中央配当 | 単勝的中率 |
|---|---|---|---|---|
| **10〜15倍** | **627**（**45%**） | ★**31.7%** | ⚠**330円** | **10.2%** |
| 15〜20倍 | 283 | 21.9% | 410円 | 5.7% |
| 20〜30倍 | 257 | 16.0% | 530円 | 3.9% |
| ★**30倍以上** | 231 | ⚠**11.3%** | ★**820円** | 1.3% |
　★**軸の45%は10〜15倍**＝**そこが「当たるがかっこつかない」帯**。
　★**30倍以上は中央820円まで上がるが、的中は11.3%に落ちる**。⚠**綺麗に交換されている**。

**★★★★④単勝 —— ★いちばん「かっこつく」のはここだった**
| | 値 |
|---|---|
| **的中率** | ★**6.7%** |
| ★**中央配当** | ★★**1,320円**（**25% 1,150 / 75% 1,630 / 最大 4,890円**） |
| **頻度** | ★**年8.5本**（⚠**朝9時運用なら年5.3本**） |
　★★**6番人気の馬の単勝が1,320円**——**これが年5回**。
　★**三連単A4点（的中1.8%・年1.4本）より★3.7倍多く当たり、★1点100円で4分の1の値段**。
　★**平÷中は1.16で裾に頼っていない**（**三連単A4点は1.24**）。
　⚠**ただし単勝のROIは102.1%・必要年数2,028年**。**(226)で後半+14.5ptは
　　★4,890円1本の裾と判明している**。⚠**「儲かる」根拠にはならない**。

■ ★予想の答え合わせ（判定基準24・★このセッション4回目）
| 予想 | 実際 | |
|---|---|---|
| **的中時の中央オッズは13〜15倍** | ★**13.8倍** | ★**当たり** |
| **大きくは寄らない** | ⚠**−2.0倍で、自分で決めた閾値1.5倍を超えた** | ⚠**外れ** |
　★**閾値を先に書いておいたので、自分の願望で「寄っていない」と読むことができなかった**。

実行: python3 ml/audit_ana_look.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import features as F
from audit_crosspool import load_races, payoff
from audit_ana_odds import MIN_HORSES
from audit_ana_marg import wf_predict
from audit_ana_board import NPLACE, load_fuku_boards, qpool
from audit_ana_band import PN_FLOOR
from audit_ana_hole import GAP
from audit_ana_fix import LFIX
from train_prod import add_odds_features

KNOWN_N, KNOWN_HIT, KNOWN_ODMED = 1398, 23.5, 15.8


def dist(v, lab, unit="円"):
    a = np.asarray(v, float)
    if not len(a):
        print(f"　{lab:<22}—")
        return
    print(f"　{lab:<22}中央 {np.median(a):>7,.0f}{unit} / "
          f"25% {np.percentile(a,25):>7,.0f} / 75% {np.percentile(a,75):>7,.0f} / "
          f"最大 {a.max():>8,.0f}{unit}　(n={len(a):,})")


def main():
    print("(237) ★★★**複勝が当たったとき、どれくらい「かっこつく」のか**")
    print("★**記述であって検定ではない**。⚠**マスは1つも増やしていない**\n")

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

    AX = {"od": [], "rk": [], "fuku": [], "tan": []}
    for rid, g in sub.groupby("raceid"):
        rid = str(rid)
        r, bd = races.get(rid), boards.get(rid)
        if r is None or bd is None:
            continue
        nums = {u for u, _, _ in r["horses"]}
        if len(nums) < MIN_HORSES:
            continue
        gg = g[g["umaban"].astype(int).isin(nums)]
        ub = gg["umaban"].astype(int).to_numpy()
        if len(gg) < MIN_HORSES or not all(int(u) in bd for u in ub):
            continue
        od = gg["odds"].to_numpy(float)
        pv = gg["p"].to_numpy(float)
        if not np.isfinite(od).all() or (od <= 0).any() or pv.sum() <= 0:
            continue
        pn = pv / pv.sum() * NPLACE
        qp, _R = qpool([bd[int(u)] for u in ub], "harm")
        cand = np.where((pn >= PN_FLOOR) & (pn - qp >= GAP) & (od >= LFIX))[0]
        if not len(cand):
            continue
        i = int(cand[int(np.argmax(pn[cand]))])
        ax = int(ub[i])
        v0 = payoff(r, "複勝", [ax])
        if v0 is None:
            continue
        rank = np.argsort(np.argsort(od)) + 1
        AX["od"].append(float(od[i]))
        AX["rk"].append(int(rank[i]))
        AX["fuku"].append(float(v0))
        vt = payoff(r, "単勝", [ax])
        AX["tan"].append(float(vt) if vt is not None else 0.0)

    od = np.array(AX["od"]); rk = np.array(AX["rk"])
    fu = np.array(AX["fuku"]); ta = np.array(AX["tan"])
    hit = fu > 0
    win = ta > 0
    hr = 100.0 * hit.mean()

    print("■ ★★**内部対照**")
    ok = [("全数", len(od), KNOWN_N, 0, len(od) == KNOWN_N),
          ("複勝の的中率(%)", hr, KNOWN_HIT, 0.5, abs(hr - KNOWN_HIT) <= 0.5),
          ("軸の中央オッズ", float(np.median(od)), KNOWN_ODMED, 0.5,
           abs(np.median(od) - KNOWN_ODMED) <= 0.5)]
    for nm, got, want, tol, good in ok:
        print(f"　{nm:<16}{got:>10.1f} vs {want:>8.1f}　{'★通った' if good else '⚠落ちた'}")
    if not all(g for *_, g in ok):
        print("\n⚠⚠**内部対照が落ちた。読まない**（判定基準32）。")
        return

    print(f"\n■ ★**①複勝配当の分布**（**的中 {hit.sum():,}本 / {len(od):,}本 = {hr:.1f}%**）")
    h = fu[hit]
    dist(h, "複勝配当")
    for th in (300, 400, 500, 700, 1000):
        print(f"　　★**{th:,}円以上**: {100*np.mean(h >= th):>5.1f}%"
              f"　(★全体では {100*np.mean(fu >= th):>4.1f}%＝"
              f"**約{100/max(np.mean(fu>=th),1e-9)/100:>4.1f}本に1本**)")

    print(f"\n■ ★★★**②的中したときの軸は、どれくらい穴だったのか**　← ★**ここが罠の本体**")
    print(f"{'':<24}{'中央オッズ':>12}{'平均オッズ':>12}{'中央人気':>10}{'平均人気':>10}")
    for lab, msk in (("★軸ぜんぶ", np.ones(len(od), bool)),
                     ("★★複勝が当たった軸", hit),
                     ("　うち1着（単勝的中）", win),
                     ("⚠外れた軸", ~hit)):
        print(f"{lab:<24}{np.median(od[msk]):>11.1f}倍{od[msk].mean():>11.1f}倍"
              f"{np.median(rk[msk]):>9.0f}番{rk[msk].mean():>9.1f}番")

    d_med = float(np.median(od[hit]) - np.median(od))
    print(f"\n　★**ゲート2の答え: 的中時の中央オッズは 全体より {d_med:+.1f}倍**")
    if abs(d_med) < 1.5:
        print("　★★**ほとんど寄っていない → 23.4%はそのまま「穴が来た」と言ってよい**")
    else:
        print("　⚠**短い側に寄っている → 「平均21.2倍の馬が23.4%で来る」とは書けない**")

    print(f"\n■ ★**③オッズ帯ごとの複勝的中率**（⚠**軸の下限は10.0倍**）")
    print(f"{'帯':<14}{'本数':>8}{'複勝的中率':>12}{'中央配当':>12}{'単勝的中率':>12}")
    for lo, hi, lab in ((10, 15, "10〜15倍"), (15, 20, "15〜20倍"),
                        (20, 30, "20〜30倍"), (30, 1e9, "30倍以上")):
        s = (od >= lo) & (od < hi)
        if s.sum() < 30:
            continue
        hh = fu[s & hit]
        print(f"{lab:<14}{s.sum():>8,}{100*np.mean(fu[s]>0):>11.1f}%"
              f"{np.median(hh) if len(hh) else 0:>11,.0f}円{100*np.mean(ta[s]>0):>11.1f}%")

    print(f"\n■ ★**④参考: 単勝（軸が1着）**——★**いちばんかっこつく形**")
    print(f"　★**的中率 {100*win.mean():.1f}%**（**{len(od)/11:.0f}本/年 × これ ＝ "
          f"年{len(od)/11*win.mean():.1f}本**）")
    dist(ta[win], "単勝配当")
    print(f"　⚠**朝9時運用だと本数は約2/3**（**年80本前後**）"
          f"　→ ★**単勝的中は年{80*win.mean():.1f}本前後**")


if __name__ == "__main__":
    main()
