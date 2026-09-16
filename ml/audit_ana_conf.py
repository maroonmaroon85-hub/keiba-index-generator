"""⚠⚠★★★★**(248sns) 取り下げ（2026-09-15）—— ★実行せず・結果を見ていない**

★**利用者が直後に「いや違う」と方向を否定した**ので、**走らせたまま読まずに止めた**。
★**事前登録だけが commit fa317c5 に残っている**。⚠**結果は一度も見ていない**。
★★**追加マスは0**——**開けていないため**。★**記録にも加算していない**。
★**このファイルは消さない**——**「何を測ろうとして、なぜ止めたか」を残すため**。
　⚠**もし後で同じことを測るなら、★この事前登録をそのまま使ってよい**
　　（**手を入れずに残してある。★見てから書き直したものではない**）。
　⚠★**ただし条件が変わった**——**2026-09-15に単勝の下限が 10.0 → 8.0倍になった**。
　　★**下の事前登録は「単勝≥10.0倍」で書いてある**。**使う前にそこを直すこと**。

──────────────────── ★以下、取り下げた時点の事前登録（★手を入れていない） ────────────────────

(248sns) ★★★**「自信あり」印は成立するか** —— 利用者の案（2026-09-15）

★**利用者の案**: **「ズレ0.15を満たすレースは『自信あり』の形で発信する」**。
　★**0.15 は 0.10 の★部分集合**なので、**同じ配信の中で強弱をつけられる**。★**筋が通る**。

■ ⚠★★**先に落とし穴を書く（★測る理由）**
　★**軸は「候補の中で pn 最大の1頭」**。⚠**下限を下げると候補が増えるので、
　　★argmax が別の馬に移りうる**——**ズレ 0.10〜0.15 の間に pn の高い馬がいれば、そちらが軸になる**。
　★★**つまり同じレースでも「0.15の◎」と「0.10の◎」が★別の馬になりうる**。
　⚠**そうなると「自信あり」が別の馬を指すことになり、★配信として破綻する**。
　★**まずそこを数えるのが、この測定の本体**。

■ ★測るもの
　★**条件は現行のSNS**（**pn≥0.15 / 単勝≥10.0倍 / 障害除外**）。**ズレの下限だけ 0.15 と 0.10**。
　1. ★★**0.15が立つレースで、0.15の◎と0.10の◎が★同じ馬である割合**
　2. ★**「自信あり」（0.15が立つ）と「それ以外」（0.10だけ）の複勝的中率・ROI・中央配当**
　3. **頻度**（**1日あたり何本が「自信あり」になるか**）

■ ★★★★**ゲート2（判定基準42）—— ★何を返せば「自信ありと呼べない」か**
　★★**①軸が食い違うなら、そもそも配信として成り立たない**（**別の馬を指す**）。
　★**②的中率が「それ以外」と同じなら、★『自信あり』という言葉に中身が無い**。
　　⚠**その場合は「ズレが大きい」と事実だけ書くべきで、『自信あり』とは書けない**。
　★**③逆に的中率がはっきり高ければ、★言葉に中身がある**。
　⚠★**「0.15の方がROIが高い」は★既知**（101.8% vs 93.9%）。**そこは新しい情報ではない**。
　　★**新しいのは①と②**。

■ ★★内部対照（⚠**落ちたら読まない**）
　★**0.15 が 1,321本 / 複勝的中 24.0%**、★**0.10 が 5,015本 / 22.0%**
　　（**どちらも (243sns) の平地・障害除外**）。

■ 予想（⚠**当てにしない**・★このセッションは7勝6敗）
　★**軸の一致は 80〜90%**と見る（**ズレが大きい馬は pn も高いことが多いはず**）。
　★**「自信あり」の的中率は 24.0%、「それ以外」は 21%前後**と見る＝**差は3pt程度**。
　⚠**3ptの差だと「自信あり」と呼ぶには弱い**と見る。★**そのときは言葉を変える提案をする**。
　★**もし軸の一致が7割を切るなら、★この案は形を変えないと使えない**。

実行: python3 ml/audit_ana_conf.py
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

HI, LO = 0.15, 0.10
KNOWN = {HI: (1321, 24.0), LO: (5015, 22.0)}


def is_jump(dist):
    try:
        v = float(dist)
    except (TypeError, ValueError):
        return False
    return v > 0 and (v % 100 != 0 or v > 3600)


def main():
    print("(248sns) ★★★**「自信あり」印は成立するか**（利用者の案）")
    print(f"★**ズレ {HI} を満たすレースを「自信あり」として、{LO} の配信の中で強弱をつける**")
    print("⚠★**落とし穴: 下限を下げると argmax が別の馬に移りうる**——★**まずそこを数える**\n")

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
    sub = d.loc[m, ["raceid", "umaban", "odds", "date", "distance"]].copy()
    sub["p"] = pred[m]

    same, diff = 0, 0
    A = {"conf": {"fu": [], "od": []}, "rest": {"fu": [], "od": []}}
    nhi = nlo = 0
    yrs = set()
    for rid, g0 in sub.groupby("raceid"):
        rid = str(rid)
        r, bd = races.get(rid), boards.get(rid)
        if r is None or bd is None or is_jump(g0["distance"].iloc[0]):
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
        pn = pv / pv.sum() * NPLACE
        qp, _R = qpool([bd[int(u)] for u in ub], "harm")
        gap = pn - qp
        yrs.add(int(gg["date"].iloc[0].year))

        def axis(G):
            c = np.where((pn >= PN_FLOOR) & (gap >= G) & (od >= LFIX))[0]
            if not len(c):
                return None, None
            i = int(c[int(np.argmax(pn[c]))])
            return int(ub[i]), float(od[i])

        ah, oh = axis(HI)
        al, ol = axis(LO)
        if al is None:
            continue
        vl = payoff(r, "複勝", [al])
        if vl is None:
            continue
        nlo += 1
        if ah is not None:
            vh = payoff(r, "複勝", [ah])
            if vh is None:
                continue
            nhi += 1
            same += (ah == al)
            diff += (ah != al)
            A["conf"]["fu"].append(float(vh)); A["conf"]["od"].append(oh)
        else:
            A["rest"]["fu"].append(float(vl)); A["rest"]["od"].append(ol)

    ny = max(len(yrs), 1)
    print("■ ★★**内部対照**")
    ok = True
    for G, nm, n in ((HI, "0.15", nhi), (LO, "0.10", nlo)):
        wn, wh = KNOWN[G]
        g = abs(n - wn) <= 5
        ok &= g
        print(f"　該当({nm})　{n:>6,} vs {wn:>6,}　{'★通った' if g else '⚠落ちた'}")
    cf = np.array(A["conf"]["fu"]); rs = np.array(A["rest"]["fu"])
    hc = 100.0 * np.mean(cf > 0) if len(cf) else 0.0
    g2 = abs(hc - KNOWN[HI][1]) <= 0.5
    ok &= g2
    print(f"　複勝的中(0.15)　{hc:>5.1f}% vs {KNOWN[HI][1]:>5.1f}%　{'★通った' if g2 else '⚠落ちた'}")
    if not ok:
        print("\n⚠⚠**内部対照が落ちた。読まない**（判定基準32）。")
        return

    print(f"\n■ ★★★★**①軸は一致するか**（**0.15が立つ {nhi:,}レース**）")
    print(f"　★**同じ馬 {same:,}本（{100*same/max(nhi,1):.1f}%）**"
          f"　／　⚠**別の馬 {diff:,}本（{100*diff/max(nhi,1):.1f}%）**")
    v = ("★★**配信として成立する**" if 100*same/max(nhi, 1) >= 95
         else "⚠**食い違いが多い。★形を変えないと使えない**")
    print(f"　→ {v}")

    print(f"\n■ ★★**②「自信あり」と「それ以外」**（**11年 / {ny}年**）")
    print(f"{'':<14}{'本数':>8}{'年あたり':>9}{'1日':>7}"
          f"{'★複勝的中':>11}{'★複勝ROI':>11}{'中央配当':>10}{'中央オッズ':>11}")
    for k, nm in (("conf", "★自信あり"), ("rest", "それ以外")):
        a = np.array(A[k]["fu"]); o = np.array(A[k]["od"])
        if not len(a):
            continue
        h = a[a > 0]
        print(f"{nm:<14}{len(a):>8,}{len(a)/ny:>9.0f}{len(a)/ny/104:>7.2f}"
              f"{100*np.mean(a>0):>10.1f}%{roi_of(a):>10.1f}%"
              f"{(np.median(h) if len(h) else 0):>9,.0f}円{np.median(o):>10.1f}倍")
    hr_c = 100.0 * np.mean(cf > 0)
    hr_r = 100.0 * np.mean(rs > 0) if len(rs) else 0.0
    dd = hr_c - hr_r
    print(f"\n■ ★★★★**ゲート2の答え**")
    print(f"　★**的中率の差 {dd:+.1f}pt**（**自信あり {hr_c:.1f}% / それ以外 {hr_r:.1f}%**）")
    if dd >= 5:
        print("　→ ★★**『自信あり』と呼べる。★言葉に中身がある**")
    elif dd >= 2:
        print("　→ ⚠**差はあるが小さい**。★**『自信あり』より「ズレが特に大きい」と事実で書く方が正確**")
    else:
        print("　→ ⚠⚠**差が無い。★『自信あり』とは書けない**（**言葉に中身が無い**）")
    print(f"　⚠**ROIの差（{roi_of(cf):.1f}% vs {roi_of(rs):.1f}%）は既知**"
          "——**(240sns)で測ってある。★新しい情報ではない**")


if __name__ == "__main__":
    main()
