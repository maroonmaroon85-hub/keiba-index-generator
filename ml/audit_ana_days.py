"""(231) ★★★**該当0本の開催日は何%か** —— 軸の頻度を「レース」ではなく「日」で数える

★★**動機（2026-09-08・利用者の指定「はかって」）**
　★**ANA_SNS_BRIEF.md は「5日に2日（39.1%）は該当0本」と書いている**が、
　⚠**リポジトリ内にこの39.1%を出した測定が見つからない**。**1日あたり1.2本までしか無い**。
　★**SNSの体感は「1日1.2本」ではなく「今日は買えるのか / 何日当たっていないか」で決まる**
　　ので、**日単位の分布が要る**。

■ ★★★**これは検定ではない。★純粋な記述である**
　★**新しいマスを1つも作らない**——**券種も紐も点数も出てこない**。
　★**(210)で凍結した軸判定の出力を、レース単位から日単位に組み替えるだけ**。
　⚠**だから「良い数字が出た」という読み方をしてはいけない**（判定基準43）。

■ ★軸（**(210)で凍結・動かさない**）
　**pn ≥ 0.15 かつ ズレ ≥ 0.15 かつ 単勝オッズ ≥ 10.0倍 の中で pn 最大の1頭**。
　★**候補が無ければ見送り**。⚠**買い目は複勝1点のみ**（**見せ場の設計はこの測定の外**）。

■ ★★★**分母の定義（★先に決める・ここを曖昧にすると数字が動く）**
| 記号 | 分母 | ★**意味** |
|---|---|---|
| **(a)** | ★**判定が走った日** | **板が引けて軸判定まで到達したレースが1本以上ある開催日** |
| **(b)** | ★**開催日ぜんぶ** | **モデルの予測が出たレースが1本以上ある開催日** |
　★**(a)は「軸が本当に立たなかった日」**、**(b)は「利用者が朝に画面を見る日」**。
　⚠★**(b)の0本には「板が無くて判定できなかった日」が混ざる**——
　　**両方出さないと必ず誤読する**（**判定基準26: 物差しを替える**）。

■ ★出す量（**4つ**）
　1. **開催日ごとの該当本数の分布**（**0 / 1 / 2 / 3本以上**）
　2. ★**該当0本の日の割合** ← ★**ブリーフの39.1%と突き合わせる**
　3. **複勝が1本以上当たった日の割合**
　4. ★★**連続で当たらない開催日数**（**中央 / 95%点 / 最大**）← ★**SNSの体感そのもの**

■ ★★ゲート2（判定基準42）—— **この測定は何を返せば「意味なし」か**
　★**記述なので偽の返り値は無い**。⚠**だから対照を1つ置く**:
　★★**同じ日付集合の上で、的中フラグだけを無作為に並べ替えた場合の連続外れ日数**を併記する。
　★**的中が日にランダムに散っているだけなら、実測はこの対照と一致する**。
　★★**一致しなければ「当たりが日に固まっている」＝★体感はこの数字より悪い**。
　⚠**並べ替えは該当本数の構造を壊さないよう、★日ごとの該当本数は固定したまま
　　「その日に当たったか」だけを入れ替える**（**NRAND=10種・SEEDは(209)と同じ**）。

■ ★★内部対照（**⚠これが合わなければ経路が違う。結果を読まない**）
　★**該当レースの総数が 1,383本**（**(218)の参考行・複勝1点の本数**）**±5**。
　★**的中率が 23.4% ±0.5pt**。★**軸の中央オッズが 15.9倍 ±0.5**。

■ 予想（⚠**当てにしない**・判定基準24。★**前セッションの私は10回外した**）
　★**(a)の0本の日は 35〜45%**と見る（**ブリーフの39.1%が正しければここに入る**）。
　★**(b)はそれより高い**と見る（**板の欠けが混ざるため**）。
　★★**連続外れの最大は20開催日を超える**と見る（**年104日なので2ヶ月以上**）。
　⚠**もし実測が対照より明確に長ければ、それは「当たりが日に固まっている」ことを意味する**。

実行: python3 ml/audit_ana_days.py
"""
import os
import sys
from binascii import crc32

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import features as F
from audit_crosspool import load_races, payoff
from audit_ana_odds import MIN_HORSES, band_of, BANDS
from audit_ana_marg import wf_predict
from audit_ana_board import NPLACE, load_fuku_boards, qpool
from audit_ana_band import PN_FLOOR
from audit_ana_hole import GAP, NRAND, SEED
from audit_ana_fix import LFIX
from train_prod import add_odds_features

KNOWN_N, KNOWN_HIT, KNOWN_ODMED = 1383, 23.4, 15.9
N_TOL, HIT_TOL, OD_TOL = 5, 0.5, 0.5


def runs_of_misses(hits):
    """★**当たった日と当たった日の間に挟まる「当たらない開催日」の本数**を並べる。
    ⚠**先頭の連（最初の的中より前）と末尾の連は★打ち切られている**ので落とす
    （**判定基準43: 端の1標本を結論に混ぜない**）。"""
    idx = [i for i, h in enumerate(hits) if h]
    if len(idx) < 2:
        return []
    return [b - a - 1 for a, b in zip(idx, idx[1:])]


def describe(runs):
    if not runs:
        return "—"
    a = np.array(runs, float)
    return (f"中央 {np.median(a):>4.0f}日 / 95%点 {np.percentile(a, 95):>4.0f}日 / "
            f"最大 {a.max():>4.0f}日　(連 {len(a)}本)")


def main():
    print("(231) ★★★**該当0本の開催日は何%か** —— 軸の頻度を「日」で数える")
    print("★**これは記述であって検定ではない**。⚠**新しいマスは1つも作っていない**\n")

    races = {r["rid"]: r for r in load_races()}
    print("★複勝の板(type=2)を読む…")
    boards = load_fuku_boards()
    print(f"　**{len(boards):,}レース分**")

    d = F.to_model(F.load_files())
    f = F.build_features(d)
    keep = (f["n_prior"] >= 1) & d["odds"].notna() & (d["odds"] > 0)
    d, f = d[keep].reset_index(drop=True), f[keep].reset_index(drop=True)
    y = (d["finish"] <= 3).astype(int).to_numpy()
    fx, _ = F.encode_categoricals(f)
    fx = add_odds_features(fx, d["odds"].to_numpy(float), d["raceid"].to_numpy())
    pred = wf_predict(d, fx, y, 3)
    msk = ~np.isnan(pred)
    sub = d.loc[msk, ["raceid", "umaban", "odds", "date"]].copy()
    sub["p"] = pred[msk]

    # ★日ごとに: ran = 判定が走ったレース数 / cand = 該当本数 / hit = 複勝的中本数
    day = {}
    axod = []
    for rid, g in sub.groupby("raceid"):
        rid = str(rid)
        dt = g["date"].iloc[0].date()
        rec = day.setdefault(dt, {"seen": 0, "ran": 0, "cand": 0, "hit": 0})
        rec["seen"] += 1                       # ★(b)の分母: 予測が出たレース
        r = races.get(rid)
        bd = boards.get(rid)
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
        gap = pn - qp
        rec["ran"] += 1                        # ★(a)の分母: 軸判定まで到達したレース
        cand = np.where((pn >= PN_FLOOR) & (gap >= GAP) & (od >= LFIX))[0]
        if not len(cand):
            continue
        i = int(cand[int(np.argmax(pn[cand]))])
        ax = int(ub[i])
        v0 = payoff(r, "複勝", [ax])
        if v0 is None:
            continue
        rec["cand"] += 1
        rec["hit"] += 1 if v0 > 0 else 0
        axod.append(float(od[i]))

    days = sorted(day)
    nc = np.array([day[t]["cand"] for t in days])
    nh = np.array([day[t]["hit"] for t in days])
    ran = np.array([day[t]["ran"] for t in days])

    # ── ★★内部対照（⚠落ちたら読まない） ──────────────────────────
    tot, hits = int(nc.sum()), int(nh.sum())
    hr = 100.0 * hits / tot if tot else 0.0
    odmed = float(np.median(axod)) if axod else 0.0
    ok = [("該当レース総数", tot, KNOWN_N, N_TOL, abs(tot - KNOWN_N) <= N_TOL),
          ("複勝の的中率(%)", hr, KNOWN_HIT, HIT_TOL, abs(hr - KNOWN_HIT) <= HIT_TOL),
          ("軸の中央オッズ", odmed, KNOWN_ODMED, OD_TOL,
           abs(odmed - KNOWN_ODMED) <= OD_TOL)]
    print("\n■ ★★**内部対照**（**(210)(218)と一致するか**）")
    for nm, got, want, tol, good in ok:
        print(f"　{nm:<16}{got:>10.1f} vs {want:>7.1f} (±{tol})"
              f"　{'★一致' if good else '⚠ずれた'}")
    if not all(g for *_, g in ok):
        print("\n⚠⚠**内部対照が落ちた。経路が違う。★結果を読まない**（判定基準32/37）。")
        return

    # ── ① 日ごとの該当本数の分布 ─────────────────────────────
    ma = ran > 0                                # (a) 判定が走った日
    mb = np.ones(len(days), bool)               # (b) 開催日ぜんぶ
    print(f"\n■ ★**①開催日ごとの該当本数**"
          f"　(a)判定が走った日 **{ma.sum():,}日** / (b)開催日ぜんぶ **{mb.sum():,}日**")
    print(f"{'該当':<8}{'(a) 日数':>10}{'(a) 割合':>10}{'(b) 日数':>10}{'(b) 割合':>10}")
    for lo, hi, nm in [(0, 0, "★0本"), (1, 1, "1本"), (2, 2, "2本"), (3, 99, "3本以上")]:
        sa = ((nc >= lo) & (nc <= hi) & ma).sum()
        sb = ((nc >= lo) & (nc <= hi) & mb).sum()
        print(f"{nm:<8}{sa:>10,}{100*sa/max(ma.sum(),1):>9.1f}%"
              f"{sb:>10,}{100*sb/max(mb.sum(),1):>9.1f}%")
    print(f"{'平均':<8}{nc[ma].mean():>10.2f}{'':>10}{nc[mb].mean():>10.2f}")

    # ── ② 的中した日 ────────────────────────────────────
    print("\n■ ★**②複勝が1本以上当たった日**")
    for nm, m in [("(a) 判定が走った日", ma), ("(b) 開催日ぜんぶ", mb)]:
        w = (nh > 0) & m
        print(f"　{nm:<20}{w.sum():>6,}日 / {m.sum():>6,}日"
              f"　= ★**{100*w.sum()/max(m.sum(),1):>5.1f}%**")

    # ── ③ 連続で当たらない開催日数 ───────────────────────────
    print("\n■ ★★**③連続で当たらない開催日数**（⚠**端の連は打ち切られているので落とす**）")
    for nm, m in [("(a) 判定が走った日", ma), ("(b) 開催日ぜんぶ", mb)]:
        print(f"　{nm:<20}{describe(runs_of_misses((nh > 0)[m]))}")

    # ── ★★ゲート2: 的中を日に無作為に散らした対照 ────────────────
    print("\n■ ★★**ゲート2の対照**——**的中が日にランダムに散っているだけなら、ここに一致する**")
    print(f"　⚠**日ごとの該当本数は固定したまま、「その日に当たったか」だけを"
          f"入れ替える**（**{NRAND}種**）")
    for nm, m in [("(a) 判定が走った日", ma), ("(b) 開催日ぜんぶ", mb)]:
        base = (nh > 0)[m]
        med, p95, mx = [], [], []
        for sd in range(NRAND):
            g = np.random.default_rng([SEED + sd, crc32(nm.encode())])
            r = runs_of_misses(g.permutation(base))
            if r:
                med.append(np.median(r)); p95.append(np.percentile(r, 95)); mx.append(max(r))
        if med:
            print(f"　{nm:<20}中央 {np.mean(med):>4.1f}日 / 95%点 {np.mean(p95):>4.1f}日 / "
                  f"最大 {np.mean(mx):>4.1f}日　(★{NRAND}種の平均)")

    print("\n★**読み方**: ★**実測の最大が対照の最大より明確に長ければ、"
          "当たりが日に固まっている**＝⚠**SNSの体感はこの数字より悪い**。")
    print("⚠**一致していれば、外れの連は単に的中率23.4%から出るばらつきである**"
          "（**判定基準43**）。")


if __name__ == "__main__":
    main()
