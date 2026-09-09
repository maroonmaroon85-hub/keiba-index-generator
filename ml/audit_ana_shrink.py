"""(235) ★★★★**該当が354→36に減ったのは、どの条件のせいか** —— 3条件を年別に分解

★★**動機（2026-09-09・利用者の問い）**: ★**「穴馬の数自体が年々減っている？」**

★**(234)で、軸の該当が 354本(2016) → 36本(2026) に減っていると出た**。
★**複勝板のあるレースは年3,100前後で一定**なので、⚠**データの問題ではない**。
★★**軸は3つの条件の合成なので、どれが減ったのかを分ければ原因が分かる**:
```
軸 = pn ≥ 0.15  かつ  ズレ ≥ 0.15  かつ  単勝オッズ ≥ 10.0倍
```
| もし減っているのが | ★**意味** |
|---|---|
| ★**単勝≥10倍の馬** | ★**「穴馬そのものが減った」**（**出走頭数や人気の分布が変わった**） |
| ★**pn≥0.15の馬** | ★**モデルが確率を低く出すようになった**（**校正が変わった**） |
| ★★**ズレ≥0.15の馬** | ★★**モデルが市場に近づいた**（**(234)で立てた仮説**） |

────────────────────────────────────────────────────────────
★★★ 事前登録（2026-09-09・**結果を見る前にコミットする**）
────────────────────────────────────────────────────────────

■ ★**測るもの（すべて年別・1レースあたりの頭数）**
　**① 単勝≥10倍 ／ ② pn≥0.15 ／ ③ ズレ≥0.15 ／ ④ ①かつ② ／ ⑤ ①かつ③ ／
　　⑥ ②かつ③ ／ ★⑦ 3つすべて（＝軸の候補）**
　★**あわせて 平均出走頭数・ズレの平均と分位点・pnの平均**も出す。

■ ★★ゲート2（判定基準42）——**この測定は何を返せば「原因が分からない」か**
　★**どの条件も横ばいなら、3つとも平らな線を返す**——**そのとき該当の減少は説明できず、
　　★別の原因（対象レースの構成が変わった等）を探すことになる**。
■ ★★★内部対照（**決定的**）: ★**⑦の年別合計が (234) の該当本数と一致すること**
　（**354 / 231 / 218 / 159 / 96 / 87 / 62 / 43 / 56 / 41 / 36**）。⚠**ずれたら読まない**。

■ ⚠**この測定は探索ではない**。★**マスを増やさない。買い目も基準も変えない**。
　★**軸の定義は動かさない**。**ただ年別に分けて数えるだけ**。

■ 予想（⚠**当てにしない**・★**私は10回外した**）
　★**減っているのは③ズレだと見る**（**(234)で立てた仮説**）。
　⚠**もし①（単勝≥10倍の馬）が減っているなら、それは「穴馬そのものが減った」**——
　　★**そのときは仮説が外れで、原因は市場側にある**。

実行: python3 ml/audit_ana_shrink.py    自己テスト: python3 ml/audit_ana_shrink.py --selftest
"""
import sys

import numpy as np

sys.path.insert(0, "ml")
import features as F
from audit_crosspool import load_races
from audit_ana_odds import MIN_HORSES
from audit_ana_board import NPLACE, load_fuku_boards, qpool
from audit_ana_band import PN_FLOOR
from audit_ana_fix import LFIX
from audit_ana_hole import GAP
from audit_ana_marg import wf_predict
from train_prod import add_odds_features

# ⚠★★訂正（2026-09-09・実行後・結果を読む前）: **初版はここに (234) の年別
#   [354,231,...,36]（合計1,383）を置いて対照が落ちた**。★**1,383は「±20%の乱が
#   軸＋紐3頭ぶん引けたレース」だけ**で、★**この測定は乱を使わないので全数1,398が正しい**。
#   ⚠**昨日 ANA_RULE.md に「1,398/1,383/1,356の3つの数」の注記を書いた翌日に、同じ穴に落ちた**
#   （★**判定基準37の7回目**）。★**対照は「合計1,398」＋「各年が(234)を下回らない」に直す**。
KNOWN_LOW = [354, 231, 218, 159, 96, 87, 62, 43, 56, 41, 36]   # ★(234)＝乱の条件つき
KNOWN_SUM = 1398                                               # ★★軸が立った全数


def selftest():
    ok = True
    print(f"★軸の3条件: **pn ≥ {PN_FLOOR}** かつ **ズレ ≥ {GAP}** かつ **単勝 ≥ {LFIX}倍**")
    print("★測るのは年別・1レースあたりの頭数（①〜⑦）＋ 平均頭数・ズレの分位点")
    print(f"★★★内部対照（★訂正版）: **⑦の年別合計が {KNOWN_SUM:,}（軸が立った全数）**"
          f"　かつ **各年が(234)の値を下回らない**")
    print("　★(234)の年別（乱の条件つき・合計1,383）: "
          + " / ".join(str(x) for x in KNOWN_LOW))
    print(f"　⚠**差の15本は乱の抽選が失敗したレース**"
          f"　{'★OK' if KNOWN_SUM - sum(KNOWN_LOW) == 15 else '⚠NG'}")
    ok &= KNOWN_SUM - sum(KNOWN_LOW) == 15
    print("★★ゲート2: **どの条件も横ばいなら3本とも平らな線を返す**"
          "　→ ★**そのとき該当減少は説明できず、別の原因を探すことになる**")
    print("⚠**探索ではない。マスを増やさない。軸の定義も動かさない**")
    print("★自己テスト: " + ("全部OK" if ok else "⚠NG"))
    return 0 if ok else 1


def main():
    print("(235) ★★★★**該当が354→36に減ったのは、どの条件のせいか**")
    print("★利用者の問い: **「穴馬の数自体が年々減っている？」**\n")
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
    sub = d.loc[m, ["raceid", "umaban", "odds", "date"]].copy()
    sub["p"] = pred[m]

    acc = {}
    for rid, g in sub.groupby("raceid"):
        rid = str(rid)
        r, bd = races.get(rid), boards.get(rid)
        if r is None or bd is None:
            continue
        nums = {u for u, _, _ in r["horses"]}
        gg = g[g["umaban"].astype(int).isin(nums)]
        ub = gg["umaban"].astype(int).to_numpy()
        if len(gg) < MIN_HORSES or not all(int(u) in bd for u in ub):
            continue
        od = gg["odds"].to_numpy(float)
        pv = gg["p"].to_numpy(float)
        if not np.isfinite(od).all() or (od <= 0).any() or pv.sum() <= 0:
            continue
        yr = int(gg["date"].iloc[0].year)
        pn = pv / pv.sum() * NPLACE
        qp, _ = qpool([bd[int(u)] for u in ub], "harm")
        gap = pn - qp
        c1, c2, c3 = od >= LFIX, pn >= PN_FLOOR, gap >= GAP
        a = acc.setdefault(yr, {k: 0.0 for k in
                                ("R", "n", "1", "2", "3", "12", "13", "23", "7", "hit")})
        a["R"] += 1
        a["n"] += len(ub)
        a["1"] += c1.sum(); a["2"] += c2.sum(); a["3"] += c3.sum()
        a["12"] += (c1 & c2).sum(); a["13"] += (c1 & c3).sum(); a["23"] += (c2 & c3).sum()
        a["7"] += (c1 & c2 & c3).sum()
        a["hit"] += 1 if (c1 & c2 & c3).any() else 0
        a.setdefault("gap", []).append(float(np.max(gap)))
        a.setdefault("pn", []).append(float(np.max(pn)))

    ys = sorted(k for k in acc if k >= 2016)
    got = [int(acc[u]["hit"]) for u in ys]
    okc = (sum(got) == KNOWN_SUM
           and all(a >= b for a, b in zip(got, KNOWN_LOW)))
    print(f"★★★内部対照（★訂正版）")
    print(f"　実測 {got}　合計 {sum(got):,}（★{KNOWN_SUM:,}であること）")
    print(f"　(234) {KNOWN_LOW}　合計 {sum(KNOWN_LOW):,}（乱の条件つき）")
    print(f"　★各年が(234)以上 かつ 合計が{KNOWN_SUM:,}　"
          f"{'★★立った' if okc else '⚠⚠落ちた'}")
    if not okc:
        print("⚠⚠**対照が落ちた。読まない**（判定基準32）。")
        return

    print(f"\n{'':<26}" + "".join(f"{u:>8}" for u in ys))
    print(f"{'対象レース':<26}" + "".join(f"{int(acc[u]['R']):>8}" for u in ys))
    print(f"{'★平均出走頭数':<24}" + "".join(f"{acc[u]['n']/acc[u]['R']:>8.1f}" for u in ys))
    rows = [("★① 単勝≥10倍", "1"), ("② pn≥0.15", "2"), ("★③ ズレ≥0.15", "3"),
            ("④ ①かつ②", "12"), ("⑤ ①かつ③", "13"), ("⑥ ②かつ③", "23"),
            ("★★⑦ 3つすべて", "7")]
    print(f"\n■ ★**1レースあたりの頭数**")
    for lab, k in rows:
        print(f"{lab:<24}" + "".join(f"{acc[u][k]/acc[u]['R']:>8.3f}" for u in ys))
    print(f"\n■ ★**そのレースの最大値（平均）**")
    print(f"{'★ズレの最大':<25}" + "".join(f"{np.mean(acc[u]['gap']):>8.3f}" for u in ys))
    print(f"{'pnの最大':<26}" + "".join(f"{np.mean(acc[u]['pn']):>8.3f}" for u in ys))
    print(f"\n■ ★**2016→2026 の倍率**")
    a0, a1 = acc[ys[0]], acc[ys[-1]]
    for lab, k in rows:
        v0, v1 = a0[k] / a0["R"], a1[k] / a1["R"]
        print(f"{lab:<24}{v0:>8.3f} → {v1:>7.3f}"
              f"　★**{(v1/v0 if v0 else float('nan')):>5.2f}倍**")
    print(f"{'★ズレの最大':<25}{np.mean(a0['gap']):>8.3f} → {np.mean(a1['gap']):>7.3f}"
          f"　★**{np.mean(a1['gap'])/np.mean(a0['gap']):>5.2f}倍**")
    print("\n⚠**枠連の運用には触れない**。**設定変更は提案しない**。")


if __name__ == "__main__":
    sys.exit(selftest() if "--selftest" in sys.argv else (main() or 0))
