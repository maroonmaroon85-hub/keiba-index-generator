"""(135) ★★★枠連のD=+0.0182は「本命が絡むか」で稼いでいるのか、「本命が飛んだとき誰が来るか」で稼いでいるのか

★ユーザー発案（2026-08-12）。(134)への訂正から出た問い
　(134)で測ったのは「**荒れそうなレースを選ぶ**」＝レース選択で、答えは「向きが逆・(112)と同じ信号」。
　だがユーザーの意図は違った:
　> 荒れるレースで人気馬を買うわけじゃない。**荒れる前提で買える馬**を（見つけたい）
　★これは**レース選択ではなく、条件付き分布の構造**の話である。**まだ測っていない**。

★★分解（対数スコアの連鎖律・厳密に足してDになる）
　本命の枠 f＝単勝最少オッズ馬の枠。的中した枠組が f を含むかどうかを B とする。
```
　q(組) = Q_B · q̃(組|B)          Q_B = Σ_{その側の組} q,  q̃ = q/Q_B
　d = log q − log q_pool
　  = [log Q_B − log Q_pool,B]                     ← D_binary（本命が絡むかを当てる分）
　  + [log q̃ − log q̃_pool]                        ← D_cond （★誰が来るかを当てる分）
　D = E[d] = E[D_binary] + E[D_cond]   ★どちらも**全レースにわたる期待値**
```
　★**「本命が飛んだレースだけを取り出してDを測る」のとは違う**——それは結果で標本を選ぶので
　　(99)のバイアスを踏む。**ここでは指標関数で重みを付けているだけなので不偏**。
　★**D_cond をさらに B=1側 / B=0側 に分けて寄与を出す**（和は D_cond に一致する）。

★★事前登録（測る前に宣言する）
　1. 分解は上の2項のみ。**後から項を増やさない**。
　2. **q_pool は枠連の板(type=3)から全枠組で正規化**して作る（板が無いと Q_pool,B が作れない）。
　　 q は **λ補正Harville（ウォークフォワード）→ 枠**（＝(125)(129)(130)と同じ現行の道具）。
　3. **検算**: D_binary + D_cond が、板基準の全体D（既知の +0.0182 と同水準）に一致すること。
　　 **一致しなければ先に道具を疑う**（判定基準11）。
　4. **年分割で符号が揃うか**を両項について見る。
　5. **★運用が変わる条件を先に書く**: **D_cond が全体Dの半分以上（≥+0.0091）を占め、
　　 かつ99%CI下端が0を超え、かつ年分割で8/10年以上正**なら、
　　 「**本命が飛んだときの構造に情報がある**」＝**本命を外した買い目に意味がある**ことになり、
　　 **買い目の設計を見直す価値がある**。それ未満なら**記述にとどめ、運用は変えない**。
　6. **予想**: ★**D_binary が大半を占め、D_cond は小さい**と予想する（6:4〜8:2）。
　　 理由: (88)④で市場の較正ズレが構造として残ったのは**極端な本命の帯だけ**であり、
　　 　　　**人気順の較正は1〜8番人気すべて妙味なし**((88)⑤)。
　　 　　　＝**「誰が来るか」の順序構造に市場の穴は見つかっていない**。
　　 ⚠**ただし(88)は単勝の話で、枠連プールで同じとは限らない**。**そこが未知だから測る**。
　　 ⚠予想はあてにしない（この日 (131)で外している）。

実行: python3 ml/audit_cond_split.py [開始年(既定2015)]
"""
import math
import sys

import numpy as np

sys.path.insert(0, "ml")
from audit_capacity_d import mkt_waku_dist
from audit_crosspool import PAYBACK, load_races, probs, zq
from audit_crosspool2 import realized
from audit_lbs import build_matrix, fit_lambda
from nk_parse import nk_raceid
from waku_umatan import waku_of

NEED = -math.log(PAYBACK["枠連"])
D_ALL = 0.0182


def mci(x, alpha=0.01):
    x = np.asarray(x, float)
    if len(x) < 2:
        return float("nan"), float("nan"), float("nan")
    m = x.mean()
    se = x.std(ddof=1) / math.sqrt(len(x))
    z = zq(alpha)
    return m, m - z * se, m + z * se


def load_boards():
    from nk_odds_bulk import iter_records
    out = {}
    for rec in iter_records(3):
        r8 = nk_raceid(rec["race_id"])
        if not r8:
            continue
        d = {}
        for k, v in rec["odds"].items():
            if len(k) != 4 or not k.isdigit():
                continue
            o = v[0] if isinstance(v, (list, tuple)) else v
            if o and o > 0:
                d[(int(k[:2]), int(k[2:]))] = float(o)
        if d:
            out[r8] = d
    return out


def main():
    y0 = int(sys.argv[1]) if len(sys.argv) > 1 else 2015
    races = load_races()
    boards = load_boards()
    print(f"レース {len(races)} / 枠連の板 {len(boards)}", flush=True)

    P, i1, i2, i3, yrs = build_matrix(races, y0)
    lam = {}
    for yy in sorted(set(yrs.tolist())):
        tr = yrs < yy
        if tr.sum() < 3000:
            lam[yy] = None
            continue
        ok3 = tr & (i3 >= 0)
        lam[yy] = (fit_lambda(P[tr], i1[tr], i2[tr]),
                   fit_lambda(P[ok3], i1[ok3], i2[ok3], stage3=True, ic=i3[ok3]))

    rows = []
    for r in races:
        yy = r["year"]
        if yy < y0 or not lam.get(yy) or not r["wakuren"]:
            continue
        bd = boards.get(r["rid"])
        if not bd:
            continue
        rl = realized(r)
        if rl is None:
            continue
        a, b, _ = rl
        num2k = {u: k for k, (u, _, _) in enumerate(r["horses"])}
        if a not in num2k or b not in num2k:
            continue
        p = probs(r["horses"])
        l2, _l3 = lam[yy]
        md = mkt_waku_dist(r, p, l2)
        if not md:
            continue
        keys = [t for t in sorted(md) if t in bd]
        if len(keys) < 3:
            continue
        key = tuple(sorted((waku_of(a, r["n"]), waku_of(b, r["n"]))))
        if key not in keys:
            continue
        # 同じ支持集合の上で両方を正規化する（判定基準14②）
        inv = np.array([1.0 / bd[t] for t in keys])
        qp = inv / inv.sum()                       # 板の分布 q_pool
        qm = np.array([md[t] for t in keys])
        qm = qm / qm.sum()                         # λ補正Harville→枠 の分布 q
        # 本命の枠
        fav = int(np.argmax(p))
        wf = waku_of(r["horses"][fav][0], r["n"])
        inB = np.array([wf in t for t in keys])
        if inB.all() or not inB.any():
            continue                               # 片側が空だと分解が定義できない
        i = keys.index(key)
        side = bool(inB[i])
        Qm = qm[inB].sum() if side else qm[~inB].sum()
        Qp = qp[inB].sum() if side else qp[~inB].sum()
        if Qm <= 0 or Qp <= 0:
            continue
        d_bin = math.log(Qm) - math.log(Qp)
        d_cond = (math.log(qm[i] / Qm)) - (math.log(qp[i] / Qp))
        rows.append((yy, d_bin, d_cond, 1.0 if side else 0.0))

    arr = np.array(rows, float)
    yy, db, dc, side = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
    tot = db + dc
    print(f"\n対象 {len(rows)} レース（{int(yy.min())}〜{int(yy.max())}）")
    print(f"本命の枠が的中組に入った割合 {100*side.mean():.1f}%")

    m, lo, hi = mci(tot)
    print(f"\n★検算: D_binary + D_cond = {m:+.4f} 99%CI[{lo:+.4f},{hi:+.4f}]"
          f"  ※板基準の全体Dは +0.0182 と同水準のはず")

    print("\n── ★分解（和は上の全体Dに一致する） ──")
    for nm, v in (("D_binary  本命が絡むかを当てる分", db),
                  ("D_cond    ★誰が来るかを当てる分", dc)):
        a2, l2, h2 = mci(v)
        star = "★" if l2 > 0 else ("−" if h2 < 0 else " ")
        print(f"  {nm:34s} {a2:+.4f}  99%CI[{l2:+.4f},{h2:+.4f}] {star}"
              f"  （全体の {100*a2/m:>5.1f}%）")

    print("\n── D_cond の内訳（和は D_cond に一致する） ──")
    for nm, msk in (("本命の枠が絡んだレース", side > 0.5),
                    ("★本命の枠が飛んだレース", side < 0.5)):
        v = dc * msk
        a2, l2, h2 = mci(v)
        print(f"  {nm:26s} 寄与 {a2:+.4f}  99%CI[{l2:+.4f},{h2:+.4f}]"
              f"  (n={int(msk.sum())})")

    print("\n── 年別 ──")
    pb = pc = 0
    ys = sorted(set(int(x) for x in yy))
    for y in ys:
        m2 = yy == y
        ab, _, _ = mci(db[m2])
        ac, lc, hc = mci(dc[m2])
        pb += ab > 0
        pc += ac > 0
        print(f"  {y}  n={int(m2.sum()):5d}  D_binary={ab:+.4f}   "
              f"D_cond={ac:+.4f} 99%CI[{lc:+.4f},{hc:+.4f}]")
    print(f"  → 正の年  D_binary {pb}/{len(ys)}   D_cond {pc}/{len(ys)}")

    print("\n── ★事前登録5: 運用が変わる条件に当てる ──")
    ac, lc, _ = mci(dc)
    ok = (ac >= 0.5 * m) and (lc > 0) and (pc >= 8)
    print(f"  条件: D_cond ≥ 全体の半分({0.5*m:+.4f}) かつ CI下端>0 かつ 年8/10以上")
    print(f"  実測: D_cond={ac:+.4f} / 下端={lc:+.4f} / 正の年={pc}")
    print("  → " + ("★条件を満たす。**本命を外した買い目に意味がある**"
                    if ok else "**条件を満たさない。記述にとどめ、運用は変えない**"))


if __name__ == "__main__":
    main()
