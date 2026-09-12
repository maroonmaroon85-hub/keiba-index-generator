"""(127) ★★★(89)⑥の留保を外す — **枠連D=+0.0182は本物か、Harvilleの誤差相殺か**

★これはプロジェクトで唯一プラスの数字の正体を問う実験
　枠連 D=+0.0182 は5券種で**唯一の正**。(89)⑥はこう留保していた:
　> D_枠連 > D_馬連 は「**枠連プールが甘い**」とも
　> 「**Harvilleは枠に集約すると誤差が相殺されて精度が上がる**」とも読める。
　> **この2つは現データで分離できない**（分離には枠連の発走前オッズが要る）。
　(96)③も「宿題3は**不要にならなかった**。留保を外すにはやはりオッズが要る」と書いている。

★★分離の仕掛け（ここが肝・枠連の板だけでは足りない）
　いまのDは `q = λ補正Harville(単勝)` を枠に集約して作っている。**Harvilleが入っている**ので、
　枠連プールに勝っても「プールが甘い」のか「Harvilleが枠で当たる」のか分からない。
　→ **馬連プール(type=4)を枠に集約すれば、Harvilleを一切使わない枠組分布が作れる**。
　　 馬連は馬のペアに値段を付けた**別の市場**なので、集約は単なる足し算（近似ゼロ）。
　　　`q_枠(a,b) = Σ_{i∈a, j∈b} q_馬連(i,j)`
　**馬連→枠 が 枠連プールに勝てば → 枠連プールが甘い（本物）**
　**勝てなければ → +0.0182 は Harville の誤差相殺だった**

★★事前登録（測る前に宣言）
　1. 比較は3つ: **(a) 馬連→枠 vs 枠連プール** / (b) λHarville(単勝)→枠 vs 枠連プール /
　　 (c) 馬連→枠 vs λHarville→枠。**後から増やさない**。
　2. **判定は(a)**。99%CI下端が0を超えたら「**枠連プールが甘いのは本物**」。
　　 0をまたぐ・負なら「**+0.0182 はHarvilleの誤差相殺だった**」＝(112)(114)(117)の土台が揺らぐ。
　3. **年分割で符号が揃うか**も見る。
　4. **道具の検算**: 的中組で「板×100 と実配当」を突き合わせる。(113)は0.14%だった。
　5. **予想**: **(a)は正に出る**（枠連プールは本物に甘い）。理由は(96)②——
　　 λ補正しても枠連だけが正で他は全部負であり、これは枠集約の副作用では説明しにくいから。
　　 ⚠今日3回予想を外している。**この予想もあてにしないこと**。
　6. ★**この実験は「儲かるか」ではなく「土台が本物か」を問う**。
　　 (a)が負なら、**(112)(114)(117)で積み上げた枠連の利得が全部道具の癖だった**ことになる。
　　 **その可能性を先に書いておく**。

必要なデータ: `--type 3`（枠連の板）と **`--type 4`（馬連の板）**の両方。
実行: python3 ml/audit_waku_vs_umaren.py [開始年(既定2015)]
"""
import math
import sys

import numpy as np

sys.path.insert(0, "ml")
from audit_capacity_d import mkt_waku_dist
from audit_crosspool import PAYBACK, load_races, payoff, probs, zq
from audit_crosspool2 import PAYKEY, realized
from audit_lbs import build_matrix, fit_lambda, q_of_lbs
from nk_parse import nk_raceid
from waku_umatan import waku_of


def mci(x, alpha=0.01):
    x = np.asarray(x, float)
    if len(x) < 2:
        return float("nan"), float("nan"), float("nan")
    m = x.mean()
    se = x.std(ddof=1) / math.sqrt(len(x))
    return m, m - zq(alpha) * se, m + zq(alpha) * se


def load_type(t, keylen):
    from nk_odds_bulk import iter_records
    out = {}
    for rec in iter_records(t):
        rid8 = nk_raceid(rec["race_id"])
        if not rid8:
            continue
        d = {}
        for k, v in rec["odds"].items():
            if len(k) != keylen or not k.isdigit():
                continue
            o = v[0] if isinstance(v, (list, tuple)) else v
            if o and o > 0:
                d[(int(k[:2]), int(k[2:]))] = float(o)
        if d:
            out[rid8] = d
    return out


def main():
    y0 = int(sys.argv[1]) if len(sys.argv) > 1 else 2015
    wb = load_type(3, 4)          # 枠連の板  {(枠a,枠b): odds}
    ub = load_type(4, 4)          # 馬連の板  {(馬a,馬b): odds}
    if not wb or not ub:
        sys.exit("枠連(--type 3)と馬連(--type 4)の板が両方要る。"
                 "Macで `python3 ml/nk_odds_bulk.py --type 3` と `--type 4` を回して push すること。")
    races = load_races()
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

    rows, bad, hits = [], 0, 0
    for r in races:
        yy = r["year"]
        if yy < y0 or not lam.get(yy) or not r["wakuren"]:
            continue
        W, U = wb.get(r["rid"]), ub.get(r["rid"])
        if not W or not U:
            continue
        rl = realized(r)
        if rl is None:
            continue
        a, b, c = rl
        num2k = {u: k for k, (u, _, _) in enumerate(r["horses"])}
        if a not in num2k or b not in num2k:
            continue
        v = payoff(r, PAYKEY["枠連"], sorted((waku_of(a, r["n"]), waku_of(b, r["n"]))))
        if not v or v <= 0:
            continue
        key = tuple(sorted((waku_of(a, r["n"]), waku_of(b, r["n"]))))
        if key not in W:
            continue
        hits += 1
        if abs(W[key] * 100 - v) > max(10.0, v * 0.01):
            bad += 1

        # ★馬連プールを枠に集約する（Harvilleを一切使わない）
        agg = {}
        for (x, y), o in U.items():
            if o <= 0:
                continue
            wx, wy = sorted((waku_of(x, r["n"]), waku_of(y, r["n"])))
            agg[(wx, wy)] = agg.get((wx, wy), 0.0) + 1.0 / o
        # ⚠馬連には**同枠同士のゾロ目**も自然に含まれる（同じ枠の2頭のペア）。集約で正しく入る。
        p = probs(r["horses"])
        l2, l3 = lam[yy]
        md = mkt_waku_dist(r, p, l2)
        if not md:
            continue
        keys = [k for k in sorted(md) if k in W and k in agg]
        if key not in keys or len(keys) < 3:
            continue
        sW = sum(1.0 / W[k] for k in keys)
        sU = sum(agg[k] for k in keys)
        sH = sum(md[k] for k in keys)
        j = keys.index(key)
        lw = math.log((1.0 / W[key]) / sW)      # 枠連プール
        lu = math.log(agg[key] / sU)            # 馬連→枠（Harvilleなし）
        lh = math.log(md[key] / sH)             # λHarville(単勝)→枠
        rows.append((yy, lu - lw, lh - lw, lu - lh))

    if not rows:
        sys.exit("突き合わせできたレースが無い")
    arr = np.array([[x[1], x[2], x[3]] for x in rows])
    ys = np.array([x[0] for x in rows])
    print(f"(127) 枠連D=+0.0182 は本物か、Harvilleの誤差相殺か（{y0}年以降・{len(rows):,}レース）")
    print(f"★道具の検算: 的中 {hits:,} 件のうち 板×100 と実配当がずれたもの **{bad}**"
          f"（{bad/max(hits,1):.2%}）  ※(113)は0.14%\n")

    labs = ["(a) ★馬連→枠 − 枠連プール（Harvilleなし。**これが判定**）",
            "(b) λHarville→枠 − 枠連プール（従来のD）",
            "(c) 馬連→枠 − λHarville→枠"]
    for i, lab in enumerate(labs):
        m, lo, hi = mci(arr[:, i])
        mark = "★正" if lo > 0 else ("★負" if hi < 0 else "")
        print(f"{lab}\n    {m:+.4f}  99%CI [{lo:+.4f},{hi:+.4f}]  {mark}")
    print("\n★年分割（(a)）")
    pos = 0
    yl = sorted(set(ys.tolist()))
    for yy in yl:
        mask = ys == yy
        if mask.sum() < 100:
            continue
        mm = arr[mask, 0].mean()
        pos += mm > 0
        print(f"   {yy}  {int(mask.sum()):>5}本  {mm:+.4f}")
    print(f"   → {pos}/{len(yl)} 年で正")

    print("\n" + "=" * 92)
    print("★読み方（事前登録のとおり）")
    print("  ・(a)が★正 → **枠連プールが甘いのは本物**。(112)(114)(117)の土台は健全。")
    print("  ・(a)が0をまたぐ/負 → **+0.0182 はHarvilleの誤差相殺だった**。")
    print("    その場合、枠連で積み上げた利得は**全部道具の癖**だったことになる。")
    print("  ・(c)は「馬連プールと単勝プールのどちらが枠を当てるか」。副次的な記述。")


if __name__ == "__main__":
    main()
