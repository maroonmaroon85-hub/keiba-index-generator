"""(139) ★★★運用の「枠連スコア下位N%除外」を**(127)の新しい土台**で測り直す（2026-08-13）

★なぜ今やるか（判定基準15そのもの）
　(117)は「除外は本物。40%まで増やすほうがDは高い」と結論し、**運用を20%→40%に変えた**。
　だがその測定は **q = λ補正Harville(単勝)→枠** の上で行われている。
　**(127)で土台が変わった**——**馬連プール→枠** のほうが **+0.0093 良いq**（D +0.0182→+0.0266）。
　→ **道具を新しくしたら比較・層別の結論は測り直す**。**除外は比較そのもの**なので危ない側。

★★★本当の争点（ここが(139)の肝）
　**除外が拾っていたものと、馬連プールが知っていたものが同じかもしれない**。
　(119)で**(112)の選別と(117)の選別が同じ信号だった**という前例がある。
　枠連スコアはモデルの枠別確率から作る量で、**Harvilleの弱点を避ける方向に効いていた可能性**がある。
　もしそうなら、**Harvilleを使わない土台では除外の上積みは消える**。
　→ ★**消えたら「除外は道具の癖への対処だった」＝運用を戻す判断が要る**。
　　 **残ったら「除外は市場の甘さを拾っている」＝(117)の結論は土台を替えても立つ**。

★★事前登録（測る前に宣言する）
　1. **除外率の梯子 0/10/20/30/40/50%**（(117)と同じ。**後から水準を増やさない**）。
　2. **土台は2つ並べる**: `λHarville→枠`（従来）と `馬連→枠`（(127)の新しい土台）。
　　 **どちらも枠連の板を q_pool として同じ支持集合の上で正規化**する（判定基準14①）。
　3. **★判定の本体は「外した側 vs 残した側」の2標本検定**（判定基準13後半:
　　 **入れ子の部分集合どうしのCIを見比べても有意性は判定できない**）。
　4. **プラセボ**: **同数を無作為に除外**して同じ梯子を引く。**200回引いて平均**（判定基準13）。
　　 ★これは判定基準28で言う**不偏な部分標本の形**なので**有効なプラセボ**（構造上0にならず、
　　 　**全体のDを再現するはず**）。**平坦にならなければ実装を疑う**。
　5. **★運用判定**: 新しい土台で **40%除外の「外した側 vs 残した側」の差が
　　 99%CI下端>0 かつ 年8/11以上**なら**除外を続ける**。
　　 **満たさなければ、除外はHarvilleの癖への対処だったことになり、運用を見直す**。
　6. **★★恒等式ではないので検算に使わない**（判定基準27で踏んだばかり）。
　　 「残した側のD」と「全体のD」と「外した側のD」の関係は**加重平均で閉じる**——
　　 ⚠**これは群の割合が同じ2群への分割なので今度こそ閉じるはずだが、必ず数値で確かめる**。
　　 　（(138)が閉じなかったのは**層と部分集合が交差していた**から。ここは**単純な2分割**。）
　7. **予想（判定基準24に従い根拠の種類を明記）**
　　 ・**［類推・あてにしない］除外の効果は残るが小さくなる**。根拠は「馬連→枠のほうが良いqなので
　　 　 直す余地が減っているはず」という**機構の話**＝(137)で外したのと同じ種類。**当てにしない**。
　　 ・**［当てにしてよい根拠は無い］**。(138)で「恒等式」と自称して外したので、
　　 　 **今回は当てにしてよい予想を持っていないと明記する**。

実行: python3 ml/audit_excl_umaren.py [開始年(既定2015)]
"""
import glob
import math
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "ml")
from audit_capacity_d import mkt_waku_dist
from audit_cond_split import load_boards
from audit_crosspool import load_races, probs, zq
from audit_crosspool2 import realized
from audit_lbs import build_matrix, fit_lambda
from audit_waku_vs_umaren import load_type
from waku_umatan import bracket_probs, waku_of, waku_score, wakuren_buy

MODEL_CACHE = "data/cache/exp_L2-top3_2015"
CUTS = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50]
NPLA = 200                       # ★判定基準13: プラセボは反復して平均する
RNG = np.random.default_rng(20260813)


def two_sample(a, b, alpha=0.01):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan"), float("nan")
    d = a.mean() - b.mean()
    se = math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return d, d - zq(alpha) * se, d + zq(alpha) * se


def load_model_p():
    fs = sorted(glob.glob(f"{MODEL_CACHE}/*.csv"))
    if not fs:
        sys.exit(f"{MODEL_CACHE} が無い。")
    out = {}
    for f in fs:
        for rid, u, p in pd.read_csv(f)[["raceid", "umaban", "p"]].itertuples(index=False):
            out.setdefault(str(rid), {})[int(u)] = float(p)
    return out


def main():
    y0 = int(sys.argv[1]) if len(sys.argv) > 1 else 2015
    mp = load_model_p()
    races = load_races()
    wb, ub = load_boards(), load_type(4, 4)
    if not wb or not ub:
        sys.exit("枠連(--type 3)と馬連(--type 4)の板が両方要る。")

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
        W, U = wb.get(r["rid"]), ub.get(r["rid"])
        mm = mp.get(str(r["rid"]))
        if not W or not U or not mm:
            continue
        rl = realized(r)
        if rl is None:
            continue
        a, b, _ = rl
        n = r["n"]
        nums = [u for u, _, _ in r["horses"]]
        if a not in nums or b not in nums or any(u not in mm for u in nums):
            continue
        pm = np.array([mm[u] for u in nums], float)
        if pm.sum() <= 0:
            continue
        order = sorted(nums, key=lambda u: -mm[u])
        key = tuple(sorted((waku_of(a, n), waku_of(b, n))))
        if key not in W:
            continue
        p = probs(r["horses"])
        l2, _ = lam[yy]
        md = mkt_waku_dist(r, p, l2)
        if not md:
            continue
        agg = {}
        for (x, y), o in U.items():
            if o <= 0:
                continue
            wx, wy = sorted((waku_of(x, n), waku_of(y, n)))
            agg[(wx, wy)] = agg.get((wx, wy), 0.0) + 1.0 / o
        keys = [k for k in sorted(md) if k in W and k in agg]
        if key not in keys or len(keys) < 3:
            continue
        sW = sum(1.0 / W[k] for k in keys)
        lw = math.log((1.0 / W[key]) / sW)
        du = math.log(agg[key] / sum(agg[k] for k in keys)) - lw     # 馬連→枠
        dh = math.log(md[key] / sum(md[k] for k in keys)) - lw       # λHarville→枠
        sc = waku_score(wakuren_buy(order, n, 2), bracket_probs(nums, pm, n))
        rows.append((yy, sc, du, dh))

    ys = np.array([x[0] for x in rows])
    SC = np.array([x[1] for x in rows])
    DU = np.array([x[2] for x in rows])
    DH = np.array([x[3] for x in rows])
    n = len(rows)
    print(f"(139) 「枠連スコア下位N%除外」を(127)の新しい土台で測り直す（{y0}年以降・{n:,}レース）")
    print("★運用が毎週これでレースを外している（(117)で20%→40%に変えた）\n")

    for tag, D in (("★★馬連→枠（(127)の新しい土台）", DU), ("λHarville→枠（(117)が使った従来の土台）", DH)):
        print(f"■ {tag}")
        print(f"{'除外率':>7}{'残R数':>9}{'残したD':>11}{'外したD':>11}"
              f"{'★差(残−外)':>13}{'99%CI':>22}{'年':>7}{'プラセボ':>11}")
        for c in CUTS:
            if c == 0:
                print(f"{c:>6.0%}{n:>9,}{D.mean():>+11.4f}{'—':>11}{'—':>13}{'—':>22}"
                      f"{'—':>7}{D.mean():>+11.4f}")
                continue
            th = np.quantile(SC, c)
            keep, drop = SC >= th, SC < th
            d, lo, hi = two_sample(D[keep], D[drop])
            yl = sorted(set(ys.tolist()))
            pos = sum(1 for yy in yl
                      if (ys == yy).sum() >= 100 and (drop & (ys == yy)).sum() >= 10
                      and (D[keep & (ys == yy)].mean() - D[drop & (ys == yy)].mean()) > 0)
            pl = []
            for _ in range(NPLA):
                idx = RNG.permutation(n)[: int(keep.sum())]
                pl.append(D[idx].mean())
            mark = "★" if lo > 0 else ("★負" if hi < 0 else "")
            ci = "[" + format(lo, "+.4f") + "," + format(hi, "+.4f") + "]"
            print(f"{c:>6.0%}{keep.sum():>9,}{D[keep].mean():>+11.4f}{D[drop].mean():>+11.4f}"
                  f"{d:>+13.4f}{ci:>22}{pos:>4}/{len(yl)}{np.mean(pl):>+11.4f} {mark}")
        print()

    # ── 事前登録6: 今度こそ加重平均が閉じるか、数値で確かめる（判定基準27）──
    th = np.quantile(SC, 0.40)
    keep, drop = SC >= th, SC < th
    w = (keep.sum() * DU[keep].mean() + drop.sum() * DU[drop].mean()) / n
    print("── ★事前登録6の検算（判定基準27: 恒等式は名乗るのではなく数値で閉じることを見せる）──")
    print(f"  40%除外・馬連→枠:  残した側 {DU[keep].mean():+.5f}（{keep.sum():,}本）"
          f" / 外した側 {DU[drop].mean():+.5f}（{drop.sum():,}本）")
    print(f"  加重平均 {w:+.5f}  vs  全体 {DU.mean():+.5f}"
          f"   → {'★閉じた（単純な2分割なので今度は恒等式）' if abs(w - DU.mean()) < 1e-9 else '⚠閉じない'}")

    # ── 新旧の土台で「除外の上積み」を直接比べる ──
    print("\n── ★★除外の上積みは土台を替えても残るか（40%除外・対応あり）──")
    du_gain = DU[keep].mean() - DU.mean()
    dh_gain = DH[keep].mean() - DH.mean()
    print(f"  馬連→枠（新）      全体 {DU.mean():+.4f} → 残した側 {DU[keep].mean():+.4f}"
          f"   上積み {du_gain:+.4f}")
    print(f"  λHarville→枠（旧） 全体 {DH.mean():+.4f} → 残した側 {DH[keep].mean():+.4f}"
          f"   上積み {dh_gain:+.4f}")
    dd, lod, hid = two_sample(DU[keep] - DH[keep], DU[drop] - DH[drop])
    print(f"  ★「新−旧」の差が残した側と外した側で違うか: {dd:+.4f} [{lod:+.4f},{hid:+.4f}]"
          f" {'★' if (lod > 0 or hid < 0) else '（違わない）'}")

    print("\n" + "=" * 100)
    print("★読み方（事前登録のとおり）")
    print("  ・40%除外で差の99%CI下端>0 かつ 年8/11以上 → **除外は続ける**。")
    print("  ・満たさなければ **除外はHarvilleの癖への対処だった**ことになり、運用を見直す。")
    print("  ・プラセボ（無作為に同数除外）は**全体のDと同じ値で平坦になるはず**。")
    print("    平坦でなければ実装を疑う（判定基準28の形＝不偏な部分標本）。")


if __name__ == "__main__":
    main()
