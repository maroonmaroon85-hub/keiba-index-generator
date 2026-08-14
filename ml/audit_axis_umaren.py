"""(140) ★★★(112)の「軸の複勝が甘い裾」を**(127)の新しい土台**で測り直す（2026-08-13）

★なぜやるか（判定基準15。(139)の直後）
　(112)は**このプロジェクト最大の単一のレバー**で、枠連の D を **+0.0182 → +0.0394（裾2%）**
　に倍増させた。だがそれは **q = λ補正Harville(単勝)→枠** の上での数字。
　**(127)で土台が変わった**（馬連プール→枠 のほうが +0.0093 良い）。**測り直す**。

★★★(139)が作った、検証できる含意
　(139)で **(117)の除外の上積みは +0.0040 → +0.0015（62%減）**になった。
　**(119)は「(112)の選別と(117)の選別は同じ信号だった」と実測している**。
　→ ★**同じ信号なら、(112)の利得も同じくらい縮むはず**。
　⚠**これは恒等式ではない**（(119)は実測であって保存則ではない）。判定基準27で踏んだばかりなので
　　**「含意」であって「当てにしてよい予想」ではない**と明記しておく。
　★**縮まなければ(119)の「同じ信号」のほうを疑うことになる**。**どちらに転んでも情報がある**。

★★事前登録（測る前に宣言する）
　1. **裾は 100/20/10/5/2%**（(111)(112)と同じ水準。**後から増やさない**）。
　　 選択基準は(112)と同一: `E = 複勝の払戻率0.8 ÷ 軸(1番人気)の3着以内確率 × 100円` の**小さい側**。
　　 **λ2/λ3 はウォークフォワード推定**（(112)と同じ）。**発走前に分かる量だけで作る**。
　2. **土台は2つ並べる**: `馬連→枠`（新）と `λHarville→枠`（旧）。
　　 **q_pool は枠連の板**。3つとも**同じ支持集合の上で正規化**（判定基準14①）。
　　 ⚠(112)は払戻ベースの d を使っていたが、ここは**板ベース**。(139)で
　　 　**板ベースの旧土台が(117)を完全に再現した**ので、比較可能とみなす。
　3. **★判定の本体は「裾 vs それ以外」の2標本検定**（判定基準13後半）。
　　 入れ子（裾のCIと全体のCI）を見比べない。
　4. ⚠**検出力を先に書く**: 裾2%は約620レース。(112)の実測CI幅は±0.025程度。
　　 → **±0.025より小さい変化は見えない**。**「縮んだ」を精密には言えない**可能性が高い。
　　 　 **だから点推定の大きさと単調性(ρ)も併せて報告する**。
　5. **プラセボ**: **E をレース間でシャッフル**して同じ裾を取る。**200回引いて平均**（判定基準13）。
　　 ★判定基準28の**不偏な部分標本**の形なので、**全体のDと同じ値で平坦になるはず**。
　　 　平坦でなければ実装を疑う。
　6. **★検算（判定基準27: 名乗らずに数値で見せる）**: 裾とそれ以外の**加重平均が全体に一致**すること
　　 （単純な2分割なので閉じるはず。(139)では厳密に閉じた）。
　7. **★運用判定**: 新しい土台で**裾2%の「裾 vs それ以外」の99%CI下端>0**なら
　　 **(112)の選択は土台を替えても立つ**。またぐなら**(139)と同じ結論**＝
　　 **「Harvilleの癖への対処だった疑いが強い」**。⚠どちらでも**必要量0.2549には遠い**。
　8. **予想**: ★**当てにしてよい予想は持っていない**と明記する（(138)の反省）。
　　 上の「(119)からの含意」は**実測に基づく含意であって恒等式ではない**。

実行: python3 ml/audit_axis_umaren.py [開始年(既定2015)]
"""
import math
import sys

import numpy as np

sys.path.insert(0, "ml")
from audit_capacity_d import mkt_waku_dist
from audit_cond_split import load_boards
from audit_crosspool import PAYBACK, load_races, probs, zq
from audit_crosspool2 import realized
from audit_lbs import build_matrix, fit_lambda
from audit_waku_vs_umaren import load_type
from audit_fuku_lbs import top3_probs   # ★(112)が使ったのはこちら（soft_axis版は引数が違う）
from waku_umatan import waku_of

TAILS = [1.00, 0.20, 0.10, 0.05, 0.02]     # ★(111)(112)と同じ。後から増やさない
NPLA = 200
RNG = np.random.default_rng(20260813)


def mci(x, alpha=0.01):
    x = np.asarray(x, float)
    if len(x) < 2:
        return float("nan"), float("nan"), float("nan")
    m = x.mean()
    se = x.std(ddof=1) / math.sqrt(len(x))
    return m, m - zq(alpha) * se, m + zq(alpha) * se


def two_sample(a, b, alpha=0.01):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan"), float("nan")
    d = a.mean() - b.mean()
    se = math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return d, d - zq(alpha) * se, d + zq(alpha) * se


def main():
    y0 = int(sys.argv[1]) if len(sys.argv) > 1 else 2015
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
        if not W or not U:
            continue
        rl = realized(r)
        if rl is None:
            continue
        a, b, _ = rl
        n = r["n"]
        nums = [u for u, _, _ in r["horses"]]
        if a not in nums or b not in nums:
            continue
        p = probs(r["horses"])
        l2, l3 = lam[yy]
        # ★(112)と同じ選択基準（発走前に分かる。オッズだけ）
        t3 = top3_probs(p, 1.0, l2, l3)
        q_axis = float(t3[int(np.argmax(p))])
        if not (0 < q_axis < 1):
            continue
        e_axis = PAYBACK["複勝"] / q_axis * 100

        md = mkt_waku_dist(r, p, l2)
        if not md:
            continue
        agg = {}
        for (x, y), o in U.items():
            if o <= 0:
                continue
            wx, wy = sorted((waku_of(x, n), waku_of(y, n)))
            agg[(wx, wy)] = agg.get((wx, wy), 0.0) + 1.0 / o
        key = tuple(sorted((waku_of(a, n), waku_of(b, n))))
        keys = [k for k in sorted(md) if k in W and k in agg]
        if key not in keys or len(keys) < 3:
            continue
        lw = math.log((1.0 / W[key]) / sum(1.0 / W[k] for k in keys))
        du = math.log(agg[key] / sum(agg[k] for k in keys)) - lw
        dh = math.log(md[key] / sum(md[k] for k in keys)) - lw
        rows.append((yy, e_axis, du, dh))

    ys = np.array([x[0] for x in rows])
    E = np.array([x[1] for x in rows])
    DU = np.array([x[2] for x in rows])
    DH = np.array([x[3] for x in rows])
    n = len(rows)
    nyr = len(set(ys.tolist()))
    print(f"(140) (112)の「軸の複勝が甘い裾」を(127)の新しい土台で測り直す"
          f"（{y0}年以降・{n:,}レース・{nyr}年）")
    print("★選択基準は(112)と同一（E＝複勝の払戻率0.8÷軸の3着以内確率×100円 の小さい側）")
    print("⚠検出力を先に書いた: 裾2%は約620本でCI幅±0.025程度。それより小さい変化は見えない\n")

    for tag, D in (("★★馬連→枠（(127)の新しい土台）", DU),
                   ("λHarville→枠（(112)が使った従来の土台）", DH)):
        print(f"■ {tag}")
        print(f"{'裾':>6}{'R数':>8}{'年間R':>7}{'裾のD':>11}{'それ以外':>11}"
              f"{'★差':>11}{'99%CI':>22}{'必要量比':>10}{'プラセボ':>11}")
        ds = []
        for t in TAILS:
            if t >= 1.0:
                print(f"{t:>5.0%}{n:>8,}{n/nyr:>7.0f}{D.mean():>+11.4f}{'—':>11}{'—':>11}"
                      f"{'—':>22}{D.mean()/0.2549:>9.1%}{D.mean():>+11.4f}")
                continue
            th = np.quantile(E, t)
            sel = E <= th
            m = D[sel].mean()
            d, lo, hi = two_sample(D[sel], D[~sel])
            ds.append(d)
            pl = []
            for _ in range(NPLA):
                idx = RNG.permutation(n)[: int(sel.sum())]
                pl.append(D[idx].mean())
            mark = "★" if lo > 0 else ("★負" if hi < 0 else "")
            ci = "[" + format(lo, "+.4f") + "," + format(hi, "+.4f") + "]"
            print(f"{t:>5.0%}{sel.sum():>8,}{sel.sum()/nyr:>7.0f}{m:>+11.4f}{D[~sel].mean():>+11.4f}"
                  f"{d:>+11.4f}{ci:>22}{m/0.2549:>9.1%}{np.mean(pl):>+11.4f} {mark}")
        rho = np.corrcoef([t for t in TAILS[1:]], ds)[0, 1]
        print(f"  → 単調性 ρ={rho:+.3f}（**負なら裾を詰めるほど良い**＝(112)は −0.900 だった）\n")

    # ── 事前登録6の検算（判定基準27: 数値で閉じることを見せる）──
    th = np.quantile(E, 0.02)
    sel = E <= th
    w = (sel.sum() * DU[sel].mean() + (~sel).sum() * DU[~sel].mean()) / n
    print("── ★検算（判定基準27）──")
    print(f"  裾2%・馬連→枠: 裾 {DU[sel].mean():+.5f}（{sel.sum():,}本）"
          f" / それ以外 {DU[~sel].mean():+.5f}（{(~sel).sum():,}本）")
    print(f"  加重平均 {w:+.6f}  vs  全体 {DU.mean():+.6f}"
          f"  → {'★閉じた' if abs(w - DU.mean()) < 1e-9 else '⚠閉じない'}")

    # ── (139)が作った含意の答え合わせ ──
    print("\n── ★★(139)の含意の答え合わせ（(119)『(112)と(117)は同じ信号』は成り立つか）──")
    gu = DU[sel].mean() - DU.mean()
    gh = DH[sel].mean() - DH.mean()
    print(f"  裾2%の上積み  λHarville→枠（旧）{gh:+.4f}  →  馬連→枠（新）{gu:+.4f}"
          f"   （{1 - gu/gh:.0%} 縮んだ）" if gh > 0 else "")
    print(f"  参考: (139)の除外40%の上積みは +0.0040 → +0.0015（62%減）")
    dd, lod, hid = two_sample(DU[sel] - DH[sel], DU[~sel] - DH[~sel])
    print(f"  ★「新−旧」が裾とそれ以外で違うか: {dd:+.4f} [{lod:+.4f},{hid:+.4f}]"
          f" {'★' if (lod > 0 or hid < 0) else '（違わない）'}")

    print("\n" + "=" * 104)
    print("★読み方（事前登録のとおり）")
    print("  ・裾2%で新しい土台の差の99%CI下端>0 → **(112)の選択は土台を替えても立つ**。")
    print("  ・またぐなら **(139)と同じ**＝Harvilleの癖への対処だった疑いが強い。")
    print("  ・⚠どちらに転んでも**必要量0.2549には遠い**。運用（買う/買わない）は変わらない。")
    print("  ・プラセボは全体のDと同じ値で平坦になるはず（判定基準28の不偏な部分標本）。")


if __name__ == "__main__":
    main()
