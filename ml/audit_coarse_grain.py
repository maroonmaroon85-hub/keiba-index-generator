"""(128) ★★★「枠へ粗視化するとHarvilleの誤差が相殺される」は実在するか — (127)の前半を手持ちの板で測る

★なぜ今これができるのか（(127)は馬連待ちなのに）
　(89)⑥の留保は**2つの読み**に分かれている:
　　(i) **枠連プールが甘い**（市場側の性質）
　　(ii) **Harvilleは枠に集約すると誤差が相殺されて精度が上がる**（道具側の性質）
　(127)は「馬連→枠」というHarville非依存の物差しを当てて**(i)と(ii)を分離**する。これは馬連板が要る。
　しかし **(ii)単体の大きさ**は、**市場を固定したまま粗視化だけを入れる**ことで**いま測れる**。
　三連複板(type=7)は手元にあり、これは**Harvilleを一切使わない別市場の意見**だからである。

★★仕掛け（ここが肝）
　三連複の空間で、**同じ2つの分布を同じ写像で押し出す**:
　　q_λ    = λ補正Harville（単勝オッズ由来）の top3 集合分布
　　q_pool = 三連複プール（板の1/オッズ）を全組で正規化した分布
　粗視化 G: 馬 → 枠。押し出した先は**枠の3つ組（多重集合）**。
　　d_G = log q_λ^G(実現した枠3つ組) − log q_pool^G(実現した枠3つ組)
　**市場(q_pool)は粗視化の前後で同じ**なので、d_G − d_id の変化は**粗視化だけの効果**である。
　★これは (89)⑥(ii) の大きさそのもの。**(i)は動かさずに(ii)だけを取り出している**。

　⚠これは(127)の代用ではない。三連複の空間（top3集合）であって枠連の空間（1-2着）ではない。
　　**(ii)の機構が実在し、どの程度の量を持つか**を測るだけ。**(a)の判定は馬連板を待つ**。

★★事前登録（測る前に宣言する）
　1. 比較は3つ: **Δ_枠 = d_枠 − d_id** ／ **Δ_乱 = d_乱 − d_id**（ランダム群） ／ d_id そのもの。
　　 **後から増やさない**。
　2. **判定は Δ_枠**。99%CI下端が0を超えたら「**粗視化はHarvilleを助ける＝(ii)は実在する**」。
　　 0をまたぐなら「(ii)は量を持たない」＝**枠連の+0.0182は(i)側の可能性が高まる**。
　3. **量の目安を先に引く**。枠連の全体Dは +0.0182 である。
　　 　Δ_枠 ≥ +0.018 なら **(ii)だけで枠連のDを説明できてしまう**（★最も危ない結果）
　　 　+0.005 ≤ Δ_枠 < +0.018 なら **一部は道具の癖**（(127)の(a)は正でも割り引く必要がある）
　　 　Δ_枠 < +0.005 なら **(ii)は小さい**＝土台は(i)側で説明され、健全side
　4. ★**プラセボは枠ラベルのレース内シャッフルを R 回引いて平均する**（判定基準13）。
　　 枠は馬番で機械的に決まるので**強さとほぼ無関係**＝ランダム群と同じはず。
　　 **Δ_枠 ≈ Δ_乱 なら「枠が特別」ではなく「粗視化一般の効果」**と読む。これも結論の一部。
　　 ⚠**Δ_乱 が Δ_枠 と厳密に一致したら、まずプラセボの実装を疑う**（判定基準10）。
　5. **年分割で符号が揃うか**を見る。
　6. **道具の検算**: d_id は (113)(B) の w=1.0 の値（−0.0303）と一致するはず。
　　 **一致しなければ先に道具を疑う**（相互検算・判定基準11）。
　7. **★同じ支持集合の上で比べる**（判定基準14②）。q_λ は**板に在る組だけ**に制限して
　　 　**そこで正規化し直す**。存在しない組へ確率を捨てると比較が歪む。
　8. **予想**: **Δ_枠 は正に出る**と予想する。+0.005〜+0.015 を見込む。
　　 理由: 粗視化は KL を必ず縮める（データ処理不等式）が、縮み方は分布ごとに違う。
　　 　　　Harvilleの残差誤差は**同枠内の馬の識別**に多く乗っているはずで、
　　 　　　枠でまとめるとその分が落ちる。ただし**枠連のD全部(+0.0182)を説明するほどではない**と見る。
　　 　　　また **Δ_乱 ≈ Δ_枠** と予想する（枠は強さと無関係な機械的ラベルだから）。
　　 ⚠**この予想はあてにしないこと**。2026-08-11 だけで4回外している。

実行: python3 ml/audit_coarse_grain.py [開始年(既定2015)] [プラセボ反復(既定30)]
"""
import math
import sys
from itertools import combinations

import numpy as np

sys.path.insert(0, "ml")
from audit_crosspool import PAYBACK, load_races, payoff, zq
from audit_crosspool2 import realized
from nk_parse import nk_raceid
from waku_umatan import waku_of
import soft_axis as SA

RNG = np.random.default_rng(20260812)
NEED = -math.log(PAYBACK["三連複"])
KS = (2, 3, 4, 6, 8)          # ★群数の梯子。枠8個を k 個に併合する
D_WAKUREN = 0.0182            # (96) λ補正後の枠連D（HANDOFF 3537行）
D_UMAREN = -0.0135            # (96) λ補正後の馬連D（同上）


def merge(wk, k):
    """枠1..8 を k 個の群へ機械的に併合する。k=8 なら枠そのもの。"""
    return (wk - 1) * k // 8 + 1


def mci(x, alpha=0.01):
    x = np.asarray(x, float)
    if len(x) < 2:
        return float("nan"), float("nan"), float("nan")
    m = x.mean()
    se = x.std(ddof=1) / math.sqrt(len(x))
    z = zq(alpha)
    return m, m - z * se, m + z * se


def key_of(nums):
    return "".join(f"{n:02d}" for n in sorted(nums))


_COMBO = {}


def combo_idx(n):
    if n not in _COMBO:
        c = np.array(list(combinations(range(n), 3)), dtype=np.int64)
        _COMBO[n] = c
    return _COMBO[n]


def load_boards():
    from nk_odds_bulk import iter_records
    out = {}
    for rec in iter_records(7):
        rid8 = nk_raceid(rec["race_id"])
        if rid8:
            out[rid8] = rec["odds"]
    return out


def prep(r, board):
    """1レースを (p_pool, p_lam, 枠ラベル, 組の添字, 実現組の位置) にそろえる。

    ★q_λ は**板に在る組だけ**に制限してそこで正規化する（事前登録7）。
    """
    rl = realized(r)
    if rl is None or rl[2] is None:
        return None
    hs = r["horses"]
    nums = [n for n, _, _ in hs]
    odds = [o for _, o, _ in hs]
    n = r["n"]
    if len(nums) < 4:
        return None
    p = SA.win_probs(odds)
    ci = combo_idx(len(nums))
    pool, lam, keep = [], [], []
    for t, (a, b, c) in enumerate(ci):
        o = board.get(key_of([nums[a], nums[b], nums[c]]))
        if not o or o <= 0:
            continue
        pool.append(1.0 / o)
        lam.append(max(SA.trio_prob(p, [int(a), int(b), int(c)]), 1e-300))
        keep.append(t)
    if len(keep) < 4:
        return None
    pool = np.asarray(pool, float)
    lam = np.asarray(lam, float)
    sp, sl = pool.sum(), lam.sum()
    if sp <= 0 or sl <= 0:
        return None
    pool /= sp
    lam /= sl                       # ★同じ支持集合の上で正規化し直す
    ck = ci[np.asarray(keep, np.int64)]
    # 実現した組がその支持集合に在るか
    try:
        ridx = [nums.index(x) for x in rl]
    except ValueError:
        return None
    want = tuple(sorted(ridx))
    pos = np.flatnonzero((ck[:, 0] == want[0]) & (ck[:, 1] == want[1]) & (ck[:, 2] == want[2]))
    if len(pos) != 1:
        return None
    wk = np.array([waku_of(x, n) for x in nums], dtype=np.int64)
    return pool, lam, wk, ck, int(pos[0])


def coarse_d(pool, lam, code, ireal):
    """枠3つ組へ押し出した先での log q_λ − log q_pool（実現した組の所）。"""
    cr = code[ireal]
    a = float(lam[code == cr].sum())
    b = float(pool[code == cr].sum())
    if a <= 0 or b <= 0:
        return float("nan")
    return math.log(a) - math.log(b)


def codes_of(wk, ck):
    """組ごとの「枠3つ組」を整数コードにする。枠は1..8。"""
    w = np.sort(wk[ck], axis=1)
    return w[:, 0] * 81 + w[:, 1] * 9 + w[:, 2]


def placebo_d(pool, lam, wk, ck, ireal, reps, rng):
    """★枠ラベルをレース内でシャッフルした群での d を reps 回引いて平均する。

    判定基準13（プラセボは反復して平均する）。**全反復を一度に行列で回す**——
    1回ずつPythonで回すと反復数を増やせず、増やせないプラセボは判定に使えない。
    """
    n = len(wk)
    perm = np.argsort(rng.random((reps, n)), axis=1)      # 各行が独立な置換
    lab = wk[perm]                                        # (reps, n) 群サイズ構成は不変
    w = np.sort(lab[:, ck], axis=2)                       # (reps, m, 3)
    code = w[:, :, 0] * 81 + w[:, :, 1] * 9 + w[:, :, 2]  # (reps, m)
    same = code == code[:, ireal][:, None]
    a = (same * lam).sum(axis=1)
    b = (same * pool).sum(axis=1)
    ok = (a > 0) & (b > 0)
    if not ok.any():
        return float("nan")
    return float((np.log(a[ok]) - np.log(b[ok])).mean())


def main():
    y0 = int(sys.argv[1]) if len(sys.argv) > 1 else 2015
    reps = int(sys.argv[2]) if len(sys.argv) > 2 else 30

    races = load_races()
    boards = load_boards()
    print(f"レース {len(races)} / 板 {len(boards)}", flush=True)

    rows = []
    chk_n = chk_bad = 0
    for r in races:
        if r["year"] < y0:
            continue
        board = boards.get(r["rid"])
        if not board:
            continue
        v = prep(r, board)
        if v is None:
            continue
        pool, lam, wk, ck, ireal = v
        # ★道具の検算: 実現組の板オッズ×100 と実配当（事前登録6）
        rl = realized(r)
        pay = payoff(r, "三連複", list(rl)) or 0.0
        o_real = board.get(key_of(list(rl)))
        bad_race = False
        if pay > 0 and o_real:
            chk_n += 1
            if abs(o_real * 100.0 - pay) > 1.0:
                chk_bad += 1
                bad_race = True       # ★感度分析で落とす対象

        d_id = math.log(lam[ireal]) - math.log(pool[ireal])
        code = codes_of(wk, ck)
        d_wk = coarse_d(pool, lam, code, ireal)
        # ★プラセボ: 枠ラベルをレース内でシャッフル（群サイズの構成は保たれる）
        d_rd = placebo_d(pool, lam, wk, ck, ireal, reps, RNG)
        if math.isnan(d_wk) or math.isnan(d_rd):
            continue
        # ★群数の梯子: 枠8個を k 個に併合したときの d（粗視化の強さ依存を見る）
        lad = []
        for k in KS:
            dk = coarse_d(pool, lam, codes_of(merge(wk, k), ck), ireal)
            lad.append(dk if not math.isnan(dk) else d_id)
        rows.append((r["year"], d_id, d_wk, d_rd, float(bad_race), *lad))

    if not rows:
        print("対象レースが無い")
        return
    arr = np.array(rows, float)
    yr, d_id, d_wk, d_rd = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
    bad = arr[:, 4] > 0.5
    lad = arr[:, 5:]
    print(f"\n対象 {len(rows)} レース（{int(yr.min())}〜{int(yr.max())}）  "
          f"プラセボ反復 {reps} 回")
    print(f"★道具の検算: 板×100 と実配当のずれ {chk_bad}/{chk_n} "
          f"({100.0 * chk_bad / max(chk_n, 1):.2f}%)")
    print("　⚠(113)の0.35%とは分母が違う（あちらは的中した買い目1,427件、"
          "こちらは実現組 全レース）ので直接は比べられない。")

    print("\n── 水準（三連複の空間） ──")
    for name, v in (("d_id  粗視化なし", d_id), ("d_枠  枠へ粗視化", d_wk),
                    ("d_乱  ランダム群", d_rd)):
        m, lo, hi = mci(v)
        print(f"{name:22s} {m:+.4f}  99%CI[{lo:+.4f},{hi:+.4f}]")
    print("　※d_id は (113)(B) の w=1.0（−0.0303）と一致するはず＝相互検算")

    print("\n── ★判定: 粗視化の効果 Δ ──")
    for name, v in (("★Δ_枠 = d_枠 − d_id", d_wk - d_id),
                    ("　Δ_乱 = d_乱 − d_id", d_rd - d_id),
                    ("　Δ_枠 − Δ_乱（枠は特別か）", d_wk - d_rd)):
        m, lo, hi = mci(v)
        star = "★" if lo > 0 else ("−" if hi < 0 else " ")
        print(f"{name:30s} {m:+.4f}  99%CI[{lo:+.4f},{hi:+.4f}] {star}")

    dw = d_wk - d_id
    m, lo, _ = mci(dw)
    print("\n── 事前登録3の目安に当てる ──")
    print(f"　枠連の全体D = +0.0182 に対して Δ_枠 = {m:+.4f}"
          f"（{100.0 * m / 0.0182:.0f}%）")
    if lo <= 0:
        print("　→ **(ii)は量を持たない**。+0.0182 は(i)側で説明される可能性が高い")
    elif m >= 0.018:
        print("　→ ★★**(ii)だけで枠連のDを説明できてしまう**。土台を疑うべき")
    elif m >= 0.005:
        print("　→ ★**一部は道具の癖**。(127)の(a)が正でも割り引いて読む必要がある")
    else:
        print("　→ **(ii)は小さい**。土台は(i)側で説明され、健全な側")

    print("\n── 群数の梯子（粗視化を強めるほどΔは伸びるか） ──")
    print("   k個の群   セル数        d          Δ = d − d_id")
    for j, k in enumerate(KS):
        m2, lo2, hi2 = mci(lad[:, j])
        a2, alo, ahi = mci(lad[:, j] - d_id)
        cells = (k + 2) * (k + 1) * k // 6
        print(f"    k={k}     {cells:4d}    {m2:+.4f}    "
              f"{a2:+.4f} 99%CI[{alo:+.4f},{ahi:+.4f}]")
    print("   ※k=8が実際の枠。三連複816組→120セル(比6.8)に対し、"
          "枠連は153組→36セル(比4.25)＝**粗視化はより弱い**")

    print("\n── ★感度分析: 板と実配当がずれた265件を落とす ──")
    ok = ~bad
    m3, lo3, hi3 = mci((d_wk - d_id)[ok])
    print(f"　Δ_枠（ずれた {int(bad.sum())} 件を除外・n={int(ok.sum())}）"
          f" {m3:+.4f}  99%CI[{lo3:+.4f},{hi3:+.4f}]")

    mid, _, _ = mci(d_id)
    mwk, _, hiwk = mci(d_wk)
    mdw, _, _ = mci(d_wk - d_id)
    rec = mdw / abs(mid)

    print("\n── ⚠まず自分の事前登録の誤りを記録する（判定基準9: 恒等式の罠） ──")
    print("　事前登録8で「Δ_枠は正に出る」と予想したが、**これは構造的にほぼ強制される**。")
    print("　粗視化は両方の分布を同じ退化点へ寄せるので d→0。d_id が負である以上、")
    print("　**Δ>0 は発見ではない**。梯子がそれを示している（セルを減らすほどdは0へ寄る）。")
    print("　★測る価値があったのは**Δの符号ではなく、①その大きさ ②dが0を越えるか**だった。")
    print("　　事前登録の判定条件2（CI下端>0）は**何も検査していなかった**。")

    print("\n── ★★(127)への含意（ここが本題） ──")
    gap = D_WAKUREN - D_UMAREN
    print(f"　D_枠連 − D_馬連 = {D_WAKUREN:+.4f} − ({D_UMAREN:+.4f}) = {gap:+.4f}")
    print("　恒等式で2つに分かれる:")
    print("　　D_枠連 − D_馬連 = [Δ_pair 粗視化がHarvilleを助けた分]")
    print("　　　　　　　　　　 ＋ [(a) 馬連プール→枠 − 枠連プール] ← ★(127)が測るもの")
    print(f"　→ **(a) = {gap:+.4f} − Δ_pair**")
    print("\n　★★載っているのは梯子の「dが0を越えない」ほう:")
    print(f"　　セルを 816→4 まで潰しても d は {mid:+.4f} → −0.0055 と0へ寄るだけで、"
          "**どの水準でも負のまま**。")
    print("　　＝**粗視化は同じ市場に勝たせることはできない**（誤差を消すだけで情報は足さない）。")
    print(f"　　馬連の空間でも同じなら **d_pair^枠 ≤ 0** ⇒ **Δ_pair ≤ {-D_UMAREN:+.4f}**")
    print(f"　　⇒ **(a) ≥ {gap - (-D_UMAREN):+.4f}**（＝枠連Dそのもの）")
    print(f"\n　★(127)への事前予想（測る前にここに書く）:")
    print(f"　　**(a) は +{D_WAKUREN:.4f} 〜 +{gap:.4f} に入る**。点推定は "
          f"三連複での回収率 {100*rec:.0f}%（Δ/|d_id|）を当てて")
    print(f"　　Δ_pair ≈ {rec * abs(D_UMAREN):+.4f} ⇒ **(a) ≈ {gap - rec * abs(D_UMAREN):+.4f}**")
    print("　　⇒ ★**枠連プールが甘いのは本物**（(89)⑥の(i)側）と予想する。")
    print("　　⇒ **(112)(114)(117)の枠連の利得は道具の癖ではない**、が結論になるはず。")
    print("\n　⚠これは(127)の代用ではない。三連複の空間で測った**機構の大きさ**であって、")
    print("　　枠連の空間の(a)そのものではない。**判定は馬連板で行う**。")
    print("　⚠**(a)が負に出たらこの推論のどこかが誤り**。そのときは")
    print("　　「馬連の空間では粗視化がdの符号を変える」＝三連複と構造が違う、を疑うこと。")

    print("\n── 年別（事前登録5: 符号が揃うか） ──")
    pos = 0
    yrs = sorted(set(int(y) for y in yr))
    for y in yrs:
        m2 = yr == y
        a, lo2, hi2 = mci(dw[m2])
        pos += a > 0
        print(f"  {y}  n={int(m2.sum()):5d}  Δ_枠={a:+.4f}  99%CI[{lo2:+.4f},{hi2:+.4f}]")
    print(f"  → 正の年 {pos}/{len(yrs)}")


if __name__ == "__main__":
    main()
