"""(160) ★★★(141)に**一度も置いていなかった対照**を置く。**装置と交絡を疑う**。

⚠★**第1版の事前登録は誤りだったので、ここに残して訂正する（判定基準30の作法）**:
　❌**「全組を100円ずつ買えばROIは払戻率に一致する。定義であって仮説ではない」——誤り**。
　　実測 **61.9%**（枠連・払戻率77.5%）。**装置ではなく私の主張が間違っていた**。
　★**払戻率の恒等式が成り立つのは「板に比例して賭けたとき」**（組kに `1/odds_k` 単位）。
　　そのとき 1レースの費用は `Σ_k 1/odds_k = 1/R`、払戻は `(1/odds_w)·odds_w = 1` なので **ROI = R**。
　⚠**均等買いは別物**。`ROI = (Σ_k p_k·odds_k)/N` で、**組数の多い大穴側に重みが寄る**。
　　→ **人気薄バイアスのぶんだけ払戻率を下回る**。**61.9%はその大きさ**。
　★★**教訓: 「定義だから確かめなくてよい」と書く前に、その恒等式を1行でも導くこと**。→ 判定基準37。

★★★**そして、この失敗で(141)に本当に必要な対照が見えた**——
　**枠連の均等買いベースラインが61.9%しかない＝このプールの人気薄バイアスは非常に強い**。
　⚠**(141)の「比が高い組を買う」は、「オッズの低い組を買う」と分離されていない**。
　　**比 q/q_pool が高い組は、たいてい人気側**。**だとすれば117.3%は
　　「比の裾」ではなく「人気薄バイアスを避けただけ」かもしれない**。**一度も分けていない**。

測るもの（**事前登録。この4つだけ**）

★①**恒等式の対照（正しい版）**: **板に比例して買う** → **ROIは払戻率0.775に一致しなければならない**。
　**これは本当に恒等式**（上の導出）。**±1.0ptに乗らなければ②③④を読まない**（判定基準32）。
　あわせて**板のオーバーラウンド `Σ_k 1/odds_k` の中央値**（**1.290であるはず**）を出す。

★★②**交絡の分離 — オッズ層別**。**組を自分自身の板オッズで層に切り**、
　**層の中で「全部買う」と「比≥1.290だけ買う」を比べる**。
　★**層内でも比が勝つなら、(141)は人気薄バイアスの言い換えではない**。
　⚠**層内で差が消えるなら、(141)は「人気側を買う」の言い換え**＝**新しい情報は無い**。
　⚠**これは(141)以降ずっと欠けていた対照**。

★③**両側の裾** — 高い側だけでなく**低い側**も出す。
　⚠⚠**両側とも100%を超えたら、非効率ではなく装置の偏り**＝**(141)以降を全部疑う**。

★④**単調性** — 比の十分位でROIが単調に上がるか。**両端だけ跳ねるなら裾の少数が効いているだけ**。

⚠**新しい主張はしない**。**既存の主張の検算**。**確定オッズのオラクルであることも変わらない**。

実行: python3 ml/audit_tail_controls.py [開始年(既定2015)]
"""
import math
import sys

import numpy as np

sys.path.insert(0, "ml")
from audit_cond_split import load_boards
from audit_crosspool import load_races, payoff, zq
from audit_crosspool2 import realized
from audit_overlay_all import load_board
from waku_umatan import waku_of

R_WAKU = 0.775
TH = 1.0 / R_WAKU
NDEC = 10
NCMP = 4
ODDS_EDGES = [0, 10, 25, 60, 150, 1e9]     # 板オッズの層（事前に決め打ち・データを見ていない）


def build(y0):
    """(year, ratio配列, 板オッズ配列, 的中フラグ配列, 実配当) を作る。

    q は (141) と同じ **馬連の板→枠へ厳密集約**。q_pool は **枠連の板**。
    """
    ub, wb = load_board(4, 4), load_boards()
    rows = []
    for r in load_races():
        if r["year"] < y0 or not r.get("wakuren"):
            continue
        rl = realized(r)
        if rl is None:
            continue
        a, b, _ = rl
        n, nums = r["n"], [u for u, _, _ in r["horses"]]
        if a not in nums or b not in nums:
            continue
        U, W = ub.get(r["rid"]), wb.get(r["rid"])
        if not U or not W:
            continue
        key = tuple(sorted((waku_of(a, n), waku_of(b, n))))
        if key not in W:
            continue
        v = payoff(r, "枠連(人気順)", [key[0], key[1]])
        if not v or v <= 0:
            continue
        ag = {}
        for k, o in U.items():
            if len(k) != 4 or not k.isdigit() or o <= 0:
                continue
            i, j = int(k[:2]), int(k[2:])
            if i not in nums or j not in nums or i == j:
                continue
            k2 = tuple(sorted((waku_of(i, n), waku_of(j, n))))
            ag[k2] = ag.get(k2, 0.0) + 1.0 / o
        keys = [k for k in sorted(ag) if k in W]
        if key not in keys or len(keys) < 3:
            continue
        qq = np.array([ag[k] for k in keys]); qq /= qq.sum()
        odds = np.array([W[k] for k in keys])
        inv = 1.0 / odds
        qp = inv / inv.sum()
        rows.append((r["year"], qq / qp, odds,
                     np.array([k == key for k in keys]), v))
    return rows


def roi(rows, mask_fn, stake_fn=None):
    """mask_fn(ratio, odds) が True の組を買ったときの (ROI, レース数, 点数, se, z)。

    stake_fn(odds) が None なら1点100円の均等買い。
    """
    cost = ret = 0.0
    nb = 0
    prof = []
    for _, rat, odds, win, v in rows:
        m = mask_fn(rat, odds)
        if not m.any():
            continue
        st = np.full(m.sum(), 100.0) if stake_fn is None else stake_fn(odds[m])
        c = st.sum()
        # 的中組の払戻 = 賭け金 × (実配当/100)
        g = float((st[win[m]] * (v / 100.0)).sum()) if (m & win).any() else 0.0
        cost += c; ret += g; nb += int(m.sum())
        prof.append(g - c)
    if cost <= 0 or len(prof) < 2:
        return None
    p = np.array(prof)
    se = p.std(ddof=1) / math.sqrt(len(p)) * len(p) / cost * 100.0
    return 100.0 * ret / cost, len(p), nb, se, zq(0.01 / NCMP)


HDR = f"{'':>26}{'レース':>9}{'点数':>11}{'ROI':>10}{'99%CI(Bonf)':>20}"


def line(name, res):
    if res is None:
        print(f"{name:>26}  ——")
        return
    r, nr, nb, se, z = res
    print(f"{name:>26}{nr:>9,}{nb:>11,}{r:>9.1f}%{f'[{r-z*se:.1f},{r+z*se:.1f}]':>20}")


def main():
    y0 = int(sys.argv[1]) if len(sys.argv) > 1 else 2015
    print(f"(160) ★(141)に置いていなかった対照（{y0}年以降・枠連プール）")
    print("　q の出どころ: **馬連の板→枠へ厳密集約**（(141)と同一）")
    print("　q_pool の出どころ: **枠連の板**")
    rows = build(y0)
    print(f"　{len(rows):,}レース\n")

    ors = np.array([float((1.0 / r[2]).sum()) for r in rows])
    print("■ ★①恒等式の対照（**正しい版**）")
    print(f"　板のオーバーラウンド Σ1/odds の中央値 **{np.median(ors):.4f}**"
          f"（1/0.775 = {TH:.4f} のはず）　四分位 [{np.quantile(ors,.25):.4f}, {np.quantile(ors,.75):.4f}]")
    print(HDR)
    prop = roi(rows, lambda rat, o: np.ones_like(rat, dtype=bool),
               stake_fn=lambda o: 100.0 / o)
    line("★板に比例して買う", prop)
    flat = roi(rows, lambda rat, o: np.ones_like(rat, dtype=bool))
    line("（参考）全組を均等買い", flat)
    ok1 = prop is not None and abs(prop[0] - 100 * R_WAKU) <= 1.0
    print(f"　→ 比例買い {prop[0]:.2f}% vs 払戻率 {100*R_WAKU:.1f}%　差 {prop[0]-100*R_WAKU:+.2f}pt"
          f" → **{'★①立った' if ok1 else '⚠⚠①が落ちた'}**")
    print(f"　⚠**均等買いは {flat[0]:.1f}%**＝**払戻率を {100*R_WAKU-flat[0]:.1f}pt 下回る**"
          f"（**このプールの人気薄バイアスの強さ**）。**②はこれを打ち消すための層別**")
    if not ok1:
        print("\n⚠⚠**①が落ちた。②③④は読まない**（判定基準32）。原因究明が先。")
        return

    print("\n■ ★★②交絡の分離 — **板オッズの層の中で比べる**")
    print("　★**層内でも比が勝てば、(141)は「人気側を買う」の言い換えではない**")
    print(f"{'オッズ層':>26}{'全部買う':>12}{'比≥1.290':>12}{'差(pt)':>10}{'99%CI(差)':>22}")
    for lo, hi in zip(ODDS_EDGES[:-1], ODDS_EDGES[1:]):
        base = roi(rows, lambda rat, o, lo=lo, hi=hi: (o >= lo) & (o < hi))
        sel = roi(rows, lambda rat, o, lo=lo, hi=hi: (o >= lo) & (o < hi) & (rat >= TH))
        nm = f"{lo:g}〜{hi:g}倍" if hi < 1e8 else f"{lo:g}倍〜"
        if base is None or sel is None:
            print(f"{nm:>26}  ——")
            continue
        d = sel[0] - base[0]
        # 差のseは保守的に両者のseを独立とみなして合成（★共通部分があるので過大評価＝安全側）
        sd = math.hypot(base[3], sel[3])
        z = base[4]
        print(f"{nm:>26}{base[0]:>11.1f}%{sel[0]:>11.1f}%{d:>+10.1f}"
              f"{f'[{d-z*sd:+.1f},{d+z*sd:+.1f}]':>22}")
    print("　⚠**差のCIは両者を独立として合成した保守値**（実際は同じレースを共有＝過大評価・安全側）")

    print("\n■ ★③両側の裾（**高い側しか見てこなかったのを両側にする**）")
    print(HDR)
    for t in [TH, 1.10, 1.00]:
        line(f"高い側 ratio ≥ {t:.3f}", roi(rows, lambda rat, o, t=t: rat >= t))
    for t in [1.00, 0.90, 1.0 / TH]:
        line(f"低い側 ratio ≤ {t:.3f}", roi(rows, lambda rat, o, t=t: rat <= t))

    print("\n■ ★④単調性（比の十分位・全体の分位で切る）")
    allrat = np.concatenate([r[1] for r in rows])
    edges = np.quantile(allrat, np.linspace(0, 1, NDEC + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    print(HDR)
    prev, mono = None, 0
    for d in range(NDEC):
        lo, hi = edges[d], edges[d + 1]
        res = roi(rows, lambda rat, o, lo=lo, hi=hi: (rat >= lo) & (rat < hi))
        line(f"D{d+1} [{lo:.2f},{hi:.2f})", res)
        if res is not None:
            if prev is not None and res[0] > prev:
                mono += 1
            prev = res[0]
    print(f"　→ 隣り合う十分位で上がったのは **{mono}/{NDEC-1}** 回"
          f" → **{'★単調に近い＝裾の大穴だけではない' if mono >= 7 else '⚠単調ではない'}**")

    print("\n" + "=" * 96)
    print("★読み方（**事前登録のとおり**）")
    print("  ★★**②が本命**。**層内で差が消えたら、(141)は人気薄バイアスの言い換え**＝新情報は無い。")
    print("  ⚠⚠**③で両側とも100%超なら装置の偏り**＝(141)以降を全部疑う。")
    print("  ⚠**確定オッズのオラクルであることは何も変わらない**。**張れる時点は(148)待ち**。")


if __name__ == "__main__":
    main()
