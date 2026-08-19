"""(163) ★★★(141)の枠連117.3%に、**まだ一度も置いていない対照を3つ**置く。

★★なぜ——**構造仮説が死に(161)、100%を超える主張は枠連1本になった**。
　**残り1本になったからこそ、その1本を疑う番**。**(160)で3つ潰したが、まだ残っている**。

測るもの（**事前登録。この3つだけ。これ以上増やさない**）

★①**ゾロ目（同枠）の分離** — ⚠**(127)の第2パスで「ゾロ目の寄与は D=−0.078 と逆向き」**と
　分かっていたのに、**比の裾の測定では一度も分離していない**（(141)(158)(160)(161)すべて）。
　★**判定**: **ゾロ目を除いても100%を超えるか**。
　⚠⚠**利益がゾロ目に集中していたら、それは「枠連の非効率」ではなく「ゾロ目という特殊な札の癖」**。
　　**ゾロ目は同枠2頭が1着2着に入る事象**で、**馬連からの集約が最も特殊になる場所**
　　（`Σ_{i<j, 共に枠a}` で、他の枠組と組の数が違う）。**別物として扱う理由がある**。

★★②**頭数別** — `waku_of` は**頭数で馬→枠の割り当てが変わる**（8で割った余りで枠の大きさが違う）。
　★**判定**: **特定の頭数帯にだけ利益が寄っていないか**。
　⚠⚠**寄っていたら割り当てのバグを疑う**。**寄っていなければ、割り当ては少なくとも系統的には壊れていない**。
　★**これは「効果の頑健性」ではなく「道具の検算」として読む**（判定基準32の使い方）。

★③**脱落率の頭数別** — ⚠**`waku_of` が間違っていると、実現した枠組が板や払戻と一致せず、
　そのレースは黙って落ちる**。**0件は成功の顔をして出てくる**（(161)第1版の教訓）。
　★**判定**: **頭数別の脱落率が平坦か**。**どこかで跳ねたらそこが壊れている**。

⚠**新しい主張はしない**。**既存の1本の検算**。
⚠⚠**確定オッズのオラクルであることは何も変わらない**。**張れる時点は(148)待ち**。

実行: python3 ml/audit_waku_confound.py [開始年(既定2015)]
"""
import math
import sys

import numpy as np

sys.path.insert(0, "ml")
from audit_cond_split import load_boards
from audit_crosspool import LINE, load_races, payoff, zq
from audit_crosspool2 import realized
from audit_overlay_all import load_board
from waku_umatan import waku_of

R = LINE["枠連(人気順)"]
TH = 1.0 / R
NCMP = 3


def build(y0):
    """(頭数, 比, 板オッズ, 的中, ゾロ目フラグ, 実配当) をレース単位で作る。

    あわせて**脱落の理由を頭数別に数える**（③）。
    """
    ub, wb = load_board(4, 4), load_boards()
    rows, drop = [], {}

    def note(n, why):
        d = drop.setdefault(n, {"総数": 0, "板なし": 0, "枠組が板に無い": 0,
                                "払戻が引けない": 0, "採用": 0})
        d[why] += 1

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
        note(n, "総数")
        U, W = ub.get(r["rid"]), wb.get(r["rid"])
        if not U or not W:
            note(n, "板なし")
            continue
        key = tuple(sorted((waku_of(a, n), waku_of(b, n))))
        if key not in W:
            note(n, "枠組が板に無い")
            continue
        v = payoff(r, "枠連(人気順)", [key[0], key[1]])
        if not v or v <= 0:
            note(n, "払戻が引けない")
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
            note(n, "枠組が板に無い")
            continue
        q = np.array([ag[k] for k in keys]); q /= q.sum()
        odds = np.array([W[k] for k in keys])
        inv = 1.0 / odds; qp = inv / inv.sum()
        zoro = np.array([k[0] == k[1] for k in keys])
        note(n, "採用")
        rows.append((n, q / qp, odds, np.array([k == key for k in keys]), zoro, v))
    return rows, drop


def roi(rows, mask_fn):
    cost = ret = 0.0
    nb = hit = 0
    prof = []
    for n, rat, odds, win, zoro, v in rows:
        m = mask_fn(rat, odds, zoro, n)
        if not m.any():
            continue
        c = 100.0 * int(m.sum())
        g = v if (m & win).any() else 0.0
        cost += c; ret += g; nb += int(m.sum()); hit += int((m & win).any())
        prof.append(g - c)
    if cost <= 0 or len(prof) < 2:
        return None
    p = np.array(prof)
    se = p.std(ddof=1) / math.sqrt(len(p)) * len(p) / cost * 100.0
    return 100.0 * ret / cost, len(prof), nb, hit, se, zq(0.01 / NCMP)


def line(lab, res):
    if res is None:
        print(f"{lab:>26}  ——")
        return
    r, nr, nb, hit, se, z = res
    print(f"{lab:>26}{nr:>9,}{nb:>10,}{hit:>7,}{r:>9.1f}%"
          f"{f'[{r-z*se:.1f},{r+z*se:.1f}]':>21}")


HDR = f"{'':>26}{'レース':>9}{'点数':>10}{'的中R':>7}{'ROI':>10}{'99%CI(Bonf)':>21}"


def main():
    y0 = int(sys.argv[1]) if len(sys.argv) > 1 else 2015
    print(f"(163) ★(141)の枠連に置いていなかった対照3つ（{y0}年以降）")
    print("　q の出どころ: **馬連の板→枠へ厳密集約**（(141)と同一）")
    print("　q_pool の出どころ: **枠連の板**")
    rows, drop = build(y0)
    print(f"　{len(rows):,}レース　主閾値 {TH:.3f}（払戻率 {R:.3f}）\n")

    def show(lab, fn):
        """⚠**第1版は line() に lambda をそのまま渡していた**（roi を通し忘れ）。
        **check_sigs.py は引数の数しか見ないので素通りした**——**冒頭の限界のとおり**。"""
        line(lab, roi(rows, fn))

    print("■ ★①ゾロ目（同枠）の分離 — **(141)以降ずっと混ざっていた**")
    print(HDR)
    show("全部（=(141)の形）", lambda rat, o, z, n: rat >= TH)
    show("★ゾロ目を除く", lambda rat, o, z, n: (rat >= TH) & ~z)
    show("ゾロ目だけ", lambda rat, o, z, n: (rat >= TH) & z)
    show("（参考）全組・ゾロ目除く", lambda rat, o, z, n: ~z)
    show("（参考）全組・ゾロ目だけ", lambda rat, o, z, n: z)
    zs = float(sum(int(r[4].sum()) for r in rows))
    ts = float(sum(len(r[4]) for r in rows))
    print(f"　ゾロ目は全組の {100*zs/ts:.1f}%")

    print("\n■ ★★②頭数別 — **waku_of の割り当ては頭数で変わる。道具の検算として読む**")
    print(HDR)
    bands = [(9, 11), (12, 13), (14, 15), (16, 16), (17, 18)]
    for lo, hi in bands:
        show(f"{lo}〜{hi}頭",
             lambda rat, o, z, n, lo=lo, hi=hi: (rat >= TH) if lo <= n <= hi
             else np.zeros_like(rat, dtype=bool))
    print("　⚠**特定の帯にだけ寄っていたら割り当てのバグを疑う**")

    print("\n■ ★③脱落率の頭数別 — **0件は成功の顔をして出てくる**")
    print(f"{'頭数':>8}{'総数':>9}{'採用':>9}{'採用率':>9}"
          f"{'板なし':>9}{'枠組が板に無い':>15}{'払戻が引けない':>15}")
    for n in sorted(drop):
        d = drop[n]
        if d["総数"] < 50:
            continue
        print(f"{n:>8}{d['総数']:>9,}{d['採用']:>9,}"
              f"{100*d['採用']/d['総数']:>8.1f}%{d['板なし']:>9,}"
              f"{d['枠組が板に無い']:>15,}{d['払戻が引けない']:>15,}")

    print("\n" + "=" * 92)
    print("★読み方（**事前登録のとおり**）")
    print("  ★★**①でゾロ目を除いても100%を超えれば、(141)はゾロ目の癖ではない**。")
    print("  ⚠⚠**ゾロ目だけに寄っていたら、それは枠連の非効率ではなく特殊な札の癖**。")
    print("  ★**②③は道具の検算**。**帯で跳ねたら waku_of を疑う**。")
    print("  ⚠**確定オッズのオラクルであることは何も変わらない**。**張れる時点は(148)待ち**。")


if __name__ == "__main__":
    main()
