"""(158) ★★★★★馬単の板で **空白のマスを埋める** — **(141)を独立な経路で検証できる**

★★★これが今日いちばん重い一手（2026-08-16に馬単の板 type=6 が揃った）
　★**経路の強弱が本質だと分かった**（今日の監査）:
| 経路 | qの作り方 | 枠連 | 複勝 | 馬連 | 三連複 |
|---|---|---|---|---|---|
| ★強い | 別の市場の板を**厳密に集約** | ★**117.3%**(141) | 89.0%(146) | ❌**空白** | ❌空白 |
| ⚠弱い | 単勝＋λ補正Harville | 93.9%(141) | 46.3%(151) | 陰性(150) | 陰性(150) |
　★**100%を超えたのは「強い経路 × 最も不人気なプール」の1マスだけ**。

★★馬単の板から**厳密に**作れるもの（**Harville不使用・近似ゼロ**）
　**A 馬単 → 馬連**: `q_馬連({i,j}) = q_馬単(i,j) + q_馬単(j,i)`　← **馬連の「強い」マスが埋まる**
　**B 馬単 → 枠連**: `q_枠({a,b}) = Σ_{(i,j): {waku(i),waku(j)}={a,b}} q_馬単(i,j)`
　　 ★★**これが本命**——**(141)の +0.0266 / 117.3% を、馬連プールとは別の市場で検証できる**。
　　 **三連単(保留1)を待たずに、今日できる独立検証**。
　**C 馬単そのもの**: ⚠**厳密な上位プールが無い**（三連単が要る）ので**λHarvilleの弱い経路だけ**。
　　 → **構造仮説（非効率は不人気プールに住む）の検定にはなるが、弱い**。**そう明記する**。

★★★事前登録（**測る前に書いている**）
　1. **測るのは A / B / C の3つだけ**。**後から増やさない**。
　2. **閾値の梯子 1.00 / 1.10 / ★1/払戻率 / 1.50 / 2.00**。**主判定は 1/払戻率**。
　　 （馬連 0.775→1.290 / 枠連 0.775→1.290 / ⚠**馬単は0.750→1.333が正しい**。(162)で訂正）
　　 ⚠**(158)は馬単プールに賭けていないので、この誤りの影響を受けない**（馬単は q 側にしか出てこない）。
　3. **賭け方**: 比が閾値以上の**全組を1点100円**。該当無しは見送り。**選ばない**。
　4. **主判定（Bonferroni α=0.01/3）**: **ROIの99%CI下端 > 100%** かつ
　　 **年別で「91%を割る年が2つ以下」** かつ **2021-2026でも100%超**（(142)の教訓）。
　5. ★★**陽性対照（判定基準32）**: **同じスクリプトで「馬連→枠連」も測る**。
　　 **(141)の実測 117.3% を ±5pt で再現しなければ、A・B・C を読まない**。
　　 ★**これは「別経路の検証」と「対照」を兼ねる**——**Bが対照(馬連→枠)と一致するか**が答え。
　6. **プラセボ**: 無作為に同数買う。**解析的**（判定基準23）。
　7. **道具の検算**: 的中組で「板×100 と実配当」のずれ。**(127)は0.07%だった**。
　8. ★**予想**: ★**当てにしてよい予想は持っていない**（類推はこの4日で5連敗）。
　　 **恒等式から言えることだけ**: **AとBは近似ゼロの集約**なので、
　　 **Bが対照と大きく食い違うなら、それは「プールごとの違い」であって計算の誤差ではない**。
　　 **向きは決まらない**。

⚠⚠★**陽性でも「儲かる道」ではない**。**(141)と同じ立場＝確定オッズのオラクル**。
　**張れる時点で成立するかは(148)＝時系列オッズ25開催日待ちのまま**。**ここを混同しない**。

実行: python3 ml/audit_umatan_tail.py [開始年(既定2015)]
"""
import math
import sys

import numpy as np

sys.path.insert(0, "ml")
from audit_cond_split import load_boards
from audit_crosspool import PAYBACK, load_races, payoff, zq
from audit_crosspool2 import PAYKEY, realized
from audit_overlay_all import load_board
from waku_umatan import waku_of

THS = [1.00, 1.10, None, 1.50, 2.00]      # None は 1/払戻率（券種ごとに入れ替える）
KNOWN_141, TOL = 117.3, 5.0
NCMP = 3


def summarize(name, rows, z, need_lo=True):
    """rows: [(year, cost, ret)]。ROI・99%CI・年91%割れ・2021-・プラセボ。"""
    if not rows:
        print(f"{name:>16}   該当なし")
        return None
    pr = np.array([v - c for _, c, v in rows], float)
    cost = sum(c for _, c, _ in rows)
    ret = sum(v for _, _, v in rows)
    mc = cost / len(pr)
    se = pr.std(ddof=1) / math.sqrt(len(pr)) if len(pr) > 1 else float("nan")
    lo, hi = 1 + (pr.mean() - z * se) / mc, 1 + (pr.mean() + z * se) / mc
    ys = np.array([y for y, _, _ in rows])
    bad = 0
    for u in sorted(set(ys.tolist())):
        if (ys == u).sum() >= 30:
            cc = sum(c for (y, c, _) in rows if y == u)
            vv = sum(v for (y, _, v) in rows if y == u)
            bad += int(vv / max(cc, 1) < 0.91)
    c21 = sum(c for y, c, _ in rows if y >= 2021)
    v21 = sum(v for y, _, v in rows if y >= 2021)
    roi = 100 * ret / cost
    mark = "★★買える" if (need_lo and lo > 1.0) else ""
    print(f"{name:>16}{len(pr):>8,}{roi:>8.1f}%"
          f"{'[' + format(100*lo, '.1f') + ',' + format(100*hi, '.1f') + ']':>19}"
          f"{bad:>8}{100*v21/max(c21,1):>9.1f}% {mark}")
    return roi


def main():
    y0 = int(sys.argv[1]) if len(sys.argv) > 1 else 2015
    tb = load_board(6, 4)                 # 馬単（順序あり・キーは "0102"=1着1番2着2番）
    ub = load_board(4, 4)                 # 馬連
    wb = load_boards()                    # 枠連 {rid: {(a,b): odds}}
    if not tb or not ub or not wb:
        sys.exit("馬単(type=6)・馬連(type=4)・枠連(type=3) の板が要る。")
    races = load_races()
    print(f"(158) 馬単の板で空白のマスを埋める（{y0}年以降）")
    print(f"　馬単 {len(tb):,} / 馬連 {len(ub):,} / 枠連 {len(wb):,} レース")
    print("★経路: A 馬単→馬連（厳密）/ B ★馬単→枠連（厳密）/ C λHarville→馬単（弱い）")
    print("　対照: 馬連→枠連（厳密）＝(141)の117.3%を再現するか\n")

    rows = {k: [] for k in ("A 馬単→馬連", "B 馬単→枠連", "対照 馬連→枠連")}
    chk_n = chk_bad = 0
    for r in races:
        yy = r["year"]
        if yy < y0:
            continue
        rl = realized(r)
        if rl is None:
            continue
        a, b, _ = rl
        n, hs = r["n"], r["horses"]
        nums = [u for u, _, _ in hs]
        if a not in nums or b not in nums:
            continue
        T, U, W = tb.get(r["rid"]), ub.get(r["rid"]), wb.get(r["rid"])
        if not T:
            continue
        # 馬単の板 → 正規化前の重み
        tw = {}
        for k, o in T.items():
            if len(k) != 4 or not k.isdigit() or o <= 0:
                continue
            i, j = int(k[:2]), int(k[2:])
            if i in nums and j in nums and i != j:
                tw[(i, j)] = 1.0 / o
        if len(tw) < 6:
            continue

        # ── A: 馬単 → 馬連（厳密）。q_pool は馬連の板 ──
        if U:
            ua = {}
            for (i, j), w in tw.items():
                ua[tuple(sorted((i, j)))] = ua.get(tuple(sorted((i, j))), 0.0) + w
            ud = {}
            for k, o in U.items():
                if len(k) == 4 and k.isdigit() and o > 0:
                    x, y = int(k[:2]), int(k[2:])
                    if x in nums and y in nums:
                        ud[tuple(sorted((x, y)))] = o
            keys = [k for k in sorted(ua) if k in ud]
            key = tuple(sorted((a, b)))
            if key in keys and len(keys) >= 5:
                v = payoff(r, PAYKEY["馬連"], [key[0], key[1]])
                if v and v > 0:
                    chk_n += 1
                    if abs(ud[key] * 100 - v) > max(10.0, v * 0.01):
                        chk_bad += 1
                    qq = np.array([ua[k] for k in keys]); qq /= qq.sum()
                    inv = np.array([1.0 / ud[k] for k in keys]); qp = inv / inv.sum()
                    rows["A 馬単→馬連"].append(
                        (yy, qq / qp, np.array([ud[k] for k in keys]),
                         np.array([k == key for k in keys]), "馬連"))

        # ── B/対照: → 枠連（厳密）。q_pool は枠連の板 ──
        if W and r.get("wakuren"):
            wkey = tuple(sorted((waku_of(a, n), waku_of(b, n))))
            if wkey in W:
                v = payoff(r, "枠連(人気順)", [wkey[0], wkey[1]])
                if v and v > 0:
                    def agg(src):
                        out = {}
                        for (i, j), w in src.items():
                            k2 = tuple(sorted((waku_of(i, n), waku_of(j, n))))
                            out[k2] = out.get(k2, 0.0) + w
                        return out
                    for lab, src in (("B 馬単→枠連", tw),
                                     ("対照 馬連→枠連",
                                      {(int(k[:2]), int(k[2:])): 1.0 / o
                                       for k, o in (U or {}).items()
                                       if len(k) == 4 and k.isdigit() and o > 0
                                       and int(k[:2]) in nums and int(k[2:]) in nums})):
                        if not src:
                            continue
                        ag = agg(src)
                        keys = [k for k in sorted(ag) if k in W]
                        if wkey not in keys or len(keys) < 3:
                            continue
                        qq = np.array([ag[k] for k in keys]); qq /= qq.sum()
                        inv = np.array([1.0 / W[k] for k in keys]); qp = inv / inv.sum()
                        rows[lab].append((yy, qq / qp, np.array([W[k] for k in keys]),
                                          np.array([k == wkey for k in keys]), "枠連"))

    z = zq(0.01 / NCMP)
    print(f"★道具の検算: 馬連の的中 {chk_n:,}件のうち板×100と実配当のずれ {chk_bad}"
          f"（{chk_bad/max(chk_n,1):.2%}）  ※(127)は0.07%\n")

    ctrl = None
    for lab in ("対照 馬連→枠連", "B 馬連→枠連".replace("馬連→", "馬単→"), "A 馬単→馬連"):
        data = rows.get(lab)
        if not data:
            print(f"■ {lab}: 突き合わせできたレースが無い\n")
            continue
        kind = data[0][4]
        R = PAYBACK[kind]
        ths = [t if t is not None else 1.0 / R for t in THS]
        print(f"■ {lab}（{kind}・払戻率 {R:.3f} → 利益に要る比 {1/R:.3f}・{len(data):,}レース）")
        print(f"{'閾値':>16}{'買ったR':>8}{'ROI':>9}{'99%CI(Bonf)':>19}"
              f"{'年91%割れ':>8}{'2021-':>10}")
        for t in ths:
            sel_rows, plr, plc = [], 0.0, 0.0
            for (yy, rat, odds, win, _) in data:
                sel = rat >= t
                if not sel.any():
                    continue
                cc = 100.0 * int(sel.sum())
                vv = float(odds[sel & win].sum() * 100.0)
                sel_rows.append((yy, cc, vv))
                plr += (100.0 * odds[win].sum() / len(odds)) * int(sel.sum())
                plc += cc
            nm = f"{t:.3f}" + ("★" if abs(t - 1.0 / R) < 1e-9 else "")
            roi = summarize(nm, sel_rows, z)
            if roi is not None and abs(t - 1.0 / R) < 1e-9 and lab.startswith("対照"):
                ctrl = roi
        print(f"　（プラセボ＝無作為に同数: {100*plr/max(plc,1):.1f}%・解析的）\n")

    print("=" * 96)
    if ctrl is not None:
        ok = abs(ctrl - KNOWN_141) <= TOL
        print(f"★①陽性対照: 馬連→枠連の主閾値 **{ctrl:.1f}%** vs (141)の {KNOWN_141}%"
              f"　差 {ctrl-KNOWN_141:+.1f}pt → **{'★立った' if ok else '⚠立っていない'}**")
        if not ok:
            print("⚠**立っていない。A・B を読まないこと**（判定基準32）。")
    print("★読み方（事前登録のとおり。**後から足していない**）")
    print("  ★**Bが対照とほぼ一致すれば、(141)は「馬連プール固有の癖」ではない**＝**独立に確認された**。")
    print("  ⚠**Bが大きく食い違えば、それは『プールごとの違い』**。**どちらが正しいかは決まらない**。")
    print("  ⚠⚠**陽性でも確定オッズのオラクル**。**張れる時点は(148)＝時系列オッズ待ちのまま**。")


if __name__ == "__main__":
    main()
