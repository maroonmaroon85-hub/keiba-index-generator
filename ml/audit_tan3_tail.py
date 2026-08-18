"""(161) ★★★三連単の板で**馬単そのもの**を測る。**構造仮説の最も鋭い検定**。

⚠**データ待ち**（`data/nk_odds/type8_*.jsonl.gz`）。**届く前に事前登録を書いておく**。
⚠**判定基準(124)の教訓**: **寝かせたスクリプトは寝ている間に腐る**。回す前に `ml/check_sigs.py`。

★★なぜ書くか
　(158)で**強い経路のROI実測が3点そろい、売上の少ない順とROIの順が一致した**:
　**枠連（最も不人気）117% ＞ 複勝 89% ＞ 馬連 43%**。
　→ ★**馬単は枠連の次に不人気**。**構造仮説が正しければ馬単は2番目に良く、100%を跨ぐ可能性がある**。
　→ ★**馬単の比の裾は三連単の板からしか厳密に作れない**。**これがこの板を集める唯一の理由**。

★★★**事前登録の予測（データを見る前に書く。当たっても外れても消さない）**
　1. ★**馬単は100%前後**（**構造仮説が正しければ跨ぐ**）。⚠**当てにいかない。外れたら構造仮説を捨てる**。
　2. **三連複・三連単は陰性**（**最も売れているプール**）。
　3. ★**三連単→馬連は(158)Aの42.8%を再現するはず**——**別ルートで測る同じマス**＝**陰性の対照**。
　4. ★**三連単→枠連は116%前後**（(141)117.3 / (158)B 114.7 / (158)対照 116.4 と並ぶはず）。
　⚠**この4つは「予想を持たない」という方針の例外ではない**。**構造仮説という既存の主張の含意**であり、
　　**外れたときに何を捨てるかまで書いてあるから登録している**（判定基準24の形）。

経路（**すべて厳密集約・Harville不使用**。キーは "010203"＝1着1番2着2番3着3番）
　★**A 三連単→馬単**  `q(i,j) = Σ_k q(i,j,k)`                       ← ★**本命。空白のマス**
　　**B 三連単→三連複** `q({i,j,k}) = Σ_{6通りの並べ替え}`
　　**C 三連単→馬連**  `q({i,j}) = Σ_k q(i,j,k) + Σ_k q(j,i,k)`      ← ★**(158)Aの陰性の再現対照**
　　**D 三連単→枠連**  `Σ_{(i,j,k): {waku(i),waku(j)}={a,b}}`        ← ★**(141)の再現対照**
　　**対照 馬連→枠連**（(158)で116.4%）                                ← ★**判定基準32の陽性対照**

判定（**先に書く**）
　★**主判定は A の閾値 1/0.775 = 1.290**。**99%CI下端 > 100%**・**年91%割れ ≤2**・**2021- が100%超**。
　**Bonferroni は5経路で α=0.01/5**。
　⚠**ゲート1（判定基準32）**: **対照 馬連→枠連 が116.4%を±5ptで再現しなければ何も読まない**。
　⚠**ゲート2（判定基準37）**: **各プールで「板に比例して買う」ROIが払戻率±1.0ptに乗ること**。
　　**乗らなければそのプールは読まない**。★**(160)で作った対照を最初から入れる**（判定基準29）。

(160)で作った対照を**本命Aに最初から当てる**
　①**比例買い**（上のゲート2）／②★**オッズ層別**（「人気側を買っているだけ」の分離）／
　③**両側の裾**（両側100%超なら装置の偏り）／④**単調性**（裾の少数の大穴の排除）

⚠⚠**陽性でも確定オッズのオラクル**。**(143)の「発走10分前で最終オッズの金の75.6%は未投入」は生きている**。
**張れる時点を決めるのは(148)＝時系列オッズだけ**。**このスクリプトは運用に何も足さない**。

★**実装上の制約（先に潰しておく）**: **三連単の板は42,181レース×最大4,896組≒2億エントリ**。
　**`load_board` のように全部辞書に載せると確実にメモリで落ちる**。
　→ ★**`iter_records(8)` で1レースずつ読み、その場で集約して生の板は捨てる**。

実行: python3 ml/audit_tan3_tail.py [開始年(既定2015)]
"""
import math
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, "ml")
from audit_cond_split import load_boards
from audit_crosspool import load_races, payoff, zq
from audit_crosspool2 import realized
from audit_overlay_all import load_board
from waku_umatan import waku_of

R = {"馬単": 0.775, "三連複": 0.750, "馬連": 0.775, "枠連": 0.775}
PAYKEY = {"馬単": "馬単", "三連複": "三連複", "馬連": "馬連", "枠連": "枠連(人気順)"}
THS = [1.00, 1.10, None, 1.50, 2.00]      # None は 1/払戻率（主閾値）
NCMP = 5
ODDS_EDGES = [0, 10, 25, 60, 150, 1e9]    # (160)と同じ。データを見ずに決めてある
NDEC = 10
KNOWN_CTRL, TOL = 116.4, 5.0              # (158)の対照 馬連→枠連


def aggregate(od, nums, n):
    """三連単の板 {"010203": オッズ} を4つの券種へ**厳密に**集約する。

    返り値は {券種: {キー: 重み}}。重みは 1/オッズ の和（正規化前）。
    """
    out = {k: defaultdict(float) for k in ("馬単", "三連複", "馬連", "枠連")}
    ns = set(nums)
    for k, o in od.items():
        if len(k) != 6 or not k.isdigit() or o <= 0:
            continue
        i, j, m = int(k[:2]), int(k[2:4]), int(k[4:])
        if i not in ns or j not in ns or m not in ns:
            continue
        if i == j or j == m or i == m:
            continue
        w = 1.0 / o
        out["馬単"][(i, j)] += w
        out["三連複"][tuple(sorted((i, j, m)))] += w
        out["馬連"][tuple(sorted((i, j)))] += w
        out["枠連"][tuple(sorted((waku_of(i, n), waku_of(j, n))))] += w
    return out


def rows_for(kind, ag, board, realkey, v, yy):
    """(年, 比の配列, 板オッズ配列, 的中フラグ, 実配当) を作る。板に無いキーは捨てる。"""
    keys = [k for k in sorted(ag) if k in board]
    if realkey not in keys or len(keys) < 3:
        return None
    q = np.array([ag[k] for k in keys]); q /= q.sum()
    odds = np.array([board[k] for k in keys])
    inv = 1.0 / odds; qp = inv / inv.sum()
    return (yy, q / qp, odds, np.array([k == realkey for k in keys]), v)


def roi(rows, mask_fn, stake_fn=None):
    cost = ret = 0.0
    nb = hit = 0
    prof, yl, cl = [], [], []
    for yy, rat, odds, win, v in rows:
        m = mask_fn(rat, odds)
        if not m.any():
            continue
        st = np.full(int(m.sum()), 100.0) if stake_fn is None else stake_fn(odds[m])
        c = float(st.sum())
        g = float((st[win[m]] * (v / 100.0)).sum()) if (m & win).any() else 0.0
        cost += c; ret += g; nb += int(m.sum()); hit += int((m & win).any())
        prof.append(g - c); yl.append(yy); cl.append(c)
    if cost <= 0 or len(prof) < 2:
        return None
    p = np.array(prof)
    se = p.std(ddof=1) / math.sqrt(len(p)) * len(p) / cost * 100.0
    return dict(roi=100.0 * ret / cost, nr=len(p), nb=nb, hit=hit, se=se,
                z=zq(0.01 / NCMP), prof=p, yy=np.array(yl),
                costs=np.array(cl), cost=cost)


def year_bad(res):
    """91%を割った年の数（(141)以降の共通指標）。**事前登録の判定に使う**。"""
    bad = 0
    for y in sorted(set(res["yy"].tolist())):
        m = res["yy"] == y
        c = float(res["costs"][m].sum())
        if c <= 0:
            continue
        if 100.0 * (res["prof"][m].sum() + c) / c < 91.0:
            bad += 1
    return bad


def table(name, rows, rate):
    th_main = 1.0 / rate
    print(f"\n■ {name}（払戻率 {rate:.3f} → 利益に要る比 {th_main:.3f}・{len(rows):,}レース）")
    print(f"{'閾値':>9}{'買ったR':>9}{'点数':>10}{'的中R':>7}{'ROI':>9}"
          f"{'99%CI(Bonf)':>21}{'年91%割れ':>11}{'2021-':>9}")
    main = None
    for t in THS:
        tt = th_main if t is None else t
        res = roi(rows, lambda rat, o, tt=tt: rat >= tt)
        if res is None:
            continue
        r21 = roi([x for x in rows if x[0] >= 2021], lambda rat, o, tt=tt: rat >= tt)
        star = "\u2605" if t is None else " "
        lo = res["roi"] - res["z"] * res["se"]
        hi = res["roi"] + res["z"] * res["se"]
        ci = "[%.1f,%.1f]" % (lo, hi)
        s21 = ("%.1f%%" % r21["roi"]) if r21 else "\u2014"
        print(f"{tt:>8.3f}{star}{res['nr']:>9,}{res['nb']:>10,}{res['hit']:>7,}"
              f"{res['roi']:>8.1f}%{ci:>21}{year_bad(res):>11}{s21:>9}")
        if t is None:
            main = res
    return main


def controls(name, rows, rate):
    """(160)の対照4つ。①はゲート、②が本命。"""
    print(f"\n── ★(160)の対照: {name} ──")
    prop = roi(rows, lambda rat, o: np.ones_like(rat, dtype=bool),
               stake_fn=lambda o: 100.0 / o)
    flat = roi(rows, lambda rat, o: np.ones_like(rat, dtype=bool))
    ok = prop is not None and abs(prop["roi"] - 100 * rate) <= 1.0
    print(f"　①比例買い {prop['roi']:.2f}% vs 払戻率 {100*rate:.1f}%"
          f"（差 {prop['roi']-100*rate:+.2f}pt）→ **{'★立った' if ok else '⚠⚠落ちた'}**"
          f"　／均等買い {flat['roi']:.1f}%")
    if not ok:
        print("　⚠⚠**ゲート2が落ちた。このプールは読まない**（判定基準37）。")
        return False
    th = 1.0 / rate
    print(f"　★②オッズ層別（**「人気側を買っているだけ」の分離**）")
    print(f"{'オッズ層':>16}{'全部買う':>11}{f'比≥{th:.3f}':>11}{'差(pt)':>10}{'99%CI(差)':>21}")
    for lo, hi in zip(ODDS_EDGES[:-1], ODDS_EDGES[1:]):
        b = roi(rows, lambda rat, o, lo=lo, hi=hi: (o >= lo) & (o < hi))
        s = roi(rows, lambda rat, o, lo=lo, hi=hi: (o >= lo) & (o < hi) & (rat >= th))
        nm = f"{lo:g}〜{hi:g}倍" if hi < 1e8 else f"{lo:g}倍〜"
        if b is None or s is None:
            print(f"{nm:>16}  ——")
            continue
        d = s["roi"] - b["roi"]; sd = math.hypot(b["se"], s["se"]); z = b["z"]
        print(f"{nm:>16}{b['roi']:>10.1f}%{s['roi']:>10.1f}%{d:>+10.1f}"
              f"{f'[{d-z*sd:+.1f},{d+z*sd:+.1f}]':>21}")
    print("　★③両側の裾")
    for lab, fn in (("高い側 ≥1/R", lambda rat, o: rat >= th),
                    ("低い側 ≤R", lambda rat, o: rat <= rate)):
        res = roi(rows, fn)
        if res:
            ci = "[%.1f,%.1f]" % (res["roi"] - res["z"] * res["se"],
                                  res["roi"] + res["z"] * res["se"])
            print(f"{lab:>16}  ROI {res['roi']:>6.1f}%  {ci}")
    allrat = np.concatenate([r[1] for r in rows])
    edges = np.quantile(allrat, np.linspace(0, 1, NDEC + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    prev, mono, vals = None, 0, []
    for d in range(NDEC):
        res = roi(rows, lambda rat, o, lo=edges[d], hi=edges[d + 1]: (rat >= lo) & (rat < hi))
        if res is None:
            continue
        vals.append(f"{res['roi']:.0f}")
        if prev is not None and res["roi"] > prev:
            mono += 1
        prev = res["roi"]
    print(f"　★④単調性 十分位ROI {' → '.join(vals)}%"
          f"　上がった回数 **{mono}/{len(vals)-1}**")
    print("　⚠**④の単調性は半分自動**（ROI ∝ p_true/q_pool）。**100%超の証拠ではない**")
    return True


def selftest():
    """★データ無しでも回せる集約の検算。**(124)の「寝かせたスクリプトは腐る」対策**。"""
    nums, n = [1, 2, 3, 4], 4          # 4頭立て → 枠は1頭1枠
    od = {}
    v = 0
    for i in nums:
        for j in nums:
            for m in nums:
                if len({i, j, m}) < 3:
                    continue
                v += 1
                od["%02d%02d%02d" % (i, j, m)] = 1.0 / v     # 重み = v
    ag = aggregate(od, nums, n)
    assert len(od) == 24, len(od)
    tot = sum(od and (1.0 / o) for o in od.values())
    ok = True

    def chk(lab, got, want):
        nonlocal ok
        good = abs(got - want) < 1e-9
        ok &= good
        print(f"  {'★OK ' if good else '⚠NG '}{lab}: {got:.6f} vs {want:.6f}")

    # ① 各券種の総和は三連単の総和に一致する（取りこぼしが無い）
    for kind in ("馬単", "三連複", "馬連", "枠連"):
        chk(f"総和の保存 {kind}", sum(ag[kind].values()), tot)
    # ② 馬単(i,j) は 3着を潰した和
    want = sum(1.0 / od["%02d%02d%02d" % (1, 2, m)] for m in nums if m not in (1, 2))
    chk("馬単(1,2) = Σ_k 三連単(1,2,k)", ag["馬単"][(1, 2)], want)
    # ③ 馬連{i,j} = 馬単(i,j) + 馬単(j,i)
    chk("馬連{1,2} = 馬単(1,2)+馬単(2,1)", ag["馬連"][(1, 2)],
        ag["馬単"][(1, 2)] + ag["馬単"][(2, 1)])
    # ④ 三連複{i,j,k} は6通りの並べ替えの和
    want = sum(1.0 / od["%02d%02d%02d" % p]
               for p in ((1, 2, 3), (1, 3, 2), (2, 1, 3),
                         (2, 3, 1), (3, 1, 2), (3, 2, 1)))
    chk("三連複{1,2,3} = Σ_6通り", ag["三連複"][(1, 2, 3)], want)
    # ⑤ 4頭立ては1頭1枠なので 枠連 == 馬連
    chk("枠連{1,2} == 馬連{1,2}（4頭立て）", ag["枠連"][(1, 2)], ag["馬連"][(1, 2)])
    # ⑥ 3着だけが違う組は馬単で必ず潰れる（順序は保たれる）
    chk("馬単(2,1) ≠ 馬単(1,2)（順序は潰さない）",
        1.0 if ag["馬単"][(2, 1)] != ag["馬単"][(1, 2)] else 0.0, 1.0)
    print(f"\n{'★自己テスト全通過' if ok else '⚠⚠自己テストに失敗がある'}")
    return ok


def main():
    if "--selftest" in sys.argv:
        sys.exit(0 if selftest() else 1)
    y0 = int(sys.argv[1]) if len(sys.argv) > 1 else 2015
    print("(161) 三連単の板で**馬単そのもの**を測る（構造仮説の最鋭の検定）")
    print("★q の出どころ: **三連単の板を厳密集約**（Harville不使用）")
    print("★q_pool の出どころ: **各券種の板**")
    races = {r["rid"]: r for r in load_races() if r["year"] >= y0}
    ub, wb = load_board(4, 4), load_boards()
    print(f"　レース {len(races):,} / 馬連の板 {len(ub):,} / 枠連の板 {len(wb):,}")

    from nk_odds_bulk import iter_records
    from nk_parse import nk_raceid
    boards = {"馬単": load_board(6, 4), "三連複": load_board(7, 6),
              "馬連": ub, "枠連": wb}
    # ★三連単の板は巨大なので **1レースずつ読んで即集約し、生の板は捨てる**
    rows = {k: [] for k in boards}
    ctrl_rows = []                       # 対照 馬連→枠連
    seen = 0
    for rec in iter_records(8):
        rid = nk_raceid(rec["race_id"])
        r = races.get(rid) if rid else None
        if r is None or not r.get("wakuren"):
            continue
        rl = realized(r)
        if rl is None:
            continue
        a, b, c = rl
        n, nums = r["n"], [u for u, _, _ in r["horses"]]
        if a not in nums or b not in nums or c not in nums:
            continue
        od = {k: (v[0] if isinstance(v, (list, tuple)) else v)
              for k, v in rec["odds"].items()}
        od = {k: float(v) for k, v in od.items() if v and float(v) > 0}
        if not od:
            continue
        seen += 1
        ag = aggregate(od, nums, n)
        real = {"馬単": (a, b), "三連複": tuple(sorted((a, b, c))),
                "馬連": tuple(sorted((a, b))),
                "枠連": tuple(sorted((waku_of(a, n), waku_of(b, n))))}
        for kind in boards:
            bd = boards[kind].get(rid)
            if not bd:
                continue
            if kind in ("馬単", "三連複"):     # 板のキーが文字列なのでタプル→文字列に直す
                bd = {tuple(int(k[i:i + 2]) for i in range(0, len(k), 2)): o
                      for k, o in bd.items()}
                if kind == "三連複":
                    bd = {tuple(sorted(k)): o for k, o in bd.items()}
            v = payoff(r, PAYKEY[kind], list(real[kind]))
            if not v or v <= 0:
                continue
            row = rows_for(kind, ag[kind], bd, real[kind], v, r["year"])
            if row:
                rows[kind].append(row)
        # ★対照: 馬連の板→枠（(158)と同一の作り方）
        bdu, bdw = ub.get(rid), wb.get(rid)
        if bdu and bdw:
            agw = defaultdict(float)
            for k, o in bdu.items():
                if len(k) != 4 or not k.isdigit() or o <= 0:
                    continue
                i, j = int(k[:2]), int(k[2:])
                if i in nums and j in nums and i != j:
                    agw[tuple(sorted((waku_of(i, n), waku_of(j, n))))] += 1.0 / o
            v = payoff(r, "枠連(人気順)", list(real["枠連"]))
            if v and v > 0:
                row = rows_for("枠連", agw, bdw, real["枠連"], v, r["year"])
                if row:
                    ctrl_rows.append(row)
    print(f"　三連単の板と突き合わせできた {seen:,} レース")

    print("\n" + "=" * 96)
    print("■ ⚠ゲート1（判定基準32）: 対照 馬連→枠連 が(158)の116.4%を±5ptで再現するか")
    cm = table("対照 馬連→枠連", ctrl_rows, R["枠連"]) if ctrl_rows else None
    if cm is None:
        sys.exit("\n⚠⚠対照が作れない。**何も読まない**。")
    d = cm["roi"] - KNOWN_CTRL
    ok = abs(d) <= TOL
    print(f"\n★対照 {cm['roi']:.1f}% vs (158)の {KNOWN_CTRL}%　差 {d:+.1f}pt"
          f" → **{'★立った' if ok else '⚠⚠落ちた'}**")
    if not ok:
        print("⚠⚠**ゲート1が落ちた。以下を読まない**（判定基準32）。")
        return

    mains = {}
    for kind in ("馬単", "三連複", "馬連", "枠連"):
        if not rows[kind]:
            print(f"\n■ 三連単→{kind}: 突き合わせできたレースが無い")
            continue
        mains[kind] = table(f"三連単→{kind}（厳密）", rows[kind], R[kind])

    print("\n" + "=" * 96)
    print("■ ★★本命 A（三連単→馬単）に(160)の対照を当てる")
    if rows["馬単"]:
        controls("三連単→馬単", rows["馬単"], R["馬単"])

    print("\n" + "=" * 96)
    print("★★事前登録した予測との突き合わせ（**当たっても外れても消さない**）")
    pred = [("1. 馬単は100%前後（構造仮説が正しければ跨ぐ）", "馬単", None),
            ("2. 三連複は陰性（最も売れているプール側）", "三連複", None),
            ("3. 三連単→馬連は(158)Aの42.8%を再現するはず", "馬連", 42.8),
            ("4. 三連単→枠連は116%前後（141/158と並ぶ）", "枠連", 116.0)]
    for lab, kind, expect in pred:
        m = mains.get(kind)
        if m is None:
            print(f"　{lab} → **測れず**")
            continue
        s = f"　{lab} → **実測 {m['roi']:.1f}%**"
        if expect is not None:
            s += f"（予測 {expect:.1f}%・差 {m['roi']-expect:+.1f}pt）"
        print(s)
    print("\n⚠⚠**陽性でも確定オッズのオラクル**。**張れる時点は(148)＝時系列オッズ待ちのまま**。")
    print("⚠**このスクリプトは運用に何も足さない**。")


if __name__ == "__main__":
    main()
