"""(162) ★★★**払戻率そのものを板から実測する**。**コードの定数を疑う**。

★★なぜ書くか——**(161)でゲート2が落ちた**。原因は板でも三連単でもなく、
　**`ml/audit_crosspool.py` の `LINE` が `"馬単": 0.775` としていたこと**の疑いが濃い。
　**馬単の板が言う払戻率は74.91%で、三連複(75.0%)の74.94%とほぼ同じ**だった。
　⚠**払戻率はプロジェクト全体の「必要量 |log R|」の土台**なので、**記憶ではなく実測で決める**。

測り方（**2通り出して突き合わせる。片方だけでは板と配当のどちらがずれたか分からない**）
　★①**板が含意する払戻率** `R_implied = 1 / Σ_k (1/odds_k)`。**払戻データを一切使わない**。
　★②**比例買いの実測ROI**（組kに `1/odds_k` 単位）。**実配当を使う**。
　　 **①と②が一致すれば板と配当は整合**。**食い違えば配当データ側を疑う**（(160)の恒等式）。

★**恒等式（(160)で導いたもの）**: 板に比例して賭けると
　**費用 `Σ_k 1/odds_k`、払戻 `(1/odds_w)·odds_w = 1` → ROI = 1/Σ_k(1/odds_k) = R_implied**。
⚠**均等買いは別物**（大穴側に重みが寄る）。**参考として並べるが判定には使わない**。

★**事前登録の予測**: **馬単だけが定数と食い違い、実測は0.750に寄る**。
　⚠**外れたら「定数が誤り」という見立てのほうを捨てる**。

実行: python3 ml/audit_takeout.py [開始年(既定2015)]
"""
import sys

import numpy as np

sys.path.insert(0, "ml")
from audit_cond_split import load_boards
from audit_crosspool import LINE, load_races, payoff
from audit_crosspool2 import realized
from audit_overlay_all import load_board
from waku_umatan import waku_of

# (type, キー長, 表示名, payoff用の名前, コードが言う払戻率)
POOLS = [
    (3, None, "枠連", "枠連(人気順)", LINE["枠連(人気順)"]),
    (4, 4, "馬連", "馬連", LINE["馬連"]),
    (6, 4, "馬単", "馬単", LINE["馬単"]),
    (7, 6, "三連複", "三連複", LINE["三連複"]),
    (8, 6, "三連単", "三連単", LINE["三連単"]),
]


def realkey(kind, r, rl):
    a, b, c = rl
    n = r["n"]
    if kind == "枠連":
        return tuple(sorted((waku_of(a, n), waku_of(b, n))))
    if kind == "馬連":
        return tuple(sorted((a, b)))
    if kind == "馬単":
        return (a, b)
    if kind == "三連複":
        return tuple(sorted((a, b, c)))
    return (a, b, c)                      # 三連単


def tup(k):
    return tuple(int(k[i:i + 2]) for i in range(0, len(k), 2))


def measure(t, klen, kind, paykey, races):
    """(R_implied中央値, 比例買いROI, 均等買いROI, レース数) を返す。

    ★三連単は板が巨大なので **1レースずつ読んで即集計する**（(161)と同じ理由）。
    """
    ors, cost, ret, fcost, fret, nr = [], 0.0, 0.0, 0.0, 0.0, 0
    if kind == "枠連":
        src = ((rid, bd) for rid, bd in load_boards().items())
    elif t == 8:
        from nk_odds_bulk import iter_records
        from nk_parse import nk_raceid

        def gen():
            for rec in iter_records(8):
                rid = nk_raceid(rec["race_id"])
                if not rid:
                    continue
                d = {}
                for k, v in rec["odds"].items():
                    if len(k) != klen or not k.isdigit():
                        continue
                    o = v[0] if isinstance(v, (list, tuple)) else v
                    if o and float(o) > 0:
                        d[tup(k)] = float(o)
                if d:
                    yield rid, d
        src = gen()
    else:
        src = ((rid, {tup(k): o for k, o in bd.items()})
               for rid, bd in load_board(t, klen).items())

    for rid, bd in src:
        r = races.get(rid)
        if r is None:
            continue
        if kind in ("枠連",) and not r.get("wakuren"):
            continue
        rl = realized(r)
        if rl is None:
            continue
        nums = {u for u, _, _ in r["horses"]}
        if not set(rl) <= nums:
            continue
        key = realkey(kind, r, rl)
        if key not in bd:
            continue
        v = payoff(r, paykey, list(key))
        if not v or v <= 0:
            continue
        odds = np.array(list(bd.values()), float)
        inv = 1.0 / odds
        ors.append(float(inv.sum()))
        # ★比例買い: 組kに 100/odds_k 円
        cost += 100.0 * inv.sum()
        ret += 100.0 / bd[key] * (v / 100.0)
        # 参考: 均等買い
        fcost += 100.0 * len(odds)
        fret += v
        nr += 1
    if not nr:
        return None
    return float(np.median(ors)), 100.0 * ret / cost, 100.0 * fret / fcost, nr


def main():
    y0 = int(sys.argv[1]) if len(sys.argv) > 1 else 2015
    print(f"(162) ★払戻率そのものを板から実測する（{y0}年以降）")
    print("　★①R_implied = 1/Σ(1/odds)（**払戻データを使わない**）")
    print("　★②比例買いの実測ROI（**実配当を使う**）。①と②が一致すれば板と配当は整合\n")
    races = {r["rid"]: r for r in load_races() if r["year"] >= y0}
    print(f"{'券種':>8}{'レース':>9}{'Σ1/odds':>11}{'①R_implied':>13}"
          f"{'②比例買い':>12}{'コードの定数':>13}{'差(②−定数)':>13}{'均等買い':>10}")
    bad = []
    for t, klen, kind, paykey, rate in POOLS:
        res = measure(t, klen, kind, paykey, races)
        if res is None:
            print(f"{kind:>8}  データ無し")
            continue
        med, prop, flat, nr = res
        d = prop - 100 * rate
        mark = "★" if abs(d) <= 1.0 else "⚠⚠"
        print(f"{kind:>8}{nr:>9,}{med:>11.4f}{100/med:>12.2f}%"
              f"{prop:>11.2f}%{100*rate:>12.1f}%{d:>+12.2f}{mark}{flat:>9.1f}%")
        if abs(d) > 1.0:
            bad.append((kind, prop, 100 * rate))

    print("\n" + "=" * 96)
    if not bad:
        print("★**すべての券種でコードの定数と実測が±1.0pt以内**。**定数は正しい**。")
    else:
        for kind, prop, rate in bad:
            print(f"⚠⚠**{kind}: 実測 {prop:.2f}% vs コードの定数 {rate:.1f}%"
                  f"（差 {prop-rate:+.2f}pt）**")
        print("★**±1ptに収まらない券種は、`ml/audit_crosspool.py` の `LINE` を疑う**。")
        print("⚠**払戻率は「必要量 |log R|」と「利益に要る比 1/R」の両方の土台**——"
              "**直すと過去の判定の閾値が動く**（判定基準29で遡って当て直すこと）。")
    print("\n⚠**①と②が食い違う券種があれば、そこは板と配当データの整合を疑う**"
          "（(160)の恒等式が成り立たない）。")


if __name__ == "__main__":
    main()
