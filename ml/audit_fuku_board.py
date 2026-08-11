"""(124) ★★★複勝の板で、(99)②が「原理的に無理」とした**層別したDの計算**を可能にする

★(99)②が残した構造的限界（そのまま引用）
　> **層別した複勝プールのDは、過去の複勝オッズが無い限り計算できない**
　> （来なかった馬の p_pool が観測できない）。**層別で複勝を見るときはDではなく実測ROIを使う**。
　理由: 複勝の払戻は**着内に来た馬にしか公表されない**。だから d は「3着以内に来たレース」でしか
　作れず、馬の属性で層別した瞬間「来なかった場合」が抜け落ちる。1番人気は約63%しか来ないので、
　来た63%だけを見ればDが跳ね上がる。実際(99)②では **〜1.6倍で D=+0.2395＝必要量の107%**
　と出たが、同じ帯の実測ROIは93〜95%で**両立しなかった**。

★**(113)でその「過去の複勝オッズ」が全期間で取れるようになった**（type=2）。限界が外れる。

★★測り方を変える必要がある（ここが肝）
　従来の d は「払戻を受け取った側」しか見ない。板があるなら**二値の対数スコア**で書ける:
　　d_i = y_i·log(q_i / q_pool_i) + (1−y_i)·log((1−q_i) / (1−q_pool_i))
　　　y_i = その馬が3着以内に来たか / q_i = 我々の3着以内確率（λ補正Harville）
　　　q_pool_i = 複勝プールの含意確率（板から）
　**これは来なかった馬についても計算できる**ので、層別しても偏らない。
　⚠板の複勝オッズは **[下限, 上限] の範囲**（他の入着馬しだいで払戻が変わるため）。
　　下限を使うと q_pool を過大評価する。**下限・中点の両方で出して、結論が変わらないか見る**。
　　あわせて **Σq_pool = 3 に正規化した版**も出す（3頭が着内に入るので）。

★★事前登録（測る前に宣言）
　1. 層は **単勝オッズ帯**（〜1.6 / 1.6-2.5 / 2.5-4 / 4-7 / 7-15 / 15-40 / 40〜）の7区分。
　　 (94)③(99)②と同じ切り方に合わせる。**後から増やさない**。
　2. q_pool の作り方は **下限そのまま / 下限をΣ=3に正規化 / 中点をΣ=3に正規化** の3通りを全部出す。
　　 **3通りで結論が変わったら「決着していない」と書く**（都合の良い1つを選ばない）。
　3. **判定**: 必要量 **0.2231**。層のDの99%CI下端がこれを超えたら「その帯は儲かる」。
　4. **★整合性検査（(99)②と同じ形で必ず発火させる）**: 同じ帯の**実測ROI**を並べる。
　　 **Dが必要量を超えているのにROIが100%未満なら、どちらかが間違っている**。
　　 (99)②ではこの検査でDの側の誤りを見つけた。今回は板があるので**Dの側が正しいはず**だが、
　　 　合わなければ**また道具を疑う**。結論を先に決めない。
　5. **予想**: **どの帯も必要量に届かない**。理由は実測ROIが93〜96.8%だから。
　　 (99)②の +0.2395 は消えて、**+0.01〜+0.03程度に落ち着く**と予想する。
　　 ⚠(117)(122)で予想を外している。この予想も外れる前提で読むこと。

実行（板を集めたあと）: python3 ml/audit_fuku_board.py [開始年(既定2015)]
　　★先にMacで: python3 ml/nk_odds_bulk.py --type 2    （約18時間・放置可）
"""
import math
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "ml")
from audit_crosspool import load_races, payoff, probs, zq
from audit_lbs import build_matrix, fit_lambda, stage_w
from audit_fuku_lbs import top3_probs
from nk_parse import nk_raceid

PAYBACK = 0.80
NEED = -math.log(PAYBACK)                      # 0.2231
BANDS = [(0, 1.6), (1.6, 2.5), (2.5, 4), (4, 7), (7, 15), (15, 40), (40, 1e9)]


def mci(x, alpha=0.01):
    x = np.asarray(x, float)
    if len(x) < 2:
        return float("nan"), float("nan"), float("nan")
    m = x.mean()
    se = x.std(ddof=1) / math.sqrt(len(x))
    return m, m - zq(alpha) * se, m + zq(alpha) * se


def load_boards():
    """race_id(8桁) → {馬番: (下限, 上限)}。type=2 の板から。"""
    from nk_odds_bulk import iter_records
    out = {}
    for rec in iter_records(2):
        rid8 = nk_raceid(rec["race_id"])
        if not rid8:
            continue
        # ★type=2 は `[下限, 上限]` で保存されている（nk_odds_combo.RANGE_TYPES）。
        # 　古い形（float 1つ）で保存された分にも耐えるようにしておく。
        out[rid8] = {int(k): ((float(v[0]), float(v[1])) if isinstance(v, (list, tuple))
                              else (float(v), float(v)))
                     for k, v in rec["odds"].items()}
    return out


def qpool_variants(od, nums):
    """3通りの q_pool を返す。**どれか1つを選ばない**（事前登録2）。"""
    lo = np.array([od[u][0] for u in nums], float)
    hi = np.array([od[u][1] for u in nums], float)
    mid = (lo + hi) / 2.0
    raw = PAYBACK / lo
    n3 = raw * (3.0 / raw.sum()) if raw.sum() > 0 else raw
    m3 = (PAYBACK / mid)
    m3 = m3 * (3.0 / m3.sum()) if m3.sum() > 0 else m3
    return {"下限そのまま": raw, "下限をΣ=3に": n3, "中点をΣ=3に": m3}


def main():
    y0 = int(sys.argv[1]) if len(sys.argv) > 1 else 2015
    boards = load_boards()
    if not boards:
        sys.exit("複勝の板が無い。Macで `python3 ml/nk_odds_bulk.py --type 2` を回して push すること。")
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

    rows = []
    for r in races:
        yy = r["year"]
        if yy < y0 or not lam.get(yy):
            continue
        od = boards.get(r["rid"])
        if not od:
            continue
        hs = r["horses"]
        nums = [u for u, _, _ in hs]
        if any(u not in od or od[u][0] <= 0 for u in nums):
            continue
        p = probs(hs)
        l2, l3 = lam[yy]
        t3 = top3_probs(p, l2, l3)
        y = np.array([1.0 if f <= 3 else 0.0 for _, _, f in hs])
        if y.sum() != 3:
            continue                       # 同着などは落とす（(99)と同じ扱い）
        qs = qpool_variants(od, nums)
        pay = {u: (payoff(r, "複勝", [u]) or 0.0) for u in nums}
        for k, u in enumerate(nums):
            for tag, qp in qs.items():
                qq = min(max(float(qp[k]), 1e-6), 1 - 1e-6)
                q = min(max(float(t3[k]), 1e-6), 1 - 1e-6)
                d = (y[k] * math.log(q / qq)
                     + (1 - y[k]) * math.log((1 - q) / (1 - qq)))
                rows.append({"year": yy, "odds": hs[k][1], "umaban": u, "tag": tag,
                             "d": d, "y": y[k], "pay": pay[u]})

    df = pd.DataFrame(rows)
    print(f"(124) 複勝の板で層別Dを計算する（{y0}年以降・板のあるレース {len(boards):,}）")
    print(f"★(99)②が『過去の複勝オッズが無い限り計算できない』とした部分。必要量 {NEED:.4f}\n")

    for tag in ("下限そのまま", "下限をΣ=3に", "中点をΣ=3に"):
        g0 = df[df["tag"] == tag]
        if g0.empty:
            continue
        m0, lo0, hi0 = mci(g0["d"])
        print(f"■ q_pool = {tag}   全体 D={m0:+.4f} [{lo0:+.4f},{hi0:+.4f}]")
        print(f"{'オッズ帯':<12}{'頭数':>9}{'D':>10}{'99%CI':>22}"
              f"{'着内率':>8}{'実測ROI':>9}{'判定':>7}")
        for a, b in BANDS:
            g = g0[(g0["odds"] >= a) & (g0["odds"] < b)]
            if len(g) < 300:
                continue
            m, lo, hi = mci(g["d"])
            roi = g["pay"].sum() / (len(g) * 100) * 100
            mark = "★儲かる" if lo > NEED else ""
            lab = f"{a}-" + ("" if b > 1e8 else str(b))
            ci = "[" + format(lo, "+.4f") + "," + format(hi, "+.4f") + "]"
            hit = g["y"].mean()
            print(f"{lab:<12}{len(g):>9,}{m:>+10.4f}{ci:>22}"
                  f"{hit:>8.1%}{roi:>9.1f}{mark:>7}")
        print()

    print("=" * 92)
    print("★読み方（事前登録のとおり）")
    print("  ・3通りで結論が変われば **決着していない**。都合の良い1つを選ばない。")
    print("  ・★儲かる が付いた帯があっても、**同じ行の実測ROIが100%未満なら矛盾**。")
    print("    (99)②はその矛盾でDの側の誤りを見つけた。今回も合わなければ**また道具を疑う**。")
    print("  ・どの帯も届かなければ、(99)②の +0.2395 は**観測の偏りだった**と確定する。")


if __name__ == "__main__":
    main()
