"""(144) ★★★「締切に向けて上がる馬を推測できればいい」を詰める（2026-08-13・ユーザー発案）

★ユーザーの問い
　> その確定オッズで上がってくる馬を推測できればいいのでは？
　> 一定の購入者は締切に向けて分かって買っているわけだから

★★機構は正しい。**「締切前の金は情報を持っている」は既に実証済み**
　(109)① が `q ∝ p_final·(p_final/p_prev)^γ` の γ を当てはめ、**発走に近づくほど単調に大きい**と測った:
　　前日22時 +0.15 ／ 当日9時 +0.25 ／ 30分前 +0.45 ／ **10分前 +0.90**
　→ **smart money は実在する**。ユーザーの見立ては当たっている。

★★★だが「予測できれば勝てる」には**既に上界がある**（判定基準30を自分に当てて範囲を明記する）
　**予測できる最良の場合＝確定オッズを完全に言い当てた場合**。それは**(141)(142)そのもの**。
```
　クラス: 「両プールの確定オッズを予測して、その予測で買う」
　その中の最良: 予測が完全に当たる＝確定オッズを知っている＝(141)
　実測（(142)）: 2015-2020 → ROI 129.9% [107.5,152.4] ★
　　　　　　　　 **2021-2026 → ROI 107.1% [91.0,123.2] ← 100%を含む**
```
　→ ★**直近6年では、確定オッズを完全に予測できたとしても 100% を示せない**。
　　 **予測は必ず不完全**なので、**実現値はこれ以下**。
　⚠**この上界が効く範囲を明記する（判定基準30）**: 「**確定オッズの予測だけを使う**」戦略に対してのみ。
　　 **オッズ以外の情報（モデル等）を足せばこの上界の外に出られる**——ただしそちらは
　　 (100)〜(107)で「モデルの寄与は重み0.9%」と繰り返し測られている。

★★そこで、このスクリプトが実際に測るのは**運用に直結する別の量**
　**「発走X分前に買うと、確定に比べてどれだけ損をしているか」**。
　★現運用は**発走30分前〜10分前**に判定している（(112運用②)）。
　　**その選択が最適かを、一度も数字で確かめていない**。
　　(109)は「動きを信号にできるか」を測ったが、**「いつ買うのが良いか」は測っていない**。

★★事前登録
　1. 時点は **前日21時 / 当日9時 / 60 / 30 / 20 / 10 / 5分前 / 確定**。**後から増やさない**。
　2. 量は **単勝の対数スコア**（勝ち馬の log p）と、**枠連の D**（λ補正Harville→枠 vs 枠連の板）。
　　 **確定を基準（0）とした差**で出す。**負なら「その時点で買うと損」**。
　3. **判定**: 10分前と5分前の差が **単勝で 0.01 nat 未満**なら「**10分前で十分**」。
　　 それ以上なら「**もっと引きつけるべき**」＝**運用を変える**。
　4. ⚠**標本は396レース**。CIは±0.03程度。**0.01の差は判定できない可能性が高い**ので、
　　 **点推定の並びと単調性も併せて見る**。**先に書いておく**。
　5. **予想**: ★**当てにしてよい予想は持っていない**（類推はこの3日で4連敗している）。

実行: python3 ml/audit_late_money.py
"""
import math
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "ml")
from audit_capacity_d import mkt_waku_dist
from audit_cond_split import load_boards
from audit_crosspool import load_races, zq
from audit_crosspool2 import realized
from audit_lbs import build_matrix, fit_lambda
from odds_ts import load_dir, odds_at
from waku_umatan import waku_of

WHENS = [("prev", 21, 0), ("day", 9, 0), ("before", 60), ("before", 30),
         ("before", 20), ("before", 10), ("before", 5), ("final",)]
LAB = {("prev", 21, 0): "前日21時", ("day", 9, 0): "当日9時", ("before", 60): "60分前",
       ("before", 30): "30分前", ("before", 20): "20分前", ("before", 10): "10分前",
       ("before", 5): "5分前", ("final",): "確定"}


def mci(x, alpha=0.01):
    x = np.asarray(x, float)
    if len(x) < 2:
        return float("nan"), float("nan"), float("nan")
    m = x.mean()
    se = x.std(ddof=1) / math.sqrt(len(x))
    return m, m - zq(alpha) * se, m + zq(alpha) * se


def main():
    ts = load_dir("data/odds_ts")
    wb = load_boards()
    races = {r["rid"]: r for r in load_races()}
    P, i1, i2, i3, _ = build_matrix(list(races.values()), 2015)
    l2 = fit_lambda(P, i1, i2)

    win_ls = {w: [] for w in WHENS}     # 単勝: log p(勝ち馬)
    wak_d = {w: [] for w in WHENS}      # 枠連: log q − log q_pool
    for rid, rec in ts.items():
        r = races.get(rid)
        if r is None:
            continue
        rl = realized(r)
        if rl is None:
            continue
        a, b, _ = rl
        n = r["n"]
        nums = [u for u, _, _ in r["horses"]]
        if a not in nums or len(nums) != rec["odds"].shape[1]:
            continue
        ia = nums.index(a)
        W = wb.get(rid)
        key = tuple(sorted((waku_of(a, n), waku_of(b, n))))
        got = {}
        for w in WHENS:
            o = odds_at(rec, w)
            if o is None or np.isnan(o).any() or (o <= 0).any():
                continue
            p = 1.0 / np.asarray(o, float)
            p = p / p.sum()
            got[w] = p
        if len(got) < len(WHENS):
            continue                     # ★全時点そろうレースだけ（対応ありにする）
        for w, p in got.items():
            win_ls[w].append(math.log(max(p[ia], 1e-300)))
        if W and r.get("wakuren") and b in nums:
            ok = True
            tmp = {}
            for w, p in got.items():
                hs = [(u, float(pi), 0) for u, pi in zip(nums, p)]
                md = mkt_waku_dist({"horses": hs, "n": n}, p, l2)
                if not md:
                    ok = False
                    break
                keys = [k for k in sorted(md) if k in W]
                if key not in keys or len(keys) < 3:
                    ok = False
                    break
                inv = np.array([1.0 / W[k] for k in keys])
                qp = inv / inv.sum()
                qq = np.array([md[k] for k in keys])
                qq /= qq.sum()
                j = keys.index(key)
                tmp[w] = math.log(qq[j]) - math.log(qp[j])
            if ok:
                for w, v in tmp.items():
                    wak_d[w].append(v)

    nw = len(win_ls[("final",)])
    nk = len(wak_d[("final",)])
    print(f"(144) いつ買うのが良いか（単勝 {nw}レース / 枠連 {nk}レース・**全時点そろうものだけ**）")
    print("★機構は実証済み: (109)①で γ は発走に近づくほど単調に大きい（前日+0.15 → 10分前+0.90）")
    print("★★だが『確定を完全に予測できた場合』の上界は既に出ている →")
    print("　 (142) 2021-2026 は **ROI 107.1% [91.0,123.2]＝100%を含む**。")
    print("　 **完全予測でも直近では示せない**。予測は不完全なので実現値はこれ以下。\n")

    fin_w = np.array(win_ls[("final",)])
    print("■ 単勝の対数スコア（確定を0とした差。**負＝その時点で買うと損**）")
    print(f"{'時点':>10}{'差':>10}{'99%CI':>22}{'確定との差の意味':>22}")
    for w in WHENS:
        d = np.array(win_ls[w]) - fin_w
        m, lo, hi = mci(d)
        ci = "[" + format(lo, "+.4f") + "," + format(hi, "+.4f") + "]"
        print(f"{LAB[w]:>10}{m:>+10.4f}{ci:>22}{'（基準）' if w == ('final',) else '':>22}")

    if nk > 30:
        fin_k = np.array(wak_d[("final",)])
        print("\n■ 枠連の D（λ補正Harville→枠 vs 枠連の板。確定を0とした差）")
        print(f"{'時点':>10}{'D':>10}{'確定との差':>12}{'99%CI':>22}")
        for w in WHENS:
            arr = np.array(wak_d[w])
            d = arr - fin_k
            m, lo, hi = mci(d)
            ci = "[" + format(lo, "+.4f") + "," + format(hi, "+.4f") + "]"
            print(f"{LAB[w]:>10}{arr.mean():>+10.4f}{m:>+12.4f}{ci:>22}")

    print("\n" + "=" * 92)
    print("★読み方（事前登録のとおり）")
    print("  ・10分前と5分前の差が**単勝で0.01 nat未満**なら『10分前で十分』＝運用は変えない。")
    print("  ・それ以上なら**もっと引きつけるべき**＝運用を変える。")
    print("  ⚠396レースなのでCIは±0.03級。**0.01は判定できない**（事前に書いた）。")
    print("  ★**ユーザーの筋（締切前の金は情報を持つ）は正しい**。だが利用の上界は(142)で")
    print("    **直近6年 107.1% [91.0,123.2]**＝**完全予測でも100%を示せない**。")


if __name__ == "__main__":
    main()
