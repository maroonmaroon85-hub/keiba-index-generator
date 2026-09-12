"""(154) ★★★運用の「枠連スコア下位40%除外」は**張れる時点でも同じ判定か**

★★なぜ穴が残っていたか（2026-08-16の監査）
　(117)で**運用を20%除外→40%除外に変えた**。**いま実際に使っている**。
　⚠**`waku_score` は連続量**（買う枠組の確率の積）**で、その閾値でレースを外す**。
　★**(152)と同じ形**——**レース内の順位ではなく、レースをまたぐ閾値**。
　**qは単勝オッズ全体の関数なので締切に向けて動く**。**判定基準31は守ってくれない**。
　→ **一度も測っていない**。**(152)を書いたときに同じ形が運用にもう1つあると気づいた**。

★★事前登録（**測る前に書いている**。(152)と同じ形なので同じ骨格）
　1. **時点は 前日21時 / 当日9時 / 30分前 / ★10分前 / 確定**。**後から増やさない**。
　2. **除外率は 10 / 20 / 30 / ★40（現行） / 50%**。**後から増やさない**。
　3. **閾値は「全レースの確定オッズ」で決めた絶対値**。**この396レースの分位ではない**。
　4. ★★**陽性対照（判定基準32）**: **(144)の「10分前の単勝は確定より0.0832 nat 悪い」**。
　　 **±0.02 nat で再現しなければ以下を読まない**。**(152)で +0.0756 と再現できている**。
　5. **主判定**: ★**現行40%で、10分前の「除外する／しない」が確定と一致する割合**。
　　 **一致率が高ければ運用は安全**。**低ければ「除外の効果」は確定オッズのオラクル**。
　6. ⚠**ROIは出さない**。**396レースでは判定不能**（(152)と同じ理由）。
　7. ★**予想**: **当てにしてよい予想は持っていない**。
　　 **恒等式から言えることだけ**: **除外率が真ん中(40〜50%)ほど閾値付近の密度が高い**ので、
　　 **同じ揺れでも入れ替わりは多くなる**。**10%や90%より不利**。**向きだけは分かっている**。

⚠★**この実験は「除外が効くか」を測っていない**（それは(117)(139)）。
　**「除外する／しないの判定が時点で変わるか」だけ**。**混同しないこと**。

★★★実行済みの結果（2026-08-16・330レース）**運用のバグを見つけた**
　★①陽性対照: 10分前の対数スコア差 **+0.0740**（既知 +0.0832）→ **立った**。

■②「除外する／しない」が確定と一致する割合（**現行=40%**）
| 時点 | 確定で除外 | その時点で除外 | 判定一致 | 誤って外す |
|---|---|---|---|---|
| 前日21時 | 118 | **206** | 60.1% | **34.6%** |
| ★**当日9時（いまの運用）** | 131 | **229** | **66.1%** | ★**31.8%** |
| 30分前 | 131 | 197 | 75.2% | 22.4% |
| 10分前 | 131 | **173** | 79.4% | 16.7% |

■③スコアそのものの動き（比＝確定/その時点）
| 時点 | 比の中央値 | \|変化\|>10% | 順位相関ρ |
|---|---|---|---|
| 前日21時 | **1.271** | 82.4% | 0.452 |
| 当日9時 | **1.267** | 81.8% | 0.733 |
| 30分前 | 1.194 | 74.5% | 0.808 |
| 10分前 | **1.123** | 67.0% | 0.844 |

★★★**これは誤差ではなく系統的なずれ**——**朝のスコアは確定の約1/1.27＝21%低い**。
　**閾値は確定オッズ（学習用DS CSV）で作った分位**なのに、**発走前のオッズに当てている**
　（`train_prod.py:135` が `np.percentile(scores, k)` で作り、`predict_nk.py:113` が読む）。
　→ ★**「40%除外」の設定で、実際には約60%を除外している**。

★**実運用の記録で裏が取れた**（`data/reco/reco_*.json`）:
　39% / 70% / 53% / 57% / 70% / 63%（2026-08-01〜08-16）。**設定は40%**。

⚠**これが害かどうかは別問題**——(139)は除外率を上げるほど点推定が良い（40%で+0.0039、
　50%で+0.0045・単調）と実測している。**「買わない方向」なので損はしていない可能性が高い**。
　★**問題は「文書と実装が食い違っていること」と「(117)(139)の測定が真の40%（確定オッズ）で
　　行われたので、いま走っているものを記述していないこと」**。

★**直し方の候補（運用変更なのでユーザー判断）**:
　(a) **その日のレースの中での分位にする**（＝今日の下位40%を外す）。**分布ずれに強く、意図どおり**
　(b) 発走前オッズで分位を作り直す（**学習データに発走前オッズが要る**。いま396レースしかない）
　(c) 何もせず「実際は約60%除外」と文書に書く

⚠**この実験は「除外が効くか」を測っていない**（それは(117)(139)）。**判定の時点安定性だけ**。

実行: python3 ml/audit_excl_ts.py
"""
import math
import sys

import numpy as np

sys.path.insert(0, "ml")
from audit_crosspool import load_races, probs, zq
from audit_crosspool2 import realized
from odds_ts import load_dir, odds_at
from waku_umatan import bracket_probs, waku_of, waku_score, wakuren_buy

RATES = [0.10, 0.20, 0.30, 0.40, 0.50]
WHEN = [("前日21時", ("prev", 21, 0)), ("当日9時", ("day", 9, 0)),
        ("30分前", ("before", 30)), ("★10分前", ("before", 10)), ("確定", ("final",))]
KNOWN_GAP = 0.0832


def score_of(od, nums, n):
    """単勝オッズ → waku_score（運用と同じ計算）。取れなければ None。"""
    o = np.asarray(od, float)
    ok = np.isfinite(o) & (o > 0)
    if ok.sum() < 3:
        return None
    inv = np.where(ok, 1.0 / np.where(ok, o, 1.0), 0.0)
    p = inv / inv.sum()
    order = np.argsort(-p)
    ns = [nums[k] for k in order]
    bp = bracket_probs(ns, [p[k] for k in order], n)
    return float(waku_score(wakuren_buy(ns, n, 1), bp))


def main():
    ts = load_dir()
    if not ts:
        sys.exit("data/odds_ts が無い。")
    races = {r["rid"]: r for r in load_races()}

    # 事前登録3: 閾値は全レース(確定)で決めた絶対値
    allv = []
    for r in races.values():
        if not r.get("wakuren") or realized(r) is None:
            continue
        s = score_of([o for _, o, _ in r["horses"]], [u for u, _, _ in r["horses"]], r["n"])
        if s is not None:
            allv.append(s)
    allv = np.array(allv)
    cuts = {p: float(np.quantile(allv, p)) for p in RATES}

    per, gaps = {lab: {} for lab, _ in WHEN}, {lab: [] for lab, _ in WHEN}
    for rid in sorted(set(ts) & set(races)):
        r = races[rid]
        if not r.get("wakuren") or realized(r) is None:
            continue
        nums = [u for u, _, _ in r["horses"]]
        win = realized(r)[0]
        if win not in nums:
            continue
        wk = nums.index(win)
        for lab, w in WHEN:
            od = odds_at(ts[rid], w)
            if od is None or len(od) != len(nums):
                continue
            s = score_of(od, nums, r["n"])
            if s is None:
                continue
            per[lab][rid] = s
            o = np.asarray(od, float)
            ok = np.isfinite(o) & (o > 0)
            if ok.sum() < 3 or not ok[wk]:
                continue
            inv = np.where(ok, 1.0 / np.where(ok, o, 1.0), 0.0)
            gaps[lab].append(math.log(inv[wk] / inv.sum()))

    z = zq(0.01)
    print(f"(154) 「枠連スコア下位40%除外」は張れる時点でも同じ判定か（{len(per['確定']):,}レース）")
    print("　閾値は**全レースの確定オッズ**で決めた絶対値: "
          + " / ".join(f"{int(p*100)}%={cuts[p]:.4f}" for p in RATES) + "\n")

    print("■ ★①陽性対照（判定基準32）— (144)の +0.0832（(152)では +0.0756 で再現済み）")
    base = np.array(gaps["確定"])
    got = None
    for lab, _ in WHEN:
        v = np.array(gaps[lab])
        if lab == "確定" or len(v) < 2:
            continue
        c = min(len(v), len(base))
        d = base[:c] - v[:c]
        if lab == "★10分前":
            got = float(d.mean())
    ok = got is not None and abs(got - KNOWN_GAP) <= 0.02
    print(f"　10分前の対数スコア差 {got:+.4f}（既知 {KNOWN_GAP:+.4f}・差 {got-KNOWN_GAP:+.4f}）")
    print(f"　→ ★**①は立ったか: {'★立った' if ok else '⚠立っていない'}**")
    if not ok:
        print("　⚠**以下を読まないこと**（判定基準32）。")

    print("\n■ ★★②主判定 — 「除外する／しない」が確定と一致する割合")
    fin = per["確定"]
    print(f"{'時点':>10}{'除外率':>7}{'確定で除外':>10}{'時点で除外':>10}"
          f"{'判定一致':>9}{'誤って残す':>10}{'誤って外す':>10}")
    for lab, _ in WHEN:
        if lab == "確定":
            continue
        ids = sorted(set(per[lab]) & set(fin))
        if not ids:
            continue
        for p in RATES:
            c = cuts[p]
            f = np.array([fin[i] < c for i in ids])
            g = np.array([per[lab][i] < c for i in ids])
            mark = " ←現行" if abs(p - 0.40) < 1e-9 else ""
            print(f"{lab:>10}{int(p*100):>6}%{f.sum():>10}{g.sum():>10}"
                  f"{np.mean(f == g):>8.1%}{np.mean(f & ~g):>9.1%}{np.mean(~f & g):>9.1%}{mark}")

    print("\n■ ③スコアそのものの動き")
    print(f"{'時点':>10}{'n':>6}{'確定との比の中央値':>20}{'|変化|>10%':>11}{'順位相関ρ':>11}")
    for lab, _ in WHEN:
        if lab == "確定":
            continue
        ids = sorted(set(per[lab]) & set(fin))
        if len(ids) < 5:
            continue
        a = np.array([per[lab][i] for i in ids])
        b = np.array([fin[i] for i in ids])
        ra, rb = np.argsort(np.argsort(a)).astype(float), np.argsort(np.argsort(b)).astype(float)
        rel = b / np.maximum(a, 1e-12)
        print(f"{lab:>10}{len(ids):>6}{np.median(rel):>20.4f}"
              f"{np.mean(np.abs(rel-1) > 0.10):>10.1%}{float(np.corrcoef(ra, rb)[0,1]):>11.3f}")

    print("\n" + "=" * 92)
    print("★読み方（事前登録のとおり）")
    print("  ・★**主判定は②の「★10分前・40%」の判定一致率**。**高ければ運用は安全**。")
    print("  ⚠**これは「除外が効くか」ではない**（それは(117)(139)）。**判定の時点安定性だけ**。")
    print("  ⚠**ROIは出していない**（396レースでは判定不能・事前登録6）。")


if __name__ == "__main__":
    main()
