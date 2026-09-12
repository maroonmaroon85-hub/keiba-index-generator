"""(223) ★★★★**1日に何時にオッズを取れば、最も多くのレースを拾えるか** —— 収集時刻の設計

★★**動機（2026-09-07・利用者の指定）**:
　★**「1開催日あたり、何時何時にオッズ収集すれば最大レース取れそうか調査してほしい。
　　　レース1分前とかは現実的じゃない」**

■ ★★★**なぜこれが効くのか（(220)で分かったこと）**
　★**(220)の実測**: **確定版で軸が立つ9本のうち、朝9時でも立つのは6本（66.7%）**。
　★★**軸そのものは6/6で同一だった**——⚠**失われたのは「10.0倍の線を割った3本」だけ**。
　→ ★★**拾えるレース数を決めているのは、★軸の同一性ではなく「10倍の線をまたぐか」**。
　→ ★**だから主指標は「10倍の線をまたぐ馬の割合」**。**小さいほど多く拾える**。

■ ★**収集時刻の trade-off**
　★**遅く取るほど確定に近い**（(219): 09時 8.4% → 13時 4.4% → 15時 1.8% → 締切直前 0.9%）。
　⚠**ただし、あるレースの発走を過ぎた時刻の値は、そのレースには使えない**。
　→ ★**1回だけなら「遅く取ると午前のレースを落とす」「早く取ると午後が不正確」**。
　→ ★★**複数回に分ければ両立する。★その最適な組を総当たりで探す**。

────────────────────────────────────────────────────────────
★★★ 事前登録（2026-09-07・**結果を見る前にコミットする**）
────────────────────────────────────────────────────────────

■ ★対象: **data/odds_ts の396レース**（2026-06-20〜07-26・11開催日）。
　⚠★**これは別セッションが(148)で集めたもの。★読むだけ**（**集め方には触れない**）。
■ ★**収集時刻の候補**: **09:00〜16:00 を30分刻み**（**15点**）。
■ ★**割り当て規則**: **各レースは「発走前で最も遅い収集時刻」の値を使う**。
　★**発走前の収集が1つも無いレースは「拾えない」**。

■ ★★★★**主指標（★先に決める・これ1つ）**
　★★**「10.0倍の線をまたぐ馬の割合」の全レース平均**（**拾えないレースは★またぎ100%として数える**）。
　★**理由**: **(220)で、軸が落ちる唯一の原因がこれだったため**。⚠**小さいほど良い**。
　★**副指標として併記**: **カバー率 / 人気1番の一致率 / 対数相関 / 平均変化率**。

■ ★★**探索**: **1回・2回・3回・4回のそれぞれで、15点から選ぶ組を総当たり**
　（**15 + 105 + 455 + 1,365 = 1,940通り**）。★**主指標が最小の組を出す**。

■ ★★ゲート2（判定基準42）——**この測定は何を返せば「時刻は関係ない」か**
　★**時刻が関係ないなら、どの組でも主指標は同じ値を返し、回数を増やしても改善しない**。
　★**「1回の最良」と「4回の最良」の差も併記する**——**それが時刻設計の価値の上限**。
■ ★★★内部対照（**決定的**）: **「締切直前を毎レース取る」＝またぎ 0.9%**（**(219)と一致**）。
　★**これが理論上の下限**。⚠**合わなければ読まない**。

■ 予想（⚠**当てにしない**・★**私は8回外した**）
　★**2回で大半が取れると見る**（**午前1回・午後1回**）。
　★**3回目以降の改善は小さいと見る**。⚠**外れたら仮説の方を書き換える**。

■ ★★★★**(225) 追記（2026-09-07・★利用者の指摘「1回やっても4回やっても99.4%なの？」）**
　★★**指摘のとおり、カバー率は飽和していて、私の主指標の置き方が甘かった**。
　★**カバー率が測っているのは「発走前にオッズが1枚でも取れたか」だけ**——
　　**09:30より前に発走するレースはほぼ無いので、1回で99.4%に飽和する**。
　　⚠**残り0.6%は取りこぼしではなく、突き合わせ不能な2本**。
　★★**レースを落としているのは時刻ではなく「10倍の線」**。★**しかも落ち方には向きがある**:
| 用語 | 意味 |
|---|---|
| ★**取りこぼし** | **確定では10倍以上なのに、収集時点では10倍未満** → ★**買えたはずのレースを見送る** |
| ⚠**空振り** | **確定では10倍未満なのに、収集時点では10倍以上** → ⚠**買うべきでない馬を軸にする** |
　★**またぎ率はこの2つの合計だったので、分けて出す**。

■ ★★★★**結果（実測済み・2026-09-07）—— ★回数を増やしても、あまり良くならない**
　★**内部対照が立った**: **毎レース締切直前＝またぎ 0.9%**（**(219)と一致・理論上の下限**）。
　★**突き合わせできたレース 362本**。

| 回数 | ★**最良の収集時刻** | ★**またぎ** | ★★**取りこぼし** | ⚠**空振り** | 人気1一致 | 平均変化 | カバー率 |
|---|---|---|---|---|---|---|---|
| **1回** | ★**09:30** | **8.9%** | ★**7.7%** | **5.1%** | 72.8% | 38.6% | 99.4% |
| **2回** | ★**09:30 / 11:30** | **7.7%** | **6.1%** | 4.7% | 72.8% | 37.3% | 99.4% |
| **3回** | ★**09:30 / 11:30 / 14:30** | **7.0%** | **5.4%** | 4.4% | 74.4% | 35.5% | 99.4% |
| **4回** | 09:30 / 11:30 / 14:00 / 16:00 | **6.7%** | ★**5.1%** | 4.1% | 75.6% | 34.1% | 99.4% |
| *(参考)* **毎レース締切直前** | ⚠**現実的でない** | ★**0.9%** | — | — | 98.3% | 4.0% | — |
　★★**「最大レース取れそうか」への答え**: **09:30の1回で取りこぼし7.7%、4回でも5.1%**（**−2.6pt**）。
　⚠★**そして取りこぼしと同じくらい「空振り」がある**（**5.1% vs 4.1%・回数を増やしてもほぼ減らない**）。
　　★**朝の時点で10倍以上に見えて確定では10倍を割る馬**を軸にしてしまう。
　　⚠**これは取りこぼしより質が悪い**——**買った上で、11年の測定とは別の母集団の馬に賭けることになる**。
　★**(220)の「9本→6本」は取りこぼしの側だけ**。⚠**空振りの側は見ていなかった**
　　——**実際には「3本落として、別の何本かを拾っている」**はず。

■ ⚠★**回数を増やす価値は小さい**
　**1回→2回 −1.2pt / 2回→3回 −0.7pt / 3回→4回 −0.3pt**。
　★**4回やっても 6.7%で、締切直前の 0.9% には +5.8pt 届かない**。
　★★**つまり「決まった時刻に何回か取る」方式では、朝1回とあまり変わらない**。
　★**カバー率は1回でも99.4%**——**09:30より前に発走するレースはほぼ無い**。
　　→ ★**「拾えるレース数」は時刻で増えない。増えるのは「確定への近さ」だけ**。

■ ★★★**なぜ改善しないのか（★構造）**
　★**あるレースの発走前で最も遅い収集を使う以上、収集時刻とそのレースの発走の間隔が
　　平均で1〜2時間空く**。**その間の値動きが、またぎの主因**。
　★**間隔を詰めるには収集回数を増やすしかないが、30分刻みでも改善は逓減した**
　　（**−1.2 → −0.7 → −0.3pt**）。
　⚠**「マイレース1分前」が効くのは、まさにこの間隔をゼロにするから**。

■ ★**実務への示唆**（⚠**推奨ではなく、測定が示す形**）
　★**09:30 の1回で、4回やった場合の改善分（−2.2pt）の大半を捨てても、差は2.2pt**。
　★**運用を軽くしたいなら 09:30 の1回**、★**もう少し詰めるなら 09:30 + 11:30 + 14:30 の3回**。
　⚠**どちらでも「確定オッズで測った169マスの数字」には戻らない**。

■ ⚠★**私の予想は9回目の外し**
　★**「2回で大半が取れる」は当たり**（**1回→2回で−1.2pt、そこから先は−1.0ptしかない**）。
　⚠**「3回目以降の改善は小さい」も当たり**。★**外したのは「そもそも改善幅が小さい」ことを
　　見込んでいなかった点**——**4回でも締切直前に +5.8pt 届かない**。

実行: python3 ml/audit_ana_sched.py    自己テスト: python3 ml/audit_ana_sched.py --selftest
"""
import sys
from itertools import combinations

import numpy as np
import pandas as pd

sys.path.insert(0, "ml")
from odds_ts import load_dir
import features as F

LFIX = 10.0
GRID = [(h, m) for h in range(9, 17) for m in (0, 30)][:15]   # ★09:00〜16:00 の30分刻み
KMAX = 4
KNOWN_CROSS = 0.9          # ★内部対照: 締切直前のまたぎ率[%]（(219)）
CROSS_TOL = 0.3


def label(t):
    return f"{t[0]:02d}:{t[1]:02d}"


def row_at(rec, t):
    """★時刻 t（当日）以前で最後の単勝スナップショットの行番号。無ければ None"""
    lim = pd.Timestamp(rec["date"].date()) + pd.Timedelta(hours=t[0], minutes=t[1])
    ok = np.where((rec["times"] <= lim) & (rec["kubun"] == "1"))[0]
    return int(ok[-1]) if len(ok) else None


def usable(rec, t):
    """★その収集時刻が、このレースに使えるか（★発走前か）"""
    lim = pd.Timestamp(rec["date"].date()) + pd.Timedelta(hours=t[0], minutes=t[1])
    return lim < rec["post"]


def selftest():
    ok = True
    print(f"★収集時刻の候補 **{len(GRID)}点**: {' '.join(label(t) for t in GRID)}")
    ok &= len(GRID) == 15 and GRID[0] == (9, 0) and GRID[-1] == (16, 0)
    n = sum(len(list(combinations(range(len(GRID)), k))) for k in range(1, KMAX + 1))
    print(f"★探索する組: " + " + ".join(
        str(len(list(combinations(range(len(GRID)), k)))) for k in range(1, KMAX + 1))
        + f" = **{n:,}通り**　{'★OK' if n == 1940 else '⚠NG'}")
    ok &= n == 1940
    rec = {"date": pd.Timestamp("2026-07-26"),
           "post": pd.Timestamp("2026-07-26 12:30"),
           "times": pd.DatetimeIndex(["2026-07-26 09:00", "2026-07-26 11:00",
                                      "2026-07-26 12:25", "2026-07-26 15:00"]),
           "kubun": np.array(["1", "1", "1", "1"])}
    print(f"★発走 12:30 のレースに対して:")
    for t, wu, wr in (((9, 0), True, 0), ((11, 0), True, 1), ((12, 0), True, 1),
                      ((13, 0), False, 2), ((16, 0), False, 3)):
        u, r = usable(rec, t), row_at(rec, t)
        print(f"　{label(t)}　使える={u}（期待{wu}）　行={r}（期待{wr}）"
              f"　{'★OK' if (u == wu and r == wr) else '⚠NG'}")
        ok &= (u == wu and r == wr)
    print(f"★★主指標: **10.0倍の線をまたぐ馬の割合**"
          f"（★**拾えないレースはまたぎ100%として数える**）")
    print(f"★★★内部対照: **締切直前を毎レース取ると またぎ {KNOWN_CROSS}%±{CROSS_TOL}**")
    print("★自己テスト: " + ("全部OK" if ok else "⚠NG"))
    return 0 if ok else 1


def main():
    print("(223) ★★★★**1日に何時にオッズを取れば、最も多くのレースを拾えるか**")
    print("⚠**data/odds_ts は別セッションが(148)で集めたもの。★読むだけ**\n")
    ts = load_dir()
    d = F.to_model(F.load_files())
    fin = {}
    for rid, g in d.groupby("raceid"):
        o = {int(u): float(x) for u, x in zip(g["umaban"], g["odds"])
             if np.isfinite(x) and x > 0}
        if o:
            fin[str(rid)] = o

    # ★各レース × 各収集時刻 の指標を先に全部作る
    # ★★(225) またぎを★向きで分ける（利用者の指摘「1回でも4回でも99.4%なの？」から）
    #   ⚠**カバー率は「オッズが取れたか」しか見ておらず、レースを落としているのは10倍の線**。
    #   ★**確定≥10 なのに 収集時<10 → ★取りこぼし**（**買えたはずのレースを見送る**）
    #   ★**確定<10 なのに 収集時≥10 → ⚠空振り**（**買うべきでない馬を軸にする**）
    miss = {}       # (rid, t) → 取りこぼした馬の数 / 確定で10倍以上の馬の数
    false = {}      # (rid, t) → 空振りした馬の数 / 収集時に10倍以上の馬の数
    cross = {}      # (rid, t) → またぎ率
    top1 = {}
    corr = {}
    chg = {}
    use = {}
    races = []
    for rid, rec in ts.items():
        b = fin.get(rid)
        if b is None:
            continue
        races.append(rid)
        for t in GRID:
            if not usable(rec, t):
                continue
            i = row_at(rec, t)
            if i is None:
                continue
            row = rec["odds"][i]
            a = {u + 1: float(row[u]) for u in range(rec["n"])
                 if np.isfinite(row[u]) and row[u] > 0}
            sh = [u for u in a if u in b]
            if len(sh) < 5:
                continue
            va = np.array([a[u] for u in sh])
            vb = np.array([b[u] for u in sh])
            use[(rid, t)] = True
            cross[(rid, t)] = float(np.mean((va >= LFIX) != (vb >= LFIX)))
            hb, ha = vb >= LFIX, va >= LFIX
            miss[(rid, t)] = (float(np.sum(hb & ~ha) / hb.sum()) if hb.sum() else 0.0)
            false[(rid, t)] = (float(np.sum(ha & ~hb) / ha.sum()) if ha.sum() else 0.0)
            chg[(rid, t)] = float(np.mean(np.abs(va - vb) / vb))
            oa = sorted(sh, key=lambda u: a[u])
            ob = sorted(sh, key=lambda u: b[u])
            top1[(rid, t)] = float(oa[0] == ob[0])
            corr[(rid, t)] = (va, vb)
    print(f"★突き合わせできたレース **{len(races):,}本**")

    def score(sched):
        """★主指標: またぎ率の平均（★拾えないレースは100%）＋副指標"""
        cs, t1, cv, ch, cov = [], [], [], [], 0
        ms, fs = [], []
        for rid in races:
            best = None
            for t in sched:
                if use.get((rid, t)):
                    best = t                      # ★GRIDは時刻順なので最後が最も遅い
            if best is None:
                cs.append(1.0)
                continue
            cov += 1
            cs.append(cross[(rid, best)])
            ms.append(miss[(rid, best)]); fs.append(false[(rid, best)])
            t1.append(top1[(rid, best)])
            ch.append(chg[(rid, best)])
            cv.append(corr[(rid, best)])
        pa = np.concatenate([x[0] for x in cv]) if cv else np.array([1.0])
        pb = np.concatenate([x[1] for x in cv]) if cv else np.array([1.0])
        r = float(np.corrcoef(np.log(pa), np.log(pb))[0, 1]) if len(pa) > 2 else float("nan")
        return (float(np.mean(cs)), cov / max(len(races), 1),
                float(np.mean(t1)) if t1 else 0.0, float(np.mean(ch)) if ch else 1.0, r,
                float(np.mean(ms)) if ms else 1.0, float(np.mean(fs)) if fs else 0.0)

    print(f"\n{'回数':<5}{'★最良の収集時刻':<34}{'★またぎ':>9}"
          f"{'★★取りこぼし':>12}{'⚠空振り':>10}{'人気1一致':>10}{'平均変化':>9}")
    best = {}
    for k in range(1, KMAX + 1):
        cand = min((score(list(c)) + (c,) for c in combinations(GRID, k)),
                   key=lambda x: x[0])
        s, cov, t1, ch, r, ms, fs = cand[:7]
        sch = cand[7]
        best[k] = (s, sch, ms, fs)
        print(f"{k:<5}{' '.join(label(t) for t in sch):<34}{100*s:>8.1f}%"
              f"{100*ms:>11.1f}%{100*fs:>9.1f}%{100*t1:>9.1f}%{100*ch:>8.1f}%")

    # ★内部対照: 締切直前を毎レース取る
    cs = []
    for rid, rec in ts.items():
        if rid not in fin:
            continue
        i = int(np.where(rec["kubun"] == "1")[0][-1])
        row = rec["odds"][i]
        a = {u + 1: float(row[u]) for u in range(rec["n"])
             if np.isfinite(row[u]) and row[u] > 0}
        b = fin[rid]
        sh = [u for u in a if u in b]
        if len(sh) < 5:
            continue
        va = np.array([a[u] for u in sh])
        vb = np.array([b[u] for u in sh])
        cs.append(np.mean((va >= LFIX) != (vb >= LFIX)))
    lim = 100 * float(np.mean(cs))
    okc = abs(lim - KNOWN_CROSS) <= CROSS_TOL
    print(f"\n★★★内部対照（**理論上の下限＝毎レース締切直前**）: "
          f"またぎ **{lim:.1f}%**（{KNOWN_CROSS}±{CROSS_TOL}）　"
          f"{'★★立った' if okc else '⚠⚠落ちた'}")
    if not okc:
        print("⚠⚠**対照が落ちた。読まない**（判定基準32）。")
        return

    print(f"\n■ ★★★**取りこぼし＝「確定なら買えたのに、収集時点では条件を満たさない」馬の割合**")
    print(f"　⚠**カバー率は「オッズが取れたか」しか見ていない。レースを落とすのはこちら**。")
    print(f"\n■ ★★**回数を増やす価値**（★**これが時刻設計の上限**）")
    for k in range(2, KMAX + 1):
        print(f"　{k-1}回 → {k}回: またぎ {100*best[k-1][0]:.1f}% → "
              f"**{100*best[k][0]:.1f}%**（**{100*(best[k][0]-best[k-1][0]):+.1f}pt**）")
    print(f"　★**毎レース締切直前（現実的でない）なら {lim:.1f}%**"
          f"　→ ★**{KMAX}回でそこまでの差は {100*best[KMAX][0]-lim:+.1f}pt**")

    print("\n⚠**枠連の運用には触れない**。**(148)の集め方にも触れない**。"
          "**設定変更は提案しない**。")


if __name__ == "__main__":
    sys.exit(selftest() if "--selftest" in sys.argv else (main() or 0))
