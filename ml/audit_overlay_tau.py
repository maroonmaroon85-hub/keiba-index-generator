"""(145) ★★★★馬連プール自身の人気-穴バイアスを補正してから枠に集約する（2026-08-13）

★なぜやるか
　(141)の q は `q_枠(a,b) = Σ_{i∈a,j∈b} 1/o_馬連(i,j)` を正規化しただけ。
　**「1/オッズ ∝ 確率」を仮定している**が、(136)で**枠連プールには強い人気-穴バイアス**が実測された
　（最も人気薄の十分位で **O/E = 0.456**＝言い値の半分以下しか来ない）。
　★**馬連プールにも同じ癖があるはず**。**あるなら補正してから集約したほうが q は良くなる**。

★★補正の形（Lo–Bacon-Shone と同じ発想。**自由度は1つだけ**）
```
　p_ij ∝ (1/o_ij)^τ         τ>1 なら人気薄を割り引く（＝バイアスを打ち消す）
　q_枠(a,b) = Σ_{i∈a, j∈b} p_ij     ← ★集約自体は近似ゼロのまま
```
　★**τ は各年について「その年より前だけ」で当てはめる（ウォークフォワード）**。**先読みを入れない**。
　★**モデルは1つも使わない**。**使うのは馬連の板と枠連の板だけ**。

★★事前登録（測る前に宣言する）
　1. **τ のグリッドは 0.9〜1.3 を 0.02 刻み**。**後から広げない**。当てはめは
　　 **的中した馬連の組の対数尤度の最大化**（板の上で正規化）。
　2. **閾値は(141)と同じ 1.00/1.10/1.290/1.50/2.00**。**後から増やさない**。
　3. **★主判定は 2021-2026（直近6年）の ROI の99%CI下端 > 100%**。
　　 (142)で**補正なしの直近は 107.1% [91.0,123.2]＝100%を含む**。**そこを超えるかどうか**。
　4. **前半 2015-2020 も併記**するが**判定には使わない**（(142)で既に★が出ている＝取りにいかない）。
　5. **プラセボ**: レース内で比を組にランダム割り当てして同数買う。200回平均。
　6. ⚠**この実験が答えないこと（先に書く）**:
　　 **(143)(144)で「張れる時点では成立しない」ことが分かっている**。**確定オッズのオラクルのまま**。
　　 → ★**目的は「儲かるか」ではなく「直近にも非効率が残っているか」**。
　　 　 **残っているなら TARGETの時系列オッズ(枠連・馬連・過去1年)を出す価値がある**。
　　 　 **残っていないなら、その収集もやらなくてよい**。**そこを決めるための実験**。
　7. **予想**: ★**当てにしてよい予想は持っていない**（類推はこの3日で4連敗）。

実行: python3 ml/audit_overlay_tau.py
"""
import math
import sys

import numpy as np

sys.path.insert(0, "ml")
from audit_cond_split import load_boards
from audit_crosspool import PAYBACK, load_races, zq
from audit_crosspool2 import realized
from audit_waku_vs_umaren import load_type
from waku_umatan import waku_of

R = PAYBACK["枠連"]
THS = [1.00, 1.10, 1.0 / R, 1.50, 2.00]
TAUS = np.arange(0.90, 1.3001, 0.02)      # ★先に宣言。後から広げない
NPLA = 200
RNG = np.random.default_rng(20260813)


def main():
    races = load_races()
    wb, ub = load_boards(), load_type(4, 4)
    if not wb or not ub:
        sys.exit("枠連(--type 3)と馬連(--type 4)の板が両方要る。")

    # ── レースごとに「馬連の板」と「的中した組の位置」を作る ──
    dat = []
    for r in races:
        yy = r["year"]
        if yy < 2015 or not r["wakuren"]:
            continue
        W, U = wb.get(r["rid"]), ub.get(r["rid"])
        if not W or not U:
            continue
        rl = realized(r)
        if rl is None:
            continue
        a, b, _ = rl
        n = r["n"]
        nums = [u for u, _, _ in r["horses"]]
        if a not in nums or b not in nums:
            continue
        pk = tuple(sorted((a, b)))
        uk = sorted(U)
        if pk not in U:
            continue
        iu = uk.index(pk)
        uo = np.array([U[k] for k in uk], float)
        wkey = tuple(sorted((waku_of(a, n), waku_of(b, n))))
        if wkey not in W:
            continue
        dat.append(dict(y=yy, n=n, uk=uk, uo=uo, iu=iu, W=W, wkey=wkey))

    # ── τ をウォークフォワードで当てはめる（的中した馬連の組の対数尤度）──
    def ll(tau, idx):
        s = 0.0
        for i in idx:
            d = dat[i]
            w = (1.0 / d["uo"]) ** tau
            s += math.log(w[d["iu"]] / w.sum())
        return s

    ys = np.array([d["y"] for d in dat])
    tau_of = {}
    for yy in sorted(set(ys.tolist())):
        tr = np.where(ys < yy)[0]
        if len(tr) < 2000:
            tau_of[yy] = 1.0
            continue
        sub = tr[RNG.permutation(len(tr))[:4000]]      # 当てはめは4000本で十分
        tau_of[yy] = float(TAUS[int(np.argmax([ll(t, sub) for t in TAUS]))])
    print("(145) 馬連プールの人気-穴バイアスを補正してから枠に集約する")
    print("★モデルは使わない。使うのは馬連の板と枠連の板だけ。τはウォークフォワード")
    print("　τ（その年より前だけで当てはめ）:",
          "  ".join(f"{y}:{tau_of[y]:.2f}" for y in sorted(tau_of))[:200], "\n")

    # ── 比の裾で買う（τ補正あり / なし）──
    def run(use_tau, mask):
        out = {}
        for t in THS:
            prof, cost, ret, nb, hit = [], 0.0, 0.0, 0, 0
            pl = np.zeros(NPLA)
            plc = np.zeros(NPLA)
            for i in np.where(mask)[0]:
                d = dat[i]
                tau = tau_of[d["y"]] if use_tau else 1.0
                w = (1.0 / d["uo"]) ** tau
                agg = {}
                for k, val in zip(d["uk"], w):
                    x, y = sorted((waku_of(k[0], d["n"]), waku_of(k[1], d["n"])))
                    agg[(x, y)] = agg.get((x, y), 0.0) + float(val)
                keys = [k for k in sorted(agg) if k in d["W"]]
                if d["wkey"] not in keys or len(keys) < 3:
                    continue
                inv = np.array([1.0 / d["W"][k] for k in keys])
                qp = inv / inv.sum()
                qq = np.array([agg[k] for k in keys])
                qq /= qq.sum()
                sel = (qq / qp) >= t
                if not sel.any():
                    continue
                odds = np.array([d["W"][k] for k in keys])
                win = np.array([k == d["wkey"] for k in keys])
                c = 100.0 * sel.sum()
                v = float(odds[sel & win].sum() * 100.0)
                prof.append(v - c)
                cost += c
                ret += v
                nb += int(sel.sum())
                hit += int((sel & win).sum())
                for j in range(NPLA):
                    m2 = np.zeros(len(keys), bool)
                    m2[RNG.permutation(len(keys))[: int(sel.sum())]] = True
                    plc[j] += c
                    pl[j] += float(odds[m2 & win].sum() * 100.0)
            if not prof:
                continue
            p = np.array(prof)
            mc = cost / len(p)
            se = p.std(ddof=1) / math.sqrt(len(p))
            z = zq(0.01)
            out[t] = (len(p), nb, hit, ret / cost,
                      1 + (p.mean() - z * se) / mc, 1 + (p.mean() + z * se) / mc,
                      float(np.mean(pl / np.maximum(plc, 1))))
        return out

    for lab, mask in (("★★2021-2026（直近6年・これが主判定）", ys >= 2021),
                      ("2015-2020（前半・判定には使わない）", ys <= 2020)):
        print(f"■ {lab}")
        a = run(False, mask)
        b = run(True, mask)
        print(f"{'閾値':>7}{'補正なしROI':>12}{'99%CI':>20}"
              f"{'★τ補正ROI':>12}{'99%CI':>20}{'点数':>9}{'プラセボ':>10}")
        for t in THS:
            if t not in a or t not in b:
                continue
            _, _, _, r0, l0, h0, _ = a[t]
            _, nb, _, r1, l1, h1, pv = b[t]
            m = "★★" if l1 > 1.0 else ""
            print(f"{t:>7.3f}{100*r0:>11.1f}%"
                  f"{'[' + format(100*l0, '.1f') + ',' + format(100*h0, '.1f') + ']':>20}"
                  f"{100*r1:>11.1f}%"
                  f"{'[' + format(100*l1, '.1f') + ',' + format(100*h1, '.1f') + ']':>20}"
                  f"{nb:>9,}{100*pv:>9.1f}% {m}")
        print()

    print("=" * 100)
    print("★読み方（事前登録のとおり）")
    print("  ・**主判定は 2021-2026 の τ補正版で 99%CI下端 > 100%**。(142)の補正なしは 107.1% [91.0,123.2]。")
    print("  ・⚠**これは確定オッズのオラクルのまま**。(143)(144)より**張れる時点では成立しない**。")
    print("    → ★目的は「直近にも非効率が残っているか」を決めること。")
    print("      **残っていれば TARGETの時系列オッズ(枠連・馬連・過去1年)を出す価値がある**。")
    print("      **残っていなければ、その収集もやらなくてよい**。")


if __name__ == "__main__":
    main()
