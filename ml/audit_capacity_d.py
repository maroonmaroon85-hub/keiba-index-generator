"""(121) ★(81)「容量を上げるとROIが上がる」を**Dで測り直す**（2026-08-11）

★なぜやり直すか
　(81)は「L5がL2より枠連ROI +2.78pt・95%CI[+0.07,+5.51]★」と結論した。だが:
　　・**1シード**（(83)で「1シードの当たり」を3例踏んでいる）
　　・**ROI**で測っている。しかも(80)自身が「**枠連で1ptの差の検出に155,091R＝約50年**」
　　　「三連複BOX4の+3.52ptは**検出限界3.55pt未満**」と書いている。
　　・CIの下端が **+0.07** ＝ ほとんど0。**(62)→(117)とまったく同じ形**。
　→ **Dで測れば桁違いに精密**（1レース1標本・対数なので裾が圧縮される）。

★測るもの
　L2（現行）と L5（高容量）それぞれのモデル確率から**枠連の q** を作り、
　**全枠組で正規化**して E[d] = E[log q − log q_pool] を出す。
　⚠**混合の正規化定数は必ず計算する**。「両モデルで同じ形だから差には効かない」と考えて
　　省いたら、L5−L2 が **−0.24 nat**（このプロジェクトの量の10倍）という明らかな作り物が出た。
　　Z = Σ q_market^(1−w)·q_model^w は**モデルごとに違う**（尖っているほど変わる）。省いてはいけない。
　さらに**市場と対数線形で混ぜたとき**のDも出す（運用に効くのはこちら）。

★★事前登録（測る前に宣言）
　1. 比較は **L2-win vs L5-win**（同じ目標で容量だけ違う組）。後から別の組に変えない。
　2. **混合重み w は 0/0.1/0.2/0.3/0.5 の5点**を宣言。ウォークフォワードでは選ばない
　　 （選ぶと「どちらが良いか」ではなく「どちらが選びやすいか」を測ることになる）。
　3. **判定**: 同じ w で L5 − L2 の差の99%CIが0を外れるか。**年分割で符号が揃うか**も見る。
　4. **予想**: **差は出ない**（|L5−L2| < 0.002）。理由は(102)——モデル混合の全成果が
　　 +0.0024しかないので、その中の容量差はさらに小さいはず。(81)の+2.78ptは
　　 **ROIの推定の暴れ**（(80)の検出限界2.20ptと同オーダー）と予想する。
　5. ⚠(117)で予想を外しているので、この予想も外れる前提で読むこと。

実行: python3 ml/audit_capacity_d.py [開始年(既定2019)]
　　　（`data/cache/exp_L2-win_2015/` と `exp_L5-win_2015/` を使う）
"""
import glob
import math
import sys
from itertools import combinations_with_replacement

import numpy as np
import pandas as pd

sys.path.insert(0, "ml")
from audit_crosspool import PAYBACK, load_races, payoff, probs, zq
from audit_crosspool2 import PAYKEY, realized
from audit_lbs import build_matrix, fit_lambda, q_of_lbs
from audit_lbs import frame_members, g_pair_unordered, stage_w
from waku_umatan import waku_of

WS = [0.0, 0.1, 0.2, 0.3, 0.5]           # ★先に宣言した混合重み
EXPERTS = {"L2": "data/cache/exp_L2-win_2015", "L5": "data/cache/exp_L5-win_2015"}


def mci(x, alpha=0.01):
    x = np.asarray(x, float)
    if len(x) < 2:
        return float("nan"), float("nan"), float("nan")
    m = x.mean()
    se = x.std(ddof=1) / math.sqrt(len(x))
    return m, m - zq(alpha) * se, m + zq(alpha) * se


def load_expert(path):
    out = {}
    for f in sorted(glob.glob(f"{path}/*.csv")):
        for rid, u, p in pd.read_csv(f)[["raceid", "umaban", "p"]].itertuples(index=False):
            out.setdefault(str(rid), {})[int(u)] = float(p)
    return out


def mkt_waku_dist(r, p, lam2):
    """λ補正Harvilleの**枠連の全枠組**の分布 {(枠a,枠b): q}。q_of_lbs と同じ式を全組に当てる。"""
    w2 = stage_w(p, lam2)
    W2 = w2.sum()
    wk = frame_members(r)
    ws = sorted(wk)
    out = {}
    for i, a in enumerate(ws):
        for b in ws[i:]:
            q = 0.0
            if a == b:
                mem = wk[a]
                for x in range(len(mem)):
                    for y in range(x + 1, len(mem)):
                        q += g_pair_unordered(p, w2, W2, mem[x], mem[y])
            else:
                for x in wk[a]:
                    for y in wk[b]:
                        q += g_pair_unordered(p, w2, W2, x, y)
            if q > 0:
                out[(a, b)] = q
    tot = sum(out.values())
    return {k: v / tot for k, v in out.items()} if tot > 0 else None


def waku_dist(nums, pw, n):
    """馬ごとの確率 → **全枠組で正規化した**枠連の分布 {(枠a,枠b): q}。

    ★2つ直した点（どちらも最初は間違えていた）:
    　1. **全枠組で正規化する**。(118)では省いてDの水準が−0.1988になった。
    　2. **1頭しかいない枠にゾロ目を作らない**。1頭では同枠同士の枠連は成立しないのに
    　　 bp[a]^2 を割り当てていた。**存在しない組に確率を捨てる量がモデルごとに違う**ので、
    　　 モデル間の比較が歪む。さらに市場側と枠組の集合が食い違い、**6割のレースが落ちていた**。
    """
    s = sum(pw.values())
    if s <= 0:
        return None
    bp, cnt = {}, {}
    for u in nums:
        w = waku_of(u, n)
        bp[w] = bp.get(w, 0.0) + pw[u] / s
        cnt[w] = cnt.get(w, 0) + 1
    ws = sorted(bp)
    out = {}
    for a, b in combinations_with_replacement(ws, 2):
        if a == b:
            if cnt[a] < 2:          # ★1頭ではゾロ目にならない
                continue
            out[(a, b)] = bp[a] * bp[a]
        else:
            out[(a, b)] = 2.0 * bp[a] * bp[b]
    tot = sum(out.values())
    return {k: v / tot for k, v in out.items()} if tot > 0 else None


def main():
    y0 = int(sys.argv[1]) if len(sys.argv) > 1 else 2019
    ex = {}
    for k, path in EXPERTS.items():
        ex[k] = load_expert(path)
        if not ex[k]:
            sys.exit(f"{path} が無い。(97)/(102)のキャッシュが要る")
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
        if yy < y0 or not lam.get(yy) or not r["wakuren"]:
            continue
        rl = realized(r)
        if rl is None:
            continue
        a, b, c = rl
        hs = r["horses"]
        nums = [u for u, _, _ in hs]
        if any(r["rid"] not in ex[k] or any(u not in ex[k][r["rid"]] for u in nums)
               for k in EXPERTS):
            continue
        num2k = {u: k for k, (u, _, _) in enumerate(hs)}
        if a not in num2k or b not in num2k:
            continue
        p_mkt = probs(hs)
        l2, l3 = lam[yy]
        q_mkt, combo = q_of_lbs("枠連", r, p_mkt, l2, l3, num2k, a, b, c)
        if q_mkt <= 0 or combo is None:
            continue
        v = payoff(r, PAYKEY["枠連"], combo)
        if not v or v <= 0:
            continue
        key = tuple(sorted(combo))
        md = mkt_waku_dist(r, p_mkt, l2)
        if not md or md.get(key, 0.0) <= 0:
            continue
        dists = {}
        ok = True
        for k in EXPERTS:
            dd = waku_dist(nums, ex[k][r["rid"]], r["n"])
            if not dd or dd.get(key, 0.0) <= 0 or set(dd) != set(md):
                ok = False
                break
            dists[k] = dd
        if not ok:
            continue
        # ★混合は**全枠組で正規化**してから実現組を読む。ここを省くと結果が壊れる
        row = {"year": yy, "lg": math.log((v + 5) / 100.0) - math.log(PAYBACK["枠連"])}
        keys = sorted(md)
        lm = np.array([math.log(md[t]) for t in keys])
        j = keys.index(key)
        row["d_mkt"] = lm[j]
        for k in EXPERTS:
            lk = np.array([math.log(dists[k][t]) for t in keys])
            for w in WS:
                z = (1 - w) * lm + w * lk
                z = z - z.max()
                row[f"{k}_{w}"] = float(z[j] - math.log(np.exp(z).sum()))
        rows.append(row)

    df = pd.DataFrame(rows)
    print(f"(121) 容量 L2 vs L5 を D で測る（{y0}年以降・{len(df):,}レース）")
    print("★(81)はROI・1シードで「L5が+2.78pt・CI[+0.07,+5.51]」と結論していた\n")

    lg = df["lg"].to_numpy()
    print(f"{'w':>5}{'L2のD':>11}{'L5のD':>11}{'L5−L2':>10}{'99%CI':>22}{'判定':>7}")
    for w in WS:
        d = {k: df[f"{k}_{w}"].to_numpy() + lg for k in EXPERTS}
        diff = d["L5"] - d["L2"]
        m, lo, hi = mci(diff)
        print(f"{w:>5.1f}{d['L2'].mean():>+11.4f}{d['L5'].mean():>+11.4f}"
              f"{m:>+10.4f}{f'[{lo:+.4f},{hi:+.4f}]':>22}"
              f"{('★L5' if lo > 0 else ('★L2' if hi < 0 else '')):>7}")

    print("\n★年分割（w=0.2）")
    w = 0.2
    diff = df[f"L5_{w}"].to_numpy() - df[f"L2_{w}"].to_numpy()
    pos = 0
    yl = sorted(df["year"].unique())
    for yy in yl:
        m = (df["year"] == yy).to_numpy()
        if m.sum() < 100:
            continue
        pos += diff[m].mean() > 0
        print(f"   {yy}  {int(m.sum()):>5}本  L5−L2 = {diff[m].mean():+.4f}")
    print(f"   → {pos}/{len(yl)} 年で L5 が上")
    print("\n★読み方（事前登録のとおり）")
    print("  ・どのwでも99%CIが0をまたぐなら、(81)の+2.78ptは**ROIの推定の暴れ**だったと確定。")
    print("  ・★L5が付いて年分割も揃うなら、**容量は本物**＝L5に乗り換える判断が要る。")
    print("  ・★L2が付いたら、(81)は**符号まで間違っていた**ことになる。")


if __name__ == "__main__":
    main()
