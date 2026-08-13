"""(137) ★★★**軸と上位馬が同じ枠に入るとき**を測り直す — ユーザー発案（2026-08-13）

★ユーザーの問い「枠連の推奨馬・上位馬が重なった時などの検証を進めて。そこ頑張れば伸びそう」
　★これは(135)のBと違って**出馬表を見た瞬間に分かる量**＝**発走前に選べる**（判定基準19を満たす）。

★★なぜ「まだ決着していない」と言えるのか（2つとも既存の記録に残っている）
　**(60)がROIで測って正の数字を出したまま、Dで測り直していない**:
　　| 上位3頭が2枠に分散（2頭同枠） | 5,236R | 167円 | **86.5%** |
　　| ゾロ目あり（軸と紐が同枠）     | 3,511R | 200円 | **85.6%** |
　　| ゾロ目なし                     | 25,259R| 193円 | **84.3%** |
　⚠この +2.5pt は **(a)単一分割（(78)前）** で、**(b)枠連のROI検出限界2.20ptすれすれ**。
　　**(62)は(60)のカバー数の主張を訂正した**が、**枠の重なりの行は訂正も再測定もされていない**。
　→ **判定基準16そのもの**（ROIで測った数字は物差しを替えると決着することがある）。

★★もうひとつ、**(127)で新しい道具が手に入った直後だから今やる**
　既存の記録に **「ゾロ目のみ D=−0.0726（Harvilleが苦手）」**（HANDOFF）とある。
　(96)はHarvilleの系統誤差を「**人気馬の2着確率を過大評価する**」と特定しており、
　**ゾロ目＝同じ枠の2頭が1着2着を占める事象**なので、**Harvilleが最も壊れる形**に当たる。
　★**(127)の「馬連プール→枠」はHarvilleを通らない**ので、**ここが直っているはず**。
　→ **判定基準15**（道具を新しくしたら層別の結論は測り直す）にも真正面から該当する。

★★事前登録（測る前に宣言する。**後から層を増やさない**）
　**層（3つ。すべて発走前に観測できる）**
　　S1 **軸と紐1が同枠か**（＝買い目がゾロ目になるか）… 2層
　　S2 **モデル上位3頭が2枠に収まるか / 3枠に散るか** … 2層（(60)が+2.5ptと言った当の量）
　　S3 **枠集中度**＝モデルの枠別確率のハーフィンダール Σp_枠²の**十分位** … 10層
　**量（判定基準8のとおり、目的ごとに変える）**
　　A. **D**（q の質）: `馬連→枠` と `λHarville→枠` の両方を**枠連プール基準**で。
　　B. **1レース期待損失[円]**（買い方の質）: ROI差は検出限界2.20pt未満で測れないので使わない。

　1. **★主判定（レース選択に使えるか）**: S1/S2/S3 のいずれかで **D_馬連の層間差が +0.005超**
　　 かつ99%CI下端>0 かつ**年8/11以上**で符号が揃うこと。((125)と同じ基準を流用する)
　　 それ未満なら**記述にとどめ、運用は変えない**。
　2. **★副判定（機構）**: **ゾロ目層で `D_馬連 − D_λH` が他層より大きいか**。
　　 大きければ「(60)の85.6%とHarvilleの−0.0726は**同じものの表裏**だった」と言える。
　3. **★運用判定（買い方を変えるか）**: ゾロ目になるレースで
　　 **(i) そのままゾロ目を買う vs (ii) 紐を「別枠の次点馬」に替える** を**対応あり**で比べ、
　　 **期待損失の差の99%CIが0を除外**したら買い方を変える。またがるなら**変えない**。
　4. **多重比較**: 層は 2+2+10=14。**Bonferroni（α=0.01/14）**。
　5. ⚠**プラセボについて（判定基準23を先に当てる）**: ここで見るのは**層間差**なので、
　　 **層の割り当てを無作為にすれば差は構造上0**になる＝**統計的には何も検査しない**。
　　 → **プラセボは実装の検査に格下げ**し（無作為な2分割・30回・0になるはず）、
　　 　**統計的な守りはBonferroniと年分割に担わせる**。
　6. **道具の検算**: 的中組で「板×100 と実配当」を突き合わせる。(127)は0.07%だった。
　7. **★予想（判定基準24に従い、根拠の種類を明記する）**
　　 ・**［機構からの予想］ゾロ目層で `D_馬連 − D_λH` が他層より大きく出る**。
　　 　根拠は(96)の系統誤差の**向きが特定されている**こと。**類推ではない**ので、ここは当てにいく。
　　 ・**［類推からの予想・あてにしない］(60)の「上位3頭が2枠で+2.5pt」は縮む**。
　　 　根拠は「(78)が同種の単一分割の数字を2pt動かした」だけ＝**別の場所での出来事**。
　　 　⚠判定基準24のとおり**類推に載せた予想は当たらない**。実際(135)(136)で2回外している。
　　 ・**［機構からの予想］どの層もROIで100%には届かない**。(89)④の上界は層で分けても動かない。

実行: python3 ml/audit_waku_overlap.py [開始年(既定2015)]
必要: `--type 3`（枠連の板）と `--type 4`（馬連の板）と `data/cache/exp_L2-top3_2015`
"""
import glob
import math
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "ml")
from audit_capacity_d import mkt_waku_dist
from audit_cond_split import load_boards
from audit_crosspool import PAYBACK, load_races, payoff, probs, zq
from audit_crosspool2 import PAYKEY, realized
from audit_lbs import build_matrix, fit_lambda
from audit_waku_vs_umaren import load_type
from waku_umatan import bracket_probs, waku_of

MODEL_CACHE = "data/cache/exp_L2-top3_2015"
NGRP = 2 + 2 + 10                     # S1(2) + S2(2) + S3(10)
ALPHA = 0.01 / NGRP                   # Bonferroni
NPLA = 30                             # プラセボ＝実装の検査（上の5.）
RNG = np.random.default_rng(20260813)


def mci(x, alpha=ALPHA):
    x = np.asarray(x, float)
    if len(x) < 2:
        return float("nan"), float("nan"), float("nan")
    m = x.mean()
    se = x.std(ddof=1) / math.sqrt(len(x))
    return m, m - zq(alpha) * se, m + zq(alpha) * se


def load_model_p():
    """(97)が作ったウォークフォワードのモデル確率 → {raceid: {馬番: p}}。"""
    fs = sorted(glob.glob(f"{MODEL_CACHE}/*.csv"))
    if not fs:
        sys.exit(f"{MODEL_CACHE} が無い。先に (97)/(102) の実験を回してキャッシュを作ること。")
    out = {}
    for f in fs:
        for rid, u, p in pd.read_csv(f)[["raceid", "umaban", "p"]].itertuples(index=False):
            out.setdefault(str(rid), {})[int(u)] = float(p)
    return out


def two_sample(a, b, alpha=ALPHA):
    """互いに素な2群の差（判定基準13の後半: 入れ子のCIを見比べない）。"""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan"), float("nan")
    d = a.mean() - b.mean()
    se = math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return d, d - zq(alpha) * se, d + zq(alpha) * se


def main():
    y0 = int(sys.argv[1]) if len(sys.argv) > 1 else 2015
    mp = load_model_p()
    races = load_races()
    wb = load_boards()            # 枠連の板 {(枠a,枠b): odds}
    ub = load_type(4, 4)          # 馬連の板 {(馬a,馬b): odds}
    if not wb or not ub:
        sys.exit("枠連(--type 3)と馬連(--type 4)の板が両方要る。")
    print(f"レース {len(races)} / 枠連の板 {len(wb)} / 馬連の板 {len(ub)}", flush=True)

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

    rows, bad, hits = [], 0, 0
    for r in races:
        yy = r["year"]
        if yy < y0 or not lam.get(yy) or not r["wakuren"]:
            continue
        W, U = wb.get(r["rid"]), ub.get(r["rid"])
        if not W or not U:
            continue
        mm = mp.get(str(r["rid"]))
        if not mm:
            continue
        rl = realized(r)
        if rl is None:
            continue
        a, b, _ = rl
        n = r["n"]
        nums = [u for u, _, _ in r["horses"]]
        if a not in nums or b not in nums:
            continue
        pm = np.array([mm.get(u, 0.0) for u in nums], float)
        if pm.sum() <= 0 or (pm > 0).sum() < 3:
            continue
        order = [nums[i] for i in np.argsort(-pm)]      # モデル降順
        key = tuple(sorted((waku_of(a, n), waku_of(b, n))))
        if key not in W:
            continue
        v = payoff(r, PAYKEY["枠連"], list(key))
        if not v or v <= 0:
            continue
        hits += 1
        if abs(W[key] * 100 - v) > max(10.0, v * 0.01):
            bad += 1

        # ── 3つの q を同じ支持集合の上で作る（判定基準14①：必ず全組で正規化）──
        p = probs(r["horses"])
        l2, _ = lam[yy]
        md = mkt_waku_dist(r, p, l2)
        if not md:
            continue
        agg = {}
        for (x, y), o in U.items():
            if o <= 0:
                continue
            wx, wy = sorted((waku_of(x, n), waku_of(y, n)))
            agg[(wx, wy)] = agg.get((wx, wy), 0.0) + 1.0 / o
        keys = [k for k in sorted(md) if k in W and k in agg]
        if key not in keys or len(keys) < 3:
            continue
        sW = sum(1.0 / W[k] for k in keys)
        sU = sum(agg[k] for k in keys)
        sH = sum(md[k] for k in keys)
        lw = math.log((1.0 / W[key]) / sW)
        lu = math.log(agg[key] / sU)
        lh = math.log(md[key] / sH)

        # ── 層（すべて発走前に観測できる）──
        w_axis = waku_of(order[0], n)
        w_h1 = waku_of(order[1], n)
        s1 = int(w_axis == w_h1)                                   # ゾロ目になるか
        s2 = int(len({waku_of(h, n) for h in order[:3]}) <= 2)     # 上位3頭が2枠以内か
        bp = bracket_probs(nums, pm, n)
        hhi = sum(x * x for x in bp.values())                      # 枠集中度

        # ── 買い方（判定基準8: ROIではなく1レース期待損失[円]）──
        # (i) そのまま: 軸枠 × 紐1枠（ゾロ目ならゾロ目を買う）
        buy_i = tuple(sorted((w_axis, w_h1)))
        # (ii) 別枠に替える: モデル順で**軸と違う枠**に出てくる最初の馬を紐にする
        alt = next((h for h in order[1:] if waku_of(h, n) != w_axis), None)
        buy_ii = tuple(sorted((w_axis, waku_of(alt, n)))) if alt else None
        pay_i = v if buy_i == key else 0.0
        pay_ii = (v if buy_ii == key else 0.0) if buy_ii else float("nan")

        rows.append((yy, s1, s2, hhi, lu - lw, lh - lw, pay_i, pay_ii))

    if not rows:
        sys.exit("突き合わせできたレースが無い")
    yrsA = np.array([x[0] for x in rows])
    S1 = np.array([x[1] for x in rows])
    S2 = np.array([x[2] for x in rows])
    HHI = np.array([x[3] for x in rows])
    DU = np.array([x[4] for x in rows])      # 馬連→枠  −  枠連プール
    DH = np.array([x[5] for x in rows])      # λHarville→枠 − 枠連プール
    PI = np.array([x[6] for x in rows])
    PII = np.array([x[7] for x in rows])
    n = len(rows)

    print(f"\n(137) 軸と上位馬が同枠に入るとき（{y0}年以降・{n:,}レース）")
    print(f"★道具の検算: 的中 {hits:,} 件のうち板×100と実配当のずれ **{bad}**"
          f"（{bad/max(hits,1):.2%}）  ※(127)は0.07%")
    print(f"★Bonferroni: {NGRP}層なので各層は {100*(1-ALPHA):.4f}%CI で判定する")
    print(f"　ゾロ目になるレース {S1.sum():,}（{S1.mean():.1%}） / "
          f"上位3頭が2枠以内 {S2.sum():,}（{S2.mean():.1%}）")

    # ── A. D を層別に ────────────────────────────────────────────
    def show(tag, mask, lab):
        mu, lo, hi = mci(DU[mask])
        mh, _, _ = mci(DH[mask])
        print(f"  {lab:<26}{int(mask.sum()):>7}本  "
              f"馬連→枠 {mu:+.4f} [{lo:+.4f},{hi:+.4f}]   "
              f"λH→枠 {mh:+.4f}   差 {mu-mh:+.4f}")

    print("\n── A. D を層別に（枠連プール基準・★★は(127)の新しい土台） ──")
    print("\n■ S1 軸と紐1が同枠か（＝買い目がゾロ目になるか）")
    show("S1", S1 == 1, "ゾロ目になる")
    show("S1", S1 == 0, "ならない")
    d, lo, hi = two_sample(DU[S1 == 1], DU[S1 == 0])
    dh, _, _ = two_sample(DH[S1 == 1], DH[S1 == 0])
    print(f"  → 層間差(馬連→枠) {d:+.4f} [{lo:+.4f},{hi:+.4f}] "
          f"{'★' if (lo > 0 or hi < 0) else ''}   （従来のλH基準なら {dh:+.4f}）")

    print("\n■ S2 モデル上位3頭が2枠以内に収まるか（★(60)が+2.5ptと言った当の量）")
    show("S2", S2 == 1, "2枠以内（重なる）")
    show("S2", S2 == 0, "3枠に散る")
    d2, lo2, hi2 = two_sample(DU[S2 == 1], DU[S2 == 0])
    print(f"  → 層間差(馬連→枠) {d2:+.4f} [{lo2:+.4f},{hi2:+.4f}] "
          f"{'★' if (lo2 > 0 or hi2 < 0) else ''}")

    print("\n■ S3 枠集中度（モデルの枠別確率のΣp²）の十分位")
    dec = np.clip((np.argsort(np.argsort(HHI)) * 10) // n, 0, 9)
    ms = []
    for k in range(10):
        m, lo3, hi3 = mci(DU[dec == k])
        ms.append(m)
        print(f"  第{k+1:>2}十分位（{'散る' if k == 0 else '集中' if k == 9 else ''}）"
              f"{int((dec == k).sum()):>7}本  {m:+.4f} [{lo3:+.4f},{hi3:+.4f}]")
    rho = np.corrcoef(np.arange(10), np.array(ms))[0, 1]
    d3, lo3, hi3 = two_sample(DU[dec == 9], DU[dec == 0])
    print(f"  → 単調性 ρ={rho:+.3f}　最上位−最下位 {d3:+.4f} [{lo3:+.4f},{hi3:+.4f}] "
          f"{'★' if (lo3 > 0 or hi3 < 0) else ''}")

    # ── プラセボ（実装の検査。構造上0になるはず）──
    pl = []
    for _ in range(NPLA):
        z = RNG.random(n) < S1.mean()
        pl.append(DU[z].mean() - DU[~z].mean())
    print(f"\n  ⚠プラセボ（無作為な2分割・{NPLA}回平均）= {np.mean(pl):+.5f}"
          f"　※**0になるのが構造上あたりまえ**。統計はBonferroniと年分割が守っている（判定基準23）")

    # ── B. 買い方（1レース期待損失[円]・対応あり）─────────────────
    print("\n── B. ★ゾロ目になるレースで、紐を別枠に替えるべきか（判定基準8: 損失[円]で見る） ──")
    z = (S1 == 1) & ~np.isnan(PII)
    if z.sum() >= 100:
        li = 100.0 - PI[z]                 # どちらも常に1点＝100円
        lii = 100.0 - PII[z]
        mi, loi, hii = mci(li)
        mii, loii, hiii = mci(lii)
        dd, lod, hid = mci(li - lii)       # ★対応あり（同じレース）
        print(f"  対象 {int(z.sum()):,}レース（ゾロ目になる回）")
        print(f"  (i)  そのままゾロ目を買う   1R期待損失 {mi:>6.1f}円 [{loi:.1f},{hii:.1f}]"
              f"   ROI {100*PI[z].mean()/100:>5.1f}%  的中 {(PI[z] > 0).mean():.1%}")
        print(f"  (ii) 紐を別枠の次点に替える 1R期待損失 {mii:>6.1f}円 [{loii:.1f},{hiii:.1f}]"
              f"   ROI {100*PII[z].mean()/100:>5.1f}%  的中 {(PII[z] > 0).mean():.1%}")
        mark = "★(ii)が良い" if lod > 0 else ("★(i)が良い" if hid < 0 else "**区別できない**")
        print(f"  → 対応ある差 (i)−(ii) = {dd:+.1f}円 [{lod:+.1f},{hid:+.1f}]  {mark}")
        pos = sum(1 for yy in sorted(set(yrsA[z].tolist()))
                  if (li - lii)[yrsA[z] == yy].mean() > 0)
        print(f"     年分割: (ii)が良い年 {pos}/{len(set(yrsA[z].tolist()))}")
    else:
        print("  ゾロ目のレースが足りない")

    print("\n" + "=" * 100)
    print("★読み方（事前登録のとおり。**後から層を足していない**）")
    print("  ・A の層間差が **+0.005超・CI下端>0・年8/11以上** → レース選択の変数として使える。")
    print("    それ未満なら**記述にとどめ、運用は変えない**（(125)と同じ基準）。")
    print("  ・★**ゾロ目層で「差(馬連−λH)」が大きい** → (60)の85.6%と既知の「ゾロ目のみ−0.0726」は")
    print("    **同じものの表裏**だった＝Harvilleの壊れ方であって、市場の甘さではなかったことになる。")
    print("  ・B のCIが0をまたぐなら**買い方は変えない**。ROIで見ると必ず過大評価する（判定基準22）。")


if __name__ == "__main__":
    main()
