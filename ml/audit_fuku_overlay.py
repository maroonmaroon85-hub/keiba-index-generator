"""(146) ★★★★★複勝を**三連複プール**で値付けして比の裾を買う — **残っている中で一番見込みがある**

⚠**複勝の板（`--type 2`）が届いてから回す**。届くまでは何もせず終了する。

★★★なぜ複勝が一番見込みがあるのか（4つとも枠連より有利）
　1. **控除率が一番低い**。払戻率 **0.80**（枠連0.775・三連複0.75）。
　　 → **利益に要る比は 1/0.80 = 1.250**（枠連は1.290）。**ハードルが低い**。
　2. **集約が厳密**。三連複は「上位3頭の集合」の分布なので
　　 　`P(馬i が3着以内) = Σ_{j,k} q_三連複({i,j,k})` が**厳密に成り立つ**。
　　 　**Harvilleもモデルもλも要らない**——(127)で枠連に効いたのと同じ形。
　3. **プールが厚い**。三連複は日本で最も売れる券種の一つで、**複勝も深い**。
　　 → **(142)(b)の容量の問題が枠連よりずっと軽い**（枠連は薄くて1点1万円で配当−4%だった）。
　4. **このプロジェクトの最良ROIは複勝 96.8%**((106))＝**100%に一番近づいた場所**。
　　 ⚠ただし(106)は **max-q（最も当たりやすい馬）で選んだ**もので、
　　 　**比 q/q_pool の裾で選んだことは一度も無い**。**判定基準30の「平均で閉じても裾では閉じていない」**。

★★事前登録（測る前に宣言する。**データが来る前に書いてある**）
　1. **q（我々の確率）**: 三連複の板から `q_i = Σ_{j,k} q_trio({i,j,k})`。
　　 **板の全三連複で正規化してから足す**（判定基準14①）。**Σ_i q_i = 3 になるはず＝検算に使う**。
　2. **q_pool（複勝プール）**: 複勝の板から `1/オッズ`。
　　 ⚠**複勝のオッズは [下限, 上限] の範囲**（最低保証の再配分で着順の組み合わせに依存するため）。
　　 　→ ★**下限を主とする**（1/下限＝含意確率が大きい＝**比が小さくなる＝保守側**）。
　　 　　 **上限版も感度として併記する**。**主判定は下限版**。
　　 　 **Σ_i q_pool,i = 3 に正規化する**（★3頭が必ず3着以内に入るので。**これも検算になる**）。
　3. **対象は8頭以上**（7頭以下は複勝が2着まで＝別の券種になる）。
　4. **閾値の梯子 1.00 / 1.10 / **1.250（=1/払戻率）** / 1.40 / 1.60**。**後から増やさない**。
　5. **賭け方**: 比が閾値以上の**全馬を1点100円ずつ**。該当が無いレースは見送り。**選ばない**。
　6. **★判定は race単位の収支**（馬の間は従属なので馬単位でCIを作らない）。
　　 **運用判定: ROIの99%CI下端 > 100% かつ 年8/11以上でプラス**。
　7. **★時間分割を最初から本判定に入れる**（(142)の教訓）。
　　 **2021-2026 だけでも CI下端 > 100%** を要求する。**全期間で出ても直近で出なければ採らない**。
　8. **プラセボ**: レース内で比を馬にランダム割り当てして同数買う。200回平均。
　9. ⚠**確定オッズのオラクルである**ことは枠連と同じ。**(143)(144)の壁は複勝にも掛かる**。
　　 ★ただし**複勝は三連複と同じ「上位3頭」の情報**を見ており、**単勝より値動きが緩い可能性がある**。
　　 　**そこは(146)では分からない**。**通ったら次に時系列で確かめる**。**順序を守る**。
　10. **予想**: ★**当てにしてよい予想は持っていない**（類推はこの3日で4連敗）。

実行: python3 ml/audit_fuku_overlay.py [開始年(既定2015)]
"""
import math
import sys
import numpy as np

sys.path.insert(0, "ml")
from audit_crosspool import PAYBACK, load_races, payoff, zq

R = PAYBACK["複勝"]
THS = [1.00, 1.10, 1.0 / R, 1.40, 1.60]
NPLA = 200
RNG = np.random.default_rng(20260813)


def load_board(t, keylen):
    """{race_id8: {キー: 値}}。複勝(t=2)は [下限, 上限] を持つ。"""
    from nk_odds_bulk import iter_records
    from nk_parse import nk_raceid
    out = {}
    for rec in iter_records(t):
        r8 = nk_raceid(rec["race_id"])
        if not r8:
            continue
        d = {}
        for k, v in rec["odds"].items():
            if len(k) != keylen or not k.isdigit():
                continue
            vals = list(v) if isinstance(v, (list, tuple)) else [v]
            vals = [float(x) for x in vals if x and float(x) > 0]
            if vals:
                d[k] = vals
        if d:
            out[r8] = d
    return out


def main():
    y0 = int(sys.argv[1]) if len(sys.argv) > 1 else 2015
    fb = load_board(2, 2)          # 複勝の板 {'01': [下限, 上限]}
    tb = load_board(7, 6)          # 三連複の板 {'010203': [オッズ]}
    if not fb:
        sys.exit("複勝の板(--type 2)がまだ無い。Macで `python3 ml/nk_odds_bulk.py --type 2` を回して push すること。")
    if not tb:
        sys.exit("三連複の板(--type 7)が無い。")
    races = {r["rid"]: r for r in load_races()}
    print(f"(146) 複勝を三連複プールで値付けして比の裾を買う")
    print(f"　複勝の板 {len(fb):,} / 三連複の板 {len(tb):,} / レース {len(races):,}")
    print(f"★利益に要る比 = 1/払戻率 = {1/R:.3f}（枠連の1.290より低い）\n")

    dat = []
    for rid, r in races.items():
        if r["year"] < y0 or r["n"] < 8:
            continue
        F, T = fb.get(rid), tb.get(rid)
        if not F or not T:
            continue
        nums = [u for u, _, _ in r["horses"]]
        idx = {u: i for i, u in enumerate(nums)}
        # ── q: 三連複の板 → 各馬の3着以内確率（厳密。Σ=3 になる）──
        tri, w = [], []
        for k, v in T.items():
            a, b, c = int(k[:2]), int(k[2:4]), int(k[4:])
            if a in idx and b in idx and c in idx:
                tri.append((idx[a], idx[b], idx[c]))
                w.append(1.0 / v[0])
        if len(tri) < 10:
            continue
        w = np.array(w)
        w /= w.sum()
        q = np.zeros(len(nums))
        for (i, j, k), ww in zip(tri, w):
            q[i] += ww
            q[j] += ww
            q[k] += ww
        # ── q_pool: 複勝の板（下限 / 上限）──
        qp = {}
        for which, pos in (("lo", 0), ("hi", -1)):
            v = np.zeros(len(nums))
            ok = True
            for u, i in idx.items():
                key = f"{u:02d}"
                if key not in F:
                    ok = False
                    break
                v[i] = 1.0 / F[key][min(pos, len(F[key]) - 1)]
            if not ok or v.sum() <= 0:
                qp = None
                break
            qp[which] = v / v.sum() * 3.0
        if qp is None:
            continue
        pay = np.array([payoff(r, "複勝", (u,)) or 0.0 for u in nums])
        dat.append(dict(y=r["year"], q=q, qp=qp, pay=pay))

    if not dat:
        sys.exit("突き合わせできたレースが無い")
    ys = np.array([d["y"] for d in dat])
    print(f"対象 {len(dat):,}レース（{ys.min()}〜{ys.max()}）")
    print(f"★検算: Σq（三連複由来）の中央値 = {np.median([d['q'].sum() for d in dat]):.4f}"
          f"（**3.0000 が正しい**）")
    print(f"★検算: Σq_pool（複勝の板・下限）は 3.0 に正規化済み\n")

    def run(which, mask):
        out = {}
        for t in THS:
            prof, cost, ret, nb, hit, yl = [], 0.0, 0.0, 0, 0, []
            pl = np.zeros(NPLA)
            plc = np.zeros(NPLA)
            for i in np.where(mask)[0]:
                d = dat[i]
                sel = (d["q"] / np.maximum(d["qp"][which], 1e-12)) >= t
                if not sel.any():
                    continue
                c = 100.0 * sel.sum()
                v = float(d["pay"][sel].sum())
                prof.append(v - c)
                yl.append(d["y"])
                cost += c
                ret += v
                nb += int(sel.sum())
                hit += int((d["pay"][sel] > 0).sum())
                for j in range(NPLA):
                    m2 = np.zeros(len(d["q"]), bool)
                    m2[RNG.permutation(len(d["q"]))[: int(sel.sum())]] = True
                    plc[j] += c
                    pl[j] += float(d["pay"][m2].sum())
            if not prof:
                continue
            p = np.array(prof)
            mc = cost / len(p)
            se = p.std(ddof=1) / math.sqrt(len(p))
            z = zq(0.01)
            ya = np.array(yl)
            pos = sum(1 for u in sorted(set(ya.tolist()))
                      if (ya == u).sum() >= 30 and p[ya == u].mean() > 0)
            out[t] = (len(p), nb, hit, ret / cost,
                      1 + (p.mean() - z * se) / mc, 1 + (p.mean() + z * se) / mc,
                      float(np.mean(pl / np.maximum(plc, 1))), pos, len(set(ya.tolist())))
        return out

    for lab, mask in (("全期間", np.ones(len(dat), bool)),
                      ("★★2021-2026（(142)の教訓で本判定に入れた）", ys >= 2021),
                      ("2015-2020", ys <= 2020)):
        print(f"■ {lab}")
        print(f"{'閾値':>7}{'買ったR':>9}{'点数':>10}{'的中':>8}{'ROI(下限)':>11}"
              f"{'99%CI':>22}{'年+':>7}{'ROI(上限)':>11}{'プラセボ':>10}")
        a = run("lo", mask)
        b = run("hi", mask)
        for t in THS:
            if t not in a:
                continue
            nR, nb, hit, roi, lo, hi, pv, pos, ny = a[t]
            roi2 = b[t][3] if t in b else float("nan")
            m = "★★買える" if lo > 1.0 else ""
            print(f"{t:>7.3f}{nR:>9,}{nb:>10,}{hit:>8,}{100*roi:>10.1f}%"
                  f"{'[' + format(100*lo, '.1f') + ',' + format(100*hi, '.1f') + ']':>22}"
                  f"{pos:>4}/{ny}{100*roi2:>10.1f}%{100*pv:>9.1f}% {m}")
        print()

    print("=" * 104)
    print("★読み方（事前登録のとおり）")
    print("  ・**全期間と 2021-2026 の両方で 99%CI下端 > 100% かつ年8/11以上**なら★。")
    print("    **全期間だけで出ても直近で出なければ採らない**（(142)の教訓）。")
    print("  ・ROI(上限)は感度。**主判定は下限版**（保守側）。両者が大きく違えば範囲の扱いを疑う。")
    print("  ⚠**確定オッズのオラクル**。通っても**次に時系列で確かめる**まで運用に入れない（(143)(144)）。")


if __name__ == "__main__":
    main()
