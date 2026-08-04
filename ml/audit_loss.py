"""D5: ★★買い方を **ROIではなく「1レースあたりの期待損失(円)」** で順位付けし直す。

★動機（(80)のD2で数字が出たので初めて言えるようになったこと）
`audit_tails.py` §5 が、この標本量での**検出限界**を出した:

    枠連 軸枠×紐枠2 … 2.20pt（1ptの差を検出するには 155,091R ＝ 約50年分）
    三連複 BOX上位4 … 3.55pt（現行の対人気順 +3.52pt は**この限界に届いていない**）
    複勝 top1      … 0.82pt

つまり**買い方どうしの1〜2ptのROI差は、このデータでは原理的に測れない**。
ところが本プロジェクトは①〜(79)を通じて**一貫してROIで買い方を選んで**きた（(77)の29通り横断も
「ROIで現行を置き換えるものは無い」という結論）。**測れない量で選んでいた**ことになる。

★そこで量を変える。どの買い方もROI<100%なので、実際に財布から出ていくのは
    **1レース期待損失 = 平均コスト − 平均払戻 [円]**
で、これは**コストという誤差ゼロの項を含む**。ROI差が誤差に埋もれていても、
コスト差(例: 194円 vs 100円)は**測定を要さずに確定**している。

  ⚠ ただし「コストが誤差ゼロだから損失差も誤差ゼロ」ではない。
    損失差 = コスト差 − 払戻差 であり、**払戻差には標本誤差がある**。
    だが払戻差は**同じレースでの対応あり**で測れる（買い増した券の追加払戻ぶんだけ）ので、
    ROIどうしの差より分散がずっと小さい。**そこを実測して確かめる**のがこのスクリプト。

★事前宣言する仮説（(77)②の3点セット a.事前宣言）
  H1: 期待損失で並べ替えると、**点数の少ない買い方が上位に来る**。
  H2: 現行 枠連 軸枠×紐枠2（平均194円）は、**枠連 軸枠×紐枠1（100円）と 複勝 top1（100円）に負ける**。
  H3: 損失差のCIは、同じ比較のROI差のCIより**相対的に狭い**（＝コスト項が誤差を持たないぶん）。
  ※H1が外れる（＝点数を増やした方が損失が小さい）なら、その買い方は払戻がコスト増を上回っており、
    それはそれで重要な発見になる。どちらに転んでも結論が出る設計。

守った作法: 3(人気順の対照) 6(プラセボ対照＝同じ一致率のノイズ順) 5c(対応あり比較)
           2(標本誤差＋検出限界) 7(的中率も併記) 判定基準5(年別で安定性)
　　　　　 ★シード対照(判定基準1)は本スクリプトでは当たらない。`audit_units.py` が作った
　　　　　 シード平均の予測を再利用しているため。**シード幅は `audit_himo1.py` で別途出す**。

前提: `python3 ml/audit_units.py` で /tmp/units_races.pkl を作っておくこと（学習は不要）。
実行: python3 ml/audit_loss.py
"""
import itertools
import pickle
import sys

import numpy as np

sys.path.insert(0, "ml")
from place_wide import load_place_wide
from pocket_eval import load_payout_a
from waku_umatan import load_wu, waku_of

PAYOUT = "data/payout/a.csv"
SIGMAS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1.0]
N_DRAW = 6


# ---- 買い方の定義。order（馬番の並び）と頭数から (点数, 払戻合計) を返す ----
def wakuren(order, n, k, pay):
    """軸枠 × 上位k頭の枠。重複除去するので点数はk以下。"""
    wa = waku_of(order[0], n)
    cs = sorted({tuple(sorted((wa, waku_of(h, n)))) for h in order[1:1 + k]})
    return len(cs) * 100.0, sum(pay.get(c, 0) for c in cs)


def sanrenpuku_box(order, n, k, pay):
    cs = list(itertools.combinations(sorted(order[:k]), 3))
    return len(cs) * 100.0, sum(pay.get(tuple(sorted(c)), 0) for c in cs)


def fukusho(order, n, k, pay):
    return k * 100.0, sum(pay.get(int(h), 0) for h in order[:k])


MENU = [
    # (表示名, 関数, k, 券種キー)
    ("枠連 軸枠×紐枠1", wakuren, 1, "wk"),
    ("枠連 軸枠×紐枠2 ★現行", wakuren, 2, "wk"),
    ("枠連 軸枠×紐枠3", wakuren, 3, "wk"),
    ("三連複 BOX上位3", sanrenpuku_box, 3, "s3"),
    ("三連複 BOX上位4", sanrenpuku_box, 4, "s3"),
    ("複勝 top1", fukusho, 1, "fk"),
    ("複勝 top2", fukusho, 2, "fk"),
]


def boot_mean(x, rng, n=4000):
    b = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)])
    return np.percentile(b, 2.5), np.percentile(b, 97.5)


def main():
    with open("/tmp/units_races.pkl", "rb") as fh:
        races = pickle.load(fh)
    wu, pa, pw = load_wu(PAYOUT), load_payout_a(PAYOUT), load_place_wide(PAYOUT)

    # 券種ごとに成立したレースだけを使う
    pays = {}
    for r in races:
        w, s3, fw = wu.get(r["rid"]), pa.get(r["rid"]), pw.get(r["rid"])
        pays[r["rid"]] = {
            "wk": w["wakuren"] if (w and w["wakuren"]) else None,
            "s3": s3["sanrenpuku"] if (s3 and s3["sanrenpuku"] and r["n"] >= 9) else None,
            "fk": fw["fuku"] if (fw and fw.get("fuku")) else None,
        }

    pops = {r["rid"]: r["uma"][np.argsort(r["lo"], kind="mergesort")] for r in races}

    def run(order_of, sel_key):
        """order_of(r) が返す並びで、券種sel_keyの全買い方を評価する。"""
        out = {}
        for nm, fn, k, key in MENU:
            if key != sel_key:
                continue
            cost, pay, rid = [], [], []
            for r in races:
                p = pays[r["rid"]][key]
                if p is None:
                    continue
                c, v = fn(order_of(r), r["n"], k, p)
                cost.append(c)
                pay.append(v)
                rid.append(r["rid"])
            out[nm] = (np.array(cost), np.array(pay), rid)
        return out

    # --- プラセボの σ を券種ごとに合わせる（(79)①と同じ作法）---
    def placebo_order(sg, seed):
        g = np.random.default_rng(seed)
        return {r["rid"]: r["uma"][np.argsort(r["lo"] + g.normal(0, sg, len(r["uma"])),
                                              kind="mergesort")] for r in races}

    def agree_rate(orders, nm, fn, k, key):
        """買い目が人気順と一致する率。"""
        a = []
        for r in races:
            p = pays[r["rid"]][key]
            if p is None:
                continue
            wa_m = fn(orders[r["rid"]], r["n"], k, p)
            wa_p = fn(pops[r["rid"]], r["n"], k, p)
            a.append(wa_m == wa_p)
        return float(np.mean(a))

    print(f"読み込み {len(races):,}R（{races[0]['year']}〜{races[-1]['year']}）")
    print("プラセボの σ を『枠連 軸枠×紐枠2』の買い目一致率に合わせる（(79)①と同じ）…")
    mo = {r["rid"]: r["mo"] for r in races}
    nm2, fn2, k2, key2 = MENU[1]
    am = agree_rate(mo, nm2, fn2, k2, key2)
    curve = [(sg, agree_rate(placebo_order(sg, 300), nm2, fn2, k2, key2)) for sg in SIGMAS]
    xs = np.array([c[1] for c in curve])
    sg = float(np.interp(am, xs[::-1], np.array(SIGMAS)[::-1]))
    print(f"  モデルの一致率 {am*100:.1f}% → σ={sg:.3f}")
    pls = [placebo_order(sg, 400 + i) for i in range(N_DRAW)]

    rng = np.random.default_rng(0)
    print(f"\n{'='*118}")
    print("★買い方を『1レース期待損失(円)』で並べ替える（同じ順序＝モデル。人気順/プラセボは対照）")
    print(f"{'買い方':<22}{'R':>8}{'平均コスト':>10}{'ROI(等額)':>11}{'的中率':>8}"
          f"{'★期待損失':>10}{'その95%CI':>18}{'人気順の損失':>12}{'プラセボの損失':>14}")
    rows = []
    for nm, fn, k, key in MENU:
        m = run(lambda r: mo[r["rid"]], key)[nm]
        p = run(lambda r: pops[r["rid"]], key)[nm]
        lcost, lpay = [], []
        for pl in pls:
            c, v, _ = run(lambda r, pl=pl: pl[r["rid"]], key)[nm]
            lcost.append(c)
            lpay.append(v)
        lc, lv = np.mean(lcost, axis=0), np.mean(lpay, axis=0)

        mc, mv, rid = m
        pc, pv, _ = p
        loss = mc - mv                       # 1レースの純損失（円）
        lo, hi = boot_mean(loss, rng)
        rows.append((nm, key, fn, k, rid, mc, mv, pc, pv, lc, lv))
        print(f"{nm:<22}{len(mc):>8,}{mc.mean():>9.1f}円{(mv/mc).mean()*100:>10.2f}%"
              f"{(mv > 0).mean()*100:>7.2f}%{loss.mean():>9.1f}円"
              f"{f'[{lo:.1f},{hi:.1f}]':>18}{(pc-pv).mean():>11.1f}円{(lc-lv).mean():>13.1f}円")

    # --- 対応あり比較: 現行(枠連 紐2)との1レース損失差 ---
    base = [r for r in rows if r[0].startswith("枠連 軸枠×紐枠2")][0]
    bl = dict(zip(base[4], base[5] - base[6]))          # rid -> 現行の損失
    bpos = {x: i for i, x in enumerate(base[4])}        # rid -> 現行側の行番号
    print(f"\n{'='*118}")
    print("★現行（枠連 軸枠×紐枠2）との対応あり比較 ＝ 同じレースで測った『1レース損失の差』")
    print("  （マイナス＝現行より損が小さい。ROI差ではなく円で見る）")
    print(f"{'買い方':<22}{'共通R':>8}{'損失差':>10}{'95%CI':>20}{'内訳:コスト差':>14}"
          f"{'内訳:払戻差':>13}{'ROI差(参考)':>13}{'そのCI':>20}")
    for nm, key, fn, k, rid, mc, mv, pc, pv, lc, lv in rows:
        idx = [i for i, x in enumerate(rid) if x in bl]
        d = (mc[idx] - mv[idx]) - np.array([bl[rid[i]] for i in idx])
        lo, hi = boot_mean(d, rng)
        bi = np.array([bpos[rid[i]] for i in idx])
        bc, bv = base[5][bi], base[6][bi]
        dr = (mv[idx] / mc[idx]) - (bv / bc)
        rlo, rhi = boot_mean(dr * 100, rng)
        print(f"{nm:<22}{len(idx):>8,}{d.mean():>+9.1f}円{f'[{lo:+.1f},{hi:+.1f}]':>20}"
              f"{(mc[idx]-bc).mean():>+13.1f}円{(mv[idx]-bv).mean():>+12.1f}円"
              f"{dr.mean()*100:>+12.2f}pt{f'[{rlo:+.2f},{rhi:+.2f}]':>20}")

    # --- H3: 損失差とROI差、どちらが相対的に精密か ---
    print(f"\n{'='*118}")
    print("★H3の検証: 同じ比較を『損失差(円)』と『ROI差(pt)』で測ったときの、CI幅 / 点推定")
    print("  （小さいほど精密。損失差の方が小さければ、コスト項に誤差が無いぶん得をしている）")
    print(f"{'買い方':<22}{'損失差CI幅/|点推定|':>22}{'ROI差CI幅/|点推定|':>22}{'判定':>14}")
    for nm, key, fn, k, rid, mc, mv, pc, pv, lc, lv in rows:
        if nm.startswith("枠連 軸枠×紐枠2"):
            continue
        idx = [i for i, x in enumerate(rid) if x in bl]
        d = (mc[idx] - mv[idx]) - np.array([bl[rid[i]] for i in idx])
        lo, hi = boot_mean(d, rng)
        bi = np.array([bpos[rid[i]] for i in idx])
        bc, bv = base[5][bi], base[6][bi]
        dr = (mv[idx] / mc[idx]) - (bv / bc)
        rlo, rhi = boot_mean(dr * 100, rng)
        a = (hi - lo) / abs(d.mean()) if d.mean() else np.inf
        b = (rhi - rlo) / abs(dr.mean() * 100) if dr.mean() else np.inf
        print(f"{nm:<22}{a:>22.2f}{b:>22.2f}{('損失差が精密' if a < b else 'ROI差が精密'):>14}")

    # --- 年別（判定基準5）---
    print(f"\n{'='*118}")
    print("★年別の1レース期待損失（円・モデル順）")
    yrs = sorted({r["year"] for r in races})
    yof = {r["rid"]: r["year"] for r in races}
    print(f"{'買い方':<22}" + "".join(f"{y:>7}" for y in yrs) + f"{'現行に勝った年':>14}")
    blyr = {}
    for nm, key, fn, k, rid, mc, mv, pc, pv, lc, lv in rows:
        ys = np.array([yof[x] for x in rid])
        loss = mc - mv
        cells = []
        win = 0
        for y in yrs:
            s = ys == y
            v = loss[s].mean() if s.sum() else np.nan
            cells.append(v)
            if nm.startswith("枠連 軸枠×紐枠2"):
                blyr[y] = v
            elif not np.isnan(v) and v < blyr.get(y, np.inf):
                win += 1
        print(f"{nm:<22}" + "".join(f"{c:>7.0f}" for c in cells) +
              (f"{'—':>14}" if nm.startswith("枠連 軸枠×紐枠2") else f"{f'{win}/{len(yrs)}':>14}"))


if __name__ == "__main__":
    main()
