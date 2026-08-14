"""(148) ★★★★★(141)を**発走前の両プール**で決着させる — **唯一100%を超えた数字の最終判定**

⚠**枠連と馬連の時系列オッズが要る**。無ければ何もせず終了する。
　`data/odds_ts_waku/` と `data/odds_ts_umaren/`（TARGETから出す。手順は NEXT_SESSION）。

★★なぜこれが決定的か
　(141)は**両プールの確定オッズ**で ROI 117.3% [103.9,130.8]（全期間）を出した。
　**このプロジェクトで唯一100%を超えた数字**。だが**確定オッズは張り終えたあとに決まる**。
　(143)(144)で「10分前では駄目」と書いたが、⚠**あれは単勝由来のλHarvilleで測ったもの**で、
　**(141)が使う馬連ルートは一度も測っていない**（→(147)で判断を撤回した。**判定基準25**）。
　★**方向も違う**: (143)(144)が示したのは「**各プールの水準は遅く固まる**」。
　　(141)が使うのは「**2つのプールの食い違い**」。**両方が遅く固まっても食い違いは安定しうる**。
　→ ★**両プールの時系列を持って初めて答えが出る**。**それがこのスクリプト**。

★★事前登録（測る前に宣言する。**データが来る前に書いてある**）
　1. **時点は 前日21時 / 当日9時 / 60 / 30 / 20 / 10 / 5分前 / 確定**。**後から増やさない**。
　2. **閾値は(141)と同じ 1.290（=1/払戻率）に固定**。**動かさない**。
　3. **★2通り出す（両方とも事前に宣言する）**
　　 **(A) 現実版**: **時点tのオッズで選び、確定オッズで払い戻す**。
　　 　　←**これが実際にできること**。**主判定はこちら**。
　　 **(B) 対照版**: **時点tのオッズで選び、その時点のオッズで払い戻す**（架空）。
　　 　　←**「選別の質」だけを取り出す**。(A)と(B)の差が**払戻の流れ（他人の後乗り）の分**。
　4. **★主判定: (A)の10分前で ROIの99%CI下端 > 100%**。
　　 かつ**年別で「91%を割る年が2つ以下」**（★ユーザーの許容水準・2026-08-13。
　　 「毎年100%超えは要求しない。91%は許容する。40%とかは無理だけど」）。
　5. ⚠**検出力を先に書く**: 1年分＝約3,400レース。(141)は31,130レースで CI幅±13pt だった。
　　 → **1年では ±38pt 級**。**ROIだけでは判定できない見込みが高い**。
　　 ★**だから同時に精度の出る量を2つ出す**:
　　 　**① 再現率／適合率**（確定で選ぶ集合を時点tでどれだけ復元できるか。**組が数万あるので精密**）
　　 　**② 各時点の D**（馬連(t)→枠 vs 枠連(t)）。**(144)と同じ形だが今度は馬連ルート**。
　　 **①②が「10分前でも食い違いが在る」と言い、(A)が0をまたぐだけなら「標本不足」と書く**。
　　 **①②が「10分前には無い」と言うなら、そこで閉じる**。**この読み分けを先に決めておく**。
　6. **プラセボ**: レース内で比を組にランダム割り当てして同数買う。**解析的に出す**（判定基準23）。
　7. **予想**: ★**当てにしてよい予想は持っていない**（類推はこの3日で4連敗）。

実行: python3 ml/audit_overlay_ts.py
"""
import math
import sys

import numpy as np

sys.path.insert(0, "ml")
from audit_cond_split import load_boards
from audit_crosspool import PAYBACK, load_races, zq
from audit_crosspool2 import realized
from odds_ts_combo import load_dir, odds_at
from waku_umatan import waku_of

R = PAYBACK["枠連"]
TH = 1.0 / R
WHENS = [("prev", 21, 0), ("day", 9, 0), ("before", 60), ("before", 30),
         ("before", 20), ("before", 10), ("before", 5), ("final",)]
LAB = {("prev", 21, 0): "前日21時", ("day", 9, 0): "当日9時", ("before", 60): "60分前",
       ("before", 30): "30分前", ("before", 20): "20分前", ("before", 10): "10分前",
       ("before", 5): "5分前", ("final",): "確定"}
DW, DU = "data/odds_ts_waku", "data/odds_ts_umaren"


def main():
    tw, tu = load_dir(DW), load_dir(DU)
    if not tw or not tu:
        sys.exit(f"{DW} と {DU} の両方が要る。TARGETから枠連・馬連の時系列オッズを出すこと。\n"
                 f"　まず `python3 ml/odds_ts_combo.py {DW}` で中身が読めるか確かめる。")
    wb = load_boards()                       # 確定の枠連の板（払戻に使う）
    races = {r["rid"]: r for r in load_races()}
    ids = sorted(set(tw) & set(tu) & set(wb) & set(races))
    print(f"(148) (141)を発走前の両プールで決着させる")
    print(f"　枠連の時系列 {len(tw):,} / 馬連の時系列 {len(tu):,} / 突き合わせ {len(ids):,}レース")
    print(f"★閾値は(141)と同じ {TH:.3f} に固定。**主判定は(A)の10分前**\n")

    rows = []
    for rid in ids:
        r = races[rid]
        if not r.get("wakuren"):
            continue
        rl = realized(r)
        if rl is None:
            continue
        a, b, _ = rl
        n = r["n"]
        nums = [u for u, _, _ in r["horses"]]
        if a not in nums or b not in nums:
            continue
        key = tuple(sorted((waku_of(a, n), waku_of(b, n))))
        Wf = wb[rid]
        if key not in Wf:
            continue
        per = {}
        for w in WHENS:
            ow, ou = odds_at(tw[rid], w), odds_at(tu[rid], w)
            if not ow or not ou:
                continue
            agg = {}
            for (x, y), o in ou.items():
                if o <= 0 or x not in nums or y not in nums:
                    continue
                wx, wy = sorted((waku_of(x, n), waku_of(y, n)))
                agg[(wx, wy)] = agg.get((wx, wy), 0.0) + 1.0 / o
            keys = [k for k in sorted(agg) if k in ow]
            if key not in keys or len(keys) < 3:
                continue
            inv = np.array([1.0 / ow[k] for k in keys])
            qp = inv / inv.sum()
            qq = np.array([agg[k] for k in keys])
            qq /= qq.sum()
            j = keys.index(key)
            sel = {keys[i] for i in range(len(keys)) if qq[i] / qp[i] >= TH}
            per[w] = dict(sel=sel, ow=ow, d=math.log(qq[j]) - math.log(qp[j]))
        if ("final",) not in per:
            continue
        rows.append(dict(y=r["year"], key=key, Wf=Wf, per=per))

    if not rows:
        sys.exit("突き合わせできたレースが無い")
    print(f"対象 {len(rows):,}レース\n")

    print("■ ★★(A) 現実版: **時点tで選び、確定オッズで払い戻す**（これが実際にできること）")
    print(f"{'時点':>10}{'買ったR':>9}{'点数':>8}{'的中':>6}{'ROI':>9}{'99%CI':>20}"
          f"{'再現率':>8}{'適合率':>8}{'D':>10}{'(B)ROI':>9}{'プラセボ':>9}")
    for w in WHENS:
        prof, cost, ret, nb, hit, yl = [], 0.0, 0.0, 0, 0, []
        bret, bcost = 0.0, 0.0
        plr, plc = 0.0, 0.0
        inter = [0, 0, 0]
        ds = []
        for r in rows:
            p = r["per"].get(w)
            if p is None:
                continue
            fs = r["per"][("final",)]["sel"]
            inter[0] += len(p["sel"] & fs)
            inter[1] += len(fs)
            inter[2] += len(p["sel"])
            ds.append(p["d"])
            if not p["sel"]:
                continue
            c = 100.0 * len(p["sel"])
            v = 100.0 * r["Wf"][r["key"]] if r["key"] in p["sel"] else 0.0
            prof.append(v - c)
            yl.append(r["y"])
            cost += c
            ret += v
            nb += len(p["sel"])
            hit += int(r["key"] in p["sel"])
            # (B) その時点のオッズで払い戻す（架空）
            bcost += c
            bret += 100.0 * p["ow"].get(r["key"], 0.0) if r["key"] in p["sel"] else 0.0
            # プラセボ（解析的）: 無作為に同数選ぶ期待払戻 = そのレースの平均払戻 × 点数
            m = len(p["ow"])
            plr += (100.0 * r["Wf"][r["key"]] / m) * len(p["sel"]) if m else 0.0
            plc += c
        if not prof:
            print(f"{LAB[w]:>10}   該当なし")
            continue
        pr = np.array(prof)
        mc = cost / len(pr)
        se = pr.std(ddof=1) / math.sqrt(len(pr))
        z = zq(0.01)
        lo, hi = 1 + (pr.mean() - z * se) / mc, 1 + (pr.mean() + z * se) / mc
        mark = "★★" if lo > 1.0 else ""
        print(f"{LAB[w]:>10}{len(pr):>9,}{nb:>8,}{hit:>6}{100*ret/cost:>8.1f}%"
              f"{'[' + format(100*lo, '.0f') + ',' + format(100*hi, '.0f') + ']':>20}"
              f"{inter[0]/max(inter[1],1):>7.1%}{inter[0]/max(inter[2],1):>8.1%}"
              f"{np.mean(ds):>+10.4f}{100*bret/max(bcost,1):>8.1f}%"
              f"{100*plr/max(plc,1):>8.1f}% {mark}")

    print("\n" + "=" * 100)
    print("★読み方（事前登録のとおり。**後から足していない**）")
    print("  ・**主判定は(A)の10分前で 99%CI下端 > 100%**、かつ年別で91%割れが2つ以下。")
    print("  ⚠1年分だとCI幅は±38pt級＝**ROIだけでは判定できない見込み**。**先に書いてある**。")
    print("  ★**その場合は 再現率／適合率／D で読む**:")
    print("    ・**10分前でも再現率が高く D が正** → 食い違いは早くから在る＝**標本不足**と書く。")
    print("    ・**10分前で D が負** → **食い違いは締切間際に生成される**＝**そこで閉じる**。")
    print("  ・(A)と(B)の差 = **他人の後乗りで配当が削られる分**。")


if __name__ == "__main__":
    main()
