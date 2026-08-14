"""(148) ★★★★★(141)を**発走前の両プール**で決着させる — **唯一100%を超えた数字の最終判定**

⚠**枠連と馬連の時系列オッズが要る**（TARGETの「指定時系列オッズ(CSV形式)」）。
　`data/odds_ts_waku/` と `data/odds_ts_umaren/`。無ければ何もせず終了する。

★★なぜこれが決定的か
　(141)は**両プールの確定オッズ**で ROI 117.3% [103.9,130.8]（全期間）を出した。
　**このプロジェクトで唯一100%を超えた数字**。だが**確定オッズは張り終えたあとに決まる**。
　(143)(144)で「10分前では駄目」と書いたが、⚠**あれは単勝由来のλHarvilleで測ったもの**で、
　**(141)が使う馬連ルートは一度も測っていない**（→(147)で判断を撤回。**判定基準25**）。
　★**方向も違う**: (143)(144)が示したのは「**各プールの水準は遅く固まる**」。
　　(141)が使うのは「**2つのプールの食い違い**」。**両方が遅く固まっても食い違いは安定しうる**。

⚠★★★**事前登録から1点ずれた（データを見る前に書いた宣言との差。正直に記録する）**
　当初は **前日21時/当日9時/60/30/20/10/5分前/確定 の8時点**と宣言した。
　**実データ（TARGETの指定時系列オッズ）は1レース4時点しか無い**:
　　**前日夜（〜21時）／当日朝（〜9時）／★直前／確定**
　★**「直前」は実測36レースで発走の11分前でほぼ一定**（10:00発走→09:49、10:30→10:19…）。
　→ ★**主判定の「10分前」はこの「直前」スナップで置き換える**。**中身は宣言どおり**。
　　 **60/30/20/5分前は取得できないので落とす**。**判定そのものは変えていない**。
　⚠**これはデータの制約による差し替えであって、結果を見てからの変更ではない**（測る前に書いている）。

★★事前登録（残りは当初のまま）
　1. **閾値は(141)と同じ 1.290（=1/払戻率）に固定**。**動かさない**。
　2. **★2通り出す**
　　 **(A) 現実版**: **時点tのオッズで選び、確定オッズで払い戻す** ←**これが実際にできること。主判定**。
　　 **(B) 対照版**: **時点tのオッズで選び、その時点のオッズで払い戻す**（架空）。
　　 　　(A)と(B)の差が**他人の後乗りで配当が削られる分**。
　3. **★主判定: (A)の「直前」で ROIの99%CI下端 > 100%**、
　　 かつ**年別で「91%を割る年が2つ以下」**（ユーザーの許容水準・2026-08-13）。
　4. ⚠**検出力を先に書く**: 1年分＝約3,400レース。(141)は31,130レースでCI幅±13ptだった。
　　 → **1年では±38pt級**。**ROIだけでは判定できない見込みが高い**。
　　 ★**だから同時に精度の出る量を2つ出す**:
　　 　**① 再現率／適合率**（確定で選ぶ集合を時点tでどれだけ復元できるか。**組が多いので精密**）
　　 　**② 各時点の D**（馬連(t)→枠 vs 枠連(t)）。**(144)と同じ形だが今度は馬連ルート**。
　　 **①②が「直前でも食い違いが在る」と言い、(A)が0をまたぐだけなら「標本不足」と書く**。
　　 **①②が「直前には無い」と言うなら、そこで閉じる**。**この読み分けを先に決めておく**。
　5. **プラセボ**: 無作為に同数買う。**解析的に出す**（判定基準23）。
　6. **道具の検算**: 時系列の「確定」と netkeiba の板（(141)が使ったもの）を突き合わせる。
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
from odds_ts_combo import LABELS, load_dir
from waku_umatan import waku_of

R = PAYBACK["枠連"]
TH = 1.0 / R
DW, DU = "data/odds_ts_waku", "data/odds_ts_umaren"


def main():
    tw, tu = load_dir(DW), load_dir(DU)
    if not tw or not tu:
        sys.exit(f"{DW} と {DU} の両方が要る。TARGETの「指定時系列オッズ(CSV形式)」で\n"
                 f"　枠連と馬連を出すこと。まず `python3 ml/odds_ts_combo.py {DW}` で読めるか確かめる。")
    wb = load_boards()
    races = {r["rid"]: r for r in load_races()}
    ids = sorted(set(tw) & set(tu) & set(races))
    print("(148) (141)を発走前の両プールで決着させる")
    print(f"　枠連の時系列 {len(tw):,} / 馬連の時系列 {len(tu):,} / レースデータと一致 {len(ids):,}")
    print(f"★閾値は(141)と同じ {TH:.3f} に固定。**主判定は(A)の「直前」（実測=発走11分前）**")
    print("⚠事前登録の8時点は取得できず4時点。**「10分前」を「直前」で置き換えた**（冒頭に明記）\n")

    rows, chk_n, chk_bad = [], 0, 0
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
        per = {}
        for lab in LABELS:
            sw = tw[rid]["snaps"].get(lab)
            su = tu[rid]["snaps"].get(lab)
            if not sw or not su or not sw[1] or not su[1]:
                continue
            agg = {}
            for (x, y), o in su[1].items():
                if o <= 0 or x not in nums or y not in nums:
                    continue
                wx, wy = sorted((waku_of(x, n), waku_of(y, n)))
                agg[(wx, wy)] = agg.get((wx, wy), 0.0) + 1.0 / o
            keys = [k for k in sorted(agg) if k in sw[1]]
            if key not in keys or len(keys) < 3:
                continue
            inv = np.array([1.0 / sw[1][k] for k in keys])
            qp = inv / inv.sum()
            qq = np.array([agg[k] for k in keys])
            qq /= qq.sum()
            j = keys.index(key)
            per[lab] = dict(sel={keys[i] for i in range(len(keys)) if qq[i] / qp[i] >= TH},
                            odds=sw[1], d=math.log(qq[j]) - math.log(qp[j]))
        if "確定" not in per:
            continue
        # 道具の検算: 時系列の確定 vs netkeibaの板
        nb = wb.get(rid)
        if nb and key in nb:
            chk_n += 1
            o1, o2 = per["確定"]["odds"].get(key), nb[key]
            if o1 and abs(o1 - o2) > max(0.2, o2 * 0.02):
                chk_bad += 1
        rows.append(dict(y=r["year"], key=key, per=per))

    if not rows:
        sys.exit("突き合わせできたレースが無い")
    print(f"対象 {len(rows):,}レース")
    print(f"★道具の検算: 時系列の「確定」と netkeibaの板 のずれ {chk_bad}/{chk_n}"
          f"（{chk_bad/max(chk_n,1):.1%}）\n")

    print("■ ★★(A) 時点tで選び、確定オッズで払い戻す（これが実際にできること）")
    print(f"{'時点':>7}{'買ったR':>9}{'点数':>7}{'的中':>6}{'ROI':>9}{'99%CI':>20}"
          f"{'再現率':>8}{'適合率':>8}{'D':>10}{'(B)ROI':>9}{'プラセボ':>9}")
    for lab in LABELS:
        prof, cost, ret, nbt, hit, yl = [], 0.0, 0.0, 0, 0, []
        bret, bcost, plr, plc = 0.0, 0.0, 0.0, 0.0
        inter, ds = [0, 0, 0], []
        for r in rows:
            p = r["per"].get(lab)
            if p is None:
                continue
            fs = r["per"]["確定"]["sel"]
            fo = r["per"]["確定"]["odds"]
            inter[0] += len(p["sel"] & fs)
            inter[1] += len(fs)
            inter[2] += len(p["sel"])
            ds.append(p["d"])
            if not p["sel"] or r["key"] not in fo:
                continue
            c = 100.0 * len(p["sel"])
            v = 100.0 * fo[r["key"]] if r["key"] in p["sel"] else 0.0
            prof.append(v - c)
            yl.append(r["y"])
            cost += c
            ret += v
            nbt += len(p["sel"])
            hit += int(r["key"] in p["sel"])
            bcost += c
            bret += 100.0 * p["odds"].get(r["key"], 0.0) if r["key"] in p["sel"] else 0.0
            plr += (100.0 * fo[r["key"]] / len(fo)) * len(p["sel"])
            plc += c
        if not prof:
            print(f"{lab:>7}   該当なし")
            continue
        pr = np.array(prof)
        mc = cost / len(pr)
        se = pr.std(ddof=1) / math.sqrt(len(pr)) if len(pr) > 1 else float("nan")
        z = zq(0.01)
        lo, hi = 1 + (pr.mean() - z * se) / mc, 1 + (pr.mean() + z * se) / mc
        mark = "★★" if lo > 1.0 else ""
        print(f"{lab:>7}{len(pr):>9,}{nbt:>7,}{hit:>6}{100*ret/cost:>8.1f}%"
              f"{'[' + format(100*lo, '.0f') + ',' + format(100*hi, '.0f') + ']':>20}"
              f"{inter[0]/max(inter[1],1):>7.1%}{inter[0]/max(inter[2],1):>8.1%}"
              f"{np.mean(ds):>+10.4f}{100*bret/max(bcost,1):>8.1f}%"
              f"{100*plr/max(plc,1):>8.1f}% {mark}")

    print("\n" + "=" * 100)
    print("★読み方（事前登録のとおり。**後から足していない**）")
    print("  ・**主判定は(A)の「直前」（発走11分前）で 99%CI下端 > 100%**、年別で91%割れが2つ以下。")
    print("  ⚠1年分だとCI幅±38pt級＝**ROIだけでは判定できない見込み**。**先に書いてある**。")
    print("  ★**その場合は 再現率／適合率／D で読む**:")
    print("    ・**直前でも再現率が高く D が正** → 食い違いは早くから在る＝**標本不足**と書く。")
    print("    ・**直前で D が負** → **食い違いは締切間際に生成される**＝**そこで閉じる**。")
    print("  ・(A)と(B)の差 = **他人の後乗りで配当が削られる分**。")


if __name__ == "__main__":
    main()
