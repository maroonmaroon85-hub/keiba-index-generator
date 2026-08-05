"""(89) 券種間（プール間）の整合性 — **モデル不使用・既存データのみ**。

★HANDOFF「次にやること 1-b」は「配当A は払戻＝レース後の値なので使えない／netkeibaで
　これから集める」としていたが、**これは半分だけ正しい**。
　・任意の組み合わせの発走前オッズは確かに取れない（オーバーレイの事前計算は不可）
　・しかし**払戻＝的中時に実際に受け取る額**なので、**買い方のROIは完全に測れる**
　→ 「単勝プール由来の確率で他券種を買ったときのROI」は13年46,917Rで今日測れる。

★この検証の要点（これまで一度もやっていない比較）
　本プロジェクトは「モデルROI vs 人気順ROI」を何度も測ったが、
　**人気順ROI をその券種の控除率線と比べたことが無い**。
　パリミュチュエルでは、そのプールが完璧に較正されていれば
　**どんな買い方をしてもROIは厳密に払戻率(1−控除率)になる**。
　したがって「人気順で買ったときのROI − 払戻率」が、そのプールの誤りの符号と大きさそのもの。
　単勝プールの情報(=人気順)で他プールを買って払戻率を超えるなら、
　**そのプールは単勝プールより下手**＝(1-b)が探していた「市場が市場自身と矛盾している箇所」。

────────────────────────────────────────────────────────────────
★測る前に宣言する（(86)の作法）
────────────────────────────────────────────────────────────────
【券種と払戻率の線】単勝80.0 / 複勝80.0 / 枠連77.5 / 馬連77.5 / ワイド77.5 /
　　　　　　　　　馬単77.5 / 三連複75.0 / 三連単72.5 （JRA公示・2014年6月以降）
【買い方】各券種で「単勝オッズから作った確率」が最大の1点（本命1点）。
　　　　　枠連だけは(i)人気順の上位2頭の枠 と(ii)Harvilleで枠を集計した最上位 の2通り
　　　　　（枠は複数頭の合算なのでオッズの大きさが効く唯一の券種）。
【判定】9券種でBonferroni α=0.05/9。
　(a) ROIの99.44%CI下端 > **払戻率の線** → そのプールは単勝プールより下手（＝入口）
　(b) ROIのCI下端 > 100% → 儲かる
　(c) 前後半で符号一致 / (d) 年ごとに一貫（14年中何年が線超えか）
【★片側にしか情報が無いことの明示】
　Harvilleには既知の偏りがある（下位馬の連対確率を過大評価する）。
　よってROIが線を**下回っても**「プールが正しい」とは言えない（Harvilleのせいかもしれない）。
　**線を上回った場合だけ**が意味のある発見。これを事前に認めておく。
【第2部・どちらのプールが鋭いか（対数スコア）】
　的中した組み合わせについてだけなら、両プールの確率が両方とも観測できる:
　　単勝プール由来 = Harville / 馬連プール由来 = 払戻率 ÷ (払戻/100)
　両方とも組み合わせ空間上の正規化された分布なので、**実現結果への対数スコアで比較できる**
　（真の分布を知らなくても比較は妥当。固有スコアリングルール）。
　控除率の値に依存しないよう、**「両者が引き分けになる払戻率 t*」**を出す。
　t* < 実際の払戻率 なら **そのプールの方が単勝プールより鋭い**。
────────────────────────────────────────────────────────────────

実行: python3 ml/audit_crosspool.py
"""
import csv
import io
import math
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, "ml")
from pocket_eval import _slots            # noqa: E402
from waku_umatan import waku_of           # noqa: E402

PATH = "data/payout/a.csv"
NBET = 9
ALPHA = 0.05 / NBET

# JRA公示の払戻率（1−控除率）
PAYBACK = {"単勝": 0.800, "複勝": 0.800, "枠連": 0.775, "馬連": 0.775, "ワイド": 0.775,
           "馬単": 0.775, "三連複": 0.750, "三連単": 0.725}


def zq(alpha):
    from statistics import NormalDist
    return NormalDist().inv_cdf(1 - alpha / 2)


# ───────────────────────── 読み込み ─────────────────────────
def load_races(path=PATH):
    with open(path, "rb") as fh:
        txt = fh.read().decode("shift_jis", "replace")
    races = []
    for r in csv.reader(io.StringIO(txt)):
        if len(r) < 224:
            continue
        rid = r[14].strip()
        if len(rid) != 8:
            continue
        try:
            year = 2000 + int(r[0])
            n = int(r[13])
        except ValueError:
            continue
        # 馬ごと: col15から4列 (着順, 異常区分, 人気, 単勝オッズ) を馬番順に
        horses = []
        for i in range(18):
            b = 15 + i * 4
            try:
                fin, abn, pop, od = int(r[b]), int(r[b + 1]), int(r[b + 2]), float(r[b + 3])
            except (ValueError, IndexError):
                continue
            if abn == 0 and od > 0 and fin > 0:
                horses.append((i + 1, od, fin))
        if len(horses) < 5 or len(horses) != n:
            continue
        races.append(dict(
            rid=rid, year=year, n=n, horses=horses,
            tan=_slots(r, 87, 3, 2, 1, 1), fuku=_slots(r, 93, 5, 2, 1, 1),
            wakuren=_slots(r, 103, 3, 4, 2, 2), umaren=_slots(r, 115, 3, 4, 2, 2),
            wide=_slots(r, 127, 7, 4, 2, 2), umatan=_slots(r, 155, 6, 4, 2, 2),
            puku=_slots(r, 179, 3, 5, 3, 3), tan3=_slots(r, 194, 6, 5, 3, 3)))
    return races


def probs(horses):
    inv = np.array([1.0 / o for _, o, _ in horses])
    return inv / inv.sum()


# ───────────────────────── Harville ─────────────────────────
def h_pair_unordered(p, i, j):
    return p[i] * p[j] / (1 - p[i]) + p[j] * p[i] / (1 - p[j])


def h_tri_unordered(p, idx):
    s = 0.0
    a, b, c = idx
    for x, y, z in ((a, b, c), (a, c, b), (b, a, c), (b, c, a), (c, a, b), (c, b, a)):
        d1, d2 = 1 - p[x], 1 - p[x] - p[y]
        if d1 > 1e-12 and d2 > 1e-12:
            s += p[x] * p[y] * p[z] / (d1 * d2)
    return s


def h_tri_ordered(p, x, y, z):
    d1, d2 = 1 - p[x], 1 - p[x] - p[y]
    return p[x] * p[y] * p[z] / (d1 * d2) if (d1 > 1e-12 and d2 > 1e-12) else 0.0


def h_pair_ordered(p, x, y):
    return p[x] * p[y] / (1 - p[x])


# ───────────────────────── 買い目 ─────────────────────────
def bets_for(race):
    """単勝オッズだけから各券種の本命1点を決める。返り値: {券種: (組, 点数)}"""
    hs = race["horses"]
    p = probs(hs)
    order = np.argsort(-p)                    # 人気順（オッズの小さい順）
    nums = [hs[k][0] for k in order]
    n = race["n"]
    out = {}
    out["単勝"] = (nums[0],)
    out["複勝"] = (nums[0],)
    out["馬連"] = tuple(sorted(nums[:2]))
    out["ワイド"] = tuple(sorted(nums[:2]))
    out["馬単"] = (nums[0], nums[1])
    out["三連複"] = tuple(sorted(nums[:3]))
    out["三連単"] = (nums[0], nums[1], nums[2])
    # 枠連(i) 人気順の上位2頭の枠
    w1, w2 = waku_of(nums[0], n), waku_of(nums[1], n)
    out["枠連(人気順)"] = tuple(sorted((w1, w2)))
    # 枠連(ii) Harvilleで枠ペアを集計して最大（枠は複数頭の合算＝大きさが効く唯一の券種）
    wk = defaultdict(list)
    for k, (num, _, _) in enumerate(hs):
        wk[waku_of(num, n)].append(k)
    best, bestv = None, -1.0
    frames = sorted(wk)
    for a in range(len(frames)):
        for b in range(a, len(frames)):
            fa, fb = frames[a], frames[b]
            v = 0.0
            if fa == fb:
                mem = wk[fa]
                for x in range(len(mem)):
                    for y in range(x + 1, len(mem)):
                        v += h_pair_unordered(p, mem[x], mem[y])
            else:
                for x in wk[fa]:
                    for y in wk[fb]:
                        v += h_pair_unordered(p, x, y)
            if v > bestv:
                best, bestv = (fa, fb), v
    out["枠連(Harville)"] = best
    return out


KEYMAP = {"単勝": "tan", "複勝": "fuku", "馬連": "umaren", "ワイド": "wide",
          "馬単": "umatan", "三連複": "puku", "三連単": "tan3",
          "枠連(人気順)": "wakuren", "枠連(Harville)": "wakuren"}
LINE = {"単勝": 0.800, "複勝": 0.800, "馬連": 0.775, "ワイド": 0.775, "馬単": 0.775,
        "三連複": 0.750, "三連単": 0.725, "枠連(人気順)": 0.775, "枠連(Harville)": 0.775}


def payoff(race, kind, combo):
    """その買い目の払戻[円/100円]。的中していなければ0。"""
    slots = race[KEYMAP[kind]]
    if not slots:
        return None                       # その券種が発売されていない（枠連は8頭未満など）
    if kind in ("単勝", "複勝"):
        return float(slots.get((combo[0],), 0))
    if kind in ("馬単", "三連単"):
        return float(slots.get(tuple(combo), 0))
    key = tuple(sorted(combo))
    for k, v in slots.items():
        if tuple(sorted(k)) == key:
            return float(v)
    return 0.0


def ci(x, alpha):
    x = np.asarray(x, float)
    m = x.mean()
    se = x.std(ddof=1) / math.sqrt(len(x))
    z = zq(alpha)
    return m, m - z * se, m + z * se


def main():
    races = load_races()
    print(f"配当A: {len(races):,}レース（全馬にオッズ・頭数一致） "
          f"{races[0]['year']}〜{races[-1]['year']}")
    z = zq(ALPHA)

    # ───────── 第1部: 人気順/Harville本命のROI vs 払戻率の線 ─────────
    pay = defaultdict(list)
    yearly = defaultdict(lambda: defaultdict(list))
    for r in races:
        b = bets_for(r)
        for kind, combo in b.items():
            v = payoff(r, kind, combo)
            if v is None:
                continue
            pay[kind].append(v / 100.0)
            yearly[kind][r["year"]].append(v / 100.0)

    print(f"\n{'='*108}")
    print(f"【第1部】単勝プール由来の本命1点を各プールで買ったときのROI vs そのプールの払戻率")
    print(f"　Bonferroni α=0.05/{NBET}={ALPHA:.4f}（z={z:.2f}）")
    print("=" * 108)
    print(f"{'券種':<16}{'R数':>8}{'的中率':>8}{'ROI':>8}"
          f"{'99.44%CI':>18}{'払戻率線':>9}{'線との差':>9}{'(a)':>4}{'(b)':>4}"
          f"{'前半/後半':>16}{'年で線超':>9}")
    print("-" * 108)
    rows = []
    for kind in ["単勝", "複勝", "枠連(人気順)", "枠連(Harville)", "馬連", "ワイド",
                 "馬単", "三連複", "三連単"]:
        x = np.array(pay[kind])
        if len(x) < 100:
            continue
        m, lo, hi = ci(x, ALPHA)
        line = LINE[kind]
        half = len(x) // 2
        m1, m2 = x[:half].mean(), x[half:].mean()
        yrs = sorted(yearly[kind])
        nyr = sum(1 for y in yrs if np.mean(yearly[kind][y]) > line)
        a = "○" if lo > line else "×"
        bb = "○" if lo > 1.0 else "×"
        c = "○" if (m1 > line) == (m2 > line) else "×"
        rows.append((kind, m, lo, line, a, bb, c, nyr, len(yrs)))
        print(f"{kind:<16}{len(x):>8,}{100*(x>0).mean():>7.1f}%{100*m:>7.1f}%"
              f"  [{100*lo:>5.1f},{100*hi:>5.1f}]{100*line:>8.1f}%{100*(m-line):>+8.2f}pt"
              f"{a:>4}{bb:>4}{100*m1:>7.1f}/{100*m2:>6.1f}{c:>3}{nyr:>5}/{len(yrs)}")

    na = sum(1 for r in rows if r[4] == "○")
    nb = sum(1 for r in rows if r[5] == "○")
    print(f"\n  ★(a)払戻率線をCIで超えた券種: **{na}件** "
          f"{[r[0] for r in rows if r[4]=='○']}")
    print(f"  ★(b)100%を超えた券種: **{nb}件** {[r[0] for r in rows if r[5]=='○']}")

    # ───────── 第2部: 対数スコアで「どちらのプールが鋭いか」 ─────────
    print(f"\n{'='*108}")
    print("【第2部】対数スコア — 単勝プール由来(Harville) vs そのプール自身の値付け")
    print("　的中組についてだけなら両方の確率が観測できる。分割になっている券種のみ。")
    print("　t* = 両者が引き分けになる払戻率。**t* < 実際の払戻率 ならそのプールの方が鋭い**。")
    print("=" * 108)
    parts = {"枠連": "wakuren", "馬連": "umaren", "馬単": "umatan",
             "三連複": "puku", "三連単": "tan3"}
    print(f"{'券種':<10}{'R数':>9}{'平均log q(Harville)':>22}{'平均log(払戻/100)':>20}"
          f"{'t*':>9}{'実際':>8}{'鋭いのは':>12}")
    print("-" * 108)
    for kind, key in parts.items():
        lh, lp = [], []
        for r in races:
            slots = r[key]
            if not slots:
                continue
            hs = r["horses"]
            p = probs(hs)
            num2k = {num: k for k, (num, _, _) in enumerate(hs)}
            fin = {num: f for num, _, f in hs}
            first = [num for num in fin if fin[num] == 1]
            second = [num for num in fin if fin[num] == 2]
            third = [num for num in fin if fin[num] == 3]
            if len(first) != 1 or len(second) != 1:
                continue
            a, b = first[0], second[0]
            if kind in ("三連複", "三連単") and len(third) != 1:
                continue
            c = third[0] if third else None
            # 実現組の払戻（1点分）
            if kind == "枠連":
                combo = tuple(sorted((waku_of(a, r["n"]), waku_of(b, r["n"]))))
            elif kind == "馬連":
                combo = tuple(sorted((a, b)))
            elif kind == "馬単":
                combo = (a, b)
            elif kind == "三連複":
                combo = tuple(sorted((a, b, c)))
            else:
                combo = (a, b, c)
            v = payoff(r, kind if kind != "枠連" else "枠連(人気順)", combo)
            if not v or v <= 0:
                continue
            # Harville側
            if kind == "枠連":
                wa, wb = combo
                wk = defaultdict(list)
                for k, (num, _, _) in enumerate(hs):
                    wk[waku_of(num, r["n"])].append(k)
                q = 0.0
                if wa == wb:
                    mem = wk[wa]
                    for x in range(len(mem)):
                        for y in range(x + 1, len(mem)):
                            q += h_pair_unordered(p, mem[x], mem[y])
                else:
                    for x in wk[wa]:
                        for y in wk[wb]:
                            q += h_pair_unordered(p, x, y)
            elif kind == "馬連":
                q = h_pair_unordered(p, num2k[a], num2k[b])
            elif kind == "馬単":
                q = h_pair_ordered(p, num2k[a], num2k[b])
            elif kind == "三連複":
                q = h_tri_unordered(p, (num2k[a], num2k[b], num2k[c]))
            else:
                q = h_tri_ordered(p, num2k[a], num2k[b], num2k[c])
            if q <= 0:
                continue
            lh.append(math.log(q))
            lp.append(math.log(v / 100.0))
        if len(lh) < 500:
            continue
        mh, mp = float(np.mean(lh)), float(np.mean(lp))
        tstar = math.exp(mh + mp)
        actual = PAYBACK[kind]
        who = "**そのプール**" if tstar < actual else "単勝プール(Harville)"
        print(f"{kind:<10}{len(lh):>9,}{mh:>22.4f}{mp:>20.4f}"
              f"{100*tstar:>8.1f}%{100*actual:>7.1f}%{who:>14}")

    print("\n  読み方: t* は「Harvilleと同じ精度になるための払戻率」。")
    print("  実際の払戻率がこれを上回っていれば、そのプールは単勝プールより多くを知っている")
    print("  ＝ 単勝→他券種の裁定余地は無い。逆なら、そのプールは単勝プールに負けている。")


if __name__ == "__main__":
    main()
