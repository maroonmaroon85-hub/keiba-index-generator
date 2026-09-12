"""(131) ★★★WIN5は資金的に実行可能か — **板を集める前に、手持ちの単勝オッズだけで決める**

★なぜ「情報」ではなく「資金」を測るのか
　`ROI_MAP.md` の結論は「**ROIを上げる道はDを上げるかRを上げるかの2つしかなく、I側は枯れた**」。
　残ったのは **II-28 WIN5のキャリーオーバー**だけである。だが**WIN5は情報で勝つ券種ではない**:
```
　WIN5の払戻率 τ ≈ 0.70（控除率30%）→ 必要量 |log τ| = 0.3567
　　★これは三連単の 0.3216 より**高い**。**全券種で最も不利**。
　　我々の最良のDは +0.0394（(112)枠連の裾2%）＝必要量の 11%。**情報では絶対に届かない**。
　キャリーオーバー C が current pool P に乗ると  R_eff = τ + C/P
　　★R_eff ≥ 1 の条件:  C ≥ (1−τ)·P ＝ **Cが売上の30%以上**
```
　★**つまりWIN5は「キャリーオーバーに賭ける」買い方であって、予想の巧拙は主役ではない**。
　　そしてキャリーオーバーを取りに行くなら **プールの確率質量を広く覆う**必要がある
　　（自分の買い目が当たらなければ、R_eff がいくら1を超えていても取り分は0）。
　→ ★**先に決まるのは「いくら要るか」**。ここが retail の桁を超えるなら、
　　 **C や P を集める前にこの道は閉じる**。**それは手持ちのデータで今日決められる**。

★★事前登録（測る前に宣言する）
　1. **WIN5の対象5レースは公表選定で、過去分は手元に無い**。→ **その日の最終5レース**で代用する
　　 （WIN5は各開催の後半のレースが選ばれるため）。**近似であることを明記する**。
　　 **感度分析としてその日から無作為に5レース選ぶ場合も併記**する。
　2. 覆う確率質量は **10 / 25 / 50 / 80 / 90%** の5点。**後から増やさない**。
　　 各点で「**最小の買い目点数**」と「**その金額（100円/点）**」を出す。
　　 ★最小点数は**確率の高い組から順に取る**（これが最適な覆い方）。
　3. **判定（先に宣言）**: **確率質量の50%を覆うのに 1,000万円を超える**なら、
　　 **この道はretailの資金では実行不能**と判定し、**WIN5の収集は行わない**。
　　 1,000万円以下なら**収集する価値がある**（次にC/Pの実績を集める）。
　　 ⚠この閾値は**期待値ではなく実行可能性**の線。ユーザーの資金で動かせるかで引いている。
　4. **予想**: **50%を覆うのに数百万〜数千万円かかり、判定は「実行不能」側に出る**と予想する。
　　 理由: 5レース×十数頭の積は 10^6 のオーダーで、確率質量は本命側に集中するとはいえ
　　 　　　1番人気の勝率は3割程度しかなく、5つ重ねると本命ど真ん中でも 0.3^5≈0.24% にしかならない。
　　 ⚠**予想はあてにしない**。

⚠**この計算が使っていない前提（Macで要確認）**:
　- **τ=0.70**（WIN5の払戻率）。**未検証**。JRAの公表値を確認すること。
　- **キャリーオーバーの上限**と**発生頻度**、**週あたり売上P**。**すべて未取得**。
　- **自分の購入Bはプールを希釈する**（R_eff = τ + C/(P+B)）。ここでは B≪P として無視している。

実行: python3 ml/audit_win5_capital.py
"""
import csv
import io
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, "ml")

PATH = "data/payout/a.csv"
TAU = 0.70                                   # ⚠未検証（WIN5の払戻率）
COVERS = (0.10, 0.25, 0.50, 0.80, 0.90)      # ★先に宣言した5点
LIMIT_YEN = 10_000_000                       # ★事前登録3の判定線
RNG = np.random.default_rng(20260812)


def load_days(path=PATH):
    """日付 → その日のレース一覧 [(レース番号, 単勝確率)]。標準の a.csv をそのまま読む。"""
    with open(path, "rb") as fh:
        txt = fh.read().decode("shift_jis", "replace")
    days = defaultdict(list)
    for r in csv.reader(io.StringIO(txt)):
        if len(r) < 224:
            continue
        rid = r[14].strip()
        if len(rid) != 8:
            continue
        try:
            y, mo, d = 2000 + int(r[0]), int(r[1]), int(r[2])
            n = int(r[13])
            rno = int(rid[6:8])
        except ValueError:
            continue
        odds = []
        for i in range(18):
            b = 15 + i * 4
            try:
                fin, abn, _pop, od = (int(r[b]), int(r[b + 1]),
                                      int(r[b + 2]), float(r[b + 3]))
            except (ValueError, IndexError):
                continue
            if abn == 0 and od > 0 and fin > 0:
                odds.append(od)
        if len(odds) < 5 or len(odds) != n:
            continue
        inv = np.array([1.0 / o for o in odds])
        days[(y, mo, d)].append((rno, rid[:2], inv / inv.sum()))
    return days


def coverage(ps, covers=COVERS):
    """5レースの確率ベクトル → 各被覆率に必要な最小点数。

    ★確率の高い組から順に取るのが最適な覆い方。積分布を全部作って降順に足す。
    """
    v = ps[0]
    for p in ps[1:]:
        v = np.multiply.outer(v, p).ravel()
    v.sort()
    v = v[::-1]
    cs = np.cumsum(v)
    out = []
    for c in covers:
        k = int(np.searchsorted(cs, c) + 1)
        out.append(min(k, len(v)))
    return out, int(len(v)), float(np.prod([p.max() for p in ps]))


def pick_last5(rs):
    """★代用: その日の最終5レース（レース番号の大きい順）。"""
    return [p for _rno, _jyo, p in sorted(rs, key=lambda x: -x[0])[:5]]


def pick_rand5(rs):
    idx = RNG.choice(len(rs), size=5, replace=False)
    return [rs[i][2] for i in idx]


def summarize(label, rows, ncombo, pfav):
    print(f"\n── {label}（{len(rows)}日分） ──")
    a = np.array(rows, float)
    print(f"  組の総数（Π頭数）  中央値 {np.median(ncombo):,.0f} 通り"
          f"（四分位 {np.percentile(ncombo, 25):,.0f}〜{np.percentile(ncombo, 75):,.0f}）")
    pf = np.array(pfav, float)
    print(f"  5レースとも1番人気が勝つ確率  中央値 {100 * np.median(pf):.3f}%"
          f"  → 平均すると {1 / max(np.mean(pf), 1e-12):,.0f} 回に1回")
    print("\n  覆う確率質量   必要点数(中央値)      金額(中央値)      金額(第3四分位)")
    for j, c in enumerate(COVERS):
        med = np.median(a[:, j])
        q3 = np.percentile(a[:, j], 75)
        print(f"     {int(c * 100):3d}%      {med:>12,.0f} 点   "
              f"{100 * med:>14,.0f} 円   {100 * q3:>14,.0f} 円")


def main():
    days = load_days()
    print(f"開催日 {len(days)} 日を読み込んだ")
    print(f"⚠前提: WIN5の払戻率 τ={TAU}（**未検証**）→ 必要量 |log τ| = "
          f"{-np.log(TAU):.4f}（三連単0.3216より不利）")

    rows_l, nc_l, pf_l = [], [], []
    rows_r, nc_r, pf_r = [], [], []
    for _k, rs in sorted(days.items()):
        if len(rs) < 5:
            continue
        for pick, rows, nc, pf in ((pick_last5, rows_l, nc_l, pf_l),
                                   (pick_rand5, rows_r, nc_r, pf_r)):
            ps = pick(rs)
            if len(ps) != 5:
                continue
            tot = int(np.prod([len(p) for p in ps]))
            if tot > 4_000_000:
                continue
            cov, n, pfav = coverage(ps)
            rows.append(cov)
            nc.append(n)
            pf.append(pfav)

    if not rows_l:
        sys.exit("対象日が無い")
    summarize("★本命の代用: その日の最終5レース", rows_l, nc_l, pf_l)
    summarize("感度分析: その日から無作為に5レース", rows_r, nc_r, pf_r)

    a = np.array(rows_l, float)
    med50 = float(np.median(a[:, COVERS.index(0.50)])) * 100
    print("\n── ★事前登録3: 判定 ──")
    print(f"　確率質量50%を覆う金額の中央値 = **{med50:,.0f}円**"
          f"（判定線 {LIMIT_YEN:,}円）")
    if med50 > LIMIT_YEN:
        print("　→ ★**retailの資金では実行不能**。**WIN5の収集は行わない**。")
    else:
        print("　→ ★**実行可能側**。次に C/P の実績（売上とキャリーオーバー）を集める価値がある。")

    print("\n── ★★事後に導いた決定則（事前登録の5点から算術で出しただけ・新しい測定はしていない） ──")
    print("　プール比例で買えば取り分は「賭け金 × R_eff × 被覆率」になる。")
    print("　⇒ **利益の条件は R_eff × 被覆率 ≥ 1**、すなわち **必要な被覆率 = 1/R_eff**。")
    print("　★**部分被覆は R_eff をそのまま割り引く**。50%しか覆わないなら R_eff は2倍要る。")
    print("\n  被覆率   必要なR_eff   そのとき必要なC/P      金額(中央値)")
    for j, c in enumerate(COVERS):
        need_r = 1.0 / c
        need_cp = need_r - TAU
        med = np.median(a[:, j]) * 100
        print(f"   {int(c * 100):3d}%      {need_r:>5.2f}       C ≥ {need_cp:>5.2f}×P    "
              f"{med:>12,.0f} 円")
    print("\n　⚠**この表は2つを仮定している**（どちらも未検証）:")
    print("　　① **プールの分布が我々のqに比例する**（＝群衆が単勝の積どおりに買う）。")
    print("　　　 実際にずれているなら、そのずれ自体が利得にも損失にもなる。")
    print("　　② 自分の購入Bがプールを希釈しない（B≪P）。**Bが数百万なら成立しない可能性がある**。")
    print("　★**①②を確かめるにはWIN5の票数分布が要る**。払戻だけでは足りない。")


if __name__ == "__main__":
    main()
