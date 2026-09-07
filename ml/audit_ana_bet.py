"""(211) ★★★★**買い目を最終決定する** —— 11年を「選ぶため」に使い切り、実運用で1回だけ試す

★★**動機（2026-09-07・利用者の指定）**: **「この穴馬軸で買い目を考えよう」**。

★★★**なぜ今日なら許されるか（★設計上の核心）**
　⚠**(209)が既に「穴側 × 券種8 × L6＝48マス」を測り、主判定は落ちた**。
　　→ **同じ11年でもう一度買い目を探すのは3回目の読み**であり、**それ自体は検定にならない**。
　★**だが前向きの標本はまだ1本も無い**（**監視の開始は本日 2026-09-07**）。
　★★**だから「11年で選び、これからの実運用で1回だけ試す」= (201)と同じ設計**にできる。
　　**後半が本物の未来**なので、**選択の下駄は前向き標本に一切乗らない**。
　⚠⚠**明日以降に買い目を変えると、貯めた標本は使えなくなる**（**昨日書いた停止規則**）。
　　★**変えるなら今日が最後**。**この測定はそのための一度きりの決定である**。

■ ★★★**この測定は検定ではない。「記述」と「決定」である**
　★**11年の数字は、買い目を1つ選ぶためだけに使う**。**有意性の主張には使わない**。
　★**検定は前向きに1回だけ**（**2026-09-07以降・99%CI下端が100%超か**）。

────────────────────────────────────────────────────────────
★★★ 事前登録（2026-09-07・**結果を見る前にコミットする**）
────────────────────────────────────────────────────────────

■ ★経路（判定基準25）: **q = MLモデル / q_pool = 複勝の板**。⚠**q側は弱い**。
■ ★★軸は**(210)で固定済み・動かさない**:
　**pn ≥ 0.15 かつ ズレ ≥ 0.15 かつ 単勝オッズ ≥ 10.0倍 の中で pn 最大の1頭**。
　★**候補が無ければ見送る**。**軸は探索しない**（**探索するのは紐と券種だけ**）。

■ ★**紐の並べ方 2通り**（★**(194)は「順序はほぼ無関係」と出したが、あれは人気側の母集団**
　　**＝判定基準25で持ち込めない。だからこの母集団で測り直す**）
| 記号 | 並べ方 |
|---|---|
| **P** | **モデルの推奨度 pn の降順**（**(198)で三連単はこれでないと壊れると確認済み**） |
| **G** | **ズレ gap の降順**（★**この線の優位はズレに宿っているという仮説**） |

■ ★★★**(212) 追記（2026-09-07・★馬単を足す）**
　★**利用者の指摘: 「馬単がないのってなんで？」**。→ ★**根拠があって外したのではなく、
　　(209)で私が券種リストに書き忘れ、(211)がそれを引き継いだだけ**だった。
　★**LINE(0.750)にも payoff() にも KEYMAP にも馬単はあり、42,181レース全部で引ける**。
　⚠**「測って落とした」ではなく「測っていない」**＝**取りこぼしの修復である**。
　★★**馬単はこの目的にとって最も効きそうな位置にいる**——
　　**三連複と同じ払戻率0.750なのに、必要な馬が1頭少ない**。
　　→ **配当は馬連より高く、的中率は三連複より高いはず**。
　★**追加の設計コストはゼロ**: **(211)は「11年は選ぶために使い切り、検定は前向きに1回」**
　　なので、★**選ぶ材料が増えても前向きの検定は汚れない**。
　★**買い方は「軸を1着に固定して紐へ流す」**（**三連単Pと同じ向き**）。**k=1,2 × 紐2 = 4マス追加**。
　⚠**マスは 15 → 19 になる**。**選ぶ基準は一切変えない**。

■ ★★★**(213) 追記（2026-09-07・★利用者の指定2件）**
　★**① 紐のPとGの中身を出す**——**上位3頭の平均/中央オッズ・平均/中央人気・PとGの重なり**。
　　⚠**これまで「紐の並べ方」としか書いておらず、中身を一度も見せていなかった**。
　★**② 三連単の紐を増やす**——**利用者の読み筋: 「ROIが下がっても、的中率が上がって
　　必要年数が下がる方が良い」**。★**これは必要年数の式 need∝s²/(ROI−100)² と整合する**
　　——**点数を増やすと ROI は払戻率へ寄るが、s（分散）が大きく下がる**。
　　**どちらが勝つかは測らないと分からない**。
　★**買い方: 軸を1着に固定し、2-3着は紐 k 頭の順列 → k(k−1)点**。**k ∈ {3,4,5}**（6/12/20点）。
　★**馬単も k=3 を足す**（**(212)でk=2が全マス中2番目に短い55年だったため、勾配を見る**）。
　⚠**マスは 19 → 27 になる**。**選ぶ基準は一切変えない**。
　⚠★**これは3回目の材料追加であり、11年はもう選択に使い切っている**。
　　★**前向きの検定は1回きりで、そこは汚れない**（**(211)の設計**）。

■ ★**券種 14通り**（**1点=100円。点数ぶん賭ける**）
| 券種 | k | 点数 | 払戻率 | ★**必要な優位比 1/払戻率** |
|---|---|---|---|---|
| **複勝** | — | 1 | 0.800 | **1.250** |
| **ワイド** | 1 / 2 | 1 / 2 | 0.775 | **1.290** |
| **馬連** | 1 / 2 | 1 / 2 | 0.775 | **1.290** |
| **三連複** | 2 / 3 | 1 / 3 | 0.750 | **1.333** |
| **三連単** | 2 | 2 | 0.725 | **1.379** |
　★**複勝は紐に依らないので1マス**。→ **8×2 − 1 = 15マス**。

■ ★★★★**選ぶ基準（★先に決める・これ1つ）**
　★★**「乱との対応差の99%CI下端が 0 を超えるマスのうち、★必要年数が最小のもの」**
　★**理由**: **(210)で決着に644年と出た。★いま効いている制約は「儲け」ではなく「年数」**。
　　**必要年数 = need / 年あたりの該当レース数、need = (z·s/(ROI−100))²**
　　→ ★**ROI・分散・本数の3つを同時に最適化する唯一の量**。
　⚠**ROI最大では選ばない**（**(88)の帯の癖と裾を拾うため**・(200)で実測）。
　⚠**下端が0を超えるマスが1つも無ければ、★複勝1点のまま変えない**。

■ ★★ゲート2（判定基準42）——**この測定は何を返せば「意味なし」か**
　★**買い方に情報が無いなら、対応差は全マスで厳密に0** → ★**下端は全マスで0を割り、
　　基準は「該当なし＝複勝を維持」を返す**。★**これが偽のときの返り値である**。
　★**払戻率の差だけで説明がつくかも見る**: **各マスの優位比 ROI/払戻率 を併記し、
　　1.250（複勝）からどれだけ増幅したかを出す**（**(197)の増幅則 1.062→1.101 と突き合わせる**）。

■ ⚠ゲート1: (88)③④を再現（±3pt）／ ★ゲート板: 復元R 0.800±0.020 ／
　★陽性対照: 三連複BOX上位4 81.9%±1.5pt。
■ ★★★内部対照: **L=10・複勝 が 102.1% ±0.2pt かつ 1,356本 ±5**（**(210)と厳密に一致**）。
　⚠★**訂正（2026-09-07・結果を見る前）**: **初版はこれを1,398本で書いて落ちた**。
　　★**(210)の1,356本は「同一オッズ帯の乱が引けたレースだけ」という条件つき**で、
　　**私はその条件を落として全数と比べていた**＝**別の量を対照にした**（**判定基準37**）。
　　★**(210)と同じ乱（同一帯・複勝のみ・10種）を(211)内で組み直して対照とする**。
　⚠**同じ理由で陽性対照も誤っていた**——**box4は「払戻−400円」の純額なので roi_of では読めない**。
　　**(210)と同じ 100×(Σ+400n)/(400n) に直した**。
　★**どちらもマスの結果を1つも見ない段階で直している**（**判定基準32**）。

■ ★穴度も併記する（**利用者の目的は高配当**）:
　**的中時の平均配当 / 中央配当 / 平均÷中央**（★**1.3倍を超えたら裾依存として警告する**）。

■ 予想（⚠**当てにしない**・判定基準24。★**私は(198)(201)(203)(209)で4回外した**）
　★**複勝が残ると見る**——**増幅の実測は3.7%（1.062→1.101）で、ワイドに必要な
　　3.2%（1.290/1.250）とほぼ同じ**。**三連複6.6%・三連単10.3%には届かない**。
　★**紐はPとGでほとんど差が出ないと見る**。⚠**外れたら仮説の方を書き換える**。

■ ★★★★**結果（実測済み・2026-09-07）**
　★対照: **ゲート1 4本とも立つ / 復元R 0.8188 / 三連複BOX4 81.9%** ／
　★★**内部対照 L=10・複勝 = 102.1%・1,356本 → (210)と厳密に再現**。

■ ★★★★**事前登録した基準の答え: 該当なし → 複勝1点のまま変えない**
　★**15マスすべてで、乱との対応差の99%CI下端が0を割った**（**−0.1 〜 −159.1円**）。
　★**理由は分散**——**組の札は乱の払戻も跳ねるので、対応差の se が桁で大きくなる**。
　⚠**ROIが高いマスはある**（下記）**が、「乱より良い」と言えるマスは1つも無い**。

| 紐 | 券種 | 点 | ROI | 対応差 | ★下端 | 的中率 | 平均配当 | 平÷中 | ★必要年数 |
|---|---|---|---|---|---|---|---|---|---|
| **P** | ★**複勝** | 1 | **101.5%** | +9.6円 | **−0.1** | **23.4%** | 435円 | 1.18 | **999年** |
| P | ワイド | 1 | 101.4% | +4.6円 | −12.4 | 12.9% | 784円 | 1.21 | 2,425年 |
| P | 馬連 | 2 | 108.3% | +17.0円 | −13.3 | 7.6% | 2,852円 | ⚠1.39 | 231年 |
| P | 三連複 | 1 | 108.6% | +21.7円 | −17.5 | 3.8% | 2,888円 | 1.21 | 334年 |
| P | 三連単 | 2 | **136.7%** | +41.0円 | −40.8 | **0.9%** | 29,092円 | 1.12 | 111年 |
| ★**G** | ★**馬連** | 2 | ★**140.0%** | **+48.7円** | **−26.3** | 4.7% | **5,956円** | ⚠**2.30** | ★**37年** |
| G | 三連単 | 2 | **49.0%** | −46.7円 | −159.1 | 0.1% | 67,835円 | 1.00 | 到達せず |
　★**現行（複勝1点・(210)）: 1,356本・102.1%・必要年数 542年**（z=2.576。(210)は2比較でz=2.807の644年）

■ ★★**予想は半分外れた**（判定基準24・★**5回目**）
　★**当たり**: **複勝が残った**。**ワイドは必要な+3.2%に対し実測+3.1%/+1.8%で届かなかった**
　　——**増幅則(197)の予測どおり、ワイドは本当に「際どくて足りない」だった**。
　⚠**外れ**: ★**紐をズレ順(G)にすると馬連2点で優位比が 1.806（複勝比142%）まで跳ねた**。
　　★**(194)の「順序はほぼ無関係」は、やはり人気側の母集団の話だった**（**判定基準25の実例**）。
　⚠**ただしGは裾依存が強い**（**平÷中 2.14〜2.74**）。**Pは1.12〜1.50**。
　★**三連単Pの優位比1.886は必要1.379を大きく超えるが、的中率0.9%＝12本しか当たっていない**。

■ ★★★**基準の出力（⚠これは基準の返り値であって、買うかどうかの判断ではない）**
　★★**買い目を決めるのは利用者である**。**私は材料を出すところまで**。
　★**「高配当を取る」方向のマスは全部、乱と区別できないところで止まっている**。
　⚠**「区別がつかない」であって「乱と同じ」ではない**（**下端が割れるのは誤差が大きいから**）。
　★**組の札にすると1本あたりの情報は増えるが、分散がそれ以上に増える**——
　　**(199)で対数配当の中央値の比が1.000だったことと同じ現象**。
　★**★G・馬連2点（37年）は唯一「年数」が現実的な候補だが、下端が−26.3で基準を通らない**。
　　★**再開条件を書いて残す**（下記）。

■ ★**残した未測定（★再開条件つき）**
　★**G・馬連2点**: ★**再開条件 = 「裾を外した指標（対数配当・中央値）でも優位比が1.29を超えるとき」**。
　　⚠**平÷中 2.30 なので、いまは裾で説明がつく可能性が高い**（**(200)と同じ形**）。

■ ★★★★**結果2（(212)(213) 実測済み・2026-09-07）**
　★**内部対照 L=10・複勝 = 102.1% / 1,356本 → 再現**。★**マスは90**。

**★①(212) 馬単**——★**私の書き忘れが、いちばん短い必要年数の場所だった**
| 紐 | 券種 | 点 | ROI | 下端 | 的中率 | 平均配当 | 平÷中 | ★必要年数 |
|---|---|---|---|---|---|---|---|---|
| P | 馬単A | 2 | **136.5%** | −36.2 | 3.5% | 7,868円 | 1.44 | ★**55年** |
| P | 馬単A | 1 | 126.0% | −43.8 | 2.3% | 5,445円 | **1.14** | 64年 |
　★**P馬単2点は、三連単Pとほぼ同じROI（136.5 vs 136.7%）で的中率4倍・必要年数半分・裾も軽い**。

**★②(213) 点数の梯子**——★**必要年数は一度下がって、また上がる**（**単調ではない**）
| P馬単A | 1点 | ★2点 | 3点 | 4点 | 5点 |
|---|---|---|---|---|---|
| 的中率 | 2.3% | 3.5% | 4.3% | 4.8% | 5.3% |
| ROI | 126.0% | **136.5%** | 116.3% | 106.0% | 103.6% |
| ★必要年数 | 64年 | ★**55年** | 142年 | 725年 | 1,857年 |
　★**的中率は単調に上がるが、ROIの落ち方が速い**。**P三連単Aは 76→111→★68→103→499年**。
　★**増えた買い目が拾う当たりは平均配当を押し上げない**（**P馬単Aの平均配当 5,445→9,794円**
　　**に対し賭け金は5倍**）。⚠**利用者の読み筋は「途中まで正しい」**。

**★★③(213d) 軸の着順**——★★**紐によって最適が逆転する**
| 紐 | 券種 | ★A(軸1着) | B(軸2着) | C(軸3着) | M(マルチ) |
|---|---|---|---|---|---|
| **P** | 馬単2点 | ★**136.5%**/55年 | 90.2%/到達せず | — | 103.5%/1,241年 |
| **P** | 三連単3点 | ★**137.5%**/68年 | 92.0%/到達せず | 83.1%/到達せず | 95.3%/到達せず |
| **G** | 馬単2点 | 134.3%/127年 | ★**144.1%**/48年 | — | 114.8%/374年 |
| **G** | 馬単4点 | 98.5%/到達せず | 94.0%/到達せず | — | ★**139.2%/39年** |
　★★**Pなら軸を1着、Gなら軸を2着**。⚠**(209)から1着固定で通したのはPには合っていたがGには合っていなかった**。
　★**筋は通る**: **軸は21.2倍・6.6番人気の穴**。**紐が人気(P・2.3番人気)なら軸1着が配当の源**。
　　**紐も穴(G・6.6番人気)なら軸は2着で十分**——★**G三連単A1点は的中率0.0%**（**1着を穴に要求すると当たらない**）。

**★★④紐の中身（上位3頭）**——★**PとGはまったく別物だった**
| 紐 | 平均オッズ | 中央オッズ | 平均人気 | 中央人気 |
|---|---|---|---|---|
| **P** | **5.3倍** | 4.5倍 | ★**2.3番** | 2番 |
| **G** | **46.1倍** | 16.2倍 | ★**6.6番** | 6番 |
　★**重なりは平均1.20頭/3頭・完全一致2.8%・全く別17.8%**。**軸は共通（21.2倍・6.6番人気）**。
　★**Gの平均人気6.6番は軸と同じ**＝**Gは軸と同じ性格の馬を紐にしている**。
　　→ **Gの配当が高く（5,956〜241,202円）裾依存も強い（平÷中 2.1〜5.7）理由**。

■ ⚠⚠★**探索の規模についての警告**
　★**マスは15→19→27→90に増えた**。⚠**この中の最良値を選ぶのは、
　　(209)で落ちた48マス探索の2倍の規模の探索である**。
　★**下端が0を超えたマスは90中1つも無い**。**最も0に近いのは P馬単B・5点の −7.7**（ROI 107.7%）。
　★★**基準の返り値は「該当なし」のまま**。⚠**買うかどうかは利用者が決める**。

実行: python3 ml/audit_ana_bet.py    自己テスト: python3 ml/audit_ana_bet.py --selftest
"""
import math
import sys
from zlib import crc32

import numpy as np

sys.path.insert(0, "ml")
import features as F
from audit_crosspool import LINE, load_races, payoff, zq
from audit_ana_odds import BANDS, MIN_HORSES, band_of, gate1, roi_of
from audit_ana_marg import WF_BOX4, WF_TOL, wf_predict
from audit_ana_board import BOARD_R, BOARD_TOL, NPLACE, load_fuku_boards, qpool
from audit_ana_band import PN_FLOOR
from audit_ana_ladder import FINE
from audit_ana_hole import BETS as BETS209, GAP, NRAND, SEED
from audit_ana_hole import tickets as tickets209
from audit_ana_fix import LFIX
from train_prod import add_odds_features

# ★★(212) 馬単を足す。⚠**(209)で私が券種リストに書き忘れ、(211)がそれを引き継いだ**。
# ★**根拠があって外したのではない**。**LINEにもpayoff()にもKEYMAPにも馬単はあり、42,181R全部で引ける**。
# ★★(213c) 点数の梯子。⚠**k頭指定だと 2→6→12→20点と飛ぶ**ので、**点数を直接刻む**。
#   **三連単流 = 軸1着固定・紐の順位で並べた買い目を上から n 点**
# ⚠★**(213d) 軸の着順を1着固定から解放する（利用者の指摘）**
#   ★**軸は pn（＝3着以内の確率）で選んでいるのに、買い目は1着を要求していた**＝**選び方と買い方がずれている**。
#   **A=軸1着固定 / B=軸2着固定 / C=軸3着固定 / D=着順不問（軸を1〜3着すべてに置く）**
TAN3_N = [1, 2, 3, 4, 5, 6, 8, 10, 12]
UT_N = [1, 2, 3, 4, 5]   # ★乱が紐を5頭までしか引かないので5点まで
BETS = BETS209 \
    + [(f"馬単{ap}", n) for ap in ("A", "B", "M") for n in UT_N
       if not (ap == "M" and n % 2)] \
    + [(f"三連単{ap}", n) for ap in ("A", "B", "C", "M") for n in TAN3_N
       if not (ap == "M" and n % 3)]


def tan3_pairs():
    """★三連単の買い目の順序。**紐の順位の和が小さい順**（**(1,2)(2,1)(1,3)(3,1)…**）"""
    return sorted([(i, j) for i in range(5) for j in range(5) if i != j],
                  key=lambda t: (t[0] + t[1], t[0], t[1]))


def rate(kind):
    return LINE["三連単" if kind.startswith("三連単")
                else ("馬単" if kind.startswith("馬単") else kind)]
HIMO = ["P", "G"]
KNOWN_ROI, KNOWN_N, ROI_TOL, N_TOL = 102.1, 1356, 0.2, 5
TAIL_WARN = 1.3
ALPHA = 0.01


def need_himo(kind, k):
    """★その買い目に必要な紐の頭数"""
    if kind.startswith("三連単") and len(kind) > 3:
        n = k // 3 if kind[-1] == "M" else k
        return max(max(t) for t in tan3_pairs()[:max(n, 1)]) + 1
    if kind.startswith("馬単") and len(kind) > 2:
        return k // 2 if kind[-1] == "M" else k
    if kind == "複勝":
        return 0
    return 2 if kind == "三連単" else k


def tickets(kind, k, ax, himo):
    """★買い目。**馬単・三連単流はどちらも「軸を1着に固定して紐へ流す」**"""
    if kind.startswith("馬単") and len(kind) > 2:
        ap = kind[-1]
        m = k // 2 if ap == "M" else k
        hs = himo[:m]
        if len(hs) < m or m < 1:
            return None
        if ap == "A":
            return [("馬単", [ax, h]) for h in hs]
        if ap == "B":
            return [("馬単", [h, ax]) for h in hs]
        return [t for h in hs for t in (("馬単", [ax, h]), ("馬単", [h, ax]))]
    if kind.startswith("三連単") and len(kind) > 3:
        ap = kind[-1]
        n = k // 3 if ap == "M" else k          # ★Mは1組が3通りになるので点数を揃える
        if n < 1:
            return None
        pr = tan3_pairs()[:n]
        if len(himo) < max(max(t) for t in pr) + 1:
            return None
        out = []
        for i, j in pr:
            a, b = himo[i], himo[j]
            if ap == "A":
                out.append(("三連単", [ax, a, b]))
            elif ap == "B":
                out.append(("三連単", [a, ax, b]))
            elif ap == "C":
                out.append(("三連単", [a, b, ax]))
            else:
                out += [("三連単", [ax, a, b]), ("三連単", [a, ax, b]),
                        ("三連単", [a, b, ax])]
        return out or None
    hs = himo[:k]
    if len(hs) < k:
        return None
    if kind == "馬単":
        return [("馬単", [ax, h]) for h in hs]
    return tickets209(kind, k, ax, himo)


def cells():
    """★15マス。★複勝は紐に依らないので P のみ"""
    return [(h, kind, k) for h in HIMO for kind, k in BETS
            if not (kind == "複勝" and h != "P")]


def need_years(v, per_year, z):
    """★必要年数。★ROIが100%以下なら到達しない（None）"""
    roi = roi_of(v)
    if roi <= 100.0 or per_year <= 0:
        return None, roi
    nd = (z * float(np.std(v, ddof=1)) / (roi - 100.0)) ** 2
    return nd / per_year, roi


def selftest():
    ok = True
    z = zq(ALPHA)
    cs = cells()
    print(f"★マス数 **{len(cs)}**（券種{len(BETS)} × 紐{len(HIMO)} − 複勝の重複1）"
          f"　{'★OK' if len(cs) == 2 * len(BETS) - 1 else '⚠NG'}")
    ok &= len(cs) == 2 * len(BETS) - 1
    hm = [2, 3, 4, 5, 6]
    print("★★(213d) ★**軸の着順を1着固定から解放する**（利用者の指定）")
    print("　★**A=軸1着 / B=軸2着 / C=軸3着 / M=マルチ**　⚠**馬連は着順が無いので対象外**")
    print("　★★**軸は pn（3着以内の確率）で選んでいるのに、"
          "これまで1着だけを買っていた**＝**選び方と買い方がずれていた**")
    print(f"\n　■ **馬単**（{UT_N}点）")
    for n in UT_N:
        row = []
        for ap in ("A", "B", "M"):
            if ap == "M" and n % 2:
                row.append("M:—")
                continue
            t = tickets(f"馬単{ap}", n, 1, hm)
            row.append(f"{ap}:{len(t)}点")
            ok &= t is not None and len(t) == n
        print(f"　{n:>3}　" + " / ".join(row))
    print(f"　★M・2点 = {[x[1] for x in tickets('馬単M', 2, 1, hm)]}（**同じ1頭の裏表**）")
    ok &= tickets("馬単M", 2, 1, hm) == [("馬単", [1, 2]), ("馬単", [2, 1])]
    print(f"\n　■ **三連単**（{TAN3_N}点）")
    for n in TAN3_N:
        row = []
        for ap in ("A", "B", "C", "M"):
            if ap == "M" and n % 3:
                row.append("M:—")
                continue
            t = tickets(f"三連単{ap}", n, 1, hm)
            row.append(f"{ap}:{len(t)}点")
            ok &= t is not None and len(t) == n
        print(f"　{n:>3}　" + " / ".join(row) + f"　必要な紐 {need_himo('三連単A', n)}頭")
    okA = tickets("三連単A", 2, 1, hm) == [("三連単", [1, 2, 3]), ("三連単", [1, 3, 2])]
    print(f"　★**A・2点は(211)の三連単k=2と厳密に同じ**　{'★OK' if okA else '⚠NG'}")
    ok &= okA
    tM = tickets("三連単M", 3, 1, hm)
    print(f"　★M・3点 = {[x[1] for x in tM]}（**同じ3頭の軸1・2・3着**）")
    ok &= tM == [("三連単", [1, 2, 3]), ("三連単", [2, 1, 3]), ("三連単", [2, 3, 1])]
    t = tickets("馬単", 2, 1, [2, 3, 4])
    print(f"★★(212) 馬単を追加: {t}　{'★OK（軸を1着に固定して流す）' if t == [('馬単', [1, 2]), ('馬単', [1, 3])] else '⚠NG'}")
    ok &= t == [("馬単", [1, 2]), ("馬単", [1, 3])]
    print(f"★軸は(210)で固定: pn≥{PN_FLOOR} かつ ズレ≥{GAP} かつ 単勝≥{LFIX}倍 の中で pn最大")
    print(f"\n{'券種':<8}{'k':>3}{'点数':>5}{'払戻率':>8}{'★必要な優位比':>14}{'★複勝比':>10}")
    for kind, k in BETS:
        t = tickets(kind, k, 1, [2, 3, 4, 5, 6, 7])
        need = 1.0 / rate(kind)
        print(f"{kind:<10}{k:>3}{len(t) if t else 0:>5}{100*rate(kind):>7.1f}%"
              f"{need:>14.3f}{need/(1.0/LINE['複勝']):>9.1%}")
        ok &= t is not None
    print("　★**(197)の実測増幅は 1.062→1.101 ＝ +3.7%**"
          "　→ ★**ワイド/馬連(+3.2%)は際どい・三連複(+6.6%)と三連単(+10.3%)は届かない見込み**")

    print(f"\n★★選ぶ基準の検算（**必要年数 = (z·s/(ROI−100))² / 年あたり本数**）")
    rng = np.random.default_rng(0)
    flat = need_years(np.full(50, 100.0), 1, z)[0] is None
    print(f"　⚠**ROIちょうど100%なら到達しない**: {'★OK' if flat else '⚠NG'}")
    ok &= flat
    a = np.concatenate([np.full(500, 0.0), np.full(500, 204.2)])
    ny, roi = need_years(a, 100.0, z)
    s = float(np.std(a, ddof=1))
    exp = ((z * s / (roi - 100.0)) ** 2) / 100.0
    print(f"　★手計算と一致: ROI {roi:.1f}% / s {s:.1f} / 必要年数 {ny:,.1f} vs {exp:,.1f}"
          f"　{'★OK' if abs(ny-exp) < 1e-6 else '⚠NG'}")
    ok &= abs(ny - exp) < 1e-6

    print(f"\n★ゲート2の検算（**買い方に情報が無ければ対応差は0**）")
    d0 = rng.normal(0, 100.0, 4000)
    lo = d0.mean() - z * d0.std(ddof=1) / math.sqrt(len(d0))
    print(f"　差が0の腕の下端 {lo:+.2f}円　{'★OK（0を割る）' if lo < 0 else '⚠NG'}")
    ok &= lo < 0
    print(f"\n★★★内部対照（決定的）: **L=10・複勝 が {KNOWN_ROI}% ±{ROI_TOL}pt "
          f"かつ {KNOWN_N:,}本 ±{N_TOL}**（**(210)と一致**）")
    print(f"★裾の警告閾値: **平均÷中央 > {TAIL_WARN}倍**")
    print("★自己テスト: " + ("全部OK" if ok else "⚠NG"))
    return 0 if ok else 1


def main():
    z = zq(ALPHA)
    cs = cells()
    print("(211) ★★★★**買い目を最終決定する** —— 11年で選び、実運用で1回だけ試す")
    print("★軸は(210)で固定。⚠**これは検定ではない。買い目を1つ選ぶための記述である**\n")

    races = {r["rid"]: r for r in load_races()}
    rows, bad = gate1(list(races.values()))
    print("⚠**ゲート1**: (88)③④を別パーサで再現・許容±3pt")
    for nm, n, roi, known, dd, okg in rows:
        print(f"　{nm:<12}{roi:>7.1f}% vs {known:>5.1f}%　差 {dd:+.1f}pt"
              f"　{'★立った' if okg else '⚠落ちた'}")
    if bad:
        print("\n⚠⚠**ゲート1が落ちた。読まない**。")
        return

    print("\n★複勝の板(type=2)を読む…")
    boards = load_fuku_boards()
    print(f"　**{len(boards):,}レース分**")

    d = F.to_model(F.load_files())
    f = F.build_features(d)
    keep = (f["n_prior"] >= 1) & d["odds"].notna() & (d["odds"] > 0)
    d, f = d[keep].reset_index(drop=True), f[keep].reset_index(drop=True)
    y = (d["finish"] <= 3).astype(int).to_numpy()
    fx, _ = F.encode_categoricals(f)
    fx = add_odds_features(fx, d["odds"].to_numpy(float), d["raceid"].to_numpy())
    pred = wf_predict(d, fx, y, 3)
    msk = ~np.isnan(pred)
    sub = d.loc[msk, ["raceid", "umaban", "odds", "date"]].copy()
    sub["p"] = pred[msk]

    K = {c: {"a": [], "r": [], "hit": []} for c in cs}
    HD = {"P": {"od": [], "rk": []}, "G": {"od": [], "rk": []}, "ov": []}
    fix, fix210, Rs, box4, yrs = [], [], [], [], set()
    axod, axrk = [], []
    for rid, g in sub.groupby("raceid"):
        rid = str(rid)
        r = races.get(rid)
        bd = boards.get(rid)
        if r is None or bd is None:
            continue
        nums = {u for u, _, _ in r["horses"]}
        if len(nums) < MIN_HORSES:
            continue
        gg = g[g["umaban"].astype(int).isin(nums)]
        ub = gg["umaban"].astype(int).to_numpy()
        if len(gg) < MIN_HORSES or not all(int(u) in bd for u in ub):
            continue
        od = gg["odds"].to_numpy(float)
        pv = gg["p"].to_numpy(float)
        if not np.isfinite(od).all() or (od <= 0).any() or pv.sum() <= 0:
            continue
        pn = pv / pv.sum() * NPLACE
        qp, Rb = qpool([bd[int(u)] for u in ub], "harm")
        Rs.append(Rb)
        gap = pn - qp
        order_p = [int(u) for u in ub[np.argsort(-pv, kind="mergesort")]]
        from itertools import combinations as _cb
        bx = [payoff(r, "三連複", list(c)) for c in _cb(sorted(order_p[:4]), 3)]
        if not any(x is None for x in bx):
            box4.append(sum(bx) - 400.0)
        order_g = [int(u) for u in ub[np.argsort(-gap, kind="mergesort")]]
        bi = np.array([band_of(float(o), BANDS) for o in od])
        pos = {int(u): q for q, u in enumerate(ub)}
        rank = np.argsort(np.argsort(od)) + 1
        yrs.add(int(gg["date"].iloc[0].year))

        cand = np.where((pn >= PN_FLOOR) & (gap >= GAP) & (od >= LFIX))[0]
        if not len(cand):
            continue
        i = int(cand[int(np.argmax(pn[cand]))])
        ax = int(ub[i])
        v0 = payoff(r, "複勝", [ax])
        if v0 is None:
            continue
        fix.append(v0)
        axod.append(float(od[i])); axrk.append(int(rank[i]))
        # ★★★(210)と厳密に同じ内部対照: 乱は「同一オッズ帯」から10種・複勝のみ
        ok210 = True
        for sd in range(NRAND):
            g3 = np.random.default_rng([SEED + sd, crc32(rid.encode())])
            pl = [int(ub[q]) for q in range(len(ub)) if bi[q] == bi[i] and q != i]
            if not pl or payoff(r, "複勝", [int(g3.choice(pl))]) is None:
                ok210 = False
                break
        if ok210:
            fix210.append(v0)
        HM = {"P": [u for u in order_p if u != ax],
              "G": [u for u in order_g if u != ax]}
        # ★(213) 紐の中身を記述する（**上位3頭**で見る）
        for hkey in ("P", "G"):
            for u in HM[hkey][:3]:
                HD[hkey]["od"].append(float(od[pos[u]]))
                HD[hkey]["rk"].append(int(rank[pos[u]]))
        HD["ov"].append(len(set(HM["P"][:3]) & set(HM["G"][:3])))
        # ★乱: 軸と紐を、それぞれオッズ±20%以内の無作為な馬に置き換える（10種平均）
        draws, okd = [], True
        for sd in range(NRAND):
            g2 = np.random.default_rng([SEED + sd, crc32(rid.encode())])
            out = []
            for u in [ax] + HM["P"][:3]:
                k0 = pos[u]
                pl = [int(ub[q]) for q in range(len(ub))
                      if int(ub[q]) not in out
                      and od[k0] / FINE <= od[q] <= od[k0] * FINE]
                if not pl:
                    okd = False
                    break
                out.append(int(g2.choice(pl)))
            if not okd:
                break
            # ★(213b) 紐を5頭まで伸ばす。★最初の4頭は同じrng系列なので既存の値は動かない
            for u in HM["P"][3:5]:
                k0 = pos[u]
                pl = [int(ub[q]) for q in range(len(ub))
                      if int(ub[q]) not in out
                      and od[k0] / FINE <= od[q] <= od[k0] * FINE]
                if not pl:
                    break
                out.append(int(g2.choice(pl)))
            draws.append(out)
        if not okd:
            continue
        # ★点数の多いマスは「乱が同じ数だけ引けた」ときだけ測る（★既存マスは不変）
        nmin = min(len(t) - 1 for t in draws)
        for h, kind, k in cs:
            if need_himo(kind, k) > nmin:
                continue
            tk = tickets(kind, k, ax, HM[h])
            if tk is None:
                continue
            va = [payoff(r, nm2, sel) for nm2, sel in tk]
            if any(x is None for x in va):
                continue
            cost = 100.0 * len(tk)
            acc, okr = [], True
            for t in draws:
                tr = tickets(kind, k, t[0], t[1:])
                if tr is None:
                    okr = False
                    break
                vr = [payoff(r, nm2, sel) for nm2, sel in tr]
                if any(x is None for x in vr):
                    okr = False
                    break
                acc.append(sum(vr) / cost)
            if not okr:
                continue
            c = K[(h, kind, k)]
            c["a"].append(sum(va) / cost)
            c["r"].append(float(np.mean(acc)))
            c["hit"].append(sum(va))

    ny = max(len(yrs), 1)
    print(f"\n★対象 **{len(fix):,}レース / {ny}年**"
          f"　軸: 平均 {np.mean(axod):.1f}倍 / 中央 {np.median(axod):.1f}倍 / "
          f"平均 {np.mean(axrk):.1f}番人気")

    Rm = float(np.mean(Rs))
    okb = abs(Rm - BOARD_R) <= BOARD_TOL
    bx4 = np.asarray(box4, float)
    b4 = (100.0 * (bx4.sum() + 400.0 * len(bx4)) / (400.0 * len(bx4))
          if len(bx4) else float("nan"))
    okx = abs(b4 - WF_BOX4) <= WF_TOL
    print(f"★ゲート板: 復元R {Rm:.4f}（{BOARD_R}±{BOARD_TOL}）{'★OK' if okb else '⚠NG'}"
          f"　／ 陽性対照 三連複BOX4 {b4:.1f}%（{WF_BOX4}±{WF_TOL}）{'★OK' if okx else '⚠NG'}")
    fa = np.array(fix)
    f2 = np.array(fix210)
    fr, fn = roi_of(f2), len(f2)
    print(f"　（参考: 乱の抽選を課さない全数は {roi_of(fa):.1f}% / {len(fa):,}本）")
    okc = abs(fr - KNOWN_ROI) <= ROI_TOL and abs(fn - KNOWN_N) <= N_TOL
    print(f"★★★内部対照: L=10・複勝 **{fr:.1f}%**（{KNOWN_ROI}±{ROI_TOL}）"
          f"・**{fn:,}本**（{KNOWN_N:,}±{N_TOL}）　{'★★再現' if okc else '⚠⚠落ちた'}")
    if not (okb and okx and okc):
        print("\n⚠⚠**対照が落ちた。読まない**（判定基準32）。")
        return

    print(f"\n{'='*112}")
    print("■ ★★**全マスの記述**（⚠**検定ではない。選ぶための材料**）")
    print(f"{'紐':<3}{'券種':<7}{'点':>3}{'本数':>7}{'ROI':>8}{'優位比':>8}"
          f"{'★対応差':>10}{'99%CI下端':>11}{'的中率':>8}{'平均配当':>10}{'平÷中':>7}{'★必要年数':>12}")
    best, rowsout = None, []
    for c in cs:
        h, kind, k = c
        a = np.asarray(K[c]["a"], float) * 100.0   # ★倍率→100円あたりの円
        rv = np.asarray(K[c]["r"], float) * 100.0
        if len(a) < 100:
            continue
        pts = len(tickets(kind, k, 1, [2, 3, 4, 5, 6, 7]))
        roi = roi_of(a)
        dd = a - rv
        mu = dd.mean()
        lo = mu - z * dd.std(ddof=1) / math.sqrt(len(dd))
        hv = np.array([x for x in K[c]["hit"] if x > 0])
        hm = hv.mean() if len(hv) else 0.0
        hmd = np.median(hv) if len(hv) else 0.0
        tail = hm / hmd if hmd > 0 else float("nan")
        per_year = len(a) / ny
        yy, _ = need_years(a, per_year, z)
        ratio = roi / 100.0 / rate(kind)
        rowsout.append((c, len(a), roi, ratio, mu, lo, 100*np.mean(a > 0), hm, tail, yy, pts))
        print(f"{h:<3}{kind:<7}{pts:>3}{len(a):>7}{roi:>7.1f}%{ratio:>8.3f}"
              f"{mu:>+9.1f}円{lo:>+10.1f}{100*np.mean(a>0):>7.1f}%{hm:>9,.0f}円"
              f"{tail:>7.2f}{('%.0f年' % yy) if yy else '★到達せず':>12}")
        if lo > 0 and yy is not None and (best is None or yy < best[9]):
            best = rowsout[-1]

    print(f"\n{'='*112}")
    print("■ ★★★★**事前登録した基準の答え**")
    print("　★基準: **対応差の99%CI下端が0を超えるマスのうち、必要年数が最小**")
    if best is None:
        print("　⚠⚠**該当なし** → ★★**複勝1点のまま変えない**（**事前登録どおり**）")
    else:
        (h, kind, k), n2, roi, ratio, mu, lo, hit, hm, tail, yy, pts = best
        print(f"　★★**選ばれた買い目: 紐{h} / {kind} / k={k} / {pts}点**")
        print(f"　　**{n2:,}本・ROI {roi:.1f}%・対応差 {mu:+.1f}円（下端 {lo:+.1f}）"
              f"・的中率 {hit:.1f}%・平均配当 {hm:,.0f}円・★必要年数 {yy:,.0f}年**")
        if tail > TAIL_WARN:
            print(f"　⚠**裾依存の警告: 平均÷中央 {tail:.2f}倍 > {TAIL_WARN}**"
                  f"（**平均は少数の高配当に引かれている**）")
    fy, _ = need_years(f2, fn / ny, z)
    print(f"　★**比較: 現行（複勝1点）は {fn:,}本・{fr:.1f}%・"
          f"必要年数 {(format(fy, ',.0f') + '年') if fy else '到達せず'}**")

    print(f"\n■ ★★**紐の中身（上位3頭・PとGで何が違うか）**")
    print(f"{'紐':<4}{'平均オッズ':>12}{'中央オッズ':>12}{'平均人気':>10}{'中央人気':>10}")
    for hkey in ("P", "G"):
        o = np.array(HD[hkey]["od"]); rk = np.array(HD[hkey]["rk"])
        print(f"{hkey:<4}{o.mean():>11.1f}倍{np.median(o):>11.1f}倍"
              f"{rk.mean():>9.1f}番{np.median(rk):>9.0f}番")
    print(f"　★**PとGの上位3頭の重なり: 平均 {np.mean(HD['ov']):.2f}頭 / 3頭**"
          f"（**完全一致 {100*np.mean(np.array(HD['ov'])==3):.1f}%** / "
          f"**全く別 {100*np.mean(np.array(HD['ov'])==0):.1f}%**）")
    print(f"　★軸そのものは共通（**平均 {np.mean(axod):.1f}倍 / {np.mean(axrk):.1f}番人気**）")

    print(f"\n■ ★**増幅則との突き合わせ**（**(197)の実測は 1.062→1.101 ＝ +3.7%**）")
    base = next((r for r in rowsout if r[0] == ("P", "複勝", 0)), None)
    if base:
        print(f"{'紐':<3}{'券種':<7}{'k':>3}{'優位比':>8}{'★複勝比':>10}{'必要な比':>10}{'★判定':>10}")
        for r0 in rowsout:
            (h, kind, k), *_ = r0
            ratio = r0[3]
            g = ratio / base[3]
            nd = (1.0 / rate(kind)) / (1.0 / LINE["複勝"])
            print(f"{h:<3}{kind:<7}{k:>3}{ratio:>8.3f}{g:>9.1%}{nd:>9.1%}"
                  f"{'★上回る' if g >= nd else '⚠届かない':>10}")

    print(f"\n{'='*112}")
    print("■ ★★★**ここで決めたことは、明日以降は変えない**")
    print("　★**前向きの標本は本日から貯め始める**。**読むのは年1回**。")
    print("　★★**主判定はこれ1つ**: "
          "**「2026-09-07以降に集めたぶんだけで、ROIの99%CI下端が100%を超えるか」**")
    print("　⚠**途中で読んで「今年は良かった」と言わない**（判定基準43）。")
    print("　⚠**規則を変えたら、集めた標本は使えなくなる**。")
    print("\n⚠**枠連の運用には触れない**。**設定変更は提案しない**。")


if __name__ == "__main__":
    sys.exit(selftest() if "--selftest" in sys.argv else (main() or 0))
