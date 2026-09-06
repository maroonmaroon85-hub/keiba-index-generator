"""(206) ★**水準のCIを測る** —— 「期待値がマイナス」と言い切れるのはどの腕か（★記述のみ）

⚠⚠**私の誤り**: **利用者に「期待値がマイナスの買い目です」と書いた**。
　★**それは点推定をそのまま期待値と断定したもので、言い切れない**。
　★**このセッションで私が5回訂正してきたのと同じ形の誤り**（**幅を見ずに数字を断定した**）。
　⚠**測ってきたのは「対照との対応差」のCIばかりで、★水準のCIは一度も計算していない**。

★★**この測定は判定ではない**（**新しい仮説を作らない**）。
　**既に報告した数字の不確かさを、正しく出し直すだけ**。**だから多重比較の補正は要らない**。

■ ★出すもの（**各腕について**）
　**n / ROI / 標準誤差 / 99%CI（正規） / ブートストラップ99%区間（10,000回） /
　　★ROIが100%を超える割合 / ★100%を除外できるか**
■ ★**読み方**:
| ★**99%CI下端 > 100%** | ★**99%CI上端 < 100%** | ⚠**100%をまたぐ** |
|---|---|---|
| **プラスと言い切れる** | ★**マイナスと言い切れる** | ⚠**どちらとも言えない** |

実行: python3 ml/audit_ana_level.py
"""
import math
import sys
from itertools import combinations

import numpy as np

sys.path.insert(0, "ml")
import features as F
from audit_crosspool import load_races, payoff, zq
from audit_ana_odds import MIN_HORSES, gate1, roi_of
from audit_ana_marg import WF_BOX4, WF_TOL, wf_predict
from audit_ana_board import BOARD_R, BOARD_TOL, NPLACE, load_fuku_boards, qpool
from audit_ana_band import PN_FLOOR
from train_prod import add_odds_features

NBOOT = 10000
SEED = 20260906
ALPHA = 0.01


def main():
    z = zq(ALPHA / 2)          # ★両側99%（判定ではないので補正しない）
    print("(206) ★**水準のCIを測る** — 「期待値がマイナス」と言い切れるのはどの腕か")
    print("⚠**私は点推定をそのまま期待値と断定した。それは言い切れない**\n")

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

    ARMS = ["複勝(ズレ≥0.15)", "複勝(ズレ≥0.02)", "馬連(紐1)", "三連複", "三連単(紐2)"]
    NPT = {"複勝(ズレ≥0.15)": 1, "複勝(ズレ≥0.02)": 1, "馬連(紐1)": 1,
           "三連複": 1, "三連単(紐2)": 2}
    K = {a: [] for a in ARMS}
    Rs, box4, nall = [], [], 0
    for rid, g in sub.groupby("raceid"):
        r = races.get(str(rid))
        bd = boards.get(str(rid))
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
        nall += 1
        pn = pv / pv.sum() * NPLACE
        qp, Rb = qpool([bd[int(u)] for u in ub], "harm")
        Rs.append(Rb)
        gap = pn - qp
        order_p = [int(u) for u in ub[np.argsort(-pv, kind="mergesort")]]
        bx = [payoff(r, "三連複", list(c))
              for c in combinations(sorted(order_p[:4]), 3)]
        if not any(x is None for x in bx):
            box4.append(sum(bx) - 400.0)
        # ★複勝(ズレ≥0.15): (203)の帯[0.15,∞)・帯の中で推奨度最大
        c15 = np.where((pn >= PN_FLOOR) & (gap >= 0.15))[0]
        if len(c15):
            ai = int(c15[int(np.argmax(pn[c15]))])
            v = payoff(r, "複勝", [int(ub[ai])])
            if v is not None:
                K["複勝(ズレ≥0.15)"].append(v)
        # ★他の腕: (200)と同じ（軸 = pn≥0.15 かつ gap≥0.02 のうち gap最大、紐 = p降順）
        c02 = np.where((pn >= PN_FLOOR) & (gap >= 0.02))[0]
        if not len(c02):
            continue
        ai = int(c02[int(np.argmax(gap[c02]))])
        axu = int(ub[ai])
        himo = [u for u in order_p if u != axu]
        if len(himo) < 2:
            continue
        vs = {"複勝(ズレ≥0.02)": [payoff(r, "複勝", [axu])],
              "馬連(紐1)": [payoff(r, "馬連", [axu, himo[0]])],
              "三連複": [payoff(r, "三連複", sorted([axu, himo[0], himo[1]]))],
              "三連単(紐2)": [payoff(r, "三連単", [axu, himo[0], himo[1]]),
                              payoff(r, "三連単", [axu, himo[1], himo[0]])]}
        if any(v is None for a in vs for v in vs[a]):
            continue
        for a in vs:
            K[a].append(sum(vs[a]))
    print(f"\n★対象 **{nall:,}レース**")

    med = float(np.median(Rs))
    okR = abs(med - BOARD_R) <= BOARD_TOL
    print(f"■ ★ゲート板: 復元R = **{med:.4f}** → **{'★立った' if okR else '⚠⚠落ちた'}**")
    box4 = np.asarray(box4, float)
    g0 = 100.0 * (box4.sum() + 400.0 * len(box4)) / (400.0 * len(box4))
    okb = abs(g0 - WF_BOX4) <= WF_TOL
    print(f"■ ⚠**陽性対照**: BOX上位4 **{g0:.1f}%** vs {WF_BOX4}%"
          f" → **{'★立った' if okb else '⚠⚠落ちた'}**")
    if not (okR and okb):
        print("\n⚠⚠**ゲートが落ちた。読まない**（判定基準32）。")
        return

    rng = np.random.default_rng(SEED)
    print(f"\n{'='*118}")
    print("■ ★★**水準のCI**（**判定ではない・既報の数字の不確かさを出し直すだけ**）")
    print("　★読み方: **下端>100%→プラスと言い切れる / 上端<100%→マイナスと言い切れる / "
          "またぐ→どちらとも言えない**")
    print(f"\n{'腕':<18}{'点数':>5}{'R数':>9}{'ROI':>8}{'標準誤差':>10}"
          f"{'★99%CI(正規)':>22}{'★ブートストラップ99%':>24}{'100%超の割合':>14}{'★結論':>22}")
    for a in ARMS:
        v = np.asarray(K[a], float)
        if len(v) < 100:
            continue
        cost = 100.0 * NPT[a]
        roi = 100.0 * v.mean() / cost
        se = 100.0 * v.std(ddof=1) / math.sqrt(len(v)) / cost
        lo, hi = roi - z * se, roi + z * se
        idx = rng.integers(0, len(v), size=(NBOOT, len(v)))
        b = 100.0 * v[idx].mean(axis=1) / cost
        blo, bhi = np.percentile(b, [0.5, 99.5])
        pg = 100.0 * np.mean(b > 100.0)
        if lo > 100.0:
            tag = "★★プラスと言える"
        elif hi < 100.0:
            tag = "★マイナスと言える"
        else:
            tag = "⚠どちらとも言えない"
        print(f"{a:<18}{NPT[a]:>5}{len(v):>9,}{roi:>7.1f}%{se:>9.2f}pt"
              f"{f'[{lo:.1f}, {hi:.1f}]':>22}{f'[{blo:.1f}, {bhi:.1f}]':>24}"
              f"{pg:>13.1f}%{tag:>22}")

    print("\n⚠**この表は「賭けとしての期待値」を直接測っている**"
          "——**対照との対応差とは別の量**。")
    print("★**対応差（機構が在るか）と水準（勝てるか）は別の問い**"
          "——**このセッションで有意だったのは前者だけ**。")
    print("\n⚠**枠連の運用には触れない**。**設定変更は提案しない**。")


if __name__ == "__main__":
    sys.exit(main() or 0)
