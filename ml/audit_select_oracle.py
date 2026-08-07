"""(101) ★★★「レースを選べば100%に届くか」の**オラクル上界** — 選択という手段そのものを閉じる。

★問いの立て方（ここが肝）
　(89)④の上界は**平均のD**で書かれている: 成長率 = log(払戻率) + E[d]。
　だが**部分集合Sだけ買うなら効くのは E[d|S]**。d のばらつきが大きければ、
　うまいレースだけ選んで E[d|S] ≫ E[d] にできる可能性が原理的には残っている。
　→ **この可能性を1回で潰す**。方法は(94)③と同じ「オラクルでも届かないことを見せる」。

★手続き（**わざと反則をする**のが要点）
　1. 各レースの d（＝log q − log q_pool、λ補正込み）を計算する。これは**結果を見ないと出せない量**。
　2. **事前に分かる特徴だけ**から d を予測するモデルを作る。ただし
　　 **同じデータで学習して同じデータで予測する＝完全に過学習させる**。
　3. 予測の上位10%だけ買ったときの **実際の E[d|S]** を出す。
　★これは「未来を知っている人が、事前情報だけで最善の選択をした場合」の上界になる。
　　**この値が必要量に届かなければ、どんな選択ルールも届かない**。反則しても届かないので。
　4. 比較として、**dそのものの上位10%**（＝完全なオラクル。事前情報ですらない）も出す。
　　 これは絶対に到達不可能な天井。**必要量がこの天井すら超えていたら話は完全に終わる**。

★★事前登録
　1. **予想**: 事前情報オラクルの E[d|上位10%] は必要量（枠連0.2549）に届かない。
　2. **判定**: 届かなければ「**レース選択という手段は閉じた**」と書ける。
　　 届いたら、**そこから普通の（過学習でない）選択ルールを作る価値がある**ので宿題に格上げする。
　3. **★完全オラクルの値も必ず出す**。これが届かないなら、そもそも d の分布に十分な裾が無い
　　 ＝**買い目の選択でも、レースの選択でも、絶対に到達できない**という最も強い形の結論になる。
　4. 特徴は**発走前に分かるものだけ**。着順・払戻に依存する量を1つでも入れたら無効。

実行: python3 ml/audit_select_oracle.py [開始年(既定2015)]
"""
import math
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, "ml")
from audit_crosspool import PAYBACK, load_races, payoff, probs, zq
from audit_crosspool2 import PARTS, PAYKEY, realized
from audit_lbs import build_matrix, fit_lambda, q_of_lbs
from waku_umatan import waku_of


def pre_race_features(r, p):
    """★発走前に分かるものだけ。着順・払戻に触れたら無効。"""
    n = r["n"]
    srt = np.sort(p)[::-1]
    wk = defaultdict(int)
    for num, _, _ in r["horses"]:
        wk[waku_of(num, n)] += 1
    sizes = np.array(sorted(wk.values()))
    top_frame = waku_of(r["horses"][int(np.argmax(p))][0], n)
    return {
        "n": n,
        "n_frames": len(wk),
        "frame_max": int(sizes.max()),
        "frame_min": int(sizes.min()),
        "top_frame_size": wk[top_frame],
        "p1": float(srt[0]),
        "p2": float(srt[1]) if len(srt) > 1 else 0.0,
        "p3": float(srt[2]) if len(srt) > 2 else 0.0,
        "p1_p2": float(srt[0] - srt[1]) if len(srt) > 1 else 0.0,
        "ent": float(-(p * np.log(np.maximum(p, 1e-12))).sum()),      # オッズ分布の乱れ
        "hhi": float((p ** 2).sum()),                                  # 集中度
        "sum_inv": float(sum(1.0 / o for _, o, _ in r["horses"])),     # 控除率の実効値
        "place": int(r["rid"][:2]),
        "year": r["year"],
    }


def mci(x, alpha=0.01):
    x = np.asarray(x, float)
    m = x.mean()
    se = x.std(ddof=1) / math.sqrt(len(x))
    z = zq(alpha)
    return m, m - z * se, m + z * se


def main():
    y0 = int(sys.argv[1]) if len(sys.argv) > 1 else 2015
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
    print(f"(101) レース選択のオラクル上界（{y0}年以降）")
    print("★事前情報だけで**完全に過学習**させても必要量に届かないなら、選択という手段は閉じる\n")

    rows = []
    for r in races:
        yy = r["year"]
        if yy < y0 or not lam.get(yy):
            continue
        rl = realized(r)
        if rl is None:
            continue
        a, b, c = rl
        num2k = {num: k for k, (num, _, _) in enumerate(r["horses"])}
        if a not in num2k or b not in num2k or (c is not None and c not in num2k):
            continue
        p = probs(r["horses"])
        l2, l3 = lam[yy]
        f = pre_race_features(r, p)
        for kind, key in PARTS.items():
            if not r[key]:
                continue
            q, combo = q_of_lbs(kind, r, p, l2, l3, num2k, a, b, c)
            if q <= 0 or combo is None:
                continue
            v = payoff(r, PAYKEY[kind], combo)
            if not v or v <= 0:
                continue
            d = math.log(q) + math.log((v + 5) / 100.0) - math.log(PAYBACK[kind])
            rows.append(dict(f, kind=kind, d=d))
    df = pd.DataFrame(rows)
    print(f"対象 {len(df):,}件\n")

    import lightgbm as lgb
    FEAT = [c for c in df.columns if c not in ("kind", "d")]
    print("⚠**第1版の設計ミスを直した版**: 最初は『わざと過学習させたモデル』を上界にしたが、")
    print("　オッズ由来の連続値（p1/ent/hhi/sum_inv）が**レースの指紋になる**ので、モデルは")
    print("　レースを丸暗記して完全オラクルとほぼ同値（+0.3561 vs +0.3575）になった。上界として無意味。")
    print("　→ **ウォークフォワードで測る**。問いは『dは事前情報から予測できるか』の一点。\n")
    print(f"{'券種':<8}{'件数':>8}{'平均d':>10}{'σ(d)':>9}{'予測の相関':>11}"
          f"{'★実際に選べる':>15}{'完全オラクル':>13}{'必要量':>9}{'判定':>10}")
    for kind in PARTS:
        g = df[df["kind"] == kind].reset_index(drop=True)
        if len(g) < 2000:
            continue
        yrs_ = g["year"].to_numpy()
        y = g["d"].to_numpy()
        pred = np.full(len(g), np.nan)
        for yy in sorted(set(yrs_.tolist())):
            tr, te = yrs_ < yy, yrs_ == yy
            if tr.sum() < 3000:
                continue
            m = lgb.LGBMRegressor(n_estimators=300, num_leaves=31, learning_rate=0.05,
                                  verbose=-1).fit(g.loc[tr, FEAT], y[tr])
            pred[te] = m.predict(g.loc[te, FEAT])
        ok = ~np.isnan(pred)
        yy_, pp = y[ok], pred[ok]
        k = int(len(yy_) * 0.1)
        r = float(np.corrcoef(yy_, pp)[0, 1])
        top_wf = yy_[np.argsort(-pp)[:k]].mean()          # ★実運用で選べる値
        top_all = np.sort(yy_)[::-1][:k].mean()           # 完全オラクル（到達不可能）
        need = -math.log(PAYBACK[kind])
        mark = "★届く" if top_wf >= need else "届かない"
        print(f"{kind:<8}{len(yy_):>8,}{yy_.mean():>+10.4f}{yy_.std():>9.3f}{r:>+11.4f}"
              f"{top_wf:>+15.4f}{top_all:>+13.4f}{need:>9.4f}{mark:>10}")

    print("\n" + "=" * 104)
    print("★読み方")
    print("  ・**完全オラクルは必要量を超える**（枠連 +0.358 > 0.255）。つまり**裾は実在する**。")
    print("    儲かるレースは存在する。問題は『事前に見分けられるか』の一点に絞られる。")
    print("  ・『実際に選べる』はウォークフォワード＝実運用と同じ手続き。")
    print("    ここが必要量に届かないなら、**レースを選ぶという手段は閉じている**。")
    print("  ・『予測の相関』が0付近なら、d は事前情報から**まったく予測できない**ということ。")
    print("    d は配当の裾で決まるので、これを予測できるなら配当そのものを予測できることになる。")


if __name__ == "__main__":
    main()
