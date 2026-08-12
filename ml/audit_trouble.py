"""(123) ★★「前走で不利を受けた馬」を市場は過小評価しているか（2026-08-11・ユーザー発案）

★問いの立て方（ここを間違えると必ず失敗する）
　**「不利を検出できるか」を測っても意味が無い**。前走の不利は新聞にもnetkeibaの短評にも載る。
　**誰でも知っている情報は既にオッズに入っている**。(75)で「市場との差分を学習する」が
　全滅したのと同じ構造で、決定的だったのは「残差だけで順位付けするとAUC 0.3605＝逆相関。
　残差が最大になるのは**市場が最低評価なのに走った馬**なので、残差順は人気薄順になる」。
　→ 測るべきは **「市場が不利を過小評価しているか」**。それだけが信号になる。

★測り方
　市場の含意勝率 p を **exp(β·z(不利指標))** で傾けて正規化し、
　**E[log q'(勝ち馬) − log p(勝ち馬)]** を β の関数として出す。
　β>0 で上がる → 市場は不利を**割り引き切れていない**（買い材料）
　β=0 が最良    → 市場は**正しく織り込んでいる**
　β<0 で上がる → 市場は**過剰に補正している**（「前走不利」は買われすぎ）

★★事前登録（測る前に4つの指標を全部宣言する。後から良かったものを選ばない）
　前走のレース内で順位化して頭数で正規化する。**大きいほど「不利があったっぽい」**:
　　T1 **脚は使えたのに着順が悪い** = 着順の順位 − 上がり3Fの順位
　　　（詰まって出し切れず、空いてから伸びた典型）
　　T2 **道中は前にいたのに沈んだ** = 着順の順位 − 道中平均位置の順位
　　T3 **着差は僅少なのに掲示板外** = 6着以下で、かつ着差(秒)が小さい
　　T4 **T1〜T3の平均**（合成）
　β は **−0.4/−0.2/−0.1/0/+0.1/+0.2/+0.4** の7点。後から増やさない。
　**プラセボ**: 指標をレース内でシャッフルして同じ手続き。

★★予想（外れたら記録する）
　**β=0 が最良、つまり効かない**と予想する。むしろ **β<0 側が良い**可能性がある——
　「前走不利＝次は買い」は最も人口に膾炙した買い方なので、**買われすぎ**ているはず。
　⚠T2は「単に失速した馬」も拾う。**不利の代理として弱い**ことを先に認めておく。

実行: python3 ml/audit_trouble.py [開始年(既定2015)]
"""
import math
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "ml")
import features as F
from audit_crosspool import zq

BETAS = [-0.4, -0.2, -0.1, 0.0, 0.1, 0.2, 0.4]      # ★先に宣言
RNG = np.random.default_rng(20260811)


def mci(x, alpha=0.01):
    x = np.asarray(x, float)
    m = x.mean()
    se = x.std(ddof=1) / math.sqrt(len(x))
    return m, m - zq(alpha) * se, m + zq(alpha) * se


def main():
    y0 = int(sys.argv[1]) if len(sys.argv) > 1 else 2015
    d = F.to_model(F.load_files())
    d = d.dropna(subset=["odds", "finish"])
    d = d[d["odds"] > 0].copy()
    d["year"] = d["date"].dt.year

    # ── 前走の「不利があったっぽさ」を作る（レース内で順位化 → 頭数で正規化）──
    g = d.groupby("raceid")
    d["r_fin"] = g["finish"].rank()
    d["r_ag"] = g["agari"].rank()             # 上がりは小さいほど速い＝順位1が最速
    d["r_pass"] = g["passavg"].rank()         # 道中平均位置。小さいほど前
    n = d["fieldsize"].to_numpy(float)
    d["T1"] = (d["r_fin"] - d["r_ag"]) / n
    d["T2"] = (d["r_fin"] - d["r_pass"]) / n
    # T3: 6着以下で着差が小さいほど大きい。margin は秒差
    d["T3"] = np.where(d["finish"] >= 6, 1.0 / (1.0 + d["margin"].abs().fillna(9.9)), 0.0)
    for c in ("T1", "T2", "T3"):
        d[c] = d[c].fillna(0.0)
    z = d[["T1", "T2", "T3"]]
    d["T4"] = ((z - z.mean()) / z.std()).mean(axis=1)

    # ★**前走**の値を今走に持ってくる（同じレースの値を使ったらリーク）
    d = d.sort_values(["horse", "date"])
    for c in ("T1", "T2", "T3", "T4"):
        d[f"p_{c}"] = d.groupby("horse")[c].shift(1)
    d = d[d["year"] >= y0].dropna(subset=["p_T4"]).copy()

    # 市場の含意勝率（レース内で正規化）
    inv = 1.0 / d["odds"].to_numpy()
    d["p_mkt"] = inv / d.groupby("raceid")["odds"].transform(lambda s: (1.0 / s).sum()).to_numpy()
    d["win"] = (d["finish"] == 1).astype(int)

    # レース単位にまとめる（全馬そろっているレースだけ）
    races, drop = [], 0
    for rid, gg in d.groupby("raceid"):
        if gg["win"].sum() != 1 or len(gg) < 5:
            drop += 1
            continue
        races.append(gg)
    print(f"(123) 前走の不利を市場は過小評価しているか（{y0}年以降・{len(races):,}レース）")
    print("★測るのは「不利を検出できるか」ではなく「市場が過小評価しているか」\n")

    for tag in ("T1", "T2", "T3", "T4"):
        col = f"p_{tag}"
        rows = {b: [] for b in BETAS}
        pla = {b: [] for b in BETAS}
        for gg in races:
            p = gg["p_mkt"].to_numpy(float)
            t = gg[col].to_numpy(float)
            sd = t.std()
            tz = (t - t.mean()) / sd if sd > 1e-9 else np.zeros_like(t)
            j = int(np.argmax(gg["win"].to_numpy()))
            ts = tz[RNG.permutation(len(tz))]
            for b in BETAS:
                for arr, tt in ((rows, tz), (pla, ts)):
                    q = p * np.exp(b * tt)
                    q = q / q.sum()
                    arr[b].append(math.log(max(q[j], 1e-300)) - math.log(max(p[j], 1e-300)))
        print(f"■ {tag}" + {"T1": "（脚は使えたのに着順が悪い）",
                            "T2": "（道中は前にいたのに沈んだ）",
                            "T3": "（着差は僅少なのに掲示板外）",
                            "T4": "（3つの合成）"}[tag])
        print(f"{'β':>6}{'E[Δlog q]':>12}{'99%CI':>22}{'プラセボ':>11}{'実測−プラセボ':>14}")
        best = None
        for b in BETAS:
            m, lo, hi = mci(rows[b])
            pm = float(np.mean(pla[b]))
            if best is None or m > best[1]:
                best = (b, m)
            print(f"{b:>6.2f}{m:>+12.5f}{f'[{lo:+.5f},{hi:+.5f}]':>22}"
                  f"{pm:>+11.5f}{m-pm:>+14.5f}")
        print(f"  → 最良は β={best[0]:+.2f}（{best[1]:+.5f}）\n")

    print("=" * 92)
    print("★読み方（事前登録のとおり）")
    print("  ・β=0 が最良 → **市場は不利を正しく織り込んでいる**。この筋は閉じる。")
    print("  ・β>0 が最良で99%CIが0を外れる → **市場は不利を過小評価**＝新しい信号。")
    print("  ・β<0 が最良 → **「前走不利」は買われすぎ**。逆張りの材料になる（が控除率は越えない）。")
    print("  ⚠ここで出るのは **単勝プールに対する増分**。必要量(単勝0.2231)と比べること。")


if __name__ == "__main__":
    main()
