"""(126) ★★「確かめる術がない」と棚上げした3項目を **D で測り直す**（2026-08-11）

★なぜやるか（今日の(117)と同じ構図）
　HANDOFFの「**検討軸として残すもの（否定されたのではなく確かめる術がない）**」にはこうある:
　　・血統(父)の道悪適性 × 馬場((43)): +6.25pt・CI[−8.86,+22.45]
　　　**確定には約7倍のレース数（20万R＝60年分）が必要**
　　・父のコース適性((44)B): +3.00pt・CI[−4.75,+10.67]
　　・脚質 × 頭数((44)C): −8.93pt
　**この「60年必要」はROIで計算した数字**。今日(117)で、(62)の「ROIでは84.5→85.6%が限界」が
　**Dで測ったら全水準★・10/10年で正**だった。**ROIで諦めた場所はDなら決着しうる**。
　(91)も「Dなら割っても持つ（枠連で全体CI幅0.009）」と書いている。ROIは1ptに50年。**桁が2つ違う**。
　→ **「データが足りない」ではなく「物差しが粗かった」**かもしれない。**新データは1行も要らない**。

★測り方（(123)と同じ形。あちらは市場の織り込みを測って β=0 が最良だった）
　市場の含意勝率 p を `exp(β·z(指標))` で傾けて **E[Δlog q(勝ち馬)]** を β の関数として見る。
　β>0 で上がる → **市場がその軸を過小評価している**＝信号。β=0 が最良 → 織り込み済み。

★★指標（3つとも先に宣言。**すべて「その行より前の実績だけ」で作る**＝リーク防止）
　S1 **父の道悪適性 × 馬場**: 父の「道悪での市場残差(y − p_mkt)」の拡張平均。
　　　**道悪のレースでだけ効かせる**（良馬場では0）。件数で縮小（n/(n+50)）。
　S2 **父のコース適性**: 父の「その場での残差」の拡張平均 − 「全場での残差」の拡張平均。
　S3 **脚質 × 頭数**: その馬の過去の道中位置（passavg/頭数）の拡張平均から先行度を作り、
　　　**今走の頭数のz**を掛ける。(44)Cの「多頭数で先行有利」がそのまま指標になる。
　β は **−0.4/−0.2/−0.1/0/+0.1/+0.2/+0.4** の7点。**プラセボ**（レース内シャッフル）を必ず並べる。

★★予想
　**3つとも β=0 が最良**（＝織り込み済み）と予想する。(123)で「公開情報は過不足なく
　織り込まれている」がまた確認されたばかりで、血統も脚質も公開情報だから。
　⚠今日3回予想を外している（(117)(122)(123)の一部）。**この予想もあてにしないこと**。
　★ただし**目的は「儲かるか」ではなく「決着させるか」**。β=0が最良と**精度よく言える**なら、
　　3項目を「確かめる術がない」から「**確かめた。無い**」に格上げできる。それが成果。

⚠リークの限界（正直に書く）: 拡張平均は日付順の累積から自分の行を引いて作る。
　**同じ日の同じ父の他レース**はわずかに混じる。件数に対して無視できる量だが、ゼロではない。

実行: python3 ml/audit_leftovers_d.py [開始年(既定2015)]
"""
import math
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "ml")
import features as F
from audit_crosspool import zq

BETAS = [-0.4, -0.2, -0.1, 0.0, 0.1, 0.2, 0.4]
K = 50.0                                  # 縮小の強さ（件数がこれ未満なら効きを弱める）
RNG = np.random.default_rng(20260811)


def mci(x, alpha=0.01):
    x = np.asarray(x, float)
    m = x.mean()
    se = x.std(ddof=1) / math.sqrt(len(x))
    return m, m - zq(alpha) * se, m + zq(alpha) * se


def expanding(d, key, val, cnt):
    """`key` ごとの**自分より前だけ**の (合計, 件数)。日付順に並んでいる前提。"""
    g = d.groupby(key, observed=True)
    return (g[val].cumsum() - d[val]), (g[cnt].cumsum() - d[cnt])


def main():
    y0 = int(sys.argv[1]) if len(sys.argv) > 1 else 2015
    d = F.to_model(F.load_files())
    d = d.dropna(subset=["odds", "finish", "fieldsize"])
    d = d[(d["odds"] > 0) & (d["sire"] != "")].copy()
    d = d.sort_values("date").reset_index(drop=True)
    d["y"] = (d["finish"] == 1).astype(float)
    inv = 1.0 / d["odds"].to_numpy()
    d["p_mkt"] = inv / d.groupby("raceid")["odds"].transform(lambda s: (1.0 / s).sum()).to_numpy()
    d["resid"] = d["y"] - d["p_mkt"]
    d["one"] = 1.0

    # ── S1 父の道悪適性 × 馬場 ──
    wet = d["cond"].isin(["稍", "重", "不"])
    d["wr"] = d["resid"].where(wet, 0.0)
    d["wc"] = wet.astype(float)
    sw, cw = expanding(d, "sire", "wr", "wc")
    d["S1"] = (sw / (cw + K)).fillna(0.0) * wet.astype(float)

    # ── S2 父のコース適性（その場 − 全場）──
    d["sc_key"] = d["sire"].astype(str) + "|" + d["course"].astype(str)
    sc, cc = expanding(d, "sc_key", "resid", "one")
    sa, ca = expanding(d, "sire", "resid", "one")
    d["S2"] = ((sc / (cc + K)) - (sa / (ca + K))).fillna(0.0)

    # ── S3 脚質 × 頭数（先行度 × 今走の頭数z）──
    d["passratio"] = (d["passavg"] / d["fieldsize"]).clip(0, 1)
    d["pr"] = d["passratio"].fillna(0.5)
    sp, cp = expanding(d, "horse", "pr", "one")
    lead = 1.0 - (sp / (cp + 3.0)).fillna(0.5)          # 先行度（大きいほど前で運ぶ）
    fs = d["fieldsize"].to_numpy(float)
    d["S3"] = lead.to_numpy() * ((fs - fs.mean()) / fs.std())

    d = d[d["date"].dt.year >= y0].copy()
    print(f"(126) 「確かめる術がない」3項目を D で測る（{y0}年以降）")
    print("★ROIで『60年必要』とされたもの。**新データは1行も要らない**\n")

    races = [g for _, g in d.groupby("raceid") if g["y"].sum() == 1 and len(g) >= 5]
    print(f"対象 {len(races):,}レース\n")

    LAB = {"S1": "父の道悪適性 × 馬場（(43)・ROIでは+6.25pt / 60年必要）",
           "S2": "父のコース適性（(44)B・ROIでは+3.00pt）",
           "S3": "脚質 × 頭数（(44)C・ROIでは−8.93pt）"}
    for tag in ("S1", "S2", "S3"):
        rows = {b: [] for b in BETAS}
        pla = {b: [] for b in BETAS}
        for gg in races:
            p = gg["p_mkt"].to_numpy(float)
            t = gg[tag].to_numpy(float)
            sd = t.std()
            tz = (t - t.mean()) / sd if sd > 1e-12 else np.zeros_like(t)
            j = int(np.argmax(gg["y"].to_numpy()))
            ts = tz[RNG.permutation(len(tz))]
            for b in BETAS:
                for arr, tt in ((rows, tz), (pla, ts)):
                    q = p * np.exp(b * tt)
                    q /= q.sum()
                    arr[b].append(math.log(max(q[j], 1e-300)) - math.log(max(p[j], 1e-300)))
        print(f"■ {tag}  {LAB[tag]}")
        print(f"{'β':>6}{'E[Δlog q]':>12}{'99%CI':>24}{'プラセボ':>11}{'実測−プラセボ':>14}")
        best = None
        for b in BETAS:
            m, lo, hi = mci(rows[b])
            pm = float(np.mean(pla[b]))
            if best is None or m > best[1]:
                best = (b, m, lo)
            ci = "[" + format(lo, "+.5f") + "," + format(hi, "+.5f") + "]"
            print(f"{b:>6.2f}{m:>+12.5f}{ci:>24}{pm:>+11.5f}{m-pm:>+14.5f}")
        mark = "★市場が過小評価" if (best[0] > 0 and best[2] > 0) else ""
        print(f"  → 最良は β={best[0]:+.2f}（{best[1]:+.5f}）{mark}\n")

    print("=" * 96)
    print("★読み方（事前登録のとおり）")
    print("  ・β=0 が最良 → **織り込み済み**。3項目を『確かめる術がない』から")
    print("    **『確かめた。無い』に格上げできる**。これが目的。")
    print("  ・β>0 が最良で99%CI下端が0超 → **市場が過小評価**＝新しい信号。")
    print("  ⚠ここで出るのは単勝プールに対する増分。必要量(単勝0.2231)と比べること。")


if __name__ == "__main__":
    main()
