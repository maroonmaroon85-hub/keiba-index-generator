"""(136) ★★★枠連プールの較正を**組み合わせ単位**で検定する — **モデルを一切使わない**（2026-08-12）

★なぜやるか（(135)の直後の宿題）
　(135)で分かったのは「**優位の93%は『誰が来るか』側**」。ただし切り分けに使った B（本命の枠が
　的中組に入るか）は**結果**であり、**発走前には選べない**（判定基準19として記録した）。
　→ **発走前に選べる形で同じ問いを立て直す**のがこれ。

★★この実験の一番の値打ち: **(127)の答えを待たずに済む**
　いま積み上がっている数字は全部 `q = λ補正Harville(単勝)→枠` を通っており、(89)⑥の
　「**枠連プールが甘いのか / Harvilleが枠集約で誤差を相殺しているのか**」が未分離のまま。
　★**ここではモデルもHarvilleも使わない**。**板の値段と、実際に来たかどうか**だけを突き合わせる。
　　→ **もしプールの較正が崩れている区分が見つかれば、それはHarvilleと無関係に本物**。
　　→ **見つからなければ、「プールは組の値付けを正しくやっている」という独立の証拠**になり、
　　　 (135)の +0.0161 は**Harville側の性質**である疑いが強まる。**どちらに転んでも情報がある**。

★★測り方（レース単位の不偏統計量。(99)のバイアスを踏まない形）
```
　q_pool(k) = (1/o_k) / Σ_j (1/o_j)      ← 板の全枠組で正規化＝控除率を抜く
　区分 g について、レース r ごとに
　　X_r = 1{的中組 ∈ g} − Σ_{k∈g} q_pool(k)
　帰無仮説のもとで E[X_r] = 0。**レース間は独立**なのでそのまま平均とCIが取れる。
```
　★**結果で標本を選んでいない**（全レースが全区分に寄与する）ので(99)の罠に落ちない。

⚠★**判定基準9（恒等式の罠）を先に潰しておく**
　レース内で q_pool を正規化しているので **Σ_全組 q_pool = 1 = Σ_全組 1{的中}** が恒等的に成立し、
　**「全体」の O/E は必ず 1.0000 になる**。**これは検査ではない**。
　**情報があるのは部分集合（区分）だけ**。全体の行は「道具が正しく動いている」検算として出す。

★★事前登録（測る前に宣言する。**後から区分を増やさない**）
　L1 **本命の枠を含む組 / 含まない組**（2区分）← (135)のBを**買い目の側**に移したもの。**主判定**
　L2 **q_pool の十分位**（10区分）← 人気-穴のバイアスがあるならここに出る
　L3 **ゾロ目（同枠2頭）/ 異なる枠**（2区分）
　1. 統計量は上の X_r。**99%CI**。区分が計14あるので **Bonferroni（α=0.01/14）**で判定する。
　2. **年分割で符号が揃うか**（8/11年以上）も併せて要求する。
　3. ⚠★**プラセボについて — 判定基準13を自分に当てて、素直に引くのは無意味だと分かった**。
　　 **任意の固定した区分について E[X_r] は帰無仮説のもとで厳密に0**である
　　 （区分に n_c 組が入るなら、並べ替えの上での期待値は (n_c/K)(Σ1{的中} − Σq_pool) = 0）。
　　 → **無作為な区分を引いても必ず0が出る**ので、**統計的には何も検査していない**（判定基準9）。
　　 ★そこで**プラセボは「実装の検査」として位置づけを下げ**、
　　 　**本命の枠を無作為な枠に差し替えた版を30回引いて平均**する（L1のみ・0になるはず）。
　　 **統計的な守りは Bonferroni と年分割**が担う。**プラセボに守らせない**。
　4. ★**運用が変わる条件**: ある区分の **O/E ≥ 1/払戻率 = 1.290**（＝控除率を超える）で
　　 上の1.2.を満たすこと。**それ未満なら記述にとどめ、運用は変えない**。
　　 ⚠**O/E が 1 を有意に超えても、1.29 に届かなければ賭けとしては負け**。ここを混同しない。
　5. **予想**: ★**L1は平坦**（(135)の D_binary=+0.0012・CIが0をまたぐ、と整合するはず）。
　　 **L2には人気-穴のバイアスが出る**（(105)「妙味の無い組は買われない」は市場全体の性質）。
　　 **L3は平坦**。**どれも1.29には遠く届かない**。
　　 ⚠**直前の(135)で予想を大きく外している**（6:4と言って実測は6.7:93.3）。**あてにしないこと**。
　6. ★**検算**: 全体のO/Eが 1.0000 になること（上の恒等式）。ならなければ**先に道具を疑う**。

実行: python3 ml/audit_pool_calib.py [開始年(既定2013=板の全期間)]
"""
import math
import sys

import numpy as np

sys.path.insert(0, "ml")
from audit_cond_split import load_boards
from audit_crosspool import PAYBACK, load_races, probs, zq
from audit_crosspool2 import realized
from waku_umatan import waku_of

NPLA = 30                                    # プラセボ＝**実装の検査**。統計は守らない（上の3.）
NGRP = 2 + 10 + 2                            # L1(2) + L2(10) + L3(2) = 14区分
ALPHA = 0.01 / NGRP                          # Bonferroni
NEED_OE = 1.0 / PAYBACK["枠連"]              # 控除率を超えるのに要る O/E
RNG = np.random.default_rng(20260812)


def mci(x, alpha):
    x = np.asarray(x, float)
    if len(x) < 2:
        return float("nan"), float("nan"), float("nan")
    m = x.mean()
    se = x.std(ddof=1) / math.sqrt(len(x))
    z = zq(alpha)
    return m, m - z * se, m + z * se


NEED_D = 0.2549                              # 枠連で儲かるのに要るD（(89)）


def dequiv(O, E, n):
    """★区分のずれを**D換算**する（事後の換算であって新しい検定ではない）。

    その区分の真の頻度 O/n と板の言い値 E/n のKLダイバージェンス
    ＝**そのラベルだけを正しく直したときに得られるDの上限**。**必要量と直接比べられる**。
    """
    d = 0.0
    for o, e in zip(O, E):
        if o > 0 and e > 0:
            d += (o / n) * math.log((o / n) / (e / n))
    return d


def report(name, labs, O, E, X, ys, pla, n):
    """O=的中数 / E=板の期待値 / X=レース単位の差 / pla=プラセボの平均差"""
    print(f"\n■ {name}")
    print(f"{'区分':<22}{'的中O':>9}{'板E':>11}{'O/E':>8}{'E[X]':>10}"
          f"{f'{100*(1-ALPHA):.3f}%CI':>22}{'プラセボ':>10}{'正の年':>8}")
    for i, lb in enumerate(labs):
        m, lo, hi = mci(X[:, i], ALPHA)
        oe = O[i] / E[i] if E[i] > 0 else float("nan")
        pos = 0
        yl = sorted(set(ys.tolist()))
        for yy in yl:
            k = ys == yy
            if k.sum() >= 100 and X[k, i].mean() * (1 if m >= 0 else -1) > 0:
                pos += 1
        ci = "[" + format(lo, "+.4f") + "," + format(hi, "+.4f") + "]"
        sig = "★" if (lo > 0 or hi < 0) else ""
        buy = "◎買える" if (lo > 0 and oe >= NEED_OE) else ""
        print(f"{lb:<22}{O[i]:>9.0f}{E[i]:>11.1f}{oe:>8.3f}{m:>+10.4f}{ci:>22}"
              f"{pla[i]:>+10.4f}{pos:>5}/{len(yl)} {sig}{buy}")
    d = dequiv(O, E, n)
    print(f"  ★D換算 = {d:+.4f}  ＝必要量({NEED_D:.4f})の **{100*d/NEED_D:.1f}%**"
          f"　※このラベルを完璧に直しても得られるのはここまで")


def main():
    y0 = int(sys.argv[1]) if len(sys.argv) > 1 else 2013
    races = load_races()
    boards = load_boards()
    print(f"レース {len(races)} / 枠連の板 {len(boards)}", flush=True)
    print(f"★モデルもHarvilleも使わない。板の値段と的中だけ。")
    print(f"★Bonferroni: {NGRP}区分なので各区分は {100*(1-ALPHA):.4f}%CI で判定する")
    print(f"★控除率を超えるのに要る O/E = {NEED_OE:.3f}（払戻率 {PAYBACK['枠連']:.3f}）\n", flush=True)

    L1 = ["本命の枠を含む", "本命の枠を含まない"]
    L2 = [f"第{i+1}十分位{'（最も人気薄）' if i == 0 else '（最も人気）' if i == 9 else ''}"
          for i in range(10)]
    L3 = ["ゾロ目（同枠2頭）", "異なる枠"]
    nO = np.zeros(NGRP)
    nE = np.zeros(NGRP)
    xs, yy_list = [], []
    pla_sum = np.zeros(NGRP)

    for r in races:
        if r["year"] < y0 or not r["wakuren"]:
            continue
        bd = boards.get(r["rid"])
        if not bd:
            continue
        rl = realized(r)
        if rl is None:
            continue
        a, b, _ = rl
        num2k = {u: k for k, (u, _, _) in enumerate(r["horses"])}
        if a not in num2k or b not in num2k:
            continue
        keys = sorted(bd)
        if len(keys) < 3:
            continue
        key = tuple(sorted((waku_of(a, r["n"]), waku_of(b, r["n"]))))
        if key not in bd:
            continue

        inv = np.array([1.0 / bd[t] for t in keys])
        qp = inv / inv.sum()
        p = probs(r["horses"])
        fav = int(np.argmax(p))
        wf = waku_of(r["horses"][fav][0], r["n"])

        win = np.array([t == key for t in keys], float)
        # ── 区分の割り当て（各レース・各組を14列のどれかに立てる）──
        M = np.zeros((len(keys), NGRP))
        hasf = np.array([wf in t for t in keys])
        M[hasf, 0] = 1.0
        M[~hasf, 1] = 1.0
        # q_pool の十分位は**そのレースの中で**切る（レース間で値域が違うため）
        dec = np.clip((np.argsort(np.argsort(qp)) * 10) // len(keys), 0, 9)
        for j in range(len(keys)):
            M[j, 2 + dec[j]] = 1.0
        zoro = np.array([t[0] == t[1] for t in keys])
        M[zoro, 12] = 1.0
        M[~zoro, 13] = 1.0

        o = win @ M
        e = qp @ M
        nO += o
        nE += e
        xs.append(o - e)
        yy_list.append(r["year"])

        # ── プラセボ（★実装の検査。統計的には0が出るのが分かっている）──
        # 　本命の枠を**無作為な枠**に差し替えてL1を作り直す。0から離れたら実装を疑う。
        frames = sorted({w for t in keys for w in t})
        for _ in range(NPLA):
            wr = frames[int(RNG.integers(len(frames)))]
            hr = np.array([wr in t for t in keys], float)
            pla_sum[0] += float(win @ hr - qp @ hr)
            pla_sum[1] += float(win @ (1 - hr) - qp @ (1 - hr))

    if not xs:
        sys.exit("突き合わせできたレースが無い")
    X = np.array(xs)
    ys = np.array(yy_list)
    n = len(X)
    pla = pla_sum / (NPLA * n)
    print(f"対象 {n:,} レース（{ys.min()}〜{ys.max()}）\n")

    tot_o, tot_e = nO[:2].sum(), nE[:2].sum()
    print(f"★検算（判定基準9）: 全体の O/E = {tot_o/tot_e:.4f}（**1.0000 で正しい。これは検査ではない**）")

    report("L1 ★主判定 — 本命の枠を含むか（(135)のBを買い目の側に移したもの）", L1,
           nO[:2], nE[:2], X[:, :2], ys, pla[:2], n)
    report("L2 q_pool の十分位（★第1＝q_poolが最小＝**人気薄**。第10＝人気）", L2,
           nO[2:12], nE[2:12], X[:, 2:12], ys, pla[2:12], n)
    report("L3 ゾロ目か", L3, nO[12:], nE[12:], X[:, 12:], ys, pla[12:], n)

    print("\n" + "=" * 100)
    print("★読み方（事前登録のとおり。**後から足していない**）")
    print("  ・★が付く＝**板の値付けがその区分でずれている**（Bonferroni込み）。**モデル無しの事実**。")
    print(f"  ・◎買える＝ずれが**控除率を超えている**（O/E ≥ {NEED_OE:.3f}）。**ここが付かなければ賭けにならない**。")
    print("  ・★が付いても◎が付かないのが既定の予想。**『有意』と『儲かる』を混同しないこと**。")
    print("  ・**L1が平坦なら**、(135)の D_binary≒0 と整合し、")
    print("    **「本命が絡むか」をプールは正しく値付けしている**が独立に確認されたことになる。")
    print("  ・**L1・L2・L3のどれも平坦なら**、(135)の +0.0161 は")
    print("    **発走前に選べる粗いラベルでは取り出せない**＝**Harville側の性質である疑いが強まる**。")
    print("    → **(127)の答え合わせに使えるが、代わりにはならない**。(127)は馬連の板で決める。")
    print("  ★**D換算の行が本体**。『有意に外れている』を『どれだけ稼げるか』に翻訳した数字であり、")
    print("    **必要量に対して何%かがそのまま結論**になる。")


if __name__ == "__main__":
    main()
