"""(112)運用版 — 「軸の複勝が甘いレース」を発走前に判定する。**標準ライブラリのみ**。

★何をする道具か
　(111)(112)で見つけた買い方を実運用で出すための部品。
　　**軸＝1番人気（単勝オッズ最小）／ 紐＝2・3番人気 → 三連複1点**
　　買うのは「**軸の複勝の期待払戻 E が小さいレース**」だけ。
　E は **単勝オッズだけから計算できる**ので、Mac でもクラウドでも同じ値が出る。
　　　E = 複勝の払戻率(0.8) ÷ (軸の3着以内確率) × 100 [円]

★確率の作り方（(96)で系統誤差を直した版）
　単勝オッズ → 正規化して勝率 p → **λ補正Harville**で3着以内確率へ。
　　1着: p をそのまま（(97)より τ≈1.0 で補正不要）
　　2着: p^λ2 に比例（λ2 = 0.848）
　　3着: p^λ3 に比例（λ3 = 0.720）
　★λは(96)で各年ウォークフォワード推定した値の最新年。11年間 0.846〜0.876 / 0.720〜0.744 で安定。

★★実測（(111)(112)・2015年以降・11年）— **どの数字を信じるべきか**
| 裾 | E の閾値 | 年間R数 | 三連複 軸→上位3頭 ROI | 99%CI | E[d｜S]（枠連） |
|---|---|---|---|---|---|
| 20% | ≤100円 | 596 | 81.4% | — | +0.0217 |
| 10% | ≤94円 | 298 | 79.3% | — | +0.0197 |
| 5% | ≤90円 | 149 | 83.2% | [70.7,95.7] | +0.0301 |
| **2%** | **≤86円** | **60** | **96.0%** | **[76.0,116.0]** | +0.0394 |
⚠**正直に書く**: 三連複の96.0%は**2%裾でだけ跳ねており単調でない**（81.4→79.3→83.2→96.0）。
　CIも[76.0,116.0]で何も区別できていない。**「最良のビンの罠」の可能性が高い**（判定基準6）。
　(112)でDを測ると三連複は **E[d|S]=−0.0213（必要量0.2877）** で、**情報量としては足りていない**。
　→ **この買い方は「期待値がプラス」ではない**。買うなら**そこを承知の上で**。
　★同じレース群で**枠連**なら D=+0.0394（必要量の15.5%）と、こちらの方が情報量は上。

実行（単体確認）: python3 ml/soft_axis.py 1.8 4.2 6.1 9.0 12.5 ...
"""
import sys

LAM2 = 0.848      # ★(96)のウォークフォワード推定・最新年。11年で0.846〜0.876
LAM3 = 0.720      # 同上。11年で0.720〜0.744
FUKU_PAYBACK = 0.80

# (裾, Eの閾値[円], 年間R数の目安) — (111)の実測から
TIERS = [(0.02, 86.0, 60), (0.05, 90.0, 149), (0.10, 94.0, 298), (0.20, 100.0, 596)]


def win_probs(odds):
    """単勝オッズ → 正規化した含意勝率。"""
    inv = [1.0 / o if o and o > 0 else 0.0 for o in odds]
    s = sum(inv)
    return [x / s for x in inv] if s > 0 else [0.0] * len(odds)


def top3_probs(p, lam2=LAM2, lam3=LAM3):
    """λ補正Harvilleで各馬の3着以内確率。λ=1 なら素のHarville。O(n^2)。"""
    n = len(p)
    w2 = [x ** lam2 for x in p]
    w3 = [x ** lam3 for x in p]
    W2, W3 = sum(w2), sum(w3)
    out = []
    for i in range(n):
        # 1着
        q = p[i]
        # 2着: Σ_{x≠i} p_x · w2_i/(W2−w2_x)
        for x in range(n):
            if x == i:
                continue
            d1 = W2 - w2[x]
            if d1 > 1e-12:
                q += p[x] * w2[i] / d1
        # 3着: Σ_{x≠i} Σ_{y≠i,x} p_x · w2_y/(W2−w2_x) · w3_i/(W3−w3_x−w3_y)
        for x in range(n):
            if x == i:
                continue
            d1 = W2 - w2[x]
            if d1 <= 1e-12:
                continue
            for y in range(n):
                if y in (i, x):
                    continue
                d2 = W3 - w3[x] - w3[y]
                if d2 > 1e-12:
                    q += p[x] * (w2[y] / d1) * (w3[i] / d2)
        out.append(min(max(q, 1e-9), 1 - 1e-9))
    return out


def axis_expect(odds):
    """→ (軸のindex, 軸の複勝の期待払戻E[円], 軸の3着以内確率)。**発走前に計算できる**。"""
    p = win_probs(odds)
    if not p or max(p) <= 0:
        return None, None, None
    k = max(range(len(p)), key=lambda i: p[i])       # 軸＝1番人気
    t3 = top3_probs(p)
    return k, FUKU_PAYBACK / t3[k] * 100.0, t3[k]


def tier_of(e):
    """E が入る裾。該当しなければ None。"""
    for t, thr, _ in TIERS:
        if e <= thr:
            return t, thr
    return None


def recommend(umabans, odds):
    """→ dict。買い目は**オッズ順**（モデルではない）で決める。

    軸＝1番人気 / 紐＝2・3番人気 → 三連複1点。
    """
    if len(umabans) < 3:
        return None
    order = sorted(range(len(odds)), key=lambda i: odds[i])   # 単勝オッズの小さい順
    k, e, q = axis_expect(odds)
    if e is None:
        return None
    tier = tier_of(e)
    trio = sorted(umabans[i] for i in order[:3])
    return {"axis": umabans[order[0]], "trio": trio,
            "sanrenpuku": "-".join(str(x) for x in trio),
            "e_axis": round(e, 1), "q_axis": round(q, 4),
            "tier": (None if tier is None else tier[0]),
            "buy": tier is not None}


def main():
    odds = [float(x) for x in sys.argv[1:]]
    if len(odds) < 3:
        sys.exit("使い方: python3 ml/soft_axis.py <単勝オッズを頭数ぶん>")
    r = recommend(list(range(1, len(odds) + 1)), odds)
    print(f"軸 {r['axis']}番（{min(odds):.1f}倍）")
    print(f"軸の3着以内確率 {r['q_axis']*100:.1f}%  →  複勝の期待払戻 {r['e_axis']:.0f}円")
    if r["buy"]:
        print(f"★買う（裾{int(r['tier']*100)}%に入っている）  三連複 {r['sanrenpuku']}（1点100円）")
    else:
        print(f"見送り（Eが{TIERS[-1][1]:.0f}円を超えている＝甘くない）")


if __name__ == "__main__":
    main()
