"""(112)運用版 — 「軸の複勝が甘いレース」を発走前に判定する。**標準ライブラリのみ**。

★何をする道具か
　(111)(112)で見つけた買い方を実運用で出すための部品。
　　**人気上位3頭（1・2・3番人気）ちょうどで三連複1点**
　　★「軸→流し」ではない。**3頭固定＝組み合わせは1通り＝1点100円**。
　　　（(111)では「軸→上位3頭」と書いていたが流しに読めるので改名した。中身は同じ）
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
| 裾 | E の閾値 | 年間R数 | 三連複 人気上位3頭 ROI | 99%CI | E[d｜S]（枠連） |
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

# ★「買う」と判定する既定の水準。**最も絞る(2%・≤86円)** を既定にする。
# 　理由: 三連複のROI 96.0% は**この水準でしか出ていない**（≤90円で83.2% / ≤100円で81.4%）。
# 　既定を緩くすると「96.0%を見て買ったつもりが、実際には81.4%の買い目だった」が起きる。
DEFAULT_TIER = 0.02


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


def trio_prob(p, idx, lam2=LAM2, lam3=LAM3):
    """3頭の組が1〜3着を占める確率（λ補正Harville・6通りの順序を足す）。"""
    w2 = [x ** lam2 for x in p]
    w3 = [x ** lam3 for x in p]
    W2, W3 = sum(w2), sum(w3)
    a, b, c = idx
    tot = 0.0
    for x, y, z in ((a, b, c), (a, c, b), (b, a, c), (b, c, a), (c, a, b), (c, b, a)):
        d1 = W2 - w2[x]
        d2 = W3 - w3[x] - w3[y]
        if d1 > 1e-12 and d2 > 1e-12:
            tot += p[x] * (w2[y] / d1) * (w3[z] / d2)
    return tot


def detail(umabans, odds, tier=DEFAULT_TIER):
    """1レースの中身を全部出す。**すべて単勝オッズだけから計算している**（モデル不使用）。"""
    r = recommend(umabans, odds, tier)
    if r is None:
        return None
    p = win_probs(odds)
    t3 = top3_probs(p)
    order = sorted(range(len(odds)), key=lambda i: odds[i])
    idx = [umabans.index(x) for x in r["trio"]]
    q = trio_prob(p, idx)
    r["trio_prob"] = q
    r["trio_expect"] = (0.75 / q * 100.0) if q > 0 else None      # 三連複の払戻率は75%
    r["horses"] = [{"umaban": umabans[i], "odds": odds[i],
                    "win": p[i], "top3": t3[i],
                    "role": ("買" if i in order[:3] else ""),
                    "pop": order.index(i) + 1}
                   for i in order]
    return r


def axis_expect(odds):
    """→ (軸のindex, 軸の複勝の期待払戻E[円], 軸の3着以内確率)。**発走前に計算できる**。"""
    p = win_probs(odds)
    if not p or max(p) <= 0:
        return None, None, None
    k = max(range(len(p)), key=lambda i: p[i])       # 軸＝1番人気
    t3 = top3_probs(p)
    return k, FUKU_PAYBACK / t3[k] * 100.0, t3[k]


def tier_of(e):
    """E が入る裾（最も絞ったものを返す）。どれにも入らなければ None。"""
    for t, thr, _ in TIERS:
        if e <= thr:
            return t, thr
    return None


def threshold_of(tier):
    """裾の水準 → E の閾値[円]。"""
    for t, thr, _ in TIERS:
        if abs(t - tier) < 1e-9:
            return thr
    raise ValueError(f"tier は {[t for t, _, _ in TIERS]} のいずれか")


def recommend(umabans, odds, tier=DEFAULT_TIER):
    """→ dict。買い目は**オッズ順**（モデルではない）で決める。

    人気上位3頭ちょうどで三連複1点（流しではない）。
    `tier` を緩めると買うレースは増えるが、**実測ROIは下がる**（96.0→83.2→79.3→81.4）。
    """
    if len(umabans) < 3:
        return None
    order = sorted(range(len(odds)), key=lambda i: odds[i])   # 単勝オッズの小さい順
    k, e, q = axis_expect(odds)
    if e is None:
        return None
    thr = threshold_of(tier)
    fell = tier_of(e)                       # 実際にどの裾まで入ったか（表示用）
    trio = sorted(umabans[i] for i in order[:3])
    return {"axis": umabans[order[0]], "trio": trio,
            "sanrenpuku": "-".join(str(x) for x in trio),
            "e_axis": round(e, 1), "q_axis": round(q, 4),
            "tier": (None if fell is None else fell[0]),
            "buy_tier": tier, "buy_threshold": thr,
            "buy": e <= thr}


def main():
    odds = [float(x) for x in sys.argv[1:]]
    if len(odds) < 3:
        sys.exit("使い方: python3 ml/soft_axis.py <単勝オッズを頭数ぶん>")
    r = recommend(list(range(1, len(odds) + 1)), odds)
    print(f"軸 {r['axis']}番（{min(odds):.1f}倍）")
    print(f"軸の3着以内確率 {r['q_axis']*100:.1f}%  →  複勝の期待払戻 {r['e_axis']:.0f}円")
    if r["buy"]:
        print(f"★買う（裾{int(r['tier']*100)}%）  三連複 {r['sanrenpuku']}（1点100円）")
    else:
        near = "" if r["tier"] is None else f"／緩い基準なら裾{int(r['tier']*100)}%には入る"
        print(f"見送り（買う基準は{r['buy_threshold']:.0f}円以下{near}）")


if __name__ == "__main__":
    main()
