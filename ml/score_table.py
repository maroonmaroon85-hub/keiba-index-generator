"""
各馬のスコア表の**列定義と描画**。ここ1か所だけに置く。

★列はユーザー指定の10列で固定（省略・並べ替えをしないこと）:
　　順 / 馬番 / 馬名 / 枠 / 複勝圏内確率 / レース内シェア / 市場 / 差 / 単勝 / 人気

  複勝圏内確率 … モデルの生値（3着以内に入る確率）
  レース内シェア … そのレースの掲載馬で合計1に正規化した値
  市場 … 単勝オッズ由来の含意確率（1/オッズ を**同じ集合で**正規化）
  差 … レース内シェア − 市場（＋なら市場より高く評価している）
  ★人気 … 単勝オッズの昇順の順位。**同オッズは同順位**（実際の人気表記と同じ）。
　　　　　 ⚠**市場の列と情報は同じ**（オッズの単調変換）。**並びを目で追うため**に置いている。

`ml/predict_nk.py`（コンソール）と `ml/reco_table.py`（markdown）の両方がこれを使う。
★**標準ライブラリだけ**にしておく（reco_table を pandas/lightgbm の無い環境でも動かすため）。
"""
import unicodedata

COLS = [("順", 4), ("馬番", 6), ("馬名", 20), ("枠", 4),
        ("複勝圏内確率", 14), ("レース内シェア", 16),
        ("市場", 8), ("差", 8), ("単勝", 8), ("人気", 6)]
HEADS = [t for t, _ in COLS]


def dispw(s):
    """全角=2で数えた表示幅。`:>7` は文字数で数えるため、全角混じりだと列がずれる。"""
    return sum(2 if unicodedata.east_asian_width(c) in "WFA" else 1 for c in s)


def cell(s, w, left=False):
    # 左寄せ列（馬名）は先頭に1つ空ける。右寄せ列の直後だと数字とくっついて読めないため。
    w -= 1 if left else 0
    while dispw(s) > w - 1:          # 長すぎる馬名は列を壊さないよう切る
        s = s[:-1]
    pad = " " * max(0, w - dispw(s))
    return (" " + s + pad) if left else (pad + s)


def values(rank, umaban, name, waku, p, share, mkt, odds, pop):
    """10列ぶんの文字列。`p`/`share`/`mkt` は0〜1、`odds` は単勝倍率、`pop` は人気。

    `pop` が 0/None（オッズ不明）のときは「-」を出す。**0位と書かない**。
    """
    return [str(rank), str(umaban), name, str(waku),
            f"{p*100:.1f}%", f"{share*100:.1f}%", f"{mkt*100:.1f}%",
            f"{(share-mkt)*100:+.1f}", f"{odds:.1f}",
            f"{pop}" if pop else "-"]


def header(indent=""):
    return indent + "".join(cell(t, w, left=(t == "馬名")) for t, w in COLS)


def row(*args, indent=""):
    return indent + "".join(cell(v, w, left=(t == "馬名"))
                            for v, (t, w) in zip(values(*args), COLS))


def md_header():
    return ("| " + " | ".join(HEADS) + " |\n"
            "|" + "|".join(["---:"] * len(COLS)) + "|")


def md_row(*args):
    return "| " + " | ".join(values(*args)) + " |"


def pop_ranks(odds_by_umaban):
    """{馬番: 単勝オッズ} → {馬番: 人気}。

    ★**同オッズは同順位**にする（JRAの人気表記と同じ）。**次の順位は飛ばす**
    （1.8/1.8/3.0 → 1人気/1人気/3人気）。オッズが無い馬は 0（表では「-」）。
    """
    vs = [o for o in odds_by_umaban.values() if o and o > 0]
    return {u: (1 + sum(1 for x in vs if x < o)) if (o and o > 0) else 0
            for u, o in odds_by_umaban.items()}


def market_shares(scores):
    """{馬番: 市場含意確率}。掲載されている馬だけで正規化する（表のシェアと同じ集合）。"""
    tot = sum(1.0 / s["odds"] for s in scores if s.get("odds"))
    return {s["umaban"]: ((1.0 / s["odds"]) / tot if s.get("odds") and tot else 0.0)
            for s in scores}
