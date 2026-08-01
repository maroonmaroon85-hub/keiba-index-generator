"""
netkeiba経由のデータを、TARGET由来のアーカイブと**1行ずつ照合**する。

(70)でパイプラインは通ったが、確かめたのは単勝オッズの一致だけ（12頭・1レース）だった。
着差はタイムから作り直し、クラスは別経路から取り、賞金は0埋めし、馬IDは名寄せしている。
**どれも「作り直したもの」なので、実物と突き合わせないと信用できない**。

やり方: アーカイブに既に入っている開催日を netkeiba からも取得し、
`raceid + 馬番` で結合して各列を比較する。TARGET側の追加出力は要らない。

判定:
  ・完全一致すべき列（着順・馬名・性別・年齢・斤量・頭数・距離・馬場・馬体重・通過順）
  ・数値で許容差を見る列（着差=秒・上がり・単勝オッズ・賞金）
  ・意味が一致すればよい列（raceclass・父・母父・騎手・調教師）

実行: python3 ml/nk_verify.py [対象日(既定 20260719)]
"""
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, "ml")
import features as F


def load(path):
    raw = pd.read_csv(path, header=None, encoding="shift_jis",
                      encoding_errors="replace", dtype=str, keep_default_na=False)
    return F.to_model(raw)


def main():
    ymd = sys.argv[1] if len(sys.argv) > 1 else "20260719"
    day = pd.Timestamp(f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:8]}")
    nk = load(f"data/nk/DSnk{ymd}.CSV")
    nk = nk[nk["date"] == day]
    arc = F.to_model(F.load_files())
    arc = arc[arc["date"] == day]
    print(f"{day.date()}  netkeiba {len(nk)}行/{nk['raceid'].nunique()}R  "
          f"アーカイブ {len(arc)}行/{arc['raceid'].nunique()}R")
    if arc.empty:
        sys.exit("アーカイブ側にこの日が無い。別の日を指定すること。")

    key = ["raceid", "umaban"]
    m = arc.merge(nk, on=key, suffixes=("_a", "_n"))
    print(f"結合できた行: {len(m)}（アーカイブ側の {len(m)/len(arc)*100:.1f}%）\n")
    if m.empty:
        sys.exit("結合ゼロ。raceidの変換が誤っている可能性が高い。")

    # ★数値列を astype(str) で比べると int64 と float64 で "9" vs "9.0" になり全滅する。
    #   完全一致を見たい数値は許容差0で NUM 側に置く。
    EXACT = ["sex", "cond", "course"]
    NUM = {"finish": 0, "age": 0, "wtcarry": 0, "fieldsize": 0, "distance": 0,
           "surface": 0, "bodywt": 0, "raceclass": 0,
           "margin": 0.15, "agari": 0.05, "odds": 0.05, "prize": 1.0, "passavg": 0.01}
    NAME = ["jockey", "trainer", "sire", "damsire"]

    print(f"{'列':<12}{'一致':>8}{'比較数':>8}  {'ズレの例'}")
    bad = []
    for c in EXACT:
        a, b = m[f"{c}_a"], m[f"{c}_n"]
        ok = (a.astype(str) == b.astype(str)) | (a.isna() & b.isna())
        ex = m.loc[~ok, [f"{c}_a", f"{c}_n"]].head(2).to_numpy().tolist()
        print(f"{c:<12}{ok.mean()*100:>7.1f}%{len(a):>8}  {ex if len(ex) else ''}")
        if ok.mean() < 0.99:
            bad.append(c)
    for c, tol in NUM.items():
        a = pd.to_numeric(m[f"{c}_a"], errors="coerce")
        b = pd.to_numeric(m[f"{c}_n"], errors="coerce")
        both = a.notna() & b.notna()
        ok = (a - b).abs() <= tol
        ok = ok | (a.isna() & b.isna())
        ex = m.loc[both & ~ok, [f"{c}_a", f"{c}_n"]].head(2).to_numpy().tolist()
        print(f"{c:<12}{ok.mean()*100:>7.1f}%{int(both.sum()):>8}  (許容±{tol}) {ex if len(ex) else ''}")
        if ok.mean() < 0.99:
            bad.append(c)
    for c in NAME:
        a, b = m[f"{c}_a"].astype(str).str.strip(), m[f"{c}_n"].astype(str).str.strip()
        ok = a == b
        ex = m.loc[~ok, [f"{c}_a", f"{c}_n"]].head(3).to_numpy().tolist()
        print(f"{c:<12}{ok.mean()*100:>7.1f}%{len(a):>8}  {ex if len(ex) else ''}")
        if ok.mean() < 0.99:
            bad.append(c)

    # 名寄せの検算: 同じ馬に同じ登録番号が付いているか
    same = (m["horse_a"].astype(str) == m["horse_n"].astype(str))
    print(f"\n★名寄せ（血統登録番号の一致）: {same.mean()*100:.1f}%"
          f"（{int(same.sum())}/{len(same)}）")
    print(f"★総合: {len(bad)}列に1%超のズレ" + (f" → {bad}" if bad else " → 問題なし"))


if __name__ == "__main__":
    main()
