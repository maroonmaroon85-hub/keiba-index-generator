"""
netkeiba経由のデータを、既存アーカイブの**血統登録番号**に名寄せする。

**なぜ必要か**: 既存の `*.CSV`（TARGET由来・587,077行）は馬の一意キーが**血統登録番号**（`23102282`）だが、
netkeibaは独自の `horse_id`（`2024108137`）で、**同じ馬が別IDになる**。
このままだと2つのデータが繋がらず、`n_prior` や `avg3_fin` など**近走ベースの特徴が作れない**
（42特徴の大半がこれに依存する）。

**名寄せキーの選定（実測）**: アーカイブ全体で同じキーに複数の登録番号が割り当たる率
| キー | ユニーク | 衝突 |
|---|---|---|
| 馬名のみ | 69,752 | 537件 (0.770%) |
| 馬名＋性別 | 72,661 | 366件 (0.504%) |
| **馬名＋性別＋生年** | **73,027** | **0件 (0.000%)** |
→ **馬名＋性別＋生年**を採用。生年は「レース年 − 馬齢」で両側から計算できる。

アーカイブに居ない馬（新馬・未出走から復帰など。実測で7/26は6.7%）は netkeiba の horse_id を
そのまま残す。桁数が違う（8桁 vs 10桁）ので既存の登録番号と衝突しない。

実行: python3 ml/nk_link.py [対象glob(既定 data/nk/DSnk*.CSV)]
"""
import csv
import glob
import sys
import warnings

import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, "ml")
import features as F


def build_map():
    """(馬名, 性別, 生年) → 血統登録番号。アーカイブ全体から作る。"""
    raw = F.load_files()
    raw = raw[raw[40].str.len() > 2]
    year = pd.to_numeric("20" + raw[0].str.zfill(2), errors="coerce")
    birth = year - pd.to_numeric(raw[15], errors="coerce")
    key = (raw[13].str.strip() + "|" + raw[14].str.strip() + "|"
           + birth.astype("Int64").astype(str))
    m = pd.Series(raw[37].str.strip().to_numpy(), index=key.to_numpy())
    return m[~m.index.duplicated()].to_dict()


def main():
    pat = sys.argv[1] if len(sys.argv) > 1 else "data/nk/DSnk*.CSV"
    files = sorted(glob.glob(pat))
    if not files:
        sys.exit(f"{pat} に対象がありません")
    mp = build_map()
    print(f"アーカイブの名寄せ表: {len(mp):,}頭")
    for p in files:
        rows = list(csv.reader(open(p, encoding="shift_jis", errors="replace")))
        hit = miss = 0
        for r in rows:
            if len(r) < 41 or not r[37]:
                continue
            try:
                birth = int("20" + r[0].zfill(2)) - int(r[15])
            except ValueError:
                continue
            reg = mp.get(f"{r[13].strip()}|{r[14].strip()}|{birth}")
            if reg:
                r[37] = reg
                hit += 1
            else:
                miss += 1
        with open(p, "w", encoding="shift_jis", errors="replace", newline="") as fh:
            csv.writer(fh).writerows(rows)
        n = hit + miss
        print(f"  {p}: {hit}/{n} 名寄せ成功（{hit/n*100:.1f}%）"
              f" / 未登録 {miss}頭はnetkeibaのIDのまま")


if __name__ == "__main__":
    main()
