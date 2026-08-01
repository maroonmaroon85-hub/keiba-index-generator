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
実測: 7/19は492頭中492頭(100%)、7/26は449頭中413頭(92%)が接続。
残りは新馬・未出走馬で**本来アーカイブに居ない馬**。netkeibaのIDをそのまま残す
（桁数が違うので既存の登録番号と衝突しない）。

★**標準ライブラリだけで動く**（pandas不要）。取得〜名寄せまでを手元のMacで完結させるため。
　pandasを入れていない環境で `ModuleNotFoundError` になって止まった実績があるので、依存を足さないこと。

★実行順: `results` → **`pedigree`** → **`nk_link`**。
　`pedigree` は netkeiba の horse_id をキーに父・母父を書き戻すので、
　**先に名寄せすると引けなくなる**（父・母父が空のままになる）。

実行: python3 ml/nk_link.py [対象glob(既定 data/nk/DSnk*.CSV)]
"""
import csv
import glob
import sys


def build_map(pattern="*.CSV"):
    """(馬名, 性別, 生年) → 血統登録番号。アーカイブ全体から作る。"""
    mp = {}
    files = sorted(glob.glob(pattern))
    for i, p in enumerate(files, 1):
        with open(p, encoding="shift_jis", errors="replace", newline="") as fh:
            for r in csv.reader(fh):
                # to_model と同じ足切り: col40(レースID+馬番)が2文字超の行だけが成績行
                if len(r) < 41 or len(r[40].strip()) <= 2 or not r[37].strip():
                    continue
                try:
                    birth = int("20" + r[0].strip().zfill(2)) - int(r[15])
                except ValueError:
                    continue
                mp.setdefault(f"{r[13].strip()}|{r[14].strip()}|{birth}", r[37].strip())
        if i % 200 == 0:
            print(f"  読込 {i}/{len(files)}ファイル…")
    return mp


def main():
    pat = sys.argv[1] if len(sys.argv) > 1 else "data/nk/DSnk*.CSV"
    files = sorted(glob.glob(pat))
    if not files:
        sys.exit(f"{pat} に対象がありません")
    print("アーカイブを読み込み中（30秒〜1分）…")
    mp = build_map()
    print(f"名寄せ表: {len(mp):,}頭\n")
    for p in files:
        with open(p, encoding="shift_jis", errors="replace", newline="") as fh:
            rows = list(csv.reader(fh))
        hit = miss = skip = 0
        for r in rows:
            if len(r) < 41 or not r[37].strip():
                continue
            if len(r[37].strip()) == 8:      # 既に名寄せ済み（登録番号は8桁）
                skip += 1
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
        done = f"{hit}/{n} 名寄せ（{hit/n*100:.1f}%）" if n else "新たに名寄せする行なし"
        print(f"  {p}: {done}"
              + (f" / 未登録 {miss}頭はnetkeibaのIDのまま" if miss else "")
              + (f" / 済 {skip}行" if skip else ""))


if __name__ == "__main__":
    main()
