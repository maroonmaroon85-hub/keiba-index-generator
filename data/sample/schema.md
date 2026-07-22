# TARGET CSV → 内部モデル 対応表（Phase 2）

TARGET frontier JV からエクスポートした実CSVと、内部モデル `PreRaceData` の対応。
サンプル: 2026/07/19 函館10R 駒場特別・2勝 ダ1700m 14頭。

## ファイル構成（TARGETの出力機能）
| ファイル | TARGETの出力元 | 内容 | 文字コード |
|---|---|---|---|
| `hakodate10R_shutuba2.csv` | 出馬表 → ファイル出力 → 「出馬表・画面イメージ出力(CSV形式)」＋項目名付加 | 当該レースの出走馬一覧（発走前情報） | Shift_JIS |
| `hakodate10R_seiseki.csv` | 出馬表 → ファイル出力 → 「S: 全馬全成績CSV形式 → 2: 成績フルセットデータ」 | 各馬の近走成績＋血統・生産者等 | Shift_JIS |

- **結合キー**: 馬名（`shutuba2.馬名` ⇔ `seiseki.馬名`）。レース内で一意。
  - ※ `seiseki` には**血統登録番号(col38)**があるが `shutuba2` には無いため、Phase 2 の結合は馬名で行う。
    血統登録番号は将来の正確な結合・馬個体識別に使えるので保持する。
- **レースヘッダ情報**（course/surface/distance/pace/condition/date）は shutuba2/seiseki のどちらにも
  ヘッダ行が無いため、**CLI引数で受け取る**（pace・condition は元々「手入力の割り切り」項目）。

## shutuba2.csv（列名付き・カンマ区切り・25列）
1列目から: `枠番, B, 番, 印, M2, M3, M4, 馬名, C, 性別, 年齢, 替, 騎手, 斤量, 減M, 単勝, 馬体重, 増減, 所属, 調教師, 馬記号, 馬主, 生産者, 毛色, 誕生`

| CSV列(1始まり) | 列名 | 内部モデル(PreRaceHorse) | 備考 |
|---|---|---|---|
| 1 | 枠番 | `frame` | |
| 2 | B | `blinker` | 値が "B" なら true |
| 3 | 番 | `number` | 馬番 |
| 8 | 馬名 | `name` | 結合キー |
| 10 | 性別 | `sex` | 牡/牝/セ |
| 11 | 年齢 | `age` | |
| 13 | 騎手 | `jockey.name` | |
| 14 | 斤量 | `weightCarry` | |
| 15 | 減M | `jockey.reductionMark/Kg` | 減量記号(☆△▲)。空なら減量なし |
| 16 | 単勝 | `odds` | 想定オッズ。予想人気の算出元 |
| 17 | 馬体重 | (未使用) | |
| 20 | 調教師 | `trainer` | |
| 23 | 生産者 | `producer` | |

## seiseki.csv（列名なし・カンマ区切り・52列・1行=1過去走・日付降順）
先頭が最新走。1頭あたり複数行。

| CSV列(1始まり) | 内容 | 内部モデル(PastRun/静的属性) | 備考 |
|---|---|---|---|
| 1-3 | 年,月,日 | （間隔計算に使用） | 例 26,06,28 = 2026/06/28 |
| 5 | 開催場 | `PastRun.course` | 函館 等 |
| 10 | 馬場種別 | `PastRun.surface` | 芝/ダ |
| 12 | 距離 | `PastRun.distance` | |
| 14 | 馬名 | 結合キー | 末尾空白はtrim |
| 18 | 斤量 | `prevWeightCarry`(最新走) | 増量△判定に使用 |
| 19 | 頭数 | `PastRun.fieldSize` | |
| 21 | 着順 | `PastRun.finish` | |
| 24 | 着差 | `PastRun.margin` | 秒。圧勝◎判定に使用 |
| 25 | 人気 | (未使用) | 過去走の人気 |
| 29-32 | 通過順位1-4 | `PastRun.passing` | 0は未計測として除外 |
| 38 | 血統登録番号 | （保持・将来の結合キー） | 8桁 |
| 43 | 生産者 | `producer` | shutuba2と重複、seiseki優先可 |
| 44 | 父 | `sireLine`（父名→系統に変換） | sire-lines.ts で系統へ |
| 46 | 母父 | `damSireLine`（母父名→系統に変換） | |
| 48 | 生年月日 | (未使用) | 例 220511 |

## 内部モデルへ供給できない項目（Phase 2時点の割り切り）
- `training`（調教評価・調子）: この2CSVには無い。**中立(evaluation=B, trend=flat)** を既定にする。
  必要なら TARGET「調教一覧」CSVを別途出力して補完（将来対応）。
- `prevWeightCarry`: 最新走(seiseki先頭行)の斤量を使用。
- `weeksSinceLastRun` / `weeksBeforeLastRun`: 走破日付(seiseki col1-3)と当該レース日(CLI --date)から算出。
- レースヘッダ(course/surface/distance/pace/condition/date): CLI引数（pace/conditionは手入力項目）。

## 血統系統マッピング
父名・母父名 → 系統名 の対応は `src/pedigree/sire-lines.ts`（仮）。
未知の馬名は「その他」（中立適性）にフォールバック。系統マスタの精緻化は Phase 3 以降。
