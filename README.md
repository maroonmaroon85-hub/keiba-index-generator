# keiba-index-generator

JRA-VAN DataLab のデータ（Windows VM上の TARGET frontierJV が出力する CSV）から、
レースごとの**予想指数表（HTML）とスコア JSON** を自動生成する CLI ツール。
最終的に期待値アプリ `keiba-ev` に接続し、出口を「印」ではなく「買い目」にする。

- 取得層（Windows/TARGET でのCSV出力）は人間作業。**本リポの実装対象は Mac 側の生成層のみ。**
- 詳細仕様はフェーズ別指示書（`keibaspec/`）に基づく。**1フェーズずつ**実装し、完了条件でレビューする方針。

## 現在の状態: Phase 1 完了（ダミーデータ）

ダミーデータ 1レース分（函館6R ダ1700m 14頭）から HTML 指数表とスコア JSON を生成できる。
実データ（TARGET CSV）接続は Phase 2 以降。

## セットアップ

```bash
npm install
```

## 使い方

```bash
npm run generate -- --input data/sample/race-dummy.json --pace H --condition 良
```

- `--input`   PreRaceData 形式の JSON（Phase 1 はダミー。Phase 2 で CSV パーサに置換）
- `--pace`    想定ペース `S|M|H`（**自動推定しない手入力**。JSON の値を上書き）
- `--condition` 馬場 `良|稍|重|不`（同上）
- `--out`     出力ディレクトリ（既定 `out/`）
- `--config`  設定ファイル（既定 `config.json`）

出力:
- `out/<race_id>.html` … 12列の指数表（外部依存なし・インラインCSS）
- `out/<race_id>.score.json` … keiba-ev 用スコア JSON（`schema/score-export.json` 準拠）

型チェック: `npm run typecheck`

## 設計の大原則

### 1. リーク防止の型分離（最重要・後から直せない）
- `src/model/pre-race.ts` … **PreRaceData**（発走前に見える情報）
- `src/model/post-race.ts` … **PostRaceData**（レース後に確定＝着順・払戻等）

`scoring` / `rules` は **PreRaceData しか受け取れないシグネチャ**。PostRaceData は
Phase 3 バックテストの答え合わせ専用で、指数計算部には型として渡せない。

### 2. scoring 層の差し替え可能性（Phase 5 ML化への含み）
`scoreRace(PreRaceData, config) → ScoredRace`（各馬 `winProb`/`placeProb` を持つ）という
interface を固定し、内部実装（現状はルール合算）を将来 LightGBM 等に差し替え可能にしている。
型分離とバックテスト基盤は ML 移行後もそのまま特徴量/ラベル分離・評価基盤として使う。

## ディレクトリ構成

```
src/
  model/     PreRaceData / PostRaceData の型（リーク防止の分離）
  parser/    入力 → 内部モデル（Phase 1: JSON。Phase 2: TARGET CSV 3段構成）
  rules/     条件フラグ 1ルール1ファイル（plus/minus, condition）
  scoring/   スコア計算・S+〜C・印付与・softmax確率（PreRace→win/place_prob）
  pedigree/  血統系統マスタ・適性テーブル（仮20系統）
  render/    HTML生成（12列・インラインCSS）
  export.ts  スコアJSON整形
  cli.ts     エントリポイント
schema/      score-export.json（keiba-ev が読む正のスキーマ）
config.json  weight / 閾値 / 色 / softmax温度（すべて外出し。Phase 3 で調整）
data/sample/ ダミーデータ
out/          生成物（gitignore）
```

## 条件フラグ（初期10ルール・weightは仮置き）

| 符号 | ルール | 判定 |
|---|---|---|
| − | 延長△ | 前走比で距離が一定以上延長 |
| − | 初ダ△ | 今走ダートで近走にダート実績なし |
| − | 増量△ | 前走比で斤量が一定以上増加 |
| − | 後方△ | 追込脚質かつ想定ペースがハイでない（前有利） |
| − | ローテ△ | 前走間隔が中1週未満 or 半年以上 |
| ＋ | 同2◎ | 前走と同距離（同距離2走目） |
| ＋ | 内枠先行◎ | 内枠かつ先行脚質 |
| ＋ | 減量◎ | 減量騎手騎乗 |
| ＋ | 叩き2◎ | 休み明け2走目 |
| ＋ | 人気落ち◎ | 近走好走かつ今回人気薄（妙味） |

weight・閾値・血統適性はすべて仮置き。**妥当性は Phase 3 のバックテストで検証・調整する。**

## スコアリング

```
score = 基礎点(近走着順の加重平均、直近ほど重い)
      + Σ(plusフラグ weight) − Σ(minusフラグ weight)
      + 調教評価ボーナス
      + 血統適性(父系・母父系 × コース種別 × 距離帯)
```
- 総合評価: `config.rank.thresholds` の閾値で S+/S/A/B/C
- 印: スコア上位から ◎○▲△、下位かつ明確に劣る馬に ×
- `win_prob`: レース内スコアの softmax（温度は config）
- `place_prob`: win_prob の単調変換（初期は簡易推定、Phase 3 でキャリブレーション）

## フェーズ計画

- **Phase 1（完了）**: ダミーデータで生成器を一気通貫
- **Phase 2**: TARGET 実 CSV の接続（サンプル CSV 受領後・複数CSV結合の3段パーサ）
- **Phase 3**: バックテストと weight 調整（過去一括 CSV 受領後・PostRaceData を初使用）
- **Phase 4**: keiba-ev 連携（EV = 確率 × オッズ、EV>1.0 の買い目提示）
- **Phase 5（構想）**: scoring 層を ML（LightGBM 等）へ差し替え
