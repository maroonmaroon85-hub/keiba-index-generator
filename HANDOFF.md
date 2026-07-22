# 引き継ぎ（keiba-index-generator）

次セッションはこの文書を最初に読めば再開できる。詳細仕様は `keibaspec/`、CSV列対応は `data/sample/schema.md`。

## リポジトリ / ブランチ
- リポ: `maroonmaroon85-hub/keiba-index-generator`（game-zatsugaku-pipeline とは無関係の別プロジェクト）
- 作業ブランチ: `claude/new-session-h4ydgl`（全成果をここにpush済み）
- 目的: JRA-VAN DataLab のデータ → 予想指数表(HTML+スコアJSON) → 最終的に期待値アプリ keiba-ev へ接続し「印」でなく「買い目」を出す

## 進捗（Phase 1〜3 完了、実データ稼働）
- **Phase 1**: ダミーデータで指数表ジェネレーター。`npm run generate -- --input data/sample/race-dummy.json --pace H --condition 良`
- **Phase 2**: TARGET実CSV接続。`npm run generate -- --shutuba <出馬表.csv> --seiseki <成績.csv> --course 函館 --surface ダ --distance 1700 --date 2026-07-19 --pace M --condition 稍`
- **Phase 3**: バックテスト基盤。`npm run backtest -- --input <成績CSV> [--input ...複数可] --odds-col 49 --min-horses 6`
- **Phase 4（未着手）**: keiba-ev 連携（EV=確率×オッズ、買い目出力）。ユーザーの keiba-ev.jsx を受領してから。
- **Phase 5（構想）**: scoring層をML化。PreRace/PostRaceの型分離とバックテスト基盤はそのまま流用する前提。

## 設計の要（崩さない）
- **リーク防止の型分離**: `src/model/pre-race.ts`(発走前) と `post-race.ts`(着順・払戻)。scoring/rules は PreRaceData しか受け取れない。PostRace はバックテストの答え合わせ専用。
- **scoring interface**: `scoreRace(PreRaceData, config) → ScoredRace`（各馬 winProb/placeProb）。中身をルール合算→ML に差し替え可能に保つ。
- **設定は全て `config.json`**（weight/閾値/色/softmax温度）。src に直書きしない。

## 実測で得た知見（6開催日=627レース/4424頭で検証済み）
- 総合評価ランクは機能（複勝率 S+39%→C13% 単調）。印も ◎37.5%>○>▲>△>×10.5% と正しい順序。
- **効かないフラグを特定して調整済み**: `人気落ち◎`・`叩き2◎` は複勝率が全体基準以下 → **config で weight=0 に**（S+の勝率が19→22%改善）。
- **win_prob が過信**だったため **softmax温度 12→20 に暫定調整**。指数が実オッズほど鋭くないため完全較正はデータ蓄積＋weight改善で詰める。
- これらは1〜6開催日ぶんの暫定調整。**データを増やして再検証する前提**（16→627レースで印別・較正が本番化したのが実証済み）。

## データ取得（人間作業・環境は構築済み）
- **環境**: Mac(Intel 2017) に VMware Fusion 13 + Windows 11 Pro。JV-Link + TARGET frontier JV 導入済み。DataLab 無料体験中。
- ⏰ **無料期間 〜2026/08/03頃**。この間に過去データを取れるだけ一括出力しておく（期間後も検証・学習は継続可）。
- **共有フォルダ**: Windows `Z:` = Mac `~/keiba-data`。TARGETのCSV出力先。
- **CSVをClaudeに渡す方法**: Mac側 `~/keiba-data` の CSV を GitHub にアップ →
  `https://github.com/maroonmaroon85-hub/keiba-index-generator/upload/claude/new-session-h4ydgl/data/sample`
  にドラッグ→Commit。Claude が `git pull` して取り込む。（リモート実行環境はネット制限でDL不可）

### TARGETからの一括CSV出力手順（バックテスト用データの増やし方）
1. 出馬表レース選択画面で、上のレース一覧をクリック（フォーカス）→ **F8「一括出力」**
2. 中央列下 `[設定] 過去走数の変更` を大きく（60戦など）してあると1ファイルが厚い
3. 左列最下部 **「全馬成績CSV形式(S)」→「4: フルセット+単オッズ」**
4. 保存先 `Z:`、ファイル名 `YYYYMMDD_all.csv`（開催日ごと）→ アップ
5. **開催日を10〜15日ぶん散らして出せば十分**（活躍馬は複数日に登場し重複して埋まる）。100日は不要。

## 実CSVの形式（重要・パーサはこれ前提）
すべて **Shift_JIS**。詳細は `data/sample/schema.md`。
- **出馬表・画面イメージCSV**（`--shutuba`）: 列名付き25列。枠1/B(ブリンカー)2/馬番3/馬名8/性別10/年齢11/騎手13/斤量14/単勝16/調教師20/生産者23。
- **全馬成績フルセット+単オッズ**（`--seiseki` / backtest `--input`）: 列名なし52列。日付1-3/開催場5/馬場種別10/距離12/馬名14/斤量18/頭数19/馬番20/着順21/着差24/通過29-32/血統登録番号38/**col41=レースID+馬番(末尾2桁が馬番)**/生産者43/父44/母父46。**単オッズは col49**（`--odds-col 49`）。
- 血統: 父名/母父名→系統は `src/pedigree/sire-lines.ts`（仮マスタ。Phase3で精緻化）。

## コード構成
```
src/model/      PreRace/PostRace 型
src/parser/     csv.ts(共通Shift_JIS読込) / json.ts(ダミー) / target-csv.ts(実CSV→PreRace)
src/rules/      条件フラグ10個（1ルール1ファイル）
src/scoring/    score.ts(指数計算) / style.ts(脚質) / types.ts
src/pedigree/   master.ts(系統マスタ・適性) / sire-lines.ts(父名→系統)
src/render/     html.ts(12列HTML) / labels.ts
src/backtest/   dataset.ts(成績CSV→レース群) / metrics.ts / cli.ts
src/cli.ts      生成CLI（JSON/実CSV両対応）
config.json     weight/閾値/温度  ／ schema/score-export.json 出力スキーマ
data/sample/    実CSV各種（20260704〜19_all.csv 等）
```

## 次にやること（優先順）
1. **開催日データを増やす**（無料期間中）→ 全部結合してバックテスト（`--input` 複数）。印別・キャリブレーション・weightを本格調整。
2. weight調整の続き: `人気落ち◎` は符号反転（minus化）も検討余地。softmax温度の再詰め。
3. **Phase 4**: keiba-ev.jsx 受領 → スコアJSON接続 → EV>1.0 の買い目出力 → バックテストにEV購入シミュ追加。
4. 注意: 生成物を外部公開する構想があるなら JRA-VAN の外部提供規約確認が先（実装と別件）。

## トークン節約（CLAUDE.md方針）
1タスク1セッション。長い作業は /compact。大きいファイルは必要な行だけ読む。セッション間はこの md 経由で引き継ぐ。
