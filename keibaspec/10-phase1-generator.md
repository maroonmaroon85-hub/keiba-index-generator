# Phase 1: 指数表ジェネレーター本体（10-phase1-generator）

前提: `00-overview.md` を先に読むこと。このフェーズは**実データ不要**。環境構築（90-env-setup）と並行して進められる。

## ゴール
手書きのダミーデータ（JSON 1レース分、14頭程度）を入力に、HTML指数表とスコアJSONを出力するパイプラインを完成させる。

## タスク
1. プロジェクト雛形（TypeScript, 00-overviewのディレクトリ構成）
2. 内部モデル定義: PreRaceData / PostRaceData の型分離（00-overview参照）
3. ダミーデータ作成: `data/sample/race-dummy.json`
   - 参考画像（函館6R ダ1700m 14頭）の内容をベースに手で起こしてよい
   - 各馬に近走3走分の履歴を持たせる
4. rules/ に条件フラグ10個を実装（00-overviewのリスト）
5. scoring/ にスコア計算・S+〜C変換・印付与・win_prob/place_prob変換を実装
6. render/ にHTML生成を実装。参考画像の密度・配色に寄せる
7. スコアJSON出力と `schema/score-export.json`
8. CLI: `npm run generate -- --input data/sample/race-dummy.json --pace H --condition 良`

## 注意
- 血統系統マスタ（父系・母父系→系統→色）は主要どころ20系統程度の仮マスタでよい
- weight・閾値は仮置きしてconfig.jsonへ。妥当性はPhase 3で検証するので凝らない
- HTMLの見た目確認用に、生成したファイルパスを出力すること

## 完了条件
- ダミーデータからHTMLとJSONが生成され、HTML表に全12列が揃っている
- 印と総合評価がスコア順に整合している
- PostRaceDataを一切参照せずに指数が計算されている（型で保証）
- ここで停止してレビューを受ける。Phase 2に進まない
