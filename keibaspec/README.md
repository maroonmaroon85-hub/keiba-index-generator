# 競馬指数表ジェネレーター 指示書一式（README / 目次）

JRA-VAN DataLabのデータから予想指数表（HTML＋スコアJSON）を自動生成し、最終的に期待値アプリ（keiba-ev）へ接続するプロジェクトの指示書。

## ファイル一覧と読む順番

| # | ファイル | 内容 | 実行者 |
|---|---------|------|--------|
| 1 | `00-overview.md` | 全体像・共通仕様・進行ルール。**最初に必ず読む。全フェーズで常に参照** | Claude Code |
| 2 | `90-env-setup.md` | Mac上のWindows VM構築チェックリスト。**Phase 1と並行して人間が進める** | 人間 |
| 3 | `10-phase1-generator.md` | Phase 1: ジェネレーター本体をダミーデータで完成させる | Claude Code |
| 4 | `20-phase2-parser.md` | Phase 2: TARGET実CSVの接続（サンプルCSV受領後） | Claude Code |
| 5 | `30-phase3-backtest.md` | Phase 3: バックテストとweight調整（過去データ受領後） | Claude Code |
| 6 | `40-phase4-ev.md` | Phase 4: 期待値アプリ連携、買い目出力 | Claude Code |

## 進行フロー

```
[人間]        90: VM構築 ──▶ TARGETセットアップ ──▶ サンプルCSV出力 ──▶ 過去データ一括出力
                │（並行）                │                    │
[Claude Code] 10: Phase 1 ──レビュー──▶ 20: Phase 2 ──▶ 30: Phase 3 ──▶ 40: Phase 4
              （ダミーデータ）           （実CSV接続）      （バックテスト）   （EV連携）
```

## 進め方のルール
- Claude Codeには **1フェーズずつ** 渡す（例: 「00を読んだ上で10-phase1を実行して」）。全フェーズ一括で投げない
- 各フェーズの「完了条件」で停止させ、レビューしてから次へ
- Phase 2以降は入力データ（実CSV、過去データ、keiba-ev.jsx）が揃ってから着手

## Claude Codeに一緒に渡すもの
- [ ] このフォルダ一式
- [ ] 参考画像（函館6R指数表のスクリーンショット）… HTMLの見た目の正
- [ ] （Phase 2時）TARGETからエクスポートした1レース分のサンプルCSV
- [ ] （Phase 3時）過去数ヶ月分の一括CSV（成績・払戻含む）
- [ ] （Phase 4時）keiba-ev.jsx のソース

## 期限
DataLab無料期間: **2026/07/04起算 → 2026/08/03頃まで**（詳細は90を参照）
