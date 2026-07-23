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
- **Phase 4（実装済み・単勝/複勝）**: EV=想定確率×オッズ で買い目出力＋バックテストにEV購入シミュ追加。keiba-ev.jsx は無しでツール内蔵。`src/ev/ev.ts`。
  - 生成時: `npm run generate ...` の末尾に「買い目（EV≥threshold）」を表示。単勝は出馬表CSVの単勝オッズ使用。複勝は複勝オッズ受領後に対応（現状 単勝のみ稼働）。
  - `config.ev`: `threshold`(1.0), `evBands`, `contenderOnly`(true=モデルが平均以上(win_prob≥1/頭数)と見た馬のみ買い。人気薄×高オッズでEVだけ跳ねる馬を除外)。
  - **実測（1年）**: 単勝EV≥1戦略の回収率は **81〜86%**（contenderOnly=trueで81%）＝**まだ非利益**。ルールベース＋仮weightでは市場(控除率約20%)を越えられない、という正直な現状。買い目自体は妥当な人気馬に出る。
  - 収益化には: 特徴量/weightの改善、Phase 5(ML化)。
  - **連系券種(馬連/馬単/ワイド/三連複/三連単)**: `src/ev/harville.ts` で win_prob からHarville式で組み合わせ確率を算出。生成時に確率上位候補を表示（オッズ照合前）。**単体確率を渡すI/Fなのでルール/ML両方で使える**。EV判定・バックテスト回収率には**連系のオッズ/払戻データが別途必要**（現状のCSVは単勝オッズcol49のみ）。Harvilleは人気馬の2・3着過大評価の癖あり、精緻化はPL等で後日。

## 重要な負の知見（過信を防ぐため必読）
- **単調較正(isotonic)は効果なし**: `npm run calibrate`（train前半/test後半で分離学習・評価）で検証。temp=45で win_prob は既に十分較正されており、較正後EV回収率は 72.9→72.5%（改善なし）。較正は収益性のボトルネックではない。`src/backtest/isotonic.ts` `calibrate.ts` は分析用に残置（scoringには未統合＝効果ないため）。
- **アウトオブサンプルは約73%**: 全年で最適化したweightの**in-sample 84% は一部オーバーフィット**。test期間の正直な単勝EV回収率は**約73%**（非利益）。weightの微調整で越えるのは困難。
- **示唆**: 回収率100%超えはルールベースの微調整では届かない。**Phase 5(ML化)＝より強い特徴量/モデル**が本命の道。指数の「順位付け・印」は既に有用なので、そのまま予想補助としては使える。
- **Phase 5（実現可能性 実証済み・本実装は今後）**: scoring層をLightGBM等に差し替える。`ml/feasibility.py` で検証済み。
  - **結果（アウトオブサンプル28338頭・日付分割）**: ルールベースを全指標で上回る。top-pick複勝率 47→**56.2%**、勝率19→**26.9%**、**単勝EV≥1回収率 73→90.3%**、AUC 0.748。まだ非利益(100%未満)だが微調整の限界を大きく突破。
  - **最重要特徴量は 母父(damsire)→父(sire)**。血統がML最大の信号＝血統特徴の深掘りが有望。
  - 特徴量は「近走(prior runs)＋条件」のみ、**単勝オッズは特徴に入れない**（市場追随回避、EV評価にだけ使用）。実行: `pip install lightgbm scikit-learn pandas numpy` → `python3 ml/feasibility.py`（リポ直下のDS*.CSVを読む）。
  - **特徴量追加の実験結果（正直な頭打ち）**: 近走系(上がりcol33/賞金col37/着順安定度/季節)を足すと AUC 0.748→**0.751**と小幅改善。ただし血統のターゲットエンコーディング(sire/damsire×コース/距離)は**逆効果**（生カテゴリと冗長＋train/test分布シフト）→既定OFF(`python3 ml/feasibility.py te` で有効化可)。EV回収率は88〜90%でノイズ範囲、**特徴量積み増しは頭打ち**。次のボトルネックは**データ量(多年)/信号の質**（オッズ変動・騎手厩舎の調子・区間タイム等）。
  - **次の本実装**: ①データを多年に拡張（無料期間中に）②複勝/エキゾチック予測（複勝は当てやすい）③モデルを scoring interface(PreRaceData→win_prob)に統合（Python学習→予測JSONをTSが読む構成）。回収率100%超えが最終目標。指数の順位付け(top-pick複勝率56%)は既に実用域。

## 設計の要（崩さない）
- **リーク防止の型分離**: `src/model/pre-race.ts`(発走前) と `post-race.ts`(着順・払戻)。scoring/rules は PreRaceData しか受け取れない。PostRace はバックテストの答え合わせ専用。
- **scoring interface**: `scoreRace(PreRaceData, config) → ScoredRace`（各馬 winProb/placeProb）。中身をルール合算→ML に差し替え可能に保つ。
- **設定は全て `config.json`**（weight/閾値/色/softmax温度）。src に直書きしない。

## 実測で得た知見（約1年=98開催日 / 7,709レース / 77,522頭で検証済み）
- 総合評価ランクは機能（複勝率 **S+47%→C15%** 単調）。印も **◎47%>○41%>▲36%>△31%>×12%** と正しい順序。統計的に安定。
- フラグ調整（実測反映済み・config/コードに反映）:
  - `叩き2◎` は複勝率26.1%≒全体26.0% で無効 → **weight=0**。
  - `人気落ち` は複勝率17.7%≪全体26.0%（8,107サンプル）で**負のシグナル**と判明 → **`人気落ち△`(minus, weight4)に反転**。反転後 S+ 複勝率42.7→47.2%・勝率17.4→19.7%と上位ランク改善。
  - 効くフラグ: `内枠先行◎`(32%)・`同2◎`(29%)、マイナス材料は全部正しく機能（`後方△`17%等）。
  - **weight最適化（全年グリッド探索）**: フラグは効きが強く過小評価だったため**全フラグ weight ×2**（例 同2◎5→10, 後方△5→10）、基礎点の近走加重を `[0.5,0.3,0.2]→[0.45,0.35,0.2]`、血統は残す(0にすると悪化)。結果 EV回収率 81→**84%**、S+複勝47.5%。フラグ half は明確に悪化＝フラグが信号を持つ証拠。
- **win_prob 較正済み**: 温度を全年データのスイープで **12→45** に。予測≒実測（10-20%帯 予測13.6/実12.7、20-30%帯 23.2/22.5）。低確率帯はわずかに過信。EV(Phase4)に効く。`placeProb.multiplier` も 1.9→**2.4**（winProb→実複勝率の比が中位で約2.4）。温度は rank/印(相対順位)に影響しない＝ランク評価は不変。
- **バックテストの回し方**: `npm run backtest -- $(for f in DS*.CSV; do echo -n "--input $f "; done) --odds-col 49 --min-horses 6`（`--input`複数対応。raceId+馬番で重複除去）。

## データ取得（人間作業・環境は構築済み）
- **環境**: Mac(Intel 2017) に VMware Fusion 13 + Windows 11 Pro。JV-Link + TARGET frontier JV 導入済み。DataLab 無料体験中。
- ⏰ **無料期間 〜2026/08/03頃**。この間に過去データを取れるだけ一括出力しておく（期間後も検証・学習は継続可）。
- **共有フォルダ**: Windows `Z:` = Mac `~/keiba-data`。TARGETのCSV出力先。
- **取得済みデータ**: 2025/07〜2026/05 の約1年・98開催日を **リポ直下 `DS*.CSV`**（TARGET既定名 DSyymmdd.CSV、Shift_JIS・52列・単オッズ col49）としてコミット済み。直近の `data/sample/2026*_all.csv` 6日分も同形式。バックテストはこれらを全部 `--input` に渡す。
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

## 進行中のデータ収集プラン（ユーザー合意済み）
1. **単勝2年分**を先に集める: 全馬成績→4(フルセット+単オッズ)を **2024年半ば〜** まで遡って一括出力（既存は2025/07〜2026/05）。操作は既存と同じ、新データ種別ではない。溜まったら `ml/feasibility.py`（DS*.CSVを自動で全部拾う）で再学習し、ML OOS回収率90%が動くか測る。分布ずれ解消で伸びる仮説。
2. **その後 連系(払戻)データ**: この実行環境はネット制限で**JRA/netkeiba等に一切アクセス不可**（WebFetchが全ドメイン403。ユーザーのMacなら閲覧可）。→ 払戻はユーザーがJRA結果ページを貼る運用。**`src/backtest/jra-payout.ts` の `parseJraResult(text)` でJRA結果ページのテキスト→RacePayout を抽出（実データで検証済み）。** 連系の 確率(harville.ts)・購入シミュ(exotic-sim.ts)・払戻パース(jra-payout.ts) は完成・テスト済み。
   - **買い方研究フレームワーク完成**: `src/ev/strategies.ts`(box/formation/topK生成) + `src/backtest/exotic-research.ts`(戦略総当たり→回収率降順)。ボックス上位N頭・軸×相手・Harville上位K点を券種別に一括比較し「馬連はbox何頭が最適か」等を実測できる。実払戻でテスト済み(三連複box3→470%等)。
   - **残りの本実装**: ①多数レースの払戻を集める（JRA結果ページを貼る→`parseJraResult`）②(日付+開催場+レース番号)↔TARGET raceId の対応で払戻を scoredレースに結合 ③`research()` を回して最適な買い方を決定。scoreは rule/ML どちらの win_prob でも可。ユーザーのゴールは「連系の最適な買い方の研究」。
- ⏰ 無料期間〜2026/08/03。CSV化すれば期限後も再学習/検証は無制限。

## 次にやること（優先順）
1. **開催日データを増やす**（無料期間中）→ 全部結合してバックテスト（`--input` 複数）。印別・キャリブレーション・weightを本格調整。
2. weight調整の続き: `人気落ち◎` は符号反転（minus化）も検討余地。softmax温度の再詰め。
3. **Phase 4**: keiba-ev.jsx 受領 → スコアJSON接続 → EV>1.0 の買い目出力 → バックテストにEV購入シミュ追加。
4. 注意: 生成物を外部公開する構想があるなら JRA-VAN の外部提供規約確認が先（実装と別件）。

## トークン節約（CLAUDE.md方針）
1タスク1セッション。長い作業は /compact。大きいファイルは必要な行だけ読む。セッション間はこの md 経由で引き継ぐ。
