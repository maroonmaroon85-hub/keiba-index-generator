# ml/ — Phase 5 機械学習パイプライン

ルールベース指数を LightGBM 確率モデルへ置き換えるための学習コード。
特徴量は「近走(prior runs)＋条件」のみで、**単勝オッズは特徴に入れない**（市場追随回避、EV評価専用）。

## セットアップ
```bash
pip install lightgbm scikit-learn pandas numpy joblib
```

## 使い方
```bash
# リポ直下(DS*.CSVがある場所)で実行
python3 ml/train.py        # 全DS*.CSVで学習→OOS指標→ml/model/に保存
python3 ml/feasibility.py  # 実験用(モデル保存なし)。"te"引数で血統ターゲットエンコーディング有効
```

## ファイル
- `features.py` — 共有の特徴量エンジニアリング（DS形式CSV→特徴量）。train/将来のpredictで同一ロジック。
- `train.py` — 学習＋アウトオブサンプル評価＋モデル保存（`ml/model/`）。**データが増えたら再実行するだけ。**
- `feasibility.py` — 実現可能性の検証用（ルールベースとの比較を表示）。

## 現状の成績（約1年・アウトオブサンプル）
- 1年(77k頭): AUC 0.751 / top-pick複勝率 56.7% / 単勝EV≥1回収率 約90%
- 2年(13.6万頭, DS240706〜DS260523): AUC 0.755 / top複勝57.4% / 勝率28.3% / 単勝EV回収 87.3%
- データ量は逓減リターン（1→2年でAUC+0.004）。単勝の収益化は未達(~87%)。最重要特徴は母父→父(血統)。

## 次の本実装（未着手）
1. **推論(predict.py)**: あるレースの seiseki＋出馬表 から特徴量を作り、保存モデルで各馬 win_prob を出力（JSON）。
   features.py を再利用。カテゴリは `ml/model/cat_maps.json` で整数コード化（未知=-1）。
2. **TS統合**: 生成CLIに `--ml <pred.json>` モードを足し、win_prob を ML 由来に差し替え → 印/EV/連系がML確率で動く。
3. データ多年化 → 再学習 → 複勝/連系（払戻データ受領後）へ展開。
