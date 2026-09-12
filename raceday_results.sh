#!/bin/sh
# ★★★★開催日の夜、Macでやることを1本にまとめたもの
#
#     sh raceday_results.sh 20260913
#
# ■ やること
#   1. ハブブランチに合わせる（★朝の買い目の凍結を取り込む）
#   2. 結果と払戻   ml/nk_fetch.py results  → data/nk/DSnk<日付>.CSV ・ data/nk/pay<日付>.csv
#   3. 名寄せ       ml/nk_link.py           （★次の開催日の予想に効く）
#   4. commit → push
#
# ⚠★**レース数が足りないとき**（開催途中に取った一覧がキャッシュに残っている）は
#   　`RACEDAY_REFRESH=1 sh raceday_results.sh <日付>` で一覧だけ捨てて取り直す。
#   　★2026-08-09 に 13/36 のまま固まった前例がある。★**レース数を必ず目で見ること**。
#
# ★このあとクラウド（Claudeのセッション）で: python3 ml/raceday_night.py <日付>
set -e

BRANCH="${RACEDAY_BRANCH:-claude/dreamy-keller-2p8bmf}"   # ★運用ハブ
YMD="$1"
[ -n "$YMD" ] || { echo "日付が要る: sh raceday_results.sh 20260913"; exit 1; }
cd "$(dirname "$0")"

if [ -n "$(git status --porcelain)" ]; then
  echo "⚠ コミットしていない変更がある。先に片付けること:"; git status --short; exit 1
fi
git fetch origin "$BRANCH"
if ! git merge --ff-only "origin/$BRANCH" >/dev/null 2>&1; then
  echo "⚠ いまのブランチ（$(git rev-parse --abbrev-ref HEAD)）からハブへ早送りできない。"
  echo "　 ★初回だけ: git checkout -B $BRANCH origin/$BRANCH"
  exit 1
fi
echo "★ハブ $BRANCH に合わせた（$(git rev-parse --short HEAD)）"

echo ""; echo "=== 結果と払戻 ==="
if [ -n "$RACEDAY_REFRESH" ]; then
  python3 ml/nk_fetch.py results "$YMD" --refresh
else
  python3 ml/nk_fetch.py results "$YMD"
fi

echo ""; echo "=== 名寄せ ==="
python3 ml/nk_link.py

echo ""; echo "=== commit / push ==="
git add data/nk
if git diff --cached --quiet; then
  echo "（新しく取れたものが無い）"
else
  git commit -q -m "$YMD の結果と払戻"
  n=1
  until git push -u origin "HEAD:$BRANCH"; do
    [ "$n" -ge 4 ] && { echo "⚠ push に4回失敗した"; exit 1; }
    sleep $((1 << n)); n=$((n + 1))
  done
fi

echo ""
echo "★終わり。★クラウドのセッションで次を走らせること:"
echo "    python3 ml/raceday_night.py $YMD"
