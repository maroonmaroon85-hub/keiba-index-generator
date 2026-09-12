#!/bin/sh
# ★★★★開催日の朝、Macでやることを1本にまとめたもの
#
#     sh raceday_fetch.sh 20260913
#
# ■ やること（★この順番でないといけない）
#   1. ハブブランチに合わせる（★3セッションぶんのデータが1箇所に集まる）
#   2. 出馬表＋単勝オッズ   ml/nk_fetch.py entries
#   3. 朝9時の複勝板        ml/nk_place_morn.py
#   4. commit → push
#
# ⚠★**2と3を同時に走らせないこと**。`nk_place_morn.py` は起動時に `pgrep` で
#   　`nk_fetch` が走っていないかを見て、走っていれば自分から止まる（設計の中心・規則4b）。
#   　→ ★**このスクリプトは順番に走らせるので衝突しない**。
#
# ⚠★**朝9時の複勝板は後から遡れない**（`ANA_RULE.md` §6）。
#   ★**取り損ねた開催日は、穴馬もSNSも永久に出せない**。★**先に3を確実に通すこと**。
#
# ★このあとクラウド（Claudeのセッション）で: python3 ml/raceday.py <日付>
set -e

BRANCH="${RACEDAY_BRANCH:-claude/dreamy-keller-2p8bmf}"   # ★運用ハブ
YMD="$1"
[ -n "$YMD" ] || { echo "日付が要る: sh raceday_fetch.sh 20260913"; exit 1; }
cd "$(dirname "$0")"

# ---- 1. ハブに合わせる
if [ -n "$(git status --porcelain)" ]; then
  echo "⚠ コミットしていない変更がある。先に片付けること:"; git status --short; exit 1
fi
git fetch origin "$BRANCH"
if ! git merge --ff-only "origin/$BRANCH" >/dev/null 2>&1; then
  echo "⚠ いまのブランチ（$(git rev-parse --abbrev-ref HEAD)）からハブへ早送りできない。"
  echo "　 ★初回だけこれを叩いてから、もう一度このスクリプトを走らせること:"
  echo "　   git checkout -B $BRANCH origin/$BRANCH"
  exit 1
fi
echo "★ハブ $BRANCH に合わせた（$(git rev-parse --short HEAD)）"

# ---- 2. 出馬表＋単勝オッズ
echo ""; echo "=== 出馬表＋単勝オッズ ==="
python3 ml/nk_fetch.py entries "$YMD"

# ---- 3. 朝9時の複勝板（★2の後。並走させない）
echo ""; echo "=== 朝9時の複勝板 ==="
python3 ml/nk_place_morn.py "$YMD" || {
  echo ""
  echo "⚠★**複勝板の取得が落ちた**。終了コード2は「連続失敗で自分から止まった」。"
  echo "　 ⚠**自動で叩き直さないこと**（`ml/nk_place_morn.py` の禁止事項1）。"
  echo "　 ★出馬表は取れているので、枠連（本命）だけは出せる。"
  exit 2
}

# ---- 4. 記録して渡す
echo ""; echo "=== commit / push ==="
git add data/nk data/nk_odds_morn
if git diff --cached --quiet; then
  echo "（新しく取れたものが無い）"
else
  git commit -q -m "$YMD の出馬表と朝9時の複勝板"
  n=1
  until git push -u origin "HEAD:$BRANCH"; do
    [ "$n" -ge 4 ] && { echo "⚠ push に4回失敗した"; exit 1; }
    sleep $((1 << n)); n=$((n + 1))
  done
fi

echo ""
echo "★終わり。★クラウドのセッションで次を走らせること:"
echo "    python3 ml/raceday.py $YMD"
