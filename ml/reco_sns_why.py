"""(251sns) ★★★★**◎を推す理由を★競馬の言葉で出す** —— 利用者の指定（2026-09-16）

★**利用者の指定**: **「オッズだけで推薦するのはファンがつかない」**（**note有料展開を見据えて**）。
　⚠**それまでの案（複勝の板の序列が逆）は却下された**——**板の話はオッズの話だから**。

■ ★★何をするのか
　★**モデルの出力を `pred_contrib` で1頭ぶん44個に分解し、★日本語に直す**。
　★**特徴量は全部名前が付いている**（**前走着順・距離変化・クラス替わり・乗り替わり…**）
　　→ ★**そのまま競馬の言葉になる**。

■ ★★★★これが成立する理由（★ここが設計の肝）
　★**`log_odds` と `mkt_prob` が★特徴量に入っている**（**44個のうち2個**）。
　→ ★**寄与を「オッズ由来」と「それ以外42個」に★割れる**。
　★★**◎は定義上オッズが低い評価を与える馬**（**単勝8倍以上**）。
　　→ ⚠**それでも pn が高い＝★オッズ以外の42個が押し上げている**ということ。
　★★**その42個の中身を名前で出せば、「市場が見ていない要素」を★具体名で言える**。

■ ⚠⚠★★★★**言ってはいけないこと（★先に書く）**
　1. ⚠**これは「モデルがなぜそう思ったか」であって、★「事実の因果」ではない**。
　　　★**「前走の上がりが効いた」は、★モデルの中でその特徴量が効いたという意味**。
　　　⚠**「この馬は前走で不利を受けた」のような★観察の主張にすり替えないこと**。
　2. ⚠**モデルは市場を超えていない**（**AUC −0.0054・上回った年はほぼ無い**）。
　　　→ ★**「AIが市場より正確」とは書けない**。**書けるのは「うちの指数はこう見ている」まで**。
　3. ⚠**寄与が大きい＝当たる、ではない**。**11年の複勝ROIは94.6%＝100%未満**。

■ ★★手続き（⚠**`reco_ana_day.py` も `reco_sns_day.py` も1行も書き換えない**）
　★**`reco_sns_day` を import して、その中で `D.load_model` と `F.build_features` を
　　★捕まえるだけ**。**スコアリングは1行も複製しない**——**複製した瞬間に別物になるため**。

実行: python3 ml/reco_sns_why.py 20260913 [--top 5]

■ ⚠⚠★★★★**結果（2026-09-16・9/13の◎4頭）—— ★求められた用途には使えない**

★**利用者が欲しかったのは「前走で不利があった」「距離短縮で一変」型の★競馬記事の理由**。
⚠⚠**このモデルからは★出ない**。**実測がそう言っている**。

★**◎4頭ぶんの寄与を合計した内訳**:
| 種類 | 項目 | 寄与 |
|---|---|---|
| ⚠**素性・条件** | **母父** | **+0.958** |
| ⚠**素性・条件** | **厩舎** | **+0.641** |
| ⚠**素性・条件** | **父** | **+0.397** |
| ⚠**素性・条件** | **頭数** | **+0.138** |
| ★**経過・調子** | **近走の賞金** | ★**+0.123** |
| ⚠**素性・条件** | **騎手** | **+0.101** |
| ★**経過・調子** | **近3走の着順** | ★**+0.060** |
| ★**経過・調子** | **前走の着順** | ★**+0.035** |
| ★**経過・調子** | **前走からの間隔** | ★**+0.034** |

★★★**◎を押し上げているのは、ほぼ全部★素性（血統・厩舎・騎手）と条件（頭数）**。
　⚠**前走の内容・距離替わり・上がりは★桁が一つ小さい**（**+0.03〜+0.12 対 +0.4〜+1.0**）。
　→ ⚠⚠**「この馬は前走がこうだったから買える」とは★書けない**。**モデルがそう見ていない**。
　→ ⚠**書けるのは「母父◯◯」「◯◯厩舎」まで**。**毎回同じことしか言えない＝記事にならない**。

■ ★★**これは失敗ではなく、★答えである**
　★**「モデルから記事の理由を取り出す」という道が★閉じたことが分かった**。
　★**44個の名前付き特徴量があっても、★効いているのが素性なら記事にはならない**。
　⚠**もし血統・厩舎を落として経過だけで学習し直せば経過ベースの理由は出る**が、
　　★**それは別のモデル＝★別の軸**になり、**11年の測定が全部使えなくなる**（判定基準25）。
　★**追加マスは0**——**ROIを1つも測っていない**（**モデルの中身を見ただけ**）。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

# ★★44個の特徴量 → 競馬の言葉（★`feature_cols.json` と1対1で対応させてある）
JA = {
    "n_prior": "キャリア", "last1_fin": "前走の着順", "last2_fin": "2走前の着順",
    "last3_fin": "3走前の着順", "avg3_fin": "近3走の着順", "best3_fin": "近3走のベスト",
    "last_margin": "前走の着差", "last_passratio": "前走の位置取り",
    "avg3_passratio": "近走の脚質", "last_agari": "前走の上がり",
    "best3_agari": "近3走のベストの上がり", "avg3_prize": "近走の賞金",
    "fin_std": "着順の安定度", "dist_change": "距離の替わり", "same_dist": "同距離",
    "surf_change": "芝ダートの替わり", "wt_change": "斤量の増減",
    "weeks_since": "前走からの間隔", "weeks_before": "前走までの間隔",
    "bodywt": "馬体重", "bodywt_change": "馬体重の増減",
    "jockey_form": "騎手の調子", "trainer_form": "厩舎の調子",
    "month": "時期", "season": "季節", "raceclass": "クラス",
    "class_change": "クラスの替わり", "last_class": "前走のクラス",
    "sire_season_lift": "父の季節適性", "damsire_season_lift": "母父の季節適性",
    "distance": "距離", "surface": "馬場種別", "fieldsize": "頭数",
    "wtcarry": "斤量", "age": "年齢", "sex": "性別", "cond": "馬場状態",
    "course": "コース", "sire": "父", "damsire": "母父",
    "jockey": "騎手", "trainer": "厩舎",
    "log_odds": "★単勝オッズ", "mkt_prob": "★市場の評価",
}
ODDS_FEATS = {"log_odds", "mkt_prob"}       # ★市場由来＝これだけ別勘定にする

# ★★**「記事に書ける理由」になる特徴量**（**その馬の★経過・調子**）と、
#   ⚠**素性・条件**（**父・母父・騎手・厩舎・頭数・距離…＝★毎回同じことしか言えない**）を分ける。
FORM = {
    "last1_fin", "last2_fin", "last3_fin", "avg3_fin", "best3_fin", "last_margin",
    "last_passratio", "avg3_passratio", "last_agari", "best3_agari", "avg3_prize",
    "fin_std", "dist_change", "same_dist", "surf_change", "wt_change",
    "weeks_since", "weeks_before", "bodywt_change", "class_change", "last_class",
    "jockey_form", "trainer_form", "n_prior",
}


def phrase(name, val):
    """★値の向きが言葉を変えるものだけ、★向きを付ける。⚠それ以外は名前のまま。"""
    ja = JA.get(name, name)
    try:
        v = float(val)
    except (TypeError, ValueError):
        return ja
    if not np.isfinite(v):
        return f"{ja}（不明）"
    if name == "dist_change":
        return "距離短縮" if v < 0 else ("距離延長" if v > 0 else "同距離")
    if name == "class_change":
        return "格上挑戦" if v > 0 else ("クラス下げ" if v < 0 else "同クラス")
    if name == "bodywt_change":
        return f"馬体重{'増' if v > 0 else '減'}（{v:+.0f}kg）" if v else "馬体重は同じ"
    if name == "wt_change":
        return f"斤量{'増' if v > 0 else '減'}（{v:+.1f}kg）" if v else "斤量は同じ"
    if name == "surf_change":
        return "芝ダート替わり" if v else "同じ馬場種別"
    if name == "weeks_since":
        if v >= 20:
            return f"長期休養明け（{v:.0f}週）"
        return f"{'詰めた間隔' if v <= 2 else '間隔'}（{v:.0f}週）"
    if name in ("last1_fin", "last2_fin", "last3_fin"):
        return f"{ja}（{v:.0f}着）"
    if name == "last_agari":
        return f"前走の上がり（{v:.1f}秒）"
    if name == "bodywt":
        return f"馬体重（{v:.0f}kg）"
    return ja


def capture(ymd):
    """★★印を出すついでに、★中の材料を捕まえて返す（⚠**向こうのファイルは書き換えない**）。

    → **(cap, picks)**。`cap` に **d2（過去走＋今日）/ X（特徴量行列）/ cols / boosters**、
    `picks` に **(そのレースの馬番配列, 軸の馬番)** が★軸の立った順に入る。
    ★**(251sns) と (252sns) が同じ捕まえ方を共有する**——**二重に書くと別物になるため**。
    """
    import features as F
    import reco_ana_day as D
    import reco_sns_day as S

    cap = {}

    # ★① 特徴量を組む直前の表を捕まえる（★行の身元＝raceid/馬番 を取るため）
    frozen_bf = F.build_features

    def _bf(d2):
        cap["d2"] = d2
        return frozen_bf(d2)

    # ★② boosterを包んで、★predictに渡る行列そのものを捕まえる
    frozen_lm = D.load_model

    class Probe:
        def __init__(self, b):
            self.b = b

        def predict(self, X, **kw):
            cap["X"] = X                 # ★これが fx[cols].values
            return self.b.predict(X, **kw)

    def _lm(md=None):
        boosters, cat_maps, cols, meta = frozen_lm(md)
        cap["boosters"], cap["cols"] = boosters, cols
        return [Probe(b) for b in boosters], cat_maps, cols, meta

    # ★③ 軸の判定そのものを捕まえる（⚠**軸を再計算しない**——**再計算は別物になりうる**）
    frozen_ah = D.axis_and_himo
    picks = []

    def _ah(ub, od, pv, board):
        r = frozen_ah(ub, od, pv, board)
        if r[0] is not None:
            picks.append((np.asarray(ub, int).copy(), int(r[0])))
        return r

    F.build_features, D.load_model, D.axis_and_himo = _bf, _lm, _ah
    try:
        S.main()                         # ★印はこれまでどおり出る（★出力は変えない）
    finally:
        F.build_features, D.load_model, D.axis_and_himo = frozen_bf, frozen_lm, frozen_ah
    return cap, picks


def axis_rows(cap, picks):
    """★軸の行番号を、★今日の行の中から引く（⚠**軸を再計算しない**）。→ (sub, rows)"""
    import numpy as np
    d2 = cap["d2"]
    sub = d2[d2["date"].eq(d2["date"].max())].reset_index(drop=True)
    umab = sub["umaban"].astype(int).to_numpy()
    rids = sub["raceid"].astype(str).to_numpy()
    rows = []
    for ub, ax in picks:
        for rid in dict.fromkeys(rids):
            m = np.where(rids == rid)[0]
            if len(m) == len(ub) and (umab[m] == ub).all():
                w = m[np.where(ub == ax)[0]]
                if len(w):
                    rows.append(int(w[0]))
                break
    return sub, rows


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        sys.exit("日付が要る: python3 ml/reco_sns_why.py 20260913 [--top 5]")
    top = 5
    for i, a in enumerate(sys.argv):
        if a == "--top" and i + 1 < len(sys.argv):
            top = int(sys.argv[i + 1])
    cap, picks = capture(args[0])

    if "X" not in cap or "d2" not in cap:
        print("\n⚠**寄与を出せなかった**（**該当0本、または途中で落ちている**）")
        return 0

    X, cols = np.asarray(cap["X"], float), cap["cols"]
    sub, rows = axis_rows(cap, picks)
    if len(sub) != len(X):
        print(f"\n⚠⚠**行数が合わない（{len(sub)} vs {len(X)}）。読まない**（判定基準32）")
        return 2

    # ★3シードぶんの寄与を平均する（★予測の平均と同じ扱い）
    C = np.mean([b.predict(X, pred_contrib=True) for b in cap["boosters"]], axis=0)
    contrib, base = C[:, :-1], C[:, -1]
    io_ = np.array([c in ODDS_FEATS for c in cols])

    print(f"\n{'='*92}")
    print("(251sns) ★★★★**◎を推している中身**（⚠**モデルの寄与の分解。★事実の因果ではない**）")
    print("★**44個の特徴量のうち、★オッズ由来は2個だけ**"
          "（`log_odds` / `mkt_prob`）。★**残り42個を「オッズ以外」として並べる**。\n")

    if not rows:
        print("⚠**◎が1頭も立っていない（または行を引けなかった）。★何も出さない**")
        return 0

    import reco_ana_day as D
    nm = {}
    try:
        for rc in D.load_entries(f"data/nk/entries{args[0]}.json"):
            for h in rc["horses"]:
                nm[(rc["raceid"], int(h["umaban"]))] = h["name"]
    except Exception:
        pass

    agg = {}
    for k in rows[:top]:
        r = sub.iloc[int(k)]
        rid, u = str(r["raceid"]), int(r["umaban"])
        nonodds = float(contrib[k, ~io_].sum())
        oddsc = float(contrib[k, io_].sum())
        name = nm.get((rid, u), "")
        print(f"■ **{rid} {u}番 {name}**")
        print(f"　★**オッズ以外の寄与 {nonodds:+.3f}**　/　**オッズ由来 {oddsc:+.3f}**"
              f"　（基準 {base[k]:+.3f}）")
        idx = [j for j in np.argsort(-contrib[k]) if not io_[j]][:6]
        for j in idx:
            if contrib[k, j] <= 0:
                break
            print(f"　　★{phrase(cols[j], X[k, j]):<24}{contrib[k, j]:+.3f}")
            agg[cols[j]] = agg.get(cols[j], 0.0) + float(contrib[k, j])
        neg = [j for j in np.argsort(contrib[k]) if not io_[j]][:2]
        for j in neg:
            if contrib[k, j] >= 0:
                break
            print(f"　　⚠{phrase(cols[j], X[k, j]):<24}{contrib[k, j]:+.3f}")
        print()

    print(f"■ ★★**◎ {len(rows)}頭ぶんを合計すると、どの項目が押し上げているか**")
    for nmf, v in sorted(agg.items(), key=lambda x: -x[1])[:10]:
        kind = "★経過・調子" if nmf in FORM else "⚠素性・条件"
        print(f"　{kind}　{JA.get(nmf, nmf):<22}{v:+.3f}")
    print()
    print("⚠⚠★**読み方**: **「前走の上がりが効いた」は★モデルの中でその特徴量が効いた、という意味**。")
    print("　⚠**「この馬は前走で不利を受けた」のような★観察の主張にすり替えないこと**。")
    print("　⚠**モデルは市場を超えていない**（**AUC −0.0054**）。**書けるのは「うちの指数はこう見ている」まで**。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
