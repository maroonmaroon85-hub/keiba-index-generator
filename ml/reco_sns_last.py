"""(252sns) ★★★★**◎の前走の中身を、★競馬の言葉で出す** —— 利用者の指定（2026-09-16）

★**利用者の選択**: **「A. 前走の中身を別に取る」**。
　★**(251sns)で「モデルからは記事の理由が出ない」ことが分かった**ため
　（**◎を押し上げているのは母父+0.958/厩舎+0.641/父+0.397＝★素性ばかり**）。

■ ★★★★**取りに行く必要は無かった —— ★既に手元にある**
　★**過去走のパネル（`features.to_model`）は1走ごとに次を持っている**:
　　**着順 / 着差 / ★4角までの通過順平均 / ★上がり3F / 馬体重 / 頭数 /
　　　クラス / 距離 / 馬場種別 / 馬場状態 / 単勝オッズ / 騎手**
　★★**しかも★全馬ぶんある**——**だから「そのレースで上がり何位か」を★計算できる**。
　　→ ★**「メンバー最速の上がり」「14番手から差した」が★言える**。**これが記事の言葉**。
　⚠**新しいスクレイピングを一切していない**（**`data/odds_ts` にも触っていない**）。

■ ⚠**取れないもの（★先に書く）**
　⚠**短評・レース回顧のテキスト / ラップ / ★「不利を受けた」の記述**。
　→ ★**「前走は不利があった」とは書けない**。**書けるのは「14番手から上がり最速」まで**。
　　★**そこから「展開が向かなかった」と解釈するのは★利用者の主張**であってモデルの主張ではない。

■ ★★手続き（⚠**`reco_ana_day.py` も `reco_sns_day.py` も1行も書き換えない**）
　★**`reco_sns_why.capture()` を再利用する**——**軸の判定も過去走も、★向こうが作った物をそのまま使う**。
　⚠**軸を再計算しない**（**再計算した瞬間に別物になりうる**）。

実行: python3 ml/reco_sns_last.py 20260913 [--n 3]

■ ★★★★**(253sns) コーナーごとの通過順を足した（2026-09-16）—— ★取りに行っていない**

★**利用者**: **「（短評が）あれば使えそうだけどね」**。
⚠★**短評は取れない**——**`db.netkeiba` の結果ページに★列が無い**
　（**着順/枠/馬番/馬名/性齢/斤量/騎手/タイム/着差/通過/上り/単勝/馬体重/調教師**）。
　★**短評は netkeiba の有料側**。⚠**そこを取りに行くのは規約上も筋が悪いので勧めない**。

★★**代わりに、★使っていない材料が手元にあった**——**生CSVの★28〜31列＝コーナーごとの通過順**。
　⚠**`features.to_model` が4つを平均して `passavg` にしており、★形が消えていた**。
　★**`features.py` は書き換えず、`to_model` を包んで生の入力から作り直した**。

★**これが効いた実例**:
| | 平均だけ | ★**コーナーごと** |
|---|---|---|
| **レッドホットの未勝利勝ち** | 「道中12番手」 | ★★**`12-11-14-12`＝4角12番手から1着まで差した** |
| **ランウェイミューズ 8/15** | 「道中11番手」 | ★**`13-12-8`＝13番手から8番手まで押し上げ** |
　★★**「4角で後方なのに上位で入った」は★平均に潰すと消える**。**そこが一番の売り**。
　★**近走で何回それをやったかも数えた**（**レッドホットは近3走で2回**）。

■ ⚠★**実装で踏んだもの**: **`astype(str)` を通しても要素が float で来た**
　（**concat の中身がまちまちなため**）。★**`str(x)` で必ず文字にしてから判定する**。
　⚠**最初は `except: pass` で握り潰していて、★「0行拾えた」と静かに壊れていた**。
　★**握り潰さず理由を出すようにした**。

■ ★★★★**結果（2026-09-16・9/13の◎4頭）—— ★記事になる**
　★**(251sns)の寄与では何も言えなかった4頭が、★前走の中身では全部書けた**:
| ◎ | ★**書ける材料** |
|---|---|
| **レッドホット**(18.5倍) | ★**未勝利を12番手からメンバー最速で差し切り→昇級初戦8着も上がり2位** |
| **ランウェイミューズ**(11.6倍) | ★**前走2着・道中12番手からメンバー最速** |
| **シルフレイ**(8.6倍) | ★**4ヶ月ぶり（17週）の休み明け** |
| **コスモベスケーデン**(17.5倍) | **200mの距離短縮・芝→ダート替わり** |
　★★**同じ◎でも、★モデルの寄与は「母父」「厩舎」しか返さなかった**。
　　→ ★**記事の材料は「モデルの中」ではなく「★生の過去走」にあった**。

■ ⚠⚠★★**初版のバグ2つ（★投稿に嘘が乗るところだった。★2026-09-16に修正）**
　1. ⚠**上がりの★順位を見ずに「上がりは速い」と書いていた**
　　　→ **シルフレイの前走は 上がり7位/11頭 なのに「速い」と出ていた**。
　　　★**順位が3位以内のときだけ言うようにした**（**「上がり2位」と順位も出す**）。
　2. ⚠**休み明けの閾値が20週で高すぎ、★17週ぶりを取りこぼしていた**
　　　→ **シルフレイの「4ヶ月ぶり」が★見出しの事実なのに落ちていた**。
　　　★**26週/13週/8週の3段にした**。
　★★**有料で読ませるなら、★この種の嘘が一番まずい**。**順位や日数は必ず実数から出すこと**。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

CLS = {0: "新馬", 1: "未勝利", 2: "1勝", 3: "2勝", 4: "3勝",
       5: "OP", 6: "L", 7: "G3", 8: "G2", 9: "G1"}


def ordinal(v, arr, bigger_is_better=False):
    """★そのレースの中で何位か。→ (順位, 母数) / 出せなければ (None, n)。"""
    a = np.asarray(arr, float)
    ok = np.isfinite(a)
    if not np.isfinite(v) or ok.sum() < 3:
        return None, int(ok.sum())
    r = int((a[ok] > v).sum() + 1) if bigger_is_better else int((a[ok] < v).sum() + 1)
    return r, int(ok.sum())


def corners_of(cs):
    """"12-12-11-10" → [12,12,11,10]。⚠読めなければ空。"""
    return [int(x) for x in (cs or "").split("-") if x.isdigit()]


def corner_phrase(cs, n, fin=None):
    """★★コーナーごとの通過順 → ★記事の言葉。⚠無ければ None。

    ★★**いちばん効くのは「★最終コーナーで後方なのに上位で入った」**——
    　**レッドホットの未勝利勝ちが `12-11-14-12` で、★4角12番手のまま差し切っている**。
    　★**「道中11番手」では消えてしまう情報**。**平均に潰さない理由がこれ**。
    """
    v = corners_of(cs)
    if not v or not n:
        return None
    lab = lambda p: "逃げ・先行" if p <= n * 0.25 else ("中団" if p <= n * 0.6 else "後方")
    if len(v) == 1:
        return f"道中{v[0]}番手（{lab(v[0])}）"
    first, last = v[0], v[-1]
    tail = f"{'-'.join(str(x) for x in v)}（{lab(first)}→{lab(last)}）"
    corner_no = {2: "3角", 3: "4角", 4: "4角"}.get(len(v), "最終コーナー")
    if fin and fin <= 3 and last > n * 0.55:
        return f"★★{corner_no}{last}番手から{fin}着まで差した　{tail}"
    if first - last >= 4:
        return f"★{first}番手から{last}番手まで押し上げ　{tail}"
    if last - first >= 4:
        return f"⚠{first}番手から{last}番手まで下げた　{tail}"
    return f"通過 {tail}"


def one_run(r, field, corner=None):
    """★1走を日本語1行に。`field` は★そのレースの全馬（上がり順位を出すため）。"""
    d = r["date"]
    cls = CLS.get(int(r["raceclass"]) if pd.notna(r["raceclass"]) else -1, "")
    surf = "ダ" if int(r["surface"]) == 1 else "芝"
    n = int(r["fieldsize"]) if pd.notna(r["fieldsize"]) else 0
    fin = int(r["finish"]) if pd.notna(r["finish"]) else 0
    head = (f"{d:%y/%m/%d} {str(r['course'])[:4]:<4}{surf}{int(r['distance']):>4}m "
            f"{str(r['cond'])[:2]:<2} {cls:<3} {n:>2}頭　★**{fin:>2}着**")

    bits = []
    if pd.notna(r["odds"]):
        rk, _ = ordinal(float(r["odds"]), field["odds"].to_numpy(float))
        if rk:
            bits.append(f"{rk}番人気")
    if pd.notna(r["margin"]):
        m = float(r["margin"])
        bits.append("先頭" if fin == 1 else f"{m:.1f}差")
    cp = corner_phrase(corner, n, fin)
    if cp:
        bits.append(cp)                 # ★★コーナーごとが取れたら★そちらを使う
    elif pd.notna(r["passavg"]) and n:
        p = float(r["passavg"])
        pos = "逃げ・先行" if p <= n * 0.25 else ("中団" if p <= n * 0.6 else "★後方")
        bits.append(f"道中{p:.0f}番手（{pos}）")
    if pd.notna(r["agari"]):
        a = float(r["agari"])
        rk, m = ordinal(a, field["agari"].to_numpy(float))
        tag = ""
        if rk == 1:
            tag = "★**メンバー最速**"
        elif rk and m and rk <= 3:
            tag = f"★上がり{rk}位"
        elif rk:
            tag = f"上がり{rk}位"
        bits.append(f"上がり{a:.1f}秒 {tag}".strip())
    if pd.notna(r["bodywt"]):
        bits.append(f"{int(r['bodywt'])}kg")
    return head + "　" + " / ".join(bits)


def readable(runs, today_row, agari_rank=None, last_corners=None, n_deep=0):
    """★★「記事に書ける材料」を★事実として拾う（⚠解釈はしない）。

    `agari_rank` は★前走の上がり順位（**そのレースの全馬から出した実数**）。
    ⚠**渡されないときは上がりの話を★一切しない**——**順位なしで「速い」とは書けない**。
    """
    out = []
    if not len(runs):
        return out
    last = runs.iloc[-1]
    n = int(last["fieldsize"]) if pd.notna(last["fieldsize"]) else 0
    # ⚠★**初版のバグ（2026-09-16 に修正）**: ★上がりの★順位を見ずに「速い」と書いていた。
    #   **シルフレイの前走は 上がり7位/11頭 なのに「上がりは速い」と出ていた**＝★投稿に嘘が乗る。
    #   ★**順位が上位3位以内のときだけ言う**。
    if pd.notna(last["agari"]) and pd.notna(last["finish"]) and agari_rank:
        if int(last["finish"]) >= 4 and agari_rank <= 3:
            lab = "メンバー最速" if agari_rank == 1 else f"上がり{agari_rank}位"
            out.append(("AGARI", f"前走は掲示板を外しているが{lab}"))
    if last_corners and n:
        lastpos, lfin = last_corners[-1], int(last["finish"]) if pd.notna(last["finish"]) else 99
        if lfin <= 3 and lastpos > n * 0.55:
            out.append(("POS", f"★前走は4角{lastpos}番手から{lfin}着まで差している"))
        elif lastpos > n * 0.6:
            out.append(("POS", "前走は後方からの競馬"))
    elif pd.notna(last["passavg"]) and n and float(last["passavg"]) > n * 0.6:
        out.append(("POS", "前走は後方からの競馬"))
    # ★★近走で「4角後方→3着以内」が複数あれば、★それが売りになる
    if n_deep >= 2:
        out.append(("DEEP", f"★近{len(runs)}走のうち{n_deep}回、4角後方から3着以内に来ている"))
    dd = float(today_row["distance"]) - float(last["distance"])
    if dd <= -200:
        out.append(("DIST", f"{abs(dd):.0f}m の距離短縮"))
    elif dd >= 200:
        out.append(("DIST", f"{dd:.0f}m の距離延長"))
    if int(today_row["surface"]) != int(last["surface"]):
        out.append(("SURF", "芝⇄ダートの替わり"))
    # ⚠★**初版のバグ（2026-09-16 に修正）**: ★閾値20週が高すぎて休み明けを取りこぼす。
    #   **シルフレイは 5/17 → 9/13 の★17週ぶりなのに何も出ていなかった**＝★見出しの事実を落とす。
    w = (today_row["date"] - last["date"]).days / 7.0
    if w >= 26:
        out.append(("REST", f"★半年以上（{w:.0f}週）ぶりの実戦"))
    elif w >= 13:
        out.append(("REST", f"★{w/4.3:.0f}ヶ月ぶり（{w:.0f}週）の休み明け"))
    elif w >= 8:
        out.append(("REST", f"{w:.0f}週ぶり"))
    elif w <= 2:
        out.append(("REST", f"中{int(w*7)-1}日の詰めた間隔"))
    f3 = runs["finish"].tail(3).dropna()
    if len(f3) >= 2 and (f3 <= 3).all():
        out.append(("FORM", "近走は連続で3着以内"))
    if len(f3) >= 3 and (f3 >= 6).all():
        out.append(("FORM", "⚠近3走はいずれも6着以下"))
    return out


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        sys.exit("日付が要る: python3 ml/reco_sns_last.py 20260913 [--n 3]")
    nlast = 3
    for i, a in enumerate(sys.argv):
        if a == "--n" and i + 1 < len(sys.argv):
            nlast = int(sys.argv[i + 1])

    import reco_sns_why as W
    import reco_ana_day as D

    cap, picks = W.capture(args[0])
    if "d2" not in cap:
        print("\n⚠**前走を出せなかった**（**該当0本、または途中で落ちている**）")
        return 0
    sub, rows = W.axis_rows(cap, picks)
    if not rows:
        print("\n⚠**◎が1頭も立っていない。★何も出さない**")
        return 0

    d2 = cap["d2"]
    corners = cap.get("corner") or {}
    if not corners:
        print("⚠**コーナーごとの通過順を拾えなかった**"
              "——**道中の平均だけで書く（★情報が減る）**")
    today = d2["date"].max()
    past = d2[d2["date"].lt(today)]
    by_race = {k: v for k, v in past.groupby("raceid")}

    nm = {}
    try:
        for rc in D.load_entries(f"data/nk/entries{args[0]}.json"):
            for h in rc["horses"]:
                nm[(rc["raceid"], int(h["umaban"]))] = h["name"]
    except Exception:
        pass

    print(f"\n{'='*92}")
    print("(252sns) ★★★★**◎の前走の中身**（★**手元の過去走から。新しく取っていない**）")
    print("⚠**短評・ラップ・「不利」は取れない**。★**書けるのは着順・着差・通過順・上がりまで**\n")

    for k in rows:
        r = sub.iloc[int(k)]
        rid, u = str(r["raceid"]), int(r["umaban"])
        h = r["horse"]
        runs = past[past["horse"].eq(h)].sort_values("date").tail(nlast)
        print(f"■ ★★**{rid} {u}番 {nm.get((rid, u), '')}**")
        if not len(runs):
            print("　⚠**過去走が引けない**（**初出走、または照合できていない**）\n")
            continue
        for _, run in runs.iterrows():
            fld = by_race.get(run["raceid"])
            ck = f"{run['raceid']}{int(run['umaban']):02d}"
            print("　" + one_run(run, fld if fld is not None else runs,
                                 corners.get(ck)))
        lastrun = runs.iloc[-1]
        fld = by_race.get(lastrun["raceid"])
        arank = None
        if fld is not None and pd.notna(lastrun["agari"]):
            arank, _ = ordinal(float(lastrun["agari"]), fld["agari"].to_numpy(float))
        lc = corners_of(corners.get(f"{lastrun['raceid']}{int(lastrun['umaban']):02d}"))
        n_deep = 0
        for _, run in runs.iterrows():
            vv = corners_of(corners.get(f"{run['raceid']}{int(run['umaban']):02d}"))
            nn = int(run["fieldsize"]) if pd.notna(run["fieldsize"]) else 0
            ff = int(run["finish"]) if pd.notna(run["finish"]) else 99
            if vv and nn and ff <= 3 and vv[-1] > nn * 0.55:
                n_deep += 1
        mats = readable(runs, r, arank, lc, n_deep)
        if mats:
            print("　★**書ける材料**: " + " / ".join(m for _, m in mats))
        print()

    print("⚠⚠★**ここから先は★利用者の主張になる**——")
    print("　★**「14番手から上がり最速」は事実**。⚠**「展開が向かなかった」は★解釈**。")
    print("　★**解釈を書くのは構わないが、★モデルの主張として書かないこと**。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
