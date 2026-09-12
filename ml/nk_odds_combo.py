"""三連複など**組み合わせ券種のオッズ**を取って保存する。**標準ライブラリのみ・Macで実行**。

★何のために要るか
　(112)の買い目を運用していて「三連複が1.9倍しかつかないレースは避けるべきでは」という
　問いが出た。だが**現データでは検証できない**——過去の払戻データには
　**的中した組の配当しか残っていない**ので、外れた組が何倍だったかが分からない。
　→ **発走前のオッズ板そのもの**を集めれば検証できるようになる。これはその収集器。

★実測で判明した type の割り当て（2026-08-08・8頭立てのレースで確認）
| type | 件数(8頭) | キー | 券種 |
|---|---|---|---|
| 1 | 8 | `'01'` | 単勝（`odds['2']` に複勝も同梱） |
| 2 | 8 | `'01'` | 複勝（値は [下限, 上限, 人気]） |
| 3 | — | — | **枠連**（8頭立ては未発売なので返らない。9頭以上で確認すること） |
| 4 | 28=C(8,2) | `'0102'` | 馬連 |
| 5 | 28 | `'0102'` | ワイド（値は [下限, 上限, 人気]） |
| 6 | 56=P(8,2) | `'0102'` | 馬単 |
| **7** | **56=C(8,3)** | **`'010203'`** | **三連複** ← 既定 |
| 8 | 336=P(8,3) | `'010203'` | 三連単 |
　値は `[オッズ, 上限(範囲を持つ券種のみ), 人気順]` の3要素。範囲を持たない券種は2つめが `'0'`。

★負荷への配慮（(70)⑤の方針）
　1.5秒間隔。**同じレース・同じ時点は取り直さない**（キーに時刻を入れないので2回目はキャッシュ）。
　⚠ただし**発走前オッズは時点が意味を持つ**ので、`--now` を付けたときだけ取り直す。

使い方（Macで実行）:
    python3 ml/nk_odds_combo.py 20260808                 # その日の全レースの三連複
    python3 ml/nk_odds_combo.py 20260808 中京 5          # 1レースだけ
    python3 ml/nk_odds_combo.py 20260808 --type 3        # 枠連（宿題3）
    python3 ml/nk_odds_combo.py 20260808 中京 5 --now    # 時点を変えて取り直す
    python3 ml/nk_odds_combo.py --old 202507020501       # ★過去レースが返るかの確認

保存先: data/nk_odds/<race_id>_type<N>_<取得時刻>.json
　　　　{"race_id":…, "type":7, "at":"…", "odds":{"020305":1.9, …}}
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nk_fetch import ODDS_API as API     # ★URLの定義は nk_fetch に1本化した

OUT = "data/nk_odds"
NAMES = {1: "単勝", 2: "複勝", 3: "枠連", 4: "馬連", 5: "ワイド",
         6: "馬単", 7: "三連複", 8: "三連単"}


RANGE_TYPES = {2, 5}      # ★複勝・ワイドは [下限, 上限] の範囲を持つ。下限だけだと過小評価


def parse_combo(txt, t, keep_range=None):
    """→ ({組キー: オッズ}, 取得時点, {捨てた生の値: 件数})。組キーは '020305'（馬番2桁ずつ）。

    ⚠値は `['1.9', '0', '3']` の形（[オッズ, 上限, 人気順]）。
    　★**範囲を持つ券種（複勝type=2・ワイドtype=5）は `[下限, 上限]` のまま返す**。
    　　下限だけにすると q_pool を系統的に過大評価する。(124)で複勝の層別Dを出すのに要る。
    　　**集めてからでは直せない**（18時間の取り直しになる）ので先に入れた。
    　範囲を持たない券種は float を返す（従来どおり）。
    　カンマ区切り（'2,403.0'）が入るので必ず除去する。

    ★**負の値は番兵であってオッズではない**（2026-08-08に実測して判明）。
    　新潟4Rで三連複 `010208→-3.0`、枠連 `0505→-3.0` が出た。オッズは定義上正なので、
    　これは取消・発売中止などを表す netkeiba 側の印。**素通しすると「最低オッズ」が壊れ、
    　これから引く閾値がすべて狂う**ので、**0以下は必ず落とす**。
    　何を落としたかは3つめの戻り値で分かるようにしてある（黙って消さない）。
    """
    try:
        j = json.loads(txt)
    except (ValueError, TypeError):
        return {}, "", {}
    d = j.get("data")
    if not isinstance(d, dict):
        return {}, "", {}
    at = str(d.get("official_datetime", "") or "")
    od = d.get("odds")
    if not isinstance(od, dict):
        return {}, at, {}
    tbl = od.get(str(t))
    if not isinstance(tbl, dict):
        return {}, at, {}
    if keep_range is None:
        keep_range = t in RANGE_TYPES
    out, drop = {}, {}
    for k, v in tbl.items():
        vals = list(v) if isinstance(v, (list, tuple)) else [v]

        def num(x):
            y = str(x).replace(",", "").strip()
            try:
                return float(y)
            except ValueError:
                return None

        f = num(vals[0]) if vals else None
        if f is None or f <= 0:         # ★番兵。オッズではない
            drop[str(vals[0] if vals else v)] = drop.get(str(vals[0] if vals else v), 0) + 1
            continue
        if keep_range:
            # ★範囲を持つ券種は **[下限, 上限]** を残す。上限が0/欠損なら下限で埋める。
            hi = num(vals[1]) if len(vals) > 1 else None
            out[str(k)] = [f, hi if (hi and hi >= f) else f]
        else:
            out[str(k)] = f
    return out, at, drop


def fetch(rid, t, now=False):
    from nk_fetch import CACHE, get
    key = (f"combo_{rid}_type{t}_{int(time.time())}.json" if now
           else f"combo_{rid}_type{t}.json")
    b = get(API.format(t=t, rid=rid), key,
            referer=f"https://race.netkeiba.com/odds/index.html?race_id={rid}")
    if not b:
        return {}, "", {}
    odds, at, drop = parse_combo(b.decode("utf-8", "replace"), t)
    if not odds and not now:
        try:                       # 空はキャッシュに残さない（後で取り直せるように）
            os.remove(os.path.join(CACHE, key))
        except OSError:
            pass
    return odds, at, drop


def save(rid, t, odds, at):
    os.makedirs(OUT, exist_ok=True)
    stamp = (at or "").replace(":", "").replace(" ", "_").replace("-", "")
    f = f"{OUT}/{rid}_type{t}_{stamp or int(time.time())}.json"
    json.dump({"race_id": rid, "type": t, "at": at, "odds": odds},
              open(f, "w", encoding="utf-8"), ensure_ascii=False)
    return f


def main():
    a = sys.argv[1:]
    if not a:
        sys.exit(__doc__)
    t = 7
    if "--type" in a:
        i = a.index("--type")
        t = int(a[i + 1])
        del a[i:i + 2]
    now = "--now" in a
    if now:
        a.remove("--now")
    if not a:
        sys.exit("日付(YYYYMMDD) か --old <race_id> を指定して")

    # ★過去レースが返るかの確認モード（遡って集められるかが分かる）
    if a[0] == "--old":
        if len(a) < 2:
            sys.exit("--old のあとに netkeiba の race_id（12桁）を書いて")
        rid = a[1]
        odds, at, drop = fetch(rid, t)
        print(f"race_id={rid} type={t}（{NAMES.get(t, t)}）")
        if odds:
            ex = list(odds.items())[:3]
            print(f"  ★取れた: {len(odds)}件 @{at}  例: "
                  + " / ".join(f"{k}→{v}" for k, v in ex))
            print("  → **過去レースも返る＝遡って集められる**")
        else:
            print(f"  取れなかった（@{at}）→ 過去分は諦めて、今後の分だけ貯める")
        if drop:
            print(f"  ※オッズでない値を落とした: {drop}")
        return

    # ★一覧ファイルから遡って集めるモード（`nk_odds_targets.py` が作る）
    if a[0] == "--list":
        if len(a) < 2:
            sys.exit("--list のあとに race_id 一覧ファイルを書いて")
        rids = [x.strip() for x in open(a[1], encoding="utf-8") if x.strip()]
        return collect(rids, t, now, label=lambda r: r)

    from nk_fetch import race_ids_of_day
    from nk_parse import PLACES
    ymd = a[0]
    place = next((x for x in a[1:] if not x.isdigit()), None)
    rno = next((int(x) for x in a[1:] if x.isdigit()), None)
    ids = race_ids_of_day(ymd)
    if not ids:
        sys.exit("レース一覧が取れなかった")
    sel = [r for r in ids
           if (place is None or PLACES.get(r[4:6], "") == place)
           and (rno is None or int(r[10:12]) == rno)]
    if not sel:
        sys.exit(f"該当なし。その日の開催: "
                 f"{' '.join(sorted({PLACES.get(r[4:6],'?') for r in ids}))}")

    collect(sel, t, now,
            label=lambda r: f"{PLACES.get(r[4:6],'')}{int(r[10:12])}R")


def collect(rids, t, now, label):
    """race_id のリストを順に取って保存する。日付指定でも一覧ファイルでも共通。"""
    name = NAMES.get(t, t)
    print(f"{name}（type={t}）を {len(rids)}レース ぶん取る\n")
    n_ok, dropped = 0, {}
    for i, rid in enumerate(rids):
        odds, at, drop = fetch(rid, t, now)
        for k, v in drop.items():
            dropped[k] = dropped.get(k, 0) + v
        lab = label(rid)
        if not odds:
            print(f"  {lab}  まだ出ていない（発売前か未発売）")
            continue
        f = save(rid, t, odds, at)
        n_ok += 1
        def _lo(v):
            return v[0] if isinstance(v, (list, tuple)) else v
        best = min(odds.items(), key=lambda kv: _lo(kv[1]))
        prog = f"[{i+1}/{len(rids)}] " if len(rids) > 60 else ""
        print(f"  {prog}{lab}  {len(odds)}件 @{at}  最低オッズ {best[0]}→{_lo(best[1])}倍  → {f}")
    print(f"\n{n_ok}/{len(rids)}レース 保存した。")
    if dropped:
        # ★黙って消さない。負の値は取消などの番兵なので、何が来たかを毎回見せる。
        print("※オッズでない値を落とした（0以下・数値でない）: "
              + " / ".join(f"{k!r}×{v}" for k, v in sorted(dropped.items())))
    print(f"★このデータが貯まると「{name}が○倍以下なら見送る」を検証できるようになる。")


if __name__ == "__main__":
    main()
