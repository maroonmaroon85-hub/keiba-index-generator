"""(244) ★★★★**当日の穴馬の買い目を出す** —— 朝9時の単勝＋朝9時の複勝板から

★★**なぜ要るか（2026-09-12 に判明した穴）**
　⚠**`reco_ana.py` は未走レースでは動かない**——`load_races()` が **着順>0 の馬しか取らない**
　（`audit_crosspool.py:98`）。★**「朝9時に `reco_ana.py` で推奨を出す」は成り立たなかった**。
　★**これはその穴を埋めるスクリプト**（**当日・未走レース専用**）。

■ ★**入力（★どちらも朝9時のもの**）
| | ★**中身** | ★**作るもの** |
|---|---|---|
| `data/nk/entries<日付>.json` | **出馬表＋単勝オッズ** | ★**枠連側の `nk_fetch.py entries`**（⚠**こちらは読むだけ**） |
| `data/nk_odds_morn/place<日付>.jsonl` | ★**複勝の板[下限,上限]** | ★**`ml/nk_place_morn.py`**（2026-09-12から） |

■ ★★**軸（★一切変えていない）**: `pn ≥ 0.15 かつ ズレ ≥ 0.15 かつ 単勝 ≥ 10.0倍 の中で pn 最大`
　**pn** = モデルの複勝確率をレース内で合計3に正規化／**qp** = 複勝板の調和平均を合計3に正規化
　**ズレ = pn − qp**

■ ★★**買い目（★一切変えていない・`ANA_RULE.md` §4）**
　★**P馬単M4点**（紐＝モデル上位2頭・マルチ）＋★**X三連単A4点**（紐＝モデル上位2頭＋人気1頭・軸1着）
　＝ ★**8点 800円/レース**

■ ⚠★★★**これは凍結値（113.4% / 128.8%）を引き継がない**（`ANA_RULE.md` §6・理由は4つ）
| | ⚠**相違** |
|---|---|
| **1** | **qp が朝9時の板**（凍結は確定板） |
| **2** | **単勝が朝9時**（凍結は確定。★(219)で平均39.3%動く） |
| **3** | ★**モデルが `ml/model_ana_fwd`（固定）**。⚠**凍結値は `wf_predict`（walk-forward）** |
| **4** | ⚠**過去走に繋がらない馬は落ちる**→★**pn の Σ=3 が欠けた頭数の上で行われる** |
　★**3は `train_ana_fwd.py` で「作り方」を揃えた**（**2025年までで学習・L2・3シード**）が、
　　⚠**同じ予測ではない**。★**1・2・4 は消せない**。

■ ⚠★**前向きの標本としての主判定は、まだ登録していない**
　★**買い目を出すことと、それを何で判定するかは別**。★**判定は結果を見る前に登録すること**。

実行: python3 ml/reco_ana_day.py 20260912      自己テスト: python3 ml/reco_ana_day.py --selftest
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features as F
from predict import load_model
from predict_nk import load_entries          # ★枠連側の読み込みを再利用（★読むだけ）
from nk_link import build_map
from audit_ana_board import NPLACE, qpool
from audit_ana_band import PN_FLOOR
from audit_ana_hole import GAP
from audit_ana_fix import LFIX
from audit_ana_odds import MIN_HORSES
from audit_ana_bet import tickets

MODEL = "ml/model_ana_fwd"
MORN = "data/nk_odds_morn"
BUY = [("P馬単M4点", "P", "馬単M", 4), ("X三連単A4点", "X", "三連単A", 4)]


def load_morn_boards(ymd):
    """`place<日付>.jsonl` → {raceid8: ({馬番: (下限,上限)}, fetched_at)}。★同じレースは最新を使う。"""
    p = f"{MORN}/place{ymd}.jsonl"
    if not os.path.exists(p):
        return {}, p
    out = {}
    for line in open(p, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except ValueError:
            continue                       # ⚠書き込み中に落ちた末尾
        d = {}
        for k, v in (r.get("odds") or {}).items():
            if not str(k).isdigit():
                continue
            lo, hi = (float(v[0]), float(v[1])) if isinstance(v, (list, tuple)) else (float(v),) * 2
            if lo > 0 and hi > 0:
                d[int(k)] = (lo, hi)
        if not d:
            continue
        prev = out.get(r["raceid"])
        if prev is None or r["fetched_at"] > prev[1]:   # ★同じレースは新しい方
            out[r["raceid"]] = (d, r["fetched_at"])
    return out, p


def axis_and_himo(ub, od, pv, board):
    """→ (軸, 紐P, 紐X, pn, qp, gap) / 軸が立たなければ None。★軸の定義は変えない。"""
    pn = pv / pv.sum() * NPLACE
    qp, _ = qpool([board[int(u)] for u in ub], "harm")
    gap = pn - qp
    c = np.where((pn >= PN_FLOOR) & (gap >= GAP) & (od >= LFIX))[0]
    if not len(c):
        return None, None, None, pn, qp, gap
    i = int(c[int(np.argmax(pn[c]))])
    ax = int(ub[i])
    op = [int(u) for u in ub[np.argsort(-pv, kind="mergesort")] if int(u) != ax]
    oq = [int(u) for u in ub[np.argsort(od, kind="mergesort")] if int(u) != ax]
    X = op[:2] + [u for u in oq if u not in op[:2]]
    return ax, op, X, pn, qp, gap


def selftest():
    ok = True
    print(f"★★軸（★変えない）: **pn ≥ {PN_FLOOR} かつ ズレ ≥ {GAP} かつ 単勝 ≥ {LFIX}倍 の中で pn 最大**")
    print(f"★★買い目（★変えない）: " + " ＋ ".join(f"**{b[0]}**" for b in BUY) + "　= 8点 800円/レース")
    print(f"★モデル: **{MODEL}**（⚠**枠連の ml/model_prod ではない**）")
    ok &= MODEL != "ml/model_prod"
    meta = f"{MODEL}/meta.json"
    if os.path.exists(meta):
        m = json.load(open(meta))
        print(f"　学習 {m['date_from']}〜{m['date_to']} / {m['rows_train']:,}行 / {m['train_filter']}")
        ok &= m["date_to"] < "2026-01-01"
        print(f"　★**2026年を学習に使っていない** {'★OK' if m['date_to'] < '2026-01-01' else '⚠NG'}")
    else:
        print(f"　⚠**{meta} が無い。先に `python3 ml/train_ana_fwd.py`**")
        ok = False
    # ★軸の検算（★合成データ・通信しない）
    ub = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    od = np.array([2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 40.0, 80.0])
    pv = np.array([.30, .22, .14, .10, .13, .05, .04, .02])
    bd = {1: (1.1, 1.3), 2: (1.3, 1.6), 3: (1.8, 2.3), 4: (2.5, 3.2),
          5: (4.0, 5.5), 6: (5.0, 7.0), 7: (8.0, 12.0), 8: (15.0, 22.0)}
    ax, op, X, pn, qp, gap = axis_and_himo(ub, od, pv, bd)
    j = int(np.argmax(pn - qp))
    print(f"★軸の検算（合成8頭）: 軸={ax} / 紐P={op[:3] if op else None} / 紐X={X[:3] if X else None}")
    print(f"　ズレ最大の馬 {int(ub[j])}（pn {pn[j]:.3f} − qp {qp[j]:.3f} = {gap[j]:+.3f}"
          f" / 単勝 {od[j]:.1f}倍）")
    good = ax == 5 and op[:2] == [1, 2]
    print(f"　★**単勝10倍以上＋ズレ最大＝5番が軸／紐Pはモデル上位 [1,2]** {'★OK' if good else '⚠NG'}")
    ok &= good
    t = tickets("馬単M", 4, 5, [1, 2])
    print(f"★買い目の検算: P馬単M4点 → {[x[1] for x in t]}")
    ok &= t is not None and len(t) == 4
    print("⚠★**凍結値(113.4%/128.8%)は引き継がない**（ANA_RULE.md §6 の4つの理由）")
    print("⚠★**前向きの主判定はまだ登録していない**")
    print("⚠**枠連の運用には触れない**")
    print("★自己テスト: " + ("全部OK" if ok else "⚠NG"))
    return 0 if ok else 1


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        sys.exit("日付が要る: python3 ml/reco_ana_day.py 20260912")
    ymd = args[0]
    today = pd.Timestamp(f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:8]}")
    print(f"(244) ★★★★**{today.date()} の穴馬の買い目**（★朝9時の単勝＋朝9時の複勝板）\n")

    ep = f"data/nk/entries{ymd}.json"
    if not os.path.exists(ep):
        sys.exit(f"⚠{ep} が無い。★先に `python3 ml/nk_fetch.py entries {ymd}`")
    races = load_entries(ep)
    boards, bp = load_morn_boards(ymd)
    if not boards:
        sys.exit(f"⚠{bp} が無いか空。★先に `python3 ml/nk_place_morn.py {ymd}`")
    ats = sorted({v[1][:16] for v in boards.values()})
    print(f"★出馬表 **{len(races)}レース**（単勝 {races[0].get('odds_at','?')} 時点）")
    print(f"★複勝の板 **{len(boards)}レース**（{' / '.join(ats)} 時点）")

    boosters, cat_maps, cols, meta = load_model(MODEL)
    print(f"★モデル {MODEL}（学習 {meta['date_from']}〜{meta['date_to']} / "
          f"{meta['rows_train']:,}行・シード{meta['n_seed']}本）")

    # ★過去走のパネル（★`train_ana_fwd.load_panel` と同じ）
    from train_ana_fwd import load_panel
    d, nds = load_panel()
    dup = d["raceid"].isin({rc["raceid"] for rc in races})
    if dup.any():
        print(f"　※対象レースが過去走に{int(dup.sum())}行含まれていたので除外")
        d = d[~dup].reset_index(drop=True)
    mp = build_map()
    last = d.sort_values("date").groupby("horse").tail(1).set_index("horse")

    synth, miss = [], {}
    for rc in races:
        surf = 1 if rc["surface"] == "ダ" else 0
        for h in rc["horses"]:
            reg = mp.get(f"{h['name']}|{h['sex']}|{today.year - h['age']}")
            if reg is None or reg not in last.index:
                miss.setdefault(rc["raceid"], []).append(h["name"])
                continue
            lr = last.loc[reg]
            if isinstance(lr, pd.DataFrame):
                lr = lr.iloc[0]
            synth.append({
                "date": today, "course": rc["place"], "surface": surf,
                "distance": rc["distance"] or lr["distance"], "cond": rc["cond"], "horse": reg,
                "sex": h["sex"], "age": h["age"], "wtcarry": h["wtcarry"],
                "fieldsize": len(rc["horses"]), "finish": np.nan, "margin": np.nan,
                "passavg": np.nan, "agari": np.nan, "prize": np.nan,
                "raceid": rc["raceid"], "umaban": h["umaban"],
                "jockey": h["jockey"] or str(lr["jockey"]),
                "trainer": h["trainer"] or str(lr["trainer"]),
                "bodywt": np.nan, "sire": lr["sire"], "damsire": lr["damsire"],
                "raceclass": F._classcode(rc["name"]), "odds": h["odds"],
                "finratio": np.nan, "passratio": np.nan,
            })
    d2 = pd.concat([d, pd.DataFrame(synth)], ignore_index=True)
    d2 = d2.sort_values(["horse", "date"]).reset_index(drop=True)
    f = F.build_features(d2)
    is_today = d2["date"].eq(today).to_numpy()
    fx = F.encode_categoricals(f[is_today], cat_maps)
    sub = d2.loc[is_today, ["raceid", "umaban", "odds"]].copy()
    o = sub["odds"].to_numpy(float)
    inv = 1.0 / o
    fx["log_odds"] = np.log(o)
    fx["mkt_prob"] = inv / pd.Series(inv).groupby(sub["raceid"].to_numpy()).transform("sum").to_numpy()
    sub["p"] = np.mean([b.predict(fx[cols].values) for b in boosters], axis=0)

    nm = {(rc["raceid"], h["umaban"]): h["name"] for rc in races for h in rc["horses"]}
    hit, skip, total = [], [], 0
    for rc in races:
        rid = rc["raceid"]
        lab = f"{rc['place']}{rc['r']:>2}R {rc['name'][:12]}"
        g = sub[sub["raceid"] == rid]
        bd = boards.get(rid, (None, ""))[0]
        lost = miss.get(rid, [])
        if bd is None:
            skip.append((lab, "★朝9時の板が無い")); continue
        if len(g) < MIN_HORSES:
            skip.append((lab, f"⚠出走{len(g)}頭（過去走に繋がらない {len(lost)}頭）")); continue
        ub = g["umaban"].astype(int).to_numpy()
        if not all(int(u) in bd for u in ub):
            skip.append((lab, "⚠板に無い馬がいる")); continue
        od, pv = g["odds"].to_numpy(float), g["p"].to_numpy(float)
        if not np.isfinite(od).all() or (od <= 0).any() or pv.sum() <= 0:
            skip.append((lab, "⚠単勝オッズが欠けている")); continue
        ax, op, X, pn, qp, gap = axis_and_himo(ub, od, pv, bd)
        if ax is None:
            skip.append((lab, f"該当なし（ズレ最大 {gap.max():+.3f}）")); continue

        print(f"\n{'='*92}\n★★**{lab}**　{rc['surface']}{rc['distance']}m {len(ub)}頭"
              + (f"　⚠**過去走に繋がらない {len(lost)}頭を除いている**" if lost else ""))
        order = np.argsort(-pv, kind="mergesort")
        rank = {int(u): r + 1 for r, u in enumerate(ub[np.argsort(od, kind="mergesort")])}
        print(f"{'馬番':>4}{'馬名':<16}{'単勝':>8}{'人気':>5}{'★推奨度':>9}{'市場':>8}{'★ズレ':>8}  役割")
        for k in order:
            u = int(ub[k])
            role = ("★軸（穴馬）" if u == ax else
                    "紐1（モデル順）" if op and u == op[0] else
                    "紐2（モデル順）" if op and len(op) > 1 and u == op[1] else
                    "紐3（人気順）" if X and len(X) > 2 and u == X[2] else "")
            print(f"{u:>4}{nm.get((rid,u),'')[:15]:<16}{od[k]:>8.1f}{rank[u]:>5}"
                  f"{pn[k]:>9.3f}{qp[k]:>8.3f}{gap[k]:>+8.3f}  {role}")
        cost = 0
        for name, hk, kind, npt in BUY:
            himo = op if hk == "P" else X
            t = tickets(kind, npt, ax, himo)
            if t is None:
                print(f"　⚠**{name}: 紐が足りない**"); continue
            cost += 100 * len(t)
            print(f"　★**買 {name}**　" + " / ".join("-".join(str(x) for x in s) for _, s in t))
        total += cost
        print(f"　★**購入額 {cost:,}円**")
        hit.append(lab)

    print(f"\n{'='*92}")
    print(f"★★**該当 {len(hit)}レース / {len(races)}レース　★合計 {total:,}円**")
    for lab in hit:
        print(f"　★{lab}")
    if skip:
        print(f"\n（該当なし {len(skip)}レース）")
        for lab, why in skip[:40]:
            print(f"　{lab}　{why}")
    print(f"\n⚠★**凍結値(113.4%/128.8%)は引き継がない**（ANA_RULE.md §6・理由4つ）")
    print(f"⚠★**前向きの主判定はまだ登録していない**")
    print("⚠**枠連の運用には触れていない**")
    return 0


if __name__ == "__main__":
    sys.exit(selftest() if "--selftest" in sys.argv else main())
