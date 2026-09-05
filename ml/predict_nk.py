"""
netkeiba の出馬表JSON（`ml/nk_fetch.py entries` の出力）から買い目を出す。

`ml/predict.py` は TARGET の DG 形式を読む。こちらは同じ推奨ロジックを netkeiba 経由の
入力に対して行う版で、**モデルも買い方も判定も predict.py と同一**（`ml/model_prod/`）:
  ・本命 枠連 軸枠×紐枠2（平均194円・ROI 84.5%・的中30.0%・的中時545円）
  ・対抗 三連複 BOX上位4（400円・ROI 84.5%・的中20.2%・的中時1,674円）
  ・枠連スコア（積ベース）**下位40%は除外**（(62)で20%→**(117)でDで測り直して40%に**。
    E[d|S] 20%:+0.0218 → 40%:+0.0235。★それでも必要量の9.2%＝損が減るだけ）
  ・9頭未満は対象外（枠連が発売されない）

DG版との違いは3つだけ:
  1. 馬の同定が **馬名＋性別＋生年 → 血統登録番号**（(70)④。netkeibaは独自IDのため）
  2. **馬場状態が本物**（DGには無いので predict.py は「良」固定。ただし(63)よりモデルは馬場を使わない）
  3. 過去走に `data/nk/DSnk*.CSV` も含める（アーカイブより新しい開催を埋めるため）

★ROI 84.5% は「長期的に投下額の15.5%が減る」という意味。1日の結果には何の意味も無い。

実行:
  python3 ml/nk_fetch.py entries 20260801      # 手元Macで取得（オッズは実行時点のもの）
  python3 ml/predict_nk.py data/nk/entries20260801.json [--out out/reco.json]
"""
import argparse
import glob
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features as F
import score_table as ST
import soft_axis as SA
from nk_link import build_map
from predict import MODEL_DIR, MIN_FIELD, load_model
from waku_umatan import bracket_probs, waku_of, waku_score, wakuren_buy

ALT_DIR = "ml/model_prod_l5"     # --dual で併記する高容量モデル（(83)）


def load_entries(path):
    j = json.load(open(path, encoding="utf-8"))
    out = []
    for e in j:
        hs = []
        for h in e["horses"]:
            try:
                hs.append({"waku": int(h["waku"]), "umaban": int(h["umaban"]), "name": h["name"],
                           "sex": h["sex"], "age": int(h["age"] or 0), "jockey": h["jockey"],
                           "trainer": h["trainer"],
                           "wtcarry": float(h["wtcarry"] or 0) or np.nan,
                           "odds": float(h["odds"]) if h.get("odds") else np.nan})
            except (ValueError, TypeError):
                continue
        out.append({"raceid": e["raceid"], "place": e.get("place", ""), "r": e.get("r", 0),
                    "name": e.get("name", ""), "surface": e.get("surface", ""),
                    "distance": int(e.get("distance") or 0), "cond": e.get("cond", "良"),
                    "odds_at": e.get("odds_at", ""), "horses": hs})
    return out


def soft_of(rc):
    """★甘い軸の三連複（(111)(112)）。**単勝オッズだけで決まる**——モデルも枠連も要らない。"""
    um = [h["umaban"] for h in rc["horses"]]
    od = [h["odds"] for h in rc["horses"]]
    if len(od) < 3 or not all(o and o > 0 for o in od):
        return None
    return SA.recommend(um, od)


def keep_soft(out_races, rc, soft):
    """枠連の対象外レースでも、**買うと判定した甘い軸だけは記録に残す**。

    `nk_score.py` はこの JSON を採点するので、ここで落とすと
    **実際に買った買い目が成績に入らない**（8/08の中京5Rで実際に起きた）。
    枠連・三連複BOXの買い目は無いので `waku_na` を立て、採点側でそこだけ飛ばす。
    """
    if not (soft and soft["buy"]):
        return
    out_races.append({"raceid": rc["raceid"], "label": rc.get("label")
                      or f"{rc['place']}{rc['r']}R", "r": rc["r"],
                      "fieldsize": len(rc["horses"]), "axis": soft["axis"],
                      "excluded": False, "waku_na": True,
                      "odds_at": rc.get("odds_at", ""), "soft_axis": soft})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("entries")
    ap.add_argument("--date", default=None, help="開催日 YYYY-MM-DD（既定はファイル名から）")
    ap.add_argument("--out", default=None)
    ap.add_argument("--excl", type=int, default=40,
                    help="枠連スコア下位N%%を除外（既定40。(117)で20→40に変更）")
    ap.add_argument("--no-exclude", action="store_true")
    ap.add_argument("--partners", type=int, default=1,
                    help="紐の数。★既定1＝本命((80)期待損失/(155)ROIの両方で紐1が上)。2にすると旧来の紐2")
    ap.add_argument("--dual", action="store_true",
                    help="高容量モデル(ml/model_prod_l5)の買い目も併記し、JSONにも記録する。"
                         "★(83)より差は未確定なので**比較用**。作るには "
                         "`python3 ml/train_prod.py 3 l5`（20〜30分）")
    ap.add_argument("--sanrentan", action="store_true",
                    help="三連単(2着固定×紐3・6点600円)も出す。★ROI79.3%%で三連複BOX4(84.5%%)より悪い")
    args = ap.parse_args()
    ymd = args.date or os.path.basename(args.entries).replace("entries", "")[:8]
    today = pd.Timestamp(f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:8]}") if len(ymd) == 8 else pd.Timestamp(args.date)

    boosters, cat_maps, cols, meta = load_model()
    # ★除外率は(117)で測り直して 20% → **40%** にした。
    # 　E[d|S] が 20%で+0.0218 → 40%で+0.0235（単調・全水準★・10/10年で正）。
    # 　⚠それでも必要量0.2549の9.2%。**損が減るだけで儲かるようにはならない**。
    pcts = meta.get("waku_score_pcts") or {}
    th = float(pcts.get(str(args.excl), meta["waku_score_p20"]))
    if str(args.excl) not in pcts:
        print(f"⚠ meta に {args.excl}% の閾値が無いので下位20%を使う"
              "（`python3 ml/train_prod.py` を回し直すと全水準が入る）")
    races = load_entries(args.entries)
    print(f"モデル: {MODEL_DIR}（{meta['target']} / シード{meta['n_seed']}本平均 / "
          f"{meta['date_from']}〜{meta['date_to']}）")
    print(f"出馬表: {args.entries} {len(races)}レース / 対象日 {today.date()}")
    print("過去走を読み込み中…")

    # ★空のDSnkは飛ばす。まだ走っていない日に `nk_fetch.py results` を当てると
    # 　0バイトのCSVができ、pandasが EmptyDataError で止まる（8/09で実際に踏んだ）。
    ds = [p for p in sorted(glob.glob("data/nk/DSnk*.CSV")) if os.path.getsize(p) > 0]
    frames = [F.load_files()] + [
        pd.read_csv(p, header=None, encoding="shift_jis", encoding_errors="replace",
                    dtype=str, keep_default_na=False) for p in ds]
    d = F.to_model(pd.concat(frames, ignore_index=True))
    # ★予測対象のレースが過去走データに既に入っていると、下の synth と二重になり
    #   同じ馬が2回並ぶ（実際に8/01の成績取込後に踏んだ）。先に落とす。
    dup = d["raceid"].isin({rc["raceid"] for rc in races})
    if dup.any():
        print(f"  ※対象レースが過去走に{int(dup.sum())}行含まれていたので除外（二重計上を防ぐ）")
        d = d[~dup].reset_index(drop=True)
    print(f"  過去走 {len(d):,}行（〜{d['date'].max().date()}）")
    mp = build_map()
    last = d.sort_values("date").groupby("horse").tail(1).set_index("horse")

    synth, miss = [], {}
    for rc in races:
        surf = 1 if rc["surface"] == "ダ" else 0
        for h in rc["horses"]:
            birth = today.year - h["age"]
            reg = mp.get(f"{h['name']}|{h['sex']}|{birth}")
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
    sub = d2.loc[is_today, ["raceid", "umaban", "odds", "fieldsize"]].copy()
    o = sub["odds"].to_numpy(float)
    inv = 1.0 / o
    fx["log_odds"] = np.log(o)
    fx["mkt_prob"] = inv / pd.Series(inv).groupby(sub["raceid"].to_numpy()).transform("sum").to_numpy()
    sub["p"] = np.mean([b.predict(fx[cols].values) for b in boosters], axis=0)
    # ★並行運用: 高容量モデルの予測も同じ特徴量から作る（(83)。差が未確定なので比較用）
    sub2 = None
    if args.dual:
        b2, cm2, cols2, meta2 = load_model(ALT_DIR)
        fx2 = F.encode_categoricals(f[is_today], cm2)
        fx2["log_odds"] = np.log(o)
        fx2["mkt_prob"] = fx["mkt_prob"]
        sub2 = sub.copy()
        sub2["p"] = np.mean([b.predict(fx2[cols2].values) for b in b2], axis=0)
        print(f"  併記: {ALT_DIR}（容量{meta2.get('capacity','?').upper()} / "
              f"シード{meta2['n_seed']}本平均）")

    print(f"\n===== {today.date()} 買い目（枠連 軸枠×紐枠2 ／ 三連複 BOX上位4） =====")
    print(f"※枠連スコア下位20%（<{th:.4f}）は除外。ROI84.5%＝**長期で投下額の15.5%が減る**。1日の結果に意味は無い")
    out_races = []
    for rc in races:
        head = (f"{rc['place']}{rc['r']:>2}R {rc['surface']}{rc['distance']:<5} "
                f"{rc['name'][:10]:10s} {len(rc['horses'])}頭 馬場{rc['cond']}")
        g = sub[sub["raceid"] == rc["raceid"]]
        lost = miss.get(rc["raceid"], [])
        # ★(111)(112) 甘い軸の三連複は**オッズだけで決まる**ので、
        # 　枠連の発売可否ともモデルの予想可否とも無関係。**下の continue より前に出す**。
        # 　⚠以前はこの計算が continue の後にあったため、**8頭立てが丸ごと落ちていた**。
        # 　　少頭数ほど断然人気が出やすい＝2%裾が出やすいので、いちばん要るレースを捨てていた
        # 　　（実際、2026-08-08に買った1本目の中京5Rが8頭立てで、記録から漏れた）。
        soft = soft_of(rc)
        if soft and soft["buy"]:
            print(f"       ★甘い軸 三連複 {soft['sanrenpuku']}（人気上位3頭・1点100円）"
                  f"  軸{soft['axis']}番・複勝の期待払戻{soft['e_axis']:.0f}円"
                  f"（裾{int(soft['tier']*100)}%）")
        if len(rc["horses"]) < MIN_FIELD:
            print(f"    {head}  対象外（{MIN_FIELD}頭未満は枠連の発売なし）")
            keep_soft(out_races, rc, soft)
            continue
        if len(g) < 3 or not np.isfinite(g["odds"].to_numpy(float)).all():
            print(f"    {head}  予想不可（過去走のある馬{len(g)}頭 / オッズ欠損）")
            keep_soft(out_races, rc, soft)
            continue
        gg = g.sort_values("p", ascending=False, kind="mergesort")
        nums = gg["umaban"].astype(int).tolist()
        n = len(rc["horses"])
        wk = {h["umaban"]: h["waku"] for h in rc["horses"]}
        info = {h["umaban"]: h for h in rc["horses"]}
        # ★★紐の数は既定1（本命）。(80)は期待損失、(155)はROIでも紐1が上と実測。
        #   (155): 8つの除外水準のうち7つで紐1のROIが上（40%除外で 86.9% vs 86.1%）、
        #   期待損失は半分（13.1円 vs 26.5円）。→ **2026-08-16に既定を紐2→紐1に変えた**。
        #   ⚠それまで紐2で出していたので、口頭で「上位1点だけ買って」と補っていた。
        pairs = sorted({tuple(sorted((wk[nums[0]], wk[h])))
                        for h in nums[1:1 + args.partners]})
        # ⚠⚠★**除外の判定は必ず紐2で作る**。`train_prod.py:113` が `wakuren_buy(nums,n,2)`
        #   でスコアの分位を作っているので、**買い目の紐数を変えると閾値の土台がずれる**
        #   （2026-08-16に紐1へ変えた瞬間、スコアが半減して**30/30レース全除外**になった）。
        #   ★**判定基準34そのもの**——**閾値は「較正した時点・条件」と揃える**。
        pairs_for_score = sorted({tuple(sorted((wk[nums[0]], wk[h]))) for h in nums[1:3]})
        # ★併記は除外レースでも記録する（後で「除外の判定自体が正しかったか」を見るため）
        alt = None
        if sub2 is not None:
            g2 = sub2[sub2["raceid"] == rc["raceid"]]
            nums2 = g2.sort_values("p", ascending=False, kind="mergesort")["umaban"].astype(int).tolist()
            pr2 = sorted({tuple(sorted((wk[nums2[0]], wk[h]))) for h in nums2[1:3]})
            alt = {"axis": nums2[0],
                   "wakuren": [f"{a}-{b}" for a, b in pr2],
                   "sanrenpuku_box": [f"{a}-{b}-{c}"
                                      for a, b, c in itertools.combinations(nums2[:4], 3)],
                   "same_as_current": bool(set(pr2) == set(pairs))}
        praw = gg["p"].to_numpy(float)
        pmap = {u: float(v) for u, v in zip(nums, praw)}
        q = praw / praw.sum()
        bp = {}
        for u, pv in zip(nums, q):
            bp[wk[u]] = bp.get(wk[u], 0.0) + float(pv)
        sc = waku_score(pairs_for_score, bp)   # ★除外の判定は紐2固定（上のコメント参照）
        skip = (not args.no_exclude) and sc < th
        ax = info[nums[0]]
        print(f"{'×除外' if skip else '○   '}{head}  軸 {nums[0]}番{ax['name'][:9]}"
              f"({ax['odds']:.1f}倍/{wk[nums[0]]}枠) スコア{sc:.3f}")
        if lost:
            print(f"       ※過去走なし{len(lost)}頭: {'/'.join(lost[:3])}（来たら外れる）")
        trio = [f"{a}-{b}-{c}" for a, b, c in itertools.combinations(nums[:4], 3)]
        # 三連単は**2着固定**が最良((54): 6点600円・的中5.98%・配当7,958円・ROI79.3%[75,84])。
        # 軸の8割は1番人気で、1番人気の勝ちは最も買われている結果＝過剰人気。
        # 「1番人気が取りこぼす」側に賭ける2着固定の方が配当が5割増える。
        # ただし三連複BOX4(84.5%)に届かないので既定では出さない（--sanrentan で明示的に出す）。
        tan = [f"{a}-{nums[0]}-{b}" for a, b in itertools.permutations(nums[1:4], 2)]
        if soft and not soft["buy"]:
            print(f"       （甘い軸: 期待払戻{soft['e_axis']:.0f}円"
                  f"＝買う基準{soft['buy_threshold']:.0f}円以下に届かず見送り）")
        if not skip:
            print(f"       枠連 {' '.join(f'{a}-{b}' for a, b in pairs)}"
                  f"（{len(pairs)}点 {len(pairs)*100}円）"
                  + ("  ※軸と紐が同枠で1点" if len(pairs) == 1 else ""))
            # ★(80)③で最下位と確定。本命より1レース +42.9円[+32.7,+53.1] 余計に失い 0/11年。
            #   ROIで見ると85.2%対81.8%で「3.4pt差」に見えるが、4点買うので損失は5倍になる。
            #   出力自体は残す（比較のため）が、**推奨していないことを行に明記する**。
            print(f"       三連複BOX {' '.join(trio)}（4点 400円）"
                  f"　⚠買うな: 1R損失−72.7円（本命−14.8円の約5倍・0/11年）")
            if args.sanrentan:
                print(f"       三連単 2着固定 {' '.join(tan)}（6点 600円 / ROI79.3%・非推奨）")
            # ★各馬の指数。「どのくらい推奨されているか」が順位だけでは分からないため。
            #   複勝圏内確率 = モデルの生値 / レース内シェア = レース内で合計1に正規化したもの
            #   市場 = 単勝オッズ由来の含意確率（同じ集合で正規化）
            #   差 = レース内シェア − 市場（＋なら市場より高く評価している）
            mk = {u: (1 / info[u]["odds"]) for u in nums}
            msum = sum(mk.values())
            if alt is not None:
                print(f"       ── 高容量(L5)の買い目 "
                      f"{'（現行と同じ）' if alt['same_as_current'] else '★現行と違う'}")
                print(f"          枠連 {' '.join(alt['wakuren'])}"
                      f" ／ 三連複BOX {' '.join(alt['sanrenpuku_box'])}"
                      f"  軸 {alt['axis']}番{info[alt['axis']]['name'][:9]}")
            print(ST.header("      "))
            pop = ST.pop_ranks({u: info[u]["odds"] for u in nums})
            for i, u in enumerate(nums):
                print(ST.row(i + 1, u, info[u]["name"], wk[u], pmap[u],
                             float(q[i]), mk[u] / msum, info[u]["odds"], pop[u],
                             indent="      "))
            # ★買った枠の中身。枠連はその枠から**どれか1頭**が来れば当たるので、
            #   同じ枠に複数いるならそれだけ当たりやすい。馬連等に読み替えるときの材料にもなる。
            rank = {u: i + 1 for i, u in enumerate(nums)}
            axw = wk[nums[0]]
            order = [axw] + [w for pr in pairs for w in pr if w != axw]
            seen, lines = set(), []
            for w in order:
                if w in seen:
                    continue
                seen.add(w)
                mem = sorted([u for u in nums if wk[u] == w], key=lambda u: rank[u])
                lines.append(f"{'軸' if w == axw else '  '}{w}枠 "
                             + " ".join(f"{u}番{info[u]['name'][:7]}({info[u]['odds']:.1f}倍"
                                        f"/{rank[u]}位)" for u in mem))
            print("       枠の中身  " + "\n                 ".join(lines))
        out_races.append({"raceid": rc["raceid"],   # ★答え合わせの結合キー。無いと払戻と繋げない
                          "track": rc["place"], "r": rc["r"], "label": f"{rc['place']}{rc['r']}R",
                          "fieldsize": n, "axis": nums[0], "axis_name": ax["name"],
                          "axis_odds": ax["odds"], "axis_waku": wk[nums[0]], "top5": nums[:5],
                          "waku_score": round(float(sc), 5), "excluded": bool(skip),
                          "scores": [{"umaban": u, "name": info[u]["name"], "waku": wk[u],
                                      "p": round(pmap[u], 5), "share": round(float(q[i]), 5),
                                      "odds": info[u]["odds"]} for i, u in enumerate(nums)],
                          "odds_at": rc["odds_at"],
                          "wakuren": [f"{a}-{b}" for a, b in pairs], "sanrenpuku_box": trio,
                          "sanrentan_2nd": tan,
                          **({"soft_axis": soft} if soft else {}),
                          **({"l5": alt} if alt else {})})

    buy = [r for r in out_races if not r["excluded"] and not r.get("waku_na")]
    cost = sum(len(r["wakuren"]) * 100 for r in buy)
    print(f"\n購入対象 {len(buy)}レース / 判定 {len(out_races)}レース（除外 {len(out_races)-len(buy)}）")
    print(f"枠連 合計 {cost:,}円 ／ 三連複BOX4も買うなら +{len(buy)*400:,}円")
    print(f"期待損失は投下額の約15%（枠連{int(cost*0.155):,}円ぶん）。増えることは期待しない。")
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        # ★キー名は `waku_score_p20` のままにしない。(117)で除外率を40%にしたので、
        # 　中身が40%の閾値なのに名前がp20だと必ず取り違える。**実際の水準も一緒に書く**。
        json.dump({"date": str(today.date()), "excl_pct": args.excl,
                   "waku_score_th": th,
                   # ★★後から「どの設定で出した推奨か」を復元できるように残す（2026-08-16）
                   #   紐の数は(155)で1に変えた。判定時刻はオッズの取得時刻（朝/取り直し）。
                   "partners": args.partners,
                   "odds_at": max((r.get("odds_at") or "") for r in out_races) or None,
                   # ★★`p` が何の確率かを必ず書き残す（2026-08-30・ユーザー要望）。
                   #   ⚠**(168)で「枠連は上位2頭で決まるのに top3 で学習している」という
                   #   　ずれが見つかった**（測ったが差は出ず、目標は top3 のまま置いた）。
                   #   ★**もし将来この目標を変えるなら、`p` の意味が黙って変わり、
                   #   　過去の推奨ファイルまで解釈できなくなる**。だから毎回書く。
                   #   ★★**運用上の約束**: **目標を何に変えても、複勝圏内(finish<=3)の確率は
                   #   　数値として必ず残す**——**全馬を見比べるときの物差しになっているため**。
                   #   　目標を変える場合は `p` とは別に `p_fuku` を足すこと。**消さない**。
                   "p_meaning": meta.get("target", "top3 (finish<=3)"),
                   "model_dir": MODEL_DIR,
                   "races": out_races},
                  open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"保存: {args.out}")


if __name__ == "__main__":
    main()
