"""
netkeiba の出馬表JSON（`ml/nk_fetch.py entries` の出力）から買い目を出す。

`ml/predict.py` は TARGET の DG 形式を読む。こちらは同じ推奨ロジックを netkeiba 経由の
入力に対して行う版で、**モデルも買い方も判定も predict.py と同一**（`ml/model_prod/`）:
  ・本命 枠連 軸枠×紐枠2（平均194円・ROI 84.5%・的中30.0%・的中時545円）
  ・対抗 三連複 BOX上位4（400円・ROI 84.5%・的中20.2%・的中時1,674円）
  ・枠連スコア（積ベース）下位20%は除外（(62)。回収率を上げる絞りは存在しない）
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
from nk_link import build_map
from predict import MODEL_DIR, MIN_FIELD, load_model
from waku_umatan import bracket_probs, waku_of, waku_score, wakuren_buy


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("entries")
    ap.add_argument("--date", default=None, help="開催日 YYYY-MM-DD（既定はファイル名から）")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-exclude", action="store_true")
    ap.add_argument("--sanrentan", action="store_true",
                    help="三連単(2着固定×紐3・6点600円)も出す。★ROI79.3%%で三連複BOX4(84.5%%)より悪い")
    args = ap.parse_args()
    ymd = args.date or os.path.basename(args.entries).replace("entries", "")[:8]
    today = pd.Timestamp(f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:8]}") if len(ymd) == 8 else pd.Timestamp(args.date)

    boosters, cat_maps, cols, meta = load_model()
    th = meta["waku_score_p20"]
    races = load_entries(args.entries)
    print(f"モデル: {MODEL_DIR}（{meta['target']} / シード{meta['n_seed']}本平均 / "
          f"{meta['date_from']}〜{meta['date_to']}）")
    print(f"出馬表: {args.entries} {len(races)}レース / 対象日 {today.date()}")
    print("過去走を読み込み中…")

    frames = [F.load_files()] + [
        pd.read_csv(p, header=None, encoding="shift_jis", encoding_errors="replace",
                    dtype=str, keep_default_na=False) for p in sorted(glob.glob("data/nk/DSnk*.CSV"))]
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

    print(f"\n===== {today.date()} 買い目（枠連 軸枠×紐枠2 ／ 三連複 BOX上位4） =====")
    print(f"※枠連スコア下位20%（<{th:.4f}）は除外。ROI84.5%＝**長期で投下額の15.5%が減る**。1日の結果に意味は無い")
    out_races = []
    for rc in races:
        head = (f"{rc['place']}{rc['r']:>2}R {rc['surface']}{rc['distance']:<5} "
                f"{rc['name'][:10]:10s} {len(rc['horses'])}頭 馬場{rc['cond']}")
        g = sub[sub["raceid"] == rc["raceid"]]
        lost = miss.get(rc["raceid"], [])
        if len(rc["horses"]) < MIN_FIELD:
            print(f"    {head}  対象外（{MIN_FIELD}頭未満は枠連の発売なし）")
            continue
        if len(g) < 3 or not np.isfinite(g["odds"].to_numpy(float)).all():
            print(f"    {head}  予想不可（過去走のある馬{len(g)}頭 / オッズ欠損）")
            continue
        gg = g.sort_values("p", ascending=False, kind="mergesort")
        nums = gg["umaban"].astype(int).tolist()
        n = len(rc["horses"])
        wk = {h["umaban"]: h["waku"] for h in rc["horses"]}
        info = {h["umaban"]: h for h in rc["horses"]}
        pairs = sorted({tuple(sorted((wk[nums[0]], wk[h]))) for h in nums[1:3]})
        praw = gg["p"].to_numpy(float)
        pmap = {u: float(v) for u, v in zip(nums, praw)}
        q = praw / praw.sum()
        bp = {}
        for u, pv in zip(nums, q):
            bp[wk[u]] = bp.get(wk[u], 0.0) + float(pv)
        sc = waku_score(pairs, bp)
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
        if not skip:
            print(f"       枠連 {' '.join(f'{a}-{b}' for a, b in pairs)}"
                  f"（{len(pairs)}点 {len(pairs)*100}円）"
                  + ("  ※軸と紐が同枠で1点" if len(pairs) == 1 else ""))
            print(f"       三連複BOX {' '.join(trio)}（4点 400円）")
            if args.sanrentan:
                print(f"       三連単 2着固定 {' '.join(tan)}（6点 600円 / ROI79.3%・非推奨）")
            # ★各馬の指数。「どのくらい推奨されているか」が順位だけでは分からないため。
            #   複勝圏内確率 = モデルの生値 / レース内シェア = レース内で合計1に正規化したもの
            #   市場 = 単勝オッズ由来の含意確率（同じ集合で正規化）
            #   差 = レース内シェア − 市場（＋なら市場より高く評価している）
            mk = {u: (1 / info[u]["odds"]) for u in nums}
            msum = sum(mk.values())
            print(ST.header("      "))
            for i, u in enumerate(nums):
                print(ST.row(i + 1, u, info[u]["name"], wk[u], pmap[u],
                             float(q[i]), mk[u] / msum, info[u]["odds"], indent="      "))
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
                          "sanrentan_2nd": tan})

    buy = [r for r in out_races if not r["excluded"]]
    cost = sum(len(r["wakuren"]) * 100 for r in buy)
    print(f"\n購入対象 {len(buy)}レース / 判定 {len(out_races)}レース（除外 {len(out_races)-len(buy)}）")
    print(f"枠連 合計 {cost:,}円 ／ 三連複BOX4も買うなら +{len(buy)*400:,}円")
    print(f"期待損失は投下額の約15%（枠連{int(cost*0.155):,}円ぶん）。増えることは期待しない。")
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump({"date": str(today.date()), "waku_score_p20": th, "races": out_races},
                  open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"保存: {args.out}")


if __name__ == "__main__":
    main()
