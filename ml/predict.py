"""
未来レース推論（実運用スクリプト）。出馬表(DG)＋各馬の過去走(DS)＋手元の全過去データから、
検証で最良だった構成の買い目を出す。

**採用している構成**（HANDOFF.md 冒頭・(45)(48)(55)(62)(63)）:
  ・モデル … `ml/model_prod/`（**オッズ入り・top3目標・シード平均**）。作るのは `ml/train_prod.py`。
    ※`ml/model/`（train.py の産物）は**オッズなし・win目標**で別物。こちらは使わない。
  ・本命   … **枠連 軸枠×紐枠2**（平均194円・ROI 84.5%・的中30.0%・的中時545円）
  ・対抗   … **三連複 BOX上位4**（400円・ROI 84.5%・的中20.2%・的中時1,674円）
  ・絞り込み … (62)より**回収率を上げる絞りは存在しない**。使えるのは
    「枠連スコア（積ベース）下位20%を**除外**する」だけ（84.5%→84.7%）。
  ・賭け金 … 全券種でROI<100%＝**ケリー最適f=0%**。金額の算出はしない（娯楽費として決めるほかない）。

**期待値の前提を誤解しないための注記**:
  ROI 84.5% は**長期的に15.5%ずつ減る**という意味。控除率22.5%の枠連で市場を10pt以上上回っているが、
  それでも**プラスにはならない**。「損失を小さくしつつ楽しむ」ための出力であり、増やすためのものではない。

**未来レースで劣化する要素（正直に書いておく）**:
  1. **オッズ** … 学習・検証は**確定オッズ**。前日/直前オッズで買えば買い目は変わる。
     (59)の感度分析ではσ=0.2で軸が約8%のレースで入れ替わったが、**本当の検証は未実施**。
  2. **馬体重** … DGには無いので欠損のまま。学習時は有った特徴なのでその分は不利。
  3. **馬場状態** … DGには無いので「良」固定。ただし(63)より**モデルは馬場を一度も使っていない**
     （重要度0.00%・書き換えても買い目の変化0.0%）ので実質影響なし。
  4. **騎手/調教師の集計** … 手元の全過去データ(*.CSV)を母集団にして計算する。DSだけだと母数が
     足りず学習時と分布がずれるため、このスクリプトは**過去データ全部を読み込む**（実行に2分前後）。

出力対象は**9頭立て以上**のみ。枠連は8頭以下では**発売されない**（実データで確認: 8頭以下1,944R中0R）。

使い方:
  python3 ml/train_prod.py                     # 先にモデルを作る（データが増えたら再実行）
  python3 ml/predict.py --shutuba data/upcoming/DG260725.CSV \
                        --seiseki data/upcoming/DS260725.CSV \
                        --date 2026-07-25 --out out/reco_260725.json
  # レース後: python3 ml/check_result.py --reco out/reco_260725.json --payout data/payout/a.csv --type wakuren
"""
import argparse
import csv
import io
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features as F
from waku_umatan import bracket_probs, waku_of, waku_score, wakuren_buy

MODEL_DIR = "ml/model_prod"
MIN_FIELD = 9        # 枠連の発売下限。三連複BOX4の実測もこの範囲。


def _f(x):
    try:
        return float(str(x).strip())
    except ValueError:
        return np.nan


def _i(x):
    try:
        return int(str(x).strip())
    except ValueError:
        return 0


def parse_shutuba(path):
    """DG（画面イメージCSV・レースデータ行頭付き）→ レース配列。

    構造は「レースデータ列名行＋値行 → 各馬列名行＋各馬行」の繰り返し。
    ★列は固定ではない: TARGETの出力設定によって列が増減し、**レースごとに列名行が付く**。
      実データでも 単勝オッズ列がある race（23列）と無い race（22列）が同じファイルに混在する。
    そのため**列番号ではなく列名で引く**。単勝が無ければ odds=NaN（そのレースは予想不可になる）。
    """
    with open(path, "rb") as fh:
        txt = fh.read().decode("shift_jis", "replace")
    rows = [r for r in csv.reader(io.StringIO(txt)) if r]
    races, cur, rh, hh = [], None, None, None

    def pick(r, hdr, name, cast=None, default=""):
        try:
            v = r[hdr.index(name)].strip()
        except (ValueError, IndexError):
            return default if cast is None else cast("")
        return v if cast is None else cast(v)

    for r in rows:
        c0 = str(r[0]).strip()
        if c0 == "年":                    # レースデータ列名行
            rh = [c.strip() for c in r]
            continue
        if c0 == "枠番":                  # 各馬列名行（レースごとに出る）
            hh = [c.strip() for c in r]
            continue
        if rh and c0.isdigit() and len(c0) == 4 and len(r) >= 10:   # レースデータ値行(年=2026)
            if cur:
                races.append(cur)
            cur = {"meta": {"track": pick(r, rh, "場所"), "r": pick(r, rh, "R", _i),
                            "cls": pick(r, rh, "略レース名"),
                            "surface": pick(r, rh, "芝・ダート"),
                            "distance": pick(r, rh, "距離", _i), "field": pick(r, rh, "頭数", _i)},
                   "horses": []}
            continue
        if cur is not None and hh and c0.isdigit():                  # 各馬行(先頭=枠番)
            cur["horses"].append({
                "waku": _i(c0), "umaban": pick(r, hh, "番", _i), "name": pick(r, hh, "馬名"),
                "sex": pick(r, hh, "性別"), "age": pick(r, hh, "年齢", _i),
                "jockey": pick(r, hh, "騎手"), "wtcarry": pick(r, hh, "斤量", _f),
                "odds": pick(r, hh, "単勝", _f), "trainer": pick(r, hh, "調教師")})
    if cur:
        races.append(cur)
    return races


def load_model():
    """ml/model_prod/ を読む。シードごとのモデルを全部返し、予測は平均を取る（(30)のシード差対策）。"""
    if not os.path.exists(f"{MODEL_DIR}/meta.json"):
        sys.exit(f"{MODEL_DIR}/meta.json が無い。先に `python3 ml/train_prod.py` を実行してください。")
    meta = json.load(open(f"{MODEL_DIR}/meta.json"))
    boosters = [lgb.Booster(model_file=f"{MODEL_DIR}/{n}") for n in meta["models"]]
    cat_maps = json.load(open(f"{MODEL_DIR}/cat_maps.json"))
    cols = json.load(open(f"{MODEL_DIR}/feature_cols.json"))
    return boosters, cat_maps, cols, meta


def build_today(races, seiseki, date, hist_pattern):
    """DG＋DS＋全過去データ → (build_features済みの特徴, 今日の行のメタ)。

    今日の行は「結果が空の1走」として過去走の後ろに足す（近走特徴が同じロジックで作られる）。
    馬場は「良」固定・馬体重は欠損（DGに無い）。どちらも docstring の注記どおり。
    """
    raw_ds = pd.read_csv(seiseki, header=None, encoding="shift_jis",
                         encoding_errors="replace", dtype=str, keep_default_na=False)
    raw_hist = F.load_files(hist_pattern)
    raw = pd.concat([raw_hist, raw_ds], ignore_index=True)
    d = F.to_model(raw)                     # raceid+horse で重複除去されるのでDSの重複は問題ない
    name2reg = dict(zip(raw_ds[13].str.strip(), raw_ds[37].str.strip()))
    last_of = d.sort_values("date").groupby("horse").tail(1).set_index("horse")
    today = pd.Timestamp(date)

    synth, missing = [], {}
    for ri, rc in enumerate(races):
        m = rc["meta"]
        surf = 1 if m["surface"] == "ダート" else 0
        for h in rc["horses"]:
            reg = name2reg.get(h["name"])
            if reg is None or reg not in last_of.index:
                missing.setdefault(ri, []).append(h["name"])
                continue
            lr = last_of.loc[reg]
            if isinstance(lr, pd.DataFrame):
                lr = lr.iloc[0]
            synth.append({
                "date": today, "course": m["track"], "surface": surf,
                "distance": m["distance"] or lr["distance"], "cond": "良", "horse": reg,
                "sex": h["sex"] or lr["sex"], "age": h["age"] or lr["age"],
                "wtcarry": h["wtcarry"] if h["wtcarry"] == h["wtcarry"] else lr["wtcarry"],
                "fieldsize": len(rc["horses"]), "finish": np.nan, "margin": np.nan,
                "passavg": np.nan, "agari": np.nan, "prize": np.nan,
                "raceid": f"S{ri:02d}", "umaban": h["umaban"],
                "jockey": h["jockey"] or str(lr["jockey"]), "trainer": h["trainer"] or str(lr["trainer"]),
                "bodywt": np.nan, "sire": lr["sire"], "damsire": lr["damsire"],
                "raceclass": F._classcode(m["cls"]), "odds": h["odds"],
                "finratio": np.nan, "passratio": np.nan,
            })
    d2 = pd.concat([d, pd.DataFrame(synth)], ignore_index=True)
    d2 = d2.sort_values(["horse", "date"]).reset_index(drop=True)
    f = F.build_features(d2)
    return d2, f, missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shutuba", required=True, help="出馬表 DG*.CSV")
    ap.add_argument("--seiseki", required=True, help="対象馬の過去走 DS*.CSV")
    ap.add_argument("--date", required=True, help="開催日 YYYY-MM-DD")
    ap.add_argument("--out", default=None, help="推奨をJSON保存（後日 check_result.py で答え合わせ）")
    ap.add_argument("--hist", default="*.CSV", help="騎手/血統集計に使う過去データのglob（既定=カレントの*.CSV）")
    ap.add_argument("--no-exclude", action="store_true", help="枠連スコア下位20%の除外をしない")
    args = ap.parse_args()

    boosters, cat_maps, cols, meta = load_model()
    th = meta["waku_score_p20"]
    print(f"モデル: {MODEL_DIR}（{meta['target']} / シード{meta['n_seed']}本平均 / "
          f"{meta['date_from']}〜{meta['date_to']} {meta['rows']:,}行）")

    races = parse_shutuba(args.shutuba)
    print(f"出馬表: {len(races)}レース。過去データを読み込み中（2分前後）…")
    d2, f, missing = build_today(races, args.seiseki, args.date, args.hist)

    is_today = d2["date"].eq(pd.Timestamp(args.date)).to_numpy() & d2["raceid"].str.startswith("S").to_numpy()
    fx = F.encode_categoricals(f[is_today], cat_maps)
    sub = d2.loc[is_today, ["raceid", "umaban", "odds", "fieldsize"]].copy()
    odds = sub["odds"].to_numpy(float)
    inv = 1.0 / odds
    fx["log_odds"] = np.log(odds)
    fx["mkt_prob"] = inv / pd.Series(inv).groupby(sub["raceid"].to_numpy()).transform("sum").to_numpy()
    sub["p"] = np.mean([b.predict(fx[cols].values) for b in boosters], axis=0)

    info = {(ri, h["umaban"]): h for ri, rc in enumerate(races) for h in rc["horses"]}
    out_races = []
    print(f"\n===== {args.date} 買い目（枠連 軸枠×紐枠2 ／ 三連複 BOX上位4） =====")
    print(f"※枠連スコア下位20%（<{th:.4f}）は「買ってはいけないレース」として除外（(62)）"
          f"{'…今回は--no-excludeで無効' if args.no_exclude else ''}")
    for ri, rc in enumerate(races):
        m = rc["meta"]
        lab = f"{m['track']}{m['r']}R"
        head = f"{lab:8s} {m['surface']}{m['distance']} {m['cls'][:8]:8s} {len(rc['horses'])}頭"
        g = sub[sub["raceid"] == f"S{ri:02d}"]
        lost = missing.get(ri, [])
        if len(rc["horses"]) < MIN_FIELD:
            print(f"  {head}  対象外（{MIN_FIELD}頭未満は枠連の発売がない）")
            continue
        if len(g) < 3:
            print(f"  {head}  予想不可（過去走のある馬が{len(g)}頭しかない）")
            continue
        if not np.isfinite(g["odds"].to_numpy(float)).all() or (g["odds"] <= 0).any():
            print(f"  {head}  予想不可（単勝オッズ未取得。オッズはモデルの主要特徴）")
            continue
        gg = g.sort_values("p", ascending=False, kind="mergesort")
        nums = gg["umaban"].astype(int).tolist()
        n = len(rc["horses"])
        # 枠番はDGの値を使う（頭数からの導出 waku_of と一致するか検算し、違えばDG優先で警告）
        wk = {u: info[(ri, u)]["waku"] for u in nums}
        bad = [u for u in nums if wk[u] and wk[u] != waku_of(u, n)]
        pairs = sorted({tuple(sorted((wk[nums[0]], wk[h]))) for h in nums[1:3]})
        bp = bracket_probs(nums, gg["p"].to_numpy(), n)
        # 枠のモデル確率はDGの枠番で組み直す（bracket_probs は導出枠なので、ずれる場合に備える）
        if bad:
            bp = {}
            q = gg["p"].to_numpy(float)
            q = q / q.sum()
            for u, pv in zip(nums, q):
                bp[wk[u]] = bp.get(wk[u], 0.0) + float(pv)
        sc = waku_score(pairs, bp)
        skip = (not args.no_exclude) and sc < th
        mark = "×除外" if skip else "○"
        top4 = nums[:4]
        trio = [f"{a}-{b}-{c}" for a, b, c in itertools.combinations(top4, 3)]
        ax = info[(ri, nums[0])]
        print(f"{mark:4s}{head}  軸 {nums[0]}番{ax['name'][:8]}({ax['odds']:.1f}倍/{wk[nums[0]]}枠)  "
              f"スコア{sc:.3f}")
        if bad:
            print(f"      ⚠DGの枠番と頭数からの導出が不一致（DG優先）: 馬番 {bad}")
        if lost:
            print(f"      ※過去走なしで順位付け対象外 {len(lost)}頭: {'/'.join(lost[:4])}"
                  f"（検証時も同じ扱い＝この馬が来たら外れる）")
        if not skip:
            print(f"      枠連 {' '.join(f'{a}-{b}' for a, b in pairs)}"
                  f"（{len(pairs)}点 {len(pairs)*100}円）"
                  + ("  ※軸と紐が同枠のため1点" if len(pairs) == 1 else ""))
            print(f"      三連複BOX {' '.join(trio)}（4点 400円）")
            print(f"      順位 " + " ".join(
                f"{i+1}:{u}番({info[(ri, u)]['odds']:.1f})" for i, u in enumerate(nums[:5])))
        out_races.append({
            "track": m["track"], "r": m["r"], "label": lab, "fieldsize": n,
            "surface": m["surface"], "distance": m["distance"], "cls": m["cls"],
            "axis": nums[0], "axis_name": ax["name"], "axis_odds": ax["odds"],
            "axis_waku": wk[nums[0]], "top5": nums[:5],
            "waku_score": round(float(sc), 5), "excluded": bool(skip),
            "wakuren": [f"{a}-{b}" for a, b in pairs], "sanrenpuku_box": trio,
        })

    buy = [r for r in out_races if not r["excluded"]]
    cost = sum(len(r["wakuren"]) * 100 for r in buy)
    print(f"\n購入対象 {len(buy)}レース / 判定対象 {len(out_races)}レース"
          f"（除外 {len(out_races)-len(buy)}）")
    print(f"枠連の合計 {cost:,}円 ／ 三連複BOX4も買う場合は +{len(buy)*400:,}円")
    print("※長期の目安はどちらもROI 84.5%＝**この投下額の約15%が期待損失**。増えることは期待しない。")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump({"date": args.date, "model": meta["date_to"], "waku_score_p20": th,
                   "races": out_races}, open(args.out, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        print(f"\n推奨を保存: {args.out}"
              f"（レース後に `python3 ml/check_result.py --reco {args.out} "
              f"--payout data/payout/a.csv --type wakuren`）")


if __name__ == "__main__":
    main()
