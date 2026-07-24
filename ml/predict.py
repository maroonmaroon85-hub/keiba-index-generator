"""
未来レース推論。学習済みモデル(ml/model/)で、出馬表(DG…画面イメージCSV)＋各馬の過去走(DS…全馬成績)
から各馬の win_prob を出し、レースごとにワイドの買い目(上位4頭ボックス=6点)を出力する。

前提/割り切り:
- 前日で「今回の距離・馬場・クラス・オッズ」は未確定。→ 距離/馬場/クラスは各馬の前走から補完。
  （支配的特徴は血統＋近走なので概ね有効。条件系は小さい寄与。オッズ確定後に再実行すると精度向上）
- 過去走がDSに無い馬(新馬/初出走)はモデル対象外→ win_prob 0 として扱う。

使い方:
  python3 ml/predict.py --shutuba DG260725.CSV --seiseki data/upcoming/DS260725.CSV
"""
import sys, json, argparse
import numpy as np
import pandas as pd
import lightgbm as lgb
import features as F

MODEL_DIR = "ml/model"

def parse_shutuba(path):
    """DG(出馬表・画面イメージCSV, ヘッダ行でレース区切り) → レース配列。各要素は出走馬dictのリスト。"""
    dg = pd.read_csv(path, header=None, encoding="shift_jis", dtype=str, keep_default_na=False)
    races, cur = [], []
    for _, r in dg.iterrows():
        if str(r[0]).startswith("枠番"):
            if cur: races.append(cur); cur = []
            continue
        cur.append({
            "umaban": F_int(r[2]), "name": str(r[7]).strip(),
            "sex": str(r[9]).strip(), "age": F_int(r[10]), "wtcarry": F_num(r[13]),
        })
    if cur: races.append(cur)
    return races

def F_int(x):
    try: return int(str(x).strip())
    except: return 0
def F_num(x):
    try: return float(str(x).strip())
    except: return np.nan

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shutuba", required=True)
    ap.add_argument("--seiseki", required=True)
    ap.add_argument("--topbox", type=int, default=4, help="ワイドの上位Nボックス(既定4→6点)")
    args = ap.parse_args()

    model = lgb.Booster(model_file=f"{MODEL_DIR}/model.txt")
    cat_maps = json.load(open(f"{MODEL_DIR}/cat_maps.json"))
    cols = json.load(open(f"{MODEL_DIR}/feature_cols.json"))

    races = parse_shutuba(args.shutuba)
    raw = pd.read_csv(args.seiseki, header=None, encoding="shift_jis", encoding_errors="replace", dtype=str, keep_default_na=False)
    d = F.to_model(raw)  # 各馬の過去走(結果あり)。horse(=血統登録番号),date昇順。

    # 出馬表は馬名しか無い→ 馬名→血統登録番号(col37) の対応表でDS(登録番号キー)に橋渡し。
    name2reg = dict(zip(raw[13].str.strip(), raw[37].str.strip()))
    # 登録番号→過去走の最新行（血統・前走条件の補完元）。
    last_of = d.sort_values("date").groupby("horse").tail(1).set_index("horse")
    TODAY = pd.Timestamp("2026-07-25")

    # 各出走馬の「当日行」を合成（前走から距離/馬場/クラス/血統を補完）。
    synth = []
    tags = []  # (race_idx, umaban, name)
    for ri, rc in enumerate(races):
        for h in rc:
            nm = h["name"]
            reg = name2reg.get(nm)
            if reg is None or reg not in last_of.index:
                continue  # 過去走なし(新馬等)→対象外
            lr = last_of.loc[reg]
            if isinstance(lr, pd.DataFrame): lr = lr.iloc[0]
            synth.append({
                "date": TODAY, "course": lr["course"], "surface": lr["surface"],
                "distance": lr["distance"], "cond": "良", "horse": reg,
                "sex": h["sex"] or lr["sex"], "age": h["age"] or lr["age"],
                "wtcarry": h["wtcarry"] if not np.isnan(h["wtcarry"]) else lr["wtcarry"],
                "fieldsize": len(rc), "finish": np.nan, "margin": np.nan,
                "passavg": np.nan, "agari": np.nan, "prize": np.nan,
                "raceid": f"S{ri:02d}", "umaban": h["umaban"],
                "sire": lr["sire"], "damsire": lr["damsire"],
                "raceclass": np.nan,  # 当日クラス不明→欠損(class_changeも欠損=モデルは他で補う)
                "odds": np.nan, "finratio": np.nan, "passratio": np.nan,
            })
            tags.append((ri, h["umaban"], nm))

    synth = pd.DataFrame(synth)
    d2 = pd.concat([d, synth], ignore_index=True).sort_values(["horse", "date"]).reset_index(drop=True)
    f = F.build_features(d2)
    fx = F.encode_categoricals(f, cat_maps)[cols]

    # 合成(当日)行だけ抽出して予測。
    is_today = d2["date"].eq(TODAY).values
    pred = model.predict(fx.values)
    d2 = d2.assign(pred=pred)
    today = d2[is_today].copy()
    today["key"] = list(zip(today["raceid"], today["umaban"]))

    # レースごとに正規化(win_prob) → ワイド上位ボックス。gap≥0.20 のレースを選抜(研究の好ポケット)。
    GAP_TH = 0.20
    print(f"\n===== 2026-07-25 予想（ML win_prob / 条件は前走補完・オッズ未確定） =====")
    print(f"※races {len(races)}  対象馬 {len(today)}（過去走ありのみ）")
    print(f"※★=1番人気が抜けたレース(gap≥{GAP_TH:.2f})＝研究上の好ポケット。ダート＆人気馬2-5倍が重なると更に良(オッズ確定後に判定可)\n")
    picks = []
    for ri, rc in enumerate(races):
        sub = today[today["raceid"] == f"S{ri:02d}"].copy()
        if len(sub) < 2:
            print(f"[R{ri+1:02d}] 過去走ありの馬が少なく予想不可（新馬/未勝利の可能性）"); continue
        sub["wp"] = sub["pred"] / sub["pred"].sum()
        sub = sub.sort_values("wp", ascending=False)
        wp = sub["wp"].values
        gap = float(wp[0] - wp[1])
        star = "★" if gap >= GAP_TH else "  "
        names = {int(t[1]): t[2] for t in tags if t[0] == ri}
        rank = "  ".join(f"{int(r.umaban)}{names.get(int(r.umaban),'')[:6]}({r.wp*100:.0f}%)" for r in sub.head(6).itertuples())
        top = [int(r.umaban) for r in sub.head(args.topbox).itertuples()]
        pairs = [f"{a}-{b}" for i, a in enumerate(top) for b in top[i+1:]]
        print(f"{star}[R{ri+1:02d}] {len(rc)}頭 gap{gap*100:.0f}%  上位: {rank}")
        print(f"      ワイド上位{args.topbox}頭ボックス({len(pairs)}点): {'  '.join(pairs)}\n")
        if gap >= GAP_TH:
            picks.append((ri+1, gap, top, pairs))
    print(f"===== ★選抜レース（gap≥{GAP_TH:.2f}・1番人気が抜けた回）: {len(picks)}レース =====")
    for r, g, top, pairs in sorted(picks, key=lambda x:-x[1]):
        print(f"  R{r:02d} (gap{g*100:.0f}%)  ワイド: {'  '.join(pairs)}")
    print("\n注: これは順位付けベースの買い目。ブランケットで全部買うと長期回収率~77%(負け)。")
    print("    土曜朝に単勝オッズが出たら、★の中で『ダート & 人気馬2-5倍』に更に絞ると実測92%台まで上がる。")

if __name__ == "__main__":
    main()
