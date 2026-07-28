"""
未来レース推論。学習済みモデル(ml/model/)で、出馬表(DG…画面イメージCSV・レースデータ行頭付き)
＋各馬の過去走(DS…全馬成績)から各馬の win_prob を出し、レースごとに買い目を出力する。
実オッズ・実条件(芝ダ/距離/クラス)入りDGなら、研究の好ポケット(gap≥0.20 & 人気馬2-5倍 & ダート)で全レースを一括選抜する。

DG形式(レースデータ先頭行付き・項目名レース毎):
  各レースが [年,月,日,場所,R,略レース名,条件表記,芝・ダート,距離,頭数,…] の列名行＋値行、
  続いて [枠番,B,番,…,斤量,減M,単勝,…] の列名行＋各馬行。

使い方:
  python3 ml/predict.py --shutuba data/upcoming/DG260725.CSV --seiseki data/upcoming/DS260725.CSV
"""
import json, argparse, csv, io
import numpy as np
import pandas as pd
import lightgbm as lgb
import features as F

MODEL_DIR = "ml/model"
GAP_TH = 0.20

# 好ポケット168Rの実測から求めたケリー最適f（1レースに資金の何%を賭けるか）と対数成長。
# 券種ごとに (最適f, 対数成長, 点数, 表示名)。predict はこれを分数ケリーで割り引いて金額を出す。
KELLY = {
    "sanrentan": (0.026, 4.21, 36, "三連単 一軸マルチ"),
    "sanrenpuku": (0.023, 2.20, 6, "三連複 軸1×紐4"),
    "umaren": (0.021, 1.69, 4, "馬連 軸1×紐4"),
}

def stake_line(key, bankroll, kfrac, points):
    """分数ケリーでの賭け金と1点あたりを返す。資金不足なら理由つきで警告。"""
    f, growth, _, name = KELLY[key]
    if bankroll <= 0:
        return f"({points}点/{points*100}円・成長{growth:.2f})"
    target = bankroll * f * kfrac          # 理論上の投下額
    minimum = points * 100                  # 100円×点数（JRAの最小単位）
    if target < minimum:
        need = int(minimum / (f * kfrac))
        return (f"({points}点・成長{growth:.2f}) ⚠資金不足: 適正{int(target):,}円 < 最低{minimum:,}円"
                f"／この券種には資金{need:,}円必要")
    per = int(target / points / 100) * 100  # 1点あたり(100円単位に丸め)
    total = per * points
    return f"({points}点 × {per:,}円 = {total:,}円・成長{growth:.2f})"

def _f(x):
    try: return float(str(x).strip())
    except: return np.nan
def _i(x):
    try: return int(str(x).strip())
    except: return 0

def parse_shutuba(path):
    """DG → レース配列。各要素 {meta:{track,r,surface,distance,cls,field}, horses:[{umaban,name,sex,age,wtcarry,odds}]}。"""
    with open(path, "rb") as fh:
        txt = fh.read().decode("shift_jis", "replace")
    rows = [r for r in csv.reader(io.StringIO(txt)) if r]
    races, cur = [], None
    for r in rows:
        c0 = str(r[0]).strip()
        if c0 == "年":      # レースデータ列名行
            continue
        if c0 == "枠番":    # 各馬列名行
            continue
        if c0.isdigit() and len(c0) == 4 and len(r) >= 10:  # レースデータ値行(年=2026)
            if cur: races.append(cur)
            cur = {"meta": {"track": str(r[3]).strip(), "r": _i(r[4]), "cls": str(r[5]).strip(),
                            "surface": str(r[7]).strip(), "distance": _i(r[8]), "field": _i(r[9])},
                   "horses": []}
            continue
        if cur is not None and c0.isdigit():  # 各馬行(先頭=枠番の数値)
            cur["horses"].append({"umaban": _i(r[2]), "name": str(r[7]).strip(), "sex": str(r[9]).strip(),
                                  "age": _i(r[10]), "wtcarry": _f(r[13]), "odds": _f(r[15])})
    if cur: races.append(cur)
    return races

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shutuba", required=True)
    ap.add_argument("--seiseki", required=True)
    ap.add_argument("--topbox", type=int, default=4)
    ap.add_argument("--date", required=True, help="開催日 YYYY-MM-DD（出馬表の日付）")
    ap.add_argument("--out", default=None, help="推奨をJSON保存するパス（後日 check_result.py で答え合わせ）")
    ap.add_argument("--bankroll", type=int, default=0, help="資金(円)。指定すると分数ケリーで各券種の賭け金を出す")
    ap.add_argument("--kelly", type=float, default=0.5, help="分数ケリー係数(既定0.5=半ケリー。1/4なら0.25)")
    args = ap.parse_args()

    model = lgb.Booster(model_file=f"{MODEL_DIR}/model.txt")
    cat_maps = json.load(open(f"{MODEL_DIR}/cat_maps.json"))
    cols = json.load(open(f"{MODEL_DIR}/feature_cols.json"))

    races = parse_shutuba(args.shutuba)
    raw = pd.read_csv(args.seiseki, header=None, encoding="shift_jis", encoding_errors="replace", dtype=str, keep_default_na=False)
    d = F.to_model(raw)
    name2reg = dict(zip(raw[13].str.strip(), raw[37].str.strip()))
    last_of = d.sort_values("date").groupby("horse").tail(1).set_index("horse")
    TODAY = pd.Timestamp(args.date)

    synth, tags = [], []
    for ri, rc in enumerate(races):
        m = rc["meta"]
        surf = 1 if m["surface"] == "ダート" else 0
        for h in rc["horses"]:
            reg = name2reg.get(h["name"])
            if reg is None or reg not in last_of.index:
                continue
            lr = last_of.loc[reg]
            if isinstance(lr, pd.DataFrame): lr = lr.iloc[0]
            synth.append({
                "date": TODAY, "course": m["track"], "surface": surf,
                "distance": m["distance"] or lr["distance"], "cond": "良", "horse": reg,
                "sex": h["sex"] or lr["sex"], "age": h["age"] or lr["age"],
                "wtcarry": h["wtcarry"] if not np.isnan(h["wtcarry"]) else lr["wtcarry"],
                "fieldsize": len(rc["horses"]), "finish": np.nan, "margin": np.nan,
                "passavg": np.nan, "agari": np.nan, "prize": np.nan,
                "raceid": f"S{ri:02d}", "umaban": h["umaban"],
                "sire": lr["sire"], "damsire": lr["damsire"],
                "raceclass": F._classcode(m["cls"]), "odds": h["odds"],
                "finratio": np.nan, "passratio": np.nan,
            })
            tags.append((ri, h["umaban"], h["name"], h["odds"]))

    synth = pd.DataFrame(synth)
    d2 = pd.concat([d, synth], ignore_index=True).sort_values(["horse", "date"]).reset_index(drop=True)
    f = F.build_features(d2)
    fx = F.encode_categoricals(f, cat_maps)[cols]
    is_today = d2["date"].eq(TODAY).values
    d2 = d2.assign(pred=model.predict(fx.values))
    today = d2[is_today]

    print(f"\n===== {args.date} 予想（ML win_prob / 実オッズ・実条件） =====")
    print(f"※★=好ポケット合致（gap≥{GAP_TH:.2f} & 本命2-5倍 & ダート）。◎gap=1番人気の抜け具合")
    if args.bankroll:
        kn = {1.0: "フルケリー", 0.5: "半ケリー", 0.25: "1/4ケリー"}.get(args.kelly, f"{args.kelly}ケリー")
        print(f"※資金{args.bankroll:,}円・{kn}で賭け金を算出。1レースあたりの額なので、"
              f"同日に複数★が出たらその数だけ資金を使う点に注意")
    print()
    pockets = []
    for ri, rc in enumerate(races):
        m = rc["meta"]; lab = f"{m['track']}{m['r']}R"
        sub = today[today["raceid"] == f"S{ri:02d}"].copy()
        info = {t[1]: (t[2], t[3]) for t in tags if t[0] == ri}
        if len(sub) < 2:
            print(f"  {lab:8s} {m['surface']}{m['distance']} {m['cls']}  予想不可(新馬/過去走なし)"); continue
        sub["wp"] = sub["pred"] / sub["pred"].sum()
        sub = sub.sort_values("wp", ascending=False)
        wp = sub["wp"].values; gap = float(wp[0] - wp[1])
        favN = int(sub.iloc[0]["umaban"]); favOdds = info.get(favN, ("", np.nan))[1]
        dirt = m["surface"] == "ダート"
        top = [int(x) for x in sub.head(args.topbox)["umaban"]]
        pairs = [f"{a}-{b}" for i, a in enumerate(top) for b in top[i+1:]]
        # 馬連 軸1×紐4(4点)。紐は全券種で①②③④に統一（実行しやすさ）。
        # 紐3(119.5%/成長1.22)より紐4(118.3%/成長1.69)の方が的中頻度が上がり分散が減るため成長率は上。
        axis = None  # 下で紐(t4)確定後に組む
        # 本線=三連単 一軸マルチ×相手4(36点)。好ポケット実測141.9%、ケリー対数成長4.21で最大。
        # 対抗=三連複 軸1×相手4(6点)。実測121.3%・対数成長2.20。資金が少ない場合はこちら。
        t4 = [int(x) for x in sub.head(5)["umaban"]][1:5]
        trio = [f"{top[0]}-{t4[i]}-{t4[j]}" for i in range(len(t4)) for j in range(i+1, len(t4))]
        hlist = t4  # 紐＝モデル2-5位（優先順）
        axis = [f"{top[0]}-{h}" for h in t4]  # 馬連 軸×紐①②③④
        tan = [f"{a}-{b}-{c}" for i, a in enumerate([top[0]] + t4) for j, b in enumerate([top[0]] + t4)
               for k, c in enumerate([top[0]] + t4)
               if len({a, b, c}) == 3 and top[0] in (a, b, c) and not (a != top[0] and b != top[0] and c != top[0])]
        favnm = info.get(favN, ("",""))[0][:7]
        has_odds = favOdds == favOdds
        od = f"{favOdds:.1f}倍" if has_odds else "オッズ未取得"
        # ポケット入口: gap≥閾値 & ダート。オッズがあれば2-5倍で確定/除外、無ければ「要確認」。
        entry = gap >= GAP_TH and dirt
        if entry:
            if has_odds:
                status = "★合致" if 2 <= favOdds < 5 else ("除外(堅すぎ)" if favOdds < 2 else "除外(人気薄)")
            else:
                status = "△要オッズ確認(2-5倍なら合致)"
            pockets.append((lab, favN, favnm, od, gap, pairs, status, m["track"], m["r"], axis, trio, tan, hlist))
        star = "◆" if entry else ("・" if gap >= GAP_TH else "  ")
        print(f"{star}{lab:8s} {m['surface']}{m['distance']} {m['cls'][:6]:6s} 頭{len(sub):2d} gap{gap*100:2.0f}% ◎{favN}{favnm}({od})")
    print(f"\n===== ◆好ポケット入口(gap≥{GAP_TH:.2f}&ダート)のレース: {len(pockets)} =====")
    for lab, favN, nm, od, gap, pairs, status, _tr, _r, axis, trio, tan, hlist in pockets:
        print(f"  {lab} {status}  ◎{favN}{nm}({od}) gap{gap*100:.0f}%")
        # 本線=馬連 軸1×相手3。実測119.5%(168R)・ブートストラップでプラスの見込み69%。
        # 参考=ワイドBOX4。実測89.2%だが「負け」の確率87%＝安定して負ける。分散は小さい。
        # 軸＋紐の優先順位だけ表示（組み合わせ全列挙は見づらいので出さない）。
        himo = " ".join(f"{chr(9312+i)}{h}" for i, h in enumerate(hlist))
        print(f"      軸 {favN}  ／  紐(優先順) {himo}")
        print(f"      ★本線     三連単 一軸マルチ 軸{favN}×紐①②③④ {stake_line('sanrentan', args.bankroll, args.kelly, len(tan))}")
        print(f"      ○対抗     三連複 軸{favN}×紐①②③④ {stake_line('sanrenpuku', args.bankroll, args.kelly, len(trio))}")
        print(f"      ・参考     馬連 軸{favN}×紐①②③④ {stake_line('umaren', args.bankroll, args.kelly, len(axis))}")
    if not pockets:
        print("  なし")

    # 穴馬候補: モデル上位4以内 かつ 単勝10-20倍。
    # 実測(OOS17万頭)の裏付け: 20倍超は単勝58-64%/複勝63-78%で最悪＝買わない。
    # 10-20倍×モデル上位は単勝~79-88%/複勝~79-84%と穴の中では最も減りにくい帯。
    # 単体で儲かる訳ではないので、ワイドの「相手」に混ぜて配当を伸ばす用途を想定。
    print("\n===== 穴馬候補（モデル上位4以内 × 単勝10-20倍 / ワイド相手向き） =====")
    found = 0
    for ri, rc in enumerate(races):
        m = rc["meta"]; lab = f"{m['track']}{m['r']}R"
        sub = today[today["raceid"] == f"S{ri:02d}"].copy()
        if len(sub) < 4: continue
        sub["wp"] = sub["pred"] / sub["pred"].sum()
        sub = sub.sort_values("wp", ascending=False).head(4)
        info = {t[1]: (t[2], t[3]) for t in tags if t[0] == ri}
        for rank, row in enumerate(sub.itertuples(), 1):
            u = int(row.umaban); nm, od = info.get(u, ("", np.nan))
            if od == od and 10 <= od < 20:
                print(f"  {lab:8s} {m['surface']}{m['distance']}  {u}番 {nm}({od:.1f}倍) モデル{rank}位 wp{row.wp*100:.0f}%")
                found += 1
    if not found:
        print("  なし（オッズ未取得のレースは判定不可。オッズ確定後に再実行）")

    # 推奨をJSON保存（後日 ml/check_result.py で実払戻と突合＝成績の記録を残す）。
    if args.out:
        reco = {"date": args.date, "topbox": args.topbox,
                "races": [{"track": tr, "r": r, "label": lab, "status": st,
                           "fav": favN, "fav_name": nm, "fav_odds": od,
                           "gap": round(gap, 4), "himo": hlist, "sanrentan_multi": tan,
                           "umaren_axis": axis, "sanrenpuku_axis": trio, "wide": pairs}
                          for lab, favN, nm, od, gap, pairs, st, tr, r, axis, trio, tan, hlist in pockets]}
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(reco, fh, ensure_ascii=False, indent=1)
        print(f"\n推奨を保存: {args.out}（レース後に ml/check_result.py で答え合わせ）")

if __name__ == "__main__":
    main()
