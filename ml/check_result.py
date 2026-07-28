"""
推奨の答え合わせ。predict.py が保存した推奨JSONと、レース後にTARGETから出した払戻CSV(配当B)を
突き合わせて、実際の購入額・払戻・回収率を出す。毎回これを回して成績を積み上げる（自己申告でなく実測で管理）。

配当Bの列（payout-parser.ts と同じ）:
  col0-2=年月日(yy,mm,dd) col4=場所名 col6=R  ワイド=col49から (馬番,馬番,配当)×5スロット

使い方:
  python3 ml/check_result.py --reco out/reco_260725.json --payout data/payout/haraimodoshiB_260725.csv
  # 複数まとめて通算成績:
  python3 ml/check_result.py --reco "out/reco_*.json" --payout "data/payout/haraimodoshiB_*.csv"
"""
import argparse, csv, io, json, glob

def load_wide(paths):
    """払戻CSV群 → {(yymmdd, 場所, R): {"a-b": 配当}}。"""
    out = {}
    for p in paths:
        with open(p, "rb") as fh:
            txt = fh.read().decode("shift_jis", "replace")
        for r in csv.reader(io.StringIO(txt)):
            if len(r) < 58:
                continue
            key = (f"{r[0].strip()}{r[1].strip().zfill(2)}{r[2].strip().zfill(2)}", r[4].strip(), int(r[6] or 0))
            wins = {}
            for s in range(5):  # ワイドは col49 から 3列×5スロット
                b = 49 + s * 3
                try:
                    u1, u2, pay = int(r[b]), int(r[b + 1]), int(r[b + 2])
                except ValueError:
                    continue
                if u1 > 0 and u2 > 0 and pay > 0:
                    wins["-".join(map(str, sorted((u1, u2))))] = pay
            um = {}
            for s in range(3):  # 馬連は col40 から 3列×3スロット
                b = 40 + s * 3
                try:
                    u1, u2, pay = int(r[b]), int(r[b + 1]), int(r[b + 2])
                except ValueError:
                    continue
                if u1 > 0 and u2 > 0 and pay > 0:
                    um["-".join(map(str, sorted((u1, u2))))] = pay
            if wins or um:
                out[key] = {"wide": wins, "umaren": um}
    return out

def norm(combo):
    return "-".join(sorted(combo.split("-"), key=int))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reco", required=True, help="推奨JSON（glob可）")
    ap.add_argument("--payout", required=True, help="配当B CSV（glob可）")
    ap.add_argument("--only-star", action="store_true", help="★合致レースのみ集計")
    ap.add_argument("--type", choices=["umaren", "wide"], default="umaren",
                    help="集計する買い目（既定=umaren:本線の馬連 軸1×相手3 / wide:参考のワイドBOX4）")
    args = ap.parse_args()

    wide = load_wide(sorted(glob.glob(args.payout)))
    if not wide:
        print("払戻CSVを読めませんでした（配当Bの形式か確認）"); return

    tot_bet = tot_ret = 0
    rows = []
    for rp in sorted(glob.glob(args.reco)):
        reco = json.load(open(rp, encoding="utf-8"))
        ymd = reco["date"].replace("-", "")[2:]  # YYYY-MM-DD → yymmdd
        for rc in reco["races"]:
            if args.only_star and not rc["status"].startswith("★"):
                continue
            pay = wide.get((ymd, rc["track"], rc["r"]))
            if pay is None:
                rows.append((rc["label"], rc["status"], None, None, None)); continue
            wins = pay["umaren" if args.type == "umaren" else "wide"]
            combos = rc.get("umaren_axis", []) if args.type == "umaren" else rc["wide"]
            bet = len(combos) * 100
            hit = [(c, wins[norm(c)]) for c in combos if norm(c) in wins]
            ret = sum(p for _, p in hit)
            tot_bet += bet; tot_ret += ret
            rows.append((rc["label"], rc["status"], bet, ret, hit))

    tname = "馬連 軸1×相手3(本線)" if args.type == "umaren" else "ワイドBOX4(参考)"
    print(f"\n===== 推奨の答え合わせ（{tname}/{'★合致のみ' if args.only_star else '全推奨'}） =====")
    for lab, st, bet, ret, hit in rows:
        if bet is None:
            print(f"  {lab:8s} {st:24s} 払戻データ未取得"); continue
        h = " ".join(f"{c}:{p}円" for c, p in hit) or "的中なし"
        print(f"  {lab:8s} {st:24s} 購入{bet:5d}円 払戻{ret:6d}円 ({ret/bet*100:5.1f}%)  {h}")
    if tot_bet:
        print(f"\n  通算: 購入{tot_bet}円 / 払戻{tot_ret}円 / 収支{tot_ret-tot_bet:+}円 / 回収率 {tot_ret/tot_bet*100:.1f}%")
        exp = ("実測119.5%/168R・プラスの見込み69%(95%CI 62-195%)＝分散大"
               if args.type == "umaren" else "実測89.2%だが負けの確率87%＝安定して負ける")
        print(f"  ※好ポケットの長期実測は{exp}。単発の上下に一喜一憂しないための記録です。")

if __name__ == "__main__":
    main()
