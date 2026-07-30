"""
推奨の答え合わせ。predict.py が保存した推奨JSONと、レース後にTARGETから出した払戻CSVを
突き合わせて、実際の購入額・払戻・回収率を出す。毎回これを回して成績を積み上げる（印象でなく実測で管理）。

払戻CSVは2形式に自動対応（列数で判別）:
  - 配当A（224列）: 馬単/三連複/三連単を含むフルセット ← 三連単の検証にはこちら必須
  - 配当B（136列）: 単複枠連/馬連/ワイドのみ
列レイアウトは src/backtest/payout-parser.ts の冒頭コメントと同じ。

使い方:
  # 本線=枠連 軸枠×紐枠2（既定）
  python3 ml/check_result.py --reco out/reco_260801.json --payout data/payout/a.csv
  # 通算成績（複数週まとめて）
  python3 ml/check_result.py --reco "out/reco_*.json" --payout data/payout/a.csv
  # 対抗=三連複 BOX上位4 / 単価を変える
  python3 ml/check_result.py --reco "out/reco_*.json" --payout data/payout/a.csv --type sanrenpuku_box --unit 500
  # predict.py が除外したレースも含めて（除外ルールが効いていたか確認する用）
  python3 ml/check_result.py --reco "out/reco_*.json" --payout data/payout/a.csv --include-excluded
"""
import argparse, csv, io, json, glob

# 券種 → (推奨JSONのキー, 着順どおりか, 表示名, 長期実測の目安)
# 現行の predict.py が出すのは wakuren / sanrenpuku_box の2つだけ。
# 下の3つ（sanrentan/sanrenpuku/umaren）は**否定済みの旧戦略**のキーで、過去の推奨JSONの読み直し用に残す。
BETS = {
    "wakuren": ("wakuren", False, "枠連 軸枠×紐枠2",
                "OOS 28,770R実測 ROI 84.5%・的中30.0%・的中時545円（控除率22.5%＝天井77.5%を10pt上回るがプラスではない）"),
    "sanrenpuku_box": ("sanrenpuku_box", False, "三連複 BOX上位4",
                       "OOS 28,628R実測 ROI 84.5%・的中20.2%（控除率25%＝天井75%）"),
    "sanrentan": ("sanrentan_multi", True, "三連単 一軸マルチ×紐4 【旧・否定済み】",
                  "旧「実測141.9%/168R」は(25)(26)で否定（実際は56.4%）。参考値として見ないこと"),
    "sanrenpuku": ("sanrenpuku_axis", False, "三連複 軸1×紐4 【旧】", "(48)(55)で BOX上位4 に更新済み"),
    "umaren": ("umaren_axis", False, "馬連 軸1×紐4 【旧】", "旧118.3%/168Rは(26)で否定"),
    "wide": ("wide", False, "ワイド BOX4 【旧】", "実測89.2%＝負けの確率87%"),
}

def _slots(r, start, count, width, nhorses, pay_idx):
    """(馬番×nhorses, 配当) のスロット列を読む。payout>0 のものだけ返す。"""
    out = {}
    for s in range(count):
        b = start + s * width
        try:
            pay = int(r[b + pay_idx])
            horses = [int(r[b + i]) for i in range(nhorses)]
        except (ValueError, IndexError):
            continue
        if pay > 0 and all(h > 0 for h in horses):
            out[tuple(horses)] = pay
    return out

def load_payouts(paths):
    """払戻CSV群 → {(yymmdd, 場所, R): {券種: {組: 配当}}}。配当A/Bを列数で自動判別。"""
    out = {}
    for p in paths:
        with open(p, "rb") as fh:
            txt = fh.read().decode("shift_jis", "replace")
        for r in csv.reader(io.StringIO(txt)):
            if len(r) < 58:
                continue
            try:
                key = (f"{r[0].strip()}{r[1].strip().zfill(2)}{r[2].strip().zfill(2)}",
                       r[4].strip(), int(r[6] or 0))
            except ValueError:
                continue
            if len(r) >= 224:  # 配当A: 枠連103 馬連115 ワイド127 馬単155 三連複179 三連単194
                d = {"wakuren": _slots(r, 103, 3, 4, 2, 2),
                     "umaren": _slots(r, 115, 3, 4, 2, 2), "wide": _slots(r, 127, 7, 4, 2, 2),
                     "umatan": _slots(r, 155, 6, 4, 2, 2), "sanrenpuku": _slots(r, 179, 3, 5, 3, 3),
                     "sanrentan": _slots(r, 194, 6, 5, 3, 3)}
            else:              # 配当B: 枠連31 馬連40 ワイド49（3連系なし）
                d = {"wakuren": _slots(r, 31, 3, 3, 2, 2),
                     "umaren": _slots(r, 40, 3, 3, 2, 2), "wide": _slots(r, 49, 5, 3, 2, 2),
                     "umatan": {}, "sanrenpuku": {}, "sanrentan": {}}
            if any(d.values()):
                out[key] = d
    return out

# 券種名 → 払戻CSV側のキー（買い方が違っても払戻表は同じ券種を見る）
PAYKEY = {"sanrenpuku_box": "sanrenpuku"}


def key_of(combo, ordered):
    """推奨の "14-16-8" → 突合キー。着順券種は順序を保持、それ以外は昇順に正規化。"""
    nums = [int(x) for x in combo.split("-")]
    return tuple(nums) if ordered else tuple(sorted(nums))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reco", required=True, help="推奨JSON（glob可）")
    ap.add_argument("--payout", required=True, help="払戻CSV 配当A/B（glob可）")
    ap.add_argument("--only-star", action="store_true", help="旧JSON用: ★合致レースのみ集計")
    ap.add_argument("--include-excluded", action="store_true",
                    help="predict.py が枠連スコア下位20%で除外したレースも集計に含める")
    ap.add_argument("--type", choices=list(BETS), default="wakuren", help="集計する券種（既定=枠連）")
    ap.add_argument("--unit", type=int, default=100, help="1点あたりの賭け金（既定100円）")
    args = ap.parse_args()

    pays = load_payouts(sorted(glob.glob(args.payout)))
    if not pays:
        print("払戻CSVを読めませんでした（配当A/Bの形式か確認）"); return
    rkey, ordered, tname, note = BETS[args.type]
    pkey = PAYKEY.get(args.type, args.type)

    tot_bet = tot_ret = tot_hit = 0
    rows = []
    for rp in sorted(glob.glob(args.reco)):
        reco = json.load(open(rp, encoding="utf-8"))
        ymd = reco["date"].replace("-", "")[2:]  # YYYY-MM-DD → yymmdd
        for rc in reco["races"]:
            st = rc.get("status") or ("×除外" if rc.get("excluded") else "○")
            if args.only_star and not st.startswith("★"):
                continue
            if rc.get("excluded") and not args.include_excluded:
                continue
            combos = rc.get(rkey) or []
            if not combos:
                continue
            p = pays.get((ymd, rc["track"], rc["r"]))
            if p is None:
                rows.append((reco["date"], rc["label"], st, None, None, "払戻データ未取得")); continue
            wins = p[pkey]
            if not wins:
                rows.append((reco["date"], rc["label"], st, None, None, "この券種の払戻なし(配当Aが必要)")); continue
            bet = len(combos) * args.unit
            hit = [(c, wins[key_of(c, ordered)]) for c in combos if key_of(c, ordered) in wins]
            ret = sum(pay * args.unit // 100 for _, pay in hit)  # 100円あたり配当 × 口数
            tot_bet += bet; tot_ret += ret; tot_hit += len(hit)
            rows.append((reco["date"], rc["label"], st, bet, ret, hit))

    print(f"\n===== 答え合わせ: {tname}（1点{args.unit:,}円{'・★のみ' if args.only_star else ''}） =====")
    for date, lab, st, bet, ret, hit in rows:
        if bet is None:
            print(f"  {date} {lab:8s} {st:22s} {hit}"); continue
        h = " ".join(f"{c}:{p:,}円" for c, p in hit) or "的中なし"
        print(f"  {date} {lab:8s} {st:22s} 購入{bet:6,}円 払戻{ret:8,}円 ({ret/bet*100:6.1f}%)  {h}")
    if tot_bet:
        print(f"\n  通算 {len(rows)}レース: 購入{tot_bet:,}円 / 払戻{tot_ret:,}円 / "
              f"収支{tot_ret-tot_bet:+,}円 / 回収率 {tot_ret/tot_bet*100:.1f}% / 的中{tot_hit}本")
        print(f"  ※長期の目安: {note}")
        print("  ※単発の上下は無意味。数十レース単位で見るための記録です。")
    else:
        print("  集計対象がありませんでした（★が無い／払戻CSVの日付が合わない等）")

if __name__ == "__main__":
    main()
