"""
netkeiba経由の払戻（`data/nk/pay*.csv`）で、記録した買い目（`data/reco/reco_*.json`）を答え合わせする。

`ml/check_result.py` は TARGET の配当A/B（固定列のCSV）を読む作りなので、netkeiba側の
tidy形式（raceid, kind, combo, payout）には対応していない。こちらはその橋渡し。

★1日の結果には意味が無い。枠連の的中率は30%、三連複BOX4は20%なので、
　当たっても外れても偶然の範囲。**数十レース積み上げてから読むこと**。
　長期の目安は 枠連84.5% / 三連複BOX上位4 84.5%（(55)(56)）。

実行: python3 ml/nk_score.py [reco glob(既定 data/reco/reco_*.json)]
"""
import csv
import glob
import json
import sys


def load_pays(pattern="data/nk/pay*.csv"):
    """{raceid: {券種: {組(tuple): 配当}}}。組は昇順に正規化（着順どおりの券種は除く）。"""
    ordered = {"馬単", "三連単"}
    out = {}
    for p in sorted(glob.glob(pattern)):
        with open(p, encoding="utf-8", newline="") as fh:
            for r in csv.DictReader(fh):
                nums = [int(x) for x in r["combo"].split("-") if x.strip().isdigit()]
                if not nums:
                    continue
                key = tuple(nums) if r["kind"] in ordered else tuple(sorted(nums))
                out.setdefault(r["raceid"], {}).setdefault(r["kind"], {})[key] = int(r["payout"])
    return out


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    pat = args[0] if args else "data/reco/reco_*.json"
    files = sorted(glob.glob(pat))
    if not files:
        sys.exit(f"{pat} に推奨JSONがありません")
    pays = load_pays()
    print(f"払戻: {len(pays)}レース読込\n")

    # (表示名, 推奨JSONのキー, 1点の金額, 着順どおりか)
    # ★三連単は**買わない**（(54)でROI79.3%＝三連複BOX4の84.5%に届かない）。
    # 　推奨JSONには常に入っているが、合計に混ぜると「買ってもいない券種の負け」が
    # 　成績に載って読み違える。--sanrentan を付けたときだけ集計する。
    BETS = [("枠連", "wakuren", 100, False), ("三連複", "sanrenpuku_box", 100, False)]
    if "--sanrentan" in sys.argv:
        BETS.append(("三連単", "sanrentan_2nd", 100, True))
    # ★並行運用((83))。--dual で予測したJSONには "l5" が入っているので、同じ集計を別枠で回す。
    #   ⚠これは**決着には使えない**。1.6ptの差を実測で捉えるには約68,000レース＝43年かかる。
    #   目的は「L5が実際に動くか」「的中率が実際に下がるか」の確認と、重大な異常の検出。
    ALT = [("枠連(L5)", "wakuren", 100, False), ("三連複(L5)", "sanrenpuku_box", 100, False)]
    tot = {k: [0, 0, 0] for k, _, _, _ in BETS}
    tot.update({k: [0, 0, 0] for k, _, _, _ in ALT})
    n_diff = n_dual = 0     # 購入, 払戻, 的中本数
    soft = [0, 0, 0, []]    # ★甘い軸の三連複1点: 購入, 払戻, 的中, 明細
    for fp in files:
        reco = json.load(open(fp, encoding="utf-8"))
        print(f"=== {reco['date']}（{fp}）")
        for rc in reco["races"]:
            # ★除外レースでも**甘い軸だけは採点する**。
            # 　以前はここで `continue` していたので、除外レースに買い目が出ると
            # 　**実際に買った1点が成績に入らない**（9頭未満で同じ穴を踏んでいる）。
            # 　(117)で除外を20%→40%にしたぶん、該当する確率はちょうど2倍になった。
            skip_bets = bool(rc.get("excluded"))
            p = pays.get(rc["raceid"] if "raceid" in rc else "")
            if p is None:      # 推奨JSONにraceidが無い版のため場所+Rから引けない場合がある
                p = next((v for k, v in pays.items()
                          if k.endswith(f"{rc['r']:02d}")and rc["label"][:2] in k), None)
            line = f"  {'×除外' if skip_bets else '    '}{rc['label']:8s} 軸{rc['axis']:>2}番"
            # ★枠連が発売されないレース（9頭未満）は、甘い軸の三連複だけを記録している。
            # 　枠連・三連複BOXの「払戻データなし」を出しても意味が無いので飛ばす。
            for name, key, unit, ordered in ([] if (rc.get("waku_na") or skip_bets) else BETS):
                combos = rc.get(key) or []
                if not combos or not p or name not in p:
                    if key != "sanrentan_2nd":   # 三連単は既定で買わないので「なし」を出さない
                        line += f"  {name}: 払戻データなし"
                    continue
                # ★三連単は**着順どおり**の券種なので組を昇順に直してはいけない
                def key_of(c, _o=ordered):
                    t = tuple(int(x) for x in c.split("-"))
                    return t if _o else tuple(sorted(t))
                bet = len(combos) * unit
                hits = [(c, p[name][key_of(c)]) for c in combos if key_of(c) in p[name]]
                ret = sum(v for _, v in hits)
                tot[name][0] += bet
                tot[name][1] += ret
                tot[name][2] += len(hits)
                line += f"  {name}: {bet}円→{ret:,}円" + (f" 的中{hits[0][0]}" if hits else "")
            # ★甘い軸の三連複1点（(112)運用）。**これが実際に買っている買い目**なのに
            # 　採点されていなかった（8/8の中京5Rが記録から漏れていた）。
            # 　★`buy` が真のレースだけ数える。年間60レースしか出ない水準なので、
            # 　　緩い裾のものを混ぜると別の戦略の成績になってしまう。
            sa = rc.get("soft_axis")
            if sa and sa.get("buy") and p and "三連複" in p:
                c = sa["sanrenpuku"]
                k = tuple(sorted(int(x) for x in c.split("-")))
                v = p["三連複"].get(k, 0)
                soft[0] += 100
                soft[1] += v
                soft[2] += v > 0
                soft[3].append((reco["date"], rc["label"], c, v))
                line += f"  ★甘い軸{c}: 100円→{v:,}円"
            l5 = None if skip_bets else rc.get("l5")
            if l5:
                n_dual += 1
                n_diff += 0 if l5.get("same_as_current") else 1
                for name, key, unit, ordered in ALT:
                    combos = l5.get(key) or []
                    base = name.replace("(L5)", "")
                    if not combos or not p or base not in p:
                        continue
                    hits = [(c, p[base][tuple(sorted(int(x) for x in c.split("-")))])
                            for c in combos
                            if tuple(sorted(int(x) for x in c.split("-"))) in p[base]]
                    tot[name][0] += len(combos) * unit
                    tot[name][1] += sum(v for _, v in hits)
                    tot[name][2] += len(hits)
                line += ("  L5:同" if l5.get("same_as_current") else "  L5:違")
            # 除外レースで甘い軸も無いなら、何も足されていないので出さない
            if not skip_bets or "★甘い軸" in line:
                print(line)

    print()
    for name, _, _, _ in BETS + (ALT if n_dual else []):
        bet, ret, hit = tot[name]
        if not bet:
            continue
        print(f"{name:<6} 購入{bet:,}円 / 払戻{ret:,}円 / 収支{ret-bet:+,}円 / "
              f"回収率{ret/bet*100:.1f}% / 的中{hit}本")
    if n_dual:
        print(f"\n※L5併記 {n_dual}レース中 {n_diff}レース（{n_diff/n_dual*100:.0f}%）で買い目が現行と違う。"
              "★この比較で優劣は決着しない（1.6ptの差には約68,000レース＝43年必要）。"
              "見ているのは『L5が動くか』『的中率が実際に下がるか』だけ")
    # ★甘い軸の三連複（(112)運用）。**現在ほんとうに買っているのはこれだけ**
    if soft[0]:
        bet, ret, hit, rows = soft
        print(f"\n★甘い軸の三連複1点（裾2%・買うと判定した分だけ）")
        for d, lab, c, v in rows:
            print(f"   {d} {lab:8s} {c:10s} {'的中 ' + format(v, ',') + '円' if v else '外れ'}")
        print(f"   購入{bet:,}円 / 払戻{ret:,}円 / 収支{ret-bet:+,}円 / "
              f"回収率{ret/bet*100:.1f}% / 的中{hit}/{bet//100}本")
        print("   ⚠**年間60レースしか出ない**。(111)の実測は656レースで99%CI[76.0,116.0]＝"
              "判定不能。**数年かけて積む標本**であって、今の数字は読まないこと。")
    else:
        print("\n★甘い軸の三連複: 買うと判定したレースはまだ無い"
              "（年間60レース程度なので、出ない週が正常）")

    print("\n※1日の結果に意味は無い（枠連の的中率は30%、三連複BOX4は20%）。"
          "長期の目安は両方とも84.5%。数十レース積んでから読むこと。")


if __name__ == "__main__":
    main()
