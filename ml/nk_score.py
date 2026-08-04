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
    pat = sys.argv[1] if len(sys.argv) > 1 else "data/reco/reco_*.json"
    files = sorted(glob.glob(pat))
    if not files:
        sys.exit(f"{pat} に推奨JSONがありません")
    pays = load_pays()
    print(f"払戻: {len(pays)}レース読込\n")

    # (表示名, 推奨JSONのキー, 1点の金額, 着順どおりか)
    BETS = [("枠連", "wakuren", 100, False), ("三連複", "sanrenpuku_box", 100, False),
            ("三連単", "sanrentan_2nd", 100, True)]
    tot = {k: [0, 0, 0] for k, _, _, _ in BETS}     # 購入, 払戻, 的中本数
    for fp in files:
        reco = json.load(open(fp, encoding="utf-8"))
        print(f"=== {reco['date']}（{fp}）")
        for rc in reco["races"]:
            if rc.get("excluded"):
                continue
            p = pays.get(rc["raceid"] if "raceid" in rc else "")
            if p is None:      # 推奨JSONにraceidが無い版のため場所+Rから引けない場合がある
                p = next((v for k, v in pays.items()
                          if k.endswith(f"{rc['r']:02d}")and rc["label"][:2] in k), None)
            line = f"  {rc['label']:8s} 軸{rc['axis']:>2}番"
            for name, key, unit, ordered in BETS:
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
            print(line)

    print()
    for name, _, _, _ in BETS:
        bet, ret, hit = tot[name]
        if not bet:
            continue
        print(f"{name:<6} 購入{bet:,}円 / 払戻{ret:,}円 / 収支{ret-bet:+,}円 / "
              f"回収率{ret/bet*100:.1f}% / 的中{hit}本")
    print("\n※1日の結果に意味は無い（枠連の的中率は30%、三連複BOX4は20%）。"
          "長期の目安は両方とも84.5%。数十レース積んでから読むこと。")


if __name__ == "__main__":
    main()
