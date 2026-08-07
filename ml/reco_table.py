"""
記録した買い目（`data/reco/reco_*.json`）を**表**で出す。

列は `ml/score_table.py` の9列固定（順/馬番/馬名/枠/複勝圏内確率/レース内シェア/市場/差/単勝）で、
`ml/predict_nk.py` のコンソール出力と同じ定義・同じ並び。列の意味は score_table.py を見ること。

★標準ライブラリだけで動く（pandas/lightgbm 不要）。手元のMacでも回せるようにするため。

実行:
  python3 ml/reco_table.py data/reco/reco_20260802.json          # markdown（既定）
  python3 ml/reco_table.py data/reco/reco_20260802.json --text    # 等幅テキスト
  python3 ml/reco_table.py data/reco/reco_20260802.json --all     # 除外レースも出す
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import score_table as ST


def rows_of(rc):
    """(順, 馬番, 馬名, 枠, 複勝圏内確率, レース内シェア, 市場, 単勝) を上位から。"""
    sc = rc.get("scores") or []
    mkt = ST.market_shares(sc)
    for i, s in enumerate(sc, 1):
        yield (i, s["umaban"], s["name"], s["waku"], s["p"], s["share"],
               mkt[s["umaban"]], s["odds"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*", default=["data/reco/reco_*.json"])
    ap.add_argument("--text", action="store_true", help="markdownでなく等幅テキストで出す")
    ap.add_argument("--all", action="store_true", help="除外レースも出す")
    args = ap.parse_args()

    files = sorted(f for p in args.paths for f in glob.glob(p))
    if not files:
        sys.exit(f"{args.paths} に推奨JSONがありません")
    for fp in files:
        reco = json.load(open(fp, encoding="utf-8"))
        print(f"# {reco['date']}（{fp}）\n")
        for rc in reco["races"]:
            if rc.get("excluded") and not args.all:
                continue
            if not rc.get("scores"):
                print(f"## {rc['label']}　※このJSONにはスコアが入っていない"
                      "（スコア表示より前に作った版）\n")
                continue
            mark = "×除外 " if rc.get("excluded") else ""
            print(f"## {mark}{rc['label']}　軸{rc['axis']}番{rc['axis_name']}"
                  f"（枠連スコア{rc['waku_score']:.3f}）")
            print(f"枠連 {' '.join(rc.get('wakuren') or [])}"
                  f"／三連複BOX {' '.join(rc.get('sanrenpuku_box') or [])}")
            # ★(111)(112) 甘い軸の三連複。オッズだけで決まるのでJSONに無ければその場で計算する
            sa = rc.get("soft_axis")
            if sa is None and rc.get("scores"):
                try:
                    import soft_axis as SA
                    sa = SA.recommend([x["umaban"] for x in rc["scores"]],
                                      [x["odds"] for x in rc["scores"]])
                except Exception:
                    sa = None
            if sa:
                if sa.get("buy"):
                    print(f"★甘い軸 三連複 {sa['sanrenpuku']}（1点100円）"
                          f"　軸{sa['axis']}番・複勝の期待払戻{sa['e_axis']:.0f}円"
                          f"（裾{int(sa['tier']*100)}%）")
                else:
                    print(f"（甘い軸: 期待払戻{sa['e_axis']:.0f}円＝閾値100円超で見送り）")
            print()
            print(ST.header() if args.text else ST.md_header())
            for r in rows_of(rc):
                print(ST.row(*r) if args.text else ST.md_row(*r))
            print()


if __name__ == "__main__":
    main()
