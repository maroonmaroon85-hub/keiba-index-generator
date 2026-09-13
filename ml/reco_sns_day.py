"""(242sns) ★★★**その日のSNS用の印を出す**（★ズレ下限を SNS 用に緩める）

★**利用者の指定（2026-09-12）**:
　**「規則側はレース選定が厳しくて、SNSはある程度数を載せたいので選定を緩めたい」**
　★**(240sns)の梯子から ★ズレ下限 0.10 を採用**（**1日1.22本 → 4.63本**）。

■ ⚠★★**これは何を変え、何を変えないか**
| | |
|---|---|
| ★**変える** | ★**ズレの下限だけ**（**0.15 → `SNS_GAP`**） |
| ★**変えない** | **pn ≥ 0.15 / ★単勝 ≥ 10.0倍 / その中で pn 最大の1頭**＝**◎は穴馬のまま** |
| ⚠**触らない** | ★**`ml/reco_ana_day.py` を1行も書き換えない**（**規則側の当日モード＝前向き標本の道具**） |
　★**やり方: `reco_ana_day` を import して、★モジュール変数 `GAP` だけ差し替える**。

■ ⚠⚠★★★**この出力に付けてはいけない数字**
　★**これは「朝9時の単勝＋朝9時の複凝板＋固定モデル」で作った軸**であり、
　★**さらにズレの下限も下げてある**。→ ⚠⚠**11年の凍結値は★二重に当てはまらない**。
　⚠**投稿に「複勝23.5%」「ROI113.4%」等を添えないこと**（`ANA_SNS_RULE.md` §8）。
　★**G=0.10 の11年実測は (240sns)**: **複勝 94.2% / 的中21.7%**。
　　⚠**ただしそれも確定オッズでの値**で、**朝9時版ではない**。

■ ★出力
　★**◎＝軸 / ○▲△＝紐P（モデル順）**。⚠**買い目の羅列は出さない**（**利用者の指定**）。
　★**該当0本なら「見送り」**。

実行: python3 ml/reco_sns_day.py 20260912 [--gap 0.10]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SNS_GAP = 0.10


def is_jump(rc):
    """★★(243sns) 障害戦か

    ★★**(245sns) 2026-09-13 訂正（★穴馬セッションの指摘）**:
    　★**`entries` の `surface` に ★"障" が入っている**——**実物で確認済み**
    　（**9/12 阪神1R 障2970m / 9/13 中山1R 障2880m**）。★**これが正確**。
    　⚠**初版は距離だけで見ていた**（**100の倍数でない or 3600m超**）——
    　　★**障3100/3200/3300m のような切りの良い障害を★取りこぼす**。
    　★**`surface` を正とし、距離は★それが取れないときの保険にする**。
    ⚠**11年の遡及測定（`audit_ana_jump.py`）は距離判定のまま**——
    　★**生データのトラック種別は 芝/ダ の2値で「障」が無いため**（`features.py:39`）。
    　★**あちらは取りこぼす方向に外れる**と明記してある。
    """
    if str(rc.get("surface") or "").strip() == "障":
        return True                     # ★これが正
    try:
        dist = float(rc.get("distance") or 0)
    except (TypeError, ValueError):
        return False
    return dist > 0 and (dist % 100 != 0 or dist > 3600)   # ★保険


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        sys.exit("日付が要る: python3 ml/reco_sns_day.py 20260912 [--gap 0.10]")
    gap_v = SNS_GAP
    for i, a in enumerate(sys.argv):
        if a == "--gap" and i + 1 < len(sys.argv):
            gap_v = float(sys.argv[i + 1])

    import reco_ana_day as D
    frozen, frozen_le = D.GAP, D.load_entries
    D.GAP = gap_v          # ★ここだけ差し替える（⚠ファイルは1行も書き換えない）

    # ★★(243sns) 障害戦を★入口で落とす（⚠これも D のファイルは書き換えない）
    #   ★**理由はROIではない**——**モデルが障害を「芝」として学習しているため
    #   （生データに『障』が無い）。★障害では軸が2倍立つのに複勝は来ない
    #   （24.0→14.3% / 22.0→17.4%）＝★ズレは妙味ではなくモデルの誤差**。
    dropped = []

    def _load_entries_flat(path):
        rs = frozen_le(path)
        keep = []
        for rc in rs:
            (dropped if is_jump(rc) else keep).append(rc)
        return keep

    D.load_entries = _load_entries_flat

    print(f"(242sns) ★★★**SNS用**：★**ズレの下限 {frozen} → {gap_v}**"
          f"（**(240sns)で選んだ水準・1日1.22本→4.63本**）")
    print("★**他の条件は据え置き**: **pn ≥ 0.15 / 単勝 ≥ 10.0倍 / pn最大の1頭**"
          "　→ ★**◎は穴馬のまま**")
    print("⚠⚠**この出力に11年の凍結値を添えないこと**"
          "——**朝9時の板＋固定モデル＋下限変更で★二重に別の量**\n")
    try:
        # ★★向こうの main() をそのまま呼ぶ（★スコアリングを複製しない）
        #   ⚠**役割欄に「★軸（穴馬）／紐1／紐2／紐3」が出るので、そこから印を作る**
        rc0 = D.main()
    finally:
        D.GAP, D.load_entries = frozen, frozen_le   # ★必ず戻す（★他を汚さない）
    if dropped:
        print(f"\n⚠★**障害戦を {len(dropped)}レース除外した**（(243sns)）:")
        for rc in dropped:
            print(f"　　{rc['place']}{rc['r']:>2}R {rc.get('name','')[:16]}"
                  f"　{rc.get('surface','')}{rc.get('distance','')}m")
        print("　★**理由はROIではない**——**モデルが障害を「芝」として学習しているため**"
              "（**生データに『障』が無い**）。")
        print("　★**障害では軸が約2倍立つのに複勝は来ない**"
              "（**24.0→14.3% / 22.0→17.4%**）＝★**ズレは妙味ではなくモデルの誤差**。")
    return rc0


if __name__ == "__main__":
    sys.exit(main())
