"""★★★★開催日の朝の一括 —— 3セッションぶんを1コマンドで出す

    python3 ml/raceday.py 20260913

■ ★何をするか
| | 中身 | 呼ぶもの |
|---|---|---|
| ① | **本命（枠連 軸枠×紐枠1）＋甘い軸の三連複の候補** | `ml/predict_nk.py` |
| ② | **穴馬（自分用・P馬単M4点＋X三連単A4点＝8点800円）** | `ml/reco_ana_day.py` |
| ③ | **SNS（印・ズレ下限0.10）** | `ml/reco_sns_day.py` |

■ ⚠★★**判定ロジックには一切触らない**
　★**このファイルは「呼ぶ・束ねる・記録する」だけ**。**軸も紐も買い目も閾値も上の3本のまま**。
　★`reco_sns_day.py` が `reco_ana_day` を書き換えずに import しているのと同じ流儀で、
　**フック（`load_morn_boards` / `tickets`）を差し込んで、出た買い目を控えるだけ**にしてある。
　→ ★**前向き標本（`ANA_RULE.md` §5-2・2026-09-12開始）は切れない**。

■ ★★なぜ要るか（**2026-09-12 に見つけた穴**）
　⚠**`reco_ana_day.py` は買い目を画面に出すだけで、どこにも保存していなかった**。
　★`ANA_RULE.md` が「2026-09-12 から前向き標本を貯める」と宣言しているのに
　**`data/reco/ana_forward.csv` は1行も存在しない**。★**朝の買い目を凍結しないと夜に採点できない**。
　→ ★**このスクリプトが `data/raceday/<日付>/tickets.json` に凍結する**。**夜の採点はそれだけを読む**。

■ ★出力
```
data/raceday/<日付>/brief.txt     ★人が読むぶん（画面に出したものそのまま）
data/raceday/<日付>/tickets.json  ★機械が読むぶん（★夜の採点の唯一の入力）
data/reco/reco_<日付>.json        ★★枠連側の正典（`ml/nk_score.py` が読む）
```
⚠★**`data/reco/reco_<日付>.json` を止めないこと**。★**そこが (112) の標本の入口**。

■ ⚠★**朝に一本化できないもの＝甘い軸の三連複の「確定」**
　★**穴馬・SNSは朝9時で凍結**（`ANA_RULE.md` §3。★変えたら標本が切れる）。
　★**甘い軸の三連複は発走30分前〜10分前**（`NEXT_SESSION.md` §7・(112運用②)）。
　→ ★**時刻が違うので朝の一括には入れられない**。**朝は候補まで**。
　　★**直前に手元で `python3 ml/nk_race.py <日付> <場> <R>` を叩いて確定させること**。

■ ★前提（無ければ止まって、Macで叩くコマンドを出す）
| 要るもの | 作るもの | どこで |
|---|---|---|
| `data/nk/entries<日付>.json` | `ml/nk_fetch.py entries` | ★**Mac** |
| `data/nk_odds_morn/place<日付>.jsonl` | `ml/nk_place_morn.py` | ★**Mac** |
| `ml/model_prod/` | `ml/train_prod.py` | ★**クラウド**（★無ければ自動で作る・約2分） |
| `ml/model_ana_fwd/` | `ml/train_ana_fwd.py` | ★**コミット済み**（作り直さない＝標本が切れる） |

実行: python3 ml/raceday.py 20260913 [--no-sync] [--sns-gap 0.10]
"""
import argparse
import contextlib
import datetime
import io
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

OUTDIR = "data/raceday"
ENTRIES = "data/nk/entries{ymd}.json"
MORN = "data/nk_odds_morn/place{ymd}.jsonl"

# ★取り込み元（★研究は各セッションのまま。★ここは「読んで取り込む」だけ）
UPSTREAM = [
    ("研究（枠連・本命）", "claude/handoff-env-check-2kexpo"),
    ("穴馬（自分用）", "claude/determined-allen-2s07uf"),
    ("SNS", "claude/admiring-gauss-o5grnz"),
]


def sh(*cmd, check=False):
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if check and r.returncode:
        raise RuntimeError(f"{' '.join(cmd)}\n{r.stdout}\n{r.stderr}")
    return r


# ---------------------------------------------------------------- 取り込み
def sync(lines):
    """★3ブランチを取り込む。⚠**コンフリクトしたら中止して、そのまま報告する**（勝手に潰さない）。"""
    if sh("git", "status", "--porcelain").stdout.strip():
        lines.append("　⚠**作業ツリーが汚れているので取り込みを飛ばした**"
                     "（`git status` を見て、コミットしてから `--no-sync` 無しで叩き直す）")
        return
    for name, br in UPSTREAM:
        f = sh("git", "fetch", "origin", f"{br}:refs/remotes/origin/{br}")
        if f.returncode:
            lines.append(f"　⚠**{name} の fetch に失敗**（{br}）: {f.stderr.strip()[:120]}")
            continue
        behind = sh("git", "rev-list", "--count", f"HEAD..origin/{br}").stdout.strip()
        if behind in ("", "0"):
            lines.append(f"　{name}　取り込み済み（新しいコミットなし）")
            continue
        m = sh("git", "merge", "--no-edit", f"origin/{br}")
        if m.returncode:
            sh("git", "merge", "--abort")
            lines.append(f"　⚠★**{name} の取り込みがコンフリクトした（中止した）**。"
                         f"**手で `git merge origin/{br}` を解いてから叩き直すこと**")
            continue
        lines.append(f"　★{name}　**{behind}コミット取り込んだ**（{br}）")


# ---------------------------------------------------------------- 前提
def need(ymd, lines):
    ok = True
    ep, mp = ENTRIES.format(ymd=ymd), MORN.format(ymd=ymd)
    if not os.path.exists(os.path.join(ROOT, ep)):
        lines.append(f"　⚠★**{ep} が無い** → ★**Macで** `python3 ml/nk_fetch.py entries {ymd}`")
        ok = False
    if not os.path.exists(os.path.join(ROOT, mp)):
        lines.append(f"　⚠★**{mp} が無い** → ★**Macで** `python3 ml/nk_place_morn.py {ymd}`")
        lines.append("　　⚠**朝9時の板は後から遡れない**（`ANA_RULE.md` §6）。"
                     "**取り損ねた開催日は穴馬・SNSが永久に出せない**")
        ok = False
    if not os.path.exists(os.path.join(ROOT, "ml/model_ana_fwd/meta.json")):
        lines.append("　⚠★**ml/model_ana_fwd が無い**（★コミット済みのはず）。"
                     "**作り直すと前向き標本が切れる**ので、先に `git status` を見ること")
        ok = False
    if not os.path.exists(os.path.join(ROOT, "ml/model_prod/meta.json")):
        lines.append("　★**ml/model_prod が無いので作る**（.gitignore なので新しい環境では毎回・約2分）")
        r = subprocess.run([sys.executable, "ml/train_prod.py"], cwd=ROOT,
                           capture_output=True, text=True)
        tail = [x for x in r.stdout.splitlines() if x.strip()][-3:]
        lines += ["　　" + x for x in tail]
        if r.returncode:
            lines.append("　⚠★**train_prod.py が落ちた**"); ok = False
    return ok


# ---------------------------------------------------------------- ①本命
def run_waku(ymd, outdir):
    """`predict_nk.py` をそのまま呼ぶ。→ (画面出力, 構造化した買い目)

    ⚠★★**出力先は `data/reco/reco_<日付>.json`**（★**枠連側の統計の正典**）。
    　★`ml/nk_score.py` が `data/reco/reco_*.json` を読んで (112) の標本を数えるので、
    　**ここに書かないと記録が静かに止まる**。
    　⚠**2026-09-13 に実際に止めた**——`data/raceday/<日付>/waku.json` にだけ書いていて、
    　　**9/13 が正典に入っていなかった**（**枠連側の指摘で発覚・同日に埋め戻した**）。
    """
    wj = f"data/reco/reco_{ymd}.json"
    r = subprocess.run([sys.executable, "ml/predict_nk.py", ENTRIES.format(ymd=ymd),
                        "--out", wj], cwd=ROOT, capture_output=True, text=True)
    text = r.stdout + (("\n" + r.stderr) if r.returncode else "")
    if r.returncode or not os.path.exists(os.path.join(ROOT, wj)):
        return text, None
    d = json.load(open(os.path.join(ROOT, wj), encoding="utf-8"))
    waku, soft = [], []
    for rc in d["races"]:
        if rc.get("wakuren") and not rc.get("excluded"):
            waku.append({"raceid": rc["raceid"], "label": rc["label"],
                         "axis": rc.get("axis"), "axis_name": rc.get("axis_name"),
                         "combos": list(rc["wakuren"]), "cost": 100 * len(rc["wakuren"])})
        s = rc.get("soft_axis")
        if s and s.get("buy"):
            soft.append({"raceid": rc["raceid"], "label": rc["label"], "axis": s["axis"],
                         "combo": s["sanrenpuku"], "e_axis": s["e_axis"],
                         "tier": s["tier"], "cost": 100})
    return text, {"odds_at": d.get("odds_at", ""), "model": d.get("model_dir", ""),
                  "wakuren": waku, "soft_sanrenpuku": soft}


# ---------------------------------------------------------------- ②③穴馬・SNS
def run_ana(ymd, sns, gap=None):
    """`reco_ana_day.main()` をそのまま走らせ、**出た買い目だけ控える**。

    ★フックは2つ。**どちらも値を変えず、通り道で見ているだけ**:
      `load_morn_boards` → 走査中のレースIDを拾う（main が `boards.get(rid)` を必ず通る）
      `tickets`          → 確定した買い目（軸・紐・組）を拾う
    ⚠**SNS側は `reco_sns_day` に `GAP` と障害戦の除外をやらせる**（★こちらでは真似しない）。
    """
    import numpy as np
    import reco_ana_day as D
    rec, cur = {"races": []}, {"rid": None}
    st = {"boards": {}, "ctx": {}}          # ★板の時刻 と 軸判定の材料（★記録用・値は変えない）
    by_kind = {k: name for name, _hk, k, _n in D.BUY}

    orig_lmb, orig_t, orig_ax = D.load_morn_boards, D.tickets, D.axis_and_himo

    def lmb(y):
        boards, p = orig_lmb(y)
        st["boards"] = boards               # ★{rid: (板, fetched_at)}

        class B(dict):
            def get(self, k, *a):
                cur["rid"] = k
                return dict.get(self, k, *a)
        return B(boards), p

    def hook_ax(ub, od, pv, board):
        """⚠**値は一切変えない**。★G紐（ズレ降順）とQ紐（単勝昇順）を作る材料を控えるだけ。"""
        r = orig_ax(ub, od, pv, board)
        ax, _op, _X, _pn, _qp, gap = r
        if ax is not None:
            st["ctx"][cur["rid"]] = {"ub": [int(u) for u in ub],
                                     "od": [float(x) for x in od],
                                     "gap": [float(x) for x in gap], "ax": int(ax)}
        return r

    def hook(kind, npt, ax, himo):
        out = orig_t(kind, npt, ax, himo)
        if out:
            r = next((x for x in rec["races"] if x["raceid"] == cur["rid"]), None)
            if r is None:
                r = {"raceid": cur["rid"], "axis": int(ax), "tickets": []}
                rec["races"].append(r)
            combos = [[int(x) for x in s] for _, s in out]
            used = []                       # ★実際に買い目に出た紐（出た順・軸を除く）
            for c in combos:
                for x in c:
                    if x != int(ax) and x not in used:
                        used.append(x)
            r["tickets"].append({
                "label": by_kind.get(kind, kind), "kind": kind, "n": npt,
                "himo": used,                              # ★この券種が実際に使った紐
                "himo_rank": [int(h) for h in himo[:4]],   # ★並べ方の上位4頭（Pならモデル順）
                "combos": combos, "cost": 100 * len(out)})
        return out

    D.load_morn_boards, D.tickets, D.axis_and_himo = lmb, hook, hook_ax
    argv, buf = sys.argv, io.StringIO()
    try:
        if sns:
            import reco_sns_day as S
            gap = S.SNS_GAP if gap is None else gap   # ★既定はSNS側が持っている値
        sys.argv = ["reco_sns_day.py" if sns else "reco_ana_day.py", ymd]
        if sns:
            sys.argv += ["--gap", str(gap)]
        with contextlib.redirect_stdout(buf):
            if sns:
                S.main()
            else:
                D.main()
    except SystemExit as e:                       # ★入力が足りないときは main が exit する
        if e.code:
            buf.write(f"\n⚠{e.code}\n")
    finally:
        sys.argv = argv
        D.load_morn_boards, D.tickets, D.axis_and_himo = orig_lmb, orig_t, orig_ax

    # ---- ★板の時刻（`ANA_MERGE_HANDOFF.md` Q1-a・★今は復元できないので必ず残す）
    for r in rec["races"]:
        b = st["boards"].get(r["raceid"])
        r["board_at"] = b[1] if b else ""

    # ---- ★「記録だけ」の G馬単M4点 / Q三連単A4点（`ANA_MERGE_HANDOFF.md` Q3）
    #   ⚠**買い目も軸も変えない**。★`ANA_RULE.md` §4-2 が「記録対象は6本」と書いているのに
    #   　当日モードが P紐 と X紐 しか作らないため、G と Q が記録できていなかった。
    #   ★**軸は `reco_ana_day` が決めたものをそのまま使い、紐の並べ替えだけここで作る**。
    if not sns:
        for r in rec["races"]:
            c = st["ctx"].get(r["raceid"])
            if not c:
                continue
            ub, od, gap, ax = (np.array(c["ub"]), np.array(c["od"]),
                               np.array(c["gap"]), c["ax"])
            G = [int(u) for u in ub[np.argsort(-gap, kind="mergesort")] if int(u) != ax]
            Q = [int(u) for u in ub[np.argsort(od, kind="mergesort")] if int(u) != ax]
            r["record_only"] = []
            for label, kind, npt, himo in (("G馬単M4点", "馬単M", 4, G),
                                           ("Q三連単A4点", "三連単A", 4, Q)):
                t = orig_t(kind, npt, ax, himo)
                if not t:
                    continue
                r["record_only"].append({
                    "label": label, "kind": kind, "n": npt, "himo": himo[:3],
                    "combos": [[int(x) for x in sel] for _, sel in t],
                    "cost": 100 * len(t), "buy": False})

    rec["gap"] = gap if sns else D.GAP
    return buf.getvalue(), rec


def entries_index(ymd):
    """→ {raceid: {場・R・レース名・馬場距離・馬番→(馬名, 単勝), 馬番→人気}}。★投稿の materials。"""
    p = os.path.join(ROOT, ENTRIES.format(ymd=ymd))
    if not os.path.exists(p):
        return {}
    d = json.load(open(p, encoding="utf-8"))
    out = {}
    for rc in (d.get("races", []) if isinstance(d, dict) else d):
        hs = rc.get("horses") or []
        name, odds = {}, {}
        for h in hs:
            try:
                u = int(h["umaban"])
            except (KeyError, TypeError, ValueError):
                continue
            name[u] = h.get("name", "")
            try:
                odds[u] = float(h["odds"])
            except (KeyError, TypeError, ValueError):
                pass
        pop = {u: i + 1 for i, u in enumerate(sorted(odds, key=lambda k: odds[k]))}
        out[rc["raceid"]] = {"place": rc["place"], "r": rc["r"], "name": rc.get("name", ""),
                             "surface": rc.get("surface", ""), "distance": rc.get("distance", ""),
                             "name_of": name, "odds_of": odds, "pop_of": pop}
    return out


def sns_post(races, idx, ymd):
    """★そのまま貼れる投稿文。★ハッシュタグは #<場><YYMMDD>。

    ⚠**数字は◎の人気だけ**（`ANA_SNS_RULE.md` §8「買い目の羅列は出さない」「凍結値を添えない」）。
    """
    out = []
    for rc in races:
        e = idx.get(rc["raceid"])
        if not e:
            continue
        blk = [f"【{e['place']}{e['r']}R {e['name']}】{e['surface']}{e['distance']}m"]
        for k in ("◎", "○", "▲", "△"):
            u = rc["marks"].get(k)
            if u is None:
                continue
            line = f"{k} {u} {e['name_of'].get(u, '')}"
            if k == "◎":
                pop = e["pop_of"].get(u)
                line += f"  {pop}番人気" if pop else ""
            blk.append(line)
        blk.append("")
        blk.append(f"#{e['place']}{ymd[2:]}")
        out.append("\n".join(blk))
    return "\n\n".join(out)


def label_map(ymd):
    p = os.path.join(ROOT, ENTRIES.format(ymd=ymd))
    if not os.path.exists(p):
        return {}
    d = json.load(open(p, encoding="utf-8"))        # ★entries は素のリスト
    out = {}
    for rc in (d.get("races", []) if isinstance(d, dict) else d):
        out[rc["raceid"]] = f"{rc['place']}{rc['r']:>2}R {rc.get('name','')[:14]}"
    return out


def marks(race):
    """★印（◎○▲△）。**◎＝軸 ／ ○▲△＝P紐（モデルの pv 降順）の1・2・3頭目**。

    ★**`ANA_SNS_RULE.md` §2「紐はPで統一」に合わせてある**（**2026-09-13 変更**）。
    ⚠**それ以前は △ だけ X紐の3頭目＝人気順だった**（`reco_ana_day` の役割欄に合わせていた）。
    ★**SNSセッションの判断で §2 側に寄せた**（`ANA_SNS_HANDOFF.md` §0-1）。**理由3つ**:
      ① **§2は測定を根拠にしている**（**P紐の朝9時一致率100% / Q(人気順)は75%**）
      ② **役割欄は `ANA_RULE.md` の買い目（X三連単A4点）のために在り、目的が違う**
      ③ **SNSの買い目は△を使わない**＝**買い目上の制約が無い**
    ⚠**2026-09-12・09-13 に出した印は変更前のもの**。★**記録は書き換えていない**（投稿済みのため）。
    """
    p = next((t for t in race["tickets"] if t["kind"] == "馬単M"), None)
    m = {"◎": race["axis"]}
    rank = (p or {}).get("himo_rank") or []      # ★P紐の上位（モデル順）
    for k, i in (("○", 0), ("▲", 1), ("△", 2)):
        if len(rank) > i:
            m[k] = rank[i]
    return m


# ---------------------------------------------------------------- 束ねる
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ymd", help="開催日 YYYYMMDD")
    ap.add_argument("--no-sync", action="store_true", help="3ブランチの取り込みを飛ばす")
    ap.add_argument("--sns-gap", type=float, default=None,
                    help="SNSのズレ下限（既定は reco_sns_day.SNS_GAP＝0.10）")
    a = ap.parse_args()
    ymd = a.ymd
    d = datetime.date(int(ymd[:4]), int(ymd[4:6]), int(ymd[6:8]))
    outdir = os.path.join(ROOT, OUTDIR, ymd)
    os.makedirs(outdir, exist_ok=True)

    head = [f"{'='*78}",
            f"★★★★{d} の開催日一括（朝）",
            f"{'='*78}", "", "■ 取り込み"]
    if a.no_sync:
        head.append("　（--no-sync）")
    else:
        sync(head)
    head.append("")
    head.append("■ 前提")
    if not need(ymd, head):
        print("\n".join(head))
        print("\n⚠★**足りないものがあるので止めた**。上のコマンドを先に叩くこと。")
        return 1
    head.append("　★揃っている")

    waku_txt, waku = run_waku(ymd, os.path.join(OUTDIR, ymd))
    ana_txt, ana = run_ana(ymd, sns=False)
    sns_txt, sns = run_ana(ymd, sns=True, gap=a.sns_gap)
    lab, eidx = label_map(ymd), entries_index(ymd)
    for r in ana["races"] + sns["races"]:
        r["label"] = lab.get(r["raceid"], r["raceid"])
    for r in sns["races"]:
        r["marks"] = marks(r)
    sns["note"] = ("★印の導出用。⚠**この tickets は買わない**——"
                   "`reco_ana_day` の買い目（P馬単M4点＋X三連単A4点＝800円）がそのまま入っているが、"
                   "`ANA_SNS_RULE.md` §3 のSNS用は 複勝1＋単勝1＋P馬連2＝400円で別物。"
                   "★夜の採点も SNS は ◎の複勝・単勝・着順しか見ない")
    sns["marks_rule"] = "◎=軸 / ○▲△=P紐(モデル順)1・2・3頭目（ANA_SNS_RULE.md §2・2026-09-13から）"

    # ---- まとめ
    n_waku = len(waku["wakuren"]) if waku else 0
    n_soft = len(waku["soft_sanrenpuku"]) if waku else 0
    c_waku = sum(x["cost"] for x in waku["wakuren"]) if waku else 0
    c_soft = sum(x["cost"] for x in waku["soft_sanrenpuku"]) if waku else 0
    c_ana = sum(t["cost"] for r in ana["races"] for t in r["tickets"])
    s = [""]
    s.append("=" * 78)
    s.append("★★今日やること")
    s.append("=" * 78)
    s.append(f"　① 本命 枠連（軸枠×紐枠1）     {n_waku:>2}レース  {c_waku:>6,}円")
    s.append(f"　② 甘い軸の三連複（候補）       {n_soft:>2}レース  {c_soft:>6,}円"
             "　⚠**直前に再判定が要る**")
    s.append(f"　③ 穴馬（自分用・8点/レース）   {len(ana['races']):>2}レース  {c_ana:>6,}円")
    s.append(f"　④ SNS（印・ズレ下限{sns['gap']}）    {len(sns['races']):>2}レース"
             "　　　　投稿だけ（買わない）")
    s.append("　" + "-" * 60)
    s.append(f"　★**自分で買う合計  {c_waku + c_soft + c_ana:,}円**"
             "（⚠②は直前の再判定で消えることがある）")
    s.append("")
    s.append("■ ① 本命 枠連")
    for x in (waku["wakuren"] if waku else []):
        s.append(f"　　{x['label']:>9}　枠連 {' / '.join(x['combos'])}"
                 f"　軸 {x['axis']}番{x['axis_name']}　{x['cost']}円")
    if not n_waku:
        s.append("　　（該当なし）")
    s.append("")
    s.append("■ ② 甘い軸の三連複　⚠**朝の候補。★発走30分前〜10分前に確定させる**")
    for x in (waku["soft_sanrenpuku"] if waku else []):
        s.append(f"　　{x['label']:>9}　三連複 {x['combo']}　軸{x['axis']}番"
                 f"・複勝の期待払戻{x['e_axis']:.0f}円（裾{int(x['tier']*100)}%）　{x['cost']}円")
        s.append(f"　　　　→ ★直前に `python3 ml/nk_race.py {ymd} "
                 f"{x['label'].split()[0][:2]} {x['label'].split()[0][2:].rstrip('R')}`")
    if not n_soft:
        s.append("　　（朝の時点では該当なし。⚠**直前に条件を満たすレースが出ることがある**）")
    s.append("")
    s.append("■ ③ 穴馬（自分用）　★1レース8点800円")
    for r in ana["races"]:
        s.append(f"　　{r['label']}　軸 {r['axis']}番")
        for t in r["tickets"]:
            s.append(f"　　　　{t['label']}　"
                     + " / ".join("-".join(str(x) for x in c) for c in t["combos"]))
    if not ana["races"]:
        s.append("　　（該当なし。⚠**開催日の39.1%は0本**＝`ANA_RULE.md` §1）")
    s.append("")
    s.append("■ ④ SNS（印）　⚠**買い目の羅列は投稿しない**（`ANA_SNS_RULE.md` §8）")
    for r in sns["races"]:
        m = r["marks"]
        s.append(f"　　{r['label']}　"
                 + "　".join(f"{k}{v}" for k, v in m.items() if v is not None))
    if not sns["races"]:
        s.append("　　（該当なし＝★「見送り」を投稿する。`ANA_SNS_RULE.md` §8）")
    s.append("")
    if sns["races"]:
        s.append("■ ④ 投稿文（★そのまま貼れる）")
        s.append("")
        for ln in sns_post(sns["races"], eidx, ymd).splitlines():
            s.append("    " + ln if ln else "")
        s.append("")
    s.append("■ ⚠投稿に添えてはいけない数字")
    s.append("　　★**11年の凍結値（113.4% / 128.8% / 複勝23.5%）は当てはまらない**"
             "——**朝9時の板＋固定モデル＋ズレ下限の変更で別の量**（`ANA_SNS_RULE.md` §8）")
    s.append("")
    s.append("■ ★このあと")
    s.append(f"　　直前（発走30分前〜10分前）　`python3 ml/nk_race.py {ymd} <場> <R>`　②を確定させる")
    s.append(f"　　夜（Mac）　`sh raceday_results.sh {ymd}`　結果を取って push")
    s.append(f"　　夜（クラウド）　`python3 ml/raceday_night.py {ymd}`　4本まとめて答え合わせ")
    s.append("")

    brief = "\n".join(head + s)
    detail = ("\n\n" + "=" * 78 + "\n★①本命 predict_nk.py の生出力\n" + "=" * 78 + "\n" + waku_txt
              + "\n\n" + "=" * 78 + "\n★③穴馬 reco_ana_day.py の生出力\n" + "=" * 78 + "\n" + ana_txt
              + "\n\n" + "=" * 78 + "\n★④SNS reco_sns_day.py の生出力\n" + "=" * 78 + "\n" + sns_txt)
    print(brief)
    print(detail)

    open(os.path.join(outdir, "brief.txt"), "w", encoding="utf-8").write(brief + detail)
    post = sns_post(sns["races"], eidx, ymd) if sns["races"] else "今日は該当0本。見送りです。"
    open(os.path.join(outdir, "sns_post.txt"), "w", encoding="utf-8").write(post + "\n")
    tickets = {
        "date": ymd,
        "generated_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "git_head": sh("git", "rev-parse", "HEAD").stdout.strip(),
        "note": "★朝に凍結した買い目。★夜の採点（ml/raceday_night.py）はこれだけを読む",
        "waku": waku, "ana": ana, "sns": sns,
    }
    json.dump(tickets, open(os.path.join(outdir, "tickets.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"★保存: {OUTDIR}/{ymd}/tickets.json ・ brief.txt ・ sns_post.txt"
          f" ／ ★正典 data/reco/reco_{ymd}.json")
    print("⚠★**買う前にコミットすること**（★結果を見る前に凍結した証拠になる）:")
    print(f"　　git add {OUTDIR}/{ymd} data/reco data/nk data/nk_odds_morn && "
          f"git commit -m '{d} の朝の買い目（結果を見る前）'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
