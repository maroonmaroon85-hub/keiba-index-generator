"""★★影武者運転 —— **クラウドでも取ってみて、Macの本番と突き合わせる**（2026-09-26から）

★**目的**: ⚠**朝9時の板は取り逃すと永久に失われる**（`ANA_RULE.md` §6）。
　→ ★**「クラウドでも取れる」を、本番を止めずに確かめる**。
　★**本番（`data/nk/` `data/nk_odds_morn/`）には1バイトも書かない**。出力は `data/shadow/<日付>/`。

■ ⚠★**守ること**
| | |
|---|---|
| **1** | ★**Macより後に走らせる**（⚠`pgrep` は同じマシンしか見ない＝**規則4b は跨がない**） |
| **2** | ★**本番の出力先を書き換えてから呼ぶ**（`nk_fetch.OUT` / `nk_place_morn.OUT`） |
| **3** | ★**買い目は作らない**。⚠**凍結記録にも正典にも触れない** |
| **4** | ★**突き合わせは「取れた数」と「板の値」**。⚠**時刻は必ずずれる**ので差として出すだけ |

実行: python3 ml/raceday_shadow.py 20260927
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

SHADOW = "data/shadow"


def load_jsonl(path):
    """→ {raceid: (最新の板, 取得時刻)}。★同じレースは新しい方（本番と同じ規則）。"""
    out = {}
    if not os.path.exists(path):
        return out
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except ValueError:
            continue
        prev = out.get(r["raceid"])
        if prev is None or r["fetched_at"] > prev[1]:
            out[r["raceid"]] = (r.get("odds") or {}, r["fetched_at"])
    return out


def compare(ymd, sh_dir):
    """★本番とシャドウを突き合わせて、★表示する行を返す。"""
    real = load_jsonl(os.path.join(ROOT, f"data/nk_odds_morn/place{ymd}.jsonl"))
    shad = load_jsonl(os.path.join(sh_dir, f"place{ymd}.jsonl"))
    out = ["", "=" * 70, "★★突き合わせ（本番＝Mac ／ 影＝クラウド）", "=" * 70]
    if not real:
        out.append("⚠**本番の板がまだ無い**（★Macがpushする前に走ったか、取れていない）")
    if not shad:
        out.append("⚠★**影の板が取れなかった**。★ここが本番になると板が失われる")
    out.append(f"　レース数　本番 {len(real)} / 影 {len(shad)}")
    only_r = sorted(set(real) - set(shad))
    only_s = sorted(set(shad) - set(real))
    if only_r:
        out.append(f"　⚠**本番にしか無い {len(only_r)}レース**: " + " ".join(only_r[:6]))
    if only_s:
        out.append(f"　⚠**影にしか無い {len(only_s)}レース**: " + " ".join(only_s[:6]))

    # ★値の比較（★同じレース・同じ馬番の [下限,上限] が一致するか）
    same = diff = 0
    ex = []
    for rid in sorted(set(real) & set(shad)):
        ro, rt = real[rid]
        so, st = shad[rid]
        for u in sorted(set(ro) & set(so), key=lambda x: int(x)):
            if list(ro[u]) == list(so[u]):
                same += 1
            else:
                diff += 1
                if len(ex) < 5:
                    ex.append(f"{rid} {u}番 本番{ro[u]} → 影{so[u]}")
    n = same + diff
    if n:
        out.append(f"　値の一致　{same}/{n} ({same/n:.0%})　⚠**差はオッズが動いたぶん**（時刻が違う）")
        out += ["　　　" + e for e in ex]
    if real and shad:
        rt = sorted(v[1] for v in real.values())[0]
        st = sorted(v[1] for v in shad.values())[0]
        out.append(f"　時刻　　　本番 {rt} / 影 {st}")
    out.append("")
    out.append("★**読み方**: ⚠**値が違うのは正常**（取得時刻が違うのでオッズは動く）。")
    out.append("　★**見るのはレース数**——★**本番と同じ数が取れていれば、クラウドで代わりが務まる**。")
    return out


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        sys.exit("日付が要る: python3 ml/raceday_shadow.py 20260927")
    ymd = args[0]
    sh_dir = os.path.join(ROOT, SHADOW, ymd)
    os.makedirs(sh_dir, exist_ok=True)

    print("=" * 70)
    print(f"★★★★{ymd} の影武者運転（★本番には書かない）")
    print("=" * 70)
    print(f"　出力先 {SHADOW}/{ymd}/　⚠**data/nk と data/nk_odds_morn は触らない**")

    # ★★出力先を書き換えてから呼ぶ（⚠本番を上書きしないための唯一の砦）
    rel = os.path.relpath(sh_dir, ROOT)
    for mod, name in (("nk_fetch", "OUT"), ("nk_place_morn", "OUT")):
        m = __import__(mod)
        if getattr(m, name) in ("data/nk", "data/nk_odds_morn"):
            setattr(m, name, rel)
        else:
            sys.exit(f"⚠**{mod}.{name} が想定と違う（{getattr(m,name)}）。★安全でないので止める**")

    import nk_fetch as F
    import nk_place_morn as P

    print("\n=== 出馬表＋単勝（影）===")
    argv = sys.argv
    try:
        sys.argv = ["nk_fetch.py", "entries", ymd]
        F.main()
    except SystemExit:
        pass
    except Exception as e:                        # ⚠落ちても突き合わせまでは進む
        print(f"⚠**出馬表で落ちた**: {type(e).__name__}: {e}")
    finally:
        sys.argv = argv

    print("\n=== 朝9時の複勝板（影）===")
    try:
        sys.argv = ["nk_place_morn.py", ymd]
        P.main()
    except SystemExit:
        pass
    except Exception as e:
        print(f"⚠**板で落ちた**: {type(e).__name__}: {e}")
    finally:
        sys.argv = argv

    print("\n".join(compare(ymd, sh_dir)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
