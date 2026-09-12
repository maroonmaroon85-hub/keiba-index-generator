"""(124)の再発防止 — **共有関数の引数が変わったのに呼び出し側が追随していない**を静的に検出する。

★★なぜ要るか（2026-08-16に実際に踏んだ）
　`audit_fuku_lbs.top3_probs` に (145)で `tau` 引数が足されたのに、
　`audit_fuku_board.py`（(124)）の呼び出し側が更新されず **TypeError で落ちる状態**だった。
　⚠**複勝の板が無くて実行されなかったので、4日間そのままだった**。
　→ ★**「データ待ち」と書いて寝かせたスクリプトは、寝ている間に腐る**。
　★**実行しなくても検出できる**ので、**道具を変えたコミットで毎回これを回す**。

★何を見るか
　`ml/*.py` の全 `def` の引数の数を集め、**直接呼び出し（`f(...)` の形）**と突き合わせる。
　⚠**属性呼び出し（`d.values()` 等）は見ない**——**組み込みメソッドと同名の関数があると
　　全部誤検出になる**（実際 `score_table.values` で43件の誤検出が出た）。
　⚠**`*args` / キーワード引数が混じる呼び出しも見ない**（静的には判定できない）。
　⚠**同名の関数が複数あるときは「どれか1つに合えばOK」**とする（誤検出を避ける保守側）。

★これが見つけないもの（限界を先に書く）
　・**引数の「数」しか見ない**。**順番や意味が変わった場合は見つからない**
　　（例: `top3_probs(p, tau, l2, l3)` の `tau` と `l2` を入れ替えても数は同じ）。
　・**キーワード引数で呼んでいる箇所**。
　・**実行時にしか分からない型の不一致**。
　→ ★**これは「安い網」であって「保証」ではない**。**総当たり実行の代わりにはならない**。

実行: python3 ml/check_sigs.py    （終了コード 1 = 不一致あり。CIに入れられる）
"""
import ast
import glob
import os
import sys
from collections import defaultdict


def collect_defs(paths):
    defs = defaultdict(list)
    for p in paths:
        try:
            t = ast.parse(open(p, encoding="utf-8").read())
        except SyntaxError as e:
            print(f"⚠構文エラー {p}: {e}")
            continue
        for node in ast.walk(t):
            if isinstance(node, ast.FunctionDef):
                a = node.args
                pos = len(a.posonlyargs) + len(a.args)
                mn = pos - len(a.defaults)
                mx = 10 ** 6 if a.vararg else pos
                defs[node.name].append((mn, mx, os.path.basename(p), node.lineno))
    return defs


def find_bad(paths, defs):
    bad = []
    for p in paths:
        try:
            t = ast.parse(open(p, encoding="utf-8").read())
        except SyntaxError:
            continue
        for node in ast.walk(t):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                continue
            name = node.func.id
            if name not in defs:
                continue
            if node.keywords or any(isinstance(x, ast.Starred) for x in node.args):
                continue
            n = len(node.args)
            if any(mn <= n <= mx for mn, mx, _, _ in defs[name]):
                continue
            bad.append((os.path.basename(p), node.lineno, name, n,
                        " / ".join(f"{mn}〜{'*' if mx >= 10**6 else mx}引数@{fl}:{ln}"
                                   for mn, mx, fl, ln in defs[name])))
    return bad


def main():
    paths = sorted(glob.glob("ml/*.py"))
    if not paths:
        sys.exit("ml/*.py が見つからない。リポジトリ直下で実行すること。")
    defs = collect_defs(paths)
    bad = find_bad(paths, defs)
    print(f"ml/*.py を {len(paths)} 本、定義 {sum(len(v) for v in defs.values())} 個で照合")
    if not bad:
        print("★引数の数が合わない直接呼び出しは **0件**")
        print("⚠**これは安い網であって保証ではない**（引数の数しか見ない。冒頭の限界を読むこと）。")
        return 0
    print(f"⚠**引数の数が合わない直接呼び出し {len(bad)}件**")
    for f, ln, nm, n, cand in bad:
        print(f"  {f}:{ln}  {nm}(...) を {n}引数で呼んでいる  ← 定義は {cand}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
