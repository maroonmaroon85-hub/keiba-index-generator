import type { ScoredRace } from "../scoring/types.js";

/**
 * Harville モデルによる連系券種の確率算出。
 * 各馬の単勝確率 p_i（win_prob）から、着順の条件付き確率を積んで組み合わせ確率を出す。
 *   P(i→j)      = p_i * p_j/(1-p_i)                      （馬単）
 *   P(i→j→k)    = p_i * p_j/(1-p_i) * p_k/(1-p_i-p_j)    （三連単）
 *   馬連/三連複 = 順序違いの和、ワイド = 両馬とも3着内
 * 単体確率を渡すだけなので、ルールベースでもML由来でも同じI/Fで使える。
 *
 * 注意: Harvilleは近似（人気馬の2・3着を過大評価する既知の癖）。EV判定には十分だが、
 * 精緻化するならPL(Plackett-Luce)や補正係数を後で入れる。
 */

export type ExoticType = "馬単" | "馬連" | "ワイド" | "三連複" | "三連単";

export interface Combo {
  type: ExoticType;
  horses: number[]; // 券面の馬番（順序券種は着順どおり）
  prob: number;
}

interface P {
  n: number;
  p: number;
}

function normalized(scored: ScoredRace): P[] {
  const raw = scored.horses.map((h) => ({ n: h.number, p: Math.max(h.winProb, 1e-6) }));
  const sum = raw.reduce((a, b) => a + b.p, 0) || 1;
  return raw.map((x) => ({ n: x.n, p: x.p / sum }));
}

/** 全組み合わせの確率を計算し、指定券種を確率降順で返す（topN件）。 */
export function exoticCombos(scored: ScoredRace, type: ExoticType, topN = 5): Combo[] {
  const ps = normalized(scored);
  const combos: Combo[] = [];
  const n = ps.length;

  const pair2nd = (i: number, j: number) => (ps[i]!.p * ps[j]!.p) / (1 - ps[i]!.p); // P(i→j)

  if (type === "馬単") {
    for (let i = 0; i < n; i++)
      for (let j = 0; j < n; j++)
        if (i !== j) combos.push({ type, horses: [ps[i]!.n, ps[j]!.n], prob: pair2nd(i, j) });
  } else if (type === "馬連") {
    for (let i = 0; i < n; i++)
      for (let j = i + 1; j < n; j++)
        combos.push({ type, horses: [ps[i]!.n, ps[j]!.n], prob: pair2nd(i, j) + pair2nd(j, i) });
  } else if (type === "三連単") {
    for (let i = 0; i < n; i++)
      for (let j = 0; j < n; j++) {
        if (j === i) continue;
        for (let k = 0; k < n; k++) {
          if (k === i || k === j) continue;
          const p = pair2nd(i, j) * (ps[k]!.p / (1 - ps[i]!.p - ps[j]!.p));
          if (p > 0) combos.push({ type, horses: [ps[i]!.n, ps[j]!.n, ps[k]!.n], prob: p });
        }
      }
  } else if (type === "三連複" || type === "ワイド") {
    // 三連複: 3頭が上位3着を占める確率（6通りの三連単の和）。
    // ワイド: 2頭がともに3着内 = その2頭を含む三連複を、残り1頭で周辺化。
    const triple = (i: number, j: number, k: number) => {
      let s = 0;
      for (const [a, b, c] of perms3(i, j, k)) {
        const denom2 = 1 - ps[a]!.p - ps[b]!.p;
        if (denom2 > 1e-9) s += pair2nd(a, b) * (ps[c]!.p / denom2);
      }
      return s;
    };
    if (type === "三連複") {
      for (let i = 0; i < n; i++)
        for (let j = i + 1; j < n; j++)
          for (let k = j + 1; k < n; k++) combos.push({ type, horses: [ps[i]!.n, ps[j]!.n, ps[k]!.n], prob: triple(i, j, k) });
    } else {
      for (let i = 0; i < n; i++)
        for (let j = i + 1; j < n; j++) {
          let s = 0;
          for (let k = 0; k < n; k++) if (k !== i && k !== j) s += triple(i, j, k);
          combos.push({ type, horses: [ps[i]!.n, ps[j]!.n], prob: s });
        }
    }
  }
  return combos.sort((a, b) => b.prob - a.prob).slice(0, topN);
}

function perms3(i: number, j: number, k: number): [number, number, number][] {
  return [
    [i, j, k],
    [i, k, j],
    [j, i, k],
    [j, k, i],
    [k, i, j],
    [k, j, i],
  ];
}
