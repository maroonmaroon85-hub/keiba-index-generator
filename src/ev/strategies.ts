import type { ScoredRace } from "../scoring/types.js";
import { exoticCombos, type Combo, type ExoticType } from "./harville.js";

/**
 * 連系の「買い方（戦略）」生成。win_prob 上位馬から購入点を作る。
 * 払戻データだけで回収率が測れる（＝当たった組の配当が分かればよい）ので、
 * オッズ全表なしで「どの買い方が一番か」を研究できる。
 *
 * 戦略:
 *  - box:        上位N頭のボックス（全通り）
 *  - formation:  軸（上位A頭）×相手（上位B頭）。順序券種は軸を上位着に固定
 *  - topK:       Harville確率の上位K点（harville.ts）
 */

const ORDERED: ExoticType[] = ["馬単", "三連単"];
const SIZE: Record<ExoticType, number> = { 馬連: 2, 馬単: 2, ワイド: 2, 三連複: 3, 三連単: 3 };

function topHorses(scored: ScoredRace, n: number): number[] {
  return [...scored.horses].sort((a, b) => b.winProb - a.winProb).slice(0, n).map((h) => h.number);
}

function combinations<T>(arr: T[], k: number): T[][] {
  if (k === 0) return [[]];
  if (arr.length < k) return [];
  const [head, ...rest] = arr;
  return [...combinations(rest, k - 1).map((c) => [head!, ...c]), ...combinations(rest, k)];
}
function permutations<T>(arr: T[], k: number): T[][] {
  if (k === 0) return [[]];
  return arr.flatMap((x, i) => permutations([...arr.slice(0, i), ...arr.slice(i + 1)], k - 1).map((p) => [x, ...p]));
}

/** 上位N頭のボックス。 */
export function boxCombos(scored: ScoredRace, type: ExoticType, topN: number): Combo[] {
  const hs = topHorses(scored, topN);
  const size = SIZE[type];
  const raw = ORDERED.includes(type) ? permutations(hs, size) : combinations(hs, size);
  return raw.map((horses) => ({ type, horses, prob: 0 }));
}

/**
 * 軸フォーメーション。軸=上位axisN頭、相手=上位(axisN+partnerN)頭。
 * 順序券種は軸を上位着（1着/上位）に置く。非順序は軸を含む組。
 */
export function formationCombos(scored: ScoredRace, type: ExoticType, axisN: number, partnerN: number): Combo[] {
  const size = SIZE[type];
  const pool = topHorses(scored, axisN + partnerN);
  const axes = pool.slice(0, axisN);
  const out: Combo[] = [];
  const seen = new Set<string>();
  const push = (horses: number[]) => {
    const key = (ORDERED.includes(type) ? horses : [...horses].sort((a, b) => a - b)).join("-");
    if (seen.has(key)) return;
    seen.add(key);
    out.push({ type, horses, prob: 0 });
  };
  if (ORDERED.includes(type)) {
    // 軸を先頭着に固定し、残り着を相手から並べる。
    for (const ax of axes) {
      const others = pool.filter((h) => h !== ax);
      for (const rest of permutations(others, size - 1)) push([ax, ...rest]);
    }
  } else {
    for (const rest of combinations(pool, size)) if (rest.some((h) => axes.includes(h))) push(rest);
  }
  return out;
}

/** Harville確率上位K点（harville.ts のラッパ）。 */
export function topKCombos(scored: ScoredRace, type: ExoticType, k: number): Combo[] {
  return exoticCombos(scored, type, k);
}
