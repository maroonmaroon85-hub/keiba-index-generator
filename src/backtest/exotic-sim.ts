import { scoreRace } from "../scoring/score.js";
import { exoticCombos, type ExoticType, type Combo } from "../ev/harville.js";
import type { BacktestRace } from "./dataset.js";
import type { Config } from "../config.js";

/**
 * 連系券種の購入シミュレーション。
 * 戦略: 各レースで Harville 確率の上位 topN 組を各100円購入し、実際の払戻と突合して回収率を出す。
 * 払戻データ（RacePayout）は TARGET の払戻CSVから作る想定（パーサは払戻CSVの実物受領後に確定）。
 * ここでは「戦略評価ロジック」を確定させておき、データが来たら即回せる状態にする。
 */

/** 1レース1券種の的中組み合わせと払戻（100円あたり円）。 */
export interface Winning {
  combo: number[]; // 馬連/三連複は順不同、馬単/三連単は着順どおり
  payout: number; // 100円あたりの払戻金
}
export interface RacePayout {
  umaren?: Winning;
  umatan?: Winning;
  sanrenpuku?: Winning;
  sanrentan?: Winning;
  wide?: Winning[]; // ワイドは的中3組
}

export interface ExoticResult {
  type: ExoticType;
  bets: number; // 購入点数
  hits: number;
  roi: number; // 回収率(%)
}

function sortNum(a: number[]): number[] {
  return [...a].sort((x, y) => x - y);
}
function sameSet(a: number[], b: number[]): boolean {
  if (a.length !== b.length) return false;
  const s = sortNum(a);
  const t = sortNum(b);
  return s.every((v, i) => v === t[i]);
}
function sameOrder(a: number[], b: number[]): boolean {
  return a.length === b.length && a.every((v, i) => v === b[i]);
}

const UNORDERED: ExoticType[] = ["馬連", "三連複", "ワイド"];

function winnersFor(type: ExoticType, pay: RacePayout): Winning[] {
  switch (type) {
    case "馬連": return pay.umaren ? [pay.umaren] : [];
    case "馬単": return pay.umatan ? [pay.umatan] : [];
    case "三連複": return pay.sanrenpuku ? [pay.sanrenpuku] : [];
    case "三連単": return pay.sanrentan ? [pay.sanrentan] : [];
    case "ワイド": return pay.wide ?? [];
  }
}

/** 純粋な戦略評価: レースごとの購入候補(combos)＋払戻 → 回収率。テスト容易なようスコアリングと分離。 */
export function evalExoticStrategy(
  entries: { raceId: string; combos: Combo[] }[],
  payouts: Map<string, RacePayout>,
  type: ExoticType,
): ExoticResult {
  const unordered = UNORDERED.includes(type);
  let bets = 0;
  let hits = 0;
  let ret = 0;
  for (const e of entries) {
    const pay = payouts.get(e.raceId);
    if (!pay) continue;
    const winners = winnersFor(type, pay);
    if (winners.length === 0) continue;
    for (const c of e.combos) {
      bets++;
      for (const w of winners) {
        const match = unordered ? sameSet(c.horses, w.combo) : sameOrder(c.horses, w.combo);
        if (match) {
          hits++;
          ret += w.payout;
          break;
        }
      }
    }
  }
  return { type, bets, hits, roi: bets > 0 ? (ret / (bets * 100)) * 100 : 0 };
}

/** スコアリング → Harville上位topN購入 → 払戻突合。 */
export function simulateExotic(
  races: BacktestRace[],
  payouts: Map<string, RacePayout>,
  config: Config,
  type: ExoticType,
  topN: number,
): ExoticResult {
  const entries = races.map((r) => ({
    raceId: r.pre.race.raceId,
    combos: exoticCombos(scoreRace(r.pre, config), type, topN),
  }));
  return evalExoticStrategy(entries, payouts, type);
}
