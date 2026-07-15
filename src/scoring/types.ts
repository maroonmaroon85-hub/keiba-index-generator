import type { PreRaceInfo, RunningStyle } from "../model/pre-race.js";

/** 1頭分の指数計算結果。 */
export interface ScoredHorse {
  number: number;
  name: string;
  score: number;
  /** 内訳（デバッグ・バックテスト用）。 */
  breakdown: {
    base: number;
    rules: number;
    training: number;
    pedigree: number;
  };
  rank: string;
  /** 印（◎○▲△× または空）。 */
  mark: string;
  /** レース内予想人気順位（1=1番人気）。odds 昇順。 */
  predictedPopularity: number;
  winProb: number;
  placeProb: number;
  style: RunningStyle;
  flags: { plus: string[]; minus: string[] };
}

/** レース単位の指数計算結果。 */
export interface ScoredRace {
  race: PreRaceInfo;
  horses: ScoredHorse[];
}
