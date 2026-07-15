import type { ScoredRace } from "./scoring/types.js";

/** schema/score-export.json に対応する出力構造。 */
export interface ScoreExport {
  race_id: string;
  race_info: {
    course: string;
    surface: string;
    distance: number;
    pace: string;
    condition: string;
    name?: string;
  };
  horses: Array<{
    number: number;
    name: string;
    score: number;
    rank: string;
    mark: string;
    win_prob: number;
    place_prob: number;
    flags: { plus: string[]; minus: string[] };
  }>;
}

/** ScoredRace → keiba-ev が読むスコアJSON。馬番昇順で出力。 */
export function toScoreExport(scored: ScoredRace): ScoreExport {
  const { race, horses } = scored;
  return {
    race_id: race.raceId,
    race_info: {
      course: race.course,
      surface: race.surface,
      distance: race.distance,
      pace: race.pace,
      condition: race.condition,
      ...(race.name ? { name: race.name } : {}),
    },
    horses: [...horses]
      .sort((a, b) => a.number - b.number)
      .map((h) => ({
        number: h.number,
        name: h.name,
        score: h.score,
        rank: h.rank,
        mark: h.mark,
        win_prob: h.winProb,
        place_prob: h.placeProb,
        flags: h.flags,
      })),
  };
}
