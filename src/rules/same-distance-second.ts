import type { Rule } from "./types.js";

/** 同2◎: 前走と同距離（＝同距離2走目）。条件を掴んでいる想定で加点。 */
export const rule: Rule = {
  name: "同2◎",
  sign: "plus",
  condition: (horse, race) => {
    const last = horse.pastRuns[0];
    if (!last) return false;
    return last.distance === race.distance;
  },
};
