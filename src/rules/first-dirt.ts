import type { Rule } from "./types.js";

/** 初ダ△: 今走がダートで、近走にダート実績が一切ない（＝初ダート）。 */
export const rule: Rule = {
  name: "初ダ△",
  sign: "minus",
  condition: (horse, race) => {
    if (race.surface !== "ダ") return false;
    return horse.pastRuns.every((r) => r.surface !== "ダ");
  },
};
