import type { Rule } from "./types.js";

/** 減量◎: 減量騎手騎乗（斤量減がある）。斤量利を評価。 */
export const rule: Rule = {
  name: "減量◎",
  sign: "plus",
  condition: (horse) => {
    return horse.jockey.reductionKg > 0;
  },
};
