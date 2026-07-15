import type { Rule } from "./types.js";

/** 増量△: 前走比で斤量が minKg 以上増加。 */
export const rule: Rule = {
  name: "増量△",
  sign: "minus",
  condition: (horse, _race, ctx) => {
    return horse.weightCarry - horse.prevWeightCarry >= ctx.params["増量_minKg"]!;
  },
};
