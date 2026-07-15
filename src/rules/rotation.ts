import type { Rule } from "./types.js";

/** ローテ△: 前走間隔が中1週未満(詰まりすぎ) or 半年以上(間隔空きすぎ)。 */
export const rule: Rule = {
  name: "ローテ△",
  sign: "minus",
  condition: (horse, _race, ctx) => {
    const w = horse.weeksSinceLastRun;
    return w < ctx.params["ローテ_shortWeeks"]! || w >= ctx.params["ローテ_longWeeks"]!;
  },
};
