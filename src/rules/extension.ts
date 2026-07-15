import type { Rule } from "./types.js";

/** 延長△: 前走より距離が minMeters 以上延長。距離延長は割引きの初期仮説。 */
export const rule: Rule = {
  name: "延長△",
  sign: "minus",
  condition: (horse, race, ctx) => {
    const last = horse.pastRuns[0];
    if (!last) return false;
    return race.distance - last.distance >= ctx.params["延長_minMeters"]!;
  },
};
