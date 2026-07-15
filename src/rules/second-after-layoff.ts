import type { Rule } from "./types.js";

/**
 * 叩き2◎: 休み明け2走目。
 * 前走の前が長期休養(layoffWeeks 以上)で、前走→今走が短間隔(freshWeeks 以内)。
 * 一度使われて上昇する典型パターンを加点。
 */
export const rule: Rule = {
  name: "叩き2◎",
  sign: "plus",
  condition: (horse, _race, ctx) => {
    return (
      horse.weeksBeforeLastRun >= ctx.params["叩き_layoffWeeks"]! &&
      horse.weeksSinceLastRun <= ctx.params["叩き_freshWeeks"]!
    );
  },
};
