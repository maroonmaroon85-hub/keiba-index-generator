import type { Rule } from "./types.js";
import { estimateStyle } from "../scoring/style.js";

/**
 * 後方△: 追込脚質(b)で、かつ想定ペースがハイでない（前有利想定）。
 * ハイペースなら追込は展開利があるので割引かない。
 */
export const rule: Rule = {
  name: "後方△",
  sign: "minus",
  condition: (horse, race) => {
    return estimateStyle(horse) === "b" && race.pace !== "H";
  },
};
