import type { Rule } from "./types.js";
import { rule as extension } from "./extension.js";
import { rule as firstDirt } from "./first-dirt.js";
import { rule as weightUp } from "./weight-up.js";
import { rule as rearUnfavored } from "./rear-unfavored.js";
import { rule as rotation } from "./rotation.js";
import { rule as sameDistanceSecond } from "./same-distance-second.js";
import { rule as innerFront } from "./inner-front.js";
import { rule as apprentice } from "./apprentice.js";
import { rule as secondAfterLayoff } from "./second-after-layoff.js";
import { rule as popularityDrop } from "./popularity-drop.js";

export type { Rule, RuleContext } from "./types.js";

/** 全ルール（minus 5 + plus 5 = 10）。 */
export const ALL_RULES: Rule[] = [
  // minus
  extension,
  firstDirt,
  weightUp,
  rearUnfavored,
  rotation,
  // plus
  sameDistanceSecond,
  innerFront,
  apprentice,
  secondAfterLayoff,
  popularityDrop,
];
