import type { Rule } from "./types.js";
import { estimateStyle } from "../scoring/style.js";

/**
 * 内枠先行◎: 内枠(frame <= maxFrame)かつ先行脚質(f)。
 * ロスの少ない立ち回りが見込める想定で加点。最内枠単独でも内枠先行に含める。
 */
export const rule: Rule = {
  name: "内枠先行◎",
  sign: "plus",
  condition: (horse, _race, ctx) => {
    const inner = horse.frame <= ctx.params["内枠_maxFrame"]!;
    return inner && estimateStyle(horse) === "f";
  },
};
