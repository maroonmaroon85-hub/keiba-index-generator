import type { Rule } from "./types.js";

/**
 * 人気落ち◎: 近走で好走(前走 recentTopN 着以内)しているのに
 * 今回の想定人気が minPopularity 番人気以下と評価が下がっている。
 * 妙味（オッズ的な過小評価）を拾う仮説。予想人気は odds 昇順の順位で判定する。
 * ※ 予想人気の順位は scoring 側で算出して horse に付与できないため、
 *   ここでは近走好走 × 想定オッズが人気薄、という PreRace 情報のみで近似する。
 */
export const rule: Rule = {
  name: "人気落ち◎",
  sign: "plus",
  condition: (horse, _race, ctx) => {
    const last = horse.pastRuns[0];
    if (!last) return false;
    const ranWell = last.finish <= ctx.params["人気落ち_recentTopN"]!;
    // 想定オッズが二桁台に近い＝人気薄の目安（minPopularity番人気相当を odds で近似）。
    const unpopular = horse.odds >= ctx.params["人気落ち_minPopularity"]! * 2;
    return ranWell && unpopular;
  },
};
