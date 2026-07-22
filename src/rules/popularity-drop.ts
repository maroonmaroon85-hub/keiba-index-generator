import type { Rule } from "./types.js";

/**
 * 人気落ち△: 近走で好走(前走 recentTopN 着以内)しているのに
 * 今回の想定人気が minPopularity 番人気以下と評価が下がっている馬。
 *
 * 当初は「妙味（過小評価）を拾う」プラス仮説だったが、1年ぶん(8,107サンプル)の
 * バックテストで複勝率17.7%（全体基準26.0%）と明確に平均以下＝市場のフェードが正しく、
 * むしろ割引くべき負のシグナルと判明したため minus に反転した（Phase 3の実測反映）。
 */
export const rule: Rule = {
  name: "人気落ち△",
  sign: "minus",
  condition: (horse, _race, ctx) => {
    const last = horse.pastRuns[0];
    if (!last) return false;
    const ranWell = last.finish <= ctx.params["人気落ち_recentTopN"]!;
    // 想定オッズが二桁台に近い＝人気薄の目安（minPopularity番人気相当を odds で近似）。
    const unpopular = horse.odds >= ctx.params["人気落ち_minPopularity"]! * 2;
    return ranWell && unpopular;
  },
};
