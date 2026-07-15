import type { PreRaceData, PreRaceHorse } from "../model/pre-race.js";
import type { Config } from "../config.js";
import type { ScoredHorse, ScoredRace } from "./types.js";
import { ALL_RULES, type Rule } from "../rules/index.js";
import { estimateStyle } from "./style.js";
import { aptitude } from "../pedigree/master.js";

/**
 * 指数計算のエントリポイント。
 *
 * 【リーク防止】受け取るのは PreRaceData のみ。PostRaceData は型として渡せない。
 * 【Phase 5 互換】この関数のシグネチャ（PreRaceData → 各馬 winProb/placeProb を持つ ScoredRace）を固定し、
 *  内部実装（ルール合算 or ML）を差し替え可能にする。
 */
export function scoreRace(data: PreRaceData, config: Config): ScoredRace {
  const { race, horses } = data;

  // 予想人気: 想定オッズ昇順の順位（同オッズは馬番で安定ソート）。
  const popularityOrder = [...horses].sort((a, b) => a.odds - b.odds || a.number - b.number);
  const popularityRank = new Map<number, number>();
  popularityOrder.forEach((h, i) => popularityRank.set(h.number, i + 1));

  // 各馬の素点を計算。
  const scored: ScoredHorse[] = horses.map((horse) => {
    const base = basePoint(horse, config);
    const { delta, plus, minus } = applyRules(horse, data, config);
    const training = trainingBonus(horse, config);
    const pedigree = pedigreeBonus(horse, data, config);
    const score = base + delta + training + pedigree;

    return {
      number: horse.number,
      name: horse.name,
      score: round1(score),
      breakdown: {
        base: round1(base),
        rules: round1(delta),
        training: round1(training),
        pedigree: round1(pedigree),
      },
      rank: "", // 後で閾値付与
      mark: "", // 後で順位付与
      predictedPopularity: popularityRank.get(horse.number)!,
      winProb: 0, // 後で softmax
      placeProb: 0,
      style: estimateStyle(horse),
      flags: { plus, minus },
    };
  });

  assignRanks(scored, config);
  assignProbabilities(scored, config);
  assignMarks(scored, config);

  return { race, horses: scored };
}

/** 基礎点: 近走着順の加重平均（直近ほど重い）。着順→(頭数-着順+1)/頭数*100。 */
function basePoint(horse: PreRaceHorse, config: Config): number {
  const weights = config.scoring.base.recencyWeights;
  let acc = 0;
  let wsum = 0;
  horse.pastRuns.forEach((run, i) => {
    const w = weights[i] ?? 0;
    if (w === 0) return;
    const pts = ((run.fieldSize - run.finish + 1) / run.fieldSize) * 100;
    acc += pts * w;
    wsum += w;
  });
  const avg = wsum > 0 ? acc / wsum : 50; // 履歴なしは中立50
  return avg * config.scoring.base.scale;
}

/** ルール適用。plus は +weight, minus は -weight。 */
function applyRules(
  horse: PreRaceHorse,
  data: PreRaceData,
  config: Config,
): { delta: number; plus: string[]; minus: string[] } {
  const ctx = { params: config.ruleParams };
  const plus: string[] = [];
  const minus: string[] = [];
  let delta = 0;
  for (const rule of ALL_RULES as Rule[]) {
    if (!rule.condition(horse, data.race, ctx)) continue;
    const weight = config.rules[rule.name] ?? 0;
    if (rule.sign === "plus") {
      plus.push(rule.name);
      delta += weight;
    } else {
      minus.push(rule.name);
      delta -= weight;
    }
  }
  return { delta, plus, minus };
}

function trainingBonus(horse: PreRaceHorse, config: Config): number {
  const ev = config.scoring.training.evaluation[horse.training.evaluation] ?? 0;
  const tr = config.scoring.training.trend[horse.training.trend] ?? 0;
  return ev + tr;
}

function pedigreeBonus(horse: PreRaceHorse, data: PreRaceData, config: Config): number {
  const { surface, distance } = data.race;
  const sire = aptitude(horse.sireLine, surface, distance);
  const damSire = aptitude(horse.damSireLine, surface, distance);
  // 適性は 0..1。中立0.5 を差し引いて -0.5..+0.5 に振ってから係数を掛ける。
  return (
    (sire - 0.5) * config.scoring.pedigree.sireWeight +
    (damSire - 0.5) * config.scoring.pedigree.damSireWeight
  );
}

function assignRanks(scored: ScoredHorse[], config: Config): void {
  const thresholds = [...config.rank.thresholds].sort((a, b) => b.min - a.min);
  for (const h of scored) {
    h.rank = thresholds.find((t) => h.score >= t.min)?.rank ?? "C";
  }
}

/** win_prob = レース内スコアの softmax。place_prob = win_prob の単調変換（Phase 3でキャリブレーション）。 */
function assignProbabilities(scored: ScoredHorse[], config: Config): void {
  const temp = config.scoring.softmaxTemperature;
  const maxScore = Math.max(...scored.map((h) => h.score));
  const exps = scored.map((h) => Math.exp((h.score - maxScore) / temp));
  const sum = exps.reduce((a, b) => a + b, 0);
  scored.forEach((h, i) => {
    const wp = exps[i]! / sum;
    h.winProb = round3(wp);
    const pp = Math.min(config.scoring.placeProb.cap, wp * config.scoring.placeProb.multiplier);
    h.placeProb = round3(pp);
  });
}

/** 印: スコア上位から ◎○▲△、下位かつトップと gapForCross 以上離れた馬に ×。 */
function assignMarks(scored: ScoredHorse[], config: Config): void {
  const order = [...scored].sort((a, b) => b.score - a.score);
  const marks = config.marks.order;
  order.forEach((h, i) => {
    if (i < marks.length) h.mark = marks[i]!;
  });
  const top = order[0]?.score ?? 0;
  // 下位から crossCount 頭、かつトップとの差が gapForCross 以上に × を付ける。
  const worst = [...order].reverse();
  let crosses = 0;
  for (const h of worst) {
    if (crosses >= config.marks.crossCount) break;
    if (h.mark !== "") continue; // 既に印がある馬には付けない
    if (top - h.score >= config.marks.gapForCross) {
      h.mark = "×";
      crosses++;
    }
  }
}

function round1(n: number): number {
  return Math.round(n * 10) / 10;
}
function round3(n: number): number {
  return Math.round(n * 1000) / 1000;
}
