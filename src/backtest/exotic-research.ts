import { scoreRace } from "../scoring/score.js";
import type { Combo, ExoticType } from "../ev/harville.js";
import { boxCombos, formationCombos, topKCombos } from "../ev/strategies.js";
import { evalExoticStrategy, type RacePayout, type ExoticResult } from "./exotic-sim.js";
import type { BacktestRace } from "./dataset.js";
import type { Config } from "../config.js";

/**
 * 連系「買い方」研究フレームワーク。
 * scoredレース群 × 払戻 に対して、複数の買い方（ボックス/フォーメーション/上位K点）を
 * 総当たりで回し、券種別に回収率・的中率・平均点数を比較する。
 * 「馬連はボックス何頭が一番か」「三連複は軸1頭×相手5頭か上位10点か」等を実測で決める。
 */

export interface StrategySpec {
  name: string;
  type: ExoticType;
  gen: (scored: ReturnType<typeof scoreRace>) => Combo[];
}

export interface StrategyReport extends ExoticResult {
  name: string;
  avgPoints: number; // 1レースあたり平均購入点数
  races: number;
}

/** 標準の買い方セットを生成（券種ごとにbox/formation/topK）。 */
export function standardStrategies(types: ExoticType[]): StrategySpec[] {
  const specs: StrategySpec[] = [];
  for (const type of types) {
    const size = type === "三連複" || type === "三連単" ? 3 : 2;
    for (const n of [size, size + 1, size + 2, size + 3]) {
      specs.push({ name: `${type} box${n}`, type, gen: (s) => boxCombos(s, type, n) });
    }
    specs.push({ name: `${type} 軸1×相手4`, type, gen: (s) => formationCombos(s, type, 1, 4) });
    if (size === 3) specs.push({ name: `${type} 軸2×相手4`, type, gen: (s) => formationCombos(s, type, 2, 4) });
    for (const k of [3, 6, 10]) specs.push({ name: `${type} 上位${k}点`, type, gen: (s) => topKCombos(s, type, k) });
  }
  return specs;
}

/** 各戦略を回して回収率降順で返す。
 * overrideProb を渡すと各馬の winProb を（raceId→馬番→確率）で差し替える。
 * ML予測(out/ml_test_pred.csv)を渡してML順位付けで買い方を評価する用途。
 * override がある場合、その raceId のレースだけを対象にする（＝MLのOOS期間に自動で絞られる）。 */
export function research(
  races: BacktestRace[],
  payouts: Map<string, RacePayout>,
  config: Config,
  specs: StrategySpec[],
  overrideProb?: Map<string, Map<number, number>>,
): StrategyReport[] {
  // 払戻のあるレースだけ、一度スコアしてキャッシュ。
  const scored = races
    .filter((r) => payouts.has(r.pre.race.raceId))
    .filter((r) => !overrideProb || overrideProb.has(r.pre.race.raceId))
    .map((r) => {
      const sc = scoreRace(r.pre, config);
      const ov = overrideProb?.get(r.pre.race.raceId);
      if (ov) for (const h of sc.horses) h.winProb = ov.get(h.number) ?? 0;
      return { raceId: r.pre.race.raceId, sc };
    });

  const reports: StrategyReport[] = specs.map((spec) => {
    const entries = scored.map((x) => ({ raceId: x.raceId, combos: spec.gen(x.sc) }));
    const totalPoints = entries.reduce((a, e) => a + e.combos.length, 0);
    const res = evalExoticStrategy(entries, payouts, spec.type);
    const races = entries.filter((e) => payouts.has(e.raceId)).length;
    return { ...res, name: spec.name, races, avgPoints: races ? totalPoints / races : 0 };
  });
  return reports.sort((a, b) => b.roi - a.roi);
}

/** コンソール表示用の整形。 */
export function formatReports(reports: StrategyReport[]): string {
  const head = "  買い方              R    平均点  的中率   回収率";
  const rows = reports.map(
    (r) =>
      `  ${r.name.padEnd(18)}${String(r.races).padStart(4)}  ${r.avgPoints.toFixed(1).padStart(6)}  ${((r.hits / Math.max(1, r.bets)) * 100).toFixed(1).padStart(5)}%  ${r.roi.toFixed(1).padStart(6)}%`,
  );
  return [head, ...rows].join("\n");
}
