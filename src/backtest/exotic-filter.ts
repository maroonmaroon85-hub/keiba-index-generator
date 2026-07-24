import { scoreRace } from "../scoring/score.js";
import { boxCombos, formationCombos, topKCombos } from "../ev/strategies.js";
import { evalExoticStrategy, type RacePayout } from "./exotic-sim.js";
import type { Combo, ExoticType } from "../ev/harville.js";
import type { BacktestRace } from "./dataset.js";
import type { Config } from "../config.js";
import type { StrategySpec } from "./exotic-research.js";

/**
 * レース絞り込み探索。
 * 固定の券種×買い方で、各レースの購入コストと払戻を出し、レース条件（頭数・1番人気の強さ・
 * 1-2番手の確率差・馬場・距離・人気馬オッズ）で層別して回収率を比較する。
 * 「どの条件のレースなら回収率が高いか（100%超の部分集合はあるか）」を実測する。
 */

export interface RaceRecord {
  raceId: string;
  cost: number; // 購入額(円)
  ret: number; // 払戻(円)
  fieldSize: number;
  top1: number; // 最大win_prob（1番人気の強さ）
  gap: number; // top1 - top2
  surface: string; // 芝/ダ
  distance: number;
  favOdds: number; // win_prob最上位馬の市場オッズ
}

/** 券種×買い方を1つ固定して、払戻のあるレースごとのコスト/払戻/条件を出す。 */
export function perRaceRecords(
  races: BacktestRace[],
  payouts: Map<string, RacePayout>,
  config: Config,
  spec: StrategySpec,
  overrideProb?: Map<string, Map<number, number>>,
): RaceRecord[] {
  const out: RaceRecord[] = [];
  for (const r of races) {
    const raceId = r.pre.race.raceId;
    if (!payouts.has(raceId)) continue;
    if (overrideProb && !overrideProb.has(raceId)) continue;
    const sc = scoreRace(r.pre, config);
    const ov = overrideProb?.get(raceId);
    if (ov) for (const h of sc.horses) h.winProb = ov.get(h.number) ?? 0;

    const combos: Combo[] = spec.gen(sc);
    const res = evalExoticStrategy([{ raceId, combos }], payouts, spec.type);
    if (res.bets === 0) continue; // 券種の払戻が無いレース
    const cost = res.bets * 100;
    const ret = res.roi * res.bets; // roi(%) * bets = ret/100 * bets*100 の整理

    const probs = sc.horses.map((h) => h.winProb).sort((a, b) => b - a);
    const top1 = probs[0] ?? 0;
    const top2 = probs[1] ?? 0;
    const favNumber = [...sc.horses].sort((a, b) => b.winProb - a.winProb)[0]?.number;
    const favOdds = r.pre.horses.find((h) => h.number === favNumber)?.odds ?? 0;

    out.push({
      raceId,
      cost,
      ret,
      fieldSize: sc.horses.length,
      top1,
      gap: top1 - top2,
      surface: r.pre.race.surface,
      distance: r.pre.race.distance,
      favOdds,
    });
  }
  return out;
}

export interface Bucket {
  label: string;
  races: number;
  hitless: number; // 参考
  cost: number;
  ret: number;
  roi: number;
}

/** records を、値→バケツラベルの関数で層別して回収率を集計。 */
export function bucketBy(
  records: RaceRecord[],
  labeler: (r: RaceRecord) => string | null,
  order?: string[],
): Bucket[] {
  const map = new Map<string, { races: number; cost: number; ret: number }>();
  for (const r of records) {
    const label = labeler(r);
    if (label === null) continue;
    const b = map.get(label) ?? { races: 0, cost: 0, ret: 0 };
    b.races++;
    b.cost += r.cost;
    b.ret += r.ret;
    map.set(label, b);
  }
  const keys = order ? order.filter((k) => map.has(k)) : [...map.keys()].sort();
  return keys.map((k) => {
    const b = map.get(k)!;
    return { label: k, races: b.races, hitless: 0, cost: b.cost, ret: b.ret, roi: b.cost ? (b.ret / b.cost) * 100 : 0 };
  });
}

export function formatBuckets(title: string, buckets: Bucket[]): string {
  const head = `\n■ ${title}\n  区分              R      回収率`;
  const rows = buckets.map(
    (b) => `  ${b.label.padEnd(14)}${String(b.races).padStart(6)}   ${b.roi.toFixed(1).padStart(6)}%`,
  );
  return [head, ...rows].join("\n");
}

/** 標準の層別軸セットで records を分析。 */
export function analyze(records: RaceRecord[]): string {
  const parts: string[] = [];
  const total = records.reduce((a, r) => ({ cost: a.cost + r.cost, ret: a.ret + r.ret }), { cost: 0, ret: 0 });
  parts.push(`全体: ${records.length}レース  回収率 ${((total.ret / total.cost) * 100).toFixed(1)}%`);

  parts.push(formatBuckets("頭数", bucketBy(records, (r) => (r.fieldSize <= 10 ? "≤10頭" : r.fieldSize <= 13 ? "11-13頭" : r.fieldSize <= 16 ? "14-16頭" : "17-18頭"),
    ["≤10頭", "11-13頭", "14-16頭", "17-18頭"])));

  parts.push(formatBuckets("1番人気の強さ(top1 win_prob)", bucketBy(records, (r) =>
    r.top1 < 0.2 ? "<0.20" : r.top1 < 0.3 ? "0.20-0.30" : r.top1 < 0.4 ? "0.30-0.40" : r.top1 < 0.5 ? "0.40-0.50" : "≥0.50",
    ["<0.20", "0.20-0.30", "0.30-0.40", "0.40-0.50", "≥0.50"])));

  parts.push(formatBuckets("1-2番手の確率差(gap)", bucketBy(records, (r) =>
    r.gap < 0.05 ? "<0.05(混戦)" : r.gap < 0.1 ? "0.05-0.10" : r.gap < 0.2 ? "0.10-0.20" : "≥0.20(抜けた)",
    ["<0.05(混戦)", "0.05-0.10", "0.10-0.20", "≥0.20(抜けた)"])));

  parts.push(formatBuckets("人気馬オッズ(win_prob最上位馬)", bucketBy(records, (r) =>
    r.favOdds <= 0 ? null : r.favOdds < 2 ? "<2.0" : r.favOdds < 3 ? "2.0-3.0" : r.favOdds < 5 ? "3.0-5.0" : r.favOdds < 10 ? "5.0-10" : "≥10",
    ["<2.0", "2.0-3.0", "3.0-5.0", "5.0-10", "≥10"])));

  parts.push(formatBuckets("馬場", bucketBy(records, (r) => (r.surface === "芝" ? "芝" : "ダート"), ["芝", "ダート"])));

  parts.push(formatBuckets("距離", bucketBy(records, (r) =>
    r.distance <= 1400 ? "≤1400" : r.distance <= 1800 ? "1401-1800" : r.distance <= 2200 ? "1801-2200" : "≥2201",
    ["≤1400", "1401-1800", "1801-2200", "≥2201"])));

  return parts.join("\n");
}
