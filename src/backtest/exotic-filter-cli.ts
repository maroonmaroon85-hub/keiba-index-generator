import { resolve } from "node:path";
import { readdirSync } from "node:fs";
import { loadConfig } from "../config.js";
import { readCsvShiftJis } from "../parser/csv.js";
import { buildDataset } from "./dataset.js";
import { loadPayouts } from "./payout-parser.js";
import { perRaceRecords, analyze } from "./exotic-filter.js";
import { boxCombos, formationCombos, topKCombos } from "../ev/strategies.js";
import type { StrategySpec } from "./exotic-research.js";
import type { ExoticType, Combo } from "../ev/harville.js";
import type { ScoredRace } from "../scoring/types.js";

/**
 * 連系レース絞り込み探索CLI。
 * 券種×買い方を1つ固定し、レース条件別に回収率を層別表示（100%超の部分集合を探す）。
 *
 * 例:
 *   npm run exotic-filter -- --dir . --payout-b data/payout/haraimodoshiB.csv \
 *     --odds-col 49 --min-horses 8 --ml out/ml_test_pred.csv --type ワイド --strategy top6
 *
 * strategy: box2..box6 / f1x4(軸1×相手4) / f2x4 / top3 / top6 / top10
 */

function makeSpec(type: ExoticType, strategy: string): StrategySpec {
  const gen = (s: ScoredRace): Combo[] => {
    if (strategy.startsWith("box")) return boxCombos(s, type, Number(strategy.slice(3)));
    if (strategy === "f1x4") return formationCombos(s, type, 1, 4);
    if (strategy === "f2x4") return formationCombos(s, type, 2, 4);
    if (strategy.startsWith("top")) return topKCombos(s, type, Number(strategy.slice(3)));
    return topKCombos(s, type, 6);
  };
  return { name: `${type} ${strategy}`, type, gen };
}

function main(): void {
  const argv = process.argv.slice(2);
  const get = (flag: string) => {
    const i = argv.indexOf(flag);
    return i >= 0 ? argv[i + 1] : undefined;
  };
  const input: string[] = [];
  const dir = get("--dir");
  if (dir) for (const f of readdirSync(dir)) if (/\.csv$/i.test(f)) input.push(resolve(dir, f));
  for (let i = 0; i < argv.length; i++) if (argv[i] === "--input" && argv[i + 1]) input.push(argv[i + 1]!);

  const payoutB = get("--payout-b");
  const payoutA = get("--payout-a");
  if (input.length === 0 || (!payoutB && !payoutA)) {
    console.error("使い方: npm run exotic-filter -- --dir <CSVフォルダ> --payout-b <配当B.csv> [--ml out/ml_test_pred.csv] [--type ワイド] [--strategy top6] [--odds-col 49] [--min-horses 8]");
    process.exit(1);
    return;
  }
  const type = (get("--type") ?? "ワイド") as ExoticType;
  const strategy = get("--strategy") ?? "top6";
  const oddsCol = get("--odds-col") ? Number(get("--odds-col")) - 1 : undefined;
  const minHorses = get("--min-horses") ? Number(get("--min-horses")) : undefined;
  const config = loadConfig(get("--config"));

  const payouts = loadPayouts(payoutB ? resolve(payoutB) : undefined, payoutA ? resolve(payoutA) : undefined);
  const races = buildDataset(input.map((p) => resolve(p)), { oddsCol, minHorses });

  let override: Map<string, Map<number, number>> | undefined;
  const ml = get("--ml");
  if (ml) {
    override = new Map();
    const rows = readCsvShiftJis(resolve(ml));
    for (let i = 1; i < rows.length; i++) {
      const [rid, umaban, prob] = rows[i]!;
      if (!rid || rid === "raceid") continue;
      const m = override.get(rid.trim()) ?? new Map<number, number>();
      m.set(Number(umaban), Number(prob));
      override.set(rid.trim(), m);
    }
  }

  const spec = makeSpec(type, strategy);
  const records = perRaceRecords(races, payouts, config, spec, override);
  console.log(`\n=== 絞り込み探索: ${spec.name}  win_prob=${ml ? "ML(OOS)" : "RULE"}  対象${records.length}レース ===`);
  console.log(analyze(records));
  console.log("\n注: 100%超のバケツ＝その条件のレースだけ買えば妙味あり（ただしR数が少ないと偶然の可能性）。");
}

main();
