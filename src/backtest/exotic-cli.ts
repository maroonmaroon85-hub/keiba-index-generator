import { resolve } from "node:path";
import { loadConfig } from "../config.js";
import { buildDataset } from "./dataset.js";
import { loadPayouts } from "./payout-parser.js";
import { research, standardStrategies, formatReports } from "./exotic-research.js";
import type { ExoticType } from "../ev/harville.js";

/**
 * 連系「買い方」研究CLI。
 * 成績フルセットCSV（scored races）× 払戻CSV（配当A/B）で、券種別に買い方の回収率を実測する。
 *
 * 使い方:
 *   npm run exotic -- --input DS*.CSV [--input ...] \
 *     --payout-b data/payout/haraimodoshiB.csv [--payout-a data/payout/haraimodoshiA.csv] \
 *     [--odds-col 49] [--min-horses 8] [--types 馬連,ワイド,三連複]
 */

interface Args {
  input: string[];
  payoutA?: string;
  payoutB?: string;
  oddsCol?: number;
  minHorses?: number;
  types: ExoticType[];
  config?: string;
}

const ALL_TYPES: ExoticType[] = ["馬連", "馬単", "ワイド", "三連複", "三連単"];

function parseArgs(argv: string[]): Args {
  const args: Args = { input: [], types: [] };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    const next = argv[i + 1];
    switch (a) {
      case "--input": if (next) args.input.push(next); i++; break;
      case "--payout-a": args.payoutA = next; i++; break;
      case "--payout-b": args.payoutB = next; i++; break;
      case "--odds-col": args.oddsCol = Number(next) - 1; i++; break;
      case "--min-horses": args.minHorses = Number(next); i++; break;
      case "--types": args.types = (next ?? "").split(",").filter(Boolean) as ExoticType[]; i++; break;
      case "--config": args.config = next; i++; break;
    }
  }
  return args;
}

function main(): void {
  const args = parseArgs(process.argv.slice(2));
  if (args.input.length === 0 || (!args.payoutA && !args.payoutB)) {
    console.error(
      "使い方: npm run exotic -- --input <成績CSV>[複数可] --payout-b <配当B.csv> [--payout-a <配当A.csv>] [--odds-col 49] [--min-horses 8] [--types 馬連,ワイド]",
    );
    process.exit(1);
    return;
  }
  const config = loadConfig(args.config);

  const payouts = loadPayouts(
    args.payoutB ? resolve(args.payoutB) : undefined,
    args.payoutA ? resolve(args.payoutA) : undefined,
  );
  console.log(`払戻: ${payouts.size}レース読み込み`);

  const races = buildDataset(args.input.map((p) => resolve(p)), {
    oddsCol: args.oddsCol,
    minHorses: args.minHorses,
  });
  console.log(`成績: ${races.length}レース構築`);

  const joined = races.filter((r) => payouts.has(r.pre.race.raceId)).length;
  console.log(`結合: ${joined}レースが払戻とマッチ`);
  if (joined === 0) {
    console.error("払戻とマッチするレースが0件。raceIdの整合を確認してください。");
    process.exit(1);
    return;
  }

  // 券種ごとに「その払戻を持つレースだけ」を対象にする。
  // 配当Bは馬連/ワイド=全13年、配当Aの三連系=768レースのみ。混ぜると三連系ROIが
  // 「払戻データの無いレースでも買った」扱いになり不当に下がるため、券種別にpayoutsを絞る。
  const types = args.types.length ? args.types : ALL_TYPES;
  const keyOf = (t: ExoticType) =>
    t === "馬連" ? "umaren" : t === "馬単" ? "umatan" : t === "ワイド" ? "wide" : t === "三連複" ? "sanrenpuku" : "sanrentan";
  const hasType = (p: unknown, t: ExoticType) => {
    const v = (p as Record<string, unknown>)[keyOf(t)];
    return Array.isArray(v) ? v.length > 0 : Boolean(v);
  };

  const allReports = [] as ReturnType<typeof research>;
  const covered: string[] = [];
  for (const t of types) {
    const sub = new Map(Array.from(payouts).filter(([, p]) => hasType(p, t)));
    if (sub.size === 0) continue;
    covered.push(`${t}:${sub.size}R`);
    const specs = standardStrategies([t]);
    allReports.push(...research(races, sub, config, specs));
  }
  const reports = allReports.sort((a, b) => b.roi - a.roi);

  console.log(`\n=== 連系 買い方研究（回収率降順） 券種別対象レース数: ${covered.join(", ")} ===`);
  console.log(formatReports(reports));

  console.log(
    "\n注: 回収率100%超が継続すれば妙味あり。ただし平均点数が多い買い方は分散大。" +
    "\n    三連複/三連単/馬単は配当A(768レース)のみ、馬連/ワイドは配当B(全13年)。",
  );
}

main();
