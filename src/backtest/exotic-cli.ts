import { resolve } from "node:path";
import { readdirSync } from "node:fs";
import { readCsvShiftJis } from "../parser/csv.js";
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
 *   npm run exotic -- --dir . \
 *     --payout-b data/payout/haraimodoshiB.csv [--payout-a data/payout/haraimodoshiA.csv] \
 *     [--odds-col 49] [--min-horses 8] [--types 馬連,ワイド,三連複]
 *   （--dir はフォルダ内 *.CSV を全読み。zshで $args が単語分割されない問題を回避。）
 */

interface Args {
  input: string[];
  payoutA?: string;
  payoutB?: string;
  oddsCol?: number;
  minHorses?: number;
  types: ExoticType[];
  config?: string;
  ml?: string; // ML予測CSV(out/ml_test_pred.csv: raceid,umaban,prob,...)
  prob?: "rule" | "ml"; // どのwin_probで買い方を作るか
}

const ALL_TYPES: ExoticType[] = ["馬連", "馬単", "ワイド", "三連複", "三連単"];

function parseArgs(argv: string[]): Args {
  const args: Args = { input: [], types: [] };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    const next = argv[i + 1];
    switch (a) {
      case "--input": if (next) args.input.push(next); i++; break;
      case "--dir": {
        // フォルダ内の *.CSV を全部 input に追加（zshの$args単語分割問題を回避）。
        if (next) {
          for (const f of readdirSync(next)) {
            if (/\.csv$/i.test(f)) args.input.push(resolve(next, f));
          }
        }
        i++;
        break;
      }
      case "--payout-a": args.payoutA = next; i++; break;
      case "--payout-b": args.payoutB = next; i++; break;
      case "--odds-col": args.oddsCol = Number(next) - 1; i++; break;
      case "--min-horses": args.minHorses = Number(next); i++; break;
      case "--types": args.types = (next ?? "").split(",").filter(Boolean) as ExoticType[]; i++; break;
      case "--config": args.config = next; i++; break;
      case "--ml": args.ml = next; i++; break;
      case "--prob": args.prob = (next === "rule" ? "rule" : "ml"); i++; break;
    }
  }
  return args;
}

function main(): void {
  const args = parseArgs(process.argv.slice(2));
  if (args.input.length === 0 || (!args.payoutA && !args.payoutB)) {
    console.error(
      "使い方: npm run exotic -- --dir <成績CSVのフォルダ> --payout-b <配当B.csv> [--payout-a <配当A.csv>] [--odds-col 49] [--min-horses 8] [--types 馬連,ワイド]\n" +
      "  （個別指定は --input <CSV> も可・複数回）",
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

  // --ml: ML予測CSV(raceid,umaban,prob,...)を Map<raceId, Map<umaban, prob>> に。
  // これがある場合、ML評価期間(OOS)のレースだけが対象になる。--prob rule なら同一レース群を
  // ルールwin_probで評価（＝ML vs ルールの公平比較）、--prob ml(既定)ならMLで評価。
  let overrideProb: Map<string, Map<number, number>> | undefined;
  const useProb = args.prob ?? (args.ml ? "ml" : "rule");
  if (args.ml) {
    overrideProb = new Map();
    const rows = readCsvShiftJis(resolve(args.ml));
    for (let i = 1; i < rows.length; i++) {
      const [raceId, umaban, prob] = rows[i]!;
      if (!raceId || raceId === "raceid") continue;
      const rid = raceId.trim();
      const m = overrideProb.get(rid) ?? new Map<number, number>();
      m.set(Number(umaban), Number(prob));
      overrideProb.set(rid, m);
    }
    console.log(`ML予測: ${overrideProb.size}レース（評価はこの期間=OOSに限定, 使用確率=${useProb}）`);
  }
  // ML群での評価かつ --prob rule のときは、絞り込みだけ override で行い、確率はルールのまま。
  const subsetOnly = args.ml && useProb === "rule";

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

  // --prob rule かつ --ml のときは、レースをML群に絞り、確率はルールのまま渡す。
  const evalRaces = subsetOnly && overrideProb
    ? races.filter((r) => overrideProb!.has(r.pre.race.raceId))
    : races;
  const passOverride = useProb === "ml" ? overrideProb : undefined;

  const allReports = [] as ReturnType<typeof research>;
  const covered: string[] = [];
  for (const t of types) {
    const sub = new Map(Array.from(payouts).filter(([, p]) => hasType(p, t)));
    if (sub.size === 0) continue;
    const specs = standardStrategies([t]);
    const rep = research(evalRaces, sub, config, specs, passOverride);
    if (rep[0]) covered.push(`${t}:${rep[0].races}R`);
    allReports.push(...rep);
  }
  const reports = allReports.sort((a, b) => b.roi - a.roi);

  const label = args.ml ? `win_prob=${useProb.toUpperCase()}（OOS期間）` : "win_prob=RULE（全期間）";
  console.log(`\n=== 連系 買い方研究（回収率降順） ${label} 券種別対象レース数: ${covered.join(", ")} ===`);
  console.log(formatReports(reports));

  console.log(
    "\n注: 回収率100%超が継続すれば妙味あり。ただし平均点数が多い買い方は分散大。" +
    "\n    三連複/三連単/馬単は配当A(768レース)のみ、馬連/ワイドは配当B(全13年)。",
  );
}

main();
