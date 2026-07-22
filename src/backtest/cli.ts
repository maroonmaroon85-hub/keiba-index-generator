import { mkdirSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { loadConfig } from "../config.js";
import { scoreRace } from "../scoring/score.js";
import { buildDataset } from "./dataset.js";
import {
  byRank,
  byMark,
  byFlag,
  calibration,
  type Sample,
  type GroupStat,
} from "./metrics.js";

interface Args {
  input: string[];
  oddsCol?: number;
  minHorses?: number;
  outDir: string;
  config?: string;
}

function parseArgs(argv: string[]): Args {
  const args: Args = { input: [], outDir: "out" };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    const next = argv[i + 1];
    switch (a) {
      case "--input": if (next) args.input.push(next); i++; break;
      case "--odds-col": args.oddsCol = Number(next) - 1; i++; break; // 1始まり列番号→0始まり
      case "--min-horses": args.minHorses = Number(next); i++; break;
      case "--out": args.outDir = next ?? "out"; i++; break;
      case "--config": args.config = next; i++; break;
    }
  }
  return args;
}

function fmt(n: number): string {
  return n.toFixed(1).padStart(5);
}
function roi(v: number | null): string {
  return v === null ? "  --  " : (v.toFixed(0) + "%").padStart(6);
}

function printGroupTable(title: string, rows: GroupStat[]): void {
  console.log(`\n■ ${title}`);
  console.log("  区分        n     勝率  連対率 複勝率  単回収");
  for (const r of rows) {
    console.log(
      `  ${r.label.padEnd(10)}${String(r.n).padStart(5)}  ${fmt(r.winRate)} ${fmt(r.quinellaRate)} ${fmt(r.placeRate)}  ${roi(r.winROI)}`,
    );
  }
}

function toCsv(rows: GroupStat[]): string {
  const head = "区分,n,勝率,連対率,複勝率,単勝回収率";
  const body = rows.map((r) =>
    [r.label, r.n, r.winRate.toFixed(1), r.quinellaRate.toFixed(1), r.placeRate.toFixed(1), r.winROI === null ? "" : r.winROI.toFixed(1)].join(","),
  );
  return [head, ...body].join("\n");
}

function main(): void {
  const args = parseArgs(process.argv.slice(2));
  if (args.input.length === 0) {
    console.error("使い方: npm run backtest -- --input <成績フルセットCSV> [--input ...(複数可)] [--odds-col 列番号] [--min-horses 5] [--out out] [--config config.json]");
    process.exit(1);
    return;
  }
  const config = loadConfig(args.config);
  const races = buildDataset(args.input.map((p) => resolve(p)), {
    oddsCol: args.oddsCol,
    minHorses: args.minHorses,
  });

  if (races.length === 0) {
    console.error("バックテスト対象レースが0件です（--min-horses を下げる、または期間の広いCSVを使ってください）。");
    process.exit(1);
    return;
  }

  const samples: Sample[] = [];
  for (const { pre, post } of races) {
    const scored = scoreRace(pre, config);
    const finishByNumber = new Map(post.horses.map((h) => [h.number, h]));
    for (const h of scored.horses) {
      const p = finishByNumber.get(h.number);
      if (!p || p.finish <= 0) continue; // 取消・除外等
      samples.push({
        rank: h.rank,
        mark: h.mark,
        flags: [...h.flags.plus, ...h.flags.minus],
        winProb: h.winProb,
        placeProb: h.placeProb,
        finish: p.finish,
        fieldSize: scored.horses.length,
        winOdds: p.finalWinOdds,
      });
    }
  }

  const rankRows = byRank(samples, config.rank.thresholds.map((t) => t.rank));
  const markRows = byMark(samples, [...config.marks.order, "×"]);
  const flagRows = byFlag(samples);
  const calRows = calibration(samples);

  console.log(`\nバックテスト: ${races.length}レース / ${samples.length}頭ぶん`);
  const withOdds = samples.filter((s) => s.winOdds > 0).length;
  console.log(withOdds > 0 ? `（単オッズあり ${withOdds}頭 → 回収率を算出）` : "（単オッズ列なし → 回収率は '--'。--odds-col で列指定すると算出）");

  printGroupTable("総合評価ランク別", rankRows);
  printGroupTable("印別", markRows);
  printGroupTable("条件フラグ別（(全体)と比較して効いているか）", flagRows);

  console.log("\n■ win_prob キャリブレーション（予測 vs 実測）");
  console.log("  予測帯       n    予測平均  実勝率  実複勝率");
  for (const c of calRows) {
    console.log(`  ${c.label.padEnd(10)}${String(c.n).padStart(5)}  ${fmt(c.predicted)}  ${fmt(c.actualWin)}  ${fmt(c.actualPlace)}`);
  }

  const outDir = resolve(args.outDir);
  mkdirSync(outDir, { recursive: true });
  writeFileSync(resolve(outDir, "backtest_rank.csv"), toCsv(rankRows), "utf-8");
  writeFileSync(resolve(outDir, "backtest_mark.csv"), toCsv(markRows), "utf-8");
  writeFileSync(resolve(outDir, "backtest_flag.csv"), toCsv(flagRows), "utf-8");
  writeFileSync(
    resolve(outDir, "backtest_calibration.csv"),
    ["予測帯,n,予測平均,実勝率,実複勝率", ...calRows.map((c) => [c.label, c.n, c.predicted.toFixed(1), c.actualWin.toFixed(1), c.actualPlace.toFixed(1)].join(","))].join("\n"),
    "utf-8",
  );
  console.log(`\nCSV出力: ${outDir}/backtest_{rank,mark,flag,calibration}.csv\n`);
}

main();
