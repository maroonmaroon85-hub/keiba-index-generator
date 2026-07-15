import { mkdirSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { loadConfig } from "./config.js";
import { loadPreRaceFromJson, overrideRaceParams } from "./parser/json.js";
import { scoreRace } from "./scoring/score.js";
import { renderHtml } from "./render/html.js";
import { toScoreExport } from "./export.js";
import type { Pace, TrackCondition } from "./model/pre-race.js";

interface Args {
  input?: string;
  pace?: Pace;
  condition?: TrackCondition;
  outDir: string;
  config?: string;
}

function parseArgs(argv: string[]): Args {
  const args: Args = { outDir: "out" };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    const next = argv[i + 1];
    switch (a) {
      case "--input": args.input = next; i++; break;
      case "--pace": args.pace = next as Pace; i++; break;
      case "--condition": args.condition = next as TrackCondition; i++; break;
      case "--out": args.outDir = next ?? "out"; i++; break;
      case "--config": args.config = next; i++; break;
    }
  }
  return args;
}

function main(): void {
  const args = parseArgs(process.argv.slice(2));
  if (!args.input) {
    console.error("使い方: npm run generate -- --input <race.json> [--pace H] [--condition 良] [--out out] [--config config.json]");
    process.exit(1);
  }

  const config = loadConfig(args.config);
  let pre = loadPreRaceFromJson(resolve(args.input));
  pre = overrideRaceParams(pre, args.pace, args.condition);

  const scored = scoreRace(pre, config);
  const html = renderHtml(pre, scored, config);
  const json = toScoreExport(scored);

  const outDir = resolve(args.outDir);
  mkdirSync(outDir, { recursive: true });
  const htmlPath = resolve(outDir, `${pre.race.raceId}.html`);
  const jsonPath = resolve(outDir, `${pre.race.raceId}.score.json`);
  writeFileSync(htmlPath, html, "utf-8");
  writeFileSync(jsonPath, JSON.stringify(json, null, 2), "utf-8");

  // コンソールに要約と生成パスを出す。
  const ranked = [...scored.horses].sort((a, b) => b.score - a.score);
  console.log(`\n${pre.race.course}${pre.race.surface}${pre.race.distance}m  ペース${pre.race.pace} 馬場${pre.race.condition}  (${scored.horses.length}頭)`);
  console.log("─".repeat(52));
  for (const h of ranked) {
    const mark = h.mark || " ";
    console.log(
      `${mark} ${String(h.number).padStart(2)} ${h.name.padEnd(10)} ` +
        `評${h.rank.padEnd(2)} 指数${String(h.score).padStart(5)} ` +
        `単${(h.winProb * 100).toFixed(1)}% 複${(h.placeProb * 100).toFixed(1)}%`,
    );
  }
  console.log("─".repeat(52));
  console.log(`HTML: ${htmlPath}`);
  console.log(`JSON: ${jsonPath}\n`);
}

main();
